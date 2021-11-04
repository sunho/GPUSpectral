#include "Renderer.h"

#include <sunho3d/renderer/shaders/DisplayTextureFrag.h>
#include <sunho3d/renderer/shaders/DisplayTextureVert.h>
#include <sunho3d/renderer/shaders/ForwardPhongFrag.h>
#include <sunho3d/renderer/shaders/ForwardPhongVert.h>
#include <sunho3d/renderer/shaders/ForwardRTFrag.h>
#include <sunho3d/renderer/shaders/ForwardRTVert.h>
#include <tiny_gltf.h>

#include "../Entity.h"
#include "../Scene.h"

#include <iostream>
using namespace sunho3d;

Renderer::Renderer(Window* window)
    : window(window), driver(window) {
    for (size_t i = 0; i < MAX_INFLIGHTS; ++i) {
        inflights[i].fence = driver.createFence();
        inflights[i].rg = std::make_unique<FrameGraph>(*this);
    }

    Program prog(ForwardPhongVert, ForwardPhongVertSize, ForwardPhongFrag,ForwardPhongFragSize);
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addUniformBuffer(0, 2)
        .addUniformBuffer(0, 2)
        .addTexture(1, 0)
        .addTexture(1, 1);
    fowradPassProgram = driver.createProgram(prog);

    Program prog2(DisplayTextureVert, DisplayTextureVertSize, DisplayTextureFrag, DisplayTextureFragSize);
    prog2.parameterLayout
        .addTexture(1, 0);
    blitProgram = driver.createProgram(prog2);

    Program prog3(ForwardRTVert, ForwardRTVertSize, ForwardRTFrag, ForwardRTFragSize);
    prog3.parameterLayout
        .addStorageBuffer(2, 0)
        .addStorageBuffer(2, 1)
        .addUniformBuffer(0, 0);
    forwardRTProgram = driver.createProgram(prog3);

    Primitive primitive;
    std::vector<float> v = {
        -1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
    };
    std::vector<uint16_t> indices = { 0, 1, 2, 3, 4, 5 };
    primitive.attibutes[0] = {
        .name = "position",
        .index = 0,
        .offset = 0,
        .stride = 8,
        .type = ElementType::FLOAT2
    };
    auto buffer0 = driver.createBufferObject(4 * v.size(), BufferUsage::VERTEX);
    driver.updateBufferObject(buffer0, { .data = (uint32_t*)v.data() }, 0);
    auto vbo = driver.createVertexBuffer(1, 6, 1, primitive.attibutes);
    driver.setVertexBuffer(vbo, 0, buffer0);

    auto ibo = driver.createIndexBuffer(indices.size());
    driver.updateIndexBuffer(ibo, { .data = (uint32_t*)indices.data() }, 0);
    primitive.indexBuffer = ibo;
    primitive.vertexBuffer = vbo;
    quadPrimitive = driver.createPrimitive(PrimitiveMode::TRIANGLES);
    driver.setPrimitiveBuffer(quadPrimitive, vbo, ibo);

    surfaceRenderTarget = driver.createDefaultRenderTarget();
}

Renderer::~Renderer() {
}

void Renderer::rasterSuite(Scene* scene) {
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver.waitFence(inflight.fence);
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
    }
    Handle<HwInflight> handle = driver.beginFrame(inflight.fence);
    inflight.handle = handle;

    FrameGraph& renderGraph = *inflight.rg;
    renderGraph.reset();

    scene->prepare();
    SceneData& sceneData = scene->getSceneData();

    auto sceneDataRef = renderGraph.declareResource<SceneData*>("SceneData");
    renderGraph.defineResource(sceneDataRef, &sceneData);

    auto shadowMapRef = renderGraph.declareResource<Handle<HwTexture>>("ShadowMap");
    auto forwardRef = renderGraph.declareResource<Handle<HwTexture>>("Forward");

    auto lightUB = renderGraph.createUniformBufferSC(sizeof(LightBuffer));

    std::map<Material*, Handle<HwUniformBuffer>> materialBuffers;

    const auto getOrCreateMaterialBuffer = [&](Material* material) {
        auto it = materialBuffers.find(material);
        if (it != materialBuffers.end()) {
            return it->second;
        }
        MaterialBuffer materialBuffer;
        materialBuffer.phong = 100;
        materialBuffer.specular = glm::vec4(0.2, 0.2, 0.2, 1.0);
        auto mb = renderGraph.createUniformBufferSC(sizeof(MaterialBuffer));
        driver.updateUniformBuffer(mb, { .data = (uint32_t*)&materialBuffer }, 0);
        materialBuffers.emplace(material, mb);
        return mb;
    };

    renderGraph.addRenderPass("generate shadow maps", { sceneDataRef }, { shadowMapRef }, [this, scene, shadowMapRef, lightUB, sceneDataRef, &renderGraph, getOrCreateMaterialBuffer](FrameGraph& fg) {
        auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        auto depth = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        ColorAttachment att = {};
        att.colors[0] = {
            .handle = color
        };
        att.targetNum = 1;
        auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att, TextureAttachment{ .handle = depth });
        SceneData* sceneData = fg.getResource<SceneData*>(sceneDataRef);
        auto camera = scene->getCamera();

        auto pos = sceneData->lightBuffer.lights[0].pos;
        camera.lookAt(pos, glm::vec3(0.0, 1.0, 0), glm::vec3(0, 1, 0));
        // tan (th/2) = t/n = 1/1
        // th = pi / 2
        camera.setProjectionFov(90.0f, 1.0f, 0.8f, 12.0f);
        sceneData->lightBuffer.lightVP[0] = camera.proj * camera.view;

        PipelineState pipe = {};
        pipe.program = this->fowradPassProgram;
        pipe.depthTest.enabled = 1;
        pipe.depthTest.write = 1;
        pipe.depthTest.compareOp = CompareOp::LESS;
        RenderPassParams params;
        driver.beginRenderPass(renderTarget, params);
        for (size_t i = 0; i < sceneData->geometries.size(); ++i) {
            auto& geom = sceneData->geometries[i];
            auto& model = sceneData->worldTransforms[i];
            pipe.bindUniformBuffer(0, 0, this->createTransformBuffer(renderGraph, camera, model));
            pipe.bindUniformBuffer(0, 1, lightUB);
            pipe.bindUniformBuffer(0, 2, getOrCreateMaterialBuffer(geom.material));
            pipe.bindTexture(1, 0, geom.material->diffuseMap);
            driver.draw(pipe, geom.primitive);
        }
        driver.endRenderPass();

        driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData->lightBuffer }, 0);
        fg.defineResource<Handle<HwTexture>>(shadowMapRef, depth);
    });

    renderGraph.addRenderPass("forward", { sceneDataRef, shadowMapRef }, {forwardRef}, [this, scene, lightUB, forwardRef, sceneDataRef, shadowMapRef, &renderGraph, getOrCreateMaterialBuffer](FrameGraph& fg) {
        auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        auto depth = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        ColorAttachment att = {};
        att.colors[0] = {
            .handle = color
        };
        att.targetNum = 1;
        auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att, TextureAttachment{ .handle = depth });
        SceneData* sceneData = fg.getResource<SceneData*>(sceneDataRef);
        Handle<HwTexture> shadowMap = fg.getResource<Handle<HwTexture>>(shadowMapRef);
        auto& camera = scene->getCamera();

        PipelineState pipe = {};
        pipe.program = this->fowradPassProgram;
        pipe.depthTest.enabled = 1;
        pipe.depthTest.write = 1;
        pipe.depthTest.compareOp = CompareOp::LESS;
        RenderPassParams params;
        driver.beginRenderPass(renderTarget, params);
        for (size_t i = 0; i < sceneData->geometries.size(); ++i) {
            auto& geom = sceneData->geometries[i];
            auto& model = sceneData->worldTransforms[i];
            pipe.bindUniformBuffer(0, 0, this->createTransformBuffer(renderGraph, camera, model));
            pipe.bindUniformBuffer(0, 1, lightUB);
            pipe.bindUniformBuffer(0, 2, getOrCreateMaterialBuffer(geom.material));
            pipe.bindTexture(1, 0, geom.material->diffuseMap);
            pipe.bindTexture(1, 1, shadowMap);
            driver.draw(pipe, geom.primitive);
        }
        driver.endRenderPass();
        fg.defineResource<Handle<HwTexture>>(forwardRef, color);
    });
    renderGraph.addRenderPass("blit", { forwardRef }, {}, [this, scene, lightUB, sceneDataRef, forwardRef, &renderGraph, getOrCreateMaterialBuffer](FrameGraph& fg) {
        Handle<HwTexture> forwardMap = fg.getResource<Handle<HwTexture>>(forwardRef);

        PipelineState pipe = {};
        pipe.program = this->blitProgram;
        RenderPassParams params;
        driver.beginRenderPass(this->surfaceRenderTarget, params);
        pipe.bindTexture(1, 0, forwardMap);
        driver.draw(pipe, this->quadPrimitive);
        driver.endRenderPass();
    });
    renderGraph.submit();
    driver.endFrame();
}

Handle<HwUniformBuffer> Renderer::createTransformBuffer(FrameGraph& rg, const Camera& camera, const glm::mat4& model) {
    auto tb = rg.createUniformBufferSC(sizeof(TransformBuffer));
    TransformBuffer transformBuffer;
    transformBuffer.MVP = camera.proj * camera.view * model;
    transformBuffer.invModelT = glm::transpose(glm::inverse(model));
    transformBuffer.model = model;
    transformBuffer.cameraPos = glm::vec4(camera.pos(), 1.0f);
    driver.updateUniformBuffer(tb, { .data = (uint32_t*)&transformBuffer }, 0);
    return tb;
}

void Renderer::run(Scene* scene) {
    rtSuite(scene);
}

void Renderer::rtSuite(Scene* scene) {
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver.waitFence(inflight.fence);
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
    }
    Handle<HwInflight> handle = driver.beginFrame(inflight.fence);
    inflight.handle = handle;

    FrameGraph& renderGraph = *inflight.rg;
    renderGraph.reset();
    inflight.instances.clear();

    scene->prepare();
    SceneData& sceneData = scene->getSceneData();

    auto tlasRef = renderGraph.declareResource<Handle<HwTLAS>>("TLAS");
    auto hitBufferRef = renderGraph.declareResource<Handle<HwBufferObject>>("Hit Buffer");
    auto renderedRef = renderGraph.declareResource<Handle<HwTexture>>("Rendered texture");
    auto sceneBufferRef = renderGraph.declareResource<Handle<HwUniformBuffer>>("RT Scene buffer");
    renderGraph.addRenderPass("build tlas and scene buffer", { }, { tlasRef, sceneBufferRef }, [this, sceneBufferRef, tlasRef, &inflight, &sceneData, &renderGraph](FrameGraph& fg) {
        ForwardRTSceneBuffer sceneBuffer = {};
        auto f = driver.getFrameSize();
        sceneBuffer.frameSize = glm::uvec2(f.width, f.height);
        sceneBuffer.instanceNum = sceneData.geometries.size();
        for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
            auto& geom = sceneData.geometries[i];
            auto& model = sceneData.worldTransforms[i];
            Handle<HwBLAS> blas = driver.createBLAS();
            auto it = this->blasMap.find(geom.primitive.getId());
            if (it == this->blasMap.end()) {
                driver.buildBLAS(blas, geom.primitive);
                this->blasMap.emplace(geom.primitive.getId(), blas);
            } else {
                blas = it->second;
            }
            RTInstance instance = {};
            instance.blas = blas;
            glm::mat4x3 t = {};
            t[0] = model[0];
            t[1] = model[1];
            t[2] = model[2];
            t[3] = model[3];
            instance.transfom = t;
            inflight.instances.push_back(instance); 
            Instance ginstance = {};
            ginstance.vertexStart = geom.vertexStart;
            ginstance.transform = model;
            sceneBuffer.instances[i] = ginstance;
        }
        RTSceneDescriptor desc = { inflight.instances.data(), inflight.instances.size() };
        auto tlas = driver.createTLAS();
        driver.buildTLAS(tlas, desc);
        renderGraph.defineResource(tlasRef, tlas);

        auto sb = renderGraph.createUniformBufferSC(sizeof(ForwardRTSceneBuffer));
        driver.updateUniformBuffer(sb, { (uint32_t*)&sceneBuffer }, 0 );
        renderGraph.defineResource(sceneBufferRef, sb);
    });

    renderGraph.addRenderPass("intersect", { tlasRef }, { hitBufferRef }, [this, hitBufferRef, tlasRef, scene, &sceneData, &renderGraph](FrameGraph& fg) {
        Handle<HwTLAS> tlas = fg.getResource<Handle<HwTLAS>>(tlasRef);
        auto f = driver.getFrameSize();
        std::vector<Ray> rays;
        rays.resize(f.width * f.height);
        for (size_t i = 0; i < f.height; ++i) {
            for (size_t j = 0; j < f.width; ++j) {
                Ray& ray = rays[i * f.width + j];
                auto dir = scene->getCamera().rayDir(glm::vec2(f.width, f.height), glm::vec2(j,i));
                dir.y *= -1.0f;
                ray.origin = scene->getCamera().pos();
                ray.dir = dir;
                ray.minTime = 0.001f;
                ray.maxTime = 100000.0f;
            }
        }
        auto rayBuffer = renderGraph.createBufferObjectSC(rays.size() * sizeof(Ray), BufferUsage::STORAGE);
        driver.updateBufferObject(rayBuffer, { reinterpret_cast<uint32_t*>(rays.data()) }, 0);
        auto hitBuffer = renderGraph.createBufferObjectSC(rays.size() * sizeof(RayHit), BufferUsage::TRANSFER_DST | BufferUsage::STORAGE);
        driver.intersectRays(tlas, f.width * f.height, rayBuffer, hitBuffer);
        renderGraph.defineResource(hitBufferRef, hitBuffer);
    });

    renderGraph.addRenderPass("render", { hitBufferRef, sceneBufferRef }, { renderedRef }, [this, &sceneData, renderedRef, sceneBufferRef, hitBufferRef, &renderGraph](FrameGraph& fg) {
        Handle<HwBufferObject> hitBuffer = fg.getResource<Handle<HwBufferObject>>(hitBufferRef);
        Handle<HwUniformBuffer> sceneBuffer = fg.getResource<Handle<HwUniformBuffer>>(sceneBufferRef);
        auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        ColorAttachment att = {};
        att.colors[0] = {
            .handle = color
        };
        att.targetNum = 1;
         
        PipelineState pipe = {};
        pipe.program = this->forwardRTProgram;
        auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att, TextureAttachment{ });
        driver.beginRenderPass(renderTarget, {});
        pipe.bindStorageBuffer(2, 0, sceneData.globalVertexBuffer);
        pipe.bindStorageBuffer(2, 1, hitBuffer);
        pipe.bindUniformBuffer(0, 0, sceneBuffer);
       
        driver.draw(pipe, this->quadPrimitive);
        driver.endRenderPass();
        renderGraph.defineResource(renderedRef, color);

    });

    renderGraph.addRenderPass("blit", { renderedRef }, {}, [this, scene, renderedRef, &renderGraph](FrameGraph& fg) {
        Handle<HwTexture> rendered = fg.getResource<Handle<HwTexture>>(renderedRef);

        PipelineState pipe = {};
        pipe.program = this->blitProgram;
        RenderPassParams params;
        driver.beginRenderPass(this->surfaceRenderTarget, params);
        pipe.bindTexture(1,0, rendered);
        driver.draw(pipe, this->quadPrimitive);
        driver.endRenderPass();
    });
    renderGraph.submit();
    driver.endFrame();
}

void Renderer::ddgiSuite(Scene* scene) {
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver.waitFence(inflight.fence);
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
    }
    Handle<HwInflight> handle = driver.beginFrame(inflight.fence);
    inflight.handle = handle;

    FrameGraph& renderGraph = *inflight.rg;
    renderGraph.reset();

    scene->prepare();
    SceneData& sceneData = scene->getSceneData();
    // 1. Generate random rays at selected probe
    // 2. For each octahedron dirs:
    //  newIrradiance[texDir] = lerp(oldIrradiance[texDir], sum(max(0,dot(texDir,rayDir)*rayRadiance)), hysteresis)
    //  newDistance[probe] = lerp(oldDistance[probe], mean(rayDistance), hysteresis) 
    // 3. Shade based on probes
    //  trilinear interpolation
    //  + weight based on distance
    // vec2 temp = texture(L.meanDistProbeGrid, vec3(octDir, p)).rg;
    // float mean = temp.x;
    // float variance = abs(temp.y - (mean * mean));

    // float t_sub_mean = distToProbe - mean;
    // float chebychev = variance / (variance + (t_sub_mean * t_sub_mean));
    // weight *= ((distToProbe <= mean) ? 1.0 : max(chebychev, 0.0));
    // Avoid zero weight
    //weight = max(0.0002, weight);


    // n rays per probe
    // m probe
    // 16 * 8 * 16
    // 32
    // Prepare probe update rays
    

    // First pass: shadow map for direct lighting (raster)

    // Second pass: probe ray tracing (rt backend)
    
    // Third pass: probe ray shading (compute)

    // Third pass: probe update (compute)

    // Fourth pass: shading (raster)
    

    renderGraph.submit();
    driver.endFrame();
}