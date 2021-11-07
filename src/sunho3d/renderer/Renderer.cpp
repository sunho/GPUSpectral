#include "Renderer.h"

#include <sunho3d/renderer/shaders/DisplayTextureFrag.h>
#include <sunho3d/renderer/shaders/DisplayTextureVert.h>
#include <sunho3d/renderer/shaders/ForwardPhongFrag.h>
#include <sunho3d/renderer/shaders/ForwardPhongVert.h>
#include <sunho3d/renderer/shaders/ForwardRTFrag.h>
#include <sunho3d/renderer/shaders/ForwardRTVert.h>
#include <sunho3d/renderer/shaders/DDGIProbeRayGen.h>
#include <sunho3d/renderer/shaders/DDGIProbeRayShade.h>
#include <sunho3d/renderer/shaders/DDGIProbeUpdate.h>
#include <sunho3d/renderer/shaders/DDGIShadeVert.h>
#include <sunho3d/renderer/shaders/DDGIShadeFrag.h>
#include <sunho3d/renderer/shaders/GBufferGenVert.h>
#include <sunho3d/renderer/shaders/GBufferGenFrag.h>
#include <sunho3d/renderer/shaders/DeferredRenderVert.h>
#include <sunho3d/renderer/shaders/DeferredRenderFrag.h>
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
    std::vector<uint32_t> indices = { 0, 1, 2, 3, 4, 5 };
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
    
    registerPrograms();
}

void Renderer::registerPrograms() {
    Program prog = { ForwardPhongVert, ForwardPhongVertSize, ForwardPhongFrag, ForwardPhongFragSize };
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addUniformBuffer(0, 1)
        .addUniformBuffer(0, 2)
        .addTexture(1, 0)
        .addTexture(1, 1);
    fowradPassProgram = driver.createProgram(prog);

    prog = { DisplayTextureVert, DisplayTextureVertSize, DisplayTextureFrag, DisplayTextureFragSize };
    prog.parameterLayout
        .addTexture(1, 0);
    blitProgram = driver.createProgram(prog);

    prog = { ForwardRTVert, ForwardRTVertSize, ForwardRTFrag, ForwardRTFragSize };
    prog.parameterLayout
        .addStorageBuffer(2, 0)
        .addStorageBuffer(2, 1)
        .addUniformBuffer(0, 0);
    forwardRTProgram = driver.createProgram(prog);

    prog = { DDGIProbeRayGen, DDGIProbeRayGenSize };
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addStorageBuffer(1, 0);
    ddgiProbeRayGenProgram = driver.createProgram(prog);

    prog = { DDGIProbeRayShade, DDGIProbeRayShadeSize };
    prog.parameterLayout
        .addStorageImage(2, 0)
        .addStorageImage(2, 1)
        .addStorageImage(2, 2)
        .addStorageImage(2, 3)
        .addTextureArray(2, 4, 32)
        .addStorageBuffer(1, 0)
        .addStorageBuffer(1, 1)
        .addUniformBuffer(0, 0);
    ddgiProbeRayShadeProgram = driver.createProgram(prog);

    prog = { GBufferGenVert, GBufferGenVertSize, GBufferGenFrag, GBufferGenFragSize };
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addTexture(1, 0);

    gbufferGenProgram = driver.createProgram(prog);

    prog = { DeferredRenderVert, DeferredRenderVertSize, DeferredRenderFrag, DeferredRenderFragSize };
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addTexture(1, 0)
        .addTexture(1, 1)
        .addTexture(1, 2);
    deferredRenderProgram = driver.createProgram(prog);

    prog = { DDGIShadeVert, DDGIShadeVertSize, DDGIShadeFrag, DDGIShadeFragSize };
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addTexture(1, 0)
        .addTexture(1, 1)
        .addTexture(1, 2);
    ddgiShadeProgram = driver.createProgram(prog);

    prog = { DDGIProbeUpdate, DDGIProbeUpdateSize };
    prog.parameterLayout
        .addStorageBuffer(0, 0)
        .addTexture(1, 0)
        .addTexture(1, 1)
        .addStorageImage(2, 0)
        .addStorageImage(2, 1)
        .addStorageImage(2, 2);
    ddgiProbeUpdateProgram = driver.createProgram(prog);
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
        RenderAttachments att = {};
        att.colors[0] = color;
        att.depth = depth;
        auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);
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
        RenderAttachments att = {};
        att.colors[0] = color;
        att.depth = depth;
        auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);
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

void Renderer::deferSuite(Scene* scene) {
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

    auto positionRef = renderGraph.declareResource<Handle<HwTexture>>("Position texture");
    auto normalRef = renderGraph.declareResource<Handle<HwTexture>>("Normal texture");
    auto diffuseRef = renderGraph.declareResource<Handle<HwTexture>>("Diffuse texture");
    auto renderedRef = renderGraph.declareResource<Handle<HwTexture>>("Rendered texture");
    
    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }

    renderGraph.addRenderPass("gbuffer generation", {}, { positionRef, normalRef, diffuseRef }, [this, scene, &sceneData, positionRef, normalRef, diffuseRef](FrameGraph& rg) {
        auto& camera = scene->getCamera();

        PipelineState pipe = {};
        pipe.program = this->gbufferGenProgram;
        pipe.depthTest.enabled = 1;
        pipe.depthTest.write = 1;
        pipe.depthTest.compareOp = CompareOp::LESS;
        RenderPassParams params;
        driver.beginRenderPass(gbuffer->renderTarget, params);
        for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
            auto& geom = sceneData.geometries[i];
            auto& model = sceneData.worldTransforms[i];
            pipe.bindUniformBuffer(0, 0, this->createTransformBuffer(rg, camera, model));
            pipe.bindTexture(1, 0, geom.material->diffuseMap);
            driver.draw(pipe, geom.primitive);
        }
        driver.endRenderPass();
        rg.defineResource<Handle<HwTexture>>(positionRef, gbuffer->positionBuffer);
        rg.defineResource<Handle<HwTexture>>(normalRef, gbuffer->normalBuffer);
        rg.defineResource<Handle<HwTexture>>(diffuseRef, gbuffer->diffuseBuffer);
    });
    renderGraph.addRenderPass("deferred render", { positionRef, normalRef, diffuseRef }, { renderedRef }, [this, scene, renderedRef , & sceneData, positionRef, normalRef, diffuseRef](FrameGraph& rg) {
        auto color = rg.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        auto depth = rg.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        RenderAttachments att = {};
        att.colors[0] = color;
        att.depth = depth;
        auto renderTarget = rg.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);
        
        auto positionBuffer = rg.getResource<Handle<HwTexture>>(positionRef);
        auto normalBuffer = rg.getResource<Handle<HwTexture>>(normalRef);
        auto diffuseBuffer = rg.getResource<Handle<HwTexture>>(diffuseRef);
        auto& camera = scene->getCamera();
        PipelineState pipe = {};
        pipe.program = this->deferredRenderProgram;
        auto lightUB = rg.createUniformBufferSC(sizeof(LightBuffer));
        driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData.lightBuffer }, 0);
        pipe.bindUniformBuffer(0, 0, lightUB);
        pipe.bindTexture(1, 0, positionBuffer);
        pipe.bindTexture(1, 1, normalBuffer);
        pipe.bindTexture(1, 2, diffuseBuffer);
        RenderPassParams params;
        driver.beginRenderPass(renderTarget, params);
        driver.draw(pipe, quadPrimitive);
        driver.endRenderPass();
        rg.defineResource<Handle<HwTexture>>(renderedRef, color);
    });
    renderGraph.addRenderPass("blit", { renderedRef }, {}, [this, scene, renderedRef, &renderGraph](FrameGraph& fg) {
        Handle<HwTexture> rendered = fg.getResource<Handle<HwTexture>>(renderedRef);

        PipelineState pipe = {};
        pipe.program = this->blitProgram;
        RenderPassParams params;
        driver.beginRenderPass(this->surfaceRenderTarget, params);
        pipe.bindTexture(1, 0, rendered);
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
    ddgiSuite(scene);
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
        RenderAttachments att = {};
        att.colors[0] = color;
         
        PipelineState pipe = {};
        pipe.program = this->forwardRTProgram;
        auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);
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
    glm::uvec3 gridNum = scene->ddgi.gridNum;
    size_t probeNum = gridNum.x * gridNum.y * gridNum.z;
        
    if (!rayGbuffer) {
        rayGbuffer = std::make_unique<GBuffer>(driver, RAYS_PER_PROBE, probeNum);
        probeTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, IRD_MAP_SIZE * IRD_MAP_PROBE_COLS, IRD_MAP_SIZE * (probeNum / IRD_MAP_PROBE_COLS));
        probeDistTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, IRD_MAP_PROBE_COLS, (probeNum / IRD_MAP_PROBE_COLS));
        probeDistSquareTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, IRD_MAP_PROBE_COLS, (probeNum / IRD_MAP_PROBE_COLS));
    }

    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }


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

    auto tlasRef = renderGraph.declareResource<Handle<HwTLAS>>("TLAS");
    auto rayBufferRef = renderGraph.declareResource<Handle<HwBufferObject>>("Ray Buffer");
    auto hitBufferRef = renderGraph.declareResource<Handle<HwBufferObject>>("Hit Buffer");
    auto sceneBufferRef = renderGraph.declareResource<Handle<HwUniformBuffer>>("RT Scene buffer");
    auto diffuseMapRef = renderGraph.declareResource<FixedVector<Handle<HwTexture>> >("Diffuse map");
    auto distanceBufferRef = renderGraph.declareResource<Handle<HwTexture>>("Distance buffer");
    auto positionBufferRef = renderGraph.declareResource<Handle<HwTexture>>("Position buffer");
    auto radianceBufferRef = renderGraph.declareResource<Handle<HwTexture>>("Radiance buffer");
    renderGraph.addRenderPass("build tlas and scene buffer", {}, { tlasRef, sceneBufferRef, diffuseMapRef }, [this, diffuseMapRef, sceneBufferRef, tlasRef, &inflight, scene, &sceneData](FrameGraph& rg) {
        DDGISceneBuffer sceneBuffer = {};
        auto f = driver.getFrameSize();
        sceneBuffer.frameSize = glm::uvec2(f.width, f.height);
        sceneBuffer.instanceNum = sceneData.geometries.size();
        sceneBuffer.gridNum = scene->ddgi.gridNum;
        sceneBuffer.sceneSize = scene->ddgi.worldSize;
        FixedVector<TextureHandle> diffuseMap(sceneBuffer.instanceNum);

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
            ginstance.diffuseMapIndex = i;
            sceneBuffer.instances[i] = ginstance;
            diffuseMap[i] = geom.material->diffuseMap;
        }
        RTSceneDescriptor desc = { inflight.instances.data(), inflight.instances.size() };
        auto tlas = driver.createTLAS();
        driver.buildTLAS(tlas, desc);
        rg.defineResource(tlasRef, tlas);

        auto sb = rg.createUniformBufferSC(sizeof(DDGISceneBuffer));
        driver.updateUniformBuffer(sb, { (uint32_t*)&sceneBuffer }, 0);
        rg.defineResource(sceneBufferRef, sb);
        rg.defineResource(diffuseMapRef, diffuseMap);
    });
    // First pass: shadow map for direct lighting (raster)

    // Second pass: probe ray tracing (rt backend)
    renderGraph.addRenderPass("probe ray gens", { sceneBufferRef }, { rayBufferRef }, [this, rayBufferRef, sceneBufferRef, scene, &sceneData](FrameGraph& rg) {
        Handle<HwUniformBuffer> sceneBuffer = rg.getResource<Handle<HwUniformBuffer>>(sceneBufferRef);
        glm::uvec3 gridNum = scene->ddgi.gridNum;
        size_t probeNum = gridNum.x * gridNum.y * gridNum.z;
        auto rayBuffer = rg.createBufferObjectSC(sizeof(Ray) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE);
        
        DDGIPushConstants constants = {};
        constants.globalRngState = rand();
        PipelineState pipe = {};
        pipe.bindUniformBuffer(0, 0, sceneBuffer);
        pipe.bindStorageBuffer(1, 0, rayBuffer);
        pipe.copyPushConstants(&constants, sizeof(constants));
        pipe.program = ddgiProbeRayGenProgram;
        driver.dispatch(pipe, probeNum, 1, 1);
        rg.defineResource(rayBufferRef, rayBuffer);
    });
    renderGraph.addRenderPass("intersect", { tlasRef, rayBufferRef }, { hitBufferRef }, [this, rayBufferRef, hitBufferRef, tlasRef, scene, &sceneData, &renderGraph](FrameGraph& rg) {
        Handle<HwTLAS> tlas = rg.getResource<Handle<HwTLAS>>(tlasRef);
        Handle<HwBufferObject> rayBuffer = rg.getResource<Handle<HwBufferObject>>(rayBufferRef);
        glm::uvec3 gridNum = scene->ddgi.gridNum;
        size_t probeNum = gridNum.x * gridNum.y * gridNum.z;
       
        auto hitBuffer = renderGraph.createBufferObjectSC(sizeof(RayHit) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_DST | BufferUsage::STORAGE);
        driver.intersectRays(tlas, probeNum * RAYS_PER_PROBE, rayBuffer, hitBuffer);
        renderGraph.defineResource(hitBufferRef, hitBuffer);
    });
    // Third pass: probe ray shading (compute)
    renderGraph.addRenderPass("probe ray gbuffer gen", { hitBufferRef }, { positionBufferRef, distanceBufferRef }, [this, positionBufferRef, distanceBufferRef, hitBufferRef, rayBufferRef, diffuseMapRef, sceneBufferRef, probeNum, scene, &sceneData](FrameGraph& rg) {
        Handle<HwTexture> distanceBuffer = rg.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, RAYS_PER_PROBE, probeNum);
        Handle<HwUniformBuffer> sceneBuffer = rg.getResource<Handle<HwUniformBuffer>>(sceneBufferRef);
        FixedVector<TextureHandle> diffuseMap = rg.getResource<FixedVector<TextureHandle>>(diffuseMapRef);
        Handle<HwBufferObject> hitBuffer = rg.getResource<Handle<HwBufferObject>>(hitBufferRef);
        
        DDGIPushConstants constants = {};
        constants.globalRngState = rand();
        PipelineState pipe = {};
        pipe.bindStorageImage(2, 0, rayGbuffer->positionBuffer);
        pipe.bindStorageImage(2, 1, rayGbuffer->normalBuffer);
        pipe.bindStorageImage(2, 2, rayGbuffer->diffuseBuffer);
        pipe.bindStorageImage(2, 3, distanceBuffer);
        pipe.bindTextureArray(2, 4, diffuseMap.data(), sceneData.geometries.size());
        pipe.bindUniformBuffer(0, 0, sceneBuffer);
        pipe.bindStorageBuffer(1, 0, sceneData.globalVertexBuffer);
        pipe.bindStorageBuffer(1, 1, hitBuffer);
        pipe.copyPushConstants(&constants, sizeof(constants));
        pipe.program = ddgiProbeRayShadeProgram;
        driver.dispatch(pipe, probeNum, 1, 1);

        rg.defineResource<Handle<HwTexture>>(positionBufferRef, rayGbuffer->positionBuffer);
        rg.defineResource<Handle<HwTexture>>(distanceBufferRef, distanceBuffer);
    });
    renderGraph.addRenderPass("probe ray shade", { positionBufferRef }, { radianceBufferRef }, [this, hitBufferRef, radianceBufferRef, rayBufferRef, diffuseMapRef, sceneBufferRef, probeNum, scene, &sceneData](FrameGraph& rg) {
        auto color = rg.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, RAYS_PER_PROBE, probeNum);
        RenderAttachments att = {};
        att.colors[0] = color;
        auto renderTarget = rg.createRenderTargetSC(RAYS_PER_PROBE, probeNum, att);

        auto positionBuffer = rayGbuffer->positionBuffer;
        auto normalBuffer = rayGbuffer->normalBuffer;
        auto diffuseBuffer = rayGbuffer->diffuseBuffer;
        auto& camera = scene->getCamera();
        PipelineState pipe = {};
        pipe.program = this->ddgiShadeProgram;
        auto lightUB = rg.createUniformBufferSC(sizeof(LightBuffer));
        driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData.lightBuffer }, 0);
        pipe.bindUniformBuffer(0, 0, lightUB);
        pipe.bindTexture(1, 0, positionBuffer);
        pipe.bindTexture(1, 1, normalBuffer);
        pipe.bindTexture(1, 2, diffuseBuffer);
        pipe.bindTexture(1, 3, probeTexture);
        RenderPassParams params;
        driver.beginRenderPass(renderTarget, params);
        driver.draw(pipe, quadPrimitive);
        driver.endRenderPass();
        rg.defineResource<Handle<HwTexture>>(radianceBufferRef, color);
    });

    auto probeTextureRef = renderGraph.declareResource<Handle<HwTexture>>("Probe texture");
    // Third pass: probe update (compute)
    renderGraph.addRenderPass("probe update", { rayBufferRef, radianceBufferRef, distanceBufferRef }, { probeTextureRef }, [this, probeTextureRef, radianceBufferRef, positionBufferRef, distanceBufferRef, hitBufferRef, rayBufferRef, diffuseMapRef, sceneBufferRef, probeNum, scene, &sceneData](FrameGraph& rg) {
        Handle<HwBufferObject> rayBuffer = rg.getResource<Handle<HwBufferObject>>(rayBufferRef);
        Handle<HwTexture> radianceBuffer = rg.getResource<Handle<HwTexture>>(radianceBufferRef);
        Handle<HwTexture> distanceBuffer = rg.getResource<Handle<HwTexture>>(distanceBufferRef);

        DDGIPushConstants constants = {};
        constants.globalRngState = rand();
        PipelineState pipe = {};
        pipe.bindStorageBuffer(0, 0, rayBuffer);
        pipe.bindTexture(1, 0, radianceBuffer);
        pipe.bindTexture(1, 1, distanceBuffer);
        pipe.bindStorageImage(2, 0, probeTexture);
        pipe.bindStorageImage(2, 1, probeDistTexture);
        pipe.bindStorageImage(2, 2, probeDistSquareTexture);
        pipe.program = ddgiProbeUpdateProgram;
        driver.dispatch(pipe, probeNum, IRD_MAP_SIZE, IRD_MAP_SIZE);
        rg.defineResource<Handle<HwTexture>>(probeTextureRef, probeTexture);
    });
    // Fourth pass: shading (raster)

    auto positionRef = renderGraph.declareResource<Handle<HwTexture>>("Position texture");
    auto normalRef = renderGraph.declareResource<Handle<HwTexture>>("Normal texture");
    auto diffuseRef = renderGraph.declareResource<Handle<HwTexture>>("Diffuse texture");
    auto renderedRef = renderGraph.declareResource<Handle<HwTexture>>("Rendered texture");

    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }

    renderGraph.addRenderPass("gbuffer generation", {}, { positionRef, normalRef, diffuseRef }, [this, scene, &sceneData, positionRef, normalRef, diffuseRef](FrameGraph& rg) {
        auto& camera = scene->getCamera();

        PipelineState pipe = {};
        pipe.program = this->gbufferGenProgram;
        pipe.depthTest.enabled = 1;
        pipe.depthTest.write = 1;
        pipe.depthTest.compareOp = CompareOp::LESS;
        RenderPassParams params;
        driver.beginRenderPass(gbuffer->renderTarget, params);
        for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
            auto& geom = sceneData.geometries[i];
            auto& model = sceneData.worldTransforms[i];
            pipe.bindUniformBuffer(0, 0, this->createTransformBuffer(rg, camera, model));
            pipe.bindTexture(1, 0, geom.material->diffuseMap);
            driver.draw(pipe, geom.primitive);
        }
        driver.endRenderPass();
        rg.defineResource<Handle<HwTexture>>(positionRef, gbuffer->positionBuffer);
        rg.defineResource<Handle<HwTexture>>(normalRef, gbuffer->normalBuffer);
        rg.defineResource<Handle<HwTexture>>(diffuseRef, gbuffer->diffuseBuffer);
    });
    renderGraph.addRenderPass("deferred render", { probeTextureRef, positionRef, normalRef, diffuseRef }, { renderedRef }, [this, scene, renderedRef, probeTextureRef , & sceneData, positionRef, normalRef, diffuseRef](FrameGraph& rg) {
        auto color = rg.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        auto depth = rg.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
        RenderAttachments att = {};
        att.colors[0] = color;
        att.depth = depth;
        auto renderTarget = rg.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);
        auto positionBuffer = rg.getResource<Handle<HwTexture>>(positionRef);
        auto normalBuffer = rg.getResource<Handle<HwTexture>>(normalRef);
        auto diffuseBuffer = rg.getResource<Handle<HwTexture>>(diffuseRef);
        auto probeIrradianceMap = rg.getResource<Handle<HwTexture>>(probeTextureRef);
        auto& camera = scene->getCamera();
        PipelineState pipe = {};
        pipe.program = this->deferredRenderProgram;
        auto lightUB = rg.createUniformBufferSC(sizeof(LightBuffer));
        driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData.lightBuffer }, 0);
        pipe.bindUniformBuffer(0, 0, lightUB);
        pipe.bindTexture(1, 0, positionBuffer);
        pipe.bindTexture(1, 1, normalBuffer);
        pipe.bindTexture(1, 2, diffuseBuffer);
        pipe.bindTexture(1, 3, probeIrradianceMap);
        RenderPassParams params;
        driver.beginRenderPass(renderTarget, params);
        driver.draw(pipe, quadPrimitive);
        driver.endRenderPass();
        rg.defineResource<Handle<HwTexture>>(renderedRef, color);
    });
    renderGraph.addRenderPass("blit", { renderedRef }, {}, [this, scene, renderedRef, &renderGraph](FrameGraph& fg) {
        Handle<HwTexture> rendered = fg.getResource<Handle<HwTexture>>(renderedRef);

        PipelineState pipe = {};
        pipe.program = this->blitProgram;
        RenderPassParams params;
        driver.beginRenderPass(this->surfaceRenderTarget, params);
        pipe.bindTexture(1, 0, rendered);
        driver.draw(pipe, this->quadPrimitive);
        driver.endRenderPass();
    });

    renderGraph.submit();
    driver.endFrame();
}