#include "Renderer.h"

#include <sunho3d/framegraph/shaders/DisplayTextureFrag.h>
#include <sunho3d/framegraph/shaders/DisplayTextureVert.h>
#include <sunho3d/framegraph/shaders/ForwardPhongFrag.h>
#include <sunho3d/framegraph/shaders/ForwardPhongVert.h>
#include <sunho3d/framegraph/shaders/ForwardRTFrag.h>
#include <sunho3d/framegraph/shaders/ForwardRTVert.h>
#include <tiny_gltf.h>

#include "Entity.h"
#include "Scene.h"

#include <iostream>
using namespace sunho3d;

RenderGraph::RenderGraph(Renderer& renderer)
    : parent(renderer), driver(renderer.getDriver()) {
}

RenderGraph::~RenderGraph() {
    for (auto d : destroyers) {
        d();
    }
}
void RenderGraph::reset() {
    for (auto d : destroyers) {
        d();
    }
    destroyers.clear();
    fg = FrameGraph();
}

void RenderGraph::addRenderPass(const std::string& name, std::vector<FGResource> inputs, std::vector<FGResource> outputs, RenderPass::RenderPassFunc func) {
    fg.addRenderPass(RenderPass(name, inputs, outputs, func));
}

void RenderGraph::submit() {
    fg.compile();
    fg.run();
}

Renderer::Renderer(Window* window)
    : window(window), driver(window) {
    for (size_t i = 0; i < MAX_INFLIGHTS; ++i) {
        inflights[i].fence = driver.createFence();
        inflights[i].rg = std::make_unique<RenderGraph>(*this);
    }

    Program prog(ForwardPhongVert, ForwardPhongVertSize, ForwardPhongFrag,ForwardPhongFragSize);
    fowradPassProgram = driver.createProgram(prog);

    Program prog2(DisplayTextureVert, DisplayTextureVertSize, DisplayTextureFrag, DisplayTextureFragSize);
    blitProgram = driver.createProgram(prog2);

    Program prog3(ForwardRTVert, ForwardRTVertSize, ForwardRTFrag, ForwardRTFragSize);
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

    RenderGraph& renderGraph = *inflight.rg;
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
            driver.bindUniformBuffer(0, this->createTransformBuffer(renderGraph, camera, model));
            driver.bindUniformBuffer(1, lightUB);
            driver.bindUniformBuffer(2, getOrCreateMaterialBuffer(geom.material));
            driver.bindTexture(0, geom.material->diffuseMap);
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
            driver.bindUniformBuffer(0, this->createTransformBuffer(renderGraph, scene->getCamera(), model));
            driver.bindUniformBuffer(1, lightUB);
            driver.bindUniformBuffer(2, getOrCreateMaterialBuffer(geom.material));
            driver.bindTexture(0, geom.material->diffuseMap);
            driver.bindTexture(1, shadowMap);
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
        driver.bindTexture(0, forwardMap);
        driver.draw(pipe, this->quadPrimitive);
        driver.endRenderPass();
    });
    renderGraph.submit();
    driver.endFrame();
}

Handle<HwUniformBuffer> Renderer::createTransformBuffer(RenderGraph& rg, const Camera& camera, const glm::mat4& model) {
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

    RenderGraph& renderGraph = *inflight.rg;
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
            ginstance.transform = t;
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
        auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att, TextureAttachment{ });
        driver.beginRenderPass(renderTarget, {});
        //driver.bindStorageBuffer(0, sceneData.globalVertexBuffer);
        driver.bindStorageBuffer(1, hitBuffer);
        driver.bindUniformBuffer(0, sceneBuffer);
        
        PipelineState pipe = {};
        pipe.program = this->forwardRTProgram;
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
        driver.bindTexture(0, rendered);
        driver.draw(pipe, this->quadPrimitive);
        driver.endRenderPass();
    });
    renderGraph.submit();
    driver.endFrame();
}
