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
#include <sunho3d/utils/HalfFloat/umHalf.h>
#include <tiny_gltf.h>

#include "../Entity.h"
#include "../Scene.h"

#include <iostream>
using namespace sunho3d;

Renderer::Renderer(Window* window)
    : window(window), driver(window) {
    for (size_t i = 0; i < MAX_INFLIGHTS; ++i) {
        inflights[i].fence = driver.createFence();
        inflights[i].rg = std::make_unique<FrameGraph>(driver);
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
        .addUniformBuffer(0,1)
        .addTexture(1, 0)
        .addTexture(1, 1)
        .addTexture(1, 2)
        .addTexture(1, 3);
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
    /*
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver.waitFence(inflight.fence);
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
    }
    Handle<HwInflight> handle = driver.beginFrame(inflight.fence);
    inflight.handle = handle;

    inflight.rg.reset();
    inflight.rg = std::make_unique<FrameGraph>(driver);
    FrameGraph& renderGraph = *inflight.rg;
    
    scene->prepare();
    SceneData& sceneData = scene->getSceneData();


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
    */
}

void Renderer::deferSuite(Scene* scene) {
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver.waitFence(inflight.fence);
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
    }
    Handle<HwInflight> handle = driver.beginFrame(inflight.fence);
    inflight.handle = handle;

    inflight.rg.reset();
    inflight.rg = std::make_unique<FrameGraph>(driver);
    FrameGraph& renderGraph = *inflight.rg;
    inflight.instances.clear();

    scene->prepare();
    SceneData& sceneData = scene->getSceneData();

    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }

    auto positionBuffer = renderGraph.importImage("position", gbuffer->positionBuffer);
    auto normalBuffer = renderGraph.importImage("normal", gbuffer->normalBuffer);
    auto diffuseBuffer = renderGraph.importImage("diffuse", gbuffer->diffuseBuffer);

    renderGraph.addFramePass({ 
        .name = "gbuffer generation",
        .resources = { 
            {positionBuffer, ResourceAccessType::ColorWrite},
            {normalBuffer, ResourceAccessType::ColorWrite },
            {diffuseBuffer, ResourceAccessType::ColorWrite },
        },
        .func = [this, scene, &sceneData](FrameGraph& rg, FrameGraphContext& ctx) {
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
                //pipe.bindTexture(1, 0, geom.material->diffuseMap);
                driver.draw(pipe, geom.primitive);
            }
            driver.endRenderPass();
        },
    });

    auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    auto depth = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    RenderAttachments att = {};
    att.colors[0] = color;
    att.depth = depth;
    auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);
    auto renderedBuffer = renderGraph.importImage("rendered", color);

    renderGraph.addFramePass({
        .name = "deferred render",
        .resources = {
            { positionBuffer, ResourceAccessType::FragmentRead },
            { normalBuffer, ResourceAccessType::FragmentRead },
            { diffuseBuffer, ResourceAccessType::FragmentRead },
            { renderedBuffer, ResourceAccessType::ColorWrite }
        },
        .func = [this, scene, &sceneData, renderTarget, positionBuffer, normalBuffer, diffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            auto& camera = scene->getCamera();
            PipelineState pipe = {};
            pipe.program = this->deferredRenderProgram;
            auto lightUB = rg.createUniformBufferSC(sizeof(LightBuffer));
            driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData.lightBuffer }, 0);
            pipe.bindUniformBuffer(0, 0, lightUB);
            ctx.bindTextureResource(pipe, 1, 0, positionBuffer);
            ctx.bindTextureResource(pipe, 1, 1, normalBuffer);
            ctx.bindTextureResource(pipe, 1, 2, diffuseBuffer);
            RenderPassParams params;
            driver.beginRenderPass(renderTarget, params);
            driver.draw(pipe, quadPrimitive);
            driver.endRenderPass();
        },
    });

    renderGraph.addFramePass({
        .name = "blit",
        .resources = {
            { renderedBuffer, ResourceAccessType::FragmentRead },
        },
        .func = [this, scene, &sceneData, renderTarget, renderedBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            PipelineState pipe = {};
            pipe.program = this->blitProgram;
            RenderPassParams params;
            driver.beginRenderPass(this->surfaceRenderTarget, params);
            ctx.bindTextureResource(pipe, 1, 0, renderedBuffer);
            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
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
    /*
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
    driver.endFrame();*/
}

template <class... Ts>
struct overload : Ts... { using Ts::operator()...; };
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

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
    glm::uvec2 probeTexSize = glm::uvec2(IRD_MAP_SIZE * IRD_MAP_PROBE_COLS, IRD_MAP_SIZE * (probeNum / IRD_MAP_PROBE_COLS));
    if (!rayGbuffer) {
        rayGbuffer = std::make_unique<GBuffer>(driver, RAYS_PER_PROBE, probeNum);
        probeTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, probeTexSize.x, probeTexSize.y);
        probeDistTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, IRD_MAP_PROBE_COLS, (probeNum / IRD_MAP_PROBE_COLS));
        probeDistSquareTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, IRD_MAP_PROBE_COLS, (probeNum / IRD_MAP_PROBE_COLS));
        std::vector<half> blank(probeTexSize.x * probeTexSize.y*4, 0.0f);
        driver.updateTexture(probeTexture, { .data = (uint32_t*)blank.data(), .size = 0 });
    }
 
    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }

    inflight.rg.reset();
    inflight.rg = std::make_unique<FrameGraph>(driver);
    FrameGraph& renderGraph = *inflight.rg;

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

    DDGISceneBuffer sb = {};
    auto f = driver.getFrameSize();
    sb.frameSize = glm::uvec2(f.width, f.height);
    sb.instanceNum = sceneData.geometries.size();
    sb.gridNum = scene->ddgi.gridNum;
    sb.sceneSize = scene->ddgi.worldSize;
    std::vector<TextureHandle> diffuseMap);

    for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
        auto& geom = sceneData.geometries[i];
        auto& model = sceneData.worldTransforms[i];
        Instance ginstance = {};
        ginstance.vertexStart = geom.vertexStart;
        ginstance.transform = model;
        InstanceMaterial material;
        std::visit(overload{
            [&](DiffuseColorMaterialData& data) {
                material.typeID = MATERIAL_DIFFUSE_COLOR;
                material.diffuseColor = data.rgb;
            },
            [&](DiffuseTextureMaterialData& data) {
                material.typeID = MATERIAL_DIFFUSE_TEXTURE;
                material.diffuseMapIndex = diffuseMap.size();
                diffuseMap.push_back(data.diffuseMap);
            },
        }, geom.material->materialData);
        sb.instances[i] = ginstance;
        //diffuseMap[i] = geom.material->diffuseMap;
    }

    auto tlas = driver.createTLAS();
    auto sceneBuffer = renderGraph.createUniformBufferSC(sizeof(DDGISceneBuffer));
    driver.updateUniformBuffer(sceneBuffer, { (uint32_t*)&sb }, 0);

    auto rb = renderGraph.createBufferObjectSC(sizeof(Ray) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE);
    auto rayBuffer = renderGraph.importBuffer("ray buffer", rb);

    auto hb = renderGraph.createBufferObjectSC(sizeof(RayHit) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_DST | BufferUsage::STORAGE);
    auto hitBuffer = renderGraph.importBuffer("hit buffer", hb);

    renderGraph.addFramePass({
        .name = "build tlas",
        .resources = {
        },
        .func = [this, scene, &sceneData, &inflight, tlas](FrameGraph& rg, FrameGraphContext& ctx) {
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
            }
            RTSceneDescriptor desc = { inflight.instances.data(), inflight.instances.size() };
            driver.buildTLAS(tlas, desc);
        },
    });

    renderGraph.addFramePass({
        .name = "probe ray gens",
        .resources = {
            { rayBuffer, ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, &sceneData, &inflight, tlas, sceneBuffer, rayBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            glm::uvec3 gridNum = scene->ddgi.gridNum;
            size_t probeNum = gridNum.x * gridNum.y * gridNum.z;
            
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            pipe.bindUniformBuffer(0, 0, sceneBuffer);
            ctx.bindStorageBufferResource(pipe, 1, 0, rayBuffer);
            pipe.copyPushConstants(&constants, sizeof(constants));
            pipe.program = ddgiProbeRayGenProgram;
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });

    renderGraph.addFramePass({
        .name = "intersect",
        .resources = {
            { rayBuffer, ResourceAccessType::ComputeRead },
            { hitBuffer, ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, &sceneData, &inflight, tlas, sceneBuffer, rayBuffer, hitBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            glm::uvec3 gridNum = scene->ddgi.gridNum;
            size_t probeNum = gridNum.x * gridNum.y * gridNum.z;

            driver.intersectRays(tlas, probeNum * RAYS_PER_PROBE, ctx.unwrapBufferHandle(rayBuffer), ctx.unwrapBufferHandle(hitBuffer));
        },
    });

    auto rayGPositionBuffer = renderGraph.importImage("ray g position", rayGbuffer->positionBuffer);
    auto rayGNormalBuffer = renderGraph.importImage("ray g normal", rayGbuffer->normalBuffer);
    auto rayGDiffuseBuffer = renderGraph.importImage("ray g diffuse", rayGbuffer->diffuseBuffer);
    auto db = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, RAYS_PER_PROBE, probeNum);
    auto distanceBuffer = renderGraph.importImage("distance buffer", db);

    renderGraph.addFramePass({
        .name = "ray gbuffer gen",
        .resources = {
            { hitBuffer, ResourceAccessType::ComputeRead },
            { rayGPositionBuffer, ResourceAccessType::ComputeWrite },
            { rayGNormalBuffer, ResourceAccessType::ComputeWrite },
            { rayGDiffuseBuffer, ResourceAccessType::ComputeWrite },
            { distanceBuffer , ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, probeNum , & sceneData, &inflight, tlas, &diffuseMap, sceneBuffer, rayBuffer, hitBuffer, distanceBuffer, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            ctx.bindStorageImageResource(pipe, 2, 0, rayGPositionBuffer);
            ctx.bindStorageImageResource(pipe, 2, 1, rayGNormalBuffer);
            ctx.bindStorageImageResource(pipe, 2, 2, rayGDiffuseBuffer);
            ctx.bindStorageImageResource(pipe, 2, 3, distanceBuffer);
            pipe.bindTextureArray(2, 4, diffuseMap.data(), diffuseMap.size());
            pipe.bindUniformBuffer(0, 0, sceneBuffer);
            pipe.bindStorageBuffer(1, 0, sceneData.globalVertexBuffer);
            ctx.bindStorageBufferResource(pipe, 1, 1, hitBuffer);
            pipe.copyPushConstants(&constants, sizeof(constants));
            pipe.program = ddgiProbeRayShadeProgram;
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });

    auto radianceColor = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, RAYS_PER_PROBE, probeNum);
    RenderAttachments att = {};
    att.colors[0] = radianceColor;
    auto radianceRenderTarget = renderGraph.createRenderTargetSC(RAYS_PER_PROBE, probeNum, att);
    auto randianceBuffer = renderGraph.importImage("radiance buffer", radianceColor);
    auto probeTex = renderGraph.importImage("probe texture", probeTexture);
    auto probeDistTex = renderGraph.importImage("probe dist texture", probeDistTexture);
    auto probeDistSquareTex = renderGraph.importImage("probe dist square texture", probeDistSquareTexture);

    renderGraph.addFramePass({
        .name = "ray gbuffer shade",
        .resources = {
            { rayGPositionBuffer, ResourceAccessType::FragmentRead },
            { rayGNormalBuffer, ResourceAccessType::FragmentRead },
            { rayGDiffuseBuffer, ResourceAccessType::FragmentRead },
            { probeTex, ResourceAccessType::FragmentRead },
            { randianceBuffer , ResourceAccessType::ColorWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, tlas, &diffuseMap, sceneBuffer, radianceRenderTarget, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer, probeTex](FrameGraph& rg, FrameGraphContext& ctx) {
            auto& camera = scene->getCamera();
            PipelineState pipe = {};
            pipe.program = this->ddgiShadeProgram;
            auto lightUB = rg.createUniformBufferSC(sizeof(LightBuffer));
            driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData.lightBuffer }, 0);
            pipe.bindUniformBuffer(0, 0, sceneBuffer);
            pipe.bindUniformBuffer(0, 1, lightUB);
            ctx.bindTextureResource(pipe, 1, 0, rayGPositionBuffer);
            ctx.bindTextureResource(pipe, 1, 1, rayGNormalBuffer);
            ctx.bindTextureResource(pipe, 1, 2, rayGDiffuseBuffer);
            ctx.bindTextureResource(pipe, 1, 3, probeTex);
            RenderPassParams params;
            driver.beginRenderPass(radianceRenderTarget, params);
            driver.draw(pipe, quadPrimitive);
            driver.endRenderPass();
        },
    });


    // Third pass: probe update (compute)
    renderGraph.addFramePass({
        .name = "probe update",
        .resources = {
            { rayBuffer, ResourceAccessType::ComputeRead },
            { randianceBuffer, ResourceAccessType::ComputeRead },
            { distanceBuffer, ResourceAccessType::ComputeRead },
            { probeTex, ResourceAccessType::ComputeWrite },
            { probeDistTex, ResourceAccessType::ComputeWrite },
            { probeDistSquareTex, ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, tlas, &diffuseMap, rayBuffer, randianceBuffer, distanceBuffer, probeTex, probeDistTex, probeDistSquareTex](FrameGraph& rg, FrameGraphContext& ctx) {
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            ctx.bindStorageBufferResource(pipe, 0, 0, rayBuffer);
            ctx.bindTextureResource(pipe, 1, 0, randianceBuffer);
            ctx.bindTextureResource(pipe, 1, 1, distanceBuffer);
            ctx.bindStorageImageResource(pipe, 2, 0, probeTex);
            ctx.bindStorageImageResource(pipe, 2, 1, probeDistTex);
            ctx.bindStorageImageResource(pipe, 2, 2, probeDistSquareTex);
            pipe.program = ddgiProbeUpdateProgram;
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });

    auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    auto depth = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    RenderAttachments att2 = {};
    att2.colors[0] = color;
    att2.depth = depth;
    auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att2);
    auto renderedImage = renderGraph.importImage("rendered", color);
    auto positionBuffer = renderGraph.importImage("g position", gbuffer->positionBuffer);
    auto normalBuffer = renderGraph.importImage("g normal", gbuffer->normalBuffer);
    auto diffuseBuffer = renderGraph.importImage("g diffuse", gbuffer->diffuseBuffer);
    
    renderGraph.addFramePass({
        .name = "gbuffer gen",
        .resources = {
            { positionBuffer, ResourceAccessType::ColorWrite },
            { normalBuffer, ResourceAccessType::ColorWrite },
            { diffuseBuffer, ResourceAccessType::ColorWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, tlas, &diffuseMap, sceneBuffer, rayBuffer, hitBuffer, distanceBuffer, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
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
        },
    });
    
    renderGraph.addFramePass({
        .name = "deferred render",
        .resources = {
            { positionBuffer, ResourceAccessType::FragmentRead },
            { normalBuffer, ResourceAccessType::FragmentRead },
            { diffuseBuffer, ResourceAccessType::FragmentRead },
            { probeTex, ResourceAccessType::FragmentRead },
            { renderedImage, ResourceAccessType::ColorWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, tlas, &diffuseMap, renderTarget, sceneBuffer, probeTex, rayBuffer, hitBuffer, distanceBuffer, positionBuffer, normalBuffer, diffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            
            auto& camera = scene->getCamera();
            PipelineState pipe = {};
            pipe.program = this->ddgiShadeProgram;
            auto lightUB = rg.createUniformBufferSC(sizeof(LightBuffer));
            driver.updateUniformBuffer(lightUB, { .data = (uint32_t*)&sceneData.lightBuffer }, 0);
            pipe.bindUniformBuffer(0, 0, sceneBuffer);
            pipe.bindUniformBuffer(0, 1, lightUB);
            ctx.bindTextureResource(pipe, 1, 0, positionBuffer);
            ctx.bindTextureResource(pipe, 1, 1, normalBuffer);
            ctx.bindTextureResource(pipe, 1, 2, diffuseBuffer);
            ctx.bindTextureResource(pipe, 1, 3, probeTex);
            RenderPassParams params;
            driver.beginRenderPass(renderTarget, params);
            driver.draw(pipe, quadPrimitive);
            driver.endRenderPass();
        },
    });
    
    renderGraph.addFramePass({
        .name = "blit",
        .resources = {
            { renderedImage, ResourceAccessType::FragmentRead },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, tlas, &diffuseMap, renderedImage](FrameGraph& rg, FrameGraphContext& ctx) {
            PipelineState pipe = {};
            pipe.program = this->blitProgram;
            RenderPassParams params;
            driver.beginRenderPass(this->surfaceRenderTarget, params);
            ctx.bindTextureResource(pipe, 1, 0, renderedImage);
            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
    });

    renderGraph.submit();
    driver.endFrame();
}