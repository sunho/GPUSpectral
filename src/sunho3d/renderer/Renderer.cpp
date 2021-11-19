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
#include <sunho3d/renderer/shaders/PointShadowGenVert.h>
#include <sunho3d/renderer/shaders/PointShadowGenFrag.h>
#include <sunho3d/utils/HalfFloat/umHalf.h>
#include <tiny_gltf.h>

#include "../Entity.h"
#include "../Scene.h"

#include <iostream>
using namespace sunho3d;

template <class... Ts>
struct overload : Ts... { using Ts::operator()...; };
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

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
    driver.updateBufferObjectSync(buffer0, { .data = (uint32_t*)v.data() }, 0);
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
        .addStorageBufferArray(1, 0, 32)
        .addStorageBufferArray(1, 1, 32)
        .addStorageBufferArray(1, 2, 32)
        .addStorageBuffer(1,3)
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
        .addStorageImage(2, 4)
        .addTextureArray(2, 5, 32)
        .addStorageBufferArray(1,0, MAX_MESH_COUNT)
        .addStorageBufferArray(1, 1, MAX_MESH_COUNT)
        .addStorageBufferArray(1, 2, MAX_MESH_COUNT)
        .addStorageBuffer(1, 3)
        .addUniformBuffer(0, 0);
    ddgiProbeRayShadeProgram = driver.createProgram(prog);

    prog = { PointShadowGenVert, PointShadowGenVertSize, PointShadowGenFrag, PointShadowGenFragSize };
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addUniformBuffer(0, 1);
    pointShadowGenProgram = driver.createProgram(prog);

    prog = { GBufferGenVert, GBufferGenVertSize, GBufferGenFrag, GBufferGenFragSize };
    prog.parameterLayout
        .addUniformBuffer(0, 0)
        .addUniformBuffer(0,1)
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
        .addTexture(1, 3)
        .addTexture(1, 4)
        .addTextureArray(1, 5, MAX_LIGHTS);
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
    if (inflight.tlas) {
        driver.destroyTLAS(inflight.tlas);
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
            {{positionBuffer}, ResourceAccessType::ColorWrite},
            {{normalBuffer}, ResourceAccessType::ColorWrite },
            {{diffuseBuffer}, ResourceAccessType::ColorWrite },
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
                auto& modelInvT = sceneData.worldTransformsInvT[i];
                pipe.bindUniformBuffer(0, 0, this->createTransformBuffer(rg, camera, model, modelInvT));
                //pipe.bindTexture(1, 0, geom.material->diffuseMap);
                driver.draw(pipe, geom.primitive.hwInstance);
            }
            driver.endRenderPass();
        },
    });

    auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, 1);
    auto depth = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, 1);
    RenderAttachments att = {};
    att.colors[0] = color;
    att.depth = depth;
    auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);
    auto renderedBuffer = renderGraph.importImage("rendered", color);

    renderGraph.addFramePass({
        .name = "deferred render",
        .resources = {
            { {positionBuffer}, ResourceAccessType::FragmentRead },
            { {normalBuffer}, ResourceAccessType::FragmentRead },
            { {diffuseBuffer}, ResourceAccessType::FragmentRead },
            { {renderedBuffer}, ResourceAccessType::ColorWrite }
        },
        .func = [this, scene, &sceneData, renderTarget, positionBuffer, normalBuffer, diffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            auto& camera = scene->getCamera();
            PipelineState pipe = {};
            pipe.program = this->deferredRenderProgram;
            auto lightUB = rg.createTempUniformBuffer((void*)&sceneData.lightBuffer, sizeof(LightBuffer));
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
            { {renderedBuffer}, ResourceAccessType::FragmentRead },
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

Handle<HwBufferObject> Renderer::createTransformBuffer(FrameGraph& rg, const Camera& camera, const glm::mat4& model, const glm::mat4& modelInvT) {
    TransformBuffer transformBuffer;
    transformBuffer.MVP = camera.proj * camera.view * model;
    transformBuffer.invModelT = modelInvT;
    transformBuffer.model = model;
    transformBuffer.cameraPos = glm::vec4(camera.pos(), 1.0f);
    return rg.createTempUniformBuffer((void*)&transformBuffer, sizeof(TransformBuffer));
}

void Renderer::run(Scene* scene) {
    FrameMarkStart("Frame")
    ddgiSuite(scene);
    FrameMarkEnd("Frame")
    ++currentFrame;
}

void Renderer::rtSuite(Scene* scene) {
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver.waitFence(inflight.fence);
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
    }
    if (inflight.tlas) {
        driver.destroyTLAS(inflight.tlas);
    }
    Handle<HwInflight> handle = driver.beginFrame(inflight.fence);
    inflight.handle = handle;
    inflight.rg.reset();
    inflight.rg = std::make_unique<FrameGraph>(driver);
    FrameGraph& renderGraph = *inflight.rg;

    inflight.instances.clear();
    scene->prepare();
    SceneData& sceneData = scene->getSceneData();

    ForwardRTSceneBuffer sb = {};
    auto f = driver.getFrameSize();
    sb.frameSize = glm::uvec2(f.width, f.height);
    sb.instanceNum = sceneData.geometries.size();
    std::vector<TextureHandle> diffuseMap;
    std::vector<InstanceMaterial> materials;
    std::vector<Handle<HwBufferObject>> materialBuffers;
    std::vector<Handle<HwBufferObject>> transformBuffers;
    std::vector<Handle<HwBufferObject>> vertexPositionBuffers;
    std::vector<Handle<HwBufferObject>> vertexNormalBuffers;
    std::vector<Handle<HwBufferObject>> vertexUVBuffers;
    std::unordered_map<uint32_t, uint32_t> primitiveIdToVB;
    {
        ZoneScopedN("Build scene buffer") auto f = driver.getFrameSize();
        sb.frameSize = glm::uvec2(f.width, f.height);
        sb.instanceNum = sceneData.geometries.size();
        auto& camera = scene->getCamera();
        glm::mat4 vp = camera.proj * camera.view;
        for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
            auto& geom = sceneData.geometries[i];
            auto& model = sceneData.worldTransforms[i];
            auto& modelInvT = sceneData.worldTransformsInvT[i];

            if (primitiveIdToVB.find(geom.primitive.hwInstance.getId()) == primitiveIdToVB.end()) {
                primitiveIdToVB.emplace(geom.primitive.hwInstance.getId(), vertexPositionBuffers.size());
                vertexPositionBuffers.push_back(geom.primitive.positionBuffer);
                vertexNormalBuffers.push_back(geom.primitive.normalBuffer);
                vertexUVBuffers.push_back(geom.primitive.uvBuffer);
            }
            Instance ginstance = {};
            ginstance.meshIndex = primitiveIdToVB[geom.primitive.hwInstance.getId()];
            ginstance.transform = model;
            InstanceMaterial material = {};
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
                           [&](EmissionMaterialData& data) {
                               material.typeID = MATERIAL_EMISSION;
                               material.diffuseColor = data.radiance;
                           },
                           [](auto k) {} },
                       geom.material->materialData);
            materials.push_back(material);
            auto materialBuffer = renderGraph.createTempUniformBuffer((void*)&material, sizeof(InstanceMaterial));
            materialBuffers.push_back(materialBuffer);
            ginstance.material = material;
            sb.instances[i] = ginstance;

            TransformBuffer transformBuffer;
            transformBuffer.MVP = vp * model;
            transformBuffer.invModelT = modelInvT;
            transformBuffer.model = model;
            transformBuffer.cameraPos = glm::vec4(camera.pos(), 1.0f);
            auto tb = renderGraph.createTempUniformBuffer((void*)&transformBuffer, sizeof(TransformBuffer));
            transformBuffers.push_back(tb);
            //diffuseMap[i] = geom.material->diffuseMap;
        }
    }
    

    auto tlas = driver.createTLAS();
    auto sceneBuffer = renderGraph.createTempUniformBuffer((void*)&sb, sizeof(ForwardRTSceneBuffer));

    renderGraph.addFramePass({
        .name = "build tlas",
        .resources = {},
        .func = [this, scene, &sceneData, &inflight, tlas](FrameGraph& rg, FrameGraphContext& ctx) {
            for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
                auto& geom = sceneData.geometries[i];
                auto& model = sceneData.worldTransforms[i];
                Handle<HwBLAS> blas = driver.createBLAS();
                auto it = this->blasMap.find(geom.primitive.hwInstance.getId());
                if (it == this->blasMap.end()) {
                    driver.buildBLAS(blas, geom.primitive.hwInstance);
                    this->blasMap.emplace(geom.primitive.hwInstance.getId(), blas);
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

    auto hb = renderGraph.createBufferObjectSC(f.width * f.height * sizeof(RayHit), BufferUsage::TRANSFER_DST | BufferUsage::STORAGE);
    auto hitBuffer = renderGraph.importBuffer("hit buffer", hb);

    renderGraph.addFramePass({
        .name = "intersect",
        .resources = {
            { {hitBuffer}, ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, &sceneData, &inflight, tlas, sceneBuffer, hitBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            std::vector<Ray> rays;
            auto f = driver.getFrameSize();
            rays.resize(f.width * f.height);
            for (size_t i = 0; i < f.height; ++i) {
                for (size_t j = 0; j < f.width; ++j) {
                    Ray& ray = rays[i * f.width + j];
                    auto dir = scene->getCamera().rayDir(glm::vec2(f.width, f.height), glm::vec2(j, i));
                    dir.y *= -1.0f;
                    ray.origin = scene->getCamera().pos();
                    ray.dir = dir;
                    ray.minTime = 0.001f;
                    ray.maxTime = 100000.0f;
                }
            }
            auto rayBuffer = rg.createBufferObjectSC(rays.size() * sizeof(Ray), BufferUsage::STORAGE);
            driver.updateBufferObjectSync(rayBuffer, { reinterpret_cast<uint32_t*>(rays.data()) }, 0); // TODO
            driver.intersectRays(tlas, f.width * f.height, rayBuffer, ctx.unwrapBufferHandle(hitBuffer));
        },
    });

    auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, 1);
    RenderAttachments att = {};
    att.colors[0] = color;
    auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);  
    auto renderedImage = renderGraph.importImage("rendered", color);
    
    renderGraph.addFramePass({
        .name = "render",
        .resources = {
            { {hitBuffer}, ResourceAccessType::ComputeRead },
        },
        .func = [this, scene, &sceneData, &inflight, renderTarget, tlas, 
&vertexPositionBuffers, &vertexNormalBuffers, &vertexUVBuffers, & diffuseMap, sceneBuffer, hitBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            DDGIPushConstants constants = {};

            PipelineState pipe = {};
            pipe.program = this->forwardRTProgram;
            driver.beginRenderPass(renderTarget, {});
            pipe.bindStorageBufferArray(1, 0, vertexPositionBuffers.data(), vertexPositionBuffers.size());
            pipe.bindStorageBufferArray(1, 1, vertexNormalBuffers.data(), vertexNormalBuffers.size());
            pipe.bindStorageBufferArray(1, 2, vertexUVBuffers.data(), vertexUVBuffers.size());
            ctx.bindStorageBufferResource(pipe, 1, 3, hitBuffer);
            pipe.bindUniformBuffer(0, 0, sceneBuffer);

            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
    });

    renderGraph.addFramePass({
        .name = "blit",
        .resources = {
            { {renderedImage}, ResourceAccessType::FragmentRead },
        },
        .func = [this, scene, &sceneData, &inflight, tlas, &diffuseMap, renderedImage](FrameGraph& rg, FrameGraphContext& ctx) {
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


void Renderer::ddgiSuite(Scene* scene) {
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    {
        ZoneScopedN("Wait for fence")
        driver.waitFence(inflight.fence);
    }
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
    }
    if (inflight.tlas) {
        driver.destroyTLAS(inflight.tlas);
    }
    Handle<HwInflight> handle;
    {
        ZoneScopedN("Begin frame") 
        handle = driver.beginFrame(inflight.fence);
    }
    inflight.instances.clear();
    inflight.handle = handle;
    glm::uvec3 gridNum = scene->ddgi.gridNum;
    size_t probeNum = gridNum.x * gridNum.y * gridNum.z;
    glm::uvec2 probeTexSize = glm::uvec2(IRD_MAP_SIZE * IRD_MAP_PROBE_COLS, IRD_MAP_SIZE * (probeNum / IRD_MAP_PROBE_COLS));
    if (!rayGbuffer) {
        rayGbuffer = std::make_unique<GBuffer>(driver, RAYS_PER_PROBE, probeNum);
        probeTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, 1, probeTexSize.x, probeTexSize.y, 1);
        probeDistTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, 1, IRD_MAP_PROBE_COLS, (probeNum / IRD_MAP_PROBE_COLS), 1);
        probeDistSquareTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, 1, IRD_MAP_PROBE_COLS, (probeNum / IRD_MAP_PROBE_COLS), 1);
        std::vector<half> blank(probeTexSize.x * probeTexSize.y*4, 0.0f);
        driver.copyTextureInitialData(probeTexture, { .data = (uint32_t*)blank.data(), .size = 0 });
    }
 
    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }

    {
        ZoneScopedN("Reset render graph")
        inflight.rg.reset();
        inflight.rg = std::make_unique<FrameGraph>(driver);
    }
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
    std::vector<TextureHandle> diffuseMap;
    std::vector<InstanceMaterial> materials;
    std::vector<Handle<HwBufferObject>> materialBuffers;
    std::vector<Handle<HwBufferObject>> transformBuffers;
    std::vector<Handle<HwBufferObject>> vertexPositionBuffers;
    std::vector<Handle<HwBufferObject>> vertexNormalBuffers;
    std::vector<Handle<HwBufferObject>> vertexUVBuffers;
    std::unordered_map<uint32_t, uint32_t> primitiveIdToVB;
    {
        ZoneScopedN("Build scene buffer")
        auto f = driver.getFrameSize();
        sb.frameSize = glm::uvec2(f.width, f.height);
        sb.instanceNum = sceneData.geometries.size();
        sb.sceneInfo.gridNum = scene->ddgi.gridNum;
        sb.sceneInfo.sceneSize = scene->ddgi.worldSize;
        sb.sceneInfo.sceneCenter = scene->ddgi.gridOrigin;
        diffuseMap.push_back(rayGbuffer->positionBuffer);
        auto& camera = scene->getCamera();
        glm::mat4 vp = camera.proj * camera.view;
        for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
            auto& geom = sceneData.geometries[i];
            auto& model = sceneData.worldTransforms[i];
            auto& modelInvT = sceneData.worldTransformsInvT[i];
            
            if (primitiveIdToVB.find(geom.primitive.hwInstance.getId()) == primitiveIdToVB.end()) {
                primitiveIdToVB.emplace(geom.primitive.hwInstance.getId(), vertexPositionBuffers.size());
                vertexPositionBuffers.push_back(geom.primitive.positionBuffer);
                vertexNormalBuffers.push_back(geom.primitive.normalBuffer);
                vertexUVBuffers.push_back(geom.primitive.uvBuffer);
            }
            Instance ginstance = {};
            ginstance.meshIndex = primitiveIdToVB[geom.primitive.hwInstance.getId()];
            ginstance.transform = model;
            InstanceMaterial material = {};
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
                           [&](EmissionMaterialData& data) {
                               material.typeID = MATERIAL_EMISSION;
                               material.diffuseColor = data.radiance;
                           },
                           [](auto k) {} },
                       geom.material->materialData);
            materials.push_back(material);
            auto materialBuffer = renderGraph.createTempUniformBuffer((void*)&material, sizeof(InstanceMaterial));
            materialBuffers.push_back(materialBuffer);
            ginstance.material = material;
            sb.instances[i] = ginstance;

            TransformBuffer transformBuffer;
            transformBuffer.MVP = vp * model;
            transformBuffer.invModelT = modelInvT;
            transformBuffer.model = model;
            transformBuffer.cameraPos = glm::vec4(camera.pos(), 1.0f);
            auto tb = renderGraph.createTempUniformBuffer((void*)&transformBuffer, sizeof(TransformBuffer));
            transformBuffers.push_back(tb);
            //diffuseMap[i] = geom.material->diffuseMap;
        }
    }
    

    inflight.tlas = driver.createTLAS();
    auto sceneBuffer = renderGraph.createTempUniformBuffer((void*)&sb, sizeof(DDGISceneBuffer));

    auto rb = renderGraph.createBufferObjectSC(sizeof(Ray) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE);
    auto rayBuffer = renderGraph.importBuffer("ray buffer", rb);

    auto hb = renderGraph.createBufferObjectSC(sizeof(RayHit) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_DST | BufferUsage::STORAGE);
    auto hitBuffer = renderGraph.importBuffer("hit buffer", hb);

    
    auto lightUB = renderGraph.createTempUniformBuffer(&sceneData.lightBuffer, sizeof(LightBuffer));
    auto rayGPositionBuffer = renderGraph.importImage("ray g position", rayGbuffer->positionBuffer);
    auto rayGNormalBuffer = renderGraph.importImage("ray g normal", rayGbuffer->normalBuffer);
    auto rayGDiffuseBuffer = renderGraph.importImage("ray g diffuse", rayGbuffer->diffuseBuffer);
    auto rayGEmissionBuffer = renderGraph.importImage("ray g emission", rayGbuffer->emissionBuffer);
    auto db = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, RAYS_PER_PROBE, probeNum, 1);
    auto distanceBuffer = renderGraph.importImage("distance buffer", db);

    auto radianceColor = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA16F, 1, RAYS_PER_PROBE, probeNum, 1);
    RenderAttachments att = {};
    att.colors[0] = radianceColor;
    auto radianceRenderTarget = renderGraph.createRenderTargetSC(RAYS_PER_PROBE, probeNum, att);
    auto randianceBuffer = renderGraph.importImage("radiance buffer", radianceColor);
    auto probeTex = renderGraph.importImage("probe texture", probeTexture);
    auto probeDistTex = renderGraph.importImage("probe dist texture", probeDistTexture);
    auto probeDistSquareTex = renderGraph.importImage("probe dist square texture", probeDistSquareTexture);

    const size_t SHADOW_MAP_SIZE = 1024;
    std::vector<ResourceHandle> shadowMapResources;
    for (size_t i = 0; i < sceneData.lightBuffer.lightNum; ++i) {
        if (shadowMaps.find(i) != shadowMaps.end()) {
            shadowMapResources.push_back(renderGraph.importImage("shadow cube map", shadowMaps.at(i)));
            continue;
        }
        auto shadowMapTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT | TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE | TextureUsage::CUBE, TextureFormat::DEPTH32F, 1, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 6);
        auto shadowMap = renderGraph.importImage("shadow cube map", shadowMapTexture);
        std::vector<ResourceHandle> cubeFaceResources;
        float nearPlane = 0.001f;
        float farPlane = 25.0f;
        glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), 1.0f, nearPlane, farPlane);
        auto& light = sceneData.lightBuffer.lights[i];
        glm::vec3 lightPos = light.pos;
        std::vector<glm::mat4> shadowTransforms;
        shadowTransforms.push_back(shadowProj *
            glm::lookAt(lightPos, lightPos + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        shadowTransforms.push_back(shadowProj *
            glm::lookAt(lightPos, lightPos + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        shadowTransforms.push_back(shadowProj *
            glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
        shadowTransforms.push_back(shadowProj *
            glm::lookAt(lightPos, lightPos + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
        shadowTransforms.push_back(shadowProj *
            glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
        shadowTransforms.push_back(shadowProj *
            glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
        for (size_t j = 0; j < 6; ++j) {
            auto depthBuffer = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT | TextureUsage::UPLOADABLE, TextureFormat::DEPTH32F, 1, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1);
            RenderAttachments att = {};
            att.depth = depthBuffer;
            auto rt = renderGraph.createRenderTargetSC(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, att);
            auto shadowFace = renderGraph.importImage("shadow cube face", depthBuffer);

            PointShadowUniformBuffer uniformBuffer;
            uniformBuffer.lightPos = light.pos;
            uniformBuffer.farPlane = farPlane;
            uniformBuffer.lightVP = shadowTransforms[j];
            auto ub = renderGraph.createTempUniformBuffer((void*)&uniformBuffer, sizeof(PointShadowUniformBuffer));


            renderGraph.addFramePass({
                    .name = "build point shadow face",
                    .resources = {
                        {{shadowFace}, ResourceAccessType::DepthWrite },
                    },
                    .func = [this, scene, &sceneData, ub, rt, &inflight, &transformBuffers](FrameGraph& rg, FrameGraphContext& ctx) {
                            PipelineState pipe = {};
                            pipe.program = this->pointShadowGenProgram;
                            pipe.depthTest.enabled = 1;
                            pipe.depthTest.write = 1;
                            pipe.depthTest.compareOp = CompareOp::LESS;
                            RenderPassParams params;
                            driver.beginRenderPass(rt, params);
                            for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
                                auto& geom = sceneData.geometries[i];
                                pipe.bindUniformBuffer(0, 0, transformBuffers[i]);
                                pipe.bindUniformBuffer(0, 1, ub);
                                  
                                driver.draw(pipe, geom.primitive.hwInstance);
                            }
                            driver.endRenderPass();
                    },
             });
            cubeFaceResources.push_back(shadowFace);
        }
        renderGraph.addFramePass({
            .name = "blit to shadow cube map",
            .resources = {
                {cubeFaceResources, ResourceAccessType::TransferRead },
                {{shadowMap}, ResourceAccessType::TransferWrite}
            },
            .func = [this, scene, &sceneData, cubeFaceResources, shadowMap, &inflight](FrameGraph& rg, FrameGraphContext& ctx) {
                auto dest = ctx.unwrapTextureHandle(shadowMap);
                for (size_t i = 0; i < 6; ++i) {
                    auto tex = ctx.unwrapTextureHandle(cubeFaceResources[i]);
                    ImageSubresource destIndex{};
                    destIndex.baseLayer = i;
                    destIndex.baseLevel = 0;
                    ImageSubresource srcIndex{};
                    srcIndex.baseLayer = 0;
                    srcIndex.baseLevel = 0;
                    driver.blitTexture(dest, destIndex, tex, srcIndex);
                }
            },
        });
        shadowMapResources.push_back(shadowMap);
        shadowMaps.emplace(i, shadowMapTexture);
    }


    renderGraph.addFramePass({
        .name = "build tlas",
        .resources = {
        },
        .func = [this, scene, &sceneData, &inflight](FrameGraph& rg, FrameGraphContext& ctx) {
            for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
                auto& geom = sceneData.geometries[i];
                auto& model = sceneData.worldTransforms[i];
                Handle<HwBLAS> blas;
                auto it = this->blasMap.find(geom.primitive.hwInstance.getId());
                if (it == this->blasMap.end()) {
                    blas = driver.createBLAS();
                    driver.buildBLAS(blas, geom.primitive.hwInstance);
                    this->blasMap.emplace(geom.primitive.hwInstance.getId(), blas);
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
            driver.buildTLAS(inflight.tlas, desc);
        },
    });

    renderGraph.addFramePass({
        .name = "probe ray gens",
        .resources = {
            { {rayBuffer}, ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, &sceneData, &inflight, sceneBuffer, rayBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
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
            { {rayBuffer}, ResourceAccessType::ComputeRead },
            { {hitBuffer}, ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, &sceneData, &inflight, sceneBuffer, rayBuffer, hitBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            glm::uvec3 gridNum = scene->ddgi.gridNum;
            size_t probeNum = gridNum.x * gridNum.y * gridNum.z;

            driver.intersectRays(inflight.tlas, probeNum * RAYS_PER_PROBE, ctx.unwrapBufferHandle(rayBuffer), ctx.unwrapBufferHandle(hitBuffer));
        },
    });


    renderGraph.addFramePass({
        .name = "ray gbuffer gen",
        .resources = {
            { {hitBuffer}, ResourceAccessType::ComputeRead },
            { {rayGNormalBuffer}, ResourceAccessType::ComputeWrite },
            { {rayGDiffuseBuffer}, ResourceAccessType::ComputeWrite },
            { {rayGEmissionBuffer}, ResourceAccessType::ComputeWrite },
            { {distanceBuffer}, ResourceAccessType::ComputeWrite },
            {{rayGPositionBuffer}, ResourceAccessType::ComputeWrite},
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, &diffuseMap, &vertexPositionBuffers, &vertexNormalBuffers, &vertexUVBuffers, rayGEmissionBuffer, sceneBuffer, rayBuffer, hitBuffer, distanceBuffer, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            ctx.bindStorageImageResource(pipe, 2, 0, rayGPositionBuffer);
            ctx.bindStorageImageResource(pipe, 2, 1, rayGNormalBuffer);
            ctx.bindStorageImageResource(pipe, 2, 2, rayGDiffuseBuffer);
            ctx.bindStorageImageResource(pipe, 2, 3, rayGEmissionBuffer);
            ctx.bindStorageImageResource(pipe, 2, 4, distanceBuffer);
            pipe.bindTextureArray(2, 5, diffuseMap.data(), diffuseMap.size());
            pipe.bindUniformBuffer(0, 0, sceneBuffer);
            pipe.bindStorageBufferArray(1, 0, vertexPositionBuffers.data(), vertexPositionBuffers.size());
            pipe.bindStorageBufferArray(1, 1, vertexNormalBuffers.data(), vertexNormalBuffers.size());
            pipe.bindStorageBufferArray(1, 2, vertexUVBuffers.data(), vertexUVBuffers.size());
            ctx.bindStorageBufferResource(pipe, 1, 3, hitBuffer);
            pipe.copyPushConstants(&constants, sizeof(constants));
            pipe.program = ddgiProbeRayShadeProgram;
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });

    renderGraph.addFramePass({
        .name = "ray gbuffer shade",
        .resources = {
            { {rayGPositionBuffer}, ResourceAccessType::FragmentRead },
            { {rayGNormalBuffer}, ResourceAccessType::FragmentRead },
            { {rayGDiffuseBuffer}, ResourceAccessType::FragmentRead },
            { {rayGEmissionBuffer}, ResourceAccessType::FragmentRead },
            { {probeTex}, ResourceAccessType::FragmentRead },
            { shadowMapResources, ResourceAccessType::FragmentRead},
            { {randianceBuffer}, ResourceAccessType::ColorWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, &shadowMapResources, lightUB, rayGEmissionBuffer, &diffuseMap, sceneBuffer, radianceRenderTarget, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer, probeTex](FrameGraph& rg, FrameGraphContext& ctx) {
            auto& camera = scene->getCamera();
            PipelineState pipe = {};
            pipe.program = this->ddgiShadeProgram;
            pipe.bindUniformBuffer(0, 0, sceneBuffer);
            pipe.bindUniformBuffer(0, 1, lightUB);
            ctx.bindTextureResource(pipe, 1, 0, rayGPositionBuffer);
            ctx.bindTextureResource(pipe, 1, 1, rayGNormalBuffer);
            ctx.bindTextureResource(pipe, 1, 2, rayGDiffuseBuffer);
            ctx.bindTextureResource(pipe, 1, 3, rayGEmissionBuffer);
            ctx.bindTextureResource(pipe, 1, 4, probeTex);
            auto sms = ctx.unwrapTextureHandleArray(shadowMapResources);
            pipe.bindTextureArray(1, 5, sms.data(), sms.size());
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
            { {rayBuffer}, ResourceAccessType::ComputeRead },
            { {randianceBuffer}, ResourceAccessType::ComputeRead },
            { {distanceBuffer}, ResourceAccessType::ComputeRead },
            { {probeTex}, ResourceAccessType::ComputeWrite },
            { {probeDistTex}, ResourceAccessType::ComputeWrite },
            { {probeDistSquareTex}, ResourceAccessType::ComputeWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, &diffuseMap, rayBuffer, randianceBuffer, distanceBuffer, probeTex, probeDistTex, probeDistSquareTex](FrameGraph& rg, FrameGraphContext& ctx) {
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
    
    auto color = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA16F, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT,1);
    auto depth = renderGraph.createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F,1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT,1);
    RenderAttachments att2 = {};
    att2.colors[0] = color;
    att2.depth = depth;
    auto renderTarget = renderGraph.createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att2);
    auto renderedImage = renderGraph.importImage("rendered", color);
    auto positionBuffer = renderGraph.importImage("g position", gbuffer->positionBuffer);
    auto normalBuffer = renderGraph.importImage("g normal", gbuffer->normalBuffer);
    auto diffuseBuffer = renderGraph.importImage("g diffuse", gbuffer->diffuseBuffer);
    auto emissionBuffer = renderGraph.importImage("g emission", gbuffer->emissionBuffer);
    
    renderGraph.addFramePass({
        .name = "gbuffer gen",
        .resources = {
            { {positionBuffer}, ResourceAccessType::ColorWrite },
            { {normalBuffer}, ResourceAccessType::ColorWrite },
            { {diffuseBuffer}, ResourceAccessType::ColorWrite },
            { {emissionBuffer}, ResourceAccessType::ColorWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, color, &materials, &materialBuffers, &transformBuffers, & diffuseMap, sceneBuffer, rayBuffer, hitBuffer, distanceBuffer, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            PipelineState pipe = {};
            pipe.program = this->gbufferGenProgram;
            pipe.depthTest.enabled = 1;
            pipe.depthTest.write = 1;
            pipe.depthTest.compareOp = CompareOp::LESS;
            RenderPassParams params;
            driver.beginRenderPass(gbuffer->renderTarget, params);
            for (size_t i = 0; i < sceneData.geometries.size(); ++i) {
                auto& geom = sceneData.geometries[i];
                pipe.bindUniformBuffer(0, 0, transformBuffers[i]);
                pipe.bindUniformBuffer(0, 1, materialBuffers[i]);
                if (materials[i].typeID == MATERIAL_DIFFUSE_TEXTURE) {
                    pipe.bindTexture(1, 0, diffuseMap[materials[i].diffuseMapIndex]);
                } else {
                    pipe.bindTexture(1, 0, rayGbuffer->positionBuffer);
                }
                driver.draw(pipe, geom.primitive.hwInstance);
            }
            driver.endRenderPass();
        },
    });
    
    renderGraph.addFramePass({
        .name = "deferred render",
        .resources = {
            { {positionBuffer}, ResourceAccessType::FragmentRead },
            { {normalBuffer}, ResourceAccessType::FragmentRead },
            { {diffuseBuffer}, ResourceAccessType::FragmentRead },
            { {emissionBuffer}, ResourceAccessType::FragmentRead },
            { {probeTex}, ResourceAccessType::FragmentRead },
             { shadowMapResources, ResourceAccessType::FragmentRead},
            { {renderedImage}, ResourceAccessType::ColorWrite },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, &shadowMapResources, emissionBuffer, &diffuseMap, renderTarget, lightUB, sceneBuffer, probeTex, rayBuffer, hitBuffer, distanceBuffer, positionBuffer, normalBuffer, diffuseBuffer](FrameGraph& rg, FrameGraphContext& ctx) {
            
            auto& camera = scene->getCamera();
            PipelineState pipe = {};
            pipe.program = this->ddgiShadeProgram;

            pipe.bindUniformBuffer(0, 0, sceneBuffer);
            pipe.bindUniformBuffer(0, 1, lightUB);
            ctx.bindTextureResource(pipe, 1, 0, positionBuffer);
            ctx.bindTextureResource(pipe, 1, 1, normalBuffer);
            ctx.bindTextureResource(pipe, 1, 2, diffuseBuffer);
            ctx.bindTextureResource(pipe, 1, 3, emissionBuffer);
            ctx.bindTextureResource(pipe, 1, 4, probeTex);
            auto sms = ctx.unwrapTextureHandleArray(shadowMapResources);
            pipe.bindTextureArray(1, 5, sms.data(), sms.size());
            RenderPassParams params;
            driver.beginRenderPass(renderTarget, params);
            driver.draw(pipe, quadPrimitive);
            driver.endRenderPass();
        },
    });
    
    renderGraph.addFramePass({
        .name = "blit",
        .resources = {
            { {renderedImage}, ResourceAccessType::FragmentRead },
        },
        .func = [this, scene, probeNum, &sceneData, &inflight, &diffuseMap, renderedImage](FrameGraph& rg, FrameGraphContext& ctx) {
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