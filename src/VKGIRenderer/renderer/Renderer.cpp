#include "Renderer.h"

#include <VKGIRenderer/utils/HalfFloat/umHalf.h>
#include <tiny_gltf.h>
#include <iostream>

#include "../Engine.h"
#include "../Entity.h"
#include "../Scene.h"

#include <iostream>
using namespace VKGIRenderer;

template <class... Ts>
struct overload : Ts... { using Ts::operator()...; };
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

Renderer::Renderer(Engine& engine, Window* window)
    : engine(engine), window(window), driver(window, engine.getBasePath()), loader(engine, *this) {
    for (size_t i = 0; i < MAX_INFLIGHTS; ++i) {
        inflights[i].fence = driver.createFence();
        inflights[i].rg = std::make_unique<FrameGraph>(driver);
    }

    Primitive primitive;
    std::vector<float> v = { -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1 };
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

    cube = loader.loadObj(engine.assetPath("cube.obj"));

    surfaceRenderTarget = driver.createDefaultRenderTarget();

    dummyTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE , TextureFormat::RGBA8, 1, 1, 1, 1);
    uint32_t zero = 0;
    driver.copyTextureInitialData(dummyTexture, { .data = &zero });
    
    registerPrograms();
}

Handle<HwProgram> Renderer::loadComputeShader(const std::string& filename) {
    auto code = driver.compileCode(engine.assetPath(filename).c_str());
    Program prog{ code };
    return driver.createProgram(prog);
}

Handle<HwProgram> Renderer::loadGraphicsShader(const std::string& vertFilename, const std::string& fragFilename) {
    auto vertCode = driver.compileCode(engine.assetPath(vertFilename).c_str());
    auto fragCode = driver.compileCode(engine.assetPath(fragFilename).c_str());
    Program prog{ vertCode, fragCode };
    return driver.createProgram(prog);
}

void Renderer::registerComputeShader(const std::string& shaderName, const std::string& filename)  {
    programs.emplace(shaderName, loadComputeShader(filename));
}

void Renderer::registerGraphicsShader(const std::string& shaderName, const std::string& vertFilename, const std::string& fragFilename) {
    programs.emplace(shaderName, loadGraphicsShader(vertFilename, fragFilename));
}

Handle<HwProgram> Renderer::getShaderProgram(const std::string& shaderName) {
    return programs.at(shaderName);
}

void Renderer::registerPrograms() {
    registerGraphicsShader("ForwardPhong", "shaders/ForwardPhong.vert", "shaders/ForwardPhong.frag");
    registerGraphicsShader("DisplayTexture", "shaders/DisplayTexture.vert", "shaders/DisplayTexture.frag");
    registerGraphicsShader("ToneMap", "shaders/ToneMap.vert", "shaders/ToneMap.frag");
    registerGraphicsShader("ForwardRT", "shaders/ForwardRT.vert", "shaders/ForwardRT.frag");
    registerComputeShader("DDGIProbeRayGen", "shaders/DDGIProbeRayGen.comp");
    registerComputeShader("DDGIProbeRayShade", "shaders/DDGIProbeRayShade.comp");
    registerGraphicsShader("PointShadowGen", "shaders/PointShadowGen.vert", "shaders/PointShadowGen.frag");
    registerGraphicsShader("GBufferGen", "shaders/GBufferGen.vert", "shaders/GBufferGen.frag");
    registerGraphicsShader("DeferredRender", "shaders/DeferredRender.vert", "shaders/DeferredRender.frag");
    registerGraphicsShader("DDGIShade", "shaders/DDGIShade.vert", "shaders/DDGIShade.frag");
    registerComputeShader("DDGIProbeDepthUpdate", "shaders/DDGIProbeDepthUpdate.comp");
    registerComputeShader("DDGIProbeIrradianceUpdate", "shaders/DDGIProbeIrradianceUpdate.comp");
    registerGraphicsShader("ProbeDebug", "shaders/ProbeDebug.vert", "shaders/ProbeDebug.frag");
}

Renderer::~Renderer() {
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
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver.waitFence(inflight.fence);
    if (inflight.handle) {
        driver.releaseInflight(inflight.handle);
        inflight.handle.reset();
    }
    if (inflight.tlas) {
        driver.destroyTLAS(inflight.tlas);
        inflight.tlas.reset();
    }
    Handle<HwInflight> handle = driver.beginFrame(inflight.fence);
    inflight.handle = handle;

    inflight.rg.reset();
    inflight.rg = std::make_unique<FrameGraph>(driver);
    InflightContext ctx = {};
    ctx.data = &inflights[currentFrame % MAX_INFLIGHTS];
    ctx.rg = inflight.rg.get();
    ctx.sceneData = &scene->getSceneData();
    ctx.scene = scene;

    prepareSceneData(ctx);

    ddgiSuite(ctx);
    ctx.rg->submit();
    driver.endFrame();

    ++currentFrame;
    FrameMarkEnd("Frame")
}

void Renderer::prepareSceneData(InflightContext& ctx) {
    ctx.data->reset(driver);
    ctx.scene->prepare();
    std::unordered_map<uint32_t, uint32_t> primitiveIdToVB;
    {
        ZoneScopedN("Build scene buffer")
        auto f = driver.getFrameSize();
        SceneBuffer sb = {};
        sb.frameSize = glm::uvec2(f.width, f.height);
        sb.instanceNum = ctx.sceneData->geometries.size();
        sb.sceneInfo.gridNum = ctx.scene->ddgi.gridNum;
        sb.sceneInfo.sceneSize = ctx.scene->ddgi.worldSize;
        sb.sceneInfo.sceneCenter = ctx.scene->ddgi.gridOrigin;
        //ctx.data->diffuseMap.push_back(rayGbuffer->positionBuffer);
        auto& camera = ctx.scene->getCamera();
        glm::mat4 vp = camera.proj * camera.view;
        for (size_t i = 0; i < ctx.sceneData->geometries.size(); ++i) {
            auto& geom = ctx.sceneData->geometries[i];
            auto& model = ctx.sceneData->worldTransforms[i];
            auto& modelInvT = ctx.sceneData->worldTransformsInvT[i];

            if (primitiveIdToVB.find(geom.primitive.hwInstance.getId()) == primitiveIdToVB.end()) {
                primitiveIdToVB.emplace(geom.primitive.hwInstance.getId(), ctx.data->vertexPositionBuffers.size());
                ctx.data->vertexPositionBuffers.push_back(geom.primitive.positionBuffer);
                ctx.data->vertexNormalBuffers.push_back(geom.primitive.normalBuffer);
                ctx.data->vertexUVBuffers.push_back(geom.primitive.uvBuffer);
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
                    material.diffuseMapIndex = ctx.data->diffuseMap.size();
                    ctx.data->diffuseMap.push_back(data.diffuseMap);
                },
                [&](EmissionMaterialData& data) {
                    material.typeID = MATERIAL_EMISSION;
                    material.diffuseColor = data.radiance;
                },
                [](auto k) {} },
                geom.material->materialData);
            ctx.data->materials.push_back(material);
            auto materialBuffer = ctx.rg->createTempUniformBuffer((void*)&material, sizeof(InstanceMaterial));
            ctx.data->materialBuffers.push_back(materialBuffer);
            ginstance.material = material;
            sb.instances[i] = ginstance;

            TransformBuffer transformBuffer;
            transformBuffer.MVP = vp * model;
            transformBuffer.invModelT = modelInvT;
            transformBuffer.model = model;
            transformBuffer.cameraPos = glm::vec4(camera.pos(), 1.0f);
            auto tb = ctx.rg->createTempUniformBuffer((void*)&transformBuffer, sizeof(TransformBuffer));
            ctx.data->transformBuffers.push_back(tb);
        }
        ctx.data->cpuSceneBuffer = sb;
        ctx.data->sceneBuffer = ctx.rg->createTempUniformBuffer((void*)&sb, sizeof(SceneBuffer));
    }

    for (size_t i = 0; i < ctx.sceneData->geometries.size(); ++i) {
        auto& geom = ctx.sceneData->geometries[i];
        auto& model = ctx.sceneData->worldTransforms[i];
        Handle<HwBLAS> blas;
        auto it = blasCache.find(geom.primitive.hwInstance.getId());
        auto hwPrimitive = geom.primitive.hwInstance;
        if (it == blasCache.end()) {
            blas = driver.createBLAS();
            blasCache.emplace(hwPrimitive.getId(), blas);
            ctx.rg->addFramePass({
                .name = "build blas",
                .func = [this, blas, hwPrimitive](FrameGraph& fg) {
                    driver.buildBLAS(blas, hwPrimitive);
                },
             });
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
        ctx.data->instances.push_back(instance);
    }

    for (size_t i = 0; i < ctx.sceneData->lightBuffer.lightNum; ++i) {
        if (shadowMapCache.find(i) != shadowMapCache.end()) {
            ctx.data->shadowMaps.push_back(shadowMapCache.at(i));
        }
        else {
            auto shadowMap = buildPointShadowMap(ctx, ctx.sceneData->lightBuffer.lights[i]);
            ctx.data->shadowMaps.push_back(shadowMap);
            shadowMapCache.emplace(i, shadowMap);
        }
    }

    if (ctx.data->diffuseMap.empty()) {
        ctx.data->diffuseMap.push_back(dummyTexture);
    }

    RTSceneDescriptor desc = { ctx.data->instances.data(), ctx.data->instances.size() };
    ctx.data->tlas = driver.createTLAS();
    ctx.rg->addFramePass({
        .name = "build tlas",
        .func = [this, desc, ctx](FrameGraph& fg) {
            driver.buildTLAS(ctx.data->tlas, desc);
        },
    });


    ctx.data->lightBuffer = ctx.rg->createTempUniformBuffer((void*)&ctx.sceneData->lightBuffer, sizeof(LightBuffer));
}

Handle<HwTexture> Renderer::buildPointShadowMap(InflightContext& ctx, LightData light) {
    const size_t SHADOW_MAP_SIZE = 1024;
    auto shadowMap = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT | TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE | TextureUsage::CUBE, TextureFormat::DEPTH32F, 1, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 6);
    std::vector<Handle<HwTexture>> cubeFaces;
    float nearPlane = 0.001f;
    float farPlane = 25.0f;
    glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), 1.0f, nearPlane, farPlane);
    glm::vec3 lightPos = light.pos;
    std::vector<glm::mat4> shadowTransforms;
    SceneData& sceneData = ctx.scene->getSceneData();
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
        auto shadowFace = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT | TextureUsage::UPLOADABLE, TextureFormat::DEPTH32F, 1, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1);
        RenderAttachments att = {};
        att.depth = shadowFace;
        auto rt = ctx.rg->createRenderTargetSC(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, att);

        PointShadowUniformBuffer uniformBuffer;
        uniformBuffer.lightPos = light.pos;
        uniformBuffer.farPlane = farPlane;
        uniformBuffer.lightVP = shadowTransforms[j];
        auto ub = ctx.rg->createTempUniformBuffer((void*)&uniformBuffer, sizeof(PointShadowUniformBuffer));

        ctx.rg->addFramePass({
            .name = "build point shadow face",
            .textures = {
                {{shadowFace}, ResourceAccessType::DepthWrite },
            },
            .func = [this, ctx, sceneData, ub, rt](FrameGraph& rg) {
                PipelineState pipe = {};
                pipe.program = getShaderProgram("PointShadowGen");
                pipe.depthTest.enabled = 1;
                pipe.depthTest.write = 1;
                pipe.depthTest.compareOp = CompareOp::LESS;
                RenderPassParams params;
                driver.beginRenderPass(rt, params);
                for (size_t i = 0; i < ctx.sceneData->geometries.size(); ++i) {
                    auto& geom = sceneData.geometries[i];
                    pipe.bindUniformBuffer(0, 0, ctx.data->transformBuffers[i]);
                    pipe.bindUniformBuffer(0, 1, ub);

                    driver.draw(pipe, geom.primitive.hwInstance);
                }
                driver.endRenderPass();
            },
        });
        cubeFaces.push_back(shadowFace);
    }
    ctx.rg->addFramePass({
        .name = "blit to shadow cube map",
        .textures = {
            {cubeFaces, ResourceAccessType::TransferRead },
            {{shadowMap}, ResourceAccessType::TransferWrite}
        },
        .func = [this, cubeFaces, shadowMap](FrameGraph& rg) {
            for (size_t i = 0; i < 6; ++i) {
                ImageSubresource destIndex{};
                destIndex.baseLayer = i;
                destIndex.baseLevel = 0;
                ImageSubresource srcIndex{};
                srcIndex.baseLayer = 0;
                srcIndex.baseLevel = 0;
                driver.blitTexture(shadowMap, destIndex, cubeFaces[i], srcIndex);
            }
        },
    });
    return shadowMap;
}

void Renderer::deferSuite(InflightContext& ctx) {
    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }

    auto positionBuffer = gbuffer->positionBuffer;
    auto normalBuffer = gbuffer->normalBuffer;
    auto diffuseBuffer = gbuffer->diffuseBuffer;

    ctx.rg->addFramePass({ 
        .name = "gbuffer generation",
        .textures = { 
            {{positionBuffer}, ResourceAccessType::ColorWrite},
            {{normalBuffer}, ResourceAccessType::ColorWrite },
            {{diffuseBuffer}, ResourceAccessType::ColorWrite },
        },
        .func = [this, ctx](FrameGraph& rg) {
            auto& camera = ctx.scene->getCamera();

            PipelineState pipe = {};
            pipe.program = getShaderProgram("GBufferGen");
            pipe.depthTest.enabled = 1;
            pipe.depthTest.write = 1;
            pipe.depthTest.compareOp = CompareOp::LESS;
            RenderPassParams params;
            driver.beginRenderPass(gbuffer->renderTarget, params);
            for (size_t i = 0; i < ctx.sceneData->geometries.size(); ++i) {
                auto& geom = ctx.sceneData->geometries[i];
                auto& model = ctx.sceneData->worldTransforms[i];
                auto& modelInvT = ctx.sceneData->worldTransformsInvT[i];
                pipe.bindUniformBuffer(0, 0, this->createTransformBuffer(rg, camera, model, modelInvT));
                //pipe.bindTexture(1, 0, geom.material->diffuseMap);
                driver.draw(pipe, geom.primitive.hwInstance);
            }
            driver.endRenderPass();
        },
    });

    auto renderedBuffer = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, 1);
    auto depth = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, 1);
    RenderAttachments att = {};
    att.colors[0] = renderedBuffer;
    att.depth = depth;
    auto renderTarget = ctx.rg->createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);

    ctx.rg->addFramePass({
        .name = "deferred render",
        .textures = {
            { {positionBuffer}, ResourceAccessType::FragmentRead },
            { {normalBuffer}, ResourceAccessType::FragmentRead },
            { {diffuseBuffer}, ResourceAccessType::FragmentRead },
            { {renderedBuffer}, ResourceAccessType::ColorWrite }
        },
        .func = [this, ctx, renderTarget, positionBuffer, normalBuffer, diffuseBuffer](FrameGraph& rg) {
            auto& camera = ctx.scene->getCamera();
            PipelineState pipe = {};
            pipe.program = getShaderProgram("DeferredRender");
            pipe.bindUniformBuffer(0, 0, ctx.data->lightBuffer);
            pipe.bindTexture(1, 0, positionBuffer);
            pipe.bindTexture(1, 1, normalBuffer);
            pipe.bindTexture(1, 2, diffuseBuffer);
            RenderPassParams params;
            driver.beginRenderPass(renderTarget, params);
            driver.draw(pipe, quadPrimitive);
            driver.endRenderPass();
        },

    });

    ctx.rg->addFramePass({
        .name = "blit",
        .textures = {
            { {renderedBuffer}, ResourceAccessType::FragmentRead },
        },
        .func = [this, ctx, renderTarget, renderedBuffer](FrameGraph& rg) {
            PipelineState pipe = {};
            pipe.program = getShaderProgram("DisplayTexture");
            RenderPassParams params;
            driver.beginRenderPass(this->surfaceRenderTarget, params);
            pipe.bindTexture(1, 0, renderedBuffer);
            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
    });
}

void Renderer::rtSuite(InflightContext& ctx) {
    auto f = driver.getFrameSize();
    auto hitBuffer = ctx.rg->createBufferObjectSC(f.width * f.height * sizeof(RayHit), BufferUsage::TRANSFER_DST | BufferUsage::STORAGE);

    ctx.rg->addFramePass({
        .name = "intersect",
        .buffers = {
            { {hitBuffer}, ResourceAccessType::ComputeWrite },
        },
        .func = [this, ctx, hitBuffer](FrameGraph& rg) {
            std::vector<Ray> rays;
            auto f = driver.getFrameSize();
            rays.resize(f.width * f.height);
            for (size_t i = 0; i < f.height; ++i) {
                for (size_t j = 0; j < f.width; ++j) {
                    Ray& ray = rays[i * f.width + j];
                    auto dir = ctx.scene->getCamera().rayDir(glm::vec2(f.width, f.height), glm::vec2(j, i));
                    dir.y *= -1.0f;
                    ray.origin = ctx.scene->getCamera().pos();
                    ray.dir = dir;
                    ray.minTime = 0.001f;
                    ray.maxTime = 100000.0f;
                }
            }
            auto rayBuffer = rg.createBufferObjectSC(rays.size() * sizeof(Ray), BufferUsage::STORAGE);
            driver.updateBufferObjectSync(rayBuffer, { reinterpret_cast<uint32_t*>(rays.data()) }, 0); // TODO
            driver.intersectRays(ctx.data->tlas, f.width * f.height, rayBuffer, hitBuffer);
        },
    });

    auto renderedImage = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA8, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, 1);
    RenderAttachments att = {};
    att.colors[0] = renderedImage;
    auto renderTarget = ctx.rg->createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att);  
    
    ctx.rg->addFramePass({
        .name = "render",
        .buffers = {
            { {hitBuffer}, ResourceAccessType::ComputeRead },
        },
        .func = [this, renderTarget, hitBuffer, ctx](FrameGraph& rg) {
            DDGIPushConstants constants = {};

            PipelineState pipe = {};
            pipe.program = getShaderProgram("ForwardRT");
            driver.beginRenderPass(renderTarget, {});
            pipe.bindStorageBufferArray(1, 0, ctx.data->vertexPositionBuffers);
            pipe.bindStorageBufferArray(1, 1, ctx.data->vertexNormalBuffers);
            pipe.bindStorageBufferArray(1, 2, ctx.data->vertexUVBuffers);
            pipe.bindStorageBuffer(1, 3, hitBuffer);
            pipe.bindUniformBuffer(0, 0, ctx.data->sceneBuffer);

            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
    });

    ctx.rg->addFramePass({
        .name = "blit",
        .textures = {
            { {renderedImage}, ResourceAccessType::FragmentRead },
        },
        .func = [this, renderedImage](FrameGraph& rg) {
            PipelineState pipe = {};
            pipe.program = getShaderProgram("DisplayTexture");
            RenderPassParams params;
            driver.beginRenderPass(this->surfaceRenderTarget, params);
            pipe.bindTexture(1, 0, renderedImage);
            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
    });
}

glm::ivec3 probIDToGrid(int id, DDGISceneInfo sceneInfo) {
    glm::ivec3 gridNum = glm::ivec3(sceneInfo.gridNum);
    int x = id % gridNum.x;
    int y = (id % (gridNum.x * gridNum.y)) / gridNum.x;
    int z = id / (gridNum.x * gridNum.y);
    return glm::ivec3(x, y, z);
}

glm::vec3 gridToPos(glm::ivec3 grid, DDGISceneInfo sceneInfo) {
    glm::vec3 gridSize = sceneInfo.sceneSize * 2.0f / glm::vec3(sceneInfo.gridNum);
    glm::vec3 ogd = glm::vec3(grid) * gridSize;
    return ogd - sceneInfo.sceneSize + sceneInfo.sceneCenter;
}

glm::vec3 probeIDToPos(int id, DDGISceneInfo sceneInfo) {
    glm::ivec3 grid = probIDToGrid(id, sceneInfo);
    return gridToPos(grid, sceneInfo);
}

struct ProbeDebugUniformBuffer {
    glm::mat4 MVP;
    glm::mat4 model;
    DDGISceneInfo sceneInfo;
    int probeId;
};


void Renderer::ddgiSuite(InflightContext& ctx) {
    glm::uvec3 gridNum = ctx.scene->ddgi.gridNum;
    size_t probeNum = gridNum.x * gridNum.y * gridNum.z;
    glm::uvec2 probeTexSize = glm::uvec2(IRD_MAP_TEX_SIZE * IRD_MAP_PROBE_COLS, IRD_MAP_TEX_SIZE * (probeNum / IRD_MAP_PROBE_COLS));
    glm::uvec2 depthTexSize = glm::uvec2(DEPTH_MAP_TEX_SIZE * DEPTH_MAP_PROBE_COLS, DEPTH_MAP_TEX_SIZE * (probeNum / DEPTH_MAP_PROBE_COLS));
    if (!rayGbuffer) {
        rayGbuffer = std::make_unique<GBuffer>(driver, RAYS_PER_PROBE, probeNum);
        probeTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, 1, probeTexSize.x, probeTexSize.y, 1);
        probeDepthTexture = driver.createTexture(SamplerType::SAMPLER2D, TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE | TextureUsage::STORAGE, TextureFormat::RGBA16F, 1, depthTexSize.x, depthTexSize.y, 1);

        std::vector<half> blank(probeTexSize.x * probeTexSize.y*4, 0.0f);
        driver.copyTextureInitialData(probeTexture, { .data = (uint32_t*)blank.data(), .size = 0 });

        std::vector<half> blank2(depthTexSize.x * depthTexSize.y * 4, 0.0f);
        driver.copyTextureInitialData(probeDepthTexture, { .data = (uint32_t*)blank2.data(), .size = 0 });
    }
 
    if (!gbuffer) {
        gbuffer = std::make_unique<GBuffer>(driver, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT);
    }

    auto rayBuffer = ctx.rg->createBufferObjectSC(sizeof(Ray) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE);
    auto hitBuffer = ctx.rg->createBufferObjectSC(sizeof(RayHit) * probeNum * RAYS_PER_PROBE, BufferUsage::TRANSFER_DST | BufferUsage::STORAGE);

    auto rayGPositionBuffer = rayGbuffer->positionBuffer;
    auto rayGNormalBuffer = rayGbuffer->normalBuffer;
    auto rayGDiffuseBuffer = rayGbuffer->diffuseBuffer;
    auto rayGEmissionBuffer = rayGbuffer->emissionBuffer;
    auto distanceBuffer = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, RAYS_PER_PROBE, probeNum, 1);

    auto randianceBuffer = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLEABLE, TextureFormat::RGBA16F, 1, RAYS_PER_PROBE, probeNum, 1);
    auto depth3 = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F, 1, RAYS_PER_PROBE, probeNum, 1);
    RenderAttachments att = {};
    att.colors[0] = randianceBuffer;
    att.depth = depth3;
    auto radianceRenderTarget = ctx.rg->createRenderTargetSC(RAYS_PER_PROBE, probeNum, att);
    std::cout << "created: " << radianceRenderTarget.getId() << std::endl;
    std::cout << "Radiance buffer: " << randianceBuffer.getId() << std::endl;

    ctx.rg->addFramePass({
        .name = "probe ray gens",
        .buffers = {
            { {rayBuffer}, ResourceAccessType::ComputeWrite },
        },
        .func = [this,  ctx, rayBuffer](FrameGraph& rg) {
            glm::uvec3 gridNum = ctx.scene->ddgi.gridNum;
            size_t probeNum = gridNum.x * gridNum.y * gridNum.z;
            
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            pipe.bindUniformBuffer(0, 0, ctx.data->sceneBuffer);
            pipe.bindStorageBuffer(1, 0, rayBuffer);
            pipe.copyPushConstants(&constants, sizeof(constants));
            pipe.program = getShaderProgram("DDGIProbeRayGen");
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });

    ctx.rg->addFramePass({
        .name = "intersect",
        .buffers = {
            { {rayBuffer}, ResourceAccessType::ComputeRead },
            { {hitBuffer}, ResourceAccessType::ComputeWrite },
        },
        .func = [this, ctx,  rayBuffer, hitBuffer](FrameGraph& rg) {
            glm::uvec3 gridNum = ctx.scene->ddgi.gridNum;
            size_t probeNum = gridNum.x * gridNum.y * gridNum.z;

            driver.intersectRays(ctx.data->tlas, probeNum * RAYS_PER_PROBE, rayBuffer, hitBuffer);
        },
    });

    ctx.rg->addFramePass({
        .name = "ray gbuffer gen",
        .textures = {
            { {rayGNormalBuffer}, ResourceAccessType::ComputeWrite },
            { {rayGDiffuseBuffer}, ResourceAccessType::ComputeWrite },
            { {rayGEmissionBuffer}, ResourceAccessType::ComputeWrite },
            { {distanceBuffer}, ResourceAccessType::ComputeWrite },
            {{rayGPositionBuffer}, ResourceAccessType::ComputeWrite},
        },
        .buffers = {
            { {hitBuffer}, ResourceAccessType::ComputeRead },
        },
        .func = [this, ctx, probeNum, rayGEmissionBuffer, rayBuffer, hitBuffer, distanceBuffer, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer](FrameGraph& rg) {
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            pipe.bindStorageImage(2, 0, rayGPositionBuffer);
            pipe.bindStorageImage(2, 1, rayGNormalBuffer);
            pipe.bindStorageImage(2, 2, rayGDiffuseBuffer);
            pipe.bindStorageImage(2, 3, rayGEmissionBuffer);
            pipe.bindStorageImage(2, 4, distanceBuffer);
            pipe.bindTextureArray(2, 5, ctx.data->diffuseMap);
            pipe.bindUniformBuffer(0, 0, ctx.data->sceneBuffer);
            pipe.bindStorageBufferArray(1, 0, ctx.data->vertexPositionBuffers);
            pipe.bindStorageBufferArray(1, 1, ctx.data->vertexNormalBuffers);
            pipe.bindStorageBufferArray(1, 2, ctx.data->vertexUVBuffers);
            pipe.bindStorageBuffer(1, 3, hitBuffer);
            pipe.copyPushConstants(&constants, sizeof(constants));
            pipe.program = getShaderProgram("DDGIProbeRayShade");
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });

    ctx.rg->addFramePass({
        .name = "ray gbuffer shade",
        .textures = {
            { {rayGPositionBuffer}, ResourceAccessType::FragmentRead },
            { {rayGNormalBuffer}, ResourceAccessType::FragmentRead },
            { {rayGDiffuseBuffer}, ResourceAccessType::FragmentRead },
            { {rayGEmissionBuffer}, ResourceAccessType::FragmentRead },
            { {probeTexture}, ResourceAccessType::FragmentRead },
            { ctx.data->shadowMaps, ResourceAccessType::FragmentRead},
            { {randianceBuffer}, ResourceAccessType::ColorWrite },
        },
        .func = [this, ctx, probeNum, rayGEmissionBuffer, radianceRenderTarget, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer](FrameGraph& rg) {
            auto& camera = ctx.scene->getCamera();
            PipelineState pipe = {};
            pipe.program = getShaderProgram("DDGIShade");
            pipe.bindUniformBuffer(0, 0, ctx.data->sceneBuffer);
            pipe.bindUniformBuffer(0, 1, ctx.data->lightBuffer);
            pipe.bindTexture(1, 0, rayGPositionBuffer);
            pipe.bindTexture(1, 1, rayGNormalBuffer);
            pipe.bindTexture(1, 2, rayGDiffuseBuffer);
            pipe.bindTexture(1, 3, rayGEmissionBuffer);
            pipe.bindTexture(1, 4, probeTexture);
            pipe.bindTexture(1, 5, probeDepthTexture);
            pipe.bindTextureArray(1, 6, ctx.data->shadowMaps);
            RenderPassParams params;
            driver.beginRenderPass(radianceRenderTarget, params);
            driver.draw(pipe, quadPrimitive);
            driver.endRenderPass();
        },
    });
    
    // Third pass: probe update (compute)
    ctx.rg->addFramePass({
        .name = "probe irradiance update",
        .textures = {
            { {randianceBuffer}, ResourceAccessType::ComputeRead },
            { {distanceBuffer}, ResourceAccessType::ComputeRead },
            { {probeTexture}, ResourceAccessType::ComputeWrite },
        },
        .buffers = {
            { {rayBuffer}, ResourceAccessType::ComputeRead },
        },
        .func = [this, probeNum, rayBuffer, randianceBuffer, distanceBuffer](FrameGraph& rg) {
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            pipe.bindStorageBuffer(0, 0, rayBuffer);
            pipe.bindTexture(1, 0, randianceBuffer);
            pipe.bindTexture(1, 1, distanceBuffer);
            pipe.bindStorageImage(2, 0, probeTexture);
            pipe.program = getShaderProgram("DDGIProbeIrradianceUpdate");
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });

    ctx.rg->addFramePass({
        .name = "probe depth update",
        .textures = {
            { {distanceBuffer}, ResourceAccessType::ComputeRead },
            { {probeDepthTexture}, ResourceAccessType::ComputeWrite },
        },
        .buffers = {
            { {rayBuffer}, ResourceAccessType::ComputeRead },
        },
        .func = [this, probeNum, rayBuffer, randianceBuffer, distanceBuffer](FrameGraph& rg) {
            DDGIPushConstants constants = {};
            constants.globalRngState = rand();
            PipelineState pipe = {};
            pipe.bindStorageBuffer(0, 0, rayBuffer);
            pipe.bindTexture(1, 0, distanceBuffer);
            pipe.bindStorageImage(2, 0, probeDepthTexture);
            pipe.program = getShaderProgram("DDGIProbeDepthUpdate");
            driver.dispatch(pipe, probeNum, 1, 1);
        },
    });
    
    auto renderedImage = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::COLOR_ATTACHMENT, TextureFormat::RGBA16F, 1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT,1);
    auto depth = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, TextureFormat::DEPTH32F,1, HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT,1);
    RenderAttachments att2 = {};
    att2.colors[0] = renderedImage;
    att2.depth = depth;
    auto renderTarget = ctx.rg->createRenderTargetSC(HwTexture::FRAME_WIDTH, HwTexture::FRAME_HEIGHT, att2);

    auto positionBuffer = gbuffer->positionBuffer;
    auto normalBuffer = gbuffer->normalBuffer;
    auto diffuseBuffer = gbuffer->diffuseBuffer;
    auto emissionBuffer = gbuffer->emissionBuffer;
    
    std::vector<Handle<HwBufferObject>> probeUniformBuffers;
    DDGISceneInfo sceneInfo = ctx.data->cpuSceneBuffer.sceneInfo;
    /*
    for (size_t i = 0; i < probeNum; ++i) {
        glm::vec3 pos = probeIDToPos(i, sceneInfo);
        glm::vec3 gridSize = sceneInfo.sceneSize * 2.0f / glm::vec3(sceneInfo.gridNum);
        auto scaler = glm::scale(glm::identity<glm::mat4>(), gridSize * 0.1f);
        auto model = glm::translate(glm::identity<glm::mat4>(), pos) * scaler;
        auto mvp = ctx.scene->getCamera().proj * ctx.scene->getCamera().view * model;
        ProbeDebugUniformBuffer ub = {};
        ub.model = model;
        ub.MVP = mvp;
        ub.probeId = i;
        ub.sceneInfo = sceneInfo;
        auto ubo = ctx.rg->createTempUniformBuffer(&ub, sizeof(ProbeDebugUniformBuffer));
        probeUniformBuffers.push_back(ubo);
    }*/

    ctx.rg->addFramePass({
        .name = "gbuffer gen",
        .textures = {
            { {positionBuffer}, ResourceAccessType::ColorWrite },
            { {normalBuffer}, ResourceAccessType::ColorWrite },
            { {diffuseBuffer}, ResourceAccessType::ColorWrite },
            { {emissionBuffer}, ResourceAccessType::ColorWrite },
        },
        .func = [this, ctx, probeUniformBuffers, probeNum, renderedImage, rayBuffer, hitBuffer, distanceBuffer, rayGPositionBuffer, rayGNormalBuffer, rayGDiffuseBuffer](FrameGraph& rg) {
            PipelineState pipe = {};
            pipe.program = getShaderProgram("GBufferGen");
            pipe.depthTest.enabled = 1;
            pipe.depthTest.write = 1;
            pipe.depthTest.compareOp = CompareOp::LESS;
            RenderPassParams params;
            driver.beginRenderPass(gbuffer->renderTarget, params);
            for (size_t i = 0; i < ctx.sceneData->geometries.size(); ++i) {
                auto& geom = ctx.sceneData->geometries[i];
                pipe.bindUniformBuffer(0, 0, ctx.data->transformBuffers[i]);
                pipe.bindUniformBuffer(0, 1, ctx.data->materialBuffers[i]);
                if (ctx.data->materials[i].typeID == MATERIAL_DIFFUSE_TEXTURE) {
                    pipe.bindTexture(1, 0, ctx.data->diffuseMap[ctx.data->materials[i].diffuseMapIndex]);
                } else {
                    pipe.bindTexture(1, 0, rayGbuffer->positionBuffer);
                }
                driver.draw(pipe, geom.primitive.hwInstance);
            }
            pipe.bindings.clear();
            /*pipe.program = getShaderProgram("ProbeDebug");
            for (size_t i = 0; i < probeNum; ++i) {
                pipe.bindUniformBuffer(0, 0, probeUniformBuffers[i]);
                pipe.bindTexture(1, 0, probeTexture);
                pipe.bindTexture(1, 1, probeDepthTexture);
                driver.draw(pipe, cube->getPrimitives().front().hwInstance);
            }*/
            driver.endRenderPass();
        },
    });
    
    ctx.rg->addFramePass({
        .name = "deferred render",
        .textures = {
            { {positionBuffer}, ResourceAccessType::FragmentRead },
            { {normalBuffer}, ResourceAccessType::FragmentRead },
            { {diffuseBuffer}, ResourceAccessType::FragmentRead },
            { {emissionBuffer}, ResourceAccessType::FragmentRead },
            { {probeTexture}, ResourceAccessType::FragmentRead },
            { {probeDepthTexture}, ResourceAccessType::FragmentRead},
             { ctx.data->shadowMaps, ResourceAccessType::FragmentRead},
            { {renderedImage}, ResourceAccessType::ColorWrite },
        },
        .func = [this, ctx, probeNum, emissionBuffer, renderTarget, rayBuffer, hitBuffer, distanceBuffer, positionBuffer, normalBuffer, diffuseBuffer](FrameGraph& rg) {
            
            auto& camera = ctx.scene->getCamera();
            PipelineState pipe = {};
            pipe.program = getShaderProgram("DDGIShade");

            pipe.bindUniformBuffer(0, 0, ctx.data->sceneBuffer);
            pipe.bindUniformBuffer(0, 1, ctx.data->lightBuffer);
            pipe.bindTexture(1, 0, positionBuffer);
            pipe.bindTexture(1, 1, normalBuffer);
            pipe.bindTexture(1, 2, diffuseBuffer);
            pipe.bindTexture(1, 3, emissionBuffer);
            pipe.bindTexture(1, 4, probeTexture);
            pipe.bindTexture(1, 5, probeDepthTexture);
            pipe.bindTextureArray(1, 6, ctx.data->shadowMaps);
            RenderPassParams params;
            driver.beginRenderPass(renderTarget, params);
            driver.draw(pipe, quadPrimitive);
            driver.endRenderPass();
        },
    });


    ctx.rg->addFramePass({
        .name = "blit",
        .textures = {
            { {renderedImage}, ResourceAccessType::FragmentRead },
        },
        .func = [this,  probeNum, renderedImage](FrameGraph& rg) {
            PipelineState pipe = {};
            pipe.program = getShaderProgram("ToneMap");
            RenderPassParams params;
            driver.beginRenderPass(this->surfaceRenderTarget, params);
            pipe.bindTexture(1, 0, renderedImage);
            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
    });
}
