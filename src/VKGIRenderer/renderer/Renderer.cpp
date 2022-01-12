#include "Renderer.h"

#include <VKGIRenderer/utils/HalfFloat/umHalf.h>
#include <tiny_gltf.h>
#include <iostream>
#include <fstream>

#include "../Engine.h"
#include "../Scene.h"

#include <iostream>
using namespace VKGIRenderer;

template <class... Ts>
struct overload : Ts... { using Ts::operator()...; };
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

Renderer::Renderer(Engine& engine, Window* window)
    : engine(engine), window(window), driver(window, engine.getBasePath()) {
    for (size_t i = 0; i < MAX_INFLIGHTS; ++i) {
        inflights[i].fence = driver.createFence();
        inflights[i].rg = std::make_unique<FrameGraph>(driver);
    }
    registerPrograms();
    surfaceRenderTarget = driver.createDefaultRenderTarget();
    std::vector<float> v = { -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1 };
    std::vector<uint32_t> indices = { 0, 1, 2, 3, 4, 5 };
    AttributeArray attributes = {};
    attributes[0] = {
        .name = "position",
        .index = 0,
        .offset = 0,
        .stride = 8,
        .type = ElementType::FLOAT2
    };
    auto buffer0 = driver.createBufferObject(4 * v.size(), BufferUsage::VERTEX, BufferType::DEVICE);
    driver.updateBufferObjectSync(buffer0, { .data = (uint32_t*)v.data() }, 0);
    auto vbo = driver.createVertexBuffer(1, 6, 1, attributes);
    driver.setVertexBuffer(vbo, 0, buffer0);

    auto ibo = driver.createIndexBuffer(indices.size());
    driver.updateIndexBuffer(ibo, { .data = (uint32_t*)indices.data() }, 0);
    quadPrimitive = driver.createPrimitive(PrimitiveMode::TRIANGLES);
    driver.setPrimitiveBuffer(quadPrimitive, vbo, ibo);
}

static std::vector<uint32_t> loadCompiledShader(const std::string& path) {
    std::ifstream cacheFile(path, std::ifstream::binary);
    if (cacheFile.is_open()) {
        std::vector<uint8_t> cacheFileContent((std::istreambuf_iterator<char>(cacheFile)), std::istreambuf_iterator<char>());
        CompiledCode compiledCode(cacheFileContent.size() / 4);
        memcpy(compiledCode.data(), cacheFileContent.data(), cacheFileContent.size());
        return compiledCode;
    }
    throw std::runtime_error("Could not open shader file: " + path);
}

Handle<HwProgram> Renderer::loadShader(const std::string& filename) {
    auto code = loadCompiledShader(engine.assetPath(filename)+".spv");
    Program prog{ code };
    return driver.createProgram(prog);
}

void Renderer::registerShader(const std::string& shaderName, const std::string& filename)  {
    programs.emplace(shaderName, loadShader(filename));
}


Handle<HwProgram> Renderer::getShaderProgram(const std::string& shaderName) {
    return programs.at(shaderName);
}

void Renderer::registerPrograms() {
    registerShader("RayMiss", "shaders/miss.glsl");
    registerShader("RayGen", "shaders/raygen.glsl");
    registerShader("RayHit", "shaders/rayhit.glsl");
    registerShader("DrawTextureVert", "shaders/DrawTexture.vert");
    registerShader("DrawTextureFrag", "shaders/DrawTexture.frag");
}

Renderer::~Renderer() {
}

void Renderer::run(const Scene& scene) {
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

    prepareSceneData(ctx, scene);
    render(ctx, scene);

    ctx.rg->submit();
    driver.endFrame();

    ++currentFrame;
    FrameMarkEnd("Frame")
}

struct BufferInstance {
    glm::mat4 transformInvT;
    uint64_t positionBuffer;
    uint64_t normalBuffer;
};

void Renderer::render(InflightContext& ctx, const Scene& scene) {
    auto tex = ctx.rg->createTextureSC(SamplerType::SAMPLER2D, TextureUsage::STORAGE | TextureUsage::SAMPLEABLE, TextureFormat::RGBA8, 1, 2048, 2048, 1);
    auto tlas = ctx.data->tlas;
    auto instanceBuffer = ctx.data->instanceBuffer;
    ctx.rg->addFramePass({
        .textures = {
            {{tex}, ResourceAccessType::RTWrite},
        },
        .func = [this, tex, tlas, instanceBuffer](FrameGraph& rg) {
            RTPipeline pipeline = {};
            pipeline.raygenGroup = getShaderProgram("RayGen");
            pipeline.missGroups.push_back(getShaderProgram("RayMiss"));
            pipeline.hitGroups.push_back(getShaderProgram("RayHit"));
            pipeline.bindTLAS(0, 0, tlas);
            pipeline.bindStorageImage(0, 1, tex);
            pipeline.bindStorageBuffer(0, 2, instanceBuffer);
            driver.traceRays(pipeline, 2048, 2048);
        },
    });

    ctx.rg->addFramePass({
        .textures = {
            {{tex}, ResourceAccessType::FragmentRead},
        },
        .func = [this, tex, tlas](FrameGraph& rg) {
            GraphicsPipeline pipe = {};
            pipe.vertex = getShaderProgram("DrawTextureVert");
            pipe.fragment= getShaderProgram("DrawTextureFrag");
            RenderPassParams params;
            driver.beginRenderPass(this->surfaceRenderTarget, params);
            pipe.bindTexture(0, 0, tex);
            driver.draw(pipe, this->quadPrimitive);
            driver.endRenderPass();
        },
    });

}

void Renderer::prepareSceneData(InflightContext& ctx, const Scene& scene) {
    std::unordered_map<uint32_t, uint32_t> primitiveIdToVB;
    std::vector<RTInstance> instances;
    std::vector<BufferInstance> binstances;
    for (auto& obj: scene.renderObjects) {
        Handle<HwBLAS> blas;
        auto it = blasCache.find(obj.mesh->id());
        if (it == blasCache.end()) {
            blas = driver.createBLAS(obj.mesh->hwInstance);
            blasCache.emplace(obj.mesh->id(), blas);
        } else {
            blas = it->second;
        }
        RTInstance instance;
        instance.blas = blas;
        instance.transfom = obj.transform;
        instances.push_back(instance);
        BufferInstance bi;
        bi.transformInvT = glm::inverse(glm::transpose(obj.transform));
        bi.positionBuffer = driver.getDeviceAddress(obj.mesh->positionBuffer);
        bi.normalBuffer = driver.getDeviceAddress(obj.mesh->normalBuffer);
        binstances.push_back(bi);
    }
    auto tlas = driver.createTLAS({ .instances=instances.data(), .count = (uint32_t)instances.size()});
    ctx.data->tlas = tlas;
    ctx.data->instanceBuffer = ctx.rg->createTempStorageBuffer(binstances.data(), binstances.size()*sizeof(BufferInstance));
}
