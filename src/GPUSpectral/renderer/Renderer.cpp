#include "Renderer.h"

#include <GPUSpectral/utils/HalfFloat/umHalf.h>
#include <tiny_gltf.h>
#include <iostream>
#include <fstream>

#include "../engine/Engine.h"
#include "Scene.h"

#include <iostream>
using namespace GPUSpectral;

template <class... Ts>
struct overload : Ts... { using Ts::operator()...; };
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

Renderer::Renderer(Engine& engine, Window* window)
    : engine(engine), window(window), driver(std::make_unique<VulkanDriver>(window, engine.getBasePath())) {
    for (size_t i = 0; i < MAX_INFLIGHTS; ++i) {
        inflights[i].fence = driver->createFence();
        inflights[i].rg = std::make_unique<FrameGraph>(*driver);
    }
    registerPrograms();
    surfaceRenderTarget = driver->createDefaultRenderTarget();
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
    auto buffer0 = driver->createBufferObject(4 * v.size(), BufferUsage::VERTEX, BufferType::DEVICE);
    driver->updateBufferObjectSync(buffer0, { .data = (uint32_t*)v.data() }, 0);
    auto vbo = driver->createVertexBuffer(1, 6, 1, attributes);
    driver->setVertexBuffer(vbo, 0, buffer0);

    auto ibo = driver->createIndexBuffer(indices.size());
    driver->updateIndexBuffer(ibo, { .data = (uint32_t*)indices.data() }, 0);
    quadPrimitive = driver->createPrimitive(PrimitiveMode::TRIANGLES);
    driver->setPrimitiveBuffer(quadPrimitive, vbo, ibo);
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
    return driver->createProgram(prog);
}

void Renderer::registerShader(const std::string& shaderName, const std::string& filename)  {
    programs.emplace(shaderName, loadShader(filename));
}


Handle<HwProgram> Renderer::getShaderProgram(const std::string& shaderName) {
    return programs.at(shaderName);
}

void Renderer::registerPrograms() {
    registerShader("RayMiss", "shaders/miss.rmiss");
    registerShader("ShadowMiss", "shaders/shadowmiss.rmiss");
    registerShader("RayGen", "shaders/raygen.rgen");
    registerShader("RayHit", "shaders/rayhit.rchit");
    registerShader("DrawTextureVert", "shaders/DrawTexture.vert");
    registerShader("DrawTextureFrag", "shaders/DrawTexture.frag");
}

Renderer::~Renderer() {
}

void Renderer::run(const Scene& scene) {
    FrameMarkStart("Frame")
    InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver->waitFence(inflight.fence);
    if (inflight.handle) {
        driver->releaseInflight(inflight.handle);
        inflight.handle.reset();
    }
    if (inflight.tlas) {
        driver->destroyTLAS(inflight.tlas);
        inflight.tlas.reset();
    }
    Handle<HwInflight> handle = driver->beginFrame(inflight.fence);
    inflight.handle = handle;

    inflight.rg.reset();
    inflight.rg = std::make_unique<FrameGraph>(*driver);
    InflightContext ctx = {};
    ctx.data = &inflights[currentFrame % MAX_INFLIGHTS];
    ctx.rg = inflight.rg.get();

    prepareSceneData(ctx, scene);
    impl->render(ctx, scene);

    ctx.rg->submit();
    driver->endFrame();

    ++currentFrame;
    FrameMarkEnd("Frame")
}

void Renderer::render(InflightContext& ctx, const Scene& scene) {
}

void Renderer::prepareSceneData(InflightContext& ctx, const Scene& scene) {
    std::unordered_map<uint32_t, uint32_t> primitiveIdToVB;
    std::vector<RTInstance> instances;
    for (auto& obj: scene.renderObjects) {
        Handle<HwBLAS> blas;
        auto it = blasCache.find(obj.mesh->getID());
        if (it == blasCache.end()) {
            blas = driver->createBLAS(obj.mesh->getPrimitive());
            blasCache.emplace(obj.mesh->getID(), blas);
        } else {
            blas = it->second;
        }
        RTInstance instance;
        instance.blas = blas;
        instance.transfom = obj.transform;
        instances.push_back(instance);
    }
    auto tlas = driver->createTLAS({ .instances=instances.data(), .count = (uint32_t)instances.size()});
    ctx.data->tlas = tlas;
}

MeshPtr Renderer::createMesh(const std::span<Mesh::Vertex> vertices, const std::span<uint32_t>& indices) {
    return std::make_shared<Mesh>(*driver, nextMeshId++, vertices, indices);
}
