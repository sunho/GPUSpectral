#include "Renderer.h"

#include <GPUSpectral/utils/HalfFloat/umHalf.h>
#include <tiny_gltf.h>
#include <fstream>
#include <iostream>

#include "../engine/Engine.h"
#include "Scene.h"

#include <iostream>
using namespace GPUSpectral;

template <class... Ts>
struct overload : Ts... { using Ts::operator()...; };
template <class... Ts>
overload(Ts...)->overload<Ts...>;

Renderer::Renderer(Engine& engine, Window* window)
    : engine(engine), window(window), driver(std::make_unique<VulkanDriver>(window, engine.getBasePath())) {
    for (size_t i = 0; i < MAX_INFLIGHTS; ++i) {
        inflights[i].fence = driver->createFence();
        inflights[i].rg = std::make_unique<FrameGraph>(*driver);
    }
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
    auto code = loadCompiledShader(engine.assetPath(filename) + ".spv");
    Program prog{ code };
    return driver->createProgram(prog);
}

Handle<HwProgram> Renderer::getShaderProgram(const std::string& shaderName) {
    auto it = programs.find(shaderName);
    if (it != programs.end()) {
        return it->second;
    }
    auto program = loadShader("shaders/" + shaderName);
    programs.emplace(shaderName, program);
    return program;
}

Handle<HwPrimitive> GPUSpectral::Renderer::getQuadPrimitive() const noexcept {
    return quadPrimitive;
}

Handle<HwRenderTarget> GPUSpectral::Renderer::getSurfaceRenderTarget() const noexcept {
    return surfaceRenderTarget;
}

Renderer::~Renderer() {
}

void Renderer::addRenderPassCreator(std::unique_ptr<RenderPassCreator> creator) {
    renderPassCreators.push_back(std::move(creator));
}

HwDriver& GPUSpectral::Renderer::getDriver() const noexcept {
    return *driver;
}

void Renderer::run(const Scene& scene) {
    FrameMarkStart("Frame")
        InflightData& inflight = inflights[currentFrame % MAX_INFLIGHTS];
    driver->waitFence(inflight.fence);
    if (inflight.handle) {
        driver->releaseInflight(inflight.handle);
        inflight.handle.reset();
    }
    Handle<HwInflight> handle = driver->beginFrame(inflight.fence);
    inflight.handle = handle;

    inflight.rg.reset();
    inflight.rg = std::make_unique<FrameGraph>(*driver);

    for (auto& creator : renderPassCreators) {
        creator->createRenderPass(*inflight.rg, scene);
    }

    inflight.rg->submit();
    driver->endFrame();

    ++currentFrame;
    FrameMarkEnd("Frame")
}

MeshPtr Renderer::createMesh(const std::span<Mesh::Vertex> vertices, const std::span<uint32_t>& indices) {
    return std::make_shared<Mesh>(*driver, nextMeshId++, vertices, indices);
}

Handle<HwBLAS> GPUSpectral::Renderer::getOrCreateBLAS(const MeshPtr& meshPtr) {
    auto it = blasCache.find(meshPtr->getID());
    if (it == blasCache.end()) {
        auto blas = driver->createBLAS(meshPtr->getPrimitive());
        blasCache.emplace(meshPtr->getID(), blas);
        return blas;
    } else {
        return it->second;
    }
}
