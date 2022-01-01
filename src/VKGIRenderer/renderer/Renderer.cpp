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
    registerShader("DDGIProbeRayShade", "shaders/DDGIProbeRayShade.comp");
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

    prepareSceneData(ctx);

    ctx.rg->submit();
    driver.endFrame();

    ++currentFrame;
    FrameMarkEnd("Frame")
}

void Renderer::prepareSceneData(InflightContext& ctx) {
    ctx.data->reset(driver);
    std::unordered_map<uint32_t, uint32_t> primitiveIdToVB;
}
