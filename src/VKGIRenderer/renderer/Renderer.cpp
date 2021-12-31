#include "Renderer.h"

#include <VKGIRenderer/utils/HalfFloat/umHalf.h>
#include <tiny_gltf.h>
#include <iostream>

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
