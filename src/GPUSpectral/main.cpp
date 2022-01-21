#include <GPUSpectral/engine/Engine.h>
#include <GPUSpectral/engine/Loader.h>
#include <GPUSpectral/renderer/PathTracer.h>   

#include <Windows.h>
#include <filesystem>

std::filesystem::path basePath() {
    char path[1024];
    GetModuleFileNameA(0, path, 1024);
    auto out = std::filesystem::path(path);
    return out.parent_path();
}


int main() {

    GPUSpectral::Engine engine(basePath(), basePath() / "assets");
    engine.init(500, 500);
    auto pathTracer = std::make_unique<GPUSpectral::PathTracer>(engine.getRenderer());
    engine.getRenderer().setRendererImpl(std::move(pathTracer));


    auto scene = loadScene(engine, engine.getRenderer(), (basePath() / "assets" / "scenes" / "test3" / "scene.xml").string());
    glm::vec3 cameraPos = glm::vec3(0.0, 1.0, 13.0f);
    glm::vec3 origin = glm::vec3(0.0, 1.0, 0.0);

    engine.getWindow().run([&]() {
        engine.getRenderer().run(scene);
    });
    
    return 0;
}
