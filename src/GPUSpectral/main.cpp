#define NOMINMAX

#include <Windows.h>
#include <filesystem>
#include "renderer/Renderer.h"
#include "loader/SceneLoader.h"

std::filesystem::path basePath() {
    char path[1024];
    GetModuleFileNameA(0, path, 1024);
    auto out = std::filesystem::path(path);
    return out.parent_path();
}

int main() {
    Renderer renderer(basePath().string());
    auto scene = loadScene(renderer, "coffee/scene.xml");
    RenderConfig config = {
        .width = 800,
        .height = 1000
    }; 
    renderer.setScene(scene, config); 
    int totalSamples = 24800;
    int spp = 256;
    for (int i = 0; i < totalSamples / spp; ++i) {
        renderer.render(spp);
    }

    return 0;
}
