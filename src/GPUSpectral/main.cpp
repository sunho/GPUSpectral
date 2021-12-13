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
    auto scene = loadScene(renderer, "test/scene.xml");
    renderer.setScene(scene);
    renderer.render();

    return 0;
}
