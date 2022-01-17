#include "Engine.h"

using namespace GPUSpectral;

Engine::Engine(const std::filesystem::path& basePath, const std::filesystem::path& assetBasePath) : assetBasePath(assetBasePath), basePath(basePath) {
}

Engine::~Engine() {
}

Window *Engine::createWindow(size_t width, size_t height) {
    return windows.construct(width, height);
}

Renderer *Engine::createRenderer(Window *window) {
    return renderers.construct(*this, window);
}

std::string GPUSpectral::Engine::assetPath(const std::string& assetName) {
    auto path = assetBasePath / assetName;
    return path.string();
}

Mesh *Engine::createMesh() {
    return meshes.construct();
}