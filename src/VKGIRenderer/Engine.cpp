#include "Engine.h"

using namespace VKGIRenderer;

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

Entity *Engine::createEntity() {
    return entities.construct();
}

Scene *Engine::createScene(Renderer *renderer) {
    return scenes.construct(renderer);
}

Material *Engine::createMaterial() {
    return materials.construct();
}

std::string VKGIRenderer::Engine::assetPath(const std::string& assetName) {
    auto path = assetBasePath / assetName;
    return path.string();
}

Mesh *Engine::createMesh() {
    return meshes.construct();
}