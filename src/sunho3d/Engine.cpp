#include "Engine.h"

using namespace sunho3d;

Engine::Engine(const std::filesystem::path& assetBasePath) : assetBasePath(assetBasePath) {
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

std::string sunho3d::Engine::assetPath(const std::string& assetName) {
    auto path = assetBasePath / assetName;
    return path.string();
}

Mesh *Engine::createMesh() {
    return meshes.construct();
}