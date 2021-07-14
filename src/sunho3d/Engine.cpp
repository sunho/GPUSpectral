#include "Engine.h"

using namespace sunho3d;

Engine::Engine() {
}

Engine::~Engine() {
}

Window *Engine::createWindow(size_t width, size_t height) {
    return windows.construct(width, height);
}

Renderer *Engine::createRenderer(Window *window) {
    return renderers.construct(window);
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
