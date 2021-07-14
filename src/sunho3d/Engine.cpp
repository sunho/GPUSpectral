#include "Engine.h"

using namespace sunho3d;

Engine::Engine() {
}

Engine::~Engine() {
}

Window* Engine::createWindow(size_t width, size_t height) {
    return windows.construct(width, height);
}

Renderer* Engine::createRenderer(Window* window, Scene* scene) {
    return renderers.construct(window, scene);
}

Entity* Engine::createEntity() {
    return entities.construct();
}

Scene* Engine::createScene() {
    return scenes.construct();
}
