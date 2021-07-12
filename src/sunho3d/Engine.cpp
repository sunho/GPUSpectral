#include "Engine.h"

using namespace sunho3d;

Engine::Engine() {
}

Engine::~Engine() {
}

Window *Engine::createWindow(size_t width, size_t height) { 
    auto window = std::make_unique<Window>(windows.getNextId(), width, height);
    auto out = window.get();
    windows.add(std::move(window));
    return out;
}

Renderer *Engine::createRenderer(Window* window, Scene* scene) {
    auto renderer = std::make_unique<Renderer>(renderers.getNextId(), window, scene);
    auto out = renderer.get();
    renderers.add(std::move(renderer));
    return out;
}

Entity *Engine::createEntity() {
    auto entity = std::make_unique<Entity>(renderers.getNextId());
    auto out = entity.get();
    entities.add(std::move(entity));
    return out;
}

Scene* Engine::createScene() {
    auto scene = std::make_unique<Scene>(scenes.getNextId());
    auto out = scene.get();
    scenes.add(std::move(scene));
    return out;
}
