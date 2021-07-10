#include "Engine.h"

using namespace sunho3d;

Engine::Engine() {
}

Engine::~Engine() {
}

Window *Engine::createWindow(size_t width, size_t height) { 
    auto window = std::make_unique<Window>(width, height);
    auto out = window.get();
    windows.push_back(std::move(window));
    return out;
}

Renderer *Engine::createRenderer(Window* window) {
    auto renderer = std::make_unique<Renderer>(window);
    auto out = renderer.get();
    renderers.push_back(std::move(renderer));
    return out;
}
