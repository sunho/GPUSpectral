#include "Renderer.h"

using namespace sunho3d;

Renderer::Renderer(Window* window) : window(window){
    driver.createLogicalDevice(window);
    driver.setupPipeline();
}

Renderer::~Renderer() {
}

void Renderer::run() { 
    window->run([&](){ driver.drawFrame(); });
}

