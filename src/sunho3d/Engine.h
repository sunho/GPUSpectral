#pragma once

#include <memory>
#include <list>
#include <stdlib.h>
#include "Window.h"
#include "Renderer.h"

namespace sunho3d {

class Engine {
public:
    Engine();
    ~Engine();
    
    Window* createWindow(size_t width, size_t height);
    Renderer* createRenderer(Window* window);
private:
    std::list<std::unique_ptr<Window>> windows;
    std::list<std::unique_ptr<Renderer>> renderers;
};

}
