#pragma once

#include "backend/vulkan/Driver.h"
#include "Window.h"

namespace sunho3d {

class Renderer {
public:
    Renderer(Window* window);
    ~Renderer();
    
    void run();
private:
    VulkanDriver driver;
    Window* window;
};

}
