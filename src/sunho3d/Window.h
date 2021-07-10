#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <functional>

namespace sunho3d {

class Engine;

class Window {
public:
    Window() = delete;
    explicit Window(size_t width, size_t height);
    ~Window();
    
    size_t getWindowWidth() const;
    size_t getWindowHeight() const;
    
    VkSurfaceKHR createSurface(VkInstance instance);

    void run(std::function<void()> drawFrame);

private:
    GLFWwindow* window;

    size_t width;
    size_t height;
};

}
