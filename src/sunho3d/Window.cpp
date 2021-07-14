#include "Window.h"

#include "Engine.h"

using namespace sunho3d;

Window::Window(size_t width, size_t height)
    : width(width), height(height) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Vulkan window", nullptr, nullptr);
}

Window::~Window() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Window::run(std::function<void()> drawFrame) {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
}

VkSurfaceKHR Window::createSurface(VkInstance instance) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    return surface;
}

size_t Window::getWindowWidth() const {
    int width_, height_;
    glfwGetFramebufferSize(window, &width_, &height_);
    return width_;
}

size_t Window::getWindowHeight() const {
    int width_, height_;
    glfwGetFramebufferSize(window, &width_, &height_);
    return height_;
}
