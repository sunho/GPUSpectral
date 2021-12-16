#pragma once
#include <VkBootstrap.h>
#include <VKGIRenderer/Window.h>
#include <vulkan/vulkan.hpp>
#include <vector>

class VulkanDevice;

struct VulkanSwapChain {
    vk::Format format;
    vk::Image image;
    vk::ImageView view;
};

class VulkanWSI {
public:
    VulkanWSI() = delete;
    VulkanWSI(VKGIRenderer::Window* window, VulkanDevice* device);
    ~VulkanWSI();

    void initSwapchain();

    void beginFrame(vk::Semaphore imageSemaphore);
    void endFrame(vk::Semaphore renderSemaphore);
    VulkanSwapChain currentSwapChain();
    vk::Extent2D getExtent();

    VkSurfaceKHR surface{};
private:
    VKGIRenderer::Window* window;
    VulkanDevice* device;

    vkb::Swapchain vkbSwapchain{};
    vk::Queue presentQueue{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    uint32_t swapchainIndex{};
};