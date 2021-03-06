#pragma once
#include <GPUSpectral/engine/Window.h>
#include <VkBootstrap.h>
#include <vector>
#include <vulkan/vulkan.hpp>

class VulkanDevice;

struct VulkanSwapChain {
    vk::Format format;
    vk::Image image;
    vk::ImageView view;
};

class VulkanWSI {
  public:
    VulkanWSI() = delete;
    VulkanWSI(GPUSpectral::Window* window, VulkanDevice* device);
    ~VulkanWSI();

    void initSwapchain();

    void beginFrame(vk::Semaphore imageSemaphore);
    void endFrame(vk::Semaphore renderSemaphore);
    VulkanSwapChain currentSwapChain();
    vk::Extent2D getExtent();

    VkSurfaceKHR surface{};

  private:
    GPUSpectral::Window* window;
    VulkanDevice* device;

    vkb::Swapchain vkbSwapchain{};
    vk::Queue presentQueue{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    uint32_t swapchainIndex{};
};