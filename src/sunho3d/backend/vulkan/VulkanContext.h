#pragma once

#include <sunho3d/Window.h>

#include <vulkan/vulkan.h>
#include <vector>

struct VulkanContext {
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties physicalDeviceProperties;
    uint32_t graphicsFamily;
    VkQueue graphicsQueue;
    VkCommandPool commandPool;
    VkRenderPass currentRenderPass;
};

struct VulkanAttachment {
    VkFormat format;
    VkImage image;
    VkImageView view;
};

struct VulkanSwapContext {
    VulkanAttachment attachment;
    VkCommandBuffer commands;
};

struct VulkanSurfaceContext {
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR format;
    VkExtent2D extent;
    VkQueue presentQueue;
    size_t size;
    VulkanSwapContext* currentContext;
    std::vector<VkSurfaceFormatKHR> availabeFormats;
    std::vector<VulkanSwapContext> swapContexts;
    int swapContextIndex{0};
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
};

void initContext(VulkanContext& context);

void initSurfaceContext(VulkanContext& context, VulkanSurfaceContext& surface, sunho3d::Window *window);
void pickPhysicalDevice(VulkanContext& context, VulkanSurfaceContext& surface);
void createLogicalDevice(VulkanContext& context, VulkanSurfaceContext& surface);
void createSwapChain(VulkanContext& context, VulkanSurfaceContext& surface, sunho3d::Window *window);
void populateSwapContexts(VulkanContext& context, VulkanSurfaceContext& surface);

void destroyContext(VulkanContext& context, VulkanSurfaceContext& surface);
