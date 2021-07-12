#pragma once

#include <sunho3d/Window.h>

#include <vulkan/vulkan.h>
#include <vector>

#include "VulkanCommands.h"

struct VulkanSwapContext;
struct VulkanSurfaceContext;

struct VulkanContext {
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties physicalDeviceProperties;
    uint32_t graphicsFamily;
    VkQueue graphicsQueue;
    VkCommandPool commandPool;
    VulkanCommands commands;
    VulkanSurfaceContext* surface;
    VulkanSwapContext* currentSwapContext;
    VkRenderPass currentRenderPass;
};

struct VulkanAttachment {
    VkFormat format;
    VkImage image;
    VkImageView view;
};

struct VulkanSwapContext {
    VulkanAttachment attachment;
};

struct VulkanSurfaceContext {
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR format;
    VkExtent2D extent;
    VkQueue presentQueue;
    size_t size;
    std::vector<VkSurfaceFormatKHR> availabeFormats;
    std::vector<VulkanSwapContext> swapContexts;
    uint32_t swapContextIndex{0};
};  

void initContext(VulkanContext& context);

void initSurfaceContext(VulkanContext& context, VulkanSurfaceContext& surface, sunho3d::Window *window);
void pickPhysicalDevice(VulkanContext& context, VulkanSurfaceContext& surface);
void createLogicalDevice(VulkanContext& context, VulkanSurfaceContext& surface);
void createSwapChain(VulkanContext& context, VulkanSurfaceContext& surface, sunho3d::Window *window);
void populateSwapContexts(VulkanContext& context, VulkanSurfaceContext& surface);

void destroyContext(VulkanContext& context, VulkanSurfaceContext& surface);
