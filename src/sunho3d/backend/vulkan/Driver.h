#pragma once

#include <sunho3d/Window.h>

#include "Context.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>

namespace sunho3d {

class VulkanDriver {
public:
    VulkanDriver(Window* window);
    ~VulkanDriver();

    void drawFrame();
private:
    void setupDebugMessenger();
    
    void fillCommands();
    void createRenderPass();
    void createPipeline();

    VkShaderModule createShaderModule(const char* code, size_t codeSize);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

    VkDebugUtilsMessengerEXT debugMessenger;
    
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> framebuffers;
    VulkanContext context;
    VulkanSurfaceContext surface;
};

}
