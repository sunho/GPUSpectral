#pragma once

#include "VulkanContext.h"
#include "VulkanHandles.h"
#include <vulkan/vulkan.h>
#include <vector>

struct VulkanPipelineKey {
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDescription> bindings;
    VulkanProgram* program;
    Viewport viewport;
    bool operator==(const VulkanPipelineKey& other) const {
        if (bindings.size() != other.bindings.size())
            return false;
        if (attributes.size() != other.attributes.size())
            return false;
        for (size_t i = 0; i < bindings.size(); ++i) {
            if (memcmp(&bindings[i], &other.bindings[i], sizeof(VkVertexInputBindingDescription)) != 0)
                return false;
        }
        for (size_t i = 0; i < attributes.size(); ++i) {
            if (memcmp(&attributes[i], &other.attributes[i], sizeof(VkVertexInputAttributeDescription)) != 0)
                return false;
        }
        return program == other.program && viewport == other.viewport;
    }
};

class VulkanPipelineCache {
public:
    VkPipeline createPipeline(VulkanContext& context, const VulkanPipelineKey& key);
    VkFramebuffer createFrameBuffer(VulkanContext& context, VkRenderPass renderPass, VulkanRenderTarget* renderTarget);
    VkRenderPass createRenderPass(VulkanContext& context, VulkanRenderTarget* renderTarget);
};
