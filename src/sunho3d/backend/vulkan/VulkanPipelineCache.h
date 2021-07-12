#pragma once

#include "VulkanContext.h"
#include "VulkanHandles.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <map>
#include <unordered_map>

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

// FIXME: hash more robustly
struct KeyHasher
{
  std::size_t operator()(const VulkanPipelineKey& k) const
  {
      return std::hash<VulkanProgram*>()(k.program);
  }
};


class VulkanPipelineCache {
public:
    void init(VulkanContext& contex);
    VkPipeline getOrCreatePipeline(VulkanContext& context, const VulkanPipelineKey& key);
    VkFramebuffer getOrCreateFrameBuffer(VulkanContext& context, VkRenderPass renderPass, VulkanRenderTarget* renderTarget);
    VkRenderPass getOrCreateRenderPass(VulkanContext& context, VulkanRenderTarget* renderTarget);
    VkDescriptorSet getOrCreateDescriptorSet(VulkanContext& context, const std::vector<VkDescriptorSetLayoutBinding>& bindings);
    VkPipelineLayout pipelineLayout;
private:
    VkDescriptorPool descriptorPool;
        VkDescriptorSetLayout descriptorSetLayout;
    std::unordered_map<VulkanPipelineKey, VkPipeline, KeyHasher> pipelines;
    std::map<std::pair<VulkanRenderTarget*, VkImageView>, VkFramebuffer> framebuffers;
    std::unordered_map<VulkanRenderTarget*, VkRenderPass> renderpasses;
};
