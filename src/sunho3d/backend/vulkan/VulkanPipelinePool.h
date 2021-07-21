#pragma once

#include <vulkan/vulkan.h>

#include <map>
#include <unordered_map>
#include <vector>

#include "../../utils/GCPool.h"
#include "VulkanContext.h"
#include "VulkanHandles.h"

struct VulkanPipeline {
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptorLayout;
    VkPipelineLayout pipelineLayout;
};

struct VulkanPipelineParams {
    AttributeArray attributes;
    size_t attributeCount;
    VulkanProgram *program;
    VkRenderPass renderPass;
    Viewport viewport;

    bool operator<(const VulkanPipelineParams &other) const {
        return std::tie(attributeCount, program, viewport, attributes, renderPass) < std::tie(other.attributeCount, other.program, other.viewport, other.attributes, renderPass);
    }
};

static bool operator<(const VkDescriptorImageInfo &lhs, const VkDescriptorImageInfo &rhs) {
    return std::tie(lhs.imageLayout, lhs.imageView, lhs.sampler) < std::tie(rhs.imageLayout, rhs.imageView, rhs.sampler);
};

struct VulkanBind {
    VkDescriptorImageInfo imageInfo;
};

struct VulkanBindings {
    std::array<VulkanBind, Program::NUM_BINDINGS> bindings;
    uint32_t bindingNum;
};

class VulkanPipelinePool {
  public:
    void init(VulkanContext &contex);
    VkPipeline getOrCreatePipeline(VulkanContext &context, const VulkanPipelineParams &params);
    VkPipeline getOrCreateComputePipeline(VulkanContext &context, VulkanProgram* program);
    VkFramebuffer getOrCreateFrameBuffer(VulkanContext &context, VkRenderPass renderPass,
                                         VulkanRenderTarget *renderTarget);
    VkRenderPass getOrCreateRenderPass(VulkanContext &context, VulkanRenderTarget *renderTarget);
    void createDescriptor(VkDscr);
    void tick();

  private:
    VkDescriptorPool descriptorPool;

    GCPool<VulkanPipelineParams, VkPipeline> pipelines;
    GCPool<VulkanProgram *, VkPipeline> computes;
    GCPool<std::pair<VulkanRenderTarget *, VkImageView>, VkFramebuffer> framebuffers;
    GCPool<VulkanRenderTarget *, VkRenderPass> renderpasses;
    
    VulkanContext *context;

};
