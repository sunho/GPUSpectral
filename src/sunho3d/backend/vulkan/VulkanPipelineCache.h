#pragma once

#include <vulkan/vulkan.h>

#include <map>
#include <unordered_map>
#include <vector>

#include "../../utils/GCPool.h"
#include "VulkanDevice.h"
#include "VulkanHandles.h"

struct VulkanPipelineKey {
    AttributeArray attributes;
    size_t attributeCount;
    VulkanProgram *program;
    Viewport viewport;
    VkRenderPass renderPass;

    bool operator<(const VulkanPipelineKey &other) const {
        return std::tie(attributeCount, program, viewport, attributes, renderPass) < std::tie(other.attributeCount, other.program, other.viewport, other.attributes, other.renderPass);
    }
};

static bool operator<(const VkDescriptorImageInfo &lhs, const VkDescriptorImageInfo &rhs) {
    return std::tie(lhs.imageLayout, lhs.imageView, lhs.sampler) < std::tie(rhs.imageLayout, rhs.imageView, rhs.sampler);
};

struct VulkanDescriptor {
    static constexpr uint32_t UBUFFER_BINDING_COUNT = 8;
    static constexpr uint32_t SAMPLER_BINDING_COUNT = 8;
    static constexpr uint32_t TARGET_BINDING_COUNT = 8;
    static constexpr uint32_t STORAGE_BINDING_COUNT = 8;

    std::array<VkDescriptorImageInfo, SAMPLER_BINDING_COUNT> samplers;
    std::array<VkDescriptorImageInfo, TARGET_BINDING_COUNT> inputAttachments;
    std::array<VkBuffer, UBUFFER_BINDING_COUNT> uniformBuffers;
    std::array<VkDeviceSize, UBUFFER_BINDING_COUNT> uniformBufferOffsets;
    std::array<VkDeviceSize, UBUFFER_BINDING_COUNT> uniformBufferSizes;

    bool operator<(const VulkanDescriptor &other) const {
        return std::tie(uniformBuffers, samplers, inputAttachments, uniformBufferOffsets, uniformBufferSizes) < std::tie(other.uniformBuffers, other.samplers, other.inputAttachments, other.uniformBufferOffsets, other.uniformBufferSizes);
    }
};

class VulkanPipelineCache {
  public:
    VulkanPipelineCache(VulkanDevice &device);
    VkPipeline getOrCreatePipeline(const VulkanPipelineKey &key);
    vk::Framebuffer getOrCreateFrameBuffer(vk::RenderPass renderPass,
                                         VulkanSwapChain swapchain, 
                                         VulkanRenderTarget *renderTarget);
    VkRenderPass getOrCreateRenderPass(VulkanSwapChain swapchain, VulkanRenderTarget *renderTarget);
    void bindDescriptor(vk::CommandBuffer cmd, const VulkanDescriptor &key);
    void setDummyTexture(vk::ImageView imageView) {
        dummyImageView = imageView;
    }
    void tick();

  private:
    struct PipelineLayout {
		std::array<VkDescriptorSetLayout, 3> descriptorSetLayout;
		vk::PipelineLayout pipelineLayout;
    };

    void setupLayouts(VulkanPipelineCache::PipelineLayout& layout, bool compute);
    void getOrCreateDescriptors(const VulkanDescriptor &key,
                                std::array<VkDescriptorSet, 3> &descripotrs);


    GCPool<VulkanPipelineKey, VkPipeline> pipelines;
    GCPool<std::pair<VulkanRenderTarget *, VkImageView>, VkFramebuffer> framebuffers;
    GCPool<VulkanRenderTarget *, VkRenderPass> renderpasses;
    GCPool<VulkanDescriptor, std::array<VkDescriptorSet, 3>> descriptorSets;
	VkDescriptorPool descriptorPool;

    VkImageView dummyImageView = VK_NULL_HANDLE;
    VkDescriptorBufferInfo dummyBufferInfo = {};
    VkWriteDescriptorSet dummyBufferWriteInfo = {};
    VkDescriptorImageInfo dummySamplerInfo = {};
    VkWriteDescriptorSet dummySamplerWriteInfo = {};
    VkDescriptorImageInfo dummyTargetInfo = {};
    VkWriteDescriptorSet dummyTargetWriteInfo = {};

    VulkanDevice& device;
    VulkanBufferObject *dummyBuffer;

    PipelineLayout graphicsLayout;
    PipelineLayout computeLayout;
};
