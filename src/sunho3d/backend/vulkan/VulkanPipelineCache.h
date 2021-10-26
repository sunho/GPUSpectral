#pragma once

#include <vulkan/vulkan.h>

#include <map>
#include <unordered_map>
#include <vector>

#include "../../utils/GCPool.h"
#include "VulkanDevice.h"
#include "VulkanHandles.h"

template <>
inline uint64_t hashStruct<VulkanAttachment>(const VulkanAttachment& attachment) {
    uint64_t seed = hashBase(attachment.valid);
    seed ^= hashStruct(attachment.view);
    seed ^= hashBase(attachment.format);
    return hashBase(seed);
}

template <>
inline uint64_t hashStruct<VulkanAttachments>(const VulkanAttachments& attachment) {
    uint64_t seed = hashStruct(attachment.depth);
    for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
			seed ^= hashStruct(attachment.colors[i]);
    }
    return hashBase(seed);
}

struct VulkanPipelineKey {
    AttributeArray attributes;
    size_t attributeCount;
    VulkanProgram *program;
    Viewport viewport;
    VkRenderPass renderPass;
    DepthTest depthTest;

    bool operator==(const VulkanPipelineKey &other) const {
        return attributes == other.attributes && attributeCount == other.attributeCount && program == other.program
        &&viewport == other.viewport && renderPass == other.renderPass && depthTest == other.depthTest;
    }
};

template <>
inline uint64_t hashStruct<Attribute>(const Attribute& attribute) {
    uint64_t seed = hashStruct(attribute.flags);
    seed ^= hashStruct(attribute.index);
    seed ^= hashStruct(attribute.offset);
    seed ^= hashStruct(attribute.stride);
    seed ^= hashStruct(attribute.type);
    seed ^= hashStruct(attribute.flags);
    return hashBase(seed);
}

template <>
inline uint64_t hashStruct<VulkanPipelineKey>(const VulkanPipelineKey& pipelineKey) {
    uint64_t seed = hashStruct(pipelineKey.attributeCount);

    for (size_t i = 0; i < MAX_VERTEX_ATTRIBUTE_COUNT; ++i) {
        seed ^= hashStruct(pipelineKey.attributes[i]);
    }
    seed ^= hashStruct(pipelineKey.program);
    seed ^= hashStruct(pipelineKey.renderPass);
    seed ^= hashStruct(pipelineKey.viewport);
    seed ^= hashStruct(pipelineKey.depthTest);
    return hashBase(seed);
}

static bool operator==(const VkDescriptorImageInfo &lhs, const VkDescriptorImageInfo &rhs) {
    return lhs.imageLayout == rhs.imageLayout && lhs.imageView == rhs.imageView && lhs.sampler == rhs.sampler;
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

    bool operator==(const VulkanDescriptor &other) const = default;
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
    void tick();

  private:
    struct PipelineLayout {
		std::array<VkDescriptorSetLayout, 3> descriptorSetLayout;
		vk::PipelineLayout pipelineLayout;
    };

    void setupLayouts(VulkanPipelineCache::PipelineLayout& layout, bool compute);
    void getOrCreateDescriptors(const VulkanDescriptor &key,
                                std::array<VkDescriptorSet, 3> &descripotrs);


    struct VulkanPipelineKeyHasher
    {
        std::size_t operator()(const VulkanPipelineKey& k) const {
            return hashStruct(k);
        }
    };
    struct FrameBufferKeyHasher {
        std::size_t operator()(const std::pair<VkRenderPass, VkImageView>& k) const {
            return hashStruct(k.first) ^ hashStruct(k.second);
        }
    };
    struct RenderPassKeyHasher {
        std::size_t operator()(const VulkanAttachments& k) const
        {
            return hashStruct<VulkanAttachments>(k);
        }
    };
    struct DescriptorKeyHasher {
        std::size_t operator()(const VulkanDescriptor& k) const {
            return hashStruct(k);
        }
    };
    GCPool<VulkanPipelineKey, VkPipeline, VulkanPipelineKeyHasher> pipelines;
    GCPool<std::pair<VkRenderPass, VkImageView>, VkFramebuffer, FrameBufferKeyHasher> framebuffers;
    GCPool<VulkanAttachments, VkRenderPass, RenderPassKeyHasher> renderpasses;
    GCPool<VulkanDescriptor, std::array<VkDescriptorSet, 3>, DescriptorKeyHasher> descriptorSets;
	VkDescriptorPool descriptorPool;

    std::unique_ptr<VulkanTexture> dummyImage;
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
