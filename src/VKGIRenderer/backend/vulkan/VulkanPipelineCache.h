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
    for (size_t i = 0; i < RenderAttachments::MAX_MRT_NUM; ++i) {
			seed ^= hashStruct(attachment.colors[i]);
    }
    return hashBase(seed);
}

struct VulkanPipelineState {
    AttributeArray attributes;
    size_t attributeCount;
    VulkanProgram *program;
    Viewport viewport;
    VkRenderPass renderPass;
    size_t attachmentCount;
    DepthTest depthTest;

    bool operator==(const VulkanPipelineState &other) const {
        return attachmentCount == other.attachmentCount && attributes == other.attributes && attributeCount == other.attributeCount && program->program.hash == other.program->program.hash
            && viewport == other.viewport && renderPass == other.renderPass && depthTest == other.depthTest;
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
inline uint64_t hashStruct<VulkanPipelineState>(const VulkanPipelineState &pipelineKey) {
    uint64_t seed = hashStruct(pipelineKey.attributeCount);

    for (size_t i = 0; i < MAX_VERTEX_ATTRIBUTE_COUNT; ++i) {
        seed ^= hashStruct(pipelineKey.attributes[i]);
    }
    seed ^= hashStruct(pipelineKey.program);
    seed ^= hashStruct(pipelineKey.renderPass);
    seed ^= hashStruct(pipelineKey.viewport);
    seed ^= hashStruct(pipelineKey.attachmentCount);
    seed ^= hashStruct(pipelineKey.depthTest);
    return hashBase(seed);
}

static bool operator==(const VkDescriptorImageInfo &lhs, const VkDescriptorImageInfo &rhs) {
    return lhs.imageLayout == rhs.imageLayout && lhs.imageView == rhs.imageView && lhs.sampler == rhs.sampler;
};

struct VulkanBinding {
    vk::DescriptorType type;
    uint32_t set;
    uint32_t binding;
    uint32_t arraySize;
    std::vector<vk::DescriptorBufferInfo> bufferInfo;
    std::vector<vk::DescriptorImageInfo> imageInfo;
};

using VulkanBindings = std::vector<VulkanBinding>;

class VulkanDescriptorAllocator {
  public:
    struct PoolSizes {
        std::vector<std::pair<vk::DescriptorType, float>> sizes = {
            { vk::DescriptorType::eSampler, 0.5f },
            { vk::DescriptorType::eCombinedImageSampler, 4.f },
            { vk::DescriptorType::eSampledImage, 4.f },
            { vk::DescriptorType::eStorageImage, 1.f },
            { vk::DescriptorType::eUniformTexelBuffer, 1.f },
            { vk::DescriptorType::eStorageTexelBuffer, 1.f },
            { vk::DescriptorType::eUniformBuffer, 2.f },
            { vk::DescriptorType::eStorageBuffer, 2.f },
            { vk::DescriptorType::eUniformBufferDynamic, 1.f },
            { vk::DescriptorType::eStorageBufferDynamic, 1.f },
            { vk::DescriptorType::eInputAttachment, 0.5f }
        };
    };

    void resetPools();
    vk::DescriptorSet allocate(vk::DescriptorSetLayout layout);

    void init(vk::Device newDevice);

    void cleanup();

    vk::Device device;

  private:
    vk::DescriptorPool grabPool();

    vk::DescriptorPool currentPool{ nullptr };
    PoolSizes descriptorSizes;
    std::vector<vk::DescriptorPool> usedPools;
    std::vector<vk::DescriptorPool> freePools;
};

struct VulkanPipeline {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
};

class VulkanPipelineCache {
  public:
    VulkanPipelineCache(VulkanDevice &device);
    VulkanPipeline getOrCreateGraphicsPipeline(const VulkanPipelineState &state);
    VulkanPipeline getOrCreateComputePipeline(const VulkanProgram &program);

    vk::Framebuffer getOrCreateFrameBuffer(vk::RenderPass renderPass,
                                         VulkanSwapChain swapchain, 
                                         VulkanRenderTarget *renderTarget);
    VkRenderPass getOrCreateRenderPass(VulkanSwapChain swapchain, VulkanRenderTarget *renderTarget);
    void bindDescriptor(vk::CommandBuffer cmd, const VulkanProgram &program, const VulkanBindings& bindings);
    void tick();

  private:
    struct PipelineLayout {
        std::array<vk::DescriptorSetLayout, ProgramParameterLayout::MAX_SET> descriptorSetLayout;
        vk::PipelineLayout pipelineLayout;
    };

    using DescriptorSets = std::array<vk::DescriptorSet, ProgramParameterLayout::MAX_SET>;

    VulkanDescriptorAllocator &currentDescriptorAllocator() {
        return descriptorAllocators[currentFrame % descriptorAllocators.size()];
    }
    PipelineLayout getOrCreatePipelineLayout(const ProgramParameterLayout &layout, bool compute);
   
    struct KeyHasher {
        std::size_t operator()(const VulkanAttachments& k) const {
            return hashStruct<VulkanAttachments>(k);
        }
        std::size_t operator()(const std::pair<VkRenderPass, VkImageView> &k) const {
            return hashStruct(k.first) ^ hashStruct(k.second);
        }
        std::size_t operator()(const VulkanPipelineState &k) const {
            return hashStruct(k);
        }
        std::size_t operator()(const ProgramParameterLayout &k) const {
            return hashStruct<ProgramParameterLayout>(k);
        }
        std::size_t operator()(const BindingMap &k) const {
            return hashStruct(k);
        }
        std::size_t operator()(const ProgramHash &k) const {
            return k;
        }
    };

    GCPool<ProgramHash, VkPipeline, KeyHasher> computePipelines;
    GCPool<VulkanPipelineState, VkPipeline, KeyHasher> graphicsPipelines;
    GCPool<std::pair<VkRenderPass, VkImageView>, VkFramebuffer, KeyHasher> framebuffers;
    GCPool<VulkanAttachments, VkRenderPass, KeyHasher> renderpasses;
    GCPool<ProgramParameterLayout, PipelineLayout, KeyHasher> pipelineLayouts;
    std::array<VulkanDescriptorAllocator, 2> descriptorAllocators;
    size_t currentFrame{0};
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
};
