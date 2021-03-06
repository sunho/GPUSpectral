#pragma once

#include <vulkan/vulkan.h>

#include <map>
#include <unordered_map>
#include <vector>

#include "../../utils/GCPool.h"
#include "VulkanDevice.h"
#include "VulkanHandles.h"

template <>
inline uint64_t hashStruct<ProgramHash>(const ProgramHash& hash) {
    return hashBase(hash);
}

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

struct VulkanRTPipelineState {
    VulkanProgram* raygenGroup;
    std::vector<VulkanProgram*> missGroups;
    std::vector<VulkanProgram*> hitGroups;
    std::vector<VulkanProgram*> callableGroups;
    ProgramParameterLayout parameterLayout;

    size_t getGroupCount() const {
        return 1 + missGroups.size() + hitGroups.size() + callableGroups.size();
    }

    bool operator==(const VulkanRTPipelineState& other) const {
        if (raygenGroup->program.hash != other.raygenGroup->program.hash) {
            return false;
        }
        if (missGroups.size() != other.missGroups.size()) {
            return false;
        }
        if (hitGroups.size() != other.hitGroups.size()) {
            return false;
        }
        if (callableGroups.size() != other.callableGroups.size()) {
            return false;
        }
        if (parameterLayout != other.parameterLayout) {
            return false;
        }
        for (size_t i = 0; i < missGroups.size(); ++i) {
            if (missGroups[i]->program.hash != other.missGroups[i]->program.hash) {
                return false;
            }
        }
        for (size_t i = 0; i < hitGroups.size(); ++i) {
            if (hitGroups[i]->program.hash != other.hitGroups[i]->program.hash) {
                return false;
            }
        }
        for (size_t i = 0; i < callableGroups.size(); ++i) {
            if (callableGroups[i]->program.hash != other.callableGroups[i]->program.hash) {
                return false;
            }
        }
        return true;
    }
};

template <>
inline uint64_t hashStruct<VulkanRTPipelineState>(const VulkanRTPipelineState& state) {
    uint64_t seed = hashStruct(state.raygenGroup->program.hash);
    for (auto& p : state.hitGroups) {
        seed ^= hashStruct(p->program.hash);
    }
    for (auto& p : state.missGroups) {
        seed ^= hashStruct(p->program.hash);
    }
    for (auto& p : state.callableGroups) {
        seed ^= hashStruct(p->program.hash);
    }
    seed ^= hashStruct(state.parameterLayout);
    return hashBase(seed);
}

struct VulkanPipelineState {
    AttributeArray attributes;
    size_t attributeCount;
    VulkanProgram* vertex;
    VulkanProgram* fragment;
    Viewport viewport;
    VkRenderPass renderPass;
    size_t attachmentCount;
    DepthTest depthTest;
    ProgramParameterLayout parameterLayout;

    bool operator==(const VulkanPipelineState& other) const {
        return parameterLayout == other.parameterLayout && attachmentCount == other.attachmentCount && attributes == other.attributes && attributeCount == other.attributeCount && vertex->program.hash == other.vertex->program.hash &&
               fragment->program.hash == other.fragment->program.hash && viewport == other.viewport && renderPass == other.renderPass && depthTest == other.depthTest;
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
inline uint64_t hashStruct<VulkanPipelineState>(const VulkanPipelineState& pipelineKey) {
    uint64_t seed = hashStruct(pipelineKey.attributeCount);

    for (size_t i = 0; i < MAX_VERTEX_ATTRIBUTE_COUNT; ++i) {
        seed ^= hashStruct(pipelineKey.attributes[i]);
    }
    seed ^= hashStruct(pipelineKey.vertex->program.hash);
    seed ^= hashStruct(pipelineKey.fragment->program.hash);
    seed ^= hashStruct(pipelineKey.renderPass);
    seed ^= hashStruct(pipelineKey.viewport);
    seed ^= hashStruct(pipelineKey.attachmentCount);
    seed ^= hashStruct(pipelineKey.depthTest);
    seed ^= hashStruct(pipelineKey.parameterLayout);
    return hashBase(seed);
}

static bool operator==(const VkDescriptorImageInfo& lhs, const VkDescriptorImageInfo& rhs) {
    return lhs.imageLayout == rhs.imageLayout && lhs.imageView == rhs.imageView && lhs.sampler == rhs.sampler;
};

struct VulkanBinding {
    vk::DescriptorType type;
    uint32_t set;
    uint32_t binding;
    uint32_t arraySize;
    std::vector<vk::DescriptorBufferInfo> bufferInfo;
    std::vector<vk::DescriptorImageInfo> imageInfo;
    std::vector<vk::AccelerationStructureKHR> tlasInfo;
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

static inline VkStridedDeviceAddressRegionKHR getSbtEntryStridedDeviceAddressRegion(VulkanDevice& device, vk::Buffer buffer, uint32_t handleCount) {
    VkStridedDeviceAddressRegionKHR stridedDeviceAddressRegion{};
    stridedDeviceAddressRegion.deviceAddress = device.getBufferDeviceAddress(buffer);
    stridedDeviceAddressRegion.stride = device.shaderGroupHandleSizeAligned;
    stridedDeviceAddressRegion.size = handleCount * device.shaderGroupHandleSizeAligned;
    return stridedDeviceAddressRegion;
}

// TODO RAII
struct VulkanShaderBindingTable {
    VulkanShaderBindingTable() = default;
    VulkanShaderBindingTable(VulkanDevice& device, size_t handlesCount) {
        buffer = new VulkanBufferObject(device, device.shaderGroupHandleSize * handlesCount, BufferUsage::SBT | BufferUsage::BDA, BufferType::HOST_COHERENT);
        stridedDeviceAddressRegion = getSbtEntryStridedDeviceAddressRegion(device, buffer->buffer, handlesCount);
    }
    VkStridedDeviceAddressRegionKHR stridedDeviceAddressRegion{};
    VulkanBufferObject* buffer{ nullptr };
};

struct VulkanShaderBindingTables {
    VulkanShaderBindingTable raygen;
    VulkanShaderBindingTable miss;
    VulkanShaderBindingTable hit;
    VulkanShaderBindingTable callable;
};

class VulkanPipelineCache {
  public:
    VulkanPipelineCache(VulkanDevice& device);
    VulkanPipeline getOrCreateGraphicsPipeline(const VulkanPipelineState& state);
    VulkanPipeline getOrCreateComputePipeline(const VulkanProgram& program);
    VulkanPipeline getOrCreateRTPipeline(const VulkanRTPipelineState& state);
    VulkanShaderBindingTables getOrCreateSBT(const VulkanRTPipelineState& state);

    vk::Framebuffer getOrCreateFrameBuffer(vk::RenderPass renderPass,
                                           VulkanSwapChain swapchain,
                                           VulkanRenderTarget* renderTarget);
    VkRenderPass getOrCreateRenderPass(VulkanSwapChain swapchain, VulkanRenderTarget* renderTarget);
    void bindDescriptor(vk::CommandBuffer cmd, const vk::PipelineBindPoint& bindPoint, const ProgramParameterLayout& layout, const VulkanBindings& bindings);
    void tick();

  private:
    struct PipelineLayout {
        std::array<vk::DescriptorSetLayout, ProgramParameterLayout::MAX_SET> descriptorSetLayout;
        vk::PipelineLayout pipelineLayout;
    };

    using DescriptorSets = std::array<vk::DescriptorSet, ProgramParameterLayout::MAX_SET>;

    VulkanDescriptorAllocator& currentDescriptorAllocator() {
        return descriptorAllocators[currentFrame % descriptorAllocators.size()];
    }
    PipelineLayout getOrCreatePipelineLayout(const ProgramParameterLayout& layout);

    struct KeyHasher {
        std::size_t operator()(const VulkanRTPipelineState& k) const {
            return hashStruct<VulkanRTPipelineState>(k);
        }
        std::size_t operator()(const VulkanAttachments& k) const {
            return hashStruct<VulkanAttachments>(k);
        }
        std::size_t operator()(const std::pair<VkRenderPass, VkImageView>& k) const {
            return hashStruct(k.first) ^ hashStruct(k.second);
        }
        std::size_t operator()(const VulkanPipelineState& k) const {
            return hashStruct(k);
        }
        std::size_t operator()(const ProgramParameterLayout& k) const {
            return hashStruct<ProgramParameterLayout>(k);
        }
        std::size_t operator()(const BindingMap& k) const {
            return hashStruct(k);
        }
        std::size_t operator()(const ProgramHash& k) const {
            return k;
        }
    };

    GCPool<ProgramHash, VkPipeline, KeyHasher> computePipelines;
    GCPool<VulkanRTPipelineState, VkPipeline, KeyHasher> rtPipelines;
    GCPool<VulkanRTPipelineState, VulkanShaderBindingTables, KeyHasher> rtSBTs;
    GCPool<VulkanPipelineState, VkPipeline, KeyHasher> graphicsPipelines;
    GCPool<std::pair<VkRenderPass, VkImageView>, VkFramebuffer, KeyHasher> framebuffers;
    GCPool<VulkanAttachments, VkRenderPass, KeyHasher> renderpasses;
    GCPool<ProgramParameterLayout, PipelineLayout, KeyHasher> pipelineLayouts;
    std::array<VulkanDescriptorAllocator, 3> descriptorAllocators;
    size_t currentFrame{ 0 };
    VkDescriptorPool descriptorPool;

    std::unique_ptr<VulkanTexture> dummyImage;
    VkDescriptorBufferInfo dummyBufferInfo = {};
    VkWriteDescriptorSet dummyBufferWriteInfo = {};
    VkDescriptorImageInfo dummySamplerInfo = {};
    VkWriteDescriptorSet dummySamplerWriteInfo = {};
    VkDescriptorImageInfo dummyTargetInfo = {};
    VkWriteDescriptorSet dummyTargetWriteInfo = {};

    VulkanDevice& device;
    VulkanBufferObject* dummyBuffer;
};
