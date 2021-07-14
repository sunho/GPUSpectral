#pragma once

#include <vulkan/vulkan.h>

#include <map>
#include <unordered_map>
#include <vector>

#include "VulkanContext.h"
#include "VulkanHandles.h"

struct VulkanPipelineKey {
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDescription> bindings;
    VulkanProgram *program;
    Viewport viewport;
    bool operator==(const VulkanPipelineKey &other) const {
        if (bindings.size() != other.bindings.size())
            return false;
        if (attributes.size() != other.attributes.size())
            return false;
        for (size_t i = 0; i < bindings.size(); ++i) {
            if (memcmp(&bindings[i], &other.bindings[i], sizeof(VkVertexInputBindingDescription)) !=
                0)
                return false;
        }
        for (size_t i = 0; i < attributes.size(); ++i) {
            if (memcmp(&attributes[i], &other.attributes[i],
                       sizeof(VkVertexInputAttributeDescription)) != 0)
                return false;
        }
        return program == other.program && viewport == other.viewport;
    }
};

// FIXME: hash more robustly
struct VulkanPipelineKeyHasher {
    std::size_t operator()(const VulkanPipelineKey &k) const {
        return std::hash<VulkanProgram *>()(k.program);
    }
};

static constexpr uint32_t UBUFFER_BINDING_COUNT = 8;
static constexpr uint32_t SAMPLER_BINDING_COUNT = 8;
static constexpr uint32_t TARGET_BINDING_COUNT = 8;

#pragma pack(push, 1)
struct VulkanDescriptorKey {
    std::array<VkBuffer, UBUFFER_BINDING_COUNT> uniformBuffers;
    std::array<VkDescriptorImageInfo, SAMPLER_BINDING_COUNT> samplers;
    std::array<VkDescriptorImageInfo, TARGET_BINDING_COUNT> inputAttachments;
    std::array<VkDeviceSize, UBUFFER_BINDING_COUNT> uniformBufferOffsets;
    std::array<VkDeviceSize, UBUFFER_BINDING_COUNT> uniformBufferSizes;

    bool operator==(const VulkanDescriptorKey &other) const {
        return memcmp(this, &other, sizeof(VulkanDescriptorKey)) == 0;
    }
};
#pragma pack(pop)

template <class A>
class PodHash {
  public:
    size_t operator()(const A &a) const {
        // it is possible to write hash func here char by char without using std::string
        const std::string str =
            std::string(reinterpret_cast<const std::string::value_type *>(&a), sizeof(A));
        return std::hash<std::string>()(str);
    }
};

class VulkanPipelineCache {
  public:
    void init(VulkanContext &contex);
    VkPipeline getOrCreatePipeline(VulkanContext &context, const VulkanPipelineKey &key);
    VkFramebuffer getOrCreateFrameBuffer(VulkanContext &context, VkRenderPass renderPass,
                                         VulkanRenderTarget *renderTarget);
    VkRenderPass getOrCreateRenderPass(VulkanContext &context, VulkanRenderTarget *renderTarget);
    void bindDescriptors(VulkanContext &context, const VulkanDescriptorKey &key);
    void setDummyTexture(VkImageView imageView) {
        dummyImageView = imageView;
    }
    VkPipelineLayout pipelineLayout;

  private:
    void setupDescriptorLayout(VulkanContext &context);
    void getOrCreateDescriptors(VulkanContext &context, const VulkanDescriptorKey &key,
                                std::array<VkDescriptorSet, 3> &descripotrs);

    VkDescriptorPool descriptorPool;
    std::array<VkDescriptorSetLayout, 3> descriptorSetLayout;
    std::unordered_map<VulkanPipelineKey, VkPipeline, VulkanPipelineKeyHasher> pipelines;
    std::map<std::pair<VulkanRenderTarget *, VkImageView>, VkFramebuffer> framebuffers;
    std::unordered_map<VulkanRenderTarget *, VkRenderPass> renderpasses;
    std::unordered_map<VulkanDescriptorKey, std::array<VkDescriptorSet, 3>,
                       PodHash<VulkanDescriptorKey>>
        descriptorSets;

    VkImageView dummyImageView = VK_NULL_HANDLE;
    VkDescriptorBufferInfo dummyBufferInfo = {};
    VkWriteDescriptorSet dummyBufferWriteInfo = {};
    VkDescriptorImageInfo dummySamplerInfo = {};
    VkWriteDescriptorSet dummySamplerWriteInfo = {};
    VkDescriptorImageInfo dummyTargetInfo = {};
    VkWriteDescriptorSet dummyTargetWriteInfo = {};

    VulkanBufferObject *dummyBuffer;
};
