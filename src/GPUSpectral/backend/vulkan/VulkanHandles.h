#pragma once

#include <GPUSpectral/utils/Hash.h>
#include "../Handles.h"
#include "VulkanBuffer.h"
#include "VulkanDevice.h"
#include "VulkanTexture.h"
#include "VulkanTypes.h"

#include <stdexcept>

struct VulkanBufferObject;

static constexpr const size_t VULKAN_VERTEX_BUFFERS_MAX = 12;

struct VulkanVertexBuffer : public HwVertexBuffer {
    VulkanVertexBuffer() = default;
    VulkanVertexBuffer(uint32_t vertexCount, uint8_t attributeCount,
                       const AttributeArray &attributes)
        : HwVertexBuffer(vertexCount, attributeCount, attributes) {
    }
    std::array<VulkanBufferObject *, VULKAN_VERTEX_BUFFERS_MAX> buffers;
};

struct VulkanIndexBuffer : public HwIndexBuffer {
    explicit VulkanIndexBuffer(VulkanDevice &device, uint32_t count)
        : HwIndexBuffer(count), buffer(std::make_unique<VulkanBufferObject>(device, count * 4, BufferUsage::BDA | BufferUsage::INDEX | BufferUsage::STORAGE | BufferUsage::ACCELERATION_STRUCTURE_INPUT, BufferType::DEVICE)) {
    }

    std::unique_ptr<VulkanBufferObject> buffer;
};

struct VulkanProgram : public HwProgram {
    VulkanProgram() = default;
    explicit VulkanProgram(VulkanDevice &device, const Program &program);

    VkShaderModule shaderModule;

  private:
    void parseParameterLayout(const CompiledCode &code);
};

struct VulkanAttachment {
    uint32_t valid{};
    VkImageView view{};
    VkFormat format{};

    VulkanAttachment() = default;
    VulkanAttachment(VulkanTexture *texture)
        : view(texture->view), format((VkFormat)texture->vkFormat), valid(true) {
    }
    bool operator==(const VulkanAttachment &other) const = default;
};

struct VulkanAttachments {
    std::array<VulkanAttachment, RenderAttachments::MAX_MRT_NUM> colors;
    VulkanAttachment depth{};
    bool operator==(const VulkanAttachments &other) const = default;
};

struct VulkanRenderTarget : public HwRenderTarget {
    VulkanRenderTarget();
    VulkanRenderTarget(uint32_t w, uint32_t h, VulkanAttachments attachments);
    vk::Extent2D getExtent(VulkanDevice &device) const;

    bool surface;
    VulkanAttachments attachments;
    size_t attachmentCount;
};

struct VulkanPrimitive : public HwPrimitive {
    explicit VulkanPrimitive(PrimitiveMode mode)
        : HwPrimitive(mode) {
    }
    VulkanVertexBuffer *vertex{};
    VulkanIndexBuffer *index{};
};

struct VulkanFence : public HwFence {
    VulkanFence() = default;
    explicit VulkanFence(VulkanDevice &device) {
        auto fi = vk::FenceCreateInfo();
        fi.flags = vk::FenceCreateFlagBits::eSignaled;
        fence = device.device.createFence(fi);
    }
    vk::Fence fence{};
};