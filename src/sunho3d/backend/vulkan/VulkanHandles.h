#pragma once

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
    explicit VulkanVertexBuffer(uint32_t vertexCount, uint8_t attributeCount,
                                const AttributeArray &attributes)
        : HwVertexBuffer(vertexCount, attributeCount, attributes) {
    }
    std::array<VulkanBufferObject *, VULKAN_VERTEX_BUFFERS_MAX> buffers;
};

struct VulkanIndexBuffer : public HwIndexBuffer {
    explicit VulkanIndexBuffer(VulkanDevice &device, uint32_t count)
        : HwIndexBuffer(count) {
        buffer = new VulkanBufferObject(device, count * 2, BufferUsage::INDEX);
    }

    ~VulkanIndexBuffer() {
        delete buffer;
    }
    VulkanBufferObject *buffer;
};

static VkShaderModule createShaderModule(VulkanDevice &device, const char *code,
                                         uint32_t codeSize) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code);
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

struct VulkanProgram : public HwProgram {
    VulkanProgram() = default;
    explicit VulkanProgram(VulkanDevice &device, const Program &program)
        : HwProgram(program) {
        if (program.type == ProgramType::PIPELINE) {
            vertex = createShaderModule(device, program.vertex().data(), program.vertex().size());
            fragment = createShaderModule(device, program.frag().data(), program.frag().size());
        } else {
            compute = createShaderModule(device, program.compute().data(), program.compute().size());
        }
    }

    VkShaderModule vertex;
    VkShaderModule fragment;
    VkShaderModule compute;
};

struct VulkanRenderTarget : public HwRenderTarget {
    VulkanRenderTarget()
        : HwRenderTarget(0, 0), surface(true) {
    }
    explicit VulkanRenderTarget(uint32_t w, uint32_t h, std::array<VulkanAttachment, ColorAttachment::MAX_MRT_NUM> color,
                                VulkanAttachment depth)
        : HwRenderTarget(w, h), color(color), depth(depth), surface(false) {
        for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
            const VulkanAttachment &spec = color[i];
            VulkanTexture *texture = spec.texture;
            if (texture == nullptr) {
                continue;
            }
        }
    }
    bool surface;
    std::array<VulkanAttachment, ColorAttachment::MAX_MRT_NUM> color;
    VulkanAttachment depth;
};

struct VulkanPrimitive : public HwPrimitive {
    explicit VulkanPrimitive(PrimitiveMode mode)
        : HwPrimitive(mode) {
    }
    VulkanVertexBuffer *vertex{};
    VulkanIndexBuffer *index{};
};

struct VulkanUniformBuffer : public HwUniformBuffer {
    VulkanUniformBuffer() = default;
    explicit VulkanUniformBuffer(VulkanDevice &ctx, uint32_t size)
        : HwUniformBuffer(size) {
        buffer = new VulkanBufferObject(ctx, size, BufferUsage::UNIFORM);
    }
    ~VulkanUniformBuffer() {
        delete buffer;
    }
    VulkanBufferObject *buffer;
};

struct VulkanFence : public HwFence {
    VulkanFence() = default;
    explicit VulkanFence(VulkanDevice &device) : fence(device.device.createFence(vk::FenceCreateInfo())) {
    }
    vk::Fence fence{};
};