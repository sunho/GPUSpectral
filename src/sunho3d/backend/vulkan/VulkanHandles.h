#pragma once

#include "../Handles.h"
#include "VulkanBuffer.h"
#include "VulkanContext.h"
#include "VulkanTexture.h"

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
    explicit VulkanIndexBuffer(VulkanContext &ctx, uint32_t count)
        : HwIndexBuffer(count) {
        buffer = new VulkanBufferObject(ctx, count * 2, BufferUsage::INDEX);
    }

    ~VulkanIndexBuffer() {
        delete buffer;
    }
    VulkanBufferObject *buffer;
};

static VkShaderModule createShaderModule(VulkanContext &context, const char *code,
                                         uint32_t codeSize) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code);
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

struct VulkanProgram : public HwProgram {
    VulkanProgram() = default;
    explicit VulkanProgram(VulkanContext &ctx, const Program &program)
        : HwProgram(program) {
        if (program.type == ProgramType::PIPELINE) {
            vertex = createShaderModule(ctx, program.vertex().data(), program.vertex().size());
            fragment = createShaderModule(ctx, program.frag().data(), program.frag().size());
        } else {
            compute = createShaderModule(ctx, program.compute().data(), program.compute().size());
        }
    }

    VkShaderModule vertex;
    VkShaderModule fragment;
    VkShaderModule compute;
};

struct VulkanRenderTarget : public HwRenderTarget {
    VulkanRenderTarget() = default;
    explicit VulkanRenderTarget(uint32_t w, uint32_t h, VulkanAttachment depth)
        : HwRenderTarget(w, h), surface(true), depth(depth) {
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
            color[i].format = texture->vkFormat;
            color[i].view = texture->view;
        }

        const VulkanAttachment &depthSpec = depth;
        VulkanTexture *depthTexture = depth.texture;
        if (depthTexture) {
            depth.view = depthTexture->view;
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
    explicit VulkanUniformBuffer(VulkanContext &ctx, uint32_t size)
        : HwUniformBuffer(size) {
        buffer = new VulkanBufferObject(ctx, size, BufferUsage::UNIFORM);
    }
    ~VulkanUniformBuffer() {
        delete buffer;
    }
    VulkanBufferObject *buffer;
};
