#pragma once

#include "../Handles.h"
#include "VulkanBuffer.h"
#include "VulkanContext.h"

struct VulkanBufferObject;

static constexpr const size_t VULKAN_VERTEX_BUFFERS_MAX = 12;

struct VulkanVertexBuffer : public HwVertexBuffer {
    VulkanVertexBuffer() = default;
    explicit VulkanVertexBuffer(uint32_t vertexCount, uint8_t attributeCount,
                                const AttributeArray& attributes)
        : HwVertexBuffer(vertexCount, attributeCount, attributes) {
    }
    std::array<VulkanBufferObject*, VULKAN_VERTEX_BUFFERS_MAX> buffers;
};

struct VulkanIndexBuffer : public HwIndexBuffer {
    VulkanIndexBuffer() = default;
    explicit VulkanIndexBuffer(uint32_t count) : HwIndexBuffer(count) {
    }
    void allocate(VulkanContext& ctx) {
        buffer = new VulkanBufferObject(count * 2);
        buffer->allocate(ctx, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }
    VulkanBufferObject* buffer;
};

static VkShaderModule createShaderModule(VulkanContext& context, const char* code,
                                         uint32_t codeSize) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code);
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

struct VulkanProgram : public HwProgram {
    VulkanProgram() = default;
    explicit VulkanProgram(const Program& program) : HwProgram(program) {
    }
    void compile(VulkanContext& ctx) {
        auto& vertexCode = program.codes[0];
        auto& fragCode = program.codes[1];
        vertex = createShaderModule(ctx, vertexCode.data(), vertexCode.size());
        fragment = createShaderModule(ctx, fragCode.data(), fragCode.size());
    }
    VkShaderModule vertex;
    VkShaderModule fragment;
};

struct VulkanRenderTarget : public HwRenderTarget {
    VulkanRenderTarget() = default;
    explicit VulkanRenderTarget(uint32_t w, uint32_t h, VulkanAttachment depth)
        : HwRenderTarget(w, h), surface(true), depth(depth) {
    }
    explicit VulkanRenderTarget(uint32_t w, uint32_t h, VulkanAttachment color,
                                VulkanAttachment depth)
        : HwRenderTarget(w, h), color(color), depth(depth), surface(false) {
    }
    bool surface;
    VulkanAttachment color;
    VulkanAttachment depth;
};

struct VulkanPrimitive : public HwPrimitive {
    explicit VulkanPrimitive(PrimitiveMode mode) : HwPrimitive(mode) {
    }
    VulkanVertexBuffer* vertex{};
    VulkanIndexBuffer* index{};
};

struct VulkanUniformBuffer : public HwUniformBuffer {
    VulkanUniformBuffer() = default;
    explicit VulkanUniformBuffer(uint32_t size) : HwUniformBuffer(size) {
    }
    void allocate(VulkanContext& ctx) {
        buffer = new VulkanBufferObject(size);
        buffer->allocate(ctx, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }
    VulkanBufferObject* buffer;
};
