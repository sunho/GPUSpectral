#pragma once
#include <vulkan/vulkan.hpp>
#include "../DriverTypes.h"
#include "../PipelineState.h"

struct VulkanTexture;

struct VulkanPipeline {
    vk::Pipeline pipeline{};
    vk::DescriptorSetLayout descriptorLayout{};
    vk::PipelineLayout pipelineLayout{};
};

vk::Format translateTextureFormat(TextureFormat format);
vk::Format translateElementFormat(ElementType type, bool normalized, bool integer);
vk::CompareOp translateCompareOp(CompareOp op);
vk::BufferUsageFlags translateBufferUsage(BufferUsage usage);
