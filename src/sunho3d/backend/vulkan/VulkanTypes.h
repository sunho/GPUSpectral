#pragma once
#include <vulkan/vulkan.hpp>
#include "../DriverTypes.h"

struct VulkanTexture;

struct VulkanPipeline {
    vk::Pipeline pipeline{};
    vk::DescriptorSetLayout descriptorLayout{};
    vk::PipelineLayout pipelineLayout{};
};

struct VulkanAttachment {
    VulkanTexture *texture{};
};

vk::Format translateTextureFormat(TextureFormat format);
vk::Format translateElementFormat(ElementType type, bool normalized, bool integer);
vk::BufferUsageFlags translateBufferUsage(BufferUsage usage);
