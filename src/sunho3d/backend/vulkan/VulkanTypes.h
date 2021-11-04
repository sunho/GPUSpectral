#pragma once
#include <vulkan/vulkan.hpp>
#include "../DriverTypes.h"
#include "../PipelineState.h"

struct VulkanTexture;

vk::Format translateTextureFormat(TextureFormat format);
vk::Format translateElementFormat(ElementType type, bool normalized, bool integer);
vk::CompareOp translateCompareOp(CompareOp op);
vk::BufferUsageFlags translateBufferUsage(BufferUsage usage);
vk::DescriptorType translateDescriptorType(ProgramParameterType type);