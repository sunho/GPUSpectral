#pragma once

#include "../DriverBase.h"
#include "VulkanContext.h"

class VulkanTexture : public HwTexture {
  public:
    VulkanTexture(VulkanContext& context, SamplerType type, TextureUsage usage, uint8_t levels,
                  TextureFormat format, uint32_t width, uint32_t height);
    void update2DImage(VulkanContext& context, const BufferDescriptor& data);
    void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width,
                           uint32_t height, uint32_t miplevel);
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkFormat vkFormat;
    VkSampler sampler;
};
