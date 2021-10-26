#pragma once

#include "../DriverBase.h"
#include "VulkanDevice.h"
#include "VulkanTypes.h"

class VulkanTexture : public HwTexture {
  public:
    VulkanTexture(VulkanDevice &device, SamplerType type, TextureUsage usage, uint8_t levels,
                  TextureFormat format, uint32_t width, uint32_t height);
    ~VulkanTexture();
    void update2DImage(VulkanDevice &device, const BufferDescriptor &data);
    void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width,
                           uint32_t height, uint32_t miplevel);
    vk::Image image;
    vk::ImageView view;
    vk::Format vkFormat;
    vk::Sampler sampler;
    vk::ImageLayout imageLayout;

  private:
    AllocatedImage _image;
    VulkanDevice &device;
};
