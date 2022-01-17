#pragma once

#include "../DriverBase.h"
#include "VulkanDevice.h"
#include "VulkanTypes.h"

class VulkanTexture : public HwTexture {
  public:
    VulkanTexture(VulkanDevice &device, SamplerType type, TextureUsage usage, uint8_t levels,
                  TextureFormat format, uint32_t width, uint32_t height, uint32_t layers);
    ~VulkanTexture();
    void copyInitialData(const BufferDescriptor &data, ImageLayout finalLayout = ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    void copyBuffer(vk::CommandBuffer cmd, vk::Buffer buffer, uint32_t width,
                           uint32_t height, const ImageSubresource& subresource);
    void blitImage(vk::CommandBuffer cmd, VulkanTexture& srcImage, uint32_t width, uint32_t height, const ImageSubresource& srcSubresource, const ImageSubresource& dstSubresource);
    vk::Image image;
    vk::ImageView view;
    vk::Format vkFormat;
    vk::Sampler sampler;
    vk::ImageAspectFlags aspect;
    vk::ImageLayout vkImageLayout;
    ImageLayout imageLayout{ImageLayout::GENERAL};
    uint32_t layers;

  private:
    AllocatedImage _image;
    VulkanDevice &device;
};
