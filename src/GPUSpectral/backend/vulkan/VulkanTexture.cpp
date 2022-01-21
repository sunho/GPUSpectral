#include "VulkanTexture.h"

#include <Tracy.hpp>
#include "VulkanBuffer.h"

inline static AllocatedImage createImage(VulkanDevice &device, uint8_t levels, uint32_t width, uint32_t height, uint32_t layers, vk::Format format,
                                         vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::ImageCreateFlags createFlags) {
    vk::ImageCreateInfo imageInfo = {};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = levels;
    imageInfo.arrayLayers = layers;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.flags = createFlags;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    return device.allocateImage(imageInfo, VMA_MEMORY_USAGE_GPU_ONLY);
}

inline static VkImageView createImageView(VulkanDevice &device, vk::Image image, vk::Format format, uint8_t levels, uint32_t layers,
                                          vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo viewInfo = {};
    viewInfo.image = image;
    // TODO:
    viewInfo.viewType = layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::eCube;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = levels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layers;

    return device.device.createImageView(viewInfo);
}

VulkanTexture::VulkanTexture(VulkanDevice &device, SamplerType type, TextureUsage usage,
                             uint8_t levels, TextureFormat format, uint32_t w, uint32_t h, uint32_t layers)
    : HwTexture(type, levels, format, w, h), device(device), layers(layers) {
    ZoneScopedN("Texture create") if (width == HwTexture::FRAME_WIDTH) {
        width = device.wsi->getExtent().width;
    }
    if (height == HwTexture::FRAME_HEIGHT) {
        height = device.wsi->getExtent().height;
    }

    const vk::ImageUsageFlags blittable = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
    vk::ImageUsageFlags flags{};
    vk::ImageCreateFlags createFlags{};
    if (usage & TextureUsage::SAMPLEABLE) {
        flags |= vk::ImageUsageFlagBits::eSampled;
    }
    if (usage & TextureUsage::STORAGE) {
        flags |= vk::ImageUsageFlagBits::eStorage;
    }
    if (usage & TextureUsage::DEPTH_ATTACHMENT) {
        flags |= vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
        flags |= blittable;
    }
    if (usage & TextureUsage::UPLOADABLE) {
        flags |= blittable;
    }
    if (usage & TextureUsage::INPUT_ATTACHMENT) {
        flags |= vk::ImageUsageFlagBits::eInputAttachment;
    }
    if (usage & TextureUsage::COLOR_ATTACHMENT) {
        flags |= vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
        flags |= blittable;
    }
    if (usage & TextureUsage::CUBE) {
        createFlags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    vkFormat = translateTextureFormat(format);
    aspect = usage & TextureUsage::DEPTH_ATTACHMENT ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

    vkImageLayout = vk::ImageLayout::eUndefined;
    imageLayout = ImageLayout::NONE;

    _image = createImage(device, levels, width, height, layers, vkFormat, vk::ImageTiling::eOptimal, flags, createFlags);
    image = _image.image;
    view = createImageView(device, image, vkFormat, levels, layers, aspect);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = VK_FALSE;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physicalDevice, &properties);
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    sampler = device.device.createSampler(samplerInfo);
}

VulkanTexture::~VulkanTexture() {
    vkDestroySampler(device.device, sampler, nullptr);
    vkDestroyImageView(device.device, view, nullptr);
    _image.destroy(device);
}

void VulkanTexture::copyInitialData(const BufferDescriptor &data, ImageLayout finalLayout) {
    auto bi = vk::BufferCreateInfo().setSize(width * height * getTextureFormatSize(format)).setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    auto staging = device.allocateBuffer(bi, VMA_MEMORY_USAGE_CPU_ONLY);
    {
        ZoneScopedN("Texture staging") void *d;
        staging.map(device, &d);
        memcpy(d, data.data, width * height * getTextureFormatSize(format));
        staging.unmap(device);
    }
    ZoneScopedN("Texture upload")
        device.immediateSubmit([&](vk::CommandBuffer cmd) {
            vk::ImageSubresourceRange range;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = 1;
            range.aspectMask = aspect;

            vk::ImageMemoryBarrier imageBarrier_toTransfer = {};
            imageBarrier_toTransfer.oldLayout = vk::ImageLayout::eUndefined;
            imageBarrier_toTransfer.newLayout = vk::ImageLayout::eTransferDstOptimal;
            imageBarrier_toTransfer.image = _image.image;
            imageBarrier_toTransfer.subresourceRange = range;

            imageBarrier_toTransfer.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
            imageBarrier_toTransfer.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, imageBarrier_toTransfer);

            vk::BufferImageCopy copyRegion = {};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;

            copyRegion.imageSubresource.aspectMask = aspect;
            copyRegion.imageSubresource.mipLevel = 0;
            copyRegion.imageSubresource.baseArrayLayer = 0;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageExtent = vk::Extent3D().setWidth(width).setHeight(height).setDepth(1);

            cmd.copyBufferToImage(staging.buffer, _image.image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

            vk::ImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

            imageBarrier_toReadable.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            imageBarrier_toReadable.newLayout = translateImageLayout(finalLayout);
            imageBarrier_toReadable.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            imageBarrier_toReadable.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, imageBarrier_toReadable);
        });
    staging.destroy(device);

    imageLayout = finalLayout;
    vkImageLayout = translateImageLayout(finalLayout);
}

void VulkanTexture::copyBuffer(vk::CommandBuffer cmd, vk::Buffer buffer, uint32_t width, uint32_t height, const ImageSubresource &subresource) {
    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = aspect;
    region.imageSubresource.mipLevel = subresource.baseLevel;
    region.imageSubresource.baseArrayLayer = subresource.baseLayer;
    region.imageSubresource.layerCount = subresource.layerCount;
    region.imageOffset = vk::Offset3D{ 0, 0, 0 };
    region.imageExtent = vk::Extent3D{ width, height, 1 };
    cmd.copyBufferToImage(buffer, image, vkImageLayout, 1, &region);
}

void VulkanTexture::blitImage(vk::CommandBuffer cmd, VulkanTexture &srcImage, uint32_t width, uint32_t height, const ImageSubresource &srcSubresource, const ImageSubresource &dstSubresource) {
    vk::ImageBlit blit{};
    blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
    blit.srcOffsets[1] = vk::Offset3D{ (int32_t)width, (int32_t)height, 1 };
    blit.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
    blit.dstOffsets[1] = vk::Offset3D{ (int32_t)width, (int32_t)height, 1 };
    blit.srcSubresource.aspectMask = aspect;
    blit.dstSubresource.aspectMask = aspect;
    blit.srcSubresource.baseArrayLayer = srcSubresource.baseLayer;
    blit.srcSubresource.mipLevel = srcSubresource.baseLevel;
    blit.srcSubresource.layerCount = srcSubresource.layerCount;
    blit.dstSubresource.baseArrayLayer = dstSubresource.baseLayer;
    blit.dstSubresource.mipLevel = dstSubresource.baseLevel;
    blit.dstSubresource.layerCount = dstSubresource.layerCount;
    cmd.blitImage(srcImage.image, srcImage.vkImageLayout, image, vkImageLayout, 1, &blit, vk::Filter::eNearest);
}
