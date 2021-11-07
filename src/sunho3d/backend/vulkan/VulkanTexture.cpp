#include "VulkanTexture.h"

#include "VulkanBuffer.h"

inline static AllocatedImage createImage(VulkanDevice &device, uint32_t width, uint32_t height, vk::Format format,
                               vk::ImageTiling tiling, vk::ImageUsageFlags usage) {
    vk::ImageCreateInfo imageInfo = {};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    return device.allocateImage(imageInfo, VMA_MEMORY_USAGE_GPU_ONLY);
}

static VkImageView createImageView(VulkanDevice &device, vk::Image image, vk::Format format,
                                   vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo viewInfo = {};
    viewInfo.image = image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    return device.device.createImageView(viewInfo);
}

VulkanTexture::VulkanTexture(VulkanDevice &device, SamplerType type, TextureUsage usage,
                             uint8_t levels, TextureFormat format, uint32_t w, uint32_t h)
    : HwTexture(type, levels, format, w, h), device(device) {
    if (width == HwTexture::FRAME_WIDTH) {
        width = device.wsi->getExtent().width;
    }
    if (height == HwTexture::FRAME_HEIGHT) {
        height = device.wsi->getExtent().height;
    }
    
    const vk::ImageUsageFlags blittable =  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
    vk::ImageUsageFlags flags{};
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
    vkFormat = translateTextureFormat(format);
    vk::ImageAspectFlags aspect = usage & TextureUsage::DEPTH_ATTACHMENT ? vk::ImageAspectFlagBits::eDepth :
                                                                         vk::ImageAspectFlagBits::eColor;

    /*
    imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    if (usage & TextureUsage::DEPTH_ATTACHMENT) {
        imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    if (usage & TextureUsage::COLOR_ATTACHMENT) {
        imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }*/

    imageLayout = vk::ImageLayout::eGeneral;

    _image = createImage(device, width, height, vkFormat, vk::ImageTiling::eOptimal, flags);
    image = _image.image;
    view = createImageView(device, image, vkFormat, aspect);

    
    //if ((usage & TextureUsage::DEPTH_ATTACHMENT) || (usage & TextureUsage::COLOR_ATTACHMENT)) {
    device.immediateSubmit([=](vk::CommandBuffer cmd) {
        vk::ImageSubresourceRange range;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;
        range.aspectMask = aspect;
		vk::ImageMemoryBarrier imageBarrier_toTransfer = {};
		imageBarrier_toTransfer.oldLayout = vk::ImageLayout::eUndefined;
		imageBarrier_toTransfer.newLayout = vk::ImageLayout::eGeneral;
		imageBarrier_toTransfer.image = image;
		imageBarrier_toTransfer.subresourceRange = range;

		imageBarrier_toTransfer.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
		imageBarrier_toTransfer.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, imageBarrier_toTransfer);
    });
    //}

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

void VulkanTexture::update2DImage(VulkanDevice &device, const BufferDescriptor &data) {
    // VulkanBufferObject *buffer = new VulkanBufferObject(device, width * height * 4, BufferUsage::TRANSFER_SRC);
    // buffer->upload(data);
    auto bi = vk::BufferCreateInfo().setSize(width * height * 4).setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    auto staging = device.allocateBuffer(bi, VMA_MEMORY_USAGE_CPU_ONLY);
    void* d;
    staging.map(device, &d);
    memcpy(d, data.data, width*height*4);
    staging.unmap(device);
    device.immediateSubmit([&](vk::CommandBuffer cmd) {
		vk::ImageSubresourceRange range;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;
        range.aspectMask = vk::ImageAspectFlagBits::eColor;

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

        copyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = vk::Extent3D().setWidth(width).setHeight(height).setDepth(1);

        cmd.copyBufferToImage(staging.buffer, _image.image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
    
        vk::ImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

        imageBarrier_toReadable.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        imageBarrier_toReadable.newLayout = vk::ImageLayout::eGeneral;
        imageBarrier_toReadable.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        imageBarrier_toReadable.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, imageBarrier_toReadable);
    });
    staging.destroy(device);
}

void VulkanTexture::copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                                      uint32_t width, uint32_t height, uint32_t miplevel) {
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = miplevel;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };
    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}
