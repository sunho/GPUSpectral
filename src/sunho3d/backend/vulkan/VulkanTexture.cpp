#include "VulkanTexture.h"

#include "VulkanBuffer.h"

static void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                                  VkImageLayout newLayout, VkImageAspectFlags aspect) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    switch (newLayout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_GENERAL:
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            break;
        default: {
            throw std::runtime_error("WTF");
        }
    }
    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
}

inline

    static void
    createImage(VulkanContext &context, uint32_t width, uint32_t height, VkFormat format,
                VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                VkImage &image, VkDeviceMemory &imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(context.device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(context.device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(context.device, image, imageMemory, 0);
}

static VkFormat translateTextureFormat(TextureFormat format) {
    switch (format) {
        case TextureFormat::R8:
            return VK_FORMAT_R8_UNORM;
        case TextureFormat::R8_SNORM:
            return VK_FORMAT_R8_SNORM;
        case TextureFormat::R8UI:
            return VK_FORMAT_R8_UINT;
        case TextureFormat::R8I:
            return VK_FORMAT_R8_SINT;
        case TextureFormat::R16F:
            return VK_FORMAT_R16_SFLOAT;
        case TextureFormat::R16UI:
            return VK_FORMAT_R16_UINT;
        case TextureFormat::R16I:
            return VK_FORMAT_R16_SINT;
        case TextureFormat::DEPTH16:
            return VK_FORMAT_D16_UNORM;
        case TextureFormat::R32F:
            return VK_FORMAT_R32_SFLOAT;
        case TextureFormat::R32UI:
            return VK_FORMAT_R32_UINT;
        case TextureFormat::R32I:
            return VK_FORMAT_R32_SINT;
        case TextureFormat::RG16F:
            return VK_FORMAT_R16G16_SFLOAT;
        case TextureFormat::RG16UI:
            return VK_FORMAT_R16G16_UINT;
        case TextureFormat::RG16I:
            return VK_FORMAT_R16G16_SINT;
        case TextureFormat::R11F_G11F_B10F:
            return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case TextureFormat::RGBA8:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case TextureFormat::SRGB8_A8:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case TextureFormat::RGBA8_SNORM:
            return VK_FORMAT_R8G8B8A8_SNORM;
        case TextureFormat::RGB10_A2:
            return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case TextureFormat::RGBA8UI:
            return VK_FORMAT_R8G8B8A8_UINT;
        case TextureFormat::RGBA8I:
            return VK_FORMAT_R8G8B8A8_SINT;
        case TextureFormat::DEPTH32F:
            return VK_FORMAT_D32_SFLOAT;
        case TextureFormat::DEPTH24_STENCIL8:
            return VK_FORMAT_D24_UNORM_S8_UINT;
        case TextureFormat::DEPTH32F_STENCIL8:
            return VK_FORMAT_D32_SFLOAT_S8_UINT;
        default: {
            assert(false);
        }
    }
}

static VkImageView createImageView(VulkanContext &context, VkImage image, VkFormat format,
                                   VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(context.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

VulkanTexture::VulkanTexture(VulkanContext &context, SamplerType type, TextureUsage usage,
                             uint8_t levels, TextureFormat format, uint32_t width, uint32_t height)
    : HwTexture(type, levels, format, width, height) {
    const VkImageUsageFlags blittable =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageUsageFlags flags{};
    if (usage & TextureUsage::SAMPLEABLE) {
        flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (usage & TextureUsage::DEPTH_ATTACHMENT) {
        flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        flags |= blittable;
    }
    if (usage & TextureUsage::UPLOADABLE) {
        flags |= blittable;
    }
    if (usage & TextureUsage::INPUT_ATTACHMENT) {
        flags |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    }
    if (usage & TextureUsage::COLOR_ATTACHMENT) {
        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    vkFormat = translateTextureFormat(format);
    VkImageAspectFlags aspect = usage & TextureUsage::DEPTH_ATTACHMENT ? VK_IMAGE_ASPECT_DEPTH_BIT :
                                                                         VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageLayout newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (usage & TextureUsage::DEPTH_ATTACHMENT) {
        newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    if (usage & TextureUsage::COLOR_ATTACHMENT) {
        newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    createImage(context, width, height, vkFormat, VK_IMAGE_TILING_OPTIMAL, flags,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, memory);
    view = createImageView(context, image, vkFormat, aspect);
    auto cmd = context.beginSingleCommands();
    transitionImageLayout(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, newLayout, aspect);
    context.endSingleCommands(cmd);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physicalDevice, &properties);
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    if (vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void VulkanTexture::update2DImage(VulkanContext &context, const BufferDescriptor &data) {
    VulkanBufferObject buffer(width * height * 4);
    buffer.allocate(context, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    buffer.upload(context, data);

    const VkCommandBuffer cmdbuffer = context.beginSingleCommands();
    transitionImageLayout(cmdbuffer, image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    copyBufferToImage(cmdbuffer, buffer.buffer, image, width, height, 0);
    transitionImageLayout(cmdbuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    context.endSingleCommands(cmdbuffer);
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
