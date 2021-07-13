#include "VulkanBuffer.h"

void VulkanBufferObject::allocate(VulkanContext& ctx, VkBufferUsageFlags usage) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
       throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(ctx.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    allocInfo.memoryTypeIndex = ctx.findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
       throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(ctx.device, buffer, memory, 0);
}

void VulkanBufferObject::upload(VulkanContext& ctx, const BufferDescriptor& descriptor) {
    void* data;
    vkMapMemory(ctx.device, memory, 0, size, 0, &data);
    memcpy(data, descriptor.data, (size_t)size);
    vkUnmapMemory(ctx.device, memory);
}
