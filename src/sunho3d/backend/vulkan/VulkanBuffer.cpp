#include "VulkanBuffer.h"

#include <vulkan/vulkan.hpp>
#include <stdexcept>

VulkanBufferObject::VulkanBufferObject(VulkanContext &context, uint32_t size, BufferUsage usage)
    : context(context), HwBufferObject(size) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = context.translateBufferUsage(usage) | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(context.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(context.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    allocInfo.memoryTypeIndex = context.findMemoryType(memRequirements.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(context.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(context.device, buffer, memory, 0);
}

VulkanBufferObject::~VulkanBufferObject() {
    vkDestroyBuffer(context.device, buffer, nullptr);
    vkFreeMemory(context.device, memory, nullptr);
}

void VulkanBufferObject::upload(const BufferDescriptor &descriptor) {
    void *data;
    vkMapMemory(context.device, memory, 0, size, 0, &data);
    memcpy(data, descriptor.data, (size_t)size);
    vkUnmapMemory(context.device, memory);
}
