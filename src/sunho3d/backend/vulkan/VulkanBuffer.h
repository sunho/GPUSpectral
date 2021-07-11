#pragma once

#include "../DriverBase.h"
#include "../Handles.h"
#include "VulkanContext.h"

#include <vulkan/vulkan.h>

struct VulkanBufferObject : public HwBufferObject {
    VulkanBufferObject() = default;
    explicit VulkanBufferObject(uint32_t size) : HwBufferObject(size) { }
    void allocate(VulkanContext& ctx, VkBufferUsageFlags usage);
    void upload(VulkanContext& ctx, const BufferDescriptor& descriptor);
    VkDeviceMemory memory;
    VkBuffer buffer;
};
