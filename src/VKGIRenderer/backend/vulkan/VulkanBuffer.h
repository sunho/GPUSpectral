#pragma once

#include <vulkan/vulkan.hpp>
#include <stdexcept>

#include "../DriverBase.h"
#include "../Handles.h"
#include "VulkanDevice.h"

struct VulkanBufferObject : public HwBufferObject {
    explicit VulkanBufferObject(VulkanDevice &device, uint32_t size, BufferUsage usage, BufferType type);
    ~VulkanBufferObject();
    void uploadSync(const BufferDescriptor &descriptor);
    void copy(vk::CommandBuffer cmd, const VulkanBufferObject &obj);
    std::vector<char> download();
    void* mapped{ nullptr };
    vk::Buffer buffer;
    BufferType type;

  private:
    VulkanDevice &device;
    AllocatedBuffer _buffer;
};
