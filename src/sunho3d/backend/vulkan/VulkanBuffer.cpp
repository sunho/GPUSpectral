#include "VulkanBuffer.h"
#include "VulkanTypes.h"

#include <vulkan/vulkan.hpp>
#include <stdexcept>

VulkanBufferObject::VulkanBufferObject(VulkanDevice &device, uint32_t size, BufferUsage usage)
    : device(device), HwBufferObject(size) {
    
    auto bufferInfo = vk::BufferCreateInfo()
        .setSize(size)
        .setUsage(translateBufferUsage(usage) | vk::BufferUsageFlagBits::eTransferDst);

    _buffer = device.allocateBuffer(bufferInfo, VMA_MEMORY_USAGE_GPU_ONLY);
    buffer = _buffer.buffer;
}

VulkanBufferObject::~VulkanBufferObject() {
    _buffer.destroy(device);
}

void VulkanBufferObject::upload(const BufferDescriptor &descriptor) {
    auto bi = vk::BufferCreateInfo()
            .setSize(size)
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    auto staging = device.allocateBuffer(bi, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data;
    staging.map(device, data);
    memcpy(data, descriptor.data, size);
    staging.unmap(device);
    device.immediateSubmit([=](vk::CommandBuffer cmd) {
		vk::BufferCopy copy = {};
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = size;
        cmd.copyBuffer(staging.buffer, _buffer.buffer, 1, &copy);
	});
}
