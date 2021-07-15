#include "VulkanCommands.h"

#include "VulkanContext.h"

VkSemaphore VulkanCommands::renderFinishedSemaphore() {
    return renderFinishedSemaphores[currentIndex];
}

VkSemaphore VulkanCommands::imageAvailableSemaphore() {
    return imageAvailableSemaphores[currentIndex];
}

VkFence VulkanCommands::fence() {
    return fences[currentIndex];
}

uint32_t VulkanCommands::next() {
    uint32_t out = currentIndex;
    currentIndex = (currentIndex + 1) % VULKAN_COMMANDS_SIZE;
    return out;
}

VkCommandBuffer VulkanCommands::get() {
    return cmdBuffers[currentIndex];
}

VulkanCommands::VulkanCommands(VulkanContext &context)
    : context(context) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = context.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)cmdBuffers.size();

    if (vkAllocateCommandBuffers(context.device, &allocInfo, cmdBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < VULKAN_COMMANDS_SIZE; ++i) {
        if (vkCreateSemaphore(context.device, &semaphoreInfo, nullptr,
                              &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(context.device, &semaphoreInfo, nullptr,
                              &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(context.device, &fenceInfo, nullptr, &fences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }
    }
}

VulkanCommands::~VulkanCommands() {
    for (int i = 0; i < VULKAN_COMMANDS_SIZE; ++i) {
        vkDestroySemaphore(context.device, imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(context.device, renderFinishedSemaphores[i], nullptr);
        vkDestroyFence(context.device, fences[i], nullptr);
    }
}
