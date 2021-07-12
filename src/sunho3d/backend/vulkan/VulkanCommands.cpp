#include "VulkanCommands.h"

#include "VulkanContext.h"

VkSemaphore VulkanCommands::renderFinishedSemaphore() { 
    return renderFinishedSemaphores[currentFrame];
}


VkSemaphore VulkanCommands::imageAvailableSemaphore() { 
    return imageAvailableSemaphores[currentFrame];
}


VkFence VulkanCommands::fence() { 
    return fences[currentFrame];
}


void VulkanCommands::next() { 
    currentFrame = (currentFrame + 1) % VULKAN_COMMANDS_SIZE;
}


VkCommandBuffer VulkanCommands::get() { 
    return cmdBuffers[currentFrame];
}

VulkanCommands::VulkanCommands(VulkanContext &context) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = context.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) cmdBuffers.size();

    if (vkAllocateCommandBuffers(context.device, &allocInfo, cmdBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < VULKAN_COMMANDS_SIZE; ++i) {
         if (vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
             vkCreateFence(context.device, &fenceInfo, nullptr, &fences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }
    }
}
