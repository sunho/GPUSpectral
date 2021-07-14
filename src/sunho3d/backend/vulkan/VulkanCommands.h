#pragma once

#include <vulkan/vulkan.h>

#include <array>

constexpr static const size_t VULKAN_COMMANDS_SIZE = 10;

struct VulkanContext;

class VulkanCommands {
  public:
    VulkanCommands() = default;
    explicit VulkanCommands(VulkanContext &context);

    VkCommandBuffer get();
    void next();
    VkFence fence();
    VkSemaphore imageAvailableSemaphore();
    VkSemaphore renderFinishedSemaphore();

  private:
    std::array<VkCommandBuffer, VULKAN_COMMANDS_SIZE> cmdBuffers;
    std::array<VkFence, VULKAN_COMMANDS_SIZE> fences;
    std::array<VkSemaphore, VULKAN_COMMANDS_SIZE> imageAvailableSemaphores;
    std::array<VkSemaphore, VULKAN_COMMANDS_SIZE> renderFinishedSemaphores;
    size_t currentFrame{ 0 };
};
