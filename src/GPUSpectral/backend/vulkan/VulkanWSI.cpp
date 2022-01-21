#include "VulkanWSI.h"
#include "VulkanDevice.h"

VulkanWSI::VulkanWSI(GPUSpectral::Window* window, VulkanDevice* device)
    : window(window), device(device) {
    surface = window->createSurface(device->instance);
}

VulkanWSI::~VulkanWSI() {
    device->instance.destroySurfaceKHR(surface);
    vkb::destroy_swapchain(vkbSwapchain);
}

void VulkanWSI::initSwapchain() {
    vkb::SwapchainBuilder swapchain_builder{ device->vkbDevice };
    auto swapRet = swapchain_builder.set_old_swapchain(vkbSwapchain).build();
    if (!swapRet) {
        throw std::runtime_error("error create swap chain");
    }
    vkb::destroy_swapchain(vkbSwapchain);
    vkbSwapchain = swapRet.value();
    swapchainImages = vkbSwapchain.get_images().value();
    swapchainImageViews = vkbSwapchain.get_image_views().value();

    auto presentQueueRet = device->vkbDevice.get_queue(vkb::QueueType::present);
    if (!presentQueueRet) {
        throw std::runtime_error("error getting present queue");
    }
    presentQueue = presentQueueRet.value();
}

void VulkanWSI::beginFrame(vk::Semaphore imageSemaphore) {
    VkResult result = vkAcquireNextImageKHR(device->device, vkbSwapchain.swapchain, UINT64_MAX, imageSemaphore, VK_NULL_HANDLE, &swapchainIndex);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("error window begin frame");
    }
}

void VulkanWSI::endFrame(vk::Semaphore renderSemaphore) {
    std::array<vk::SwapchainKHR, 1> swapChains = { vkbSwapchain.swapchain };
    std::array<vk::Semaphore, 1> waitSemapohres = { renderSemaphore };
    auto pi = vk::PresentInfoKHR()
                  .setWaitSemaphores(waitSemapohres)
                  .setSwapchains(swapChains)
                  .setPImageIndices(&swapchainIndex);

    auto res = presentQueue.presentKHR(pi);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("error present");
    }
}

VulkanSwapChain VulkanWSI::currentSwapChain() {
    return {
        vk::Format(vkbSwapchain.image_format),
        swapchainImages[swapchainIndex],
        swapchainImageViews[swapchainIndex]
    };
}

vk::Extent2D VulkanWSI::getExtent() {
    return vkbSwapchain.extent;
}
