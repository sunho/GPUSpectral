#include "VulkanContext.h"

#include <sunho3d/Window.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <cassert>
#include <stdexcept>

#include "VulkanTexture.h"

static const std::vector<const char *> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
//, "VK_KHR_portability_subset"

static const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

inline bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

inline SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device,
                                                     VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                                  details.presentModes.data());
    }

    return details;
}

inline QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }
        i++;
    }

    return indices;
}

inline bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices = findQueueFamilies(device, surface);
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
        swapChainAdequate =
            !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }
    return indices.isComplete() && swapChainAdequate && extensionsSupported;
}

inline int rateDeviceSuitability(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    int score = 0;
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    }
    score += deviceProperties.limits.maxImageDimension2D;
    return score;
}

inline bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
        bool layerFound = false;

        for (const auto &layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }
    return true;
}

inline std::vector<const char *> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    extensions.push_back("VK_KHR_get_physical_device_properties2");
    return extensions;
}

void initContext(VulkanContext &context) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "secret";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "sunho3d";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (!checkValidationLayerSupport()) {
        throw std::runtime_error("validation layer not available!");
    }

    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    if (vkCreateInstance(&createInfo, nullptr, &context.instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void pickPhysicalDevice(VulkanContext &context, VulkanSurfaceContext &surface) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(context.instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(context.instance, &deviceCount, devices.data());

    std::multimap<int, VkPhysicalDevice> candidates;
    for (const auto &device : devices) {
        if (isDeviceSuitable(device, surface.surface)) {
            int score = rateDeviceSuitability(device);
            candidates.insert(std::make_pair(score, device));
        }
    }

    if (candidates.rbegin()->first > 0) {
        context.physicalDevice = candidates.rbegin()->second;
    } else {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void initSurfaceContext(VulkanContext &context, VulkanSurfaceContext &surface,
                        sunho3d::Window *window) {
    surface.surface = window->createSurface(context.instance);
    context.surface = &surface;
}

VkSurfaceFormatKHR
chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    for (const auto &availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
    for (const auto &availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities, sunho3d::Window *window) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        int width = window->getWindowWidth();
        int height = window->getWindowHeight();

        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

void createSwapChain(VulkanContext &context, VulkanSurfaceContext &surface, sunho3d::Window *window) {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(context.physicalDevice, surface.surface);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface.surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    QueueFamilyIndices indices = findQueueFamilies(context.physicalDevice, surface.surface);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(),
                                      indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;      // Optional
        createInfo.pQueueFamilyIndices = nullptr;  // Optional
    }
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(context.device, &createInfo, nullptr, &surface.swapChain) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(context.device, surface.swapChain, &imageCount, nullptr);

    surface.format = surfaceFormat;
    surface.extent = extent;
    surface.size = imageCount;
}

void createLogicalDevice(VulkanContext &context, VulkanSurfaceContext &surface) {
    QueueFamilyIndices indices = findQueueFamilies(context.physicalDevice, surface.surface);

    float queuePriority = 1.0f;

    VkPhysicalDeviceFeatures deviceFeatures{};
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),
                                               indices.presentFamily.value() };
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = queueCreateInfos.size();
    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = deviceExtensions.size();
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(context.physicalDevice, &createInfo, nullptr, &context.device) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }
    vkGetDeviceQueue(context.device, indices.graphicsFamily.value(), 0, &context.graphicsQueue);
    vkGetDeviceQueue(context.device, indices.presentFamily.value(), 0, &surface.presentQueue);

    QueueFamilyIndices queueFamilyIndices =
        findQueueFamilies(context.physicalDevice, surface.surface);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(context.device, &poolInfo, nullptr, &context.commandPool) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    context.commands = new VulkanCommands(context);
}

void populateSwapContexts(VulkanContext &context, VulkanSurfaceContext &surface) {
    uint32_t imageCount = surface.size;
    surface.swapContexts.resize(imageCount);
    std::vector<VkImage> swapChainImages;
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(context.device, surface.swapChain, &imageCount, swapChainImages.data());

    for (size_t i = 0; i < surface.size; ++i) {
        surface.swapContexts[i].attachment.image = swapChainImages[i];
    }

    for (size_t i = 0; i < surface.size; i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = surface.format.format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(context.device, &createInfo, nullptr,
                              &surface.swapContexts[i].attachment.view) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    context.currentSwapContext = &surface.swapContexts[0];
    surface.swapContextIndex = 0;

    context.emptyTexture =
        new VulkanTexture(context, SamplerType::SAMPLER2D,
                          TextureUsage::UPLOADABLE | TextureUsage::SAMPLEABLE |
                              TextureUsage::COLOR_ATTACHMENT | TextureUsage::INPUT_ATTACHMENT,
                          1, TextureFormat::RGBA8, 1, 1);

    surface.depthTexture = new VulkanTexture(context, SamplerType::SAMPLER2D, TextureUsage::DEPTH_ATTACHMENT, 1,
                                             TextureFormat::DEPTH32F, surface.extent.width, surface.extent.height);
}

void destroyContext(VulkanContext &context, VulkanSurfaceContext &surface) {
    delete context.commands;
    vkDestroyCommandPool(context.device, context.commandPool, nullptr);

    for (auto &ctx : surface.swapContexts) {
        vkDestroyImageView(context.device, ctx.attachment.view, nullptr);
    }
    vkDestroySwapchainKHR(context.device, surface.swapChain, nullptr);
    vkDestroyDevice(context.device, nullptr);

    vkDestroySurfaceKHR(context.instance, surface.surface, nullptr);
}

VkCommandBuffer VulkanContext::beginSingleCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void VulkanContext::endSingleCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

VkFormat VulkanContext::translateTextureFormat(TextureFormat format) {
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

VkFormat VulkanContext::translateElementFormat(ElementType type, bool normalized, bool integer) {
    using ElementType = ElementType;
    if (normalized) {
        switch (type) {
            case ElementType::BYTE:
                return VK_FORMAT_R8_SNORM;
            case ElementType::UBYTE:
                return VK_FORMAT_R8_UNORM;
            case ElementType::SHORT:
                return VK_FORMAT_R16_SNORM;
            case ElementType::USHORT:
                return VK_FORMAT_R16_UNORM;
            case ElementType::BYTE2:
                return VK_FORMAT_R8G8_SNORM;
            case ElementType::UBYTE2:
                return VK_FORMAT_R8G8_UNORM;
            case ElementType::SHORT2:
                return VK_FORMAT_R16G16_SNORM;
            case ElementType::USHORT2:
                return VK_FORMAT_R16G16_UNORM;
            case ElementType::BYTE3:
                return VK_FORMAT_R8G8B8_SNORM;
            case ElementType::UBYTE3:
                return VK_FORMAT_R8G8B8_UNORM;
            case ElementType::SHORT3:
                return VK_FORMAT_R16G16B16_SNORM;
            case ElementType::USHORT3:
                return VK_FORMAT_R16G16B16_UNORM;
            case ElementType::BYTE4:
                return VK_FORMAT_R8G8B8A8_SNORM;
            case ElementType::UBYTE4:
                return VK_FORMAT_R8G8B8A8_UNORM;
            case ElementType::SHORT4:
                return VK_FORMAT_R16G16B16A16_SNORM;
            case ElementType::USHORT4:
                return VK_FORMAT_R16G16B16A16_UNORM;
            default:
                return VK_FORMAT_UNDEFINED;
        }
    }
    switch (type) {
        case ElementType::BYTE:
            return integer ? VK_FORMAT_R8_SINT : VK_FORMAT_R8_SSCALED;
        case ElementType::UBYTE:
            return integer ? VK_FORMAT_R8_UINT : VK_FORMAT_R8_USCALED;
        case ElementType::SHORT:
            return integer ? VK_FORMAT_R16_SINT : VK_FORMAT_R16_SSCALED;
        case ElementType::USHORT:
            return integer ? VK_FORMAT_R16_UINT : VK_FORMAT_R16_USCALED;
        case ElementType::HALF:
            return VK_FORMAT_R16_SFLOAT;
        case ElementType::INT:
            return VK_FORMAT_R32_SINT;
        case ElementType::UINT:
            return VK_FORMAT_R32_UINT;
        case ElementType::FLOAT:
            return VK_FORMAT_R32_SFLOAT;
        case ElementType::BYTE2:
            return integer ? VK_FORMAT_R8G8_SINT : VK_FORMAT_R8G8_SSCALED;
        case ElementType::UBYTE2:
            return integer ? VK_FORMAT_R8G8_UINT : VK_FORMAT_R8G8_USCALED;
        case ElementType::SHORT2:
            return integer ? VK_FORMAT_R16G16_SINT : VK_FORMAT_R16G16_SSCALED;
        case ElementType::USHORT2:
            return integer ? VK_FORMAT_R16G16_UINT : VK_FORMAT_R16G16_USCALED;
        case ElementType::HALF2:
            return VK_FORMAT_R16G16_SFLOAT;
        case ElementType::FLOAT2:
            return VK_FORMAT_R32G32_SFLOAT;
        case ElementType::BYTE3:
            return VK_FORMAT_R8G8B8_SINT;
        case ElementType::UBYTE3:
            return VK_FORMAT_R8G8B8_UINT;
        case ElementType::SHORT3:
            return VK_FORMAT_R16G16B16_SINT;
        case ElementType::USHORT3:
            return VK_FORMAT_R16G16B16_UINT;
        case ElementType::HALF3:
            return VK_FORMAT_R16G16B16_SFLOAT;
        case ElementType::FLOAT3:
            return VK_FORMAT_R32G32B32_SFLOAT;
        case ElementType::BYTE4:
            return integer ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_SSCALED;
        case ElementType::UBYTE4:
            return integer ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_USCALED;
        case ElementType::SHORT4:
            return integer ? VK_FORMAT_R16G16B16A16_SINT : VK_FORMAT_R16G16B16A16_SSCALED;
        case ElementType::USHORT4:
            return integer ? VK_FORMAT_R16G16B16A16_UINT : VK_FORMAT_R16G16B16A16_USCALED;
        case ElementType::HALF4:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case ElementType::FLOAT4:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

VkBufferUsageFlags VulkanContext::translateBufferUsage(BufferUsage usage) {
    switch (usage) {
        case BufferUsage::TRANSFER_SRC:
            return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        case BufferUsage::VERTEX:
            return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        case BufferUsage::INDEX:
            return VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        case BufferUsage::UNIFORM:
            return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }
}
