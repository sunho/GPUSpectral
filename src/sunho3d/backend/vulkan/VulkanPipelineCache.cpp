#include "VulkanPipelineCache.h"
#include <iostream>

void VulkanDescriptorAllocator::init(vk::Device newDevice) {
    device = newDevice;
}

void VulkanDescriptorAllocator::cleanup() {
    for (auto p : freePools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    for (auto p : usedPools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
}

vk::DescriptorPool createPool(vk::Device device, const VulkanDescriptorAllocator::PoolSizes &poolSizes, int count, vk::DescriptorPoolCreateFlags flags) {
    std::vector<vk::DescriptorPoolSize> sizes;
    sizes.reserve(poolSizes.sizes.size());
    for (auto sz : poolSizes.sizes) {
        sizes.push_back({ sz.first, uint32_t(sz.second * count) });
    }
    vk::DescriptorPoolCreateInfo pi = {};
    pi.flags = flags;
    pi.maxSets = count;
    pi.poolSizeCount = (uint32_t)sizes.size();
    pi.pPoolSizes = sizes.data();

    return device.createDescriptorPool(pi, nullptr);
}

vk::DescriptorPool VulkanDescriptorAllocator::grabPool() {
    if (freePools.size() > 0) {
        VkDescriptorPool pool = freePools.back();
        freePools.pop_back();
        return pool;
    } else {
        return createPool(device, descriptorSizes, 1024, {});
    }
}

vk::DescriptorSet VulkanDescriptorAllocator::allocate(vk::DescriptorSetLayout layout) {
    if (!currentPool) {
        currentPool = grabPool();
        usedPools.push_back(currentPool);
    }

    vk::DescriptorSetAllocateInfo allocInfo = {};
    allocInfo.pSetLayouts = &layout;
    allocInfo.descriptorPool = currentPool;
    allocInfo.descriptorSetCount = 1;

    //try to allocate the descriptor set
    vk::DescriptorSet descriptorSet;
    vk::Result res = device.allocateDescriptorSets(&allocInfo, &descriptorSet);
    switch (res) {
        case vk::Result::eSuccess:
            //all good, return
            return descriptorSet;
        case vk::Result::eErrorFragmentedPool:
        case vk::Result::eErrorOutOfPoolMemory:
            break;
        default:
            throw std::runtime_error("pool allocation error");
    }
    //allocate a new pool and retry
    currentPool = grabPool();
    usedPools.push_back(currentPool);
    res = device.allocateDescriptorSets(&allocInfo, &descriptorSet);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("pool allocation error");
    }
    return descriptorSet;
}

void VulkanDescriptorAllocator::resetPools() {
    //reset all used pools and add them to the free pools
    for (auto p : usedPools) {
        device.resetDescriptorPool(p);
        freePools.push_back(p);
    }

    //clear the used pools, since we've put them all in the free pools
    usedPools.clear();

    //reset the current pool handle back to null
    currentPool = nullptr;
}


VulkanPipelineCache::VulkanPipelineCache(VulkanDevice &device) : device(device) {
    pipelines.setDestroyer([=](VkPipeline pipeline) {
        vkDestroyPipeline(this->device.device, pipeline, nullptr);
    });
    framebuffers.setDestroyer([=](VkFramebuffer framebuffer) {
        vkDestroyFramebuffer(this->device.device, framebuffer, nullptr);
    });
    renderpasses.setDestroyer([=](VkRenderPass renderPass) {
        vkDestroyRenderPass(this->device.device, renderPass, nullptr);
    });
    for (auto& alloc : descriptorAllocators) {
        alloc.init(device.device);
    }
}

void VulkanPipelineCache::tick() {
    pipelines.tick();
    framebuffers.tick();
    renderpasses.tick();
    ++currentFrame;
    descriptorAllocators[currentFrame % 3].resetPools();
}

VulkanPipelineCache::PipelineLayout VulkanPipelineCache::getOrCreatePipelineLayout(const ProgramParameterLayout &layout, bool compute) {
    auto it = pipelineLayouts.get(layout);
    if (it) {
        return *it;
    }
    
    VulkanPipelineCache::PipelineLayout pipelineLayout{};
    vk::PipelineLayoutCreateInfo createInfo{};
    vk::PushConstantRange pushConstants{};
    pushConstants.offset = 0;
    pushConstants.size = MAX_PUSH_CONSTANT_SIZE;
    pushConstants.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment;
    createInfo.pPushConstantRanges = &pushConstants;
    createInfo.pushConstantRangeCount = 1;


    for (size_t i = 0; i < ProgramParameterLayout::MAX_SET; ++i) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        for (size_t j = 0; j < ProgramParameterLayout::MAX_BINDINGS; ++j) {
            auto &field = layout.fields[i * ProgramParameterLayout::MAX_BINDINGS + j];
            if (field) {
                vk::DescriptorSetLayoutBinding binding{};
                binding.descriptorType = translateDescriptorType(field.type());
                binding.binding = j;
                binding.descriptorCount = field.arraySize();
                binding.stageFlags = vk::ShaderStageFlagBits::eAll;
                bindings.push_back(binding);
            }
        }
        vk::DescriptorSetLayoutCreateInfo di{};
        di.setBindings(bindings);
        pipelineLayout.descriptorSetLayout[i] = device.device.createDescriptorSetLayout(di);
    }
    createInfo.setLayoutCount = 3;
    createInfo.pSetLayouts = pipelineLayout.descriptorSetLayout.data();
    pipelineLayout.pipelineLayout = device.device.createPipelineLayout(createInfo);

    pipelineLayouts.add(layout, pipelineLayout);
    return pipelineLayout;
}

void VulkanPipelineCache::bindDescriptor(vk::CommandBuffer cmd, const VulkanPipelineState &state, const VulkanBindings& bindings) {
    const bool compute = state.program->program.type == ProgramType::COMPUTE;
    auto pipelineLayout = getOrCreatePipelineLayout(state.program->program.parameterLayout, state.program->program.type == ProgramType::COMPUTE);
    DescriptorSets descriptorSets{};
    for (size_t i = 0; i < ProgramParameterLayout::MAX_SET; ++i) {
        descriptorSets[i] = currentDescriptorAllocator().allocate(pipelineLayout.descriptorSetLayout[i]);
    }

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(bindings.size());
    for (auto &binding : bindings) {
        vk::WriteDescriptorSet write{};
        write.dstBinding = binding.binding;
        write.dstSet = descriptorSets[binding.set];
        write.dstArrayElement = 0;
        write.descriptorCount = binding.arraySize;
        write.descriptorType = binding.type;
        write.pBufferInfo = binding.bufferInfo.data();
        write.pImageInfo = binding.imageInfo.data();
        writes.push_back(write);
    }
    device.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
    cmd.bindDescriptorSets(compute ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics, pipelineLayout.pipelineLayout, 0, ProgramParameterLayout::MAX_SET, descriptorSets.data(), 0, nullptr);
}

VkPipeline VulkanPipelineCache::getOrCreatePipeline(const VulkanPipelineState &key) {
    auto it = pipelines.get(key);
    if (it) {
        return *it;
    }
    VkPipeline out;
    if (!key.program->compute) {
        out = createGraphicsPipeline(key);
    } else {
        out = createComputePipeline(key);
    }

    pipelines.add(key, out);
    return out;
}

VkPipeline VulkanPipelineCache::createGraphicsPipeline(const VulkanPipelineState& state) {
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = state.program->vertex;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = state.program->fragment;
    fragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = state.attributeCount;
    std::vector<VkVertexInputAttributeDescription> attributes(state.attributeCount);
    std::vector<VkVertexInputBindingDescription> bindings(state.attributeCount);

    for (uint32_t i = 0; i < state.attributeCount; ++i) {
        Attribute attrib = state.attributes[i];
        attributes[i] = { .location = i,
                          .binding = i,
                          .format = (VkFormat)translateElementFormat(attrib.type,
                                                                   attrib.flags & Attribute::FLAG_NORMALIZED, false) };
        bindings[i] = {
            .binding = i,
            .stride = attrib.stride,
        };
    }

    vertexInputInfo.pVertexBindingDescriptions = bindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = state.attributeCount;
    vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = (float)state.viewport.left;
    viewport.y = (float)state.viewport.top;
    viewport.width = (float)state.viewport.width;
    viewport.height = (float)state.viewport.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = { .width = (uint32_t)std::numeric_limits<int32_t>::max(), .height = (uint32_t)std::numeric_limits<int32_t>::max() };

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = state.depthTest.enabled ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = state.depthTest.write ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = (VkCompareOp)translateCompareOp(state.depthTest.compareOp);
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;  // Optional
    depthStencil.maxDepthBounds = 1.0f;  // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {};  // Optional
    depthStencil.back = {};   // Optional

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;  // Optional
    rasterizer.depthBiasClamp = 0.0f;           // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;     // Optional

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;           // Optional
    multisampling.pSampleMask = nullptr;             // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE;  // Optional
    multisampling.alphaToOneEnable = VK_FALSE;       // Optional

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;   // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;  // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;              // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;   // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;  // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;              // Optional

    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;  // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;  // Optional
    colorBlending.blendConstants[1] = 0.0f;  // Optional
    colorBlending.blendConstants[2] = 0.0f;  // Optional
    colorBlending.blendConstants[3] = 0.0f;  // Optional

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_LINE_WIDTH };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;


    auto pipelineLayout = getOrCreatePipelineLayout(state.program->program.parameterLayout, false);
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;  // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;  // Optional
    pipelineInfo.layout = pipelineLayout.pipelineLayout;
    pipelineInfo.renderPass = state.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // Optional
    pipelineInfo.basePipelineIndex = -1;               // Optional

    VkPipeline out;
    if (vkCreateGraphicsPipelines(device.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                  &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    return out;
}

VkPipeline VulkanPipelineCache::createComputePipeline(const VulkanPipelineState &state) {
    vk::ComputePipelineCreateInfo createInfo;
    auto pipelineLayout = getOrCreatePipelineLayout(state.program->program.parameterLayout, true);
    createInfo.layout = pipelineLayout.pipelineLayout;
    vk::Pipeline pipeline = device.device.createComputePipeline(nullptr, createInfo, nullptr);
    return pipeline;
}

VkRenderPass VulkanPipelineCache::getOrCreateRenderPass(VulkanSwapChain swapchain, VulkanRenderTarget *renderTarget) {
    auto it = renderpasses.get(renderTarget->attachments);
    if (it) {
        return *it;
    }
    VkRenderPass out;
    std::vector<VkAttachmentDescription> attachments;
    bool hasDepth = false;
    VkAttachmentReference depthAttachmentRef{};
    std::vector<VkAttachmentReference> colorAttachmentRef;

    if (renderTarget->surface) {
        attachments.push_back({ .format = (VkFormat)swapchain.format,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR });
        colorAttachmentRef.push_back({ .attachment = 0,
                                       .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
    } else {
        for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
            auto &color = renderTarget->attachments.colors[i];
            if (color.valid) {
                attachments.push_back({ .format = color.format,
                                        .samples = VK_SAMPLE_COUNT_1_BIT,
                                        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                        .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .finalLayout = VK_IMAGE_LAYOUT_GENERAL });

                colorAttachmentRef.push_back({ .attachment = (uint32_t)i,
                                               .layout = VK_IMAGE_LAYOUT_GENERAL });
            }
        }
        if (renderTarget->attachments.depth.valid) {
            depthAttachmentRef.attachment = attachments.size();
            attachments.push_back({ .format = renderTarget->attachments.depth.format,
                                    .samples = VK_SAMPLE_COUNT_1_BIT,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .finalLayout = VK_IMAGE_LAYOUT_GENERAL });

            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_GENERAL;
            hasDepth = true;
        }
    }

    VkSubpassDescription subpass{ .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  .colorAttachmentCount = (uint32_t)colorAttachmentRef.size(),
                                  .pColorAttachments = colorAttachmentRef.data(),
                                  .pDepthStencilAttachment = hasDepth ? &depthAttachmentRef : nullptr };

    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = (uint32_t)attachments.size(),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };

    if (vkCreateRenderPass(device.device, &renderPassInfo, nullptr, &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    renderpasses.add(renderTarget->attachments, out);
    return out;
}

vk::Framebuffer VulkanPipelineCache::getOrCreateFrameBuffer(vk::RenderPass renderPass,
                                                        VulkanSwapChain swapchain, 
                                                          VulkanRenderTarget *renderTarget) {
    auto it = framebuffers.get(
        std::make_pair(renderPass, renderTarget->surface ? swapchain.view : nullptr));
    if (it) {
        return *it;
    }

    std::vector<VkImageView> attachments;
    if (renderTarget->surface) {
        attachments.push_back(swapchain.view);
    } else {
        for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
            auto &color = renderTarget->attachments.colors[i];
            if (color.valid) {
                attachments.push_back(color.view);
            }
        }
        if (renderTarget->attachments.depth.valid) {
            attachments.push_back(renderTarget->attachments.depth.view);
        }
    }
    VkFramebuffer framebuffer;
    auto extent = renderTarget->getExtent(device);
    VkFramebufferCreateInfo framebufferInfo{ .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                             .renderPass = renderPass,
                                             .attachmentCount = (uint32_t)attachments.size(),
                                             .pAttachments = attachments.data(),
                                             .width = extent.width,
                                             .height = extent.height,
                                             .layers = 1 };

    if (vkCreateFramebuffer(device.device, &framebufferInfo, nullptr, &framebuffer) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }

    framebuffers.add(std::make_pair(renderPass, renderTarget->surface ? swapchain.view : nullptr),
                     framebuffer);
    return framebuffer;
}
