#include "VulkanPipelineCache.h"

void VulkanPipelineCache::init(VulkanContext &context) {
    dummyBufferWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dummyBufferWriteInfo.pNext = nullptr;
    dummyBufferWriteInfo.dstArrayElement = 0;
    dummyBufferWriteInfo.descriptorCount = 1;
    dummyBufferWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    dummyBufferWriteInfo.pImageInfo = nullptr;
    dummyBufferWriteInfo.pBufferInfo = &dummyBufferInfo;
    dummyBufferWriteInfo.pTexelBufferView = nullptr;

    dummySamplerWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dummySamplerWriteInfo.pNext = nullptr;
    dummySamplerWriteInfo.dstArrayElement = 0;
    dummySamplerWriteInfo.descriptorCount = 1;
    dummySamplerWriteInfo.pImageInfo = &dummySamplerInfo;
    dummySamplerWriteInfo.pBufferInfo = nullptr;
    dummySamplerWriteInfo.pTexelBufferView = nullptr;
    dummySamplerWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    dummyTargetInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dummyTargetWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dummyTargetWriteInfo.pNext = nullptr;
    dummyTargetWriteInfo.dstArrayElement = 0;
    dummyTargetWriteInfo.descriptorCount = 1;
    dummyTargetWriteInfo.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    dummyTargetWriteInfo.pImageInfo = &dummyTargetInfo;
    dummyTargetWriteInfo.pBufferInfo = nullptr;
    dummyTargetWriteInfo.pTexelBufferView = nullptr;
    setupDescriptorLayout(context);

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 16,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    };

    dummyBuffer = new VulkanBufferObject(context, 16, BufferUsage::UNIFORM);
    dummyBufferInfo.buffer = dummyBuffer->buffer;
    dummyBufferInfo.range = bufferInfo.size;

    this->context = &context;

    pipelines.setDestroyer([=](VkPipeline pipeline) {
        vkDestroyPipeline(this->context->device, pipeline, nullptr);
    });
    framebuffers.setDestroyer([=](VkFramebuffer framebuffer) {
        vkDestroyFramebuffer(this->context->device, framebuffer, nullptr);
    });
    descriptorSets.setDestroyer([=](std::array<VkDescriptorSet, 3> descriptors) {
        vkFreeDescriptorSets(this->context->device, descriptorPool, 3, descriptors.data());
    });
    renderpasses.setDestroyer([=](VkRenderPass renderPass) {
        vkDestroyRenderPass(this->context->device, renderPass, nullptr);
    });
}

void VulkanPipelineCache::tick() {
    pipelines.tick();
    framebuffers.tick();
    descriptorSets.tick();
    renderpasses.tick();
}

void VulkanPipelineCache::setupDescriptorLayout(VulkanContext &context) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    pipelineLayoutInfo.pushConstantRangeCount = 0;     // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr;  // Optional

    VkDescriptorSetLayoutBinding binding = {};
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;

    std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::UBUFFER_BINDING_COUNT> ubindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    for (uint32_t i = 0; i < VulkanDescriptor::UBUFFER_BINDING_COUNT; i++) {
        binding.binding = i;
        ubindings[i] = binding;
    }
    VkDescriptorSetLayoutCreateInfo dlinfo = {};
    dlinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlinfo.bindingCount = ubindings.size();
    dlinfo.pBindings = ubindings.data();
    vkCreateDescriptorSetLayout(context.device, &dlinfo, nullptr, &descriptorSetLayout[0]);

    std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::SAMPLER_BINDING_COUNT> sbindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    for (uint32_t i = 0; i < VulkanDescriptor::SAMPLER_BINDING_COUNT; i++) {
        binding.binding = i;
        sbindings[i] = binding;
    }
    dlinfo.bindingCount = sbindings.size();
    dlinfo.pBindings = sbindings.data();
    vkCreateDescriptorSetLayout(context.device, &dlinfo, nullptr, &descriptorSetLayout[1]);

    std::array<VkDescriptorSetLayoutBinding, VulkanDescriptor::TARGET_BINDING_COUNT> bindings;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 0; i < VulkanDescriptor::TARGET_BINDING_COUNT; i++) {
        binding.binding = i;
        bindings[i] = binding;
    }
    dlinfo.bindingCount = bindings.size();
    dlinfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(context.device, &dlinfo, nullptr, &descriptorSetLayout[2]);

    pipelineLayoutInfo.setLayoutCount = 3;
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayout.data();

    if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    VkDescriptorPoolSize poolSizes[3] = {};
    VkDescriptorPoolCreateInfo poolInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                         .maxSets = 1024 * 3,
                                         .poolSizeCount = 3,
                                         .pPoolSizes = poolSizes };
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = poolInfo.maxSets * VulkanDescriptor::UBUFFER_BINDING_COUNT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = poolInfo.maxSets * VulkanDescriptor::SAMPLER_BINDING_COUNT;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    poolSizes[2].descriptorCount = poolInfo.maxSets * VulkanDescriptor::TARGET_BINDING_COUNT;
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void VulkanPipelineCache::getOrCreateDescriptors(VulkanContext &context,
                                                 const VulkanDescriptor &key,
                                                 std::array<VkDescriptorSet, 3> &descriptors) {
    auto it = descriptorSets.get(key);
    if (it) {
        descriptors = *it;
        return;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 3;
    allocInfo.pSetLayouts = descriptorSetLayout.data();
    if (vkAllocateDescriptorSets(context.device, &allocInfo, descriptors.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }
    descriptorSets.add(key, descriptors);
}

void VulkanPipelineCache::bindDescriptor(VulkanContext &context, const VulkanDescriptor &key) {
    std::array<VkDescriptorSet, 3> descriptors;
    getOrCreateDescriptors(context, key, descriptors);

    dummySamplerInfo.imageLayout = dummyTargetInfo.imageLayout;
    dummySamplerInfo.imageView = dummyImageView;
    dummyTargetInfo.imageView = dummyImageView;

    if (dummySamplerInfo.sampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerInfo{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                         .magFilter = VK_FILTER_NEAREST,
                                         .minFilter = VK_FILTER_NEAREST,
                                         .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                         .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                         .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                         .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                         .anisotropyEnable = VK_FALSE,
                                         .maxAnisotropy = 1,
                                         .compareEnable = VK_FALSE,
                                         .compareOp = VK_COMPARE_OP_ALWAYS,
                                         .minLod = 0.0f,
                                         .maxLod = 1.0f,
                                         .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                                         .unnormalizedCoordinates = VK_FALSE };
        vkCreateSampler(context.device, &samplerInfo, nullptr, &dummySamplerInfo.sampler);
    }

    VkDescriptorBufferInfo descriptorBuffers[VulkanDescriptor::UBUFFER_BINDING_COUNT];
    VkDescriptorImageInfo descriptorSamplers[VulkanDescriptor::SAMPLER_BINDING_COUNT];
    VkDescriptorImageInfo descriptorInputAttachments[VulkanDescriptor::TARGET_BINDING_COUNT];
    VkWriteDescriptorSet
        descriptorWrites[VulkanDescriptor::UBUFFER_BINDING_COUNT + VulkanDescriptor::SAMPLER_BINDING_COUNT + VulkanDescriptor::TARGET_BINDING_COUNT];
    uint32_t nwrites = 0;
    VkWriteDescriptorSet *writes = descriptorWrites;
    nwrites = 0;
    for (uint32_t binding = 0; binding < VulkanDescriptor::UBUFFER_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet &writeInfo = writes[nwrites++];
        if (key.uniformBuffers[binding]) {
            VkDescriptorBufferInfo &bufferInfo = descriptorBuffers[binding];
            bufferInfo.buffer = key.uniformBuffers[binding];
            bufferInfo.offset = key.uniformBufferOffsets[binding];
            bufferInfo.range = key.uniformBufferSizes[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeInfo.pImageInfo = nullptr;
            writeInfo.pBufferInfo = &bufferInfo;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = dummyBufferWriteInfo;
        }
        writeInfo.dstSet = descriptors[0];
        writeInfo.dstBinding = binding;
    }
    for (uint32_t binding = 0; binding < VulkanDescriptor::SAMPLER_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet &writeInfo = writes[nwrites++];
        if (key.samplers[binding].sampler) {
            VkDescriptorImageInfo &imageInfo = descriptorSamplers[binding];
            imageInfo = key.samplers[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeInfo.pImageInfo = &imageInfo;
            writeInfo.pBufferInfo = nullptr;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = dummySamplerWriteInfo;
        }
        writeInfo.dstSet = descriptors[1];
        writeInfo.dstBinding = binding;
    }
    for (uint32_t binding = 0; binding < VulkanDescriptor::TARGET_BINDING_COUNT; binding++) {
        VkWriteDescriptorSet &writeInfo = writes[nwrites++];
        if (key.inputAttachments[binding].imageView) {
            VkDescriptorImageInfo &imageInfo = descriptorInputAttachments[binding];
            imageInfo = key.inputAttachments[binding];
            writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfo.pNext = nullptr;
            writeInfo.dstArrayElement = 0;
            writeInfo.descriptorCount = 1;
            writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            writeInfo.pImageInfo = &imageInfo;
            writeInfo.pBufferInfo = nullptr;
            writeInfo.pTexelBufferView = nullptr;
        } else {
            writeInfo = dummyTargetWriteInfo;
        }
        writeInfo.dstSet = descriptors[2];
        writeInfo.dstBinding = binding;
    }
    vkUpdateDescriptorSets(context.device, nwrites, writes, 0, nullptr);
    vkCmdBindDescriptorSets(context.commands->get(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 3, descriptors.data(), 0, nullptr);
}

VkPipeline VulkanPipelineCache::getOrCreatePipeline(VulkanContext &context,
                                                    const VulkanPipelineKey &key) {
    auto it = pipelines.get(key);
    if (it) {
        return *it;
    }

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = key.program->vertex;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = key.program->fragment;
    fragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = key.attributeCount;
    std::vector<VkVertexInputAttributeDescription> attributes(key.attributeCount);
    std::vector<VkVertexInputBindingDescription> bindings(key.attributeCount);

    for (uint32_t i = 0; i < key.attributeCount; ++i) {
        Attribute attrib = key.attributes[i];
        attributes[i] = { .location = i,
                          .binding = i,
                          .format = context.translateElementFormat(attrib.type,
                                                                   attrib.flags & Attribute::FLAG_NORMALIZED, false) };
        bindings[i] = {
            .binding = i,
            .stride = attrib.stride,
        };
    }

    vertexInputInfo.pVertexBindingDescriptions = bindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = key.attributeCount;
    vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = (float)key.viewport.left;
    viewport.y = (float)key.viewport.top;
    viewport.width = (float)key.viewport.width;
    viewport.height = (float)key.viewport.height;
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
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
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
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = key.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // Optional
    pipelineInfo.basePipelineIndex = -1;               // Optional
    VkPipeline out;
    if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                  &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    pipelines.add(key, out);
    return out;
}

VkRenderPass VulkanPipelineCache::getOrCreateRenderPass(VulkanContext &context,
                                                        VulkanRenderTarget *renderTarget) {
    auto it = renderpasses.get(renderTarget);
    if (it) {
        return *it;
    }
    VkRenderPass out;
    std::vector<VkAttachmentDescription> attachments;
    VkAttachmentReference depthAttachmentRef{};
    std::vector<VkAttachmentReference> colorAttachmentRef;

    if (renderTarget->surface) {
        attachments.push_back({ .format = context.surface->format.format,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR });
        colorAttachmentRef.push_back({ .attachment = 0,
                                       .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

        attachments.push_back({ .format = VK_FORMAT_D32_SFLOAT,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL });

        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    } else {
        for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
            auto &texture = renderTarget->color[i].texture;
            if (texture) {
                attachments.push_back({ .format = texture->vkFormat,
                                        .samples = VK_SAMPLE_COUNT_1_BIT,
                                        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                        .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .finalLayout = VK_IMAGE_LAYOUT_GENERAL });

                colorAttachmentRef.push_back({ .attachment = (uint32_t)i,
                                               .layout = VK_IMAGE_LAYOUT_GENERAL });
            }
        }
        if (renderTarget->depth.texture) {
            depthAttachmentRef.attachment = attachments.size();
            attachments.push_back({ .format = renderTarget->depth.texture->vkFormat,
                                    .samples = VK_SAMPLE_COUNT_1_BIT,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .finalLayout = VK_IMAGE_LAYOUT_GENERAL });

            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    VkSubpassDescription subpass{ .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  .colorAttachmentCount = (uint32_t)colorAttachmentRef.size(),
                                  .pColorAttachments = colorAttachmentRef.data(),
                                  .pDepthStencilAttachment = &depthAttachmentRef };

    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = (uint32_t)attachments.size(),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };

    if (vkCreateRenderPass(context.device, &renderPassInfo, nullptr, &out) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    renderpasses.add(renderTarget, out);
    return out;
}

VkFramebuffer VulkanPipelineCache::getOrCreateFrameBuffer(VulkanContext &context,
                                                          VkRenderPass renderPass,
                                                          VulkanRenderTarget *renderTarget) {
    auto it = framebuffers.get(
        std::make_pair(renderTarget, context.currentSwapContext->attachment.view));
    if (it) {
        return *it;
    }

    std::vector<VkImageView> attachments;
    if (renderTarget->surface) {
        attachments.push_back(context.currentSwapContext->attachment.view);
        attachments.push_back(renderTarget->depth.view);
    } else {
        for (size_t i = 0; i < ColorAttachment::MAX_MRT_NUM; ++i) {
            auto &texture = renderTarget->color[i].texture;
            if (texture) {
                attachments.push_back(texture->view);
            }
        }
        if (renderTarget->depth.texture) {
            attachments.push_back(renderTarget->depth.texture->view);
        }
    }
    VkFramebuffer framebuffer;
    VkFramebufferCreateInfo framebufferInfo{ .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                             .renderPass = renderPass,
                                             .attachmentCount = (uint32_t)attachments.size(),
                                             .pAttachments = attachments.data(),
                                             .width = renderTarget->width,
                                             .height = renderTarget->height,
                                             .layers = 1 };

    if (vkCreateFramebuffer(context.device, &framebufferInfo, nullptr, &framebuffer) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }

    framebuffers.add(std::make_pair(renderTarget, context.currentSwapContext->attachment.view),
                     framebuffer);
    return framebuffer;
}
