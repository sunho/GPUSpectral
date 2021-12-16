#include "FrameGraph.h"
#include <stack>
#include <Tracy.hpp>
#include <iostream>


FrameGraph::FrameGraph(VKGIRenderer::VulkanDriver& driver)
    : driver(driver) {
}

void FrameGraph::submit() {
    compile();
    run();
}

void FrameGraph::addFramePass(FramePass pass) {
    passes.push_back(pass);
}

Handle<HwBufferObject> FrameGraph::createTempUniformBuffer(void* data, size_t size) {
    auto buffer = createBufferObjectSC(size, BufferUsage::UNIFORM | BufferUsage::TRANSFER_DST);
    auto staging = createBufferObjectSC(size, BufferUsage::STAGING);
    driver.updateStagingBufferObject(staging, { .data = (uint32_t*)data, .size = size }, 0);
    driver.copyBufferObject(buffer, staging);
    return buffer;
}


FrameGraph::~FrameGraph() {
    for (auto d : destroyers) {
        d();
    }
}

static void dfs(const std::vector<std::vector<size_t>>& graph, std::vector<bool>& visited, std::vector<size_t>& runOrder, size_t i) {
    if (visited[i])
        return;
    visited[i] = true;
    for (auto j : graph[i]) {
        if (!visited[j])
            dfs(graph, visited, runOrder, j);
    }
    runOrder.push_back(i);
}

static std::pair<BarrierStageMask, BarrierAccessFlag> convertAccessTypeToBarrierStage(const ResourceAccessType& type) {
    switch (type) {
        case ResourceAccessType::ColorWrite:
            return std::make_pair(BarrierStageMask::COLOR_ATTACHMENT_OUTPUT, BarrierAccessFlag::COLOR_WRITE);
        case ResourceAccessType::ComputeRead:
            return std::make_pair(BarrierStageMask::COMPUTE, BarrierAccessFlag::SHADER_READ);
        case ResourceAccessType::ComputeWrite:
            return std::make_pair(BarrierStageMask::COMPUTE, BarrierAccessFlag::SHADER_WRITE);
        case ResourceAccessType::FragmentRead:
            return std::make_pair(BarrierStageMask::FRAGMENT_SHADER, BarrierAccessFlag::SHADER_READ);
        case ResourceAccessType::DepthWrite:
            return std::make_pair(BarrierStageMask::EARLY_FRAGMENT_TESTS | BarrierStageMask::LATE_FRAGMENT_TESTS, BarrierAccessFlag::DEPTH_STENCIL_WRITE);     
        case ResourceAccessType::TransferRead:
            return std::make_pair(BarrierStageMask::TRANSFER, BarrierAccessFlag::TRANSFER_READ);
        case ResourceAccessType::TransferWrite:
            return std::make_pair(BarrierStageMask::TRANSFER, BarrierAccessFlag::TRANSFER_WRITE);
    }
}

static Barrier generateBufferBarrier(const BakedPassResource& prev, const BakedPassResource& current) {
    Barrier barrier = {};
    auto src = convertAccessTypeToBarrierStage(prev.accessType);
    barrier.srcStage = src.first;
    barrier.srcAccess = src.second;
    auto dest = convertAccessTypeToBarrierStage(current.accessType);
    barrier.dstStage = dest.first;
    barrier.dstAccess = dest.second;
    return barrier;
}

static ImageLayout decideImageLayout(const ResourceAccessType& type) {
    switch (type) {
        case ResourceAccessType::ColorWrite:
            return ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        case ResourceAccessType::ComputeRead:
            return ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        case ResourceAccessType::ComputeWrite:
            return ImageLayout::GENERAL;
        case ResourceAccessType::FragmentRead:
            return ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        case ResourceAccessType::DepthWrite:
            return ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
        case ResourceAccessType::TransferRead:
            return ImageLayout::TRANSFER_SRC_OPTIMAL;
        case ResourceAccessType::TransferWrite:
            return ImageLayout::TRANSFER_DST_OPTIMAL;
    }
}

static Barrier generateImageBarrier(const BakedPassResource& prev, const BakedPassResource& current, Handle<HwTexture> image) {
    Barrier barrier = {};
    auto src = convertAccessTypeToBarrierStage(prev.accessType);
    barrier.srcStage = src.first;
    barrier.srcAccess = src.second;
    auto dest = convertAccessTypeToBarrierStage(current.accessType);
    barrier.dstStage = dest.first;
    barrier.dstAccess = dest.second;
    barrier.image = image;
    barrier.initialLayout = decideImageLayout(prev.accessType);
    barrier.finalLayout = decideImageLayout(current.accessType);
    return barrier;
}

void FrameGraph::compile() {
    ZoneScopedN("Frame graph compile")
    std::unordered_map<size_t, size_t> resourceIdToRp;
    std::vector<BakedPass> bakedPasses(passes.size());
    std::vector<std::vector<size_t>> graph(passes.size());
    for (size_t i = 0; i < passes.size(); ++i) {
        bakedPasses[i] = BakedPass(*this, passes[i]);
    }

    for (size_t i = 0; i < bakedPasses.size(); ++i) {
        for (auto r : bakedPasses[i].outputs) {
            resourceIdToRp.emplace(r.resource.getId(), i);
        }
    }

    for (size_t i = 0; i < bakedPasses.size(); ++i) {
        for (auto r : bakedPasses[i].inputs) {
            auto it = resourceIdToRp.find(r.resource.getId());
            if (it != resourceIdToRp.end()) {
                graph[i].push_back(it->second);
            }
        }
    }

    /*
    // topological sort
    // in dfs spanning tree
    // we should place what's on the left early (cross edge)
    // and what's on the bottom early
    std::vector<size_t> runOrder;
    std::vector<bool> visited(bakedPasses.size());
    for (int i = 0; i < bakedPasses.size(); ++i) {
        dfs(graph, visited, runOrder, i);
    }
    for (auto idx : runOrder) {
        bakedGraph.passes.push_back(bakedPasses[idx]);
    }
    */
    for (auto pass : bakedPasses) {
        bakedGraph.passes.push_back(pass);
    }

    for (int i = 0; i < bakedGraph.passes.size(); ++i) {
        auto& pass = bakedGraph.passes[i];
        for (auto [_, res] : pass.resources) {
            bakedGraph.useChains.emplace(std::make_pair(res.resource.getId(), i));
            bakedGraph.usedRes.emplace(res.resource.getId());
        }
    }

    for (auto res : bakedGraph.usedRes) {
        auto iterBegin = bakedGraph.useChains.lower_bound(std::make_pair(res, 0));
        auto iterEnd = bakedGraph.useChains.upper_bound(std::make_pair(res, std::numeric_limits<uint32_t>::max()));
        for (auto iter = iterBegin; iter != iterEnd; ++iter) {
            auto& pass = bakedGraph.passes[iter->second];
            if (iter != iterBegin) {
                auto b = std::prev(iter);
                auto& prevPass = bakedGraph.passes[b->second];
                auto prevRes = prevPass.resources.at(iter->first);
                auto res = pass.resources.at(iter->first);
                auto type = res.resource.getType();
                if (isWriteAccessType(prevRes.accessType)) { 
                    if (type == ResourceType::Image) {
                        auto res = pass.resources.at(iter->first);
                        // hack: we should do this translation in vulkanrenderpass.attachments
                        if (prevRes.accessType == ResourceAccessType::ColorWrite || prevRes.accessType == ResourceAccessType::DepthWrite) {
                            auto image = *getResource<Handle<HwTexture>>(res.resource);
                            Barrier barrier = {};
                            auto src = convertAccessTypeToBarrierStage(prevRes.accessType);
                            barrier.srcStage = src.first;
                            barrier.srcAccess = src.second;
                            auto dest = convertAccessTypeToBarrierStage(res.accessType);
                            barrier.dstStage = dest.first;
                            barrier.dstAccess = dest.second;
                            barrier.image = image;
                            barrier.initialLayout = ImageLayout::GENERAL;
                            barrier.finalLayout = decideImageLayout(res.accessType);
                            pass.barriers.push_back(barrier);
                        } else {
                            auto image = *getResource<Handle<HwTexture>>(res.resource);
                            auto barrier = generateImageBarrier(prevRes, res, image);
                            pass.barriers.push_back(barrier);
                        }
                    } else {
                        pass.barriers.push_back(generateBufferBarrier(prevRes, res));
                    }
                } else {
                    if (isWriteAccessType(res.accessType)) {
                        if (type == ResourceType::Image) {
                            auto image = *getResource<Handle<HwTexture>>(res.resource);
                            auto barrier = generateImageBarrier(prevRes, res, image);
                            pass.barriers.push_back(barrier);
                        } else {
                            auto barrier = generateBufferBarrier(prevRes, res);
                            barrier.srcAccess = BarrierAccessFlag::NONE;
                            barrier.dstAccess = BarrierAccessFlag::NONE;
                            pass.barriers.push_back(barrier);
                        }
                    }
                }
            } else {
                auto res = pass.resources.at(iter->first);
                auto type = res.resource.getType();
                if (type == ResourceType::Image) {
                    auto image = *getResource<Handle<HwTexture>>(res.resource);
                    Barrier barrier = {};

                    auto dest = convertAccessTypeToBarrierStage(res.accessType);
                    barrier.dstStage = dest.first;
                    barrier.dstAccess = dest.second;
                    barrier.srcStage = BarrierStageMask::TOP_OF_PIPE;
                    barrier.srcAccess = BarrierAccessFlag::NONE;
                    barrier.image = image;
                    barrier.initialLayout = driver.getTextureImageLayout(image);
                    barrier.finalLayout = decideImageLayout(res.accessType);
                    pass.barriers.push_back(barrier);
                } else {
                    auto barrier = generateBufferBarrier(res, res);
                    barrier.srcStage = BarrierStageMask::TRANSFER;
                    barrier.srcAccess = BarrierAccessFlag::TRANSFER_WRITE;
                    barrier.dstAccess = BarrierAccessFlag::SHADER_READ;
                    pass.barriers.push_back(barrier);
                }
            }
        }
    }
}

void FrameGraph::run() {
    for (auto pass : bakedGraph.passes) {
        ZoneTransientN(zone, pass.name.c_str(), true)
        driver.setProfileSectionName(pass.name.c_str());
        for (auto& barrier : pass.barriers) {
            driver.setBarrier(barrier);
        }
        std::cout << "pass start name: " << pass.name << std::endl;
        pass.func(*this);
        std::cout << "pass end name: " << pass.name << std::endl;
    }
}

FrameGraph::BakedPass::BakedPass(FrameGraph& fg, FramePass pass) : name(pass.name), func(pass.func) {
    for (auto buf : pass.buffers) {
        for (auto b : buf.resource) {
            auto res = fg.getOrRegisterBufferResource(b);
            if (isWriteAccessType(buf.accessType)) {
                outputs.push_back({ res, buf.accessType });
            }
            else {
                inputs.push_back({ res, buf.accessType });
            }
            resources.emplace(res.getId(), BakedPassResource{ res, buf.accessType });
        }
    }

    for (auto tex : pass.textures) {
        for (auto t : tex.resource) {
            auto res = fg.getOrRegisterTextureResource(t);
            if (isWriteAccessType(tex.accessType)) {
                outputs.push_back({ res, tex.accessType });
            }
            else {
                inputs.push_back({ res, tex.accessType });
            }
            resources.emplace(res.getId(), BakedPassResource{ res, tex.accessType });
        }
    }

}