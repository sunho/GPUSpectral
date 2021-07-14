#pragma once

#include <unordered_map>

#include "../utils/FixedVector.h"
#include "RenderPass.h"
#include "Resource.h"

class FrameGraph {
  public:
    FrameGraph() = default;
    ~FrameGraph();

    void addRenderPass(const RenderPass& pass);
    void compile();
    void run();

    template <typename T>
    FGResource declareResource(const std::string& name) {
        resources.emplace(nextId, FixedVector<char>(sizeof(T)));
        FGResource resource = {
            .name = name,
            .id = nextId++
        };
        return resource;
    }

    template <typename T>
    T getResource(const FGResource& resource) {
        auto& data = resources.find(resource.id)->second;
        return *reinterpret_cast<T*>(data.data());
    }

    template <typename T>
    void defineResource(const FGResource& resource, const T& t) {
        auto& data = resources.find(resource.id)->second;
        auto addr = reinterpret_cast<T*>(data.data());
        *addr = t;
    }

  private:
    std::unordered_map<uint32_t, FixedVector<char>> resources;
    std::vector<RenderPass> renderPasses;
    std::vector<size_t> runOrder;
    uint32_t nextId{ 1 };
};
