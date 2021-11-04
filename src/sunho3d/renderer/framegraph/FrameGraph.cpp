#include "FrameGraph.h"
#include "../Renderer.h"
#include <stack>


FrameGraph::FrameGraph(sunho3d::Renderer& renderer)
    : parent(renderer), driver(renderer.getDriver()) {
}

void FrameGraph::reset() {
    for (auto d : destroyers) {
        d();
    }
    destroyers.clear();
    resources.clear();
    renderPasses.clear();
    runOrder.clear();
}

void FrameGraph::addRenderPass(const std::string& name, std::vector<FGResource> inputs, std::vector<FGResource> outputs, RenderPass::RenderPassFunc func) {
    addRenderPass(RenderPass(name, inputs, outputs, func));
}

void FrameGraph::submit() {
    compile();
    run();
}


FrameGraph::~FrameGraph() {
    for (auto d : destroyers) {
        d();
    }
}

void FrameGraph::addRenderPass(const RenderPass& pass) {
    renderPasses.push_back(pass);
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

void FrameGraph::compile() {
    std::unordered_map<size_t, size_t> resourceIdToRp;
    std::vector<std::vector<size_t>> graph(renderPasses.size());

    for (size_t i = 0; i < renderPasses.size(); ++i) {
        for (auto r : renderPasses[i].getOutputs()) {
            resourceIdToRp.emplace(r.id, i);
        }
    }

    for (size_t i = 0; i < renderPasses.size(); ++i) {
        for (auto r : renderPasses[i].getInputs()) {
            auto it = resourceIdToRp.find(r.id);
            if (it != resourceIdToRp.end()) {
                graph[i].push_back(it->second);
            }
        }
    }

    // topological sort
    // in dfs spanning tree
    // we should place what's on the left early (cross edge)
    // and what's on the bottom early
    runOrder.clear();
    std::vector<bool> visited(renderPasses.size());
    for (int i = 0; i < renderPasses.size(); ++i) {
        dfs(graph, visited, runOrder, i);
    }
}

void FrameGraph::run() {
    for (auto index : runOrder) {
        renderPasses[index].run(*this);
    }
}
