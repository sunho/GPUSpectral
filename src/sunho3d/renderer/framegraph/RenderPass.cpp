#include "RenderPass.h"

RenderPass::RenderPass(const std::string& name, std::vector<FGResource> inputs, std::vector<FGResource> outputs, RenderPassFunc func)
    : name(name), inputs(inputs), outputs(outputs), func(func) {
}

void RenderPass::run(FrameGraph& fg) {
    func(fg);
}
