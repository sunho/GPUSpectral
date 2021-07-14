#pragma once

#include <string>
#include <vector>

#include "Resource.h"

class FrameGraph;

class RenderPass {
  public:
    using RenderPassFunc = std::function<void(FrameGraph& fg)>;
    explicit RenderPass(const std::string& name, std::vector<FGResource> inputs, std::vector<FGResource> outputs, RenderPassFunc func);

    void run(FrameGraph& fg);
    const std::string& getName() const {
        return name;
    }
    void setName(const std::string& name) {
        this->name = name;
    }

    const std::vector<FGResource>& getInputs() const {
        return inputs;
    }
    const std::vector<FGResource>& getOutputs() const {
        return outputs;
    }

  private:
    std::string name;
    std::vector<FGResource> inputs;
    std::vector<FGResource> outputs;
    RenderPassFunc func;
};
