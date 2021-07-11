#pragma once

#include "Handles.h"
#include "DriverTypes.h"

struct PipelineState {
    PipelineState() = default;
    Handle<HwProgram> program{};
    Viewport scissor{ 0, 0,
                      (uint32_t)std::numeric_limits<int32_t>::max(),
                      (uint32_t)std::numeric_limits<int32_t>::max()
    };
};
