#pragma once

#include "DriverTypes.h"
#include "Handles.h"

enum class CompareOp : uint8_t {
    NEVER,
    LESS,
    EQUAL,
    LESS_OR_EQUAL,
    GREATER,
    NOT_EQUAL,
    GREATER_OR_EQUAL,
    ALWAYS
};

struct DepthTest {
    uint8_t enabled{0};
    uint8_t write{0};
    CompareOp compareOp{CompareOp::NEVER};
    bool operator==(const DepthTest& other) const {
        return (bool)enabled == (bool)other.enabled && (bool)write == (bool)other.write && compareOp == other.compareOp;
    }
};

struct PipelineState {
    PipelineState() = default;
    Handle<HwProgram> program{};
    Viewport scissor{ 0, 0, (uint32_t)std::numeric_limits<int32_t>::max(),
                      (uint32_t)std::numeric_limits<int32_t>::max() };
    DepthTest depthTest{};
};
