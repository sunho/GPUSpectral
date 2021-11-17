#pragma once

#include "DriverTypes.h"
#include "Handles.h"
#include "Program.h"

#include <map>

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

struct BindingIndex {
    uint32_t set{};
    uint32_t binding{};
    bool operator==(const BindingIndex& other) const = default;
    bool operator<(const BindingIndex& other) const {
        return std::tie(set, binding) < std::tie(other.set, other.binding);
    }
};

union BindingHandle {
    Handle<HwTexture> texture;
    Handle<HwBufferObject> buffer;
};

struct Binding {
    constexpr static size_t MAX_SIZE = 32;
    Binding() {
        memset(handles.data(), 0xFF, handles.size() * sizeof(BindingHandle)); // TODO: this is based on internal of HandleBase should change it
    }
    ProgramParameterType type{};
    std::array<BindingHandle, MAX_SIZE> handles{};
};

using BindingMap = std::map<BindingIndex, Binding>;

struct PipelineState {
    PipelineState() = default;
    Handle<HwProgram> program{};
    Viewport scissor{ 0, 0, (uint32_t)std::numeric_limits<int32_t>::max(),
                      (uint32_t)std::numeric_limits<int32_t>::max() };
    DepthTest depthTest{};
    BindingMap bindings{};
    std::vector<uint8_t> pushConstants{};

    PipelineState& copyPushConstants(void* buf, size_t size) {
        pushConstants.resize(size);
        memcpy(pushConstants.data(), buf, size);
        return *this;
    }

    PipelineState& bindUniformBuffer(uint32_t set, uint32_t binding, Handle<HwBufferObject> buffer) {
        Binding b = {};
        b.type = ProgramParameterType::UNIFORM;
        b.handles[0] = {.buffer = buffer};
        bindings[{ set, binding }] = b;
        return *this;
    }

    PipelineState& bindStorageBufferArray(uint32_t set, uint32_t binding, Handle<HwBufferObject>* buffers, size_t size) {
        Binding b = {};
        b.type = ProgramParameterType::STORAGE;
        for (size_t i = 0; i < size; ++i) {
            b.handles[i] = { .buffer = buffers[i] };
        }
        bindings[{ set, binding }] = b;
        return *this;
    }

    PipelineState& bindTextureArray(uint32_t set, uint32_t binding, Handle<HwTexture>* texture, size_t size) {
        Binding b = {};
        b.type = ProgramParameterType::TEXTURE;
        for (size_t i = 0; i < size; ++i) {
            b.handles[i] = { .texture = texture[i] };
        }
        bindings[{ set, binding }] = b;
        return *this;
    }

    PipelineState& bindTexture(uint32_t set, uint32_t binding, Handle<HwTexture> texture) {
        Binding b = {};
        b.type = ProgramParameterType::TEXTURE;
        b.handles[0] = { .texture = texture };
        bindings[{ set, binding }] = b;
        return *this;
    }

     PipelineState& bindStorageImage(uint32_t set, uint32_t binding, Handle<HwTexture> texture) {
        Binding b = {};
        b.type = ProgramParameterType::IMAGE;
        b.handles[0] = { .texture = texture };
        bindings[{ set, binding }] = b;
        return *this;
    }

    PipelineState& bindStorageBuffer(uint32_t set, uint32_t binding, Handle<HwBufferObject> buffer) {
        Binding b = {};
        b.type = ProgramParameterType::STORAGE;
        b.handles[0] = {.buffer = buffer};
        bindings[{ set, binding }] = b;
        return *this;
    }
};

