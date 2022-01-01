#pragma once

#include <array>
#include <string>
#include <vector>
#include <variant>
#include <VKGIRenderer/utils/Hash.h>
#include <VKGIRenderer/utils/FixedVector.h>

enum class ProgramType {
    VERTEX,
    FRAGMENT,
    COMPUTE
};

enum class ProgramParameterType : uint8_t {
    UNIFORM = 1,
    TEXTURE,
    IMAGE,
    ATTACHMENT,
    STORAGE,
    TLAS
};

struct ProgramParameterLayout {
    constexpr static size_t MAX_SET = 3;
    constexpr static size_t MAX_BINDINGS = 8;
    constexpr static size_t TABLE_SIZE = MAX_SET * MAX_BINDINGS;
    struct LayoutField {
        LayoutField() = default;
        LayoutField(ProgramParameterType type, uint32_t arraySize) {
            value = (static_cast<uint8_t>(type) << 24) | (arraySize & 0xFFFFFF);
        }
        ProgramParameterType type() const {
            return static_cast<ProgramParameterType>(value >> 24);
        }
        uint32_t arraySize() const {
            return value & 0xFFFFFF;
        }
        operator bool() const {
            return value != 0;
        }
        bool operator==(const LayoutField& other) const {
            return value == other.value;
        }
        uint32_t value{0};
    };

    ProgramParameterLayout() {
        std::fill(fields.begin(), fields.end(), LayoutField());
    }

    ProgramParameterLayout& addUniformBuffer(uint32_t set, uint32_t binding) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::UNIFORM, 1 };
        return *this;
    }

    ProgramParameterLayout& addUniformBufferArray(uint32_t set, uint32_t binding, size_t size) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::UNIFORM, (uint32_t)size };
        return *this;
    }

    ProgramParameterLayout& addStorageBuffer(uint32_t set, uint32_t binding) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::STORAGE, 1 };
        return *this;
    }

    ProgramParameterLayout& addAttachment(uint32_t set, uint32_t binding) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::ATTACHMENT, 1 };
        return *this;
    }

    ProgramParameterLayout& addTexture(uint32_t set, uint32_t binding) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::TEXTURE, 1 };
        return *this;
    }

    ProgramParameterLayout& addTextureArray(uint32_t set, uint32_t binding, size_t size) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::TEXTURE, (uint32_t)size };
        return *this;
    }

    ProgramParameterLayout& addStorageBufferArray(uint32_t set, uint32_t binding, size_t size) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::STORAGE, (uint32_t)size };
        return *this;
    }


    ProgramParameterLayout& addStorageImage(uint32_t set, uint32_t binding) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::IMAGE, 1 };
        return *this;
    }

    ProgramParameterLayout& addStorageImageArray(uint32_t set, uint32_t binding, size_t size) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::IMAGE, (uint32_t)size };
        return *this;
    }

    bool operator==(const ProgramParameterLayout& other) const {
        return fields == other.fields;
    }

    std::array<LayoutField, TABLE_SIZE> fields{};
};

using ProgramHash = uint64_t;
using CompiledCode = std::vector<uint32_t>;

struct ProgramParameter {
    uint32_t set{};
    uint32_t binding{};
    ProgramParameterType type{};
    uint32_t arraySize{1};
};

struct Program {
    Program(CompiledCode compCode) {
        _code = compCode;
        hash = hashBuffer<uint32_t>(_code.data(), _code.size());
    }

    const CompiledCode& code() const {
        return _code;
    }

    ProgramParameterLayout parameterLayout;
    ProgramType type;
    ProgramHash hash;
  private:
    CompiledCode _code;
};

