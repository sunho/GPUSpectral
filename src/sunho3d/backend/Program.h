#pragma once

#include <array>
#include <string>
#include <vector>
#include <variant>

using ProgramCode = std::vector<char>;

enum class ProgramType {
    PIPELINE,
    COMPUTE
};

enum class ProgramParameterType : uint8_t {
    UNIFORM = 1,
    TEXTURE,
    ATTACHMENT,
    STORAGE
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
        uint32_t value{0};
    };

    ProgramParameterLayout() {
        std::fill(fields.begin(), fields.end(), LayoutField());
    }

    ProgramParameterLayout& addUniformBuffer(uint32_t set, uint32_t binding) {
        fields[set * MAX_BINDINGS + binding] = { ProgramParameterType::UNIFORM, 1 };
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
    bool operator==(const ProgramParameterLayout& other) const {
        return fields == other.fields;
    }

    std::array<LayoutField, TABLE_SIZE> fields{};
};

struct ProgramParameter {
    uint32_t set{};
    uint32_t binding{};
    ProgramParameterType type{};
    uint32_t arraySize{1};
};

struct Program {
    explicit Program(const char* vertCode, size_t vertSize, const char* fragCode, size_t fragSize)
        : type(ProgramType::PIPELINE) {
        codes[0] = std::vector(vertCode, vertCode + vertSize);
		codes[1] = std::vector(fragCode, fragCode + fragSize);
    }

    const ProgramCode& vertex() const {
        return codes[0];
    }
    
    const ProgramCode& frag() const {
        return codes[1];
    }
    
    const ProgramCode& compute() const {
        return codes[0];
    }

    ProgramParameterLayout parameterLayout;
    ProgramType type;


  private:
    std::array<ProgramCode, 2> codes;
};

