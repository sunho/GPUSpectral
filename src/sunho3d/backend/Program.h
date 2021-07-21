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

enum class BindingType {
    UNIFORM,
    TEXTURE,
    ATTACHMENT,
    STORAGE
};

struct Binding {
    constexpr static const int32_t nullIndex = -1;
    Binding() = default;
    int32_t index{nullIndex};
    BindingType type{};
};

struct Program {
    constexpr static const size_t NUM_BINDINGS = 12;

    explicit Program(const char* vertCode, size_t vertSize, const char* fragCode, size_t fragSize) : type(ProgramType::PIPELINE) {
        codes[0] = std::vector(vertCode, vertCode + vertSize);
		codes[1] = std::vector(fragCode, fragCode + fragSize);
        std::fill(bindings.begin(), bindings.end(), Binding());
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
    std::array<Binding, NUM_BINDINGS> bindings;
    ProgramType type;

  private:
    std::array<ProgramCode, 2> codes;
};
