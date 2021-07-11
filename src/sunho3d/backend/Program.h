#pragma once

#include <string>
#include <vector>
#include <array>

enum class ProgramType : uint8_t {
    VERTEX = 0,
    FRAGMENT = 1
};

using ProgramCode = std::vector<char>;

struct Program {
    std::string name;
    std::array<ProgramCode, 2> codes;
};
