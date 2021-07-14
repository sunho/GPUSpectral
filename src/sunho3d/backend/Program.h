#pragma once

#include <array>
#include <string>
#include <vector>

enum class ProgramType : uint8_t { VERTEX = 0, FRAGMENT = 1 };

using ProgramCode = std::vector<char>;

struct Program {
    std::string name;
    std::array<ProgramCode, 2> codes;
};
