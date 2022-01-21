#pragma once
#include <fmt/format.h>
#include <iostream>

template <typename... Args>
inline static void Log(const char* format, Args... args) {
    std::cout << fmt::format(format, args...) << std::endl;
}