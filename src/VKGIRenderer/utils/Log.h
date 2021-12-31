#pragma once
#include <iostream>

template<typename... Args>
inline static void Log(const char* format, Args... args)
{
    std::cout << std::format(format, args...) << std::endl;
}