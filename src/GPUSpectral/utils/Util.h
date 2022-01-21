#pragma once
#include <string>
#include <vector>

#define M_PI 3.14159265358979323846f

inline static std::vector<std::string> Split(const std::string& str, const char* delim, bool allow_empty) {
    if (str.empty())
        return {};
    std::vector<std::string> ret;

    size_t start_index = 0;
    size_t index = 0;
    while ((index = str.find_first_of(delim, start_index)) != std::string::npos) {
        if (allow_empty || index > start_index)
            ret.push_back(str.substr(start_index, index - start_index));
        start_index = index + 1;

        if (allow_empty && (index == str.size() - 1))
            ret.emplace_back();
    }

    if (start_index < str.size())
        ret.push_back(str.substr(start_index));
    return ret;
}

inline static std::vector<std::string> Split(const std::string& str, const char* delim) {
    return Split(str, delim, true);
}

inline static size_t align(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}
