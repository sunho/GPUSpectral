#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Transform {
    float x{};
    float y{};
    float z{};
    float sx{ 1.0f };
    float sy{ 1.0f };
    float sz{ 1.0f };
};
