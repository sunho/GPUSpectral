#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_GTC_type_precision
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_precision.hpp>

struct Transform {
    float x{};
    float y{};
    float z{};
    float sx{ 1.0f };
    float sy{ 1.0f };
    float sz{ 1.0f };
    float rx{ 0.0f };
    float ry{ 0.0f };
    float rz{ 0.0f };

    glm::mat4 toMatrix() const {
        // [I,e]
        glm::mat4 T = glm::translate(glm::identity<glm::mat4>(), glm::vec3(x, y, z));
        // I * [sx,sy,sz,1]
        glm::mat4 S = glm::scale(T, glm::vec3(sx, sy, sz));
        glm::mat4 Rx = glm::rotate(S, glm::radians(rx), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 Ry = glm::rotate(Rx, glm::radians(ry), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 Rz = glm::rotate(Ry, glm::radians(rz), glm::vec3(0.0f, 0.0f, 1.0f));
        return Rz;
    }
};
