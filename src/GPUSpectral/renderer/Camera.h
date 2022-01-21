#pragma once

#include "Transform.h"

namespace GPUSpectral {

class Camera {
public:
    Camera();
    Camera(const glm::mat4& view, float fov);

    void lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up); 

    void setView(glm::mat4 view);

    void setToWorld(glm::mat4 toWorld); 

    void setFov(float fovY, float aspect, float near, float far);

    [[nodiscard]] glm::vec3 getPosition() const noexcept;

    [[nodiscard]] glm::mat4 getToWorld() const noexcept;

    [[nodiscard]] float getFov() const noexcept;

    [[nodiscard]] glm::vec3 rayDir(glm::vec2 size, glm::vec2 fragCoord) const noexcept;

private:
    glm::mat4 toWorld;
    glm::mat4 view;
    glm::mat4 proj;
    float fov{};
};
}
