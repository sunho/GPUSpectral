#pragma once

#include "Transform.h"

struct Camera {
    glm::mat4 toWorld;
    glm::mat4 view;
    glm::mat4 proj;
    float fov{};

    
    void lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) {
        view = glm::lookAt(eye, center, up);
    }
    
    void setView(glm::mat4 view) {
        this->view = view;
        toWorld = glm::inverse(view);
    }

    void setToWorld(glm::mat4 toWorld) {
        this->toWorld = toWorld;
        view = glm::inverse(toWorld);
    }

    glm::vec3 pos() const {
        return glm::vec3(toWorld[3]);
    }

    // tan theta / 2 = t / n
    // given n and theta we can solve for t
    // if we got t we can easily get t, l, b cause we know aspect ratio and
    // the center of the plane is paralle to the center of the camera (it "spreads" out)
    void setProjectionFov(float fovY, float aspect, float near, float far) {
        proj = glm::perspective(fovY, aspect, near, far);
        proj[1][1] *= -1;
        fov = fovY;
    }

    glm::vec3 rayDir(glm::vec2 size, glm::vec2 fragCoord) {
        glm::vec2 xy = fragCoord - size / 2.0f;
        float z = size.y / tan(fov);
        auto dir = glm::normalize(glm::vec3(xy.x, xy.y, -z));
        auto hdir = glm::vec4(dir, 0.0);
        auto out = glm::vec3(glm::transpose(view) * hdir);
        return out;
    }
};
