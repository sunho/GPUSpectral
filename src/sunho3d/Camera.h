#pragma once

#include "Transform.h"

struct Camera {
    glm::mat4 view;
    glm::mat4 proj;

    void lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) {
        view = glm::lookAt(eye, center, up);
    }
    // view matrix is [u v w e]^-1
    //                [0 0 0 0]
    // (inverse of transforming world basis into camera basis)
    // obviously this contains the affine section -x_e, -y_e, -z_e
    glm::vec3 pos() const {
        return -glm::vec3(view[3]);
    }

    // tan theta / 2 = t / n
    // given n and theta we can solve for t
    // if we got t we can easily get t, l, b cause we know aspect ratio and
    // the center of the plane is paralle to the center of the camera (it "spreads" out)
    void setProjectionFov(float fovY, float aspect, float near, float far) {
        proj = glm::perspective(fovY, aspect, near, far);
        proj[1][1] *= -1;
    }
};
