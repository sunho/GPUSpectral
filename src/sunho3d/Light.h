#pragma once

#include "Transform.h"
#include "utils/ResourceList.h"

namespace sunho3d {

class Light : public IdResource {
  public:
    enum class Type {
        DIRECTIONAL,
        POINT,
    };

    explicit Light(Type type)
        : type(type) {
    }

    Type getType() const {
        return type;
    }

    const Transform& getTransform() const {
        return transform;
    }
    void setTransform(const Transform& transform) {
        this->transform = transform;
    }

    const glm::vec3& getDirection() const {
        return direction;
    }
    void setDirection(const glm::vec3& direction) {
        this->direction = direction;
    }

    const float getIntensity() const {
        return intensity;
    }
    void setIntensity(float intensity) {
        this->intensity = intensity;
    }

    const float getRadius() const {
        return radius;
    }
    void setRadius(float radius) {
        this->radius = radius;
    }

  private:
    Transform transform{};
    float intensity{};
    float radius{};
    glm::vec3 direction;
    Type type;
};

}  // namespace sunho3d
