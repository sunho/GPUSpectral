#pragma once

#include <vector>

#include "utils/ResourceList.h"

namespace sunho3d {
class Entity;
class Scene : public IdResource {
  public:
    Scene();
    void addEntity(Entity *entity);
    // private:
    std::vector<Entity *> entities;
};

}  // namespace sunho3d
