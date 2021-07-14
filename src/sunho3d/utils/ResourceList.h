#pragma once

#include <map>
#include <memory>

#include "FixedVector.h"

namespace sunho3d {

class IdResource {
  public:
    explicit IdResource() = default;
    uint32_t id() const {
        return id_;
    }
    void setId(uint32_t id) {
        id_ = id;
    }

  private:
    uint32_t id_{};
};

template <typename T>
class ResourceList {
  public:
    using ResourceData = FixedVector<char>;

    ResourceList() = default;
    ~ResourceList() {
        for (auto [_, resource] : data) {
            T* addr = reinterpret_cast<T*>(resource.data());
            destruct(addr);
        }
    }

    template <typename... ARGS>
    T* construct(ARGS&&... args) noexcept {
        ResourceData resource(sizeof(T));
        T* addr = reinterpret_cast<T*>(resource.data());
        new (addr) T(std::forward<ARGS>(args)...);
        addr->setId(nextId);
        data.emplace(nextId++, std::move(resource));
        return addr;
    }

    void destruct(T* resource) noexcept {
        resource->~T();
        data.erase(resource->id());
    }

    T* get(uint32_t id) {
        ResourceData& resource = data.find(id)->second;
        T* addr = reinterpret_cast<T*>(resource.data());
        return addr;
    }

  private:
    std::map<uint32_t, ResourceData> data;
    uint32_t nextId{ 1 };
};

}  // namespace sunho3d
