#pragma once

#include <map>
#include <memory>

namespace sunho3d {

class IdResource {
public:
    explicit IdResource(uint32_t id) : id_(id) { }
    uint32_t id() const { return id_; }
    
private:
    uint32_t id_;
};

template <typename T>
class ResourceList {
public:
    ResourceList() = default;
    void add(std::unique_ptr<T> obj) {
        data.emplace(obj->id(), std::move(obj));
    }

    uint32_t getNextId() {
        return nextId++;
    }
    
    void get(uint32_t id) {
        return data.find(id)->second;
    }
private:
    std::map<uint32_t, std::unique_ptr<T>> data;
    uint32_t nextId{1};
};

}
