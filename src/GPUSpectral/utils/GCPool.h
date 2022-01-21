#pragma once

#include <functional>
#include <map>
#include <optional>

template <typename K, typename V, typename Hasher>
class GCPool {
  public:
    const static constexpr uint32_t INITIAL_REF_COUNT = 8;
    template <typename G>
    struct Holder {
        uint32_t refCount;
        G value;
    };

    GCPool() = default;
    void add(K key, V obj) {
        objs.emplace(key, Holder<V>{ INITIAL_REF_COUNT, obj });
    }

    std::optional<V> get(K key) {
        auto it = objs.find(key);
        if (it != objs.end()) {
            it->second.refCount = INITIAL_REF_COUNT;
            return it->second.value;
        }
        return std::nullopt;
    }

    void setDestroyer(std::function<void(V)> destoryer) {
        this->destroyer = destoryer;
    }

    void tick() {
        for (auto it = objs.begin(); it != objs.end();) {
            --it->second.refCount;
            if (it->second.refCount == 0) {
                destroyer(it->second.value);
                objs.erase(it++);
            } else {
                ++it;
            }
        }
    }

  private:
    std::function<void(V)> destroyer;
    std::unordered_map<K, Holder<V>, Hasher> objs;
};
