#pragma oncee

#include <map>
#include <optional>

template <typename K, typename V>
class GCPool {
  public:
    GCPool() = default;
    void add(K key, V obj) {
        objs.emplace(key, Holder{ INITIAL_REF_COUNT, obj });
    }

    std::optional<V> get(K key) {
        auto it = objs.find(key);
        if (it != objs.end()) {
            ++it->second.refCount;
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

    const static constexpr uint32_t INITIAL_REF_COUNT = 2;
    struct Holder {
        uint32_t refCount{};
        V value;
        Holder(const Holder& other) = delete;
        Holder& operator=(const Holder& other) = delete;
        Holder(Holder&& other) = default;
        Holder& operator=(Holder&& other) = default;
    };
    std::map<K, Holder> objs;
};
