#pragma once

#include <unordered_map>
#include <set>

#include "../../utils/FixedVector.h"
#include "../../backend/vulkan/VulkanDriver.h"
#include "Resource.h"


class FrameGraph;
struct FrameGraphContext;

using FramePassFunc = std::function<void(FrameGraph& fg)>;

enum class ResourceAccessType {
    TransferRead,
    TransferWrite,
    ComputeWrite,
    DepthWrite,
    ColorWrite,
    ComputeRead,
    FragmentRead,
    RTWrite,
    RTRead
};

static inline bool isWriteAccessType(const ResourceAccessType& access) {
    switch (access) {
        case ResourceAccessType::ComputeWrite:
        case ResourceAccessType::ColorWrite:
        case ResourceAccessType::DepthWrite:
        case ResourceAccessType::TransferWrite:
        case ResourceAccessType::RTWrite:
            return true;
        default:
            return false;
    }
}


static inline bool isReadAccessType(const ResourceAccessType& access) {
    switch (access) {
        case ResourceAccessType::FragmentRead:
        case ResourceAccessType::ComputeRead:
        case ResourceAccessType::TransferRead:
        case ResourceAccessType::RTRead:
            return true;
        default:
            return false;
    }
}

enum class ResourceType {
    None,
    Image,
    Buffer
};

class ResourceHandle {
  public:
    using HandleId = uint32_t;
    static constexpr const HandleId nullId = std::numeric_limits<HandleId>::max();

    ResourceHandle() = default;
    explicit ResourceHandle(HandleId id, ResourceType type)
        : id(id), type(type) {
    }

    operator bool() const {
        return id != nullId;
    }

    bool operator==(const ResourceHandle& rhs) const {
        return id == rhs.id;
    }
    bool operator!=(const ResourceHandle& rhs) const {
        return id != rhs.id;
    }

    HandleId getId() const {
        return id;
    }

    ResourceType getType() const {
        return type;
    }

  private:
    HandleId id{ nullId };
    ResourceType type{ResourceType::None};
};

struct BakedPassResource {
    ResourceHandle resource;
    ResourceAccessType accessType;
};

struct FramePassTexture {
    std::vector<Handle<HwTexture>> resource;
    ResourceAccessType accessType;
};

struct FramePassBuffer {
    std::vector<Handle<HwBufferObject>> resource;
    ResourceAccessType accessType;
};

struct FramePass {
    std::string name;
    std::vector<FramePassTexture> textures;
    std::vector<FramePassBuffer> buffers;
    FramePassFunc func;
};

struct Resource {
    ResourceType type;
    FixedVector<char> data;
};

class FrameGraph {
  public:
    friend class FrameGraphContext;
    friend class BakedPass;
    FrameGraph(VKGIRenderer::VulkanDriver& driver);
    ~FrameGraph();

    void submit();

    void addFramePass(FramePass pass);
    Handle<HwBufferObject> createTempUniformBuffer(void* data, size_t size);
    Handle<HwBufferObject> createTempStorageBuffer(void* data, size_t size);
    Handle<HwBufferObject> createPermenantUniformBuffer(void* data, size_t size);

#define SCRATCH_IMPL(RESOURCENAME, METHODNAME)                        \
    template <typename... ARGS>                                       \
    Handle<Hw##RESOURCENAME> METHODNAME##SC(ARGS&&... args) {         \
        auto handle = driver.METHODNAME(std::forward<ARGS>(args)...); \
        destroyers.push_back([handle, this]() {                       \
            this->driver.destroy##RESOURCENAME(handle);               \
        });                                                           \
        return handle;                                                \
    }

    SCRATCH_IMPL(RenderTarget, createDefaultRenderTarget)
    SCRATCH_IMPL(RenderTarget, createRenderTarget)
    SCRATCH_IMPL(BufferObject, createBufferObject)
    SCRATCH_IMPL(Texture, createTexture)

#undef SCRATCH_IMPL

    void compile();
    void run();

  private:
    struct BakedPass {
        explicit BakedPass(FrameGraph& fg, FramePass pass);
        BakedPass() = default;
        std::string name;
        FramePassFunc func;
        std::vector<BakedPassResource> outputs;
        std::vector<BakedPassResource> inputs;
        std::unordered_map<uint32_t, BakedPassResource> resources;
        std::vector<Barrier> barriers;
        std::vector<Barrier> postBarriers;
    };

    struct BakedGraph {
        std::vector<BakedPass> passes;
        std::set<std::pair<ResourceHandle::HandleId, uint32_t>> useChains;
        std::set<ResourceHandle::HandleId> usedRes;
    };

    template <typename T>
    T* getResource(const ResourceHandle& resource) {
        auto& res = resources.find(resource.getId())->second;
        return reinterpret_cast<T*>(res.data.data());
    }

    template <typename T>
    ResourceHandle declareResource(ResourceType type) {
        Resource resource = {
            .type = type,
            .data = FixedVector<char>(sizeof(T)),
        };
        resources.emplace(nextId, resource);
        return ResourceHandle(nextId++, type);
    }

    ResourceHandle getOrRegisterTextureResource(Handle<HwTexture> tex) {
        if (registeredTextures.find(tex.getId()) != registeredTextures.end()) {
            return registeredTextures.at(tex.getId());
        }
        auto res = declareResource<Handle<HwTexture>>(ResourceType::Image);
        *getResource<Handle<HwTexture>>(res) = tex;
        registeredTextures.emplace(tex.getId(), res);
        return res;
    }

    ResourceHandle getOrRegisterBufferResource(Handle<HwBufferObject> buf) {
        if (registeredBuffers.find(buf.getId()) != registeredBuffers.end()) {
            return registeredBuffers.at(buf.getId());
        }
        auto res = declareResource<Handle<HwBufferObject>>(ResourceType::Buffer);
        *getResource<Handle<HwBufferObject>>(res) = buf;
        registeredBuffers.emplace(buf.getId(), res);
        return res;
    }

    std::unordered_map<uint32_t, Resource> resources;
    std::unordered_map<HandleBase::HandleId, ResourceHandle> registeredTextures;
    std::unordered_map<HandleBase::HandleId, ResourceHandle> registeredBuffers;
    std::vector<FramePass> passes;
    BakedGraph bakedGraph;
    uint32_t nextId{ 1 };

    template <typename T, typename... ARGS>
    T callDriverMethod(T (VKGIRenderer::VulkanDriver::*mf)(ARGS...), ARGS&&... args) noexcept {
        return (driver.*mf)(std::forward<ARGS>(args)...);
    }

    std::vector<std::function<void()>> destroyers;
    VKGIRenderer::VulkanDriver& driver;
};
