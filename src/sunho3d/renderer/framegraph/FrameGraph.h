#pragma once

#include <unordered_map>
#include <set>

#include "../../utils/FixedVector.h"
#include "../../backend/vulkan/VulkanDriver.h"
#include "Resource.h"


class FrameGraph;
struct FrameGraphContext;

using FramePassFunc = std::function<void(FrameGraph& fg, FrameGraphContext& ctx)>;

enum class ResourceAccessType {
    ComputeWrite,
    DepthWrite,
    ColorWrite,
    ComputeRead,
    FragmentRead
};

static inline bool isWriteAccessType(const ResourceAccessType& access) {
    switch (access) {
        case ResourceAccessType::ComputeWrite:
        case ResourceAccessType::ColorWrite:
        case ResourceAccessType::DepthWrite:
            return true;
        default:
            return false;
    }
}


static inline bool isReadAccessType(const ResourceAccessType& access) {
    switch (access) {
        case ResourceAccessType::FragmentRead:
        case ResourceAccessType::ComputeRead:
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

struct FramePassResource {
    ResourceHandle resource;
    ResourceAccessType accessType;
};

struct FramePass {
    std::string name;
    std::vector<FramePassResource> resources;
    FramePassFunc func;
};

struct Resource {
    std::string name;
    ResourceType type;
    FixedVector<char> data;
};

class FrameGraph {
  public:
    friend class FrameGraphContext;
    FrameGraph(sunho3d::VulkanDriver& driver);
    ~FrameGraph();

    void submit();

    void addFramePass(FramePass pass);
    ResourceHandle importImage(std::string name, Handle<HwTexture> image);
    ResourceHandle importBuffer(std::string name, Handle<HwBufferObject> buffer);

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
    SCRATCH_IMPL(UniformBuffer, createUniformBuffer)

#undef SCRATCH_IMPL

    void compile();
    void run();

  private:
    struct BakedPass {
        explicit BakedPass(FramePass pass);
        BakedPass() = default;
        std::string name;
        FramePassFunc func;
        std::vector<FramePassResource> outputs;
        std::vector<FramePassResource> inputs;
        std::unordered_map<uint32_t, FramePassResource> resources;
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
    ResourceHandle declareResource(const std::string& name, ResourceType type) {
        Resource resource = {
            .name = name,
            .type = type,
            .data = FixedVector<char>(sizeof(T)),
        };
        resources.emplace(nextId, resource);
        return ResourceHandle(nextId++, type);
    }

    std::unordered_map<uint32_t, Resource> resources;
    std::vector<FramePass> passes;
    BakedGraph bakedGraph;
    uint32_t nextId{ 1 };

    template <typename T, typename... ARGS>
    T callDriverMethod(T (sunho3d::VulkanDriver::*mf)(ARGS...), ARGS&&... args) noexcept {
        return (driver.*mf)(std::forward<ARGS>(args)...);
    }

    std::vector<std::function<void()>> destroyers;
    sunho3d::VulkanDriver& driver;
};

struct FrameGraphContext {
    FrameGraphContext(FrameGraph& parent)
        : parent(parent) {
    }

    void bindTextureResource(PipelineState& pipe, uint32_t set, uint32_t binding, ResourceHandle handle);
    void bindStorageImageResource(PipelineState& pipe, uint32_t set, uint32_t binding, ResourceHandle handle);
    void bindStorageBufferResource(PipelineState& pipe, uint32_t set, uint32_t binding, ResourceHandle handle);

    Handle<HwTexture> unwrapTextureHandle(ResourceHandle handle);
    Handle<HwBufferObject> unwrapBufferHandle(ResourceHandle handle);


  private:
    FrameGraph& parent;
};