#pragma once

#include <numeric>

struct HwVertexBuffer;
struct HwIndexBuffer;
struct HwProgram;
struct HwRenderTarget;
struct HwBufferObject;
struct HwPrimitive;
struct HwTexture;
struct HwFence;
struct HwInflight;
struct HwTLAS;
struct HwBLAS;

class HandleBase {
  public:
    using HandleId = uint32_t;
    static constexpr const HandleId nullId = std::numeric_limits<HandleId>::max();

    HandleBase() = default;
    explicit HandleBase(HandleId id)
        : id(id) {
    }

    operator bool() const {
        return id != nullId;
    }

    bool operator==(const HandleBase &rhs) const {
        return id == rhs.id;
    }
    bool operator!=(const HandleBase &rhs) const {
        return id != rhs.id;
    }

    HandleId getId() const {
        return id;
    }

    void reset() {
        id = nullId;
    }

  protected:
    HandleId id{ nullId };
};

template <typename T>
struct Handle : public HandleBase {
    Handle()
        : HandleBase() {
    }
    template <typename B, typename = std::enable_if_t<std::is_base_of<T, B>::value>>
    Handle(Handle<B> const &base) noexcept
        : HandleBase(base) {
    }
    explicit Handle(HandleId id)
        : HandleBase(id) {
    }
    operator bool() const {
        return id != nullId;
    }
};

using VertexBufferHandle = Handle<HwVertexBuffer>;
using IndexBufferHandle = Handle<HwIndexBuffer>;
using ProgramHandle = Handle<HwProgram>;
using RenderTargetHandle = Handle<HwRenderTarget>;
using BufferObjectHandle = Handle<HwBufferObject>;
using PrimitiveHandle = Handle<HwPrimitive>;
using TextureHandle = Handle<HwTexture>;
using FenceHandle = Handle<HwFence>;
using InflightHandle = Handle<HwInflight>;
using TLASHandle = Handle<HwTLAS>;
using BLASHandle = Handle<HwBLAS>;