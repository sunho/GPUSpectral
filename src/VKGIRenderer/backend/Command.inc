#define EXPAND(x) x

#define APPLY0(M,...)
#define APPLY1(M, A, ...) EXPAND(M(A))
#define APPLY2(M, A, ...) EXPAND(M(A)), EXPAND(APPLY1(M, __VA_ARGS__))
#define APPLY3(M, A, ...) EXPAND(M(A)), EXPAND(APPLY2(M, __VA_ARGS__))
#define APPLY4(M, A, ...) EXPAND(M(A)), EXPAND(APPLY3(M, __VA_ARGS__))
#define APPLY5(M, A, ...) EXPAND(M(A)), EXPAND(APPLY4(M, __VA_ARGS__))
#define APPLY6(M, A, ...) EXPAND(M(A)), EXPAND(APPLY5(M, __VA_ARGS__))
#define APPLY7(M, A, ...) EXPAND(M(A)), EXPAND(APPLY6(M, __VA_ARGS__))
#define APPLY8(M, A, ...) EXPAND(M(A)), EXPAND(APPLY7(M, __VA_ARGS__))
#define APPLY9(M, A, ...) EXPAND(M(A)), EXPAND(APPLY8(M, __VA_ARGS__))
#define APPLY10(M, A, ...) EXPAND(M(A)), EXPAND(APPLY9(M, __VA_ARGS__))
#define APPLY11(M, A, ...) EXPAND(M(A)), EXPAND(APPLY10(M, __VA_ARGS__))
#define APPLY12(M, A, ...) EXPAND(M(A)), EXPAND(APPLY11(M, __VA_ARGS__))
#define APPLY_N__(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, X, ...) APPLY##X
#define APPLY(M, ...) EXPAND(EXPAND(APPLY_N__(M, __VA_ARGS__, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))(M, __VA_ARGS__))

#define PAIR_ARGS_0(M, ...)
#define PAIR_ARGS_1(M, X, Y, ...) M(X, Y)
#define PAIR_ARGS_2(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_1(M, __VA_ARGS__))
#define PAIR_ARGS_3(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_2(M, __VA_ARGS__))
#define PAIR_ARGS_4(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_3(M, __VA_ARGS__))
#define PAIR_ARGS_5(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_4(M, __VA_ARGS__))
#define PAIR_ARGS_6(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_5(M, __VA_ARGS__))
#define PAIR_ARGS_7(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_6(M, __VA_ARGS__))
#define PAIR_ARGS_8(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_7(M, __VA_ARGS__))
#define PAIR_ARGS_9(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_8(M, __VA_ARGS__))
#define PAIR_ARGS_10(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_9(M, __VA_ARGS__))
#define PAIR_ARGS_11(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_10(M, __VA_ARGS__))
#define PAIR_ARGS_12(M, X, Y, ...) M(X, Y), EXPAND(PAIR_ARGS_11(M, __VA_ARGS__))

#define PAIR_ARGS_N__(_0, E1, _1, E2, _2, E3, _3, E4, _4, E5, _5, E6, _6, E7, _7, E8, _8, E9, _9, E10, _10, E11, _11, E12, _12, X, ...) PAIR_ARGS_##X

#define PAIR_ARGS_N(M, ...) \
    EXPAND(EXPAND(PAIR_ARGS_N__(0, ##__VA_ARGS__, 12, E, 11, E, 10, E, 9, E, 8, E, 7, E, 6, E, 5, E, 4, E, 3, E, 2, E, 1, E, 0))(M, __VA_ARGS__))

#define ARG(T, P) T P

#define PARAM(T, P) P

#define VOIDCOMMAND(N, ...) \
    DECL_VOIDCOMMAND(N, PAIR_ARGS_N(ARG, ##__VA_ARGS__), PAIR_ARGS_N(PARAM, ##__VA_ARGS__))

#define COMMAND(R, N, ...) \
    DECL_COMMAND(R, N, PAIR_ARGS_N(ARG, ##__VA_ARGS__), PAIR_ARGS_N(PARAM, ##__VA_ARGS__))

#define PARAM_1( P, ...) P
#define ARG_1( T, P, ...) T P
#define VOIDCOMMAND0(N) DECL_VOIDCOMMAND(N, ARG_1(int, dummy=0), PARAM_1(dummy))
#define COMMAND0(R, N) DECL_COMMAND(R, N, ARG_1(int, dummy=0), PARAM_1(dummy))

COMMAND(VertexBufferHandle, createVertexBuffer, uint32_t, bufferCount, uint32_t, vertexCount, uint8_t, attributeCount, AttributeArray, attributes)
COMMAND(IndexBufferHandle, createIndexBuffer, uint32_t, indexCount)
COMMAND(ProgramHandle, createProgram, Program, program)
COMMAND(CompiledCode, compileCode, const char *, path)
COMMAND0(FenceHandle, createFence)
COMMAND0(RenderTargetHandle, createDefaultRenderTarget)
COMMAND(RenderTargetHandle, createRenderTarget, uint32_t, width, uint32_t, height, RenderAttachments, attachments)
COMMAND(BufferObjectHandle, createBufferObject, uint32_t, size, BufferUsage, usage, BufferType, type)
COMMAND(PrimitiveHandle, createPrimitive, PrimitiveMode, mode)
COMMAND(TextureHandle, createTexture, SamplerType, type, TextureUsage, usage, TextureFormat, format, uint8_t, levels, uint32_t, width, uint32_t, height, uint32_t, layers)
COMMAND(InflightHandle, beginFrame, FenceHandle, fence)
VOIDCOMMAND(waitFence, FenceHandle, fence)
VOIDCOMMAND(setVertexBuffer, VertexBufferHandle, handle, uint32_t, index, BufferObjectHandle, bufferObject)
VOIDCOMMAND(updateBufferObjectSync, BufferObjectHandle, handle, BufferDescriptor, data, uint32_t, offset)
VOIDCOMMAND(updateCPUBufferObject, BufferObjectHandle, handle, BufferDescriptor, data, uint32_t, offset)
VOIDCOMMAND(copyBufferObject, BufferObjectHandle, dest, BufferObjectHandle, src)
VOIDCOMMAND(copyTextureInitialData, TextureHandle, handle, BufferDescriptor, data)
VOIDCOMMAND(copyBufferToTexture, TextureHandle, handle, ImageSubresource, subresource, BufferObjectHandle, src)
VOIDCOMMAND(blitTexture, TextureHandle, dest, ImageSubresource, destSubresource, TextureHandle, src, ImageSubresource, srcSubresource)
VOIDCOMMAND(updateIndexBuffer, IndexBufferHandle, handle, BufferDescriptor, data, uint32_t, offset)
VOIDCOMMAND(setPrimitiveBuffer, PrimitiveHandle, handle, VertexBufferHandle, vertexBuffer, IndexBufferHandle, indexBuffer)
VOIDCOMMAND(beginRenderPass, RenderTargetHandle, renderTarget, RenderPassParams, params)
VOIDCOMMAND(setBarrier, Barrier, barrier)
VOIDCOMMAND(setProfileSectionName, const char *, name)
COMMAND(ImageLayout, getTextureImageLayout, TextureHandle, handle)
VOIDCOMMAND0(endRenderPass)
VOIDCOMMAND(draw, GraphicsPipeline, pipeline, PrimitiveHandle, primitive)
VOIDCOMMAND(dispatch, ComputePipeline, pipeline, size_t, groupCountX, size_t, groupCountY, size_t, groupCountZ)
COMMAND0(Extent2D, getFrameSize)
COMMAND(BLASHandle, createBLAS, PrimitiveHandle, primitive)
COMMAND(TLASHandle, createTLAS, RTSceneDescriptor, descriptor)
VOIDCOMMAND(destroyBLAS, BLASHandle, handle)
VOIDCOMMAND(destroyTLAS, TLASHandle, handle)
VOIDCOMMAND(traceRays, RTPipeline, pipeline, size_t, width, size_t, height)
VOIDCOMMAND0(endFrame)
VOIDCOMMAND(releaseInflight, InflightHandle, handle)
VOIDCOMMAND(destroyVertexBuffer, VertexBufferHandle, handle)
VOIDCOMMAND(destroyIndexBuffer, IndexBufferHandle, handle)
VOIDCOMMAND(destroyBufferObject, BufferObjectHandle, handle)
VOIDCOMMAND(destroyPrimitive, PrimitiveHandle, handle)
VOIDCOMMAND(destroyTexture, TextureHandle, handle)
VOIDCOMMAND(destroyRenderTarget, RenderTargetHandle, handle)
VOIDCOMMAND(destroyFence, FenceHandle, handle)

#undef PAIR_ARGS_1
#undef PAIR_ARGS_2
#undef PAIR_ARGS_3
#undef PAIR_ARGS_4
#undef PAIR_ARGS_5
#undef PAIR_ARGS_6
#undef PAIR_ARGS_7
#undef PAIR_ARGS_8
#undef PAIR_ARGS_N__
#undef PAIR_ARGS_N

#undef PARAM
#undef ARG
#undef EXPAND

#undef PARAM_1
#undef ARG_1

#undef COMMAND
#undef VOIDCOMMAND
#undef COMMAND0
#undef VOIDCOMMAND0
