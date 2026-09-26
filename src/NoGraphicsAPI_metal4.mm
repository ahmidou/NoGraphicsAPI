#include <NoGraphicsAPI/NoGraphicsAPI.hpp>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Keep assert expressions type-checked in release without evaluating them.
#if defined(NDEBUG)
#undef assert
#define assert(expression) ((void)sizeof(static_cast<bool>(expression)))
#endif

// Metal 4 backend. Every allocation lives in one device-wide residency set attached to every queue. Metal 4 orders nothing
// between encoders, so barrier() is the only ordering, as on Vulkan: it applies to the open encoder at once and to every encoder
// opened later in the same command buffer (Metal barriers attach to encoders, so one cannot reach another command buffer).
// Descriptor heaps hold 8-byte Metal resource IDs: texture slots are views in an MTLTextureViewPool, sampler slots are
// sampler gpuResourceIDs. Shaders read those IDs through Slang handles in the root; heaps are never bound to the GPU.
// Buffer copies need MTLBuffer objects, so heaps are found by address in a device registry. Unlike Vulkan, GPU heap
// creation/destruction must therefore not run concurrently with other heap creation/destruction or with recording copies.

namespace gpu {

    namespace {

        constexpr uint32 max_queues = 16;
        constexpr uint32 max_color_attachments = 8;
        constexpr uint64 descriptor_size = sizeof(MTLResourceID);
        constexpr uint64 root_chunk_size = 64 * 1024;
        constexpr uint64 root_alignment = 16;
        constexpr uint64 texture_heap_alignment = 64 * 1024;
        constexpr MTLStages compute_encoder_stages = MTLStageDispatch | MTLStageBlit;
        constexpr MTLStages render_encoder_stages = MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh;
        constexpr MTLRenderStages render_argument_stages = MTLRenderStageVertex | MTLRenderStageFragment | MTLRenderStageObject | MTLRenderStageMesh;

        [[noreturn]] void fail(const char* operation, NSError* error) noexcept
        {
            fprintf(stderr, "NoGraphicsAPI Metal: %s failed%s%s\n", operation, error ? ": " : "", error ? error.localizedDescription.UTF8String : "");
            abort();
        }

        template<typename T> struct Array
        {
            T* data = nullptr;
            uint32 size = 0;
            uint32 capacity = 0;

            Array() = default;
            Array(const Array&) = delete;
            Array& operator=(const Array&) = delete;
            ~Array()
            {
                delete[] data;
            }

            T& push() noexcept
            {
                if (size == capacity)
                {
                    capacity = capacity ? capacity * 2 : 16;
                    T* grown = new T[capacity];
                    for (uint32 index = 0; index < size; ++index)
                        grown[index] = static_cast<T&&>(data[index]);
                    delete[] data;
                    data = grown;
                }
                return data[size++];
            }

            void remove(uint32 index) noexcept
            {
                data[index] = static_cast<T&&>(data[--size]);
                data[size] = T{};
            }
        };

    } // namespace

    struct GpuHeapOwner
    {
        Device* device = nullptr;
        id<MTLBuffer> buffer = nil;
        id<MTLTextureViewPool> view_pool = nil; // Texture descriptor heaps only.
    };

    struct TextureHeapOwner
    {
        Device* device = nullptr;
        id<MTLHeap> heap = nil;
    };

    struct HeapRecord
    {
        byte* gpu = nullptr;
        byte* cpu = nullptr;
        uint64 size = 0;
        GpuHeapOwner* owner = nullptr;
    };

    struct SamplerRecord
    {
        SamplerDesc desc = {};
        id<MTLSamplerState> state = nil;
    };

    struct DepthStencilRecord
    {
        DepthStencilState desc = {};
        id<MTLDepthStencilState> state = nil;
    };

    struct RenderView
    {
        id<MTLTexture> texture = nil;
        uint32 width = 0;
        uint32 height = 0;
    };

    struct Device
    {
        id<MTLDevice> device = nil;
        id<MTL4Compiler> compiler = nil;
        id<MTL4CommandQueue> queues[max_queues]{};
        uint32 queue_count = 0;
        id<MTLResidencySet> residency = nil;
        id<MTLSharedEvent> idle_event = nil;
        uint64 idle_value = 0;
        CAMetalLayer* layer = nil;
        id<CAMetalDrawable> drawable = nil;
        RenderView swapchain_view{};
        Array<HeapRecord> heaps;
        Array<SamplerRecord> samplers;
        Array<DepthStencilRecord> depth_stencil_states;
        DeviceCaps caps{};
        char device_name[256]{};
    };

    struct TimelineSemaphore
    {
        id<MTLSharedEvent> event = nil;
    };

    struct Texture
    {
        Device* device = nullptr;
        id<MTLTexture> texture = nil;
        uint32 width = 0;
        uint32 height = 0;
        uint32 depth = 0;
        uint32 layer_count = 0;
        Format format = Format::rgba8_unorm;
    };

    struct PSO
    {
        id<MTLComputePipelineState> compute = nil;
        id<MTLRenderPipelineState> render = nil;
        MTLSize threads_per_threadgroup{};
        MTLCullMode cull = MTLCullModeNone;
        MTLWinding winding = MTLWindingClockwise;
        float depth_bias = 0.0f;
        float depth_bias_slope = 0.0f;
        float depth_bias_clamp = 0.0f;
    };

    struct CommandBuffer
    {
        CommandPool* pool = nullptr;
        Device* device = nullptr;
        id<MTL4CommandBuffer> command_buffer = nil;
        id<MTL4ArgumentTable> argument_table = nil;
        id<MTL4ComputeCommandEncoder> compute = nil;
        id<MTL4RenderCommandEncoder> render = nil;
        const PSO* pso = nullptr;
        // Stages of barrier() calls recorded so far in this command buffer; every encoder opened later waits on them.
        MTLStages barrier_source = 0;
        MTLStages barrier_destination = 0;
        bool swapchain = false;
        CommandBuffer* next = nullptr;
    };

    struct CommandPool
    {
        Device* device = nullptr;
        uint32 queue_index = 0;
        id<MTL4CommandAllocator> allocator = nil;
        CommandBuffer* first = nullptr;
        CommandBuffer* last = nullptr;
        CommandBuffer* next_buffer = nullptr;
        Array<id<MTLBuffer>> root_chunks;
        uint32 root_chunk = 0;
        uint64 root_offset = 0;
    };

    namespace {

        MTLPixelFormat to_mtl(Format format) noexcept
        {
            switch (format)
            {
            case Format::r8_srgb: return MTLPixelFormatR8Unorm_sRGB;
            case Format::rg8_srgb: return MTLPixelFormatRG8Unorm_sRGB;
            case Format::rgba8_srgb: return MTLPixelFormatRGBA8Unorm_sRGB;
            case Format::bgra8_srgb: return MTLPixelFormatBGRA8Unorm_sRGB;
            case Format::r8_unorm: return MTLPixelFormatR8Unorm;
            case Format::rg8_unorm: return MTLPixelFormatRG8Unorm;
            case Format::rgba8_unorm: return MTLPixelFormatRGBA8Unorm;
            case Format::bgra8_unorm: return MTLPixelFormatBGRA8Unorm;
            case Format::r16_unorm: return MTLPixelFormatR16Unorm;
            case Format::rg16_unorm: return MTLPixelFormatRG16Unorm;
            case Format::rgba16_unorm: return MTLPixelFormatRGBA16Unorm;
            case Format::r8_uint: return MTLPixelFormatR8Uint;
            case Format::rg8_uint: return MTLPixelFormatRG8Uint;
            case Format::rgba8_uint: return MTLPixelFormatRGBA8Uint;
            case Format::r16_uint: return MTLPixelFormatR16Uint;
            case Format::rg16_uint: return MTLPixelFormatRG16Uint;
            case Format::rgba16_uint: return MTLPixelFormatRGBA16Uint;
            case Format::r32_uint: return MTLPixelFormatR32Uint;
            case Format::rg32_uint: return MTLPixelFormatRG32Uint;
            case Format::rgba32_uint: return MTLPixelFormatRGBA32Uint;
            case Format::r16_float: return MTLPixelFormatR16Float;
            case Format::rg16_float: return MTLPixelFormatRG16Float;
            case Format::rgba16_float: return MTLPixelFormatRGBA16Float;
            case Format::r32_float: return MTLPixelFormatR32Float;
            case Format::rg32_float: return MTLPixelFormatRG32Float;
            case Format::rgba32_float: return MTLPixelFormatRGBA32Float;
            case Format::rgb10a2_unorm: return MTLPixelFormatRGB10A2Unorm;
            case Format::rg11b10_float: return MTLPixelFormatRG11B10Float;
            case Format::d16_unorm: return MTLPixelFormatDepth16Unorm;
            case Format::d32_float: return MTLPixelFormatDepth32Float;
            case Format::s8_uint: return MTLPixelFormatStencil8;
            case Format::d32_float_s8_uint: return MTLPixelFormatDepth32Float_Stencil8;
            case Format::eac_rg: return MTLPixelFormatEAC_RG11Unorm;
            case Format::astc_4x4_srgb: return MTLPixelFormatASTC_4x4_sRGB;
            case Format::astc_4x4_unorm: return MTLPixelFormatASTC_4x4_LDR;
            case Format::bc3_srgb: return MTLPixelFormatBC3_RGBA_sRGB;
            case Format::bc3_unorm: return MTLPixelFormatBC3_RGBA;
            case Format::bc5_rg: return MTLPixelFormatBC5_RGUnorm;
            case Format::bc6h_ufloat: return MTLPixelFormatBC6H_RGBUfloat;
            case Format::bc6h_sfloat: return MTLPixelFormatBC6H_RGBFloat;
            case Format::bc7_srgb: return MTLPixelFormatBC7_RGBAUnorm_sRGB;
            case Format::bc7_unorm: return MTLPixelFormatBC7_RGBAUnorm;
            // Packed 16-bit formats differ in channel order from Vulkan's; BGRA uint and D24S8 have no Apple silicon equivalent.
            case Format::rgba4_unorm:
            case Format::r5g5b5a1_unorm:
            case Format::r5g6b5_unorm:
            case Format::bgra8_uint:
            case Format::d24_unorm_s8_uint:
            case Format::undefined: return MTLPixelFormatInvalid;
            }
            return MTLPixelFormatInvalid;
        }

        bool is_compressed(Format format) noexcept
        {
            return get_texture_format_info(format).block_extent.x > 1;
        }

        bool is_bc(Format format) noexcept
        {
            return format == Format::bc3_srgb || format == Format::bc3_unorm || format == Format::bc5_rg || format == Format::bc6h_ufloat ||
                   format == Format::bc6h_sfloat || format == Format::bc7_srgb || format == Format::bc7_unorm;
        }

        MTLCompareFunction to_mtl(CompareOp op) noexcept
        {
            switch (op)
            {
            case CompareOp::never: return MTLCompareFunctionNever;
            case CompareOp::less: return MTLCompareFunctionLess;
            case CompareOp::equal: return MTLCompareFunctionEqual;
            case CompareOp::less_equal: return MTLCompareFunctionLessEqual;
            case CompareOp::greater: return MTLCompareFunctionGreater;
            case CompareOp::not_equal: return MTLCompareFunctionNotEqual;
            case CompareOp::greater_equal: return MTLCompareFunctionGreaterEqual;
            case CompareOp::always: return MTLCompareFunctionAlways;
            }
            return MTLCompareFunctionAlways;
        }

        MTLStencilOperation to_mtl(StencilOp op) noexcept
        {
            switch (op)
            {
            case StencilOp::keep: return MTLStencilOperationKeep;
            case StencilOp::zero: return MTLStencilOperationZero;
            case StencilOp::replace: return MTLStencilOperationReplace;
            case StencilOp::increment_clamp: return MTLStencilOperationIncrementClamp;
            case StencilOp::decrement_clamp: return MTLStencilOperationDecrementClamp;
            case StencilOp::invert: return MTLStencilOperationInvert;
            case StencilOp::increment_wrap: return MTLStencilOperationIncrementWrap;
            case StencilOp::decrement_wrap: return MTLStencilOperationDecrementWrap;
            }
            return MTLStencilOperationKeep;
        }

        MTLBlendFactor to_mtl(BlendFactor factor) noexcept
        {
            switch (factor)
            {
            case BlendFactor::zero: return MTLBlendFactorZero;
            case BlendFactor::one: return MTLBlendFactorOne;
            case BlendFactor::source_color: return MTLBlendFactorSourceColor;
            case BlendFactor::one_minus_source_color: return MTLBlendFactorOneMinusSourceColor;
            case BlendFactor::destination_color: return MTLBlendFactorDestinationColor;
            case BlendFactor::one_minus_destination_color: return MTLBlendFactorOneMinusDestinationColor;
            case BlendFactor::source_alpha: return MTLBlendFactorSourceAlpha;
            case BlendFactor::one_minus_source_alpha: return MTLBlendFactorOneMinusSourceAlpha;
            case BlendFactor::destination_alpha: return MTLBlendFactorDestinationAlpha;
            case BlendFactor::one_minus_destination_alpha: return MTLBlendFactorOneMinusDestinationAlpha;
            case BlendFactor::source_alpha_saturate: return MTLBlendFactorSourceAlphaSaturated;
            }
            return MTLBlendFactorOne;
        }

        MTLBlendOperation to_mtl(BlendOp op) noexcept
        {
            switch (op)
            {
            case BlendOp::add: return MTLBlendOperationAdd;
            case BlendOp::subtract: return MTLBlendOperationSubtract;
            case BlendOp::reverse_subtract: return MTLBlendOperationReverseSubtract;
            case BlendOp::minimum: return MTLBlendOperationMin;
            case BlendOp::maximum: return MTLBlendOperationMax;
            }
            return MTLBlendOperationAdd;
        }

        MTLSamplerAddressMode to_mtl(AddressMode mode) noexcept
        {
            switch (mode)
            {
            case AddressMode::repeat: return MTLSamplerAddressModeRepeat;
            case AddressMode::mirrored_repeat: return MTLSamplerAddressModeMirrorRepeat;
            case AddressMode::clamp_to_edge: return MTLSamplerAddressModeClampToEdge;
            }
            return MTLSamplerAddressModeRepeat;
        }

        MTLLoadAction to_mtl(LoadOp op) noexcept
        {
            switch (op)
            {
            case LoadOp::load: return MTLLoadActionLoad;
            case LoadOp::clear: return MTLLoadActionClear;
            case LoadOp::discard: return MTLLoadActionDontCare;
            }
            return MTLLoadActionDontCare;
        }

        MTLStoreAction to_mtl(StoreOp op) noexcept
        {
            return op == StoreOp::store ? MTLStoreActionStore : MTLStoreActionDontCare;
        }

        bool has_stage(Stage stages, Stage stage) noexcept
        {
            return (static_cast<uint64>(stages) & static_cast<uint64>(stage)) != 0;
        }

        MTLStages to_mtl(Stage stage) noexcept
        {
            if (has_stage(stage, Stage::all_commands))
                return compute_encoder_stages | render_encoder_stages;
            MTLStages result = 0;
            if (has_stage(stage, Stage::indirect))
                result |= MTLStageVertex | MTLStageObject | MTLStageMesh | MTLStageDispatch;
            if (has_stage(stage, Stage::index_input) || has_stage(stage, Stage::vertex))
                result |= MTLStageVertex;
            if (has_stage(stage, Stage::task))
                result |= MTLStageObject;
            if (has_stage(stage, Stage::mesh))
                result |= MTLStageMesh;
            if (has_stage(stage, Stage::depth_stencil_tests) || has_stage(stage, Stage::fragment) || has_stage(stage, Stage::color_output))
                result |= MTLStageFragment;
            if (has_stage(stage, Stage::compute))
                result |= MTLStageDispatch;
            if (has_stage(stage, Stage::transfer))
                result |= MTLStageBlit;
            return result;
        }

        MTLTextureType to_mtl(TextureType type, uint32 layer_count) noexcept
        {
            switch (type)
            {
            case TextureType::one_d: return layer_count > 1 ? MTLTextureType1DArray : MTLTextureType1D;
            case TextureType::two_d: return layer_count > 1 ? MTLTextureType2DArray : MTLTextureType2D;
            case TextureType::three_d: return MTLTextureType3D;
            case TextureType::cube: return MTLTextureTypeCube;
            case TextureType::two_d_array: return MTLTextureType2DArray;
            case TextureType::cube_array: return MTLTextureTypeCubeArray;
            }
            return MTLTextureType2D;
        }

        bool has_usage(TextureUsage usage, TextureUsage flag) noexcept
        {
            return (static_cast<uint32>(usage) & static_cast<uint32>(flag)) != 0;
        }

        MTLTextureDescriptor* make_texture_descriptor(const TextureDesc& desc) noexcept
        {
            MTLTextureDescriptor* result = [MTLTextureDescriptor new];
            result.textureType = to_mtl(desc.type, desc.layer_count);
            result.pixelFormat = to_mtl(desc.format);
            result.width = desc.extent.x;
            result.height = desc.extent.y;
            result.depth = desc.extent.z;
            result.mipmapLevelCount = desc.mip_levels;
            result.arrayLength = desc.type == TextureType::cube || desc.type == TextureType::cube_array ? desc.layer_count / 6 : desc.layer_count;
            result.storageMode = MTLStorageModePrivate;
            result.hazardTrackingMode = MTLHazardTrackingModeUntracked;
            MTLTextureUsage usage = MTLTextureUsageUnknown;
            if (has_usage(desc.usage, TextureUsage::sampled))
                usage |= MTLTextureUsageShaderRead;
            if (has_usage(desc.usage, TextureUsage::storage))
                usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            if (has_usage(desc.usage, TextureUsage::color_attachment) || has_usage(desc.usage, TextureUsage::depth_stencil_attachment))
                usage |= MTLTextureUsageRenderTarget;
            if (desc.mutable_format)
                usage |= MTLTextureUsagePixelFormatView;
            result.usage = usage;
            return result;
        }

        void track_allocation(Device* device, id<MTLAllocation> allocation) noexcept
        {
            [device->residency addAllocation:allocation];
            [device->residency commit];
        }

        void untrack_allocation(Device* device, id<MTLAllocation> allocation) noexcept
        {
            [device->residency removeAllocation:allocation];
            [device->residency commit];
        }

        // Finds the heap containing [address, address + size) in the registry, by GPU or CPU address.
        const HeapRecord& find_heap(Device* device, const void* address, bool cpu) noexcept
        {
            const byte* pointer = static_cast<const byte*>(address);
            for (uint32 index = 0; index < device->heaps.size; ++index)
            {
                const HeapRecord& record = device->heaps.data[index];
                const byte* base = cpu ? record.cpu : record.gpu;
                if (base && pointer >= base && pointer < base + record.size)
                    return record;
            }
            assert(false && "address is not inside a live GpuHeap");
            abort();
        }

        // Opens a compute encoder if none is open. Compute work may not be recorded inside a render pass.
        id<MTL4ComputeCommandEncoder> compute_encoder(CommandBuffer* commands) noexcept
        {
            assert(!commands->render && "compute, copy, and dispatch commands must be recorded outside a render pass");
            if (!commands->compute)
            {
                commands->compute = [commands->command_buffer computeCommandEncoder];
                [commands->compute setArgumentTable:commands->argument_table];
                if (commands->barrier_destination & compute_encoder_stages)
                    [commands->compute barrierAfterQueueStages:commands->barrier_source
                                                  beforeStages:commands->barrier_destination & compute_encoder_stages
                                             visibilityOptions:MTL4VisibilityOptionDevice];
            }
            return commands->compute;
        }

        void end_compute_encoder(CommandBuffer* commands) noexcept
        {
            if (!commands->compute)
                return;
            [commands->compute endEncoding];
            commands->compute = nil;
        }

        // Copies the root into the pool's root ring and points argument table slot 0 at it (the Slang root at [[buffer(0)]]).
        void emit_root_data(CommandBuffer* commands, ByteSpan root) noexcept
        {
            assert(root.size <= 256);
            if (root.size == 0)
                return;
            CommandPool* pool = commands->pool;
            pool->root_offset = (pool->root_offset + root_alignment - 1) & ~(root_alignment - 1);
            if (pool->root_chunk == pool->root_chunks.size || pool->root_offset + root.size > root_chunk_size)
            {
                if (pool->root_chunk < pool->root_chunks.size && pool->root_offset != 0)
                    ++pool->root_chunk;
                if (pool->root_chunk == pool->root_chunks.size)
                {
                    id<MTLBuffer> chunk = [pool->device->device
                        newBufferWithLength:root_chunk_size
                                    options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined | MTLResourceHazardTrackingModeUntracked];
                    if (!chunk)
                        fail("root buffer allocation", nil);
                    pool->root_chunks.push() = chunk;
                    track_allocation(pool->device, chunk);
                }
                pool->root_offset = 0;
            }
            id<MTLBuffer> chunk = pool->root_chunks.data[pool->root_chunk];
            memcpy(static_cast<byte*>(chunk.contents) + pool->root_offset, root.data, root.size);
            [commands->argument_table setAddress:chunk.gpuAddress + pool->root_offset atIndex:0];
            pool->root_offset += root.size;
        }

        id<MTLLibrary> load_library(Device* device, ByteSpan metallib) noexcept
        {
            assert(metallib.data && metallib.size && "Metal PSOs need ShaderCode::metallib");
            dispatch_data_t data = dispatch_data_create(metallib.data, metallib.size, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError* error = nil;
            id<MTLLibrary> library = [device->device newLibraryWithData:data error:&error];
            if (!library)
                fail("metallib loading", error);
            return library;
        }

        MTL4LibraryFunctionDescriptor* function_descriptor(id<MTLLibrary> library, NSString* name) noexcept
        {
            MTL4LibraryFunctionDescriptor* result = [MTL4LibraryFunctionDescriptor new];
            result.library = library;
            result.name = name;
            return result;
        }

        void submit_commands(Device* device, const SubmitDesc& desc, uint32 queue_index, id<CAMetalDrawable> drawable) noexcept
        {
            assert(queue_index < device->queue_count && desc.completion.semaphore);
            assert(desc.commands.data || desc.commands.size == 0);
            id<MTL4CommandQueue> queue = device->queues[queue_index];
            for (size_t index = 0; index < desc.waits.size; ++index)
            {
                assert(desc.waits.data[index].semaphore);
                [queue waitForEvent:desc.waits.data[index].semaphore->event value:desc.waits.data[index].value];
            }
            if (drawable)
                [queue waitForDrawable:drawable];
            constexpr size_t batch_size = 16;
            for (size_t first = 0; first < desc.commands.size; first += batch_size)
            {
                __unsafe_unretained id<MTL4CommandBuffer> batch[batch_size];
                const size_t count = desc.commands.size - first < batch_size ? desc.commands.size - first : batch_size;
                for (size_t index = 0; index < count; ++index)
                {
                    assert(
                        desc.commands.data[first + index] && !desc.commands.data[first + index]->compute && !desc.commands.data[first + index]->render &&
                        "submitted command buffers must be ended"
                    );
                    batch[index] = desc.commands.data[first + index]->command_buffer;
                }
                [queue commit:batch count:count];
            }
            if (drawable)
            {
                [queue signalDrawable:drawable];
                [drawable present];
            }
            [queue signalEvent:desc.completion.semaphore->event value:desc.completion.value];
        }

        id<MTLDepthStencilState> depth_stencil_state(Device* device, const DepthStencilState& state) noexcept
        {
            for (uint32 index = 0; index < device->depth_stencil_states.size; ++index)
            {
                if (memcmp(&device->depth_stencil_states.data[index].desc, &state, sizeof(state)) == 0)
                    return device->depth_stencil_states.data[index].state;
            }
            MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
            desc.depthCompareFunction = state.depth_test ? to_mtl(state.depth_compare) : MTLCompareFunctionAlways;
            desc.depthWriteEnabled = state.depth_test && state.depth_write;
            if (state.stencil_test)
            {
                const StencilFaceState* faces[2]{ &state.front, &state.back };
                MTLStencilDescriptor* stencils[2]{ [MTLStencilDescriptor new], [MTLStencilDescriptor new] };
                for (uint32 face = 0; face < 2; ++face)
                {
                    stencils[face].stencilCompareFunction = to_mtl(faces[face]->compare);
                    stencils[face].stencilFailureOperation = to_mtl(faces[face]->fail);
                    stencils[face].depthFailureOperation = to_mtl(faces[face]->depth_fail);
                    stencils[face].depthStencilPassOperation = to_mtl(faces[face]->pass);
                    stencils[face].readMask = state.stencil_read_mask;
                    stencils[face].writeMask = state.stencil_write_mask;
                }
                desc.frontFaceStencil = stencils[0];
                desc.backFaceStencil = stencils[1];
            }
            id<MTLDepthStencilState> result = [device->device newDepthStencilStateWithDescriptor:desc];
            if (!result)
                fail("depth-stencil state creation", nil);
            DepthStencilRecord& record = device->depth_stencil_states.push();
            memcpy(&record.desc, &state, sizeof(state));
            record.state = result;
            return result;
        }

    } // namespace

    // ---------------------------------------------------------------------------------------------------------------------
    // Device
    // ---------------------------------------------------------------------------------------------------------------------

    DeviceInit create_device(const DeviceDesc& desc) noexcept
    {
        assert(desc.desired_queue_count != 0);
        @autoreleasepool
        {
            id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
            if (!mtl_device || ![mtl_device supportsFamily:MTLGPUFamilyMetal4])
                return { .error = Error::unsupported };

            Device* device = new Device{};
            device->device = mtl_device;
            const uint32 requested = desc.desired_queue_count + desc.desired_compute_queue_count + desc.desired_copy_queue_count;
            if (requested > max_queues)
            {
                delete device;
                return { .error = Error::unsupported };
            }
            NSError* error = nil;
            device->residency = [mtl_device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
            device->compiler = [mtl_device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
            device->idle_event = [mtl_device newSharedEvent];
            if (!device->residency || !device->compiler || !device->idle_event)
            {
                delete device;
                return { .error = Error::driver_error };
            }
            for (uint32 index = 0; index < requested; ++index)
            {
                device->queues[index] = [mtl_device newMTL4CommandQueue];
                if (!device->queues[index])
                {
                    delete device;
                    return { .error = Error::driver_error };
                }
                [device->queues[index] addResidencySet:device->residency];
            }
            device->queue_count = requested;

            if (desc.window)
            {
                device->layer = (__bridge CAMetalLayer*)desc.window;
                device->layer.device = mtl_device;
                device->layer.pixelFormat = to_mtl(desc.swapchain_format);
                device->layer.maximumDrawableCount = desc.desired_swapchain_image_count < 2   ? 2
                                                     : desc.desired_swapchain_image_count > 3 ? 3
                                                                                              : desc.desired_swapchain_image_count;
                [device->queues[0] addResidencySet:device->layer.residencySet];
            }

            strncpy(device->device_name, mtl_device.name.UTF8String, sizeof(device->device_name) - 1);
            device->caps = {
                .device_name = device->device_name,
                .queue_count = requested,
                .general_queue_count = desc.desired_queue_count,
                .compute_queue_count = desc.desired_compute_queue_count,
                .copy_queue_count = desc.desired_copy_queue_count,
                .copy_texture_granularity = { .x = 1, .y = 1, .z = 1 },
                .max_push_data_size = 256,
                .texture_heap_alignment = texture_heap_alignment,
                .texture_descriptor_size = descriptor_size,
                .sampler_descriptor_size = descriptor_size,
                .timestamp_period_ns = 1.0f,
                .sub_texel_precision_bits = 8,
                .texture_compression_bc = mtl_device.supportsBCTextureCompression,
                .texture_compression_astc = [mtl_device supportsFamily:MTLGPUFamilyApple2],
                .storage_input_output16 = true,
            };
            return { .device = device };
        }
    }

    void destroy_device(Device* device) noexcept
    {
        if (!device)
            return;
        for (uint32 index = 0; index < device->queue_count; ++index)
            [device->queues[index] removeResidencySet:device->residency];
        if (device->layer)
            [device->queues[0] removeResidencySet:device->layer.residencySet];
        delete device;
    }

    const DeviceCaps& get_device_caps(const Device* device) noexcept
    {
        assert(device);
        return device->caps;
    }

    bool supports_texture_format(const Device* device, Format format, TextureUsage usage) noexcept
    {
        assert(device);
        const TextureFormatInfo info = get_texture_format_info(format);
        if (to_mtl(format) == MTLPixelFormatInvalid)
            return false;
        if ((info.depth || info.stencil) && (has_usage(usage, TextureUsage::color_attachment) || has_usage(usage, TextureUsage::storage)))
            return false;
        if (!info.depth && !info.stencil && has_usage(usage, TextureUsage::depth_stencil_attachment))
            return false;
        if (is_compressed(format) && (has_usage(usage, TextureUsage::storage) || has_usage(usage, TextureUsage::color_attachment)))
            return false;
        if (is_bc(format))
            return device->caps.texture_compression_bc;
        if (format == Format::astc_4x4_srgb || format == Format::astc_4x4_unorm)
            return device->caps.texture_compression_astc;
        return true;
    }

    uint32x2 get_drawable_extent(Device* device) noexcept
    {
        assert(device);
        if (!device->layer)
            return {};
        const CGSize size = device->layer.drawableSize;
        return { .x = static_cast<uint32>(size.width), .y = static_cast<uint32>(size.height) };
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Timelines
    // ---------------------------------------------------------------------------------------------------------------------

    TimelineSemaphore* create_timeline_semaphore(Device* device, uint64 initial_value) noexcept
    {
        assert(device);
        TimelineSemaphore* semaphore = new TimelineSemaphore{ .event = [device->device newSharedEvent] };
        if (!semaphore->event)
            fail("shared event creation", nil);
        semaphore->event.signaledValue = initial_value;
        return semaphore;
    }

    void destroy_timeline_semaphore(TimelineSemaphore* semaphore) noexcept
    {
        delete semaphore;
    }

    uint64 timeline_completed_value(const TimelineSemaphore* semaphore) noexcept
    {
        assert(semaphore);
        return semaphore->event.signaledValue;
    }

    void wait_timeline(TimelinePoint point) noexcept
    {
        assert(point.semaphore);
        [point.semaphore->event waitUntilSignaledValue:point.value timeoutMS:UINT64_MAX];
    }

    void wait_idle(Device* device) noexcept
    {
        assert(device && !device->drawable && "wait_idle is not allowed while a swapchain image is acquired");
        for (uint32 index = 0; index < device->queue_count; ++index)
            [device->queues[index] signalEvent:device->idle_event value:++device->idle_value];
        [device->idle_event waitUntilSignaledValue:device->idle_value timeoutMS:UINT64_MAX];
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Presentation
    // ---------------------------------------------------------------------------------------------------------------------

    SwapchainFrame acquire(CommandBuffer* commands) noexcept
    {
        assert(commands && !commands->swapchain && !commands->render);
        Device* device = commands->device;
        assert(device->layer && !device->drawable);
        const uint32x2 extent = get_drawable_extent(device);
        if (extent.x == 0 || extent.y == 0)
            return {};
        @autoreleasepool
        {
            device->drawable = [device->layer nextDrawable];
        }
        if (!device->drawable)
            return {};
        device->swapchain_view = {
            .texture = device->drawable.texture,
            .width = static_cast<uint32>(device->drawable.texture.width),
            .height = static_cast<uint32>(device->drawable.texture.height),
        };
        commands->swapchain = true;
        return {
            .render_view = &device->swapchain_view,
            .extent = { .x = device->swapchain_view.width, .y = device->swapchain_view.height },
        };
    }

    void submit_and_present(Device* device, const SubmitDesc& desc) noexcept
    {
        assert(device && device->drawable);
#if !defined(NDEBUG)
        bool contains_swapchain = false;
        for (size_t index = 0; index < desc.commands.size; ++index)
            contains_swapchain = contains_swapchain || desc.commands.data[index]->swapchain;
        assert(contains_swapchain && "presentation submission must include the acquired command buffer");
#endif
        for (size_t index = 0; index < desc.commands.size; ++index)
            desc.commands.data[index]->swapchain = false;
        submit_commands(device, desc, 0, device->drawable);
        device->drawable = nil;
        device->swapchain_view.texture = nil;
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Memory
    // ---------------------------------------------------------------------------------------------------------------------

    GpuHeap create_gpu_heap(Device* device, uint64 byte_count, MemoryType memory) noexcept
    {
        assert(device && byte_count);
        MTLResourceOptions options = MTLResourceHazardTrackingModeUntracked;
        switch (memory)
        {
        case MemoryType::cpu_visible: options |= MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined; break;
        case MemoryType::gpu_only: options |= MTLResourceStorageModePrivate; break;
        case MemoryType::readback:
        case MemoryType::texture_descriptor_heap:
        case MemoryType::sampler_descriptor_heap: options |= MTLResourceStorageModeShared; break;
        }
        GpuHeapOwner* owner = new GpuHeapOwner{ .device = device, .buffer = [device->device newBufferWithLength:byte_count options:options] };
        if (!owner->buffer)
            fail("buffer allocation", nil);
        if (memory == MemoryType::texture_descriptor_heap)
        {
            MTLResourceViewPoolDescriptor* pool_desc = [MTLResourceViewPoolDescriptor new];
            pool_desc.resourceViewCount = byte_count / descriptor_size;
            NSError* error = nil;
            owner->view_pool = [device->device newTextureViewPoolWithDescriptor:pool_desc error:&error];
            if (!owner->view_pool)
                fail("texture view pool creation", error);
        }
        track_allocation(device, owner->buffer);
        const GpuHeap heap{
        .range = {
            .cpu = memory == MemoryType::gpu_only ? nullptr : static_cast<byte*>(owner->buffer.contents),
            .gpu = reinterpret_cast<byte*>(static_cast<uintptr>(owner->buffer.gpuAddress)),
            .size = byte_count,
        },
        .owner = owner,
    };
        device->heaps.push() = { .gpu = heap.range.gpu, .cpu = heap.range.cpu, .size = byte_count, .owner = owner };
        return heap;
    }

    void destroy_gpu_heap(const GpuHeap& heap) noexcept
    {
        if (!heap.owner)
            return;
        Device* device = heap.owner->device;
        for (uint32 index = 0; index < device->heaps.size; ++index)
        {
            if (device->heaps.data[index].owner == heap.owner)
            {
                device->heaps.remove(index);
                break;
            }
        }
        untrack_allocation(device, heap.owner->buffer);
        delete heap.owner;
    }

    TextureHeap create_texture_heap(Device* device, uint64 byte_count) noexcept
    {
        assert(device && byte_count);
        MTLHeapDescriptor* desc = [MTLHeapDescriptor new];
        desc.type = MTLHeapTypePlacement;
        desc.storageMode = MTLStorageModePrivate;
        desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        desc.size = byte_count;
        TextureHeapOwner* owner = new TextureHeapOwner{ .device = device, .heap = [device->device newHeapWithDescriptor:desc] };
        if (!owner->heap)
            fail("texture heap allocation", nil);
        track_allocation(device, owner->heap);
        return { .size = byte_count, .owner = owner };
    }

    void destroy_texture_heap(const TextureHeap& heap) noexcept
    {
        if (!heap.owner)
            return;
        untrack_allocation(heap.owner->device, heap.owner->heap);
        delete heap.owner;
    }

    SizeAlign get_texture_size_align(Device* device, const TextureDesc& desc) noexcept
    {
        assert(device);
        const MTLSizeAndAlign size_align = [device->device heapTextureSizeAndAlignWithDescriptor:make_texture_descriptor(desc)];
        assert(texture_heap_alignment % size_align.align == 0);
        return { .size = size_align.size, .align = size_align.align };
    }

    Texture* create_texture(CommandBuffer* commands, const TextureDesc& desc, const TextureHeap& heap, uint64 offset) noexcept
    {
        assert(commands && heap.owner);
        Texture* texture = new Texture{
            .device = commands->device,
            .texture = [heap.owner->heap newTextureWithDescriptor:make_texture_descriptor(desc) offset:offset],
            .width = desc.extent.x,
            .height = desc.extent.y,
            .depth = desc.extent.z,
            .layer_count = desc.layer_count,
            .format = desc.format,
        };
        if (!texture->texture)
            fail("placed texture creation", nil);
        // Placed textures are resident through their heap, but GPU shader validation only accepts resources added individually.
        track_allocation(commands->device, texture->texture);
        return texture;
    }

    void destroy_texture(Texture* texture) noexcept
    {
        if (!texture)
            return;
        untrack_allocation(texture->device, texture->texture);
        delete texture;
    }

    RenderView* create_render_view(Texture* texture, const RenderViewDesc& desc) noexcept
    {
        assert(texture);
        const uint32 width = texture->width >> desc.mip_level;
        const uint32 height = texture->height >> desc.mip_level;
        RenderView* view = new RenderView{
            .texture = [texture->texture newTextureViewWithPixelFormat:texture->texture.pixelFormat
                                                           textureType:MTLTextureType2D
                                                                levels:NSMakeRange(desc.mip_level, 1)
                                                                slices:NSMakeRange(desc.slice, 1)],
            .width = width ? width : 1,
            .height = height ? height : 1,
        };
        if (!view->texture)
            fail("render view creation", nil);
        return view;
    }

    void destroy_render_view(RenderView* render_view) noexcept
    {
        delete render_view;
    }

    void write_texture_descriptor(
        Device* device,
        void* cpu_destination,
        const Texture* texture,
        TextureDescriptorType,
        const TextureDescriptorDesc& desc
    ) noexcept
    {
        assert(device && cpu_destination && texture);
        const HeapRecord& heap = find_heap(device, cpu_destination, true);
        assert(heap.owner->view_pool && "texture descriptors must be written into a texture_descriptor_heap");
        const uint64 offset = static_cast<uint64>(static_cast<byte*>(cpu_destination) - heap.cpu);
        assert(offset % descriptor_size == 0);

        MTLPixelFormat format = to_mtl(desc.format == Format::undefined ? texture->format : desc.format);
        if (desc.aspect == TextureAspect::stencil || (desc.aspect == TextureAspect::automatic && texture->format == Format::s8_uint))
            format = texture->format == Format::d32_float_s8_uint ? MTLPixelFormatX32_Stencil8 : MTLPixelFormatStencil8;
        MTLTextureViewDescriptor* view = [MTLTextureViewDescriptor new];
        view.pixelFormat = format;
        view.textureType = texture->texture.textureType;
        view.levelRange = NSMakeRange(desc.base_mip, desc.mip_count ? desc.mip_count : texture->texture.mipmapLevelCount - desc.base_mip);
        view.sliceRange = NSMakeRange(desc.base_layer, desc.layer_count ? desc.layer_count : texture->layer_count - desc.base_layer);
        const MTLResourceID id = [heap.owner->view_pool setTextureView:texture->texture descriptor:view atIndex:offset / descriptor_size];
        memcpy(cpu_destination, &id, sizeof(id));
    }

    void write_sampler_descriptor(Device* device, void* cpu_destination, const SamplerDesc& desc) noexcept
    {
        assert(device && cpu_destination);
        id<MTLSamplerState> state = nil;
        for (uint32 index = 0; index < device->samplers.size && !state; ++index)
        {
            if (memcmp(&device->samplers.data[index].desc, &desc, sizeof(desc)) == 0)
                state = device->samplers.data[index].state;
        }
        if (!state)
        {
            MTLSamplerDescriptor* sampler = [MTLSamplerDescriptor new];
            sampler.minFilter = desc.min_filter == Filter::linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            sampler.magFilter = desc.mag_filter == Filter::linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            sampler.mipFilter = desc.mip_filter == Filter::linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
            sampler.sAddressMode = to_mtl(desc.address_u);
            sampler.tAddressMode = to_mtl(desc.address_v);
            sampler.rAddressMode = to_mtl(desc.address_w);
            sampler.maxAnisotropy = desc.anisotropic ? 4 : 1;
            sampler.compareFunction = desc.compare_enabled ? to_mtl(desc.compare) : MTLCompareFunctionNever;
            sampler.lodMaxClamp = FLT_MAX;
            sampler.supportArgumentBuffers = YES;
            state = [device->device newSamplerStateWithDescriptor:sampler];
            if (!state)
                fail("sampler creation", nil);
            SamplerRecord& record = device->samplers.push();
            memcpy(&record.desc, &desc, sizeof(desc));
            record.state = state;
        }
        const MTLResourceID id = state.gpuResourceID;
        memcpy(cpu_destination, &id, sizeof(id));
    }

    uint64 get_texture_handle(const GpuHeap& texture_descriptor_heap, uint32 slot) noexcept
    {
        assert(texture_descriptor_heap.range.cpu && (slot + 1) * descriptor_size <= texture_descriptor_heap.range.size);
        uint64 handle = 0;
        memcpy(&handle, texture_descriptor_heap.range.cpu + slot * descriptor_size, sizeof(handle));
        return handle;
    }

    uint64 get_sampler_handle(const GpuHeap& sampler_descriptor_heap, uint32 slot) noexcept
    {
        return get_texture_handle(sampler_descriptor_heap, slot);
    }

    void* get_native_texture(const Texture* texture) noexcept
    {
        assert(texture);
        return (__bridge void*)texture->texture;
    }

    void* get_native_timeline_semaphore(const TimelineSemaphore* semaphore) noexcept
    {
        assert(semaphore);
        return (__bridge void*)semaphore->event;
    }

    void* get_native_residency_set(const Device* device) noexcept
    {
        assert(device);
        return (__bridge void*)device->residency;
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------------------------------------------------------

    PSO* create_graphics_pso(Device* device, const GraphicsPSODesc& desc) noexcept
    {
        assert(device && desc.color_targets.size <= max_color_attachments);
        @autoreleasepool
        {
            MTL4RenderPipelineDescriptor* pipeline = [MTL4RenderPipelineDescriptor new];
            pipeline.vertexFunctionDescriptor = function_descriptor(load_library(device, desc.vertex.metallib), @"vertexMain");
            if (desc.fragment.metallib.size)
                pipeline.fragmentFunctionDescriptor = function_descriptor(load_library(device, desc.fragment.metallib), @"fragmentMain");
            pipeline.rasterSampleCount = 1;
            pipeline.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            for (size_t index = 0; index < desc.color_targets.size; ++index)
            {
                const ColorTargetDesc& target = desc.color_targets.data[index];
                MTL4RenderPipelineColorAttachmentDescriptor* attachment = pipeline.colorAttachments[index];
                attachment.pixelFormat = to_mtl(target.format);
                attachment.blendingState = target.blend.enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
                attachment.sourceRGBBlendFactor = to_mtl(target.blend.color.source);
                attachment.destinationRGBBlendFactor = to_mtl(target.blend.color.destination);
                attachment.rgbBlendOperation = to_mtl(target.blend.color.operation);
                attachment.sourceAlphaBlendFactor = to_mtl(target.blend.alpha.source);
                attachment.destinationAlphaBlendFactor = to_mtl(target.blend.alpha.destination);
                attachment.alphaBlendOperation = to_mtl(target.blend.alpha.operation);
                // Vulkan's mask is R=1, G=2, B=4, A=8; Metal's is A=1, B=2, G=4, R=8.
                attachment.writeMask = ((target.write_mask & 1u) ? MTLColorWriteMaskRed : 0) | ((target.write_mask & 2u) ? MTLColorWriteMaskGreen : 0) |
                                       ((target.write_mask & 4u) ? MTLColorWriteMaskBlue : 0) | ((target.write_mask & 8u) ? MTLColorWriteMaskAlpha : 0);
            }
            NSError* error = nil;
            PSO* pso = new PSO{
                .render = [device->compiler newRenderPipelineStateWithDescriptor:pipeline compilerTaskOptions:nil error:&error],
                .cull = desc.rasterization.cull == CullMode::none ? MTLCullModeNone : MTLCullModeBack,
                // Metal's viewport is flipped (see set_viewport), which mirrors winding; this keeps Vulkan's front-face result.
                .winding = desc.rasterization.cull == CullMode::counter_clockwise ? MTLWindingCounterClockwise : MTLWindingClockwise,
                .depth_bias = desc.rasterization.depth_bias_constant,
                .depth_bias_slope = desc.rasterization.depth_bias_slope,
                .depth_bias_clamp = desc.rasterization.depth_bias_clamp,
            };
            if (!pso->render)
                fail("render pipeline creation", error);
            return pso;
        }
    }

    PSO* create_mesh_pso(Device*, const MeshPSODesc&) noexcept
    {
        assert(false && "Metal 4: mesh pipelines are not implemented yet");
        return nullptr;
    }

    PSO* create_compute_pso(Device* device, const ShaderCode& compute) noexcept
    {
        assert(device);
        @autoreleasepool
        {
            MTL4ComputePipelineDescriptor* pipeline = [MTL4ComputePipelineDescriptor new];
            pipeline.computeFunctionDescriptor = function_descriptor(load_library(device, compute.metallib), @"computeMain");
        assert(compute.threads_per_threadgroup.x != 0 && "Metal compute shaders need ShaderCode::threads_per_threadgroup");
        const MTLSize threads = MTLSizeMake(compute.threads_per_threadgroup.x, compute.threads_per_threadgroup.y ? compute.threads_per_threadgroup.y : 1,
                                            compute.threads_per_threadgroup.z ? compute.threads_per_threadgroup.z : 1);
        pipeline.requiredThreadsPerThreadgroup = threads;
            NSError* error = nil;
            PSO* pso = new PSO{ .compute = [device->compiler newComputePipelineStateWithDescriptor:pipeline compilerTaskOptions:nil error:&error] };
            if (!pso->compute)
                fail("compute pipeline creation", error);
        pso->threads_per_threadgroup = threads;
            return pso;
        }
    }

    void destroy_pso(PSO* pso) noexcept
    {
        delete pso;
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Command pools and buffers
    // ---------------------------------------------------------------------------------------------------------------------

    CommandPool* create_command_pool(Device* device, uint32 queue_index) noexcept
    {
        assert(device && queue_index < device->queue_count);
        CommandPool* pool = new CommandPool{ .device = device, .queue_index = queue_index, .allocator = [device->device newCommandAllocator] };
        if (!pool->allocator)
            fail("command allocator creation", nil);
        return pool;
    }

    void destroy_command_pool(CommandPool* pool) noexcept
    {
        if (!pool)
            return;
        while (pool->first)
        {
            CommandBuffer* commands = pool->first;
            assert(!commands->swapchain && "an acquired swapchain image must be presented before destroying its command pool");
            pool->first = commands->next;
            delete commands;
        }
        for (uint32 index = 0; index < pool->root_chunks.size; ++index)
            untrack_allocation(pool->device, pool->root_chunks.data[index]);
        delete pool;
    }

    void reset_command_pool(CommandPool* pool) noexcept
    {
        assert(pool);
        [pool->allocator reset];
        pool->next_buffer = pool->first;
        pool->root_chunk = 0;
        pool->root_offset = 0;
    }

    CommandBuffer* begin_commands(CommandPool* pool) noexcept
    {
        assert(pool);
        CommandBuffer* commands = pool->next_buffer;
        if (commands)
        {
            pool->next_buffer = commands->next;
        }
        else
        {
            MTL4ArgumentTableDescriptor* table = [MTL4ArgumentTableDescriptor new];
            table.maxBufferBindCount = 1;
            NSError* error = nil;
            commands = new CommandBuffer{
                .pool = pool,
                .device = pool->device,
                .command_buffer = [pool->device->device newCommandBuffer],
                .argument_table = [pool->device->device newArgumentTableWithDescriptor:table error:&error],
            };
            if (!commands->command_buffer || !commands->argument_table)
                fail("command buffer creation", error);
            if (pool->last)
                pool->last->next = commands;
            else
                pool->first = commands;
            pool->last = commands;
        }
        assert(!commands->swapchain);
        commands->pso = nullptr;
        commands->barrier_source = 0;
        commands->barrier_destination = 0;
        [commands->command_buffer beginCommandBufferWithAllocator:pool->allocator];
        return commands;
    }

    void end_commands(CommandBuffer* commands) noexcept
    {
        assert(commands && !commands->render && "end the render pass before ending the command buffer");
        end_compute_encoder(commands);
        [commands->command_buffer endCommandBuffer];
    }

    void submit(Device* device, const SubmitDesc& desc, uint32 queue_index) noexcept
    {
        assert(device);
#if !defined(NDEBUG)
        for (size_t index = 0; index < desc.commands.size; ++index)
            assert(!desc.commands.data[index]->swapchain && "swapchain commands require submit_and_present");
#endif
        submit_commands(device, desc, queue_index, nil);
    }

    void set_texture_descriptor_heap(CommandBuffer* commands, GpuRange heap) noexcept
    {
        assert(commands && heap.gpu);
    }

    void set_sampler_descriptor_heap(CommandBuffer* commands, GpuRange heap) noexcept
    {
        assert(commands && heap.gpu);
    }

    void write_timestamp(CommandBuffer* commands, uint64* gpu_destination, Stage) noexcept
    {
        assert(commands && gpu_destination);
    }

    void barrier(CommandBuffer* commands, Stage before, Access, Stage after, Access) noexcept
    {
        assert(commands);
        const MTLStages source = to_mtl(before);
        const MTLStages destination = to_mtl(after);
        if (!source || !destination)
            return;
        commands->barrier_source |= source;
        commands->barrier_destination |= destination;
        // The open encoder: earlier encoders through a queue barrier, its own earlier commands through an encoder barrier.
        id<MTL4CommandEncoder> encoder = commands->render ? static_cast<id<MTL4CommandEncoder>>(commands->render) : commands->compute;
        const MTLStages encoder_stages = commands->render ? render_encoder_stages : compute_encoder_stages;
        if (!encoder || !(destination & encoder_stages))
            return;
        [encoder barrierAfterQueueStages:source beforeStages:destination & encoder_stages visibilityOptions:MTL4VisibilityOptionDevice];
        if (source & encoder_stages)
            [encoder barrierAfterEncoderStages:source & encoder_stages beforeEncoderStages:destination & encoder_stages
                             visibilityOptions:MTL4VisibilityOptionDevice];
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Copies
    // ---------------------------------------------------------------------------------------------------------------------

    namespace {

        struct TextureCopyRegion
        {
            MTLOrigin origin;
            MTLSize size;
            uint64 row_pitch;
            uint64 slice_pitch;
            uint32 slice_count;
        };

        TextureCopyRegion make_copy_region(const Texture& texture, const TextureCopyDesc& copy) noexcept
        {
            uint32 width = texture.width >> copy.mip_level;
            uint32 height = texture.height >> copy.mip_level;
            uint32 depth = texture.depth >> copy.mip_level;
            width = width ? width : 1;
            height = height ? height : 1;
            depth = depth ? depth : 1;
            const TextureFormatInfo info = get_texture_format_info(texture.format);
            const MTLSize size = MTLSizeMake(
                copy.extent.x ? copy.extent.x : width - copy.offset.x,
                copy.extent.y ? copy.extent.y : height - copy.offset.y,
                copy.extent.z ? copy.extent.z : depth - copy.offset.z
            );
            const uint64 row_pitch =
                copy.row_pitch_bytes ? copy.row_pitch_bytes : (size.width + info.block_extent.x - 1) / info.block_extent.x * info.bytes_per_block;
            const uint64 rows = (size.height + info.block_extent.y - 1) / info.block_extent.y;
            return {
                .origin = MTLOriginMake(copy.offset.x, copy.offset.y, copy.offset.z),
                .size = size,
                .row_pitch = row_pitch,
                .slice_pitch = copy.slice_pitch_bytes ? copy.slice_pitch_bytes : row_pitch * rows * (texture.depth > 1 ? size.depth : 1),
                .slice_count = copy.slice_count ? copy.slice_count : texture.layer_count - copy.base_slice,
            };
        }

    } // namespace

    void copy_memory(CommandBuffer* commands, GpuRange source, GpuRange destination) noexcept
    {
        assert(commands && source.size <= destination.size);
        const HeapRecord& from = find_heap(commands->device, source.gpu, false);
        const HeapRecord& to = find_heap(commands->device, destination.gpu, false);
        [compute_encoder(commands) copyFromBuffer:from.owner->buffer
                                     sourceOffset:static_cast<NSUInteger>(static_cast<byte*>(source.gpu) - from.gpu)
                                         toBuffer:to.owner->buffer
                                destinationOffset:static_cast<NSUInteger>(static_cast<byte*>(destination.gpu) - to.gpu)
                                             size:source.size];
    }

    void copy_memory_to_texture(CommandBuffer* commands, GpuRange source, Texture* destination, const TextureCopyDesc& copy) noexcept
    {
        assert(commands && destination);
        const HeapRecord& from = find_heap(commands->device, source.gpu, false);
        const TextureCopyRegion region = make_copy_region(*destination, copy);
        const NSUInteger offset = static_cast<NSUInteger>(static_cast<byte*>(source.gpu) - from.gpu);
        id<MTL4ComputeCommandEncoder> encoder = compute_encoder(commands);
        for (uint32 slice = 0; slice < region.slice_count; ++slice)
            [encoder copyFromBuffer:from.owner->buffer
                       sourceOffset:offset + slice * region.slice_pitch
                  sourceBytesPerRow:region.row_pitch
                sourceBytesPerImage:region.slice_pitch
                         sourceSize:region.size
                          toTexture:destination->texture
                   destinationSlice:copy.base_slice + slice
                   destinationLevel:copy.mip_level
                  destinationOrigin:region.origin];
    }

    void copy_texture_to_memory(CommandBuffer* commands, Texture* source, GpuRange destination, const TextureCopyDesc& copy) noexcept
    {
        assert(commands && source);
        const HeapRecord& to = find_heap(commands->device, destination.gpu, false);
        const TextureCopyRegion region = make_copy_region(*source, copy);
        const NSUInteger offset = static_cast<NSUInteger>(static_cast<byte*>(destination.gpu) - to.gpu);
        id<MTL4ComputeCommandEncoder> encoder = compute_encoder(commands);
        for (uint32 slice = 0; slice < region.slice_count; ++slice)
            [encoder copyFromTexture:source->texture
                             sourceSlice:copy.base_slice + slice
                             sourceLevel:copy.mip_level
                            sourceOrigin:region.origin
                              sourceSize:region.size
                                toBuffer:to.owner->buffer
                       destinationOffset:offset + slice * region.slice_pitch
                  destinationBytesPerRow:region.row_pitch
                destinationBytesPerImage:region.slice_pitch];
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // Rendering
    // ---------------------------------------------------------------------------------------------------------------------

    void begin_render_pass(CommandBuffer* commands, const RenderingDesc& desc, RenderingFlags flags) noexcept
    {
        assert(commands && !commands->render && desc.colors.size <= max_color_attachments);
        const RenderView* area = desc.colors.size         ? desc.colors.data[0].render_view
                                 : desc.depth.render_view ? desc.depth.render_view
                                                          : desc.stencil.render_view;
        assert(area && "begin_render_pass requires an attachment to determine the render area");
        end_compute_encoder(commands);

        MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
        for (size_t index = 0; index < desc.colors.size; ++index)
        {
            const ColorAttachment& attachment = desc.colors.data[index];
            assert(attachment.render_view);
            pass.colorAttachments[index].texture = attachment.render_view->texture;
            pass.colorAttachments[index].loadAction = to_mtl(attachment.load);
            pass.colorAttachments[index].storeAction = to_mtl(attachment.store);
            pass.colorAttachments[index].clearColor = MTLClearColorMake(attachment.clear.x, attachment.clear.y, attachment.clear.z, attachment.clear.w);
        }
        if (desc.depth.render_view)
        {
            pass.depthAttachment.texture = desc.depth.render_view->texture;
            pass.depthAttachment.loadAction = to_mtl(desc.depth.load);
            pass.depthAttachment.storeAction = to_mtl(desc.depth.store);
            pass.depthAttachment.clearDepth = desc.depth.clear;
        }
        if (desc.stencil.render_view)
        {
            pass.stencilAttachment.texture = desc.stencil.render_view->texture;
            pass.stencilAttachment.loadAction = to_mtl(desc.stencil.load);
            pass.stencilAttachment.storeAction = to_mtl(desc.stencil.store);
            pass.stencilAttachment.clearStencil = desc.stencil.clear;
        }
        MTL4RenderEncoderOptions options = MTL4RenderEncoderOptionNone;
        if (static_cast<uint32>(flags) & static_cast<uint32>(RenderingFlags::suspending))
            options |= MTL4RenderEncoderOptionSuspending;
        if (static_cast<uint32>(flags) & static_cast<uint32>(RenderingFlags::resuming))
            options |= MTL4RenderEncoderOptionResuming;
        commands->render = [commands->command_buffer renderCommandEncoderWithDescriptor:pass options:options];
        if (!commands->render)
            fail("render encoder creation", nil);
        [commands->render setArgumentTable:commands->argument_table atStages:render_argument_stages];
        if (commands->barrier_destination & render_encoder_stages)
            [commands->render barrierAfterQueueStages:commands->barrier_source
                                         beforeStages:commands->barrier_destination & render_encoder_stages
                                    visibilityOptions:MTL4VisibilityOptionDevice];

        set_viewport(commands, { .width = static_cast<float>(area->width), .height = static_cast<float>(area->height) });
        set_scissor(commands, { .width = area->width, .height = area->height });
        set_depth_stencil(commands, {});
    }

    void end_render_pass(CommandBuffer* commands) noexcept
    {
        assert(commands && commands->render);
        [commands->render endEncoding];
        commands->render = nil;
    }

    void set_viewport(CommandBuffer* commands, const Viewport& viewport) noexcept
    {
        assert(commands && commands->render);
        // Vulkan clip space has +y down, Metal's has +y up. Flipping the viewport makes the same shader output land on the same pixels.
        [commands->render setViewport:{
                                          .originX = viewport.x,
                                          .originY = viewport.y + viewport.height,
                                          .width = viewport.width,
                                          .height = -viewport.height,
                                          .znear = viewport.min_depth,
                                          .zfar = viewport.max_depth,
                                      }];
    }

    void set_scissor(CommandBuffer* commands, const Scissor& scissor) noexcept
    {
        assert(commands && commands->render && scissor.x >= 0 && scissor.y >= 0);
        [commands->render setScissorRect:{
                                             .x = static_cast<NSUInteger>(scissor.x),
                                             .y = static_cast<NSUInteger>(scissor.y),
                                             .width = scissor.width,
                                             .height = scissor.height,
                                         }];
    }

    void set_depth_stencil(CommandBuffer* commands, const DepthStencilState& state) noexcept
    {
        assert(commands && commands->render);
        [commands->render setDepthStencilState:depth_stencil_state(commands->device, state)];
        if (state.stencil_test)
            [commands->render setStencilFrontReferenceValue:state.front.reference backReferenceValue:state.back.reference];
    }

    void bind_pso(CommandBuffer* commands, const PSO* pso) noexcept
    {
        assert(commands && pso);
        commands->pso = pso;
        if (pso->compute)
        {
            [compute_encoder(commands) setComputePipelineState:pso->compute];
            return;
        }
        assert(commands->render && "graphics PSOs must be bound inside a render pass");
        [commands->render setRenderPipelineState:pso->render];
        [commands->render setCullMode:pso->cull];
        [commands->render setFrontFacingWinding:pso->winding];
        [commands->render setDepthBias:pso->depth_bias slopeScale:pso->depth_bias_slope clamp:pso->depth_bias_clamp];
    }

    void draw(CommandBuffer* commands, ByteSpan root, uint32 vertex_count, uint32 instance_count, uint32 first_vertex, uint32 first_instance) noexcept
    {
        assert(commands && commands->render);
        emit_root_data(commands, root);
        [commands->render drawPrimitives:MTLPrimitiveTypeTriangle
                             vertexStart:first_vertex
                             vertexCount:vertex_count
                           instanceCount:instance_count
                            baseInstance:first_instance];
    }

    void draw_indexed(
        CommandBuffer* commands,
        ByteSpan root,
        GpuRange indices,
        IndexType type,
        uint32 index_count,
        uint32 instance_count,
        uint32 first_index,
        int32 vertex_offset,
        uint32 first_instance
    ) noexcept
    {
        assert(commands && commands->render);
        emit_root_data(commands, root);
        const uint64 index_size = type == IndexType::uint16 ? 2 : 4;
        [commands->render drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                     indexCount:index_count
                                      indexType:type == IndexType::uint16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                                    indexBuffer:reinterpret_cast<uintptr>(indices.gpu) + first_index * index_size
                              indexBufferLength:indices.size - first_index * index_size
                                  instanceCount:instance_count
                                     baseVertex:vertex_offset
                                   baseInstance:first_instance];
    }

    // Indirect argument layouts match Vulkan's: {count, instances, first, (vertex offset,) first instance}.
    void draw_indirect(CommandBuffer* commands, ByteSpan root, GpuRange arguments, uint32 draw_count, uint32 stride) noexcept
    {
        assert(commands && commands->render);
        emit_root_data(commands, root);
        const uint64 step = stride ? stride : 4 * sizeof(uint32);
        for (uint32 index = 0; index < draw_count; ++index)
            [commands->render drawPrimitives:MTLPrimitiveTypeTriangle indirectBuffer:reinterpret_cast<uintptr>(arguments.gpu) + index * step];
    }

    void draw_indexed_indirect(
        CommandBuffer* commands,
        ByteSpan root,
        GpuRange indices,
        IndexType type,
        GpuRange arguments,
        uint32 draw_count,
        uint32 stride
    ) noexcept
    {
        assert(commands && commands->render);
        emit_root_data(commands, root);
        const uint64 step = stride ? stride : 5 * sizeof(uint32);
        for (uint32 index = 0; index < draw_count; ++index)
            [commands->render drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                          indexType:type == IndexType::uint16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                                        indexBuffer:reinterpret_cast<uintptr>(indices.gpu)
                                  indexBufferLength:indices.size
                                     indirectBuffer:reinterpret_cast<uintptr>(arguments.gpu) + index * step];
    }

    void dispatch(CommandBuffer* commands, ByteSpan root, uint32x3 group_count) noexcept
    {
        assert(commands && commands->pso && commands->pso->compute && "bind a compute PSO before dispatching");
        id<MTL4ComputeCommandEncoder> encoder = compute_encoder(commands);
        emit_root_data(commands, root);
        [encoder dispatchThreadgroups:MTLSizeMake(group_count.x, group_count.y, group_count.z) threadsPerThreadgroup:commands->pso->threads_per_threadgroup];
    }

    void dispatch_indirect(CommandBuffer* commands, ByteSpan root, GpuRange arguments) noexcept
    {
        assert(commands && commands->pso && commands->pso->compute && "bind a compute PSO before dispatching");
        id<MTL4ComputeCommandEncoder> encoder = compute_encoder(commands);
        emit_root_data(commands, root);
        [encoder dispatchThreadgroupsWithIndirectBuffer:reinterpret_cast<uintptr>(arguments.gpu) threadsPerThreadgroup:commands->pso->threads_per_threadgroup];
    }

    void draw_meshlets(CommandBuffer*, ByteSpan, uint32x3) noexcept
    {
        assert(false && "Metal 4: mesh pipelines are not implemented yet");
    }

    void draw_meshlets_indirect(CommandBuffer*, ByteSpan, GpuRange, uint32, uint32) noexcept
    {
        assert(false && "Metal 4: mesh pipelines are not implemented yet");
    }

} // namespace gpu
