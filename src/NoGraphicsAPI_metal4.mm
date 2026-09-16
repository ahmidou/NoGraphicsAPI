#include <NoGraphicsAPI/NoGraphicsAPI.hpp>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <assert.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <string.h>

// Keep assert expressions type-checked in release without evaluating them.
#if defined(NDEBUG)
#undef assert
#define assert(expression) ((void)sizeof(static_cast<bool>(expression)))
#endif

namespace gpu {

    namespace {

        struct DeviceImpl
        {
            id<MTLDevice> device;
            id<MTLCommandQueue> queue;
            CAMetalLayer* layer;
            id<CAMetalDrawable> current_drawable;
            DeviceCaps caps;
            Format swapchain_format;
            char device_name_buf[256];
        };

        struct TimelineSemaphoreImpl
        {
            id<MTLSharedEvent> event;
            DeviceImpl* device;
        };

        struct RenderViewImpl
        {
            id<MTLTexture> texture;
            uint32 width;
            uint32 height;
        };

        struct CommandBufferImpl
        {
            id<MTLCommandBuffer> command_buffer;
            id<MTLRenderCommandEncoder> render_encoder;
            DeviceImpl* device;
        };

        // ---------------------------------------------------------------------------
        // Format conversion helpers
        // ---------------------------------------------------------------------------

        MTLPixelFormat to_mtl(Format format) noexcept
        {
            switch (format)
            {
            case Format::bgra8_srgb: return MTLPixelFormatBGRA8Unorm_sRGB;
            case Format::rgba8_srgb: return MTLPixelFormatRGBA8Unorm_sRGB;
            case Format::bgra8_unorm: return MTLPixelFormatBGRA8Unorm;
            case Format::rgba8_unorm: return MTLPixelFormatRGBA8Unorm;
            case Format::r8_unorm: return MTLPixelFormatR8Unorm;
            case Format::rg8_unorm: return MTLPixelFormatRG8Unorm;
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
            case Format::d32_float: return MTLPixelFormatDepth32Float;
            case Format::d32_float_s8_uint: return MTLPixelFormatDepth32Float_Stencil8;
            case Format::d16_unorm: return MTLPixelFormatDepth16Unorm;
            case Format::s8_uint: return MTLPixelFormatStencil8;
            case Format::astc_4x4_srgb: return MTLPixelFormatASTC_4x4_sRGB;
            case Format::astc_4x4_unorm: return MTLPixelFormatASTC_4x4_LDR;
            case Format::bc7_srgb: return MTLPixelFormatBC7_RGBAUnorm_sRGB;
            case Format::bc7_unorm: return MTLPixelFormatBC7_RGBAUnorm;
            default: return MTLPixelFormatInvalid;
            }
        }

        MTLLoadAction to_mtl_load(LoadOp op) noexcept
        {
            switch (op)
            {
            case LoadOp::load: return MTLLoadActionLoad;
            case LoadOp::clear: return MTLLoadActionClear;
            case LoadOp::discard: return MTLLoadActionDontCare;
            }
            return MTLLoadActionDontCare;
        }

        MTLStoreAction to_mtl_store(StoreOp op) noexcept
        {
            switch (op)
            {
            case StoreOp::store: return MTLStoreActionStore;
            case StoreOp::discard: return MTLStoreActionDontCare;
            }
            return MTLStoreActionDontCare;
        }

    } // anonymous namespace

    // ---------------------------------------------------------------------------
    // Device lifecycle
    // ---------------------------------------------------------------------------

    DeviceInit create_device(const DeviceDesc& desc) noexcept
    {
        id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
        if (!mtl_device)
            return { .error = Error::unsupported };

        id<MTLCommandQueue> queue = [mtl_device newCommandQueue];
        if (!queue)
            return { .error = Error::driver_error };

        auto* d = new DeviceImpl{};
        d->device = mtl_device;
        d->queue = queue;
        d->current_drawable = nil;
        d->swapchain_format = desc.swapchain_format;

        // Copy device name
        const char* name = [[mtl_device name] UTF8String];
        if (name)
        {
            strncpy(d->device_name_buf, name, sizeof(d->device_name_buf) - 1);
            d->device_name_buf[sizeof(d->device_name_buf) - 1] = '\0';
        }
        else
        {
            d->device_name_buf[0] = '\0';
        }

        // Configure CAMetalLayer if a window (layer) is provided
        if (desc.window)
        {
            CAMetalLayer* layer = (__bridge CAMetalLayer*)desc.window;
            layer.device = mtl_device;
            layer.pixelFormat = to_mtl(desc.swapchain_format);
            layer.framebufferOnly = YES;
            d->layer = layer;
        }
        else
        {
            d->layer = nil;
        }

        d->caps = {
            .device_name = d->device_name_buf,
            .max_push_data_size = 256,
            .texture_heap_alignment = 16,
            .texture_descriptor_size = 0,
            .sampler_descriptor_size = 0,
            .timestamp_period_ns = 1.0f,
            .sub_texel_precision_bits = 8,
            .texture_compression_bc = false,
            .texture_compression_astc = [mtl_device supportsFamily:MTLGPUFamilyApple2],
            .storage_input_output16 = false,
        };

        return {
            .device = reinterpret_cast<Device*>(d),
        };
    }

    void destroy_device(Device* device) noexcept
    {
        if (!device)
            return;
        auto* d = reinterpret_cast<DeviceImpl*>(device);
        d->current_drawable = nil;
        d->queue = nil;
        d->device = nil;
        d->layer = nil;
        delete d;
    }

    const DeviceCaps& get_device_caps(const Device* device) noexcept
    {
        assert(device && "get_device_caps called with a null device");
        auto* d = reinterpret_cast<const DeviceImpl*>(device);
        return d->caps;
    }

    uint32x2 get_drawable_extent(Device* device) noexcept
    {
        assert(device && "get_drawable_extent called with a null device");
        auto* d = reinterpret_cast<DeviceImpl*>(device);
        if (!d->layer)
            return {};
        CGSize size = d->layer.drawableSize;
        return {
            .x = static_cast<uint32>(size.width),
            .y = static_cast<uint32>(size.height),
        };
    }

    bool supports_texture_format(const Device* device, Format format, TextureUsage usage) noexcept
    {
        assert(device);
        (void)usage;
        MTLPixelFormat mtl = to_mtl(format);
        return mtl != MTLPixelFormatInvalid;
    }

    // ---------------------------------------------------------------------------
    // Swapchain
    // ---------------------------------------------------------------------------

    SwapchainFrame acquire(Device* device) noexcept
    {
        assert(device && "acquire called with a null device");
        auto* d = reinterpret_cast<DeviceImpl*>(device);
        assert(d->layer && "acquire called on a headless device");

        id<CAMetalDrawable> drawable = [d->layer nextDrawable];
        if (!drawable)
            return {};

        d->current_drawable = drawable;

        auto* rv = new RenderViewImpl{};
        rv->texture = drawable.texture;
        rv->width = static_cast<uint32>([drawable.texture width]);
        rv->height = static_cast<uint32>([drawable.texture height]);

        return {
            .render_view = reinterpret_cast<RenderView*>(rv),
            .extent = { .x = rv->width, .y = rv->height },
        };
    }

    void submit_and_present(Device* device, Span<CommandBuffer* const> commands, TimelinePoint completion) noexcept
    {
        assert(device && commands.data && commands.size != 0);
        auto* d = reinterpret_cast<DeviceImpl*>(device);

        for (size_t i = 0; i < commands.size; ++i)
        {
            auto* cb = reinterpret_cast<CommandBufferImpl*>(commands.data[i]);
            assert(cb && cb->device == d);

            // End any active render encoder
            if (cb->render_encoder)
            {
                [cb->render_encoder endEncoding];
                cb->render_encoder = nil;
            }

            // On the last command buffer: signal timeline, present, commit
            if (i == commands.size - 1)
            {
                // Signal timeline semaphore before present+commit
                if (completion.semaphore)
                {
                    auto* ts = reinterpret_cast<TimelineSemaphoreImpl*>(completion.semaphore);
                    [cb->command_buffer encodeSignalEvent:ts->event value:completion.value];
                }

                if (d->current_drawable)
                {
                    [cb->command_buffer presentDrawable:d->current_drawable];
                }

                [cb->command_buffer commit];
                d->current_drawable = nil;
            }
            else
            {
                [cb->command_buffer commit];
            }

            delete cb;
        }
    }

    // ---------------------------------------------------------------------------
    // Commands
    // ---------------------------------------------------------------------------

    CommandBuffer* begin_commands(Device* device) noexcept
    {
        assert(device && "begin_commands called with a null device");
        auto* d = reinterpret_cast<DeviceImpl*>(device);

        id<MTLCommandBuffer> mtl_cb = [d->queue commandBuffer];
        assert(mtl_cb && "failed to create MTLCommandBuffer");

        auto* cb = new CommandBufferImpl{};
        cb->command_buffer = mtl_cb;
        cb->render_encoder = nil;
        cb->device = d;

        return reinterpret_cast<CommandBuffer*>(cb);
    }

    void begin_render_pass(CommandBuffer* commands, const RenderingDesc& desc) noexcept
    {
        assert(commands);
        auto* cb = reinterpret_cast<CommandBufferImpl*>(commands);

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];

        for (size_t i = 0; i < desc.colors.size; ++i)
        {
            const ColorAttachment& attachment = desc.colors.data[i];
            assert(attachment.render_view);
            auto* rv = reinterpret_cast<RenderViewImpl*>(attachment.render_view);

            pass.colorAttachments[i].texture = rv->texture;
            pass.colorAttachments[i].loadAction = to_mtl_load(attachment.load);
            pass.colorAttachments[i].storeAction = to_mtl_store(attachment.store);
            pass.colorAttachments[i].clearColor = MTLClearColorMake(attachment.clear.x, attachment.clear.y, attachment.clear.z, attachment.clear.w);
        }

        if (desc.depth.render_view)
        {
            auto* rv = reinterpret_cast<RenderViewImpl*>(desc.depth.render_view);
            pass.depthAttachment.texture = rv->texture;
            pass.depthAttachment.loadAction = to_mtl_load(desc.depth.load);
            pass.depthAttachment.storeAction = to_mtl_store(desc.depth.store);
            pass.depthAttachment.clearDepth = desc.depth.clear;
        }

        if (desc.stencil.render_view)
        {
            auto* rv = reinterpret_cast<RenderViewImpl*>(desc.stencil.render_view);
            pass.stencilAttachment.texture = rv->texture;
            pass.stencilAttachment.loadAction = to_mtl_load(desc.stencil.load);
            pass.stencilAttachment.storeAction = to_mtl_store(desc.stencil.store);
            pass.stencilAttachment.clearStencil = desc.stencil.clear;
        }

        cb->render_encoder = [cb->command_buffer renderCommandEncoderWithDescriptor:pass];
        assert(cb->render_encoder && "failed to create render command encoder");
    }

    void end_render_pass(CommandBuffer* commands) noexcept
    {
        assert(commands);
        auto* cb = reinterpret_cast<CommandBufferImpl*>(commands);
        if (cb->render_encoder)
        {
            [cb->render_encoder endEncoding];
            cb->render_encoder = nil;
        }
    }

    // ---------------------------------------------------------------------------
    // Sync
    // ---------------------------------------------------------------------------

    TimelineSemaphore* create_timeline_semaphore(Device* device, uint64 initial_value) noexcept
    {
        assert(device && "create_timeline_semaphore called with a null device");
        auto* d = reinterpret_cast<DeviceImpl*>(device);

        id<MTLSharedEvent> event = [d->device newSharedEvent];
        assert(event && "failed to create MTLSharedEvent");
        event.signaledValue = initial_value;

        auto* ts = new TimelineSemaphoreImpl{};
        ts->event = event;
        ts->device = d;

        return reinterpret_cast<TimelineSemaphore*>(ts);
    }

    void destroy_timeline_semaphore(TimelineSemaphore* semaphore) noexcept
    {
        if (!semaphore)
            return;
        auto* ts = reinterpret_cast<TimelineSemaphoreImpl*>(semaphore);
        ts->event = nil;
        delete ts;
    }

    uint64 timeline_completed_value(const TimelineSemaphore* semaphore) noexcept
    {
        assert(semaphore && "timeline_completed_value received a null semaphore");
        auto* ts = reinterpret_cast<const TimelineSemaphoreImpl*>(semaphore);
        return ts->event.signaledValue;
    }

    void wait_timeline(TimelinePoint point) noexcept
    {
        assert(point.semaphore && "wait_timeline requires a live timeline semaphore");
        auto* ts = reinterpret_cast<TimelineSemaphoreImpl*>(point.semaphore);

        if (ts->event.signaledValue >= point.value)
            return;

        dispatch_semaphore_t sema = dispatch_semaphore_create(0);
        MTLSharedEventListener* listener = [[MTLSharedEventListener alloc] init];

        [ts->event notifyListener:listener
                          atValue:point.value
                            block:^(id<MTLSharedEvent>, uint64) {
                              dispatch_semaphore_signal(sema);
                            }];

        dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
    }

    void wait_idle(Device* device) noexcept
    {
        assert(device && "wait_idle called with a null device");
        auto* d = reinterpret_cast<DeviceImpl*>(device);

        id<MTLCommandBuffer> cb = [d->queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // ---------------------------------------------------------------------------
    // Non-presenting submit
    // ---------------------------------------------------------------------------

    void submit(Span<CommandBuffer* const> commands, TimelinePoint completion) noexcept
    {
        assert(commands.data && commands.size != 0);

        for (size_t i = 0; i < commands.size; ++i)
        {
            auto* cb = reinterpret_cast<CommandBufferImpl*>(commands.data[i]);
            assert(cb);

            // End any active render encoder
            if (cb->render_encoder)
            {
                [cb->render_encoder endEncoding];
                cb->render_encoder = nil;
            }

            // On the last command buffer, signal the timeline
            if (i == commands.size - 1 && completion.semaphore)
            {
                auto* ts = reinterpret_cast<TimelineSemaphoreImpl*>(completion.semaphore);
                [cb->command_buffer encodeSignalEvent:ts->event value:completion.value];
            }

            [cb->command_buffer commit];
            delete cb;
        }
    }

    // ---------------------------------------------------------------------------
    // Barrier (no-op for clear screen)
    // ---------------------------------------------------------------------------

    void barrier(CommandBuffer* commands, Stage before, Access before_access, Stage after, Access after_access) noexcept
    {
        (void)commands;
        (void)before;
        (void)before_access;
        (void)after;
        (void)after_access;
    }

    // ---------------------------------------------------------------------------
    // Stubs — assert(false) for unimplemented functions
    // ---------------------------------------------------------------------------

    GpuHeap create_gpu_heap(Device*, uint64, MemoryType) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return {};
    }

    void destroy_gpu_heap(const GpuHeap&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    TextureHeap create_texture_heap(Device*, uint64) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return {};
    }

    void destroy_texture_heap(const TextureHeap&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    SizeAlign get_texture_size_align(Device*, const TextureDesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return {};
    }

    Texture* create_texture(Device*, const TextureDesc&, const TextureHeap&, uint64) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return nullptr;
    }

    void destroy_texture(Texture*) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    RenderView* create_render_view(Texture*, const RenderViewDesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return nullptr;
    }

    void destroy_render_view(RenderView*) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void write_texture_descriptor(Device*, void*, const Texture*, TextureDescriptorType, const TextureDescriptorDesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void write_sampler_descriptor(Device*, void*, const SamplerDesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    PSO* create_graphics_pso(Device*, const GraphicsPSODesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return nullptr;
    }

    PSO* create_mesh_pso(Device*, const MeshPSODesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return nullptr;
    }

    PSO* create_compute_pso(Device*, Span<const uint32>) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
        return nullptr;
    }

    void destroy_pso(PSO*) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void set_texture_descriptor_heap(CommandBuffer*, GpuRange) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void set_sampler_descriptor_heap(CommandBuffer*, GpuRange) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void copy_memory(CommandBuffer*, GpuRange, GpuRange) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void copy_memory_to_texture(CommandBuffer*, GpuRange, Texture*, const TextureCopyDesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void copy_texture_to_memory(CommandBuffer*, Texture*, GpuRange, const TextureCopyDesc&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void write_timestamp(CommandBuffer*, uint64*, Stage) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void set_viewport(CommandBuffer*, const Viewport&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void set_scissor(CommandBuffer*, const Scissor&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void set_depth_stencil(CommandBuffer*, const DepthStencilState&) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void bind_pso(CommandBuffer*, const PSO*) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void draw(CommandBuffer*, ByteSpan, uint32, uint32, uint32, uint32) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void draw_indexed(CommandBuffer*, ByteSpan, GpuRange, IndexType, uint32, uint32, uint32, int32, uint32) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void draw_indirect(CommandBuffer*, ByteSpan, GpuRange, uint32, uint32) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void draw_indexed_indirect(CommandBuffer*, ByteSpan, GpuRange, IndexType, GpuRange, uint32, uint32) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void dispatch(CommandBuffer*, ByteSpan, uint32x3) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void dispatch_indirect(CommandBuffer*, ByteSpan, GpuRange) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void draw_meshlets(CommandBuffer*, ByteSpan, uint32x3) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

    void draw_meshlets_indirect(CommandBuffer*, ByteSpan, GpuRange, uint32, uint32) noexcept
    {
        assert(false && "Metal 4: not yet implemented");
    }

} // namespace gpu

// ---------------------------------------------------------------------------
// Apple-only: expose native handles for imgui_impl_metal
// ---------------------------------------------------------------------------

#if defined(__APPLE__)
void* get_native_device(gpu::Device* device) noexcept
{
    auto* d = reinterpret_cast<gpu::DeviceImpl*>(device);
    return (__bridge void*)d->device;
}

void* get_native_command_queue(gpu::Device* device) noexcept
{
    auto* d = reinterpret_cast<gpu::DeviceImpl*>(device);
    return (__bridge void*)d->queue;
}
#endif
