// Native-handle interop on Metal: NoGraphicsAPI clears a texture and signals its timeline; a separate raw MTL4CommandQueue, as a UI
// renderer would own, waits on the MTLSharedEvent, then samples and blits that texture into a readback buffer.
// Modes pick how the foreign queue makes NoGraphicsAPI memory resident: the device residency set (default), the texture alone in the
// foreign queue's own set (what imgui_impl_metal4 does for user textures), or nothing (--no-residency, which GPU shader validation rejects).
// --same-queue submits the raw command buffer on NoGraphicsAPI's own queue (get_native_queue) instead: no event wait, and the device
// residency set is already there; a consumer queue barrier in the raw encoder orders its reads after the clear, as barrier() would.
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>

#import <Metal/Metal.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace gpu;

static const uint32 size = 16;
static const ClearColor clear_color{ .x = 0.25f, .y = 0.5f, .z = 0.75f, .w = 1.0f };

static const char* kernel_source = R"(
#include <metal_stdlib>
using namespace metal;
kernel void sample_texture(texture2d<float> source [[texture(0)]], device float4* destination [[buffer(0)]], uint2 id [[thread_position_in_grid]])
{
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    destination[id.y * 16 + id.x] = source.sample(nearest, (float2(id) + 0.5) / 16.0, level(0));
}
)";

static bool check(bool condition, const char* what) noexcept
{
    if (!condition)
        fprintf(stderr, "native interop test failed: %s\n", what);
    return condition;
}

int main(int argc, char** argv)
{
    const bool texture_residency = argc > 1 && strcmp(argv[1], "--texture-residency") == 0;
    const bool no_residency = argc > 1 && strcmp(argv[1], "--no-residency") == 0;
    const bool same_queue = argc > 1 && strcmp(argv[1], "--same-queue") == 0;
    const DeviceInit init = create_device({});
    if (init.error == Error::unsupported)
    {
        fprintf(stderr, "No supported GPU; skipping.\n");
        return 77;
    }
    if (init.error != Error::none)
        return 1;
    Device* device = init.device;
    CommandPool* pool = create_command_pool(device);
    TimelineSemaphore* timeline = create_timeline_semaphore(device);

    const TextureDesc desc{
        .extent = { .x = size, .y = size, .z = 1 },
        .usage = TextureUsage::sampled | TextureUsage::color_attachment | TextureUsage::transfer_source,
    };
    const TextureHeap heap = create_texture_heap(device, get_texture_size_align(device, desc).size);
    Texture* texture = create_texture(device, desc, heap, 0);
    CommandBuffer* commands = begin_commands(pool);
    initialize_texture(commands, texture);
    end_commands(commands);
    submit(device, { .commands = { commands }, .completion = { .semaphore = timeline, .value = 1 } });
    wait_timeline({ .semaphore = timeline, .value = 1 });
    RenderView* view = create_render_view(texture);

    id<MTLTexture> native_texture = (__bridge id<MTLTexture>)get_native_texture(texture);
    id<MTLSharedEvent> native_event = (__bridge id<MTLSharedEvent>)get_native_timeline_semaphore(timeline);
    id<MTLResidencySet> native_residency = (__bridge id<MTLResidencySet>)get_native_residency_set(device);
    bool passed = check(native_texture && native_event && native_residency, "native getters return non-null objects");
    passed = check([native_texture conformsToProtocol:@protocol(MTLTexture)] && native_texture.width == size && native_texture.height == size &&
                       native_texture.pixelFormat == MTLPixelFormatRGBA8Unorm && native_texture.heap != nil,
                   "the native texture is the placed MTLTexture") && passed;
    passed = check([native_event conformsToProtocol:@protocol(MTLSharedEvent)] && native_event.signaledValue == 1, "the native semaphore is the MTLSharedEvent") &&
             passed;
    // Under GPU shader validation the backend makes placed textures resident one by one instead of through their heap.
    passed = check([native_residency conformsToProtocol:@protocol(MTLResidencySet)] &&
                       ([native_residency containsAllocation:native_texture.heap] || [native_residency containsAllocation:native_texture]),
                   "the native residency set holds the texture or its heap") && passed;

    id<MTL4CommandQueue> native_queue = (__bridge id<MTL4CommandQueue>)get_native_queue(device);
    passed = check([native_queue conformsToProtocol:@protocol(MTL4CommandQueue)] && native_queue.device == native_texture.device,
                   "the native queue is the device's MTL4CommandQueue") && passed;

    // Clear on NoGraphicsAPI's queue and signal 2 without a CPU wait: only the foreign queue's GPU-side wait (or, on the same queue,
    // the consumer barrier) orders the read.
    reset_command_pool(pool);
    commands = begin_commands(pool);
    const ColorAttachment color{ .render_view = view, .load = LoadOp::clear, .clear = clear_color };
    begin_render_pass(commands, { .colors = { color } });
    end_render_pass(commands);
    end_commands(commands);
    submit(device, { .commands = { commands }, .completion = { .semaphore = timeline, .value = 2 } });

    id<MTLDevice> mtl_device = native_texture.device;
    NSError* error = nil;
    id<MTL4CommandQueue> queue = same_queue ? native_queue : [mtl_device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [mtl_device newCommandAllocator];
    id<MTL4CommandBuffer> command_buffer = [mtl_device newCommandBuffer];
    id<MTLBuffer> readback = [mtl_device newBufferWithLength:size * size * (sizeof(float) * 4 + 4) options:MTLResourceStorageModeShared];
    id<MTLResidencySet> residency = [mtl_device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    id<MTLLibrary> library = [mtl_device newLibraryWithSource:@(kernel_source) options:nil error:&error];
    id<MTL4Compiler> compiler = [mtl_device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"sample_texture";
    MTL4ComputePipelineDescriptor* pipeline_desc = [MTL4ComputePipelineDescriptor new];
    pipeline_desc.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = library ? [compiler newComputePipelineStateWithDescriptor:pipeline_desc compilerTaskOptions:nil error:&error] : nil;
    MTL4ArgumentTableDescriptor* table_desc = [MTL4ArgumentTableDescriptor new];
    table_desc.maxBufferBindCount = 1;
    table_desc.maxTextureBindCount = 1;
    id<MTL4ArgumentTable> table = [mtl_device newArgumentTableWithDescriptor:table_desc error:&error];
    if (!queue || !allocator || !command_buffer || !readback || !residency || !pipeline || !table)
    {
        fprintf(stderr, "native interop test setup failed: %s\n", error ? error.localizedDescription.UTF8String : "");
        return 1;
    }
    [residency addAllocation:readback];
    if (texture_residency)
        [residency addAllocation:native_texture];
    [residency commit];
    [queue addResidencySet:residency];
    const bool add_native_residency = !texture_residency && !no_residency && !same_queue;
    if (add_native_residency)
        [queue addResidencySet:native_residency];

    [command_buffer beginCommandBufferWithAllocator:allocator];
    id<MTL4ComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    // The stages NoGraphicsAPI writes textures in (render passes, dispatches, copies) before this encoder's reads, caches flushed: what
    // barrier(color_output | compute | transfer -> compute | transfer) records. Queue barriers reach earlier command buffers on the queue.
    if (same_queue)
        [encoder barrierAfterQueueStages:MTLStageFragment | MTLStageDispatch | MTLStageBlit
                            beforeStages:MTLStageDispatch | MTLStageBlit
                       visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder setComputePipelineState:pipeline];
    [table setTexture:native_texture.gpuResourceID atIndex:0];
    [table setAddress:readback.gpuAddress atIndex:0];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(size, size, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [encoder copyFromTexture:native_texture
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(size, size, 1)
                    toBuffer:readback
           destinationOffset:size * size * sizeof(float) * 4
      destinationBytesPerRow:size * 4
    destinationBytesPerImage:size * size * 4];
    [encoder endEncoding];
    [command_buffer endCommandBuffer];
    if (!same_queue)
        [queue waitForEvent:native_event value:2];
    __unsafe_unretained id<MTL4CommandBuffer> batch[1]{ command_buffer };
    // The API validation layer alone misses a missing residency (the texture happens to be resident through NoGraphicsAPI's queue).
    // GPU shader validation catches it but only zeroes the read; MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 prints the report.
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    __block NSError* gpu_error = nil;
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        gpu_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    [queue commit:batch count:1 options:options];
    dispatch_semaphore_wait(completed, DISPATCH_TIME_FOREVER);

    const float* sampled = static_cast<const float*>(readback.contents);
    const uint8* copied = static_cast<const uint8*>(readback.contents) + size * size * sizeof(float) * 4;
    const uint8 expected[4]{ 64, 128, 191, 255 };
    bool sample_valid = true;
    bool copy_valid = true;
    for (uint32 pixel = 0; pixel < size * size; ++pixel)
    {
        sample_valid = sample_valid && fabsf(sampled[pixel * 4 + 0] - clear_color.x) < 1e-2f && fabsf(sampled[pixel * 4 + 1] - clear_color.y) < 1e-2f &&
                       fabsf(sampled[pixel * 4 + 2] - clear_color.z) < 1e-2f && fabsf(sampled[pixel * 4 + 3] - clear_color.w) < 1e-2f;
        for (uint32 channel = 0; channel < 4; ++channel)
            copy_valid = copy_valid && abs(copied[pixel * 4 + channel] - expected[channel]) <= 1;
    }
    if (gpu_error)
        fprintf(stderr, "raw command buffer error: %s\n", gpu_error.description.UTF8String);
    passed = check(!gpu_error, "the raw command buffer completes without errors") && passed;
    if (!sample_valid)
        fprintf(stderr, "sampled texel 0: %g %g %g %g\n", sampled[0], sampled[1], sampled[2], sampled[3]);
    passed = check(sample_valid, "raw commands sample the cleared texture after the event wait or queue barrier") && passed;
    passed = check(copy_valid, "raw commands blit the cleared texture after the event wait or queue barrier") && passed;

    [queue removeResidencySet:residency];
    if (add_native_residency)
        [queue removeResidencySet:native_residency];
    wait_idle(device);
    destroy_render_view(view);
    destroy_texture(texture);
    destroy_texture_heap(heap);
    destroy_timeline_semaphore(timeline);
    destroy_command_pool(pool);
    destroy_device(device);
    if (passed)
        printf("Native interop test passed.\n");
    return passed ? 0 : 1;
}
