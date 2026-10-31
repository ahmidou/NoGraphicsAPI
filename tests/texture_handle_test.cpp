// get_texture_handle: a compute shader reads a cleared texture through a Slang `.Handle` in its root.
#include "shader_code.h"
#include <math.h>

using namespace gpu;

struct TextureHandleRoot
{
    uint64 source;
    float* destination;
};
static_assert(sizeof(TextureHandleRoot) == 16);

int main()
{
    const DeviceInit init = create_device();
    if (init.error == Error::unsupported)
        return 77;
    if (init.error != Error::none)
        return 1;
    Device* device = init.device;
    Span<byte> code = load_test_shader(NOGRAPHICSAPI_TEXTURE_HANDLE_SHADER);
    if (!code.data)
        return 1;
    PSO* pso = create_compute_pso(device, { .code = { code.data, code.size }, .entry_point = "computeMain" });
    free(code.data);
    if (!pso)
        return 1;
    CommandPool* pool = create_command_pool(device);
    TimelineSemaphore* timeline = create_timeline_semaphore(device);
    const TextureDesc desc{ .extent = { .x = 1, .y = 1, .z = 1 }, .usage = TextureUsage::sampled | TextureUsage::color_attachment };
    const TextureHeap heap = create_texture_heap(device, get_texture_size_align(device, desc).size);
    Texture* texture = create_texture(device, desc, heap, 0);
    RenderView* view = create_render_view(texture);
    TextureDescriptorHeap* descriptors = create_texture_descriptor_heap(device, 8);
    write_texture_descriptor(descriptors, 5, texture, TextureDescriptorType::sampled);
    const GpuHeap memory = create_gpu_heap(device, 64, MemoryType::readback);
    *reinterpret_cast<TextureHandleRoot*>(memory.range.cpu) = { .source = get_texture_handle(descriptors, 5),
                                                                .destination = reinterpret_cast<float*>(memory.range.gpu + 16) };

    CommandBuffer* commands = begin_commands(pool);
    initialize_texture(commands, texture);
    const ColorAttachment color{ .render_view = view, .load = LoadOp::clear, .clear = { .x = 0.25f, .y = 0.5f, .z = 0.75f, .w = 1.0f } };
    begin_render_pass(commands, { .colors = { color } });
    end_render_pass(commands);
    barrier(commands, Stage::color_output, Access::color_write, Stage::compute, Access::shader_read);
    set_texture_descriptor_heap(commands, descriptors);
    bind_pso(commands, pso);
    dispatch(commands, memory.range.gpu, { .x = 1, .y = 1, .z = 1 });
    barrier(commands, Stage::compute, Access::shader_write, Stage::host, Access::host_read);
    end_commands(commands);
    submit(device, { .commands = { commands }, .completion = { .semaphore = timeline, .value = 1 } });
    wait_timeline({ .semaphore = timeline, .value = 1 });

    const float* texel = reinterpret_cast<const float*>(memory.range.cpu + 16);
    const bool passed = fabsf(texel[0] - 0.25f) < 1e-2f && fabsf(texel[1] - 0.5f) < 1e-2f && fabsf(texel[2] - 0.75f) < 1e-2f && fabsf(texel[3] - 1.0f) < 1e-2f;
    if (!passed)
        fprintf(stderr, "Texture handle test failed: texel %g %g %g %g\n", texel[0], texel[1], texel[2], texel[3]);

    wait_idle(device);
    destroy_gpu_heap(memory);
    destroy_texture_descriptor_heap(descriptors);
    destroy_render_view(view);
    destroy_texture(texture);
    destroy_texture_heap(heap);
    destroy_timeline_semaphore(timeline);
    destroy_command_pool(pool);
    destroy_pso(pso);
    destroy_device(device);
    if (passed)
        printf("Texture handle test passed.\n");
    return passed ? 0 : 1;
}
