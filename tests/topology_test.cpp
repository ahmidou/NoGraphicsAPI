// GraphicsPSODesc::topology: a line list, a line strip and a triangle strip, drawn white over black and read back.
#include "shader_code.h"
#include "topology_shared.h"
#include <stdlib.h>

using namespace gpu;

namespace {
    struct Fixture
    {
        Device* device = nullptr;
        CommandPool* pool = nullptr;
        TimelineSemaphore* timeline = nullptr;
        Texture* target = nullptr;
        RenderView* target_view = nullptr;
        GpuHeap roots = {};
        GpuHeap readback = {};
        uint64 submitted = 0;
    };

    constexpr uint32 pixel_count = topology_test_size * topology_test_size;

    void submit_and_wait(Fixture& fixture, CommandBuffer* commands)
    {
        end_commands(commands);
        submit(fixture.device, { .commands = { commands }, .completion = { .semaphore = fixture.timeline, .value = ++fixture.submitted } });
        wait_timeline({ .semaphore = fixture.timeline, .value = fixture.submitted });
    }

    // Checks the pixels the lines cross are lit and the pixels two away from them are not. The lines' end pixels are left
    // unchecked: the diamond-exit rule may or may not light them.
    bool draw_and_check(Fixture& fixture, PSO* pso, Topology topology, uint32 first_point, uint32 point_count, const char* what)
    {
        reset_command_pool(fixture.pool);
        CommandBuffer* commands = begin_commands(fixture.pool);
        *reinterpret_cast<TopologyRoot*>(fixture.roots.range.cpu) = { .first_point = first_point };
        const ColorAttachment color{ .render_view = fixture.target_view, .load = LoadOp::clear, .clear = { .x = 0, .y = 0, .z = 0, .w = 1 } };
        begin_render_pass(commands, { .colors = { color } });
        bind_pso(commands, pso);
        draw(commands, fixture.roots.range.gpu, point_count);
        end_render_pass(commands);
        barrier(commands, Stage::color_output, Access::color_write, Stage::transfer, Access::transfer_read);
        copy_texture_to_memory(commands, fixture.target, { .gpu = fixture.readback.range.gpu, .size = pixel_count * 4 });
        barrier(commands, Stage::transfer, Access::transfer_write, Stage::host, Access::host_read);
        submit_and_wait(fixture, commands);

        const byte* pixels = fixture.readback.range.cpu;
        bool valid = true;
        for (uint32 y = 0; y < topology_test_size; ++y)
        {
            for (uint32 x = 0; x < topology_test_size; ++x)
            {
                const bool lit = pixels[(y * topology_test_size + x) * 4] == 255;
                const int32 row_distance = abs(static_cast<int32>(y) - 4);
                const int32 column_distance = abs(static_cast<int32>(x) - 10);
                if (topology == Topology::triangle_strip)
                    valid = valid && lit;
                else if (topology == Topology::line_list)
                    valid = valid && (row_distance == 0 && x >= 1 && x <= 14 ? lit : row_distance >= 2 ? !lit : true);
                else if ((row_distance == 0 && x >= 1 && x <= 9) || (column_distance == 0 && y >= 6 && y <= 14))
                    valid = valid && lit;
                else if ((row_distance >= 2 || x >= 12) && (column_distance >= 2 || y <= 2))
                    valid = valid && !lit;
            }
        }
        if (!valid)
            fprintf(stderr, "Topology test failed: %s\n", what);
        return valid;
    }
} // namespace

int main()
{
    const DeviceInit init = create_device();
    if (init.error == Error::unsupported)
        return 77;
    if (init.error != Error::none)
        return 1;
    Fixture fixture{ .device = init.device };
    Span<byte> vertex = load_test_shader(NOGRAPHICSAPI_TOPOLOGY_VERTEX_SHADER);
    Span<byte> fragment = load_test_shader(NOGRAPHICSAPI_TOPOLOGY_FRAGMENT_SHADER);
    if (!vertex.data || !fragment.data)
        return 1;
    fixture.pool = create_command_pool(fixture.device);
    fixture.timeline = create_timeline_semaphore(fixture.device);
    fixture.roots = create_gpu_heap(fixture.device, sizeof(TopologyRoot));
    fixture.readback = create_gpu_heap(fixture.device, pixel_count * 4, MemoryType::readback);
    const TextureDesc desc{ .extent = { .x = topology_test_size, .y = topology_test_size, .z = 1 },
                            .usage = TextureUsage::color_attachment | TextureUsage::transfer_source };
    const TextureHeap heap = create_texture_heap(fixture.device, get_texture_size_align(fixture.device, desc).size);
    fixture.target = create_texture(fixture.device, desc, heap, 0);
    fixture.target_view = create_render_view(fixture.target);
    CommandBuffer* commands = begin_commands(fixture.pool);
    initialize_texture(commands, fixture.target);
    submit_and_wait(fixture, commands);

    const ColorTargetDesc color_target{ .format = desc.format };
    const Topology topologies[3]{ Topology::line_list, Topology::line_strip, Topology::triangle_strip };
    PSO* psos[3]{};
    for (uint32 i = 0; i < 3; ++i)
        psos[i] = create_graphics_pso(
            fixture.device,
            { .vertex = { .code = { vertex.data, vertex.size }, .entry_point = "vertexMain" },
              .fragment = { .code = { fragment.data, fragment.size }, .entry_point = "fragmentMain" },
              .color_targets = { color_target },
              .topology = topologies[i] }
        );
    bool passed = psos[0] && psos[1] && psos[2];
    passed = passed && draw_and_check(fixture, psos[0], Topology::line_list, 0, 2, "a line list draws a one-pixel line");
    passed = passed && draw_and_check(fixture, psos[1], Topology::line_strip, 2, 3, "a line strip joins its points");
    passed = passed && draw_and_check(fixture, psos[2], Topology::triangle_strip, 5, 4, "a triangle strip's two triangles cover the target");

    wait_idle(fixture.device);
    for (PSO* pso : psos)
        destroy_pso(pso);
    destroy_render_view(fixture.target_view);
    destroy_texture(fixture.target);
    destroy_texture_heap(heap);
    destroy_gpu_heap(fixture.readback);
    destroy_gpu_heap(fixture.roots);
    destroy_timeline_semaphore(fixture.timeline);
    destroy_command_pool(fixture.pool);
    destroy_device(fixture.device);
    free(fragment.data);
    free(vertex.data);
    if (passed)
        printf("Topology test passed.\n");
    return passed ? 0 : 1;
}
