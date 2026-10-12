// End-to-end shader pipeline test: compute into GPU pointers, buffer and texture copies, texture/sampler handles read by
// compute and fragment shaders, a full-screen draw whose orientation must match on every backend, and line and strip topologies.
#include "shader_pipeline_shared.h"
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace gpu;

static const uint32 fill_count = 256;
static const uint32 pixel_count = pipeline_test_size * pipeline_test_size;

struct ShaderFile
{
    void* data = nullptr;
    size_t size = 0;
};

static ShaderFile read_shader(const char* path) noexcept
{
    ShaderFile file{};
    FILE* stream = fopen(path, "rb");
    if (!stream)
        return file;
    fseek(stream, 0, SEEK_END);
    file.size = static_cast<size_t>(ftell(stream));
    fseek(stream, 0, SEEK_SET);
    file.data = malloc(file.size);
    if (fread(file.data, 1, file.size, stream) != file.size)
        file.size = 0;
    fclose(stream);
    return file;
}

static ShaderCode shader_code(const ShaderFile& file, uint32x3 threads_per_threadgroup = {}) noexcept
{
#if defined(__APPLE__)
    return { .metallib = { file.data, file.size }, .threads_per_threadgroup = threads_per_threadgroup };
#else
    return { .spirv = { static_cast<const uint32*>(file.data), file.size / sizeof(uint32) }, .threads_per_threadgroup = threads_per_threadgroup };
#endif
}

struct Fixture
{
    Device* device = nullptr;
    CommandPool* pool = nullptr;
    TimelineSemaphore* timeline = nullptr;
    uint64 value = 0;
    GpuHeap upload{};
    GpuHeap gpu_memory{};
    GpuHeap readback{};
    GpuHeap texture_descriptors{};
    GpuHeap sampler_descriptors{};
    TextureHeap texture_heap{};
    Texture* source = nullptr;
    Texture* target = nullptr;
    RenderView* target_view = nullptr;
};

static CommandBuffer* begin(Fixture& fixture) noexcept
{
    if (fixture.value)
        wait_timeline({ .semaphore = fixture.timeline, .value = fixture.value });
    reset_command_pool(fixture.pool);
    return begin_commands(fixture.pool);
}

static void submit_and_wait(Fixture& fixture, CommandBuffer* commands) noexcept
{
    end_commands(commands);
    submit(fixture.device, { .commands = { commands }, .completion = { .semaphore = fixture.timeline, .value = ++fixture.value } });
    wait_timeline({ .semaphore = fixture.timeline, .value = fixture.value });
}

static bool check(bool condition, const char* what) noexcept
{
    if (!condition)
        fprintf(stderr, "shader pipeline test failed: %s\n", what);
    return condition;
}

static uint8 pattern(uint32 x, uint32 y, uint32 channel) noexcept
{
    const uint8 values[4]{ static_cast<uint8>(x * 16), static_cast<uint8>(y * 16), static_cast<uint8>((x ^ y) * 16), 255 };
    return values[channel];
}

static bool test_fill_and_copy(Fixture& fixture, PSO* fill) noexcept
{
    uint32* upload = reinterpret_cast<uint32*>(fixture.upload.range.cpu);
    memset(upload, 0, fill_count * sizeof(uint32));
    CommandBuffer* commands = begin(fixture);
    bind_pso(commands, fill);
    const FillRoot root{ .destination = reinterpret_cast<uint32*>(fixture.upload.range.gpu), .count = fill_count, .value = 1000 };
    dispatch(commands, root, { .x = fill_count / fill_group_size, .y = 1, .z = 1 });
    barrier(commands, Stage::compute, Access::shader_write, Stage::transfer, Access::transfer_read);
    copy_memory(commands, { .gpu = fixture.upload.range.gpu, .size = fill_count * sizeof(uint32) }, gpu_range(fixture.gpu_memory));
    barrier(commands, Stage::transfer, Access::transfer_write, Stage::transfer, Access::transfer_read);
    copy_memory(commands, { .gpu = fixture.gpu_memory.range.gpu, .size = fill_count * sizeof(uint32) }, gpu_range(fixture.readback));
    barrier(commands, Stage::transfer, Access::transfer_write, Stage::host, Access::host_read);
    submit_and_wait(fixture, commands);

    const uint32* result = reinterpret_cast<const uint32*>(fixture.readback.range.cpu);
    bool valid = true;
    for (uint32 index = 0; index < fill_count; ++index)
        valid = valid && upload[index] == 1000 + index && result[index] == 1000 + index;
    return check(valid, "compute fill through a GPU pointer, then copy_memory through gpu_only memory");
}

static bool test_texture_handles(Fixture& fixture, PSO* read, uint64 texture, uint64 sampler) noexcept
{
    byte* upload = fixture.upload.range.cpu;
    for (uint32 y = 0; y < pipeline_test_size; ++y)
        for (uint32 x = 0; x < pipeline_test_size; ++x)
            for (uint32 channel = 0; channel < 4; ++channel)
                upload[(y * pipeline_test_size + x) * 4 + channel] = pattern(x, y, channel);

    CommandBuffer* commands = begin(fixture);
    copy_memory_to_texture(commands, gpu_range(fixture.upload), fixture.source);
    barrier(commands, Stage::transfer, Access::transfer_write, Stage::compute | Stage::transfer, Access::shader_read | Access::transfer_read);
    copy_texture_to_memory(commands, fixture.source, { .gpu = fixture.readback.range.gpu, .size = pixel_count * 4 });
    set_texture_descriptor_heap(commands, gpu_range(fixture.texture_descriptors));
    set_sampler_descriptor_heap(commands, gpu_range(fixture.sampler_descriptors));
    bind_pso(commands, read);
    const TextureReadRoot root{
        .texture = texture,
        .sampler = sampler,
        .destination = reinterpret_cast<float4*>(fixture.readback.range.gpu + pixel_count * 4),
    };
    dispatch(commands, root, { .x = pipeline_test_size / texture_group_size, .y = pipeline_test_size / texture_group_size, .z = 1 });
    barrier(commands, Stage::compute | Stage::transfer, Access::shader_write | Access::transfer_write, Stage::host, Access::host_read);
    submit_and_wait(fixture, commands);

    const byte* copied = fixture.readback.range.cpu;
    const float4* sampled = reinterpret_cast<const float4*>(fixture.readback.range.cpu + pixel_count * 4);
    bool copy_valid = memcmp(copied, upload, pixel_count * 4) == 0;
    bool sample_valid = true;
    for (uint32 y = 0; y < pipeline_test_size; ++y)
        for (uint32 x = 0; x < pipeline_test_size; ++x)
            for (uint32 channel = 0; channel < 4; ++channel)
                sample_valid = sample_valid && fabsf(sampled[y * pipeline_test_size + x][channel] - pattern(x, y, channel) / 255.0f) < 1e-3f;
    return check(copy_valid, "copy_memory_to_texture / copy_texture_to_memory round trip") &&
           check(sample_valid, "compute sampling a texture through texture and sampler handles");
}

static bool test_draw(Fixture& fixture, PSO* draw_pso, uint64 texture, uint64 sampler, bool sample_texture) noexcept
{
    CommandBuffer* commands = begin(fixture);
    set_texture_descriptor_heap(commands, gpu_range(fixture.texture_descriptors));
    set_sampler_descriptor_heap(commands, gpu_range(fixture.sampler_descriptors));
    const ColorAttachment color{ .render_view = fixture.target_view, .load = LoadOp::clear, .clear = { .x = 1, .y = 0, .z = 1, .w = 1 } };
    begin_render_pass(commands, { .colors = { color } });
    bind_pso(commands, draw_pso);
    const DrawRoot root{ .texture = texture, .sampler = sampler, .sample_texture = sample_texture ? 1u : 0u };
    draw(commands, root, 3);
    end_render_pass(commands);
    barrier(commands, Stage::color_output, Access::color_write, Stage::transfer, Access::transfer_read);
    copy_texture_to_memory(commands, fixture.target, { .gpu = fixture.readback.range.gpu, .size = pixel_count * 4 });
    barrier(commands, Stage::transfer, Access::transfer_write, Stage::host, Access::host_read);
    submit_and_wait(fixture, commands);

    const byte* pixels = fixture.readback.range.cpu;
    bool valid = true;
    for (uint32 y = 0; y < pipeline_test_size; ++y)
    {
        for (uint32 x = 0; x < pipeline_test_size; ++x)
        {
            const byte* pixel = pixels + (y * pipeline_test_size + x) * 4;
            if (sample_texture)
            {
                for (uint32 channel = 0; channel < 4; ++channel)
                    valid = valid && pixel[channel] == pattern(x, y, channel);
            }
            else
            {
                // uv = (pixel + 0.5) / size with (0, 0) at the top-left, the same on every backend.
                const float expected_r = (x + 0.5f) / pipeline_test_size * 255.0f;
                const float expected_g = (y + 0.5f) / pipeline_test_size * 255.0f;
                valid = valid && fabsf(pixel[0] - expected_r) <= 1.0f && fabsf(pixel[1] - expected_g) <= 1.0f && pixel[2] == 0 && pixel[3] == 255;
            }
        }
    }
    return check(valid, sample_texture ? "fragment shader sampling through handles" : "full-screen draw covers the target with top-left uv origin");
}

// Draws one topology white over black, then checks the pixels the lines cross are lit and the pixels two away from them are not.
// The lines' end pixels are left unchecked: the diamond-exit rule may or may not light them.
static bool test_topology(Fixture& fixture, PSO* pso, Topology topology, uint32 first_point, uint32 point_count, const char* what) noexcept
{
    CommandBuffer* commands = begin(fixture);
    const ColorAttachment color{ .render_view = fixture.target_view, .load = LoadOp::clear, .clear = { .x = 0, .y = 0, .z = 0, .w = 1 } };
    begin_render_pass(commands, { .colors = { color } });
    bind_pso(commands, pso);
    draw(commands, LinesRoot{ .first_point = first_point }, point_count);
    end_render_pass(commands);
    barrier(commands, Stage::color_output, Access::color_write, Stage::transfer, Access::transfer_read);
    copy_texture_to_memory(commands, fixture.target, { .gpu = fixture.readback.range.gpu, .size = pixel_count * 4 });
    barrier(commands, Stage::transfer, Access::transfer_write, Stage::host, Access::host_read);
    submit_and_wait(fixture, commands);

    const byte* pixels = fixture.readback.range.cpu;
    bool valid = true;
    for (uint32 y = 0; y < pipeline_test_size; ++y)
    {
        for (uint32 x = 0; x < pipeline_test_size; ++x)
        {
            const bool lit = pixels[(y * pipeline_test_size + x) * 4] == 255;
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
    return check(valid, what);
}

int main()
{
    Fixture fixture{};
    const DeviceInit init = create_device({});
    if (init.error == Error::unsupported)
    {
        fprintf(stderr, "No supported GPU; skipping.\n");
        return 77;
    }
    if (init.error != Error::none)
        return 1;
    fixture.device = init.device;

    const ShaderFile fill_file = read_shader(NOGRAPHICSAPI_PIPELINE_FILL_SHADER);
    const ShaderFile read_file = read_shader(NOGRAPHICSAPI_PIPELINE_TEXTURE_SHADER);
    const ShaderFile vertex_file = read_shader(NOGRAPHICSAPI_PIPELINE_DRAW_VERTEX_SHADER);
    const ShaderFile fragment_file = read_shader(NOGRAPHICSAPI_PIPELINE_DRAW_FRAGMENT_SHADER);
    const ShaderFile lines_vertex_file = read_shader(NOGRAPHICSAPI_PIPELINE_LINES_VERTEX_SHADER);
    const ShaderFile lines_fragment_file = read_shader(NOGRAPHICSAPI_PIPELINE_LINES_FRAGMENT_SHADER);
    if (!fill_file.size || !read_file.size || !vertex_file.size || !fragment_file.size || !lines_vertex_file.size || !lines_fragment_file.size)
    {
        fprintf(stderr, "Cannot read shader pipeline test shaders.\n");
        return 1;
    }

    const DeviceCaps& caps = get_device_caps(fixture.device);
    fixture.pool = create_command_pool(fixture.device);
    fixture.timeline = create_timeline_semaphore(fixture.device);
    fixture.upload = create_gpu_heap(fixture.device, 64 * 1024);
    fixture.gpu_memory = create_gpu_heap(fixture.device, 64 * 1024, MemoryType::gpu_only);
    fixture.readback = create_gpu_heap(fixture.device, 64 * 1024, MemoryType::readback);
    fixture.texture_descriptors = create_gpu_heap(fixture.device, 4 * caps.texture_descriptor_size, MemoryType::texture_descriptor_heap);
    fixture.sampler_descriptors = create_gpu_heap(fixture.device, 4 * caps.sampler_descriptor_size, MemoryType::sampler_descriptor_heap);

    const TextureDesc source_desc{
        .extent = { .x = pipeline_test_size, .y = pipeline_test_size, .z = 1 },
        .usage = TextureUsage::sampled | TextureUsage::transfer_source | TextureUsage::transfer_destination,
    };
    const TextureDesc target_desc{
        .extent = { .x = pipeline_test_size, .y = pipeline_test_size, .z = 1 },
        .usage = TextureUsage::color_attachment | TextureUsage::transfer_source,
    };
    const SizeAlign source_placement = get_texture_size_align(fixture.device, source_desc);
    const SizeAlign target_placement = get_texture_size_align(fixture.device, target_desc);
    const uint64 target_offset = (source_placement.size + target_placement.align - 1) / target_placement.align * target_placement.align;
    fixture.texture_heap = create_texture_heap(fixture.device, target_offset + target_placement.size);

    CommandBuffer* commands = begin(fixture);
    fixture.source = create_texture(commands, source_desc, fixture.texture_heap, 0);
    fixture.target = create_texture(commands, target_desc, fixture.texture_heap, target_offset);
    submit_and_wait(fixture, commands);
    fixture.target_view = create_render_view(fixture.target);

    // Slot 1, not 0, so handles are proven to carry the slot.
    write_texture_descriptor(
        fixture.device,
        fixture.texture_descriptors.range.cpu + caps.texture_descriptor_size,
        fixture.source,
        TextureDescriptorType::sampled
    );
    write_sampler_descriptor(
        fixture.device,
        fixture.sampler_descriptors.range.cpu + caps.sampler_descriptor_size,
        { .min_filter = Filter::nearest,
          .mag_filter = Filter::nearest,
          .mip_filter = Filter::nearest,
          .address_u = AddressMode::clamp_to_edge,
          .address_v = AddressMode::clamp_to_edge }
    );
    const uint64 texture = get_texture_handle(fixture.texture_descriptors, 1);
    const uint64 sampler = get_sampler_handle(fixture.sampler_descriptors, 1);

    PSO* fill = create_compute_pso(fixture.device, shader_code(fill_file, { .x = fill_group_size, .y = 1, .z = 1 }));
    PSO* read = create_compute_pso(fixture.device, shader_code(read_file, { .x = texture_group_size, .y = texture_group_size, .z = 1 }));
    const ColorTargetDesc color_target{ .format = Format::rgba8_unorm };
    PSO* draw_pso = create_graphics_pso(
        fixture.device,
        {
            .vertex = shader_code(vertex_file),
            .fragment = shader_code(fragment_file),
            .color_targets = { color_target },
        }
    );

    PSO* line_list = create_graphics_pso(fixture.device, { .vertex = shader_code(lines_vertex_file), .fragment = shader_code(lines_fragment_file),
                                                           .color_targets = { color_target }, .topology = Topology::line_list });
    PSO* line_strip = create_graphics_pso(fixture.device, { .vertex = shader_code(lines_vertex_file), .fragment = shader_code(lines_fragment_file),
                                                            .color_targets = { color_target }, .topology = Topology::line_strip });
    PSO* triangle_strip = create_graphics_pso(fixture.device, { .vertex = shader_code(lines_vertex_file), .fragment = shader_code(lines_fragment_file),
                                                                .color_targets = { color_target }, .topology = Topology::triangle_strip });

    bool passed = test_fill_and_copy(fixture, fill);
    passed = test_texture_handles(fixture, read, texture, sampler) && passed;
    passed = test_draw(fixture, draw_pso, texture, sampler, false) && passed;
    passed = test_draw(fixture, draw_pso, texture, sampler, true) && passed;
    passed = test_topology(fixture, line_list, Topology::line_list, 0, 2, "a line list draws a one-pixel line") && passed;
    passed = test_topology(fixture, line_strip, Topology::line_strip, 2, 3, "a line strip joins its points") && passed;
    passed = test_topology(fixture, triangle_strip, Topology::triangle_strip, 5, 4, "a triangle strip's two triangles cover the target") && passed;

    wait_idle(fixture.device);
    destroy_pso(triangle_strip);
    destroy_pso(line_strip);
    destroy_pso(line_list);
    destroy_pso(draw_pso);
    destroy_pso(read);
    destroy_pso(fill);
    destroy_render_view(fixture.target_view);
    destroy_texture(fixture.target);
    destroy_texture(fixture.source);
    destroy_texture_heap(fixture.texture_heap);
    destroy_gpu_heap(fixture.sampler_descriptors);
    destroy_gpu_heap(fixture.texture_descriptors);
    destroy_gpu_heap(fixture.readback);
    destroy_gpu_heap(fixture.gpu_memory);
    destroy_gpu_heap(fixture.upload);
    destroy_timeline_semaphore(fixture.timeline);
    destroy_command_pool(fixture.pool);
    destroy_device(fixture.device);
    free(lines_fragment_file.data);
    free(lines_vertex_file.data);
    free(fragment_file.data);
    free(vertex_file.data);
    free(read_file.data);
    free(fill_file.data);
    if (passed)
        printf("Shader pipeline test passed.\n");
    return passed ? 0 : 1;
}
