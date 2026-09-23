#include "shader_pipeline_shared.h"

[[vk::push_constant]] ConstantBuffer<TextureReadRoot> root;

[shader("compute")]
[numthreads(texture_group_size, texture_group_size, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= pipeline_test_size || thread.y >= pipeline_test_size) return;
    const float2 uv = (float2(thread.xy) + 0.5) / pipeline_test_size;
    root.destination[thread.y * pipeline_test_size + thread.x] = root.texture.SampleLevel(root.sampler, uv, 0);
}
