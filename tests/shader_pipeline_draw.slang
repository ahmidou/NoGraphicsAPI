#include "shader_pipeline_shared.h"

[[vk::push_constant]] ConstantBuffer<DrawRoot> root;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// One triangle covering the target. Vulkan clip space: uv (0, 0) is the top-left pixel.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VertexID)
{
    VertexOutput output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * 2 - 1, 0, 1);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target0
{
    if (root.sample_texture != 0) return root.texture.SampleLevel(root.sampler, input.uv, 0);
    return float4(input.uv, 0, 1);
}
