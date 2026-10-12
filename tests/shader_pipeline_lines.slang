#include "shader_pipeline_shared.h"

[[vk::push_constant]] ConstantBuffer<LinesRoot> root;

// Points in target pixels, (0, 0) at the top-left: a line list's line, a line strip's two lines, a triangle strip's quad.
static const float2 points[] = {
    float2(0, 4.5), float2(16, 4.5),
    float2(0, 4.5), float2(10.5, 4.5), float2(10.5, 16),
    float2(0, 0), float2(16, 0), float2(0, 16), float2(16, 16),
};

[shader("vertex")]
float4 vertexMain(uint vertex_id : SV_VertexID) : SV_Position
{
    return float4(points[root.first_point + vertex_id] / (pipeline_test_size * 0.5) - 1, 0, 1);
}

[shader("fragment")]
float4 fragmentMain() : SV_Target0
{
    return float4(1, 1, 1, 1);
}
