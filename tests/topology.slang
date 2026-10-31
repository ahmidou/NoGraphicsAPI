#include <NoGraphicsAPI/shader.slang>
#include "topology_shared.h"

GPU_ROOT(TopologyRoot, root);

// Points in target pixels, (0, 0) at the top-left: a line list's line, a line strip's two lines, a triangle strip's quad.
static const float2 points[] = {
    float2(0, 4.5), float2(16, 4.5),
    float2(0, 4.5), float2(10.5, 4.5), float2(10.5, 16),
    float2(0, 0), float2(16, 0), float2(0, 16), float2(16, 16),
};

[shader("vertex")]
float4 vertexMain(uint vertex_index : SV_VertexID) : SV_Position
{
    const float2 point = points[root.first_point + vertex_index] / (topology_test_size * 0.5f) - 1.0f;
    return float4(point, 0.0f, 1.0f);
}

[shader("fragment")]
float4 fragmentMain() : SV_Target0
{
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
}
