#include "shader_pipeline_shared.h"

[[vk::push_constant]] ConstantBuffer<FillRoot> root;

[shader("compute")]
[numthreads(fill_group_size, 1, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x < root.count) root.destination[thread.x] = root.value + thread.x;
}
