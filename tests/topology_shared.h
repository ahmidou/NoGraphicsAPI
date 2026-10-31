#pragma once
#include <NoGraphicsAPIUtility/shader_types.h>

static const uint32 topology_test_size = 16;

struct TopologyRoot
{
    uint32 first_point;
    uint32 padding[3];
};

#if !defined(__SLANG__)
static_assert(sizeof(TopologyRoot) == 16);
#endif
