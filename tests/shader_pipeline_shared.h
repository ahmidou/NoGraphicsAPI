#pragma once

#include <NoGraphicsAPIUtility/shader_types.h>

// Handles are 64-bit values written by the CPU from gpu::get_texture_handle / get_sampler_handle.
#if defined(__SLANG__)
typedef Texture2D<float4>.Handle TextureHandle;
typedef SamplerState.Handle SamplerHandle;
#else
typedef uint64 TextureHandle;
typedef uint64 SamplerHandle;
#endif

static const uint32 pipeline_test_size = 16;
static const uint32 fill_group_size = 64;
static const uint32 texture_group_size = 8;

struct FillRoot
{
    uint32* destination;
    uint32 count;
    uint32 value;
};

struct TextureReadRoot
{
    TextureHandle texture;
    SamplerHandle sampler;
    float4* destination;
};

struct DrawRoot
{
    TextureHandle texture;
    SamplerHandle sampler;
    uint32 sample_texture;
};

struct LinesRoot
{
    uint32 first_point;
};
