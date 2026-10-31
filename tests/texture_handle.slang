#include <NoGraphicsAPI/shader.slang>
#include <NoGraphicsAPIUtility/shader_types.h>

// texture_handle_test.cpp's root: a texture by its get_texture_handle() value, and where its first texel goes.
struct TextureHandleRoot
{
    Texture2D<float4>.Handle source;
    float4* destination;
};

GPU_ROOT(TextureHandleRoot, root);

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    *root.destination = root.source.Load(int3(0, 0, 0));
}
