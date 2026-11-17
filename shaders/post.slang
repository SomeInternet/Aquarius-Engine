enum EngineStateFlags : uint32_t {
    IBL = 1 << 0,
    RAY_RECONSTRUCTION = 1 << 1,
    SHARC = 1 << 2,
    SHARC_HEATMAP = 1 << 3,
    GPU_HASH_MAP_HEATMAP = 1 << 4
};

struct PushConstants {
    uint32_t outputImageIdx;
    uint32_t swapRedBlue;
    EngineStateFlags flags;
};

[vk::push_constant]
PushConstants pushConstants;

[[vk::binding(0, 0)]] 
RWTexture2D<float4> radianceImage;

[[vk::binding(1, 0)]]
RWTexture2D<float4> swapchainImages[];

float3 reinhard(float3 color) {
    return color / (color + float3(1));
}

float3 gammaCorrect(float3 color) {
    return pow(color, 1.f / 2.2f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 globalId: SV_DispatchThreadID) { //SV means system value. We specified the dimensions of the workgroups, and in the dispatch command we specified the number of workgroups
    uint2 size;
    radianceImage.GetDimensions(size.x, size.y); //Store image dimensions in x, y

    if (globalId.x < size.x && globalId.y < size.y) {
        float3 color = ((EngineStateFlags::GPU_HASH_MAP_HEATMAP | EngineStateFlags::SHARC_HEATMAP) & pushConstants.flags) ? 
        radianceImage[globalId.xy].rgb :
        gammaCorrect(reinhard(radianceImage[globalId.xy].rgb));

        float4 displayed = float4(color, 1.f);
        if (pushConstants.swapRedBlue != 0) displayed = displayed.bgra;
        swapchainImages[pushConstants.outputImageIdx][globalId.xy] = displayed;
    }
}