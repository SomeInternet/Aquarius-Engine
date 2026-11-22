
#include "util.slang"

struct RasterPushConstants {
    float4x4 viewProj;
    float4x4 model; // Might not need this
    float4x4 prevModel; //For motion vectors?
    Material *materials;

    float4 *posBuffer;
    Vertex *vertBuffer;

    float3 camPos;
    int materialIdx;
    int instanceIdx;
};

struct GBufferFSInput {
    float4 fragPos : SV_Position;
    float3 worldPos;
    float3 nor;
    float3 tan;
    half2 uv;
};

//NOTE: If anyone changes this, make sure to update the color attachment info in the raster pipeline setup!
struct GBufferFSOutput {
    [[vk::location(0)]] float4 albedoMetallic : SV_Target0;
    [[vk::location(1)]] float3 pos : SV_Target1; //Add IOR/Transmissiveness to this and make it a float4?
    [[vk::location(2)]] float4 normalRoughness : SV_Target2;
    [[vk::location(3)]] float2 motionVectors : SV_Target3;
    [[vk::location(4)]] float depth : SV_Target4;
};

//GBuffer albedo image
[[vk::binding(0, 0)]]
RWTexture2D<float4> engineImages[];

[[vk::binding(1, 0)]]
Texture2D<float4> sceneImages[];

[[vk::binding(2, 0)]]
SamplerState samplers[];

[[vk::push_constant]]
RasterPushConstants pc;

//diffuse albedo, specular albedo, specular hit dist to be written out in reconstruction pass
//Depth already written out in the depth prepass

[shader("vertex")]
GBufferFSInput vertMain(uint32_t idx: SV_VertexID) {
    // Do the usual projection
    GBufferFSInput output = {};
    output.worldPos = pc.posBuffer[idx].xyz;
    output.fragPos = mul(pc.viewProj, mul(pc.model, float4(output.worldPos, 1.f)));
    output.nor = fromOctahedral(pc.vertBuffer[idx].nor);
    output.tan = fromOctahedral(pc.vertBuffer[idx].tan);
    output.uv = pc.vertBuffer[idx].uv;
    return output;
}

inline float4 sampleTex(int imgIdx, int sampIdx, half2 uv, float4 defaultValue) {
    return (imgIdx != -1) ? sceneImages[NonUniformResourceIndex(imgIdx)].SampleLevel(samplers[NonUniformResourceIndex(sampIdx + 1)], uv, 0.f) : defaultValue;
}

[earlydepthstencil]
[shader("fragment")]
GBufferFSOutput fragMain(GBufferFSInput input) {
    GBufferFSOutput output = {};

    Material mat = pc.materials[pc.materialIdx];

    output.albedoMetallic.xyz = mat.color.xyz * sampleTex(mat.colorImgIdx, mat.colorSamplerIdx, input.uv, float4(1.f)).xyz;

    float2 metallicRoughness = mat.metallicRoughness * sampleTex(mat.metallicRoughnessImgIdx, mat.metallicRoughnessSamplerIdx, input.uv, float4(1.f)).xy;
    output.albedoMetallic.w = metallicRoughness.x;
    output.normalRoughness.w = metallicRoughness.y;

    output.normalRoughness.xyz = input.nor;
    if (mat.norImgIdx != -1) {
        float3 mNor = sampleTex(mat.norImgIdx, mat.norSamplerIdx, input.uv, float4(0.f)).xyz;

        PointInfo pointInfo = {};
        pointInfo.nor = input.nor;
        pointInfo.tan = input.tan;
        applyNormalMap(pointInfo, mNor);

        output.normalRoughness.xyz = pointInfo.nor;
    }

    float4 clipSpacePrevPos = mul(pc.viewProj, mul(pc.prevModel, float4(input.worldPos, 1.f)));
    output.motionVectors = (input.fragPos - clipSpacePrevPos / clipSpacePrevPos.w).xy;

    output.pos = mul(pc.model, float4(input.worldPos, 1.f)).xyz;
    output.depth = length(output.pos - pc.camPos);

    return output;

    //Compute albedo, write out
    //Compute normals, write out
    //Write out world position
    //Write out motion vectors...?

    //Compute pbr parameters, write out
    //TODO: Maybe extract the material parameter finding logic into a helper function? I could use that here...

    //Compute motion vectors
}