
#include "util.slang"

struct RasterPushConstants {
    CameraData *cam;
    Material *materials;
    VkInstance *instances;
    VkInstance *prevInstances;

    float4 *posBuffer;
    Vertex *vertBuffer;

    int materialIdx;
    int instanceIdx;
};

struct GBufferFSInput {
    float4 fragPos : SV_Position;
    float3 worldPos;
    float3 nor;
    float3 tan;
    half2 uv;
    float4 currClip;
    float4 prevClip;
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
    VkInstance inst = pc.instances[pc.instanceIdx];
    float3 objPos = pc.posBuffer[idx].xyz;
    Vertex v = pc.vertBuffer[idx];

    output.fragPos = instanceToClip(pc.cam->currViewProj, inst, objPos, output.worldPos);
    output.currClip = output.fragPos;
    output.prevClip = mul(pc.cam->prevViewProj, float4(instanceTransformPoint(pc.prevInstances[pc.instanceIdx], objPos), 1.f));

    output.nor = instanceTransformNormal(inst, fromOctahedral(v.nor));
    output.tan = instanceTransformDir(inst, fromOctahedral(v.tan));
    output.uv = v.uv;
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

    float3 nor = normalize(input.nor);
    output.normalRoughness.xyz = nor;
    if (mat.norImgIdx != -1) {
        float3 mNor = sampleTex(mat.norImgIdx, mat.norSamplerIdx, input.uv, float4(0.f)).xyz;

        float3 tan = input.tan - nor * dot(nor, input.tan);
        float tanLen = length(tan);

        PointInfo pointInfo = {};
        pointInfo.nor = nor;
        pointInfo.tan = tanLen > 1e-6f ? tan / tanLen : fallbackTangent(nor);
        applyNormalMap(pointInfo, mNor);

        output.normalRoughness.xyz = pointInfo.nor;
    }

    float2 currNdc = input.currClip.xy / input.currClip.w;
    float2 prevNdc = input.prevClip.xy / input.prevClip.w;
    output.motionVectors = (prevNdc - currNdc) * 0.5f;

    output.pos = input.worldPos;
    output.depth = length(output.pos - pc.cam->camPos);

    return output;

    //Compute albedo, write out
    //Compute normals, write out
    //Write out world position
    //Write out motion vectors...?

    //Compute pbr parameters, write out
    //TODO: Maybe extract the material parameter finding logic into a helper function? I could use that here...

    //Compute motion vectors
}