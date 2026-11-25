
#include "util.slang"

struct RasterPushConstants {
    CameraData *cam;
    Material *materials;
    VkInstance *instances;
    VkInstance *prevInstances;

    float3 *posBuffer;
    Vertex *vertBuffer;

    int materialIdx;
    int instanceIdx;
};

struct GBufferFSInput {
    float4 fragPos : SV_Position;
    float3 worldPos;
    float3 nor;
    float3 tan;
    float2 uv;
    float4 currClip;
    float4 prevClip;
};

//NOTE: If anyone changes this, make sure to update the color attachment info in the raster pipeline setup!
struct GBufferFSOutput {
    [[vk::location(0)]] float4 radianceScore : SV_Target0;
    [[vk::location(1)]] float4 albedoEmission : SV_Target1;
    [[vk::location(2)]] float4 metallicTransmissionIor : SV_Target2;
    [[vk::location(3)]] float depth : SV_Target3;
    [[vk::location(4)]] float4 normalRoughness : SV_Target4;
    [[vk::location(5)]] float2 motionVectors : SV_Target5;
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
    float3 objPos = pc.posBuffer[idx];
    Vertex v = pc.vertBuffer[idx];

    output.fragPos = instanceToClip(pc.cam->currViewProj, inst, objPos, output.worldPos);
    output.currClip = output.fragPos;
    output.prevClip = mul(pc.cam->prevViewProj, float4(instanceTransformPoint(pc.prevInstances[pc.instanceIdx], objPos), 1.f));

    output.nor = instanceTransformNormal(inst, fromOctahedral(v.nor));
    output.tan = instanceTransformDir(inst, fromOctahedral(v.tan));
    output.uv = float2(v.uv);
    return output;
}

inline float4 sampleTex(int imgIdx, int sampIdx, float2 uv, float4 defaultValue) {
    return (imgIdx != -1) ? sceneImages[NonUniformResourceIndex(imgIdx)].SampleLevel(samplers[NonUniformResourceIndex(sampIdx + 1)], uv, 0.f) : defaultValue;
}

[earlydepthstencil]
[shader("fragment")]
GBufferFSOutput fragMain(GBufferFSInput input) {
    GBufferFSOutput output = {};

    Material mat = pc.materials[pc.materialIdx];

    output.albedoEmission.xyz = mat.color.xyz * sampleTex(mat.colorImgIdx, mat.colorSamplerIdx, input.uv, float4(1.f)).xyz;

    float2 metallicRoughness = mat.metallicRoughness * sampleTex(mat.metallicRoughnessImgIdx, mat.metallicRoughnessSamplerIdx, input.uv, float4(1.f)).xy;
    output.metallicTransmissionIor = float4(metallicRoughness.x, mat.transmissiveness, mat.ior - 1.f, 0.f);
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

    float3 viewDir = pc.cam->camPos - input.worldPos;
    output.depth = length(viewDir);

    if (length(mat.emission) > 0.f) {
        output.albedoEmission = float4(mat.emission.xyz * sampleTex(mat.emissionImgIdx, mat.emissionSamplerIdx, input.uv, float4(1.f)).xyz, mat.emission.w);
    } else { // Score computation
        //
        float3 startingBeta = envBRDFApprox2(lerp(output.albedoEmission.xyz, float3(.4f), metallicRoughness.x), metallicRoughness.y * metallicRoughness.y, 
            abs(dot(output.normalRoughness.xyz, normalize(viewDir))));

        // BetaLum is the max channel of the throughput (We don't want to sample emissive surfaces, so we only fill this out where there isn't emission)
        float betaLum = saturate(max(startingBeta.x, max(startingBeta.y, startingBeta.z)));

        float fresnelFac = (1.f - betaLum);
        fresnelFac *= fresnelFac;
        fresnelFac *= fresnelFac;

        float sparsityScore = min(output.normalRoughness.w, fresnelFac);

        float maxRoughnessToAbort = 1.f;
        float roughnessAdjust = saturate((maxRoughnessToAbort - sparsityScore) / maxRoughnessToAbort);
        float mirrorMaxRateOfSparsification = 1.f;

        float threshold = lerp(.4f, max(mirrorMaxRateOfSparsification, .4f), roughnessAdjust);
        output.radianceScore.w = threshold;
    }

    

    return output;

    //Compute albedo, write out
    //Compute normals, write out
    //Write out world position
    //Write out motion vectors...?

    //Compute pbr parameters, write out
    //TODO: Maybe extract the material parameter finding logic into a helper function? I could use that here...

    //Compute motion vectors
}