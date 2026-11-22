
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

struct GBufferFSOutput {
    [[vk::location(0)]] float4 albedoMetallic : SV_Target0;
    [[vk::location(1)]] float4 normalRoughness : SV_Target1;
    [[vk::location(2)]] float2 motionVectors : SV_Target2;
    [[vk::location(3)]] float depth : SV_Target3;
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
    output.worldPos = mul(pc.model, float4(pc.posBuffer[idx].xyz, 1.f)).xyz;
    output.nor = fromOctahedral(pc.vertBuffer[idx].nor);
    output.tan = fromOctahedral(pc.vertBuffer[idx].tan);
    output.uv = pc.vertBuffer[idx].uv;
    return output;
}

[shader("fragment")]
void fragMain() {
    GBufferFSOutput output = {};

    Material mat = pc.materials[pc.materialIdx];
    
    output.albedoMetallic.xyz = pc.materials[pc.materialIdx].color * (pc.materials[pc.m)


    //Compute albedo, write out
    //Compute normals, write out
    //Write out world position
    //Write out motion vectors...?

    //Compute pbr parameters, write out
    //TODO: Maybe extract the material parameter finding logic into a helper function? I could use that here...

    //Compute motion vectors
}