struct DepthPushConstants {
    float4x4 viewProj;
    float4x4 model;

    float4 *posBuffer;

    float3 camPos;
    int instanceIdx;
};


[[vk::push_constant]]
DepthPushConstants pc;

//Very very light pass
[shader("vertex")]
float4 vertMain(uint32_t idx : SV_VertexID) : SV_Position {
    return mul(pc.viewProj, mul(pc.model, float4(pc.posBuffer[idx].xyz, 1.f)));
}