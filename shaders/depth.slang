#include "util.slang"

struct DepthPushConstants {
    CameraData *cam;
    VkInstance *instances;
    float4 *posBuffer;
    int instanceIdx;
};


[[vk::push_constant]]
DepthPushConstants pc;

//Very very light pass
[shader("vertex")]
float4 vertMain(uint32_t idx : SV_VertexID) : SV_Position {
    float3 worldPos;
    return instanceToClip(pc.cam->currViewProj, pc.instances[pc.instanceIdx], pc.posBuffer[idx].xyz, worldPos);
}