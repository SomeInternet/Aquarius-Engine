#pragma once
//Macros
#define MAX_BOUNCES 10
#define EPSILON 0.001f
#define INVALID_HIT -0.00031415f
#define PI 3.1415926535897932384626433832795028841971f
#define INFINITY 0x1.fffffep+127f

#define ENGINE_IMAGES 10
#define MAX_SWAPCHAIN_SIZE 4
#define HDR_IMAGES 1

struct DownrezLayer {
    uint32_t size;
    uint32_t *list;
};

struct HashMap {
    uint32_t *checkSumBuffer;
    uint3 *radianceBuffer;
    uint32_t *countBuffer;
    uint32_t *childAddressBuffer;
    // Header for the finest-LOD slot list. size is reset each frame; list holds hash slots.
    DownrezLayer *finestLayer;
};

struct PushConstantsDownrez {
    HashMap *hashMap;

    uint32_t *childAdded;
    DownrezLayer *currLayer;
    DownrezLayer *nextLayer;
};

//Struct definitions
struct Vertex {
	half2 uv;
	half2 nor;
    half2 tan;
    float w; //Handedness of the tangent
};

struct DeviceSubMesh {
    float3 *posBuffer;
    Vertex *vertexBuffer;
    uint32_t *indexBuffer;

    int materialIdx;
    int padding;
};

struct Material {
    float4 color;
    float4 emission; // Also store emissive strength in here?
    float2 metallicRoughness;
    float transmissiveness;
    float ior;

    int colorImgIdx;
    int colorSamplerIdx;

    int metallicRoughnessImgIdx;
    int metallicRoughnessSamplerIdx;

    int norImgIdx;
    int norSamplerIdx;

    int emissionImgIdx;
    int emissionSamplerIdx;
};

struct PBR {
    float4 color;
    float2 metallicRoughness;
    float transmissiveness;
    float etaA;
    float etaB;
};

struct LightSample {
    float4 dir;
    float3 radiance;
    float pr;
};

struct AliasTableEntry {
    float threshold;
    uint32_t alias;
};

struct LightTriangle {
    float3 p0;
    float3 p1;
    float3 p2;
    uint32_t matIdx;
    half2 uv0;
    half2 uv1;
    half2 uv2;
};

struct SceneData {
    DeviceSubMesh *deviceSubMeshes;
    Material *materials;

    LightTriangle *lightTriangles;
    AliasTableEntry *lightAliasTable;
    AliasTableEntry *envAliasTable;

    uint32_t nLights = 0;
    float totalLightWeight = 0.f;
    float totalEnvWeight = 0.f;
};

static const uint SPARSE_PIXEL_CAPACITY = 1920u * 1080u;

struct SparsePixel {
    int16_t x;
    int16_t y;
};

struct SparsePixelList {
    uint count;
    SparsePixel pixels[SPARSE_PIXEL_CAPACITY];
};

enum EngineStateFlags : uint32_t {
    IBL = 1 << 0,
    RAY_RECONSTRUCTION = 1 << 1,
    SHARC = 1 << 2,
    SHARC_HEATMAP = 1 << 3,
    GPU_HASH_MAP_HEATMAP = 1 << 4
};

struct CameraData {
    float4x4 currViewProj;
    float4x4 prevViewProj;

    float3 camPos;
    float3 camForward;
    float3 camRight;
    float3 camUp;

    float fov;
};

struct PushConstantsPathtracer {
    float3 camPos;
    float3 camForward;
    float3 camRight;
    float3 camUp;

    RaytracingAccelerationStructure *tlas;
    SceneData *sceneData;
    HashMap *hashMap;

    float fov;
    uint32_t totalFrames;
    uint32_t frameNum;
    uint32_t maxBounces;

    float lensRadius = 0.f;
    float focusDist = 1.f;

    EngineStateFlags flags;
    int heatMapLod;

    SparsePixelList *sparsePixels;
};

struct VkInstance {
    float4 rows[3];
    uint4 padding;
};

inline float3 instanceTransformPoint(VkInstance inst, float3 p) {
    float4 p4 = float4(p, 1.f);
    return float3(dot(inst.rows[0], p4), dot(inst.rows[1], p4), dot(inst.rows[2], p4));
}

inline float3 instanceTransformDir(VkInstance inst, float3 d) {
    return float3(dot(inst.rows[0].xyz, d), dot(inst.rows[1].xyz, d), dot(inst.rows[2].xyz, d));
}

inline float3 instanceTransformNormal(VkInstance inst, float3 n) {
    float3 c0 = float3(inst.rows[0].x, inst.rows[1].x, inst.rows[2].x);
    float3 c1 = float3(inst.rows[0].y, inst.rows[1].y, inst.rows[2].y);
    float3 c2 = float3(inst.rows[0].z, inst.rows[1].z, inst.rows[2].z);

    float3 a = cross(c1, c2);
    float3 res = a * n.x + cross(c2, c0) * n.y + cross(c0, c1) * n.z;
    return dot(c0, a) < 0.f ? -res : res;
}

inline float4 instanceToClip(float4x4 viewProj, VkInstance inst, float3 p, out float3 worldPos) {
    worldPos = instanceTransformPoint(inst, p);
    return mul(viewProj, float4(worldPos, 1.f));
}

inline float3 fallbackTangent(float3 n) {
    return abs(n.x) > abs(n.y) ? float3(-n.z, 0.f, n.x) / sqrt(n.x * n.x + n.z * n.z) :
        float3(0.f, n.z, -n.y) / sqrt(n.y * n.y + n.z * n.z);
}

struct HitPayload {
    float3 throughput = {};

    //TODO: Add weight for RR?
    float3 rayOrigin = {};
    float3 rayDir = {};
    float pdf = 0.f; //BSDF pdf of rayDir, or the light pdf of whatever was hit when the path ends on an emitter or the environment

    float3 directLight = {}; //Light sampling contribution, not yet attenuated

    uint2 seed = 0;
    bool gBufferWritten = false;

    // (review this) sharc adds
    float3 hitNor = {}; // ws normal, facing incoming ray
    float hitLen = 0.f; // length of curr segment
    float roughness = 1.f; // rghness of surface

};

struct PointInfo {
    float3 nor = {};
    float3 tan = {};
    float2 uv = {};
};

// Helper functions
half2 toOctahedral(float3 v) {
    if (length(v) == 0.f) return half2(-2.h);

    v /= (abs(v.x) + abs(v.y) + abs(v.z));
    half2 sign = sign(v.xy);
    if (sign.x == 0) sign.x = 1.h;
    if (sign.y == 0) sign.y = 1.h;

    return (v.z >= 0.f) ? half2(v.xy) : (1.h - abs(float2(v.yx))) * sign;
}

float3 fromOctahedral(half2 u) {
    if (u.x == -2.h && u.y == -2.h) return float3(0);

    float3 v = float3(u.x, u.y, 1.f - abs(u.x) - abs(u.y));

    float2 signNotZeroxy = sign(v.xy);
    signNotZeroxy.x = signNotZeroxy.x == 0.f ? 1.f : signNotZeroxy.x;
    signNotZeroxy.y = signNotZeroxy.y == 0.f ? 1.f : signNotZeroxy.y;

    if (v.z < 0) v.xy = (float2(1.f) - abs(v.yx)) * signNotZeroxy;

    return normalize(v);
}

uint32_t pcg(uint32_t x) {
    uint32_t state = x * 747796405 + 2891336453;
    uint32_t word = (state >> ((state >> 28) + 4) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

uint2 pcg2(uint2 v) { return uint2(pcg(v.x), pcg(v.y)); }

//From ShaderToy https://www.shadertoy.com/view/4tXyWN
static uint2 seed;
float rand() {
    seed += uint2(1);
    uint2 q = 1103515245U * ((seed >> 1U) ^ (seed.yx));
    uint n = 1103515245U * ((q.x) ^ (q.y >> 3U));
    return float(n >> 8) * (1.f / 16777216.f);
}

inline uint32_t sampleAliasTable(AliasTableEntry *aliasTable, uint32_t n, float rand1, float rand2) {
    AliasTableEntry entry = aliasTable[(uint32_t)(rand1 * n)];
    return (rand2 > entry.threshold) ? entry.alias : (uint32_t)(rand1 * n);
}

//Uniformly samples a point on the triangle by area
inline float3 sampleLightTriangle(LightTriangle lightTriangle, float2 xi, out float2 uv, out int matIdx, out float3 nor) {
    float s = sqrt(xi.x);
    float u = 1.f - s;
    float v = xi.y * s;
    matIdx = lightTriangle.matIdx;
    uv = float2(u * lightTriangle.uv0 + v * lightTriangle.uv1 + (1.f - u - v) * lightTriangle.uv2);
    nor = normalize(cross(lightTriangle.p1 - lightTriangle.p0, lightTriangle.p2 - lightTriangle.p0));
    return u * lightTriangle.p0 + v * lightTriangle.p1 + (1.f - u - v) * lightTriangle.p2;
}

//Solid angle pdf of sampleLight landing on a point of an emissive triangle
inline float triangleLightPdf(SceneData *sceneData, Material mat, float dist2, float cosLight) {
    if (cosLight <= 0.f) return 0.f;
    float power = (mat.emission.r + mat.emission.g + mat.emission.b) * mat.emission.w;
    return power / (sceneData->totalEnvWeight + sceneData->totalLightWeight) * dist2 / cosLight;
}

inline PointInfo getPointInfo(uint subMeshIdx, uint primitiveIdx, float3 barycentrics, PushConstantsPathtracer pc, out float3 gNor) {
    {
        float3 *posBuffer = pc.sceneData->deviceSubMeshes[subMeshIdx].posBuffer;
        float3 p0 = posBuffer[pc.sceneData->deviceSubMeshes[subMeshIdx].indexBuffer[primitiveIdx * 3]];
        float3 p1 = posBuffer[pc.sceneData->deviceSubMeshes[subMeshIdx].indexBuffer[primitiveIdx * 3 + 1]];
        float3 p2 = posBuffer[pc.sceneData->deviceSubMeshes[subMeshIdx].indexBuffer[primitiveIdx * 3 + 2]];
        gNor = normalize(cross(p1 - p0, p2 - p0));
    }

    Vertex *vertices = pc.sceneData->deviceSubMeshes[subMeshIdx].vertexBuffer;

    float3 tangent = float3(0.f);

    PointInfo res = {};
    [ForceUnroll] //Loop unrolling, yay!
    for (int i = 0; i < 3; ++i) {
        Vertex v = vertices[pc.sceneData->deviceSubMeshes[subMeshIdx].indexBuffer[primitiveIdx * 3 + i]];

        res.nor += barycentrics[i] * fromOctahedral(v.nor);
        res.uv += barycentrics[i] * float2(v.uv.x, v.uv.y);
        tangent += barycentrics[i] * fromOctahedral(v.tan);
    }

    res.nor = normalize(res.nor);

    if (length(tangent) == 0.f) { //When we have to compute the tangent
        tangent = fallbackTangent(res.nor);
    }
    res.tan = normalize(tangent - res.nor * dot(res.nor, tangent)); // TODO: might have to change this for left-handed spaces
    return res;
}

inline float2 toDiskUniform(float2 xi) {
    float r = sqrt(xi.x);
    float theta = 2.f * PI * xi.y;

    return float2(r * cos(theta), r * sin(theta));
}

inline float2 toDiskConcentric(float2 xi) {
    float2 uOffset = 2.f * xi - float2(1.f);

    if (uOffset.x == 0 && uOffset.y == 0) return float2(0.f);

    float r = (abs(uOffset.x) > abs(uOffset.y)) ? uOffset.x : uOffset.y;
    float theta = (abs(uOffset.x) > abs(uOffset.y)) ? PI * uOffset.y / (uOffset.x * 4.f) : PI / 2.f - PI * uOffset.x / (uOffset.y * 4.f);

    return float2(r * cos(theta), r * sin(theta));
}

inline float3 toHemisphereCosineWeighted(float2 xi) {
    float2 diskSample = toDiskConcentric(xi);

    return float3(diskSample.x, diskSample.y, sqrt(1.f - diskSample.x * diskSample.x - diskSample.y * diskSample.y));
}

inline float3 toHemisphereUniform(float2 xi) {
    return float3(cos(2.f * PI * xi.y) * sqrt(1.f - xi.x * xi.x), sin(2.f * PI * xi.y) * sqrt(1.f - xi.x * xi.x), xi.x);
}

inline void applyNormalMap(inout PointInfo pointInfo, float3 mNor) {
    mNor = normalize(mNor * 2.f - 1.f);

    pointInfo.nor = normalize(mNor.x * pointInfo.tan + mNor.y * cross(pointInfo.nor, pointInfo.tan) + mNor.z * pointInfo.nor);
    pointInfo.tan = normalize(pointInfo.tan - pointInfo.nor * dot(pointInfo.tan, pointInfo.nor));
}

inline float3 mulNor(PointInfo pointInfo, float3 v) {
    return float3(dot(v, pointInfo.tan), dot(v, cross(pointInfo.nor, pointInfo.tan)), dot(v, pointInfo.nor));
}

inline float3 mulNorTranspose(PointInfo pointInfo, float3 v) {
    return v.x * pointInfo.tan + v.y * cross(pointInfo.nor, pointInfo.tan) + v.z * pointInfo.nor;
}