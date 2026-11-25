// Picks pixels to perform the sparse pass
// Probably roll in the sparse pass which pixels to use for SHaRC + ORCA and which to only use for ORCA

#include "bsdf.slang"

#define BLOCK_X 8 
#define BLOCK_Y 8
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

// matching writeImageDesciptor
#define IMG_NORMAL_ROUGHNESS 3
#define IMG_VIEW_DEPTH 4
#define IMG_ALBEDO_METALLIC 8
#define IMG_SCORE 9

//for float atomic additin (fit for 32 bit atm)
#define INTIFIED_SCORE_SCALE 1024.0f

struct SelectionState {
    uint scoreSumIntified;
}

struct PushConstantsSelection {
    CameraData *cam;
    SelectionState *state;
    SparsePixelList *sparsePixels;

    uint2 imageSize;
    uint frameNum;

    float budget; // est. like w * h /16
    float thresholdRate; // 1/16
    float mirrorMaxRate;
    float roughnessMax; // treat as diffuse if above this factor
    bool applySpecularHeuristics; // 0 to uniform threshold every pixell
    bool applyBalancing; // 0 to skip bugeting, just for debugging 
}

[[vk::binding(0, 0)]]
RWTexture2D<float4> engineImages[];

[[vk::push_constant]]
PushConstantsSelection pc;

// HELPER FUNCTIONS =========================

// reconstruct view dir from pc info
inline float3 viewDirFromPixel(uint2 px) {};

// compute pixel's score based on material prop (albedo, metallic, roughness), normal
inline float computeScore(uint2 px, float3 albedo, float metallic, float roughness, float3 normal) {};

// TODO: probably helpers for FAST noise

// SCORE + SUM PASS ============================

groupshared float gsWaveSums[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void scorePass(uint2 px: SV_DispatchThreadID, uint groupIndex: SV_GroupIndex) {

    float score = 0.0f;
    if (all(px < pc.imageSize)) {
        float depth = engineImages[IMG_VIEW_DEPTH][px].x;
        if (depth > 0.0f) {
            float4 albedoMetallic = engineImages[IMG_ALBEDO_METALLIC][px];
            float4 normalRoughness = engineImages[IMG_NORMAL_ROUGHNESS][px];

            float3 normal = normalize(normalRoughness.xyz);
            float3 viewDir = -viewDirFromPixel(px);

            score = computeScore(px, albedoMetallic.xyz, albedoMetallic.w, normalRoughness.w, normal);
        }
        engineImages[IMG_SCORE][px] = float4(score, 0.0f, 0.0f, 0.0f); // uhh is there a way to prevent this
    }

    // TODO: sum things

}

// to balance, pick, write to spb
[shader("compute")]
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void pickPass(uint2 px: SV_DispatchThreadID) {
    if (any(px >= pc.imageSize)) {
        return;
    }

    float score = engineImages[IMG_SCORE][px].x;
    if (score <= 0.0f) {
        return;
    }

    float scale = 1.0f;
    if (pc.applyBalancing != 0) {
        float total = float(pc.state->scoreSumIntified) / INTIFIED_SCORE_SCALE;
        scale = total > 0.0f ? pc.budget / total : 0.0f;
    };

    // 
};