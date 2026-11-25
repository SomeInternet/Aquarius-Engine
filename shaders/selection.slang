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

// to score (heurstic) + sum
[shader("compute")]
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void scorePass(uint2 px: SV_DispatchThreadID, uint groupIdx: SV_GroupIndex) {
    if (any (p >= pc.imageSize)) {
        //Add information to buffer
    }
}

// to balance, pick, write to spb
[shader("compute")]
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void pickPass(uint2 px: SV_DispatchThreadID) {
    if (any(px >= pc.imageSize)) {
        return;
    }
}