// for SHARC stuff across the passes

#include "include/SHARC/include/SharcCommon.h"

// #define CAPACITY (1u << 22)
// #define SCENE_SCALE 50.0f
// #define RADIANCE_SCALE 1e3f

struct SharcConfig
{
    uint entriesNum;
    float sceneScale;
    float radianceScale;
    uint padding;
};

SharcParameters createSharcParams(
    float3 cameraPos,
    SharcConfig config,
    RWStructuredBuffer<HashGridKey> sharcHashEntries,
    RWStructuredBuffer<SharcAccumulationData> sharcAccumulation,
    RWStructuredBuffer<SharcPackedData> sharcResolved,
    RWStructuredBuffer<uint> sharcLockBuffer
) 
{
    SharcParameters params;

    HashGridParameters gridParams = { 
        cameraPos, 
        SHARC_GRID_LOGARITHM_BASE, 
        config.sceneScale, 
        SHARC_GRID_LEVEL_BIAS 
    };

    HashGridData gridData = {
        config.entriesNum,
        sharcHashEntries
    };

    params.hashGridData = gridData;
    params.hashGridParameters = gridParams;

    #if !HASH_GRID_ENABLE_64_BIT_ATOMICS
        parameters.hashGridData.lockBuffer = sharcLockBuffer;
    #endif

    params.radianceScale = config.radianceScale;
    params.accumulationBuffer = sharcAccumulation;
    params.resolvedBuffer = sharcResolved;
    return params;
}