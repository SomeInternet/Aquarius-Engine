// for SHARC stuff across the passes

#include "include/SHARC/include/SharcCommon.h"

#define CAPACITY (1u << 22)
#define SCENE_SCALE 50.0f
#define RADIANCE_SCALE 1e3f

// TODO : ACTUALLY AGREE THESE
[vk::binding(5, 0)] RWStructuredBuffer<HashGridKey> hashEntriesTemp;
[vk::binding(6, 0)] RWStructuredBuffer<SharcAccumulationData> accumulationTemp;
[vk::binding(7, 0)] RWStructuredBuffer<SharcPackedData> resolvedTemp;

SharcParameters createSharcParams(float3 cameraPos) {
    SharcParameters params;

    HashGridParameters gridParams = { 
        cameraPos, 
        SHARC_GRID_LOGARITHM_BASE, 
        SCENE_SCALE, 
        SHARC_GRID_LEVEL_BIAS 
    };

    HashGridData gridData = {
        CAPACITY,
        hashEntriesTemp
    };

    params.hashGridData = gridData;
    params.hashGridParameters = gridParams;

    params.radianceScale = RADIANCE_SCALE;
    params.accumulationBuffer = accumulationTemp;
    params.resolvedBuffer = resolvedTemp;
    return params;

}