// One thread per cache entry. The host dispatch uses the same group size.
#define BLOCK_SIZE 256

// Resolve does not update or query. Defining the flags the header tests
// without a default keeps those branches out and silences the preprocessor.
#define SHARC_UPDATE 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_GLSL 0

// SHaRC's buffers are RWStructuredBuffer fields of SharcParameters. Defaults
// stay off for spherical harmonics, so each record is 16 bytes.
#include "include/SHARC/include/SharcCommon.h"

struct PushConstantsResolve {
    float3 cameraPosition;
    float logarithmBase;
    float sceneScale;
    float levelBias;

    float3 cameraPositionPrev;
    float radianceScale;

    // Device addresses of the three cache buffers. The pipeline maps the
    // storage-buffer bindings from these fields; SharcResolveEntry does not
    // load them itself.
    uint64_t hashEntriesAddress;
    uint64_t accumulationAddress;
    uint64_t resolvedAddress;

    uint32_t capacity;
    uint32_t accumulationFrameNum;
    uint32_t responsiveFrameNum;
    uint32_t staleFrameNumMax;
    uint32_t frameIndex;
};

[vk::push_constant]
PushConstantsResolve pc;

[[vk::binding(0, 0)]]
RWStructuredBuffer<uint64_t> hashEntries;

[[vk::binding(1, 0)]]
RWStructuredBuffer<SharcAccumulationData> accumulation;

[[vk::binding(2, 0)]]
RWStructuredBuffer<SharcPackedData> resolved;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    SharcParameters params;
    params.hashGridParameters.cameraPosition = pc.cameraPosition;
    params.hashGridParameters.logarithmBase = pc.logarithmBase;
    params.hashGridParameters.sceneScale = pc.sceneScale;
    params.hashGridParameters.levelBias = pc.levelBias;
    params.hashGridData.capacity = pc.capacity;
    params.hashGridData.hashEntriesBuffer = hashEntries;
    params.radianceScale = pc.radianceScale;
    params.accumulationBuffer = accumulation;
    params.resolvedBuffer = resolved;

    SharcResolveParameters resolveParams;
    resolveParams.cameraPositionPrev = pc.cameraPositionPrev;
    resolveParams.accumulationFrameNum = pc.accumulationFrameNum;
    resolveParams.responsiveFrameNum = pc.responsiveFrameNum;
    resolveParams.staleFrameNumMax = pc.staleFrameNumMax;
    resolveParams.frameIndex = pc.frameIndex;

    SharcResolveEntry(dispatchThreadId.x, params, resolveParams);
}
