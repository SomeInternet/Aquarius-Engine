//Steps for downrez
//Ping pong buffers
// 1) Size of the list (Threads check against this so we don't overwrite)
// 2) The list itself
// 3) Boolean list of "in list"
// The above buffers would be part of the Hash map, probably

// The first thread would reset the counter for the next layer

// *Assume this already exists for the first layer (but we'll need to set that up)

// For current pass and then the next "layer"
// For each thread, atomically add its radiance and its count to the hash map entry for its parent
// If the parent is not already in the list, add it (could use a boolean buffer or something to check)

#define BLOCK_SIZE 256
#define MAX_LAYER_SIZE 32767

#include "hashmap.slang"

// struct DownrezLayer {
//     uint32_t size;
//     uint32_t *list;
// };

// struct PushConstantsDownrez {
//     HashMap *hashMap;

//     uint32_t *childAdded;
//     DownrezLayer *currLayer;
//     DownrezLayer *nextLayer;
// };

[[vk::push_constant]]
PushConstantsDownrez pcdr;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint32_t threadIdx: SV_DispatchThreadID) {
    //Next layer's size zeroed before dispatch

    if (threadIdx >= min(MAX_LAYER_SIZE, pcdr.currLayer.size)) return;

    // 1) Thread operates on parent voxel (fine)
    uint32_t sourceIdx = pcdr.currLayer->list[threadIdx];
    uint3 radiance = pcdr.hashMap->radianceBuffer[sourceIdx];
    uint count = pcdr.hashMap->countBuffer[sourceIdx];

    uint destIdx = pcdr.hashMap->childAddressBuffer[sourceIdx];
    if (destIdx == HASHMAP_UNALLOCATED) return;

    // 2) Thread adds parent's radiance to child
    // Potential risk of overflow...
    InterlockedAdd(pcdr.hashMap->radianceBuffer[destIdx].x, radiance.x);
    InterlockedAdd(pcdr.hashMap->radianceBuffer[destIdx].y, radiance.y);
    InterlockedAdd(pcdr.hashMap->radianceBuffer[destIdx].z, radiance.z);
    InterlockedAdd(pcdr.hashMap->countBuffer[destIdx], count);

    //3) If child (coarse) voxel is not in the list, add the voxel to the list
    //Off the top of my head, we could use a boolean array alongside the hash table to check

    uint32_t wasChildAdded;
    InterlockedCompareExchange(pcdr.childAdded[destIdx], 0u, 1u, wasChildAdded);

    // Original value 0 means this thread won the flag and must append the parent.
    if (wasChildAdded == 0u) {
        // add child to list
        int index;
        InterlockedAdd(pcdr.nextLayer->size, 1, index);
        pcdr.nextLayer->list[index] = destIdx;
    }

    //Before the next invocation, ping pong
}