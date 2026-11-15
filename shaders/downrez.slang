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

//     DownrezLayer *currLayer;
//     DownrezLayer *nextLayer;
// };

[[vk::push_constant]]
PushConstantsDownrez pcdr;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint32_t threadIdx: SV_DispatchThreadID) {
    if (threadIdx == 0) pcdr.nextLayer->size = 0;

    if (threadIdx >= min(MAX_LAYER_SIZE, pcdr.currLayer.size)) return;

    

    //pcdr.hashMap.radianceBuffer

    //IDK if this is the right one...
    AllMemoryBarrierWithGroupSync();

    pcdr.currLayer->list[threadIdx];
    
}