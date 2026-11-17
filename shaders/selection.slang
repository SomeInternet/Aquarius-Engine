// Picks pixels to perform the sparse pass
// Probably roll in the sparse pass which pixels to use for SHaRC + ORCA and which to only use for ORCA
[vk::push_constant]
RWStructuredBuffer<float> indirect;

#include "util.slang"

#define BLOCK_SIZE 256

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint32_t threadIdx: SV_DispatchThreadID) {
    if (!(threadIdx & 0xff)) {
        //Add information to buffer
    }
}