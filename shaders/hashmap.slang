#include "util.slang"

// RWStructuredBuffer<uint32_t> hashMapCheckSums
// RWStructuredBuffer<float> hashMapValues

//This is to average the contributions of multiple vertices in a voxel, I think
// RWStructuredBuffer<uint32_t> hashMapCounts

//Data structures needed to link finer voxels to coarser voxels

#define HASHMAP_UNALLOCATED ~0
#define MIN_LOD -4
#define MAX_LOD 3
#define DIR_QUANTIZATION_BITS 4
#define SEARCH_LENGTH 100
#define NUM_HASH_MAP_ENTRIES 1036800

#define RADIANCE_CONVERSION_SCALE 100

enum HashMapResult {
    ENTRY_UNALLOCATED, ENTRY_FOUND, ENTRY_NOT_FOUND,
};

//NOTE: I imagine the hash table would consist of a buffer of entries and a buffer of counts to use only when populating?

//From a very good friend
inline uint64_t hash(uint64_t sK)
{
    sK ^= sK >> 30;
    sK *= 0xbf58476d1ce4e5b9ULL;
    sK ^= sK >> 27;
    sK *= 0x94d049bb133111ebULL;
    sK ^= sK >> 31;

    return sK;
}

// Quantizes direction, though it's left to the user to compress it down into the actual proper size afterwards. The quantized direction
//goes into the LSB.
//  TODO jitter
inline uint32_t quantizeDir(float3 dir) {
    half2 oDir = toOctahedral(dir);
    oDir = (oDir + 1.h) / 2.h;
    uint2 qiDir = min(uint2(oDir * (1 << DIR_QUANTIZATION_BITS)), uint2((1 << DIR_QUANTIZATION_BITS) - 1));
    return (qiDir.x << DIR_QUANTIZATION_BITS) | qiDir.y;
}

inline uint64_t quantizePos(float3 pos, int lod) {
    uint3 voxelPos = floor(pos / (1 << lod));

    return uint64_t(voxelPos[0] << 32) | uint64_t(voxelPos[1] << 16) | uint64_t(voxelPos[2]);
}

inline uint64_t quantize(float3 dir, float3 pos, int lod) {
     // Compute the quantized voxel

    // Layout would be (subject to change)
    //  DIRECTION (8 bits) | POSITION (48 bits) | lod (8 bits)
    return (uint64_t(quantizeDir(dir)) << 56) | (quantizePos(pos, lod) << 8) | uint64_t(lod - MIN_LOD);
}

inline void hashVertex(float3 dir, float3 pos, int lod, inout uint32_t startAddress, inout uint32_t checkSum) {
    uint64_t sK = quantize(dir, pos, lod);

    uint64_t hashed = hash(sK);
    startAddress = uint32_t(hashed >> 32);
    startAddress = (startAddress == ~0) ? startAddress - 1 : startAddress;
    checkSum = uint32_t(hashed & (~0));
    checkSum = (checkSum == ~0) ? checkSum - 1 : checkSum;
}

HashMapResult insert(RWStructuredBuffer<uint32_t> checkSums, RWStructuredBuffer<uint32_t> radiances, RWStructuredBuffer<uint32_t> counts, 
    uint32_t hashMapSize, float3 dir, float3 pos, int lod, float3 radiance,
    inout uint32_t finalAddress, inout uint32_t checkSum) {

    uint32_t startAddress;
    hashVertex(dir, pos, lod, startAddress, checkSum);

    uint32_t mapValue;
    // check the hashmap entry, find the first non-collision
    finalAddress = startAddress;
    for (uint32_t entry = 0; entry < SEARCH_LENGTH; ++entry) {
        finalAddress = (startAddress + entry) % hashMapSize;

        //Update the entry in the checksum array
        //atomicCX(hashMap[finalAddress], HASHMAP_UNALLOCATED, checksum, mapValue);

        // Atomically update value, counter
        InterlockedCompareExchange(checkSums[finalAddress], HASHMAP_UNALLOCATED, checkSum, mapValue);
        if (mapValue == HASHMAP_UNALLOCATED || mapValue == checkSum) {
            if (mapValue == HASHMAP_UNALLOCATED) {
                uint3 radianceInt = round(clamp(radiance * RADIANCE_CONVERSION_SCALE, float3(0.f), float3(32767.f)));

                //Prone to overflow, we should examine if it's an issue and address it potentially
                InterlockedAdd(radiances[finalAddress * 3], radianceInt[0]);
                InterlockedAdd(radiances[finalAddress * 3 + 1], radianceInt[1]);
                InterlockedAdd(radiances[finalAddress * 3 + 2], radianceInt[2]);
                InterlockedAdd(counts[finalAddress], 1);
            }
            // Found empty slot or existing entry. Break out of loop
            entry = SEARCH_LENGTH;
        }
    }

    if (mapValue == HASHMAP_UNALLOCATED) return HashMapResult::ENTRY_UNALLOCATED;
    else if (mapValue == checkSum) return HashMapResult::ENTRY_FOUND;
    else return HashMapResult::ENTRY_NOT_FOUND;
}

//Populate is like insert, but allocates all descendant voxels while inserting the child at the finest LOD
HashMapResult populate(RWStructuredBuffer<uint32_t> checkSums, RWStructuredBuffer<uint32_t> radiances,
                       RWStructuredBuffer<uint32_t> counts, RWStructuredBuffer<uint32_t> childAddresses,
                       uint32_t hashMapSize, float3 dir, float3 pos, float3 radiance,
    inout uint32_t finalAddress, inout uint32_t checkSum) {
    for (int i = 0; i < MIN_LOD, i >= MAX_LOD; ++i) {}
}

HashMapResult query(RWStructuredBuffer<uint32_t> checkSums, RWStructuredBuffer<uint32_t> radiances, uint32_t hashMapSize, 
float3 dir, float3 pos, int lod, 
out float3 radiance) {
    uint32_t startAddress, checkSum;
    hashVertex(dir, pos, lod, startAddress, checkSum);

    uint32_t mapValue;
    uint32_t finalAddress = startAddress;
    for (uint32_t entry = 0; entry < SEARCH_LENGTH; ++entry) {
        finalAddress = (startAddress + entry) % hashMapSize;
        if (mapValue == HASHMAP_UNALLOCATED || mapValue == checkSum) {
            // Found empty slot or existing entry. Break out of loop
            entry = SEARCH_LENGTH;
            radiance = float3(radiances[finalAddress], radiances[finalAddress + 1], radiances[finalAddress + 2]) / RADIANCE_CONVERSION_SCALE;
        }
    }

    if (mapValue == HASHMAP_UNALLOCATED) return HashMapResult::ENTRY_UNALLOCATED;
    else if (mapValue == checkSum) return HashMapResult::ENTRY_FOUND;
    return HashMapResult::ENTRY_NOT_FOUND;
}