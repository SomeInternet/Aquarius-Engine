#include "util.slang"

// RWStructuredBuffer<uint32_t> hashMapCheckSums
// RWStructuredBuffer<float> hashMapValues

//This is to average the contributions of multiple vertices in a voxel, I think
// RWStructuredBuffer<uint32_t> hashMapCounts

//Data structures needed to link finer voxels to coarser voxels

#define HASHMAP_UNALLOCATED 0xffffffffu
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
    // Potential overflow issues here
    float lodFac = exp2(lod);
    uint16_t3 voxelPos = reinterpret<uint16_t3>(int16_t3(int3(floor(pos / lodFac)) & 0xffff));

    return uint64_t(voxelPos[0]) << 32 | uint64_t(voxelPos[1]) << 16 | uint64_t(voxelPos[2]);
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

    checkSum = uint32_t(hashed & (~0));
    checkSum = (checkSum == HASHMAP_UNALLOCATED) ? checkSum - 1 : checkSum;
}

HashMapResult insert(uint32_t *checkSums, uint3 *radiances, uint32_t *counts, 
    uint32_t hashMapSize, float3 dir, float3 pos, int lod, float3 radiance,
    inout uint32_t finalAddress, inout uint32_t checkSum) {

    radiance = select(isnan(radiance), 0.f, radiance);

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
            uint3 radianceInt = round(clamp(radiance * RADIANCE_CONVERSION_SCALE, float3(0.f), float3(32767.f)));

            // Prone to overflow, we should examine if it's an issue and address it potentially
            if (radianceInt.x > 0) InterlockedAdd(radiances[finalAddress].x, radianceInt.x);
            if (radianceInt.y > 0) InterlockedAdd(radiances[finalAddress].y, radianceInt.y);
            if (radianceInt.z > 0) InterlockedAdd(radiances[finalAddress].z, radianceInt.z);

            InterlockedAdd(counts[finalAddress], 1);
            // Found empty slot or existing entry. Break out of loop
            entry = SEARCH_LENGTH;
        }
    }

    if (mapValue == HASHMAP_UNALLOCATED) return HashMapResult::ENTRY_UNALLOCATED;
    else if (mapValue == checkSum) return HashMapResult::ENTRY_FOUND;
    else return HashMapResult::ENTRY_NOT_FOUND;
}

HashMapResult allocate(uint32_t *checkSums, uint3 *radiances, uint32_t *counts, 
    uint32_t hashMapSize, float3 dir, float3 pos, int lod, inout uint32_t finalAddress, inout uint32_t checkSum) {

    uint32_t startAddress;
    hashVertex(dir, pos, lod, startAddress, checkSum);

    uint32_t mapValue;
    // check the hashmap entry, find the first non-collision
    finalAddress = startAddress;
    for (uint32_t entry = 0; entry < SEARCH_LENGTH; ++entry) {
        finalAddress = (startAddress + entry) % hashMapSize;

        // Update the entry in the checksum array
        // atomicCX(hashMap[finalAddress], HASHMAP_UNALLOCATED, checksum, mapValue);

        // Atomically update value, counter
        InterlockedCompareExchange(checkSums[finalAddress], HASHMAP_UNALLOCATED, checkSum, mapValue);
        if (mapValue == HASHMAP_UNALLOCATED || mapValue == checkSum) {
            // Found empty slot or existing entry. Break out of loop
            entry = SEARCH_LENGTH;
        }
    }

    if (mapValue == HASHMAP_UNALLOCATED) return HashMapResult::ENTRY_UNALLOCATED;
    else if (mapValue == checkSum) return HashMapResult::ENTRY_FOUND;
    else return HashMapResult::ENTRY_NOT_FOUND;
}

//Populate is like insert, but allocates all descendant voxels while inserting the child at the finest LOD
HashMapResult populate(uint32_t *checkSums, uint3 *radiances,
                       uint32_t *counts, uint32_t *childAddresses,
                       uint32_t hashMapSize, float3 dir, float3 pos, float3 radiance) {

    uint32_t finalAddress, checkSum;
    HashMapResult res = insert(checkSums, radiances, counts, hashMapSize, dir, pos, MIN_LOD, radiance, finalAddress, checkSum);
    if (res == HashMapResult::ENTRY_NOT_FOUND) return HashMapResult::ENTRY_NOT_FOUND;
    for (int i = MIN_LOD + 1; i <= MAX_LOD; ++i) {
        uint32_t address;

        //"Allocate" child and link the parent
        if (allocate(checkSums, radiances, counts, hashMapSize, dir, pos, i, address, checkSum) == HashMapResult::ENTRY_NOT_FOUND) return res;
        childAddresses[finalAddress] = address;
        finalAddress = address;
    }
    return res;
}

HashMapResult query(uint32_t *checkSums, uint3 *radiances, uint32_t *counts, uint32_t hashMapSize, 
float3 dir, float3 pos, int lod, 
out float3 radiance) {
    uint32_t startAddress, checkSum;
    hashVertex(dir, pos, lod, startAddress, checkSum);

    uint32_t mapValue;
    uint32_t finalAddress = startAddress;

    for (uint32_t entry = 0; entry < SEARCH_LENGTH; ++entry) {
        finalAddress = (startAddress + entry) % hashMapSize;
        mapValue = checkSums[finalAddress];
        finalAddress = (startAddress + entry) % hashMapSize;
        if (mapValue == HASHMAP_UNALLOCATED || mapValue == checkSum) {
            // Found empty slot or existing entry. Break out of loop
            entry = SEARCH_LENGTH;
            radiance = float3(radiances[finalAddress]) / (counts[finalAddress] * RADIANCE_CONVERSION_SCALE);
        }
    }

    if (mapValue == HASHMAP_UNALLOCATED) return HashMapResult::ENTRY_UNALLOCATED;
    else if (mapValue == checkSum) return HashMapResult::ENTRY_FOUND;
    return HashMapResult::ENTRY_NOT_FOUND;
}