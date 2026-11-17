// for things like hashmap config numbers, etc
// usage in downrez.slang, hashmap.slang, engine.cpp, hashmpa.h

#pragma once 

#define MIN_LOD -4
#define MAX_LOD 3
#define BLOCK_SIZE 256
#define MAX_LAYER_SIZE 1048575

#define HASHMAP_UNALLOCATED 0xffffffffu
#define DIR_QUANTIZATION_BITS 4
#define SEARCH_LENGTH 100
#define NUM_HASH_MAP_ENTRIES 1036800
#define RADIANCE_CONVERSION_SCALE 100