#pragma once
#include "allocator.h"

constexpr int NUM_HASH_MAP_ENTRIES = 1036800; //1920 * 1080 * 6 (LODs) / 16
constexpr int LIST_SIZE = 32767;

// Match MIN_LOD, MAX_LOD, and BLOCK_SIZE in shaders/hashmap.slang and shaders/downrez.slang.
// Each downrez dispatch folds one level into the next, stopping at the coarsest.
constexpr int HASHMAP_MIN_LOD = -4;
constexpr int HASHMAP_MAX_LOD = 3;
constexpr uint32_t DOWNREZ_BLOCK_SIZE = 256;

struct DownrezLayer {
	uint32_t size = 0;
	VkDeviceAddress list{};
};

struct HashMapUniformBuffer {
	VkDeviceAddress checkSumBuffer{};
	VkDeviceAddress radianceBuffer{};
	VkDeviceAddress countBuffer{};
	VkDeviceAddress childAddressBuffer{};
};

struct HashMap {
	AllocatedBuffer checkSumBuffer{};
	AllocatedBuffer radianceBuffer{};
	AllocatedBuffer countBuffer{};
	AllocatedBuffer childAddressBuffer{};
	AllocatedBuffer childAdded{};

	AllocatedBuffer hashMapUniformBuffer{};

	std::vector<AllocatedBuffer> downrezLists;
	std::vector<DownrezLayer> downrezLayers;
	std::vector<AllocatedBuffer> downrezLayerBuffers;

	HashMap(Allocator &allocator, int numEntries = NUM_HASH_MAP_ENTRIES): 
		checkSumBuffer(allocator.createBuffer(numEntries * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY)),
		radianceBuffer(allocator.createBuffer(numEntries * sizeof(int) * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY)),
		countBuffer(allocator.createBuffer(numEntries * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY)),
		childAddressBuffer(allocator.createBuffer(numEntries * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY)),
		childAdded(allocator.createBuffer(numEntries * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, false))
	{
		downrezLists = {
			allocator.createBuffer(sizeof(uint32_t) * LIST_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY),
			allocator.createBuffer(sizeof(uint32_t) * LIST_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY)
		};

		downrezLayers = {
			{0, downrezLists[0].address},
			{0, downrezLists[1].address}
		};

		downrezLayerBuffers = {
			allocator.uploadBuffer(&downrezLayers[0], sizeof(DownrezLayer), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU),
			allocator.uploadBuffer(&downrezLayers[1], sizeof(DownrezLayer), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU)
		};

		HashMapUniformBuffer hashMapUniforms{
			checkSumBuffer.address, radianceBuffer.address, countBuffer.address, childAddressBuffer.address
		};
		hashMapUniformBuffer = allocator.uploadBuffer(&hashMapUniforms, sizeof(HashMapUniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	}

	void resetSize(Allocator &allocator, int i) {
		memcpy(downrezLayerBuffers[i].info.pMappedData, &downrezLayers[i], sizeof(DownrezLayer));
		allocator.flush(downrezLayerBuffers[i]);
	}

	void destroy(Allocator &allocator) {
		allocator.destroyBuffer(checkSumBuffer);
		allocator.destroyBuffer(radianceBuffer);
		allocator.destroyBuffer(countBuffer);
		allocator.destroyBuffer(childAddressBuffer);
		allocator.destroyBuffer(childAdded);

		allocator.destroyBuffer(hashMapUniformBuffer);

		for (int i = 0; i < 2; ++i) {
			allocator.destroyBuffer(downrezLists[i]);
			allocator.destroyBuffer(downrezLayerBuffers[i]);
		}
	}
};