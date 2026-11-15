#pragma once
#include "allocator.h"

constexpr int NUM_HASH_MAP_ENTRIES = 1036800; //1920 * 1080 * 6 (LODs) / 16

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

	AllocatedBuffer hashMapUniformBuffer{};

	HashMap(Allocator &allocator, int numEntries = NUM_HASH_MAP_ENTRIES): 
		checkSumBuffer(allocator.createBuffer(numEntries * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY)),
		radianceBuffer(allocator.createBuffer(numEntries * sizeof(int) * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY)),
		countBuffer(allocator.createBuffer(numEntries * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY)),
		childAddressBuffer(allocator.createBuffer(numEntries * sizeof(int), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY))
	{
		HashMapUniformBuffer hashMapUniforms{
			checkSumBuffer.address, radianceBuffer.address, countBuffer.address, childAddressBuffer.address
		};
		hashMapUniformBuffer = allocator.uploadBuffer(&hashMapUniforms, sizeof(HashMapUniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	}

	void destroy(Allocator &allocator) {
		allocator.destroyBuffer(checkSumBuffer);
		allocator.destroyBuffer(radianceBuffer);
		allocator.destroyBuffer(countBuffer);
		allocator.destroyBuffer(childAddressBuffer);

		allocator.destroyBuffer(hashMapUniformBuffer);
	}
};