#pragma once
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "util.h"
#include "hashmap.h"

struct SceneData {
	VkDeviceAddress deviceSubMeshes;
	VkDeviceAddress materials;

	VkDeviceAddress lightTriangles;
	VkDeviceAddress lightAliasTable;
	VkDeviceAddress envAliasTable;

	uint32_t nLights{ 0 };
	float totalLightWeight{ 0.f };
	float totalEnvWeight{ 0.f };
};

struct PushConstantsPathtracer {
	glm::vec3 camPos;
	glm::vec3 camForward;
	glm::vec3 camRight;
	glm::vec3 camUp;

	VkDeviceAddress tlas;
	VkDeviceAddress sceneData;
	VkDeviceAddress hashMap;

	float fov;
	uint32_t totalFrames{ 0 };
	uint32_t frameNum{ 0 };
	uint32_t maxBounces{ 10 };

	float lensRadius{ 0.f };
	float focusDist{ 1.f };

	EngineStateFlags flags{ 0 };
};

struct PushConstantsPost {
	uint32_t outputImageIdx;
	uint32_t swapRedBlue;
};

// SHaRC resolve. shaders/resolve.slang PushConstantsResolve must stay in this order:
// the pipeline maps the three storage buffers from the device-address fields.
// BLOCK_SIZE in that shader is the dispatch group size.
constexpr uint32_t SHARC_RESOLVE_BLOCK_SIZE = 256;
constexpr uint32_t SHARC_CAPACITY = 1u << 22;
constexpr float SHARC_RADIANCE_SCALE = 1e3f;
constexpr float SHARC_LOGARITHM_BASE = 2.f;
constexpr float SHARC_LEVEL_BIAS = 0.f;

// Match SharcConfig shaders/sharc.slang
struct SharcConfig {
	uint32_t entriesNum{ SHARC_CAPACITY };
	float sceneScale{ 50.f };
	float radianceScale{ SHARC_RADIANCE_SCALE };
	uint32_t padding{ 0 };
};

// Supply bindings 5-9 to update.slang
struct PushDataSharc {
	PushConstantsPathtracer pathtracer;
	VkDeviceAddress hashEntriesAddress;
	VkDeviceAddress accumulationAddress;
	VkDeviceAddress resolvedAddress;
	VkDeviceAddress configAddress;
	VkDeviceAddress lockAddress;
};

// Must match SHARC_TILE_SIZE in shaders/update.slang.
constexpr uint32_t SHARC_UPDATE_TILE_SIZE = 5;

struct PushConstantsResolve {
	glm::vec3 cameraPosition;
	float logarithmBase;
	float sceneScale;
	float levelBias;

	glm::vec3 cameraPositionPrev;
	float radianceScale;

	VkDeviceAddress hashEntriesAddress;
	VkDeviceAddress accumulationAddress;
	VkDeviceAddress resolvedAddress;

	uint32_t capacity;
	uint32_t accumulationFrameNum;
	uint32_t responsiveFrameNum;
	uint32_t staleFrameNumMax;
	uint32_t frameIndex;
};
