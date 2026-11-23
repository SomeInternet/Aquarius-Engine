#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <vulkan/vulkan.h>

#include <functional>

#include "allocator.h"

//Nor and tan are octahedral-encoded, and everything is half-precision
struct Vertex {
	glm::uint32 uv{ 0 };
	glm::uint32 nor{ 0 };
	glm::uint32 tan{ 0 };
	float w{ 1.f }; //Handedness of the tangent
};

struct Material {
	glm::vec4 color{ 1.f };
	glm::vec4 emission{ 0.f }; //Also store emissive strength in here?
	float metallic{ 1.f };
	float roughness{ 1.f };
	float transmissiveness{ 0.f };
	float ior{ 1.5f };

	int colorImgIdx{ -1 };
	int colorSamplerIdx{ -1 };

	int metallicRoughnessImgIdx{ -1 };
	int metallicRoughnessSamplerIdx{ -1 };

	int norImgIdx{ -1 };
	int norSamplerIdx{ -1 };

	int emissionImgIdx{ -1 };
	int emissionSamplerIdx{ -1 };
};

struct SubMesh {
	int materialIdx{ -1 };

	uint32_t vertexCount{ 0 };
	uint32_t indexCount{ 0 };
	
	AllocatedBuffer posBuffer;
	AllocatedBuffer indexBuffer;
	AllocatedBuffer vertexBuffer;
};

struct DeviceSubMesh {
	VkDeviceAddress posBuffer;
	VkDeviceAddress vertexBuffer;
	VkDeviceAddress indexBuffer;

	int materialIdx{ -1 };
	int padding;
};

//Adapted from Nvidia's Vulkan ray tracing tutorial
struct AccelerationStructure {
	VkAccelerationStructureKHR as{};
	AllocatedBuffer buffer;

	VkAccelerationStructureGeometryKHR geometry{};
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};

	VkDeviceAddress address; //For instancing
};

struct Mesh {
	std::vector<SubMesh> subMeshes;

	AccelerationStructure blas{};
};

struct Instance {
	glm::mat4 transform{ 1.f };
	uint32_t mesh;
	int node{ -1 };
};

enum AnimationType {
	TRANS, ROT, SCALE, UNSUPPORTED
};

struct AnimSampler {
	std::vector<float> times;
	std::vector<glm::vec4> values;
	//TODO: Add different interpolation types
};

struct AnimChannel {
	int node;
	AnimationType type;
	int sampler;
};

struct Animation {
	std::vector<AnimChannel> animChannels;
	std::vector<AnimSampler> animSamplers;
	float duration{ 0.f };
};

struct Node {
	glm::vec3 trans{ 0.f };
	glm::quat rot{ 1.f, 0.f, 0.f, 0.f };
	glm::vec3 scale{ 1.f };
	int parent{ -1 };
};

struct Scene {
	bool loaded{ false };

	AllocatedBuffer materialBuffer;

	std::vector<Mesh> meshes;

	AllocatedBuffer subMeshBuffer; //Flat buffer containing the corresponding DeviceSubMeshes of the meshes

	std::vector<VkSamplerCreateInfo> samplerCreateInfos; //Descriptor heaps allocate the samplers from the create infos
	std::vector<VkImageViewCreateInfo> imageViewCreateInfos; //Descriptor heaps allocate image views from the create infos

	std::vector<AllocatedImage> images;
	std::vector<int> instanceNodes; //Tracks which nodes control which instances
	std::vector<AllocatedBuffer> instanceBuffers;

	AllocatedBuffer tlasScratchBuffer;
	
	std::vector<Node> nodes;
	std::vector<int> updateOrder;
	std::vector<glm::mat4> prevNodeTransforms;
	std::vector<glm::mat4> nodeTransforms;
	std::vector<Animation> animations;
	std::vector<Instance> instances;

	AccelerationStructure tlas{};

	AllocatedBuffer lightTriangles;
	AllocatedBuffer lightStrengthAliasTable;
	uint32_t nLights{ 0 };
	float totalLightWeight{ 0.f };

	bool animated{ false };
};

struct LightSubMesh {
	std::vector<glm::vec3> pos;
	std::vector<glm::uint32> uvs;
	std::vector<uint32_t> indices;
	uint32_t mat{ 0 };
};

struct LightMesh {
	std::vector<LightSubMesh> lightSubMeshes;
};

struct LightTriangle {
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;
	uint32_t matIdx;
	glm::uint32_t uv0;
	glm::uint32_t uv1;
	glm::uint32_t uv2;
};