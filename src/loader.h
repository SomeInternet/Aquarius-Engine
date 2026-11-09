#pragma once
#include "engine.h"

namespace loader {
	Scene loadScene(VkDevice &device, Allocator &allocator, const std::string &file, DeviceProperties &deviceProperties, vkb::DispatchTable &dispatchTable);

	void updateScene(VkDevice &device, Scene &scene, vkb::DispatchTable &dispatchTable, float time);

	void destroyScene(VkDevice &device, Allocator &allocator, Scene &scene, vkb::DispatchTable &dispatchTable);
};