#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <tiny_gltf.h>
#include <vk_mem_alloc.h>
#include <iostream>
#include <stdexcept>
#include "engine.h"

int main(int argc, char *argv[]) {
	Engine engine;

	try {
		engine.init();

		engine.run();

		engine.cleanup();
	}
	catch (const std::exception &e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}
