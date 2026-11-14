#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <tiny_gltf.h>
#include <vk_mem_alloc.h>
#include <cstddef>
#include <format>

#ifdef AQUARIUS_DLSS
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_helpers_dlssd_d3d.h>
#endif

#include "engine.h"
#include "init.h"
#include "loader.h"
#include "pushconstants.h"
#include "aliastable.h"

static_assert(sizeof(glm::vec3) == 12, "SHaRC push constants assume a packed vec3");
static_assert(sizeof(SharcConfig) == 16);
static_assert(offsetof(SharcConfig, sceneScale) == 4);
static_assert(offsetof(SharcConfig, radianceScale) == 8);
static_assert(offsetof(SharcConfig, padding) == 12);
static_assert(offsetof(PushDataSharc, pathtracer) == 0);
static_assert(offsetof(PushConstantsResolve, hashEntriesAddress) == 40);
static_assert(offsetof(PushConstantsResolve, accumulationAddress) == 48);
static_assert(offsetof(PushConstantsResolve, resolvedAddress) == 56);
static_assert(offsetof(PushConstantsResolve, capacity) == 64);
static_assert(offsetof(PushConstantsResolve, frameIndex) == 80);

//Helpers to, uh, help us set up the descriptors for the shaders
static VkDescriptorSetAndBindingMappingEXT heapMapping(uint32_t binding, uint32_t set, VkSpirvResourceTypeFlagsEXT mask, 
	VkDeviceSize heapOffset, VkDeviceSize heapArrayStride) {

	VkDescriptorSetAndBindingMappingEXT mapping{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT };
	mapping.descriptorSet = set;
	mapping.firstBinding = binding;
	mapping.bindingCount = 1;
	mapping.resourceMask = mask;
	mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
	mapping.sourceData.constantOffset.heapOffset = heapOffset;
	mapping.sourceData.constantOffset.heapArrayStride = heapArrayStride; 
	return mapping;
}

static VkDescriptorSetAndBindingMappingEXT pushAddressMapping(uint32_t binding, uint32_t set, VkSpirvResourceTypeFlagsEXT mask, uint32_t pushOffset) {
	VkDescriptorSetAndBindingMappingEXT mapping{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT };
	mapping.descriptorSet = set;
	mapping.firstBinding = binding;
	mapping.bindingCount = 1; 
	mapping.resourceMask = mask;
	mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
	mapping.sourceData.pushAddressOffset = pushOffset;
	return mapping;
}

void Engine::initWindow() {
	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags windowFlags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

	_window = SDL_CreateWindow("Aquarius Engine", _windowExtent.width, _windowExtent.height, windowFlags);
}

void Engine::initVulkan() {
	//Lambda for validating VkBootstrap calls
	//&& denotes a forwarding/universal reference, which can bind to anything (l or r value)
	const auto vkbCheck = [](auto &&result, const std::string_view op) {
			if (!result) {
				std::string errorMsg = std::format("{} failed: {}", op, result.error().message());
				for (const std::string &reason : result.detailed_failure_reasons()) {
					errorMsg += "\n " + reason;
				}
				throw std::runtime_error(errorMsg);
			}

			return result.value();
		};

	vkb::SystemInfo systemInfo =
		vkbCheck(vkb::SystemInfo::get_system_info(), "Querying Vulkan system info");

	vkb::InstanceBuilder builder = vkb::InstanceBuilder()
		.set_app_name("Aquarius Engine")
		.set_engine_name("Aquarius Engine")
		.require_api_version(1, 4, 0)
#if _DEBUG
		.request_validation_layers()
		.set_debug_callback(debugCallback)
#endif
		;

	_vkbData.instance = vkbCheck(builder.build(), "Creating Vulkan instance");
	_instance = _vkbData.instance.instance;
	_instanceDispatchTable = _vkbData.instance.make_table();

	vkb::PhysicalDeviceSelector selector{ _vkbData.instance };

	SDL_Vulkan_CreateSurface(_window, _instance, nullptr, &_surface);

	{
		const std::array requiredExtensions{
			VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
			VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
			VK_KHR_RAY_QUERY_EXTENSION_NAME,
			VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
			VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME,
			VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME
		};

		// The post pass writes the swapchain from a compute shader. Linux surfaces
		// expose B8G8R8A8_UNORM, which has no SPIR-V storage format, so the write
		// has to be unformatted.
		VkPhysicalDeviceFeatures features10{};
		features10.shaderStorageImageWriteWithoutFormat = VK_TRUE;
		features10.shaderInt64 = VK_TRUE; //SHaRC hash keys are uint64

		VkPhysicalDeviceVulkan11Features features11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
		features11.storageBuffer16BitAccess = true; //Allows us to access our half-precision octahedral-projected normals and tangents
		features11.uniformAndStorageBuffer16BitAccess = true; //SHaRC resolved records store float16 radiance
		features11.storagePushConstant16 = true;

		VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
		features12.bufferDeviceAddress = true;
		features12.descriptorIndexing = true;
		features12.shaderFloat16 = true;
		features12.runtimeDescriptorArray = true; //Allows shaders to use an arrays of resources whose size is not fixed at compile time
		features12.descriptorBindingSampledImageUpdateAfterBind = true;
		features12.descriptorBindingPartiallyBound = true;
		features12.descriptorBindingVariableDescriptorCount = true;
		features12.scalarBlockLayout = true; //So we can have proper striding in shaders

		VkPhysicalDeviceVulkan13Features features13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
		features13.dynamicRendering = true;
		features13.synchronization2 = true;

		//I was curious as to why features and extensions had to be queried separately.
		//My main takaway was that vendors can support extensions while not supporting all their features

		VkPhysicalDeviceRayTracingPipelineFeaturesKHR featuresRayTracingPipeline = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
		featuresRayTracingPipeline.rayTracingPipeline = true;

		VkPhysicalDeviceAccelerationStructureFeaturesKHR featuresAccelerationStructures = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
		featuresAccelerationStructures.accelerationStructure = true;

		VkPhysicalDeviceRayQueryFeaturesKHR featuresRayQuery = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
		featuresRayQuery.rayQuery = true;

		VkPhysicalDeviceDescriptorHeapFeaturesEXT featuresDescriptorHeap = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT };
		featuresDescriptorHeap.descriptorHeap = true;

		VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT featuresReorder = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT };
		featuresReorder.rayTracingInvocationReorder = true;

		selector.set_surface(_surface)
			.set_minimum_version(1, 4)
			.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
			.add_required_extensions(requiredExtensions.size(), requiredExtensions.data())
			.set_required_features(features10)
			.set_required_features_11(features11)
			.set_required_features_12(features12)
			.set_required_features_13(features13)
			.add_required_extension_features(featuresRayTracingPipeline)
			.add_required_extension_features(featuresAccelerationStructures)
			.add_required_extension_features(featuresDescriptorHeap)
			.add_required_extension_features(featuresRayQuery)
			.add_required_extension_features(featuresReorder);

		const auto physicalDevices = vkbCheck(selector.select_devices(), "Selecting physical device");

		//TODO: Add additional checks to physical device selection

		if (physicalDevices.size() == 0) throw std::runtime_error("No compatible Vulkan devices were found!");

		_vkbData.physicalDevice = physicalDevices[0];
		_physicalDevice = _vkbData.physicalDevice.physical_device;
	}

#ifdef AQUARIUS_DLSS
	//Query NGX extensions for DLSS
	NVSDK_NGX_FeatureDiscoveryInfo ngxInfo{};
	ngxInfo.SDKVersion = NVSDK_NGX_Version_API;
	ngxInfo.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
	ngxInfo.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
	ngxInfo.Identifier.v.ProjectDesc = { PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, PROJECT_VERSION };
	ngxInfo.ApplicationDataPath = L"./build/ngx";

	uint32_t ngxExtensionCount = 0;
	VkExtensionProperties *ngxExtensionProperties = nullptr;

	NVSDK_NGX_Result ngxResult = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(_instance, _physicalDevice, &ngxInfo, 
		&ngxExtensionCount, &ngxExtensionProperties);
	std::vector<const char*> ngxExtensionNames;
	for (uint32_t i = 0; i < ngxExtensionCount; ++i) {
		ngxExtensionNames.push_back(ngxExtensionProperties[i].extensionName);
	}
	_vkbData.physicalDevice.enable_extensions_if_present(ngxExtensionNames);
#endif

	{
		_deviceProperties.properties.pNext = &_deviceProperties.rtProperties;
		_deviceProperties.rtProperties.pNext = &_deviceProperties.asProperties;
		_deviceProperties.asProperties.pNext = &_deviceProperties.dhProperties;
		vkGetPhysicalDeviceProperties2(_physicalDevice, &_deviceProperties.properties);
	}

	//Create the logical device and queues
	//TODO: Consider moving compute and present to their own queues
	{
		_vkbData.device = vkbCheck(vkb::DeviceBuilder(_vkbData.physicalDevice).build(), "Creating Vulkan logical device");
		_device = _vkbData.device.device;

		_dispatchTable = _vkbData.device.make_table();

		_queue = vkbCheck(_vkbData.device.get_queue(vkb::QueueType::graphics), "Getting graphics queue");
		_queueFamilyIdx = vkbCheck(_vkbData.device.get_queue_index(vkb::QueueType::graphics), "Getting graphics queue family");


		const uint32_t presentQueueFamilyIdx = vkbCheck(_vkbData.device.get_queue_index(vkb::QueueType::present), "Getting present queue family");

		if (!(_vkbData.device.queue_families[_queueFamilyIdx].queueFlags & VK_QUEUE_COMPUTE_BIT)) throw std::runtime_error("Selected queue family doesn't support compute!");
		if(presentQueueFamilyIdx != _queueFamilyIdx) throw std::runtime_error("Selected queue family doesn't support present!");
	}

#ifdef AQUARIUS_DLSS
	NVSDK_NGX_VULKAN_Init_with_ProjectID(PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, PROJECT_VERSION, L"./build/ngx",
		_instance, _physicalDevice, _device, vkGetInstanceProcAddr, vkGetDeviceProcAddr);
	bool paramsLoaded = NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&_ngxParams));

	int denoisingAvailable = 0;
	auto result = _ngxParams->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &denoisingAvailable);

	_rrAvailable = static_cast<bool>(denoisingAvailable);

	if (NVSDK_NGX_SUCCEED(result) && denoisingAvailable) {
		//TODO: Enable ray reconstruction
		_rrAvailable = true;
	}
	else if (NVSDK_NGX_SUCCEED(result) && !denoisingAvailable) {
		//TODO: Add more debug messages here
		std::cout << "DLSS Denoising unavailable on this device!" << std::endl;

		int needsUpdatedDriver = 0;
		NVSDK_NGX_Result resultUpdatedDriver = _ngxParams->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);

		int minDriverVersionMajor, minDriverVersionMinor = 0;

		NVSDK_NGX_Result ResultMinDriverVersionMajor = _ngxParams->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &minDriverVersionMajor);
		NVSDK_NGX_Result ResultMinDriverVersionMinor = _ngxParams->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minDriverVersionMinor);

		if (NVSDK_NGX_SUCCEED(resultUpdatedDriver) && needsUpdatedDriver) {
			std::cout << "Outdated driver!" << std::endl;

			if (NVSDK_NGX_SUCCEED(ResultMinDriverVersionMajor) && NVSDK_NGX_SUCCEED(ResultMinDriverVersionMinor)) {
				std::cout << "Mininimum driver version required: " << minDriverVersionMajor << "." << minDriverVersionMinor << std::endl;
			}
		}
	}

	if (paramsLoaded) _deletionQueue.push([this]() {
		if (_rrAvailable) NVSDK_NGX_VULKAN_ReleaseFeature(_ngxRrHandle);
		NVSDK_NGX_VULKAN_DestroyParameters(_ngxParams);
		NVSDK_NGX_VULKAN_Shutdown1(_device);
		});
#endif
}

void Engine::createSwapChain(int width, int height) {
	const auto vkbCheck = [](auto &&result, const std::string_view op) {
		if (!result) {
			std::string errorMsg = std::format("{} failed: {}", op, result.error().message());
			for (const std::string &reason : result.detailed_failure_reasons()) {
				errorMsg += "\n " + reason;
			}
			throw std::runtime_error(errorMsg);
		}

		return result.value();
		};

	vkb::SwapchainBuilder swapChainBuilder = vkb::SwapchainBuilder(_vkbData.device)
		.set_desired_extent(width, height)
		// set_desired_format inserts at the front, so RGBA is tried first and
		// BGRA is the fallback. Windows surfaces usually have RGBA; this NVIDIA
		// Linux surface only has BGRA.
		.set_desired_format({ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
		.set_desired_format({ VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }) //self gamma correction
		.add_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) //We want to write to the swap chain images with the compute pass
		.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);

	_vkbData.swapChain = vkbCheck(swapChainBuilder.build(), "Creating Vulkan swap chain");

	_swapChain = _vkbData.swapChain.swapchain;

	_swapChainImageFormat = _vkbData.swapChain.image_format;
	_swapChainExtent = { 
		.width = _vkbData.swapChain.extent.width, 
		.height = _vkbData.swapChain.extent.height
		};

	_swapChainImages = vkbCheck(_vkbData.swapChain.get_images(), "Getting swap chain images");

	if (_swapChainImages.size() > MAX_SWAPCHAIN_SIZE) throw std::runtime_error("Swapchain maximum size exceeded!");

	for (VkImage &swapChainImage : _swapChainImages) {
		VkImageViewCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		info.pNext = nullptr;
		info.image = swapChainImage;
		info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		info.format = _swapChainImageFormat;
		info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		info.subresourceRange.baseMipLevel = 0;
		info.subresourceRange.levelCount = 1;
		info.subresourceRange.baseArrayLayer = 0;
		info.subresourceRange.layerCount = 1;

		VkImageView view;
		VK_CHECK(vkCreateImageView(_device, &info, nullptr, &view));
		_swapChainImageViews.push_back(view);
		_swapChainImageViewCreateInfos.push_back(info);
	}

	VkFormatProperties3 props3{ .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 };
	VkFormatProperties2 props2{ .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &props3 };
	vkGetPhysicalDeviceFormatProperties2(_physicalDevice, _swapChainImageFormat, &props2);
	const bool unorm8 = _swapChainImageFormat == VK_FORMAT_R8G8B8A8_UNORM || _swapChainImageFormat == VK_FORMAT_B8G8R8A8_UNORM;
	const VkFormatFeatureFlags2 storageWrite = VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
	if (!unorm8 || (props3.optimalTilingFeatures & storageWrite) != storageWrite)
		throw std::runtime_error("Swapchain format is not an 8-bit UNORM format with storage support!");
}

void Engine::initSwapChainDescriptors() {
	for (int i = 0; i < _swapChainImageViewCreateInfos.size(); ++i) {
		writeImageDescriptor(ENGINE_IMAGES + i, _swapChainImageViewCreateInfos[i], 
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	}
	_allocator.flush(_imageHeap);
}

void Engine::createRadianceImage() {
	VkExtent3D extent = VkExtent3D(_renderExtent.width, _renderExtent.height, 1);

	_radianceImage = _allocator.createImage(extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		false, &_radianceImageViewCreateInfo);

	_allocator.immediateSubmit([&](VkCommandBuffer commandBuffer) {
		util::transitionImageLayout(commandBuffer, _radianceImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		});
}

void Engine::initRadianceImageDescriptors() {
	writeImageDescriptor(0, _radianceImageViewCreateInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	_allocator.flush(_imageHeap);
}

void Engine::createDLSSImages() {
	VkExtent3D extent = VkExtent3D(_renderExtent.width, _renderExtent.height, 1);
	
	//TRANSFER_SRC so the debug view can blit these directly into the swap chain
	constexpr VkImageUsageFlags debugViewUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

	_diffuseAlbedoImage = _allocator.createImage(extent, VK_FORMAT_R8G8B8A8_UNORM, debugViewUsage, false, &_diffuseAlbedoImageInfo);
	_specularAlbedoImage = _allocator.createImage(extent, VK_FORMAT_R8G8B8A8_UNORM, debugViewUsage, false, &_specularAlbedoImageInfo);
	_normalRoughnessImage = _allocator.createImage(extent, VK_FORMAT_R16G16B16A16_SFLOAT, debugViewUsage, false, &_normalRoughnessImageInfo);
	_viewDepthImage = _allocator.createImage(extent, VK_FORMAT_R32_SFLOAT, debugViewUsage, false, &_viewDepthImageInfo);
	_motionVectorImage = _allocator.createImage(extent, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false, &_motionVectorImageInfo);
	_specHitDistImage = _allocator.createImage(extent, VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false, &_specHitDistImageInfo);
	_dlssOutputImage = _allocator.createImage(extent, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false, &_dlssOutputImageInfo);
	_orcaAlbedoImage = _allocator.createImage(extent, VK_FORMAT_R8G8B8A8_UNORM, debugViewUsage, false, &_orcaAlbedoImageInfo);
	//TODO: Create additional images for DLSS (might want too look into the depth of field guide)

	//The shaders write these as storage images, so keep them in GENERAL
	_allocator.immediateSubmit([&](VkCommandBuffer commandBuffer) {
		for (VkImage image : { _diffuseAlbedoImage.image, _specularAlbedoImage.image, _normalRoughnessImage.image, _viewDepthImage.image,
			_motionVectorImage.image, _specHitDistImage.image, _dlssOutputImage.image, _orcaAlbedoImage.image }) {
			util::transitionImageLayout(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		}
		});
}

void Engine::destroyDLSSImages() {
	_allocator.destroyImage(_diffuseAlbedoImage);
	_allocator.destroyImage(_specularAlbedoImage);
	_allocator.destroyImage(_normalRoughnessImage);
	_allocator.destroyImage(_viewDepthImage);
	_allocator.destroyImage(_motionVectorImage);
	_allocator.destroyImage(_specHitDistImage);
	_allocator.destroyImage(_dlssOutputImage);
	_allocator.destroyImage(_orcaAlbedoImage);
	//_allocator.destroyImage(_postInputImage);
}

void Engine::initDLSSImageDescriptors() {
	writeImageDescriptor(1, _diffuseAlbedoImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	writeImageDescriptor(2, _specularAlbedoImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	writeImageDescriptor(3, _normalRoughnessImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	writeImageDescriptor(4, _viewDepthImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	writeImageDescriptor(5, _motionVectorImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	writeImageDescriptor(6, _specHitDistImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	writeImageDescriptor(7, _dlssOutputImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	writeImageDescriptor(8, _orcaAlbedoImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
	_allocator.flush(_imageHeap);
}

void Engine::initCommandResources() {
	VkCommandPoolCreateInfo commandPoolInfo{};

	commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolInfo.pNext = nullptr;
	commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolInfo.queueFamilyIndex = _queueFamilyIdx;

	//Initialize command pools and buffers for each frame in flight
	_frameResources.reserve(FRAMES_IN_FLIGHT);
	for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
		_frameResources.push_back({});
		
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frameResources[i]._commandPool));

		VkCommandBufferAllocateInfo commandBufferAllocInfo = init::commandBufferAllocateInfo(_frameResources[i]._commandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(_device, &commandBufferAllocInfo, &_frameResources[i]._commandBuffer));
	}

	//Initialize the immediate command pool and buffer for the engine
	VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

	VkCommandBufferAllocateInfo immCommandBufferAllocInfo = init::commandBufferAllocateInfo(_immCommandPool, 1);
	VK_CHECK(vkAllocateCommandBuffers(_device, &immCommandBufferAllocInfo, &_immCommandBuffer));

	_deletionQueue.push([this]() { 
		for (auto &frame : _frameResources) vkDestroyCommandPool(_device, frame._commandPool, nullptr);
		vkDestroyCommandPool(_device, _immCommandPool, nullptr);
		});
}

void Engine::initSyncStructures() {
	VkFenceCreateInfo fenceCreateInfo = init::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT); //Pre-signal fence so we don't deadlock immediately
	VkSemaphoreCreateInfo semaphoreCreateInfo = init::semaphoreCreateInfo();

	for (auto &frame : _frameResources) {
		VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &frame._renderFence));
		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &frame._swapChainSemaphore));
	}

	//Render complete semaphores are signalled when the rendering operations are done and the image is ready to present
	//Numbered by swap chain image because the semaphore is unsignalled when we get handed the swap chain image
	for (int i = 0; i < MAX_SWAPCHAIN_SIZE; ++i) {
		VkSemaphore semaphore;
		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &semaphore));
		_renderCompleteSemaphores.push_back(semaphore);
	}

	VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_sharcSemaphore));

	_deletionQueue.push([this]() {
		for (auto &frame : _frameResources) {
			vkDestroyFence(_device, frame._renderFence, nullptr);
			vkDestroySemaphore(_device, frame._swapChainSemaphore, nullptr);
		}

		for (VkSemaphore semaphore : _renderCompleteSemaphores) {
			vkDestroySemaphore(_device, semaphore, nullptr);
		}

		vkDestroyFence(_device, _immFence, nullptr);
		vkDestroySemaphore(_device, _sharcSemaphore, nullptr);
		});
}

void Engine::createDLSSFeature() {
	//TODO
}

void Engine::updateRenderExtent() {
#ifdef AQUARIUS_DLSS
	uint32_t optWidth, optHeight, minWidth, minHeight, maxWidth, maxHeight = 0;
	float sharpness = 0.f;

	NGX_DLSS_GET_OPTIMAL_SETTINGS(_ngxParams, _vkbData.swapChain.extent.width, _vkbData.swapChain.extent.height, NVSDK_NGX_PerfQuality_Value_MaxQuality,
		&optWidth, &optHeight, &maxWidth, &maxHeight, &minWidth, &minHeight, &sharpness);
	//_renderExtent.width = optWidth;
	//_renderExtent.height = optHeight;
	_renderExtent.width = _vkbData.swapChain.extent.width;
	_renderExtent.height = _vkbData.swapChain.extent.height;
#else
	_renderExtent = _swapChainExtent;
#endif

#if _DEBUG
	std::cout << "Render Extent: (" << _renderExtent.width << ", " << _renderExtent.height << ")" << std::endl;
	std::cout << "Swapchain Extent: (" << _swapChainExtent.width << ", " << _swapChainExtent.height << ")" << std::endl;
#endif
}

void Engine::initDescriptorHeaps() {
	const auto &dh = _deviceProperties.dhProperties;

	{
		const size_t slotCount = N_IMAGE_DESCRIPTORS;
		_imageHeapLayout.stride = util::alignUp(dh.imageDescriptorSize, dh.imageDescriptorAlignment);
		_imageHeapLayout.reservedOffset = util::alignUp(slotCount * _imageHeapLayout.stride, std::max(dh.imageDescriptorAlignment, dh.bufferDescriptorAlignment));
		_imageHeapLayout.reservedSize = dh.minResourceHeapReservedRange;
		_imageHeapLayout.size = _imageHeapLayout.reservedOffset + _imageHeapLayout.reservedSize;

		if (_imageHeapLayout.size > dh.maxResourceHeapSize) throw std::runtime_error("Image heap exceeds maxResourceHeapSize!");

		_imageHeap = _allocator.createBuffer(_imageHeapLayout.size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT,
			VMA_MEMORY_USAGE_CPU_TO_GPU, true, dh.resourceHeapAlignment);
	}

	{
		const size_t samplerCount = N_SAMPLER_DESCRIPTORS;
		_samplerHeapLayout.stride = util::alignUp(dh.samplerDescriptorSize, dh.samplerDescriptorAlignment);
		_samplerHeapLayout.reservedOffset = samplerCount * _samplerHeapLayout.stride;
		_samplerHeapLayout.reservedSize = dh.minSamplerHeapReservedRange;
		_samplerHeapLayout.size = _samplerHeapLayout.reservedOffset + _samplerHeapLayout.reservedSize;

		if (_samplerHeapLayout.size > dh.maxSamplerHeapSize) throw std::runtime_error("Sampler heap exceeds maxSamplerHeapSize!");

		_samplerHeap = _allocator.createBuffer(_samplerHeapLayout.size, VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT,
			VMA_MEMORY_USAGE_CPU_TO_GPU, true, dh.samplerHeapAlignment);
	}

	//Default sampler, in case a glTF has sampler = -1 for some texture
	VkSamplerCreateInfo defaultSamplerInfo{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	defaultSamplerInfo.magFilter = VK_FILTER_LINEAR;
	defaultSamplerInfo.minFilter = VK_FILTER_LINEAR;
	defaultSamplerInfo.maxLod = VK_LOD_CLAMP_NONE;
	
	VkHostAddressRangeEXT defaultRange{};
	defaultRange.address = _samplerHeap.info.pMappedData;
	defaultRange.size = dh.samplerDescriptorSize;
	VK_CHECK(_dispatchTable.writeSamplerDescriptorsEXT(1, &defaultSamplerInfo, &defaultRange));
	_allocator.flush(_samplerHeap);

	_deletionQueue.push([this]() {
		_allocator.destroyBuffer(_imageHeap);
		_allocator.destroyBuffer(_samplerHeap);
		});
}

void Engine::loadHdrImage(const std::string &path) {

	//Load the hdr using stb
	int width, height, numChannels;
	float *data = stbi_loadf(path.c_str(), &width, &height, &numChannels, 4);

	if (!data) {
		std::cout << "Failed to load hdr: " << path << std::endl;
		return;
	}
	if (_hdrLoaded) {
		vkDeviceWaitIdle(_device);
		_allocator.destroyImage(_hdrImage);
	} 

	VkExtent3D extent;
	extent.width = width;
	extent.height = height;
	extent.depth = 1;
	_hdrImage = _allocator.createImage(data, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, false, &_hdrImageViewCreateInfo, 4 * sizeof(float));
	
	std::vector<double> weights;
	weights.reserve(width * height);
	for (int i = 0; i < width * height; ++i) {
		//The texture gets squished at the poles
		double rowWeight = glm::sin((static_cast<double>(i / width) + .5f) / static_cast<double>(height) * PI);
		weights.push_back(rowWeight * glm::length(glm::vec3(data[4 * i], data[4 * i + 1], data[4 * i + 2])));
	}
	std::vector<float>pdf;
	std::vector<AliasTableEntry> envAliasTable = buildAliasTable(weights, pdf, &_sceneData.totalEnvWeight);

	if (_envAliasTable.loaded) _allocator.destroyBuffer(_envAliasTable);
	_envAliasTable = _allocator.uploadBuffer(envAliasTable.data(), envAliasTable.size() * sizeof(AliasTableEntry), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	_sceneData.envAliasTable = _envAliasTable.address;
	if (_sceneDataBuffer.loaded) _allocator.destroyBuffer(_sceneDataBuffer);
	_sceneDataBuffer = _allocator.uploadBuffer(&_sceneData, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	_hdrLoaded = true;
	stbi_image_free(data);
	writeImageDescriptor(ENGINE_IMAGES + MAX_SWAPCHAIN_SIZE, _hdrImageViewCreateInfo, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	_allocator.flush(_imageHeap);
}

void Engine::writeSceneDescriptors() {
	const auto &dh = _deviceProperties.dhProperties;

	if (_scene.images.size() > N_IMAGE_DESCRIPTORS - ENGINE_IMAGES - MAX_SWAPCHAIN_SIZE - HDR_IMAGES) throw std::runtime_error("Number of images in scene exceeds maximum!");
	if (_scene.samplerCreateInfos.size() > N_SAMPLER_DESCRIPTORS - 1) throw std::runtime_error("Number of samplers in scene exceeds maximum!");

	for (size_t i = 0; i < _scene.images.size(); ++i) {
		writeImageDescriptor(ENGINE_IMAGES + HDR_IMAGES + MAX_SWAPCHAIN_SIZE + i, _scene.imageViewCreateInfos[i], VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
	_allocator.flush(_imageHeap);

	if (!_scene.samplerCreateInfos.empty()) {
		std::vector<VkHostAddressRangeEXT> hostAddressRanges;
		hostAddressRanges.reserve(_scene.samplerCreateInfos.size());
		for (int i = 0; i < _scene.samplerCreateInfos.size(); ++i) {
			VkHostAddressRangeEXT hostAddressRange{};
			hostAddressRange.address = reinterpret_cast<char *>(_samplerHeap.info.pMappedData) + (1 + i) * _samplerHeapLayout.stride;
			hostAddressRange.size = dh.samplerDescriptorSize;

			hostAddressRanges.push_back(hostAddressRange);
		}
		VK_CHECK(_dispatchTable.writeSamplerDescriptorsEXT(_scene.samplerCreateInfos.size(), _scene.samplerCreateInfos.data(), hostAddressRanges.data()));
		_allocator.flush(_samplerHeap);
	}
}

void Engine::writeImageDescriptor(uint32_t slot, const VkImageViewCreateInfo &viewInfo, VkDescriptorType type, VkImageLayout layout) {
	VkImageDescriptorInfoEXT imageDescriptorInfo{ .sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT };
	imageDescriptorInfo.pView = &viewInfo;
	imageDescriptorInfo.layout = layout;

	VkResourceDescriptorInfoEXT resourceDescriptorInfo{ .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT };
	resourceDescriptorInfo.type = type;
	resourceDescriptorInfo.data.pImage = &imageDescriptorInfo;

	VkHostAddressRangeEXT hostAddressRange{};
	hostAddressRange.address = reinterpret_cast<char *>(_imageHeap.info.pMappedData) + slot * _imageHeapLayout.stride;
	hostAddressRange.size = _deviceProperties.dhProperties.imageDescriptorSize;

	VK_CHECK(_dispatchTable.writeResourceDescriptorsEXT(1, &resourceDescriptorInfo, &hostAddressRange));
}

void Engine::loadScene(const std::string &file) {
	vkDeviceWaitIdle(_device);

	if (_sceneDataBuffer.loaded) _allocator.destroyBuffer(_sceneDataBuffer);

	if (_scene.loaded) loader::destroyScene(_device, _allocator, _scene, _dispatchTable);

	//TODO: Reset scene descriptors?

	_scene = loader::loadScene(_device, _allocator, file, _deviceProperties, _dispatchTable);

	//Set up scene data struct
	if (_sceneDataBuffer.loaded) _allocator.destroyBuffer(_sceneDataBuffer);
	_sceneData.deviceSubMeshes = _scene.subMeshBuffer.address;
	_sceneData.materials = _scene.materialBuffer.address;
	_sceneData.lightAliasTable = _scene.lightStrengthAliasTable.address;
	_sceneData.lightTriangles = _scene.lightTriangles.address;
	_sceneData.totalLightWeight = _scene.totalLightWeight;
	_sceneData.nLights = _scene.nLights;
	_sceneDataBuffer = _allocator.uploadBuffer(&_sceneData, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	writeSceneDescriptors();
}

void Engine::initPipelinePathtracer() {
	//Initialize Vulkan raytracing objects

	VkShaderModule modulePathtracer = util::loadShaderModule("./build/shaders/pathtracer.spv", _device);

	VkDeviceSize imageStride = _imageHeapLayout.stride;

	//Create the bindings for the resources
	std::array mappings = {
		//Set 0 is for SHaRC
		heapMapping(0, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 0 * imageStride, 0), //Radiance Image
		heapMapping(1, 0, VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT, (ENGINE_IMAGES + MAX_SWAPCHAIN_SIZE + HDR_IMAGES) * imageStride, imageStride), //Scene images
		heapMapping(2, 0, VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT, 0, _samplerHeapLayout.stride), //Samplers
		pushAddressMapping(3, 0, VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT, offsetof(PushDataSharc, pathtracer) + offsetof(PushConstantsPathtracer, tlas)), //Tlas
		heapMapping(4, 0, VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT, (ENGINE_IMAGES + MAX_SWAPCHAIN_SIZE) * imageStride, imageStride), //Hdr image
		pushAddressMapping(5, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, hashEntriesAddress)),
		pushAddressMapping(6, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, accumulationAddress)),
		pushAddressMapping(7, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, resolvedAddress)),
		pushAddressMapping(8, 0, VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, configAddress)),
		pushAddressMapping(9, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, lockAddress)),
		//Set 1 is for the DLSS/ORCA images
		heapMapping(0, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 1 * imageStride, 0), //Diffuse albedo
		heapMapping(1, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 2 * imageStride, 0), //Specular albedo
		heapMapping(2, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 3 * imageStride, 0), //Normal/roughness
		heapMapping(3, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 4 * imageStride, 0), //Depth
		heapMapping(4, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 5 * imageStride, 0), //Motion vectors
		heapMapping(5, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 6 * imageStride, 0), //Specular hit distance
		heapMapping(6, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 7 * imageStride, 0), //DLSS output
		heapMapping(7, 1, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 8 * imageStride, 0), //ORCA albedo
	};

	VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT };
	mappingInfo.mappingCount = static_cast<uint32_t>(mappings.size());
	mappingInfo.pMappings = mappings.data();

	//Lambda that defines a shader stage
	auto stageInfo = [&](VkShaderStageFlagBits flag, const char *entry) {
		VkPipelineShaderStageCreateInfo info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		info.pNext = &mappingInfo;
		info.stage = flag;
		info.module = modulePathtracer;
		info.pName = entry;
		return info;
		};

	std::array stages = {
		stageInfo(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "rayGeneration"),
		stageInfo(VK_SHADER_STAGE_MISS_BIT_KHR, "miss"),
		stageInfo(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "closestHit"),
		stageInfo(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, "anyHit")
		};

	//A VkRayTracingShaderGroupTypeKHR bundles together related ray tracing shader stages
	auto group = [](VkRayTracingShaderGroupTypeKHR type, uint32_t general, uint32_t closestHit, uint32_t anyHit) {
		VkRayTracingShaderGroupCreateInfoKHR info{ .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
		info.type = type;
		info.generalShader = general;
		info.closestHitShader = closestHit;
		info.anyHitShader = anyHit;
		info.intersectionShader = VK_SHADER_UNUSED_KHR; //Apparently this is for custom procedural geometry/non-triangular primitives...
		return info;
		};

	//General is for standalone shaders that don't directly participate in a geometry hit/miss lookup
	//Triangle hit is pretty true to the name.
	std::array groups = {
		group(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR), //Set the ray generation shader
		group(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR), //Set the miss shader
		group(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR), //Set the closest and any hit shaders
	};

	VkPipelineCreateFlags2CreateInfo flags{.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO};
	flags.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

	VkRayTracingPipelineCreateInfoKHR pipelineInfo{ .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	pipelineInfo.pNext = &flags;
	pipelineInfo.stageCount = stages.size();
	pipelineInfo.pStages = stages.data();
	pipelineInfo.groupCount = groups.size();
	pipelineInfo.pGroups = groups.data();
	pipelineInfo.maxPipelineRayRecursionDepth = 2;
	pipelineInfo.layout = VK_NULL_HANDLE; //We're using descriptor heaps

	VK_CHECK(_dispatchTable.createRayTracingPipelinesKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_pipelinePathtracer));

	vkDestroyShaderModule(_device, modulePathtracer, nullptr);

	_deletionQueue.push([this]() {vkDestroyPipeline(_device, _pipelinePathtracer, nullptr); });

	//Create the shader binding table
	auto &rt = _deviceProperties.rtProperties;
	uint32_t handleSize = rt.shaderGroupHandleSize;
	VkDeviceSize handleStride = util::alignUp(handleSize, rt.shaderGroupHandleAlignment);

	_rayGenerationRegion.stride = util::alignUp(handleStride, rt.shaderGroupBaseAlignment);
	_rayGenerationRegion.size = _rayGenerationRegion.stride;

	_missRegion.stride = handleStride;
	_missRegion.size = util::alignUp(handleStride, rt.shaderGroupBaseAlignment);

	_hitRegion.stride = handleStride;
	_hitRegion.size = util::alignUp(handleStride, rt.shaderGroupBaseAlignment);

	std::vector<uint8_t> handles(groups.size() * handleSize);
	VK_CHECK(_dispatchTable.getRayTracingShaderGroupHandlesKHR(_pipelinePathtracer, 0, groups.size(), handles.size(), handles.data()));

	_sbtBuffer = _allocator.createBuffer(_rayGenerationRegion.size + _missRegion.size + _hitRegion.size + _callRegion.size,
		VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU, true, rt.shaderGroupBaseAlignment);

	//Write the handles into the shader binding table
	//Shaders hold records (handles and data), where handles reference groups (bundled shaders called together)
	uint8_t *sbt = reinterpret_cast<uint8_t *> (_sbtBuffer.info.pMappedData);
	memcpy(sbt, handles.data() + 0 * handleSize, handleSize);
	memcpy(sbt + _rayGenerationRegion.size, handles.data() + 1 * handleSize, handleSize);
	memcpy(sbt + _rayGenerationRegion.size + _missRegion.size, handles.data() + 2 * handleSize, handleSize);
	_allocator.flush(_sbtBuffer);

	_rayGenerationRegion.deviceAddress = _sbtBuffer.address;
	_missRegion.deviceAddress = _sbtBuffer.address + _rayGenerationRegion.size;
	_hitRegion.deviceAddress = _sbtBuffer.address + _rayGenerationRegion.size + _missRegion.size;
	_deletionQueue.push([this]() { _allocator.destroyBuffer(_sbtBuffer); });
}

void Engine::initPipelineUpdate() {
	//Initialize Vulkan raytracing objects

	VkShaderModule moduleUpdate = util::loadShaderModule("./build/shaders/slang.spv", _device);

	VkDeviceSize imageStride = _imageHeapLayout.stride;

	//Create the bindings for the resources
	std::array mappings = {
		heapMapping(0, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 0 * imageStride, 0), //Radiance Image
		heapMapping(1, 0, VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT, (ENGINE_IMAGES + MAX_SWAPCHAIN_SIZE + HDR_IMAGES) * imageStride, imageStride), //Scene images
		heapMapping(2, 0, VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT, 0, _samplerHeapLayout.stride), //Samplers
		pushAddressMapping(3, 0, VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT, offsetof(PushDataSharc, pathtracer) + offsetof(PushConstantsPathtracer, tlas)), //Tlas
		heapMapping(4, 0, VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT, (ENGINE_IMAGES + MAX_SWAPCHAIN_SIZE) * imageStride, imageStride), //Hdr image
		pushAddressMapping(5, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, hashEntriesAddress)),
		pushAddressMapping(6, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, accumulationAddress)),
		pushAddressMapping(7, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, resolvedAddress)),
		pushAddressMapping(8, 0, VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, configAddress)),
		pushAddressMapping(9, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushDataSharc, lockAddress)),
		pushAddressMapping(0, 2, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, _hashMap),
		pushAddressMapping(1, 2, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, _hashMap),
		pushAddressMapping(2, 2, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, _hashMapValuesBuffer),
	};

	VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT };
	mappingInfo.mappingCount = static_cast<uint32_t>(mappings.size());
	mappingInfo.pMappings = mappings.data();

	//Lambda that defines a shader stage
	auto stageInfo = [&](VkShaderStageFlagBits flag, const char *entry) {
		VkPipelineShaderStageCreateInfo info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
		info.pNext = &mappingInfo;
		info.stage = flag;
		info.module = moduleUpdate;
		info.pName = entry;
		return info;
		};

	std::array stages = {
		stageInfo(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "rayGeneration"),
		stageInfo(VK_SHADER_STAGE_MISS_BIT_KHR, "miss"),
		stageInfo(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "closestHit"),
		stageInfo(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, "anyHit")
		};

	//A VkRayTracingShaderGroupTypeKHR bundles together related ray tracing shader stages
	auto group = [](VkRayTracingShaderGroupTypeKHR type, uint32_t general, uint32_t closestHit, uint32_t anyHit) {
		VkRayTracingShaderGroupCreateInfoKHR info{ .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
		info.type = type;
		info.generalShader = general;
		info.closestHitShader = closestHit;
		info.anyHitShader = anyHit;
		info.intersectionShader = VK_SHADER_UNUSED_KHR; //Apparently this is for custom procedural geometry/non-triangular primitives...
		return info;
		};

	//General is for standalone shaders that don't directly participate in a geometry hit/miss lookup
	//Triangle hit is pretty true to the name.
	std::array groups = {
		group(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR), //Set the ray generation shader
		group(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR), //Set the miss shader
		group(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR), //Set the closest and any hit shaders
	};

	VkPipelineCreateFlags2CreateInfo flags{.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO};
	flags.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

	VkRayTracingPipelineCreateInfoKHR pipelineInfo{ .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	pipelineInfo.pNext = &flags;
	pipelineInfo.stageCount = stages.size();
	pipelineInfo.pStages = stages.data();
	pipelineInfo.groupCount = groups.size();
	pipelineInfo.pGroups = groups.data();
	pipelineInfo.maxPipelineRayRecursionDepth = 2;
	pipelineInfo.layout = VK_NULL_HANDLE; //We're using descriptor heaps

	VK_CHECK(_dispatchTable.createRayTracingPipelinesKHR(VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_pipelineSharcUpdate));

	vkDestroyShaderModule(_device, moduleUpdate, nullptr);

	_deletionQueue.push([this]() {vkDestroyPipeline(_device, _pipelineSharcUpdate, nullptr); });

	//Create the shader binding table
	auto &rt = _deviceProperties.rtProperties;
	uint32_t handleSize = rt.shaderGroupHandleSize;
	VkDeviceSize handleStride = util::alignUp(handleSize, rt.shaderGroupHandleAlignment);

	_sharcUpdateRayGenerationRegion.stride = util::alignUp(handleStride, rt.shaderGroupBaseAlignment);
	_sharcUpdateRayGenerationRegion.size = _sharcUpdateRayGenerationRegion.stride;

	_sharcUpdateMissRegion.stride = handleStride;
	_sharcUpdateMissRegion.size = util::alignUp(handleStride, rt.shaderGroupBaseAlignment);

	_sharcUpdateHitRegion.stride = handleStride;
	_sharcUpdateHitRegion.size = util::alignUp(handleStride, rt.shaderGroupBaseAlignment);

	std::vector<uint8_t> handles(groups.size() * handleSize);
	VK_CHECK(_dispatchTable.getRayTracingShaderGroupHandlesKHR(_pipelineSharcUpdate, 0, groups.size(), handles.size(), handles.data()));

	_sharcUpdateSbtBuffer = _allocator.createBuffer(_sharcUpdateRayGenerationRegion.size + _sharcUpdateMissRegion.size + _sharcUpdateHitRegion.size + _sharcUpdateCallRegion.size,
		VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU, true, rt.shaderGroupBaseAlignment);

	//Write the handles into the shader binding table
	//Shaders hold records (handles and data), where handles reference groups (bundled shaders called together)
	uint8_t *sbt = reinterpret_cast<uint8_t *> (_sharcUpdateSbtBuffer.info.pMappedData);
	memcpy(sbt, handles.data() + 0 * handleSize, handleSize);
	memcpy(sbt + _sharcUpdateRayGenerationRegion.size, handles.data() + 1 * handleSize, handleSize);
	memcpy(sbt + _sharcUpdateRayGenerationRegion.size + _sharcUpdateMissRegion.size, handles.data() + 2 * handleSize, handleSize);
	_allocator.flush(_sharcUpdateSbtBuffer);

	_sharcUpdateRayGenerationRegion.deviceAddress = _sharcUpdateSbtBuffer.address;
	_sharcUpdateMissRegion.deviceAddress = _sharcUpdateSbtBuffer.address + _sharcUpdateRayGenerationRegion.size;
	_sharcUpdateHitRegion.deviceAddress = _sharcUpdateSbtBuffer.address + _sharcUpdateRayGenerationRegion.size + _sharcUpdateMissRegion.size;
	_deletionQueue.push([this]() { _allocator.destroyBuffer(_sharcUpdateSbtBuffer); });
}

void Engine::createSharcBuffers() {
	const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	_sharcHashEntries = _allocator.createBuffer(VkDeviceSize(SHARC_CAPACITY) * 8, usage, VMA_MEMORY_USAGE_GPU_ONLY, false);
	_sharcAccumulation = _allocator.createBuffer(VkDeviceSize(SHARC_CAPACITY) * 16, usage, VMA_MEMORY_USAGE_GPU_ONLY, false);
	_sharcResolved = _allocator.createBuffer(VkDeviceSize(SHARC_CAPACITY) * 16, usage, VMA_MEMORY_USAGE_GPU_ONLY, false);
	_sharcLockBuffer = _allocator.createBuffer(VkDeviceSize(SHARC_CAPACITY) * sizeof(uint32_t), usage, VMA_MEMORY_USAGE_GPU_ONLY, false);
	for (auto &frame : _frameResources) {
		frame._sharcConfigBuffer = _allocator.createBuffer(sizeof(SharcConfig), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	}

	_deletionQueue.push([this]() {
		_allocator.destroyBuffer(_sharcHashEntries);
		_allocator.destroyBuffer(_sharcAccumulation);
		_allocator.destroyBuffer(_sharcResolved);
		_allocator.destroyBuffer(_sharcLockBuffer);
		for (auto &frame : _frameResources) _allocator.destroyBuffer(frame._sharcConfigBuffer);
		});

	// Hash key 0 is an empty slot. Zeroed accumulation and resolved records match.
	_allocator.immediateSubmit([&](VkCommandBuffer commandBuffer) {
		vkCmdFillBuffer(commandBuffer, _sharcHashEntries.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(commandBuffer, _sharcAccumulation.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(commandBuffer, _sharcResolved.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(commandBuffer, _sharcLockBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

		// Make the initial clears visible to the first update/resolve dispatches.
		VkMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		VkDependencyInfo dependency{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers = &barrier;
		vkCmdPipelineBarrier2(commandBuffer, &dependency);
		});
}

void Engine::updateSharc(VkCommandBuffer commandBuffer) {
	SharcConfig config{};
	config.sceneScale = _sharcSceneScale;
	auto &configBuffer = getCurrFrame()._sharcConfigBuffer;
	memcpy(configBuffer.info.pMappedData, &config, sizeof(config));
	_allocator.flush(configBuffer);

	PushDataSharc push{};
	push.pathtracer = _pcpt;
	push.hashEntriesAddress = _sharcHashEntries.address;
	push.accumulationAddress = _sharcAccumulation.address;
	push.resolvedAddress = _sharcResolved.address;
	push.configAddress = configBuffer.address;
	push.lockAddress = _sharcLockBuffer.address;

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _pipelineSharcUpdate);
	VkPushDataInfoEXT pushData{ .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT };
	pushData.data = { &push, sizeof(push) };
	_dispatchTable.cmdPushDataEXT(commandBuffer, &pushData);

	// Sparse pathtracing dispatch
	_dispatchTable.cmdTraceRaysKHR(commandBuffer, &_sharcUpdateRayGenerationRegion, &_sharcUpdateMissRegion, &_sharcUpdateHitRegion, &_sharcUpdateCallRegion,
		(_renderExtent.width + SHARC_UPDATE_TILE_SIZE - 1) / SHARC_UPDATE_TILE_SIZE,
		(_renderExtent.height + SHARC_UPDATE_TILE_SIZE - 1) / SHARC_UPDATE_TILE_SIZE, 1);
}

void Engine::initPipelineResolve() {
	VkShaderModule moduleResolve = util::loadShaderModule("./build/shaders/resolve.spv", _device);

	std::array mappings = {
		pushAddressMapping(0, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushConstantsResolve, hashEntriesAddress)),
		pushAddressMapping(1, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushConstantsResolve, accumulationAddress)),
		pushAddressMapping(2, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT, offsetof(PushConstantsResolve, resolvedAddress))
	};

	VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT };
	mappingInfo.mappingCount = static_cast<uint32_t>(mappings.size());
	mappingInfo.pMappings = mappings.data();

	VkPipelineCreateFlags2CreateInfo flags{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO };
	flags.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

	VkPipelineShaderStageCreateInfo shaderStage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	shaderStage.pNext = &mappingInfo;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.module = moduleResolve;
	shaderStage.pName = "main";

	VkComputePipelineCreateInfo pipelineInfo{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	pipelineInfo.pNext = &flags;
	pipelineInfo.stage = shaderStage;
	pipelineInfo.layout = VK_NULL_HANDLE;

	VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_pipelineResolve));

	vkDestroyShaderModule(_device, moduleResolve, nullptr);

	_deletionQueue.push([this]() { vkDestroyPipeline(_device, _pipelineResolve, nullptr); });
}

void Engine::resolveSharc(VkCommandBuffer commandBuffer) {
	const glm::vec3 cameraPosition = _pcpt.camPos;
	if (!_sharcCameraInitialized) {
		_sharcCameraPositionPrev = cameraPosition;
		_sharcCameraInitialized = true;
	}

	const std::array<VkBuffer, 3> cacheBuffers = {
		_sharcHashEntries.buffer,
		_sharcAccumulation.buffer,
		_sharcResolved.buffer
	};

	// Update (ray tracing) writes the cache before this dispatch. The render pass
	// reads the resolved records afterwards. Both sides are still stubs, so the
	// barriers are what keeps those passes ordered once they land in this buffer.
	auto cacheDependency = [&](VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
		VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
			std::array<VkBufferMemoryBarrier2, 3> barriers{};
			for (size_t i = 0; i < cacheBuffers.size(); ++i) {
				barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
				barriers[i].srcStageMask = srcStage;
				barriers[i].srcAccessMask = srcAccess;
				barriers[i].dstStageMask = dstStage;
				barriers[i].dstAccessMask = dstAccess;
				barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barriers[i].buffer = cacheBuffers[i];
				barriers[i].size = VK_WHOLE_SIZE;
			}

			VkDependencyInfo dependency{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
			dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
			dependency.pBufferMemoryBarriers = barriers.data();
			vkCmdPipelineBarrier2(commandBuffer, &dependency);
		};

	cacheDependency(
		VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
		VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

	PushConstantsResolve pc{};
	pc.cameraPosition = cameraPosition;
	pc.logarithmBase = SHARC_LOGARITHM_BASE;
	pc.sceneScale = _sharcSceneScale;
	pc.levelBias = SHARC_LEVEL_BIAS;
	pc.cameraPositionPrev = _sharcCameraPositionPrev;
	pc.radianceScale = SHARC_RADIANCE_SCALE;
	pc.hashEntriesAddress = _sharcHashEntries.address;
	pc.accumulationAddress = _sharcAccumulation.address;
	pc.resolvedAddress = _sharcResolved.address;
	pc.capacity = SHARC_CAPACITY;
	pc.accumulationFrameNum = _sharcAccumulationFrameNum;
	pc.responsiveFrameNum = _sharcAccumulationFrameNum;
	pc.staleFrameNumMax = _sharcStaleFrameNumMax;
	pc.frameIndex = _totalFrames;

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineResolve);

	VkPushDataInfoEXT pushData{ .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT };
	pushData.offset = 0;
	pushData.data = { &pc, sizeof(PushConstantsResolve) };
	_dispatchTable.cmdPushDataEXT(commandBuffer, &pushData);

	const uint32_t groups = (SHARC_CAPACITY + SHARC_RESOLVE_BLOCK_SIZE - 1) / SHARC_RESOLVE_BLOCK_SIZE;
	vkCmdDispatch(commandBuffer, groups, 1, 1);

	cacheDependency(
		VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

	_sharcCameraPositionPrev = cameraPosition;
}

void Engine::initPipelinePost() {
	VkShaderModule modulePost = util::loadShaderModule("./build/shaders/post.spv", _device);

	VkDeviceSize imageStride = _imageHeapLayout.stride;

	std::array mappings = {
	heapMapping(0, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, 0 * imageStride, 0),
	heapMapping(1, 0, VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT, ENGINE_IMAGES * imageStride, imageStride)
	};

	VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT };
	mappingInfo.mappingCount = mappings.size();
	mappingInfo.pMappings = mappings.data();

	VkPipelineCreateFlags2CreateInfo flags{ .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO };
	flags.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

	VkPipelineShaderStageCreateInfo shaderStage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	shaderStage.pNext = &mappingInfo;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.module = modulePost;
	shaderStage.pName = "main";

	VkComputePipelineCreateInfo pipelineInfo{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	pipelineInfo.pNext = &flags;
	pipelineInfo.stage = shaderStage;
	pipelineInfo.layout = VK_NULL_HANDLE;

	VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_pipelinePost));

	vkDestroyShaderModule(_device, modulePost, nullptr);

	_deletionQueue.push([this]() {vkDestroyPipeline(_device, _pipelinePost, nullptr); });
}

void Engine::initGPUHashMap() {
	_hashMapCheckSumBuffer = _allocator.createBuffer(NUM_HASH_MAP_ENTRIES * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, false);
	_hashMapCountsBuffer = _allocator.createBuffer(NUM_HASH_MAP_ENTRIES * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, false);
	_hashMapValuesBuffer = _allocator.createBuffer(NUM_HASH_MAP_ENTRIES * sizeof(glm::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, false);

	_hashMapUniformBuffer = _allocator.createBuffer(sizeof(HashMap), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	
	HashMap hashMap{ _hashMapCheckSumBuffer.address, _hashMapCountsBuffer.address, _hashMapValuesBuffer.address };
	memcpy(_hashMapUniformBuffer.info.pMappedData, &hashMap, sizeof(HashMap));
	_allocator.flush(_hashMapUniformBuffer);

	_deletionQueue.push([=]() {
		_allocator.destroyBuffer(_hashMapCheckSumBuffer);
		_allocator.destroyBuffer(_hashMapCountsBuffer);
		_allocator.destroyBuffer(_hashMapValuesBuffer);

		_allocator.destroyBuffer(_hashMapUniformBuffer);
		});
}

void Engine::initImgui() {
	ImGui::CreateContext();

	ImGui_ImplSDL3_InitForVulkan(_window);
	ImGui_ImplVulkan_InitInfo imGuiInfo{};
	imGuiInfo.Instance = _instance;
	imGuiInfo.PhysicalDevice = _physicalDevice;
	imGuiInfo.Device = _device;
	imGuiInfo.Queue = _queue;
	imGuiInfo.QueueFamily = _queueFamilyIdx;
	imGuiInfo.DescriptorPool = VK_NULL_HANDLE; //Thank you, Nathan
	imGuiInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
	imGuiInfo.MinImageCount = 3;
	imGuiInfo.ImageCount = _swapChainImages.size();
	imGuiInfo.UseDynamicRendering = true;

	imGuiInfo.PipelineInfoMain.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
	imGuiInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
	imGuiInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapChainImageFormat;

	imGuiInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	ImGui_ImplVulkan_Init(&imGuiInfo);

	_io = &ImGui::GetIO();

	_deletionQueue.push([this]() {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL3_Shutdown();
		ImGui::DestroyContext();
		});
}

void Engine::drawImgui(VkCommandBuffer commandBuffer, VkImageView targetImageView) {
	//mouseOverImGuiWindow = io->WantCaptureMouse;

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL3_NewFrame();
	ImGui::NewFrame();

	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	static float f = 0.0f;
	static int counter = 0;

	ImGui::Begin("Aquarius Engine Analytics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

	//Debug view to make sure the DLSS/GBuffer images are appearing properly
	std::vector<const char *> debugViewOptions = {"Pathtraced", "Diffuse Albedo", "Specular Albedo", "Normals", "Depth", "ORCA Albedo"};
	const char *selected = debugViewOptions[static_cast<int>(_debugView)];

	if (ImGui::BeginCombo("Debug view", selected)) {
		for (uint8_t i = 0; i < debugViewOptions.size(); ++i) {
			bool isSelected = selected == debugViewOptions[i];
			if (ImGui::Selectable(debugViewOptions[i], isSelected)) selected = debugViewOptions[i];

			isSelected = selected == debugViewOptions[i];
			if (isSelected) _debugView = static_cast<EngineDebugView>(i);
		}
		ImGui::EndCombo();
	}

	int maxBounces = _pcpt.maxBounces;
	ImGui::SliderInt("Max bounces", &maxBounces, 1, 20);
	if (maxBounces != _pcpt.maxBounces) _frameNum = 0;
	_pcpt.maxBounces = maxBounces;

	float focusDist = _pcpt.focusDist;
	ImGui::SliderFloat("Focus distance", &focusDist, 1.f, 20.f);
	if (focusDist != _pcpt.focusDist) _frameNum = 0;
	_pcpt.focusDist = focusDist;

	float lensRadius = _pcpt.lensRadius;
	ImGui::SliderFloat("Lens radius", &lensRadius, 0.f, 1.f);
	if (lensRadius != _pcpt.lensRadius) _frameNum = 0;
	_pcpt.lensRadius = lensRadius;

	int accumulationFrames = static_cast<int>(_sharcAccumulationFrameNum);
	ImGui::SliderInt("SHaRC accumulation frames", &accumulationFrames, 1, 128);
	if (static_cast<uint32_t>(accumulationFrames) != _sharcAccumulationFrameNum) _frameNum = 0;
	_sharcAccumulationFrameNum = static_cast<uint32_t>(accumulationFrames);

	int staleFrames = static_cast<int>(_sharcStaleFrameNumMax);
	ImGui::SliderInt("SHaRC stale frames", &staleFrames, 8, 256);
	if (static_cast<uint32_t>(staleFrames) != _sharcStaleFrameNumMax) _frameNum = 0;
	_sharcStaleFrameNumMax = static_cast<uint32_t>(staleFrames);

	if (ImGui::SliderFloat("SHaRC scene scale", &_sharcSceneScale, 1.f, 200.f)) _frameNum = 0;

	float theta = _camera.getTheta();
	ImGui::InputFloat("Theta", &theta);

	float phi = _camera.getPhi();
	ImGui::InputFloat("Phi", &phi);

	glm::vec3 origin = _camera.getOrigin();
	ImGui::InputFloat3("Origin", reinterpret_cast<float *>(&origin));

	_camera.setCamera(origin, theta, phi);

	ImGui::End();
	ImGui::Render();

	VkRenderingAttachmentInfo colorAttachmentInfo = init::renderingAttachmentInfo(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkRenderingInfo renderingInfo = init::renderingInfo(_swapChainExtent, &colorAttachmentInfo, nullptr);
	
	vkCmdBeginRendering(commandBuffer, &renderingInfo);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
	vkCmdEndRendering(commandBuffer);
}

void Engine::draw() {
	FrameResources &currFrame = getCurrFrame();

	VK_CHECK(vkWaitForFences(_device, 1, &currFrame._renderFence, true, 1000000000)); //Wait for 1 fence (the fence of the current frame) for up to 1 second
	
	//Get image from swapchain
	uint32_t swapChainImageIndex;

	VkResult result = vkAcquireNextImageKHR(_device, _swapChain, 1000000000, currFrame._swapChainSemaphore, nullptr, &swapChainImageIndex); //Signals the swapchain semaphore when complete

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) { //Swapchain is out of date due to needing to be resized
		_resize = true;
		return;
	}

	VK_CHECK(vkResetFences(_device, 1, &currFrame._renderFence));
	
	VkCommandBuffer commandBuffer = currFrame._commandBuffer;

	VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));

	//Since we're re-recording every frame
	VkCommandBufferBeginInfo beginInfo = init::commandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	//Bind out descriptor heaps
	//TODO: Consider caching?
	VkBindHeapInfoEXT imageHeapBind{ .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT };
	imageHeapBind.heapRange = { _imageHeap.address, _imageHeapLayout.size };
	imageHeapBind.reservedRangeOffset = _imageHeapLayout.reservedOffset;
	imageHeapBind.reservedRangeSize = _imageHeapLayout.reservedSize;
	_dispatchTable.cmdBindResourceHeapEXT(commandBuffer, &imageHeapBind);

	VkBindHeapInfoEXT samplerHeapBind{ .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT };
	samplerHeapBind.heapRange = { _samplerHeap.address, _samplerHeapLayout.size };
	samplerHeapBind.reservedRangeOffset = _samplerHeapLayout.reservedOffset;
	samplerHeapBind.reservedRangeSize = _samplerHeapLayout.reservedSize;
	_dispatchTable.cmdBindSamplerHeapEXT(commandBuffer, &samplerHeapBind);

	//Call the pathtracer
	if (_camera.wasDirty()) _frameNum = 0; //Reset the frame if the camera's view or rotation changed

	if (_scene.animated) {
		_frameNum = 0;

		auto now = std::chrono::steady_clock::now();
		auto time = std::chrono::duration<float>(now - _start).count();

		int frameIdx = _totalFrames % FRAMES_IN_FLIGHT;

		loader::updateScene(_device, _allocator, _scene, _dispatchTable, time, frameIdx);

		//Rebuild the TLAS
		VkAccelerationStructureGeometryKHR tlasGeometry = _scene.tlas.geometry;
		tlasGeometry.geometry.instances.data.deviceAddress = _scene.instanceBuffers[frameIdx].address;

		VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo = _scene.tlas.buildInfo;
		tlasBuildInfo.pGeometries = &tlasGeometry;

		uint32_t instanceCount = _scene.instanceNodes.size();

		VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
		tlasRange.primitiveCount = instanceCount;

		//The TLAS and its scratch buffer are shared by every frame in flight
		VkMemoryBarrier2 preBuildBarrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
		preBuildBarrier.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
		preBuildBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		preBuildBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
		preBuildBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		VkDependencyInfo preBuildDependency{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		preBuildDependency.memoryBarrierCount = 1;
		preBuildDependency.pMemoryBarriers = &preBuildBarrier;
		vkCmdPipelineBarrier2(commandBuffer, &preBuildDependency);

		const VkAccelerationStructureBuildRangeInfoKHR *pTlasRange = &tlasRange;
		_dispatchTable.cmdBuildAccelerationStructuresKHR(commandBuffer, 1, &tlasBuildInfo, &pTlasRange);

		VkMemoryBarrier2 barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
		barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
		barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
		VkDependencyInfo dependency{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers = &barrier;

		vkCmdPipelineBarrier2(commandBuffer, &dependency);
	}
		
	_camera.toPushConstantsPathtracer(_pcpt);
	_pcpt.frameNum = _frameNum;
	_pcpt.tlas = _scene.tlas.address;
	_pcpt.sceneData = _sceneDataBuffer.address;
	_pcpt.totalFrames = _totalFrames;

	// SHaRC order is update, resolve, then the render query.
	updateSharc(commandBuffer);
	resolveSharc(commandBuffer);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _pipelinePathtracer);

	PushDataSharc pushPt{};
	pushPt.pathtracer = _pcpt;
	pushPt.hashEntriesAddress = _sharcHashEntries.address;
	pushPt.accumulationAddress = _sharcAccumulation.address;
	pushPt.resolvedAddress = _sharcResolved.address;
	pushPt.configAddress = getCurrFrame()._sharcConfigBuffer.address;
	pushPt.lockAddress = _sharcLockBuffer.address;

	VkPushDataInfoEXT pushDataPt{ .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT };
	pushDataPt.offset = 0;
	pushDataPt.data = { &pushPt, sizeof(pushPt)};

	// Previous frame's trace may still be touching these shared buffers
	VkMemoryBarrier2 preClear{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
	preClear.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
	preClear.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
	preClear.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
	preClear.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;

	VkDependencyInfo preDep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	preDep.memoryBarrierCount = 1;
	preDep.pMemoryBarriers = &preClear;

	vkCmdPipelineBarrier2(commandBuffer, &preDep);

	//Clear GPU Hash table (ORCA reset cache)
	vkCmdFillBuffer(commandBuffer, _hashMapCheckSumBuffer.buffer, 0, VK_WHOLE_SIZE, ~0u);
	vkCmdFillBuffer(commandBuffer, _hashMapCountsBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
	vkCmdFillBuffer(commandBuffer, _hashMapValuesBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

	//Post barrier
	VkMemoryBarrier2 postClear{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
	postClear.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
	postClear.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	postClear.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
	postClear.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

	VkDependencyInfo postDep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
	postDep.memoryBarrierCount = 1;
	postDep.pMemoryBarriers = &postClear;

	vkCmdPipelineBarrier2(commandBuffer, &postDep);
	
	//Sparse pass
	_dispatchTable.cmdPushDataEXT(commandBuffer, &pushDataPt);
	_dispatchTable.cmdTraceRaysKHR(commandBuffer, &_rayGenerationRegion, &_missRegion, &_hitRegion, &_callRegion, _renderExtent.width, _renderExtent.height, 1);

	VkImage swapChainImage = _swapChainImages[swapChainImageIndex];
	VkImageLayout swapChainLayout;

	if (_debugView == EngineDebugView::PATHTRACED) {
		//Transition the swap chain image layout for writing from the post process compute
		util::transitionImageLayout(commandBuffer, swapChainImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		swapChainLayout = VK_IMAGE_LAYOUT_GENERAL;

		//Call the post process
		PushConstantsPost pcp{};
		pcp.outputImageIdx = swapChainImageIndex;
		// Storage writes map shader component 0 onto the format's first channel.
		// For B8G8R8A8 that channel is blue, so the post shader swaps R and B.
		pcp.swapRedBlue = _swapChainImageFormat == VK_FORMAT_B8G8R8A8_UNORM ? 1 : 0;

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelinePost);
		VkPushDataInfoEXT pushDataPost{ .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT };
		pushDataPost.offset = 0;
		pushDataPost.data = { &pcp, sizeof(PushConstantsPost) };
		_dispatchTable.cmdPushDataEXT(commandBuffer, &pushDataPost);
		vkCmdDispatch(commandBuffer, (_swapChainExtent.width + 15) / 16, (_swapChainExtent.height + 15) / 16, 1);
	}
	else {
		//Blit the selected g-buffer image straight into the swap chain
		const AllocatedImage *debugImage = nullptr;
		switch (_debugView) {
		case EngineDebugView::DIFFUSE_ALBEDO: debugImage = &_diffuseAlbedoImage; break;
		case EngineDebugView::SPECULAR_ALBEDO: debugImage = &_specularAlbedoImage; break;
		case EngineDebugView::NORMALS: debugImage = &_normalRoughnessImage; break;
		case EngineDebugView::DEPTH: debugImage = &_viewDepthImage; break;
		case EngineDebugView::ORCA_ALBEDO: debugImage = &_orcaAlbedoImage; break;
		default: break;
		}

		util::transitionImageLayout(commandBuffer, swapChainImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		swapChainLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

		VkImageSubresourceRange clearRange = init::imageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
		VkClearColorValue clearValue = { { 0.f, 0.f, 0.f, 1.f } };
		vkCmdClearColorImage(commandBuffer, swapChainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &clearRange);

		util::transitionImageLayout(commandBuffer, debugImage->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

		VkImageBlit2 blitRegion{ .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2 };
		blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		blitRegion.srcOffsets[1] = { static_cast<int32_t>(_renderExtent.width), static_cast<int32_t>(_renderExtent.height), 1 };
		blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		blitRegion.dstOffsets[1] = { static_cast<int32_t>(_swapChainExtent.width), static_cast<int32_t>(_swapChainExtent.height), 1 };

		VkBlitImageInfo2 blitInfo{ .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2 };
		blitInfo.srcImage = debugImage->image;
		blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
		blitInfo.dstImage = swapChainImage;
		blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		blitInfo.regionCount = 1;
		blitInfo.pRegions = &blitRegion;
		blitInfo.filter = VK_FILTER_NEAREST; //Linear filtering isn't guaranteed for R32_SFLOAT (depth)

		vkCmdBlitImage2(commandBuffer, &blitInfo);
	}

	if (_showGui) {
		util::transitionImageLayout(commandBuffer, swapChainImage, swapChainLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		drawImgui(commandBuffer, _swapChainImageViews[swapChainImageIndex]);

		//Transition image for presentation
		util::transitionImageLayout(commandBuffer, swapChainImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	}
	else {
		util::transitionImageLayout(commandBuffer, swapChainImage, swapChainLayout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	}


	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkCommandBufferSubmitInfo commandBufferSubmitInfo = init::commandBufferSubmitInfo(commandBuffer);

	// The cache is one buffer shared by every frame in flight. The semaphore chains
	// this frame's resolve and ray tracing behind the previous frame's.
	const VkPipelineStageFlags2 sharcStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
	std::array<VkSemaphoreSubmitInfo, 2> waits = {
		init::semaphoreSubmitInfo(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, currFrame._swapChainSemaphore),
		init::semaphoreSubmitInfo(sharcStages, _sharcSemaphore)
	};
	std::array<VkSemaphoreSubmitInfo, 2> signals = {
		init::semaphoreSubmitInfo(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, _renderCompleteSemaphores[swapChainImageIndex]),
		init::semaphoreSubmitInfo(sharcStages, _sharcSemaphore)
	};
	const uint32_t waitCount = _sharcSemaphoreWait ? 2u : 1u;

	//Submit
	VkSubmitInfo2 submit = init::submitInfo(&commandBufferSubmitInfo, signals.data(), waits.data(), signals.size(), waitCount);
	VK_CHECK(vkQueueSubmit2(_queue, 1, &submit, currFrame._renderFence));
	_sharcSemaphoreWait = true;

	//TODO: Present
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &_swapChain;
	presentInfo.pWaitSemaphores = &_renderCompleteSemaphores[swapChainImageIndex];
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapChainImageIndex;

	result = vkQueuePresentKHR(_queue, &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		_resize = true;
		return;
	}
	++_totalFrames;
	++_frameNum;
}

void Engine::resize() {
	int width, height;
	SDL_GetWindowSizeInPixels(_window, &width, &height);

	if (width == 0 || height == 0) return;

	_windowExtent = VkExtent2D(width, height);

	vkDeviceWaitIdle(_device);

	//Destroy and recreate the swap chain
	destroySwapChain();
	createSwapChain(width, height);
	initSwapChainDescriptors();

	//Destroy and recreate the radiance image
	updateRenderExtent();
	_allocator.destroyImage(_radianceImage);
	createRadianceImage();
	initRadianceImageDescriptors();

	//Destroy and recreate the DLSS images
	destroyDLSSImages();
	createDLSSImages();
	initDLSSImageDescriptors();

	_frameNum = 0;
	_resize = false;
}

void Engine::destroySwapChain() {
	for (auto swapChainImageView : _swapChainImageViews) vkDestroyImageView(_device, swapChainImageView, nullptr);
	_swapChainImages.clear();
	_swapChainImageViews.clear();
	_swapChainImageViewCreateInfos.clear();

	vkDestroySwapchainKHR(_device, _swapChain, nullptr);
}

void Engine::init() {
	initWindow();
	initVulkan();
	createSwapChain(WIDTH, HEIGHT);
	initCommandResources();
	initSyncStructures();

	_allocator.init(_instance, _physicalDevice, _device);
	_allocator.setCommandInfo(_queue, _immCommandBuffer, _immFence);

	updateRenderExtent();
	createRadianceImage();
	createDLSSImages();
	initDescriptorHeaps();

	loadHdrImage("./hdrs/kloofendal_48d_partly_cloudy_puresky_4k.hdr");
	initSwapChainDescriptors();
	initRadianceImageDescriptors();
	initDLSSImageDescriptors();

	_loaded = true;

	//Load default scene
	std::string path = "./models/cornellbox_anims.glb";
	loadScene(path);

	createSharcBuffers();
	initGPUHashMap();

	initPipelinePathtracer();
	initPipelineUpdate();
	initPipelineResolve();
	initPipelinePost();

	initImgui();

	_start = std::chrono::steady_clock::now();
}

void Engine::run() {
	SDL_Event e;

	bool quit = false;

	while (!quit) {

		//Handle events
		while (SDL_PollEvent(&e) != 0) {
			ImGui_ImplSDL3_ProcessEvent(&e);

			if (e.type == SDL_EVENT_QUIT) quit = true;

			if (e.type == SDL_EVENT_WINDOW_MINIMIZED) _minimized = true;

			if (e.type == SDL_EVENT_WINDOW_RESTORED) _minimized = false;

			if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) _resize = true;

			if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_S) _showGui = !_showGui;

			//TODO: Handle camera movement
			if (!_io->WantCaptureMouse) {
				_camera.processEvent(e);
			}
		}

		if (_minimized) {
			SDL_Delay(10);
			continue;
		}

		if (_resize) resize();

		draw();
	}
}

void Engine::cleanup() {
	if (_loaded) {
		vkDeviceWaitIdle(_device);

		if (_sceneDataBuffer.loaded) _allocator.destroyBuffer(_sceneDataBuffer);
		if (_scene.loaded) loader::destroyScene(_device, _allocator, _scene, _dispatchTable);

		if (_envAliasTable.loaded) _allocator.destroyBuffer(_envAliasTable);

		destroySwapChain();
		destroyDLSSImages();
		if (_hdrLoaded) _allocator.destroyImage(_hdrImage);
		_allocator.destroyImage(_radianceImage);

		_deletionQueue.flush();

		_allocator.cleanup();

		SDL_Vulkan_DestroySurface(_instance, _surface, nullptr);
		SDL_DestroyWindow(_window);

		vkb::destroy_device(_vkbData.device);
		vkb::destroy_instance(_vkbData.instance); //Handles debug messenger destruction
	}

	_loaded = false;
}