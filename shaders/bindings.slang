#pragma once
#include "util.slang"
#include "sharc_util.slang"

[vk::binding(5, 0)]
RWStructuredBuffer<HashGridKey> sharcHashEntries;

[vk::binding(6, 0)]
RWStructuredBuffer<SharcAccumulationData> sharcAccumulation;

[vk::binding(7, 0)]
RWStructuredBuffer<SharcPackedData> sharcResolved;

[vk::binding(8, 0)]
StructuredBuffer<SharcConfig> sharcConfigBuffer;

[vk::binding(9, 0)]
RWStructuredBuffer<uint> sharcLockBuffer;

//Inputs
[vk::binding(0, 0)]
RWTexture2D<float4> radianceImage;

[vk::binding(1, 0)] //TODO: Change this if you change the max number of swap chain images
Texture2D<float4> sceneImages[];

[vk::binding(2, 0)]
SamplerState samplers[];

[vk::binding(3, 0)]
RaytracingAccelerationStructure tlas;

[vk::binding(4, 0)]
Texture2D<float4> hdrImage;

// Set 0 bindings 5-9 are reserved for the SHaRC buffers
// G-buffer, DLSS and ORCA images
[vk::binding(0, 1)]
RWTexture2D<float4> diffuseAlbedoImage;

[vk::binding(1, 1)]
RWTexture2D<float4> specularAlbedoImage;

[vk::binding(2, 1)]
RWTexture2D<float4> normalRoughnessImage;

[vk::binding(3, 1)]
RWTexture2D<float4> depthImage;

[vk::binding(4, 1)]
RWTexture2D<float4> motionVectorImage;

[vk::binding(5, 1)]
RWTexture2D<float4> specHitDistImage;

[vk::binding(6, 1)]
RWTexture2D<float4> dlssOutputImage;

[vk::binding(7, 1)]
RWTexture2D<float4> orcaAlbedoImage;

//[vk::binding(0, 2)]
//ConstantBuffer<HashMap> hashMap;

[[vk::push_constant]]
PushConstantsPathtracer pc;