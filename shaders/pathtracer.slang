// SHARC defs
#define SHARC_QUERY 0
#define SHARC_UPDATE 0

//Holy shit, I think I love slang
#include "bsdf.slang"

#include "sampling.slang"
#include "bindings.slang"
#include "hashmap.slang"
// SHARC

// Primary hit read from the raster G-buffer instead of tracing primary rays.
// World position is reconstructed from the camera ray and the stored ray length.
struct RasterPrimaryHit {
    float4 albedoMetallic; // orcaAlbedoImage
    float3 pos; // calculated using camPos + cameraDir * depth
    float4 normalRoughness; // normalRoughnessImage
    float2 motionVectors; // motionVectorImage
    float depth; // depthImage

    float3 cameraDir;
    float3 normal; // derived from normalRoughnessImage
    float roughness; // derived from normalRoughnessImage
    bool valid; // whether the pixel has a real rasterized surface(aka if that pixel is on a surface)

    // Filled by samplePrimaryBounce. The G-buffer does not store a light sample or a bounce.
    float3 directLight;
    float3 throughput;
    float pdf;
    bool hasBounce;
    float3 bounceOrigin;
    float3 bounceDir;
};

// given pixel on the image, returns 3D world-space direction vector from camera's pinhole through that pixel on the virtual image plane 
float3 pinholeCameraDirection(float2 samplePos, float2 imageSize) {
    float aspect = imageSize.x / imageSize.y;
    float2 ndc = float2((samplePos.x - imageSize.x / 2.f) / (imageSize.x / 2.f), (imageSize.y / 2.f - samplePos.y) / (imageSize.y / 2.f));
    return normalize(pc.camForward + pc.camRight * tan(pc.fov / 2.f) * ndc.x * aspect + pc.camUp * tan(pc.fov / 2.f) * ndc.y);
}

RasterPrimaryHit loadRasterPrimaryHit(uint2 pixel, float2 imageSize) {
    RasterPrimaryHit primary = {};
    primary.cameraDir = pinholeCameraDirection(float2(pixel) + 0.5f, imageSize);

    primary.albedoMetallic = orcaAlbedoImage[pixel];
    primary.normalRoughness = normalRoughnessImage[pixel];
    primary.motionVectors = motionVectorImage[pixel].xy;
    primary.depth = depthImage[pixel].x;

    primary.normal = primary.normalRoughness.xyz;
    primary.roughness = primary.normalRoughness.w;
    // Uncovered pixels are cleared to zero.
    primary.valid = primary.depth > 0.f && primary.depth < INFINITY; // if pixel is on a surface, the depth is positive and finite distance 
    primary.pos = pc.camPos + primary.cameraDir * primary.depth;

    primary.directLight = float3(0.f);
    primary.throughput = float3(1.f);
    primary.pdf = 0.f;
    primary.hasBounce = false;
    primary.bounceOrigin = primary.pos + primary.normal * EPSILON;
    primary.bounceDir = float3(0.f);
    return primary;
}

// The raster pass replaced the camera ray, so the primary surface is shaded here.
// Similar to closestHit: a light sample, then a BSDF sample for the next ray.
// The G-buffer has no IOR or transmission, so this is an opaque surface at the glTF default IOR.
void samplePrimaryBounce(inout RasterPrimaryHit primary) {
    if (!primary.valid) return;

    float3 worldNor = primary.normal;
    float norLen = length(worldNor);
    if (norLen <= EPSILON) return;
    worldNor /= norLen; // normalizes the normal 
    // cameraDir points from the eye toward the hit. A front face points back at the eye.
    if (dot(worldNor, primary.cameraDir) > 0.f) worldNor = -worldNor;
    primary.normal = worldNor;

    PointInfo pointInfo = {};
    pointInfo.nor = worldNor;
    pointInfo.tan = fallbackTangent(worldNor);

    float3 wo = -mulNor(pointInfo, primary.cameraDir);
    if (wo.z < EPSILON) return;

    PBR pbr = {};
    pbr.color = float4(primary.albedoMetallic.xyz, 1.f);
    pbr.metallicRoughness = float2(clamp(primary.albedoMetallic.w, 0.f, 1.f), max(primary.roughness, EPSILON));
    pbr.transmissiveness = 0.f;
    pbr.etaA = 1.f;
    pbr.etaB = 1.5f;
    // sample 1 light from hit position, evaluate BSDF toward it 
    {
        LightSample ls = sampleLight(pc.sceneData, primary.pos);
        if (ls.pr > 0.f) {
            float3 wiLight = normalize(mulNor(pointInfo, ls.dir.xyz));
            if (wiLight.z != 0.f) {
                float3 bsdf;
                float bsdfPdf;
                float3 diffuseAlbedo = float3(0.f);
                float3 specularAlbedo = float3(0.f);
                evalCookTorrance2(wo, wiLight, pbr, bsdf, diffuseAlbedo, specularAlbedo, bsdfPdf);

                if (any(bsdf > 0.f) && shadowRayCast(primary.pos + worldNor * sign(wiLight.z) * EPSILON, ls.dir.xyz, ls.dir.w - 2.f * EPSILON))
                    primary.directLight = bsdf * ls.radiance * powerHeuristic(1, ls.pr, 1, bsdfPdf) / ls.pr;
            }
        }
    }

    // sampleWi picks random outgoing direction, and evaluates that direction's BSDF value and prob 
    float2 xi = float2(rand(), rand());
    float lobe = rand();
    float3 wi;
    bool valid = sampleWi(wo, pbr, lobe, xi, wi);
    float3 bsdf = float3(0.f);
    float pdf = 0.f;
    if (valid) evalCookTorrance(wo, wi, pbr, bsdf, pdf);
    if (!valid || pdf <= 0.f) return;

    primary.throughput = bsdf / pdf;
    primary.pdf = pdf;
    primary.hasBounce = true;
    primary.bounceDir = normalize(mulNorTranspose(pointInfo, wi));
    primary.bounceOrigin = primary.pos + worldNor * sign(wi.z) * EPSILON;
}

[shader("raygeneration")]
void rayGeneration() {
    seed = pcg2(DispatchRaysIndex().xy) ^ pcg(pc.totalFrames);

    float2 imageSize = (float2)DispatchRaysDimensions().xy;
    uint2 pixel = DispatchRaysIndex().xy;

    RasterPrimaryHit primary = loadRasterPrimaryHit(pixel, imageSize);
    samplePrimaryBounce(primary);

    float3 throughput = primary.throughput;
    float3 accumulatedLight = primary.directLight;
    float prevBsdfPdf = primary.pdf;
    float prevRoughness = primary.roughness;
    uint depth = 0; // track bounces (sharc)

    float3 cachedRadiance = float3(0.f);
    bool hashMapQueried = false;
    float3 dir = primary.bounceDir; // incoming direction of the reconstruction ray

    // The camera ray is gone. This trace is the first indirect hit.
    if (primary.valid && primary.hasBounce) {
        depth = 1;

        RayDesc ray;
        ray.Origin = primary.bounceOrigin;
        ray.Direction = primary.bounceDir;
        ray.TMin = EPSILON;
        ray.TMax = INFINITY;

        HitPayload payload = {};
        payload.gBufferWritten = true; // the raster pass owns the primary G-buffer
        payload.seed = seed;

        HitObject hitObject = HitObject::TraceRay(tlas, 0, 0xFF, 0, 0, 0, ray, payload);
        ReorderThread(hitObject);
        HitObject::Invoke<HitPayload>(tlas, hitObject, payload);
        seed = payload.seed;

        if (EngineStateFlags::GPU_HASH_MAP_HEATMAP & pc.flags) {
            bool found = false;
            if (hitObject.IsHit()) {
                float3 hitPos = ray.Origin + ray.Direction * payload.hitLen;
                if (query(pc.hashMap->checkSumBuffer, pc.hashMap->radianceBuffer, pc.hashMap->countBuffer, NUM_HASH_MAP_ENTRIES, dir, hitPos,
                pc.heatMapLod, cachedRadiance, pc.flags & EngineStateFlags::GPU_HASH_MAP_HEATMAP) == HashMapResult::ENTRY_FOUND) {
                    found = true;
                }
            }
            if (!found) cachedRadiance = float3(0.f);
        } else if (hitObject.IsHit() && (payload.pdf > 0.f || payload.throughput.b < 0.f)) {
            hashMapQueried = true;
            float3 hitPos = ray.Origin + ray.Direction * payload.hitLen;
            bool found = false;
            for(int j = MIN_LOD; j <= MAX_LOD; ++j) {
                if (query(pc.hashMap->checkSumBuffer, pc.hashMap->radianceBuffer, pc.hashMap->countBuffer, NUM_HASH_MAP_ENTRIES, dir, hitPos,
                j, cachedRadiance, pc.flags & EngineStateFlags::GPU_HASH_MAP_HEATMAP) == HashMapResult::ENTRY_FOUND) {
                    found = true;
                    break;
                }
            }
            if (!found) cachedRadiance = float3(0.f);

            // The cached tail already includes the bounce BSDF. Scale by the primary throughput.
            cachedRadiance *= throughput;
            cachedRadiance = select(isnan(cachedRadiance), float3(0.f), cachedRadiance);
        }

        if (hitObject.IsHit()) accumulatedLight += throughput * select(isnan(payload.directLight), float3(0.f), payload.directLight);

        // A negative blue channel means the path ended, on an emitter or the environment
        if (!hitObject.IsHit() || payload.throughput.b < 0.f) {
            payload.throughput.b *= -1.f;
            // MIS weight of the BSDF sample against the chance of light sampling having picked the same point (payload.pdf)
            float misWeight = prevBsdfPdf > 0.f ? powerHeuristic(1, prevBsdfPdf, 1, payload.pdf) : 1.f;
            accumulatedLight += throughput * payload.throughput.rgb * misWeight;
        }
    }

    if (pc.flags & EngineStateFlags::SHARC_HEATMAP) {
        accumulatedLight = depth == 0   ? float3(0.f, 0.f, 1.f)
                           : depth == 1 ? float3(0.f, 1.f, 0.f)
                                        : float3(1.f, 0.f, 0.f);
    }

    /*
    if (pc.flags & EngineStateFlags::RAY_RECONSTRUCTION) {
        radianceImage[DispatchRaysIndex().xy] = radianceImage[DispatchRaysIndex().xy] * float(pc.frameNum) / float(pc.frameNum + 1) +
            float4(accumulatedLight, 1.f) * 1.f / float(pc.frameNum + 1);
    } else {
        radianceImage[DispatchRaysIndex().xy] = radianceImage[DispatchRaysIndex().xy] * float(pc.frameNum) / float(pc.frameNum + 1) +
            float4(accumulatedLight, 1.f) * 1.f / float(pc.frameNum + 1);
    }
    */

    // Direct + first indirect light, plus the attenuated hashmap radiance for the rest of the path
    if (EngineStateFlags::GPU_HASH_MAP_HEATMAP & pc.flags) radianceImage[DispatchRaysIndex().xy] = float4(cachedRadiance, 1.f);
    else radianceImage[DispatchRaysIndex().xy] = float4(accumulatedLight + cachedRadiance, 1.f);
}

[shader("closesthit")]
void closestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes isect) {
    seed = payload.seed;
    payload.directLight = float3(0.f);

    PointInfo pointInfo;
    bool entering;
    float3 gNor;
    {
        float3 barycentrics = float3(1.f - isect.barycentrics.x - isect.barycentrics.y, isect.barycentrics);
        pointInfo = getPointInfo(InstanceID() + GeometryIndex(), PrimitiveIndex(), barycentrics, pc, gNor);

        entering = dot(ObjectRayDirection(), gNor) < 0.f;
        if (dot(ObjectRayDirection(), gNor) > 0.f) gNor *= -1.f;
        if (dot(gNor, pointInfo.nor) < 0.f) {
            pointInfo.nor *= -1.f;
            pointInfo.tan *= -1.f;
        }
    }

    payload.hitLen = RayTCurrent();
    payload.hitNor = normalize(mul(gNor, WorldToObject3x4()).xyz);

    // TODO: Consider an alternative to my current solution to a material index being 0
    // TODO: Drop the image and sampler indices once I'm done with them?
    PBR pbr;
    {
        Material material = pc.sceneData->materials[max(pc.sceneData->deviceSubMeshes[InstanceID() + GeometryIndex()].materialIdx, 0)];

        // Emissive
        // We terminate early on emissive materials
        if (length(material.emission.rgb) > 0.f) {
            float3 emissionColor = material.emission.rgb * (material.emissionImgIdx != -1 ? 
            sceneImages[NonUniformResourceIndex(material.emissionImgIdx)]
                           .SampleLevel(samplers[NonUniformResourceIndex(material.emissionSamplerIdx + 1)], pointInfo.uv, 0.f) .rgb : 
            float3(1.f));

            payload.throughput = emissionColor * material.emission[3], 1.f;
            payload.throughput.b = min(-payload.throughput.b, -EPSILON);

            //Chance of light sampling having picked this point, for the MIS weight in raygen
            float3 worldGNor = normalize(mul(gNor, WorldToObject3x4()).xyz);
            payload.pdf = triangleLightPdf(pc.sceneData, material, RayTCurrent() * RayTCurrent(), abs(dot(worldGNor, WorldRayDirection())));
            return;
        }

        // APPLY MATERIAL MAPS
        //  Normal map
        if (material.norImgIdx >= 0) {
            float3 mNor = sceneImages[NonUniformResourceIndex(material.norImgIdx)]
                    .SampleLevel(samplers[NonUniformResourceIndex(material.norSamplerIdx + 1)], pointInfo.uv, 0.f) .rgb;

            applyNormalMap(pointInfo, mNor);
        }

        // Metallic-Roughness
        pbr.metallicRoughness = material.metallicRoughness * (material.metallicRoughnessImgIdx != -1 ? 
        sceneImages[NonUniformResourceIndex(material.metallicRoughnessImgIdx)]
                       .SampleLevel(samplers[NonUniformResourceIndex(material.metallicRoughnessSamplerIdx + 1)], pointInfo.uv, 0.f) .bg : 
        float2(1.f));

        // Using .y instead of .g because the Slang syntax highlighting extension behaves really funky... :(
        pbr.metallicRoughness.y = max(pbr.metallicRoughness.y, EPSILON);

        // Albedo
        pbr.color = material.color * (material.colorImgIdx != -1 ? 
        sceneImages[NonUniformResourceIndex(material.colorImgIdx)]
        .SampleLevel( samplers[NonUniformResourceIndex(material.colorSamplerIdx + 1)], pointInfo.uv, 0.f) : 
        float4(1.f));

        if (pbr.color.a < .2f) {
            payload.throughput = float3(1.f);
            payload.pdf = 0.f; //Tells raygen nothing scattered here
            payload.rayDir = WorldRayDirection();
            payload.rayOrigin = WorldRayOrigin() + (RayTCurrent() + EPSILON) * WorldRayDirection();
            payload.seed = seed;
            return;
        }

        pbr.transmissiveness = material.transmissiveness;
        if (material.transmissiveness == 0.f) entering = true;
        pbr.etaA = entering ? 1.f : material.ior;
        pbr.etaB = entering ? material.ior : 1.f;
    }

    float3 wo = -normalize(mulNor(pointInfo, mul(WorldToObject3x4(), float4(WorldRayDirection(), 0.f)).xyz));
    if (wo.z < EPSILON) {
        pointInfo.nor = gNor;
        pointInfo.tan = normalize(pointInfo.tan - gNor * dot(pointInfo.tan, gNor));
        wo = -normalize(mulNor(pointInfo, mul(WorldToObject3x4(), float4(WorldRayDirection(), 0.f)).xyz));
    }
    float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 worldNor = normalize(mul(ObjectToWorld3x4(), float4(mulNorTranspose(pointInfo, float3(0.f, 0.f, 1.f)), 0.f)));
    
    // fill query fields for SHARC
    payload.hitLen = RayTCurrent();
    payload.hitNor = normalize(mul(gNor, WorldToObject3x4()).xyz);
    payload.roughness = pbr.metallicRoughness.y;

    // Light sampling, weighted against the chance of the BSDF sampling the same direction
    float3 gAlbedo;
    float3 gSpecular;
    {
        LightSample ls = sampleLight(pc.sceneData, hitPos);
        float3 wiLight = normalize(mulNor(pointInfo, mul(WorldToObject3x4(), float4(ls.dir.xyz, 0.f)).xyz));

        if (ls.pr > 0.f && wiLight.z != 0.f) {
            float3 bsdf;
            float bsdfPdf;
            evalCookTorrance2(wo, wiLight, pbr, bsdf, gAlbedo, gSpecular, bsdfPdf);

            //The bsdf already includes the cosine term
            if (any(bsdf > 0.f) && shadowRayCast(hitPos + worldNor * sign(wiLight.z) * EPSILON, ls.dir.xyz, ls.dir.w - 2.f * EPSILON)) {
                payload.directLight = bsdf * ls.radiance * powerHeuristic(1, ls.pr, 1, bsdfPdf) / ls.pr;
            }
        }
    }

    float3 wi;
    {
        //TODO: Optimize
        float2 xi = float2(rand(), rand());
        float rand = rand();

        bool valid = sampleWi(wo, pbr, rand, xi, wi);
        if (valid) evalCookTorrance(wo, wi, pbr, payload.throughput, payload.pdf);
        if (!valid || payload.pdf <= 0.f) {
            payload.throughput = float3(0, 0, -EPSILON);
            payload.pdf = 0.f;
            return;
        }
    }

    payload.rayDir = normalize(mul(ObjectToWorld3x4(), float4(mulNorTranspose(pointInfo, wi), 0.f)).xyz);
    payload.rayOrigin = hitPos + worldNor * sign(wi.z) * EPSILON;
    payload.seed = seed;
}

[shader("miss")]
void miss(inout HitPayload payload) {
    float2 uv = float2(atan2(WorldRayDirection().z, WorldRayDirection().x), asin(WorldRayDirection().y));
    uv *= float2(1.f / (2.f * PI), -1.f / PI); //v is flipped
    uv += float2(.5f);
    payload.throughput = hdrImage.SampleLevel(samplers[0], uv, 0.f).rgb;
    payload.pdf = envLightPdf(pc.sceneData, uv);
    payload.throughput.b = min(-payload.throughput.b, -EPSILON);
}

[shader("anyhit")]
void anyHit() {

}
