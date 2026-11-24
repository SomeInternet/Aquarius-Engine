// Holy shit, I think I love slang
#define SHARC_UPDATE 1
#define SHARC_QUERY 1

#define HASH_GRID_COMPACT 0
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 0

#include "bsdf.slang"
#include "bindings.slang"
#include "sampling.slang"
#include "hashmap.slang"

#define SPARSE_TILE_SIZE 4

struct UpdateHitPayload
{
    // Existing values: Single-segment BSDF value, or terminal radiance with a negative blue channel.
    float3 throughput;
    float3 rayOrigin;
    float3 rayDir;
    float pdf;
    float3 directLight;
    uint2 seed;

    // Surface data shared by ORCA and SHARC
    float3 positionWorld;
    float3 normalWorld;
    uint surfaceValid;

    // Unweighted local radiance, reset for each SHARC segment.
    float3 sharcLocalRadiance;
};

inline uint2 pickSparsePixel(uint2 imageSize) {
    uint2 tileIndex = DispatchRaysIndex().xy;
    uint2 tileOrigin = tileIndex * SPARSE_TILE_SIZE;

    uint2 tileExtent = min(
        uint2(SPARSE_TILE_SIZE, SPARSE_TILE_SIZE),
        imageSize - tileOrigin);

    uint tilePixelCount = tileExtent.x * tileExtent.y;

    uint tilePhase =
        pcg(tileIndex.x ^ pcg(tileIndex.y)) % tilePixelCount;
    uint tileSample =
        (tilePhase + pc.totalFrames) % tilePixelCount;

    uint2 tileOffset = uint2(
        tileSample % tileExtent.x,
        tileSample / tileExtent.x);

    return tileOrigin + tileOffset; // choose one pixel for 4x4 tile
}

[shader("raygeneration")]
void rayGeneration() {
    uint2 imageSize;
    radianceImage.GetDimensions(imageSize.x, imageSize.y);

    uint index = DispatchRaysIndex().x;
    if (index >  min(pc->sparsePixels.count, SPARSE_PIXEL_CAPACITY))
        return;

    SparsePixel px = pc->sparsePixels.pixels[index];
    uint2 pixel = uint2(px.x, px.y);

    if (any(pixel >= imageSize))
        return;

    float2 samplePos = float2(pixel);

    seed = pcg2(pixel) ^ pcg(pc.totalFrames);

    RayDesc ray;
    {
        float2 imageSizeFloat = float2(imageSize);
        float aspect = imageSizeFloat.x / imageSizeFloat.y;
        float2 ndc = float2((samplePos.x - imageSizeFloat.x / 2.f) / (imageSizeFloat.x / 2.f), (imageSizeFloat.y / 2.f - samplePos.y) / (imageSizeFloat.y / 2.f));
        float3 dir = normalize(pc.camForward + pc.camRight * tan(pc.fov / 2.f) * ndc.x * aspect + pc.camUp * tan(pc.fov / 2.f) * ndc.y);

        ray.Origin = pc.camPos; 
        ray.Direction = dir;
    }
    ray.TMin = EPSILON; // Avoid self-intersection
    ray.TMax = INFINITY;

    // All sparse paths populate ORCA; ~67% update SHaRC when enabled.
    bool updateSharc = rand() < (2.f / 3.f) && (pc.flags & EngineStateFlags::SHARC) != 0;

    // SHARC init
    SharcParameters sharcParameters = createSharcParams(pc.camPos, sharcConfigBuffer[0], sharcHashEntries, sharcAccumulation, sharcResolved, sharcLockBuffer);
    SharcState sharcState;
    SharcInit(sharcState);

    UpdateHitPayload payload = {};
    uint surfaceHitCount = 0;

    // View dir and pos of the first indirect hit, plus the radiance gathered past it
    float3 indirectHitDir = float3(0.f), indirectHitPos = float3(0.f);
    float3 orcaThroughput = float3(1.f); // throughput past the second bounce (first indirect hit)
    float3 orcaRadiance = float3(0.f);
    float prevBsdfPdf = 0.f;
    for (uint i = 0; i < pc.maxBounces; ++i) {
        payload.seed = seed;
        payload.throughput = float3(0.f);
        payload.pdf = 0.f;
        payload.directLight = float3(0.f);
        payload.sharcLocalRadiance = float3(0.f);
        payload.surfaceValid = 0;

        HitObject hitObject = HitObject::TraceRay(tlas, 0, 0xFF, 0, 0, 0, ray, payload);
        ReorderThread(hitObject);
        HitObject::Invoke<UpdateHitPayload>(tlas, hitObject, payload);
        seed = payload.seed;

        // Miss, return environment radiance
        if (!hitObject.IsHit())
        {
            payload.throughput.b *= -1.f;

            // Environment radiance arriving past the first indirect hit
            if (surfaceHitCount >= 2) {
                float misWeight = prevBsdfPdf > 0.f ? powerHeuristic(1, prevBsdfPdf, 1, payload.pdf) : 1.f;
                orcaRadiance += orcaThroughput * payload.throughput * misWeight;
            }

            if (updateSharc)
                SharcUpdateMiss(sharcParameters, sharcState, payload.sharcLocalRadiance);

            break;
        }

        // Terminal hit: if the path is at the bounce limit, surface is valid, path continues, and SHARC is enabled, then query the SHaRC cache.
        bool atBounceLimit = (i + 1 == pc.maxBounces);
        bool continuingSurface = payload.surfaceValid != 0 && payload.throughput.b >= 0.f && payload.pdf > 0.f;

        if (atBounceLimit && continuingSurface && surfaceHitCount >= 2 && (pc.flags & EngineStateFlags::SHARC) != 0)
        {
            SharcHitData terminalHit = {};
            terminalHit.positionWorld = payload.positionWorld;
            terminalHit.normalWorld = payload.normalWorld;

            float3 tailRadiance = float3(0.f);
            if (SharcGetCachedRadiance(sharcParameters, terminalHit, tailRadiance, false))
            {
                orcaRadiance += orcaThroughput * tailRadiance;

                if (updateSharc)
                    SharcUpdateMiss(sharcParameters, sharcState, tailRadiance);

                break;
            }
        }

        // Valid hit
        // ORCA skips direct lighting at the first indirect hit itself.
        if (payload.surfaceValid != 0)
        {
            if (++surfaceHitCount == 2)
            {
                // Store view dir and pos of first indirect hit
                indirectHitDir = ray.Direction;
                indirectHitPos = payload.positionWorld;
            }
            else if (surfaceHitCount >= 2)
            {
                // Radiance past the first indirect hit
                if (payload.throughput.b < 0.f) {
                    float3 emitted = float3(payload.throughput.rg, -payload.throughput.b);
                    float misWeight = prevBsdfPdf > 0.f ? powerHeuristic(1, prevBsdfPdf, 1, payload.pdf) : 1.f;
                    orcaRadiance += orcaThroughput * emitted * misWeight;
                } else {
                    orcaRadiance += orcaThroughput * select(isnan(payload.directLight), float3(0.f), payload.directLight);
                }
            }
        }

        // SHARC handles each surface independently of ORCA's accumulated values.
        if (updateSharc && payload.surfaceValid != 0)
        {
            SharcHitData sharcHitData;
            sharcHitData.positionWorld = payload.positionWorld;
            sharcHitData.normalWorld = payload.normalWorld;

            // Update ORCA even if SHARC stops updating, since ORCA needs to accumulate radiance past the first indirect hit.
            updateSharc = SharcUpdateHit(sharcParameters, sharcState, sharcHitData, payload.sharcLocalRadiance, rand());
            // if (!SharcUpdateHit(sharcParameters, sharcState, sharcHitData, payload.sharcLocalRadiance, rand()))
            //     break;
        }

        // No more path
        if (payload.throughput.b < 0.f)
            break;

        // Alpha-cut pass-through
        if (payload.pdf <= 0.f)
        {
            ray.Origin = payload.rayOrigin;
            ray.Direction = payload.rayDir;
            continue;
        }

        // Single segment throughput
        float3 segmentThroughput = payload.throughput / payload.pdf;

        if (updateSharc)
            SharcSetThroughput(sharcState, segmentThroughput);

        // Accumulate throughput past the second bounce (first indirect hit)
        if (surfaceHitCount >= 2)
            orcaThroughput *= select(isnan(segmentThroughput), float3(0.f), segmentThroughput);
        prevBsdfPdf = payload.pdf;

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDir;

    }

    // Path terminus: cache the radiance into the view dir and pos of the first indirect hit,
    // and record the new finest slot for downrez
    // if (surfaceHitCount >= 2 && length(orcaRadiance) > 0.f)
    if (surfaceHitCount >= 2) {
        uint address;
        HashMapResult result = populate(pc.hashMap->checkSumBuffer, pc.hashMap->radianceBuffer, pc.hashMap->countBuffer,
                                        pc.hashMap->childAddressBuffer, NUM_HASH_MAP_ENTRIES, indirectHitDir, indirectHitPos,
                                        orcaRadiance, address, pc.flags & EngineStateFlags::GPU_HASH_MAP_HEATMAP);
        if (result == HashMapResult::ENTRY_UNALLOCATED) {
            uint index;
            InterlockedAdd(pc.hashMap->finestLayer->size, 1, index);
            if (index < MAX_LAYER_SIZE)
                pc.hashMap->finestLayer->list[index] = address;
        }
    }
}

[shader("closesthit")]
void closestHit(inout UpdateHitPayload payload, in BuiltInTriangleIntersectionAttributes isect) {
    seed = payload.seed;

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

    // SHARC & ORCA populate geometric data
    payload.positionWorld = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.normalWorld = normalize(mul(gNor, WorldToObject3x4()).xyz);
    payload.surfaceValid = 1;

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

            float3 emittedRadiance = emissionColor * material.emission[3];

            //SHARC local radiance
            payload.sharcLocalRadiance = emittedRadiance;
            payload.throughput = emittedRadiance;
            payload.throughput.b = min(-payload.throughput.b, -EPSILON);

            //Chance of light sampling having picked this point, for the MIS weight in raygen
            payload.pdf = triangleLightPdf(pc.sceneData, material, RayTCurrent() * RayTCurrent(), abs(dot(payload.normalWorld, WorldRayDirection())));
            payload.seed = seed;
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
            payload.surfaceValid = 0;
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
    float3 hitPos = payload.positionWorld;
    float3 worldNor = normalize(mul(ObjectToWorld3x4(), float4(mulNorTranspose(pointInfo, float3(0.f, 0.f, 1.f)), 0.f)));

    //Light sampling, weighted against the chance of the BSDF sampling the same direction
    {
        LightSample ls = sampleLight(pc.sceneData, hitPos);
        float3 wiLight = normalize(mulNor(pointInfo, mul(WorldToObject3x4(), float4(ls.dir.xyz, 0.f)).xyz));

        if (ls.pr > 0.f && wiLight.z != 0.f) {
            float3 bsdf;
            float bsdfPdf;
            evalCookTorrance(wo, wiLight, pbr, bsdf, bsdfPdf);

            //The bsdf already includes the cosine term
            if (any(bsdf > 0.f) && shadowRayCast(hitPos + worldNor * sign(wiLight.z) * EPSILON, ls.dir.xyz, ls.dir.w - 2.f * EPSILON)) {
                payload.directLight = bsdf * ls.radiance * powerHeuristic(1, ls.pr, 1, bsdfPdf) / ls.pr;
                // SHARC local radiance
                payload.sharcLocalRadiance = select(isnan(payload.directLight), float3(0.f), payload.directLight);
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
void miss(inout UpdateHitPayload payload) {
    float2 uv = float2(atan2(WorldRayDirection().z, WorldRayDirection().x), asin(WorldRayDirection().y));
    uv *= float2(1.f / (2.f * PI), -1.f / PI); //v is flipped
    uv += float2(.5f);
    payload.sharcLocalRadiance = hdrImage.SampleLevel(samplers[0], uv, 0.f).rgb;
    payload.throughput = payload.sharcLocalRadiance;
    payload.pdf = envLightPdf(pc.sceneData, uv);
    payload.throughput.b = min(-payload.throughput.b, -EPSILON);
}

[shader("anyhit")]
void anyHit() {

}