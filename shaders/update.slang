// Holy shit, I think I love slang
#define SHARC_UPDATE 1
#define SHARC_QUERY 0

#define HASH_GRID_COMPACT 0

#include "bsdf.slang"
#include "sharc_util.slang"

// SHARC
//  SHARC hash table
[vk::binding(5, 0)]
RWStructuredBuffer<HashGridKey> sharcHashEntries;

// Per-frame radiance accumulation
[vk::binding(6, 0)]
RWStructuredBuffer<SharcAccumulationData> sharcAccumulation;

// Temporally resolved cache
[vk::binding(7, 0)]
RWStructuredBuffer<SharcPackedData> sharcResolved;
[vk::binding(8, 0)]
StructuredBuffer<SharcConfig> sharcConfigBuffer;

[vk::binding(9, 0)]
RWStructuredBuffer<uint> sharcLockBuffer;

//Inputs
[vk::binding(0, 0)]
RWTexture2D<float4> radianceImage;

[vk::binding(0, 1)]
RWTexture2D<float4> diffuseAlbedoImage;

[vk::binding(0, 2)]
RWTexture2D<float4> specularAlbedoImage;

[vk::binding(0, 3)]
RWTexture2D<float4> normalRoughnessImage;

[vk::binding(0, 4)]
RWTexture2D<float4> depthImage;

[vk::binding(0, 5)]
RWTexture2D<float4> motionVectorImage;

[vk::binding(0, 6)]
RWTexture2D<float4> specHitDistImage;

[vk::binding(0, 7)]
RWTexture2D<float4> dlssOutputImage;

[vk::binding(1, 0)] //TODO: Change this if you change the max number of swap chain images
Texture2D<float4> sceneImages[];

[vk::binding(2, 0)]
SamplerState samplers[];

[vk::binding(3, 0)]
RaytracingAccelerationStructure tlas;

[vk::binding(4, 0)]
Texture2D<float4> hdrImage;

[[vk::push_constant]]
PushConstantsPathtracer pc;

struct UpdateHitPayload
{
    // Existing values
    float3 throughput;
    float3 rayOrigin;
    float3 rayDir;
    float pdf;
    float3 directLight;
    uint2 seed;

    // More information returned to ray generation for SHARC.
    float3 sharcPositionWorld;
    float3 sharcNormalWorld;
    float3 sharcLocalRadiance;
    uint sharcSurfaceValid;
};

inline float envLightPdf(SceneData *sceneData, float2 uv) {
    if (sceneData->totalEnvWeight <= 0.f) return 0.f;

    float sinTheta = sin(uv.y * PI);
    if (sinTheta <= 0.f) return 0.f;

    uint width, height;
    hdrImage.GetDimensions(width, height);
    int2 pixel = clamp(int2(uv * float2(width, height)), int2(0), int2(width - 1, height - 1));

    float rowWeight = sin((float(pixel.y) + .5f) / float(height) * PI);
    float weight = rowWeight * length(hdrImage.Load(int3(pixel, 0)).rgb);

    return weight / (sceneData->totalEnvWeight + sceneData->totalLightWeight) * float(width * height) / (2.f * PI * PI * sinTheta);
}

inline LightSample sampleLight(SceneData *sceneData, float3 origin) {
    LightSample ls = {};
    float totalWeight = sceneData->totalEnvWeight + sceneData->totalLightWeight;
    if (totalWeight <= 0.f) return ls;

    if (rand() < sceneData->totalEnvWeight / totalWeight) { // Sample from environment map
        uint width, height;
        hdrImage.GetDimensions(width, height);
        uint32_t pixel = sampleAliasTable(sceneData->envAliasTable, width * height, rand(), rand());

        float2 uv = (float2(pixel % width, pixel / width) + float2(rand(), rand())) / float2(width, height);
        float phi = (uv.x - .5f) * 2.f * PI;
        float latitude = (.5f - uv.y) * PI;

        ls.dir = float4(cos(latitude) * cos(phi), sin(latitude), cos(latitude) * sin(phi), INFINITY);
        ls.radiance = hdrImage.SampleLevel(samplers[0], uv, 0.f).rgb;
        ls.pr = envLightPdf(sceneData, uv);
    } else {
        uint32_t light = sampleAliasTable(sceneData->lightAliasTable, sceneData->nLights, rand(), rand());
        float2 uv;
        int matIdx;
        float3 lightNor;
        float3 p = sampleLightTriangle(sceneData->lightTriangles[light], float2(rand(), rand()), uv, matIdx, lightNor);

        float3 toLight = p - origin;
        float dist2 = dot(toLight, toLight);
        float dist = sqrt(dist2);
        ls.dir = float4(toLight / dist, dist);

        Material mat = sceneData->materials[matIdx];
        ls.radiance = mat.emission.rgb * mat.emission.w * ((mat.emissionImgIdx != -1) ?
        sceneImages[NonUniformResourceIndex(mat.emissionImgIdx)]
            .SampleLevel(samplers[NonUniformResourceIndex(mat.emissionSamplerIdx + 1)], uv, 0.f).rgb : float3(1.f));

        ls.pr = triangleLightPdf(sceneData, mat, dist2, abs(dot(lightNor, ls.dir.xyz)));
    }

    return ls;
}

//Tests if the ray hit anything in the scene before it would reach whatever is tMax away in the direction
//Having hit nothing means that we can see the light
inline bool shadowRayCast(float3 origin, float3 dir, float tMax) {
    RayDesc r;
    r.Origin = origin;
    r.Direction = dir;
    r.TMin = EPSILON;
    r.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, r);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

[shader("raygeneration")]
void rayGeneration() {
    static const uint SHARC_TILE_SIZE = 5;
    uint renderWidth;
    uint renderHeight;
    radianceImage.GetDimensions(renderWidth, renderHeight);
    uint2 imageSize = uint2(renderWidth, renderHeight);

    uint2 tileIndex = DispatchRaysIndex().xy;
    uint2 tileOrigin = tileIndex * SHARC_TILE_SIZE;

    if (any(tileOrigin >= imageSize)) // origin of tile outside image
        return;

    // Account for partial tiles at right/bottom edges
    uint2 tileExtent = min(
        uint2(SHARC_TILE_SIZE, SHARC_TILE_SIZE),
        imageSize - tileOrigin);

    uint tilePixelCount = tileExtent.x * tileExtent.y;

    uint tilePhase =
        pcg(tileIndex.x ^ pcg(tileIndex.y)) % tilePixelCount;
    uint tileSample =
        (tilePhase + pc.totalFrames) % tilePixelCount;

    uint2 tileOffset = uint2(
        tileSample % tileExtent.x,
        tileSample / tileExtent.x);

    uint2 pixel = tileOrigin + tileOffset; //  choose one pixel for 5x5 tile

    seed = pcg2(pixel) ^ pcg(pc.totalFrames);

    float2 samplePos = float2(pixel) + float2(rand(), rand()); // jitter pixel

    RayDesc ray;
    {
        float2 imageSizeFloat = float2(imageSize);
        float aspect = imageSizeFloat.x / imageSizeFloat.y;
        float2 ndc = float2((samplePos.x - imageSizeFloat.x / 2.f) / (imageSizeFloat.x / 2.f), (imageSizeFloat.y / 2.f - samplePos.y) / (imageSizeFloat.y / 2.f));
        float3 dir = normalize(pc.camForward + pc.camRight * tan(pc.fov / 2.f) * ndc.x * aspect + pc.camUp * tan(pc.fov / 2.f) * ndc.y);

        float2 lensSample = toDiskConcentric(float2(rand(), rand()));
        //Get the direction of the ray by finding the point on the focus plane, subtracting the point on the lens we're sampling from, and normalizing
        ray.Origin = pc.camPos + pc.camUp * lensSample.x * pc.lensRadius + pc.camRight * lensSample.y * pc.lensRadius;
        ray.Direction = normalize(pc.camPos + dir * pc.focusDist / dot(dir, pc.camForward) - ray.Origin);
    }
    ray.TMin = EPSILON; // Avoid self-intersection
    ray.TMax = INFINITY;

    // SHARC init
    SharcParameters sharcParameters = createSharcParams(pc.camPos, sharcConfigBuffer[0], sharcHashEntries, sharcAccumulation, sharcResolved, sharcLockBuffer);

    SharcState sharcState;
    SharcInit(sharcState);

    float3 rrPathThroughput = float3(1.f);

    UpdateHitPayload payload = {};
    for (uint i = 0; i < pc.maxBounces; ++i) {
        payload.seed = seed;
        payload.throughput = float3(0.f);
        payload.pdf = 0.f;
        payload.directLight = float3(0.f);
        payload.sharcLocalRadiance = float3(0.f);
        payload.sharcSurfaceValid = 0;

        HitObject hitObject = HitObject::TraceRay(tlas, 0, 0xFF, 0, 0, 0, ray, payload);
        ReorderThread(hitObject);
        HitObject::Invoke<UpdateHitPayload>(tlas, hitObject, payload);
        seed = payload.seed;

        // Miss, return environment radiance
        if (!hitObject.IsHit())
        {
            payload.throughput.b *= -1.f;

            SharcUpdateMiss( sharcParameters, sharcState, payload.throughput);

            break;
        }

        // Valid hit
        if (payload.sharcSurfaceValid != 0)
        {
            SharcHitData sharcHitData;
            sharcHitData.positionWorld = payload.sharcPositionWorld;
            sharcHitData.normalWorld = payload.sharcNormalWorld;

            if (!SharcUpdateHit(sharcParameters, sharcState, sharcHitData, payload.sharcLocalRadiance, rand()))
                break;
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
        rrPathThroughput *= segmentThroughput;

        // RR
        if (i > 2)
        {
            float q = clamp(max(rrPathThroughput.x, max(rrPathThroughput.y, rrPathThroughput.z)), 0.05f, 1.f);

            if (rand() >= q)
                break;

            // Include rr to throughput
            segmentThroughput /= q;
            rrPathThroughput /= q;
        }

        SharcSetThroughput(sharcState, segmentThroughput);

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDir;
        
    }
}

[shader("closesthit")]
void closestHit(inout UpdateHitPayload payload, in BuiltInTriangleIntersectionAttributes isect) {
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
            float3 worldGNor = normalize(mul(gNor, WorldToObject3x4()).xyz);
            payload.pdf = triangleLightPdf(pc.sceneData, material, RayTCurrent() * RayTCurrent(), abs(dot(worldGNor, WorldRayDirection())));
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
            payload.sharcSurfaceValid = 0;
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

    float3 worldGeometryNormal = normalize(mul(gNor, WorldToObject3x4()).xyz);
    // SHARC populate geometric data
    payload.sharcPositionWorld = hitPos;
    payload.sharcNormalWorld = worldGeometryNormal;
    payload.sharcSurfaceValid = 1;

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
    payload.throughput = hdrImage.SampleLevel(samplers[0], uv, 0.f).rgb;
    payload.pdf = envLightPdf(pc.sceneData, uv);
    payload.throughput.b = min(-payload.throughput.b, -EPSILON);
}

[shader("anyhit")]
void anyHit() {

}