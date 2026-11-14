// SHARC defs
#define SHARC_QUERY 1
#define SHARC_UPDATE 1

//Holy shit, I think I love slang
#include "bsdf.slang"

#include "sampling.slang"
#include "bindings.slang"
// SHARC 

[shader("raygeneration")]
void rayGeneration() {
    seed = pcg2(DispatchRaysIndex().xy) ^ pcg(pc.totalFrames);

    float2 samplePos = (float2)DispatchRaysIndex().xy;
    float2 imageSize = (float2)DispatchRaysDimensions().xy;

    samplePos += float2(rand(), rand()); //Add jitter for anti-aliasing

    RayDesc ray;
    {
        float aspect = imageSize.x / imageSize.y;
        float2 ndc = float2((samplePos.x - imageSize.x / 2.f) / (imageSize.x / 2.f), (imageSize.y / 2.f - samplePos.y) / (imageSize.y / 2.f));
        float3 dir = normalize(pc.camForward + pc.camRight * tan(pc.fov / 2.f) * ndc.x * aspect + pc.camUp * tan(pc.fov / 2.f) * ndc.y);

        float2 lensSample = toDiskConcentric(float2(rand(), rand()));
        //Get the direction of the ray by finding the point on the focus plane, subtracting the point on the lens we're sampling from, and normalizing
        ray.Origin = pc.camPos + pc.camUp * lensSample.x * pc.lensRadius + pc.camRight * lensSample.y * pc.lensRadius;
        ray.Direction = normalize(pc.camPos + dir * pc.focusDist / dot(dir, pc.camForward) - ray.Origin);
    }
    ray.TMin = EPSILON; // Avoid self-intersection
    ray.TMax = INFINITY;

    float3 throughput = float3(1.f);
    float3 accumulatedLight = float3(0.f);
    float prevBsdfPdf = 0.f; // 0 when no light was sampled at the previous vertex (camera ray), so hitting a light gets the full weight
    
    float prevRoughness = 1.f;
    
    #if SHARC_QUERY
        SharcParameters params = createSharcParams(pc.camPos, sharcConfigBuffer[0], sharcHashEntries, sharcAccumulation, sharcResolved, sharcLockBuffer);
    #endif

    HitPayload payload = {};
    for (uint i = 0; i < pc.maxBounces; ++i) {
        payload.seed = seed;

        //For SER
        HitObject hitObject = HitObject::TraceRay(tlas, 0, 0xFF, 0, 0, 0, ray, payload);
        ReorderThread(hitObject);
        HitObject::Invoke<HitPayload>(tlas, hitObject, payload);
        seed = payload.seed;

        // SHARC QUERY/RENDER stage integration
        // we not considering the first (direct) bounce, if cache hit we use cache and terminate
        #if SHARC_QUERY
        if (hitObject.IsHit() && prevBsdfPdf > 0.f && (payload.throughput.b >= 0.f) && (payload.pdf > 0.f)) {
            float3 hitPos = ray.Origin + ray.Direction * payload.hitLen;
            float voxelSize = HashGridGetVoxelSize(HashGridGetLevel(hitPos, params.hashGridParameters), params.hashGridParameters);
            // cannot be shorter than voxel length
            bool lengthCheck = payload.hitLen > voxelSize * sqrt(3.f);

            // cone spread eq from SHARC guide:
            // 2.0∗ray.length∗sqrt(0.5∗a2/(1−a2)
            float roughness = prevRoughness;
            float alphaSq = roughness * roughness * roughness * roughness; // a is material SQUARED
            bool widthCheck = (2.0f * payload.hitLen * sqrt(0.5f * alphaSq / max(1.0f - alphaSq, 1e-6f))) > voxelSize;

            SharcHitData hitData;
            hitData.positionWorld = hitPos;
            hitData.normalWorld = payload.hitNor;

            float3 cacheData;
            bool getSuccess = SharcGetCachedRadiance(params, hitData, cacheData, false);
            if (lengthCheck && widthCheck && getSuccess) {
                accumulatedLight += throughput * cacheData;
                break;
            } 
        }
        #endif

        if (hitObject.IsHit()) accumulatedLight += throughput * select(isnan(payload.directLight), float3(0.f), payload.directLight);

        //A negative blue channel means the path ended, on an emitter or the environment
        if (!hitObject.IsHit() || payload.throughput.b < 0.f) {
            payload.throughput.b *= -1.f;
            //MIS weight of the BSDF sample against the chance of light sampling having picked the same point (payload.pdf)
            float misWeight = prevBsdfPdf > 0.f ? powerHeuristic(1, prevBsdfPdf, 1, payload.pdf) : 1.f;
            accumulatedLight += throughput * payload.throughput.rgb * misWeight;
            break;
        }

        //A pdf of 0 means we passed straight through an alpha-cut surface, so the throughput and pdf carry over
        if (payload.pdf > 0.f) {
            float3 weight = payload.throughput.rgb / payload.pdf;
            throughput *= select(isnan(weight), float3(0.f), weight);
            prevBsdfPdf = payload.pdf;
            prevRoughness = payload.roughness;
        }

        // RR
        if (i > 2) {
            float q = clamp(max(throughput.x, max(throughput.y, throughput.z)), .05f, 1.f);
            if (rand() >= q) break;
            throughput /= q;
        }

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDir;
    }

    if (pc.flags & EngineStateFlags::RAY_RECONSTRUCTION) {
        radianceImage[DispatchRaysIndex().xy] = radianceImage[DispatchRaysIndex().xy] * float(pc.frameNum) / float(pc.frameNum + 1) +
            float4(accumulatedLight, 1.f) * 1.f / float(pc.frameNum + 1);
    } else {
        radianceImage[DispatchRaysIndex().xy] = radianceImage[DispatchRaysIndex().xy] * float(pc.frameNum) / float(pc.frameNum + 1) +
            float4(accumulatedLight, 1.f) * 1.f / float(pc.frameNum + 1);
    }
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

    //Write out information to the shared DLSS/ORCA GBuffer images
    if (!payload.gBufferWritten) {
        payload.gBufferWritten = true;
        diffuseAlbedoImage[DispatchRaysIndex().xy] = float4(gAlbedo, pbr.metallicRoughness.x);
        specularAlbedoImage[DispatchRaysIndex().xy] = float4(gSpecular, 1.f);
        normalRoughnessImage[DispatchRaysIndex().xy] = float4(worldNor, pbr.metallicRoughness.y);
        depthImage[DispatchRaysIndex().xy] = RayTCurrent();
        // TODO: Specular hit dist image via raycast...?
        // TODO: Motion vectors
        orcaAlbedoImage[DispatchRaysIndex().xy] = pbr.color;

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

    if (!payload.gBufferWritten) {
        payload.gBufferWritten = true;
        diffuseAlbedoImage[DispatchRaysIndex().xy] = float4(0.f);
        specularAlbedoImage[DispatchRaysIndex().xy] = float4(0.f);
        normalRoughnessImage[DispatchRaysIndex().xy] = float4(0.f);
        orcaAlbedoImage[DispatchRaysIndex().xy] = float4(float3(0.f), 1.f);
        depthImage[DispatchRaysIndex().xy] = INFINITY;
    }
}

[shader("anyhit")]
void anyHit() {

}