#include "bsdf.slang"
#include "bindings.slang"

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