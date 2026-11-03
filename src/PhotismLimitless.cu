#include "PhotismParams.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct half4 {
    half x, y, z, w;
};

inline __device__ float4 Half4ToFloat4(half4 v) {
    float4 out;
    out.x = __half2float(v.x);
    out.y = __half2float(v.y);
    out.z = __half2float(v.z);
    out.w = __half2float(v.w);
    return out;
}

inline __device__ half4 Float4ToHalf4(float4 v) {
    half4 out;
    out.x = __float2half_rn(v.x);
    out.y = __float2half_rn(v.y);
    out.z = __float2half_rn(v.z);
    out.w = __float2half_rn(v.w);
    return out;
}

// -----------------------------------------------------------------------------
// Procedural Fast Noise & Curl Functions
// -----------------------------------------------------------------------------
inline __device__ float fract_dev(float x) {
    return x - floorf(x);
}

inline __device__ float hash21(float2 p) {
    float3 p3 = make_float3(fract_dev(p.x * 0.1031f), fract_dev(p.y * 0.1030f), fract_dev(p.x * 0.0973f));
    float d = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    p3.x += d; p3.y += d; p3.z += d;
    return fract_dev((p3.x + p3.y) * p3.z);
}

inline __device__ float noise2D(float2 p) {
    float2 i = make_float2(floorf(p.x), floorf(p.y));
    float2 f = make_float2(fract_dev(p.x), fract_dev(p.y));
    float2 u = make_float2(f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f),
                           f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f));

    float a = hash21(i);
    float b = hash21(make_float2(i.x + 1.0f, i.y));
    float c = hash21(make_float2(i.x, i.y + 1.0f));
    float d = hash21(make_float2(i.x + 1.0f, i.y + 1.0f));

    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
}

inline __device__ float fbm3(float2 p) {
    float v = 0.0f;
    float a = 0.5f;
    v += a * noise2D(p); p.x *= 2.02f; p.y *= 2.03f; a *= 0.5f;
    v += a * noise2D(p); p.x *= 2.01f; p.y *= 2.04f; a *= 0.5f;
    v += a * noise2D(p);
    return v;
}

// -----------------------------------------------------------------------------
// Phase 0: Trivial Invert + Vignette Test Kernel
// -----------------------------------------------------------------------------
__global__ void kTrivialCUDA(
    float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    float vignetteAmount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float4 pixel;
    if (is16f) {
        pixel = Half4ToFloat4(((half4*)destImg)[y * destPitch + x]);
    } else {
        pixel = destImg[y * destPitch + x];
    }

    float nx = (static_cast<float>(x) / static_cast<float>(width) - 0.5f) * 2.0f;
    float ny = (static_cast<float>(y) / static_cast<float>(height) - 0.5f) * 2.0f;
    float r2 = nx * nx + ny * ny;
    float vig = 1.0f - fminf(1.0f, r2 * vignetteAmount * 0.5f);

    pixel.x = (1.0f - pixel.x) * vig;
    pixel.y = (1.0f - pixel.y) * vig;
    pixel.z = (1.0f - pixel.z) * vig;

    if (is16f) {
        ((half4*)destImg)[y * destPitch + x] = Float4ToHalf4(pixel);
    } else {
        destImg[y * destPitch + x] = pixel;
    }
}

// -----------------------------------------------------------------------------
// Phase 1: Smear Mode CUDA Pipeline
// -----------------------------------------------------------------------------
__global__ void kSmear_PrepassSobel(
    const float4* srcImg,
    float2* outGrad, // RG = (dx, dy)
    int srcPitch,
    int is16f,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    auto getLuma = [&](int cx, int cy) -> float {
        cx = max(0, min(width - 1, cx));
        cy = max(0, min(height - 1, cy));
        float4 p;
        if (is16f) {
            p = Half4ToFloat4(((const half4*)srcImg)[cy * srcPitch + cx]);
        } else {
            p = srcImg[cy * srcPitch + cx];
        }
        return 0.2126f * p.z + 0.7152f * p.y + 0.0722f * p.x; // BGRA order
    };

    float tl = getLuma(x - 1, y - 1);
    float tc = getLuma(x,     y - 1);
    float tr = getLuma(x + 1, y - 1);
    float ml = getLuma(x - 1, y);
    float mr = getLuma(x + 1, y);
    float bl = getLuma(x - 1, y + 1);
    float bc = getLuma(x,     y + 1);
    float br = getLuma(x + 1, y + 1);

    float dx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
    float dy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);

    outGrad[y * width + x] = make_float2(dx * 0.125f, dy * 0.125f);
}

__global__ void kSmear_AdvectFeedback(
    const float4* srcImg,
    const float2* gradImg,
    const float4* historyIn,
    float4* historyOut,
    float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    PhotismParamsGPU params)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float aspect = static_cast<float>(width) / static_cast<float>(height);
    float u = static_cast<float>(x) / static_cast<float>(width);
    float v = static_cast<float>(y) / static_cast<float>(height);
    float2 p = make_float2((u - 0.5f) * aspect * 2.0f, (v - 0.5f) * 2.0f);
    float r = sqrtf(p.x * p.x + p.y * p.y);

    // 1. Curl noise flow field
    float flowScale = 1.5f + (params.complexity * 0.01f) * 2.5f;
    float2 noiseCoord = make_float2(p.x * flowScale + params.phase * 0.13f,
                                    p.y * flowScale - params.phase * 0.09f);
    float eps = 0.01f;
    float psi = fbm3(noiseCoord);
    float dx_n = fbm3(make_float2(noiseCoord.x + eps, noiseCoord.y)) - psi;
    float dy_n = fbm3(make_float2(noiseCoord.x, noiseCoord.y + eps)) - psi;
    float2 vCurl = make_float2(dy_n / eps, -dx_n / eps);

    // 2. Image edge-steered flow
    float2 sGrad = gradImg[y * width + x];
    float gradLen = sqrtf(sGrad.x * sGrad.x + sGrad.y * sGrad.y) + 1e-5f;
    float2 tangent = make_float2(-sGrad.y / gradLen, sGrad.x / gradLen);
    float2 vEdge = make_float2(tangent.x * 0.08f + sGrad.x * 0.04f,
                               tangent.y * 0.08f + sGrad.y * 0.04f);

    // 3. Radial and rotational swirl drive
    float2 radial = make_float2(p.x / max(r, 1e-4f), p.y / max(r, 1e-4f));
    float2 swirl = make_float2(-radial.y, radial.x);

    float effReact = params.react * 0.01f;
    float effPulse = params.pulse * 0.01f * effReact;
    float effLow   = params.driveLow * 0.01f * effReact;
    float effMid   = params.driveMid * 0.01f * effReact;
    float effChaos = (params.chaos * 0.01f) * (1.0f + effMid * 1.5f);

    float zoomPulse = effPulse * 0.25f + effLow * 0.10f;
    float twistRate = (params.twist * 0.01f) * 2.0f;

    float2 velocity = make_float2(
        vCurl.x * effChaos * 0.5f + vEdge.x + radial.x * zoomPulse + swirl.x * twistRate * (0.25f + r),
        vCurl.y * effChaos * 0.5f + vEdge.y + radial.y * zoomPulse + swirl.y * twistRate * (0.25f + r)
    );

    // Semi-Lagrangian Backtrace
    float dt = params.dt > 0.0f ? params.dt : 0.01667f;
    float2 backP = make_float2(p.x - velocity.x * dt * 2.0f,
                               p.y - velocity.y * dt * 2.0f);

    float backU = backP.x / (aspect * 2.0f) + 0.5f;
    float backV = backP.y * 0.5f + 0.5f;

    float hx = fmaxf(0.0f, fminf(static_cast<float>(width - 1), backU * width));
    float hy = fmaxf(0.0f, fminf(static_cast<float>(height - 1), backV * height));
    int x0 = static_cast<int>(floorf(hx));
    int y0 = static_cast<int>(floorf(hy));
    int x1 = min(width - 1, x0 + 1);
    int y1 = min(height - 1, y0 + 1);
    float fx = hx - x0;
    float fy = hy - y0;

    float4 h00 = historyIn[y0 * width + x0];
    float4 h10 = historyIn[y0 * width + x1];
    float4 h01 = historyIn[y1 * width + x0];
    float4 h11 = historyIn[y1 * width + x1];

    float4 advectedHist;
    if (params.trails > 50.0f) {
        float2 backP2 = make_float2(p.x - velocity.x * dt * 1.0f,
                                    p.y - velocity.y * dt * 1.0f);
        float backU2 = backP2.x / (aspect * 2.0f) + 0.5f;
        float backV2 = backP2.y * 0.5f + 0.5f;
        float hx2 = fmaxf(0.0f, fminf(static_cast<float>(width - 1), backU2 * width));
        float hy2 = fmaxf(0.0f, fminf(static_cast<float>(height - 1), backV2 * height));
        int x0_2 = static_cast<int>(floorf(hx2));
        int y0_2 = static_cast<int>(floorf(hy2));
        int x1_2 = min(width - 1, x0_2 + 1);
        int y1_2 = min(height - 1, y0_2 + 1);
        float fx2 = hx2 - x0_2;
        float fy2 = hy2 - y0_2;
        float4 h00_2 = historyIn[y0_2 * width + x0_2];
        float4 h10_2 = historyIn[y0_2 * width + x1_2];
        float4 h01_2 = historyIn[y1_2 * width + x0_2];
        float4 h11_2 = historyIn[y1_2 * width + x1_2];
        float4 tap2;
        tap2.x = (1.0f - fx2) * (1.0f - fy2) * h00_2.x + fx2 * (1.0f - fy2) * h10_2.x + (1.0f - fx2) * fy2 * h01_2.x + fx2 * fy2 * h11_2.x;
        tap2.y = (1.0f - fx2) * (1.0f - fy2) * h00_2.y + fx2 * (1.0f - fy2) * h10_2.y + (1.0f - fx2) * fy2 * h01_2.y + fx2 * fy2 * h11_2.y;
        tap2.z = (1.0f - fx2) * (1.0f - fy2) * h00_2.z + fx2 * (1.0f - fy2) * h10_2.z + (1.0f - fx2) * fy2 * h01_2.z + fx2 * fy2 * h11_2.z;

        float4 tap1;
        tap1.x = (1.0f - fx) * (1.0f - fy) * h00.x + fx * (1.0f - fy) * h10.x + (1.0f - fx) * fy * h01.x + fx * fy * h11.x;
        tap1.y = (1.0f - fx) * (1.0f - fy) * h00.y + fx * (1.0f - fy) * h10.y + (1.0f - fx) * fy * h01.y + fx * fy * h11.y;
        tap1.z = (1.0f - fx) * (1.0f - fy) * h00.z + fx * (1.0f - fy) * h10.z + (1.0f - fx) * fy * h01.z + fx * fy * h11.z;

        advectedHist.x = tap1.x * 0.65f + tap2.x * 0.35f;
        advectedHist.y = tap1.y * 0.65f + tap2.y * 0.35f;
        advectedHist.z = tap1.z * 0.65f + tap2.z * 0.35f;
        advectedHist.w = 1.0f;
    } else {
        advectedHist.x = (1.0f - fx) * (1.0f - fy) * h00.x + fx * (1.0f - fy) * h10.x + (1.0f - fx) * fy * h01.x + fx * fy * h11.x;
        advectedHist.y = (1.0f - fx) * (1.0f - fy) * h00.y + fx * (1.0f - fy) * h10.y + (1.0f - fx) * fy * h01.y + fx * fy * h11.y;
        advectedHist.z = (1.0f - fx) * (1.0f - fy) * h00.z + fx * (1.0f - fy) * h10.z + (1.0f - fx) * fy * h01.z + fx * fy * h11.z;
        advectedHist.w = 1.0f;
    }

    // Decay: range 0.005s (clears <= 3 frames) to 4.0s (trails persist)
    float decayNorm = params.decay * 0.01f;
    float halfLife = 0.005f * powf(800.0f, decayNorm);
    float retention = expf(-0.693147f * dt / halfLife);

    float trailMix = params.trails * 0.01f;

    // Heat shimmer
    float heatScale = (params.heat * 0.01f) * (1.0f + effLow * 0.5f);
    float2 refrUV = make_float2(u + vCurl.x * heatScale * 0.008f,
                                v + vCurl.y * heatScale * 0.008f);
    int su = max(0, min(width - 1, static_cast<int>(refrUV.x * width)));
    int sv = max(0, min(height - 1, static_cast<int>(refrUV.y * height)));

    int sPitch = params.srcPitch > 0 ? params.srcPitch : destPitch;
    float4 fresh;
    if (is16f) {
        fresh = Half4ToFloat4(((const half4*)srcImg)[sv * sPitch + su]);
    } else {
        fresh = srcImg[sv * sPitch + su];
    }

    float injectGain = (params.inputGain * 0.01f) * 1.5f + effPulse * 0.8f;
    float4 freshInjected = make_float4(fresh.x * injectGain, fresh.y * injectGain, fresh.z * injectGain, 1.0f);

    float4 feedback;
    feedback.x = fmaxf(freshInjected.x, advectedHist.x * retention * trailMix);
    feedback.y = fmaxf(freshInjected.y, advectedHist.y * retention * trailMix);
    feedback.z = fmaxf(freshInjected.z, advectedHist.z * retention * trailMix);
    feedback.w = 1.0f;

    // Liquid fire white core: highlight accumulation lifts channels towards white blowout
    float maxCh = fmaxf(feedback.x, fmaxf(feedback.y, feedback.z));
    if (maxCh > 0.65f) {
        float bloom = powf(maxCh - 0.65f, 1.1f) * (2.2f + (params.heat * 0.01f) * 4.0f);
        feedback.x += bloom;
        feedback.y += bloom;
        feedback.z += bloom;
    }

    historyOut[y * width + x] = feedback;

    // Tone map & finish
    float exposure = powf(2.0f, (params.brightness - 50.0f) * 0.04f);
    float4 tonemapped;
    tonemapped.x = 1.0f - expf(-feedback.x * exposure);
    tonemapped.y = 1.0f - expf(-feedback.y * exposure);
    tonemapped.z = 1.0f - expf(-feedback.z * exposure);
    tonemapped.w = 1.0f;

    if (params.vignette > 0.0f) {
        float vig = 1.0f - fminf(1.0f, (r * r) * (params.vignette * 0.01f));
        tonemapped.x *= vig;
        tonemapped.y *= vig;
        tonemapped.z *= vig;
    }

    float wet = params.dryWet * 0.01f;
    float4 outColor;
    outColor.x = fresh.x * (1.0f - wet) + tonemapped.x * wet;
    outColor.y = fresh.y * (1.0f - wet) + tonemapped.y * wet;
    outColor.z = fresh.z * (1.0f - wet) + tonemapped.z * wet;
    outColor.w = 1.0f;

    if (is16f) {
        ((half4*)destImg)[y * destPitch + x] = Float4ToHalf4(outColor);
    } else {
        destImg[y * destPitch + x] = outColor;
    }
}

// -----------------------------------------------------------------------------
// Host Entry Points for CUDA Kernels
// -----------------------------------------------------------------------------
extern "C" void LaunchTrivialKernel_CUDA(
    float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    float vignetteAmount)
{
    dim3 threads(16, 16);
    dim3 grid((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

    kTrivialCUDA<<<grid, threads>>>(destImg, destPitch, is16f, width, height, vignetteAmount);
}

extern "C" void LaunchSmearPipeline_CUDA(
    const float4* srcImg,
    float2* gradBuffer,
    const float4* historyIn,
    float4* historyOut,
    float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    PhotismParamsGPU params)
{
    dim3 threads(16, 16);
    dim3 grid((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

    int sPitch = params.srcPitch > 0 ? params.srcPitch : destPitch;
    kSmear_PrepassSobel<<<grid, threads>>>(srcImg, gradBuffer, sPitch, is16f, width, height);

    kSmear_AdvectFeedback<<<grid, threads>>>(srcImg, gradBuffer, historyIn, historyOut, destImg,
                                            destPitch, is16f, width, height, params);
}
