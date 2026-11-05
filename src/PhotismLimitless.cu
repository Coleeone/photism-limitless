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
// Procedural Fast Noise, Curl & Helper Functions
// -----------------------------------------------------------------------------
inline __device__ float fract_dev(float x) {
    return x - floorf(x);
}

inline __device__ float smoothstep_dev(float e0, float e1, float x) {
    float t = fminf(1.0f, fmaxf(0.0f, (x - e0) / fmaxf(1e-5f, e1 - e0)));
    return t * t * (3.0f - 2.0f * t);
}

inline __device__ float hash21(float2 p) {
    float3 p3 = make_float3(fract_dev(p.x * 0.1031f), fract_dev(p.y * 0.1030f), fract_dev(p.x * 0.0973f));
    float d = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    p3.x += d; p3.y += d; p3.z += d;
    return fract_dev((p3.x + p3.y) * p3.z);
}

inline __device__ float2 hash22(float2 p) {
    float n = hash21(p);
    return make_float2(n, hash21(make_float2(p.x + n + 19.19f, p.y + 7.73f)));
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

inline __device__ float4 sampleSrcBilinear(
    const float4* srcImg,
    int sPitch,
    int is16f,
    int width,
    int height,
    float u,
    float v)
{
    // Mirrored repeat for seamless tunnel/spiral wrapping when coordinates exceed [0,1]
    float wu = fract_dev(u);
    float wv = fract_dev(v);
    int iu = static_cast<int>(floorf(u));
    int iv = static_cast<int>(floorf(v));
    if (iu & 1) wu = 1.0f - wu;
    if (iv & 1) wv = 1.0f - wv;

    float fx = fmaxf(0.0f, fminf(static_cast<float>(width - 1), wu * (width - 1)));
    float fy = fmaxf(0.0f, fminf(static_cast<float>(height - 1), wv * (height - 1)));
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    int x1 = min(width - 1, x0 + 1);
    int y1 = min(height - 1, y0 + 1);
    float tx = fx - x0;
    float ty = fy - y0;

    float4 c00, c10, c01, c11;
    if (is16f) {
        const half4* hSrc = reinterpret_cast<const half4*>(srcImg);
        c00 = Half4ToFloat4(hSrc[y0 * sPitch + x0]);
        c10 = Half4ToFloat4(hSrc[y0 * sPitch + x1]);
        c01 = Half4ToFloat4(hSrc[y1 * sPitch + x0]);
        c11 = Half4ToFloat4(hSrc[y1 * sPitch + x1]);
    } else {
        c00 = srcImg[y0 * sPitch + x0];
        c10 = srcImg[y0 * sPitch + x1];
        c01 = srcImg[y1 * sPitch + x0];
        c11 = srcImg[y1 * sPitch + x1];
    }

    float4 res;
    res.x = (1.0f - tx) * (1.0f - ty) * c00.x + tx * (1.0f - ty) * c10.x + (1.0f - tx) * ty * c01.x + tx * ty * c11.x;
    res.y = (1.0f - tx) * (1.0f - ty) * c00.y + tx * (1.0f - ty) * c10.y + (1.0f - tx) * ty * c01.y + tx * ty * c11.y;
    res.z = (1.0f - tx) * (1.0f - ty) * c00.z + tx * (1.0f - ty) * c10.z + (1.0f - tx) * ty * c01.z + tx * ty * c11.z;
    res.w = c00.w;
    return res;
}

inline __device__ float3 applyHueRotation(float3 rgb, float angleRad) {
    if (fabsf(angleRad) < 1e-4f) return rgb;
    float cosA = cosf(angleRad);
    float sinA = sinf(angleRad);
    float oneMinusCos = 1.0f - cosA;
    float k = 0.57735027f; // 1/sqrt(3)
    float dotVal = k * (rgb.x + rgb.y + rgb.z);
    float3 crossVal = make_float3(k * (rgb.z - rgb.y), k * (rgb.x - rgb.z), k * (rgb.y - rgb.x));
    return make_float3(
        rgb.x * cosA + crossVal.x * sinA + k * dotVal * oneMinusCos,
        rgb.y * cosA + crossVal.y * sinA + k * dotVal * oneMinusCos,
        rgb.z * cosA + crossVal.z * sinA + k * dotVal * oneMinusCos
    );
}

inline __device__ float3 applyPaletteTint(float3 rgb, int palette, float phase) {
    if (palette <= PALETTE_CUSTOM) return rgb;
    float luma = 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
    float palHue = static_cast<float>(palette - 1) * 0.261799f + phase * 0.05f;
    float3 tint = make_float3(
        0.5f + 0.5f * cosf(palHue),
        0.5f + 0.5f * cosf(palHue + 2.094395f),
        0.5f + 0.5f * cosf(palHue + 4.188790f)
    );
    return make_float3(
        rgb.x * 0.55f + luma * tint.x * 0.65f,
        rgb.y * 0.55f + luma * tint.y * 0.65f,
        rgb.z * 0.55f + luma * tint.z * 0.65f
    );
}

// -----------------------------------------------------------------------------
// Phase 0: Trivial Invert + Vignette Test Kernel (Used by test_harness.cu)
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
// Sobel Gradient Prepass
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

// -----------------------------------------------------------------------------
// Full Photism Limitless 21-Mode Shader Pipeline
// -----------------------------------------------------------------------------
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

    int sPitch = params.srcPitch > 0 ? params.srcPitch : destPitch;
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    float u = static_cast<float>(x) / static_cast<float>(width);
    float v = static_cast<float>(y) / static_cast<float>(height);

    // Base unwarped source pixel (for Dry/Wet blending)
    float4 rawDry;
    if (is16f) {
        rawDry = Half4ToFloat4(((const half4*)srcImg)[y * sPitch + x]);
    } else {
        rawDry = srcImg[y * sPitch + x];
    }

    // Effective Audio & Tune Modifiers
    float effReact = params.react * 0.01f;
    float effPulse = params.pulse * 0.01f * effReact;
    float effLow   = params.driveLow * 0.01f * effReact;
    float effMid   = params.driveMid * 0.01f * effReact;
    float effHigh  = params.driveHigh * 0.01f * effReact;
    float effChaos = (params.chaos * 0.01f) * (1.0f + effMid * 1.5f);
    float dt = params.dt > 0.0f ? params.dt : 0.01667f;

    // Normalized centered coordinates p in [-aspect..aspect, -1..1]
    float2 p = make_float2((u - 0.5f) * aspect * 2.0f, (v - 0.5f) * 2.0f);

    // Global Coordinate Modifiers: Symmetry, Tiles, Slice, Lens
    if (params.symmetry >= 1.0f) {
        float sectors = floorf(params.symmetry);
        float angle = atan2f(p.y, p.x);
        float radius = sqrtf(p.x * p.x + p.y * p.y);
        float sectorAngle = 6.2831853f / fmaxf(1.0f, sectors);
        float a = fmodf(angle + 3.14159265f, sectorAngle);
        if (a < 0.0f) a += sectorAngle;
        a = fabsf(a - sectorAngle * 0.5f);
        p.x = radius * cosf(a);
        p.y = radius * sinf(a);
    }

    if (params.tiles >= 1.5f) {
        float tCount = floorf(params.tiles);
        float tu = fract_dev((p.x / (aspect * 2.0f) + 0.5f) * tCount);
        float tv = fract_dev((p.y * 0.5f + 0.5f) * tCount);
        p.x = (tu - 0.5f) * aspect * 2.0f;
        p.y = (tv - 0.5f) * 2.0f;
    }

    if (params.slice > 0.0f) {
        float sliceAmt = (params.slice * 0.01f) * (0.08f + effPulse * 0.15f);
        float band = floorf((p.y * 0.5f + 0.5f) * (12.0f + params.complexity * 0.2f));
        float shift = (hash21(make_float2(band, floorf(params.phase * 4.0f))) - 0.5f) * 2.0f;
        p.x += shift * sliceAmt * aspect;
    }

    if (params.lens > 0.0f) {
        float lensAmt = (params.lens * 0.01f) * 0.18f;
        float ribs = 16.0f + (params.complexity * 0.35f);
        if (params.lensStyle == LENS_GLASS_RIBS) {
            p.x += sinf(p.x * ribs + params.phase) * lensAmt;
        } else if (params.lensStyle == LENS_GLASS_FAN) {
            float ang = atan2f(p.y, p.x);
            p.x += cosf(ang * ribs) * lensAmt;
            p.y += sinf(ang * ribs) * lensAmt;
        } else if (params.lensStyle == LENS_ANAMORPHIC) {
            p.x *= (1.0f - lensAmt * 1.5f);
        } else if (params.lensStyle == LENS_MELT) {
            p.y -= fbm3(make_float2(p.x * 3.0f, params.phase * 0.4f)) * lensAmt * 2.0f;
        } else if (params.lensStyle == LENS_GLITCH) {
            float gy = floorf(p.y * 24.0f);
            p.x += (hash21(make_float2(gy, floorf(params.phase * 8.0f))) - 0.5f) * lensAmt * 3.0f;
        } else if (params.lensStyle == LENS_PIXELS) {
            float q = fmaxf(4.0f, 80.0f * (1.0f - params.lens * 0.008f));
            p.x = floorf(p.x * q) / q;
            p.y = floorf(p.y * q) / q;
        } else if (params.lensStyle == LENS_FLY_EYE) {
            float2 cell = make_float2(fract_dev(p.x * 6.0f) - 0.5f, fract_dev(p.y * 6.0f) - 0.5f);
            p.x += cell.x * lensAmt * 2.5f;
            p.y += cell.y * lensAmt * 2.5f;
        }
    }

    float r = sqrtf(p.x * p.x + p.y * p.y);
    float theta = atan2f(p.y, p.x);
    float2 warpedUV = make_float2(p.x / (aspect * 2.0f) + 0.5f, p.y * 0.5f + 0.5f);

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
    float2 radial = make_float2(p.x / fmaxf(r, 1e-4f), p.y / fmaxf(r, 1e-4f));
    float2 swirl  = make_float2(-radial.y, radial.x);
    float zoomPulse = effPulse * 0.25f + effLow * 0.10f;
    float twistRate = (params.twist * 0.01f) * 2.0f;

    float2 velocity = make_float2(
        vCurl.x * effChaos * 0.5f + vEdge.x + radial.x * zoomPulse + swirl.x * twistRate * (0.25f + r),
        vCurl.y * effChaos * 0.5f + vEdge.y + radial.y * zoomPulse + swirl.y * twistRate * (0.25f + r)
    );

    float4 modeColor = rawDry;
    bool useAdvectionFeedback = false;
    float modeTrailScale = 1.0f;

    // -------------------------------------------------------------------------
    // Mode-Specific Shader Engines (Modes 1 .. 21)
    // -------------------------------------------------------------------------
    switch (params.mode) {
        case MODE_NONE: {
            // Clean canvas with global tune modifiers (Symmetry, Tiles, Lens, Vignette, Brightness)
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, warpedUV.y);
            useAdvectionFeedback = false;
            break;
        }

        case MODE_SMEAR: {
            // Look A: Liquid fire curl-noise advection + Sobel edge flow
            float heatScale = (params.heat * 0.01f) * (1.0f + effLow * 0.5f);
            float2 refrUV = make_float2(warpedUV.x + vCurl.x * heatScale * 0.008f,
                                        warpedUV.y + vCurl.y * heatScale * 0.008f);
            int su = max(0, min(width - 1, static_cast<int>(refrUV.x * width)));
            int sv = max(0, min(height - 1, static_cast<int>(refrUV.y * height)));
            if (is16f) {
                modeColor = Half4ToFloat4(((const half4*)srcImg)[sv * sPitch + su]);
            } else {
                modeColor = srcImg[sv * sPitch + su];
            }
            useAdvectionFeedback = true;
            break;
        }

        case MODE_DROSTE: {
            // Look E: Log-polar recursive Escher spiral
            float safeR = fmaxf(r, 1e-4f);
            float branches = fmaxf(1.0f, floorf(1.0f + (params.complexity * 0.01f) * 3.0f));
            float spiralAngle = theta + (1.2f + (params.shape * 0.01f) * 1.5f) * logf(safeR)
                                + twistRate - params.phase * 1.4f;
            float du = fract_dev((spiralAngle / 6.2831853f) * branches);
            float dv = fract_dev(0.65f * logf(safeR) - params.phase * 0.45f - effPulse * 0.2f);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, du, dv);
            // Seam glow along spiral boundary
            float seam = smoothstep_dev(0.45f, 0.5f, fabsf(dv - 0.5f));
            float glow = seam * (params.heat * 0.01f) * 0.6f;
            modeColor.x += glow * 0.3f;
            modeColor.y += glow * 0.5f;
            modeColor.z += glow * 0.9f;
            useAdvectionFeedback = (params.trails > 20.0f);
            modeTrailScale = 0.6f;
            break;
        }

        case MODE_INFINITE: {
            // Look H: Infinity mirror recursive tunnel box
            float boxDist = fmaxf(fabsf(p.x / aspect), fabsf(p.y));
            float safeBox = fmaxf(boxDist, 1e-3f);
            float depthFreq = 2.0f + (params.complexity * 0.01f) * 5.0f;
            float z = -logf(safeBox) * depthFreq - params.phase * 1.2f - effLow * 0.5f;
            float layerIdx = floorf(z);
            float layerFract = fract_dev(z);
            float rot = layerIdx * (twistRate * 0.25f);
            float cosR = cosf(rot), sinR = sinf(rot);
            float2 rp = make_float2((p.x / safeBox) * cosR - (p.y / safeBox) * sinR,
                                    (p.x / safeBox) * sinR + (p.y / safeBox) * cosR);
            float iu = rp.x / (aspect * 2.0f) + 0.5f;
            float iv = rp.y * 0.5f + 0.5f;
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, iu, iv);
            float atten = expf(-fmaxf(0.0f, -logf(safeBox)) * (0.25f + (1.0f - params.fill * 0.01f) * 0.5f));
            float neonFrame = smoothstep_dev(0.88f, 0.98f, layerFract) * (0.4f + (params.heat * 0.01f) * 1.2f);
            modeColor.x = modeColor.x * atten + neonFrame * (0.5f + 0.5f * sinf(layerIdx * 0.7f));
            modeColor.y = modeColor.y * atten + neonFrame * (0.5f + 0.5f * sinf(layerIdx * 0.7f + 2.1f));
            modeColor.z = modeColor.z * atten + neonFrame * (0.5f + 0.5f * sinf(layerIdx * 0.7f + 4.2f));
            useAdvectionFeedback = (params.trails > 30.0f);
            modeTrailScale = 0.5f;
            break;
        }

        case MODE_CORRIDOR: {
            // Look B: 3D perspective architectural corridor / vortex tunnel
            float shapeMix = params.shape * 0.01f;
            float boxR = fmaxf(fabsf(p.x / aspect), fabsf(p.y));
            float geomR = fmaxf(1e-3f, r * (1.0f - shapeMix) + boxR * shapeMix);
            float depth = -1.3f * logf(geomR) + 0.15f / geomR;
            float tunnelAngle = theta + twistRate * 0.35f * depth + params.phase * 0.25f;
            float rep = fmaxf(1.0f, floorf(1.0f + (params.complexity * 0.01f) * 3.0f));
            float tu = fract_dev((tunnelAngle / 6.2831853f) * rep);
            float tv = fract_dev(depth * 0.35f - params.phase * 0.85f - effPulse * 0.25f);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, tu, tv);
            // Center vanishing horizon glow
            float centerGlow = expf(-geomR * (4.5f - (params.fill * 0.01f) * 2.5f)) * (0.5f + (params.heat * 0.01f) * 1.4f);
            modeColor.x += centerGlow * 0.6f;
            modeColor.y += centerGlow * 0.75f;
            modeColor.z += centerGlow * 1.0f;
            useAdvectionFeedback = (params.trails > 25.0f);
            modeTrailScale = 0.55f;
            break;
        }

        case MODE_CUBE: {
            // 3D room box perspective fold
            float ax = fabsf(p.x / aspect);
            float ay = fabsf(p.y);
            float maxAxis = fmaxf(ax, fmaxf(ay, 1e-3f));
            float cu, cv;
            if (maxAxis < 0.35f) {
                // Back wall
                cu = (p.x / aspect) / 0.7f + 0.5f;
                cv = p.y / 0.7f + 0.5f;
            } else if (ax >= ay) {
                // Left/Right walls
                cu = fract_dev((0.35f / ax) - params.phase * 0.4f);
                cv = (p.y / ax) * 0.5f + 0.5f;
            } else {
                // Floor/Ceiling walls
                cu = ((p.x / aspect) / ay) * 0.5f + 0.5f;
                cv = fract_dev((0.35f / ay) - params.phase * 0.4f);
            }
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, cu, cv);
            float edgeSeam = expf(-fabsf(ax - ay) * 28.0f) + expf(-fabsf(maxAxis - 0.35f) * 28.0f);
            float seamBoost = edgeSeam * (0.2f + (params.heat * 0.01f) * 0.8f);
            modeColor.x += seamBoost * 0.4f;
            modeColor.y += seamBoost * 0.7f;
            modeColor.z += seamBoost * 1.0f;
            useAdvectionFeedback = (params.trails > 30.0f);
            modeTrailScale = 0.5f;
            break;
        }

        case MODE_PLANET: {
            // Spherical globe / Tiny Planet projection with outer atmospheric halo
            float sphereRad = 0.72f + (params.fill * 0.01f) * 0.25f + effLow * 0.08f;
            if (r <= sphereRad) {
                float nr = r / sphereRad;
                float lat = asinf(fminf(1.0f, nr));
                float lon = theta + twistRate * (1.0f - nr) + params.phase * 0.5f;
                float pu = fract_dev(lon / 6.2831853f + 0.5f);
                float pv = lat / 1.5707963f;
                modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, pu, pv);
                float limb = powf(1.0f - nr * nr, 0.25f);
                float fresnel = powf(nr, 3.0f) * (params.heat * 0.01f);
                modeColor.x = modeColor.x * limb + fresnel * 0.9f;
                modeColor.y = modeColor.y * limb + fresnel * 0.5f;
                modeColor.z = modeColor.z * limb + fresnel * 0.3f;
            } else {
                // Outer atmospheric halo
                float halo = expf(-(r - sphereRad) * 6.0f);
                float2 bgUV = make_float2(fract_dev(theta / 6.2831853f), fract_dev(1.0f / fmaxf(r, 0.1f) - params.phase * 0.2f));
                float4 bg = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, bgUV.x, bgUV.y);
                modeColor.x = bg.x * 0.25f + halo * 0.7f;
                modeColor.y = bg.y * 0.25f + halo * 0.4f;
                modeColor.z = bg.z * 0.25f + halo * 0.2f;
                modeColor.w = 1.0f;
            }
            useAdvectionFeedback = (params.trails > 30.0f);
            modeTrailScale = 0.5f;
            break;
        }

        case MODE_WINDOWS: {
            // Multi-screen staggered picture-in-picture matrix
            float gridN = floorf(2.0f + (params.complexity * 0.01f) * 6.0f);
            float2 gCoord = make_float2(warpedUV.x * gridN, warpedUV.y * gridN);
            float2 cellID = make_float2(floorf(gCoord.x), floorf(gCoord.y));
            float2 cellUV = make_float2(fract_dev(gCoord.x), fract_dev(gCoord.y));
            float cellHash = hash21(cellID);
            float zoom = 0.7f + 0.3f * sinf(params.phase * 1.5f + cellHash * 6.28f) - effPulse * 0.15f;
            float2 subUV = make_float2((cellUV.x - 0.5f) * zoom + 0.5f, (cellUV.y - 0.5f) * zoom + 0.5f);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, subUV.x, subUV.y);
            // Window mullion bezel
            float bezel = smoothstep_dev(0.0f, 0.04f, cellUV.x) * smoothstep_dev(0.0f, 0.04f, 1.0f - cellUV.x) *
                          smoothstep_dev(0.0f, 0.04f, cellUV.y) * smoothstep_dev(0.0f, 0.04f, 1.0f - cellUV.y);
            modeColor.x *= bezel;
            modeColor.y *= bezel;
            modeColor.z *= bezel;
            break;
        }

        case MODE_MELT: {
            // Look G: Viscous downward gravity liquid paint drip
            float dripFreq = 3.0f + (params.complexity * 0.01f) * 8.0f;
            float colNoise = fbm3(make_float2(p.x * dripFreq, params.phase * 0.35f));
            float dripOffset = colNoise * (0.12f + (params.chaos * 0.01f) * 0.25f + effLow * 0.2f);
            float2 meltUV = make_float2(warpedUV.x + vCurl.x * 0.015f,
                                        warpedUV.y - dripOffset);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, meltUV.x, meltUV.y);
            // Bias advection velocity strongly downward (in UV space, +y is down)
            velocity.x = vCurl.x * 0.35f * effChaos + vEdge.x;
            velocity.y = (0.85f + effLow * 1.2f + colNoise * 0.9f) + vEdge.y;
            useAdvectionFeedback = true;
            break;
        }

        case MODE_SHATTER: {
            // Voronoi crystalline polygon shard fracture
            float shardScale = 4.0f + (params.complexity * 0.01f) * 14.0f;
            float2 sp = make_float2(p.x * shardScale, p.y * shardScale);
            float2 i_sp = make_float2(floorf(sp.x), floorf(sp.y));
            float2 f_sp = make_float2(fract_dev(sp.x), fract_dev(sp.y));
            float minDist1 = 8.0f;
            float minDist2 = 8.0f;
            float2 bestCell = make_float2(0.0f, 0.0f);
            for (int oy = -1; oy <= 1; ++oy) {
                for (int ox = -1; ox <= 1; ++ox) {
                    float2 neighbor = make_float2(static_cast<float>(ox), static_cast<float>(oy));
                    float2 cellId = make_float2(i_sp.x + neighbor.x, i_sp.y + neighbor.y);
                    float2 rnd = hash22(cellId);
                    float2 pt = make_float2(neighbor.x + 0.5f + 0.4f * sinf(params.phase * 0.8f + 6.2831f * rnd.x) - f_sp.x,
                                            neighbor.y + 0.5f + 0.4f * cosf(params.phase * 0.8f + 6.2831f * rnd.y) - f_sp.y);
                    float d = pt.x * pt.x + pt.y * pt.y;
                    if (d < minDist1) {
                        minDist2 = minDist1;
                        minDist1 = d;
                        bestCell = rnd;
                    } else if (d < minDist2) {
                        minDist2 = d;
                    }
                }
            }
            float fractureAmt = (0.04f + (params.chaos * 0.01f) * 0.14f) * (1.0f + effPulse * 1.5f);
            float2 shardUV = make_float2(warpedUV.x + (bestCell.x - 0.5f) * fractureAmt,
                                         warpedUV.y + (bestCell.y - 0.5f) * fractureAmt);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, shardUV.x, shardUV.y);
            float crack = 1.0f - smoothstep_dev(0.0f, 0.06f, sqrtf(minDist2) - sqrtf(minDist1));
            float crackGlow = crack * (params.heat * 0.01f) * 1.5f;
            modeColor.x += crackGlow;
            modeColor.y += crackGlow;
            modeColor.z += crackGlow;
            useAdvectionFeedback = (params.trails > 40.0f);
            modeTrailScale = 0.5f;
            break;
        }

        case MODE_PRISM: {
            // Look F: Per-channel RGB chromatic dispersion tears
            float disp = (0.015f + (params.heat * 0.01f) * 0.06f + effHigh * 0.05f) * (1.0f + effChaos);
            float2 dir = make_float2(vCurl.x * 0.6f + sGrad.x * 2.5f + radial.x * 0.35f,
                                     vCurl.y * 0.6f + sGrad.y * 2.5f + radial.y * 0.35f);
            float4 colB = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x - dir.x * disp, warpedUV.y - dir.y * disp);
            float4 colG = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, warpedUV.y);
            float4 colR = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x + dir.x * disp, warpedUV.y + dir.y * disp);
            modeColor = make_float4(colB.x, colG.y, colR.z, colG.w);
            useAdvectionFeedback = (params.trails > 20.0f);
            modeTrailScale = 0.75f;
            break;
        }

        case MODE_TERRAIN: {
            // 3D luminance heightfield elevation contour ridges
            float4 baseSample = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, warpedUV.y);
            float luma = 0.2126f * baseSample.z + 0.7152f * baseSample.y + 0.0722f * baseSample.x;
            float elev = (luma - 0.5f) * (0.12f + (params.shape * 0.01f) * 0.2f + effLow * 0.15f);
            float2 terrUV = make_float2(warpedUV.x + sGrad.x * 0.25f, warpedUV.y - elev);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, terrUV.x, terrUV.y);
            float lines = 25.0f + (params.complexity * 0.01f) * 55.0f;
            float ridge = powf(0.5f + 0.5f * sinf(warpedUV.y * lines * 6.2831f - luma * 14.0f - params.phase * 2.0f), 4.0f);
            float ridgeMix = 0.35f + 0.65f * ridge;
            modeColor.x *= ridgeMix;
            modeColor.y *= ridgeMix;
            modeColor.z *= ridgeMix;
            break;
        }

        case MODE_BREATHE: {
            // Radial harmonic chest-expansion breathing wave
            float waveFreq = 3.0f + (params.complexity * 0.01f) * 6.0f;
            float breathAmp = (0.06f + (params.chaos * 0.01f) * 0.12f) * sinf(params.phase * 2.2f - r * waveFreq)
                              + effLow * 0.12f + effPulse * 0.15f;
            float scaleR = fmaxf(0.05f, 1.0f - breathAmp);
            float2 bUV = make_float2((p.x * scaleR) / (aspect * 2.0f) + 0.5f,
                                     (p.y * scaleR) * 0.5f + 0.5f);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, bUV.x, bUV.y);
            useAdvectionFeedback = (params.trails > 20.0f);
            modeTrailScale = 0.7f;
            break;
        }

        case MODE_MOSH: {
            // Quantized 16x16 macroblock datamosh displacement
            float blockSz = fmaxf(8.0f, 40.0f - (params.complexity * 0.01f) * 28.0f);
            float bx = floorf(static_cast<float>(x) / blockSz) * blockSz;
            float by = floorf(static_cast<float>(y) / blockSz) * blockSz;
            int bxi = max(0, min(width - 1, static_cast<int>(bx)));
            int byi = max(0, min(height - 1, static_cast<int>(by)));
            float2 bGrad = gradImg[byi * width + bxi];
            float2 bNoise = hash22(make_float2(bx * 0.1f, by * 0.1f + floorf(params.phase * 6.0f)));
            float moshStrength = (0.04f + (params.chaos * 0.01f) * 0.2f) * (1.0f + effPulse * 2.0f);
            float2 moshUV = make_float2(warpedUV.x + (bGrad.x * 4.0f + (bNoise.x - 0.5f)) * moshStrength,
                                        warpedUV.y + (bGrad.y * 4.0f + (bNoise.y - 0.5f)) * moshStrength);
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, moshUV.x, moshUV.y);
            velocity.x = (bNoise.x - 0.5f) * 3.0f * effChaos;
            velocity.y = (bNoise.y - 0.5f) * 3.0f * effChaos;
            useAdvectionFeedback = true;
            break;
        }

        case MODE_SORT: {
            // Vertical column pixel-sorting streak stretch gated by luminance threshold
            float4 curPx = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, warpedUV.y);
            float luma = 0.2126f * curPx.z + 0.7152f * curPx.y + 0.0722f * curPx.x;
            float thresh = 0.15f + (1.0f - params.fill * 0.01f) * 0.55f;
            float colRand = hash21(make_float2(floorf(warpedUV.x * 160.0f), floorf(params.phase * 2.0f)));
            float sortSpan = (0.15f + (params.trails * 0.01f) * 0.45f) * (0.5f + 0.5f * colRand) * (1.0f + effLow);
            float sortedV = warpedUV.y - fmaxf(0.0f, luma - thresh) * sortSpan;
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, sortedV);
            velocity = make_float2(0.0f, 1.5f + effLow * 2.0f);
            useAdvectionFeedback = (params.trails > 25.0f);
            modeTrailScale = 0.75f;
            break;
        }

        case MODE_CORAL: {
            // Reaction-diffusion cellular Turing pattern modulation
            float cellFreq = 6.0f + (params.complexity * 0.01f) * 18.0f;
            float n1 = fbm3(make_float2(p.x * cellFreq + vCurl.x * 0.8f, p.y * cellFreq + vCurl.y * 0.8f));
            float n2 = fbm3(make_float2(p.x * cellFreq * 2.1f - params.phase * 0.3f, p.y * cellFreq * 2.1f + params.phase * 0.3f));
            float coralPattern = sinf((n1 * 4.0f + n2 * 2.0f) * 3.14159f);
            float2 coralUV = make_float2(warpedUV.x + coralPattern * 0.025f * (1.0f + effChaos),
                                         warpedUV.y + coralPattern * 0.025f * (1.0f + effChaos));
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, coralUV.x, coralUV.y);
            float ridge = smoothstep_dev(0.2f, 0.85f, fabsf(coralPattern));
            float boost = 0.65f + 0.7f * ridge;
            modeColor.x *= boost;
            modeColor.y *= boost;
            modeColor.z *= boost;
            useAdvectionFeedback = (params.trails > 25.0f);
            break;
        }

        case MODE_SCANLINE: {
            // Slit-scan horizontal row phase delay + CRT electron beam modulation
            float scanFreq = 80.0f + (params.complexity * 0.01f) * 220.0f;
            float wave = sinf(warpedUV.y * 18.0f - params.phase * 3.5f) * (0.02f + (params.chaos * 0.01f) * 0.08f + effLow * 0.06f);
            float4 scanPx = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x + wave, warpedUV.y);
            float luma = 0.2126f * scanPx.z + 0.7152f * scanPx.y + 0.0722f * scanPx.x;
            float beam = 0.55f + 0.45f * sinf(warpedUV.y * scanFreq * 3.14159f + luma * 6.0f);
            modeColor = make_float4(scanPx.x * beam, scanPx.y * beam, scanPx.z * beam, scanPx.w);
            break;
        }

        case MODE_THRESH: {
            // Look I: High-contrast graphic screen-print posterization with inked Sobel contours
            float4 basePx = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, warpedUV.y);
            float luma = 0.2126f * basePx.z + 0.7152f * basePx.y + 0.0722f * basePx.x;
            float bands = floorf(2.0f + (params.complexity * 0.01f) * 6.0f);
            float qLuma = floorf(luma * bands + 0.5f) / bands;
            float edgeInk = smoothstep_dev(0.08f, 0.22f, gradLen * (2.5f + (params.contrast * 0.01f) * 4.0f));
            float scale = (qLuma / fmaxf(luma, 1e-3f)) * (1.0f - edgeInk * 0.85f);
            modeColor = make_float4(basePx.x * scale, basePx.y * scale, basePx.z * scale, basePx.w);
            break;
        }

        case MODE_SOLAR: {
            // Look I: Psychedelic Sabattier solarization curve with animated threshold inversion
            float4 basePx = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, warpedUV.y);
            float freq = 1.5f + (params.complexity * 0.01f) * 3.5f;
            float phaseShift = params.phase * 0.8f + effLow * 1.5f;
            modeColor.x = 0.5f + 0.5f * sinf(basePx.x * freq * 6.28318f + phaseShift);
            modeColor.y = 0.5f + 0.5f * sinf(basePx.y * freq * 6.28318f + phaseShift + 0.6f);
            modeColor.z = 0.5f + 0.5f * sinf(basePx.z * freq * 6.28318f + phaseShift + 1.2f);
            modeColor.w = basePx.w;
            useAdvectionFeedback = (params.trails > 35.0f);
            modeTrailScale = 0.5f;
            break;
        }

        case MODE_HALFTONE: {
            // Look C: Rotated 30° pop-art LED halftone dot matrix
            float angle = 0.5235988f; // 30 degrees
            float cosA = cosf(angle), sinA = sinf(angle);
            float gridDensity = 24.0f + (params.complexity * 0.01f) * 96.0f;
            float2 rotP = make_float2((p.x * cosA - p.y * sinA) * gridDensity * 0.5f,
                                      (p.x * sinA + p.y * cosA) * gridDensity * 0.5f);
            float2 cellIdx = make_float2(floorf(rotP.x) + 0.5f, floorf(rotP.y) + 0.5f);
            float2 local = make_float2(rotP.x - cellIdx.x, rotP.y - cellIdx.y);
            // Unrotate cell center back to screen UV
            float2 centerP = make_float2((cellIdx.x * cosA + cellIdx.y * sinA) / (gridDensity * 0.5f),
                                         (-cellIdx.x * sinA + cellIdx.y * cosA) / (gridDensity * 0.5f));
            float2 centerUV = make_float2(centerP.x / (aspect * 2.0f) + 0.5f, centerP.y * 0.5f + 0.5f);
            float4 cellColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, centerUV.x, centerUV.y);
            float luma = 0.2126f * cellColor.z + 0.7152f * cellColor.y + 0.0722f * cellColor.x;
            float dotRad = 0.52f * sqrtf(fmaxf(0.0f, luma)) * (0.75f + 0.45f * (params.contrast * 0.01f) + effPulse * 0.25f);
            float dist = sqrtf(local.x * local.x + local.y * local.y);
            float mask = 1.0f - smoothstep_dev(dotRad - 0.06f, dotRad + 0.06f, dist);
            float lampGlow = expf(-dist * dist * 4.5f) * (0.25f + (params.heat * 0.01f) * 0.75f);
            float bgFill = (params.fill * 0.01f) * 0.25f;
            float mult = fmaxf(bgFill, mask + lampGlow * mask);
            modeColor = make_float4(cellColor.x * mult, cellColor.y * mult, cellColor.z * mult, cellColor.w);
            break;
        }

        case MODE_VHS: {
            // Look J: Analog tape tracking wobble, CRT scanlines, and horizontal chroma bleed
            float tapeBand = sinf(warpedUV.y * 6.0f + params.phase * 4.0f) * sinf(warpedUV.y * 23.0f - params.phase * 9.0f);
            float tearLine = smoothstep_dev(0.82f, 0.98f, sinf(warpedUV.y * 3.14159f - params.phase * 1.8f));
            float jitter = (tapeBand * 0.008f + tearLine * 0.045f) * (0.4f + (params.chaos * 0.01f) * 1.5f + effPulse);
            float chromaBleed = 0.006f + (params.heat * 0.01f) * 0.018f;
            float4 cB = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x + jitter - chromaBleed, warpedUV.y);
            float4 cG = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x + jitter, warpedUV.y);
            float4 cR = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x + jitter + chromaBleed, warpedUV.y);
            float scanline = 0.82f + 0.18f * sinf(static_cast<float>(y) * 3.14159f);
            float staticNoise = (hash21(make_float2(warpedUV.x * 300.0f, warpedUV.y * 300.0f + params.phase * 30.0f)) - 0.5f)
                                * tearLine * 0.25f;
            modeColor = make_float4(
                fmaxf(0.0f, cB.x * scanline + staticNoise),
                fmaxf(0.0f, cG.y * scanline + staticNoise),
                fmaxf(0.0f, cR.z * scanline + staticNoise),
                cG.w
            );
            break;
        }

        default: {
            modeColor = sampleSrcBilinear(srcImg, sPitch, is16f, width, height, warpedUV.x, warpedUV.y);
            useAdvectionFeedback = true;
            break;
        }
    }

    // -------------------------------------------------------------------------
    // Semi-Lagrangian Feedback History Accumulation
    // -------------------------------------------------------------------------
    float4 feedback;
    if (useAdvectionFeedback) {
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
        float trailMix = (params.trails * 0.01f) * modeTrailScale;

        float injectGain = (params.inputGain * 0.01f) * 1.5f + effPulse * 0.8f;
        float4 freshInjected = make_float4(modeColor.x * injectGain, modeColor.y * injectGain, modeColor.z * injectGain, 1.0f);

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
    } else {
        float gain = (params.mode == MODE_NONE) ? 1.0f : ((params.inputGain * 0.01f) * 1.25f + effPulse * 0.5f);
        feedback = make_float4(modeColor.x * gain, modeColor.y * gain, modeColor.z * gain, 1.0f);
    }

    historyOut[y * width + x] = feedback;

    // -------------------------------------------------------------------------
    // Global Color Finishing: Hue Drift, Palette, Exposure, Vignette & Dry/Wet
    // -------------------------------------------------------------------------
    float3 rgb = make_float3(feedback.z, feedback.y, feedback.x); // Convert BGRA -> RGB for color math

    if (params.hueDrift > 0.0f) {
        float hueAngle = (params.hueDrift * 0.01f) * (params.phase * 0.8f + r * 1.5f);
        rgb = applyHueRotation(rgb, hueAngle);
    }

    if (params.palette > PALETTE_CUSTOM) {
        rgb = applyPaletteTint(rgb, params.palette, params.phase);
    }

    float4 tonemapped;
    if (params.mode == MODE_NONE) {
        float expScale = powf(2.0f, (params.brightness - 50.0f) * 0.02f);
        tonemapped.x = fminf(1.0f, fmaxf(0.0f, rgb.z * expScale));
        tonemapped.y = fminf(1.0f, fmaxf(0.0f, rgb.y * expScale));
        tonemapped.z = fminf(1.0f, fmaxf(0.0f, rgb.x * expScale));
        tonemapped.w = 1.0f;
    } else {
        float exposure = powf(2.0f, (params.brightness - 50.0f) * 0.04f);
        tonemapped.x = 1.0f - expf(-fmaxf(0.0f, rgb.z) * exposure);
        tonemapped.y = 1.0f - expf(-fmaxf(0.0f, rgb.y) * exposure);
        tonemapped.z = 1.0f - expf(-fmaxf(0.0f, rgb.x) * exposure);
        tonemapped.w = 1.0f;
    }

    if (params.vignette > 0.0f) {
        float vig = 1.0f - fminf(1.0f, (r * r) * (params.vignette * 0.01f));
        tonemapped.x *= vig;
        tonemapped.y *= vig;
        tonemapped.z *= vig;
    }

    float wet = params.dryWet * 0.01f;
    float4 outColor;
    outColor.x = rawDry.x * (1.0f - wet) + tonemapped.x * wet;
    outColor.y = rawDry.y * (1.0f - wet) + tonemapped.y * wet;
    outColor.z = rawDry.z * (1.0f - wet) + tonemapped.z * wet;
    outColor.w = rawDry.w;

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
