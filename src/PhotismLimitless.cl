R"===(
__kernel void kTrivialCL(
    __global float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    float vignetteAmount)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float4 pixel;
    if (is16f) {
        pixel = vload_half4(y * destPitch + x, (const __global half*)destImg);
    } else {
        pixel = destImg[y * destPitch + x];
    }

    float nx = ((float)x / (float)width - 0.5f) * 2.0f;
    float ny = ((float)y / (float)height - 0.5f) * 2.0f;
    float r2 = nx * nx + ny * ny;
    float vig = 1.0f - min(1.0f, r2 * vignetteAmount * 0.5f);

    pixel.x = (1.0f - pixel.x) * vig;
    pixel.y = (1.0f - pixel.y) * vig;
    pixel.z = (1.0f - pixel.z) * vig;

    if (is16f) {
        vstore_half4_rtz(pixel, y * destPitch + x, (__global half*)destImg);
    } else {
        destImg[y * destPitch + x] = pixel;
    }
}

float hash21_cl(float2 p) {
    float3 p3 = (float3)(fract(p.x * 0.1031f, (float*)0), fract(p.y * 0.1030f, (float*)0), fract(p.x * 0.0973f, (float*)0));
    float d = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    p3 += d;
    return fract((p3.x + p3.y) * p3.z, (float*)0);
}

float noise2D_cl(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p, (float2*)0);
    float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float a = hash21_cl(i);
    float b = hash21_cl((float2)(i.x + 1.0f, i.y));
    float c = hash21_cl((float2)(i.x, i.y + 1.0f));
    float d = hash21_cl((float2)(i.x + 1.0f, i.y + 1.0f));

    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
}

float fbm3_cl(float2 p) {
    float v = 0.0f;
    float a = 0.5f;
    v += a * noise2D_cl(p); p.x *= 2.02f; p.y *= 2.03f; a *= 0.5f;
    v += a * noise2D_cl(p); p.x *= 2.01f; p.y *= 2.04f; a *= 0.5f;
    v += a * noise2D_cl(p);
    return v;
}

__kernel void kSmearCL(
    __global const float4* srcImg,
    __global const float4* historyIn,
    __global float4* historyOut,
    __global float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    float react,
    float chaos,
    float trails,
    float heat,
    float decay,
    float brightness,
    float inputGain,
    float speed,
    float complexity,
    float twist,
    float vignette,
    float dryWet,
    float driveLow,
    float driveMid,
    float driveHigh,
    float pulse,
    float phase,
    float dt)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float aspect = (float)width / (float)height;
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    float2 p = (float2)((u - 0.5f) * aspect * 2.0f, (v - 0.5f) * 2.0f);
    float r = sqrt(p.x * p.x + p.y * p.y);

    // 1. Curl noise flow field
    float flowScale = 1.5f + (complexity * 0.01f) * 2.5f;
    float2 noiseCoord = (float2)(p.x * flowScale + phase * 0.13f,
                                 p.y * flowScale - phase * 0.09f);
    float eps = 0.01f;
    float psi = fbm3_cl(noiseCoord);
    float dx_n = fbm3_cl((float2)(noiseCoord.x + eps, noiseCoord.y)) - psi;
    float dy_n = fbm3_cl((float2)(noiseCoord.x, noiseCoord.y + eps)) - psi;
    float2 vCurl = (float2)(dy_n / eps, -dx_n / eps);

    // 2. Image edge-steered flow
    int l_x = max(0, x - 1);
    int r_x = min(width - 1, x + 1);
    int t_y = max(0, y - 1);
    int b_y = min(height - 1, y + 1);
    float4 p_l = is16f ? vload_half4(y * destPitch + l_x, (const __global half*)srcImg) : srcImg[y * destPitch + l_x];
    float4 p_r = is16f ? vload_half4(y * destPitch + r_x, (const __global half*)srcImg) : srcImg[y * destPitch + r_x];
    float4 p_t = is16f ? vload_half4(t_y * destPitch + x, (const __global half*)srcImg) : srcImg[t_y * destPitch + x];
    float4 p_b = is16f ? vload_half4(b_y * destPitch + x, (const __global half*)srcImg) : srcImg[b_y * destPitch + x];
    float lum_l = 0.2126f * p_l.z + 0.7152f * p_l.y + 0.0722f * p_l.x;
    float lum_r = 0.2126f * p_r.z + 0.7152f * p_r.y + 0.0722f * p_r.x;
    float lum_t = 0.2126f * p_t.z + 0.7152f * p_t.y + 0.0722f * p_t.x;
    float lum_b = 0.2126f * p_b.z + 0.7152f * p_b.y + 0.0722f * p_b.x;
    float2 sGrad = (float2)((lum_r - lum_l) * 0.5f, (lum_b - lum_t) * 0.5f);
    float gradLen = sqrt(sGrad.x * sGrad.x + sGrad.y * sGrad.y) + 1e-5f;
    float2 tangent = (float2)(-sGrad.y / gradLen, sGrad.x / gradLen);
    float2 vEdge = (float2)(tangent.x * 0.08f + sGrad.x * 0.04f,
                            tangent.y * 0.08f + sGrad.y * 0.04f);

    float effReact = react * 0.01f;
    float effPulse = pulse * 0.01f * effReact;
    float effLow   = driveLow * 0.01f * effReact;
    float effMid   = driveMid * 0.01f * effReact;
    float effChaos = (chaos * 0.01f) * (1.0f + effMid * 1.5f);

    float zoomPulse = effPulse * 0.25f + effLow * 0.10f;
    float twistRate = (twist * 0.01f) * 2.0f;
    float2 radial = (float2)(p.x / max(r, 1e-4f), p.y / max(r, 1e-4f));
    float2 swirl  = (float2)(-radial.y, radial.x);

    float2 velocity = (float2)(
        vCurl.x * effChaos * 0.5f + vEdge.x + radial.x * zoomPulse + swirl.x * twistRate * (0.25f + r),
        vCurl.y * effChaos * 0.5f + vEdge.y + radial.y * zoomPulse + swirl.y * twistRate * (0.25f + r)
    );

    float fdt = dt > 0.0f ? dt : 0.01667f;
    float2 backP = (float2)(p.x - velocity.x * fdt * 2.0f,
                            p.y - velocity.y * fdt * 2.0f);

    float backU = clamp(backP.x / (aspect * 2.0f) + 0.5f, 0.0f, 1.0f);
    float backV = clamp(backP.y * 0.5f + 0.5f, 0.0f, 1.0f);

    int hx = clamp((int)(backU * (float)width), 0, width - 1);
    int hy = clamp((int)(backV * (float)height), 0, height - 1);

    float4 advectedHist = historyIn[hy * width + hx];

    float decayNorm = decay * 0.01f;
    float halfLife = 0.005f * pow(800.0f, decayNorm);
    float retention = exp(-0.693147f * fdt / max(halfLife, 1e-4f));
    float trailMix = trails * 0.01f;

    float4 fresh;
    if (is16f) {
        fresh = vload_half4(y * destPitch + x, (const __global half*)srcImg);
    } else {
        fresh = srcImg[y * destPitch + x];
    }

    float injectGain = (inputGain * 0.01f) * 1.5f + effPulse * 0.8f;
    float4 freshInjected = (float4)(fresh.x * injectGain, fresh.y * injectGain, fresh.z * injectGain, 1.0f);

    float4 feedback;
    feedback.x = max(freshInjected.x, advectedHist.x * retention * trailMix);
    feedback.y = max(freshInjected.y, advectedHist.y * retention * trailMix);
    feedback.z = max(freshInjected.z, advectedHist.z * retention * trailMix);
    feedback.w = 1.0f;

    // Liquid fire white core: highlight accumulation lifts channels towards white blowout
    float maxCh = max(feedback.x, max(feedback.y, feedback.z));
    if (maxCh > 0.65f) {
        float bloom = pow(maxCh - 0.65f, 1.1f) * (2.2f + (heat * 0.01f) * 4.0f);
        feedback.x += bloom;
        feedback.y += bloom;
        feedback.z += bloom;
    }

    historyOut[y * width + x] = feedback;

    float exposure = pow(2.0f, (brightness - 50.0f) * 0.04f);
    float4 tonemapped;
    tonemapped.x = 1.0f - exp(-feedback.x * exposure);
    tonemapped.y = 1.0f - exp(-feedback.y * exposure);
    tonemapped.z = 1.0f - exp(-feedback.z * exposure);
    tonemapped.w = 1.0f;

    if (vignette > 0.0f) {
        float vig = 1.0f - min(1.0f, (r * r) * (vignette * 0.01f));
        tonemapped.x *= vig;
        tonemapped.y *= vig;
        tonemapped.z *= vig;
    }

    float wet = dryWet * 0.01f;
    float4 outColor;
    outColor.x = fresh.x * (1.0f - wet) + tonemapped.x * wet;
    outColor.y = fresh.y * (1.0f - wet) + tonemapped.y * wet;
    outColor.z = fresh.z * (1.0f - wet) + tonemapped.z * wet;
    outColor.w = 1.0f;

    if (is16f) {
        vstore_half4_rtz(outColor, y * destPitch + x, (__global half*)destImg);
    } else {
        destImg[y * destPitch + x] = outColor;
    }
}
)===";
