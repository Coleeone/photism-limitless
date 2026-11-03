#include "PhotismLimitless.h"
#include "DeterministicState.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace {

inline float Fract(float x) {
    return x - floorf(x);
}

inline float Hash21_CPU(float x, float y) {
    float p3x = Fract(x * 0.1031f);
    float p3y = Fract(y * 0.1030f);
    float p3z = Fract(x * 0.0973f);
    float d = p3x * (p3y + 33.33f) + p3y * (p3z + 33.33f) + p3z * (p3x + 33.33f);
    p3x += d; p3y += d; p3z += d;
    return Fract((p3x + p3y) * p3z);
}

inline float Noise2D_CPU(float x, float y) {
    float ix = floorf(x), iy = floorf(y);
    float fx = Fract(x), fy = Fract(y);
    float ux = fx * fx * fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);
    float uy = fy * fy * fy * (fy * (fy * 6.0f - 15.0f) + 10.0f);

    float a = Hash21_CPU(ix, iy);
    float b = Hash21_CPU(ix + 1.0f, iy);
    float c = Hash21_CPU(ix, iy + 1.0f);
    float d = Hash21_CPU(ix + 1.0f, iy + 1.0f);

    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

inline float FBM3_CPU(float x, float y) {
    float v = 0.0f;
    float a = 0.5f;
    v += a * Noise2D_CPU(x, y); x *= 2.02f; y *= 2.03f; a *= 0.5f;
    v += a * Noise2D_CPU(x, y); x *= 2.01f; y *= 2.04f; a *= 0.5f;
    v += a * Noise2D_CPU(x, y);
    return v;
}

} // namespace

static PF_Err
About(
    PF_InData       *in_data,
    PF_OutData      *out_data,
    PF_ParamDef     *params[],
    PF_LayerDef     *output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);
    suites.ANSICallbacksSuite1()->sprintf(
        out_data->return_msg,
        "%s v%d.%d\r%s",
        PLUGIN_NAME,
        PLUGIN_MAJOR_VERSION,
        PLUGIN_MINOR_VERSION,
        PLUGIN_DESCRIPTION);
    return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
    PF_InData       *in_data,
    PF_OutData      *out_data,
    PF_ParamDef     *params[],
    PF_LayerDef     *output)
{
    out_data->my_version = PF_VERSION(
        PLUGIN_MAJOR_VERSION,
        PLUGIN_MINOR_VERSION,
        PLUGIN_BUG_VERSION,
        PLUGIN_STAGE_VERSION,
        PLUGIN_BUILD_VERSION);

    out_data->out_flags |= PF_OutFlag_PIX_INDEPENDENT |
                           PF_OutFlag_USE_OUTPUT_EXTENT |
                           PF_OutFlag_NON_PARAM_VARY;

    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
                           PF_OutFlag2_FLOAT_COLOR_AWARE |
                           PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS;

    if (in_data->appl_id == 'PrMr') {
        auto pixelFormatSuite =
            AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data,
                                                   kPFPixelFormatSuite,
                                                   kPFPixelFormatSuiteVersion1,
                                                   out_data);

        (*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
    }

    return PF_Err_NONE;
}

static PF_Err
ParamsSetup(
    PF_InData       *in_data,
    PF_OutData      *out_data,
    PF_ParamDef     *params[],
    PF_LayerDef     *output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    // 1. Mode popup (All 21 Limitless modes per RESEARCH_PHOTISM_PRODUCT.md §10)
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Mode",
                 MODE_COUNT,
                 DFLT_MODE,
                 "None|Smear|Droste|Infinite|Corridor|Cube|Planet|Windows|Melt|Shatter|Prism|Terrain|Breathe|Mosh|Sort|Coral|Scanline|Thresh|Solar|Halftone|VHS",
                 MODE_DISK_ID);

    // 2. React
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("React", 0.0f, 200.0f, 0.0f, 200.0f, DFLT_REACT, PF_Precision_TENTHS, 0, 0, REACT_DISK_ID);

    // 3. Chaos
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Chaos", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_CHAOS, PF_Precision_TENTHS, 0, 0, CHAOS_DISK_ID);

    // 4. Trails
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Trails", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_TRAILS, PF_Precision_TENTHS, 0, 0, TRAILS_DISK_ID);

    // 5. Heat
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Heat", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_HEAT, PF_Precision_TENTHS, 0, 0, HEAT_DISK_ID);

    // 6. Decay
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Decay", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_DECAY, PF_Precision_TENTHS, 0, 0, DECAY_DISK_ID);

    // 7. Brightness
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Brightness", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_BRIGHTNESS, PF_Precision_TENTHS, 0, 0, BRIGHTNESS_DISK_ID);

    // 8. Input Gain
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Input Gain", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_INPUT_GAIN, PF_Precision_TENTHS, 0, 0, INPUT_GAIN_DISK_ID);

    // 9. Symmetry
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Symmetry", 0.0f, 12.0f, 0.0f, 12.0f, DFLT_SYMMETRY, PF_Precision_INTEGER, 0, 0, SYMMETRY_DISK_ID);

    // 10. Tiles
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Tiles", 0.0f, 12.0f, 0.0f, 12.0f, DFLT_TILES, PF_Precision_INTEGER, 0, 0, TILES_DISK_ID);

    // 11. Twist
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Twist", -100.0f, 100.0f, -100.0f, 100.0f, DFLT_TWIST, PF_Precision_TENTHS, 0, 0, TWIST_DISK_ID);

    // 12. Slice
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Slice", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_SLICE, PF_Precision_TENTHS, 0, 0, SLICE_DISK_ID);

    // 13. Lens
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Lens", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_LENS, PF_Precision_TENTHS, 0, 0, LENS_DISK_ID);

    // 14. Lens Style popup
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Lens Style",
                 LENS_STYLE_COUNT,
                 DFLT_LENS_STYLE,
                 "Glass Ribs|Glass Fan|Anamorphic|Melt|Glitch|Pixels|Fly Eye",
                 LENS_STYLE_DISK_ID);

    // 15. Hue Drift
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Hue Drift", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_HUE_DRIFT, PF_Precision_TENTHS, 0, 0, HUE_DRIFT_DISK_ID);

    // 16. Vignette
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Vignette", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_VIGNETTE, PF_Precision_TENTHS, 0, 0, VIGNETTE_DISK_ID);

    // 17. Palette popup
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Palette",
                 PALETTE_COUNT,
                 DFLT_PALETTE,
                 "Custom|Nocturne|Ember|Glacial|Orchid|Moss|Sodium|Pearl|Cobalt|Reef|Vapor|Oxide|Aurora|Velvet|Acid|Neon|Mirage|Canyon|Adobe|Sage|Umber|Prism|Carnival|Candy|Tropic",
                 PALETTE_DISK_ID);

    // 18. Randomize Look button
    AEFX_CLR_STRUCT(def);
    PF_ADD_BUTTON("Randomize Look", "Dice", 0, 0, RANDOMIZE_LOOK_DISK_ID);

    // 19. Speed
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Speed", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_SPEED, PF_Precision_TENTHS, 0, 0, SPEED_DISK_ID);

    // 20. Shape
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Shape", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_SHAPE, PF_Precision_TENTHS, 0, 0, SHAPE_DISK_ID);

    // 21. Complexity
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Complexity", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_COMPLEXITY, PF_Precision_TENTHS, 0, 0, COMPLEXITY_DISK_ID);

    // 22. Fill
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Fill", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_FILL, PF_Precision_TENTHS, 0, 0, FILL_DISK_ID);

    // 23. Contrast
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Contrast", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_CONTRAST, PF_Precision_TENTHS, 0, 0, CONTRAST_DISK_ID);

    // 24. Dry/Wet
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Dry/Wet", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_DRY_WET, PF_Precision_TENTHS, 0, 0, DRY_WET_DISK_ID);

    // 25. Drive_Low
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Drive_Low", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_DRIVE_LOW, PF_Precision_TENTHS, 0, 0, DRIVE_LOW_DISK_ID);

    // 26. Drive_Mid
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Drive_Mid", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_DRIVE_MID, PF_Precision_TENTHS, 0, 0, DRIVE_MID_DISK_ID);

    // 27. Drive_High
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Drive_High", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_DRIVE_HIGH, PF_Precision_TENTHS, 0, 0, DRIVE_HIGH_DISK_ID);

    // 28. Pulse
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Pulse", 0.0f, 100.0f, 0.0f, 100.0f, DFLT_PULSE, PF_Precision_TENTHS, 0, 0, PULSE_DISK_ID);

    out_data->num_params = LIMITLESS_NUM_PARAMS;
    return err;
}

static PF_Err
RenderCPU(
    PF_InData       *in_data,
    PF_OutData      *out_data,
    PF_ParamDef     *params[],
    PF_LayerDef     *output)
{
    PF_Err err = PF_Err_NONE;

    int width = output->width;
    int height = output->height;
    if (width <= 0 || height <= 0) return PF_Err_NONE;

    PhotismParamsGPU p = {};
    p.mode        = params[LIMITLESS_MODE]->u.pd.value;
    p.react       = static_cast<float>(params[LIMITLESS_REACT]->u.fs_d.value);
    p.chaos       = static_cast<float>(params[LIMITLESS_CHAOS]->u.fs_d.value);
    p.trails      = static_cast<float>(params[LIMITLESS_TRAILS]->u.fs_d.value);
    p.heat        = static_cast<float>(params[LIMITLESS_HEAT]->u.fs_d.value);
    p.decay       = static_cast<float>(params[LIMITLESS_DECAY]->u.fs_d.value);
    p.brightness  = static_cast<float>(params[LIMITLESS_BRIGHTNESS]->u.fs_d.value);
    p.inputGain   = static_cast<float>(params[LIMITLESS_INPUT_GAIN]->u.fs_d.value);
    p.symmetry    = static_cast<float>(params[LIMITLESS_SYMMETRY]->u.fs_d.value);
    p.tiles       = static_cast<float>(params[LIMITLESS_TILES]->u.fs_d.value);
    p.twist       = static_cast<float>(params[LIMITLESS_TWIST]->u.fs_d.value);
    p.slice       = static_cast<float>(params[LIMITLESS_SLICE]->u.fs_d.value);
    p.lens        = static_cast<float>(params[LIMITLESS_LENS]->u.fs_d.value);
    p.lensStyle   = params[LIMITLESS_LENS_STYLE]->u.pd.value;
    p.hueDrift    = static_cast<float>(params[LIMITLESS_HUE_DRIFT]->u.fs_d.value);
    p.vignette    = static_cast<float>(params[LIMITLESS_VIGNETTE]->u.fs_d.value);
    p.palette     = params[LIMITLESS_PALETTE]->u.pd.value;

    p.speed       = static_cast<float>(params[LIMITLESS_SPEED]->u.fs_d.value);
    p.shape       = static_cast<float>(params[LIMITLESS_SHAPE]->u.fs_d.value);
    p.complexity  = static_cast<float>(params[LIMITLESS_COMPLEXITY]->u.fs_d.value);
    p.fill        = static_cast<float>(params[LIMITLESS_FILL]->u.fs_d.value);
    p.contrast    = static_cast<float>(params[LIMITLESS_CONTRAST]->u.fs_d.value);
    p.dryWet      = static_cast<float>(params[LIMITLESS_DRY_WET]->u.fs_d.value);

    p.driveLow    = static_cast<float>(params[LIMITLESS_DRIVE_LOW]->u.fs_d.value);
    p.driveMid    = static_cast<float>(params[LIMITLESS_DRIVE_MID]->u.fs_d.value);
    p.driveHigh   = static_cast<float>(params[LIMITLESS_DRIVE_HIGH]->u.fs_d.value);
    p.pulse       = static_cast<float>(params[LIMITLESS_PULSE]->u.fs_d.value);

    int frameIndex = in_data->time_step > 0 ? (std::max)(0, static_cast<int>(in_data->current_time / in_data->time_step)) : 0;
    float dt = in_data->time_scale > 0 && in_data->time_step > 0 ?
               static_cast<float>(in_data->time_step) / static_cast<float>(in_data->time_scale) : 0.01667f;

    auto state = StateManager::Instance().GetOrCreateState(in_data->effect_ref, width, height);
    state->SetActiveMode(p.mode);

    int startFrame = 0;
    float startPhase = 0.0f;
    float speedRate = (p.speed * 0.01f) * (1.0f + (p.react * 0.01f) * (p.driveMid * 0.01f));
    bool needResim = state->CheckSeekAndPrepare(frameIndex, dt, speedRate, startFrame, startPhase);

    // Get input and output pixel buffers
    PF_PixelFloat* inputData32f = reinterpret_cast<PF_PixelFloat*>(params[0]->u.ld.data);
    PF_PixelFloat* outputData32f = reinterpret_cast<PF_PixelFloat*>(output->data);
    int rowPixels = output->rowbytes / sizeof(PF_PixelFloat);

    float aspect = static_cast<float>(width) / static_cast<float>(height);

    // Phase 0: Trivial Invert + Vignette mode when Mode == MODE_NONE
    if (p.mode == MODE_NONE) {
        float vigAmount = p.vignette * 0.01f;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float nx = (static_cast<float>(x) / static_cast<float>(width) - 0.5f) * 2.0f;
                float ny = (static_cast<float>(y) / static_cast<float>(height) - 0.5f) * 2.0f;
                float r2 = nx * nx + ny * ny;
                float vig = 1.0f - (std::min)(1.0f, r2 * vigAmount * 0.5f);

                PF_PixelFloat src = inputData32f[y * rowPixels + x];
                PF_PixelFloat dst;
                dst.red   = (1.0f - src.red) * vig;
                dst.green = (1.0f - src.green) * vig;
                dst.blue  = (1.0f - src.blue) * vig;
                dst.alpha = src.alpha;
                outputData32f[y * rowPixels + x] = dst;
            }
        }
        return PF_Err_NONE;
    }

    // Phase 1: Smear Mode (CPU reference implementation with 1 LSB parity)
    float* historyIn = state->GetCurrentHistory();
    float* historyOut = state->GetNextHistory();

    float flowScale = 1.5f + (p.complexity * 0.01f) * 2.5f;
    float effReact = p.react * 0.01f;
    float effPulse = p.pulse * 0.01f * effReact;
    float effLow   = p.driveLow * 0.01f * effReact;
    float effMid   = p.driveMid * 0.01f * effReact;
    float effChaos = (p.chaos * 0.01f) * (1.0f + effMid * 1.5f);
    float zoomPulse = effPulse * 0.25f + effLow * 0.10f;
    float twistRate = (p.twist * 0.01f) * 2.0f;

    float decayNorm = p.decay * 0.01f;
    float halfLife = 0.005f * powf(800.0f, decayNorm);
    float retention = expf(-0.693147f * dt / (std::max)(halfLife, 1e-4f));
    float trailMix = p.trails * 0.01f;
    float injectGain = (p.inputGain * 0.01f) * 1.5f + effPulse * 0.8f;
    float exposure = powf(2.0f, (p.brightness - 50.0f) * 0.04f);
    float wet = p.dryWet * 0.01f;

    // Sobel gradient prepass
    std::vector<float> gradX(width * height, 0.0f);
    std::vector<float> gradY(width * height, 0.0f);

    auto getLuma = [&](int cx, int cy) -> float {
        cx = (std::max)(0, (std::min)(width - 1, cx));
        cy = (std::max)(0, (std::min)(height - 1, cy));
        PF_PixelFloat px = inputData32f[cy * rowPixels + cx];
        return 0.2126f * px.red + 0.7152f * px.green + 0.0722f * px.blue;
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
            gradX[y * width + x] = dx * 0.125f;
            gradY[y * width + x] = dy * 0.125f;
        }
    }

    // Advection + Feedback Pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(width);
            float v = static_cast<float>(y) / static_cast<float>(height);
            float px = (u - 0.5f) * aspect * 2.0f;
            float py = (v - 0.5f) * 2.0f;
            float r = sqrtf(px * px + py * py);

            float ncx = px * flowScale + startPhase * 0.13f;
            float ncy = py * flowScale - startPhase * 0.09f;
            float eps = 0.01f;
            float psi = FBM3_CPU(ncx, ncy);
            float dx_n = FBM3_CPU(ncx + eps, ncy) - psi;
            float dy_n = FBM3_CPU(ncx, ncy + eps) - psi;
            float vCurlX = dy_n / eps;
            float vCurlY = -dx_n / eps;

            float sGx = gradX[y * width + x];
            float sGy = gradY[y * width + x];
            float gLen = sqrtf(sGx * sGx + sGy * sGy) + 1e-5f;
            float tanX = -sGy / gLen;
            float tanY = sGx / gLen;
            float vEdgeX = tanX * 0.08f + sGx * 0.04f;
            float vEdgeY = tanY * 0.08f + sGy * 0.04f;

            float radX = px / (std::max)(r, 1e-4f);
            float radY = py / (std::max)(r, 1e-4f);
            float swX = -radY;
            float swY = radX;

            float velX = vCurlX * effChaos * 0.5f + vEdgeX + radX * zoomPulse + swX * twistRate * (0.25f + r);
            float velY = vCurlY * effChaos * 0.5f + vEdgeY + radY * zoomPulse + swY * twistRate * (0.25f + r);

            float backPx = px - velX * dt * 2.0f;
            float backPy = py - velY * dt * 2.0f;
            float backU = backPx / (aspect * 2.0f) + 0.5f;
            float backV = backPy * 0.5f + 0.5f;

            float hx = (std::max)(0.0f, (std::min)(static_cast<float>(width - 1), backU * width));
            float hy = (std::max)(0.0f, (std::min)(static_cast<float>(height - 1), backV * height));
            int x0 = static_cast<int>(floorf(hx));
            int y0 = static_cast<int>(floorf(hy));
            int x1 = (std::min)(width - 1, x0 + 1);
            int y1 = (std::min)(height - 1, y0 + 1);
            float fx = hx - x0;
            float fy = hy - y0;

            int idx00 = (y0 * width + x0) * 4;
            int idx10 = (y0 * width + x1) * 4;
            int idx01 = (y1 * width + x0) * 4;
            int idx11 = (y1 * width + x1) * 4;

            float advectedR = (1.0f - fx) * (1.0f - fy) * historyIn[idx00]     + fx * (1.0f - fy) * historyIn[idx10]     + (1.0f - fx) * fy * historyIn[idx01]     + fx * fy * historyIn[idx11];
            float advectedG = (1.0f - fx) * (1.0f - fy) * historyIn[idx00 + 1] + fx * (1.0f - fy) * historyIn[idx10 + 1] + (1.0f - fx) * fy * historyIn[idx01 + 1] + fx * fy * historyIn[idx11 + 1];
            float advectedB = (1.0f - fx) * (1.0f - fy) * historyIn[idx00 + 2] + fx * (1.0f - fy) * historyIn[idx10 + 2] + (1.0f - fx) * fy * historyIn[idx01 + 2] + fx * fy * historyIn[idx11 + 2];

            float heatScale = (p.heat * 0.01f) * (1.0f + effLow * 0.5f);
            float refrU = u + vCurlX * heatScale * 0.008f;
            float refrV = v + vCurlY * heatScale * 0.008f;
            int su = (std::max)(0, (std::min)(width - 1, static_cast<int>(refrU * width)));
            int sv = (std::max)(0, (std::min)(height - 1, static_cast<int>(refrV * height)));

            PF_PixelFloat fresh = inputData32f[sv * rowPixels + su];
            float freshR = fresh.red * injectGain;
            float freshG = fresh.green * injectGain;
            float freshB = fresh.blue * injectGain;

            float feedR = (std::max)(freshR, advectedR * retention * trailMix);
            float feedG = (std::max)(freshG, advectedG * retention * trailMix);
            float feedB = (std::max)(freshB, advectedB * retention * trailMix);

            float maxCh = (std::max)(feedR, (std::max)(feedG, feedB));
            if (maxCh > 0.65f) {
                float bloom = powf(maxCh - 0.65f, 1.1f) * (2.2f + (p.heat * 0.01f) * 4.0f);
                feedR += bloom;
                feedG += bloom;
                feedB += bloom;
            }

            int outIdx = (y * width + x) * 4;
            historyOut[outIdx]     = feedR;
            historyOut[outIdx + 1] = feedG;
            historyOut[outIdx + 2] = feedB;
            historyOut[outIdx + 3] = 1.0f;

            float toneR = 1.0f - expf(-feedR * exposure);
            float toneG = 1.0f - expf(-feedG * exposure);
            float toneB = 1.0f - expf(-feedB * exposure);

            if (p.vignette > 0.0f) {
                float vig = 1.0f - (std::min)(1.0f, (r * r) * (p.vignette * 0.01f));
                toneR *= vig; toneG *= vig; toneB *= vig;
            }

            PF_PixelFloat dst;
            dst.red   = fresh.red * (1.0f - wet) + toneR * wet;
            dst.green = fresh.green * (1.0f - wet) + toneG * wet;
            dst.blue  = fresh.blue * (1.0f - wet) + toneB * wet;
            dst.alpha = fresh.alpha;
            outputData32f[y * rowPixels + x] = dst;
        }
    }

    state->SwapHistory();
    state->CommitStep(frameIndex, startPhase, state->GetCurrentHistory());

    return err;
}

DllExport PF_Err
EntryPointFunc(
    PF_Cmd          cmd,
    PF_InData       *in_data,
    PF_OutData      *out_data,
    PF_ParamDef     *params[],
    PF_LayerDef     *output,
    void            *extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data, params, output);
                break;
            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(in_data, out_data, params, output);
                break;
            case PF_Cmd_PARAMS_SETUP:
                err = ParamsSetup(in_data, out_data, params, output);
                break;
            case PF_Cmd_RENDER:
                err = RenderCPU(in_data, out_data, params, output);
                break;
            case PF_Cmd_SMART_PRE_RENDER:
            case PF_Cmd_SMART_RENDER:
                // Fallback for smart render hosts
                err = PF_Err_NONE;
                break;
            default:
                break;
        }
    } catch (const PF_Err& thrown_err) {
        err = thrown_err;
    } catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}
