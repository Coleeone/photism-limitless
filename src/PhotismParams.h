#pragma once

// 21 Limitless Modes (Full Specification per RESEARCH_PHOTISM_PRODUCT.md §10)
enum PhotismMode {
    MODE_NONE = 1,
    MODE_SMEAR = 2,
    MODE_DROSTE = 3,
    MODE_INFINITE = 4,
    MODE_CORRIDOR = 5,
    MODE_CUBE = 6,
    MODE_PLANET = 7,
    MODE_WINDOWS = 8,
    MODE_MELT = 9,
    MODE_SHATTER = 10,
    MODE_PRISM = 11,
    MODE_TERRAIN = 12,
    MODE_BREATHE = 13,
    MODE_MOSH = 14,
    MODE_SORT = 15,
    MODE_CORAL = 16,
    MODE_SCANLINE = 17,
    MODE_THRESH = 18,
    MODE_SOLAR = 19,
    MODE_HALFTONE = 20,
    MODE_VHS = 21,
    MODE_COUNT = 21
};

// Lens Styles
enum PhotismLensStyle {
    LENS_GLASS_RIBS = 1,
    LENS_GLASS_FAN = 2,
    LENS_ANAMORPHIC = 3,
    LENS_MELT = 4,
    LENS_GLITCH = 5,
    LENS_PIXELS = 6,
    LENS_FLY_EYE = 7,
    LENS_STYLE_COUNT = 7
};

// Palettes
enum PhotismPalette {
    PALETTE_CUSTOM = 1,
    PALETTE_NOCTURNE = 2,
    PALETTE_EMBER = 3,
    PALETTE_GLACIAL = 4,
    PALETTE_ORCHID = 5,
    PALETTE_MOSS = 6,
    PALETTE_SODIUM = 7,
    PALETTE_PEARL = 8,
    PALETTE_COBALT = 9,
    PALETTE_REEF = 10,
    PALETTE_VAPOR = 11,
    PALETTE_OXIDE = 12,
    PALETTE_AURORA = 13,
    PALETTE_VELVET = 14,
    PALETTE_ACID = 15,
    PALETTE_NEON = 16,
    PALETTE_MIRAGE = 17,
    PALETTE_CANYON = 18,
    PALETTE_ADOBE = 19,
    PALETTE_SAGE = 20,
    PALETTE_UMBER = 21,
    PALETTE_PRISM = 22,
    PALETTE_CARNIVAL = 23,
    PALETTE_CANDY = 24,
    PALETTE_TROPIC = 25,
    PALETTE_COUNT = 25
};

// Default Values for Smear Preset (EFFECT_ANALYSIS.md & REVERSE_ENGINEERING_PLAN.md)
#define DFLT_MODE           MODE_SMEAR
#define DFLT_REACT          36.0f
#define DFLT_CHAOS          23.0f
#define DFLT_TRAILS         82.0f
#define DFLT_HEAT           55.0f
#define DFLT_DECAY          10.0f
#define DFLT_BRIGHTNESS     47.0f
#define DFLT_INPUT_GAIN     78.0f
#define DFLT_SYMMETRY       0.0f
#define DFLT_TILES          0.0f
#define DFLT_TWIST          0.0f
#define DFLT_SLICE          0.0f
#define DFLT_LENS           0.0f
#define DFLT_LENS_STYLE     LENS_GLASS_RIBS
#define DFLT_HUE_DRIFT      0.0f
#define DFLT_VIGNETTE       0.0f
#define DFLT_PALETTE        PALETTE_CUSTOM

#define DFLT_SPEED          54.0f
#define DFLT_SHAPE          0.0f
#define DFLT_COMPLEXITY     100.0f
#define DFLT_FILL           0.0f
#define DFLT_CONTRAST       98.0f
#define DFLT_DRY_WET        100.0f

#define DFLT_DRIVE_LOW      0.0f
#define DFLT_DRIVE_MID      0.0f
#define DFLT_DRIVE_HIGH     0.0f
#define DFLT_PULSE          0.0f

// Uniforms / Parameters packed for GPU Kernel Dispatch
struct PhotismParamsGPU {
    int   mode;
    float react;
    float chaos;
    float trails;
    float heat;
    float decay;
    float brightness;
    float inputGain;
    float symmetry;
    float tiles;
    float twist;
    float slice;
    float lens;
    int   lensStyle;
    float hueDrift;
    float vignette;
    int   palette;

    float speed;
    float shape;
    float complexity;
    float fill;
    float contrast;
    float dryWet;

    float driveLow;
    float driveMid;
    float driveHigh;
    float pulse;

    float phase;
    float dt;
    int   frameIndex;
    int   width;
    int   height;
    int   destPitch;
    int   srcPitch;
    int   is16f;
};
