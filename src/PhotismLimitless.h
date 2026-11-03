#pragma once

#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"

#ifdef AE_OS_WIN
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#include "PrSDKTypes.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "PrSDKAESupport.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "Smart_Utils.h"
#include "AEFX_SuiteHelper.h"
#include "AEGP_SuiteHandler.h"

#include "PhotismParams.h"

#define PLUGIN_NAME "Photism Limitless"
#define PLUGIN_MATCH_NAME "Photism Limitless"
#define PLUGIN_CATEGORY "Photism"
#define PLUGIN_DESCRIPTION "Cloned Limitless audio-reactive visualizer engine for Premiere Pro."

#define PLUGIN_MAJOR_VERSION 1
#define PLUGIN_MINOR_VERSION 0
#define PLUGIN_BUG_VERSION   0
#define PLUGIN_STAGE_VERSION PF_Stage_DEVELOP
#define PLUGIN_BUILD_VERSION 1

// Parameter Enum Index
enum {
    LIMITLESS_INPUT = 0,

    // Global tune group
    LIMITLESS_MODE,
    LIMITLESS_REACT,
    LIMITLESS_CHAOS,
    LIMITLESS_TRAILS,
    LIMITLESS_HEAT,
    LIMITLESS_DECAY,
    LIMITLESS_BRIGHTNESS,
    LIMITLESS_INPUT_GAIN,
    LIMITLESS_SYMMETRY,
    LIMITLESS_TILES,
    LIMITLESS_TWIST,
    LIMITLESS_SLICE,
    LIMITLESS_LENS,
    LIMITLESS_LENS_STYLE,
    LIMITLESS_HUE_DRIFT,
    LIMITLESS_VIGNETTE,
    LIMITLESS_PALETTE,
    LIMITLESS_RANDOMIZE_LOOK,

    // Limitless scene group
    LIMITLESS_SPEED,
    LIMITLESS_SHAPE,
    LIMITLESS_COMPLEXITY,
    LIMITLESS_FILL,
    LIMITLESS_CONTRAST,
    LIMITLESS_DRY_WET,

    // Baked drive group
    LIMITLESS_DRIVE_LOW,
    LIMITLESS_DRIVE_MID,
    LIMITLESS_DRIVE_HIGH,
    LIMITLESS_PULSE,

    LIMITLESS_NUM_PARAMS
};

// Disk IDs
enum {
    MODE_DISK_ID = 1,
    REACT_DISK_ID,
    CHAOS_DISK_ID,
    TRAILS_DISK_ID,
    HEAT_DISK_ID,
    DECAY_DISK_ID,
    BRIGHTNESS_DISK_ID,
    INPUT_GAIN_DISK_ID,
    SYMMETRY_DISK_ID,
    TILES_DISK_ID,
    TWIST_DISK_ID,
    SLICE_DISK_ID,
    LENS_DISK_ID,
    LENS_STYLE_DISK_ID,
    HUE_DRIFT_DISK_ID,
    VIGNETTE_DISK_ID,
    PALETTE_DISK_ID,
    RANDOMIZE_LOOK_DISK_ID,

    SPEED_DISK_ID,
    SHAPE_DISK_ID,
    COMPLEXITY_DISK_ID,
    FILL_DISK_ID,
    CONTRAST_DISK_ID,
    DRY_WET_DISK_ID,

    DRIVE_LOW_DISK_ID,
    DRIVE_MID_DISK_ID,
    DRIVE_HIGH_DISK_ID,
    PULSE_DISK_ID
};

#ifdef __cplusplus
extern "C" {
#endif

DllExport PF_Err
EntryPointFunc(
    PF_Cmd          cmd,
    PF_InData       *in_data,
    PF_OutData      *out_data,
    PF_ParamDef     *params[],
    PF_LayerDef     *output,
    void            *extra);

#ifdef __cplusplus
}
#endif
