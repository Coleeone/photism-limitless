#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "AE_General.r"

resource 'PiPL' (16000) {
    {
        Kind {
            AEEffect
        },
        Name {
            "Photism Limitless"
        },
        Category {
            "Photism"
        },
#ifdef AE_OS_WIN
        CodeWin64X86 {"EntryPointFunc"},
#else
        CodeMacARM64 {"EntryPointFunc"},
        CodeMacIntel64 {"EntryPointFunc"},
#endif
        AE_PiPL_Version {
            2,
            0
        },
        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },
        AE_Effect_Version {
            524289
        },
        AE_Effect_Info_Flags {
            0
        },
        AE_Effect_Global_OutFlags {
            0x00000440
        },
        AE_Effect_Global_OutFlags_2 {
            0x00001040
        },
        AE_Effect_Match_Name {
            "Photism Limitless"
        },
        AE_Reserved_Info {
            8
        }
    }
};
