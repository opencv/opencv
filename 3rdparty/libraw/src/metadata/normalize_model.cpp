/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *

 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/dcraw_defs.h"
#include "../../internal/libraw_cameraids.h"

void LibRaw::GetNormalizedModel()
{

  int i, j;
  char *ps;
  int try_xml = 0;

  static const struct
  {
    unsigned long long id;
    char t_model[20];
  } unique[] =
// clang-format off
    {
      { CanonID_EOS_M50,           "EOS M50"}, // Kiss M
      { CanonID_EOS_M6_Mark_II,    "EOS M6 Mark II"},
      { CanonID_EOS_M200,          "EOS M200"},
      { CanonID_EOS_D30,           "EOS D30"},
      { CanonID_EOS_D60,           "EOS D60"},
      { CanonID_EOS_M3,            "EOS M3"},
      { CanonID_EOS_M10,           "EOS M10"},
      { CanonID_EOS_M5,            "EOS M5"},
      { CanonID_EOS_M100,          "EOS M100"},
      { CanonID_EOS_M6,            "EOS M6"},
      { CanonID_EOS_1D,            "EOS-1D"},
      { CanonID_EOS_1Ds,           "EOS-1Ds"},
      { CanonID_EOS_10D,           "EOS 10D"},
      { CanonID_EOS_1D_Mark_III,   "EOS-1D Mark III"},
      { CanonID_EOS_300D,          "EOS 300D"}, // Digital Rebel / Kiss Digital
      { CanonID_EOS_1D_Mark_II,    "EOS-1D Mark II"},
      { CanonID_EOS_20D,           "EOS 20D"},
      { CanonID_EOS_450D,          "EOS 450D"}, // Digital Rebel XSi / Kiss X2
      { CanonID_EOS_1Ds_Mark_II,   "EOS-1Ds Mark II"},
      { CanonID_EOS_350D,          "EOS 350D"}, // Digital Rebel XT / Kiss Digital N
      { CanonID_EOS_40D,           "EOS 40D"},
      { CanonID_EOS_5D,            "EOS 5D"},
      { CanonID_EOS_1Ds_Mark_III,  "EOS-1Ds Mark III"},
      { CanonID_EOS_5D_Mark_II,    "EOS 5D Mark II"},
      { CanonID_EOS_1D_Mark_II_N,  "EOS-1D Mark II N"},
      { CanonID_EOS_30D,           "EOS 30D"},
      { CanonID_EOS_400D,          "EOS 400D"}, // Digital Rebel XTi / Kiss Digital X
      { CanonID_EOS_7D,            "EOS 7D"},
      { CanonID_EOS_500D,          "EOS 500D"},   // Rebel T1i / Kiss X3
      { CanonID_EOS_1000D,         "EOS 1000D"}, // Digital Rebel XS / Kiss F
      { CanonID_EOS_50D,           "EOS 50D"},
      { CanonID_EOS_1D_X,          "EOS-1D X"},
      { CanonID_EOS_550D,          "EOS 550D"}, // Rebel T2i / Kiss X4
      { CanonID_EOS_1D_Mark_IV,    "EOS-1D Mark IV"},
      { CanonID_EOS_5D_Mark_III,   "EOS 5D Mark III"},
      { CanonID_EOS_600D,          "EOS 600D"}, // Rebel T3i / Kiss X5
      { CanonID_EOS_60D,           "EOS 60D"},
      { CanonID_EOS_1100D,         "EOS 1100D"}, // Rebel T3 / Kiss X50
      { CanonID_EOS_7D_Mark_II,    "EOS 7D Mark II"},
      { CanonID_EOS_650D,          "EOS 650D"}, // Rebel T4i / Kiss X6i
      { CanonID_EOS_6D,            "EOS 6D"},
      { CanonID_EOS_1D_C,          "EOS-1D C"},
      { CanonID_EOS_70D,           "EOS 70D"},
      { CanonID_EOS_700D,          "EOS 700D"},   // Rebel T5i / Kiss X7i
      { CanonID_EOS_1200D,         "EOS 1200D"}, // Rebel T5 / Kiss X70 / Hi
      { CanonID_EOS_1D_X_Mark_II,  "EOS-1D X Mark II"},
      { CanonID_EOS_M,             "EOS M"},
      { CanonID_EOS_100D,          "EOS 100D"}, // Rebel SL1 / Kiss X7
      { CanonID_EOS_760D,          "EOS 760D"}, // Rebel T6s / 8000D
      { CanonID_EOS_5D_Mark_IV,    "EOS 5D Mark IV"},
      { CanonID_EOS_80D,           "EOS 80D"},
      { CanonID_EOS_M2,            "EOS M2"},
      { CanonID_EOS_5DS,           "EOS 5DS"},
      { CanonID_EOS_750D,          "EOS 750D"}, // Rebel T6i / Kiss X8i
      { CanonID_EOS_5DS_R,         "EOS 5DS R"},
      { CanonID_EOS_1300D,         "EOS 1300D"}, // Rebel T6 / Kiss X80
      { CanonID_EOS_800D,          "EOS 800D"},   // Rebel T7i / Kiss X9i
      { CanonID_EOS_6D_Mark_II,    "EOS 6D Mark II"},
      { CanonID_EOS_77D,           "EOS 77D"},     // 9000D
      { CanonID_EOS_200D,          "EOS 200D"},   // Rebel SL2 / Kiss X9
      { CanonID_EOS_R5,            "EOS R5"},
      { CanonID_EOS_3000D,         "EOS 3000D"}, // Rebel T100 / 4000D
      { CanonID_EOS_R,             "EOS R"},
      { CanonID_EOS_1D_X_Mark_III, "EOS-1D X Mark III"},
      { CanonID_EOS_1500D,         "EOS 1500D"}, // Rebel T7 / 2000D / Kiss X90
      { CanonID_EOS_RP,            "EOS RP"},
      { CanonID_EOS_850D,          "EOS 850D"},  // EOS Rebel T8i / X10i
      { CanonID_EOS_250D,          "EOS 250D"}, // Rebel SL3 / 200D II / Kiss X10
      { CanonID_EOS_90D,           "EOS 90D"},
      { CanonID_EOS_R3,            "EOS R3"},
      { CanonID_EOS_R6,            "EOS R6"},
      { CanonID_EOS_R7,            "EOS R7"},
      { CanonID_EOS_R10,           "EOS R10"},
      { CanonID_EOS_M50_Mark_II,   "EOS M50 Mark II"}, // M50m2, Kiss M2
    },
#if 0
    olyque[] = {
      { OlyID_E_20,            "E-20"},
      { OlyID_E_20,            "E-20,E-20N,E-20P"},
      { OlyID_E_1,             "E-1"},
      { OlyID_E_300,           "E-300"},
      { OlyID_SP_550UZ,        "SP-550UZ"},
      { OlyID_SP_550UZ,        "SP550UZ"},
      { OlyID_SP_510UZ,        "SP-510UZ"},
      { OlyID_SP_510UZ,        "SP510UZ"},
      { OlyID_SP_560UZ,        "SP-560UZ"},
      { OlyID_SP_560UZ,        "SP560UZ"},
      { OlyID_SP_570UZ,        "SP-570UZ"},
      { OlyID_SP_570UZ,        "SP570UZ"},
      { OlyID_SP_565UZ,        "SP-565UZ"},
      { OlyID_SP_565UZ,        "SP565UZ"},
      { OlyID_XZ_1,            "XZ-1"},
      { OlyID_XZ_2,            "XZ-2"},
      { OlyID_XZ_10,           "XZ-10"},
      { OlyID_STYLUS_1,        "Stylus 1"},
      { OlyID_STYLUS_1,        "STYLUS1"},
      { OlyID_STYLUS_1,        "STYLUS1,1s"},
      { OlyID_SH_2,            "SH-2"},
      { OlyID_TG_4,            "TG-4"},
      { OlyID_TG_5,            "TG-5"},
      { OlyID_TG_6,            "TG-6"},
      { OlyID_E_10,            "E-10"},
      { OlyID_AIR_A01,         "AIR A01"},
      { OlyID_AIR_A01,         "AIR-A01"},
      { OlyID_E_330,           "E-330"},
      { OlyID_E_500,           "E-500"},
      { OlyID_E_400,           "E-400"},
      { OlyID_E_510,           "E-510"},
      { OlyID_E_3,             "E-3"},
      { OlyID_E_410,           "E-410"},
      { OlyID_E_420,           "E-420"},
      { OlyID_E_30,            "E-30"},
      { OlyID_E_520,           "E-520"},
      { OlyID_E_P1,            "E-P1"},
      { OlyID_E_620,           "E-620"},
      { OlyID_E_P2,            "E-P2"},
      { OlyID_E_PL1,           "E-PL1"},
      { OlyID_E_450,           "E-450"},
      { OlyID_E_600,           "E-600"},
      { OlyID_E_P3,            "E-P3"},
      { OlyID_E_5,             "E-5"},
      { OlyID_E_PL2,           "E-PL2"},
      { OlyID_E_M5,            "E-M5"},
      { OlyID_E_PL3,           "E-PL3"},
      { OlyID_E_PM1,           "E-PM1"},
      { OlyID_E_PL1s,          "E-PL1s"},
      { OlyID_E_PL5,           "E-PL5"},
      { OlyID_E_PM2,           "E-PM2"},
      { OlyID_E_P5,            "E-P5"},
      { OlyID_E_PL6,           "E-PL6"},
      { OlyID_E_PL7,           "E-PL7"},
      { OlyID_E_M1,            "E-M1"},
      { OlyID_E_M10,           "E-M10"},
      { OlyID_E_M5_Mark_II,    "E-M5 Mark II"},
      { OlyID_E_M5_Mark_II,    "E-M5MarkII"},
      { OlyID_E_M5_Mark_II,    "E-M5_M2"},
      { OlyID_E_M10_Mark_II,   "E-M10 Mark II"}, // Clauss piX 5oo
      { OlyID_E_M10_Mark_II,   "E-M10MarkII"},
      { OlyID_E_M10_Mark_II,   "E-M10_M2"},
      { OlyID_PEN_F,           "PEN-F"},
      { OlyID_E_PL8,           "E-PL8"},
      { OlyID_E_M1_Mark_II,    "E-M1 Mark II"},
      { OlyID_E_M1_Mark_II,    "E-M1MarkII"},
      { OlyID_E_M1_Mark_II,    "E-M1_M2"},
      { OlyID_E_M10_Mark_III,  "E-M10 Mark III"},
      { OlyID_E_M10_Mark_III,  "E-M10_M3"},
      { OlyID_E_PL9,           "E-PL9"},
      { OlyID_E_M1X,           "E-M1X"},
      { OlyID_E_PL10,          "E-PL10"},
      { OlyID_E_M10_Mark_IV,   "E-M10 Mark IV"},
      { OlyID_E_M10_Mark_IV,   "E-M10MarkIV"},
      { OlyID_E_M10_Mark_IV,   "E-M10_M4"},
      { OlyID_E_M5_Mark_III,   "E-M5 Mark III"},
      { OlyID_E_M5_Mark_III,   "E-M5MarkIII"},
      { OlyID_E_M5_Mark_III,   "E-M5_M3"},
      { OlyID_E_M1_Mark_III,   "E-M1 Mark III"},
      { OlyID_E_M1_Mark_III,   "E-M1MarkIII"},
      { OlyID_E_M1_Mark_III,   "E-M1_M3"},
      { OlyID_E_P7             "E-P7"},
      { OlyID_C_3030Z,         "C-3030Z"},
      { OlyID_C_3030Z,         "C3030Z"},
      { OlyID_C_5050Z,         "C-5050Z"},
      { OlyID_C_5050Z,         "C5050Z"},
      { OlyID_C_350Z,          "C-350Z"},
      { OlyID_C_350Z,          "X200,D560Z,C350Z"},
      { OlyID_C_740UZ,         "C-740UZ"},
      { OlyID_C_740UZ,         "C740UZ"},
      { OlyID_C_5060WZ,        "C-5060WZ"},
      { OlyID_C_5060WZ,        "C5060WZ"},
      { OlyID_C_8080WZ,        "C-8080WZ"},
      { OlyID_C_8080WZ,        "C8080WZ"},
      { OlyID_C_770UZ,         "C-770UZ"},
      { OlyID_C_770UZ,         "C770UZ"},
      { OlyID_C_7070WZ,        "C-7070WZ"},
      { OlyID_C_7070WZ,        "C7070WZ"},
      { OlyID_C_7000Z,         "C-7000Z"},
      { OlyID_C_7000Z,         "C70Z,C7000Z"},
      { OlyID_SP_500UZ,        "SP-500UZ"},
      { OlyID_SP_500UZ,        "SP500UZ"},
      { OlyID_SP_310,          "SP-310"},
      { OlyID_SP_310,          "SP310"},
      { OlyID_SP_350,          "SP-350"},
      { OlyID_SP_350,          "SP350"},
      { OlyID_SP_320,          "SP-320"},
      { OlyID_SP_320,          "SP320"},
    },

    penique[] = {
      { PentaxID_Optio_S,      "Optio S"},
      { PentaxID_Optio_S_V101, "Optio S V1.01"},
      { PentaxID_staristD,     "*istD"},
      { PentaxID_staristD,     "*ist D"},
      { PentaxID_Optio_33WR,   "Optio 33WR"},
      { PentaxID_Optio_S4,     "Optio S4"},
      { PentaxID_Optio_750Z,   "Optio 750Z"},
      { PentaxID_staristDS,    "*istDS"},
      { PentaxID_staristDS,    "*ist DS"},
      { PentaxID_staristDL,    "*istDL"},
      { PentaxID_staristDL,    "*ist DL"},
      { PentaxID_staristDS2,   "*istDS2"},
      { PentaxID_staristDS2,   "*ist DS2"},
      { PentaxID_GX_1S,        "GX-1S"},        // Samsung
      { PentaxID_staristDL2,   "*istDL2"},
      { PentaxID_staristDL2,   "*ist DL2"},
      { PentaxID_GX_1L,        "GX-1L"},        // Samsung
      { PentaxID_K100D,        "K100D"},
      { PentaxID_K110D,        "K110D"},
      { PentaxID_K100D_Super,  "K100D Super"},
      { PentaxID_K10D,         "K10D"},
      { PentaxID_GX10,         "GX10"},         // Samsung
      { PentaxID_GX10,         "GX-10"},        // Samsung
      { PentaxID_K20D,         "K20D"},
      { PentaxID_GX20,         "GX20"},         // Samsung
      { PentaxID_GX20,         "GX-20"},        // Samsung
      { PentaxID_K200D,        "K200D"},
      { PentaxID_K2000,        "K2000"},
      { PentaxID_K_m,          "K-m"},
      { PentaxID_K_7,          "K-7"},
      { PentaxID_K_x,          "K-x"},
      { PentaxID_645D,         "645D"},
      { PentaxID_K_r,          "K-r"},
      { PentaxID_K_5,          "K-5"},
      { PentaxID_Q,            "Q"},
      { PentaxID_K_01,         "K-01"},
      { PentaxID_K_30,         "K-30"},
      { PentaxID_Q10,          "Q10"},
      { PentaxID_K_5_II,       "K-5 II"},
      { PentaxID_K_5_II_s,     "K-5 II s"},
      { PentaxID_Q7,           "Q7"},
      { PentaxID_MX_1,         "MX-1"},
      { PentaxID_K_50,         "K-50"},
      { PentaxID_K_3,          "K-3"},
      { PentaxID_K_500,        "K-500"},
      { PentaxID_645Z,         "645Z"},
      { PentaxID_K_S1,         "K-S1"},
      { PentaxID_K_S2,         "K-S2"},         // Ricoh
      { PentaxID_Q_S1,         "Q-S1"},
      { PentaxID_K_1,          "K-1"},          // Ricoh
      { PentaxID_K_3_II,       "K-3 II"},       // Ricoh
      { PentaxID_GR_III,       "GR III"},       // Ricoh
      { PentaxID_K_70,         "K-70"},         // Ricoh
      { PentaxID_KP,           "KP"},           // Ricoh
      { PentaxID_K_1_Mark_II,  "K-1 Mark II"},  // Ricoh
      { PentaxID_K_3_III,      "K-3 Mark III"}, // Ricoh
      { PentaxID_GR_IIIx,      "GR IIIx"},
    },
#endif
    sonique[] = {
      { SonyID_DSC_R1,         "DSC-R1"},
      { SonyID_DSLR_A100,      "DSLR-A100"},
      { SonyID_DSLR_A900,      "DSLR-A900"},
      { SonyID_DSLR_A700,      "DSLR-A700"},
      { SonyID_DSLR_A200,      "DSLR-A200"},
      { SonyID_DSLR_A350,      "DSLR-A350"},
      { SonyID_DSLR_A300,      "DSLR-A300"},
      { SonyID_DSLR_A900_APSC, "DSLR-A900"},
      { SonyID_DSLR_A380,      "DSLR-A380"},    // DSLR-A390
      { SonyID_DSLR_A330,      "DSLR-A330"},
      { SonyID_DSLR_A230,      "DSLR-A230"},
      { SonyID_DSLR_A290,      "DSLR-A290"},
      { SonyID_DSLR_A850,      "DSLR-A850"},
      { SonyID_DSLR_A850_APSC, "DSLR-A850"},
      { SonyID_DSLR_A550,      "DSLR-A550"},
      { SonyID_DSLR_A500,      "DSLR-A500"},
      { SonyID_DSLR_A450,      "DSLR-A450"},
      { SonyID_NEX_5,          "NEX-5"},
      { SonyID_NEX_3,          "NEX-3"},
      { SonyID_SLT_A33,        "SLT-A33"},
      { SonyID_SLT_A55,        "SLT-A55"},      // SLT-A55V
      { SonyID_DSLR_A560,      "DSLR-A560"},
      { SonyID_DSLR_A580,      "DSLR-A580"},
      { SonyID_NEX_C3,         "NEX-C3"},
      { SonyID_SLT_A35,        "SLT-A35"},
      { SonyID_SLT_A65,        "SLT-A65"},      // SLT-A65V
      { SonyID_SLT_A77,        "SLT-A77"},      // SLT-A77V
      { SonyID_NEX_5N,         "NEX-5N"},
      { SonyID_NEX_7,          "NEX-7"},        // Hasselblad Lunar
      { SonyID_NEX_VG20,       "NEX-VG20"},
      { SonyID_SLT_A37,        "SLT-A37"},
      { SonyID_SLT_A57,        "SLT-A57"},
      { SonyID_NEX_F3,         "NEX-F3"},
      { SonyID_SLT_A99,        "SLT-A99"},      // SLT-A99V / Hasselblad HV
      { SonyID_NEX_6,          "NEX-6"},
      { SonyID_NEX_5R,         "NEX-5R"},
      { SonyID_DSC_RX100,      "DSC-RX100"},    // Hasselblad Stellar
      { SonyID_DSC_RX1,        "DSC-RX1"},
      { SonyID_NEX_VG900,      "NEX-VG900"},
      { SonyID_NEX_VG30,       "NEX-VG30"},
      { SonyID_ILCE_3000,      "ILCE-3000"},    // ILCE-3500
      { SonyID_SLT_A58,        "SLT-A58"},
      { SonyID_NEX_3N,         "NEX-3N"},
      { SonyID_ILCE_7,         "ILCE-7"},
      { SonyID_NEX_5T,         "NEX-5T"},
      { SonyID_DSC_RX100M2,    "DSC-RX100M2"},  // Hasselblad Stellar II
      { SonyID_DSC_RX10,       "DSC-RX10"},
      { SonyID_DSC_RX1R,       "DSC-RX1R"},
      { SonyID_ILCE_7R,        "ILCE-7R"},      // Hasselblad Lusso
      { SonyID_ILCE_6000,      "ILCE-6000"},
      { SonyID_ILCE_5000,      "ILCE-5000"},
      { SonyID_DSC_RX100M3,    "DSC-RX100M3"},
      { SonyID_ILCE_7S,        "ILCE-7S"},
      { SonyID_ILCA_77M2,      "ILCA-77M2"},
      { SonyID_ILCE_5100,      "ILCE-5100"},
      { SonyID_ILCE_7M2,       "ILCE-7M2"},
      { SonyID_DSC_RX100M4,    "DSC-RX100M4"},
      { SonyID_DSC_RX10M2,     "DSC-RX10M2"},
      { SonyID_DSC_RX1RM2,     "DSC-RX1RM2"},
      { SonyID_ILCE_QX1,       "ILCE-QX1"},
      { SonyID_ILCE_7RM2,      "ILCE-7RM2"},
      { SonyID_ILCE_7SM2,      "ILCE-7SM2"},
      { SonyID_ILCA_68,        "ILCA-68"},
      { SonyID_ILCA_99M2,      "ILCA-99M2"},
      { SonyID_DSC_RX10M3,     "DSC-RX10M3"},
      { SonyID_DSC_RX100M5,    "DSC-RX100M5"},
      { SonyID_ILCE_6300,      "ILCE-6300"},
      { SonyID_ILCE_9,         "ILCE-9"},
      { SonyID_ILCE_6500,      "ILCE-6500"},
      { SonyID_ILCE_7RM3,      "ILCE-7RM3"},
      { SonyID_ILCE_7M3,       "ILCE-7M3"},
      { SonyID_DSC_RX0,        "DSC-RX0"},
      { SonyID_DSC_RX10M4,     "DSC-RX10M4"},
      { SonyID_DSC_RX100M6,    "DSC-RX100M6"},
      { SonyID_DSC_HX99,       "DSC-HX99"},
      { SonyID_DSC_RX100M5A,   "DSC-RX100M5A"},
      { SonyID_ILCE_6400,      "ILCE-6400"},
      { SonyID_DSC_RX0M2,      "DSC-RX0M2"},
      { SonyID_DSC_RX100M7,    "DSC-RX100M7"},
      { SonyID_ILCE_7RM4,      "ILCE-7RM4"},
      { SonyID_ILCE_9M2,       "ILCE-9M2"},
      { SonyID_ILCE_6600,      "ILCE-6600"},
      { SonyID_ILCE_6100,      "ILCE-6100"},
      { SonyID_ZV_1,           "ZV-1"},
      { SonyID_ILCE_7C,        "ILCE-7C"},
      { SonyID_ZV_E10,         "ZV-E10"},
      { SonyID_ILCE_7SM3,      "ILCE-7SM3"},
      { SonyID_ILCE_1,         "ILCE-1"},
      { SonyID_ILME_FX3,       "ILME-FX3"},
      { SonyID_ILCE_7RM3A,     "ILCE-7RM3A"},
      { SonyID_ILCE_7RM4A,     "ILCE-7RM4A"},
      { SonyID_ILCE_7M4,       "ILCE-7M4"},
    };

  static const char *orig;

  static const char fujialias[][16] = {
    "@DBP for GX680", "DX-2000",
    "@F500EXR", "F505EXR",
    "@F600EXR", "F605EXR",
    "@F770EXR", "F775EXR",
    "@HS10", "HS10 HS11",
    "@HS20EXR", "HS22EXR",
    "@HS30EXR", "HS33EXR", "HS35EXR",
    "@S5100", "S5500",
    "@S5200", "S5600",
    "@S6000fd", "S6500fd",
    "@S9000", "S9500",
    "@S9100", "S9600",
    "@S200EXR", "S205EXR",
    "@X-T1 IR", "X-T1IR",
    "@GFX 100S", "GFX100S",
    "@GFX 50S II", "GFX50S II"
  };

  static const char kodakalias[][16] = {
    "@DCS Pro 14N", "Camerz ZDS 14", // Camerz rebadge make: "Photo Control"
    "@DCS720X", "SCS2000",
    "@DCS520C", "EOS D2000C", "EOS D2000", // EOS rebadge make: Canon
    "@DCS560C", "EOS D6000C", "EOS D6000", // EOS rebadge make: Canon
    "@DCS460M", "DCS460A", // 'A' was supposed to stand for 'achromatic', marketing changed it to 'M'
    "@DCS460",  "DCS460C", "DCS460D",
    "@DCS465",  "DCS465C", "DCS465D",
    "@EOSDCS1", "EOSDCS1B", "EOSDCS1C",
    "@EOSDCS3", "EOSDCS3B", "EOSDCS3C",
  };

  static const struct
  {
    const char *Kmodel;
    ushort mount;
  } Kodak_mounts[] = {
      {"DCS465", LIBRAW_MOUNT_DigitalBack},
      {"DCS5", LIBRAW_MOUNT_Canon_EF},
      {"DCS Pro SLR/c", LIBRAW_MOUNT_Canon_EF},
      {"DCS", LIBRAW_MOUNT_Nikon_F},
      {"EOS", LIBRAW_MOUNT_Canon_EF},
      {"NC2000", LIBRAW_MOUNT_Nikon_F}, // AP "News Camera"
      {"Pixpro S-1", LIBRAW_MOUNT_mFT},
      {"ProBack", LIBRAW_MOUNT_DigitalBack},
      {"SCS1000", LIBRAW_MOUNT_Canon_EF},
  };

  static const char *KodakMonochrome[] = {
      "DCS420M",    "DCS420A",  "DCS420I",
      "DCS460M",    "DCS460A",  "DCS460I",
      "DCS465M",    "DCS465A",  "DCS465I",
      "DCS560M",    "DCS660M",  "DCS760M", "EOS D2000M", "EOS D6000M",
      "EOSDCS1M",   "EOSDCS1I",
      "EOSDCS3M",   "EOSDCS3I",
      "EOSDCS5M",   "EOSDCS5I",
      "NC2000M",    "NC2000A",  "NC2000I",
  };

  static const char leafalias[][16] = {
      // Leaf re-badged to Mamiya
    "@Aptus-II 5",  "DM22",
    "@Aptus-II 6",  "DM28",
    "@Aptus-II 7",  "DM33",
    "@Aptus-II 8",  "DM40",
    "@Aptus-II 10", "DM56",
  };

  static const char KonicaMinolta_aliases[][24] = {
    "@DG-5D", "DYNAX 5D", "MAXXUM 5D", "ALPHA-5 DIGITAL", "ALPHA SWEET DIGITAL",
    "@DG-7D", "DYNAX 7D", "MAXXUM 7D", "ALPHA-7 DIGITAL",
  };

  static const char nikonalias[][16] = {
      "@COOLPIX 2100",  "E2100",         "@COOLPIX 2500",  "E2500",
      "@COOLPIX 3200",  "E3200",         "@COOLPIX 3700",  "E3700",
      "@COOLPIX 4300",  "E4300",         "@COOLPIX 4500",  "E4500",
      "@COOLPIX 5000",  "E5000",         "@COOLPIX 5400",  "E5400",
      "@COOLPIX 5700",  "E5700",         "@COOLPIX 8400",  "E8400",
      "@COOLPIX 8700",  "E8700",         "@COOLPIX 8800",  "E8800",
      "@COOLPIX 700",   "E700",          "@COOLPIX 800",   "E800",
      "@COOLPIX 880",   "E880",          "@COOLPIX 900",   "E900",
      "@COOLPIX 950",   "E950",          "@COOLPIX 990",   "E990",
      "@COOLPIX 995",   "E995",          "@COOLPIX P7700", "COOLPIX Deneb",
      "@COOLPIX P7800", "COOLPIX Kalon",
  };

  static const char olyalias[][32] = { // Olympus
    "@AIR A01", "AIR-A01",
    "@C-3030Z", "C3030Z",
    "@C-5050Z", "C5050Z",
    "@C-5060WZ", "C5060WZ",
    "@C-7000Z", "C7000Z", "C70Z,C7000Z", "C70Z",
    "@C-7070WZ", "C7070WZ",
    "@C-8080WZ", "C8080WZ",
    "@C-350Z", "C350Z", "X200,D560Z,C350Z", "X200", "D560Z",
    "@C-740UZ", "C740UZ",
    "@C-770UZ", "C770UZ",
    "@E-20", "E-20,E-20N,E-20P", "E-20N", "E-20P",
    "@E-M10 Mark II", "E-M10MarkII", "E-M10_M2", "piX 5oo",
    "@E-M10 Mark III", "E-M10MarkIII", "E-M10_M3",
    "@E-M10 Mark IV", "E-M10MarkIV", "E-M10_M4",
    "@E-M1 Mark II", "E-M1MarkII", "E-M1_M2",
    "@E-M1 Mark III", "E-M1MarkIII", "E-M1_M3",
    "@E-M5 Mark II", "E-M5MarkII", "E-M5_M2",
    "@E-M5 Mark III", "E-M5MarkIII", "E-M5_M3",
    "@SH-2", "SH-3",
    "@SP-310", "SP310",
    "@SP-320", "SP320",
    "@SP-350", "SP350",
    "@SP-500UZ", "SP500UZ",
    "@SP-510UZ", "SP510UZ",
    "@SP-550UZ", "SP550UZ",
    "@SP-560UZ", "SP560UZ",
    "@SP-565UZ", "SP565UZ",
    "@SP-570UZ", "SP570UZ",
    "@Stylus 1", "STYLUS1", "STYLUS1s", "STYLUS1,1s",
  };

  static const char panalias[][16] = { // Panasonic, PanaLeica
// fixed lens
    "@DMC-FX150", "DMC-FX180",
    "@DC-FZ1000M2", "DC-FZ10002", "V-Lux 5",
    "@DMC-FZ1000", "V-LUX (Typ 114)",
    "@DMC-FZ2500", "DMC-FZ2000", "DMC-FZH1",
    "@DMC-FZ100", "V-LUX 2",
    "@DMC-FZ150", "V-LUX 3",
    "@DMC-FZ200", "V-LUX 4",
    "@DMC-FZ300", "DMC-FZ330",
    "@DMC-FZ35", "DMC-FZ38",
    "@DMC-FZ40", "DMC-FZ42", "DMC-FZ45", "DC-FZ40", "DC-FZ42", "DC-FZ45",
    "@DMC-FZ50", "V-LUX 1", "V-LUX1",
    "@DMC-FZ70", "DMC-FZ72",
    "@DC-FZ80", "DC-FZ81", "DC-FZ82", "DC-FZ83", "DC-FZ85",
    "@DMC-LC1", "DIGILUX 2", "Digilux 2", "DIGILUX2",
    "@DMC-LF1", "C (Typ 112)",
    "@DC-LX100M2", "D-Lux 7",
    "@DMC-LX100", "D-LUX (Typ 109)", "D-Lux (Typ 109)",
    "@DMC-LX1", "D-Lux2", "D-LUX2", "D-LUX 2",
    "@DMC-LX2", "D-LUX 3", "D-LUX3",
    "@DMC-LX3", "D-LUX 4",
    "@DMC-LX5", "D-LUX 5",
    "@DMC-LX7", "D-LUX 6",
    "@DMC-LX9", "DMC-LX10", "DMC-LX15",
    "@DMC-ZS100", "DMC-ZS110", "DMC-TZ100", "DMC-TZ101", "DMC-TZ110", "DMC-TX1",
    "@DC-ZS200", "DC-ZS220", "DC-TZ200", "DC-TZ202", "DC-TZ220", "DC-TX2", "C-Lux", "CAM-DC25",
    "@DMC-ZS40", "DMC-TZ60", "DMC-TZ61",
    "@DMC-ZS50", "DMC-TZ70", "DMC-TZ71",
    "@DMC-ZS60", "DMC-TZ80", "DMC-TZ81", "DMC-TZ82", "DMC-TZ85",
    "@DC-ZS70", "DC-TZ90", "DC-TZ91", "DC-TZ92", "DC-TZ93",
    "@DC-ZS80", "DC-TZ95", "DC-TZ96", "DC-TZ97",

// interchangeable lens
    "@DC-G100",  "DC-G110",
    "@DC-G99",   "DC-G90",   "DC-G91",  "DC-G95",
    "@DMC-G7",   "DMC-G70",
    "@DMC-G8",   "DMC-G80",  "DMC-G81", "DMC-G85",
    "@DMC-GH4",  "AG-GH4",   "CGO4",
    "@DC-GF10",  "DC-GF90",  "DC-GX880",
    "@DC-GF9",   "DC-GX850", "DC-GX800",
    "@DMC-GM1",  "DMC-GM1S",
    "@DMC-GX85", "DMC-GX80", "DMC-GX7MK2",
    "@DC-GX9",   "DC-GX7MK3",
    "@DMC-L1",   "DIGILUX 3", "DIGILUX3", // full 4/3 mount, not m43
  };

  static const char phase1alias[][16] = {
    "@H20",  "H 20",
    "@H25",  "H 25",
    "@P20+", "P 20+",
    "@P20",  "P 20",
    "@P21+", "P 21+", "M18", // "Mamiya M18"
    "@P21",  "P 21",
    "@P25+", "P 25+", "M22", // "Mamiya M22"
    "@P25",  "P 25",
    "@P30+", "P 30+", "M31", // "Mamiya M31"
    "@P30",  "P 30",
    "@P40+", "P 40+",
    "@P40",  "P 40",
    "@P45+", "P 45+",
    "@P45",  "P 45",
    "@P65+", "P 65+",
    "@P65",  "P 65",
  };

  static const char SamsungPentax_aliases[][16] = {
    "@*istDL2", "*ist DL2", "GX-1L",
    "@*istDS2", "*ist DS2", "GX-1S",
    "@*istDL",  "*ist DL",
    "@*istDS",  "*ist DS",
    "@*istD",   "*ist D",
    "@K10D", "GX10", "GX-10",
    "@K20D", "GX20", "GX-20",
    "@K-m", "K2000",
  };

  static const char samsungalias[][64] = {
    "@EX1", "TL500",
    "@NX U", "EK-GN100", "EK-GN110", "EK-GN120", "EK-KN120", "Galaxy NX",
    "@NX mini", "NXF1",
    "@WB2000", "TL350",
      //    "@WB5000", "WB5000/HZ25W", // no spaces around the slash separating names
      //    "@WB5500", "WB5500 / VLUU WB5500 / SAMSUNG HZ50W",
      //    "@WB500", "WB510 / VLUU WB500 / SAMSUNG HZ10W",
      //    "@WB550", "WB560 / VLUU WB550 / SAMSUNG HZ15W",
      //    "@WB650", "SAMSUNG WB650 / VLUU WB650 / SAMSUNG WB660" aka HZ35W
  };

//clang-format on
  if (makeIs(LIBRAW_CAMERAMAKER_VLUU)) {
	  setMakeFromIndex(LIBRAW_CAMERAMAKER_Samsung);
  }

  if (makeIs(LIBRAW_CAMERAMAKER_Samsung) &&
      (ilm.CameraMount == LIBRAW_MOUNT_Pentax_K)) {
	  setMakeFromIndex(LIBRAW_CAMERAMAKER_Pentax);

  } else if (makeIs(LIBRAW_CAMERAMAKER_Unknown)) {
    if (strcasestr(model, "Google")) {
		  setMakeFromIndex(LIBRAW_CAMERAMAKER_Google);
    }
#ifdef USE_6BY9RPI
	else if(strcasestr(make,"RaspberryPi"))
		setMakeFromIndex(LIBRAW_CAMERAMAKER_RaspberryPi);
#endif
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Hasselblad) && is_Sony)
  {
	  setMakeFromIndex(LIBRAW_CAMERAMAKER_Sony);
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_OmDigital))
  {
	  setMakeFromIndex(LIBRAW_CAMERAMAKER_Olympus);
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Clauss) && (OlyID == OlyID_E_M10_Mark_II))
  {
	  setMakeFromIndex(LIBRAW_CAMERAMAKER_Olympus);
  } else if (makeIs(LIBRAW_CAMERAMAKER_Canon) &&
             (!strncmp(model, "EOS D2000", 9) || // don't use unique_id here
              !strncmp(model, "EOS D6000", 9) || // because ids for Monochrome models are unknown
              !strncmp(model, "EOSDCS", 6))) {
    setMakeFromIndex(LIBRAW_CAMERAMAKER_Kodak);
//    if (unique_id == CanonID_EOS_D2000C) {
//
//    } else if (unique_id  == CanonID_EOS_D6000C) {
///
//    }

  } else if (makeIs(LIBRAW_CAMERAMAKER_PhotoControl) &&
             !strncasecmp(model, "Camerz ZDS 14", 13)) {
	  setMakeFromIndex(LIBRAW_CAMERAMAKER_Kodak);

  } else {
    strcpy(normalized_make, make);
  }

  if (makeIs(LIBRAW_CAMERAMAKER_Apple)) {
    if ((imgdata.color.UniqueCameraModel[0]) &&
        (!strncmp(model, "iPad", 4) || !strncmp(model, "iPhone", 6)))
    strcpy(model, imgdata.color.UniqueCameraModel);

  } else if (makeIs(LIBRAW_CAMERAMAKER_Kodak)) {
    if ((model[6] == ' ') &&
        (!strncmp(model, "DCS4", 4) ||
         !strncmp(model, "NC2000", 6)))
    {
      model[6] = 0;
    }
    if ((model[6] != 'A') &&
        (model[6] != 'I') &&
        (model[6] != 'M') &&
        !strncmp(model, "NC2000", 6))
    {
      model[6] = 0;
    }
  }

  else if (makeIs(LIBRAW_CAMERAMAKER_Ricoh) &&
           !strncmp(model, "GXR", 3)) {
    strcpy(ilm.body, "Ricoh GXR");
    if (!imgdata.lens.Lens[0] && imgdata.color.UniqueCameraModel[0]) {
      strcpy (imgdata.lens.Lens, imgdata.color.UniqueCameraModel);
      remove_caseSubstr (imgdata.lens.Lens, (char *)"Ricoh");
      remove_caseSubstr (imgdata.lens.Lens, (char *)"Lens");
      removeExcessiveSpaces (imgdata.lens.Lens);
    }
    if (ilm.LensID == LIBRAW_LENS_NOT_SET) {
      if (strstr(imgdata.lens.Lens, "50mm"))
        ilm.LensID = 1;
      else if (strstr(imgdata.lens.Lens, "S10"))
        ilm.LensID = 2;
      else if (strstr(imgdata.lens.Lens, "P10"))
        ilm.LensID = 3;
      else if (strstr(imgdata.lens.Lens, "28mm"))
        ilm.LensID = 5;
      else if (strstr(imgdata.lens.Lens, "A16"))
        ilm.LensID = 6;
    }
    switch (ilm.LensID) {
    case 1: // GR Lens A12 50mm F2.5 Macro
      strcpy(model, "GXR A12 50mm");
      ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_APSC;
      ilm.CameraMount = LIBRAW_MOUNT_RicohModule;
      ilm.LensMount = LIBRAW_MOUNT_FixedLens;
      ilm.FocalType = LIBRAW_FT_PRIME_LENS;
      break;
    case 2:
      strcpy(model, "GXR S10");
      ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_1div1p7INCH;
      ilm.CameraMount = LIBRAW_MOUNT_RicohModule;
      ilm.LensMount = LIBRAW_MOUNT_FixedLens;
      ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
      break;
    case 3: // Ricoh Lens P10 28-300mm F3.5-5.6 VC
      strcpy(model, "GXR P10");
      ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_1div2p3INCH;
      ilm.CameraMount = LIBRAW_MOUNT_RicohModule;
      ilm.LensMount = LIBRAW_MOUNT_FixedLens;
      ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
      break;
    case 5: // GR Lens A12 28mm F2.5
      strcpy(model, "GXR A12 28mm");
      ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_APSC;
      ilm.CameraMount = LIBRAW_MOUNT_RicohModule;
      ilm.LensMount = LIBRAW_MOUNT_FixedLens;
      ilm.FocalType = LIBRAW_FT_PRIME_LENS;
      break;
    case 6: // Ricoh Lens A16 24-85mm F3.5-5.5
      strcpy(model, "GXR A16");
      ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_APSC;
      ilm.CameraMount = LIBRAW_MOUNT_RicohModule;
      ilm.LensMount = LIBRAW_MOUNT_FixedLens;
      ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
      break;
    case 8: // Ricoh Mount A12 (Leica M lenses)
      strcpy(model, "GXR Mount A12");
      ilm.CameraFormat = LIBRAW_FORMAT_APSC;
      ilm.CameraMount = LIBRAW_MOUNT_Leica_M;
      ilm.LensID = LIBRAW_LENS_NOT_SET;
      break;
    }
  }

  strcpy(normalized_model, model);

  if (makeIs(LIBRAW_CAMERAMAKER_Canon))
  {
    if (unique_id)
    {
      if ((unique_id != CanonID_EOS_D2000C) &&
          (unique_id != CanonID_EOS_D6000C))
      {
        for (i = 0; i < int(sizeof unique / sizeof *unique); i++)
        {
          if (unique_id == unique[i].id)
          {
            strcpy(model, unique[i].t_model);
            strcpy(normalized_model, unique[i].t_model);
            break;
          }
        }
      }
    }
    else
    {
      if ((dng_version) &&
          (strlen(imgdata.color.UniqueCameraModel) > 6) &&
          strncmp(imgdata.color.UniqueCameraModel+6, "PowerShot", 9))
      {
        for (i = 0; i < int(sizeof unique / sizeof *unique); i++)
        {
          if (!strcmp(unique[i].t_model, imgdata.color.UniqueCameraModel+6))
          {
            ilm.CamID = unique_id = unique[i].id;
            strcpy(normalized_model, unique[i].t_model);
            try_xml = 1;
            break;
          }
        }
      }
    }
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Fujifilm))
  {
    for (i = 0; i < int(sizeof fujialias / sizeof *fujialias); i++)
    {
      if (fujialias[i][0] == '@')
      {
        orig = fujialias[i] + 1;
        if (!strcmp(model, orig)) break;
      }
      else if (!strcmp(model, fujialias[i]))
      {
        strcpy(normalized_model, orig);
        break;
      }
    }

  } else if (makeIs(LIBRAW_CAMERAMAKER_Hasselblad)) {
    parseHassyModel();
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Mamiya))
  {
    for (i = 0; i < int(sizeof phase1alias / sizeof *phase1alias); i++)
    { // re-badged Phase One backs
      if (phase1alias[i][0] == '@') orig = phase1alias[i] + 1;
      else if (!strcmp(model, phase1alias[i]))
      {
        setMakeFromIndex(LIBRAW_CAMERAMAKER_PhaseOne);
        strcpy(normalized_model, orig);
        break;
      }
    }
    for (i = 0; i < int(sizeof leafalias / sizeof *leafalias); i++)
    { // re-badged Leaf backs
      if (leafalias[i][0] == '@') orig = leafalias[i] + 1;
      else if (!strcmp(model, leafalias[i]))
      {
        setMakeFromIndex(LIBRAW_CAMERAMAKER_Leaf);
        strcpy(normalized_model, orig);
        break;
      }
    }

    /* repeating, because make for some Mamiya re-badged Leaf backs is set to
     * Leaf */
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Leaf))
  {
    for (i = 0; i < int(sizeof leafalias / sizeof *leafalias); i++)
    { // re-badged Leaf backs
      if (leafalias[i][0] == '@')
      {
        orig = leafalias[i] + 1;
        if (!strcmp(model, orig)) break;
      }
      else if (!strcmp(model, leafalias[i]))
      { // maybe to change regular "make" to "Mamiya" too
        strcpy(normalized_model, orig);
        break;
      }
    }
    if ((ps = strchr(normalized_model, '(')))
      *ps = 0;
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Minolta) ||
           makeIs(LIBRAW_CAMERAMAKER_Konica))
  {
    if (makeIs(LIBRAW_CAMERAMAKER_Konica) && !strncasecmp(model, "DiMAGE", 6))
    {
      setMakeFromIndex(LIBRAW_CAMERAMAKER_Minolta);
      strcpy(make, "Minolta");
    }
    else
    {
      for (i = 0;
           i<int(sizeof KonicaMinolta_aliases / sizeof *KonicaMinolta_aliases);
           i++)
      {
        if (KonicaMinolta_aliases[i][0] == '@')
        {
          orig = KonicaMinolta_aliases[i] + 1;
          if (!strcmp(model, orig))
          {
            setMakeFromIndex(LIBRAW_CAMERAMAKER_Minolta);
            strcpy(make, "Minolta");
            break;
          }
        }
        else if (!strcasecmp(model, KonicaMinolta_aliases[i]))
        {
          setMakeFromIndex(LIBRAW_CAMERAMAKER_Minolta);
          strcpy(make, "Minolta");
          strcpy(normalized_model, orig);
          break;
        }
      }
    }
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Nikon))
  {
    for (i = 0; i < int(sizeof nikonalias / sizeof *nikonalias); i++)
    {
      if (nikonalias[i][0] == '@')
      {
        orig = nikonalias[i] + 1;
        if (!strcmp(model, orig)) break;
      }
      else if (!strcmp(model, nikonalias[i]))
      {
        strcpy(normalized_model, orig);
        break;
      }
    }

  } else if (makeIs(LIBRAW_CAMERAMAKER_Olympus)) {
    for (i = 0; i < int(sizeof olyalias / sizeof *olyalias); i++) {
      if (olyalias[i][0] == '@') {
        orig = olyalias[i] + 1;
        if (!strcmp(model, orig)) break;
      } else if (!strcmp(model, olyalias[i])) {
        strcpy(normalized_model, orig);
        break;
      }
    }

    if (!OlyID) {
      if (!strcmp(normalized_model, "C-740UZ")) {
        ilm.CamID = OlyID = unique_id = OlyID_C_740UZ;

      } else if (!strcmp(normalized_model, "C-770UZ")) {
        ilm.CamID = OlyID = unique_id = OlyID_C_770UZ;
      }
    }

  } else if (makeIs(LIBRAW_CAMERAMAKER_Panasonic) ||
             makeIs(LIBRAW_CAMERAMAKER_Leica) ||
             makeIs(LIBRAW_CAMERAMAKER_Yuneec))
  {
    for (i = 0; i < int(sizeof panalias / sizeof *panalias); i++)
    {
      if (panalias[i][0] == '@')
      {
        orig = panalias[i] + 1;
        if (!strcmp(model, orig)) break;
      }
      else if (!strcmp(model, panalias[i]))
      {
        setMakeFromIndex(LIBRAW_CAMERAMAKER_Panasonic);
        strcpy(normalized_model, orig);
        break;
      }
    }
  } else if (makeIs(LIBRAW_CAMERAMAKER_Pentax)) {
    if (!unique_id) {
      if (!strcmp(model, "Optio S")) {
        ilm.CamID = unique_id = PentaxID_Optio_S;
      } else if (!strcmp(model, "Optio S V1.01")) {
        ilm.CamID = unique_id = PentaxID_Optio_S_V101;
      } else if (!strcmp(model, "Optio S4")) {
        ilm.CamID = unique_id = PentaxID_Optio_S4;
      } else if (!strcmp(model, "Optio 750Z")) {
        ilm.CamID = unique_id = PentaxID_Optio_750Z;
      } else if (!strcmp(model, "Optio 33WR")) {
        ilm.CamID = unique_id = PentaxID_Optio_33WR;
      }
    }

    for (i = 0;
    i < int(sizeof SamsungPentax_aliases / sizeof *SamsungPentax_aliases);
    i++) {
      if (SamsungPentax_aliases[i][0] == '@') {
        orig = SamsungPentax_aliases[i] + 1;
        if (!strcmp(model, orig)) break;
      } else if (!strcmp(model, SamsungPentax_aliases[i])) {
        strcpy(normalized_model, orig);
        break;
      }
    }
    if (!strncmp(model, "GR", 2)) {
	  setMakeFromIndex(LIBRAW_CAMERAMAKER_Ricoh);
      strcpy(make, "Ricoh");
    }

  } else if (makeIs(LIBRAW_CAMERAMAKER_PhaseOne))
  {
    for (i = 0; i < int(sizeof phase1alias / sizeof *phase1alias); i++)
    {
      if (phase1alias[i][0] == '@')
      {
        orig = phase1alias[i] + 1;
        if (!strcmp(model, orig)) break;
      }
      else if (!strcmp(model, phase1alias[i]))
      {
        strcpy(normalized_model, orig);
        break;
      }
    }
	if (strstr(imPhaseOne.FirmwareString, "Achromatic"))
	{
		colors = 1;
		filters = 0;
	}
  }
  else if (makeIs(LIBRAW_CAMERAMAKER_Samsung))
  {
    j = 0;
    if (strstr(model, "WB5500") || strstr(model, "HZ50W"))
    {
      strcpy(model, "WB5500");
      j++;
    }
    else if (strstr(model, "WB5000") || strstr(model, "HZ25W"))
    {
      strcpy(model, "WB5000");
      j++;
    }
    else if (strstr(model, "WB550") || strstr(model, "HZ15W"))
    {
      strcpy(model, "WB550");
      j++;
    }
    else if (strstr(model, "WB500") || strstr(model, "HZ10W"))
    {
      strcpy(model, "WB500");
      j++;
    }
    if (j)
    {
      strcpy(normalized_model, model);
    }
    else
    {
      for (i = 0; i < int(sizeof samsungalias / sizeof *samsungalias); i++)
      {
        if (samsungalias[i][0] == '@')
        {
          orig = samsungalias[i] + 1;
          if (!strcmp(model, orig)) break;
        }
        else if (!strcmp(model, samsungalias[i]))
        {
          strcpy(normalized_model, orig);
          break;
        }
      }
    }

  } else if (makeIs(LIBRAW_CAMERAMAKER_Sony)) {
    if (unique_id)
    {
      for (i = 0; i < int(sizeof sonique / sizeof *sonique); i++)
      {
        if (unique_id == sonique[i].id)
        {
          if (!strcmp(make, "Sony"))
            strcpy(model, sonique[i].t_model);
          strcpy(normalized_model, sonique[i].t_model);
          break;
        }
      }
    }

  } else if (makeIs(LIBRAW_CAMERAMAKER_Kodak)) {
    remove_caseSubstr (normalized_model, (char *)"EasyShare");
    remove_caseSubstr (normalized_model, (char *)"ZOOM");
    removeExcessiveSpaces (normalized_model);
    for (i = 0; i < int(sizeof kodakalias / sizeof *kodakalias); i++)
    {
      if (kodakalias[i][0] == '@')
      {
        orig = kodakalias[i] + 1;
        if (!strcmp(model, orig)) break;
      }
      else if (!strcmp(model, kodakalias[i]))
      {
        strcpy(normalized_model, orig);
        break;
      }
    }

    if (strstr(model, "DC25"))
    {
      strcpy(model, "DC25");
      strcpy(normalized_model, model);
    }
    else if (!strcmp(model, "40"))
    {
      strcpy(model, "DC40");
      strcpy(normalized_model, model);
    }
    else if (strstr(model, "DC50"))
    {
      strcpy(model, "DC50");
      strcpy(normalized_model, model);
    }
    else if (strstr(model, "DC120"))
    {
      strcpy(model, "DC120");
      strcpy(normalized_model, model);
    }

    for (i = 0; i < int(sizeof KodakMonochrome / sizeof *KodakMonochrome); i++)
    {
      if (!strncmp(model, KodakMonochrome[i], strlen(KodakMonochrome[i])))
      {
        colors = 1;
        filters = 0;
      }
    }
  }

  if (ilm.body[0])
  {
    if ((ilm.CameraMount != LIBRAW_MOUNT_Hasselblad_V) &&
        !strncmp(ilm.body, "Hasselblad ", 11) &&
        ((ilm.body[11] == 'C') || (ilm.body[11] == '2') ||
         (ilm.body[11] == '5') || (ilm.body[11] == '9')))
    {
      ilm.CameraFormat = LIBRAW_FORMAT_66;
      ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_V;
    }
    else if (!strncmp(ilm.body, "XF", 2) || !strncmp(ilm.body, "645DF", 5))
    {
      ilm.CameraMount = LIBRAW_MOUNT_Mamiya645;
      ilm.CameraFormat = LIBRAW_FORMAT_645;
    }
    else if (!strncmp(ilm.body, "Sinarcam", 2))
    {
      ilm.CameraMount = LIBRAW_MOUNT_LF;
      ilm.CameraFormat = LIBRAW_FORMAT_LF;
      strcat(ilm.body, " shutter system");
    }
  }

  if (makeIs(LIBRAW_CAMERAMAKER_Kodak))
  {
    if (((ilm.CameraMount == LIBRAW_MOUNT_DigitalBack) ||
         (ilm.CameraMount == LIBRAW_MOUNT_Unknown)) &&
        !strncmp(model2, "PB645", 5))
    {
      ilm.CameraFormat = LIBRAW_FORMAT_645;
      if (model2[5] == 'C')
      {
        ilm.CameraMount = LIBRAW_MOUNT_Contax645;
        strcpy(ilm.body, "Contax 645");
      }
      else if (model2[5] == 'H')
      {
        ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_H;
        strcpy(ilm.body, "Hasselblad H1/H2");
      }
      else if (model2[5] == 'M')
      {
        ilm.CameraMount = LIBRAW_MOUNT_Mamiya645;
        strcpy(ilm.body, "Mamiya 645");
      }

    } else if (!strncasecmp(model, "PIXPRO S-1", 10)) {
      ilm.CameraFormat = LIBRAW_FORMAT_FT;
    } else if (!strncasecmp(model, "PIXPRO ", 7)) {
      ilm.CameraFormat = LIBRAW_FORMAT_1div2p3INCH;
    }
  }

  else if (makeIs(LIBRAW_CAMERAMAKER_Fujifilm))
  {
    if (!strncmp(normalized_model, "DBP", 3))
    {
      strcpy(ilm.body, "Fujifilm GX680");
    }
  }

  if ((ilm.CameraFormat == LIBRAW_FORMAT_Unknown) ||
      (ilm.CameraMount == LIBRAW_MOUNT_Unknown) ||
      (ilm.CameraMount == LIBRAW_MOUNT_IL_UM))
  {

    if (makeIs(LIBRAW_CAMERAMAKER_Canon))
    {
      if (strncmp(normalized_model, "EOS", 3))
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Nikon))
    {
      if (normalized_model[0] == 'D')
      {
        ilm.CameraMount = LIBRAW_MOUNT_Nikon_F;
      }
      else
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Panasonic))
    {
      if (!strncmp(normalized_model, "DC-S", 4))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_FF;
        ilm.CameraMount = LIBRAW_MOUNT_LPS_L;
      }
      else if (!strncmp(normalized_model, "DMC-L1", 6) ||
               !strncmp(normalized_model, "DMC-L10", 7))
      {
        ilm.CameraFormat = ilm.CameraMount = LIBRAW_FORMAT_FT;
      }
      else if (!strncmp(normalized_model + 2, "-G", 2) ||
               !strncmp(normalized_model + 3, "-G", 2))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_FT;
        ilm.CameraMount = LIBRAW_MOUNT_mFT;
      }
      else
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
        ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
        if (!strncmp(normalized_model + 2, "-LX100", 6) || // DC-LX100M2
            !strncmp(normalized_model + 3, "-LX100", 6))
        { // DMC-LX100
          ilm.CameraFormat = ilm.LensFormat = LIBRAW_FORMAT_FT;
        }
        else if (!strncmp(normalized_model, "DMC-CM1", 7))
        {
          ilm.FocalType = LIBRAW_FT_PRIME_LENS;
        }
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Fujifilm))
    {
      if (!strncmp(normalized_model, "GFX ", 4))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_CROP645;
        ilm.CameraMount = LIBRAW_MOUNT_Fuji_GF;
      }
      else if (!strncmp(normalized_model, "X-", 2) &&
               (strncmp(normalized_model, "X-S1", 4) || !strncmp(normalized_model, "X-S10", 5)))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_APSC;
        ilm.CameraMount = LIBRAW_MOUNT_Fuji_X;
      }
      else if (((normalized_model[0] == 'S') && // S2Pro, S3Pro, S5Pro
                (normalized_model[2] == 'P')) ||
               !strncasecmp(normalized_model, "IS Pro", 6))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_APSC;
        ilm.CameraMount = LIBRAW_MOUNT_Nikon_F;
      }
      else if (!strncmp(normalized_model, "DBP", 3))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_68;
        ilm.CameraMount = LIBRAW_MOUNT_Fuji_GX;
      }
      else
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Samsung))
    {
      if ((normalized_model[0] == 'N') &&
          (normalized_model[1] == 'X')) // DNG converters delete makernotes
      {
        if ((normalized_model[2] == 'F') && (normalized_model[3] == '1'))
        {
          ilm.CameraMount = LIBRAW_MOUNT_Samsung_NX_M;
          ilm.CameraFormat = LIBRAW_FORMAT_1INCH;
        }
        else
        {
          ilm.CameraMount = LIBRAW_MOUNT_Samsung_NX;
          ilm.CameraFormat = LIBRAW_FORMAT_APSC;
        }
      }
      else
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Kodak))
    {
      ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      for (i = 0; i < int(sizeof Kodak_mounts / sizeof *Kodak_mounts); i++)
      {
        if (!strncmp(normalized_model, Kodak_mounts[i].Kmodel,
                     strlen(Kodak_mounts[i].Kmodel)))
        {
          ilm.CameraMount = Kodak_mounts[i].mount;
          break;
        }
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Minolta))
    {
      if (!strcmp(normalized_model, "DG-5D") ||
          !strcmp(normalized_model, "DG-7D"))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_APSC;
        ilm.CameraMount = LIBRAW_MOUNT_Minolta_A;
      }
      else if (!strncasecmp(normalized_model, "DiMAGE", 6))
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Casio) ||
             makeIs(LIBRAW_CAMERAMAKER_Creative))
    {
      ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Sigma))
    {
      if (!strncmp(normalized_model, "fp", 2))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_FF;
        ilm.CameraMount = LIBRAW_MOUNT_LPS_L;
      }
      else if (!strncasecmp(normalized_model, "SD", 2))
      {
        ilm.CameraMount = LIBRAW_MOUNT_Sigma_X3F;
        if (!strcmp(normalized_model, "SD1") || (normalized_model[4] == 'M'))
        {
          ilm.CameraFormat = LIBRAW_FORMAT_SigmaMerrill;
        }
        else if (normalized_model[11] == 'H')
        { // 'sd Quattro H'
          ilm.CameraFormat = LIBRAW_FORMAT_SigmaAPSH;
        }
        else if (normalized_model[4] == 'Q')
        { // 'sd Quattro'
          ilm.CameraFormat = LIBRAW_FORMAT_APSC;
        }
        else
        {
          ilm.CameraFormat = LIBRAW_FORMAT_SigmaAPSC;
        }
      }
      else if (!strncasecmp(normalized_model, "DP", 2))
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
        if (normalized_model[4] == 'M')
        {
          ilm.CameraFormat = LIBRAW_FORMAT_SigmaMerrill;
        }
        else if (normalized_model[4] == 'Q')
        {
          ilm.CameraFormat = LIBRAW_FORMAT_APSC;
        }
        else
        {
          ilm.CameraFormat = LIBRAW_FORMAT_SigmaAPSC;
        }
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Konica))
    {
      if (!strncmp(model, "KD-", 3))
      { // Konica KD-400Z, KD-510Z
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Mamiya))
    {
      if (!strncmp(normalized_model, "ZD", 2))
      {
        ilm.CameraFormat = LIBRAW_FORMAT_3648;
        ilm.CameraMount = LIBRAW_MOUNT_Mamiya645;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Sony))
    {
      if (!strncmp(normalized_model, "XCD-", 4))
      {
        ilm.CameraMount = LIBRAW_MOUNT_C;
      }
      else if (!strncmp(normalized_model, "DSC-V3", 6) ||
               !strncmp(normalized_model, "DSC-F828", 8))
      {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
        imSony.CameraType = LIBRAW_SONY_DSC;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Polaroid) &&
             !strncmp(normalized_model, "x530", 4))
    {
      ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Rollei) &&
             !strncmp(normalized_model, "d530flex", 8))
    {
      ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Pentax) &&
             !strncmp(normalized_model, "Optio", 5)) {
        ilm.CameraMount = LIBRAW_MOUNT_FixedLens;
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Epson) &&
             !strncmp(normalized_model, "R-D1", 4))
    {
      ilm.CameraMount = LIBRAW_MOUNT_Leica_M;
      ilm.CameraFormat = LIBRAW_FORMAT_APSC;
    }
  }

  if ((ilm.LensMount == LIBRAW_MOUNT_Canon_RF) &&
      (ilm.LensID == 61182)                    &&
      (imCanon.RF_lensID != 0))                {
    ilm.LensID = imCanon.RF_lensID;
  }

  if (ilm.LensMount == LIBRAW_MOUNT_Unknown)
  {
    if (makeIs(LIBRAW_CAMERAMAKER_Samsung))
    {
      if ((imgdata.lens.Lens[0] == 'N') &&
          (imgdata.lens.Lens[1] == 'X')) // same DNG problem
      {
        if (imgdata.lens.Lens[2] == '-')
        {
          ilm.LensMount = LIBRAW_MOUNT_Samsung_NX_M;
          ilm.LensFormat = LIBRAW_FORMAT_1INCH;
        }
        else
        {
          ilm.LensMount = LIBRAW_MOUNT_Samsung_NX;
          ilm.LensFormat = LIBRAW_FORMAT_APSC;
        }
      }
    }
  }

  if ((ilm.LensID == LIBRAW_LENS_NOT_SET) &&
      xmpdata &&
      (strlen(xmpdata) > 9))
  {
    if (makeIs(LIBRAW_CAMERAMAKER_Canon) &&
        try_xml &&
        (ps = strstr(xmpdata, "LensID=\"")))
    {
      ilm.LensID = atoi(ps + 8);
      if (ilm.LensID == 61182)
      {
        ilm.LensMount = ilm.CameraMount = LIBRAW_MOUNT_Canon_RF;
      }
    }
    else if (makeIs(LIBRAW_CAMERAMAKER_Samsung))
    {
      if ((ilm.LensMount == LIBRAW_MOUNT_Samsung_NX) &&
          (ps = strstr(xmpdata, "LensID=\"(")))
      {
        ilm.LensID = atoi(ps + 9);
      }
    }
  }

  if (ilm.CameraMount == LIBRAW_MOUNT_FixedLens)
  {
    if (ilm.CameraFormat)
      ilm.LensFormat = ilm.CameraFormat;
    if (ilm.LensMount == LIBRAW_MOUNT_Unknown)
      ilm.LensMount = LIBRAW_MOUNT_FixedLens;
  }

  if ((ilm.CameraMount != LIBRAW_MOUNT_Unknown) &&
      (ilm.CameraMount != LIBRAW_MOUNT_FixedLens) &&
      (ilm.LensMount == LIBRAW_MOUNT_Unknown)) {
    if (ilm.LensID == LIBRAW_LENS_NOT_SET) ilm.LensMount = LIBRAW_MOUNT_IL_UM;
    else ilm.LensMount = ilm.CameraMount;
    }
}

void LibRaw::SetStandardIlluminants (unsigned makerIdx, const char* /*normModel*/) {
  int i = -1;
  int c;
  if (!icWBC[LIBRAW_WBI_Ill_A][0] &&
      !icWBC[LIBRAW_WBI_D65][0]) {
    if (makerIdx == LIBRAW_CAMERAMAKER_Olympus ) {
      while (++i, icWBCCTC[i][0]) {
        if (icWBCCTC[i][0] == 3000)
          FORC4 icWBC[LIBRAW_WBI_Ill_A][c] = icWBCCTC[i][c+1];
        else if (icWBCCTC[i][0] == 6600)
          FORC4 icWBC[LIBRAW_WBI_D65][c] = icWBCCTC[i][c+1];
      }
    }
  }

  if (!icWBC[LIBRAW_WBI_Ill_A][0] && icWBC[LIBRAW_WBI_Tungsten][0])
    FORC4 icWBC[LIBRAW_WBI_Ill_A][c] = icWBC[LIBRAW_WBI_Tungsten][c];

  if (!icWBC[LIBRAW_WBI_D65][0] && icWBC[LIBRAW_WBI_FL_N][0])
    FORC4 icWBC[LIBRAW_WBI_D65][c] = icWBC[LIBRAW_WBI_FL_N][c];

  return;
}
