/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *
 LibRaw uses code from dcraw.c -- Dave Coffin's raw photo decoder,
 dcraw.c is copyright 1997-2018 by Dave Coffin, dcoffin a cybercom o net.
 LibRaw do not use RESTRICTED code from dcraw.c

 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/dcraw_defs.h"
#include "../../internal/libraw_cameraids.h"

// clang-format on
static const struct
{
    const int CorpId;
    const char *CorpName;
} CorpTable[] = {
    {LIBRAW_CAMERAMAKER_Agfa,           "AgfaPhoto"},
    {LIBRAW_CAMERAMAKER_Apple,          "Apple"},
    {LIBRAW_CAMERAMAKER_Broadcom,       "Broadcom"},
    {LIBRAW_CAMERAMAKER_Canon,          "Canon"},
    {LIBRAW_CAMERAMAKER_Casio,          "Casio"},
    {LIBRAW_CAMERAMAKER_CINE,           "CINE"},
    {LIBRAW_CAMERAMAKER_Epson,          "Epson"},
    {LIBRAW_CAMERAMAKER_Fujifilm,       "Fujifilm"},
    {LIBRAW_CAMERAMAKER_Mamiya,         "Mamiya"},
    {LIBRAW_CAMERAMAKER_Motorola,       "Motorola"},
    {LIBRAW_CAMERAMAKER_Kodak,          "Kodak"},
    {LIBRAW_CAMERAMAKER_Konica,         "Konica"},
    {LIBRAW_CAMERAMAKER_Minolta,        "Minolta"},
    {LIBRAW_CAMERAMAKER_Leica,          "Leica"},
    {LIBRAW_CAMERAMAKER_Nikon,          "Nikon"},
    {LIBRAW_CAMERAMAKER_Nokia,          "Nokia"},
    {LIBRAW_CAMERAMAKER_Olympus,        "Olympus"},
    {LIBRAW_CAMERAMAKER_OmDigital,		"OM Digital"},
    {LIBRAW_CAMERAMAKER_Ricoh,			"Ricoh"},
    {LIBRAW_CAMERAMAKER_Pentax,         "Pentax"},
    {LIBRAW_CAMERAMAKER_PhaseOne,       "Phase One"},
    {LIBRAW_CAMERAMAKER_PhaseOne,       "PhaseOne"},
    {LIBRAW_CAMERAMAKER_Samsung,        "Samsung"},
    {LIBRAW_CAMERAMAKER_Sigma,          "Sigma"},
    {LIBRAW_CAMERAMAKER_Sinar,          "Sinar"},
    {LIBRAW_CAMERAMAKER_Sony,           "Sony"},
    {LIBRAW_CAMERAMAKER_YI,             "YI"},
    // add corp. names below
    {LIBRAW_CAMERAMAKER_Alcatel,        "Alcatel"},
    {LIBRAW_CAMERAMAKER_Aptina,         "Aptina"},
    {LIBRAW_CAMERAMAKER_AVT,            "AVT"},
    {LIBRAW_CAMERAMAKER_Baumer,         "Baumer"},
    {LIBRAW_CAMERAMAKER_Clauss,         "Clauss"},
    {LIBRAW_CAMERAMAKER_Contax,         "Contax"},
    {LIBRAW_CAMERAMAKER_Creative,       "Creative"},
    {LIBRAW_CAMERAMAKER_DJI,            "DJI"},
    {LIBRAW_CAMERAMAKER_Foculus,        "Foculus"},
    {LIBRAW_CAMERAMAKER_Generic,        "Generic"},
    {LIBRAW_CAMERAMAKER_Gione,          "Gione"},
    {LIBRAW_CAMERAMAKER_GITUP,          "GITUP"},
    {LIBRAW_CAMERAMAKER_Hasselblad,     "Hasselblad"},
    {LIBRAW_CAMERAMAKER_HTC,            "HTC"},
    {LIBRAW_CAMERAMAKER_I_Mobile,       "I_Mobile"},
    {LIBRAW_CAMERAMAKER_Imacon,         "Imacon"},
    {LIBRAW_CAMERAMAKER_ISG,            "ISG"},
    {LIBRAW_CAMERAMAKER_JK_Imaging,     "JK Imaging"}, // Kodak
    {LIBRAW_CAMERAMAKER_Leaf,           "Leaf"},
    {LIBRAW_CAMERAMAKER_Lenovo,         "Lenovo"},
    {LIBRAW_CAMERAMAKER_LG,             "LG"},
    {LIBRAW_CAMERAMAKER_Logitech,       "Logitech"},
    {LIBRAW_CAMERAMAKER_Matrix,         "Matrix"},
    {LIBRAW_CAMERAMAKER_Meizu,          "Meizu"},
    {LIBRAW_CAMERAMAKER_Micron,         "Micron"},
    {LIBRAW_CAMERAMAKER_NGM,            "NGM"},
    {LIBRAW_CAMERAMAKER_OmniVison,      "OmniVison"},
    {LIBRAW_CAMERAMAKER_Panasonic,      "Panasonic"},
    {LIBRAW_CAMERAMAKER_Photron,        "Photron"},
    {LIBRAW_CAMERAMAKER_Pixelink,       "Pixelink"},
    {LIBRAW_CAMERAMAKER_Polaroid,       "Polaroid"},
    {LIBRAW_CAMERAMAKER_Rollei,         "Rollei"},
    {LIBRAW_CAMERAMAKER_RoverShot,      "RoverShot"},
    {LIBRAW_CAMERAMAKER_SMaL,           "SMaL"},
    {LIBRAW_CAMERAMAKER_ST_Micro,       "ST Micro"},
    {LIBRAW_CAMERAMAKER_THL,            "THL"},
    {LIBRAW_CAMERAMAKER_Xiaomi,         "Xiaomi"},
    {LIBRAW_CAMERAMAKER_XIAOYI,         "Xiayi"},
    {LIBRAW_CAMERAMAKER_Yuneec,         "Yuneec"},
    {LIBRAW_CAMERAMAKER_DXO,            "DxO"},
    {LIBRAW_CAMERAMAKER_RED,            "Red"},
    {LIBRAW_CAMERAMAKER_PhotoControl,   "Photo Control"},
    {LIBRAW_CAMERAMAKER_Google,         "Google"},
    {LIBRAW_CAMERAMAKER_GoPro,          "GoPro"},
    {LIBRAW_CAMERAMAKER_Parrot,         "Parrot"},
    {LIBRAW_CAMERAMAKER_Zeiss,          "Zeiss"},
    {LIBRAW_CAMERAMAKER_OnePlus,        "OnePlus"},
    {LIBRAW_CAMERAMAKER_VIVO,           "Vivo"},
    {LIBRAW_CAMERAMAKER_HMD_Global,     "HMD Global"},
    {LIBRAW_CAMERAMAKER_HUAWEI,         "Huawei"},
    {LIBRAW_CAMERAMAKER_RaspberryPi,    "RaspberryPi"},
};
// clang-format on

int LibRaw::setMakeFromIndex(unsigned makei)
{
	if (makei <= LIBRAW_CAMERAMAKER_Unknown || makei >= LIBRAW_CAMERAMAKER_TheLastOne) return 0;

	for (int i = 0; i < int(sizeof CorpTable / sizeof *CorpTable); i++)
		if ((unsigned)CorpTable[i].CorpId == makei)
		{
			strcpy(normalized_make, CorpTable[i].CorpName);
			maker_index = makei;
			return 1;
		}
	return 0;
}

const char *LibRaw::cameramakeridx2maker(unsigned maker)
{
    for (int i = 0; i < int(sizeof CorpTable / sizeof *CorpTable); i++)
        if((unsigned)CorpTable[i].CorpId == maker)
            return CorpTable[i].CorpName;
    return 0;
}


#ifdef LIBRAW_OLD_VIDEO_SUPPORT
void LibRaw::fixupArri()
{
    struct alist_t
    {
        const char *a_model;
        const char *a_software;
        ushort a_width,a_height;
        int a_black;
        unsigned a_filters;
        float a_aspect;
    }
    alist[] =
    {
        {"ALEXA65", "Alexa65  XT", 6560 ,3100, 256,0x49494949,1.f},

        {"ALEXALF", "Alexa LF Plus W", 3840 ,2160, 256,0x49494949,1.0f },
        {"ALEXALF", "Alexa LF Plus W", 4448 ,1856, 256,0x49494949,0.75f },
        {"ALEXALF", "Alexa LF Plus W", 4448 ,3096, 256,0x49494949,1.f },

        {"ALEXA", "Alexa Plus 4:3 SXT", 2880 ,1620, 256,0x61616161,.75f},
        {"ALEXA", "Alexa Plus 4:3 SXT", 3168 ,1782, 256,0x61616161,0.75f},
        {"ALEXA", "Alexa Plus 4:3 SXT", 3424 ,2202, 256,0x61616161,1.f},
        {"ALEXA", "Alexa Plus 4:3 SXT", 2592 ,2160, 256,0x61616161,1.12f},

        {"ALEXA", "Alexa Plus 4:3 XT", 2592 ,2160, 256,0x61616161,1.12f},
        {"ALEXA", "Alexa Plus 4:3 XT", 2880 ,2160, 256,0x61616161,1.f},
        {"ALEXA", "Alexa Plus 4:3 XT", 2880 ,1620, 256,0x61616161,0.75f},
        {"ALEXA", "Alexa Plus 4:3 XT", 3424 ,2202, 256,0x61616161,1.f},
    };
    for(int i = 0; i < int(sizeof(alist)/sizeof(alist[0])); i++)
        if(!strncasecmp(model,alist[i].a_model,strlen(alist[i].a_model)) && software
            && !strncasecmp(software,alist[i].a_software,strlen(alist[i].a_software))
            && width == alist[i].a_width && height == alist[i].a_height)
        {
            filters = alist[i].a_filters;
            black = alist[i].a_black;
            pixel_aspect = alist[i].a_aspect;
            strcpy(model,software);
            software[0]=0;
            return;
        }
}
#endif
/*
   Identify which camera created this file, and set global variables
   accordingly.
 */
void LibRaw::identify()
{
  // clang-format off
  static const ushort canon[][11] = {
      // raw_width, raw_height, left_margin, top_margin,
      // width_decrement, height_decrement,
      // mask01, mask03, mask11, mask13,
      // CFA_filters.
	  { 1944, 1416, 0, 0, 48, 0 }, // 00 "PowerShot Pro90 IS"
	  { 2144, 1560, 4, 8, 52, 2, 0, 0, 0, 25 }, // 01 "PowerShot S30", "PowerShot G1"
	  { 2224, 1456, 48, 6, 0, 2 }, // 02 "EOS D30"
	  { 2376, 1728, 12, 6, 52, 2 }, // 03 "PowerShot G2", "PowerShot S40", "PowerShot G3", "PowerShot S45"
	  { 2672, 1968, 12, 6, 44, 2 }, // 04 "PowerShot G5", "PowerShot S50", "PowerShot S60"
	  { 3152, 2068, 64, 12, 0, 0, 16 }, // 05 "EOS D60", "EOS 10D", "EOS 300D"
	  { 3160, 2344, 44, 12, 4, 4 }, // 06 "PowerShot G6", "PowerShot S70"
	  { 3344, 2484, 4, 6, 52, 6 }, // 07 "PowerShot Pro1"
	  { 3516, 2328, 42, 14, 0, 0 }, // 08 "EOS 350D"
	  { 3596, 2360, 74, 12, 0, 0 }, // 09 "EOS-1D Mark II", "EOS 20D", "EOS-1D Mark II N", "EOS 30D"
	  { 3744, 2784, 52, 12, 8, 12 }, // 10 "PowerShot G11", "PowerShot S90", "PowerShot G12", "PowerShot S95"
	  { 3944, 2622, 30, 18, 6, 2 }, // 11 "EOS 40D"
	  { 3948, 2622, 42, 18, 0, 2 }, // 12 "EOS 400D", "EOS 1000D"
	  { 3984, 2622, 76, 20, 0, 2, 14 }, // 13 "EOS-1D Mark III"
	  { 4032, 2656, 112, 44, 10, 0 }, // 14 APS-C crop mode: "EOS 6D Mark II"??, "EOS RP"
	  { 4104, 3048, 48, 12, 24, 12 }, // 15 "PowerShot G9"
	  { 4116, 2178, 4, 2, 0, 0 },  // 16 ??
	  { 4152, 2772, 192, 12, 0, 0 }, // 17 "PowerShot SX1 IS"
	  { 4160, 3124, 104, 11, 8, 65 }, // 18 "PowerShot S100 (new)", "PowerShot S100V", "PowerShot G15", "PowerShot S110 (new)"
	  { 4176, 3062, 96, 17, 8, 0, 0, 16, 0, 7, 0x49 }, // 19 "PowerShot SX50 HS"
	  { 4192, 3062, 96, 17, 24, 0, 0, 16, 0, 0, 0x49 }, // 20 "PowerShot G16", "PowerShot S120"
	  { 4312, 2876, 22, 18, 0, 2 }, // 21 "EOS 450D"
	  { 4352, 2850, 144, 46, 0, 0 }, // 22 APS-C crop mode: "EOS R"
	  { 4352, 2874, 62, 18, 0, 0 }, // 23 "EOS 1100D"
	  { 4476, 2954, 90, 34, 0, 0 }, // 24 "EOS 5D"
	  { 4480, 3348, 12, 10, 36, 12, 0, 0, 0, 18, 0x49 }, // 25 "PowerShot G10"
	  { 4480, 3366, 80, 50, 0, 0 }, // 26 "PowerShot G1 X Mark II"
	  { 4496, 3366, 80, 50, 12, 0 }, // 27 "PowerShot G1 X"
	  { 4768, 3516, 96, 16, 0, 0, 0, 16 }, // 28 "PowerShot SX60 HS"
	  { 4832, 3204, 62, 26, 0, 0 }, // 29 "EOS 500D"
	  { 4832, 3228, 62, 51, 0, 0 }, // 30 "EOS 50D"
	  { 5108, 3349, 98, 13, 0, 0 }, // 31 "EOS-1Ds Mark II"
	  { 5120, 3318, 142, 45, 62, 0 }, // 32  "EOS-1D Mark IV"
	  { 5280, 3528, 72, 52, 0, 0 }, // 33 "EOS M10", "EOS 650D", "EOS 700D", "EOS M", "EOS 100D", "EOS M2"
	  { 5344, 3516, 142, 51, 0, 0 }, // 34 "EOS 550D", "EOS 600D", "EOS 60D", "EOS 1200D", "EOS 1300D", "EOS 3000D"
	  { 5344, 3584, 126, 100, 0, 2 }, // 35 "EOS-1D X", "EOS-1D C"
	  { 5344, 3950, 98, 18, 0, 0, 0, 24, 0, 0 }, // 36 "PowerShot SX70 HS"
	  { 5360, 3516, 158, 51, 0, 0 }, // 37 "EOS 7D"
	  { 5568, 3708, 72, 38, 0, 0 }, // 38; "EOS 7D Mark II", "EOS 6D", "EOS 70D", "EOS-1D X MARK II"
	  { 5632, 3710, 96, 17, 0, 0, 0, 16, 0, 0, 0x49 }, // 39 "PowerShot G7 X", "PowerShot G3 X", "PowerShot G9 X", "PowerShot G5 X", "PowerShot G7 X Mark II", "PowerShot G9 X Mark II"
	  { 5712, 3774, 62, 20, 10, 2 }, // 40 "EOS-1Ds Mark III"
	  { 5792, 3804, 158, 51, 0, 0 }, // 41 "EOS 5D Mark II"
	  { 5920, 3950, 122, 80, 2, 0 }, // 42 "EOS 5D Mark III"
	  { 6096, 4051, 76, 35, 0, 0 }, // 43 "EOS 1500D"
	  { 6096, 4056, 72, 34, 0, 0 }, // 44 "EOS M3", "EOS 760D", "EOS 750D"
	  { 6288, 4056, 264, 36, 0, 0 }, // 45 "EOS M5", "EOS M100", "EOS M6", "PowerShot G1 X Mark III", "EOS 80D", "EOS 800D", "EOS 77D", "EOS 200D", "EOS 250D", "EOS M50"
	  { 6384, 4224, 120, 44, 0, 0 }, // 46 "EOS 6D Mark II", "EOS RP"
	  { 6880, 4544, 136, 42, 0, 0 }, // 47 "EOS 5D Mark IV"
	  { 6888, 4546, 146, 48, 0, 0 }, // 48 "EOS R"
	  { 7128, 4732, 144, 72, 0, 0 }, // 49 "EOS M6 II", "EOS 90D"
	  { 8896, 5920, 160, 64, 0, 0 }, // 50 "EOS 5DS", "EOS 5DS R"
      { 6192, 4152, 160, 120, 0, 0}, // EOS R3
	  { 6192, 4060, 168, 52, 24, 8, 16,48,32,0,} // EOS R10
  };

  static const libraw_custom_camera_t const_table[] = {
	  { 786432, 1024, 768, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-080C" },
	  { 1447680, 1392, 1040, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-145C" },
	  { 1920000, 1600, 1200, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-201C" },
	  { 5067304, 2588, 1958, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-510C" },
	  { 5067316, 2588, 1958, 0, 0, 0, 0, 0, 0x94, 0, 0, "AVT", "F-510C", 12 },
	  { 10134608, 2588, 1958, 0, 0, 0, 0, 9, 0x94, 0, 0, "AVT", "F-510C" },
	  { 10134620, 2588, 1958, 0, 0, 0, 0, 9, 0x94, 0, 0, "AVT", "F-510C", 12 },
	  { 16157136, 3272, 2469, 0, 0, 0, 0, 9, 0x94, 0, 0, "AVT", "F-810C" },
      { 3995136, 1632, 1224, 0, 0, 0, 0, 8, 0x61, 0, 1, "AgfaPhoto", "DC-833m" },
      { 15980544, 3264, 2448, 0, 0, 0, 0, 8, 0x61, 0, 1, "AgfaPhoto", "DC-833m" },
	  { 9631728, 2532, 1902, 0, 0, 0, 0, 96, 0x61, 0, 0, "Alcatel", "5035D" },
	  { 31850496, 4608, 3456, 0, 0, 0, 0, 0, 0x94, 0, 0, "GITUP", "GIT2 4:3" },
	  { 23887872, 4608, 2592, 0, 0, 0, 0, 0, 0x94, 0, 0, "GITUP", "GIT2 16:9" },
	  { 32257024, 4624, 3488, 8, 2, 16, 2, 0, 0x94, 0, 0, "GITUP", "GIT2P 4:3" },
	  { 24192768, 4624, 2616, 8, 2, 16, 2, 0, 0x94, 0, 0, "GITUP", "GIT2P 16:9" },
	  { 18016000, 4000, 2252, 0, 0, 0, 0, 0, 0x94, 0, 0, "GITUP", "G3DUO 16:9" },
	  //          {24000000, 4000, 3000, 0, 0, 0, 0, 0, 0x94, 0, 0, "GITUP",
      //          "G3DUO 4:3"}, // Conflict w/ Samsung WB550

      //   Android Raw dumps id start
      //   File Size in bytes Horizontal Res Vertical Flag then bayer order eg
      //   0x16 bbgr 0x94 rggb
	  { 1540857, 2688, 1520, 0, 0, 0, 0, 1, 0x61, 0, 0, "Samsung", "S3" },
	  { 2658304, 1212, 1096, 0, 0, 0, 0, 1, 0x16, 0, 0, "LG", "G3FrontMipi" },
	  { 2842624, 1296, 1096, 0, 0, 0, 0, 1, 0x16, 0, 0, "LG", "G3FrontQCOM" },
	  { 2969600, 1976, 1200, 0, 0, 0, 0, 1, 0x16, 0, 0, "Xiaomi", "MI3wMipi" },
	  { 3170304, 1976, 1200, 0, 0, 0, 0, 1, 0x16, 0, 0, "Xiaomi", "MI3wQCOM" },
	  { 3763584, 1584, 1184, 0, 0, 0, 0, 96, 0x61, 0, 0, "I_Mobile", "I_StyleQ6" },
	  { 5107712, 2688, 1520, 0, 0, 0, 0, 1, 0x61, 0, 0, "OmniVisi", "UltraPixel1" },
	  { 5382640, 2688, 1520, 0, 0, 0, 0, 1, 0x61, 0, 0, "OmniVisi", "UltraPixel2" },
	  { 5664912, 2688, 1520, 0, 0, 0, 0, 1, 0x61, 0, 0, "OmniVisi", "4688" },
	  { 5664912, 2688, 1520, 0, 0, 0, 0, 1, 0x61, 0, 0, "OmniVisi", "4688" },
	  { 5364240, 2688, 1520, 0, 0, 0, 0, 1, 0x61, 0, 0, "OmniVisi", "4688" },
	  { 6299648, 2592, 1944, 0, 0, 0, 0, 1, 0x16, 0, 0, "OmniVisi", "OV5648" },
	  { 6721536, 2592, 1944, 0, 0, 0, 0, 0, 0x16, 0, 0, "OmniVisi", "OV56482" },
	  { 6746112, 2592, 1944, 0, 0, 0, 0, 0, 0x16, 0, 0, "HTC", "OneSV" },
	  { 9631728, 2532, 1902, 0, 0, 0, 0, 96, 0x61, 0, 0, "Sony", "5mp" },
	  { 9830400, 2560, 1920, 0, 0, 0, 0, 96, 0x61, 0, 0, "NGM", "ForwardArt" },
	  { 10186752, 3264, 2448, 0, 0, 0, 0, 1, 0x94, 0, 0, "Sony", "IMX219-mipi 8mp" },
	  { 10223360, 2608, 1944, 0, 0, 0, 0, 96, 0x16, 0, 0, "Sony", "IMX" },
	  { 10782464, 3282, 2448, 0, 0, 0, 0, 0, 0x16, 0, 0, "HTC", "MyTouch4GSlide" },
	  { 10788864, 3282, 2448, 0, 0, 0, 0, 0, 0x16, 0, 0, "Xperia", "L" },
	  { 15967488, 3264, 2446, 0, 0, 0, 0, 96, 0x16, 0, 0, "OmniVison", "OV8850" },
	  { 16224256, 4208, 3082, 0, 0, 0, 0, 1, 0x16, 0, 0, "LG", "G3MipiL" },
	  { 16424960, 4208, 3120, 0, 0, 0, 0, 1, 0x16, 0, 0, "IMX135", "MipiL" },
	  { 17326080, 4164, 3120, 0, 0, 0, 0, 1, 0x16, 0, 0, "LG", "G3LQCom" },
	  { 17522688, 4212, 3120, 0, 0, 0, 0, 0, 0x16, 0, 0, "Sony", "IMX135-QCOM" },
	  { 19906560, 4608, 3456, 0, 0, 0, 0, 1, 0x16, 0, 0, "Gione", "E7mipi" },
	  { 19976192, 5312, 2988, 0, 0, 0, 0, 1, 0x16, 0, 0, "LG", "G4" },
	  { 20389888, 4632, 3480, 0, 0, 0, 0, 1, 0x16, 0, 0, "Xiaomi", "RedmiNote3Pro" },
	  { 20500480, 4656, 3496, 0, 0, 0, 0, 1, 0x94, 0, 0, "Sony", "IMX298-mipi 16mp" },
	  { 21233664, 4608, 3456, 0, 0, 0, 0, 1, 0x16, 0, 0, "Gione", "E7qcom" },
	  { 26023936, 4192, 3104, 0, 0, 0, 0, 96, 0x94, 0, 0, "THL", "5000" },
	  { 26257920, 4208, 3120, 0, 0, 0, 0, 96, 0x94, 0, 0, "Sony", "IMX214" },
	  { 26357760, 4224, 3120, 0, 0, 0, 0, 96, 0x61, 0, 0, "OV", "13860" },
	  { 41312256, 5248, 3936, 0, 0, 0, 0, 96, 0x61, 0, 0, "Meizu", "MX4" },
	  { 42923008, 5344, 4016, 0, 0, 0, 0, 96, 0x61, 0, 0, "Sony", "IMX230" },
      //   Android Raw dumps id end
	  { 20137344, 3664, 2748, 0, 0, 0, 0, 0x40, 0x49, 0, 0, "Aptina", "MT9J003", 0xffff },
	  { 2868726, 1384, 1036, 0, 0, 0, 0, 64, 0x49, 0, 8, "Baumer", "TXG14", 1078 },
	  { 6553440, 2664, 1968, 4, 4, 44, 4, 40, 0x94, 0, 2, "Canon", "PowerShot A460" }, // chdk hack
	  { 9243240, 3152, 2346, 12, 7, 44, 13, 40, 0x49, 0, 2, "Canon", "PowerShot A470" }, // chdk hack
	  { 6653280, 2672, 1992, 10, 6, 42, 2, 40, 0x94, 0, 2, "Canon", "PowerShot A530" }, // chdk hack
	  { 6573120, 2672, 1968, 12, 8, 44, 0, 40, 0x94, 0, 2, "Canon", "PowerShot A610" }, // chdk hack
	  { 9219600, 3152, 2340, 36, 12, 4, 0, 40, 0x94, 0, 2, "Canon", "PowerShot A620" }, // chdk hack
	  { 10383120, 3344, 2484, 12, 6, 44, 6, 40, 0x94, 0, 2, "Canon", "PowerShot A630" }, // chdk hack
	  { 12945240, 3736, 2772, 12, 6, 52, 6, 40, 0x94, 0, 2, "Canon", "PowerShot A640" }, // chdk hack
	  { 15636240, 4104, 3048, 48, 12, 24, 12, 40, 0x94, 0, 2, "Canon", "PowerShot A650 IS" }, // chdk hack
	  { 10341600, 3336, 2480, 6, 5, 32, 3, 40, 0x94, 0, 2, "Canon", "PowerShot A720 IS" }, // chdk hack
	  { 24724224, 4704, 3504, 8, 16, 56, 8, 40, 0x49, 0, 2, "Canon", "PowerShot A3300 IS" }, // chdk hack
	  { 18763488, 4104, 3048, 10, 22, 82, 22, 8, 0x49, 0, 0, "Canon", "PowerShot D10" }, // ? chdk hack ?
	  { 19493760, 4160, 3124, 104, 12, 8, 66, 40, 0x49, 0, 2,  "Canon", "PowerShot S100" }, // chdk hack CRW
	  { 7710960, 2888, 2136, 44, 8, 4, 0, 40, 0x94, 0, 2, "Canon", "PowerShot S3 IS" }, // chdk hack
	  { 5298000, 2400, 1766, 12, 12, 44, 2, 40, 0x94, 0, 2, "Canon", "PowerShot SD300" }, // chdk hack
	  { 18653760, 4080, 3048, 24, 12, 24, 12, 40, 0x94, 0, 2, "Canon", "PowerShot SX20 IS" }, // chdk hack
	  { 21936096, 4464, 3276, 25, 10, 73, 12, 40, 0x16, 0, 2, "Canon", "PowerShot SX30 IS" }, // chdk hack
	  { 19167840, 4176, 3060, 96, 16, 8, 0,  40, 0x94, 0, 2, "Canon", "PowerShot SX40 HS" }, // chdk hack CR2
	  { 15467760, 3720, 2772, 6, 12, 30, 0, 40, 0x94, 0, 2, "Canon", "PowerShot SX110 IS" }, // chdk hack
	  { 15534576, 3728, 2778, 12, 9, 44, 9, 40, 0x94, 0, 2, "Canon", "PowerShot SX120 IS" }, // chdk hack
	  { 19131120, 4168, 3060, 92, 16, 4, 1, 40, 0x94, 0, 2, "Canon", "PowerShot SX220 HS" }, // chdk hack
	  { 31663200, 5344, 3950, 96, 18, 0, 0, 40, 0x94, 0, 2, "Canon", "PowerShot SX710 HS" }, // chdk hack
	  { 30858240, 5248, 3920, 8, 16, 56, 16, 40, 0x94, 0, 2, "Canon", "IXUS 160" }, // chdk hack
	  { 1976352, 1632, 1211, 0, 2, 0, 1, 0, 0x94, 0, 1, "Casio", "QV-2000UX" },
	  { 3217760, 2080, 1547, 0, 0, 10, 1, 0, 0x94, 0, 1, "Casio", "QV-3*00EX" },
	  { 6218368, 2585, 1924, 0, 0, 9, 0, 0, 0x94, 0, 1, "Casio", "QV-5700" },
	  { 7816704, 2867, 2181, 0, 0, 34, 36, 0, 0x16, 0, 1, "Casio", "EX-Z60" },
	  { 2937856, 1621, 1208, 0, 0, 1, 0, 0, 0x94, 7, 13, "Casio", "EX-S20" },
	  { 4948608, 2090, 1578, 0, 0, 32, 34, 0, 0x94, 7, 1, "Casio", "EX-S100" },
	  { 6054400, 2346, 1720, 2, 0, 32, 0, 0, 0x94, 7, 1, "Casio", "QV-R41" },
	  { 7426656, 2568, 1928, 0, 0, 0, 0, 0, 0x94, 0, 1, "Casio", "EX-P505" },
	  { 7530816, 2602, 1929, 0, 0, 22, 0, 0, 0x94, 7, 1, "Casio", "QV-R51" },
	  { 7542528, 2602, 1932, 0, 0, 32, 0, 0, 0x94, 7, 1, "Casio", "EX-Z50" },
	  { 7562048, 2602, 1937, 0, 0, 25, 0, 0, 0x16, 7, 1, "Casio", "EX-Z500" },
	  { 7753344, 2602, 1986, 0, 0, 32, 26, 0, 0x94, 7, 1, "Casio", "EX-Z55" },
	  { 9313536, 2858, 2172, 0, 0, 14, 30, 0, 0x94, 7, 1, "Casio", "EX-P600" },
	  { 10834368, 3114, 2319, 0, 0, 27, 0, 0, 0x94, 0, 1, "Casio", "EX-Z750" },
	  { 10843712, 3114, 2321, 0, 0, 25, 0, 0, 0x94, 0, 1, "Casio", "EX-Z75" },
	  { 10979200, 3114, 2350, 0, 0, 32, 32, 0, 0x94, 7, 1, "Casio", "EX-P700" },
	  { 12310144, 3285, 2498, 0, 0, 6, 30, 0, 0x94, 0, 1, "Casio", "EX-Z850" },
	  { 12489984, 3328, 2502, 0, 0, 47, 35, 0, 0x94, 0, 1, "Casio", "EX-Z8" },
	  { 15499264, 3754, 2752, 0, 0, 82, 0, 0, 0x94, 0, 1, "Casio", "EX-Z1050" },
	  { 18702336, 4096, 3044, 0, 0, 24, 0, 80, 0x94, 7, 1, "Casio", "EX-ZR100" },
	  { 7684000, 2260, 1700, 0, 0, 0, 0, 13, 0x94, 0, 1, "Casio", "QV-4000" },
	  { 787456, 1024, 769, 0, 1, 0, 0, 0, 0x49, 0, 0, "Creative", "PC-CAM 600" },
	  { 28829184, 4384, 3288, 0, 0, 0, 0, 36, 0x61, 0, 0, "DJI" },
	  { 15151104, 4608, 3288, 0, 0, 0, 0, 0, 0x94, 0, 0, "Matrix" },
	  { 3840000, 1600, 1200, 0, 0, 0, 0, 65, 0x49, 0, 0, "Foculus", "531C" },
	  { 307200, 640, 480, 0, 0, 0, 0, 0, 0x94, 0, 0, "Generic" },
	  { 62464, 256, 244, 1, 1, 6, 1, 0, 0x8d, 0, 0, "Kodak", "DC20" },
	  { 124928, 512, 244, 1, 1, 10, 1, 0, 0x8d, 0, 0, "Kodak", "DC20" },
	  { 1652736, 1536, 1076, 0, 52, 0, 0, 0, 0x61, 0, 0, "Kodak", "DCS200" },
	  { 4159302, 2338, 1779, 1, 33, 1, 2, 0, 0x94, 0, 0, "Kodak", "C330" },
	  { 4162462, 2338, 1779, 1, 33, 1, 2, 0, 0x94, 0, 0, "Kodak", "C330", 3160 },
	  { 2247168, 1232, 912, 0, 0, 16, 0, 0, 0x00, 0, 0, "Kodak", "C330" },
	  { 3370752, 1232, 912, 0, 0, 16, 0, 0, 0x00, 0, 0, "Kodak", "C330" },
	  { 6163328, 2864, 2152, 0, 0, 0, 0, 0, 0x94, 0, 0, "Kodak", "C603" },
	  { 6166488, 2864, 2152, 0, 0, 0, 0, 0, 0x94, 0, 0, "Kodak", "C603", 3160 },
	  { 460800, 640, 480, 0, 0, 0, 0, 0, 0x00, 0, 0, "Kodak", "C603" },
	  { 9116448, 2848, 2134, 0, 0, 0, 0, 0, 0x00, 0, 0, "Kodak", "C603" },
	  { 12241200, 4040, 3030, 2, 0, 0, 13, 0, 0x49, 0, 0, "Kodak", "12MP" },
	  { 12272756, 4040, 3030, 2, 0, 0, 13, 0, 0x49, 0, 0, "Kodak", "12MP", 31556 },
	  { 18000000, 4000, 3000, 0, 0, 0, 0, 0, 0x00, 0, 0, "Kodak", "12MP" },
	  { 614400, 640, 480, 0, 3, 0, 0, 64, 0x94, 0, 0, "Kodak", "KAI-0340" },
	  { 15360000, 3200, 2400, 0, 0, 0, 0, 96, 0x16, 0, 0, "Lenovo", "A820" },
	  { 3884928, 1608, 1207, 0, 0, 0, 0, 96, 0x16, 0, 0, "Micron", "2010", 3212 },
	  { 1138688, 1534, 986, 0, 0, 0, 0, 0, 0x61, 0, 0, "Minolta", "RD175", 513 },
	  { 1581060, 1305, 969, 0, 0, 18, 6, 6, 0x1e, 4, 1, "Nikon", "E900" }, // "diag raw" hack
	  { 2465792, 1638, 1204, 0, 0, 22, 1, 6, 0x4b, 5, 1, "Nikon", "E950" }, // "diag raw" hack; possibly also Nikon E700, E800, E775;
	                                                                        // Olympus C-2020Z
	  { 2940928, 1616, 1213, 0, 0, 0, 7, 30, 0x94, 0, 1, "Nikon", "E2100" }, // "diag raw" hack; also Nikon E2500
	  { 4771840, 2064, 1541, 0, 0, 0, 1, 6, 0xe1, 0, 1, "Nikon", "E990" }, // "diag raw" hack; possibly also Nikon E880, E885, E995;
	                                                                       // Olympus C-3030Z
	  { 4775936, 2064, 1542, 0, 0, 0, 0, 30, 0x94, 0, 1, "Nikon", "E3700" }, // "diag raw" hack; Nikon E3100, E3200, E3500;
	                                                                         // Pentax "Optio 33WR"; possibly also Olympus C-740UZ
	  { 5865472, 2288, 1709, 0, 0, 0, 1, 6, 0xb4, 0, 1, "Nikon", "E4500" }, // "diag raw" hack; possibly also Olympus C-4040Z
	  { 5869568, 2288, 1710, 0, 0, 0, 0, 6, 0x16, 0, 1, "Nikon", "E4300" }, // "diag raw" hack; also Minolta "DiMAGE Z2"
	  { 7438336, 2576, 1925, 0, 0, 0, 1, 6, 0xb4, 0, 1, "Nikon", "E5000" }, // also Nikon E5700
	  { 8998912, 2832, 2118, 0, 0, 0, 0, 30, 0x94, 7, 1, "Nikon", "COOLPIX S6" }, // "diag raw" hack
	  { 5939200, 2304, 1718, 0, 0, 0, 0, 30, 0x16, 0, 0, "Olympus", "C-770UZ" }, // possibly also Olympus C-4100Z, C-765UZ
	  { 3178560, 2064, 1540, 0, 0, 0, 0, 0, 0x94, 0, 1, "Pentax", "Optio S V1.01" },
	  { 4841984, 2090, 1544, 0, 0, 22, 0, 0, 0x94, 7, 1, "Pentax", "Optio S" },
	  { 6114240, 2346, 1737, 0, 0, 22, 0, 0, 0x94, 7, 1, "Pentax", "Optio S4" },
	  { 10702848, 3072, 2322, 0, 0, 0, 21, 30, 0x94, 0, 1, "Pentax", "Optio 750Z" },
	  { 4147200, 1920, 1080, 0, 0, 0, 0, 0, 0x49, 0, 0, "Photron", "BC2-HD" },
	  { 4151666, 1920, 1080, 0, 0, 0, 0, 0, 0x49, 0, 0, "Photron", "BC2-HD", 8 },
	  { 13248000, 2208, 3000, 0, 0, 0, 0, 13, 0x61, 0, 0, "Pixelink", "A782" },
	  { 6291456, 2048, 1536, 0, 0, 0, 0, 96, 0x61, 0, 0, "RoverShot", "3320AF" },
	  { 311696, 644, 484, 0, 0, 0, 0, 0, 0x16, 0, 8, "ST Micro", "STV680 VGA" },
	  { 16098048, 3288, 2448, 0, 0, 24, 0, 9, 0x94, 0, 1, "Samsung", "S85" }, // hack
	  { 16215552, 3312, 2448, 0, 0, 48, 0, 9, 0x94, 0, 1, "Samsung", "S85" }, // hack
	  { 20487168, 3648, 2808, 0, 0, 0, 0, 13, 0x94, 5, 1, "Samsung", "WB550" },
	  { 24000000, 4000, 3000, 0, 0, 0, 0, 13, 0x94, 5, 1, "Samsung", "WB550" },
	  { 12582980, 3072, 2048, 0, 0, 0, 0, 33, 0x61, 0, 0, "Sinar", "", 68 }, // Sinarback 23; same res. as Leaf Volare & Cantare
	  { 33292868, 4080, 4080, 0, 0, 0, 0, 33, 0x61, 0, 0, "Sinar", "", 68 }, // Sinarback 44
	  { 44390468, 4080, 5440, 0, 0, 0, 0, 33, 0x61, 0, 0, "Sinar", "", 68 }, // Sinarback 54
	  { 1409024, 1376, 1024, 0, 0, 1, 0, 0, 0x49, 0, 0, "Sony", "XCD-SX910CR" },
	  { 2818048, 1376, 1024, 0, 0, 1, 0, 97, 0x49, 0, 0, "Sony", "XCD-SX910CR" },
  };

  libraw_custom_camera_t
      table[64 + sizeof(const_table) / sizeof(const_table[0])];


  // clang-format on

  char head[64] = {0}, *cp;
  int hlen, fsize, flen, zero_fsize = 1, i, c;
  INT64 fsize64;
  struct jhead jh;

  unsigned camera_count =
      parse_custom_cameras(64, table, imgdata.rawparams.custom_camera_strings);
  for (int q = 0; q < int(sizeof(const_table) / sizeof(const_table[0])); q++)
    memmove(&table[q + camera_count], &const_table[q], sizeof(const_table[0]));
  camera_count += sizeof(const_table) / sizeof(const_table[0]);

  tiff_flip = flip = filters = UINT_MAX; /* unknown */
  raw_height = raw_width = fuji_width = fuji_layout = cr2_slice[0] = 0;
  maximum = height = width = top_margin = left_margin = 0;
  cdesc[0] = desc[0] = artist[0] = make[0] = model[0] = model2[0] = 0;
  iso_speed = shutter = aperture = focal_len = 0;
  unique_id = 0ULL;
  tiff_nifds = 0;
  is_NikonTransfer = 0;
  is_Olympus = 0;
  OlympusDNG_SubDirOffsetValid = 0;
  is_Sony = 0;
  is_pana_raw = 0;
  maker_index = LIBRAW_CAMERAMAKER_Unknown;
  FujiCropMode = 0;
  is_PentaxRicohMakernotes = 0;
  normalized_model[0] = 0;
  normalized_make[0] = 0;
  CM_found = 0;
  memset(tiff_ifd, 0, sizeof tiff_ifd);
  libraw_internal_data.unpacker_data.crx_track_selected = -1;
  libraw_internal_data.unpacker_data.crx_track_count = -1;
  libraw_internal_data.unpacker_data.CR3_CTMDtag = 0;
  imHassy.nIFD_CM[0] = imHassy.nIFD_CM[1] = -1;
  imKodak.ISOCalibrationGain = 1.0f;
  imCommon.CameraTemperature = imCommon.SensorTemperature =
      imCommon.SensorTemperature2 = imCommon.LensTemperature =
          imCommon.AmbientTemperature = imCommon.BatteryTemperature =
              imCommon.exifAmbientTemperature = -1000.0f;

  libraw_internal_data.unpacker_data.ifd0_offset = -1LL;

  imgdata.color.ExifColorSpace = LIBRAW_COLORSPACE_Unknown;
  for (i = 0; i < LIBRAW_IFD_MAXCOUNT; i++)
  {
    tiff_ifd[i].dng_color[0].illuminant = tiff_ifd[i].dng_color[1].illuminant =
        0xffff;
    for (int q = 0; q < 4; q++)
      tiff_ifd[i].dng_levels.analogbalance[q] = 1.0f;
  }

  memset(gpsdata, 0, sizeof gpsdata);
  memset(cblack, 0, sizeof cblack);
  memset(white, 0, sizeof white);
  memset(mask, 0, sizeof mask);
  thumb_offset = thumb_length = thumb_width = thumb_height = 0;
  load_raw = 0;
  thumb_format = LIBRAW_INTERNAL_THUMBNAIL_JPEG; // default to JPEG
  data_offset = meta_offset = meta_length = tiff_bps = tiff_compress = 0;
  kodak_cbpp = zero_after_ff = dng_version = load_flags = 0;
  timestamp = shot_order = tiff_samples = black = is_foveon = 0;
  mix_green = profile_length = data_error = zero_is_bad = 0;
  pixel_aspect = is_raw = raw_color = 1;
  tile_width = tile_length = 0;
  metadata_blocks = 0;

  for (i = 0; i < 4; i++)
  {
    cam_mul[i] = i == 1;
    pre_mul[i] = i < 3;
    FORC3 cmatrix[c][i] = 0;
    FORC3 rgb_cam[c][i] = c == i;
  }
  colors = 3;
  for (i = 0; i < 0x10000; i++)
    curve[i] = i;

  order = get2();
  hlen = get4();
  fseek(ifp, 0, SEEK_SET);

  if (fread(head, 1, 64, ifp) < 64)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  libraw_internal_data.unpacker_data.lenRAFData =
      libraw_internal_data.unpacker_data.posRAFData = 0;

  fseek(ifp, 0, SEEK_END);
  fsize64 = ftell(ifp);
  if(fsize64 > LIBRAW_MAX_NONDNG_RAW_FILE_SIZE && fsize64 > LIBRAW_MAX_DNG_RAW_FILE_SIZE)
      throw LIBRAW_EXCEPTION_TOOBIG;

  flen = fsize = ftell(ifp);
  if ((cp = (char *)memmem(head, 32, (char *)"MMMM", 4)) ||
      (cp = (char *)memmem(head, 32, (char *)"IIII", 4)))
  {
    parse_phase_one(cp - head);
    if (cp - head && parse_tiff(0))
      apply_tiff();
  }
  else if (order == 0x4949 || order == 0x4d4d)
  {
    if (!memcmp(head + 6, "HEAPCCDR", 8))
    {
      data_offset = hlen;
      parse_ciff(hlen, flen - hlen, 0);
      load_raw = &LibRaw::canon_load_raw;
    }
    else if (parse_tiff(0))
      apply_tiff();
  }
  else if (!memcmp(head, "\xff\xd8\xff\xe1", 4) && !memcmp(head + 6, "Exif", 4))
  {
    fseek(ifp, 4, SEEK_SET);
    data_offset = 4 + get2();
    fseek(ifp, data_offset, SEEK_SET);
    if (fgetc(ifp) != 0xff)
      parse_tiff(12);
    thumb_offset = 0;
  }
  else if (!memcmp(head + 25, "ARECOYK", 7)) // 'KYOCERA' right-to-left
  {
    strcpy(make, "Contax");
    strcpy(model, "N Digital");
    parse_kyocera();
  }
  else if (!strcmp(head, "PXN"))
  {
    strcpy(make, "Logitech");
    strcpy(model, "Fotoman Pixtura");
  }
  else if (!strcmp(head, "qktk"))
  {
    strcpy(make, "Apple");
    strcpy(model, "QuickTake 100");
    load_raw = &LibRaw::quicktake_100_load_raw;
  }
  else if (!strcmp(head, "qktn"))
  {
    strcpy(make, "Apple");
    strcpy(model, "QuickTake 150");
    load_raw = &LibRaw::kodak_radc_load_raw;
  }
  else if (!memcmp(head, "FUJIFILM", 8))
  {
    memcpy(imFuji.SerialSignature, head + 0x10, 0x0c);
    imFuji.SerialSignature[0x0c] = 0;
    memcpy(imFuji.SensorID, imFuji.SerialSignature + 0x06, 0x04);
    imFuji.SensorID[0x04] = 0;
    strncpy(model, head + 0x1c, 0x20);
    model[0x20] = 0;
    c = 11;
    while (imFuji.SerialSignature[c] > 0 && isdigit(imFuji.SerialSignature[c]) && (c>0))
      c--;
	if(c < 11)
		unique_id = (unsigned long long)atoi(imFuji.SerialSignature+c+1);
    memcpy(imFuji.RAFVersion, head + 0x3c, 4);
    imFuji.RAFVersion[4] = 0;
    fseek(ifp, 84, SEEK_SET);
    thumb_offset = get4();
    thumb_length = get4();
    fseek(ifp, 92, SEEK_SET);
    parse_fuji(get4());
    if (thumb_offset > 120)
    {
      fseek(ifp, 120, SEEK_SET);
      is_raw += (i = get4()) ? 1 : 0;
      if (is_raw == 2 && shot_select)
        parse_fuji(i);
    }
    load_raw = &LibRaw::unpacked_load_raw;
    fseek(ifp, 100 + 28 * (shot_select > 0), SEEK_SET);
    parse_tiff(data_offset = get4());
    parse_tiff(thumb_offset + 12);
    parse_fuji_thumbnail(thumb_offset);
    apply_tiff();
  }
  else if (!memcmp(head, "RIFF", 4))
  {
    fseek(ifp, 0, SEEK_SET);
    parse_riff(100);
  }
  else if (!memcmp(head + 4, "ftypqt   ", 9))
  {
    fseek(ifp, 0, SEEK_SET);
    parse_qt(fsize);
    is_raw = 0;
  }
  else if (!memcmp(head, "\0\001\0\001\0@", 6))
  {
    fseek(ifp, 6, SEEK_SET);
    fread(make, 1, 8, ifp);
    fread(model, 1, 8, ifp);
    fread(model2, 1, 16, ifp);
    data_offset = get2();
    get2();
    raw_width = get2();
    raw_height = get2();
    load_raw = &LibRaw::nokia_load_raw;
    filters = 0x61616161;
  }
  else if (!memcmp(head, "NOKIARAW", 8))
  {
    strcpy(make, "NOKIA");
    order = 0x4949;
    fseek(ifp, 300, SEEK_SET);
    data_offset = get4();
    i = get4(); // bytes count
    width = get2();
    height = get2();

    // Data integrity check
    if (width < 1 || width > 16000 || height < 1 || height > 16000 ||
        i < (width * height) || i > (2 * width * height))
      throw LIBRAW_EXCEPTION_IO_CORRUPT;

    switch (tiff_bps = i * 8 / (width * height))
    {
    case 8:
      load_raw = &LibRaw::eight_bit_load_raw;
      break;
    case 10:
      load_raw = &LibRaw::nokia_load_raw;
      break;
    case 0:
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
      break;
    }
    raw_height = height + (top_margin = i / (width * tiff_bps / 8) - height);
    mask[0][3] = 1;
    filters = 0x61616161;
  }
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
  else if (!memcmp(head, "ARRI", 4))
  {
    order = 0x4949;
    fseek(ifp, 20, SEEK_SET);
    width = get4();
    height = get4();
    strcpy(make, "ARRI");
    fseek(ifp, 668, SEEK_SET);
    fread(model, 1, 64, ifp);
    model[63] = 0;
    fseek(ifp, 760, SEEK_SET);
    fread(software, 1, 64, ifp);
    if((unsigned char)software[0] == 0xff) software[0] = 0;
    software[63] = 0;
    data_offset = 4096;
    load_raw = &LibRaw::packed_load_raw;
    load_flags = 88;
    filters = 0x61616161;
    fixupArri();
  }
  else if (!memcmp(head, "XPDS", 4))
  {
    order = 0x4949;
    fseek(ifp, 0x800, SEEK_SET);
    fread(make, 1, 41, ifp);
    raw_height = get2();
    raw_width = get2();
    fseek(ifp, 56, SEEK_CUR);
    fread(model, 1, 30, ifp);
    data_offset = 0x10000;
    load_raw = &LibRaw::canon_rmf_load_raw;
    gamma_curve(0, 12.25, 1, 1023);
  }
  else if (!memcmp(head + 4, "RED1", 4))
  {
    strcpy(make, "Red");
    strcpy(model, "One");
    parse_redcine();
    load_raw = &LibRaw::redcine_load_raw;
    gamma_curve(1 / 2.4, 12.92, 1, 4095);
    filters = 0x49494949;
  }
#endif
  else if (!memcmp(head, "DSC-Image", 9))
    parse_rollei();
  else if (!memcmp(head, "PWAD", 4))
    parse_sinar_ia();
  else if (!memcmp(head, "\0MRM", 4))
    parse_minolta(0);
  else if (!memcmp(head, "FOVb", 4))
  {
    parse_x3f(); /* Does nothing if USE_X3FTOOLS is not defined */
  }
  else if (!memcmp(head, "CI", 2))
    parse_cine();
#ifdef USE_6BY9RPI
  else if (!memcmp(head, "BRCM", 4)) {
	fseek(ifp, 0, SEEK_SET);
	strcpy(make, "RaspberryPi");
	strcpy(model, "Pi");
	parse_raspberrypi();
	}
#endif
  else if (!memcmp(head + 4, "ftypcrx ", 8))
  {
    int err;
    unsigned long long szAtomList;
    short nesting = -1;
    short nTrack = -1;
    short TrackType;
    char AtomNameStack[129];
    strcpy(make, "Canon");

    szAtomList = ifp->size();
    err = parseCR3(0ULL, szAtomList, nesting, AtomNameStack, nTrack, TrackType);
    libraw_internal_data.unpacker_data.crx_track_count = nTrack;
    if ((err == 0 || err == -14) &&
        nTrack >= 0) // no error, or too deep nesting
      selectCRXTrack();
  }

  if (dng_version)
  {
      if (fsize64 > LIBRAW_MAX_DNG_RAW_FILE_SIZE)
          throw LIBRAW_EXCEPTION_TOOBIG;
  }
  else
  {
    if (fsize64 > LIBRAW_MAX_NONDNG_RAW_FILE_SIZE)
      throw LIBRAW_EXCEPTION_TOOBIG;
  }

  if (make[0] == 0)
    for (zero_fsize = i = 0; i < (int)camera_count; i++)
      if (fsize == (int)table[i].fsize)
      {
        strcpy(make, table[i].t_make);
        strcpy(model, table[i].t_model);
        flip = table[i].flags >> 2;
        zero_is_bad = table[i].flags & 2;
        data_offset = table[i].offset == 0xffff ? 0 : table[i].offset;
        raw_width = table[i].rw;
        raw_height = table[i].rh;
        left_margin = table[i].lm;
        top_margin = table[i].tm;
        width = raw_width - left_margin - table[i].rm;
        height = raw_height - top_margin - table[i].bm;
        filters = 0x1010101U * table[i].cf;
        colors = 4 - !((filters & filters >> 1) & 0x5555);
        load_flags = table[i].lf & 0xff;
        if (table[i].lf & 0x100) /* Monochrome sensor dump */
        {
          colors = 1;
          filters = 0;
        }
        switch (tiff_bps = (fsize - data_offset) * 8 / (raw_width * raw_height))
        {
        case 6:
          load_raw = &LibRaw::minolta_rd175_load_raw;
          ilm.CameraMount = LIBRAW_MOUNT_Minolta_A;
          break;
        case 8:
          load_raw = &LibRaw::eight_bit_load_raw;
          break;
        case 10:
          if ((fsize - data_offset) / raw_height * 3 >= raw_width * 4)
          {
            load_raw = &LibRaw::android_loose_load_raw;
            break;
          }
          else if (load_flags & 1)
          {
            load_raw = &LibRaw::android_tight_load_raw;
            break;
          }
        case 12:
          load_flags |= 128;
          load_raw = &LibRaw::packed_load_raw;
          break;
        case 16:
          order = 0x4949 | 0x404 * (load_flags & 1);
          tiff_bps -= load_flags >> 4;
          tiff_bps -= load_flags = load_flags >> 1 & 7;
          load_raw = table[i].offset == 0xffff
                         ? &LibRaw::unpacked_load_raw_reversed
                         : &LibRaw::unpacked_load_raw;
        }
        maximum = (1 << tiff_bps) - (1 << table[i].max);
        break;
      }
  if (zero_fsize)
    fsize = 0;
  if (make[0] == 0 && fsize64 < 25000000LL)
    parse_smal(0, flen);
  if (make[0] == 0)
  {
    parse_jpeg(0);
#ifdef USE_6BY9RPI
	if (!(strncmp(model, "ov", 2) && strncmp(model, "RP_", 3) && strncmp(model, "imx477", 6))) {
		//Assume that this isn't a raw unless the header can be found
		is_raw = 0;

		if (!strncasecmp(model, "RP_testc",8) 
            || !strncasecmp(model, "imx477", 6) //  from PyDNG
		    || !strncasecmp(model, "RP_imx477",9)) {
			const long offsets[] = {
				//IMX477 offsets
				3375104,  //2028x1080 12bit
				4751360,  //2028x1520 12bit
				18711040, //4056x3040 12bit
				1015808,  //1012x760 10bit
				-1        //Marker for end of table
			};
			int offset_idx;
			for (offset_idx=0; offsets[offset_idx]!=-1; offset_idx++) {
				if(!fseek (ifp, -offsets[offset_idx], SEEK_END) &&
				   fread (head, 1, 32, ifp) && !strncmp(head,"BRCM", 4)) {
					fseek(ifp, -32, SEEK_CUR);
					strcpy (make, "RaspberryPi");
                    strcpy(model, "RP_imx477"); // Force single model
					black = (offset_idx == 3) ? 64 : 256;
					parse_raspberrypi();
					break;
				}
			}
		}
		else if (!strncasecmp(model, "RP_imx", 6)) {
			const long offsets[] = {
				//IMX219 offsets
				10270208, //8MPix 3280x2464
				2678784,  //1920x1080
				2628608,  //1640x1232
				1963008,  //1640x922
				1233920,  //1280x720
				445440,   //640x480
				-1        //Marker for end of table
			};
			int offset_idx;
			for (offset_idx = 0; offsets[offset_idx] != -1; offset_idx++) {
				if (!fseek(ifp, -offsets[offset_idx], SEEK_END) &&
					fread(head, 1, 32, ifp) && !strncmp(head, "BRCM", 4)) {

					fseek(ifp, -32, SEEK_CUR);
					strcpy(make, "RaspberryPi");
                    black = 66;
					parse_raspberrypi();
					break;
				}
			}
		}
		else if (!strncasecmp(model, "RP_OV", 5) || !strncasecmp(model, "ov5647", 6)) {
			const long offsets[] = {
					6404096,  //5MPix 2592x1944
					2717696,  //1920x1080
					1625600,  //1296x972
					1233920,  //1296x730
					445440,   //640x480
					-1        //Marker for end of table
			};
			int offset_idx;
			for (offset_idx = 0; offsets[offset_idx] != -1; offset_idx++) {
				if (!fseek(ifp, -offsets[offset_idx], SEEK_END) &&
					fread(head, 1, 32, ifp) && !strncmp(head, "BRCM", 4)) {
					fseek(ifp, -32, SEEK_CUR);
					strcpy(make, "RaspberryPi");
                    strcpy(model, "ov5647"); // Force single model
                    width = raw_width;
					//Defaults
					raw_width = 2611;
					filters = 0x16161616;
                    black = 16;
					parse_raspberrypi();
					break;
				}
			}
	  }
	}// else is_raw = 0;
#else
    fseek(ifp, 0, SEEK_END);
    int sz = ftell(ifp);
    if (!strncmp(model, "RP_imx219", 9) && sz >= 0x9cb600 &&
        !fseek(ifp, -0x9cb600, SEEK_END) && fread(head, 1, 0x20, ifp) &&
        !strncmp(head, "BRCM", 4))
    {
      strcpy(make, "Broadcom");
      strcpy(model, "RPi IMX219");
      if (raw_height > raw_width)
        flip = 5;
      data_offset = ftell(ifp) + 0x8000 - 0x20;
      parse_broadcom();
      black = 66;
      maximum = 0x3ff;
      load_raw = &LibRaw::broadcom_load_raw;
      thumb_offset = 0;
      thumb_length = sz - 0x9cb600 - 1;
    }
    else if (!(strncmp(model, "ov5647", 6) && strncmp(model, "RP_OV5647", 9)) &&
             sz >= 0x61b800 && !fseek(ifp, -0x61b800, SEEK_END) &&
             fread(head, 1, 0x20, ifp) && !strncmp(head, "BRCM", 4))
    {
      strcpy(make, "Broadcom");
      if (!strncmp(model, "ov5647", 6))
        strcpy(model, "RPi OV5647 v.1");
      else
        strcpy(model, "RPi OV5647 v.2");
      if (raw_height > raw_width)
        flip = 5;
      data_offset = ftell(ifp) + 0x8000 - 0x20;
      parse_broadcom();
      black = 16;
      maximum = 0x3ff;
      load_raw = &LibRaw::broadcom_load_raw;
      thumb_offset = 0;
      thumb_length = sz - 0x61b800 - 1;
    }
    else
      is_raw = 0;
#endif
  }

  // make sure strings are terminated
  desc[511] = artist[63] = make[63] = model[63] = model2[63] = 0;

  for (i = 0; i < int(sizeof CorpTable / sizeof *CorpTable); i++)
  {
    if (strcasestr(make, CorpTable[i].CorpName))
    { /* Simplify company names */
      maker_index = CorpTable[i].CorpId;
      break;
    }
  }

  if (makeIs(LIBRAW_CAMERAMAKER_HMD_Global) && !strncasecmp(model, "Nokia", 5)) {
    maker_index = LIBRAW_CAMERAMAKER_Nokia;
  }  else if (makeIs(LIBRAW_CAMERAMAKER_JK_Imaging) && !strncasecmp(model, "Kodak", 5)) {
    maker_index = LIBRAW_CAMERAMAKER_Kodak;
  } else if (makeIs(LIBRAW_CAMERAMAKER_Ricoh) && !strncasecmp(model, "PENTAX", 6)) {
    maker_index = LIBRAW_CAMERAMAKER_Pentax;
  }

  for (i = 0; i < int(sizeof CorpTable / sizeof *CorpTable); i++) {
    if (maker_index == (unsigned)CorpTable[i].CorpId) {
      strcpy(make, CorpTable[i].CorpName);
      break;
    }
  }

  if ((makeIs(LIBRAW_CAMERAMAKER_Kodak) || makeIs(LIBRAW_CAMERAMAKER_Leica)) &&
      ((cp = strcasestr(model, " DIGITAL CAMERA")) ||
       (cp = strstr(model, "FILE VERSION")))) {
    *cp = 0;
  }

  remove_trailing_spaces(make, sizeof(make));
  remove_trailing_spaces(model, sizeof(model));

  i = int(strbuflen(make)); /* Remove make from model */
  if (!strncasecmp(model, make, i) && model[i++] == ' ')
    memmove(model, model + i, 64 - i);

  if (makeIs(LIBRAW_CAMERAMAKER_Fujifilm) && !strncmp(model, "FinePix", 7)) {
    memmove(model, model + 7, strlen(model) - 6);
    if (model[0] == ' ') {
      memmove(model, model + 1, strlen(model));
    }
  } else if ((makeIs(LIBRAW_CAMERAMAKER_Kodak) || makeIs(LIBRAW_CAMERAMAKER_Konica)) &&
             !strncmp(model, "Digital Camera ", 15)) {
    memmove(model, model + 15, strlen(model) - 14);
  }

  desc[511] = artist[63] = make[63] = model[63] = model2[63] = 0;
  if (!is_raw)
    goto notraw;

  if (!height)
    height = raw_height;
  if (!width)
    width = raw_width;

  identify_finetune_pentax();


  if (dng_version)
  {
    if (filters == UINT_MAX)
      filters = 0;
    if (!filters)
      colors = tiff_samples;
    switch (tiff_compress)
    {
    case 0: // Compression not set, assuming uncompressed
    case 1:
      // Uncompressed float: decoder set in apply_tiff for valid files; not set for non-valid with sampleformat==3
      if ((load_raw != &LibRaw::uncompressed_fp_dng_load_raw)  && (tiff_sampleformat != 3))
        load_raw = &LibRaw::packed_dng_load_raw;
      break;
    case 7:
      load_raw = &LibRaw::lossless_dng_load_raw;
      break;
    case 8:
        if (tiff_sampleformat == 3 && tiff_bps > 8 && (tiff_bps % 8 == 0) && tiff_bps <= 32)
            load_raw = &LibRaw::deflate_dng_load_raw;
        else if((tiff_sampleformat == 0 || tiff_sampleformat == 1) && tiff_bps>=8 && tiff_bps <=16)
          load_raw = &LibRaw::deflate_dng_load_raw;
        break;
#ifdef USE_GPRSDK
    case 9:
        load_raw = &LibRaw::vc5_dng_load_raw_placeholder;
        break;
#endif
    case 34892:
      load_raw = &LibRaw::lossy_dng_load_raw;
      break;
    default:
      load_raw = 0;
    }
    GetNormalizedModel();
    if (makeIs(LIBRAW_CAMERAMAKER_Leica)) {
		  if (!strcmp(model, "SL2")) 
              height -= 3;
          if (!strncasecmp(model, "Q2 MONO",7))
              height -= 18;
    }

    else if (makeIs(LIBRAW_CAMERAMAKER_Olympus) &&
        (OlyID == OlyID_STYLUS_1) && // don't use normalized_model below, it is 'Stylus 1'
        (strchr(model+6, 's') ||
         strchr(model+6, 'S')))
    {
      width -= 16;
    }
    goto dng_skip;
  }

  if (makeIs(LIBRAW_CAMERAMAKER_Canon) && !fsize && tiff_bps != 15)
  {
      bool fromtable = false;
    if (!load_raw)
      load_raw = &LibRaw::lossless_jpeg_load_raw;
    for (i = 0; i < int(sizeof canon / sizeof *canon); i++)
      if (raw_width == canon[i][0] && raw_height == canon[i][1])
      {
        width = raw_width - (left_margin = canon[i][2]);
        height = raw_height - (top_margin = canon[i][3]);
        width -= canon[i][4];
        height -= canon[i][5];
        mask[0][1] = canon[i][6];
        mask[0][3] = -canon[i][7];
        mask[1][1] = canon[i][8];
        mask[1][3] = -canon[i][9];
        if (canon[i][10])
          filters = canon[i][10] * 0x01010101U;
        fromtable = true;
      }
    if ((unique_id | 0x20000ULL) ==
        0x2720000ULL) // "PowerShot G11", "PowerShot S90": 0x2700000, 0x2720000
                      // possibly "PowerShot SX120 IS" (if not chdk hack?): 0x2710000
    {
      left_margin = 8;
      top_margin = 16;
    }
    if(!fromtable && imCanon.AverageBlackLevel) // not known, but metadata known
    {
        FORC4 cblack[c] = imCanon.ChannelBlackLevel[c];
        black = cblack[4] = cblack[5] = 0;
        // Prevent automatic BL calculation
        mask[0][3] = 1;
        mask[0][1] = 2;

        if ((imCanon.SensorWidth == raw_width) &&
            (imCanon.SensorHeight == raw_height))
        {
            left_margin = (imCanon.DefaultCropAbsolute.l+1) & 0xfffe; // round to 2
            width = imCanon.DefaultCropAbsolute.r - left_margin;
            top_margin = (imCanon.DefaultCropAbsolute.t +1)  & 0xfffe;
            height = imCanon.DefaultCropAbsolute.b - top_margin;
        }
    }
  }

  identify_finetune_by_filesize(fsize);

  if (!strcmp(model, "KAI-0340") && find_green(16, 16, 3840, 5120) < 25)
  {
    height = 480;
    top_margin = filters = 0;
    strcpy(model, "C603");
  }

  GetNormalizedModel();

  identify_finetune_dcr(head, fsize, flen);

  /* Early reject for damaged images */
  if (!load_raw || height < 22 || width < 22 ||
      (tiff_bps > 16 &&
       (load_raw != &LibRaw::deflate_dng_load_raw &&
        load_raw != &LibRaw::uncompressed_fp_dng_load_raw)) ||
      tiff_samples > 4 || colors > 4 ||
      colors < 1
      /* alloc in unpack() may be fooled by size adjust */
      || ((int)width + (int)left_margin > 65535) ||
      ((int)height + (int)top_margin > 65535))
  {
    is_raw = 0;
    RUN_CALLBACK(LIBRAW_PROGRESS_IDENTIFY, 1, 2);
    return;
  }
  if (!model[0])
  {
    sprintf(model, "%dx%d", width, height);
    strcpy(normalized_model, model);
  }

  if (!(imgdata.rawparams.options & LIBRAW_RAWOPTIONS_ZEROFILTERS_FOR_MONOCHROMETIFFS) &&
      (filters == UINT_MAX)) // Default dcraw behaviour
    filters = 0x94949494;
  else if (filters == UINT_MAX)
  {
    if (tiff_nifds > 0 && tiff_samples == 1)
    {
      colors = 1;
      filters = 0;
    }
    else
      filters = 0x94949494;
  }

  if (thumb_offset && !thumb_height)
  {
    fseek(ifp, thumb_offset, SEEK_SET);
    if (ljpeg_start(&jh, 1))
    {
      thumb_width = jh.wide;
      thumb_height = jh.high;
    }
  }

dng_skip:
  if (dng_version)
	  identify_process_dng_fields();

  /* Early reject for damaged images again (after dng fields processing) */
  if (!load_raw || height < 22 || width < 22 ||
      (tiff_bps > 16 &&
       (load_raw != &LibRaw::deflate_dng_load_raw &&
        load_raw != &LibRaw::uncompressed_fp_dng_load_raw )) ||
      ((load_raw == &LibRaw::deflate_dng_load_raw || load_raw == &LibRaw::uncompressed_fp_dng_load_raw)
        && (tiff_bps < 16 || tiff_bps > 32 || (tiff_bps % 8))   )
      ||tiff_samples > 4 || colors > 4 || colors < 1)
  {
    is_raw = 0;
    RUN_CALLBACK(LIBRAW_PROGRESS_IDENTIFY, 1, 2);
    return;
  }
  {
    // Check cam_mul range
    int cmul_ok = 1;
    FORCC if (cam_mul[c] <= 0.001f) cmul_ok = 0;
    ;

    if (cmul_ok)
    {
      double cmin = cam_mul[0], cmax;
      double cnorm[4];
      FORCC cmin = MIN(cmin, cam_mul[c]);
      FORCC cnorm[c] = cam_mul[c] / cmin;
      cmax = cmin = cnorm[0];
      FORCC
      {
        cmin = MIN(cmin, cnorm[c]);
        cmax = MIN(cmax, cnorm[c]);
      }
      if (cmin <= 0.01f || cmax > 100.f)
        cmul_ok = false;
    }
    if (!cmul_ok)
    {
      if (cam_mul[0] > 0)
        cam_mul[0] = 0;
      cam_mul[3] = 0;
    }
  }
  if ((use_camera_matrix & (((use_camera_wb || dng_version)?1:0) | 0x2)) &&
      cmatrix[0][0] > 0.125)
  {
    memcpy(rgb_cam, cmatrix, sizeof cmatrix);
    raw_color = 0;
  }
  if (raw_color && !CM_found)
    CM_found = adobe_coeff(maker_index, normalized_model);
  else if ((imgdata.color.cam_xyz[0][0] < 0.01) && !CM_found)
    CM_found = adobe_coeff(maker_index, normalized_model, 1);

  if (load_raw == &LibRaw::kodak_radc_load_raw)
    if ((raw_color) && !CM_found)
		CM_found = adobe_coeff(LIBRAW_CAMERAMAKER_Apple, "Quicktake");

  if ((maker_index != LIBRAW_CAMERAMAKER_Unknown) && normalized_model[0])
    SetStandardIlluminants (maker_index, normalized_model);

  // Clear erroneous fuji_width if not set through parse_fuji or for DNG
  if (fuji_width && !dng_version &&
      !(imgdata.process_warnings & LIBRAW_WARN_PARSEFUJI_PROCESSED))
    fuji_width = 0;

  if (fuji_width)
  {
    fuji_width = width >> int(!fuji_layout);
    filters = fuji_width & 1 ? 0x94949494 : 0x49494949;
    width = (height >> fuji_layout) + fuji_width;
    height = width - 1;
    pixel_aspect = 1;
	// Prevent incorrect-sized fuji-rotated files
	if (INT64(width)*INT64(height) > INT64(raw_width) * INT64(raw_height) * 8LL)
		is_raw = 0;
  }
  else
  {
    if (raw_height < height)
      raw_height = height;
    if (raw_width < width)
      raw_width = width;
  }
  if (!tiff_bps)
    tiff_bps = 12;
  if (!maximum)
  {
    maximum = (1 << tiff_bps) - 1;
    if (maximum < 0x10000 && curve[maximum] > 0 &&
        load_raw == &LibRaw::sony_arw2_load_raw)
      maximum = curve[maximum];
  }
  if (maximum > 0xffff)
    maximum = 0xffff;
  if (!load_raw || height < 22 || width < 22 ||
      (tiff_bps > 16 &&
       (load_raw != &LibRaw::deflate_dng_load_raw &&
        load_raw != &LibRaw::uncompressed_fp_dng_load_raw)) ||
      tiff_samples > 6 || colors > 4)
    is_raw = 0;

  if (raw_width < 22 || raw_width > 64000 || raw_height < 22 ||
      pixel_aspect < 0.1 || pixel_aspect > 10. ||
      raw_height > 64000)
    is_raw = 0;
   if(raw_width <= left_margin || raw_height <= top_margin)
       is_raw = 0;
   if (dng_version && (tiff_samples < 1 || tiff_samples > 4))
       is_raw = 0; // we do not handle DNGs with more than 4 values per pixel

#ifdef LIBRAW_OLD_VIDEO_SUPPORT
#ifdef NO_JASPER
  if (load_raw == &LibRaw::redcine_load_raw)
  {
    is_raw = 0;
    imgdata.process_warnings |= LIBRAW_WARN_NO_JASPER;
  }
#endif
#endif
#ifdef NO_JPEG
  if (load_raw == &LibRaw::kodak_jpeg_load_raw ||
      load_raw == &LibRaw::lossy_dng_load_raw)
  {
    is_raw = 0;
    imgdata.process_warnings |= LIBRAW_WARN_NO_JPEGLIB;
  }
#endif
  if (!cdesc[0])
    strcpy(cdesc, colors == 3 ? "RGBG" : "GMCY");
  if (!raw_height)
    raw_height = height;
  if (!raw_width)
    raw_width = width;
  if (filters > 999 && colors == 3)
    filters |= ((filters >> 2 & 0x22222222) | (filters << 2 & 0x88888888)) &
               filters << 1;
notraw:
  if (flip == (int)UINT_MAX)
    flip = tiff_flip;
  if (flip == (int)UINT_MAX)
    flip = 0;

  // Convert from degrees to bit-field if needed
  if (flip > 89 || flip < -89)
  {
    switch ((flip + 3600) % 360)
    {
    case 270:
      flip = 5;
      break;
    case 180:
      flip = 3;
      break;
    case 90:
      flip = 6;
      break;
    }
  }

  if (pana_bpp)
    imgdata.color.raw_bps = pana_bpp;
  else if ((load_raw == &LibRaw::phase_one_load_raw) ||
		   (load_raw == &LibRaw::phase_one_load_raw_s) ||
           (load_raw == &LibRaw::phase_one_load_raw_c))
    imgdata.color.raw_bps = ph1.format;
  else
    imgdata.color.raw_bps = tiff_bps;

  RUN_CALLBACK(LIBRAW_PROGRESS_IDENTIFY, 1, 2);
}

void LibRaw::identify_process_dng_fields()
{
	if (!dng_version) return;

	// Cleanup inset_crops if set by makernotes parser
	imgdata.sizes.raw_inset_crops[0].cleft = imgdata.sizes.raw_inset_crops[0].ctop =
        imgdata.sizes.raw_inset_crops[1].cleft = imgdata.sizes.raw_inset_crops[1].ctop = 0xffff;
    imgdata.sizes.raw_inset_crops[0].cwidth = imgdata.sizes.raw_inset_crops[0].cheight =
        imgdata.sizes.raw_inset_crops[1].cwidth = imgdata.sizes.raw_inset_crops[1].cheight = 0;


	int c;
	{
		/* copy DNG data from per-IFD field to color.dng */
		int iifd = find_ifd_by_offset(data_offset);
		int pifd = find_ifd_by_offset(thumb_offset);


#define IFDCOLORINDEX(ifd, subset, bit)                                        \
  (tiff_ifd[ifd].dng_color[subset].parsedfields & bit)                         \
      ? ifd                                                                    \
      : ((tiff_ifd[0].dng_color[subset].parsedfields & bit) ? 0 : -1)

#define IFDLEVELINDEX(ifd, bit)                                                \
  (tiff_ifd[ifd].dng_levels.parsedfields & bit)                                \
      ? ifd                                                                    \
      : ((tiff_ifd[0].dng_levels.parsedfields & bit) ? 0 : -1)

#define COPYARR(to, from) memmove(&to, &from, sizeof(from))

		if (iifd < (int)tiff_nifds && iifd >= 0)
		{
			int sidx;
			// Per field, not per structure
			if (!(imgdata.rawparams.options & LIBRAW_RAWOPTIONS_DONT_CHECK_DNG_ILLUMINANT))
			{
				int illidx[2], cmidx[2], calidx[2], abidx;
				for (int i = 0; i < 2; i++)
				{
					illidx[i] = IFDCOLORINDEX(iifd, i, LIBRAW_DNGFM_ILLUMINANT);
					cmidx[i] = IFDCOLORINDEX(iifd, i, LIBRAW_DNGFM_COLORMATRIX);
					calidx[i] = IFDCOLORINDEX(iifd, i, LIBRAW_DNGFM_CALIBRATION);
				}
				abidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_ANALOGBALANCE);
				// Data found, all in same ifd, illuminants are inited
				if (illidx[0] >= 0 && illidx[0] < (int)tiff_nifds &&
					illidx[0] == illidx[1] && illidx[0] == cmidx[0] &&
					illidx[0] == cmidx[1] &&
					tiff_ifd[illidx[0]].dng_color[0].illuminant > 0 &&
					tiff_ifd[illidx[0]].dng_color[1].illuminant > 0)
				{
					sidx = illidx[0]; // => selected IFD
					double cc[4][4], cm[4][3], cam_xyz[4][3];
					// CM -> Color Matrix
					// CC -> Camera calibration
					for (int j = 0; j < 4; j++)
						for (int i = 0; i < 4; i++)
							cc[j][i] = i == j;
					int colidx = -1;

					// IS D65 here?
					for (int i = 0; i < 2; i++)
					{
						if (tiff_ifd[sidx].dng_color[i].illuminant == LIBRAW_WBI_D65)
						{
							colidx = i;
							break;
						}
					}

					// Other daylight-type ill
					if (colidx < 0)
						for (int i = 0; i < 2; i++)
						{
							int ill = tiff_ifd[sidx].dng_color[i].illuminant;
							if (ill == LIBRAW_WBI_Daylight || ill == LIBRAW_WBI_D55 ||
								ill == LIBRAW_WBI_D75 || ill == LIBRAW_WBI_D50 ||
								ill == LIBRAW_WBI_Flash)
							{
								colidx = i;
								break;
							}
						}
					if (colidx >= 0) // Selected
					{
						// Init camera matrix from DNG
						FORCC for (int j = 0; j < 3; j++) cm[c][j] =
							tiff_ifd[sidx].dng_color[colidx].colormatrix[c][j];

						if (calidx[colidx] == sidx)
						{
							for (int i = 0; i < colors && i < 4; i++)
								FORCC
								cc[i][c] = tiff_ifd[sidx].dng_color[colidx].calibration[i][c];
						}

						if (abidx == sidx)
							for (int i = 0; i < colors && i < 4; i++)
								FORCC cc[i][c] *= tiff_ifd[sidx].dng_levels.analogbalance[i];
						int j;
						FORCC for (int i = 0; i < 3; i++)
                            for (cam_xyz[c][i] = j = 0; j < colors && j < 4; j++)
							    cam_xyz[c][i] +=
							        cc[c][j] * cm[j][i]; // add AsShotXY later * xyz[i];
						cam_xyz_coeff(cmatrix, cam_xyz);
					}
				}
			}

			bool noFujiDNGCrop = makeIs(LIBRAW_CAMERAMAKER_Fujifilm)
				&& (!strcmp(normalized_model, "S3Pro")
					|| !strcmp(normalized_model, "S5Pro")
					|| !strcmp(normalized_model, "S2Pro"));

			if (!noFujiDNGCrop) // Promote DNG Crops to raw_inset_crops
			{
				sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_CROPORIGIN);
				int sidx2 = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_CROPSIZE);
                if (sidx >= 0 && sidx == sidx2 &&
                    tiff_ifd[sidx].dng_levels.default_crop[2] > 0 &&
                    tiff_ifd[sidx].dng_levels.default_crop[3] > 0)
                {
                    int lm = tiff_ifd[sidx].dng_levels.default_crop[0];
                    int tm = tiff_ifd[sidx].dng_levels.default_crop[1];
                    int ww = tiff_ifd[sidx].dng_levels.default_crop[2];
                    int hh = tiff_ifd[sidx].dng_levels.default_crop[3];
                    if ((lm + ww < int(raw_width) + int(left_margin))
                        && (tm + hh < int(raw_height) + int(top_margin))) // Crop data is correct
                    {
                        imgdata.sizes.raw_inset_crops[0].cleft = left_margin + lm;
                        imgdata.sizes.raw_inset_crops[0].cwidth = ww;
                        imgdata.sizes.raw_inset_crops[0].ctop = top_margin + tm;
                        imgdata.sizes.raw_inset_crops[0].cheight = hh;

                        int sidx3 = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_USERCROP);
                        if (sidx3 >= 0 && sidx3 == sidx) // No need to check values range, it is checked at parse
                        {
                            int dt = int(imgdata.sizes.raw_inset_crops[0].cheight * tiff_ifd[sidx].dng_levels.user_crop[0]);
                            int dl = int(imgdata.sizes.raw_inset_crops[0].cwidth * tiff_ifd[sidx].dng_levels.user_crop[1]);
                            int db = int(imgdata.sizes.raw_inset_crops[0].cheight * tiff_ifd[sidx].dng_levels.user_crop[2]);
                            int dr = int(imgdata.sizes.raw_inset_crops[0].cwidth * tiff_ifd[sidx].dng_levels.user_crop[3]);

                            int dh = db - dt;
                            int dw = dr - dl;

                            if (dh > 0 && dw > 0
                                && dh < imgdata.sizes.raw_inset_crops[0].cheight // No need to repeat crop for 0,0,1,1
                                && dw < imgdata.sizes.raw_inset_crops[0].cwidth)
                            {
                                imgdata.sizes.raw_inset_crops[1].cleft = imgdata.sizes.raw_inset_crops[0].cleft + dl;
                                imgdata.sizes.raw_inset_crops[1].cwidth = dw;
                                imgdata.sizes.raw_inset_crops[1].ctop = imgdata.sizes.raw_inset_crops[0].ctop + dt;
                                imgdata.sizes.raw_inset_crops[1].cheight = dh;
                            }
                        }

                    }
                }
			}
			if (!(imgdata.color.dng_color[0].parsedfields &
				LIBRAW_DNGFM_FORWARDMATRIX)) // Not set already (Leica makernotes)
			{
				sidx = IFDCOLORINDEX(iifd, 0, LIBRAW_DNGFM_FORWARDMATRIX);
				if (sidx >= 0)
					COPYARR(imgdata.color.dng_color[0].forwardmatrix,
						tiff_ifd[sidx].dng_color[0].forwardmatrix);
			}
			if (!(imgdata.color.dng_color[1].parsedfields &
				LIBRAW_DNGFM_FORWARDMATRIX)) // Not set already (Leica makernotes)
			{
				sidx = IFDCOLORINDEX(iifd, 1, LIBRAW_DNGFM_FORWARDMATRIX);
				if (sidx >= 0)
					COPYARR(imgdata.color.dng_color[1].forwardmatrix,
						tiff_ifd[sidx].dng_color[1].forwardmatrix);
			}
			for (int ss = 0; ss < 2; ss++)
			{
				sidx = IFDCOLORINDEX(iifd, ss, LIBRAW_DNGFM_COLORMATRIX);
				if (sidx >= 0)
					COPYARR(imgdata.color.dng_color[ss].colormatrix,
						tiff_ifd[sidx].dng_color[ss].colormatrix);

				sidx = IFDCOLORINDEX(iifd, ss, LIBRAW_DNGFM_CALIBRATION);
				if (sidx >= 0)
					COPYARR(imgdata.color.dng_color[ss].calibration,
						tiff_ifd[sidx].dng_color[ss].calibration);

				sidx = IFDCOLORINDEX(iifd, ss, LIBRAW_DNGFM_ILLUMINANT);
				if (sidx >= 0)
					imgdata.color.dng_color[ss].illuminant =
					tiff_ifd[sidx].dng_color[ss].illuminant;
			}
			// Levels
			sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_ANALOGBALANCE);
			if (sidx >= 0)
				COPYARR(imgdata.color.dng_levels.analogbalance,
					tiff_ifd[sidx].dng_levels.analogbalance);

			sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_BASELINEEXPOSURE);
			if (sidx >= 0)
				imgdata.color.dng_levels.baseline_exposure =
				tiff_ifd[sidx].dng_levels.baseline_exposure;

			sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_WHITE);
			if (sidx >= 0 && tiff_ifd[sidx].dng_levels.dng_whitelevel[0])
				COPYARR(imgdata.color.dng_levels.dng_whitelevel,
					tiff_ifd[sidx].dng_levels.dng_whitelevel);
			else if (tiff_ifd[iifd].sample_format <= 2 && tiff_ifd[iifd].bps > 0 && tiff_ifd[iifd].bps < 32)
				FORC4
				imgdata.color.dng_levels.dng_whitelevel[c] = (1 << tiff_ifd[iifd].bps) - 1;



			sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_ASSHOTNEUTRAL);
			if (sidx >= 0)
			{
				COPYARR(imgdata.color.dng_levels.asshotneutral,
					tiff_ifd[sidx].dng_levels.asshotneutral);
				if (imgdata.color.dng_levels.asshotneutral[0])
				{
					cam_mul[3] = 0;
					FORCC
						if (fabs(imgdata.color.dng_levels.asshotneutral[c]) > 0.0001)
							cam_mul[c] = 1 / imgdata.color.dng_levels.asshotneutral[c];
				}
			}
			sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_BLACK);
			if (sidx >= 0)
			{
				imgdata.color.dng_levels.dng_fblack =
					tiff_ifd[sidx].dng_levels.dng_fblack;
				imgdata.color.dng_levels.dng_black =
					tiff_ifd[sidx].dng_levels.dng_black;
				COPYARR(imgdata.color.dng_levels.dng_cblack,
					tiff_ifd[sidx].dng_levels.dng_cblack);
				COPYARR(imgdata.color.dng_levels.dng_fcblack,
					tiff_ifd[sidx].dng_levels.dng_fcblack);
			}


			if (pifd >= 0)
			{
				sidx = IFDLEVELINDEX(pifd, LIBRAW_DNGFM_PREVIEWCS);
				if (sidx >= 0)
					imgdata.color.dng_levels.preview_colorspace =
					tiff_ifd[sidx].dng_levels.preview_colorspace;
			}
			sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_OPCODE2);
			if (sidx >= 0)
				meta_offset = tiff_ifd[sidx].opcode2_offset;

			sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_LINTABLE);
			INT64 linoff = -1;
			int linlen = 0;
			if (sidx >= 0)
			{
				linoff = tiff_ifd[sidx].lineartable_offset;
				linlen = tiff_ifd[sidx].lineartable_len;
			}

			if (linoff >= 0 && linlen > 0)
			{
				INT64 pos = ftell(ifp);
				fseek(ifp, linoff, SEEK_SET);
				linear_table(linlen);
				fseek(ifp, pos, SEEK_SET);
			}
			// Need to add curve too
		}
		/* Copy DNG black level to LibRaw's */
		if (load_raw == &LibRaw::lossy_dng_load_raw)
		{
			maximum = 0xffff;
			FORC4 imgdata.color.linear_max[c] = imgdata.color.dng_levels.dng_whitelevel[c] = 0xffff;
		}
		else
		{
			maximum = imgdata.color.dng_levels.dng_whitelevel[0];
		}
		black = imgdata.color.dng_levels.dng_black;
		if (tiff_samples == 2 &&
		    !imgdata.color.dng_levels.dng_cblack[2] &&
		    !imgdata.color.dng_levels.dng_cblack[3] &&
		    (imgdata.color.dng_levels.dng_cblack[4] == 1) &&
		    (imgdata.color.dng_levels.dng_cblack[5] == 1)
            && (imgdata.color.dng_levels.dng_cblack[LIBRAW_CBLACK_SIZE - 1] == tiff_samples)
            ) {
		  black = imgdata.color.dng_levels.dng_cblack[shot_select];
		  imgdata.color.dng_levels.dng_cblack[0] = imgdata.color.dng_levels.dng_cblack[1] = 0;
		  imgdata.color.dng_levels.dng_cblack[4] = imgdata.color.dng_levels.dng_cblack[5] = 0;
		  imgdata.color.dng_levels.dng_fcblack[0] = imgdata.color.dng_levels.dng_fcblack[1] = 0.0f;
		  imgdata.color.dng_levels.dng_fcblack[4] = imgdata.color.dng_levels.dng_fcblack[5] = 0.0f;
		}
		else if (tiff_samples == 2 && imgdata.color.dng_levels.dng_cblack[4] * imgdata.color.dng_levels.dng_cblack[5] * tiff_samples
			== imgdata.color.dng_levels.dng_cblack[LIBRAW_CBLACK_SIZE - 1])
		{
			unsigned ff = filters;
			if (filters > 999 && colors == 3)
				filters |= ((filters >> 2 & 0x22222222) | (filters << 2 & 0x88888888)) &
				filters << 1;

			/* Special case, Fuji SuperCCD dng */
			int csum[4] = { 0,0,0,0 }, ccount[4] = { 0,0,0,0 };
			int i = 6 + shot_select;
			for (unsigned row = 0; row < imgdata.color.dng_levels.dng_cblack[4]; row++)
				for (unsigned col = 0; col < imgdata.color.dng_levels.dng_cblack[5]; col++)
				{
					csum[FC(row, col)] += imgdata.color.dng_levels.dng_cblack[i];
					ccount[FC(row, col)]++;
					i += tiff_samples;
				}
			for (int q = 0; q < 4; q++)
				if (ccount[q])
					imgdata.color.dng_levels.dng_cblack[q] += csum[q] / ccount[q];
			imgdata.color.dng_levels.dng_cblack[4] = imgdata.color.dng_levels.dng_cblack[5] = 0;
			filters = ff;
		}
		else if (tiff_samples > 2 && tiff_samples <= 4 && imgdata.color.dng_levels.dng_cblack[4] * imgdata.color.dng_levels.dng_cblack[5] * tiff_samples
			== imgdata.color.dng_levels.dng_cblack[LIBRAW_CBLACK_SIZE - 1])
		{
			/* Special case, per_channel blacks in RepeatDim, average for per-channel */
			int csum[4] = { 0,0,0,0 }, ccount[4] = { 0,0,0,0 };
			int i = 6;
			for (unsigned row = 0; row < imgdata.color.dng_levels.dng_cblack[4]; row++)
				for (unsigned col = 0; col < imgdata.color.dng_levels.dng_cblack[5]; col++)
					for (unsigned q = 0; q < tiff_samples && q < 4; q++)
					{
						csum[q] += imgdata.color.dng_levels.dng_cblack[i];
						ccount[q]++;
						i++;
					}
			for (int q = 0; q < 4; q++)
				if (ccount[q])
					imgdata.color.dng_levels.dng_cblack[q] += csum[q] / ccount[q];
			imgdata.color.dng_levels.dng_cblack[4] = imgdata.color.dng_levels.dng_cblack[5] = 0;
		}

		memmove(cblack, imgdata.color.dng_levels.dng_cblack, sizeof(cblack));

		if (iifd < (int)tiff_nifds && iifd >= 0)
		{
			int sidx = IFDLEVELINDEX(iifd, LIBRAW_DNGFM_LINEARRESPONSELIMIT);
			if (sidx >= 0)
			{
				imgdata.color.dng_levels.LinearResponseLimit =
					tiff_ifd[sidx].dng_levels.LinearResponseLimit;
				if (imgdata.color.dng_levels.LinearResponseLimit > 0.1 &&
					imgdata.color.dng_levels.LinearResponseLimit <= 1.0)
				{
					// And approx promote it to linear_max:
					int bl4 = 0, bl64 = 0;
					for (int chan = 0; chan < colors && chan < 4; chan++)
						bl4 += cblack[chan];
					bl4 /= LIM(colors, 1, 4);

					if (cblack[4] * cblack[5] > 0)
					{
						unsigned cnt = 0;
						for (unsigned q = 0; q < 4096 && q < cblack[4] * cblack[5]; q++)
						{
							bl64 += cblack[q + 6];
							cnt++;
						}
						bl64 /= LIM(cnt, 1, 4096);
					}
					int rblack = black + bl4 + bl64;
                    for (int chan = 0; chan < colors && chan < 4; chan++)
                        imgdata.color.linear_max[chan] =
                            (maximum - rblack) *
                            imgdata.color.dng_levels.LinearResponseLimit +
                            rblack;
                    if (imgdata.color.linear_max[1] && !imgdata.color.linear_max[3])
                        imgdata.color.linear_max[3] = imgdata.color.linear_max[1];
				}
			}
		}
	}
}

void LibRaw::identify_finetune_pentax()
{
    if (dng_version && data_offset)
    {
        for(int i = 0; i < (int)tiff_nifds; i++)
            if (tiff_ifd[i].offset == data_offset)
            {
                if (tiff_ifd[i].phint == 34892) return; // Linear DNG made from Pentax source
                break;
            }
    }

	if (makeIs(LIBRAW_CAMERAMAKER_Pentax) ||
		makeIs(LIBRAW_CAMERAMAKER_Samsung)) {
		if (height == 2624 &&
			width == 3936) // Pentax K10D, Samsung GX10;
		{
			height = 2616;
			width = 3896;
		}
		if (height == 3136 &&
			width == 4864) // Pentax K20D, Samsung GX20;
		{
			height = 3124;
			width = 4688;
			filters = 0x16161616;
		}
	}

	if (makeIs(LIBRAW_CAMERAMAKER_Pentax)) {
		if ((width == 4352) &&
			((unique_id == PentaxID_K_r) ||
			(unique_id == PentaxID_K_x)))
		{
			width = 4309;
			filters = 0x16161616;
		}
		if ((width >= 4960) &&
			((unique_id == PentaxID_K_5) ||
			(unique_id == PentaxID_K_5_II) ||
				(unique_id == PentaxID_K_5_II_s)))
		{
			left_margin = 10;
			width = 4950;
			filters = 0x16161616;
		}
		if ((width == 6080) && (unique_id == PentaxID_K_70))
		{
			height = 4016;
			top_margin = 32;
			width = 6020;
			left_margin = 60;
		}
		if ((width == 4736) && (unique_id == PentaxID_K_7))
		{
			height = 3122;
			width = 4684;
			filters = 0x16161616;
			top_margin = 2;
		}
		if ((width == 6080) && (unique_id == PentaxID_K_3_II))
		{
			left_margin = 4;
			width = 6040;
		}
        if ((width == 6304) && (unique_id == PentaxID_K_3_III)) // From DNG ActiveArea
        {
          left_margin = 26;
          width = 6224;
          top_margin = 34;
          height = 4160;
        }
        if ((width == 6112) && (unique_id == PentaxID_KP))
		{
			// From DNG, maybe too strict
			left_margin = 54;
			top_margin = 28;
			width = 6028;
			height = raw_height - top_margin;
		}
		if ((width == 6080) && (unique_id == PentaxID_K_3))
		{
			left_margin = 4;
			width = 6040;
		}
		if ((width == 7424) && (unique_id == PentaxID_645D))
		{
			height = 5502;
			width = 7328;
			filters = 0x61616161;
			top_margin = 29;
			left_margin = 48;
		}
	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Ricoh) &&
		(height == 3014) && (width == 4096))  // Ricoh GX200
		width = 4014;
}

void LibRaw::identify_finetune_by_filesize(int fsize)
{

	if (fsize == 4771840)
	{ // hack Nikon 3mpix: E880, E885, E990, E995;
	  // Olympus C-3030Z
		if (!timestamp && nikon_e995())
			strcpy(model, "E995");
	}
	else if (fsize == 2940928)
	{ // hack Nikon 2mpix: E2100, E2500
		if (!timestamp && !nikon_e2100())
			strcpy(model, "E2500");
	}
	else if (fsize == 4775936)
	{ // hack Nikon 3mpix: E3100, E3200, E3500, E3700;
	  // Pentax "Optio 33WR";
	  // Olympus C-740UZ
		if (!timestamp)
			nikon_3700();
	}
	else if (fsize == 5869568)
	{ // hack Nikon 4mpix: E4300;
	  // hack Minolta "DiMAGE Z2"
		if (!timestamp && minolta_z2())
		{
			maker_index = LIBRAW_CAMERAMAKER_Minolta;
			strcpy(make, "Minolta");
			strcpy(model, "DiMAGE Z2");
		}
	}
}

void LibRaw::identify_finetune_dcr(char head[64], int fsize, int flen)
{
	static const short pana[][6] = {
		// raw_width, raw_height, left_margin, top_margin, width_increment,
		// height_increment
		{3130, 1743, 4, 0, -6, 0},      /* 00 */
		{3130, 2055, 4, 0, -6, 0},      /* 01 */
		{3130, 2319, 4, 0, -6, 0},      /* 02 DMC-FZ8 */
		{3170, 2103, 18, 0, -42, 20},   /* 03 */
		{3170, 2367, 18, 13, -42, -21}, /* 04 */
		{3177, 2367, 0, 0, -1, 0},      /* 05 DMC-L1 */
		{3304, 2458, 0, 0, -1, 0},      /* 06 DMC-FZ30 */
		{3330, 2463, 9, 0, -5, 0},      /* 07 DMC-FZ18 */
		{3330, 2479, 9, 0, -17, 4},     /* 08 */
		{3370, 1899, 15, 0, -44, 20},   /* 09 */
		{3370, 2235, 15, 0, -44, 20},   /* 10 */
		{3370, 2511, 15, 10, -44, -21}, /* 11 */
		{3690, 2751, 3, 0, -8, -3},     /* 12 DMC-FZ50 */
		{3710, 2751, 0, 0, -3, 0},      /* 13 DMC-L10 */
		{3724, 2450, 0, 0, 0, -2},      /* 14 */
		{3770, 2487, 17, 0, -44, 19},   /* 15 */
		{3770, 2799, 17, 15, -44, -19}, /* 16 */
		{3880, 2170, 6, 0, -6, 0},      /* 17 DMC-LX1 */
		{4060, 3018, 0, 0, 0, -2},      /* 18 DMC-FZ35, DMC-FZ38 */
		{4290, 2391, 3, 0, -8, -1},     /* 19 DMC-LX2 */
		{4330, 2439, 17, 15, -44, -19}, /* 20 "D-LUX 3" */
		{4508, 2962, 0, 0, -3, -4},     /* 21 */
		{4508, 3330, 0, 0, -3, -6},     /* 22 */
		{10480, 7794, 0, 0, -2, 0},     /* 23: G9 in high-res */
	};
	int i,c;
	struct jhead jh;

	if (makeIs(LIBRAW_CAMERAMAKER_Canon) 
        && ( !tiff_flip || unique_id == CanonID_EOS_40D)
		&& !(imgdata.rawparams.options & LIBRAW_RAWOPTIONS_CANON_IGNORE_MAKERNOTES_ROTATION)
        && imCanon.MakernotesFlip)
	{
		tiff_flip = imCanon.MakernotesFlip;
	}

	else if (makeIs(LIBRAW_CAMERAMAKER_Nikon))
	{
		if (!load_raw)
			load_raw = &LibRaw::packed_load_raw;
		if (model[0] == 'E') // Nikon E8800, E8700, E8400, E5700, E5400, E5000,
							 // others are diag hacks?
			load_flags |= !data_offset << 2 | 2;
	}
	/* Set parameters based on camera name (for non-DNG files). */

	/* Always 512 for arw2_load_raw */
	else if (makeIs(LIBRAW_CAMERAMAKER_Sony) &&
		(raw_width > 3888) && !black && !cblack[0])
	{
		black = (load_raw == &LibRaw::sony_arw2_load_raw)
			? 512
			: (128 << (tiff_bps - 12));
	}

	if (is_foveon) {
		if (height * 2 < width)
			pixel_aspect = 0.5;
		if (height > width)
			pixel_aspect = 2;
		filters = 0;

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Pentax)) {
		if ((unique_id == PentaxID_K_1) ||
			(unique_id == PentaxID_K_1_Mark_II)) {
			top_margin = 18;
			height = raw_height - top_margin;
			if (raw_width == 7392) {
				left_margin = 6;
				width = 7376;
			}

		}
		else if (unique_id == PentaxID_Optio_S_V101) { // (fsize == 3178560)
			cam_mul[0] *= 4;
			cam_mul[2] *= 4;

		}
		else if (unique_id == PentaxID_Optio_33WR) { // (fsize == 4775936)
			flip = 1;
			filters = 0x16161616;

		}
		else if (unique_id == PentaxID_staristD) {
			load_raw = &LibRaw::unpacked_load_raw;
			/* data_error = -1; */ /* No way to know why data_error was raised in dcraw.c, looks not needed esp. for unpacked_load_raw */
		}
		else if (unique_id == PentaxID_staristDS) {
			height -= 2;
		}

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Canon)) {
		if (tiff_bps == 15) { // Canon sRAW
			if (width == 3344)
				width = 3272;
			else if (width == 3872)
				width = 3866;

			if (height > width) {
				SWAP(height, width);
				SWAP(raw_height, raw_width);
			}
			if (width == 7200 &&
				height == 3888) { // Canon EOS 5DS (R);
				raw_width = width = 6480;
				raw_height = height = 4320;
			}
			filters = 0;
			tiff_samples = colors = 3;
			load_raw = &LibRaw::canon_sraw_load_raw;
		}

		if (!strcmp(normalized_model, "PowerShot 600")) {
			height = 613;
			width = 854;
			raw_width = 896;
			colors = 4;
			filters = 0xe1e4e1e4;
			load_raw = &LibRaw::canon_600_load_raw;

		}
		else if (!strcmp(normalized_model, "PowerShot A5") ||
			!strcmp(normalized_model, "PowerShot A5 Zoom")) {
			height = 773;
			width = 960;
			raw_width = 992;
			pixel_aspect = 256 / 235.0;
			filters = 0x1e4e1e4e;
			goto canon_a5;

		}
		else if (!strcmp(normalized_model, "PowerShot A50")) {
			height = 968;
			width = 1290;
			raw_width = 1320;
			filters = 0x1b4e4b1e;
			goto canon_a5;

		}
		else if (!strcmp(normalized_model, "PowerShot Pro70")) {
			height = 1024;
			width = 1552;
			filters = 0x1e4b4e1b;
		canon_a5:
			colors = 4;
			tiff_bps = 10;
			load_raw = &LibRaw::packed_load_raw;
			load_flags = 40;

		}
		else if (!strcmp(normalized_model, "PowerShot Pro90 IS") ||
			!strcmp(normalized_model, "PowerShot G1")) {
			colors = 4;
			filters = 0xb4b4b4b4;

		}
		else if (!strcmp(normalized_model, "PowerShot A610")) { // chdk hack
			if (canon_s2is()) {
				strcpy(model + 10, "S2 IS");
				strcpy(normalized_model + 10, "S2 IS");
			}

		}
		else if (!strcmp(normalized_model, "PowerShot SX220 HS")) { // chdk hack
			mask[1][3] = -4;
			top_margin = 16;
			left_margin = 92;

		}
		else if (!strcmp(normalized_model, "PowerShot S120")) { // chdk hack
			raw_width = 4192;
			raw_height = 3062;
			width = 4022;
			height = 3016;
			mask[0][0] = top_margin = 31;
			mask[0][2] = top_margin + height;
			left_margin = 120;
			mask[0][1] = 23;
			mask[0][3] = 72;

		}
		else if (!strcmp(normalized_model, "PowerShot G16")) {
			mask[0][0] = 0;
			mask[0][2] = 80;
			mask[0][1] = 0;
			mask[0][3] = 16;
			top_margin = 29;
			left_margin = 120;
			width = raw_width - left_margin - 48;
			height = raw_height - top_margin - 14;

		}
		else if (!strcmp(normalized_model, "PowerShot SX50 HS")) {
			top_margin = 17;
		}

	}

	else if (makeIs(LIBRAW_CAMERAMAKER_Nikon)) {
		if (!strcmp(model, "D1"))
		{
			imgdata.other.analogbalance[0] = cam_mul[0];
			imgdata.other.analogbalance[2] = cam_mul[2];
			imgdata.other.analogbalance[1] = imgdata.other.analogbalance[3] =
				cam_mul[1];
			cam_mul[0] = cam_mul[1] = cam_mul[2] = 1.0f;
		}

		else if (!strcmp(model, "D1X"))
		{
			width -= 4;
			pixel_aspect = 0.5;
		}
		else if (!strcmp(model, "D40X") ||
			!strcmp(model, "D60") ||
			!strcmp(model, "D80") ||
			!strcmp(model, "D3000"))
		{
			height -= 3;
			width -= 4;
		}
		else if (!strcmp(model, "D3") ||
			!strcmp(model, "D3S") ||
			!strcmp(model, "D700"))
		{
			width -= 4;
			left_margin = 2;
		}
		else if (!strcmp(model, "D3100"))
		{
			width -= 28;
			left_margin = 6;
		}
		else if (!strcmp(model, "D5000") ||
			!strcmp(model, "D90"))
		{
			width -= 42;
		}
		else if (!strcmp(model, "D5100") ||
			!strcmp(model, "D7000") ||
			!strcmp(model, "COOLPIX A"))
		{
			width -= 44;
		}
		else if (!strcmp(model, "D3200") ||
			!strcmp(model, "D600") ||
			!strcmp(model, "D610") ||
			!strncmp(model, "D800", 4)) // Nikons: D800, D800E
		{
			width -= 46;
		}
		else if (!strcmp(model, "D4") ||
			!strcmp(model, "Df"))
		{
			width -= 52;
			left_margin = 2;
		}
		else if (!strcmp(model, "D500"))
		{
			// Empty - to avoid width-1 below
		}
		else if (!strncmp(model, "D40", 3) ||
			!strncmp(model, "D50", 3) ||
			!strncmp(model, "D70", 3))
		{
			width--;
		}
		else if (!strcmp(model, "D100"))
		{
			if (load_flags) // compressed NEF
				raw_width = (width += 3) + 3;
		}
		else if (!strcmp(model, "D200"))
		{
			left_margin = 1;
			width -= 4;
			filters = 0x94949494;
		}
		else if (!strncmp(model, "D2H", 3)) // Nikons: D2H, D2Hs
		{
			left_margin = 6;
			width -= 14;
		}
		else if (!strncmp(model, "D2X", 3)) // Nikons: D2X, D2Xs
		{
			if (width == 3264) // in-camera Hi-speed crop: On
				width -= 32;
			else
				width -= 8;
		}
		else if (!strncmp(model, "D300", 4)) // Nikons: D300, D300s
		{
			width -= 32;
		}
		else if (raw_width == 4032) // Nikon "COOLPIX P7700", "COOLPIX P7800",
									// "COOLPIX P330", "COOLPIX P340"
		{
			if (!strcmp(normalized_model, "COOLPIX P7700"))
			{
				maximum = 65504;
				load_flags = 0;
			}
			else if (!strcmp(normalized_model, "COOLPIX P7800"))
			{
				maximum = 65504;
				load_flags = 0;
			}
			else if (!strcmp(model, "COOLPIX P340"))
			{
				load_flags = 0;
			}
		}
		else if (!strncmp(model, "COOLPIX P", 9) &&
			raw_width != 4032) // Nikon "COOLPIX P1000", "COOLPIX P6000",
							   // "COOLPIX P7000", "COOLPIX P7100"
		{
			load_flags = 24;
			filters = 0x94949494;
			/* the following 'if' is most probably obsolete, because we now read black
			 * level from metadata */
			if ((model[9] == '7') && /* P7000, P7100 */
				((iso_speed >= 400) || (iso_speed == 0)) &&
				!strstr(software, "V1.2")) /* v. 1.2 seen for P7000 only */
				black = 255;
		}
		else if (!strncmp(model, "COOLPIX B700", 12))
		{
			load_flags = 24;
		}
		else if (!strncmp(model, "1 ",
			2)) // Nikons: "1 AW1", "1 J1", "1 J2", "1 J3", "1 J4",
				// "1 J5", "1 S1", "1 S2", "1 V1", "1 V2", "1 V3"
		{
			height -= 2;
		}
		else if (fsize == 1581060) // hack Nikon 1mpix: E900
		{
			simple_coeff(3);
			pre_mul[0] = 1.2085f;
			pre_mul[1] = 1.0943f;
			pre_mul[3] = 1.1103f;
		}
		else if ((fsize == 4771840) &&  // hack Nikon 3mpix: E880, E885, E990
			strcmp(model, "E995")) // but not E995
		{
			filters = 0xb4b4b4b4;
			simple_coeff(3);
			pre_mul[0] = 1.196f;
			pre_mul[1] = 1.246f;
			pre_mul[2] = 1.018f;
		}
		else if ((fsize == 4775936) && // hack Nikon 3mpix: E3100, E3200, E3500
			(atoi(model + 1) < 3700)) // but not E3700;
		{
			filters = 0x49494949;
		}
		else if (fsize == 5869568) // hack Nikon 4mpix: E4300;
		{
			load_flags = 6;
		}
		else if (!strcmp(model, "E2500"))
		{
			height -= 2;
			load_flags = 6;
			colors = 4;
			filters = 0x4b4b4b4b;
		}
	}

	else if (makeIs(LIBRAW_CAMERAMAKER_Olympus)) {
		if (OlyID == OlyID_C_740UZ) { // (fsize == 4775936)
			i = find_green(12, 32, 1188864, 3576832);
			c = find_green(12, 32, 2383920, 2387016);
			if (abs(i) < abs(c)) {
				SWAP(i, c);
				load_flags = 24;
			}
			if (i < 0)
				filters = 0x61616161;
		}
		else if (OlyID == OlyID_C_770UZ) {
			height = 1718;
			width = 2304;
			filters = 0x16161616;
			load_raw = &LibRaw::packed_load_raw;
			load_flags = 30;
		}
		else {
			height += height & 1;
			if (exif_cfa)
				filters = exif_cfa;

			if (width == 4100) // Olympus E-PL2, E-PL1, E-P2, E-P1, E-620, E-600, E-5, E-30;
				width -= 4;

			if (width == 4080) // Olympus E-PM1, E-PL3, E-P3;
				width -= 24;

			if (width == 10400) // Olympus PEN-F, E-M1-II, E-M1-III, E-M1X, OM-1
				width -= 12;

			if (width == 8200) // E-M1-III in 50Mp mode, E-M1X
				width -= 30;

			if (width == 8180) // OM-1 in 50Mp
              width -= 10;

			if (width == 9280) { // Olympus E-M5 Mark II;
				width -= 6;
				height -= 6;
			}

			if (load_raw == &LibRaw::unpacked_load_raw) {
				load_flags = 4;
        if (imOly.ValidBits == 10) load_flags += 2;
			}
      tiff_bps = imOly.ValidBits;

			if ((OlyID == OlyID_E_300) ||
				(OlyID == OlyID_E_500)) {
				width -= 20;
				if (load_raw == &LibRaw::unpacked_load_raw) {
					maximum = 0xfc3;
					memset(cblack, 0, sizeof cblack);
				}
			}
			else if (OlyID == OlyID_STYLUS_1) {
				width -= 16;
				maximum = 0xfff;

			}
			else if (OlyID == OlyID_E_330) {
				width -= 30;
				if (load_raw == &LibRaw::unpacked_load_raw)
					maximum = 0xf79;

			}
			else if (OlyID == OlyID_SP_550UZ) {
				thumb_length = flen - (thumb_offset = 0xa39800);
				thumb_height = 480;
				thumb_width = 640;

			}
			else if (OlyID == OlyID_TG_4) {
				width -= 16;

			}
			else if ((OlyID == OlyID_TG_5) ||
				(OlyID == OlyID_TG_6)) {
				width -= 26;
			}
		}

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_RoverShot) &&
		(fsize == 6291456)) { // RoverShot 3320AF
		fseek(ifp, 0x300000, SEEK_SET);
		if ((order = guess_byte_order(0x10000)) == 0x4d4d)
		{
			height -= (top_margin = 16);
			width -= (left_margin = 28);
			maximum = 0xf5c0;
			strcpy(make, "ISG");
			maker_index = LIBRAW_CAMERAMAKER_ISG;
			model[0] = 0;
		}

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Fujifilm)) {
    if (!imFuji.RAFDataGeneration && (raw_width == 2944)) // S2Pro
		{
			height = 2144;
			width = 2880;
			flip = 6;
		}
		else if (load_raw != &LibRaw::packed_load_raw &&
		         strncmp(model, "X-", 2)              &&
			       filters >= 1000) // Bayer and not an X-model
			maximum = (is_raw == 2 && shot_select) ? 0x2f00 : 0x3e00;

    if (!FujiCropMode && imFuji.RAFDataGeneration && (imFuji.RAFDataGeneration != 4096))
    {
      width  = imFuji.RAFData_ImageSizeTable[0];
      height = imFuji.RAFData_ImageSizeTable[1];
    }
		else if (FujiCropMode == 1) // FF crop on GFX
		{
			width = raw_width;
			height = raw_height;
		}
        // Do we need set height = raw_height for CropMode == 2 for all cameras??
		else if (FujiCropMode == 4) // electronic shutter, high speed mode (1.25x crop)
		{
			height = raw_height;
		}

		top_margin = (raw_height >= height) ? (raw_height - height) >> 2 << 1 : 0;
		left_margin = (raw_width >= width) ? (raw_width - width) >> 2 << 1 : 0;

    if (imFuji.RAFDataGeneration && (imFuji.RAFDataGeneration != 4096)) {
      switch (raw_width) {
      case 2944:                // X-S1, X10, XF1
        filters = 0x16161616;
        break;
      case 4096:                // X20, X30, XQ1, XQ2
      case 5120:                // X-Pro1, X-E1, X-A1, X-A2, X-M1
      case 6048:                // lossless compressed X100F, X-T2, X-T20, X-Pro2, X-H1, X-E3
      case 6160:                // uncompressed (unpacked) X100F, X-T2, X-T20, X-Pro2, X-H1, X-E3
        left_margin = 0;
        break;
      case 4992:                // X-E2S, X-E2, X-T10, X-T1, X100S, X100T, X70
        left_margin = 4;
        break;
      case 6336: // X-H2S
		  top_margin = 6;
		  left_margin = 0;
		  width = 6264;
		  height = 4176;
		  break;
      case 6384:                // X-T3, X-T4, X100V, X-S10, X-T30, X-Pro3
        top_margin = 0;
        switch (FujiCropMode) {
        case 0:        // no crop
				  left_margin = 0;
				  top_margin = 6;
				  width = 6246;
				  height = 4170;
				  break;
				case 2:        // sports finder mode
				  left_margin = 624;
				  width = 5004;
				  height = raw_height;
				  break;
				case 4:        // electronic shutter, high speed mode (1.25x crop)
          left_margin = 624;
				  width = 5004;
				  break;
        }
        break;
      case 6912:                // GFX 50S, GFX 50R; FF crop
      case 9216:                // GFX 50S, GFX 50R; no crop
			  left_margin = 0;
			  top_margin = 0;
        break;
      case 8472:                // GFX 50S II
        left_margin = 0;
			  top_margin  = 0;
			  width = raw_width - 192;
        break;
      case 9696:                // GFX 100; FF crop
      case 11808:               // GFX 100; no crop
			  left_margin = 0;
			  width = raw_width - 146;
			  height = raw_height - (top_margin = 2);
			  if (tiff_bps == 16)
				  maximum = 0xffff;
      default:
      /* insert model name-based width/height/margins/etc. assignments */
        break;
      }

    } else if (!imFuji.RAFDataGeneration) {
      switch (raw_width) {
      case 2304:                // S5100
        height -= (top_margin = 6);
        break;
      case 3328:                // F550EXR, F600EXR, F770EXR, F800EXR, F900EXR,
                                // HS20EXR, HS30EXR, HS33EXR, HS50EXR
        if ((width = raw_width - 66))
          left_margin = 34;
        if (imgdata.sizes.raw_inset_crops[0].cleft == 8) // HS50EXR, F900EXR
        {
			    left_margin = 0;
			    width += 2;
			    filters = 0x16161616;
        }
        break;
      case 3664:                // "HS10 HS11"
        filters = 0x16161616;
        break;
      case 5504:                // DBP for GX680 aka DX-2000

//         7712 2752 -> 5504 3856
//         width = 688;
//         height = 30848;
//         raw_width = 688;
//         raw_height = 30848;

        left_margin = 32; // imgdata.sizes.raw_inset_crops[0].cleft
        top_margin = 8;
        width = raw_width - 2*left_margin;
        height = raw_height - 2*top_margin;

        load_raw = &LibRaw::unpacked_load_raw_FujiDBP;
        //  maximum = 0x0fff;
        filters = 0x16161616;
        load_flags = 0;
        flip = 6;
        break;
      default:
      /* insert model name-based width/height/margins/etc. assignments */
        break;
      }
    }
		if (fuji_layout)
			raw_width *= is_raw;
		if (filters == 9)
			FORC(36)
			((char *)xtrans)[c] =
			xtrans_abs[(c / 6 + top_margin) % 6][(c + left_margin) % 6];
	}

	else if (makeIs(LIBRAW_CAMERAMAKER_Konica)) {
		if (!strcmp(model, "KD-400Z")) {
			height = 1711; // 1712
			width = 2312;
			raw_width = 2336;
			goto konica_400z;
		}
		else if (!strcmp(model, "KD-510Z")) {
			goto konica_510z;
		}

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Minolta)) {
		if (fsize == 5869568) { // hack "DiMAGE Z2"
			load_flags = 30;
		}

		if (imSony.prd_StorageMethod == LIBRAW_MINOLTA_UNPACKED) {
			load_raw = &LibRaw::unpacked_load_raw;
		} else if (imSony.prd_StorageMethod == LIBRAW_MINOLTA_PACKED) {
			load_raw = &LibRaw::packed_load_raw;
      } else if (!load_raw && (maximum = 0xfff)) {
			load_raw = &LibRaw::unpacked_load_raw;
		}

      if (imSony.prd_BayerPattern == LIBRAW_MINOLTA_G2BRG1) {
			filters = 0x49494949;
      } else if (imSony.prd_BayerPattern == LIBRAW_MINOLTA_RGGB) {
			filters = 0x94949494;
      }

      if (imSony.prd_Active_bps && imSony.prd_Total_bps) {
			tiff_bps = imSony.prd_Active_bps;
      }

		if (!strncmp(model, "DiMAGE G", 8)) // hack "DiMAGE G400", "DiMAGE G500",
                                               // "DiMAGE G530", "DiMAGE G600"
		{
			if (model[8] == '4') // DiMAGE G400
			{
				height = 1716;
				width = 2304;
			}
			else if (model[8] == '5') // DiMAGE G500 / G530
			{
			konica_510z:
				height = 1956;
				width = 2607;
				raw_width = 2624;
			}
			else if (model[8] == '6') // DiMAGE G600
			{
				height = 2136;
				width = 2848;
			}
			data_offset += 14;
			filters = 0x61616161;
		konica_400z:
			load_raw = &LibRaw::unpacked_load_raw;
			maximum = 0x3df;
			order = 0x4d4d;
		}

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Samsung)) {
		if (raw_width == 4704) // Samsung NX100, NX10, NX11,
		{
			height -= top_margin = 8;
			width -= 2 * (left_margin = 8);
			load_flags = 32;
		}
		else if (!strcmp(model, "NX3000")) // Samsung NX3000; raw_width: 5600
		{
			top_margin = 38;
			left_margin = 92;
			width = 5456;
			height = 3634;
			filters = 0x61616161;
			colors = 3;
		}
		else if (raw_height == 3714) // Samsung NX2000, NX300M, NX300, NX30,
		                             // "NX U" (aka:
		                             //         "EK-GN100", "EK-GN110", "EK-GN120",
		                             //         "EK-KN120", "Galaxy NX")
		{
			height -= top_margin = 18;
			left_margin = raw_width - (width = 5536);
			if (raw_width != 5600)
				left_margin = top_margin = 0;
			filters = 0x61616161;
			colors = 3;
		}
		else if (raw_width == 5632) // Samsung NX1000, NX200, NX20, NX210
		{
			order = 0x4949;
			height = 3694;
			top_margin = 2;
			width = 5574 - (left_margin = 32 + tiff_bps);
			if (tiff_bps == 12)
				load_flags = 80;
		}
		else if (raw_width == 5664) // Samsung "NX mini"
		{
			height -= top_margin = 17;
			left_margin = 96;
			width = 5544;
			filters = 0x49494949;
		}
		else if (raw_width == 6496) // Samsung NX1, NX500
		{
			filters = 0x61616161;
			if (!black && !cblack[0] && !cblack[1] && !cblack[2] && !cblack[3])
				black = 1 << (tiff_bps - 7);
		}
		else if (!strcmp(normalized_model, "EX1")) // Samsung EX1; raw_width: 3688
		{
			order = 0x4949;
			height -= 20;
			top_margin = 2;
			if ((width -= 6) > 3682)
			{
				height -= 10;
				width -= 46;
				top_margin = 8;
			}
		}
		else if (!strcmp(normalized_model, "WB2000")) // Samsung WB2000; raw_width: 3728
		{
			order = 0x4949;
			height -= 3;
			top_margin = 2;
			if ((width -= 10) > 3718)
			{
				height -= 28;
				width -= 56;
				top_margin = 8;
			}
		}
		else if (!strcmp(model, "WB550")) // Samsung WB550; raw_width: 4000
		{
			order = 0x4949;
		}
		else if (!strcmp(model, "EX2F")) // Samsung EX2F; raw_width: 4176
		{
			height = 3030;
			width = 4040;
			top_margin = 15;
			left_margin = 24;
			order = 0x4949;
			filters = 0x49494949;
			load_raw = &LibRaw::unpacked_load_raw;
		}
	}

	else if (makeIs(LIBRAW_CAMERAMAKER_ST_Micro) && !strcmp(model, "STV680 VGA"))
	{
		black = 16;
	}
	else if (!strcmp(model, "N95"))
	{
		height = raw_height - (top_margin = 2);
	}
	else if (!strcmp(model, "640x480"))
	{
		gamma_curve(0.45, 4.5, 1, 255);
	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Hasselblad))
	{
		if (load_raw == &LibRaw::lossless_jpeg_load_raw)
			load_raw = &LibRaw::hasselblad_load_raw;

		if ((imHassy.SensorCode == 4) && !strncmp(model, "V96C", 4)) { // Hasselblad V96C
			strcpy(model, "V96C");
			strcpy(normalized_model, model);
			height -= (top_margin = 6);
			width -= (left_margin = 3) + 7;
			filters = 0x61616161;

		}
		else if ((imHassy.SensorCode == 9) && imHassy.uncropped) { // various Hasselblad '-39'
			height = 5444;
			width = 7248;
			top_margin = 4;
			left_margin = 7;
			filters = 0x61616161;

		}
		else if ((imHassy.SensorCode == 13) && imHassy.uncropped) { // Hasselblad H4D-40, H5D-40
			height -= 84;
			width -= 82;
			top_margin = 4;
			left_margin = 41;
			filters = 0x61616161;

		}
		else if ((imHassy.SensorCode == 11) && imHassy.uncropped) { // Hasselblad H5D-50
			height -= 84;
			width -= 82;
			top_margin = 4;
			left_margin = 41;
			filters = 0x61616161;

		}
		else if ((imHassy.SensorCode == 15) &&
			!imHassy.SensorSubCode && // Hasselblad H5D-50c
			imHassy.uncropped) {
			left_margin = 52;
			top_margin = 100;
			width = 8272;
			height = 6200;
			black = 256;

		}
		else if ((imHassy.SensorCode == 15) &&
			(imHassy.SensorSubCode == 2) && // various Hasselblad X1D cameras
			imHassy.uncropped) {
			top_margin = 96;
			height -= 96;
			left_margin = 48;
			width -= 106;
			maximum = 0xffff;
			tiff_bps = 16;

		}
		else if ((imHassy.SensorCode == 12) && imHassy.uncropped) { // Hasselblad H4D-60
			if (black > 500) { // (imHassy.format == LIBRAW_HF_FFF)
				top_margin = 12;
				left_margin = 44;
				width = 8956;
				height = 6708;
				memset(cblack, 0, sizeof(cblack));
				black = 512;
			}
			else { // (imHassy.format == LIBRAW_HF_3FR)
				top_margin = 8;
				left_margin = 40;
				width = 8964;
				height = 6716;
				black += load_flags = 256;
				maximum = 0x8101;
			}

		}
		else if ((imHassy.SensorCode == 17) && imHassy.uncropped) { // Hasselblad H6D-100c, A6D-100c
			left_margin = 64;
			width = 11608;
			top_margin = 108;
			height = raw_height - top_margin;
		}

		if (tiff_samples > 1)
		{
			is_raw = tiff_samples + 1;
			if (!shot_select && !half_size)
				filters = 0;
		}
	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Sinar))
	{
		if (!load_raw)
			load_raw = &LibRaw::unpacked_load_raw;
		if (is_raw > 1 && !shot_select)
			filters = 0;
		maximum = 0x3fff;
	}

	if (load_raw == &LibRaw::sinar_4shot_load_raw)
	{
		if (is_raw > 1 && !shot_select)
			filters = 0;
	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Leaf))
	{
		maximum = 0x3fff;
		fseek(ifp, data_offset, SEEK_SET);
		if (ljpeg_start(&jh, 1) && jh.bits == 15)
			maximum = 0x1fff;
		if (tiff_samples > 1)
			filters = 0;
		if (tiff_samples > 1 || tile_length < raw_height)
		{
			load_raw = &LibRaw::leaf_hdr_load_raw;
			raw_width = tile_width;
		}
		if ((width | height) == 2048)
		{
			if (tiff_samples == 1)
			{
				filters = 1;
				strcpy(cdesc, "RBTG");
				strcpy(model, "CatchLight");
				strcpy(normalized_model, model);
				top_margin = 8;
				left_margin = 18;
				height = 2032;
				width = 2016;
			}
			else
			{
				strcpy(model, "DCB2");
				strcpy(normalized_model, model);
				top_margin = 10;
				left_margin = 16;
				height = 2028;
				width = 2022;
			}
		}
		else if (width + height == 3144 + 2060)
		{
			if (!model[0])
			{
				strcpy(model, "Cantare");
				strcpy(normalized_model, model);
			}
			if (width > height)
			{
				top_margin = 6;
				left_margin = 32;
				height = 2048;
				width = 3072;
				filters = 0x61616161;
			}
			else
			{
				left_margin = 6;
				top_margin = 32;
				width = 2048;
				height = 3072;
				filters = 0x16161616;
			}
			if (!cam_mul[0] || model[0] == 'V')
				filters = 0;
			else
				is_raw = tiff_samples;
		}
		else if (width == 2116) // Leaf "Valeo 6"
		{
			strcpy(model, "Valeo 6");
			strcpy(normalized_model, model);
			height -= 2 * (top_margin = 30);
			width -= 2 * (left_margin = 55);
			filters = 0x49494949;
		}
		else if (width == 3171) // Leaf "Valeo 6"
		{
			strcpy(model, "Valeo 6");
			strcpy(normalized_model, model);
			height -= 2 * (top_margin = 24);
			width -= 2 * (left_margin = 24);
			filters = 0x16161616;
		}
	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Panasonic))
	{
		if (raw_width > 0 &&
			((flen - data_offset) / (raw_width * 8 / 7) == raw_height))
			load_raw = &LibRaw::panasonic_load_raw;
		if (!load_raw)
		{
			load_raw = &LibRaw::unpacked_load_raw;
			load_flags = 4;
		}
		zero_is_bad = 1;
		if ((height += 12) > raw_height)
			height = raw_height;
		for (i = 0; i < int(sizeof pana / sizeof *pana); i++)
			if (raw_width == pana[i][0] && raw_height == pana[i][1])
			{
				left_margin = pana[i][2];
				top_margin = pana[i][3];
				width += pana[i][4];
				height += pana[i][5];
			}
		if (!tiff_bps && pana_bpp >= 12 && pana_bpp <= 14)
			tiff_bps = pana_bpp;

        if (!strcmp(model, "DC-LX100M2") && raw_height == 3568 && raw_width == 4816 && filters == 3)
            filters = 4;

		filters = 0x01010101U *
			(uchar) "\x94\x61\x49\x16"[((filters - 1) ^ (left_margin & 1) ^
			(top_margin << 1)) &
			3];

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Contax) &&
		!strcmp(model, "N Digital")) {
		height = 2047;
		width = 3072;
		filters = 0x61616161;
		data_offset = 0x1a00;
		load_raw = &LibRaw::packed_load_raw;

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Sony)) {
		if (!strcmp(model, "DSC-F828")) { // Sony DSC-F828
			width = 3288;
			left_margin = 5;
			mask[1][3] = -17;
			data_offset = 862144;
			load_raw = &LibRaw::sony_load_raw;
			filters = 0x9c9c9c9c;
			colors = 4;
			strcpy(cdesc, "RGBE");

		}
		else if (!strcmp(model, "DSC-V3")) { // Sony DSC-V3
			width = 3109;
			left_margin = 59;
			mask[0][1] = 9;
			data_offset = 787392;
			load_raw = &LibRaw::sony_load_raw;

		}
		else if (raw_width == 3984) { // Sony DSC-R1;
			width = 3925;
			order = 0x4d4d;

		}
		else if (raw_width == 4288) { // Sony ILCE-7S, ILCE-7SM2, ILCE-7SM3, DSLR-A700, DSLR-A500;
			width -= 32;
		}
		else if (raw_width == 4600) { // Sony DSLR-A290, DSLR-A350, DSLR-A380;
			if (!strcmp(model, "DSLR-A350"))
				height -= 4;
			black = 0;

		}
		else if (raw_width == 4928) {
			// Sony DSLR-A580, NEX-C3, SLT-A35, DSC-HX99, SLT-A55,
			// NEX-5N, SLT-A37, SLT-A57, NEX-F3, NEX-6, NEX-5R, NEX-3N, NEX-5T;
			if (height < 3280)
				width -= 8;

		}
		else if (raw_width == 5504) {
			// Sony ILCE-3000, SLT-A58, DSC-RX100M3, ILCE-QX1,
			// DSC-RX10M4, DSC-RX100M6, DSC-RX100, DSC-RX100M2, DSC-RX10,
			// ILCE-5000, DSC-RX100M4, DSC-RX10M2, DSC-RX10M3,
			// DSC-RX100M5, DSC-RX100M5A;
			width -= height > 3664 ? 8 : 32;

		}
		else if (raw_width == 6048) {
			// Sony SLT-A65, DSC-RX1, SLT-A77, DSC-RX1, ILCA-77M2,
			// ILCE-7M3, NEX-7, SLT-A99, ILCE-7, DSC-RX1R, ILCE-6000,
			// ILCE-5100, ILCE-7M2, ILCA-68, ILCE-6300, ILCE-9,
			// ILCE-6500, ILCE-6400;
			width -= 24;
			if (strstr(normalized_model, "RX1") ||
				strstr(normalized_model, "A99"))
				width -= 6;

		}
		else if (raw_width == 7392) { // Sony ILCE-7R;
			width -= 30;

		}
		else if (raw_width == 8000) {
			// Sony ILCE-7RM2, ILCE-7RM2, ILCE-7RM3, DSC-RX1RM2, ILCA-99M2;
			width -= 32;

		}
		else if (raw_width == 9600) { // Sony ILCE-7RM4
			width -= 32;

		}
        else if(unique_id == SonyID_ILCE_1)
        {
          if (raw_width == 8704 && raw_height == 6144) // ILCE-1 FF@Compressed
          {
            width = 8660;
            height = 5784;
          }
          else if (raw_width == 8672) // FF uncompressed/lossy
          {
            width -= 12;
          }
          else if (raw_width == 6144 && raw_height == 4096) // APSC/Lossless
          {
            width = 5636;
            height = 3768;
          }
          else if (raw_width == 5664) // APS-C/Uncompressed or lossy
          {
              width -= 28;
          }
        }
        else if (unique_id == SonyID_ILCE_7M4)
        {
          if (raw_width == 7168 && raw_height == 5120) // ILCE-1 FF@Compressed
          {
            width = 7028;
            height = 4688;
          }
          else if (raw_width == 7040) // FF uncompressed/lossy
          {
            width -= 12;
          }
          /* FIXME: need APS-C samples, both losslesscompressed and uncompressed or lossy */
        }

        else if (!strcmp(model, "DSLR-A100")) {
			if (width == 3880) {
				height--;
				width = ++raw_width;
			}
			else {
				height -= 4;
				width -= 4;
				order = 0x4d4d;
				load_flags = 2;
			}
			filters = 0x61616161;
		}
	}

	else if (!strcmp(model, "PIXL")) {
		height -= top_margin = 4;
		width -= left_margin = 32;
		gamma_curve(0, 7, 1, 255);

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Kodak)) {

		if (!strncasecmp(model, "EasyShare", 9)) {
			data_offset = data_offset < 0x15000 ? 0x15000 : 0x17000;
			load_raw = &LibRaw::packed_load_raw;

		}
		else if (!strcmp(model, "C603") ||
			!strcmp(model, "C330") ||
			!strcmp(model, "12MP")) {
			order = 0x4949;
			if (filters && data_offset) {
				fseek(ifp, data_offset < 4096 ? 168 : 5252, SEEK_SET);
				read_shorts(curve, 256);
			}
			else
				gamma_curve(0, 3.875, 1, 255);

			load_raw = filters ? &LibRaw::eight_bit_load_raw
				: strcmp(model, "C330") ? &LibRaw::kodak_c603_load_raw
				: &LibRaw::kodak_c330_load_raw;
			load_flags = tiff_bps > 16;
			tiff_bps = 8;

		}
		else {
			if (!strncmp(model, "NC2000", 6) ||
				!strncmp(model, "EOSDCS", 6) ||
				!strncmp(model, "DCS4", 4)) {
				width -= 4;
				left_margin = 2;

			}
			else if (!strcmp(model, "DCS660M")) {
				black = 214;

			}
			else if (!strcmp(model, "EOS D2000C")) {
				filters = 0x61616161;
				if (!black) black = curve[200];
			}

			if (filters == UINT_MAX) filters = 0x61616161;

			if (!strcmp(model + 4, "20X"))
				strcpy(cdesc, "MYCY");
			if (!strcmp(model, "DC25")) {
				data_offset = 15424;
			}

			if (!strncmp(model, "DC2", 3)) {
				raw_height = 2 + (height = 242);
				if (!strncmp(model, "DC290", 5))
					iso_speed = 100;
				if (!strncmp(model, "DC280", 5))
					iso_speed = 70;
				if (flen < 100000) {
					raw_width = 256;
					width = 249;
					pixel_aspect = (4.0 * height) / (3.0 * width);
				}
				else {
					raw_width = 512;
					width = 501;
					pixel_aspect = (493.0 * height) / (373.0 * width);
				}
				top_margin = left_margin = 1;
				colors = 4;
				filters = 0x8d8d8d8d;
				simple_coeff(1);
				pre_mul[1] = 1.179f;
				pre_mul[2] = 1.209f;
				pre_mul[3] = 1.036f;
				load_raw = &LibRaw::eight_bit_load_raw;
			}
			else if (!strcmp(model, "DC40")) {
				height = 512;
				width = 768;
				data_offset = 1152;
				load_raw = &LibRaw::kodak_radc_load_raw;
				tiff_bps = 12;
				FORC4 cam_mul[c] = 1.0f;

			}
			else if (!strcmp(model, "DC50")) {
				height = 512;
				width = 768;
				iso_speed = 84;
				data_offset = 19712;
				load_raw = &LibRaw::kodak_radc_load_raw;
				FORC4 cam_mul[c] = 1.0f;

			}
			else if (!strcmp(model, "DC120")) {
				raw_height = height = 976;
				raw_width = width = 848;
				iso_speed = 160;
				pixel_aspect = height / 0.75 / width;
				load_raw = tiff_compress == 7 ? &LibRaw::kodak_jpeg_load_raw
					: &LibRaw::kodak_dc120_load_raw;

			}
			else if (!strcmp(model, "DCS200")) {
				thumb_height = 128;
				thumb_width = 192;
				thumb_offset = 6144;
				thumb_misc = 360;
				iso_speed = 140;
				thumb_format = LIBRAW_INTERNAL_THUMBNAIL_LAYER;
				black = 17;
			}
		}

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Logitech) &&
		!strcmp(model, "Fotoman Pixtura")) {
		height = 512;
		width = 768;
		data_offset = 3632;
		load_raw = &LibRaw::kodak_radc_load_raw;
		filters = 0x61616161;
		simple_coeff(2);

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Apple) &&
		!strncmp(model, "QuickTake", 9)) {
		if (head[5]) {
			strcpy(model + 10, "200");
			strcpy(normalized_model, model);
		}
		fseek(ifp, 544, SEEK_SET);
		height = get2();
		width = get2();
		data_offset = (get4(), get2()) == 30 ? 738 : 736;
		if (height > width) {
			SWAP(height, width);
			fseek(ifp, data_offset - 6, SEEK_SET);
			flip = ~get2() & 3 ? 5 : 6;
		}
		filters = 0x61616161;

	}
	else if (makeIs(LIBRAW_CAMERAMAKER_Rollei) &&
		!load_raw) {
		switch (raw_width) {
		case 1316: // Rollei d530flex
			height = 1030;
			width = 1300;
			top_margin = 1;
			left_margin = 6;
			break;
		case 2568:
			height = 1960;
			width = 2560;
			top_margin = 2;
			left_margin = 8;
		}
		filters = 0x16161616;
		load_raw = &LibRaw::rollei_load_raw;

	}
	else if (!strcmp(model, "GRAS-50S5C")) {
		height = 2048;
		width = 2440;
		load_raw = &LibRaw::unpacked_load_raw;
		data_offset = 0;
		filters = 0x49494949;
		order = 0x4949;
		maximum = 0xfffC;

	}
	else if (!strcmp(model, "BB-500CL")) {
		height = 2058;
		width = 2448;
		load_raw = &LibRaw::unpacked_load_raw;
		data_offset = 0;
		filters = 0x94949494;
		order = 0x4949;
		maximum = 0x3fff;

	}
	else if (!strcmp(model, "BB-500GE")) {
		height = 2058;
		width = 2456;
		load_raw = &LibRaw::unpacked_load_raw;
		data_offset = 0;
		filters = 0x94949494;
		order = 0x4949;
		maximum = 0x3fff;

	}
	else if (!strcmp(model, "SVS625CL")) {
		height = 2050;
		width = 2448;
		load_raw = &LibRaw::unpacked_load_raw;
		data_offset = 0;
		filters = 0x94949494;
		order = 0x4949;
		maximum = 0x0fff;
	}
}
