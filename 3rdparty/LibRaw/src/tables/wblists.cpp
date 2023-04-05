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


#define _ARR_SZ(a) (sizeof(a)/sizeof(a[0]))

static const int _tagtype_dataunit_bytes [19] = {
    1, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 2, 8, 8, 8, 8
};

libraw_static_table_t LibRaw::tagtype_dataunit_bytes(_tagtype_dataunit_bytes, _ARR_SZ(_tagtype_dataunit_bytes));

int libraw_tagtype_dataunit_bytes(int tagtype)
{
    return _tagtype_dataunit_bytes[((unsigned)tagtype <= _ARR_SZ(_tagtype_dataunit_bytes)) ? tagtype : 0];
}


static const int _Canon_wbi2std[] = { // Canon WB index to standard indexes
//      std. number                wbi - Canon number
    LIBRAW_WBI_Auto,             // 0
    LIBRAW_WBI_Daylight,         // 1
    LIBRAW_WBI_Cloudy,           // 2
    LIBRAW_WBI_Tungsten,         // 3
    LIBRAW_WBI_Fluorescent,      // 4
    LIBRAW_WBI_Flash,            // 5
    LIBRAW_WBI_Custom,           // 6
    LIBRAW_WBI_BW,               // 7
    LIBRAW_WBI_Shade,            // 8
    LIBRAW_WBI_Kelvin,           // 9
    LIBRAW_WBI_PC_Set1,          // 10
    LIBRAW_WBI_PC_Set2,          // 11
    LIBRAW_WBI_PC_Set3,          // 12
    LIBRAW_WBI_Unknown,          // 13, unlucky number "13", not used
    LIBRAW_WBI_FluorescentHigh,  // 14
    LIBRAW_WBI_Custom1,          // 15
    LIBRAW_WBI_Custom2,          // 16
    LIBRAW_WBI_Underwater,       // 17, last one for older PowerShot models
    LIBRAW_WBI_Custom3,          // 18
    LIBRAW_WBI_Custom4,          // 19
    LIBRAW_WBI_PC_Set4,          // 20
    LIBRAW_WBI_PC_Set5,          // 21
    LIBRAW_WBI_Unknown,          // 22
    LIBRAW_WBI_Auto1             // 23
};

libraw_static_table_t LibRaw::Canon_wbi2std(_Canon_wbi2std, _ARR_SZ(_Canon_wbi2std));

static const int _Canon_KeyIsZero_Len2048_linenums_2_StdWBi[] = { // Appendix A: G2, S30, S40; G3, G5, S45, S50
  LIBRAW_WBI_Custom1,
  LIBRAW_WBI_Custom2,
  LIBRAW_WBI_Daylight,
  LIBRAW_WBI_Cloudy,
  LIBRAW_WBI_Tungsten,
  LIBRAW_WBI_Fluorescent,
  LIBRAW_WBI_Unknown, // ? FluorescentHigh, Shade, Custom, Kelvin
  LIBRAW_WBI_Flash
};

libraw_static_table_t LibRaw::Canon_KeyIsZero_Len2048_linenums_2_StdWBi(_Canon_KeyIsZero_Len2048_linenums_2_StdWBi,
    _ARR_SZ(_Canon_KeyIsZero_Len2048_linenums_2_StdWBi));

static const int _Canon_KeyIs0x0410_Len3072_linenums_2_StdWBi[] = { // G6, S60, S70; offset +16
  LIBRAW_WBI_Custom1,
  LIBRAW_WBI_Custom2,
  LIBRAW_WBI_Daylight,
  LIBRAW_WBI_Cloudy,
  LIBRAW_WBI_Tungsten,
  LIBRAW_WBI_Fluorescent,
  LIBRAW_WBI_FluorescentHigh, // LIBRAW_WBI_Unknown, // ? FluorescentHigh, Shade, Custom, Kelvin
  LIBRAW_WBI_Unknown,
  LIBRAW_WBI_Underwater, // LIBRAW_WBI_Unknown,
  LIBRAW_WBI_Unknown,
  LIBRAW_WBI_Flash
};

libraw_static_table_t LibRaw::Canon_KeyIs0x0410_Len3072_linenums_2_StdWBi(_Canon_KeyIs0x0410_Len3072_linenums_2_StdWBi,
    _ARR_SZ(_Canon_KeyIs0x0410_Len3072_linenums_2_StdWBi));

static const int _Canon_KeyIs0x0410_Len2048_linenums_2_StdWBi[] = { // Pro1; offset +8
  LIBRAW_WBI_Custom1,
  LIBRAW_WBI_Custom2,
  LIBRAW_WBI_Daylight,
  LIBRAW_WBI_Cloudy,
  LIBRAW_WBI_Tungsten,
  LIBRAW_WBI_Fluorescent,
  LIBRAW_WBI_Unknown,
  LIBRAW_WBI_Flash, // LIBRAW_WBI_Unknown,
  LIBRAW_WBI_Unknown,
  LIBRAW_WBI_Unknown,
  LIBRAW_WBI_Unknown // LIBRAW_WBI_Flash
};

libraw_static_table_t LibRaw::Canon_KeyIs0x0410_Len2048_linenums_2_StdWBi(_Canon_KeyIs0x0410_Len2048_linenums_2_StdWBi,
    _ARR_SZ(_Canon_KeyIs0x0410_Len2048_linenums_2_StdWBi));

static const int _Canon_G9_linenums_2_StdWBi[] = {
    LIBRAW_WBI_Auto,
    LIBRAW_WBI_Daylight,
    LIBRAW_WBI_Cloudy,
    LIBRAW_WBI_Tungsten,
    LIBRAW_WBI_Fluorescent,
    LIBRAW_WBI_FluorescentHigh,
    LIBRAW_WBI_Flash,
    LIBRAW_WBI_Underwater,
    LIBRAW_WBI_Custom1,
    LIBRAW_WBI_Custom2
};
libraw_static_table_t LibRaw::Canon_G9_linenums_2_StdWBi(_Canon_G9_linenums_2_StdWBi, _ARR_SZ(_Canon_G9_linenums_2_StdWBi));

static const int _Canon_D30_linenums_2_StdWBi[] = {
    LIBRAW_WBI_Daylight,
    LIBRAW_WBI_Cloudy,
    LIBRAW_WBI_Tungsten,
    LIBRAW_WBI_Fluorescent,
    LIBRAW_WBI_Flash,
    LIBRAW_WBI_Custom
};
libraw_static_table_t LibRaw::Canon_D30_linenums_2_StdWBi(_Canon_D30_linenums_2_StdWBi, _ARR_SZ(_Canon_D30_linenums_2_StdWBi));

static const int _Fuji_wb_list1[] = {
    LIBRAW_WBI_FineWeather, LIBRAW_WBI_Shade, LIBRAW_WBI_FL_D,
    LIBRAW_WBI_FL_N,        LIBRAW_WBI_FL_W,  LIBRAW_WBI_Tungsten

};
libraw_static_table_t LibRaw::Fuji_wb_list1(_Fuji_wb_list1, _ARR_SZ(_Fuji_wb_list1));

static const int _FujiCCT_K[31] = {
    2500, 2550, 2650, 2700, 2800, 2850, 2950, 3000, 3100, 3200, 3300,
    3400, 3600, 3700, 3800, 4000, 4200, 4300, 4500, 4800, 5000, 5300,
    5600, 5900, 6300, 6700, 7100, 7700, 8300, 9100, 10000
};
libraw_static_table_t LibRaw::FujiCCT_K(_FujiCCT_K, _ARR_SZ(_FujiCCT_K));

static const int _Fuji_wb_list2[] = {
    LIBRAW_WBI_Auto,  0,  LIBRAW_WBI_Custom,   6,  LIBRAW_WBI_FineWeather, 1,
    LIBRAW_WBI_Shade, 8,  LIBRAW_WBI_FL_D,     10, LIBRAW_WBI_FL_N,        11,
    LIBRAW_WBI_FL_W,  12, LIBRAW_WBI_Tungsten, 2,  LIBRAW_WBI_Underwater,  35,
    LIBRAW_WBI_Ill_A, 82, LIBRAW_WBI_D65,      83
};
libraw_static_table_t LibRaw::Fuji_wb_list2(_Fuji_wb_list2, _ARR_SZ(_Fuji_wb_list2));

static const int _Pentax_wb_list1[] = {
    LIBRAW_WBI_Daylight, LIBRAW_WBI_Shade,
    LIBRAW_WBI_Cloudy,   LIBRAW_WBI_Tungsten,
    LIBRAW_WBI_FL_D,     LIBRAW_WBI_FL_N,
    LIBRAW_WBI_FL_W,     LIBRAW_WBI_Flash
};
libraw_static_table_t LibRaw::Pentax_wb_list1(_Pentax_wb_list1, _ARR_SZ(_Pentax_wb_list1));

static const int _Pentax_wb_list2[] = {
    LIBRAW_WBI_Daylight, LIBRAW_WBI_Shade, LIBRAW_WBI_Cloudy,
    LIBRAW_WBI_Tungsten, LIBRAW_WBI_FL_D,  LIBRAW_WBI_FL_N,
    LIBRAW_WBI_FL_W,     LIBRAW_WBI_Flash, LIBRAW_WBI_FL_L
};
libraw_static_table_t LibRaw::Pentax_wb_list2(_Pentax_wb_list2, _ARR_SZ(_Pentax_wb_list2));


static const int _Oly_wb_list1[] = {
    LIBRAW_WBI_Shade,    LIBRAW_WBI_Cloudy, LIBRAW_WBI_FineWeather,
    LIBRAW_WBI_Tungsten, LIBRAW_WBI_Sunset, LIBRAW_WBI_FL_D,
    LIBRAW_WBI_FL_N,     LIBRAW_WBI_FL_W,   LIBRAW_WBI_FL_WW
};
libraw_static_table_t LibRaw::Oly_wb_list1(_Oly_wb_list1, _ARR_SZ(_Oly_wb_list1));

static const int _Oly_wb_list2[] = {
    LIBRAW_WBI_Auto, 0,
    LIBRAW_WBI_Tungsten, 3000,
    0x100, 3300,
    0x100, 3600,
    0x100, 3900,
    LIBRAW_WBI_FL_W, 4000,
    0x100, 4300,
    LIBRAW_WBI_FL_D, 4500,
    0x100, 4800,
    LIBRAW_WBI_FineWeather, 5300,
    LIBRAW_WBI_Cloudy, 6000,
    LIBRAW_WBI_FL_N, 6600,
    LIBRAW_WBI_Shade, 7500,
    LIBRAW_WBI_Custom1, 0,
    LIBRAW_WBI_Custom2, 0,
    LIBRAW_WBI_Custom3, 0,
    LIBRAW_WBI_Custom4, 0
};
libraw_static_table_t LibRaw::Oly_wb_list2(_Oly_wb_list2, _ARR_SZ(_Oly_wb_list2));

static const int _Sony_SRF_wb_list[] = {
    LIBRAW_WBI_Daylight, LIBRAW_WBI_Cloudy, LIBRAW_WBI_Fluorescent,
    LIBRAW_WBI_Tungsten, LIBRAW_WBI_Flash
};
libraw_static_table_t LibRaw::Sony_SRF_wb_list(_Sony_SRF_wb_list, _ARR_SZ(_Sony_SRF_wb_list));

static const int _Sony_SR2_wb_list[] = {
    LIBRAW_WBI_Daylight, LIBRAW_WBI_Cloudy, LIBRAW_WBI_Tungsten, LIBRAW_WBI_Flash,
    4500, LIBRAW_WBI_Unknown, LIBRAW_WBI_Fluorescent
};
libraw_static_table_t LibRaw::Sony_SR2_wb_list(_Sony_SR2_wb_list, _ARR_SZ(_Sony_SR2_wb_list));

static const int _Sony_SR2_wb_list1[] = {
    LIBRAW_WBI_Daylight, LIBRAW_WBI_Cloudy, LIBRAW_WBI_Tungsten, LIBRAW_WBI_Flash,
    4500, LIBRAW_WBI_Shade, LIBRAW_WBI_FL_W, LIBRAW_WBI_FL_N, LIBRAW_WBI_FL_D,
    LIBRAW_WBI_FL_L, 8500, 6000, 3200, 2500
};
libraw_static_table_t LibRaw::Sony_SR2_wb_list1(_Sony_SR2_wb_list1, _ARR_SZ(_Sony_SR2_wb_list1));
