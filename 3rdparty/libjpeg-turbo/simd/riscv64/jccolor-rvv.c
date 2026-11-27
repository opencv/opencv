/*
 * RGB-to-YCbCr Color Conversion (64-bit RVV 1.0)
 *
 * Copyright (C) 2011, 2014, 2026, D. R. Commander.
 * Copyright (C) 2022, Institute of Software, Chinese Academy of Sciences.
 *                     Author:  Zhiyuan Tan
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "../jsimdint.h"
#include <riscv_vector.h>


/* RGB -> YCbCr conversion constants */

#define F_0_081  5329   /* FIX(0.08131) */
#define F_0_114  7471   /* FIX(0.11400) */
#define F_0_168  11059  /* FIX(0.16874) */
#define F_0_299  19595  /* FIX(0.29900) */
#define F_0_331  21709  /* FIX(0.33126) */
#define F_0_418  27439  /* FIX(0.41869) */
#define F_0_500  32768  /* FIX(0.50000) */
#define F_0_587  38470  /* FIX(0.58700) */

#define SCALEBITS  16
#define ONE_HALF  (1 << (SCALEBITS - 1))
#define SCALED_CENTERJSAMPLE  (CENTERJSAMPLE << SCALEBITS)


#include "jccolext-rvv.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define jsimd_rgb_ycc_convert_rvv  jsimd_extrgb_ycc_convert_rvv
#include "jccolext-rvv.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_rvv

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define jsimd_rgb_ycc_convert_rvv  jsimd_extrgbx_ycc_convert_rvv
#include "jccolext-rvv.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_rvv

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define jsimd_rgb_ycc_convert_rvv  jsimd_extbgr_ycc_convert_rvv
#include "jccolext-rvv.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_rvv

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define jsimd_rgb_ycc_convert_rvv  jsimd_extbgrx_ycc_convert_rvv
#include "jccolext-rvv.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_rvv

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define jsimd_rgb_ycc_convert_rvv  jsimd_extxbgr_ycc_convert_rvv
#include "jccolext-rvv.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_rvv

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define jsimd_rgb_ycc_convert_rvv  jsimd_extxrgb_ycc_convert_rvv
#include "jccolext-rvv.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef jsimd_rgb_ycc_convert_rvv
