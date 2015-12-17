/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CORE_HAL_REPLACEMENT_HPP__
#define __OPENCV_CORE_HAL_REPLACEMENT_HPP__

#include "opencv2/core/hal/interface.h"

inline int hal_ni_add8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add16s(const short*, size_t, const short*, size_t, short*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add32s(const int*, size_t, const int*, size_t, int*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add32f(const float*, size_t, const float*, size_t, float*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add64f(const double*, size_t, const double*, size_t, double*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub16s(const short*, size_t, const short*, size_t, short*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub32s(const int*, size_t, const int*, size_t, int*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub32f(const float*, size_t, const float*, size_t, float*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub64f(const double*, size_t, const double*, size_t, double*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max16s(const short*, size_t, const short*, size_t, short*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max32s(const int*, size_t, const int*, size_t, int*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max32f(const float*, size_t, const float*, size_t, float*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max64f(const double*, size_t, const double*, size_t, double*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min16s(const short*, size_t, const short*, size_t, short*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min32s(const int*, size_t, const int*, size_t, int*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min32f(const float*, size_t, const float*, size_t, float*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min64f(const double*, size_t, const double*, size_t, double*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff16s(const short*, size_t, const short*, size_t, short*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff32s(const int*, size_t, const int*, size_t, int*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff32f(const float*, size_t, const float*, size_t, float*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff64f(const double*, size_t, const double*, size_t, double*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_and8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_or8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_xor8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_not8u(const uchar*, size_t, uchar*, size_t, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_add8u hal_ni_add8u
#define cv_hal_add8s hal_ni_add8s
#define cv_hal_add16u hal_ni_add16u
#define cv_hal_add16s hal_ni_add16s
#define cv_hal_add32s hal_ni_add32s
#define cv_hal_add32f hal_ni_add32f
#define cv_hal_add64f hal_ni_add64f
#define cv_hal_sub8u hal_ni_sub8u
#define cv_hal_sub8s hal_ni_sub8s
#define cv_hal_sub16u hal_ni_sub16u
#define cv_hal_sub16s hal_ni_sub16s
#define cv_hal_sub32s hal_ni_sub32s
#define cv_hal_sub32f hal_ni_sub32f
#define cv_hal_sub64f hal_ni_sub64f
#define cv_hal_max8u hal_ni_max8u
#define cv_hal_max8s hal_ni_max8s
#define cv_hal_max16u hal_ni_max16u
#define cv_hal_max16s hal_ni_max16s
#define cv_hal_max32s hal_ni_max32s
#define cv_hal_max32f hal_ni_max32f
#define cv_hal_max64f hal_ni_max64f
#define cv_hal_min8u hal_ni_min8u
#define cv_hal_min8s hal_ni_min8s
#define cv_hal_min16u hal_ni_min16u
#define cv_hal_min16s hal_ni_min16s
#define cv_hal_min32s hal_ni_min32s
#define cv_hal_min32f hal_ni_min32f
#define cv_hal_min64f hal_ni_min64f
#define cv_hal_absdiff8u hal_ni_absdiff8u
#define cv_hal_absdiff8s hal_ni_absdiff8s
#define cv_hal_absdiff16u hal_ni_absdiff16u
#define cv_hal_absdiff16s hal_ni_absdiff16s
#define cv_hal_absdiff32s hal_ni_absdiff32s
#define cv_hal_absdiff32f hal_ni_absdiff32f
#define cv_hal_absdiff64f hal_ni_absdiff64f
#define cv_hal_and8u hal_ni_and8u
#define cv_hal_or8u hal_ni_or8u
#define cv_hal_xor8u hal_ni_xor8u
#define cv_hal_not8u hal_ni_not8u

inline int hal_ni_cmp8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp8s(const schar*, size_t, const schar*, size_t, uchar*, size_t, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp16u(const ushort*, size_t, const ushort*, size_t, uchar*, size_t, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp16s(const short*, size_t, const short*, size_t, uchar*, size_t, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp32s(const int*, size_t, const int*, size_t, uchar*, size_t, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp32f(const float*, size_t, const float*, size_t, uchar*, size_t, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp64f(const double*, size_t, const double*, size_t, uchar*, size_t, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_cmp8u hal_ni_cmp8u
#define cv_hal_cmp8s hal_ni_cmp8s
#define cv_hal_cmp16u hal_ni_cmp16u
#define cv_hal_cmp16s hal_ni_cmp16s
#define cv_hal_cmp32s hal_ni_cmp32s
#define cv_hal_cmp32f hal_ni_cmp32f
#define cv_hal_cmp64f hal_ni_cmp64f

inline int hal_ni_mul8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul16s(const short*, size_t, const short*, size_t, short*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul32s(const int*, size_t, const int*, size_t, int*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul32f(const float*, size_t, const float*, size_t, float*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul64f(const double*, size_t, const double*, size_t, double*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div16s(const short*, size_t, const short*, size_t, short*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div32s(const int*, size_t, const int*, size_t, int*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div32f(const float*, size_t, const float*, size_t, float*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div64f(const double*, size_t, const double*, size_t, double*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip16s(const short*, size_t, const short*, size_t, short*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip32s(const int*, size_t, const int*, size_t, int*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip32f(const float*, size_t, const float*, size_t, float*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip64f(const double*, size_t, const double*, size_t, double*, size_t, int, int, double) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_mul8u hal_ni_mul8u
#define cv_hal_mul8s hal_ni_mul8s
#define cv_hal_mul16u hal_ni_mul16u
#define cv_hal_mul16s hal_ni_mul16s
#define cv_hal_mul32s hal_ni_mul32s
#define cv_hal_mul32f hal_ni_mul32f
#define cv_hal_mul64f hal_ni_mul64f
#define cv_hal_div8u hal_ni_div8u
#define cv_hal_div8s hal_ni_div8s
#define cv_hal_div16u hal_ni_div16u
#define cv_hal_div16s hal_ni_div16s
#define cv_hal_div32s hal_ni_div32s
#define cv_hal_div32f hal_ni_div32f
#define cv_hal_div64f hal_ni_div64f
#define cv_hal_recip8u hal_ni_recip8u
#define cv_hal_recip8s hal_ni_recip8s
#define cv_hal_recip16u hal_ni_recip16u
#define cv_hal_recip16s hal_ni_recip16s
#define cv_hal_recip32s hal_ni_recip32s
#define cv_hal_recip32f hal_ni_recip32f
#define cv_hal_recip64f hal_ni_recip64f

inline int hal_ni_addWeighted8u(const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int, const double*) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted8s(const schar*, size_t, const schar*, size_t, schar*, size_t, int, int, const double*) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted16u(const ushort*, size_t, const ushort*, size_t, ushort*, size_t, int, int, const double*) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted16s(const short*, size_t, const short*, size_t, short*, size_t, int, int, const double*) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted32s(const int*, size_t, const int*, size_t, int*, size_t, int, int, const double*) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted32f(const float*, size_t, const float*, size_t, float*, size_t, int, int, const double*) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted64f(const double*, size_t, const double*, size_t, double*, size_t, int, int, const double*) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_addWeighted8u hal_ni_addWeighted8u
#define cv_hal_addWeighted8s hal_ni_addWeighted8s
#define cv_hal_addWeighted16u hal_ni_addWeighted16u
#define cv_hal_addWeighted16s hal_ni_addWeighted16s
#define cv_hal_addWeighted32s hal_ni_addWeighted32s
#define cv_hal_addWeighted32f hal_ni_addWeighted32f
#define cv_hal_addWeighted64f hal_ni_addWeighted64f

inline int hal_ni_split8u(const uchar*, uchar**, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split16u(const ushort*, ushort**, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split32s(const int*, int**, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split64s(const int64*, int64**, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_split8u hal_ni_split8u
#define cv_hal_split16u hal_ni_split16u
#define cv_hal_split32s hal_ni_split32s
#define cv_hal_split64s hal_ni_split64s

inline int hal_ni_merge8u(const uchar**, uchar*, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge16u(const ushort**, ushort*, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge32s(const int**, int*, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge64s(const int64**, int64*, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_merge8u hal_ni_merge8u
#define cv_hal_merge16u hal_ni_merge16u
#define cv_hal_merge32s hal_ni_merge32s
#define cv_hal_merge64s hal_ni_merge64s

#include "custom_hal.hpp"

#endif
