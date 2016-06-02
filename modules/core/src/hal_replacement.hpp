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

#ifndef OPENCV_CORE_HAL_REPLACEMENT_HPP
#define OPENCV_CORE_HAL_REPLACEMENT_HPP

#include "opencv2/core/hal/interface.h"

#if defined __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined _MSC_VER
#  pragma warning( push )
#  pragma warning( disable: 4100 )
#endif

//! @addtogroup core_hal_interface
//! @note Define your functions to override default implementations:
//! @code
//! #undef hal_add8u
//! #define hal_add8u my_add8u
//! @endcode
//! @{

/**
Add: _dst[i] = src1[i] + src2[i]_ @n
Sub: _dst[i] = src1[i] - src2[i]_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
*/
//! @addtogroup core_hal_interface_addsub Element-wise add and subtract
//! @{
inline int hal_ni_add8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_add64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

inline int hal_ni_sub8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sub64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Minimum: _dst[i] = min(src1[i], src2[i])_ @n
Maximum: _dst[i] = max(src1[i], src2[i])_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
*/
//! @addtogroup core_hal_interface_minmax Element-wise minimum or maximum
//! @{
inline int hal_ni_max8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_max64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

inline int hal_ni_min8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_min64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Absolute difference: _dst[i] = | src1[i] - src2[i] |_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
@param scale additional multiplier
*/
//! @addtogroup core_hal_interface_absdiff Element-wise absolute difference
//! @{
inline int hal_ni_absdiff8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_absdiff64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Bitwise AND: _dst[i] = src1[i] & src2[i]_ @n
Bitwise OR: _dst[i] = src1[i] | src2[i]_ @n
Bitwise XOR: _dst[i] = src1[i] ^ src2[i]_ @n
Bitwise NOT: _dst[i] = !src[i]_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
 */
//! @addtogroup core_hal_interface_logical Bitwise logical operations
//! @{
inline int hal_ni_and8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_or8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_xor8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_not8u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
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
//! @endcond

/**
Compare: _dst[i] = src1[i] op src2[i]_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
@param operation one of (CV_HAL_CMP_EQ, CV_HAL_CMP_GT, ...)
*/
//! @addtogroup core_hal_interface_compare Element-wise compare
//! @{
inline int hal_ni_cmp8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_cmp64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_cmp8u hal_ni_cmp8u
#define cv_hal_cmp8s hal_ni_cmp8s
#define cv_hal_cmp16u hal_ni_cmp16u
#define cv_hal_cmp16s hal_ni_cmp16s
#define cv_hal_cmp32s hal_ni_cmp32s
#define cv_hal_cmp32f hal_ni_cmp32f
#define cv_hal_cmp64f hal_ni_cmp64f
//! @endcond

/**
Multiply: _dst[i] = scale * src1[i] * src2[i]_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
@param scale additional multiplier
*/
//! @addtogroup core_hal_interface_multiply Element-wise multiply
//! @{
inline int hal_ni_mul8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_mul64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Divide: _dst[i] = scale * src1[i] / src2[i]_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
@param scale additional multiplier
*/
//! @addtogroup core_hal_interface_divide Element-wise divide
//! @{
inline int hal_ni_div8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_div64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

/**
Computes reciprocial: _dst[i] = scale / src[i]_
@param src_data,src_step source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
@param scale additional multiplier
 */
//! @addtogroup core_hal_interface_reciprocial Element-wise reciprocial
//! @{
inline int hal_ni_recip8u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip8s(const schar *src_data, size_t src_step, schar *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip16u(const ushort *src_data, size_t src_step, ushort *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip16s(const short *src_data, size_t src_step, short *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip32s(const int *src_data, size_t src_step, int *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip32f(const float *src_data, size_t src_step, float *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_recip64f(const double *src_data, size_t src_step, double *dst_data, size_t dst_step, int width, int height, double scale) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
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
//! @endcond

/**
Computes weighted sum of two arrays using formula: _dst[i] = a * src1[i] + b * src2[i] + c_
@param src1_data,src1_step first source image data and step
@param src2_data,src2_step second source image data and step
@param dst_data,dst_step destination image data and step
@param width,height dimensions of the images
@param scalars numbers _a_, _b_, and _c_
 */
//! @addtogroup core_hal_interface_addWeighted Element-wise weighted sum
//! @{
inline int hal_ni_addWeighted8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_addWeighted64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height, const double scalars[3]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_addWeighted8u hal_ni_addWeighted8u
#define cv_hal_addWeighted8s hal_ni_addWeighted8s
#define cv_hal_addWeighted16u hal_ni_addWeighted16u
#define cv_hal_addWeighted16s hal_ni_addWeighted16s
#define cv_hal_addWeighted32s hal_ni_addWeighted32s
#define cv_hal_addWeighted32f hal_ni_addWeighted32f
#define cv_hal_addWeighted64f hal_ni_addWeighted64f
//! @endcond

/**
@param src_data array of interleaved values (__len__ x __cn__ items) [ B, G, R, B, G, R, ...]
@param dst_data array of pointers to destination arrays (__cn__ items x __len__ items) [ [B, B, ...], [G, G, ...], [R, R, ...] ]
@param len number of elements
@param cn number of channels
 */
//! @addtogroup core_hal_interface_split Channel split
//! @{
inline int hal_ni_split8u(const uchar *src_data, uchar **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split16u(const ushort *src_data, ushort **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split32s(const int *src_data, int **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_split64s(const int64 *src_data, int64 **dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_split8u hal_ni_split8u
#define cv_hal_split16u hal_ni_split16u
#define cv_hal_split32s hal_ni_split32s
#define cv_hal_split64s hal_ni_split64s
//! @endcond

/**
@param src_data array of pointers to source arrays (__cn__ items x __len__ items) [ [B, B, ...], [G, G, ...], [R, R, ...] ]
@param dst_data destination array of interleaved values (__len__ x __cn__ items) [ B, G, R, B, G, R, ...]
@param len number of elements
@param cn number of channels
 */
//! @addtogroup core_hal_interface_merge Channel merge
//! @{
inline int hal_ni_merge8u(const uchar **src_data, uchar *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge16u(const ushort **src_data, ushort *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge32s(const int **src_data, int *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_merge64s(const int64 **src_data, int64 *dst_data, int len, int cn) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
//! @}

//! @cond IGNORED
#define cv_hal_merge8u hal_ni_merge8u
#define cv_hal_merge16u hal_ni_merge16u
#define cv_hal_merge32s hal_ni_merge32s
#define cv_hal_merge64s hal_ni_merge64s
//! @endcond

/**
@brief Dummy structure storing DFT/DCT context

Users can convert this pointer to any type they want. Initialisation and destruction should be made in Init and Free function implementations correspondingly.
Example:
@code{.cpp}
int my_hal_dftInit2D(cvhalDFT **context, ...) {
    *context = static_cast<cvhalDFT*>(new MyFilterData());
    //... init
}

int my_hal_dftFree2D(cvhalDFT *context) {
    MyFilterData *c = static_cast<MyFilterData*>(context);
    delete c;
}
@endcode
 */
struct cvhalDFT {};

/**
@param context double pointer to context storing all necessary data
@param len transformed array length
@param count estimated transformation count
@param depth array type (CV_32F or CV_64F)
@param flags algorithm options (combination of CV_HAL_DFT_INVERSE, CV_HAL_DFT_SCALE, ...)
@param needBuffer pointer to boolean variable, if valid pointer provided, then variable value should be set to true to signal that additional memory buffer is needed for operations
 */
inline int hal_ni_dftInit1D(cvhalDFT **context, int len, int count, int depth, int flags, bool *needBuffer) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
@param src source data
@param dst destination data
 */
inline int hal_ni_dft1D(cvhalDFT *context, const uchar *src, uchar *dst) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
 */
inline int hal_ni_dftFree1D(cvhalDFT *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dftInit1D hal_ni_dftInit1D
#define cv_hal_dft1D hal_ni_dft1D
#define cv_hal_dftFree1D hal_ni_dftFree1D
//! @endcond

/**
@param context double pointer to context storing all necessary data
@param width,height image dimensions
@param depth image type (CV_32F or CV64F)
@param src_channels number of channels in input image
@param dst_channels number of channels in output image
@param flags algorithm options (combination of CV_HAL_DFT_INVERSE, ...)
@param nonzero_rows number of nonzero rows in image, can be used for optimization
 */
inline int hal_ni_dftInit2D(cvhalDFT **context, int width, int height, int depth, int src_channels, int dst_channels, int flags, int nonzero_rows) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
@param src_data,src_step source image data and step
@param dst_data,dst_step destination image data and step
 */
inline int hal_ni_dft2D(cvhalDFT *context, const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
 */
inline int hal_ni_dftFree2D(cvhalDFT *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dftInit2D hal_ni_dftInit2D
#define cv_hal_dft2D hal_ni_dft2D
#define cv_hal_dftFree2D hal_ni_dftFree2D
//! @endcond

/**
@param context double pointer to context storing all necessary data
@param width,height image dimensions
@param depth image type (CV_32F or CV64F)
@param flags algorithm options (combination of CV_HAL_DFT_INVERSE, ...)
 */
inline int hal_ni_dctInit2D(cvhalDFT **context, int width, int height, int depth, int flags) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
@param src_data,src_step source image data and step
@param dst_data,dst_step destination image data and step
 */
inline int hal_ni_dct2D(cvhalDFT *context, const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
@param context pointer to context storing all necessary data
 */
inline int hal_ni_dctFree2D(cvhalDFT *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dctInit2D hal_ni_dctInit2D
#define cv_hal_dct2D hal_ni_dct2D
#define cv_hal_dctFree2D hal_ni_dctFree2D
//! @endcond

//! @}

#if defined __GNUC__
#  pragma GCC diagnostic pop
#elif defined _MSC_VER
#  pragma warning( pop )
#endif

#include "custom_hal.hpp"

#endif
