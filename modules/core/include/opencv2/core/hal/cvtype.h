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
// Copyright (C) 2000-2018, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Hamdi Sahloul hamdi[at]sahloul[dot]me
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

#ifndef OPENCV_CORE_HAL_CVTYPE_H
#define OPENCV_CORE_HAL_CVTYPE_H

/*
 * MSVC evaluates both sides of conditional expressions at compile time!
 * Lets disable that buggy behaviour, and depend on other compilers to detect real issues.
 */
#ifdef _MSC_VER
#pragma warning( disable: 4293 ) /* shift count negative or too big, undefined behavior */
#endif

//! @name OpenCV4 Data types
//! @sa cv::Types
//! @{

//ElemType register specifications
#define CV_DEPTH_SHIFT            0
#define CV_CN_SHIFT               5
#define CV_CN_LEN                 7
#define CV_CN_EXP_LEN             3
#define CV_CN_BASE_LEN            (CV_CN_LEN - CV_CN_EXP_LEN)
#define CV_DEPTH_LEN              (CV_CN_SHIFT - CV_DEPTH_SHIFT)

//ElemType exponent power limit (private)
#define __CV_CN_LIMIT_EXP(n)      (1 << (CV_CN_BASE_LEN + n))

//ElemType register masks (private)
#define CV_SANITY_DEPTH_MASK      ((1 << CV_DEPTH_LEN  ) - 1)
#define CV_SANITY_CN_MASK         ((1 << CV_CN_LEN     ) - 1)
#define CV_SANITY_CN_EXP_MASK     ((1 << CV_CN_EXP_LEN ) - 1)
#define CV_SANITY_CN_BASE_MASK    (__CV_CN_LIMIT_EXP(0)  - 1)
#define CV_MAT_DEPTH_MASK         (CV_SANITY_DEPTH_MASK << CV_DEPTH_SHIFT)
#define CV_MAT_CN_MASK            (CV_SANITY_CN_MASK    << CV_CN_SHIFT   )
#define CV_MAT_TYPE_MASK          (CV_MAT_CN_MASK | CV_MAT_DEPTH_MASK)

//ElemType register limits (public)
#define CV_DEPTH_MAX              (1 << CV_DEPTH_LEN)
#define CV_CN_MAX                 __CV_CN_LIMIT_EXP(CV_SANITY_CN_EXP_MASK) /*(1 << CV_CN_LEN )*/

//ElemType register deflate/inflate operations (private)
#define __CV_CN_MAKE_EXPONENT(cn) ((cn) <= __CV_CN_LIMIT_EXP(0) ? 0 :          \
                                  ((cn) <= __CV_CN_LIMIT_EXP(1) ? 1 :          \
                                  ((cn) <= __CV_CN_LIMIT_EXP(2) ? 2 :          \
                                  ((cn) <= __CV_CN_LIMIT_EXP(3) ? 3 :          \
                                  ((cn) <= __CV_CN_LIMIT_EXP(4) ? 4 :          \
                                  ((cn) <= __CV_CN_LIMIT_EXP(5) ? 5 :          \
                                  ((cn) <= __CV_CN_LIMIT_EXP(6) ? 6 : 7)))))))
#define __CV_CN_EXPONENT(cn_bin)  (((cn_bin) >> CV_CN_BASE_LEN) & CV_SANITY_CN_EXP_MASK)
#define __CV_CN_MAKE_BASE(cn)     (((cn) - 1) >> __CV_CN_MAKE_EXPONENT(cn)) /*& CV_SANITY_CN_BASE_MASK*/
#define __CV_CN_BASE(cn_bin)      ((cn_bin) & CV_SANITY_CN_BASE_MASK)
#define __CV_CN_DEFLATE(cn)       ((__CV_CN_MAKE_EXPONENT(cn) << CV_CN_BASE_LEN) | (__CV_CN_MAKE_BASE(cn) & CV_SANITY_CN_BASE_MASK)) /*((cn) - 1)*/
#define __CV_CN_INFLATE__(e,b)    ((b + 1) << e)
#define __CV_CN_INFLATE(cn_bin)   __CV_CN_INFLATE__(__CV_CN_EXPONENT(cn_bin), __CV_CN_BASE(cn_bin)) /*((cn_bin) + 1)*/

//ElemType register pack/unpack operations (private)
#define __CV_DEPTH_PACK(depth)    (((depth) & CV_SANITY_DEPTH_MASK) << CV_DEPTH_SHIFT)
#define __CV_DEPTH_UNPACK(flags)  (((flags) & CV_MAT_DEPTH_MASK)  >> CV_DEPTH_SHIFT)
#define __CV_CN_PACK(cn)          ((__CV_CN_DEFLATE(cn) & CV_SANITY_CN_MASK) << CV_CN_SHIFT)
#define __CV_CN_UNPACK(flags)     __CV_CN_INFLATE(((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT)
/*
TODO: static assertion on bad input channel
TODO: static assertion on cn == cn to avoid random multiple evaluation*/
#define __CV_TYPE_PACK(depth,cn)  (__CV_DEPTH_PACK(depth) | __CV_CN_PACK(cn))
/*TODO: Rewite __CV_TYPE_UNPACK*/
#define __CV_TYPE_UNPACK(flags)   __CV_TYPE_PACK(__CV_DEPTH_UNPACK(flags),__CV_CN_UNPACK(flags)) /*((flags) & CV_MAT_TYPE_MASK)*/

//ElemType register - user operations (public)
#define CV_CN_FIT(cn)             __CV_CN_INFLATE__(__CV_CN_MAKE_EXPONENT(cn), __CV_CN_MAKE_BASE(cn))
#define CV_MAT_TYPE(flags)        __CV_TYPE_UNPACK(flags)
#define CV_MAT_CN(flags)          __CV_CN_UNPACK(flags)
#define CV_MAT_DEPTH(flags)       __CV_DEPTH_UNPACK(flags)
#define CV_MAKETYPE(depth,cn)     __CV_TYPE_PACK(depth,cn)
#define CV_MAKE_TYPE              CV_MAKETYPE

//depth values (public)
#define CV_8U        0
#define CV_16U       2
#define CV_32U       8
#define CV_64U       9

#define CV_8S        1
#define CV_16S       3
#define CV_32S       4
#define CV_64S       11

#define CV_16F       7
#define CV_32F       5
#define CV_64F       6

#define CV_RAW       29
#define CV_AUTO      30
#define CV_UNDEF     31

//sizeof, fraction-bits, sign-bit of depth values (public)
//sizeof(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, CV_16F) = 1, 1, 2, 2, 4, 4, 8, 2 = 0x28442211
#define CV_ELEM_SIZE1(type)     (CV_MAT_DEPTH(type) <  8 ? ((0x28442211 >>  CV_MAT_DEPTH(type)       * 4) & 15) :     \
                                (CV_MAT_DEPTH(type) < 16 ? ((0x00008084 >> (CV_MAT_DEPTH(type) -  8) * 4) & 15) : 0))
#define CV_ELEM_EXP_BITS(type)  (CV_MAT_DEPTH(type) <  8 ? ((0x5B800000 >>  CV_MAT_DEPTH(type)       * 4) & 15) : 0)
#define CV_ELEM_SIGN_BIT(type)  (CV_MAT_DEPTH(type) <  8 ? ((0x11111010 >>  CV_MAT_DEPTH(type)       * 4) & 15) :     \
                                (CV_MAT_DEPTH(type) < 16 ? ((0x00001000 >> (CV_MAT_DEPTH(type) -  8) * 4) & 15) : 0))

#define CV_ELEM_SIZE(type)      (CV_MAT_CN(type)*CV_ELEM_SIZE1(type))

#define CV_USRTYPE1 (void)"CV_USRTYPE1 support has been dropped in OpenCV 4.0"

//ElemType values (public)
#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))

#define CV_32UC1 CV_MAKETYPE(CV_32U,1)
#define CV_32UC2 CV_MAKETYPE(CV_32U,2)
#define CV_32UC3 CV_MAKETYPE(CV_32U,3)
#define CV_32UC4 CV_MAKETYPE(CV_32U,4)
#define CV_32UC(n) CV_MAKETYPE(CV_32U,(n))

#define CV_64UC1 CV_MAKETYPE(CV_64U,1)
#define CV_64UC2 CV_MAKETYPE(CV_64U,2)
#define CV_64UC3 CV_MAKETYPE(CV_64U,3)
#define CV_64UC4 CV_MAKETYPE(CV_64U,4)
#define CV_64UC(n) CV_MAKETYPE(CV_64U,(n))

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))

#define CV_64SC1 CV_MAKETYPE(CV_64S,1)
#define CV_64SC2 CV_MAKETYPE(CV_64S,2)
#define CV_64SC3 CV_MAKETYPE(CV_64S,3)
#define CV_64SC4 CV_MAKETYPE(CV_64S,4)
#define CV_64SC(n) CV_MAKETYPE(CV_64S,(n))

#define CV_16FC1 CV_MAKETYPE(CV_16F,1)
#define CV_16FC2 CV_MAKETYPE(CV_16F,2)
#define CV_16FC3 CV_MAKETYPE(CV_16F,3)
#define CV_16FC4 CV_MAKETYPE(CV_16F,4)
#define CV_16FC(n) CV_MAKETYPE(CV_16F,(n))

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F,(n))

//! @}

#endif
