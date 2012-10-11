/*! \file pragma_include.hpp
    \brief #pragmas for auto library linking
 */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_PRAGMA_LIB_HPP__
#define __OPENCV_PRAGMA_LIB_HPP__ 1

#if (defined(_MSC_VER) && !defined(PRAGMA_COMMENT_SUPPORT)) \
  && !defined(CVAPI_EXPORTS)
#define PRAGMA_COMMENT_SUPPORT 1
#elif !defined(PRAGMA_COMMENT_SUPPORT)
#define PRAGMA_COMMENT_SUPPORT 0
#endif

#ifndef CV_MAJOR_VERSION
#pragma message("WARM: Any OpenCV header included before pragma_lib.hpp")
#include "opencv2/core/version.hpp"
#endif

// version string which contains in library's file name such as "232"
#define OPENCV_LIBVERSTR \
  CVAUX_STR(CV_MAJOR_VERSION) \
  CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

// generate #pragma arguments string
#ifndef _DEBUG // Release
#define OPENCV_COMMENT_LIB_FNAME(name) \
comment(lib, "opencv_" name OPENCV_LIBVERSTR ".lib")
#else          // Debug
#define OPENCV_COMMENT_LIB_FNAME(name) \
comment(lib, "opencv_" name OPENCV_LIBVERSTR "d.lib")
#endif

// defined macro search,
// OPENCV_DEFINE_SEARCH(modulename) is only true
// when header file of modulename is included.
#define OPENCV_DEFINE_SEARCH(modulename) \
  ( \
  defined( __OPENCV_##modulename##_HPP__ )        || \
  defined(  _OPENCV_##modulename##_HPP_ )         || \
  defined( __OPENCV_##modulename##_H__ )          || \
  defined( __OPENCV_##modulename##_##name##_C_H ) \
  )

#ifndef OPENCV_AUTO_LINK
#define OPENCV_AUTO_LINK 1
#endif

#endif // #ifndef __OPENCV_PRAGMA_LIB_HPP__
/*! \file pragma_include.hpp
    \brief #pragmas for auto library linking
 */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_PRAGMA_LIB_HPP__
#define __OPENCV_PRAGMA_LIB_HPP__ 1

#if (defined(_MSC_VER) && !defined(PRAGMA_COMMENT_SUPPORT)) \
  && !defined(CVAPI_EXPORTS)
#define PRAGMA_COMMENT_SUPPORT 1
#elif !defined(PRAGMA_COMMENT_SUPPORT)
#define PRAGMA_COMMENT_SUPPORT 0
#endif

#ifndef CV_MAJOR_VERSION
#pragma message("WARM: Any OpenCV header included before pragma_lib.hpp")
#include "opencv2/core/version.hpp"
#endif

// version string which contains in library's file name such as "232"
#define OPENCV_LIBVERSTR \
  CVAUX_STR(CV_MAJOR_VERSION) \
  CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

// generate #pragma arguments string
#ifndef _DEBUG // Release
#define OPENCV_COMMENT_LIB_FNAME(name) \
comment(lib, "opencv_" name OPENCV_LIBVERSTR ".lib")
#else          // Debug
#define OPENCV_COMMENT_LIB_FNAME(name) \
comment(lib, "opencv_" name OPENCV_LIBVERSTR "d.lib")
#endif

// defined macro search,
// OPENCV_DEFINE_SEARCH(modulename) is only true
// when header file of modulename is included.
#define OPENCV_DEFINE_SEARCH(modulename) \
  ( \
  defined( __OPENCV_##modulename##_HPP__ )        || \
  defined(  _OPENCV_##modulename##_HPP_ )         || \
  defined( __OPENCV_##modulename##_H__ )          || \
  defined( __OPENCV_##modulename##_##name##_C_H ) \
  )

#ifndef OPENCV_AUTO_LINK
#define OPENCV_AUTO_LINK 1
#endif

#endif // #ifndef __OPENCV_PRAGMA_LIB_HPP__