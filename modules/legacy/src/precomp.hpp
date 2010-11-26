/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#if _MSC_VER >= 1200
#pragma warning( disable: 4251 4710 4711 4514 4996 )
#endif

#ifdef HAVE_CONFIG_H 
#include <cvconfig.h> 
#endif

#include "opencv2/legacy/legacy.hpp"

#include "opencv2/core/internal.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"

#include "opencv2/legacy/blobtrack.hpp"
#include "opencv2/legacy/compat.hpp"

#include "_matrix.h"

typedef unsigned short ushort;

CV_INLINE bool operator == (CvSize size1, CvSize size2 );
CV_INLINE bool operator == (CvSize size1, CvSize size2 )
{
    return size1.width == size2.width && size1.height == size2.height;
}

CV_INLINE bool operator != (CvSize size1, CvSize size2 );
CV_INLINE bool operator != (CvSize size1, CvSize size2 )
{
    return size1.width != size2.width || size1.height != size2.height;
}

#endif /* __CVAUX_H__ */
