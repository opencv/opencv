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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
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

#include "precomp.hpp"

#include <deque>

#include "opencv2/core/hal/intrin.hpp"

#include "canny.hpp"
#include "canny.simd.hpp"
#include "canny.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX512_SKX,...,BASELINE based on CMakeLists.txt content

namespace cv {

int canny_simd_width()
{
    CV_CPU_DISPATCH(canny_simd_width, (), CV_CPU_DISPATCH_MODES_ALL);
}

void canny_calc_magnitude(const short* _dx, const short* _dy, int* _mag_n,
                          int width, bool L2gradient)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(canny_calc_magnitude, (_dx, _dy, _mag_n, width, L2gradient),
        CV_CPU_DISPATCH_MODES_ALL);
}

void canny_nms_row(const int* _mag_a, const int* _mag_p, const int* _mag_n,
                   const short* _dx, const short* _dy, uchar* _pmap,
                   int width, int low, int high, std::deque<uchar*>& stack)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(canny_nms_row, (_mag_a, _mag_p, _mag_n, _dx, _dy, _pmap, width, low, high, stack),
        CV_CPU_DISPATCH_MODES_ALL);
}

void canny_finalize_row(const uchar* pmap, uchar* pdst, int width)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(canny_finalize_row, (pmap, pdst, width),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
