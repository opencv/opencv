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

#ifndef _GPU_TEST_H_
#define _GPU_TEST_H_

#if defined WIN32 || defined _WIN32
#include <windows.h>
#undef min
#undef max
#endif

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "cxts.h"

/****************************************************************************************/
/*                              Warnings Disabling                                      */
/****************************************************************************************/
#if _MSC_VER > 1000
#pragma warning(disable : 4514) /* unreferenced inline function has been */
                                /* removed                               */
#pragma warning(disable : 4127) /* conditional expression is constant    */
                                /* for no warnings in _ASSERT            */
#pragma warning(disable : 4996) /* deprecated function */
#endif


static inline bool check_and_treat_gpu_exception(const cv::Exception& e, CvTS* ts)
{
    switch (e.code)
    {
    case CV_GpuNotSupported: 
        ts->printf(CvTS::LOG, "\nGpu not supported by the library"); 
        break;

    case CV_GpuApiCallError: 
        ts->printf(CvTS::LOG, "\nGPU Error: %s", e.what());
        break;

    case CV_GpuNppCallError: 
        ts->printf(CvTS::LOG, "\nNPP Error: %s", e.what());
        break;

    default:
        return false;
    }
    ts->set_failed_test_info(CvTS::FAIL_GENERIC);                        
    return true;
}

#endif 

/* End of file. */
