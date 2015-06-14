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
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
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
#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__
#ifdef __cplusplus
#include "precomp.hpp"

namespace cv
{
namespace omnidir
{

namespace internal
{
    void initializeCalibration(InputOutputArrayOfArrays objectPoints, InputOutputArrayOfArrays imagePoints, Size size, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray K, double& xi);
    void computeJacobian(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters, Mat& JTJ_inv, Mat& JTE);
    void encodeParameters(InputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, InputArray distoaration, double xi, int n, OutputArray parameters);
    void decodeParameters(InputArray paramsters, OutputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray distoration, double& xi);
    void estimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters, Mat& errors, Vec2d& std_error, double& rms, int flags);
    double computeMeanReproerr(InputArrayOfArrays imagePoints, InputArrayOfArrays proImagePoints);
    void checkFixed(Mat &G, int flags, int n);
} // internal

    
} // omnidir

} //cv
#endif
#endif