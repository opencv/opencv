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
#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__

#include "precomp.hpp"

namespace cv
{
namespace omnidir
{
    void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec, 
                       InputArray K, InputArray D, double xi, OutputArray jacobian = noArray());

    void undistortPoints(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, 
        double xi, InputArray R);
    
    void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double xi);

    void initUndistortRectifyMap(InputArray K, InputArray D, double xi, InputArray R, InputArray P, 
        const cv::Size& size, int mltype, OutputArray map1, OutputArray map2);
    
    void undistortImage(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, 
        double xi, InputArray Knew, const Size& new_size);

    double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size size, 
        InputOutputArray K, double& xi, InputOutputArray D, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll,
        int flags, TermCriteria criteria);

    double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, 
        Size imageSize, InputOutputArray K1, double& xi1, InputOutputArray D1, InputOutputArray K2, double& xi2, 
        InputOutputArray D2, OutputArray R, OutputArray T, int flags, TermCriteria criteria);

    void stereoRectify(InputArray K1, InputArray D1, double xi1, InputArray K2, InputArray D2, double xi2, const Size imageSize,
        InputArray R, InputArray tvec, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags,
        const Size& newImageSize);

namespace internal
{
    void initializeCalibration(InputOutputArrayOfArrays objectPoints, InputOutputArrayOfArrays imagePoints, Size size, 
        OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray K, double& xi);

    void computeJacobian(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters,
        Mat& JTJ_inv, Mat& JTE);

    void encodeParameters(InputArray K, InputArrayOfArrays omAll, InputArrayOfArrays tAll, InputArray distortion,
        double xi, OutputArray parameters);

    void decodeParameters(InputArray paramsters, OutputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll,
        OutputArray distortion, double& xi);

    void estimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters,
        Vec2d& std_error, double& rms);

    double computeMeanReproErr(InputArrayOfArrays imagePoints, InputArrayOfArrays proImagePoints);

    double median(InputArray row);

} // internal
} // omnidir
} //cv
#endif