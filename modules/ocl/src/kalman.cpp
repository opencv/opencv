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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//     Jin Ma, jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

using namespace cv;
using namespace cv::ocl;

KalmanFilter::KalmanFilter()
{

}

KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams, int type)
{
    init(dynamParams, measureParams, controlParams, type);
}

void KalmanFilter::init(int DP, int MP, int CP, int type)
{
    CV_Assert( DP > 0 && MP > 0 );
    CV_Assert( type == CV_32F || type == CV_64F );
    CP = cv::max(CP, 0);

    statePre.create(DP, 1, type);
    statePre.setTo(Scalar::all(0));

    statePost.create(DP, 1, type);
    statePost.setTo(Scalar::all(0));

    transitionMatrix.create(DP, DP, type);
    setIdentity(transitionMatrix, 1);

    processNoiseCov.create(DP, DP, type);
    setIdentity(processNoiseCov, 1);

    measurementNoiseCov.create(MP, MP, type);
    setIdentity(measurementNoiseCov, 1);

    measurementMatrix.create(MP, DP, type);
    measurementMatrix.setTo(Scalar::all(0));

    errorCovPre.create(DP, DP, type);
    errorCovPre.setTo(Scalar::all(0));

    errorCovPost.create(DP, DP, type);
    errorCovPost.setTo(Scalar::all(0));

    gain.create(DP, MP, type);
    gain.setTo(Scalar::all(0));

    if( CP > 0 )
    {
        controlMatrix.create(DP, CP, type);
        controlMatrix.setTo(Scalar::all(0));
    }
    else
        controlMatrix.release();

    temp1.create(DP, DP, type);
    temp2.create(MP, DP, type);
    temp3.create(MP, MP, type);
    temp4.create(MP, DP, type);
    temp5.create(MP, 1, type);
}

CV_EXPORTS const oclMat& KalmanFilter::predict(const oclMat& control)
{
    gemm(transitionMatrix, statePost, 1, oclMat(), 0, statePre);
    oclMat temp;

    if(control.data)
        gemm(controlMatrix, control, 1, statePre, 1, statePre);
    gemm(transitionMatrix, errorCovPost, 1, oclMat(), 0, temp1);
    gemm(temp1, transitionMatrix, 1, processNoiseCov, 1, errorCovPre, GEMM_2_T);
    statePre.copyTo(statePost);
    return statePre;
}

CV_EXPORTS const oclMat& KalmanFilter::correct(const oclMat& measurement)
{
    CV_Assert(measurement.empty() == false);
    gemm(measurementMatrix, errorCovPre, 1, oclMat(), 0, temp2);
    gemm(temp2, measurementMatrix, 1, measurementNoiseCov, 1, temp3, GEMM_2_T);
    Mat temp;
    solve(Mat(temp3), Mat(temp2), temp, DECOMP_SVD);
    temp4.upload(temp);
    gain = temp4.t();
    gemm(measurementMatrix, statePre, -1, measurement, 1, temp5);
    gemm(gain, temp5, 1, statePre, 1, statePost);
    gemm(gain, temp2, -1, errorCovPre, 1, errorCovPost);
    return statePost;
}
