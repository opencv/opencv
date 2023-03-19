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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include "../precomp.hpp"

using namespace cv;

class OptimalPnPRansacCallback : public PointSetRegistrator::Callback
{

public:
    OptimalPnPRansacCallback(Mat _cameraMatrix = Mat(3, 3, CV_64F))
        : cameraMatrix(_cameraMatrix) {}

    /* Pre: True */
    /* Post: compute _model with given points and return number of found models */
    int runKernel(InputArray scene, InputArray obj,
                  OutputArray _model) const CV_OVERRIDE
    {
        Mat opoints = scene.getMat(), ipoints = obj.getMat();

        bool correspondence = solvePnP(scene, obj, cameraMatrix, distCoeffs, rvec,
                                       tvec, useExtrinsicGuess, flags);

        Mat _local_model;
        hconcat(rvec, tvec, _local_model);
        _local_model.copyTo(_model);

        return correspondence;
    }

    /* Pre: True */
    /* Post: fill _err with projection errors */
    void computeError(InputArray _m1, InputArray _m2, InputArray _model,
                      OutputArray _err) const CV_OVERRIDE
    {

        Mat opoints = _m1.getMat(), ipoints = _m2.getMat(), model = _model.getMat();

        int i, count = opoints.checkVector(3);
        Mat _rvec = model.col(0);
        Mat _tvec = model.col(1);

        Mat projpoints(count, 2, CV_32FC1);
        projectPoints(opoints, _rvec, _tvec, cameraMatrix, distCoeffs, projpoints);

        const Point2f *ipoints_ptr = ipoints.ptr<Point2f>();
        const Point2f *projpoints_ptr = projpoints.ptr<Point2f>();

        _err.create(count, 1, CV_32FC1);
        float *err = _err.getMat().ptr<float>();

        for (i = 0; i < count; ++i)
            err[i] =
                (float)norm(Matx21f(ipoints_ptr[i] - projpoints_ptr[i]), NORM_L2SQR);
    }

    Mat cameraMatrix;
    Mat distCoeffs;
    int flags;
    bool useExtrinsicGuess;
    Mat rvec;
    Mat tvec;
};
