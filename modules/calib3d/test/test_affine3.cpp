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
 // Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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
 //     and / or other materials provided with the distribution.
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

#include "test_precomp.hpp"
#include "opencv2/core/affine.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>

TEST(Calib3d_Affine3f, accuracy)
{
    cv::Vec3d rvec(0.2, 0.5, 0.3);
    cv::Affine3d affine(rvec);

    cv::Mat expected;
    cv::Rodrigues(rvec, expected);


    ASSERT_EQ(0, norm(cv::Mat(affine.matrix, false).colRange(0, 3).rowRange(0, 3) != expected));
    ASSERT_EQ(0, norm(cv::Mat(affine.linear()) != expected));


    cv::Matx33d R = cv::Matx33d::eye();

    double angle = 50;
    R.val[0] = R.val[4] = std::cos(CV_PI*angle/180.0);
    R.val[3] = std::sin(CV_PI*angle/180.0);
    R.val[1] = -R.val[3];


    cv::Affine3d affine1(cv::Mat(cv::Vec3d(0.2, 0.5, 0.3)).reshape(1, 1), cv::Vec3d(4, 5, 6));
    cv::Affine3d affine2(R, cv::Vec3d(1, 1, 0.4));

    cv::Affine3d result = affine1.inv() * affine2;

    expected = cv::Mat(affine1.matrix.inv(cv::DECOMP_SVD)) * cv::Mat(affine2.matrix, false);


    cv::Mat diff;
    cv::absdiff(expected, result.matrix, diff);

    ASSERT_LT(cv::norm(diff, cv::NORM_INF), 1e-15);
}

TEST(Calib3d_Affine3f, accuracy_rvec)
{
    cv::RNG rng;
    typedef float T;

    cv::Affine3<T>::Vec3 w;
    cv::Affine3<T>::Mat3 u, vt, R;

    for(int i = 0; i < 100; ++i)
    {
        rng.fill(R, cv::RNG::UNIFORM, -10, 10, true);
        cv::SVD::compute(R, w, u, vt, cv::SVD::FULL_UV + cv::SVD::MODIFY_A);
        R = u * vt;

        //double s = (double)cv::getTickCount();
        cv::Affine3<T>::Vec3 va = cv::Affine3<T>(R).rvec();
        //std::cout << "M:" <<(cv::getTickCount() - s)*1000/cv::getTickFrequency() << std::endl;

        cv::Affine3<T>::Vec3 vo;
        //s = (double)cv::getTickCount();
        cv::Rodrigues(R, vo);
        //std::cout << "O:" <<(cv::getTickCount() - s)*1000/cv::getTickFrequency() << std::endl;

        ASSERT_LT(cv::norm(va - vo), 1e-9);
    }
}
