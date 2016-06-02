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

#include "test_precomp.hpp"
#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D

using namespace cv;
using namespace std;

TEST(SurfFeaturesFinder, CanFindInROIs)
{
    Ptr<detail::FeaturesFinder> finder = makePtr<detail::SurfFeaturesFinder>();
    Mat img  = imread(string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/lena.png");

    vector<Rect> rois;
    rois.push_back(Rect(0, 0, img.cols / 2, img.rows / 2));
    rois.push_back(Rect(img.cols / 2, img.rows / 2, img.cols - img.cols / 2, img.rows - img.rows / 2));
    detail::ImageFeatures roi_features;
    (*finder)(img, roi_features, rois);

    int tl_rect_count = 0, br_rect_count = 0, bad_count = 0;
    for (size_t i = 0; i < roi_features.keypoints.size(); ++i)
    {
        Point2f pt = roi_features.keypoints[i].pt;
        if (pt.x >= rois[0].x && pt.y >= rois[0].y && pt.x <= rois[0].br().x && pt.y <= rois[0].br().y)
            tl_rect_count++;
        else if (pt.x >= rois[1].x && pt.y >= rois[1].y && pt.x <= rois[1].br().x && pt.y <= rois[1].br().y)
            br_rect_count++;
        else
            bad_count++;
    }

    ASSERT_GT(tl_rect_count, 0);
    ASSERT_GT(br_rect_count, 0);
    ASSERT_EQ(bad_count, 0);
}

#endif
