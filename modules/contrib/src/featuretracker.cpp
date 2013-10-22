//*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                                License Agreement
//                       For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
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
#include "precomp.hpp"
#include <stdio.h>
#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/contrib/hybridtracker.hpp"

#ifdef HAVE_OPENCV_NONFREE
#include "opencv2/nonfree/nonfree.hpp"

static bool makeUseOfNonfree = initModule_nonfree();
#endif

using namespace cv;

CvFeatureTracker::CvFeatureTracker(CvFeatureTrackerParams _params) :
    params(_params)
{
    switch (params.feature_type)
    {
    case CvFeatureTrackerParams::SIFT:
        dd = Algorithm::create<Feature2D>("Feature2D.SIFT");
        if( !dd )
            CV_Error(CV_StsNotImplemented, "OpenCV has been compiled without SIFT support");
        dd->set("nOctaveLayers", 5);
        dd->set("contrastThreshold", 0.04);
        dd->set("edgeThreshold", 10.7);
        break;
    case CvFeatureTrackerParams::SURF:
        dd = Algorithm::create<Feature2D>("Feature2D.SURF");
        if( !dd )
            CV_Error(CV_StsNotImplemented, "OpenCV has been compiled without SURF support");
        dd->set("hessianThreshold", 400);
        dd->set("nOctaves", 3);
        dd->set("nOctaveLayers", 4);
        break;
    default:
        CV_Error(CV_StsBadArg, "Unknown feature type");
        break;
    }

    matcher = makePtr<BFMatcher>(int(NORM_L2));
}

CvFeatureTracker::~CvFeatureTracker()
{
}

void CvFeatureTracker::newTrackingWindow(Mat image, Rect selection)
{
    image.copyTo(prev_image);
    cvtColor(prev_image, prev_image_bw, COLOR_BGR2GRAY);
    prev_trackwindow = selection;
    prev_center.x = selection.x;
    prev_center.y = selection.y;
    ittr = 0;
}

Rect CvFeatureTracker::updateTrackingWindow(Mat image)
{
    if(params.feature_type == CvFeatureTrackerParams::OPTICAL_FLOW)
        return updateTrackingWindowWithFlow(image);
    else
        return updateTrackingWindowWithSIFT(image);
}

Rect CvFeatureTracker::updateTrackingWindowWithSIFT(Mat image)
{
    ittr++;
    std::vector<KeyPoint> prev_keypoints, curr_keypoints;
    std::vector<Point2f> prev_keys, curr_keys;
    Mat prev_desc, curr_desc;

    Rect window = prev_trackwindow;
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    rectangle(mask, Point(window.x, window.y), Point(window.x + window.width,
            window.y + window.height), Scalar(255), CV_FILLED);

    dd->operator()(prev_image, mask, prev_keypoints, prev_desc);

    window.x -= params.window_size;
    window.y -= params.window_size;
    window.width += params.window_size;
    window.height += params.window_size;
    rectangle(mask, Point(window.x, window.y), Point(window.x + window.width,
            window.y + window.height), Scalar(255), CV_FILLED);

    dd->operator()(image, mask, curr_keypoints, curr_desc);

    if (prev_keypoints.size() > 4 && curr_keypoints.size() > 4)
    {
        //descriptor->compute(prev_image, prev_keypoints, prev_desc);
        //descriptor->compute(image, curr_keypoints, curr_desc);

        matcher->match(prev_desc, curr_desc, matches);

        for (int i = 0; i < (int)matches.size(); i++)
        {
            prev_keys.push_back(prev_keypoints[matches[i].queryIdx].pt);
            curr_keys.push_back(curr_keypoints[matches[i].trainIdx].pt);
        }

        Mat T = findHomography(prev_keys, curr_keys, LMEDS);

        prev_trackwindow.x += cvRound(T.at<double> (0, 2));
        prev_trackwindow.y += cvRound(T.at<double> (1, 2));
    }

    prev_center.x = prev_trackwindow.x;
    prev_center.y = prev_trackwindow.y;
    prev_image = image;
    return prev_trackwindow;
}

Rect CvFeatureTracker::updateTrackingWindowWithFlow(Mat image)
{
    ittr++;
    Size subPixWinSize(10,10), winSize(31,31);
    Mat image_bw;
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
    std::vector<uchar> status;
    std::vector<float> err;

    cvtColor(image, image_bw, COLOR_BGR2GRAY);
    cvtColor(prev_image, prev_image_bw, COLOR_BGR2GRAY);

    if (ittr == 1)
    {
        Mat mask = Mat::zeros(image.size(), CV_8UC1);
        rectangle(mask, Point(prev_trackwindow.x, prev_trackwindow.y), Point(
                prev_trackwindow.x + prev_trackwindow.width, prev_trackwindow.y
                        + prev_trackwindow.height), Scalar(255), CV_FILLED);
        goodFeaturesToTrack(image_bw, features[1], 500, 0.01, 20, mask, 3, 0, 0.04);
        cornerSubPix(image_bw, features[1], subPixWinSize, Size(-1, -1), termcrit);
    }
    else
    {
        calcOpticalFlowPyrLK(prev_image_bw, image_bw, features[0], features[1],
                status, err, winSize, 3, termcrit);

        Point2f feature0_center(0, 0);
        Point2f feature1_center(0, 0);
        int goodtracks = 0;
        for (int i = 0; i < (int)features[1].size(); i++)
        {
            if (status[i] == 1)
            {
                feature0_center.x += features[0][i].x;
                feature0_center.y += features[0][i].y;
                feature1_center.x += features[1][i].x;
                feature1_center.y += features[1][i].y;
                goodtracks++;
            }
        }

        feature0_center.x /= goodtracks;
        feature0_center.y /= goodtracks;
        feature1_center.x /= goodtracks;
        feature1_center.y /= goodtracks;

        prev_center.x += (feature1_center.x - feature0_center.x);
        prev_center.y += (feature1_center.y - feature0_center.y);

        prev_trackwindow.x = (int)prev_center.x;
        prev_trackwindow.y = (int)prev_center.y;
    }

    swap(features[0], features[1]);
    image.copyTo(prev_image);
    return prev_trackwindow;
}

void CvFeatureTracker::setTrackingWindow(Rect _window)
{
    prev_trackwindow = _window;
}

Rect CvFeatureTracker::getTrackingWindow()
{
    return prev_trackwindow;
}

Point2f CvFeatureTracker::getTrackingCenter()
{
    Point2f center(0, 0);
    center.x = (float)(prev_center.x + prev_trackwindow.width/2.0);
    center.y = (float)(prev_center.y + prev_trackwindow.height/2.0);
    return center;
}
