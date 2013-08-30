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
#include "opencv2/contrib/hybridtracker.hpp"

using namespace cv;
using namespace std;

CvMeanShiftTracker::CvMeanShiftTracker(CvMeanShiftTrackerParams _params) : params(_params)
{
}

CvMeanShiftTracker::~CvMeanShiftTracker()
{
}

void CvMeanShiftTracker::newTrackingWindow(Mat image, Rect selection)
{
    hist.release();
    int channels[] = { 0, 0 , 1, 1};
    float hrange[] = { 0, 180 };
    float srange[] = { 0, 1 };
    const float* ranges[] = {hrange, srange};

    cvtColor(image, hsv, CV_BGR2HSV);
    inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)), mask);

    hue.create(hsv.size(), CV_8UC2);
    mixChannels(&hsv, 1, &hue, 1, channels, 2);

    Mat roi(hue, selection);
    Mat mskroi(mask, selection);
    int ch[] = {0, 1};
    int chsize[] = {32, 32};
    calcHist(&roi, 1, ch, mskroi, hist, 1, chsize, ranges);
    normalize(hist, hist, 0, 255, CV_MINMAX);

    prev_trackwindow = selection;
}

RotatedRect CvMeanShiftTracker::updateTrackingWindow(Mat image)
{
    int channels[] = { 0, 0 , 1, 1};
    float hrange[] = { 0, 180 };
    float srange[] = { 0, 1 };
    const float* ranges[] = {hrange, srange};

    cvtColor(image, hsv, CV_BGR2HSV);
    inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)), mask);
    hue.create(hsv.size(), CV_8UC2);
    mixChannels(&hsv, 1, &hue, 1, channels, 2);
    int ch[] = {0, 1};
    calcBackProject(&hue, 1, ch, hist, backproj, ranges);
    backproj &= mask;

    prev_trackbox = CamShift(backproj, prev_trackwindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
    prev_trackwindow = Rect(prev_trackwindow.x - r, prev_trackwindow.y - r, prev_trackwindow.x + r,
            prev_trackwindow.y + r) & Rect(0, 0, cols, rows);

    prev_center.x = (float)(prev_trackwindow.x + prev_trackwindow.width / 2);
    prev_center.y = (float)(prev_trackwindow.y + prev_trackwindow.height / 2);

#ifdef DEBUG_HYTRACKER
    ellipse(image, prev_trackbox, Scalar(0, 0, 255), 1, CV_AA);
#endif

    return prev_trackbox;
}

Mat CvMeanShiftTracker::getHistogramProjection(int type)
{
    Mat ms_backproj_f(backproj.size(), type);
    backproj.convertTo(ms_backproj_f, type);
    return ms_backproj_f;
}

void CvMeanShiftTracker::setTrackingWindow(Rect window)
{
    prev_trackwindow = window;
}

Rect CvMeanShiftTracker::getTrackingWindow()
{
    return prev_trackwindow;
}

RotatedRect CvMeanShiftTracker::getTrackingEllipse()
{
    return prev_trackbox;
}

Point2f CvMeanShiftTracker::getTrackingCenter()
{
    return prev_center;
}
