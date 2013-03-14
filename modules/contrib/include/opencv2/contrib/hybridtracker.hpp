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

#ifndef __OPENCV_HYBRIDTRACKER_H_
#define __OPENCV_HYBRIDTRACKER_H_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/ml.hpp"

#ifdef __cplusplus

namespace cv
{

// Motion model for tracking algorithm. Currently supports objects that do not move much.
// To add Kalman filter
struct CV_EXPORTS CvMotionModel
{
    enum {LOW_PASS_FILTER = 0, KALMAN_FILTER = 1, EM = 2};

    CvMotionModel()
    {
    }

    float low_pass_gain;    // low pass gain
};

// Mean Shift Tracker parameters for specifying use of HSV channel and CamShift parameters.
struct CV_EXPORTS CvMeanShiftTrackerParams
{
    enum {  H = 0, HS = 1, HSV = 2  };
    CvMeanShiftTrackerParams(int tracking_type = CvMeanShiftTrackerParams::HS,
            CvTermCriteria term_crit = CvTermCriteria());

    int tracking_type;
    std::vector<float> h_range;
    std::vector<float> s_range;
    std::vector<float> v_range;
    CvTermCriteria term_crit;
};

// Feature tracking parameters
struct CV_EXPORTS CvFeatureTrackerParams
{
    enum {  SIFT = 0, SURF = 1, OPTICAL_FLOW = 2 };
    CvFeatureTrackerParams(int featureType = 0, int windowSize = 0)
    {
        feature_type = featureType;
        window_size = windowSize;
    }

    int feature_type; // Feature type to use
    int window_size; // Window size in pixels around which to search for new window
};

// Hybrid Tracking parameters for specifying weights of individual trackers and motion model.
struct CV_EXPORTS CvHybridTrackerParams
{
    CvHybridTrackerParams(float ft_tracker_weight = 0.5, float ms_tracker_weight = 0.5,
            CvFeatureTrackerParams ft_params = CvFeatureTrackerParams(),
            CvMeanShiftTrackerParams ms_params = CvMeanShiftTrackerParams(),
            CvMotionModel model = CvMotionModel());

    float ft_tracker_weight;
    float ms_tracker_weight;
    CvFeatureTrackerParams ft_params;
    CvMeanShiftTrackerParams ms_params;
    int motion_model;
    float low_pass_gain;
};

// Performs Camshift using parameters from MeanShiftTrackerParams
class CV_EXPORTS CvMeanShiftTracker
{
private:
    Mat hsv, hue;
    Mat backproj;
    Mat mask, maskroi;
    MatND hist;
    Rect prev_trackwindow;
    RotatedRect prev_trackbox;
    Point2f prev_center;

public:
    CvMeanShiftTrackerParams params;

    CvMeanShiftTracker();
    explicit CvMeanShiftTracker(CvMeanShiftTrackerParams _params);
    ~CvMeanShiftTracker();
    void newTrackingWindow(Mat image, Rect selection);
    RotatedRect updateTrackingWindow(Mat image);
    Mat getHistogramProjection(int type);
    void setTrackingWindow(Rect _window);
    Rect getTrackingWindow();
    RotatedRect getTrackingEllipse();
    Point2f getTrackingCenter();
};

// Performs SIFT/SURF feature tracking using parameters from FeatureTrackerParams
class CV_EXPORTS CvFeatureTracker
{
private:
    Ptr<Feature2D> dd;
    Ptr<DescriptorMatcher> matcher;
    std::vector<DMatch> matches;

    Mat prev_image;
    Mat prev_image_bw;
    Rect prev_trackwindow;
    Point2d prev_center;

    int ittr;
    std::vector<Point2f> features[2];

public:
    Mat disp_matches;
    CvFeatureTrackerParams params;

    CvFeatureTracker();
    explicit CvFeatureTracker(CvFeatureTrackerParams params);
    ~CvFeatureTracker();
    void newTrackingWindow(Mat image, Rect selection);
    Rect updateTrackingWindow(Mat image);
    Rect updateTrackingWindowWithSIFT(Mat image);
    Rect updateTrackingWindowWithFlow(Mat image);
    void setTrackingWindow(Rect _window);
    Rect getTrackingWindow();
    Point2f getTrackingCenter();
};

// Performs Hybrid Tracking and combines individual trackers using EM or filters
class CV_EXPORTS CvHybridTracker
{
private:
    CvMeanShiftTracker* mstracker;
    CvFeatureTracker* fttracker;

    CvMat* samples;
    CvMat* labels;

    Rect prev_window;
    Point2f prev_center;
    Mat prev_proj;
    RotatedRect trackbox;

    int ittr;
    Point2f curr_center;

    inline float getL2Norm(Point2f p1, Point2f p2);
    Mat getDistanceProjection(Mat image, Point2f center);
    Mat getGaussianProjection(Mat image, int ksize, double sigma, Point2f center);
    void updateTrackerWithEM(Mat image);
    void updateTrackerWithLowPassFilter(Mat image);

public:
    CvHybridTrackerParams params;
    CvHybridTracker();
    explicit CvHybridTracker(CvHybridTrackerParams params);
    ~CvHybridTracker();

    void newTracker(Mat image, Rect selection);
    void updateTracker(Mat image);
    Rect getTrackingWindow();
};

typedef CvMotionModel MotionModel;
typedef CvMeanShiftTrackerParams MeanShiftTrackerParams;
typedef CvFeatureTrackerParams FeatureTrackerParams;
typedef CvHybridTrackerParams HybridTrackerParams;
typedef CvMeanShiftTracker MeanShiftTracker;
typedef CvFeatureTracker FeatureTracker;
typedef CvHybridTracker HybridTracker;
}

#endif

#endif
