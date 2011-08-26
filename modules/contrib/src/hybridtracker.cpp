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

CvHybridTracker::CvHybridTracker() {

}

CvHybridTracker::CvHybridTracker(HybridTrackerParams _params) :
	params(_params) {
	params.ft_params.feature_type = CvFeatureTrackerParams::SIFT;
	mstracker = new CvMeanShiftTracker(params.ms_params);
	fttracker = new CvFeatureTracker(params.ft_params);
}

CvHybridTracker::~CvHybridTracker() {
	if (mstracker != NULL)
		delete mstracker;
	if (fttracker != NULL)
		delete fttracker;
}

inline float CvHybridTracker::getL2Norm(Point2f p1, Point2f p2) {
	float distance = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y
			- p2.y);
	return sqrt(distance);
}

Mat CvHybridTracker::getDistanceProjection(Mat image, Point2f center) {
	Mat hist(image.size(), CV_64F);

	double lu = getL2Norm(Point(0, 0), center);
	double ru = getL2Norm(Point(0, image.size().width), center);
	double rd = getL2Norm(Point(image.size().height, image.size().width),
			center);
	double ld = getL2Norm(Point(image.size().height, 0), center);

	double max = (lu < ru) ? lu : ru;
	max = (max < rd) ? max : rd;
	max = (max < ld) ? max : ld;

	for (int i = 0; i < hist.rows; i++)
		for (int j = 0; j < hist.cols; j++)
			hist.at<double> (i, j) = 1.0 - (getL2Norm(Point(i, j), center)
					/ max);

	return hist;
}

Mat CvHybridTracker::getGaussianProjection(Mat image, int ksize, double sigma,
		Point2f center) {
	Mat kernel = getGaussianKernel(ksize, sigma, CV_64F);
	double max = kernel.at<double> (ksize / 2);

	Mat hist(image.size(), CV_64F);
	for (int i = 0; i < hist.rows; i++)
		for (int j = 0; j < hist.cols; j++) {
			int pos = getL2Norm(Point(i, j), center);
			if (pos < ksize / 2.0)
				hist.at<double> (i, j) = 1.0 - (kernel.at<double> (pos) / max);
		}

	return hist;
}

void CvHybridTracker::newTracker(Mat image, Rect selection) {
	prev_proj = Mat::zeros(image.size(), CV_64FC1);
	prev_center = Point2f(selection.x + selection.width / 2.0, selection.y
			+ selection.height / 2.0);
	prev_window = selection;

	mstracker->newTrackingWindow(image, selection);
	fttracker->newTrackingWindow(image, selection);

	params.em_params.covs = NULL;
	params.em_params.means = NULL;
	params.em_params.probs = NULL;
	params.em_params.nclusters = 1;
	params.em_params.weights = NULL;
	params.em_params.cov_mat_type = CvEM::COV_MAT_SPHERICAL;
	params.em_params.start_step = CvEM::START_AUTO_STEP;
	params.em_params.term_crit.max_iter = 10000;
	params.em_params.term_crit.epsilon = 0.001;
	params.em_params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	samples = cvCreateMat(2, 1, CV_32FC1);
	labels = cvCreateMat(2, 1, CV_32SC1);

	ittr = 0;
}

void CvHybridTracker::updateTracker(Mat image) {
	ittr++;

	//copy over clean images: TODO
	mstracker->updateTrackingWindow(image);
	fttracker->updateTrackingWindowWithFlow(image);

	if (params.motion_model == CvMotionModel::EM)
		updateTrackerWithEM(image);
	else
		updateTrackerWithLowPassFilter(image);

	// Regression to find new weights
	Point2f ms_center = mstracker->getTrackingEllipse().center;
	Point2f ft_center = fttracker->getTrackingCenter();

#ifdef DEBUG_HYTRACKER
	circle(image, ms_center, 3, Scalar(0, 0, 255), -1, 8);
	circle(image, ft_center, 3, Scalar(255, 0, 0), -1, 8);
	putText(image, "ms", Point(ms_center.x+2, ms_center.y), FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 255, 255));
	putText(image, "ft", Point(ft_center.x+2, ft_center.y), FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 255, 255));
#endif

	double ms_len = getL2Norm(ms_center, curr_center);
	double ft_len = getL2Norm(ft_center, curr_center);
	double total_len = ms_len + ft_len;

	params.ms_tracker_weight *= (ittr - 1);
	params.ms_tracker_weight += (ms_len / total_len);
	params.ms_tracker_weight /= ittr;
	params.ft_tracker_weight *= (ittr - 1);
	params.ft_tracker_weight += (ft_len / total_len);
	params.ft_tracker_weight /= ittr;

	circle(image, prev_center, 3, Scalar(0, 0, 0), -1, 8);
	circle(image, curr_center, 3, Scalar(255, 255, 255), -1, 8);

	prev_center = curr_center;
	prev_window.x = (int)(curr_center.x-prev_window.width/2.0);
	prev_window.y = (int)(curr_center.y-prev_window.height/2.0);

	mstracker->setTrackingWindow(prev_window);
	fttracker->setTrackingWindow(prev_window);
}

void CvHybridTracker::updateTrackerWithEM(Mat image) {
	Mat ms_backproj = mstracker->getHistogramProjection(CV_64F);
	Mat ms_distproj = getDistanceProjection(image, mstracker->getTrackingCenter());
	Mat ms_proj = ms_backproj.mul(ms_distproj);

	float dist_err = getL2Norm(mstracker->getTrackingCenter(), fttracker->getTrackingCenter());
	Mat ft_gaussproj = getGaussianProjection(image, dist_err, -1, fttracker->getTrackingCenter());
	Mat ft_distproj = getDistanceProjection(image, fttracker->getTrackingCenter());
	Mat ft_proj = ft_gaussproj.mul(ft_distproj);

	Mat proj = params.ms_tracker_weight * ms_proj + params.ft_tracker_weight * ft_proj + prev_proj;

	int sample_count = countNonZero(proj);
	cvReleaseMat(&samples);
	cvReleaseMat(&labels);
	samples = cvCreateMat(sample_count, 2, CV_32FC1);
	labels = cvCreateMat(sample_count, 1, CV_32SC1);

	int count = 0;
	for (int i = 0; i < proj.rows; i++)
		for (int j = 0; j < proj.cols; j++)
			if (proj.at<double> (i, j) > 0) {
				samples->data.fl[count * 2] = i;
				samples->data.fl[count * 2 + 1] = j;
				count++;
			}

	em_model.train(samples, 0, params.em_params, labels);

	curr_center.x = em_model.getMeans().at<double> (0, 0);
	curr_center.y = em_model.getMeans().at<double> (0, 1);
}

void CvHybridTracker::updateTrackerWithLowPassFilter(Mat image) {
	RotatedRect ms_track = mstracker->getTrackingEllipse();
	Point2f ft_center = fttracker->getTrackingCenter();

	float a = params.low_pass_gain;
	curr_center.x = (1.0 - a) * prev_center.x + a * (params.ms_tracker_weight * ms_track.center.x + params.ft_tracker_weight * ft_center.x);
	curr_center.y = (1.0 - a) * prev_center.y + a * (params.ms_tracker_weight * ms_track.center.y + params.ft_tracker_weight * ft_center.y);
}

Rect CvHybridTracker::getTrackingWindow() {
	return prev_window;
}

