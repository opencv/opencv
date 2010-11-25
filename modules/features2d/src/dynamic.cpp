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
 // Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
namespace cv
{

DynamicAdaptedFeatureDetector::DynamicAdaptedFeatureDetector(const Ptr<AdjusterAdapter>& a,
                                         int min_features, int max_features, int max_iters ) :
        escape_iters_(max_iters), min_features_(min_features), max_features_(max_features), adjuster_(a)
{}

void DynamicAdaptedFeatureDetector::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
	//for oscillation testing
	bool down = false;
	bool up = false;

	//flag for whether the correct threshhold has been reached
	bool thresh_good = false;

	//this is bad but adjuster should persist from detection to detection
	AdjusterAdapter& adjuster = const_cast<AdjusterAdapter&> (*adjuster_);

	//break if the desired number hasn't been reached.
	int iter_count = escape_iters_;

    do
    {
		keypoints.clear();

		//the adjuster takes care of calling the detector with updated parameters
		adjuster.detect(image, keypoints,mask);

        if (int(keypoints.size()) < min_features_)
        {
			down = true;
			adjuster.tooFew(min_features_, keypoints.size());
        }
        else if (int(keypoints.size()) > max_features_)
        {
			up = true;
			adjuster.tooMany(max_features_, keypoints.size());
        }
        else
			thresh_good = true;
    }
    while (--iter_count >= 0 && !(down && up) && !thresh_good && adjuster.good());
}

FastAdjuster::FastAdjuster(int init_thresh, bool nonmax) :
    thresh_(init_thresh), nonmax_(nonmax)
{}

void FastAdjuster::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
	FastFeatureDetector(thresh_, nonmax_).detect(image, keypoints, mask);
}

void FastAdjuster::tooFew(int min, int n_detected)
{
	//fast is easy to adjust
	thresh_--;
}

void FastAdjuster::tooMany(int max, int n_detected)
{
	//fast is easy to adjust
	thresh_++;
}

//return whether or not the threshhold is beyond
//a useful point
bool FastAdjuster::good() const
{
	return (thresh_ > 1) && (thresh_ < 200);
}

StarAdjuster::StarAdjuster(double initial_thresh) :
    thresh_(initial_thresh)
{}

void StarAdjuster::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
	StarFeatureDetector detector_tmp(16, thresh_, 10, 8, 3);
	detector_tmp.detect(image, keypoints, mask);
}

void StarAdjuster::tooFew(int min, int n_detected)
{
	thresh_ *= 0.9;
	if (thresh_ < 1.1)
		thresh_ = 1.1;
}

void StarAdjuster::tooMany(int max, int n_detected)
{
	thresh_ *= 1.1;
}

bool StarAdjuster::good() const
{
	return (thresh_ > 2) && (thresh_ < 200);
}

SurfAdjuster::SurfAdjuster() :
    thresh_(400.0)
{}

void SurfAdjuster::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const cv::Mat& mask) const
{
	SurfFeatureDetector detector_tmp(thresh_);
	detector_tmp.detect(image, keypoints, mask);
}

void SurfAdjuster::tooFew(int min, int n_detected)
{
	thresh_ *= 0.9;
	if (thresh_ < 1.1)
		thresh_ = 1.1;
}

void SurfAdjuster::tooMany(int max, int n_detected)
{
	thresh_ *= 1.1;
}

//return whether or not the threshhold is beyond
//a useful point
bool SurfAdjuster::good() const
{
	return (thresh_ > 2) && (thresh_ < 1000);
}

Ptr<AdjusterAdapter> AdjusterAdapter::create( const string& detectorType )
{
    Ptr<AdjusterAdapter> adapter;

    if( !detectorType.compare( "FAST" ) )
    {
        adapter = new FastAdjuster();
    }
    else if( !detectorType.compare( "STAR" ) )
    {
        adapter = new StarAdjuster();
    }
    else if( !detectorType.compare( "SURF" ) )
    {
        adapter = new SurfAdjuster();
    }

    return adapter;
}

}
