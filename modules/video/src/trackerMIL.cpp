/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

RNG TrackerMIL::rng;
/*
 *  TrackerMIL
 */

/*
 * Parameters
 */
TrackerMIL::Params::Params()
{
	samplerInitInRadius = 3;
	samplerTrackInRadius = 37.5;
	samplerSearchWinSize = 25;
	samplerInitMaxNegNum = 65;
	samplerTrackMaxPosNum = 65;
	samplerTrackMaxNegNum = 100000;
	featureSetNumFeatures = 250;
}

void TrackerMIL::Params::read( const cv::FileNode& fn )
{
	samplerInitInRadius = fn["samplerInitInRadius"];
	samplerSearchWinSize = fn["samplerSearchWinSize"];
	samplerInitInRadius = fn["samplerInitInRadius"];
	samplerTrackInRadius = fn["samplerTrackInRadius"];
	samplerTrackMaxPosNum = fn["samplerTrackMaxPosNum"];
	samplerTrackMaxNegNum = fn["samplerTrackMaxNegNum"];
	featureSetNumFeatures = fn["featureSetNumFeatures"];
}

void TrackerMIL::Params::write( cv::FileStorage& fs ) const
{
	fs << "samplerInitInRadius" << samplerInitInRadius;
	fs << "samplerSearchWinSize" << samplerSearchWinSize;
	fs << "samplerInitInRadius" << samplerInitInRadius;
	fs << "samplerTrackInRadius" << samplerTrackInRadius;
	fs << "samplerTrackMaxPosNum" << samplerTrackMaxPosNum;
	fs << "samplerTrackMaxNegNum" << samplerTrackMaxNegNum;
	fs << "featureSetNumFeatures" << featureSetNumFeatures;

}


/*
 * Constructor
 */
TrackerMIL::TrackerMIL( const TrackerMIL::Params &parameters ) :
params(parameters)
{
	initialized = false;
	rng = RNG( (int) time(0) );
}


/*
 * Destructor
 */
TrackerMIL::~TrackerMIL()
{

}

void TrackerMIL::read( const cv::FileNode& fn )
{
    params.read(fn);
}

void TrackerMIL::write( cv::FileStorage& fs ) const
{
    params.write(fs);
}

int TrackerMIL::getRandInt( const int min, const int max )
{
	return rng.uniform( min, max );
}

float TrackerMIL::getRandFloat( const float min, const float max )
{
	return rng.uniform( min, max );
}

bool TrackerMIL::initImpl( const Mat& image, const Rect& boundingBox )
{
	TrackerSamplerCSC::Params CSCparameters;
	CSCparameters.initInRad = params.samplerInitInRadius;
	CSCparameters.searchWinSize = params.samplerSearchWinSize;
	CSCparameters.initMaxNegNum = params.samplerInitMaxNegNum;
	CSCparameters.trackInPosRad = params.samplerTrackInRadius;
	CSCparameters.trackMaxPosNum = params.samplerTrackMaxPosNum;
	CSCparameters.trackMaxNegNum = params.samplerTrackMaxNegNum;

	Ptr<TrackerSamplerAlgorithm> CSCSampler = new TrackerSamplerCSC( CSCparameters );
	if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ))
		return false;

	//or add CSC sampler with default parameters
	//sampler->addTrackerSamplerAlgorithm( "CSC" );

	//Positive sampling
	Ptr<TrackerSamplerCSC> ( CSCSampler )->setMode( TrackerSamplerCSC::MODE_INIT_POS );
	sampler->sampling( image, boundingBox );
	std::vector<Mat> posSamples = sampler->getSamples();

	//Negative sampling
	Ptr<TrackerSamplerCSC> ( CSCSampler )->setMode( TrackerSamplerCSC::MODE_INIT_NEG );
	sampler->sampling( image, boundingBox );
	std::vector<Mat> negSamples = sampler->getSamples();

	//TODO compute HAAR features
	TrackerFeatureHAAR::Params HAARparameters;
	HAARparameters.numFeatures = params.featureSetNumFeatures;
	HAARparameters.rectSize = Size( boundingBox.width, boundingBox.height );
	Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
	featureSet->addTrackerFeature( trackerFeature );

	featureSet->extraction( posSamples );
	std::vector<Mat> posResponse = featureSet->getResponses();

	featureSet->extraction( negSamples );
	std::vector<Mat> negResponse = featureSet->getResponses();

	return true;
}

bool TrackerMIL::updateImpl( const Mat& image, Rect& boundingBox )
{
	return true;
}


} /* namespace cv */
