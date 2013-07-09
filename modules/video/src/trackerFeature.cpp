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

/*
 *  TrackerFeature
 */

TrackerFeature::~TrackerFeature()
{

}

void TrackerFeature::compute( const std::vector<Mat>& images, Mat& response )
{
	if( images.size() == 0 )
		return;

	computeImpl( images, response );
}


Ptr<TrackerFeature> TrackerFeature::create( const String& trackerFeatureType )
{
	if( trackerFeatureType.find("FEATURE2D") == 0 )
	{
		size_t firstSep = trackerFeatureType.find_first_of(".");
		size_t secondSep = trackerFeatureType.find_last_of(".");

		String detector = trackerFeatureType.substr( firstSep, secondSep - firstSep );
		String descriptor = trackerFeatureType.substr( secondSep, trackerFeatureType.length() - secondSep );

		return new TrackerFeatureFeature2d(detector, descriptor);
	}

	if( trackerFeatureType.find("HOG") == 0 )
	{
		return new TrackerFeatureHOG();
	}

	if( trackerFeatureType.find("HAAR") == 0 )
	{
		return new TrackerFeatureHAAR();
	}

	if( trackerFeatureType.find("LBP") == 0 )
	{
		return new TrackerFeatureLBP();
	}

	CV_Error(-1, "Tracker feature type not supported");
	return NULL;
}

String TrackerFeature::getClassName() const
{
	return className;
}

/**
 * TrackerFeatureFeature2d
 */
TrackerFeatureFeature2d::TrackerFeatureFeature2d( String detectorType, String descriptorType )
{
	className = "FEATURE2D";
}

TrackerFeatureFeature2d::~TrackerFeatureFeature2d()
{

}

bool TrackerFeatureFeature2d::computeImpl( const std::vector<Mat>& images, Mat& response )
{
	return false;
}

void TrackerFeatureFeature2d::selection( Mat& response, int npoints )
{

}

/**
 * TrackerFeatureHOG
 */
TrackerFeatureHOG::TrackerFeatureHOG()
{
	className = "HOG";
}

TrackerFeatureHOG::~TrackerFeatureHOG()
{

}

bool TrackerFeatureHOG::computeImpl( const std::vector<Mat>& images, Mat& response )
{
	return false;
}

void TrackerFeatureHOG::selection( Mat& response, int npoints )
{

}

/**
 * TrackerFeatureHAAR
 */
TrackerFeatureHAAR::TrackerFeatureHAAR()
{
	className = "HAAR";
}

TrackerFeatureHAAR::~TrackerFeatureHAAR()
{

}

bool TrackerFeatureHAAR::computeImpl( const std::vector<Mat>& images, Mat& response )
{
	return false;
}

void TrackerFeatureHAAR::selection( Mat& response, int npoints )
{

}

/**
 * TrackerFeatureLBP
 */
TrackerFeatureLBP::TrackerFeatureLBP()
{
	className = "LBP";
}

TrackerFeatureLBP::~TrackerFeatureLBP()
{

}

bool TrackerFeatureLBP::computeImpl( const std::vector<Mat>& images, Mat& response )
{
	return false;
}

void TrackerFeatureLBP::selection( Mat& response, int npoints )
{

}


} /* namespace cv */
