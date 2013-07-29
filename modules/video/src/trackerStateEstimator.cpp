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
#include "trackerMILModel.hpp"

namespace cv
{

/*
 *  TrackerStateEstimator
 */

TrackerStateEstimator::~TrackerStateEstimator()
{

}

Ptr<TrackerTargetState> TrackerStateEstimator::estimate( const std::vector<ConfidenceMap>& confidenceMaps )
{
	if( confidenceMaps.size() == 0 )
		return NULL;

	return estimateImpl( confidenceMaps );

}

void TrackerStateEstimator::update( std::vector<ConfidenceMap>& confidenceMaps )
{
	if( confidenceMaps.size() == 0 )
		return;

	return updateImpl( confidenceMaps );

}


Ptr<TrackerStateEstimator> TrackerStateEstimator::create( const String& trackeStateEstimatorType )
{

	if( trackeStateEstimatorType.find("SVM") == 0 )
	{
		return new TrackerStateEstimatorSVM();
	}

	if( trackeStateEstimatorType.find("BOOSTING") == 0 )
	{
		return new TrackerStateEstimatorBoosting();
	}


	CV_Error(-1, "Tracker state estimator type not supported");
	return NULL;
}

String TrackerStateEstimator::getClassName() const
{
	return className;
}

/**
 * TrackerStateEstimatorBoosting
 */
TrackerStateEstimatorBoosting::TrackerStateEstimatorBoosting( int numFeatures )
{
	className = "BOOSTING";
	trained = false;
	this->numFeatures = numFeatures;
}

TrackerStateEstimatorBoosting::~TrackerStateEstimatorBoosting()
{

}

void TrackerStateEstimatorBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap )
{
	currentConfidenceMap.clear();
	currentConfidenceMap = confidenceMap;
}

Ptr<TrackerTargetState> TrackerStateEstimatorBoosting::estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps )
{
	//TODO run cvBoost predict in order to compute next location
	if( currentConfidenceMap.size() == 0 )
		return NULL;

	Mat data;
	Mat responses;

	prepareData(currentConfidenceMap, data, responses );

	//TODO get the boundingbox with the highest vote
	std::vector<float> votes;
	for( size_t i = 0; i < data.rows; i++)
	{
		float vote = boostModel.predict( data.row(i), Mat(), Range::all(), false, true );
		votes.push_back( vote );
	}

	std::vector<float>::iterator maxElem = std::max_element( votes.begin(), votes.end() );
	int maxIdx = ( std::distance( votes.begin(), maxElem ) );

	return currentConfidenceMap.at( maxIdx ).first;
}

void TrackerStateEstimatorBoosting::prepareData( const ConfidenceMap& confidenceMap, Mat& trainData, Mat& responses )
{
	//TODO change with mat fast access
	//initialize trainData and responses
	trainData.create( confidenceMap.size(), numFeatures, CV_32FC1 );
	responses.create( confidenceMap.size(), 1, CV_32FC1 );


	for( size_t i = 0; i < confidenceMap.size(); i++ )
	{
		Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at(i).first;
		Mat stateFeatures = currentTargetState->getFeatures();

		for ( int j = 0; j < stateFeatures.rows; j++ )
		{
			int posIndex = numFeatures * i + j;

			//fill the trainData with the value of the feature j for sample i
			trainData.at<float>( i, j ) = stateFeatures.at<float>( j, 0 );
		}

		int classLabel = 0;
		if( currentTargetState->isTargetFg() )
			classLabel = 1;

		//fill the responses (class background or class foreground)
		responses.at<float>( i, 0 ) = classLabel;

	}
}

void TrackerStateEstimatorBoosting::updateImpl( std::vector<ConfidenceMap>& confidenceMaps )
{



	/*CvBoostParams  params( CvBoost::REAL, // boost_type
	                           100, // weak_count
	                           0.95, // weight_trim_rate
	                           2, // max_depth
	                           false, //use_surrogates
	                           0 // priors
	                         );
 */
	ConfidenceMap lastConfidenceMap = confidenceMaps.back();

	//prepare the trainData
	Mat traindata;
	Mat responses;
	prepareData( lastConfidenceMap, traindata, responses );

	//TODO update the scores of the confidence maps
	if( !trained )
	{
		//this is the first time that the classifier is built
		boostModel.train( traindata, CV_ROW_SAMPLE, responses );
		trained = true;
	}
	else
	{
		//the classifier is updated
		boostModel.train( traindata, CV_ROW_SAMPLE, responses/*, Mat(), Mat(), Mat(), Mat(), CvBoostParams(), true*/);
	}

}


/**
 * TrackerStateEstimatorSVM
 */
TrackerStateEstimatorSVM::TrackerStateEstimatorSVM( )
{
	className = "SVM";
}

TrackerStateEstimatorSVM::~TrackerStateEstimatorSVM()
{

}

Ptr<TrackerTargetState> TrackerStateEstimatorSVM::estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps )
{
	return confidenceMaps.back().back().first;
}

void TrackerStateEstimatorSVM::updateImpl( std::vector<ConfidenceMap>& confidenceMaps )
{

}

} /* namespace cv */
