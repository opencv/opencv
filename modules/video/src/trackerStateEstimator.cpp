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
 *  TrackerStateEstimator
 */

TrackerStateEstimator::~TrackerStateEstimator()
{

}

Ptr<TrackerTargetState> TrackerStateEstimator::estimate( const std::vector<ConfidenceMap>& confidenceMaps )
{
  if( confidenceMaps.empty() )
    return 0;

  return estimateImpl( confidenceMaps );

}

void TrackerStateEstimator::update( std::vector<ConfidenceMap>& confidenceMaps )
{
  if( confidenceMaps.empty() )
    return;

  return updateImpl( confidenceMaps );

}

Ptr<TrackerStateEstimator> TrackerStateEstimator::create( const String& trackeStateEstimatorType )
{

  if( trackeStateEstimatorType.find( "SVM" ) == 0 )
  {
    return new TrackerStateEstimatorSVM();
  }

  if( trackeStateEstimatorType.find( "BOOSTING" ) == 0 )
  {
    return new TrackerStateEstimatorMILBoosting();
  }

  CV_Error( -1, "Tracker state estimator type not supported" );
  return 0;
}

String TrackerStateEstimator::getClassName() const
{
  return className;
}

/**
 * TrackerStateEstimatorMILBoosting::TrackerMILTargetState
 */
TrackerStateEstimatorMILBoosting::TrackerMILTargetState::TrackerMILTargetState( const Point2f& position, int width, int height,
                                                                                bool foreground, const Mat& features )
{
  setTargetPosition( position );
  setTargetWidth( width );
  setTargetHeight( height );
  setTargetFg( foreground );
  setFeatures( features );
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setTargetFg( bool foreground )
{
  isTarget = foreground;
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setFeatures( const Mat& features )
{
  targetFeatures = features;
}

bool TrackerStateEstimatorMILBoosting::TrackerMILTargetState::isTargetFg() const
{
  return isTarget;
}

Mat TrackerStateEstimatorMILBoosting::TrackerMILTargetState::getFeatures() const
{
  return targetFeatures;
}

TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting( int nFeatures )
{
  className = "BOOSTING";
  trained = false;
  numFeatures = nFeatures;
}

TrackerStateEstimatorMILBoosting::~TrackerStateEstimatorMILBoosting()
{

}

void TrackerStateEstimatorMILBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap )
{
  currentConfidenceMap.clear();
  currentConfidenceMap = confidenceMap;
}

uint TrackerStateEstimatorMILBoosting::max_idx( const std::vector<float> &v )
{
  const float* findPtr = & ( *std::max_element( v.begin(), v.end() ) );
  const float* beginPtr = & ( *v.begin() );
  return (uint) ( findPtr - beginPtr );
}

Ptr<TrackerTargetState> TrackerStateEstimatorMILBoosting::estimateImpl( const std::vector<ConfidenceMap>& /*confidenceMaps*/ )
{
  //run ClfMilBoost classify in order to compute next location
  if( currentConfidenceMap.empty() )
    return 0;

  Mat positiveStates;
  Mat negativeStates;

  prepareData( currentConfidenceMap, positiveStates, negativeStates );

  std::vector<float> prob = boostMILModel.classify( positiveStates );

  int bestind = max_idx( prob );
  //float resp = prob[bestind];

  return currentConfidenceMap.at( bestind ).first;
}

void TrackerStateEstimatorMILBoosting::prepareData( const ConfidenceMap& confidenceMap, Mat& positive, Mat& negative )
{

  int posCounter = 0;
  int negCounter = 0;

  for ( size_t i = 0; i < confidenceMap.size(); i++ )
  {
    Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at( i ).first;
    if( currentTargetState->isTargetFg() )
      posCounter++;
    else
      negCounter++;
  }

  positive.create( posCounter, numFeatures, CV_32FC1 );
  negative.create( negCounter, numFeatures, CV_32FC1 );

  //TODO change with mat fast access
  //initialize trainData (positive and negative)

  int pc = 0;
  int nc = 0;
  for ( size_t i = 0; i < confidenceMap.size(); i++ )
  {
    Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at( i ).first;
    Mat stateFeatures = currentTargetState->getFeatures();

    if( currentTargetState->isTargetFg() )
    {
      for ( int j = 0; j < stateFeatures.rows; j++ )
      {
        //fill the positive trainData with the value of the feature j for sample i
        positive.at<float>( pc, j ) = stateFeatures.at<float>( j, 0 );
      }
      pc++;
    }
    else
    {
      for ( int j = 0; j < stateFeatures.rows; j++ )
      {
        //fill the negative trainData with the value of the feature j for sample i
        negative.at<float>( nc, j ) = stateFeatures.at<float>( j, 0 );
      }
      nc++;
    }

  }
}

void TrackerStateEstimatorMILBoosting::updateImpl( std::vector<ConfidenceMap>& confidenceMaps )
{

  if( !trained )
  {
    //this is the first time that the classifier is built
    //init MIL
    boostMILModel.init();
    trained = true;
  }

  ConfidenceMap lastConfidenceMap = confidenceMaps.back();
  Mat positiveStates;
  Mat negativeStates;

  prepareData( lastConfidenceMap, positiveStates, negativeStates );
  //update MIL
  boostMILModel.update( positiveStates, negativeStates );

}

/**
 * TrackerStateEstimatorSVM
 */
TrackerStateEstimatorSVM::TrackerStateEstimatorSVM()
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

void TrackerStateEstimatorSVM::updateImpl( std::vector<ConfidenceMap>& /*confidenceMaps*/ )
{

}

} /* namespace cv */
