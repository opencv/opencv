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

#include "trackerMILModel.hpp"

/**
 * TrackerMILModel
 */

namespace cv
{

TrackerMILModel::TrackerMILModel( const Rect& boundingBox )
{
  currentSample.clear();
  mode = MODE_POSITIVE;
  width = boundingBox.width;
  height = boundingBox.height;

  Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> initState = new TrackerStateEstimatorMILBoosting::TrackerMILTargetState( Point2f( boundingBox.x, boundingBox.y ), boundingBox.width, boundingBox.height,
                                                                    true, Mat() );
  trajectory.push_back( initState );
}

void TrackerMILModel::responseToConfidenceMap( const std::vector<Mat>& responses, ConfidenceMap& confidenceMap )
{
  if( currentSample.empty() )
  {
    CV_Error( -1, "The samples in Model estimation are empty" );
    return;
  }

  for ( size_t i = 0; i < responses.size(); i++ )
  {
    //for each column (one sample) there are #num_feature
    //get informations from currentSample
    for ( int j = 0; j < responses.at( i ).cols; j++ )
    {

      Size currentSize;
      Point currentOfs;
      currentSample.at( j ).locateROI( currentSize, currentOfs );
      bool foreground;
      if( mode == MODE_POSITIVE || mode == MODE_ESTIMATON )
      {
        foreground = true;
      }
      else if( mode == MODE_NEGATIVE )
      {
        foreground = false;
      }

      //get the column of the HAAR responses
      Mat singleResponse = responses.at( i ).col( j );

      //create the state
      Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> currentState = new TrackerStateEstimatorMILBoosting::TrackerMILTargetState(
          currentOfs, width, height, foreground, singleResponse );

      confidenceMap.push_back( std::make_pair( currentState, 0 ) );

    }

  }
}

void TrackerMILModel::modelEstimationImpl( const std::vector<Mat>& responses )
{
  responseToConfidenceMap( responses, currentConfidenceMap );

}

void TrackerMILModel::modelUpdateImpl()
{

}

void TrackerMILModel::setMode( int trainingMode, const std::vector<Mat>& samples )
{
  currentSample.clear();
  currentSample = samples;

  mode = trainingMode;
}

}
