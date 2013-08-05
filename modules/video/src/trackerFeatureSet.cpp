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
 *  TrackerFeatureSet
 */

/*
 * Constructor
 */
TrackerFeatureSet::TrackerFeatureSet()
{
  blockAddTrackerFeature = false;
}

/*
 * Destructor
 */
TrackerFeatureSet::~TrackerFeatureSet()
{

}

void TrackerFeatureSet::extraction( const std::vector<Mat>& images )
{

  clearResponses();
  responses.resize( features.size() );

  for ( size_t i = 0; i < features.size(); i++ )
  {
    Mat response;
    features[i].second->compute( images, response );
    responses[i] = response;
  }

  if( !blockAddTrackerFeature )
  {
    blockAddTrackerFeature = true;
  }
}

void TrackerFeatureSet::selection()
{

}

void TrackerFeatureSet::removeOutliers()
{

}

bool TrackerFeatureSet::addTrackerFeature( String trackerFeatureType )
{
  if( blockAddTrackerFeature )
  {
    return false;
  }
  Ptr<TrackerFeature> feature = TrackerFeature::create( trackerFeatureType );

  if( feature == 0 )
  {
    return false;
  }

  features.push_back( std::make_pair( trackerFeatureType, feature ) );

  return true;
}

bool TrackerFeatureSet::addTrackerFeature( Ptr<TrackerFeature>& feature )
{
  if( blockAddTrackerFeature )
  {
    return false;
  }

  String trackerFeatureType = feature->getClassName();
  features.push_back( std::make_pair( trackerFeatureType, feature ) );

  return true;
}

const std::vector<std::pair<String, Ptr<TrackerFeature> > >& TrackerFeatureSet::getTrackerFeature() const
{
  return features;
}

const std::vector<Mat>& TrackerFeatureSet::getResponses() const
{
  return responses;
}

void TrackerFeatureSet::clearResponses()
{
  responses.clear();
}

} /* namespace cv */
