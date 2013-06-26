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

#ifndef __OPENCV_TRACKER_HPP__
#define __OPENCV_TRACKER_HPP__

#include "opencv2/core.hpp"

/*
 * Partially based on:
 * ====================================================================================================================
 * 	- S. Salti, A. Cavallaro, L. Di Stefano, Adaptive Appearance Modeling for Video Tracking: Survey and Evaluation
 *  - X. Li, W. Hu, C. Shen, Z. Zhang, A. Dick, A. van den Hengel, A Survey of Appearance Models in Visual Object Tracking
 *
 * This Tracking API has been designed with PlantUML. If you modify this API please change UML files under modules/video/misc/
 *
 */

namespace cv
{


/************************************ Base Classes ************************************/


/**
 * \brief Abstract base class for Tracker algorithm.
 */
class CV_EXPORTS_W Tracker : public virtual Algorithm
{
public:
	/**
	 * \enum Tracker algorithms
	 * \brief List of tracker algorithms
	 */
	enum{ TRACKER_MIL = 1,    	//!< TRACKER_MIL
	      TRACKER_BOOSTING = 2	//!< TRACKER_BOOSTING
	};

	/**
	 * \enum Tracker features types
	 * \brief List of features types
	 */
	enum{ TRACKER_FEATURE_FEATURE2D = 1,   //!< TRACKER_FEATURE_FEATURE2D
		  TRACKER_FEATURE_HOG = 2,         //!< TRACKER_FEATURE_HOG
		  TRACKER_FEATURE_HAAR = 3,        //!< TRACKER_FEATURE_HAAR
		  TRACKER_FEATURE_LBP = 4,         //!< TRACKER_FEATURE_LBP
		  TRACKER_FEATURE_HISTOGRAM = 5,   //!< TRACKER_FEATURE_HISTOGRAM
		  TRACKER_FEATURE_TEMPLATE = 6,    //!< TRACKER_FEATURE_TEMPLATE
		  TRACKER_FEATURE_PIXEL = 7,       //!< TRACKER_FEATURE_PIXEL
		  TRACKER_FEATURE_CORNER = 8       //!< TRACKER_FEATURE_CORNER
	};

	virtual ~Tracker();

	/**
	 * \brief Initialize the tracker at the first frame.
	 * \param image		     The image.
	 * \param boundingBox    The bounding box.
	 * \return true the tracker is initialized, false otherwise
	 */
	bool init( const Mat& image, const Rect& boundingBox );

	/**
	 * \brief Update the tracker at the next frames.
	 * \param image          The image.
	 * \param boundingBox    The bounding box.
	 * \return true the tracker is updated, false otherwise
	 */
	bool update( const Mat& image, Rect& boundingBox );

	/**
	 * \brief Create tracker by tracker type.
	 */
	static Ptr<Tracker> create( const String& trackerType );

protected:

	virtual bool initImpl( const Mat& image, const Rect& boundingBox ) const = 0;
	virtual bool updateImpl( const Mat& image, Rect& boundingBox ) const = 0;

};





} /* namespace cv */

#endif
