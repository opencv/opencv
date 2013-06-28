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
 * 	- [AAM] S. Salti, A. Cavallaro, L. Di Stefano, Adaptive Appearance Modeling for Video Tracking: Survey and Evaluation
 *  - [AMVOT] X. Li, W. Hu, C. Shen, Z. Zhang, A. Dick, A. van den Hengel, A Survey of Appearance Models in Visual Object Tracking
 *
 * This Tracking API has been designed with PlantUML. If you modify this API please change UML files under modules/video/misc/
 *
 */

namespace cv
{


/************************************ Base Classes ************************************/

/**
 * \brief Abstract base class for TrackerFeature that represents the feature.
 */
class CV_EXPORTS_W TrackerFeature
{
public:
	virtual ~TrackerFeature();

	/**
	 * \brief Compute the features in a image
	 * \param image         The image.
	 * \param response    	Computed features.
	 */
	void compute( const Mat& image, Mat& response );

	/**
	 * \brief Create TrackerFeature by tracker feature type.
	 */
	static Ptr<TrackerFeature> create( const String& trackerFeatureType );

	/**
	 * \brief Identify most effective features
	 * \param response Collection of response for the specific TrackerFeature
	 * \param npoints Max number of features
	 */
	virtual void selection( Mat& response, int npoints ) = 0;

	/**
	 * \brief Get the name of the specific tracker feature
	 * \return The name of the tracker feature
	 */
	String getClassName() const;

protected:

	virtual bool computeImpl( const Mat& image, Mat& response ) = 0;

	String className;
};


/**
 * \brief Class that manages the extraction and selection of features
 * [AAM] Feature Extraction and Feature Set Refinement (Feature Processing and Feature Selection). See table I and section III C
 * [AMVOT] Appearance modelling -> Visual representation (Table II, section 3.1 - 3.2)
 */
class CV_EXPORTS_W TrackerFeatureSet
{
public:
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

	TrackerFeatureSet();

	~TrackerFeatureSet();

	/**
	 * \brief Extract features from the image
	 * \param image The image
	 */
	void extraction( const Mat& image );

	/**
	 * \brief Identify most effective features for all feature types
	 */
	void selection();

	/**
	 * \brief Remove outliers for all feature types
	 */
	void removeOutliers();

	/**
	 * \brief Add TrackerFeature in the collection from tracker feature type
	 * \param trackerFeatureType the tracker feature type FEATURE2D.DETECTOR.DESCRIPTOR - HOG - HAAR - LBP
	 * \return true if feature is added, false otherwise
	 */
	bool addTrackerFeature( String trackerFeatureType );

	/**
	 * \brief Add TrackerFeature in collection directly
	 * \param feature The TrackerFeature
	 * \return true if feature is added, false otherwise
	 */
	bool addTrackerFeature( Ptr<TrackerFeature>& feature );

	/**
	 * \brief Get the TrackerFeature collection
	 * \return The TrackerFeature collection
	 */
	const std::vector<std::pair<String, Ptr<TrackerFeature> > >& getTrackerFeature() const;

	/**
	 * \brief Get the reponses
	 * \return the reponses
	 */
	const std::vector<Mat>& getResponses() const;

private:

	void clearResponses();
	bool blockAddTrackerFeature;

	std::vector<std::pair<String, Ptr<TrackerFeature> > > features;	//list of features
	std::vector<Mat> responses;				//list of response after compute

};


/**
 * \brief Abstract base class for TrackerSamplerAlgorithm that represents the algorithm for the specific sampler.
 */
class CV_EXPORTS_W TrackerSamplerAlgorithm
{
public:
	/**
	 * \brief Destructor
	 */
	virtual ~TrackerSamplerAlgorithm();

	/**
	 * \brief Create TrackerSamplerAlgorithm by tracker sampler type.
	 */
	static Ptr<TrackerSamplerAlgorithm> create( const String& trackerSamplerType );

	/**
	 * \brief Computes the regions starting from a position in an image
	 * \param image The image
	 * \param position The position from which regions can be calculated
	 * \param sample The computed samples [AAM] Fig. 1 variable Sk
	 * \return true if samples is computed, false otherwise
	 */
	bool sampling( const Mat& image, Point2f position, std::vector<Mat>& sample );

	/**
	 * \brief Get the name of the specific sampler algorithm
	 * \return The name of the tracker sampler algorithm
	 */
	String getClassName() const;

protected:
	String className;

	virtual bool samplingImpl( const Mat& image, Point2f position, std::vector<Mat>& sample ) = 0;
};

/**
 * \brief Class that manages the sampler in order to select regions for the update the model of the tracker
 * [AMM] Sampling e Labeling. See table I and section III B
 */
class CV_EXPORTS_W TrackerSampler
{
public:

	/**
	 * \enum Tracker sampler types
	 * \brief List of sampler types
	 */
	enum{ TRACKER_SAMPLER_CS    = 1,   //!< TRACKER_SAMPLER_CS
		  TRACKER_SAMPLER_CSC   = 2    //!< TRACKER_SAMPLER_CSC
	};

	/**
	 * \brief Constructor
	 */
	TrackerSampler();

	/**
	 * \brief Destructor
	 */
	~TrackerSampler();

	/**
	 * \brief Computes the regions starting from a position in an image
	 * \param image The image
	 * \param position The position from which regions can be calculated
	 */
	void sampling( const Mat& image, Point2f position );

	/**
	 * Get the samples from all TrackerSamplerAlgorithm
	 * \return The samples [AAM] Fig. 1 variable Sk
	 */
	const std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm> > >& getSamples() const;

	/**
	 * \brief Add TrackerSamplerAlgorithm in the collection from tracker sampler type
	 * \param trackerSamplerAlgorithmType the tracker sampler type CSC - CS
	 * \return true if sampler is added, false otherwise
	 */
	bool addTrackerSamplerAlgorithm( String trackerSamplerAlgorithmType );

private:
	std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm> > > samplers;
	std::vector<Mat> samples;
	bool blockAddTrackerSampler;

	void clearSamples();
};



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

	virtual bool initImpl( const Mat& image, const Rect& boundingBox ) = 0;
	virtual bool updateImpl( const Mat& image, Rect& boundingBox ) = 0;

	Ptr<TrackerFeatureSet> featureSet;
	Ptr<TrackerSampler> sampler;

};


/************************************ Specific TrackerSamplerAlgorithm Classes ************************************/

/**
 * \brief TrackerSampler based on CSC (current state centered)
 */
class CV_EXPORTS_W TrackerSamplerCSC : public TrackerSamplerAlgorithm
{
public:

	TrackerSamplerCSC();

	~TrackerSamplerCSC();

	bool samplingImpl( const Mat& image, Point2f position, std::vector<Mat>& sample );

};

/**
 * \brief TrackerSampler based on CS (current state)
 */
class CV_EXPORTS_W TrackerSamplerCS : public TrackerSamplerAlgorithm
{
public:

	TrackerSamplerCS();

	~TrackerSamplerCS();

	bool samplingImpl( const Mat& image, Point2f position, std::vector<Mat>& sample );

};



/************************************ Specific TrackerFeature Classes ************************************/

/**
 * \brief TrackerFeature based on Feature2D
 */
class CV_EXPORTS_W TrackerFeatureFeature2d : public TrackerFeature
{
public:

	/**
	 * \brief Constructor
	 * \param detectorType string of FeatureDetector
	 * \param descriptorType string of DescriptorExtractor
	 */
	TrackerFeatureFeature2d( String detectorType, String descriptorType );

	~TrackerFeatureFeature2d();

	bool computeImpl( const Mat& image, Mat& response );

	void selection( Mat& response, int npoints );

private:

	std::vector<KeyPoint> keypoints;
};


/**
 * \brief TrackerFeature based on HOG
 */
class CV_EXPORTS_W TrackerFeatureHOG : public TrackerFeature
{
public:

	TrackerFeatureHOG();

	~TrackerFeatureHOG();

	bool computeImpl( const Mat& image, Mat& response );

	void selection( Mat& response, int npoints );

};


/**
 * \brief TrackerFeature based on HAAR
 */
class CV_EXPORTS_W TrackerFeatureHAAR : public TrackerFeature
{
public:

	TrackerFeatureHAAR();

	~TrackerFeatureHAAR();

	bool computeImpl( const Mat& image, Mat& response );

	void selection( Mat& response, int npoints );

};


/**
 * \brief TrackerFeature based on LBP
 */
class CV_EXPORTS_W TrackerFeatureLBP : public TrackerFeature
{
public:

	TrackerFeatureLBP();

	~TrackerFeatureLBP();

	bool computeImpl( const Mat& image, Mat& response );

	void selection( Mat& response, int npoints );

};



/************************************ Specific Tracker Classes ************************************/

/**
  \brief TrackerMIL implementation.
  For more details see B Babenko, MH Yang, S Belongie, Visual Tracking with Online Multiple Instance Learning
*/
class CV_EXPORTS_W TrackerMIL : public Tracker
{
public:
	struct CV_EXPORTS Params
	{
		Params();

		void read( const FileNode& fn );
		void write( FileStorage& fs ) const;
	};

	/**
	 * \brief TrackerMIL Constructor
	 * \param parameters        TrackerMIL parameters
	 */
	TrackerMIL(const TrackerMIL::Params &parameters = TrackerMIL::Params());

	virtual ~TrackerMIL();

	void read( const FileNode& fn );
	void write( FileStorage& fs ) const;

protected:

	bool initImpl( const Mat& image, const Rect& boundingBox );
	bool updateImpl( const Mat& image, Rect& boundingBox );

	Params params;
	AlgorithmInfo* info() const { return 0; }
};


/**
  \brief TrackerBoosting implementation.
  For more details see H Grabner, M Grabner, H Bischof, Real-time tracking via on-line boosting
*/
class CV_EXPORTS_W TrackerBoosting : public Tracker
{
public:
	struct CV_EXPORTS Params
	{
		Params();

		/**
		 * \brief Read parameters from file
		 */
		void read( const FileNode& fn );

		/**
		 * \brief Write parameters in a file
		 */
		void write( FileStorage& fs ) const;
	};

	/**
	 * \brief TrackerBoosting Constructor
	 * \param parameters        TrackerBoosting parameters
	 */
	TrackerBoosting(const TrackerBoosting::Params &parameters = TrackerBoosting::Params());

	virtual ~TrackerBoosting();

	void read( const FileNode& fn );
	void write( FileStorage& fs ) const;

protected:

	bool initImpl( const Mat& image, const Rect& boundingBox );
	bool updateImpl( const Mat& image, Rect& boundingBox );

	Params params;
	AlgorithmInfo* info() const { return 0; }
};

} /* namespace cv */

#endif
