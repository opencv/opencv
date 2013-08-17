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

#ifndef __OPENCV_TRACKER_HPP__
#define __OPENCV_TRACKER_HPP__

#include "opencv2/core.hpp"
#include "feature.hpp"
#include "onlineMIL.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

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

/************************************ TrackerFeature Base Classes ************************************/

/**
 * \brief Abstract base class for TrackerFeature that represents the feature.
 */
class CV_EXPORTS_W TrackerFeature
{
 public:
  virtual ~TrackerFeature();

  /**
   * \brief Compute the features in the images collection
   * \param images        The images.
   * \param response    	Computed features.
   */
  void compute( const std::vector<Mat>& images, Mat& response );

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

  virtual bool computeImpl( const std::vector<Mat>& images, Mat& response ) = 0;

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

  TrackerFeatureSet();

  ~TrackerFeatureSet();

  /**
   * \brief Extract features from the images collection
   * \param images The images
   */
  void extraction( const std::vector<Mat>& images );

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
   * \brief Get the responses
   * \return the responses
   */
  const std::vector<Mat>& getResponses() const;

 private:

  void clearResponses();
  bool blockAddTrackerFeature;

  std::vector<std::pair<String, Ptr<TrackerFeature> > > features;  //list of features
  std::vector<Mat> responses;				//list of response after compute

};

/************************************ TrackerSampler Base Classes ************************************/

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
   * \param boundingBox The bounding box from which regions can be calculated
   * \param sample The computed samples [AAM] Fig. 1 variable Sk
   * \return true if samples are computed, false otherwise
   */
  bool sampling( const Mat& image, Rect boundingBox, std::vector<Mat>& sample );

  /**
   * \brief Get the name of the specific sampler algorithm
   * \return The name of the tracker sampler algorithm
   */
  String getClassName() const;

 protected:
  String className;

  virtual bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample ) = 0;
};

/**
 * \brief Class that manages the sampler in order to select regions for the update the model of the tracker
 * [AAM] Sampling e Labeling. See table I and section III B
 */
class CV_EXPORTS_W TrackerSampler
{
 public:

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
   * \param boundingBox The bounding box from which regions can be calculated
   */
  void sampling( const Mat& image, Rect boundingBox );

  /**
   * Get the all samplers
   * \return The samplers
   */
  const std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm> > >& getSamplers() const;

  /**
   * Get the samples from all TrackerSamplerAlgorithm
   * \return The samples [AAM] Fig. 1 variable Sk
   */
  const std::vector<Mat>& getSamples() const;

  /**
   * \brief Add TrackerSamplerAlgorithm in the collection from tracker sampler type
   * \param trackerSamplerAlgorithmType the tracker sampler type CSC - CS
   * \return true if sampler is added, false otherwise
   */
  bool addTrackerSamplerAlgorithm( String trackerSamplerAlgorithmType );

  /**
   * \brief Add TrackerSamplerAlgorithm in collection directly
   * \param sampler The TrackerSamplerAlgorithm
   * \return true if sampler is added, false otherwise
   */
  bool addTrackerSamplerAlgorithm( Ptr<TrackerSamplerAlgorithm>& sampler );

 private:
  std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm> > > samplers;
  std::vector<Mat> samples;
  bool blockAddTrackerSampler;

  void clearSamples();
};

/************************************ TrackerModel Base Classes ************************************/

/**
 * \brief Abstract base class for TrackerTargetState that represents a possible state of the target
 * [AAM] x̄_i all the states candidates
 * Inherits this with your Target state
 */
class CV_EXPORTS_W TrackerTargetState
{
 public:
  virtual ~TrackerTargetState()
  {
  }
  ;
  /**
   * \brief Get the position
   * \return The position
   */
  Point2f getTargetPosition() const;

  /**
   * \brief Set the position
   * \param position The position
   */
  void setTargetPosition( const Point2f& position );

 protected:
  Point2f targetPosition;

};

/**
 * \brief Represents the model of the target at frame k (all states and scores)
 * [AAM] The set of the pair (x̄_k(i), C_k(i))
 */
typedef std::vector<std::pair<Ptr<TrackerTargetState>, float> > ConfidenceMap;

/**
 * \brief Represents the estimate states for all frames
 * [AAM] Xk is the trajectory of the target up to time k
 */
typedef std::vector<Ptr<TrackerTargetState> > Trajectory;

/**
 * \brief Abstract base class for TrackerStateEstimator that estimates the most likely target state.
 * [AAM] State estimator
 * [AMVOT] Statistical modeling (Fig. 3), Table III (generative) - IV (discriminative) - V (hybrid)
 */
class CV_EXPORTS_W TrackerStateEstimator
{
 public:
  virtual ~TrackerStateEstimator();

  /**
   * \brief Estimate the most likely target state
   * \param confidenceMaps The overall appearance model
   * \return The estimated state
   */
  Ptr<TrackerTargetState> estimate( const std::vector<ConfidenceMap>& confidenceMaps );

  /**
   * \brief Update the ConfidenceMap with the scores
   * \param confidenceMaps The overall appearance model
   */
  void update( std::vector<ConfidenceMap>& confidenceMaps );

  /**
   * \brief Create TrackerStateEstimator by tracker state estimator type SVM - BOOSTING.
   */
  static Ptr<TrackerStateEstimator> create( const String& trackeStateEstimatorType );

  /**
   * \brief Get the name of the specific state estimator
   * \return The name of the state estimator
   */
  String getClassName() const;

 protected:

  virtual Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps ) = 0;
  virtual void updateImpl( std::vector<ConfidenceMap>& confidenceMaps ) = 0;
  String className;
};

/**
 * \brief Abstract class that represents the model of the target. It must be instantiated by specialized tracker
 * [AAM] Ak
 */
class CV_EXPORTS_W TrackerModel
{
 public:

  /**
   * \brief Constructor
   */
  TrackerModel();

  /**
   * \brief Destructor
   */
  virtual ~TrackerModel();

  /**
   * \brief Set TrackerEstimator
   * \return true if the tracker state estimator is added, false otherwise
   */
  bool setTrackerStateEstimator( Ptr<TrackerStateEstimator> trackerStateEstimator );

  /**
   * \brief Estimate the most likely target location
   * [AAM] ME, Model Estimation table I
   * \param responses Features extracted
   */
  void modelEstimation( const std::vector<Mat>& responses );

  /**
   * \brief Update the model
   * [AAM] MU, Model Update table I
   */
  void modelUpdate();

  /**
   * \brief Run the TrackerStateEstimator
   * \return true if is possible to estimate a new state, false otherwise
   */
  bool runStateEstimator();

  /**
   * \brief Set the current estimated state
   * \param lastTargetState the current estimated state
   */
  void setLastTargetState( const Ptr<TrackerTargetState>& lastTargetState );

  /**
   * \brief Get the last target state
   * \return The last target state
   */
  Ptr<TrackerTargetState> getLastTargetState() const;

  /**
   * \brief Get the list of the confidence map
   * \return The list of the confidence map
   */
  const std::vector<ConfidenceMap>& getConfidenceMaps() const;

  /**
   * \brief Get the last confidence map
   * \return The the last confidence map
   */
  const ConfidenceMap& getLastConfidenceMap() const;

  /**
   * \brief Get the tracker state estimator
   * \return The tracker state estimator
   */
  Ptr<TrackerStateEstimator> getTrackerStateEstimator() const;

 private:

  void clearCurrentConfidenceMap();

 protected:
  std::vector<ConfidenceMap> confidenceMaps;
  Ptr<TrackerStateEstimator> stateEstimator;
  ConfidenceMap currentConfidenceMap;
  Trajectory trajectory;

  virtual void modelEstimationImpl( const std::vector<Mat>& responses ) = 0;
  virtual void modelUpdateImpl() = 0;

};

/************************************ Tracker Base Class ************************************/

/**
 * \brief Abstract base class for Tracker algorithm.
 */
class CV_EXPORTS_W Tracker : public virtual Algorithm
{
 public:

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
   * \brief Create tracker by tracker type MIL - BOOSTING.
   */
  static Ptr<Tracker> create( const String& trackerType );

 protected:

  virtual bool initImpl( const Mat& image, const Rect& boundingBox ) = 0;
  virtual bool updateImpl( const Mat& image, Rect& boundingBox ) = 0;

  bool initialized;

  Ptr<TrackerFeatureSet> featureSet;
  Ptr<TrackerSampler> sampler;
  Ptr<TrackerModel> model;

};

/************************************ Specific TrackerStateEstimator Classes ************************************/

/**
 * \brief TrackerStateEstimator based on MILBoosting
 */
class CV_EXPORTS_W TrackerStateEstimatorMILBoosting : public TrackerStateEstimator
{
 public:
  TrackerStateEstimatorMILBoosting( int numFeatures = 250 );
  ~TrackerStateEstimatorMILBoosting();

  void setCurrentConfidenceMap( ConfidenceMap& confidenceMap );

 protected:
  Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps );
  void updateImpl( std::vector<ConfidenceMap>& confidenceMaps );

 private:
  uint max_idx( const std::vector<float> &v );
  void prepareData( const ConfidenceMap& confidenceMap, Mat& positive, Mat& negative );

  ClfMilBoost boostMILModel;
  bool trained;
  int numFeatures;

  ConfidenceMap currentConfidenceMap;
};

/**
 * \brief TrackerStateEstimator based on SVM
 */
class CV_EXPORTS_W TrackerStateEstimatorSVM : public TrackerStateEstimator
{
 public:
  TrackerStateEstimatorSVM();
  ~TrackerStateEstimatorSVM();

 protected:
  Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps );
  void updateImpl( std::vector<ConfidenceMap>& confidenceMaps );
};

/************************************ Specific TrackerSamplerAlgorithm Classes ************************************/

/**
 * \brief TrackerSampler based on CSC (current state centered)
 */
class CV_EXPORTS_W TrackerSamplerCSC : public TrackerSamplerAlgorithm
{
 public:
  enum
  {
    MODE_INIT_POS = 1,  // mode for init positive samples
    MODE_INIT_NEG = 2,  // mode for init negative samples
    MODE_TRACK_POS = 3,  // mode for update positive samples
    MODE_TRACK_NEG = 4,  // mode for update negative samples
    MODE_DETECT = 5   // mode for detect samples
  };

  struct CV_EXPORTS Params
  {
    Params();
    float initInRad;        // radius for gathering positive instances during init
    float trackInPosRad;    // radius for gathering positive instances during tracking
    float searchWinSize;	// size of search window
    int initMaxNegNum;      // # negative samples to use during init
    int trackMaxPosNum;     // # positive samples to use during training
    int trackMaxNegNum;     // # negative samples to use during training
  };

  TrackerSamplerCSC( const TrackerSamplerCSC::Params &parameters = TrackerSamplerCSC::Params() );

  /**
   * \brief set the sampling mode
   */
  void setMode( int samplingMode );

  ~TrackerSamplerCSC();

 protected:

  bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample );

 private:

  Params params;
  int mode;

  std::vector<Mat> sampleImage( const Mat& img, int x, int y, int w, int h, float inrad, float outrad = 0, int maxnum = 1000000 );
};

/**
 * \brief TrackerSampler based on CS (current state)
 */
class CV_EXPORTS_W TrackerSamplerCS : public TrackerSamplerAlgorithm
{
 public:

  TrackerSamplerCS();

  ~TrackerSamplerCS();

  bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample );

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

  void selection( Mat& response, int npoints );

 protected:

  bool computeImpl( const std::vector<Mat>& images, Mat& response );

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

  void selection( Mat& response, int npoints );

 protected:

  bool computeImpl( const std::vector<Mat>& images, Mat& response );

};

/**
 * \brief TrackerFeature based on HAAR
 */
class CV_EXPORTS_W TrackerFeatureHAAR : public TrackerFeature
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    int numFeatures;  // # of rects
    Size rectSize;    // rect size

  };

  TrackerFeatureHAAR( const TrackerFeatureHAAR::Params &parameters = TrackerFeatureHAAR::Params() );

  ~TrackerFeatureHAAR();

  void selection( Mat& response, int npoints );

 protected:

  bool computeImpl( const std::vector<Mat>& images, Mat& response );

 private:

  Params params;
  Ptr<CvHaarEvaluator> featureEvaluator;
};

/**
 * \brief TrackerFeature based on LBP
 */
class CV_EXPORTS_W TrackerFeatureLBP : public TrackerFeature
{
 public:

  TrackerFeatureLBP();

  ~TrackerFeatureLBP();

  void selection( Mat& response, int npoints );

 protected:

  bool computeImpl( const std::vector<Mat>& images, Mat& response );

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
    //parameters for sampler
    float samplerInitInRadius;	// radius for gathering positive instances during init
    int samplerInitMaxNegNum;  // # negative samples to use during init
    float samplerSearchWinSize;  // size of search window
    float samplerTrackInRadius;  // radius for gathering positive instances during tracking
    int samplerTrackMaxPosNum;	// # positive samples to use during tracking
    int samplerTrackMaxNegNum;	// # negative samples to use during tracking

    int featureSetNumFeatures;  // #features

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
  };

  /**
   * \brief TrackerMIL Constructor
   * \param parameters        TrackerMIL parameters
   */
  TrackerMIL( const TrackerMIL::Params &parameters = TrackerMIL::Params() );

  virtual ~TrackerMIL();

  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

  static int getRandInt( const int min, const int max );
  static float getRandFloat( const float min = 0, const float max = 1 );

 protected:

  bool initImpl( const Mat& image, const Rect& boundingBox );
  bool updateImpl( const Mat& image, Rect& boundingBox );

  Params params;
  AlgorithmInfo* info() const;

 private:
  static RNG rng;

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
  TrackerBoosting( const TrackerBoosting::Params &parameters = TrackerBoosting::Params() );

  virtual ~TrackerBoosting();

  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:

  bool initImpl( const Mat& image, const Rect& boundingBox );
  bool updateImpl( const Mat& image, Rect& boundingBox );

  Params params;
  AlgorithmInfo* info() const;
};

} /* namespace cv */

#endif
