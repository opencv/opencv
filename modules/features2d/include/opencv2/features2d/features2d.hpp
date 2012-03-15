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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_FEATURES_2D_HPP__
#define __OPENCV_FEATURES_2D_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"

#ifdef __cplusplus
#include <limits>

namespace cv
{

/*!
 The Keypoint Class
 
 The class instance stores a keypoint, i.e. a point feature found by one of many available keypoint detectors, such as
 Harris corner detector, cv::FAST, cv::StarDetector, cv::SURF, cv::SIFT, cv::LDetector etc.
 
 The keypoint is characterized by the 2D position, scale
 (proportional to the diameter of the neighborhood that needs to be taken into account),
 orientation and some other parameters. The keypoint neighborhood is then analyzed by another algorithm that builds a descriptor
 (usually represented as a feature vector). The keypoints representing the same object in different images can then be matched using
 cv::KDTree or another method.
*/
class CV_EXPORTS_W_SIMPLE KeyPoint
{
public:
    //! the default constructor
    CV_WRAP KeyPoint() : pt(0,0), size(0), angle(-1), response(0), octave(0), class_id(-1) {}
    //! the full constructor
    KeyPoint(Point2f _pt, float _size, float _angle=-1,
            float _response=0, int _octave=0, int _class_id=-1)
            : pt(_pt), size(_size), angle(_angle),
            response(_response), octave(_octave), class_id(_class_id) {}
    //! another form of the full constructor
    CV_WRAP KeyPoint(float x, float y, float _size, float _angle=-1,
            float _response=0, int _octave=0, int _class_id=-1)
            : pt(x, y), size(_size), angle(_angle),
            response(_response), octave(_octave), class_id(_class_id) {}
    
    size_t hash() const;
    
    //! converts vector of keypoints to vector of points
    static void convert(const vector<KeyPoint>& keypoints,
                        CV_OUT vector<Point2f>& points2f,
                        const vector<int>& keypointIndexes=vector<int>());
    //! converts vector of points to the vector of keypoints, where each keypoint is assigned the same size and the same orientation
    static void convert(const vector<Point2f>& points2f,
                        CV_OUT vector<KeyPoint>& keypoints,
                        float size=1, float response=1, int octave=0, int class_id=-1);

    //! computes overlap for pair of keypoints;
    //! overlap is a ratio between area of keypoint regions intersection and
    //! area of keypoint regions union (now keypoint region is circle)
    static float overlap(const KeyPoint& kp1, const KeyPoint& kp2);

    CV_PROP_RW Point2f pt; //!< coordinates of the keypoints
    CV_PROP_RW float size; //!< diameter of the meaningful keypoint neighborhood
    CV_PROP_RW float angle; //!< computed orientation of the keypoint (-1 if not applicable)
    CV_PROP_RW float response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
    CV_PROP_RW int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
    CV_PROP_RW int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to) 
};
    
//! writes vector of keypoints to the file storage
CV_EXPORTS void write(FileStorage& fs, const string& name, const vector<KeyPoint>& keypoints);
//! reads vector of keypoints from the specified file storage node
CV_EXPORTS void read(const FileNode& node, CV_OUT vector<KeyPoint>& keypoints);    

/*
 * A class filters a vector of keypoints.
 * Because now it is difficult to provide a convenient interface for all usage scenarios of the keypoints filter class,
 * it has only 4 needed by now static methods.
 */
class CV_EXPORTS KeyPointsFilter
{
public:
    KeyPointsFilter(){}

    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    static void runByImageBorder( vector<KeyPoint>& keypoints, Size imageSize, int borderSize );
    /*
     * Remove keypoints of sizes out of range.
     */
    static void runByKeypointSize( vector<KeyPoint>& keypoints, float minSize,
                                   float maxSize=FLT_MAX );
    /*
     * Remove keypoints from some image by mask for pixels of this image.
     */
    static void runByPixelsMask( vector<KeyPoint>& keypoints, const Mat& mask );
    /*
     * Remove duplicated keypoints.
     */
    static void removeDuplicated( vector<KeyPoint>& keypoints );
    
    /*
     * Retain the specified number of the best keypoints (according to the response)
     */
    static void retainBest(vector<KeyPoint>& keypoints, int npoints);
};

    
/************************************ Base Classes ************************************/

/*
 * Abstract base class for 2D image feature detectors.
 */
class CV_EXPORTS_W FeatureDetector : public virtual Algorithm
{
public:
    virtual ~FeatureDetector();
    
    /*
     * Detect keypoints in an image.
     * image        The image.
     * keypoints    The detected keypoints.
     * mask         Mask specifying where to look for keypoints (optional). Must be a char
     *              matrix with non-zero values in the region of interest.
     */
    CV_WRAP void detect( const Mat& image, CV_OUT vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    
    /*
     * Detect keypoints in an image set.
     * images       Image collection.
     * keypoints    Collection of keypoints detected in an input images. keypoints[i] is a set of keypoints detected in an images[i].
     * masks        Masks for image set. masks[i] is a mask for images[i].
     */
    void detect( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, const vector<Mat>& masks=vector<Mat>() ) const;
    
    // Return true if detector object is empty
    CV_WRAP virtual bool empty() const;
    
    // Create feature detector by detector name.
    CV_WRAP static Ptr<FeatureDetector> create( const string& detectorType );
    
protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const = 0;
    
    /*
     * Remove keypoints that are not in the mask.
     * Helper function, useful when wrapping a library call for keypoint detection that
     * does not support a mask argument.
     */
    static void removeInvalidPoints( const Mat& mask, vector<KeyPoint>& keypoints );
};
    
    
/*
 * Abstract base class for computing descriptors for image keypoints.
 *
 * In this interface we assume a keypoint descriptor can be represented as a
 * dense, fixed-dimensional vector of some basic type. Most descriptors used
 * in practice follow this pattern, as it makes it very easy to compute
 * distances between descriptors. Therefore we represent a collection of
 * descriptors as a Mat, where each row is one keypoint descriptor.
 */
class CV_EXPORTS_W DescriptorExtractor : public virtual Algorithm
{
public:
    virtual ~DescriptorExtractor();
    
    /*
     * Compute the descriptors for a set of keypoints in an image.
     * image        The image.
     * keypoints    The input keypoints. Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Copmputed descriptors. Row i is the descriptor for keypoint i.
     */
    CV_WRAP void compute( const Mat& image, CV_OUT CV_IN_OUT vector<KeyPoint>& keypoints, CV_OUT Mat& descriptors ) const;
    
    /*
     * Compute the descriptors for a keypoints collection detected in image collection.
     * images       Image collection.
     * keypoints    Input keypoints collection. keypoints[i] is keypoints detected in images[i].
     *              Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Descriptor collection. descriptors[i] are descriptors computed for set keypoints[i].
     */
    void compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors ) const;
    
    CV_WRAP virtual int descriptorSize() const = 0;
    CV_WRAP virtual int descriptorType() const = 0;
    
    CV_WRAP virtual bool empty() const;
    
    CV_WRAP static Ptr<DescriptorExtractor> create( const string& descriptorExtractorType );
    
protected:
    virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const = 0;
    
    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    static void removeBorderKeypoints( vector<KeyPoint>& keypoints,
                                      Size imageSize, int borderSize );
};
    
    
    
/*
 * Abstract base class for simultaneous 2D feature detection descriptor extraction.
 */
class CV_EXPORTS_W Feature2D : public FeatureDetector, public DescriptorExtractor
{
public:
    /*
     * Detect keypoints in an image.
     * image        The image.
     * keypoints    The detected keypoints.
     * mask         Mask specifying where to look for keypoints (optional). Must be a char
     *              matrix with non-zero values in the region of interest.
     * useProvidedKeypoints If true, the method will skip the detection phase and will compute
     *                      descriptors for the provided keypoints 
     */
    CV_WRAP_AS(detectAndCompute) virtual void operator()( InputArray image, InputArray mask,
                                     CV_OUT vector<KeyPoint>& keypoints,
                                     OutputArray descriptors,
                                     bool useProvidedKeypoints=false ) const = 0;
    
    // Create feature detector and descriptor extractor by name.
    static Ptr<Feature2D> create( const string& name );
};
    
    
/*!
 ORB implementation.
*/
class CV_EXPORTS ORB : public Feature2D
{
public:
    // the size of the signature in bytes
    enum { kBytes = 32, HARRIS_SCORE=0, FAST_SCORE=1 };

    explicit ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 3, int edgeThreshold = 31,
                 int firstLevel = 0, int WTA_K=2, int scoreType=0, int patchSize=31 );

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;

    // Compute the ORB features and descriptors on an image
    void operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const;

    // Compute the ORB features and descriptors on an image
    void operator()( InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false ) const;
  
    AlgorithmInfo* info() const;
    
protected:

    void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    
    CV_PROP_RW int nfeatures;
    CV_PROP_RW double scaleFactor;
    CV_PROP_RW int nlevels;
    CV_PROP_RW int edgeThreshold;
    CV_PROP_RW int firstLevel;
    CV_PROP_RW int WTA_K;
    CV_PROP_RW int scoreType;
    CV_PROP_RW int patchSize;
};
    
typedef ORB OrbFeatureDetector;
typedef ORB OrbDescriptorExtractor;


/*!
 Maximal Stable Extremal Regions class.
 
 The class implements MSER algorithm introduced by J. Matas.
 Unlike SIFT, SURF and many other detectors in OpenCV, this is salient region detector,
 not the salient point detector.
 
 It returns the regions, each of those is encoded as a contour.
*/
class CV_EXPORTS_W MSER : public FeatureDetector
{
public:
    //! the full constructor
    explicit MSER( int _delta=5, int _min_area=60, int _max_area=14400,
          double _max_variation=0.25, double _min_diversity=.2,
          int _max_evolution=200, double _area_threshold=1.01,
          double _min_margin=0.003, int _edge_blur_size=5 );
    
    //! the operator that extracts the MSERs from the image or the specific part of it
    CV_WRAP_AS(detect) void operator()( const Mat& image, vector<vector<Point> >& msers,
                                        const Mat& mask=Mat() ) const; 
    AlgorithmInfo* info() const;
    
protected:
    void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    
    int delta;
    int minArea;
    int maxArea;
    double maxVariation;
    double minDiversity;
    int maxEvolution;
    double areaThreshold;
    double minMargin;
    int edgeBlurSize;
};
    
typedef MSER MserFeatureDetector;

/*!
 The "Star" Detector.
 
 The class implements the keypoint detector introduced by K. Konolige.
*/
class CV_EXPORTS_W StarDetector : public FeatureDetector
{
public:
    //! the full constructor
    CV_WRAP StarDetector(int _maxSize=45, int _responseThreshold=30,
                 int _lineThresholdProjected=10,
                 int _lineThresholdBinarized=8,
                 int _suppressNonmaxSize=5);
    
    //! finds the keypoints in the image
    CV_WRAP_AS(detect) void operator()(const Mat& image,
                CV_OUT vector<KeyPoint>& keypoints) const;
    
    AlgorithmInfo* info() const;
    
protected:
    void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    
    int maxSize;
    int responseThreshold;
    int lineThresholdProjected;
    int lineThresholdBinarized;
    int suppressNonmaxSize;
};

//! detects corners using FAST algorithm by E. Rosten
CV_EXPORTS void FAST( InputArray image, CV_OUT vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSupression=true );

class CV_EXPORTS_W FastFeatureDetector : public FeatureDetector
{
public:
    CV_WRAP FastFeatureDetector( int threshold=10, bool nonmaxSuppression=true );
    AlgorithmInfo* info() const;
    
protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    int threshold;
    bool nonmaxSuppression;
};


class CV_EXPORTS GFTTDetector : public FeatureDetector
{
public:
    GFTTDetector( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
                  int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
    AlgorithmInfo* info() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    int nfeatures;
    double qualityLevel;
    double minDistance;
    int blockSize;
    bool useHarrisDetector;
    double k;
};

typedef GFTTDetector GoodFeaturesToTrackDetector;
typedef StarDetector StarFeatureDetector;

class CV_EXPORTS_W SimpleBlobDetector : public FeatureDetector
{
public:
  struct CV_EXPORTS_W_SIMPLE Params
  {
      CV_WRAP Params();
      CV_PROP_RW float thresholdStep;
      CV_PROP_RW float minThreshold;
      CV_PROP_RW float maxThreshold;
      CV_PROP_RW size_t minRepeatability;
      CV_PROP_RW float minDistBetweenBlobs;

      CV_PROP_RW bool filterByColor;
      CV_PROP_RW uchar blobColor;

      CV_PROP_RW bool filterByArea;
      CV_PROP_RW float minArea, maxArea;

      CV_PROP_RW bool filterByCircularity;
      CV_PROP_RW float minCircularity, maxCircularity;

      CV_PROP_RW bool filterByInertia;
      CV_PROP_RW float minInertiaRatio, maxInertiaRatio;

      CV_PROP_RW bool filterByConvexity;
      CV_PROP_RW float minConvexity, maxConvexity;

      void read( const FileNode& fn );
      void write( FileStorage& fs ) const;
  };

  CV_WRAP SimpleBlobDetector(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());

  virtual void read( const FileNode& fn );
  virtual void write( FileStorage& fs ) const;

protected:
  struct CV_EXPORTS Center
  {
      Point2d location;
      double radius;
      double confidence;
  };

  virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
  virtual void findBlobs(const cv::Mat &image, const cv::Mat &binaryImage, std::vector<Center> &centers) const;

  Params params;
};

    
class CV_EXPORTS DenseFeatureDetector : public FeatureDetector
{
public:
    explicit DenseFeatureDetector( float initFeatureScale=1.f, int featureScaleLevels=1,
                                   float featureScaleMul=0.1f,
                                   int initXyStep=6, int initImgBound=0,
                                   bool varyXyStepWithScale=true,
                                   bool varyImgBoundWithScale=false );
    AlgorithmInfo* info() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    float initFeatureScale;
    int featureScaleLevels;
    float featureScaleMul;
    
    int initXyStep;
    int initImgBound;
    
    bool varyXyStepWithScale;
    bool varyImgBoundWithScale;
};

/*
 * Adapts a detector to partition the source image into a grid and detect
 * points in each cell.
 */
class CV_EXPORTS_W GridAdaptedFeatureDetector : public FeatureDetector
{
public:
    /*
     * detector            Detector that will be adapted.
     * maxTotalKeypoints   Maximum count of keypoints detected on the image. Only the strongest keypoints
     *                      will be keeped.
     * gridRows            Grid rows count.
     * gridCols            Grid column count.
     */
    CV_WRAP GridAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector,
                                        int maxTotalKeypoints=1000,
                                        int gridRows=4, int gridCols=4 );
    
    // TODO implement read/write
    virtual bool empty() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    Ptr<FeatureDetector> detector;
    int maxTotalKeypoints;
    int gridRows;
    int gridCols;
};

/*
 * Adapts a detector to detect points over multiple levels of a Gaussian
 * pyramid. Useful for detectors that are not inherently scaled.
 */
class CV_EXPORTS_W PyramidAdaptedFeatureDetector : public FeatureDetector
{
public:
    // maxLevel - The 0-based index of the last pyramid layer
    CV_WRAP PyramidAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector, int maxLevel=2 );
    
    // TODO implement read/write
    virtual bool empty() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    Ptr<FeatureDetector> detector;
    int maxLevel;
};

/** \brief A feature detector parameter adjuster, this is used by the DynamicAdaptedFeatureDetector
 *  and is a wrapper for FeatureDetector that allow them to be adjusted after a detection
 */
class CV_EXPORTS AdjusterAdapter: public FeatureDetector
{
public:   
	/** pure virtual interface
	 */
    virtual ~AdjusterAdapter() {}
    /** too few features were detected so, adjust the detector params accordingly
     * \param min the minimum number of desired features
     * \param n_detected the number previously detected
     */
    virtual void tooFew(int min, int n_detected) = 0;
    /** too many features were detected so, adjust the detector params accordingly
     * \param max the maximum number of desired features
     * \param n_detected the number previously detected
     */
    virtual void tooMany(int max, int n_detected) = 0;
    /** are params maxed out or still valid?
     * \return false if the parameters can't be adjusted any more
     */
    virtual bool good() const = 0;

    virtual Ptr<AdjusterAdapter> clone() const = 0;

    static Ptr<AdjusterAdapter> create( const string& detectorType );
};
/** \brief an adaptively adjusting detector that iteratively detects until the desired number
 * of features are detected.
 *  Beware that this is not thread safe - as the adjustment of parameters breaks the const
 *  of the detection routine...
 *  /TODO Make this const correct and thread safe
 *
 *  sample usage:
 //will create a detector that attempts to find 100 - 110 FAST Keypoints, and will at most run
 //FAST feature detection 10 times until that number of keypoints are found
 Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(20,true),100, 110, 10));

 */
class CV_EXPORTS DynamicAdaptedFeatureDetector: public FeatureDetector
{
public:

    /** \param adjaster an AdjusterAdapter that will do the detection and parameter adjustment
     *  \param max_features the maximum desired number of features
     *  \param max_iters the maximum number of times to try to adjust the feature detector params
     * 			for the FastAdjuster this can be high, but with Star or Surf this can get time consuming
     *  \param min_features the minimum desired features
     */
    DynamicAdaptedFeatureDetector( const Ptr<AdjusterAdapter>& adjaster, int min_features=400, int max_features=500, int max_iters=5 );

    virtual bool empty() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

private:
	DynamicAdaptedFeatureDetector& operator=(const DynamicAdaptedFeatureDetector&);
	DynamicAdaptedFeatureDetector(const DynamicAdaptedFeatureDetector&);

    int escape_iters_;
    int min_features_, max_features_;
    const Ptr<AdjusterAdapter> adjuster_;
};

/**\brief an adjust for the FAST detector. This will basically decrement or increment the
 * threshold by 1
 */
class CV_EXPORTS FastAdjuster: public AdjusterAdapter
{
public:
    /**\param init_thresh the initial threshold to start with, default = 20
     * \param nonmax whether to use non max or not for fast feature detection
     */
    FastAdjuster(int init_thresh=20, bool nonmax=true, int min_thresh=1, int max_thresh=200);

    virtual void tooFew(int minv, int n_detected);
    virtual void tooMany(int maxv, int n_detected);
    virtual bool good() const;

    virtual Ptr<AdjusterAdapter> clone() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    int thresh_;
    bool nonmax_;
    int init_thresh_, min_thresh_, max_thresh_;
};


/** An adjuster for StarFeatureDetector, this one adjusts the responseThreshold for now
 * TODO find a faster way to converge the parameters for Star - use CvStarDetectorParams
 */
class CV_EXPORTS StarAdjuster: public AdjusterAdapter
{
public:
    StarAdjuster(double initial_thresh=30.0, double min_thresh=2., double max_thresh=200.);

    virtual void tooFew(int minv, int n_detected);
    virtual void tooMany(int maxv, int n_detected);
    virtual bool good() const;

    virtual Ptr<AdjusterAdapter> clone() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    double thresh_, init_thresh_, min_thresh_, max_thresh_;
};

class CV_EXPORTS SurfAdjuster: public AdjusterAdapter
{
public:
    SurfAdjuster( double initial_thresh=400.f, double min_thresh=2, double max_thresh=1000 );

    virtual void tooFew(int minv, int n_detected);
    virtual void tooMany(int maxv, int n_detected);
    virtual bool good() const;

    virtual Ptr<AdjusterAdapter> clone() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    double thresh_, init_thresh_, min_thresh_, max_thresh_;
};

CV_EXPORTS Mat windowedMatchingMask( const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                                     float maxDeltaX, float maxDeltaY );



/*
 * OpponentColorDescriptorExtractor
 *
 * Adapts a descriptor extractor to compute descripors in Opponent Color Space
 * (refer to van de Sande et al., CGIV 2008 "Color Descriptors for Object Category Recognition").
 * Input RGB image is transformed in Opponent Color Space. Then unadapted descriptor extractor
 * (set in constructor) computes descriptors on each of the three channel and concatenate
 * them into a single color descriptor.
 */
class CV_EXPORTS OpponentColorDescriptorExtractor : public DescriptorExtractor
{
public:
    OpponentColorDescriptorExtractor( const Ptr<DescriptorExtractor>& descriptorExtractor );

    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;

    virtual bool empty() const;

protected:
	virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

    Ptr<DescriptorExtractor> descriptorExtractor;
};

/*
 * BRIEF Descriptor
 */
class CV_EXPORTS BriefDescriptorExtractor : public DescriptorExtractor
{
public:
    static const int PATCH_SIZE = 48;
    static const int KERNEL_SIZE = 9;

    // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
    BriefDescriptorExtractor( int bytes = 32 );

    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;

    /// @todo read and write for brief

protected:
    virtual void computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const;

    typedef void(*PixelTestFn)(const Mat&, const vector<KeyPoint>&, Mat&);

    int bytes_;
    PixelTestFn test_fn_;
};

/****************************************************************************************\
*                                          Distance                                      *
\****************************************************************************************/
template<typename T>
struct CV_EXPORTS Accumulator
{
    typedef T Type;
};

template<> struct Accumulator<unsigned char>  { typedef float Type; };
template<> struct Accumulator<unsigned short> { typedef float Type; };
template<> struct Accumulator<char>   { typedef float Type; };
template<> struct Accumulator<short>  { typedef float Type; };

/*
 * Squared Euclidean distance functor
 */
template<class T>
struct CV_EXPORTS SL2
{
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return normL2Sqr<ValueType, ResultType>(a, b, size);
    }
};

/*
 * Euclidean distance functor
 */
template<class T>
struct CV_EXPORTS L2
{
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return (ResultType)sqrt((double)normL2Sqr<ValueType, ResultType>(a, b, size));
    }
};

/*
 * Manhattan distance (city block distance) functor
 */
template<class T>
struct CV_EXPORTS L1
{
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return normL1<ValueType, ResultType>(a, b, size);
    }
};

/*
 * Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
 * bit count of A exclusive XOR'ed with B
 */
struct CV_EXPORTS Hamming
{
    typedef unsigned char ValueType;
    typedef int ResultType;

    /** this will count the bits in a ^ b
     */
    ResultType operator()( const unsigned char* a, const unsigned char* b, int size ) const
    {
        return normHamming(a, b, size);
    }
};

typedef Hamming HammingLUT;

template<int cellsize> struct CV_EXPORTS HammingMultilevel
{
    typedef unsigned char ValueType;
    typedef int ResultType;
    
    ResultType operator()( const unsigned char* a, const unsigned char* b, int size ) const
    {
        return normHamming(a, b, size, cellsize);
    }
};
    
/****************************************************************************************\
*                                      DMatch                                            *
\****************************************************************************************/
/*
 * Struct for matching: query descriptor index, train descriptor index, train image index and distance between descriptors.
 */
struct CV_EXPORTS_W_SIMPLE DMatch
{
    CV_WRAP DMatch() : queryIdx(-1), trainIdx(-1), imgIdx(-1), distance(FLT_MAX) {}
    CV_WRAP DMatch( int _queryIdx, int _trainIdx, float _distance ) :
            queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1), distance(_distance) {}
    CV_WRAP DMatch( int _queryIdx, int _trainIdx, int _imgIdx, float _distance ) :
            queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx), distance(_distance) {}

    CV_PROP_RW int queryIdx; // query descriptor index
    CV_PROP_RW int trainIdx; // train descriptor index
    CV_PROP_RW int imgIdx;   // train image index

    CV_PROP_RW float distance;

    // less is better
    bool operator<( const DMatch &m ) const
    {
        return distance < m.distance;
    }
};

/****************************************************************************************\
*                                  DescriptorMatcher                                     *
\****************************************************************************************/
/*
 * Abstract base class for matching two sets of descriptors.
 */
class CV_EXPORTS_W DescriptorMatcher : public Algorithm
{
public:
    virtual ~DescriptorMatcher();

	/*
     * Add descriptors to train descriptor collection.
     * descriptors      Descriptors to add. Each descriptors[i] is a descriptors set from one image.
     */
    CV_WRAP virtual void add( const vector<Mat>& descriptors );
    /*
     * Get train descriptors collection.
     */
    CV_WRAP const vector<Mat>& getTrainDescriptors() const;
    /*
     * Clear train descriptors collection.
     */
    CV_WRAP virtual void clear();

    /*
     * Return true if there are not train descriptors in collection.
     */
    CV_WRAP virtual bool empty() const;
    /*
     * Return true if the matcher supports mask in match methods.
     */
    CV_WRAP virtual bool isMaskSupported() const = 0;

    /*
     * Train matcher (e.g. train flann index).
     * In all methods to match the method train() is run every time before matching.
     * Some descriptor matchers (e.g. BruteForceMatcher) have empty implementation
     * of this method, other matchers really train their inner structures
     * (e.g. FlannBasedMatcher trains flann::Index). So nonempty implementation
     * of train() should check the class object state and do traing/retraining
     * only if the state requires that (e.g. FlannBasedMatcher trains flann::Index
     * if it has not trained yet or if new descriptors have been added to the train
     * collection).
     */
    CV_WRAP virtual void train();
    /*
     * Group of methods to match descriptors from image pair.
     * Method train() is run in this methods.
     */
    // Find one best match for each query descriptor (if mask is empty).
    CV_WRAP void match( const Mat& queryDescriptors, const Mat& trainDescriptors,
                CV_OUT vector<DMatch>& matches, const Mat& mask=Mat() ) const;
    // Find k best matches for each query descriptor (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    CV_WRAP void knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                   CV_OUT vector<vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const;
    // Find best matches for each query descriptor which have distance less than
    // maxDistance (in increasing order of distances).
    void radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                      vector<vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const;
    /*
     * Group of methods to match descriptors from one image to image set.
     * See description of similar methods for matching image pair above.
     */
    CV_WRAP void match( const Mat& queryDescriptors, CV_OUT vector<DMatch>& matches,
                const vector<Mat>& masks=vector<Mat>() );
    CV_WRAP void knnMatch( const Mat& queryDescriptors, CV_OUT vector<vector<DMatch> >& matches, int k,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    void radiusMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                   const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );

    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    // Clone the matcher. If emptyTrainData is false the method create deep copy of the object, i.e. copies
    // both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
    // but with empty train data.
    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

    CV_WRAP static Ptr<DescriptorMatcher> create( const string& descriptorMatcherType );
protected:
    /*
     * Class to work with descriptors from several images as with one merged matrix.
     * It is used e.g. in FlannBasedMatcher.
     */
    class CV_EXPORTS DescriptorCollection
    {
    public:
        DescriptorCollection();
        DescriptorCollection( const DescriptorCollection& collection );
        virtual ~DescriptorCollection();

        // Vector of matrices "descriptors" will be merged to one matrix "mergedDescriptors" here.
        void set( const vector<Mat>& descriptors );
        virtual void clear();

        const Mat& getDescriptors() const;
        const Mat getDescriptor( int imgIdx, int localDescIdx ) const;
        const Mat getDescriptor( int globalDescIdx ) const;
        void getLocalIdx( int globalDescIdx, int& imgIdx, int& localDescIdx ) const;

        int size() const;

    protected:
        Mat mergedDescriptors;
        vector<int> startIdxs;
    };

    // In fact the matching is implemented only by the following two methods. These methods suppose
    // that the class object has been trained already. Public match methods call these methods
    // after calling train().
    virtual void knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false ) = 0;
    virtual void radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false ) = 0;

    static bool isPossibleMatch( const Mat& mask, int queryIdx, int trainIdx );
    static bool isMaskedOut( const vector<Mat>& masks, int queryIdx );

    static Mat clone_op( Mat m ) { return m.clone(); }
	void checkMasks( const vector<Mat>& masks, int queryDescriptorsCount ) const;

    // Collection of descriptors from train images.
    vector<Mat> trainDescCollection;
};

/*
 * Brute-force descriptor matcher.
 *
 * For each descriptor in the first set, this matcher finds the closest
 * descriptor in the second set by trying each one.
 *
 * For efficiency, BruteForceMatcher is templated on the distance metric.
 * For float descriptors, a common choice would be cv::L2<float>.
 */
class CV_EXPORTS BFMatcher : public DescriptorMatcher
{
public:
    BFMatcher( int normType, bool crossCheck=false );
    virtual ~BFMatcher() {}

    virtual bool isMaskSupported() const { return true; }

    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;

protected:
    virtual void knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    virtual void radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );

    int normType;
    bool crossCheck;
};


/*
 * Flann based matcher
 */
class CV_EXPORTS_W FlannBasedMatcher : public DescriptorMatcher
{
public:
    CV_WRAP FlannBasedMatcher( const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams(),
                       const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams() );

    virtual void add( const vector<Mat>& descriptors );
    virtual void clear();

    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    virtual void train();
    virtual bool isMaskSupported() const;
	
    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;

protected:
    static void convertToDMatches( const DescriptorCollection& descriptors,
                                   const Mat& indices, const Mat& distances,
                                   vector<vector<DMatch> >& matches );

    virtual void knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
                   const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    virtual void radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                   const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );

    Ptr<flann::IndexParams> indexParams;
    Ptr<flann::SearchParams> searchParams;
    Ptr<flann::Index> flannIndex;

    DescriptorCollection mergedDescriptors;
    int addedDescCount;
};

/****************************************************************************************\
*                                GenericDescriptorMatcher                                *
\****************************************************************************************/
/*
 *   Abstract interface for a keypoint descriptor and matcher
 */
class GenericDescriptorMatcher;
typedef GenericDescriptorMatcher GenericDescriptorMatch;

class CV_EXPORTS GenericDescriptorMatcher
{
public:
    GenericDescriptorMatcher();
    virtual ~GenericDescriptorMatcher();

    /*
     * Add train collection: images and keypoints from them.
     * images       A set of train images.
     * ketpoints    Keypoint collection that have been detected on train images.
     *
     * Keypoints for which a descriptor cannot be computed are removed. Such keypoints
     * must be filtered in this method befor adding keypoints to train collection "trainPointCollection".
     * If inheritor class need perform such prefiltering the method add() must be overloaded.
     * In the other class methods programmer has access to the train keypoints by a constant link.
     */
    virtual void add( const vector<Mat>& images,
                      vector<vector<KeyPoint> >& keypoints );

    const vector<Mat>& getTrainImages() const;
    const vector<vector<KeyPoint> >& getTrainKeypoints() const;

    /*
     * Clear images and keypoints storing in train collection.
     */
    virtual void clear();
    /*
     * Returns true if matcher supports mask to match descriptors.
     */
    virtual bool isMaskSupported() = 0;
    /*
     * Train some inner structures (e.g. flann index or decision trees).
     * train() methods is run every time in matching methods. So the method implementation
     * should has a check whether these inner structures need be trained/retrained or not.
     */
    virtual void train();

    /*
     * Classifies query keypoints.
     * queryImage    The query image
     * queryKeypoints   Keypoints from the query image
     * trainImage    The train image
     * trainKeypoints   Keypoints from the train image
     */
    // Classify keypoints from query image under one train image.
    void classify( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                           const Mat& trainImage, vector<KeyPoint>& trainKeypoints ) const;
    // Classify keypoints from query image under train image collection.
    void classify( const Mat& queryImage, vector<KeyPoint>& queryKeypoints );

    /*
     * Group of methods to match keypoints from image pair.
     * Keypoints for which a descriptor cannot be computed are removed.
     * train() method is called here.
     */
    // Find one best match for each query descriptor (if mask is empty).
    void match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                vector<DMatch>& matches, const Mat& mask=Mat() ) const;
    // Find k best matches for each query keypoint (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows.
    // If compactResult is true matches vector will not contain matches for fully masked out query descriptors.
    void knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                   const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                   vector<vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const;
    // Find best matches for each query descriptor which have distance less than maxDistance (in increasing order of distances).
    void radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                      const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                      vector<vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const;
    /*
     * Group of methods to match keypoints from one image to image set.
     * See description of similar methods for matching image pair above.
     */
    void match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                vector<DMatch>& matches, const vector<Mat>& masks=vector<Mat>() );
    void knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                   vector<vector<DMatch> >& matches, int k,
                   const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    void radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                      vector<vector<DMatch> >& matches, float maxDistance,
                      const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );

    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    // Return true if matching object is empty (e.g. feature detector or descriptor matcher are empty)
    virtual bool empty() const;

    // Clone the matcher. If emptyTrainData is false the method create deep copy of the object, i.e. copies
    // both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
    // but with empty train data.
    virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

    static Ptr<GenericDescriptorMatcher> create( const string& genericDescritptorMatcherType,
                                                 const string &paramsFilename=string() );

protected:
    // In fact the matching is implemented only by the following two methods. These methods suppose
    // that the class object has been trained already. Public match methods call these methods
    // after calling train().
    virtual void knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                               vector<vector<DMatch> >& matches, int k,
                               const vector<Mat>& masks, bool compactResult ) = 0;
    virtual void radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                  vector<vector<DMatch> >& matches, float maxDistance,
                                  const vector<Mat>& masks, bool compactResult ) = 0;
    /*
     * A storage for sets of keypoints together with corresponding images and class IDs
     */
    class CV_EXPORTS KeyPointCollection
    {
    public:
        KeyPointCollection();
        KeyPointCollection( const KeyPointCollection& collection );
        void add( const vector<Mat>& images, const vector<vector<KeyPoint> >& keypoints );
        void clear();

        // Returns the total number of keypoints in the collection
        size_t keypointCount() const;
        size_t imageCount() const;

        const vector<vector<KeyPoint> >& getKeypoints() const;
        const vector<KeyPoint>& getKeypoints( int imgIdx ) const;
        const KeyPoint& getKeyPoint( int imgIdx, int localPointIdx ) const;
        const KeyPoint& getKeyPoint( int globalPointIdx ) const;
        void getLocalIdx( int globalPointIdx, int& imgIdx, int& localPointIdx ) const;

        const vector<Mat>& getImages() const;
        const Mat& getImage( int imgIdx ) const;

    protected:
        int pointCount;

        vector<Mat> images;
        vector<vector<KeyPoint> > keypoints;
        // global indices of the first points in each image, startIndices.size() = keypoints.size()
        vector<int> startIndices;

    private:
        static Mat clone_op( Mat m ) { return m.clone(); }
    };

    KeyPointCollection trainPointCollection;
};


/****************************************************************************************\
*                                VectorDescriptorMatcher                                 *
\****************************************************************************************/

/*
 *  A class used for matching descriptors that can be described as vectors in a finite-dimensional space
 */
class VectorDescriptorMatcher;
typedef VectorDescriptorMatcher VectorDescriptorMatch;

class CV_EXPORTS VectorDescriptorMatcher : public GenericDescriptorMatcher
{
public:
    VectorDescriptorMatcher( const Ptr<DescriptorExtractor>& extractor, const Ptr<DescriptorMatcher>& matcher );
    virtual ~VectorDescriptorMatcher();

    virtual void add( const vector<Mat>& imgCollection,
                      vector<vector<KeyPoint> >& pointCollection );

    virtual void clear();

    virtual void train();

    virtual bool isMaskSupported();

    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;
    virtual bool empty() const;

    virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;

protected:
    virtual void knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                               vector<vector<DMatch> >& matches, int k,
                               const vector<Mat>& masks, bool compactResult );
    virtual void radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                  vector<vector<DMatch> >& matches, float maxDistance,
                                  const vector<Mat>& masks, bool compactResult );

    Ptr<DescriptorExtractor> extractor;
    Ptr<DescriptorMatcher> matcher;
};

/****************************************************************************************\
*                                   Drawing functions                                    *
\****************************************************************************************/
struct CV_EXPORTS DrawMatchesFlags
{
    enum{ DEFAULT = 0, // Output image matrix will be created (Mat::create),
                       // i.e. existing memory of output image may be reused.
                       // Two source image, matches and single keypoints will be drawn.
                       // For each keypoint only the center point will be drawn (without
                       // the circle around keypoint with keypoint size and orientation).
          DRAW_OVER_OUTIMG = 1, // Output image matrix will not be created (Mat::create).
                                // Matches will be drawn on existing content of output image.
          NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
          DRAW_RICH_KEYPOINTS = 4 // For each keypoint the circle around keypoint with keypoint size and
                                  // orientation will be drawn.
        };
};

// Draw keypoints.
CV_EXPORTS void drawKeypoints( const Mat& image, const vector<KeyPoint>& keypoints, Mat& outImage,
                               const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT );

// Draws matches of keypints from two images on output image.
CV_EXPORTS void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                             const Mat& img2, const vector<KeyPoint>& keypoints2,
                             const vector<DMatch>& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const vector<char>& matchesMask=vector<char>(), int flags=DrawMatchesFlags::DEFAULT );

CV_EXPORTS void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                             const Mat& img2, const vector<KeyPoint>& keypoints2,
                             const vector<vector<DMatch> >& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const vector<vector<char> >& matchesMask=vector<vector<char> >(), int flags=DrawMatchesFlags::DEFAULT );

/****************************************************************************************\
*   Functions to evaluate the feature detectors and [generic] descriptor extractors      *
\****************************************************************************************/

CV_EXPORTS void evaluateFeatureDetector( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                         vector<KeyPoint>* keypoints1, vector<KeyPoint>* keypoints2,
                                         float& repeatability, int& correspCount,
                                         const Ptr<FeatureDetector>& fdetector=Ptr<FeatureDetector>() );

CV_EXPORTS void computeRecallPrecisionCurve( const vector<vector<DMatch> >& matches1to2,
                                             const vector<vector<uchar> >& correctMatches1to2Mask,
                                             vector<Point2f>& recallPrecisionCurve );

CV_EXPORTS float getRecall( const vector<Point2f>& recallPrecisionCurve, float l_precision );
CV_EXPORTS int getNearestPoint( const vector<Point2f>& recallPrecisionCurve, float l_precision );

CV_EXPORTS void evaluateGenericDescriptorMatcher( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                                  vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
                                                  vector<vector<DMatch> >* matches1to2, vector<vector<uchar> >* correctMatches1to2Mask,
                                                  vector<Point2f>& recallPrecisionCurve,
                                                  const Ptr<GenericDescriptorMatcher>& dmatch=Ptr<GenericDescriptorMatcher>() );


/****************************************************************************************\
*                                     Bag of visual words                                *
\****************************************************************************************/
/*
 * Abstract base class for training of a 'bag of visual words' vocabulary from a set of descriptors
 */
class CV_EXPORTS BOWTrainer
{
public:
    BOWTrainer();
    virtual ~BOWTrainer();

    void add( const Mat& descriptors );
    const vector<Mat>& getDescriptors() const;
    int descripotorsCount() const;

    virtual void clear();

    /*
     * Train visual words vocabulary, that is cluster training descriptors and
     * compute cluster centers.
     * Returns cluster centers.
     *
     * descriptors      Training descriptors computed on images keypoints.
     */
    virtual Mat cluster() const = 0;
    virtual Mat cluster( const Mat& descriptors ) const = 0;

protected:
    vector<Mat> descriptors;
    int size;
};

/*
 * This is BOWTrainer using cv::kmeans to get vocabulary.
 */
class CV_EXPORTS BOWKMeansTrainer : public BOWTrainer
{
public:
    BOWKMeansTrainer( int clusterCount, const TermCriteria& termcrit=TermCriteria(),
                      int attempts=3, int flags=KMEANS_PP_CENTERS );
    virtual ~BOWKMeansTrainer();

    // Returns trained vocabulary (i.e. cluster centers).
    virtual Mat cluster() const;
    virtual Mat cluster( const Mat& descriptors ) const;

protected:

    int clusterCount;
    TermCriteria termcrit;
    int attempts;
    int flags;
};

/*
 * Class to compute image descriptor using bag of visual words.
 */
class CV_EXPORTS BOWImgDescriptorExtractor
{
public:
    BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor,
                               const Ptr<DescriptorMatcher>& dmatcher );
    virtual ~BOWImgDescriptorExtractor();

    void setVocabulary( const Mat& vocabulary );
    const Mat& getVocabulary() const;
    void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& imgDescriptor,
                  vector<vector<int> >* pointIdxsOfClusters=0, Mat* descriptors=0 );
    // compute() is not constant because DescriptorMatcher::match is not constant

    int descriptorSize() const;
    int descriptorType() const;

protected:
    Mat vocabulary;
    Ptr<DescriptorExtractor> dextractor;
    Ptr<DescriptorMatcher> dmatcher;
};

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
