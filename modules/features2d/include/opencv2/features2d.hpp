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

#include "opencv2/core.hpp"
#include "opencv2/flann/miniflann.hpp"

namespace cv
{

CV_EXPORTS bool initModule_features2d();

// //! writes vector of keypoints to the file storage
// CV_EXPORTS void write(FileStorage& fs, const String& name, const std::vector<KeyPoint>& keypoints);
// //! reads vector of keypoints from the specified file storage node
// CV_EXPORTS void read(const FileNode& node, CV_OUT std::vector<KeyPoint>& keypoints);

/*
 * A class filters a vector of keypoints.
 * Because now it is difficult to provide a convenient interface for all usage scenarios of the keypoints filter class,
 * it has only several needed by now static methods.
 */
class CV_EXPORTS KeyPointsFilter
{
public:
    KeyPointsFilter(){}

    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    static void runByImageBorder( std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize );
    /*
     * Remove keypoints of sizes out of range.
     */
    static void runByKeypointSize( std::vector<KeyPoint>& keypoints, float minSize,
                                   float maxSize=FLT_MAX );
    /*
     * Remove keypoints from some image by mask for pixels of this image.
     */
    static void runByPixelsMask( std::vector<KeyPoint>& keypoints, const Mat& mask );
    /*
     * Remove duplicated keypoints.
     */
    static void removeDuplicated( std::vector<KeyPoint>& keypoints );

    /*
     * Retain the specified number of the best keypoints (according to the response)
     */
    static void retainBest( std::vector<KeyPoint>& keypoints, int npoints );
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
    CV_WRAP void detect( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

    /*
     * Detect keypoints in an image set.
     * images       Image collection.
     * keypoints    Collection of keypoints detected in an input images. keypoints[i] is a set of keypoints detected in an images[i].
     * masks        Masks for image set. masks[i] is a mask for images[i].
     */
    void detect( InputArrayOfArrays images, std::vector<std::vector<KeyPoint> >& keypoints, InputArrayOfArrays masks=noArray() ) const;

    // Return true if detector object is empty
    CV_WRAP virtual bool empty() const;

    // Create feature detector by detector name.
    CV_WRAP static Ptr<FeatureDetector> create( const String& detectorType );

protected:
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const = 0;

    /*
     * Remove keypoints that are not in the mask.
     * Helper function, useful when wrapping a library call for keypoint detection that
     * does not support a mask argument.
     */
    static void removeInvalidPoints( const Mat & mask, std::vector<KeyPoint>& keypoints );
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
    CV_WRAP void compute( const Mat& image, CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints, CV_OUT Mat& descriptors ) const;

    /*
     * Compute the descriptors for a keypoints collection detected in image collection.
     * images       Image collection.
     * keypoints    Input keypoints collection. keypoints[i] is keypoints detected in images[i].
     *              Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Descriptor collection. descriptors[i] are descriptors computed for set keypoints[i].
     */
    void compute( const std::vector<Mat>& images, std::vector<std::vector<KeyPoint> >& keypoints, std::vector<Mat>& descriptors ) const;

    CV_WRAP virtual int descriptorSize() const = 0;
    CV_WRAP virtual int descriptorType() const = 0;
    CV_WRAP virtual int defaultNorm() const = 0;

    CV_WRAP virtual bool empty() const;

    CV_WRAP static Ptr<DescriptorExtractor> create( const String& descriptorExtractorType );

protected:
    virtual void computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const = 0;

    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    static void removeBorderKeypoints( std::vector<KeyPoint>& keypoints,
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
                                     CV_OUT std::vector<KeyPoint>& keypoints,
                                     OutputArray descriptors,
                                     bool useProvidedKeypoints=false ) const = 0;

    CV_WRAP void compute( const Mat& image, CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints, CV_OUT Mat& descriptors ) const;

    // Create feature detector and descriptor extractor by name.
    CV_WRAP static Ptr<Feature2D> create( const String& name );
};

/*!
  BRISK implementation
*/
class CV_EXPORTS_W BRISK : public Feature2D
{
public:
    CV_WRAP explicit BRISK(int thresh=30, int octaves=3, float patternScale=1.0f);

    virtual ~BRISK();

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    // Compute the BRISK features on an image
    void operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;

    // Compute the BRISK features and descriptors on an image
    void operator()( InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                      OutputArray descriptors, bool useProvidedKeypoints=false ) const;

    AlgorithmInfo* info() const;

    // custom setup
    CV_WRAP explicit BRISK(std::vector<float> &radiusList, std::vector<int> &numberList,
        float dMax=5.85f, float dMin=8.2f, std::vector<int> indexChange=std::vector<int>());

    // call this to generate the kernel:
    // circle of radius r (pixels), with n points;
    // short pairings with dMax, long pairings with dMin
    CV_WRAP void generateKernel(std::vector<float> &radiusList,
        std::vector<int> &numberList, float dMax=5.85f, float dMin=8.2f,
        std::vector<int> indexChange=std::vector<int>());

protected:

    void computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

    void computeKeypointsNoOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;
    void computeDescriptorsAndOrOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                                       OutputArray descriptors, bool doDescriptors, bool doOrientation,
                                       bool useProvidedKeypoints) const;

    // Feature parameters
    CV_PROP_RW int threshold;
    CV_PROP_RW int octaves;

    // some helper structures for the Brisk pattern representation
    struct BriskPatternPoint{
        float x;         // x coordinate relative to center
        float y;         // x coordinate relative to center
        float sigma;     // Gaussian smoothing sigma
    };
    struct BriskShortPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
    };
    struct BriskLongPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
        int weighted_dx; // 1024.0/dx
        int weighted_dy; // 1024.0/dy
    };
    inline int smoothedIntensity(const cv::Mat& image,
                const cv::Mat& integral,const float key_x,
                const float key_y, const unsigned int scale,
                const unsigned int rot, const unsigned int point) const;
    // pattern properties
    BriskPatternPoint* patternPoints_;     //[i][rotation][scale]
    unsigned int points_;                 // total number of collocation points
    float* scaleList_;                     // lists the scaling per scale index [scale]
    unsigned int* sizeList_;             // lists the total pattern size per scale index [scale]
    static const unsigned int scales_;    // scales discretization
    static const float scalerange_;     // span of sizes 40->4 Octaves - else, this needs to be adjusted...
    static const unsigned int n_rot_;    // discretization of the rotation look-up

    // pairs
    int strings_;                        // number of uchars the descriptor consists of
    float dMax_;                         // short pair maximum distance
    float dMin_;                         // long pair maximum distance
    BriskShortPair* shortPairs_;         // d<_dMax
    BriskLongPair* longPairs_;             // d>_dMin
    unsigned int noShortPairs_;         // number of shortParis
    unsigned int noLongPairs_;             // number of longParis

    // general
    static const float basicSize_;
};


/*!
 ORB implementation.
*/
class CV_EXPORTS_W ORB : public Feature2D
{
public:
    // the size of the signature in bytes
    enum { kBytes = 32, HARRIS_SCORE=0, FAST_SCORE=1 };

    CV_WRAP explicit ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
        int firstLevel = 0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31 );

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    // Compute the ORB features and descriptors on an image
    void operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;

    // Compute the ORB features and descriptors on an image
    void operator()( InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false ) const;

    AlgorithmInfo* info() const;

protected:

    void computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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
  FREAK implementation
*/
class CV_EXPORTS FREAK : public DescriptorExtractor
{
public:
    /** Constructor
         * @param orientationNormalized enable orientation normalization
         * @param scaleNormalized enable scale normalization
         * @param patternScale scaling of the description pattern
         * @param nbOctave number of octaves covered by the detected keypoints
         * @param selectedPairs (optional) user defined selected pairs
    */
    explicit FREAK( bool orientationNormalized = true,
           bool scaleNormalized = true,
           float patternScale = 22.0f,
           int nOctaves = 4,
           const std::vector<int>& selectedPairs = std::vector<int>());
    FREAK( const FREAK& rhs );
    FREAK& operator=( const FREAK& );

    virtual ~FREAK();

    /** returns the descriptor length in bytes */
    virtual int descriptorSize() const;

    /** returns the descriptor type */
    virtual int descriptorType() const;

    /** returns the default norm type */
    virtual int defaultNorm() const;

    /** select the 512 "best description pairs"
         * @param images grayscale images set
         * @param keypoints set of detected keypoints
         * @param corrThresh correlation threshold
         * @param verbose print construction information
         * @return list of best pair indexes
    */
    std::vector<int> selectPairs( const std::vector<Mat>& images, std::vector<std::vector<KeyPoint> >& keypoints,
                      const double corrThresh = 0.7, bool verbose = true );

    AlgorithmInfo* info() const;

    enum
    {
        NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45
    };

protected:
    virtual void computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    void buildPattern();
    uchar meanIntensity( const Mat& image, const Mat& integral, const float kp_x, const float kp_y,
                         const unsigned int scale, const unsigned int rot, const unsigned int point ) const;

    bool orientationNormalized; //true if the orientation is normalized, false otherwise
    bool scaleNormalized; //true if the scale is normalized, false otherwise
    double patternScale; //scaling of the pattern
    int nOctaves; //number of octaves
    bool extAll; // true if all pairs need to be extracted for pairs selection

    double patternScale0;
    int nOctaves0;
    std::vector<int> selectedPairs0;

    struct PatternPoint
    {
        float x; // x coordinate relative to center
        float y; // x coordinate relative to center
        float sigma; // Gaussian smoothing sigma
    };

    struct DescriptionPair
    {
        uchar i; // index of the first point
        uchar j; // index of the second point
    };

    struct OrientationPair
    {
        uchar i; // index of the first point
        uchar j; // index of the second point
        int weight_dx; // dx/(norm_sq))*4096
        int weight_dy; // dy/(norm_sq))*4096
    };

    std::vector<PatternPoint> patternLookup; // look-up table for the pattern points (position+sigma of all points at all scales and orientation)
    int patternSizes[NB_SCALES]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
    DescriptionPair descriptionPairs[NB_PAIRS];
    OrientationPair orientationPairs[NB_ORIENPAIRS];
};


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
    CV_WRAP explicit MSER( int _delta=5, int _min_area=60, int _max_area=14400,
          double _max_variation=0.25, double _min_diversity=.2,
          int _max_evolution=200, double _area_threshold=1.01,
          double _min_margin=0.003, int _edge_blur_size=5 );

    //! the operator that extracts the MSERs from the image or the specific part of it
    CV_WRAP_AS(detect) void operator()( const Mat& image, CV_OUT std::vector<std::vector<Point> >& msers,
                                        const Mat& mask=Mat() ) const;
    AlgorithmInfo* info() const;

protected:
    void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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
                CV_OUT std::vector<KeyPoint>& keypoints) const;

    AlgorithmInfo* info() const;

protected:
    void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

    int maxSize;
    int responseThreshold;
    int lineThresholdProjected;
    int lineThresholdBinarized;
    int suppressNonmaxSize;
};

//! detects corners using FAST algorithm by E. Rosten
CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSupression=true );

CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSupression, int type );

class CV_EXPORTS_W FastFeatureDetector : public FeatureDetector
{
public:
    enum Type
    {
      TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2
    };

    CV_WRAP FastFeatureDetector( int threshold=10, bool nonmaxSuppression=true);
    CV_WRAP FastFeatureDetector( int threshold, bool nonmaxSuppression, int type);
    AlgorithmInfo* info() const;

protected:
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

    int threshold;
    bool nonmaxSuppression;
    int type;
};


class CV_EXPORTS_W GFTTDetector : public FeatureDetector
{
public:
    CV_WRAP GFTTDetector( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
                          int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
    AlgorithmInfo* info() const;

protected:
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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

  virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;
  virtual void findBlobs(const Mat &image, const Mat &binaryImage, std::vector<Center> &centers) const;

  Params params;
  AlgorithmInfo* info() const;
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
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

    double initFeatureScale;
    int featureScaleLevels;
    double featureScaleMul;

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
    CV_WRAP GridAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector=Ptr<FeatureDetector>(),
                                        int maxTotalKeypoints=1000,
                                        int gridRows=4, int gridCols=4 );

    // TODO implement read/write
    virtual bool empty() const;

    AlgorithmInfo* info() const;

protected:
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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

    static Ptr<AdjusterAdapter> create( const String& detectorType );
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

    /** \param adjuster an AdjusterAdapter that will do the detection and parameter adjustment
     *  \param max_features the maximum desired number of features
     *  \param max_iters the maximum number of times to try to adjust the feature detector params
     *          for the FastAdjuster this can be high, but with Star or Surf this can get time consuming
     *  \param min_features the minimum desired features
     */
    DynamicAdaptedFeatureDetector( const Ptr<AdjusterAdapter>& adjuster, int min_features=400, int max_features=500, int max_iters=5 );

    virtual bool empty() const;

protected:
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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
    virtual void detectImpl(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

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
    virtual void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;

    double thresh_, init_thresh_, min_thresh_, max_thresh_;
};

CV_EXPORTS Mat windowedMatchingMask( const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
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
    virtual int defaultNorm() const;

    virtual bool empty() const;

protected:
    virtual void computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const;

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
    virtual int defaultNorm() const;

    /// @todo read and write for brief

    AlgorithmInfo* info() const;

protected:
    virtual void computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const;

    typedef void(*PixelTestFn)(const Mat&, const std::vector<KeyPoint>&, Mat&);

    int bytes_;
    PixelTestFn test_fn_;
};


/****************************************************************************************\
*                                      Distance                                          *
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
    enum { normType = NORM_L2SQR };
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
    enum { normType = NORM_L2 };
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return (ResultType)std::sqrt((double)normL2Sqr<ValueType, ResultType>(a, b, size));
    }
};

/*
 * Manhattan distance (city block distance) functor
 */
template<class T>
struct CV_EXPORTS L1
{
    enum { normType = NORM_L1 };
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
    enum { normType = NORM_HAMMING };
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

template<int cellsize> struct HammingMultilevel
{
    enum { normType = NORM_HAMMING + (cellsize>1) };
    typedef unsigned char ValueType;
    typedef int ResultType;

    ResultType operator()( const unsigned char* a, const unsigned char* b, int size ) const
    {
        return normHamming(a, b, size, cellsize);
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
    CV_WRAP virtual void add( InputArrayOfArrays descriptors );
    /*
     * Get train descriptors collection.
     */
    CV_WRAP const std::vector<Mat>& getTrainDescriptors() const;
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
    CV_WRAP void match( InputArray queryDescriptors, InputArray trainDescriptors,
                CV_OUT std::vector<DMatch>& matches, InputArray mask=noArray() ) const;
    // Find k best matches for each query descriptor (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    CV_WRAP void knnMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                   CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                   InputArray mask=noArray(), bool compactResult=false ) const;
    // Find best matches for each query descriptor which have distance less than
    // maxDistance (in increasing order of distances).
    void radiusMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                      std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      InputArray mask=noArray(), bool compactResult=false ) const;
    /*
     * Group of methods to match descriptors from one image to image set.
     * See description of similar methods for matching image pair above.
     */
    CV_WRAP void match( InputArray queryDescriptors, CV_OUT std::vector<DMatch>& matches,
                        const std::vector<Mat>& masks=std::vector<Mat>() );
    CV_WRAP void knnMatch( InputArray queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                           const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false );
    void radiusMatch( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false );

    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    // Clone the matcher. If emptyTrainData is false the method create deep copy of the object, i.e. copies
    // both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
    // but with empty train data.
    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

    CV_WRAP static Ptr<DescriptorMatcher> create( const String& descriptorMatcherType );
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
        void set( const std::vector<Mat>& descriptors );
        virtual void clear();

        const Mat& getDescriptors() const;
        const Mat getDescriptor( int imgIdx, int localDescIdx ) const;
        const Mat getDescriptor( int globalDescIdx ) const;
        void getLocalIdx( int globalDescIdx, int& imgIdx, int& localDescIdx ) const;

        int size() const;

    protected:
        Mat mergedDescriptors;
        std::vector<int> startIdxs;
    };

    // In fact the matching is implemented only by the following two methods. These methods suppose
    // that the class object has been trained already. Public match methods call these methods
    // after calling train().
    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) = 0;
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) = 0;

    static bool isPossibleMatch( const Mat& mask, int queryIdx, int trainIdx );
    static bool isMaskedOut( const std::vector<Mat>& masks, int queryIdx );

    static Mat clone_op( Mat m ) { return m.clone(); }
    void checkMasks( const std::vector<Mat>& masks, int queryDescriptorsCount ) const;

    // Collection of descriptors from train images.
    std::vector<Mat> trainDescCollection;
    std::vector<UMat> utrainDescCollection;
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
class CV_EXPORTS_W BFMatcher : public DescriptorMatcher
{
public:
    CV_WRAP BFMatcher( int normType=NORM_L2, bool crossCheck=false );
    virtual ~BFMatcher() {}

    virtual bool isMaskSupported() const { return true; }

    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;

    AlgorithmInfo* info() const;
protected:
    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );

    int normType;
    bool crossCheck;
};


/*
 * Flann based matcher
 */
class CV_EXPORTS_W FlannBasedMatcher : public DescriptorMatcher
{
public:
    CV_WRAP FlannBasedMatcher( const Ptr<flann::IndexParams>& indexParams=makePtr<flann::KDTreeIndexParams>(),
                       const Ptr<flann::SearchParams>& searchParams=makePtr<flann::SearchParams>() );

    virtual void add( const std::vector<Mat>& descriptors );
    virtual void clear();

    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    virtual void train();
    virtual bool isMaskSupported() const;

    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;

    AlgorithmInfo* info() const;
protected:
    static void convertToDMatches( const DescriptorCollection& descriptors,
                                   const Mat& indices, const Mat& distances,
                                   std::vector<std::vector<DMatch> >& matches );

    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );

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
    virtual void add( const std::vector<Mat>& images,
                      std::vector<std::vector<KeyPoint> >& keypoints );

    const std::vector<Mat>& getTrainImages() const;
    const std::vector<std::vector<KeyPoint> >& getTrainKeypoints() const;

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
    void classify( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                           const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints ) const;
    // Classify keypoints from query image under train image collection.
    void classify( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints );

    /*
     * Group of methods to match keypoints from image pair.
     * Keypoints for which a descriptor cannot be computed are removed.
     * train() method is called here.
     */
    // Find one best match for each query descriptor (if mask is empty).
    void match( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints,
                std::vector<DMatch>& matches, const Mat& mask=Mat() ) const;
    // Find k best matches for each query keypoint (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows.
    // If compactResult is true matches vector will not contain matches for fully masked out query descriptors.
    void knnMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                   const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints,
                   std::vector<std::vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const;
    // Find best matches for each query descriptor which have distance less than maxDistance (in increasing order of distances).
    void radiusMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                      const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints,
                      std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const;
    /*
     * Group of methods to match keypoints from one image to image set.
     * See description of similar methods for matching image pair above.
     */
    void match( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                std::vector<DMatch>& matches, const std::vector<Mat>& masks=std::vector<Mat>() );
    void knnMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                   std::vector<std::vector<DMatch> >& matches, int k,
                   const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false );
    void radiusMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                      std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false );

    // Reads matcher object from a file node
    virtual void read( const FileNode& fn );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& fs ) const;

    // Return true if matching object is empty (e.g. feature detector or descriptor matcher are empty)
    virtual bool empty() const;

    // Clone the matcher. If emptyTrainData is false the method create deep copy of the object, i.e. copies
    // both parameters and train data. If emptyTrainData is true the method create object copy with current parameters
    // but with empty train data.
    virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

    static Ptr<GenericDescriptorMatcher> create( const String& genericDescritptorMatcherType,
                                                 const String &paramsFilename=String() );

protected:
    // In fact the matching is implemented only by the following two methods. These methods suppose
    // that the class object has been trained already. Public match methods call these methods
    // after calling train().
    virtual void knnMatchImpl( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                               std::vector<std::vector<DMatch> >& matches, int k,
                               const std::vector<Mat>& masks, bool compactResult ) = 0;
    virtual void radiusMatchImpl( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                                  std::vector<std::vector<DMatch> >& matches, float maxDistance,
                                  const std::vector<Mat>& masks, bool compactResult ) = 0;
    /*
     * A storage for sets of keypoints together with corresponding images and class IDs
     */
    class CV_EXPORTS KeyPointCollection
    {
    public:
        KeyPointCollection();
        KeyPointCollection( const KeyPointCollection& collection );
        void add( const std::vector<Mat>& images, const std::vector<std::vector<KeyPoint> >& keypoints );
        void clear();

        // Returns the total number of keypoints in the collection
        size_t keypointCount() const;
        size_t imageCount() const;

        const std::vector<std::vector<KeyPoint> >& getKeypoints() const;
        const std::vector<KeyPoint>& getKeypoints( int imgIdx ) const;
        const KeyPoint& getKeyPoint( int imgIdx, int localPointIdx ) const;
        const KeyPoint& getKeyPoint( int globalPointIdx ) const;
        void getLocalIdx( int globalPointIdx, int& imgIdx, int& localPointIdx ) const;

        const std::vector<Mat>& getImages() const;
        const Mat& getImage( int imgIdx ) const;

    protected:
        int pointCount;

        std::vector<Mat> images;
        std::vector<std::vector<KeyPoint> > keypoints;
        // global indices of the first points in each image, startIndices.size() = keypoints.size()
        std::vector<int> startIndices;

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

    virtual void add( const std::vector<Mat>& imgCollection,
                      std::vector<std::vector<KeyPoint> >& pointCollection );

    virtual void clear();

    virtual void train();

    virtual bool isMaskSupported();

    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;
    virtual bool empty() const;

    virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;

protected:
    virtual void knnMatchImpl( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                               std::vector<std::vector<DMatch> >& matches, int k,
                               const std::vector<Mat>& masks, bool compactResult );
    virtual void radiusMatchImpl( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                                  std::vector<std::vector<DMatch> >& matches, float maxDistance,
                                  const std::vector<Mat>& masks, bool compactResult );

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
CV_EXPORTS_W void drawKeypoints( const Mat& image, const std::vector<KeyPoint>& keypoints, CV_OUT Mat& outImage,
                               const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT );

// Draws matches of keypints from two images on output image.
CV_EXPORTS_W void drawMatches( const Mat& img1, const std::vector<KeyPoint>& keypoints1,
                             const Mat& img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, CV_OUT Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<char>& matchesMask=std::vector<char>(), int flags=DrawMatchesFlags::DEFAULT );

CV_EXPORTS_AS(drawMatchesKnn) void drawMatches( const Mat& img1, const std::vector<KeyPoint>& keypoints1,
                             const Mat& img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, CV_OUT Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), int flags=DrawMatchesFlags::DEFAULT );

/****************************************************************************************\
*   Functions to evaluate the feature detectors and [generic] descriptor extractors      *
\****************************************************************************************/

CV_EXPORTS void evaluateFeatureDetector( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                         std::vector<KeyPoint>* keypoints1, std::vector<KeyPoint>* keypoints2,
                                         float& repeatability, int& correspCount,
                                         const Ptr<FeatureDetector>& fdetector=Ptr<FeatureDetector>() );

CV_EXPORTS void computeRecallPrecisionCurve( const std::vector<std::vector<DMatch> >& matches1to2,
                                             const std::vector<std::vector<uchar> >& correctMatches1to2Mask,
                                             std::vector<Point2f>& recallPrecisionCurve );

CV_EXPORTS float getRecall( const std::vector<Point2f>& recallPrecisionCurve, float l_precision );
CV_EXPORTS int getNearestPoint( const std::vector<Point2f>& recallPrecisionCurve, float l_precision );

CV_EXPORTS void evaluateGenericDescriptorMatcher( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                                  std::vector<KeyPoint>& keypoints1, std::vector<KeyPoint>& keypoints2,
                                                  std::vector<std::vector<DMatch> >* matches1to2, std::vector<std::vector<uchar> >* correctMatches1to2Mask,
                                                  std::vector<Point2f>& recallPrecisionCurve,
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
    const std::vector<Mat>& getDescriptors() const;
    int descriptorsCount() const;

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
    std::vector<Mat> descriptors;
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
    BOWImgDescriptorExtractor( const Ptr<DescriptorMatcher>& dmatcher );
    virtual ~BOWImgDescriptorExtractor();

    void setVocabulary( const Mat& vocabulary );
    const Mat& getVocabulary() const;
    void compute( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0, Mat* descriptors=0 );
    void compute( const Mat& keypointDescriptors, Mat& imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0 );
    // compute() is not constant because DescriptorMatcher::match is not constant

    int descriptorSize() const;
    int descriptorType() const;

protected:
    Mat vocabulary;
    Ptr<DescriptorExtractor> dextractor;
    Ptr<DescriptorMatcher> dmatcher;
};

} /* namespace cv */

#endif
