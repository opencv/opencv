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

CV_EXPORTS bool initModule_features2d(void);

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
    CV_WRAP void compute( InputArray image, CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const;

    /*
     * Compute the descriptors for a keypoints collection detected in image collection.
     * images       Image collection.
     * keypoints    Input keypoints collection. keypoints[i] is keypoints detected in images[i].
     *              Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Descriptor collection. descriptors[i] are descriptors computed for set keypoints[i].
     */
    void compute( InputArrayOfArrays images, std::vector<std::vector<KeyPoint> >& keypoints, OutputArrayOfArrays descriptors ) const;

    CV_WRAP virtual int descriptorSize() const = 0;
    CV_WRAP virtual int descriptorType() const = 0;
    CV_WRAP virtual int defaultNorm() const = 0;

    CV_WRAP virtual bool empty() const;

    CV_WRAP static Ptr<DescriptorExtractor> create( const String& descriptorExtractorType );

protected:
    virtual void computeImpl( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const = 0;

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

    CV_WRAP void compute( InputArray image, CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const;

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

    void computeImpl( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const;
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

    void computeImpl( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const;
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
    CV_WRAP_AS(detect) void operator()( InputArray image, CV_OUT std::vector<std::vector<Point> >& msers,
                                        InputArray mask=noArray() ) const;
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

//! detects corners using FAST algorithm by E. Rosten
CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression=true );

CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression, int type );

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
  virtual void findBlobs(InputArray image, InputArray binaryImage, std::vector<Center> &centers) const;

  Params params;
  AlgorithmInfo* info() const;
};


// KAZE/AKAZE diffusivity
enum {
    DIFF_PM_G1 = 0,
    DIFF_PM_G2 = 1,
    DIFF_WEICKERT = 2,
    DIFF_CHARBONNIER = 3
};

// AKAZE descriptor type
enum {
    DESCRIPTOR_KAZE_UPRIGHT = 2, ///< Upright descriptors, not invariant to rotation
    DESCRIPTOR_KAZE = 3,
    DESCRIPTOR_MLDB_UPRIGHT = 4, ///< Upright descriptors, not invariant to rotation
    DESCRIPTOR_MLDB = 5
};

/*!
KAZE implementation
*/
class CV_EXPORTS_W KAZE : public Feature2D
{
public:
    CV_WRAP KAZE();
    CV_WRAP explicit KAZE(bool extended, bool upright, float threshold = 0.001f,
                          int octaves = 4, int sublevels = 4, int diffusivity = DIFF_PM_G2);

    virtual ~KAZE();

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    AlgorithmInfo* info() const;

    // Compute the KAZE features on an image
    void operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;

    // Compute the KAZE features and descriptors on an image
    void operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
        OutputArray descriptors, bool useProvidedKeypoints = false) const;

protected:
    void detectImpl(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask) const;
    void computeImpl(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) const;

    CV_PROP bool extended;
    CV_PROP bool upright;
    CV_PROP float threshold;
    CV_PROP int octaves;
    CV_PROP int sublevels;
    CV_PROP int diffusivity;
};

/*!
AKAZE implementation
*/
class CV_EXPORTS_W AKAZE : public Feature2D
{
public:
    CV_WRAP AKAZE();
    CV_WRAP explicit AKAZE(int descriptor_type, int descriptor_size = 0, int descriptor_channels = 3,
                   float threshold = 0.001f, int octaves = 4, int sublevels = 4, int diffusivity = DIFF_PM_G2);

    virtual ~AKAZE();

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    // Compute the AKAZE features on an image
    void operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;

    // Compute the AKAZE features and descriptors on an image
    void operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
        OutputArray descriptors, bool useProvidedKeypoints = false) const;

    AlgorithmInfo* info() const;

protected:

    void computeImpl(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) const;
    void detectImpl(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask = noArray()) const;

    CV_PROP int descriptor;
    CV_PROP int descriptor_channels;
    CV_PROP int descriptor_size;
    CV_PROP float threshold;
    CV_PROP int octaves;
    CV_PROP int sublevels;
    CV_PROP int diffusivity;
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
                        InputArrayOfArrays masks=noArray() );
    CV_WRAP void knnMatch( InputArray queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                           InputArrayOfArrays masks=noArray(), bool compactResult=false );
    void radiusMatch( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      InputArrayOfArrays masks=noArray(), bool compactResult=false );

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

    static bool isPossibleMatch( InputArray mask, int queryIdx, int trainIdx );
    static bool isMaskedOut( InputArrayOfArrays masks, int queryIdx );

    static Mat clone_op( Mat m ) { return m.clone(); }
    void checkMasks( InputArrayOfArrays masks, int queryDescriptorsCount ) const;

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

    virtual void add( InputArrayOfArrays descriptors );
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
CV_EXPORTS_W void drawKeypoints( InputArray image, const std::vector<KeyPoint>& keypoints, InputOutputArray outImage,
                               const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT );

// Draws matches of keypints from two images on output image.
CV_EXPORTS_W void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<char>& matchesMask=std::vector<char>(), int flags=DrawMatchesFlags::DEFAULT );

CV_EXPORTS_AS(drawMatchesKnn) void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, InputOutputArray outImg,
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

/****************************************************************************************\
*                                     Bag of visual words                                *
\****************************************************************************************/
/*
 * Abstract base class for training of a 'bag of visual words' vocabulary from a set of descriptors
 */
class CV_EXPORTS_W BOWTrainer
{
public:
    BOWTrainer();
    virtual ~BOWTrainer();

    CV_WRAP void add( const Mat& descriptors );
    CV_WRAP const std::vector<Mat>& getDescriptors() const;
    CV_WRAP int descriptorsCount() const;

    CV_WRAP virtual void clear();

    /*
     * Train visual words vocabulary, that is cluster training descriptors and
     * compute cluster centers.
     * Returns cluster centers.
     *
     * descriptors      Training descriptors computed on images keypoints.
     */
    CV_WRAP virtual Mat cluster() const = 0;
    CV_WRAP virtual Mat cluster( const Mat& descriptors ) const = 0;

protected:
    std::vector<Mat> descriptors;
    int size;
};

/*
 * This is BOWTrainer using cv::kmeans to get vocabulary.
 */
class CV_EXPORTS_W BOWKMeansTrainer : public BOWTrainer
{
public:
    CV_WRAP BOWKMeansTrainer( int clusterCount, const TermCriteria& termcrit=TermCriteria(),
                      int attempts=3, int flags=KMEANS_PP_CENTERS );
    virtual ~BOWKMeansTrainer();

    // Returns trained vocabulary (i.e. cluster centers).
    CV_WRAP virtual Mat cluster() const;
    CV_WRAP virtual Mat cluster( const Mat& descriptors ) const;

protected:

    int clusterCount;
    TermCriteria termcrit;
    int attempts;
    int flags;
};

/*
 * Class to compute image descriptor using bag of visual words.
 */
class CV_EXPORTS_W BOWImgDescriptorExtractor
{
public:
    CV_WRAP BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor,
                               const Ptr<DescriptorMatcher>& dmatcher );
    BOWImgDescriptorExtractor( const Ptr<DescriptorMatcher>& dmatcher );
    virtual ~BOWImgDescriptorExtractor();

    CV_WRAP void setVocabulary( const Mat& vocabulary );
    CV_WRAP const Mat& getVocabulary() const;
    void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0, Mat* descriptors=0 );
    void compute( InputArray keypointDescriptors, OutputArray imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0 );
    // compute() is not constant because DescriptorMatcher::match is not constant

    CV_WRAP_AS(compute) void compute2( const Mat& image, std::vector<KeyPoint>& keypoints, CV_OUT Mat& imgDescriptor )
    { compute(image,keypoints,imgDescriptor); }

    CV_WRAP int descriptorSize() const;
    CV_WRAP int descriptorType() const;

protected:
    Mat vocabulary;
    Ptr<DescriptorExtractor> dextractor;
    Ptr<DescriptorMatcher> dmatcher;
};

} /* namespace cv */

#endif
