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
#include "opencv2/flann/flann.hpp"

#ifdef __cplusplus
#include <limits>

extern "C" {
#endif

typedef struct CvSURFPoint
{
    CvPoint2D32f pt;

    int          laplacian;
    int          size;
    float        dir;
    float        hessian;

} CvSURFPoint;

CV_INLINE CvSURFPoint cvSURFPoint( CvPoint2D32f pt, int laplacian,
                                   int size, float dir CV_DEFAULT(0),
                                   float hessian CV_DEFAULT(0))
{
    CvSURFPoint kp;

    kp.pt        = pt;
    kp.laplacian = laplacian;
    kp.size      = size;
    kp.dir       = dir;
    kp.hessian   = hessian;

    return kp;
}

typedef struct CvSURFParams
{
    int    extended;
    double hessianThreshold;

    int    nOctaves;
    int    nOctaveLayers;

} CvSURFParams;

CVAPI(CvSURFParams) cvSURFParams( double hessianThreshold, int extended CV_DEFAULT(0) );

// If useProvidedKeyPts!=0, keypoints are not detected, but descriptors are computed
//  at the locations provided in keypoints (a CvSeq of CvSURFPoint).
CVAPI(void) cvExtractSURF( const CvArr* img, const CvArr* mask,
                           CvSeq** keypoints, CvSeq** descriptors,
                           CvMemStorage* storage, CvSURFParams params, int useProvidedKeyPts CV_DEFAULT(0)  );

/*!
 Maximal Stable Regions Parameters
*/
typedef struct CvMSERParams
{
    //! delta, in the code, it compares (size_{i}-size_{i-delta})/size_{i-delta}
    int delta;
    //! prune the area which bigger than maxArea
    int maxArea;
    //! prune the area which smaller than minArea
    int minArea;
    //! prune the area have simliar size to its children
    float maxVariation;
    //! trace back to cut off mser with diversity < min_diversity
    float minDiversity;
    
    /////// the next few params for MSER of color image
    
    //! for color image, the evolution steps
    int maxEvolution;
    //! the area threshold to cause re-initialize
    double areaThreshold;
    //! ignore too small margin
    double minMargin;
    //! the aperture size for edge blur
    int edgeBlurSize;
} CvMSERParams;

CVAPI(CvMSERParams) cvMSERParams( int delta CV_DEFAULT(5), int min_area CV_DEFAULT(60),
                           int max_area CV_DEFAULT(14400), float max_variation CV_DEFAULT(.25f),
                           float min_diversity CV_DEFAULT(.2f), int max_evolution CV_DEFAULT(200),
                           double area_threshold CV_DEFAULT(1.01),
                           double min_margin CV_DEFAULT(.003),
                           int edge_blur_size CV_DEFAULT(5) );

// Extracts the contours of Maximally Stable Extremal Regions
CVAPI(void) cvExtractMSER( CvArr* _img, CvArr* _mask, CvSeq** contours, CvMemStorage* storage, CvMSERParams params );


typedef struct CvStarKeypoint
{
    CvPoint pt;
    int size;
    float response;
} CvStarKeypoint;

CV_INLINE CvStarKeypoint cvStarKeypoint(CvPoint pt, int size, float response)
{
    CvStarKeypoint kpt;
    kpt.pt = pt;
    kpt.size = size;
    kpt.response = response;
    return kpt;
}

typedef struct CvStarDetectorParams
{
    int maxSize;
    int responseThreshold;
    int lineThresholdProjected;
    int lineThresholdBinarized;
    int suppressNonmaxSize;
} CvStarDetectorParams;

CV_INLINE CvStarDetectorParams cvStarDetectorParams(
    int maxSize CV_DEFAULT(45),
    int responseThreshold CV_DEFAULT(30),
    int lineThresholdProjected CV_DEFAULT(10),
    int lineThresholdBinarized CV_DEFAULT(8),
    int suppressNonmaxSize CV_DEFAULT(5))
{
    CvStarDetectorParams params;
    params.maxSize = maxSize;
    params.responseThreshold = responseThreshold;
    params.lineThresholdProjected = lineThresholdProjected;
    params.lineThresholdBinarized = lineThresholdBinarized;
    params.suppressNonmaxSize = suppressNonmaxSize;

    return params;
}

CVAPI(CvSeq*) cvGetStarKeypoints( const CvArr* img, CvMemStorage* storage,
        CvStarDetectorParams params CV_DEFAULT(cvStarDetectorParams()));

#ifdef __cplusplus
}

namespace cv
{
    struct CV_EXPORTS DefaultRngAuto
    {
        const uint64 old_state;

        DefaultRngAuto() : old_state(theRNG().state) { theRNG().state = (uint64)-1; }
        ~DefaultRngAuto() { theRNG().state = old_state; }

        DefaultRngAuto& operator=(const DefaultRngAuto&);
    };


// CvAffinePose: defines a parameterized affine transformation of an image patch.
// An image patch is rotated on angle phi (in degrees), then scaled lambda1 times
// along horizontal and lambda2 times along vertical direction, and then rotated again
// on angle (theta - phi).
class CV_EXPORTS CvAffinePose
{
public:
    float phi;
    float theta;
    float lambda1;
    float lambda2;
};

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
    static void convert(const std::vector<KeyPoint>& keypoints,
                        CV_OUT std::vector<Point2f>& points2f,
                        const std::vector<int>& keypointIndexes=std::vector<int>());
    //! converts vector of points to the vector of keypoints, where each keypoint is assigned the same size and the same orientation
    static void convert(const std::vector<Point2f>& points2f,
                        CV_OUT std::vector<KeyPoint>& keypoints,
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
    static void runByKeypointSize( vector<KeyPoint>& keypoints, float minSize, float maxSize=std::numeric_limits<float>::max() );
    /*
     * Remove keypoints from some image by mask for pixels of this image.
     */
    static void runByPixelsMask( vector<KeyPoint>& keypoints, const Mat& mask );
    /*
     * Remove duplicated keypoints.
     */
    static void removeDuplicated( vector<KeyPoint>& keypoints );
};

/*!
 SIFT implementation.
 
 The class implements SIFT algorithm by D. Lowe.
*/

class CV_EXPORTS SIFT
{
public:
    struct CV_EXPORTS CommonParams
    {
        static const int DEFAULT_NOCTAVES = 4;
        static const int DEFAULT_NOCTAVE_LAYERS = 3;
        static const int DEFAULT_FIRST_OCTAVE = -1;
        enum { FIRST_ANGLE = 0, AVERAGE_ANGLE = 1 };

        CommonParams();
        CommonParams( int _nOctaves, int _nOctaveLayers, int /*_firstOctave*/, int /*_angleMode*/ );
        int nOctaves, nOctaveLayers;
        int firstOctave; // it is not used now (firstOctave == 0 always)
        int angleMode;   // it is not used now
    };

    struct CV_EXPORTS DetectorParams
    {
        static double GET_DEFAULT_THRESHOLD() { return 0.04; }
        static double GET_DEFAULT_EDGE_THRESHOLD() { return 10.0; }

        DetectorParams();
        DetectorParams( double _threshold, double _edgeThreshold );
        double threshold, edgeThreshold;
    };

    struct CV_EXPORTS DescriptorParams
    {
        static double GET_DEFAULT_MAGNIFICATION() { return 3.0; }
        static const bool DEFAULT_IS_NORMALIZE = true;
        static const int DESCRIPTOR_SIZE = 128;

        DescriptorParams();
        DescriptorParams( double _magnification, bool /*_isNormalize*/, bool _recalculateAngles );
        DescriptorParams( bool _recalculateAngles );
        double magnification;
        bool isNormalize; // it is not used now (true always)
        bool recalculateAngles;
    };

    SIFT();
    //! sift-detector constructor
    SIFT( double _threshold, double _edgeThreshold,
          int _nOctaves=CommonParams::DEFAULT_NOCTAVES,
          int _nOctaveLayers=CommonParams::DEFAULT_NOCTAVE_LAYERS,
          int _firstOctave=CommonParams::DEFAULT_FIRST_OCTAVE,
          int _angleMode=CommonParams::FIRST_ANGLE );
    //! sift-descriptor constructor
    SIFT( double _magnification, bool _isNormalize=true,
          bool _recalculateAngles = true,
          int _nOctaves=CommonParams::DEFAULT_NOCTAVES,
          int _nOctaveLayers=CommonParams::DEFAULT_NOCTAVE_LAYERS,
          int _firstOctave=CommonParams::DEFAULT_FIRST_OCTAVE,
          int _angleMode=CommonParams::FIRST_ANGLE );
    SIFT( const CommonParams& _commParams,
          const DetectorParams& _detectorParams = DetectorParams(),
          const DescriptorParams& _descriptorParams = DescriptorParams() );

    //! returns the descriptor size in floats (128)
    int descriptorSize() const;
    //! finds the keypoints using SIFT algorithm
    void operator()(const Mat& img, const Mat& mask,
                    vector<KeyPoint>& keypoints) const;
    //! finds the keypoints and computes descriptors for them using SIFT algorithm.
    //! Optionally it can compute descriptors for the user-provided keypoints
    void operator()(const Mat& img, const Mat& mask,
                    vector<KeyPoint>& keypoints,
                    Mat& descriptors,
                    bool useProvidedKeypoints=false) const;

    CommonParams getCommonParams () const;
    DetectorParams getDetectorParams () const;
    DescriptorParams getDescriptorParams () const;

protected:
    CommonParams commParams;
    DetectorParams detectorParams;
    DescriptorParams descriptorParams;
};

    
/*!
 SURF implementation.
 
 The class implements SURF algorithm by H. Bay et al.
 */
class CV_EXPORTS_W SURF : public CvSURFParams
{
public:
    //! the default constructor
    CV_WRAP SURF();
    //! the full constructor taking all the necessary parameters
    CV_WRAP SURF(double _hessianThreshold, int _nOctaves=4,
         int _nOctaveLayers=2, bool _extended=false);

    //! returns the descriptor size in float's (64 or 128)
    CV_WRAP int descriptorSize() const;
    //! finds the keypoints using fast hessian detector used in SURF
    CV_WRAP_AS(detect) void operator()(const Mat& img, const Mat& mask,
                    CV_OUT vector<KeyPoint>& keypoints) const;
    //! finds the keypoints and computes their descriptors. Optionally it can compute descriptors for the user-provided keypoints
    CV_WRAP_AS(detect) void operator()(const Mat& img, const Mat& mask,
                    CV_OUT vector<KeyPoint>& keypoints,
                    CV_OUT vector<float>& descriptors,
                    bool useProvidedKeypoints=false) const;
};

/*!
 ORB implementation.
*/
class CV_EXPORTS ORB
{
public:

  /** the size of the signature in bytes */
  enum { kBytes = 32 };

  struct CV_EXPORTS CommonParams
  {
    enum { DEFAULT_N_LEVELS = 3, DEFAULT_FIRST_LEVEL = 0};

    /** default constructor */
    CommonParams(float scale_factor = 1.2f, unsigned int n_levels = DEFAULT_N_LEVELS, int edge_threshold = 31,
                 unsigned int first_level = DEFAULT_FIRST_LEVEL) :
      scale_factor_(scale_factor), n_levels_(n_levels), first_level_(first_level >= n_levels ? 0 : first_level),
      edge_threshold_(edge_threshold)
    {
      // No other patch size is supported right now
      patch_size_ = 31;
    }
    void read(const FileNode& fn);
    void write(FileStorage& fs) const;

    /** Coefficient by which we divide the dimensions from one scale pyramid level to the next */
    float scale_factor_;
    /** The number of levels in the scale pyramid */
    unsigned int n_levels_;
    /** The level at which the image is given
     * if 1, that means we will also look at the image scale_factor_ times bigger
     */
    unsigned int first_level_;
    /** How far from the boundary the points should be */
    int edge_threshold_;

    friend class ORB;
  protected:
    /** The size of the patch that will be used for orientation and comparisons */
    int patch_size_;
  };

  /** Constructor
   * @param n_features the number of desired features
   * @param detector_params parameters to use
   */
  ORB(size_t n_features = 500, const CommonParams & detector_params = CommonParams());

  /** destructor to empty the patterns */
  ~ORB();

  /** returns the descriptor size in bytes */
  int descriptorSize() const;

  /** Compute the ORB features and descriptors on an image
   * @param img the image to compute the features and descriptors on
   * @param mask the mask to apply
   * @param keypoints the resulting keypoints
   */
  void
  operator()(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::KeyPoint> & keypoints);

  /** Compute the ORB features and descriptors on an image
   * @param img the image to compute the features and descriptors on
   * @param mask the mask to apply
   * @param keypoints the resulting keypoints
   * @param descriptors the resulting descriptors
   * @param useProvidedKeypoints if true, the keypoints are used as an input
   */
  void
  operator()(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors,
             bool useProvidedKeypoints = false);

private:
  /** The size of the patch used when comparing regions in the patterns */
  static const int kKernelWidth = 5;

  /** Compute the ORB features and descriptors on an image
   * @param image the image to compute the features and descriptors on
   * @param mask the mask to apply
   * @param keypoints the resulting keypoints
   * @param descriptors the resulting descriptors
   * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
   * @param do_descriptors if true, also computes the descriptors
   */
  void
  operator()(const cv::Mat &image, const cv::Mat &mask, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors,
             bool do_keypoints, bool do_descriptors);

  /** Compute the ORB keypoints on an image
   * @param image_pyramid the image pyramid to compute the features and descriptors on
   * @param mask_pyramid the masks to apply at every level
   * @param keypoints the resulting keypoints, clustered per level
   */
  void computeKeyPoints(const std::vector<cv::Mat>& image_pyramid, const std::vector<cv::Mat>& mask_pyramid,
                        std::vector<std::vector<cv::KeyPoint> >& keypoints) const;

  /** Compute the ORB keypoint orientations
   * @param image the image to compute the features and descriptors on
   * @param integral_image the integral image of the image (can be empty, but the computation will be slower)
   * @param level the scale at which we compute the orientation
   * @param keypoints the resulting keypoints
   */
  void
  computeOrientation(const cv::Mat& image, const cv::Mat& integral_image, unsigned int level,
                     std::vector<cv::KeyPoint>& keypoints) const;

  /** Compute the ORB descriptors
   * @param image the image to compute the features and descriptors on
   * @param integral_image the integral image of the image (can be empty, but the computation will be slower)
   * @param level the scale at which we compute the orientation
   * @param keypoints the keypoints to use
   * @param descriptors the resulting descriptors
   */
  void
  computeDescriptors(const cv::Mat& image, const cv::Mat& integral_image, unsigned int level,
                     std::vector<cv::KeyPoint>& keypoints, cv::Mat & descriptors) const;

  /** Compute the integral image and upadte the cached values
   * @param image the image to compute the features and descriptors on
   * @param level the scale at which we compute the orientation
   * @param descriptors the resulting descriptors
   */
  void computeIntegralImage(const cv::Mat & image, unsigned int level, cv::Mat &integral_image);

  /** Parameters tuning ORB */
  CommonParams params_;

  /** size of the half patch used for orientation computation, see Rosin - 1999 - Measuring Corner Properties */
  int half_patch_size_;

  /** pre-computed offsets used for the Harris verification, one vector per scale */
  std::vector<std::vector<int> > orientation_horizontal_offsets_;
  std::vector<std::vector<int> > orientation_vertical_offsets_;

  /** The steps of the integral images for each scale */
  std::vector<size_t> integral_image_steps_;

  /** The number of desired features per scale */
  std::vector<size_t> n_features_per_level_;

  /** The overall number of desired features */
  size_t n_features_;

  /** the end of a row in a circular patch */
  std::vector<int> u_max_;

  /** The patterns for each level (the patterns are the same, but not their offset */
  class OrbPatterns;
  std::vector<OrbPatterns*> patterns_;
};

/*!
 Maximal Stable Extremal Regions class.
 
 The class implements MSER algorithm introduced by J. Matas.
 Unlike SIFT, SURF and many other detectors in OpenCV, this is salient region detector,
 not the salient point detector.
 
 It returns the regions, each of those is encoded as a contour.
*/
class CV_EXPORTS_W MSER : public CvMSERParams
{
public:
    //! the default constructor
    CV_WRAP MSER();
    //! the full constructor
    CV_WRAP MSER( int _delta, int _min_area, int _max_area,
          double _max_variation, double _min_diversity,
          int _max_evolution, double _area_threshold,
          double _min_margin, int _edge_blur_size );
    //! the operator that extracts the MSERs from the image or the specific part of it
    CV_WRAP_AS(detect) void operator()( const Mat& image,
        CV_OUT vector<vector<Point> >& msers, const Mat& mask ) const;
};

/*!
 The "Star" Detector.
 
 The class implements the keypoint detector introduced by K. Konolige.
*/
class CV_EXPORTS_W StarDetector : public CvStarDetectorParams
{
public:
    //! the default constructor
    CV_WRAP StarDetector();
    //! the full constructor
    CV_WRAP StarDetector(int _maxSize, int _responseThreshold,
                 int _lineThresholdProjected,
                 int _lineThresholdBinarized,
                 int _suppressNonmaxSize);
    //! finds the keypoints in the image
    CV_WRAP_AS(detect) void operator()(const Mat& image,
                CV_OUT vector<KeyPoint>& keypoints) const;
};

//! detects corners using FAST algorithm by E. Rosten
CV_EXPORTS void FAST( const Mat& image, CV_OUT vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSupression=true );

/*!
 The Patch Generator class 
*/
class CV_EXPORTS PatchGenerator
{
public:
    PatchGenerator();
    PatchGenerator(double _backgroundMin, double _backgroundMax,
                   double _noiseRange, bool _randomBlur=true,
                   double _lambdaMin=0.6, double _lambdaMax=1.5,
                   double _thetaMin=-CV_PI, double _thetaMax=CV_PI,
                   double _phiMin=-CV_PI, double _phiMax=CV_PI );
    void operator()(const Mat& image, Point2f pt, Mat& patch, Size patchSize, RNG& rng) const;
    void operator()(const Mat& image, const Mat& transform, Mat& patch,
                    Size patchSize, RNG& rng) const;
    void warpWholeImage(const Mat& image, Mat& matT, Mat& buf,
                        CV_OUT Mat& warped, int border, RNG& rng) const;
    void generateRandomTransform(Point2f srcCenter, Point2f dstCenter,
                                 CV_OUT Mat& transform, RNG& rng,
                                 bool inverse=false) const;
    void setAffineParam(double lambda, double theta, double phi);
    
    double backgroundMin, backgroundMax;
    double noiseRange;
    bool randomBlur;
    double lambdaMin, lambdaMax;
    double thetaMin, thetaMax;
    double phiMin, phiMax;
};


class CV_EXPORTS LDetector
{
public:
    LDetector();
    LDetector(int _radius, int _threshold, int _nOctaves,
              int _nViews, double _baseFeatureSize, double _clusteringDistance);
    void operator()(const Mat& image,
                    CV_OUT vector<KeyPoint>& keypoints,
                    int maxCount=0, bool scaleCoords=true) const;
    void operator()(const vector<Mat>& pyr,
                    CV_OUT vector<KeyPoint>& keypoints,
                    int maxCount=0, bool scaleCoords=true) const;
    void getMostStable2D(const Mat& image, CV_OUT vector<KeyPoint>& keypoints,
                         int maxCount, const PatchGenerator& patchGenerator) const;
    void setVerbose(bool verbose);
    
    void read(const FileNode& node);
    void write(FileStorage& fs, const String& name=String()) const;
    
    int radius;
    int threshold;
    int nOctaves;
    int nViews;
    bool verbose;
    
    double baseFeatureSize;
    double clusteringDistance;
};

typedef LDetector YAPE;

class CV_EXPORTS FernClassifier
{
public:
    FernClassifier();
    FernClassifier(const FileNode& node);
    FernClassifier(const vector<vector<Point2f> >& points,
                   const vector<Mat>& refimgs,
                   const vector<vector<int> >& labels=vector<vector<int> >(),
                   int _nclasses=0, int _patchSize=PATCH_SIZE,
                   int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                   int _nstructs=DEFAULT_STRUCTS,
                   int _structSize=DEFAULT_STRUCT_SIZE,
                   int _nviews=DEFAULT_VIEWS,
                   int _compressionMethod=COMPRESSION_NONE,
                   const PatchGenerator& patchGenerator=PatchGenerator());
    virtual ~FernClassifier();
    virtual void read(const FileNode& n);
    virtual void write(FileStorage& fs, const String& name=String()) const;
    virtual void trainFromSingleView(const Mat& image,
                                     const vector<KeyPoint>& keypoints,
                                     int _patchSize=PATCH_SIZE,
                                     int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                                     int _nstructs=DEFAULT_STRUCTS,
                                     int _structSize=DEFAULT_STRUCT_SIZE,
                                     int _nviews=DEFAULT_VIEWS,
                                     int _compressionMethod=COMPRESSION_NONE,
                                     const PatchGenerator& patchGenerator=PatchGenerator());
    virtual void train(const vector<vector<Point2f> >& points,
                       const vector<Mat>& refimgs,
                       const vector<vector<int> >& labels=vector<vector<int> >(),
                       int _nclasses=0, int _patchSize=PATCH_SIZE,
                       int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                       int _nstructs=DEFAULT_STRUCTS,
                       int _structSize=DEFAULT_STRUCT_SIZE,
                       int _nviews=DEFAULT_VIEWS,
                       int _compressionMethod=COMPRESSION_NONE,
                       const PatchGenerator& patchGenerator=PatchGenerator());
    virtual int operator()(const Mat& img, Point2f kpt, vector<float>& signature) const;
    virtual int operator()(const Mat& patch, vector<float>& signature) const;
    virtual void clear();
    virtual bool empty() const;
    void setVerbose(bool verbose);
    
    int getClassCount() const;
    int getStructCount() const;
    int getStructSize() const;
    int getSignatureSize() const;
    int getCompressionMethod() const;
    Size getPatchSize() const;
    
    struct Feature
    {
        uchar x1, y1, x2, y2;
        Feature() : x1(0), y1(0), x2(0), y2(0) {}
        Feature(int _x1, int _y1, int _x2, int _y2)
        : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
        {}
        template<typename _Tp> bool operator ()(const Mat_<_Tp>& patch) const
        { return patch(y1,x1) > patch(y2, x2); }
    };
    
    enum
    {
        PATCH_SIZE = 31,
        DEFAULT_STRUCTS = 50,
        DEFAULT_STRUCT_SIZE = 9,
        DEFAULT_VIEWS = 5000,
        DEFAULT_SIGNATURE_SIZE = 176,
        COMPRESSION_NONE = 0,
        COMPRESSION_RANDOM_PROJ = 1,
        COMPRESSION_PCA = 2,
        DEFAULT_COMPRESSION_METHOD = COMPRESSION_NONE
    };
    
protected:
    virtual void prepare(int _nclasses, int _patchSize, int _signatureSize,
                         int _nstructs, int _structSize,
                         int _nviews, int _compressionMethod);
    virtual void finalize(RNG& rng);
    virtual int getLeaf(int fidx, const Mat& patch) const;
    
    bool verbose;
    int nstructs;
    int structSize;
    int nclasses;
    int signatureSize;
    int compressionMethod;
    int leavesPerStruct;
    Size patchSize;
    vector<Feature> features;
    vector<int> classCounters;
    vector<float> posteriors;
};


/****************************************************************************************\
*                                 Calonder Classifier                                    *
\****************************************************************************************/

struct RTreeNode;

struct CV_EXPORTS BaseKeypoint
{
  int x;
  int y;
  IplImage* image;

  BaseKeypoint()
    : x(0), y(0), image(NULL)
  {}

  BaseKeypoint(int x, int y, IplImage* image)
    : x(x), y(y), image(image)
  {}
};

class CV_EXPORTS RandomizedTree
{
public:
  friend class RTreeClassifier;

  static const uchar PATCH_SIZE = 32;
  static const int DEFAULT_DEPTH = 9;
  static const int DEFAULT_VIEWS = 5000;
  static const size_t DEFAULT_REDUCED_NUM_DIM = 176;
  static float GET_LOWER_QUANT_PERC() { return .03f; }
  static float GET_UPPER_QUANT_PERC() { return .92f; }

  RandomizedTree();
  ~RandomizedTree();

  void train(std::vector<BaseKeypoint> const& base_set, RNG &rng,
             int depth, int views, size_t reduced_num_dim, int num_quant_bits);
  void train(std::vector<BaseKeypoint> const& base_set, RNG &rng,
             PatchGenerator &make_patch, int depth, int views, size_t reduced_num_dim,
             int num_quant_bits);

  // following two funcs are EXPERIMENTAL (do not use unless you know exactly what you do)
  static void quantizeVector(float *vec, int dim, int N, float bnds[2], int clamp_mode=0);
  static void quantizeVector(float *src, int dim, int N, float bnds[2], uchar *dst);

  // patch_data must be a 32x32 array (no row padding)
  float* getPosterior(uchar* patch_data);
  const float* getPosterior(uchar* patch_data) const;
  uchar* getPosterior2(uchar* patch_data);
  const uchar* getPosterior2(uchar* patch_data) const;

  void read(const char* file_name, int num_quant_bits);
  void read(std::istream &is, int num_quant_bits);
  void write(const char* file_name) const;
  void write(std::ostream &os) const;

  int classes() { return classes_; }
  int depth() { return depth_; }

  //void setKeepFloatPosteriors(bool b) { keep_float_posteriors_ = b; }
  void discardFloatPosteriors() { freePosteriors(1); }

  inline void applyQuantization(int num_quant_bits) { makePosteriors2(num_quant_bits); }

  // debug
  void savePosteriors(std::string url, bool append=false);
  void savePosteriors2(std::string url, bool append=false);

private:
  int classes_;
  int depth_;
  int num_leaves_;
  std::vector<RTreeNode> nodes_;
  float **posteriors_;        // 16-bytes aligned posteriors
  uchar **posteriors2_;     // 16-bytes aligned posteriors
  std::vector<int> leaf_counts_;

  void createNodes(int num_nodes, RNG &rng);
  void allocPosteriorsAligned(int num_leaves, int num_classes);
  void freePosteriors(int which);    // which: 1=posteriors_, 2=posteriors2_, 3=both
  void init(int classes, int depth, RNG &rng);
  void addExample(int class_id, uchar* patch_data);
  void finalize(size_t reduced_num_dim, int num_quant_bits);
  int getIndex(uchar* patch_data) const;
  inline float* getPosteriorByIndex(int index);
  inline const float* getPosteriorByIndex(int index) const;
  inline uchar* getPosteriorByIndex2(int index);
  inline const uchar* getPosteriorByIndex2(int index) const;
  //void makeRandomMeasMatrix(float *cs_phi, PHI_DISTR_TYPE dt, size_t reduced_num_dim);
  void convertPosteriorsToChar();
  void makePosteriors2(int num_quant_bits);
  void compressLeaves(size_t reduced_num_dim);
  void estimateQuantPercForPosteriors(float perc[2]);
};


inline uchar* getData(IplImage* image)
{
  return reinterpret_cast<uchar*>(image->imageData);
}

inline float* RandomizedTree::getPosteriorByIndex(int index)
{
  return const_cast<float*>(const_cast<const RandomizedTree*>(this)->getPosteriorByIndex(index));
}

inline const float* RandomizedTree::getPosteriorByIndex(int index) const
{
  return posteriors_[index];
}

inline uchar* RandomizedTree::getPosteriorByIndex2(int index)
{
  return const_cast<uchar*>(const_cast<const RandomizedTree*>(this)->getPosteriorByIndex2(index));
}

inline const uchar* RandomizedTree::getPosteriorByIndex2(int index) const
{
  return posteriors2_[index];
}

struct CV_EXPORTS RTreeNode
{
  short offset1, offset2;

  RTreeNode() {}
  RTreeNode(uchar x1, uchar y1, uchar x2, uchar y2)
    : offset1(y1*RandomizedTree::PATCH_SIZE + x1),
      offset2(y2*RandomizedTree::PATCH_SIZE + x2)
  {}

  //! Left child on 0, right child on 1
  inline bool operator() (uchar* patch_data) const
  {
    return patch_data[offset1] > patch_data[offset2];
  }
};

class CV_EXPORTS RTreeClassifier
{
public:
  static const int DEFAULT_TREES = 48;
  static const size_t DEFAULT_NUM_QUANT_BITS = 4;

  RTreeClassifier();
  void train(std::vector<BaseKeypoint> const& base_set,
             RNG &rng,
             int num_trees = RTreeClassifier::DEFAULT_TREES,
             int depth = RandomizedTree::DEFAULT_DEPTH,
             int views = RandomizedTree::DEFAULT_VIEWS,
             size_t reduced_num_dim = RandomizedTree::DEFAULT_REDUCED_NUM_DIM,
             int num_quant_bits = DEFAULT_NUM_QUANT_BITS);
  void train(std::vector<BaseKeypoint> const& base_set,
             RNG &rng,
             PatchGenerator &make_patch,
             int num_trees = RTreeClassifier::DEFAULT_TREES,
             int depth = RandomizedTree::DEFAULT_DEPTH,
             int views = RandomizedTree::DEFAULT_VIEWS,
             size_t reduced_num_dim = RandomizedTree::DEFAULT_REDUCED_NUM_DIM,
             int num_quant_bits = DEFAULT_NUM_QUANT_BITS);

  // sig must point to a memory block of at least classes()*sizeof(float|uchar) bytes
  void getSignature(IplImage *patch, uchar *sig) const;
  void getSignature(IplImage *patch, float *sig) const;
  void getSparseSignature(IplImage *patch, float *sig, float thresh) const;
  // TODO: deprecated in favor of getSignature overload, remove
  void getFloatSignature(IplImage *patch, float *sig) const { getSignature(patch, sig); }

  static int countNonZeroElements(float *vec, int n, double tol=1e-10);
  static inline void safeSignatureAlloc(uchar **sig, int num_sig=1, int sig_len=176);
  static inline uchar* safeSignatureAlloc(int num_sig=1, int sig_len=176);

  inline int classes() const { return classes_; }
  inline int original_num_classes() const { return original_num_classes_; }

  void setQuantization(int num_quant_bits);
  void discardFloatPosteriors();

  void read(const char* file_name);
  void read(std::istream &is);
  void write(const char* file_name) const;
  void write(std::ostream &os) const;

  // experimental and debug
  void saveAllFloatPosteriors(std::string file_url);
  void saveAllBytePosteriors(std::string file_url);
  void setFloatPosteriorsFromTextfile_176(std::string url);
  float countZeroElements();

  std::vector<RandomizedTree> trees_;

private:
  int classes_;
  int num_quant_bits_;
  mutable uchar **posteriors_;
  mutable unsigned short *ptemp_;
  int original_num_classes_;
  bool keep_floats_;
};

/****************************************************************************************\
*                                     One-Way Descriptor                                 *
\****************************************************************************************/

class CV_EXPORTS OneWayDescriptor
{
public:
    OneWayDescriptor();
    ~OneWayDescriptor();

    // allocates memory for given descriptor parameters
    void Allocate(int pose_count, CvSize size, int nChannels);

    // GenerateSamples: generates affine transformed patches with averaging them over small transformation variations.
    // If external poses and transforms were specified, uses them instead of generating random ones
    // - pose_count: the number of poses to be generated
    // - frontal: the input patch (can be a roi in a larger image)
    // - norm: if nonzero, normalizes the output patch so that the sum of pixel intensities is 1
    void GenerateSamples(int pose_count, IplImage* frontal, int norm = 0);

    // GenerateSamplesFast: generates affine transformed patches with averaging them over small transformation variations.
    // Uses precalculated transformed pca components.
    // - frontal: the input patch (can be a roi in a larger image)
    // - pca_hr_avg: pca average vector
    // - pca_hr_eigenvectors: pca eigenvectors
    // - pca_descriptors: an array of precomputed descriptors of pca components containing their affine transformations
    //   pca_descriptors[0] corresponds to the average, pca_descriptors[1]-pca_descriptors[pca_dim] correspond to eigenvectors
    void GenerateSamplesFast(IplImage* frontal, CvMat* pca_hr_avg,
                             CvMat* pca_hr_eigenvectors, OneWayDescriptor* pca_descriptors);

    // sets the poses and corresponding transforms
    void SetTransforms(CvAffinePose* poses, CvMat** transforms);

    // Initialize: builds a descriptor.
    // - pose_count: the number of poses to build. If poses were set externally, uses them rather than generating random ones
    // - frontal: input patch. Can be a roi in a larger image
    // - feature_name: the feature name to be associated with the descriptor
    // - norm: if 1, the affine transformed patches are normalized so that their sum is 1
    void Initialize(int pose_count, IplImage* frontal, const char* feature_name = 0, int norm = 0);

    // InitializeFast: builds a descriptor using precomputed descriptors of pca components
    // - pose_count: the number of poses to build
    // - frontal: input patch. Can be a roi in a larger image
    // - feature_name: the feature name to be associated with the descriptor
    // - pca_hr_avg: average vector for PCA
    // - pca_hr_eigenvectors: PCA eigenvectors (one vector per row)
    // - pca_descriptors: precomputed descriptors of PCA components, the first descriptor for the average vector
    // followed by the descriptors for eigenvectors
    void InitializeFast(int pose_count, IplImage* frontal, const char* feature_name,
                        CvMat* pca_hr_avg, CvMat* pca_hr_eigenvectors, OneWayDescriptor* pca_descriptors);

    // ProjectPCASample: unwarps an image patch into a vector and projects it into PCA space
    // - patch: input image patch
    // - avg: PCA average vector
    // - eigenvectors: PCA eigenvectors, one per row
    // - pca_coeffs: output PCA coefficients
    void ProjectPCASample(IplImage* patch, CvMat* avg, CvMat* eigenvectors, CvMat* pca_coeffs) const;

    // InitializePCACoeffs: projects all warped patches into PCA space
    // - avg: PCA average vector
    // - eigenvectors: PCA eigenvectors, one per row
    void InitializePCACoeffs(CvMat* avg, CvMat* eigenvectors);

    // EstimatePose: finds the closest match between an input patch and a set of patches with different poses
    // - patch: input image patch
    // - pose_idx: the output index of the closest pose
    // - distance: the distance to the closest pose (L2 distance)
    void EstimatePose(IplImage* patch, int& pose_idx, float& distance) const;

    // EstimatePosePCA: finds the closest match between an input patch and a set of patches with different poses.
    // The distance between patches is computed in PCA space
    // - patch: input image patch
    // - pose_idx: the output index of the closest pose
    // - distance: distance to the closest pose (L2 distance in PCA space)
    // - avg: PCA average vector. If 0, matching without PCA is used
    // - eigenvectors: PCA eigenvectors, one per row
    void EstimatePosePCA(CvArr* patch, int& pose_idx, float& distance, CvMat* avg, CvMat* eigenvalues) const;

    // GetPatchSize: returns the size of each image patch after warping (2 times smaller than the input patch)
    CvSize GetPatchSize() const
    {
        return m_patch_size;
    }

    // GetInputPatchSize: returns the required size of the patch that the descriptor is built from
    // (2 time larger than the patch after warping)
    CvSize GetInputPatchSize() const
    {
        return cvSize(m_patch_size.width*2, m_patch_size.height*2);
    }

    // GetPatch: returns a patch corresponding to specified pose index
    // - index: pose index
    // - return value: the patch corresponding to specified pose index
    IplImage* GetPatch(int index);

    // GetPose: returns a pose corresponding to specified pose index
    // - index: pose index
    // - return value: the pose corresponding to specified pose index
    CvAffinePose GetPose(int index) const;

    // Save: saves all patches with different poses to a specified path
    void Save(const char* path);

    // ReadByName: reads a descriptor from a file storage
    // - fs: file storage
    // - parent: parent node
    // - name: node name
    // - return value: 1 if succeeded, 0 otherwise
    int ReadByName(CvFileStorage* fs, CvFileNode* parent, const char* name);

    // ReadByName: reads a descriptor from a file node
    // - parent: parent node
    // - name: node name
    // - return value: 1 if succeeded, 0 otherwise
    int ReadByName(const FileNode &parent, const char* name);

    // Write: writes a descriptor into a file storage
    // - fs: file storage
    // - name: node name
    void Write(CvFileStorage* fs, const char* name);

    // GetFeatureName: returns a name corresponding to a feature
    const char* GetFeatureName() const;

    // GetCenter: returns the center of the feature
    CvPoint GetCenter() const;

    void SetPCADimHigh(int pca_dim_high) {m_pca_dim_high = pca_dim_high;};
    void SetPCADimLow(int pca_dim_low) {m_pca_dim_low = pca_dim_low;};

    int GetPCADimLow() const;
    int GetPCADimHigh() const;

    CvMat** GetPCACoeffs() const {return m_pca_coeffs;}

protected:
    int m_pose_count; // the number of poses
    CvSize m_patch_size; // size of each image
    IplImage** m_samples; // an array of length m_pose_count containing the patch in different poses
    IplImage* m_input_patch;
    IplImage* m_train_patch;
    CvMat** m_pca_coeffs; // an array of length m_pose_count containing pca decomposition of the patch in different poses
    CvAffinePose* m_affine_poses; // an array of poses
    CvMat** m_transforms; // an array of affine transforms corresponding to poses

    string m_feature_name; // the name of the feature associated with the descriptor
    CvPoint m_center; // the coordinates of the feature (the center of the input image ROI)

    int m_pca_dim_high; // the number of descriptor pca components to use for generating affine poses
    int m_pca_dim_low; // the number of pca components to use for comparison
};


// OneWayDescriptorBase: encapsulates functionality for training/loading a set of one way descriptors
// and finding the nearest closest descriptor to an input feature
class CV_EXPORTS OneWayDescriptorBase
{
public:

    // creates an instance of OneWayDescriptor from a set of training files
    // - patch_size: size of the input (large) patch
    // - pose_count: the number of poses to generate for each descriptor
    // - train_path: path to training files
    // - pca_config: the name of the file that contains PCA for small patches (2 times smaller
    // than patch_size each dimension
    // - pca_hr_config: the name of the file that contains PCA for large patches (of patch_size size)
    // - pca_desc_config: the name of the file that contains descriptors of PCA components
    OneWayDescriptorBase(CvSize patch_size, int pose_count, const char* train_path = 0, const char* pca_config = 0,
                         const char* pca_hr_config = 0, const char* pca_desc_config = 0, int pyr_levels = 1,
                         int pca_dim_high = 100, int pca_dim_low = 100);

    OneWayDescriptorBase(CvSize patch_size, int pose_count, const string &pca_filename, const string &train_path = string(), const string &images_list = string(),
                         float _scale_min = 0.7f, float _scale_max=1.5f, float _scale_step=1.2f, int pyr_levels = 1,
                         int pca_dim_high = 100, int pca_dim_low = 100);


    virtual ~OneWayDescriptorBase();
    void clear ();


    // Allocate: allocates memory for a given number of descriptors
    void Allocate(int train_feature_count);

    // AllocatePCADescriptors: allocates memory for pca descriptors
    void AllocatePCADescriptors();

    // returns patch size
    CvSize GetPatchSize() const {return m_patch_size;};
    // returns the number of poses for each descriptor
    int GetPoseCount() const {return m_pose_count;};

    // returns the number of pyramid levels
    int GetPyrLevels() const {return m_pyr_levels;};

    // returns the number of descriptors
    int GetDescriptorCount() const {return m_train_feature_count;};

    // CreateDescriptorsFromImage: creates descriptors for each of the input features
    // - src: input image
    // - features: input features
    // - pyr_levels: the number of pyramid levels
    void CreateDescriptorsFromImage(IplImage* src, const std::vector<cv::KeyPoint>& features);

    // CreatePCADescriptors: generates descriptors for PCA components, needed for fast generation of feature descriptors
    void CreatePCADescriptors();

    // returns a feature descriptor by feature index
    const OneWayDescriptor* GetDescriptor(int desc_idx) const {return &m_descriptors[desc_idx];};

    // FindDescriptor: finds the closest descriptor
    // - patch: input image patch
    // - desc_idx: output index of the closest descriptor to the input patch
    // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
    // - distance: distance from the input patch to the closest feature pose
    // - _scales: scales of the input patch for each descriptor
    // - scale_ranges: input scales variation (float[2])
    void FindDescriptor(IplImage* patch, int& desc_idx, int& pose_idx, float& distance, float* _scale = 0, float* scale_ranges = 0) const;

    // - patch: input image patch
    // - n: number of the closest indexes
    // - desc_idxs: output indexes of the closest descriptor to the input patch (n)
    // - pose_idx: output indexes of the closest pose of the closest descriptor to the input patch (n)
    // - distances: distance from the input patch to the closest feature pose (n)
    // - _scales: scales of the input patch
    // - scale_ranges: input scales variation (float[2])
    void FindDescriptor(IplImage* patch, int n, std::vector<int>& desc_idxs, std::vector<int>& pose_idxs,
                        std::vector<float>& distances, std::vector<float>& _scales, float* scale_ranges = 0) const;

    // FindDescriptor: finds the closest descriptor
    // - src: input image
    // - pt: center of the feature
    // - desc_idx: output index of the closest descriptor to the input patch
    // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
    // - distance: distance from the input patch to the closest feature pose
    void FindDescriptor(IplImage* src, cv::Point2f pt, int& desc_idx, int& pose_idx, float& distance) const;

    // InitializePoses: generates random poses
    void InitializePoses();

    // InitializeTransformsFromPoses: generates 2x3 affine matrices from poses (initializes m_transforms)
    void InitializeTransformsFromPoses();

    // InitializePoseTransforms: subsequently calls InitializePoses and InitializeTransformsFromPoses
    void InitializePoseTransforms();

    // InitializeDescriptor: initializes a descriptor
    // - desc_idx: descriptor index
    // - train_image: image patch (ROI is supported)
    // - feature_label: feature textual label
    void InitializeDescriptor(int desc_idx, IplImage* train_image, const char* feature_label);

    void InitializeDescriptor(int desc_idx, IplImage* train_image, const cv::KeyPoint& keypoint, const char* feature_label);

    // InitializeDescriptors: load features from an image and create descriptors for each of them
    void InitializeDescriptors(IplImage* train_image, const vector<cv::KeyPoint>& features,
                               const char* feature_label = "", int desc_start_idx = 0);

    // Write: writes this object to a file storage
    // - fs: output filestorage
    void Write (FileStorage &fs) const;
    
    // Read: reads OneWayDescriptorBase object from a file node
    // - fn: input file node    
    void Read (const FileNode &fn);

    // LoadPCADescriptors: loads PCA descriptors from a file
    // - filename: input filename
    int LoadPCADescriptors(const char* filename);

    // LoadPCADescriptors: loads PCA descriptors from a file node
    // - fn: input file node
    int LoadPCADescriptors(const FileNode &fn);

    // SavePCADescriptors: saves PCA descriptors to a file
    // - filename: output filename
    void SavePCADescriptors(const char* filename);

    // SavePCADescriptors: saves PCA descriptors to a file storage
    // - fs: output file storage
    void SavePCADescriptors(CvFileStorage* fs) const;

    // GeneratePCA: calculate and save PCA components and descriptors
    // - img_path: path to training PCA images directory
    // - images_list: filename with filenames of training PCA images
    void GeneratePCA(const char* img_path, const char* images_list, int pose_count=500);

    // SetPCAHigh: sets the high resolution pca matrices (copied to internal structures)
    void SetPCAHigh(CvMat* avg, CvMat* eigenvectors);

    // SetPCALow: sets the low resolution pca matrices (copied to internal structures)
    void SetPCALow(CvMat* avg, CvMat* eigenvectors);

    int GetLowPCA(CvMat** avg, CvMat** eigenvectors)
    {
        *avg = m_pca_avg;
        *eigenvectors = m_pca_eigenvectors;
        return m_pca_dim_low;
    };

    int GetPCADimLow() const {return m_pca_dim_low;};
    int GetPCADimHigh() const {return m_pca_dim_high;};

    void ConvertDescriptorsArrayToTree(); // Converting pca_descriptors array to KD tree

    // GetPCAFilename: get default PCA filename
    static string GetPCAFilename () { return "pca.yml"; }

    virtual bool empty() const { return m_train_feature_count <= 0 ? true : false; }

protected:
    CvSize m_patch_size; // patch size
    int m_pose_count; // the number of poses for each descriptor
    int m_train_feature_count; // the number of the training features
    OneWayDescriptor* m_descriptors; // array of train feature descriptors
    CvMat* m_pca_avg; // PCA average Vector for small patches
    CvMat* m_pca_eigenvectors; // PCA eigenvectors for small patches
    CvMat* m_pca_hr_avg; // PCA average Vector for large patches
    CvMat* m_pca_hr_eigenvectors; // PCA eigenvectors for large patches
    OneWayDescriptor* m_pca_descriptors; // an array of PCA descriptors

    cv::flann::Index* m_pca_descriptors_tree;
    CvMat* m_pca_descriptors_matrix;

    CvAffinePose* m_poses; // array of poses
    CvMat** m_transforms; // array of affine transformations corresponding to poses

    int m_pca_dim_high;
    int m_pca_dim_low;

    int m_pyr_levels;
    float scale_min;
    float scale_max;
    float scale_step;

    // SavePCAall: saves PCA components and descriptors to a file storage
    // - fs: output file storage
    void SavePCAall (FileStorage &fs) const;

    // LoadPCAall: loads PCA components and descriptors from a file node
    // - fn: input file node
    void LoadPCAall (const FileNode &fn);
};

class CV_EXPORTS OneWayDescriptorObject : public OneWayDescriptorBase
{
public:
    // creates an instance of OneWayDescriptorObject from a set of training files
    // - patch_size: size of the input (large) patch
    // - pose_count: the number of poses to generate for each descriptor
    // - train_path: path to training files
    // - pca_config: the name of the file that contains PCA for small patches (2 times smaller
    // than patch_size each dimension
    // - pca_hr_config: the name of the file that contains PCA for large patches (of patch_size size)
    // - pca_desc_config: the name of the file that contains descriptors of PCA components
    OneWayDescriptorObject(CvSize patch_size, int pose_count, const char* train_path, const char* pca_config,
                           const char* pca_hr_config = 0, const char* pca_desc_config = 0, int pyr_levels = 1);

    OneWayDescriptorObject(CvSize patch_size, int pose_count, const string &pca_filename,
                           const string &train_path = string (), const string &images_list = string (),
                           float _scale_min = 0.7f, float _scale_max=1.5f, float _scale_step=1.2f, int pyr_levels = 1);


    virtual ~OneWayDescriptorObject();

    // Allocate: allocates memory for a given number of features
    // - train_feature_count: the total number of features
    // - object_feature_count: the number of features extracted from the object
    void Allocate(int train_feature_count, int object_feature_count);


    void SetLabeledFeatures(const vector<cv::KeyPoint>& features) {m_train_features = features;};
    vector<cv::KeyPoint>& GetLabeledFeatures() {return m_train_features;};
    const vector<cv::KeyPoint>& GetLabeledFeatures() const {return m_train_features;};
    vector<cv::KeyPoint> _GetLabeledFeatures() const;

    // IsDescriptorObject: returns 1 if descriptor with specified index is positive, otherwise 0
    int IsDescriptorObject(int desc_idx) const;

    // MatchPointToPart: returns the part number of a feature if it matches one of the object parts, otherwise -1
    int MatchPointToPart(CvPoint pt) const;

    // GetDescriptorPart: returns the part number of the feature corresponding to a specified descriptor
    // - desc_idx: descriptor index
    int GetDescriptorPart(int desc_idx) const;


    void InitializeObjectDescriptors(IplImage* train_image, const vector<cv::KeyPoint>& features,
                                     const char* feature_label, int desc_start_idx = 0, float scale = 1.0f,
                                     int is_background = 0);

    // GetObjectFeatureCount: returns the number of object features
    int GetObjectFeatureCount() const {return m_object_feature_count;};

protected:
    int* m_part_id; // contains part id for each of object descriptors
    vector<cv::KeyPoint> m_train_features; // train features
    int m_object_feature_count; // the number of the positive features

};


/****************************************************************************************\
*                                    FeatureDetector                                     *
\****************************************************************************************/

/*
 * Abstract base class for 2D image feature detectors.
 */
class CV_EXPORTS FeatureDetector
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
    void detect( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    
    /*
     * Detect keypoints in an image set.
     * images       Image collection.
     * keypoints    Collection of keypoints detected in an input images. keypoints[i] is a set of keypoints detected in an images[i].
     * masks        Masks for image set. masks[i] is a mask for images[i].
     */
    void detect( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, const vector<Mat>& masks=vector<Mat>() ) const;

    // Read detector object from a file node.
    virtual void read( const FileNode& );
    // Read detector object from a file node.
    virtual void write( FileStorage& ) const;

    // Return true if detector object is empty
    virtual bool empty() const;

    // Create feature detector by detector name.
    static Ptr<FeatureDetector> create( const string& detectorType );

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const = 0;

    /*
     * Remove keypoints that are not in the mask.
     * Helper function, useful when wrapping a library call for keypoint detection that
     * does not support a mask argument.
     */
    static void removeInvalidPoints( const Mat& mask, vector<KeyPoint>& keypoints );
};

class CV_EXPORTS FastFeatureDetector : public FeatureDetector
{
public:
    FastFeatureDetector( int threshold=10, bool nonmaxSuppression=true );
    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    int threshold;
    bool nonmaxSuppression;
};


class CV_EXPORTS GoodFeaturesToTrackDetector : public FeatureDetector
{
public:
    class CV_EXPORTS Params
    {
    public:
        Params( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1.,
                int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
        void read( const FileNode& fn );
        void write( FileStorage& fs ) const;

        int maxCorners;
        double qualityLevel;
        double minDistance;
        int blockSize;
        bool useHarrisDetector;
        double k;
    };

    GoodFeaturesToTrackDetector( const GoodFeaturesToTrackDetector::Params& params=GoodFeaturesToTrackDetector::Params() );
    GoodFeaturesToTrackDetector( int maxCorners, double qualityLevel, double minDistance,
                                 int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    Params params;
};

class CV_EXPORTS MserFeatureDetector : public FeatureDetector
{
public:
    MserFeatureDetector( CvMSERParams params=cvMSERParams() );
    MserFeatureDetector( int delta, int minArea, int maxArea, double maxVariation, double minDiversity,
                         int maxEvolution, double areaThreshold, double minMargin, int edgeBlurSize );
    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    MSER mser;
};

class CV_EXPORTS StarFeatureDetector : public FeatureDetector
{
public:
    StarFeatureDetector( const CvStarDetectorParams& params=cvStarDetectorParams() );
    StarFeatureDetector( int maxSize, int responseThreshold=30, int lineThresholdProjected = 10,
                         int lineThresholdBinarized=8, int suppressNonmaxSize=5 );
    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    StarDetector star;
};

class CV_EXPORTS SiftFeatureDetector : public FeatureDetector
{
public:
    SiftFeatureDetector( const SIFT::DetectorParams& detectorParams=SIFT::DetectorParams(),
                         const SIFT::CommonParams& commonParams=SIFT::CommonParams() );
    SiftFeatureDetector( double threshold, double edgeThreshold,
                         int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES,
                         int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,
                         int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,
                         int angleMode=SIFT::CommonParams::FIRST_ANGLE );
    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    SIFT sift;
};

class CV_EXPORTS SurfFeatureDetector : public FeatureDetector
{
public:
    SurfFeatureDetector( double hessianThreshold=400., int octaves=3, int octaveLayers=4 );
    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    SURF surf;
};

/** Feature detector for the ORB feature
 * Basically fast followed by a Harris check
 */
class CV_EXPORTS OrbFeatureDetector : public cv::FeatureDetector
{
public:
  /** Default constructor
   * @param n_features the number of desired features
   * @param params parameters to use
   */
  OrbFeatureDetector(size_t n_features = 700, ORB::CommonParams params = ORB::CommonParams());

  virtual void read(const cv::FileNode&);
  virtual void write(cv::FileStorage&) const;

protected:
  virtual void
  detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask = cv::Mat()) const;
private:
  /** the ORB object we use for the computations */
  mutable ORB orb_;
  /** The parameters used */
  ORB::CommonParams params_;
  /** the number of features that need to be retrieved */
  unsigned int n_features_;
};

class CV_EXPORTS SimpleBlobDetector : public cv::FeatureDetector
{
public:
  struct CV_EXPORTS Params
  {
      Params();
      float thresholdStep;
      float minThreshold;
      float maxThreshold;
      float maxCentersDist;
      int defaultKeypointSize;
      size_t minRepeatability;
      bool computeRadius;
      bool isGrayscaleCentroid;
      int centroidROIMargin;

      bool filterByArea, filterByInertia, filterByCircularity, filterByColor, filterByConvexity;
      float minArea;
      float maxArea;
      float minCircularity;
      float minInertiaRatio;
      float minConvexity;
      uchar blobColor;
  };

  SimpleBlobDetector(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());

protected:
  struct CV_EXPORTS Center
  {
      Point2d location;
      double radius;
      double confidence;
  };

  virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
  virtual void findBlobs(const cv::Mat &image, const cv::Mat &binaryImage, std::vector<Center> &centers) const;

  Point2d computeGrayscaleCentroid(const cv::Mat &image, const std::vector<cv::Point> &contour) const;

  Params params;
};

class CV_EXPORTS DenseFeatureDetector : public FeatureDetector
{
public:
    class CV_EXPORTS Params
    {
    public:
        Params( float initFeatureScale=1.f, int featureScaleLevels=1, float featureScaleMul=0.1f,
                int initXyStep=6, int initImgBound=0, bool varyXyStepWithScale=true, bool varyImgBoundWithScale=false );
        float initFeatureScale;
        int featureScaleLevels;
        float featureScaleMul;

        int initXyStep;
        int initImgBound;

        bool varyXyStepWithScale;
        bool varyImgBoundWithScale;
    };

    DenseFeatureDetector( const DenseFeatureDetector::Params& params=DenseFeatureDetector::Params() );
    
	// TODO implement read/write

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    Params params;
};

/*
 * Adapts a detector to partition the source image into a grid and detect
 * points in each cell.
 */
class CV_EXPORTS GridAdaptedFeatureDetector : public FeatureDetector
{
public:
    /*
     * detector            Detector that will be adapted.
     * maxTotalKeypoints   Maximum count of keypoints detected on the image. Only the strongest keypoints
     *                      will be keeped.
     * gridRows            Grid rows count.
     * gridCols            Grid column count.
     */
    GridAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector, int maxTotalKeypoints=1000,
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
class CV_EXPORTS PyramidAdaptedFeatureDetector : public FeatureDetector
{
public:
    // maxLevel - The 0-based index of the last pyramid layer
    PyramidAdaptedFeatureDetector( const Ptr<FeatureDetector>& detector, int maxLevel=2 );
    
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

    virtual void tooFew(int min, int n_detected);
    virtual void tooMany(int max, int n_detected);
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

    virtual void tooFew(int min, int n_detected);
    virtual void tooMany(int max, int n_detected);
    virtual bool good() const;

    virtual Ptr<AdjusterAdapter> clone() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    double thresh_, init_thresh_, min_thresh_, max_thresh_;
    CvStarDetectorParams params_; //todo use these instead of thresh_
};

class CV_EXPORTS SurfAdjuster: public AdjusterAdapter
{
public:
    SurfAdjuster( double initial_thresh=400.f, double min_thresh=2, double max_thresh=1000 );

    virtual void tooFew(int min, int n_detected);
    virtual void tooMany(int max, int n_detected);
    virtual bool good() const;

    virtual Ptr<AdjusterAdapter> clone() const;

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    double thresh_, init_thresh_, min_thresh_, max_thresh_;
};

CV_EXPORTS Mat windowedMatchingMask( const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                                     float maxDeltaX, float maxDeltaY );

/****************************************************************************************\
*                                 DescriptorExtractor                                    *
\****************************************************************************************/

/*
 * Abstract base class for computing descriptors for image keypoints.
 *
 * In this interface we assume a keypoint descriptor can be represented as a
 * dense, fixed-dimensional vector of some basic type. Most descriptors used
 * in practice follow this pattern, as it makes it very easy to compute
 * distances between descriptors. Therefore we represent a collection of
 * descriptors as a cv::Mat, where each row is one keypoint descriptor.
 */
class CV_EXPORTS DescriptorExtractor
{
public:
    virtual ~DescriptorExtractor();

    /*
     * Compute the descriptors for a set of keypoints in an image.
     * image        The image.
     * keypoints    The input keypoints. Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Copmputed descriptors. Row i is the descriptor for keypoint i.
     */
    void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

    /*
     * Compute the descriptors for a keypoints collection detected in image collection.
     * images       Image collection.
     * keypoints    Input keypoints collection. keypoints[i] is keypoints detected in images[i].
     *              Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Descriptor collection. descriptors[i] are descriptors computed for set keypoints[i].
     */
    void compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors ) const;

    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;

    virtual int descriptorSize() const = 0;
    virtual int descriptorType() const = 0;

    virtual bool empty() const;

    static Ptr<DescriptorExtractor> create( const string& descriptorExtractorType );

protected:
    virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const = 0;

    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    static void removeBorderKeypoints( vector<KeyPoint>& keypoints,
                                       Size imageSize, int borderSize );
};

/*
 * SiftDescriptorExtractor
 */
class CV_EXPORTS SiftDescriptorExtractor : public DescriptorExtractor
{
public:
    SiftDescriptorExtractor( const SIFT::DescriptorParams& descriptorParams=SIFT::DescriptorParams(),
                             const SIFT::CommonParams& commonParams=SIFT::CommonParams() );
    SiftDescriptorExtractor( double magnification, bool isNormalize=true, bool recalculateAngles=true,
                             int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES,
                             int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,
                             int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,
                             int angleMode=SIFT::CommonParams::FIRST_ANGLE );

    virtual void read( const FileNode &fn );
    virtual void write( FileStorage &fs ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;

protected:
	virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    
    SIFT sift;
};

/*
 * SurfDescriptorExtractor
 */
class CV_EXPORTS SurfDescriptorExtractor : public DescriptorExtractor
{
public:
    SurfDescriptorExtractor( int nOctaves=4, int nOctaveLayers=2, bool extended=false );

    virtual void read( const FileNode &fn );
    virtual void write( FileStorage &fs ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;

protected:
	virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

    SURF surf;
};

/** The descriptor extractor for the ORB descriptor
 * There are two ways to speed up its computation:
 * - if you know the step size of the integral image, use setStepSize so that offsets are precomputed and cached
 * - if you know the integral image, use setIntegralImage so that it is not recomputed. This calls
 * setStepSize automatically
 */
class OrbDescriptorExtractor : public cv::DescriptorExtractor
{
public:
  /** default constructor
   * @param params parameters to use
   */
  OrbDescriptorExtractor(ORB::CommonParams params = ORB::CommonParams());

  /** destructor */
  ~OrbDescriptorExtractor()
  {
  }

  virtual int descriptorSize() const;
  virtual int descriptorType() const;

  virtual void read(const cv::FileNode&);
  virtual void write(cv::FileStorage&) const;

protected:
  void computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
private:
  /** the ORB object we use for the computations */
  mutable ORB orb_;
  /** The parameters used */
  ORB::CommonParams params_;
};

/*
 * CalonderDescriptorExtractor
 */
template<typename T>
class CV_EXPORTS CalonderDescriptorExtractor : public DescriptorExtractor
{
public:
    CalonderDescriptorExtractor( const string& classifierFile );

    virtual void read( const FileNode &fn );
    virtual void write( FileStorage &fs ) const;

    virtual int descriptorSize() const { return classifier_.classes(); }
    virtual int descriptorType() const { return DataType<T>::type; }

    virtual bool empty() const;

protected:
	virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

    RTreeClassifier classifier_;
    static const int BORDER_SIZE = 16;
};

template<typename T>
CalonderDescriptorExtractor<T>::CalonderDescriptorExtractor(const std::string& classifier_file)
{
    classifier_.read( classifier_file.c_str() );
}

template<typename T>
void CalonderDescriptorExtractor<T>::computeImpl( const cv::Mat& image,
                                              std::vector<cv::KeyPoint>& keypoints,
                                              cv::Mat& descriptors) const
{
    // Cannot compute descriptors for keypoints on the image border.
    KeyPointsFilter::runByImageBorder(keypoints, image.size(), BORDER_SIZE);

    /// @todo Check 16-byte aligned
    descriptors.create(keypoints.size(), classifier_.classes(), cv::DataType<T>::type);

    int patchSize = RandomizedTree::PATCH_SIZE;
    int offset = patchSize / 2;
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        cv::Point2f pt = keypoints[i].pt;
        IplImage ipl = image( Rect((int)(pt.x - offset), (int)(pt.y - offset), patchSize, patchSize) );
        classifier_.getSignature( &ipl, descriptors.ptr<T>(i));
    }
}

template<typename T>
void CalonderDescriptorExtractor<T>::read( const FileNode& )
{}

template<typename T>
void CalonderDescriptorExtractor<T>::write( FileStorage& ) const
{}

template<typename T>
bool CalonderDescriptorExtractor<T>::empty() const
{
    return classifier_.trees_.empty();
}

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

    virtual int descriptorSize() const;
    virtual int descriptorType() const;

    /// @todo read and write for brief

protected:
    virtual void computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const;

    typedef void(*PixelTestFn)(const Mat&, const std::vector<KeyPoint>&, Mat&);

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
struct CV_EXPORTS L2
{
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        ResultType result = ResultType();
        for( int i = 0; i < size; i++ )
        {
            ResultType diff = (ResultType)(a[i] - b[i]);
            result += diff*diff;
        }
        return (ResultType)sqrt((double)result);
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
        ResultType result = ResultType();
        for( int i = 0; i < size; i++ )
        {
            ResultType diff = a[i] - b[i];
            result += (ResultType)fabs( diff );
        }
        return result;
    }
};

/*
 * Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
 * bit count of A exclusive XOR'ed with B
 */
struct CV_EXPORTS HammingLUT
{
    typedef unsigned char ValueType;
    typedef int ResultType;

    /** this will count the bits in a ^ b
     */
    ResultType operator()( const unsigned char* a, const unsigned char* b, int size ) const;

    /** \brief given a byte, count the bits using a compile time generated look up table
     *  \param b the byte to count bits.  The look up table has an entry for all
     *  values of b, where that entry is the number of bits.
     *  \return the number of bits in byte b
     */
    static unsigned char byteBitsLookUp(unsigned char b);
};


/// Hamming distance functor, this one will try to use gcc's __builtin_popcountl
/// but will fall back on HammingLUT if not available
/// bit count of A exclusive XOR'ed with B
struct CV_EXPORTS Hamming
{
    typedef unsigned char ValueType;

    //! important that this is signed as weird behavior happens
    // in BruteForce if not
    typedef int ResultType;

    /** this will count the bits in a ^ b, using __builtin_popcountl try compiling with sse4
    */
    ResultType operator()(const unsigned char* a, const unsigned char* b, int size) const;
};


/****************************************************************************************\
*                                      DMatch                                            *
\****************************************************************************************/
/*
 * Struct for matching: query descriptor index, train descriptor index, train image index and distance between descriptors.
 */
struct CV_EXPORTS DMatch
{
    DMatch() : queryIdx(-1), trainIdx(-1), imgIdx(-1), distance(std::numeric_limits<float>::max()) {}
    DMatch( int _queryIdx, int _trainIdx, float _distance ) :
            queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1), distance(_distance) {}
    DMatch( int _queryIdx, int _trainIdx, int _imgIdx, float _distance ) :
            queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx), distance(_distance) {}

    int queryIdx; // query descriptor index
    int trainIdx; // train descriptor index
    int imgIdx;   // train image index

    float distance;

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
class CV_EXPORTS DescriptorMatcher
{
public:
    virtual ~DescriptorMatcher();

	/*
     * Add descriptors to train descriptor collection.
     * descriptors      Descriptors to add. Each descriptors[i] is a descriptors set from one image.
     */
    virtual void add( const vector<Mat>& descriptors );
    /*
     * Get train descriptors collection.
     */
    const vector<Mat>& getTrainDescriptors() const;
    /*
     * Clear train descriptors collection.
     */
    virtual void clear();

    /*
     * Return true if there are not train descriptors in collection.
     */
    virtual bool empty() const;
    /*
     * Return true if the matcher supports mask in match methods.
     */
    virtual bool isMaskSupported() const = 0;

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
    virtual void train();
    /*
     * Group of methods to match descriptors from image pair.
     * Method train() is run in this methods.
     */
    // Find one best match for each query descriptor (if mask is empty).
    void match( const Mat& queryDescriptors, const Mat& trainDescriptors,
                vector<DMatch>& matches, const Mat& mask=Mat() ) const;
    // Find k best matches for each query descriptor (in increasing order of distances).
    // compactResult is used when mask is not empty. If compactResult is false matches
    // vector will have the same size as queryDescriptors rows. If compactResult is true
    // matches vector will not contain matches for fully masked out query descriptors.
    void knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                   vector<vector<DMatch> >& matches, int k,
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
    void match( const Mat& queryDescriptors, vector<DMatch>& matches,
                const vector<Mat>& masks=vector<Mat>() );
    void knnMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
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

    static Ptr<DescriptorMatcher> create( const string& descriptorMatcherType );
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
template<class Distance>
class CV_EXPORTS BruteForceMatcher : public DescriptorMatcher
{
public:
    BruteForceMatcher( Distance d = Distance() ) : distance(d) {}
    virtual ~BruteForceMatcher() {}

    virtual bool isMaskSupported() const { return true; }

    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;

protected:
    virtual void knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    virtual void radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );

    Distance distance;

private:
    /*
     * Next two methods are used to implement specialization.
     */
    static void commonKnnMatchImpl( BruteForceMatcher<Distance>& matcher,
                    const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
                    const vector<Mat>& masks, bool compactResult );
    static void commonRadiusMatchImpl( BruteForceMatcher<Distance>& matcher,
                    const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                    const vector<Mat>& masks, bool compactResult );
};

template<class Distance>
Ptr<DescriptorMatcher> BruteForceMatcher<Distance>::clone( bool emptyTrainData ) const
{
    BruteForceMatcher* matcher = new BruteForceMatcher(distance);
    if( !emptyTrainData )
    {
        std::transform( trainDescCollection.begin(), trainDescCollection.end(),
                        matcher->trainDescCollection.begin(), clone_op );
    }
    return matcher;
}

template<class Distance>
void BruteForceMatcher<Distance>::knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
                                                const vector<Mat>& masks, bool compactResult )
{
    commonKnnMatchImpl( *this, queryDescriptors, matches, k, masks, compactResult );
}

template<class Distance>
void BruteForceMatcher<Distance>::radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches,
                                                   float maxDistance, const vector<Mat>& masks, bool compactResult )
{
    commonRadiusMatchImpl( *this, queryDescriptors, matches, maxDistance, masks, compactResult );
}

template<class Distance>
inline void BruteForceMatcher<Distance>::commonKnnMatchImpl( BruteForceMatcher<Distance>& matcher,
                          const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int knn,
                          const vector<Mat>& masks, bool compactResult )
 {
     typedef typename Distance::ValueType ValueType;
     typedef typename Distance::ResultType DistanceType;
	 CV_DbgAssert( !queryDescriptors.empty() );
     CV_Assert( DataType<ValueType>::type == queryDescriptors.type() );
     
     int dimension = queryDescriptors.cols;
     matches.reserve(queryDescriptors.rows);

     size_t imgCount = matcher.trainDescCollection.size();
     vector<Mat> allDists( imgCount ); // distances between one query descriptor and all train descriptors
     for( size_t i = 0; i < imgCount; i++ )
        allDists[i] = Mat( 1, matcher.trainDescCollection[i].rows, DataType<DistanceType>::type );

     for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
     {
         if( matcher.isMaskedOut( masks, qIdx ) )
         {
             if( !compactResult ) // push empty vector
                 matches.push_back( vector<DMatch>() );
         }
         else
         {
             // 1. compute distances between i-th query descriptor and all train descriptors
             for( size_t iIdx = 0; iIdx < imgCount; iIdx++ )
             {
                 CV_Assert( DataType<ValueType>::type == matcher.trainDescCollection[iIdx].type() ||  matcher.trainDescCollection[iIdx].empty() );
				 CV_Assert( queryDescriptors.cols == matcher.trainDescCollection[iIdx].cols || 
				            matcher.trainDescCollection[iIdx].empty() );

                 const ValueType* d1 = (const ValueType*)(queryDescriptors.data + queryDescriptors.step*qIdx);
                 allDists[iIdx].setTo( Scalar::all(std::numeric_limits<DistanceType>::max()) );
                 for( int tIdx = 0; tIdx < matcher.trainDescCollection[iIdx].rows; tIdx++ )
                 {
                     if( masks.empty() || matcher.isPossibleMatch(masks[iIdx], qIdx, tIdx) )
                     {
                         const ValueType* d2 = (const ValueType*)(matcher.trainDescCollection[iIdx].data +
                                                                  matcher.trainDescCollection[iIdx].step*tIdx);
                         allDists[iIdx].at<DistanceType>(0, tIdx) = matcher.distance(d1, d2, dimension);
                     }
                 }
             }

             // 2. choose k nearest matches for query[i]
             matches.push_back( vector<DMatch>() );
             vector<vector<DMatch> >::reverse_iterator curMatches = matches.rbegin();
             for( int k = 0; k < knn; k++ )
             {
                 DMatch bestMatch;
                 bestMatch.distance = std::numeric_limits<float>::max();
                 for( size_t iIdx = 0; iIdx < imgCount; iIdx++ )
                 {
                     if( !allDists[iIdx].empty() )
                     {
                         double minVal;
                         Point minLoc;
                         minMaxLoc( allDists[iIdx], &minVal, 0, &minLoc, 0 );
                         if( minVal < bestMatch.distance )
                                 bestMatch = DMatch( qIdx, minLoc.x, (int)iIdx, (float)minVal );
                     }
                 }
                 if( bestMatch.trainIdx == -1 )
                     break;

                 allDists[bestMatch.imgIdx].at<DistanceType>(0, bestMatch.trainIdx) = std::numeric_limits<DistanceType>::max();
                 curMatches->push_back( bestMatch );
             }
             //TODO should already be sorted at this point?
             std::sort( curMatches->begin(), curMatches->end() );
         }
     }
}

template<class Distance>
inline void BruteForceMatcher<Distance>::commonRadiusMatchImpl( BruteForceMatcher<Distance>& matcher,
                             const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                             const vector<Mat>& masks, bool compactResult )
{
    typedef typename Distance::ValueType ValueType;
    typedef typename Distance::ResultType DistanceType;
	CV_DbgAssert( !queryDescriptors.empty() );
    CV_Assert( DataType<ValueType>::type == queryDescriptors.type() );
    
    int dimension = queryDescriptors.cols;
    matches.reserve(queryDescriptors.rows);

    size_t imgCount = matcher.trainDescCollection.size();
    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        if( matcher.isMaskedOut( masks, qIdx ) )
        {
            if( !compactResult ) // push empty vector
                matches.push_back( vector<DMatch>() );
        }
        else
        {
            matches.push_back( vector<DMatch>() );
            vector<vector<DMatch> >::reverse_iterator curMatches = matches.rbegin();
            for( size_t iIdx = 0; iIdx < imgCount; iIdx++ )
            {
                CV_Assert( DataType<ValueType>::type == matcher.trainDescCollection[iIdx].type() ||
                           matcher.trainDescCollection[iIdx].empty() );
                CV_Assert( queryDescriptors.cols == matcher.trainDescCollection[iIdx].cols ||
						   matcher.trainDescCollection[iIdx].empty() );

                const ValueType* d1 = (const ValueType*)(queryDescriptors.data + queryDescriptors.step*qIdx);
                for( int tIdx = 0; tIdx < matcher.trainDescCollection[iIdx].rows; tIdx++ )
                {
                    if( masks.empty() || matcher.isPossibleMatch(masks[iIdx], qIdx, tIdx) )
                    {
                        const ValueType* d2 = (const ValueType*)(matcher.trainDescCollection[iIdx].data +
                                                                 matcher.trainDescCollection[iIdx].step*tIdx);
                        DistanceType d = matcher.distance(d1, d2, dimension);
                        if( d < maxDistance )
                            curMatches->push_back( DMatch( qIdx, tIdx, (int)iIdx, (float)d ) );
                    }
                }
            }
            std::sort( curMatches->begin(), curMatches->end() );
        }
    }
}

/*
 * BruteForceMatcher L2 specialization
 */
template<>
void BruteForceMatcher<L2<float> >::knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
                                                  const vector<Mat>& masks, bool compactResult );
template<>
void BruteForceMatcher<L2<float> >::radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches,
                                                     float maxDistance, const vector<Mat>& masks, bool compactResult );

/*
 * Flann based matcher
 */
class CV_EXPORTS FlannBasedMatcher : public DescriptorMatcher
{
public:
    FlannBasedMatcher( const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams(),
                       const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams() );

    virtual void add( const vector<Mat>& descriptors );
    virtual void clear();

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

/*
 *  OneWayDescriptorMatcher
 */
class OneWayDescriptorMatcher;
typedef OneWayDescriptorMatcher OneWayDescriptorMatch;

class CV_EXPORTS OneWayDescriptorMatcher : public GenericDescriptorMatcher
{
public:
    class CV_EXPORTS Params
    {
    public:
        static const int POSE_COUNT = 500;
        static const int PATCH_WIDTH = 24;
        static const int PATCH_HEIGHT = 24;
        static float GET_MIN_SCALE() { return 0.7f; }
        static float GET_MAX_SCALE() { return 1.5f; }
        static float GET_STEP_SCALE() { return 1.2f; }

        Params( int poseCount = POSE_COUNT,
                Size patchSize = Size(PATCH_WIDTH, PATCH_HEIGHT),
                string pcaFilename = string(),
                string trainPath = string(), string trainImagesList = string(),
                float minScale = GET_MIN_SCALE(), float maxScale = GET_MAX_SCALE(),
                float stepScale = GET_STEP_SCALE() );

        int poseCount;
        Size patchSize;
        string pcaFilename;
        string trainPath;
        string trainImagesList;

        float minScale, maxScale, stepScale;
    };

    OneWayDescriptorMatcher( const Params& params=Params() );
    virtual ~OneWayDescriptorMatcher();

    void initialize( const Params& params, const Ptr<OneWayDescriptorBase>& base=Ptr<OneWayDescriptorBase>() );

    // Clears keypoints storing in collection and OneWayDescriptorBase
    virtual void clear();

    virtual void train();

    virtual bool isMaskSupported();

    virtual void read( const FileNode &fn );
    virtual void write( FileStorage& fs ) const;

    virtual bool empty() const;

    virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;

protected:
    // Matches a set of keypoints from a single image of the training set. A rectangle with a center in a keypoint
    // and size (patch_width/2*scale, patch_height/2*scale) is cropped from the source image for each
    // keypoint. scale is iterated from DescriptorOneWayParams::min_scale to DescriptorOneWayParams::max_scale.
    // The minimum distance to each training patch with all its affine poses is found over all scales.
    // The class ID of a match is returned for each keypoint. The distance is calculated over PCA components
    // loaded with DescriptorOneWay::Initialize, kd tree is used for finding minimum distances.
    virtual void knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                               vector<vector<DMatch> >& matches, int k,
                               const vector<Mat>& masks, bool compactResult );
    virtual void radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                  vector<vector<DMatch> >& matches, float maxDistance,
                                  const vector<Mat>& masks, bool compactResult );

    Ptr<OneWayDescriptorBase> base;
    Params params;
    int prevTrainCount;
};

/*
 *  FernDescriptorMatcher
 */
class FernDescriptorMatcher;
typedef FernDescriptorMatcher FernDescriptorMatch;

class CV_EXPORTS FernDescriptorMatcher : public GenericDescriptorMatcher
{
public:
    class CV_EXPORTS Params
    {
    public:
        Params( int nclasses=0,
                int patchSize=FernClassifier::PATCH_SIZE,
                int signatureSize=FernClassifier::DEFAULT_SIGNATURE_SIZE,
                int nstructs=FernClassifier::DEFAULT_STRUCTS,
                int structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                int nviews=FernClassifier::DEFAULT_VIEWS,
                int compressionMethod=FernClassifier::COMPRESSION_NONE,
                const PatchGenerator& patchGenerator=PatchGenerator() );

        Params( const string& filename );

        int nclasses;
        int patchSize;
        int signatureSize;
        int nstructs;
        int structSize;
        int nviews;
        int compressionMethod;
        PatchGenerator patchGenerator;

        string filename;
    };

    FernDescriptorMatcher( const Params& params=Params() );
    virtual ~FernDescriptorMatcher();

    virtual void clear();

    virtual void train();

    virtual bool isMaskSupported();

    virtual void read( const FileNode &fn );
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

    void trainFernClassifier();
    void calcBestProbAndMatchIdx( const Mat& image, const Point2f& pt,
                                  float& bestProb, int& bestMatchIdx, vector<float>& signature );
    Ptr<FernClassifier> classifier;
    Params params;
    int prevTrainCount;
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
