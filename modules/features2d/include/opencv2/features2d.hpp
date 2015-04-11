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

/**
  @defgroup features2d 2D Features Framework
  @{
    @defgroup features2d_main Feature Detection and Description
    @defgroup features2d_match Descriptor Matchers

Matchers of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to
easily switch between different algorithms solving the same problem. This section is devoted to
matching descriptors that are represented as vectors in a multidimensional space. All objects that
implement vector descriptor matchers inherit the DescriptorMatcher interface.

@note
   -   An example explaining keypoint matching can be found at
        opencv_source_code/samples/cpp/descriptor_extractor_matcher.cpp
    -   An example on descriptor matching evaluation can be found at
        opencv_source_code/samples/cpp/detector_descriptor_matcher_evaluation.cpp
    -   An example on one to many image matching can be found at
        opencv_source_code/samples/cpp/matching_to_many_images.cpp

    @defgroup features2d_draw Drawing Function of Keypoints and Matches
    @defgroup features2d_category Object Categorization

This section describes approaches based on local 2D features and used to categorize objects.

@note
   -   A complete Bag-Of-Words sample can be found at
        opencv_source_code/samples/cpp/bagofwords_classification.cpp
    -   (Python) An example using the features2D framework to perform object categorization can be
        found at opencv_source_code/samples/python2/find_obj.py

  @}
 */

namespace cv
{

//! @addtogroup features2d
//! @{

// //! writes vector of keypoints to the file storage
// CV_EXPORTS void write(FileStorage& fs, const String& name, const std::vector<KeyPoint>& keypoints);
// //! reads vector of keypoints from the specified file storage node
// CV_EXPORTS void read(const FileNode& node, CV_OUT std::vector<KeyPoint>& keypoints);

/** @brief A class filters a vector of keypoints.

 Because now it is difficult to provide a convenient interface for all usage scenarios of the
 keypoints filter class, it has only several needed by now static methods.
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

/** @brief Abstract base class for 2D image feature detectors and descriptor extractors
*/
class CV_EXPORTS_W Feature2D : public virtual Algorithm
{
public:
    virtual ~Feature2D();

    /** @brief Detects keypoints in an image (first variant) or image set (second variant).

    @param image Image.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
    matrix with non-zero values in the region of interest.
     */
    CV_WRAP virtual void detect( InputArray image,
                                 CV_OUT std::vector<KeyPoint>& keypoints,
                                 InputArray mask=noArray() );

    /** @overload
    @param images Image set.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param masks Masks for each input image specifying where to look for keypoints (optional).
    masks[i] is a mask for images[i].
    */
    virtual void detect( InputArrayOfArrays images,
                         std::vector<std::vector<KeyPoint> >& keypoints,
                         InputArrayOfArrays masks=noArray() );

    /** @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
    (second variant).

    @param image Image.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
    descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
    descriptor for keypoint j-th keypoint.
     */
    CV_WRAP virtual void compute( InputArray image,
                                  CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors );

    /** @overload

    @param images Image set.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
    descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
    descriptor for keypoint j-th keypoint.
    */
    virtual void compute( InputArrayOfArrays images,
                          std::vector<std::vector<KeyPoint> >& keypoints,
                          OutputArrayOfArrays descriptors );

    /** Detects keypoints and computes the descriptors */
    CV_WRAP virtual void detectAndCompute( InputArray image, InputArray mask,
                                           CV_OUT std::vector<KeyPoint>& keypoints,
                                           OutputArray descriptors,
                                           bool useProvidedKeypoints=false );

    CV_WRAP virtual int descriptorSize() const;
    CV_WRAP virtual int descriptorType() const;
    CV_WRAP virtual int defaultNorm() const;

    //! Return true if detector object is empty
    CV_WRAP virtual bool empty() const;
};

/** Feature detectors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. All objects that implement keypoint detectors
inherit the FeatureDetector interface. */
typedef Feature2D FeatureDetector;

/** Extractors of keypoint descriptors in OpenCV have wrappers with a common interface that enables you
to easily switch between different algorithms solving the same problem. This section is devoted to
computing descriptors represented as vectors in a multidimensional space. All objects that implement
the vector descriptor extractors inherit the DescriptorExtractor interface.
 */
typedef Feature2D DescriptorExtractor;

//! @addtogroup features2d_main
//! @{

/** @brief Class implementing the BRISK keypoint detector and descriptor extractor, described in @cite LCS11 .
 */
class CV_EXPORTS_W BRISK : public Feature2D
{
public:
    /** @brief The BRISK constructor

    @param thresh AGAST detection threshold score.
    @param octaves detection octaves. Use 0 to do single scale.
    @param patternScale apply this scale to the pattern used for sampling the neighbourhood of a
    keypoint.
     */
    CV_WRAP static Ptr<BRISK> create(int thresh=30, int octaves=3, float patternScale=1.0f);

    /** @brief The BRISK constructor for a custom pattern

    @param radiusList defines the radii (in pixels) where the samples around a keypoint are taken (for
    keypoint scale 1).
    @param numberList defines the number of sampling points on the sampling circle. Must be the same
    size as radiusList..
    @param dMax threshold for the short pairings used for descriptor formation (in pixels for keypoint
    scale 1).
    @param dMin threshold for the long pairings used for orientation determination (in pixels for
    keypoint scale 1).
    @param indexChange index remapping of the bits. */
    CV_WRAP static Ptr<BRISK> create(const std::vector<float> &radiusList, const std::vector<int> &numberList,
        float dMax=5.85f, float dMin=8.2f, const std::vector<int>& indexChange=std::vector<int>());
};

/** @brief Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor

described in @cite RRKB11 . The algorithm uses FAST in pyramids to detect stable keypoints, selects
the strongest features using FAST or Harris response, finds their orientation using first-order
moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or
k-tuples) are rotated according to the measured orientation).
 */
class CV_EXPORTS_W ORB : public Feature2D
{
public:
    enum { kBytes = 32, HARRIS_SCORE=0, FAST_SCORE=1 };

    /** @brief The ORB constructor

    @param nfeatures The maximum number of features to retain.
    @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
    pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
    will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
    will mean that to cover certain scale range you will need more pyramid levels and so the speed
    will suffer.
    @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
    input_image_linear_size/pow(scaleFactor, nlevels).
    @param edgeThreshold This is size of the border where the features are not detected. It should
    roughly match the patchSize parameter.
    @param firstLevel It should be 0 in the current implementation.
    @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
    default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
    so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
    random points (of course, those point coordinates are random, but they are generated from the
    pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
    rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
    output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
    denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
    bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
    @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
    (the score is written to KeyPoint::score and is used to retain best nfeatures features);
    FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
    but it is a little faster to compute.
    @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
    pyramid layers the perceived image area covered by a feature will be larger.
    @param fastThreshold
     */
    CV_WRAP static Ptr<ORB> create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31,
        int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20);

    CV_WRAP virtual void setMaxFeatures(int maxFeatures) = 0;
    CV_WRAP virtual int getMaxFeatures() const = 0;

    CV_WRAP virtual void setScaleFactor(double scaleFactor) = 0;
    CV_WRAP virtual double getScaleFactor() const = 0;

    CV_WRAP virtual void setNLevels(int nlevels) = 0;
    CV_WRAP virtual int getNLevels() const = 0;

    CV_WRAP virtual void setEdgeThreshold(int edgeThreshold) = 0;
    CV_WRAP virtual int getEdgeThreshold() const = 0;

    CV_WRAP virtual void setFirstLevel(int firstLevel) = 0;
    CV_WRAP virtual int getFirstLevel() const = 0;

    CV_WRAP virtual void setWTA_K(int wta_k) = 0;
    CV_WRAP virtual int getWTA_K() const = 0;

    CV_WRAP virtual void setScoreType(int scoreType) = 0;
    CV_WRAP virtual int getScoreType() const = 0;

    CV_WRAP virtual void setPatchSize(int patchSize) = 0;
    CV_WRAP virtual int getPatchSize() const = 0;

    CV_WRAP virtual void setFastThreshold(int fastThreshold) = 0;
    CV_WRAP virtual int getFastThreshold() const = 0;
};

/** @brief Maximally stable extremal region extractor. :

The class encapsulates all the parameters of the MSER extraction algorithm (see
<http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions>). Also see
<http://code.opencv.org/projects/opencv/wiki/MSER> for useful comments and parameters description.

@note
   -   (Python) A complete example showing the use of the MSER detector can be found at
        opencv_source_code/samples/python2/mser.py
 */
class CV_EXPORTS_W MSER : public Feature2D
{
public:
    //! the full constructor
    CV_WRAP static Ptr<MSER> create( int _delta=5, int _min_area=60, int _max_area=14400,
          double _max_variation=0.25, double _min_diversity=.2,
          int _max_evolution=200, double _area_threshold=1.01,
          double _min_margin=0.003, int _edge_blur_size=5 );

    CV_WRAP virtual void detectRegions( InputArray image,
                                        CV_OUT std::vector<std::vector<Point> >& msers,
                                        std::vector<Rect>& bboxes ) = 0;

    CV_WRAP virtual void setDelta(int delta) = 0;
    CV_WRAP virtual int getDelta() const = 0;

    CV_WRAP virtual void setMinArea(int minArea) = 0;
    CV_WRAP virtual int getMinArea() const = 0;

    CV_WRAP virtual void setMaxArea(int maxArea) = 0;
    CV_WRAP virtual int getMaxArea() const = 0;

    CV_WRAP virtual void setPass2Only(bool f) = 0;
    CV_WRAP virtual bool getPass2Only() const = 0;
};

/** @overload */
CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression=true );

/** @brief Detects corners using the FAST algorithm

@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected corners
(keypoints).
@param type one of the three neighborhoods as defined in the paper:
FastFeatureDetector::TYPE_9_16, FastFeatureDetector::TYPE_7_12,
FastFeatureDetector::TYPE_5_8

Detects corners using the FAST algorithm by @cite Rosten06 .

@note In Python API, types are given as cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
cv2.FAST_FEATURE_DETECTOR_TYPE_7_12 and cv2.FAST_FEATURE_DETECTOR_TYPE_9_16. For corner
detection, use cv2.FAST.detect() method.
 */
CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression, int type );

//! @} features2d_main

//! @addtogroup features2d_main
//! @{

/** @brief Wrapping class for feature detection using the FAST method. :
 */
class CV_EXPORTS_W FastFeatureDetector : public Feature2D
{
public:
    enum
    {
        TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2,
        THRESHOLD = 10000, NONMAX_SUPPRESSION=10001, FAST_N=10002,
    };

    CV_WRAP static Ptr<FastFeatureDetector> create( int threshold=10,
                                                    bool nonmaxSuppression=true,
                                                    int type=FastFeatureDetector::TYPE_9_16 );

    CV_WRAP virtual void setThreshold(int threshold) = 0;
    CV_WRAP virtual int getThreshold() const = 0;

    CV_WRAP virtual void setNonmaxSuppression(bool f) = 0;
    CV_WRAP virtual bool getNonmaxSuppression() const = 0;

    CV_WRAP virtual void setType(int type) = 0;
    CV_WRAP virtual int getType() const = 0;
};

/** @overload */
CV_EXPORTS void AGAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression=true );

/** @brief Detects corners using the AGAST algorithm

@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected corners
(keypoints).
@param type one of the four neighborhoods as defined in the paper:
AgastFeatureDetector::AGAST_5_8, AgastFeatureDetector::AGAST_7_12d,
AgastFeatureDetector::AGAST_7_12s, AgastFeatureDetector::OAST_9_16

Detects corners using the AGAST algorithm by @cite mair2010_agast .

 */
CV_EXPORTS void AGAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression, int type );
//! @} features2d_main

//! @addtogroup features2d_main
//! @{

/** @brief Wrapping class for feature detection using the AGAST method. :
 */
class CV_EXPORTS_W AgastFeatureDetector : public Feature2D
{
public:
    enum
    {
        AGAST_5_8 = 0, AGAST_7_12d = 1, AGAST_7_12s = 2, OAST_9_16 = 3,
        THRESHOLD = 10000, NONMAX_SUPPRESSION = 10001,
    };

    CV_WRAP static Ptr<AgastFeatureDetector> create( int threshold=10,
                                                     bool nonmaxSuppression=true,
                                                     int type=AgastFeatureDetector::OAST_9_16 );

    CV_WRAP virtual void setThreshold(int threshold) = 0;
    CV_WRAP virtual int getThreshold() const = 0;

    CV_WRAP virtual void setNonmaxSuppression(bool f) = 0;
    CV_WRAP virtual bool getNonmaxSuppression() const = 0;

    CV_WRAP virtual void setType(int type) = 0;
    CV_WRAP virtual int getType() const = 0;
};

/** @brief Wrapping class for feature detection using the goodFeaturesToTrack function. :
 */
class CV_EXPORTS_W GFTTDetector : public Feature2D
{
public:
    CV_WRAP static Ptr<GFTTDetector> create( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
                                             int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
    CV_WRAP virtual void setMaxFeatures(int maxFeatures) = 0;
    CV_WRAP virtual int getMaxFeatures() const = 0;

    CV_WRAP virtual void setQualityLevel(double qlevel) = 0;
    CV_WRAP virtual double getQualityLevel() const = 0;

    CV_WRAP virtual void setMinDistance(double minDistance) = 0;
    CV_WRAP virtual double getMinDistance() const = 0;

    CV_WRAP virtual void setBlockSize(int blockSize) = 0;
    CV_WRAP virtual int getBlockSize() const = 0;

    CV_WRAP virtual void setHarrisDetector(bool val) = 0;
    CV_WRAP virtual bool getHarrisDetector() const = 0;

    CV_WRAP virtual void setK(double k) = 0;
    CV_WRAP virtual double getK() const = 0;
};

/** @brief Class for extracting blobs from an image. :

The class implements a simple algorithm for extracting blobs from an image:

1.  Convert the source image to binary images by applying thresholding with several thresholds from
    minThreshold (inclusive) to maxThreshold (exclusive) with distance thresholdStep between
    neighboring thresholds.
2.  Extract connected components from every binary image by findContours and calculate their
    centers.
3.  Group centers from several binary images by their coordinates. Close centers form one group that
    corresponds to one blob, which is controlled by the minDistBetweenBlobs parameter.
4.  From the groups, estimate final centers of blobs and their radiuses and return as locations and
    sizes of keypoints.

This class performs several filtrations of returned blobs. You should set filterBy\* to true/false
to turn on/off corresponding filtration. Available filtrations:

-   **By color**. This filter compares the intensity of a binary image at the center of a blob to
blobColor. If they differ, the blob is filtered out. Use blobColor = 0 to extract dark blobs
and blobColor = 255 to extract light blobs.
-   **By area**. Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
-   **By circularity**. Extracted blobs have circularity
(\f$\frac{4*\pi*Area}{perimeter * perimeter}\f$) between minCircularity (inclusive) and
maxCircularity (exclusive).
-   **By ratio of the minimum inertia to maximum inertia**. Extracted blobs have this ratio
between minInertiaRatio (inclusive) and maxInertiaRatio (exclusive).
-   **By convexity**. Extracted blobs have convexity (area / area of blob convex hull) between
minConvexity (inclusive) and maxConvexity (exclusive).

Default values of parameters are tuned to extract dark circular blobs.
 */
class CV_EXPORTS_W SimpleBlobDetector : public Feature2D
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

  CV_WRAP static Ptr<SimpleBlobDetector>
    create(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());
};

//! @} features2d_main

//! @addtogroup features2d_main
//! @{

/** @brief Class implementing the KAZE keypoint detector and descriptor extractor, described in @cite ABD12 .

@note AKAZE descriptor can only be used with KAZE or AKAZE keypoints .. [ABD12] KAZE Features. Pablo
F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. In European Conference on Computer Vision
(ECCV), Fiorenze, Italy, October 2012.
*/
class CV_EXPORTS_W KAZE : public Feature2D
{
public:
    enum
    {
        DIFF_PM_G1 = 0,
        DIFF_PM_G2 = 1,
        DIFF_WEICKERT = 2,
        DIFF_CHARBONNIER = 3
    };

    /** @brief The KAZE constructor

    @param extended Set to enable extraction of extended (128-byte) descriptor.
    @param upright Set to enable use of upright descriptors (non rotation-invariant).
    @param threshold Detector response threshold to accept point
    @param nOctaves Maximum octave evolution of the image
    @param nOctaveLayers Default number of sublevels per scale level
    @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or
    DIFF_CHARBONNIER
     */
    CV_WRAP static Ptr<KAZE> create(bool extended=false, bool upright=false,
                                    float threshold = 0.001f,
                                    int nOctaves = 4, int nOctaveLayers = 4,
                                    int diffusivity = KAZE::DIFF_PM_G2);

    CV_WRAP virtual void setExtended(bool extended) = 0;
    CV_WRAP virtual bool getExtended() const = 0;

    CV_WRAP virtual void setUpright(bool upright) = 0;
    CV_WRAP virtual bool getUpright() const = 0;

    CV_WRAP virtual void setThreshold(double threshold) = 0;
    CV_WRAP virtual double getThreshold() const = 0;

    CV_WRAP virtual void setNOctaves(int octaves) = 0;
    CV_WRAP virtual int getNOctaves() const = 0;

    CV_WRAP virtual void setNOctaveLayers(int octaveLayers) = 0;
    CV_WRAP virtual int getNOctaveLayers() const = 0;

    CV_WRAP virtual void setDiffusivity(int diff) = 0;
    CV_WRAP virtual int getDiffusivity() const = 0;
};

/** @brief Class implementing the AKAZE keypoint detector and descriptor extractor, described in @cite ANB13 . :

@note AKAZE descriptors can only be used with KAZE or AKAZE keypoints. Try to avoid using *extract*
and *detect* instead of *operator()* due to performance reasons. .. [ANB13] Fast Explicit Diffusion
for Accelerated Features in Nonlinear Scale Spaces. Pablo F. Alcantarilla, Jesús Nuevo and Adrien
Bartoli. In British Machine Vision Conference (BMVC), Bristol, UK, September 2013.
 */
class CV_EXPORTS_W AKAZE : public Feature2D
{
public:
    // AKAZE descriptor type
    enum
    {
        DESCRIPTOR_KAZE_UPRIGHT = 2, ///< Upright descriptors, not invariant to rotation
        DESCRIPTOR_KAZE = 3,
        DESCRIPTOR_MLDB_UPRIGHT = 4, ///< Upright descriptors, not invariant to rotation
        DESCRIPTOR_MLDB = 5
    };

    /** @brief The AKAZE constructor

    @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
    DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
    @param descriptor_size Size of the descriptor in bits. 0 -\> Full size
    @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
    @param threshold Detector response threshold to accept point
    @param nOctaves Maximum octave evolution of the image
    @param nOctaveLayers Default number of sublevels per scale level
    @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or
    DIFF_CHARBONNIER
     */
    CV_WRAP static Ptr<AKAZE> create(int descriptor_type=AKAZE::DESCRIPTOR_MLDB,
                                     int descriptor_size = 0, int descriptor_channels = 3,
                                     float threshold = 0.001f, int nOctaves = 4,
                                     int nOctaveLayers = 4, int diffusivity = KAZE::DIFF_PM_G2);

    CV_WRAP virtual void setDescriptorType(int dtype) = 0;
    CV_WRAP virtual int getDescriptorType() const = 0;

    CV_WRAP virtual void setDescriptorSize(int dsize) = 0;
    CV_WRAP virtual int getDescriptorSize() const = 0;

    CV_WRAP virtual void setDescriptorChannels(int dch) = 0;
    CV_WRAP virtual int getDescriptorChannels() const = 0;

    CV_WRAP virtual void setThreshold(double threshold) = 0;
    CV_WRAP virtual double getThreshold() const = 0;

    CV_WRAP virtual void setNOctaves(int octaves) = 0;
    CV_WRAP virtual int getNOctaves() const = 0;

    CV_WRAP virtual void setNOctaveLayers(int octaveLayers) = 0;
    CV_WRAP virtual int getNOctaveLayers() const = 0;

    CV_WRAP virtual void setDiffusivity(int diff) = 0;
    CV_WRAP virtual int getDiffusivity() const = 0;
};

//! @} features2d_main

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

/****************************************************************************************\
*                                  DescriptorMatcher                                     *
\****************************************************************************************/

//! @addtogroup features2d_match
//! @{

/** @brief Abstract base class for matching keypoint descriptors.

It has two groups of match methods: for matching descriptors of an image with another image or with
an image set.
 */
class CV_EXPORTS_W DescriptorMatcher : public Algorithm
{
public:
    virtual ~DescriptorMatcher();

    /** @brief Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptor
    collection.

    If the collection is not empty, the new descriptors are added to existing train descriptors.

    @param descriptors Descriptors to add. Each descriptors[i] is a set of descriptors from the same
    train image.
     */
    CV_WRAP virtual void add( InputArrayOfArrays descriptors );

    /** @brief Returns a constant link to the train descriptor collection trainDescCollection .
     */
    CV_WRAP const std::vector<Mat>& getTrainDescriptors() const;

    /** @brief Clears the train descriptor collections.
     */
    CV_WRAP virtual void clear();

    /** @brief Returns true if there are no train descriptors in the both collections.
     */
    CV_WRAP virtual bool empty() const;

    /** @brief Returns true if the descriptor matcher supports masking permissible matches.
     */
    CV_WRAP virtual bool isMaskSupported() const = 0;

    /** @brief Trains a descriptor matcher

    Trains a descriptor matcher (for example, the flann index). In all methods to match, the method
    train() is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher)
    have an empty implementation of this method. Other matchers really train their inner structures (for
    example, FlannBasedMatcher trains flann::Index ).
     */
    CV_WRAP virtual void train();

    /** @brief Finds the best match for each descriptor from a query set.

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
    descriptor. So, matches size may be smaller than the query descriptors count.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.

    In the first variant of this method, the train descriptors are passed as an input argument. In the
    second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is
    used. Optional mask (or masks) can be passed to specify which query and training descriptors can be
    matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if
    mask.at\<uchar\>(i,j) is non-zero.
     */
    CV_WRAP void match( InputArray queryDescriptors, InputArray trainDescriptors,
                CV_OUT std::vector<DMatch>& matches, InputArray mask=noArray() ) const;

    /** @brief Finds the k best matches for each descriptor from a query set.

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.

    These extended variants of DescriptorMatcher::match methods find several best matches for each query
    descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match
    for the details about query and train descriptors.
     */
    CV_WRAP void knnMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                   CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                   InputArray mask=noArray(), bool compactResult=false ) const;

    /** @brief For each query descriptor, finds the training descriptors not farther than the specified distance.

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Found matches.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.

    For each query descriptor, the methods find such training descriptors that the distance between the
    query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
    returned in the distance increasing order.
     */
    void radiusMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                      std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      InputArray mask=noArray(), bool compactResult=false ) const;

    /** @overload
    @param queryDescriptors Query set of descriptors.
    @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
    descriptor. So, matches size may be smaller than the query descriptors count.
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    */
    CV_WRAP void match( InputArray queryDescriptors, CV_OUT std::vector<DMatch>& matches,
                        InputArrayOfArrays masks=noArray() );
    /** @overload
    @param queryDescriptors Query set of descriptors.
    @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    */
    CV_WRAP void knnMatch( InputArray queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                           InputArrayOfArrays masks=noArray(), bool compactResult=false );
    /** @overload
    @param queryDescriptors Query set of descriptors.
    @param matches Found matches.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    */
    void radiusMatch( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      InputArrayOfArrays masks=noArray(), bool compactResult=false );

    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    /** @brief Clones the matcher.

    @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object,
    that is, copies both parameters and train data. If emptyTrainData is true, the method creates an
    object copy with the current parameters but with empty train data.
     */
    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

    /** @brief Creates a descriptor matcher of a given type with the default parameters (using default
    constructor).

    @param descriptorMatcherType Descriptor matcher type. Now the following matcher types are
    supported:
    -   `BruteForce` (it uses L2 )
    -   `BruteForce-L1`
    -   `BruteForce-Hamming`
    -   `BruteForce-Hamming(2)`
    -   `FlannBased`
     */
    CV_WRAP static Ptr<DescriptorMatcher> create( const String& descriptorMatcherType );
protected:
    /**
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

    //! In fact the matching is implemented only by the following two methods. These methods suppose
    //! that the class object has been trained already. Public match methods call these methods
    //! after calling train().
    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) = 0;
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) = 0;

    static bool isPossibleMatch( InputArray mask, int queryIdx, int trainIdx );
    static bool isMaskedOut( InputArrayOfArrays masks, int queryIdx );

    static Mat clone_op( Mat m ) { return m.clone(); }
    void checkMasks( InputArrayOfArrays masks, int queryDescriptorsCount ) const;

    //! Collection of descriptors from train images.
    std::vector<Mat> trainDescCollection;
    std::vector<UMat> utrainDescCollection;
};

/** @brief Brute-force descriptor matcher.

For each descriptor in the first set, this matcher finds the closest descriptor in the second set
by trying each one. This descriptor matcher supports masking permissible matches of descriptor
sets.
 */
class CV_EXPORTS_W BFMatcher : public DescriptorMatcher
{
public:
    /** @brief Brute-force matcher constructor.

    @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
    preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
    BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
    description).
    @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k
    nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
    k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
    matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
    pairs. Such technique usually produces best results with minimal number of outliers when there are
    enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
     */
    CV_WRAP BFMatcher( int normType=NORM_L2, bool crossCheck=false );
    virtual ~BFMatcher() {}

    virtual bool isMaskSupported() const { return true; }

    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;
protected:
    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );

    int normType;
    bool crossCheck;
};


/** @brief Flann-based descriptor matcher.

This matcher trains flann::Index_ on a train descriptor collection and calls its nearest search
methods to find the best matches. So, this matcher may be faster when matching a large train
collection than the brute force matcher. FlannBasedMatcher does not support masking permissible
matches of descriptor sets because flann::Index does not support this. :
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

//! @} features2d_match

/****************************************************************************************\
*                                   Drawing functions                                    *
\****************************************************************************************/

//! @addtogroup features2d_draw
//! @{

struct CV_EXPORTS DrawMatchesFlags
{
    enum{ DEFAULT = 0, //!< Output image matrix will be created (Mat::create),
                       //!< i.e. existing memory of output image may be reused.
                       //!< Two source image, matches and single keypoints will be drawn.
                       //!< For each keypoint only the center point will be drawn (without
                       //!< the circle around keypoint with keypoint size and orientation).
          DRAW_OVER_OUTIMG = 1, //!< Output image matrix will not be created (Mat::create).
                                //!< Matches will be drawn on existing content of output image.
          NOT_DRAW_SINGLE_POINTS = 2, //!< Single keypoints will not be drawn.
          DRAW_RICH_KEYPOINTS = 4 //!< For each keypoint the circle around keypoint with keypoint size and
                                  //!< orientation will be drawn.
        };
};

/** @brief Draws keypoints.

@param image Source image.
@param keypoints Keypoints from the source image.
@param outImage Output image. Its content depends on the flags value defining what is drawn in the
output image. See possible flags bit values below.
@param color Color of keypoints.
@param flags Flags setting drawing features. Possible flags bit values are defined by
DrawMatchesFlags. See details above in drawMatches .

@note
For Python API, flags are modified as cv2.DRAW_MATCHES_FLAGS_DEFAULT,
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
 */
CV_EXPORTS_W void drawKeypoints( InputArray image, const std::vector<KeyPoint>& keypoints, InputOutputArray outImage,
                               const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT );

/** @brief Draws the found matches of keypoints from two images.

@param img1 First source image.
@param keypoints1 Keypoints from the first source image.
@param img2 Second source image.
@param keypoints2 Keypoints from the second source image.
@param matches1to2 Matches from the first image to the second one, which means that keypoints1[i]
has a corresponding point in keypoints2[matches[i]] .
@param outImg Output image. Its content depends on the flags value defining what is drawn in the
output image. See possible flags bit values below.
@param matchColor Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1)
, the color is generated randomly.
@param singlePointColor Color of single keypoints (circles), which means that keypoints do not
have the matches. If singlePointColor==Scalar::all(-1) , the color is generated randomly.
@param matchesMask Mask determining which matches are drawn. If the mask is empty, all matches are
drawn.
@param flags Flags setting drawing features. Possible flags bit values are defined by
DrawMatchesFlags.

This function draws matches of keypoints from two images in the output image. Match is a line
connecting two keypoints (circles). See cv::DrawMatchesFlags.
 */
CV_EXPORTS_W void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<char>& matchesMask=std::vector<char>(), int flags=DrawMatchesFlags::DEFAULT );

/** @overload */
CV_EXPORTS_AS(drawMatchesKnn) void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), int flags=DrawMatchesFlags::DEFAULT );

//! @} features2d_draw

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

//! @addtogroup features2d_category
//! @{

/** @brief Abstract base class for training the *bag of visual words* vocabulary from a set of descriptors.

For details, see, for example, *Visual Categorization with Bags of Keypoints* by Gabriella Csurka,
Christopher R. Dance, Lixin Fan, Jutta Willamowski, Cedric Bray, 2004. :
 */
class CV_EXPORTS_W BOWTrainer
{
public:
    BOWTrainer();
    virtual ~BOWTrainer();

    /** @brief Adds descriptors to a training set.

    @param descriptors Descriptors to add to a training set. Each row of the descriptors matrix is a
    descriptor.

    The training set is clustered using clustermethod to construct the vocabulary.
     */
    CV_WRAP void add( const Mat& descriptors );

    /** @brief Returns a training set of descriptors.
    */
    CV_WRAP const std::vector<Mat>& getDescriptors() const;

    /** @brief Returns the count of all descriptors stored in the training set.
    */
    CV_WRAP int descriptorsCount() const;

    CV_WRAP virtual void clear();

    /** @overload */
    CV_WRAP virtual Mat cluster() const = 0;

    /** @brief Clusters train descriptors.

    @param descriptors Descriptors to cluster. Each row of the descriptors matrix is a descriptor.
    Descriptors are not added to the inner train descriptor set.

    The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first
    variant of the method, train descriptors stored in the object are clustered. In the second variant,
    input descriptors are clustered.
     */
    CV_WRAP virtual Mat cluster( const Mat& descriptors ) const = 0;

protected:
    std::vector<Mat> descriptors;
    int size;
};

/** @brief kmeans -based class to train visual vocabulary using the *bag of visual words* approach. :
 */
class CV_EXPORTS_W BOWKMeansTrainer : public BOWTrainer
{
public:
    /** @brief The constructor.

    @see cv::kmeans
    */
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

/** @brief Class to compute an image descriptor using the *bag of visual words*.

Such a computation consists of the following steps:

1.  Compute descriptors for a given image and its keypoints set.
2.  Find the nearest visual words from the vocabulary for each keypoint descriptor.
3.  Compute the bag-of-words image descriptor as is a normalized histogram of vocabulary words
encountered in the image. The i-th bin of the histogram is a frequency of i-th word of the
vocabulary in the given image.
 */
class CV_EXPORTS_W BOWImgDescriptorExtractor
{
public:
    /** @brief The constructor.

    @param dextractor Descriptor extractor that is used to compute descriptors for an input image and
    its keypoints.
    @param dmatcher Descriptor matcher that is used to find the nearest word of the trained vocabulary
    for each keypoint descriptor of the image.
     */
    CV_WRAP BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor,
                               const Ptr<DescriptorMatcher>& dmatcher );
    /** @overload */
    BOWImgDescriptorExtractor( const Ptr<DescriptorMatcher>& dmatcher );
    virtual ~BOWImgDescriptorExtractor();

    /** @brief Sets a visual vocabulary.

    @param vocabulary Vocabulary (can be trained using the inheritor of BOWTrainer ). Each row of the
    vocabulary is a visual word (cluster center).
     */
    CV_WRAP void setVocabulary( const Mat& vocabulary );

    /** @brief Returns the set vocabulary.
    */
    CV_WRAP const Mat& getVocabulary() const;

    /** @brief Computes an image descriptor using the set visual vocabulary.

    @param image Image, for which the descriptor is computed.
    @param keypoints Keypoints detected in the input image.
    @param imgDescriptor Computed output image descriptor.
    @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that
    pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary)
    returned if it is non-zero.
    @param descriptors Descriptors of the image keypoints that are returned if they are non-zero.
     */
    void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0, Mat* descriptors=0 );
    /** @overload
    @param keypointDescriptors Computed descriptors to match with vocabulary.
    @param imgDescriptor Computed output image descriptor.
    @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that
    pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary)
    returned if it is non-zero.
    */
    void compute( InputArray keypointDescriptors, OutputArray imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0 );
    // compute() is not constant because DescriptorMatcher::match is not constant

    CV_WRAP_AS(compute) void compute2( const Mat& image, std::vector<KeyPoint>& keypoints, CV_OUT Mat& imgDescriptor )
    { compute(image,keypoints,imgDescriptor); }

    /** @brief Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.
    */
    CV_WRAP int descriptorSize() const;

    /** @brief Returns an image descriptor type.
     */
    CV_WRAP int descriptorType() const;

protected:
    Mat vocabulary;
    Ptr<DescriptorExtractor> dextractor;
    Ptr<DescriptorMatcher> dmatcher;
};

//! @} features2d_category

//! @} features2d

} /* namespace cv */

#endif
