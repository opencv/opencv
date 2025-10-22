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

#ifndef OPENCV_FEATURES_HPP
#define OPENCV_FEATURES_HPP

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_FLANN
#include "opencv2/flann/miniflann.hpp"
#endif

/**
  @defgroup features Features Framework
  @{
    @defgroup features_main Feature Detection and Description
    @defgroup features_match Descriptor Matchers

    Matchers of keypoint descriptors in OpenCV have wrappers with a common interface that enables
    you to easily switch between different algorithms solving the same problem. This section is
    devoted to matching descriptors that are represented as vectors in a multidimensional space.
    All objects that implement vector descriptor matchers inherit the DescriptorMatcher interface.

    @defgroup features_draw Drawing Function of Keypoints and Matches

    @defgroup feature_hal Hardware Acceleration Layer
    @{
        @defgroup features_hal_interface Interface
    @}

    @defgroup features_annoy Approximate Nearest Neighbors Search in Multi-Dimensional Spaces

    This section documents OpenCV's interface to the Annoy. Annoy (Approximate Nearest Neighbors Oh Yeah)
    is a library to search for points in space that are close to a given query point. It also creates
    large read-only file-based data structures that are mmapped into memory so that many processes may
    share the same data.
  @}
 */

namespace cv
{

//! @addtogroup features_main
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
     * Remove objects from some image and a vector of points by mask for pixels of this image
     */
    static void runByPixelsMask2VectorPoint(std::vector<KeyPoint> &keypoints, std::vector<std::vector<Point> > &removeFrom, const Mat &mask);
    /*
     * Remove duplicated keypoints.
     */
    static void removeDuplicated( std::vector<KeyPoint>& keypoints );
    /*
     * Remove duplicated keypoints and sort the remaining keypoints
     */
    static void removeDuplicatedSorted( std::vector<KeyPoint>& keypoints );

    /*
     * Retain the specified number of the best keypoints (according to the response)
     */
    static void retainBest( std::vector<KeyPoint>& keypoints, int npoints );
};


/************************************ Base Classes ************************************/

/** @brief Abstract base class for 2D image feature detectors and descriptor extractors
*/
class CV_EXPORTS_W Feature2D : public Algorithm
{
public:
    virtual ~Feature2D();

    /** @brief Detects keypoints in an image (first variant) or image set (second variant).

    @param image Image.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param mask Mask specifying where to look for keypoints (optional). It must be CV_8U or CV_Bool
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
    CV_WRAP virtual void detect( InputArrayOfArrays images,
                         CV_OUT std::vector<std::vector<KeyPoint> >& keypoints,
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
    CV_WRAP virtual void compute( InputArrayOfArrays images,
                          CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint> >& keypoints,
                          OutputArrayOfArrays descriptors );

    /** Detects keypoints and computes the descriptors */
    CV_WRAP virtual void detectAndCompute( InputArray image, InputArray mask,
                                           CV_OUT std::vector<KeyPoint>& keypoints,
                                           OutputArray descriptors,
                                           bool useProvidedKeypoints=false );

    CV_WRAP virtual int descriptorSize() const;
    CV_WRAP virtual int descriptorType() const;
    CV_WRAP virtual int defaultNorm() const;

    CV_WRAP void write( const String& fileName ) const;

    CV_WRAP void read( const String& fileName );

    virtual void write( FileStorage&) const CV_OVERRIDE;

    // see corresponding cv::Algorithm method
    CV_WRAP virtual void read( const FileNode&) CV_OVERRIDE;

    //! Return true if detector object is empty
    CV_WRAP virtual bool empty() const CV_OVERRIDE;
    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;

    // see corresponding cv::Algorithm method
    CV_WRAP inline void write(FileStorage& fs, const String& name) const { Algorithm::write(fs, name); }
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


/** @brief Class for implementing the wrapper which makes detectors and extractors to be affine invariant,
described as ASIFT in @cite YM11 .
*/
class CV_EXPORTS_W AffineFeature : public Feature2D
{
public:
    /**
    @param backend The detector/extractor you want to use as backend.
    @param maxTilt The highest power index of tilt factor. 5 is used in the paper as tilt sampling range n.
    @param minTilt The lowest power index of tilt factor. 0 is used in the paper.
    @param tiltStep Tilt sampling step \f$\delta_t\f$ in Algorithm 1 in the paper.
    @param rotateStepBase Rotation sampling step factor b in Algorithm 1 in the paper.
    */
    CV_WRAP static Ptr<AffineFeature> create(const Ptr<Feature2D>& backend,
        int maxTilt = 5, int minTilt = 0, float tiltStep = 1.4142135623730951f, float rotateStepBase = 72);

    CV_WRAP virtual void setViewParams(const std::vector<float>& tilts, const std::vector<float>& rolls) = 0;
    CV_WRAP virtual void getViewParams(std::vector<float>& tilts, std::vector<float>& rolls) const = 0;
    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;
};

typedef AffineFeature AffineFeatureDetector;
typedef AffineFeature AffineDescriptorExtractor;


/** @brief Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform
(SIFT) algorithm by D. Lowe @cite Lowe04 .
*/
class CV_EXPORTS_W SIFT : public Feature2D
{
public:
    /**
    @param nfeatures The number of best features to retain. The features are ranked by their scores
    (measured in SIFT algorithm as the local contrast)

    @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
    number of octaves is computed automatically from the image resolution.

    @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
    (low-contrast) regions. The larger the threshold, the less features are produced by the detector.

    @note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When
    nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set
    this argument to 0.09.

    @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
    is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
    filtered out (more features are retained).

    @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image
    is captured with a weak camera with soft lenses, you might want to reduce the number.

    @param enable_precise_upscale Whether to enable precise upscaling in the scale pyramid, which maps
    index \f$\texttt{x}\f$ to \f$\texttt{2x}\f$. This prevents localization bias. The option
    is disabled by default.
    */
    CV_WRAP static Ptr<SIFT> create(int nfeatures = 0, int nOctaveLayers = 3,
        double contrastThreshold = 0.04, double edgeThreshold = 10,
        double sigma = 1.6, bool enable_precise_upscale = false);

    /** @brief Create SIFT with specified descriptorType.
    @param nfeatures The number of best features to retain. The features are ranked by their scores
    (measured in SIFT algorithm as the local contrast)

    @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
    number of octaves is computed automatically from the image resolution.

    @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
    (low-contrast) regions. The larger the threshold, the less features are produced by the detector.

    @note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When
    nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set
    this argument to 0.09.

    @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
    is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
    filtered out (more features are retained).

    @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image
    is captured with a weak camera with soft lenses, you might want to reduce the number.

    @param descriptorType The type of descriptors. Only CV_32F and CV_8U are supported.

    @param enable_precise_upscale Whether to enable precise upscaling in the scale pyramid, which maps
    index \f$\texttt{x}\f$ to \f$\texttt{2x}\f$. This prevents localization bias. The option
    is disabled by default.
    */
    CV_WRAP static Ptr<SIFT> create(int nfeatures, int nOctaveLayers,
        double contrastThreshold, double edgeThreshold,
        double sigma, int descriptorType, bool enable_precise_upscale = false);

    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;

    CV_WRAP virtual void setNFeatures(int maxFeatures) = 0;
    CV_WRAP virtual int getNFeatures() const = 0;

    CV_WRAP virtual void setNOctaveLayers(int nOctaveLayers) = 0;
    CV_WRAP virtual int getNOctaveLayers() const = 0;

    CV_WRAP virtual void setContrastThreshold(double contrastThreshold) = 0;
    CV_WRAP virtual double getContrastThreshold() const = 0;

    CV_WRAP virtual void setEdgeThreshold(double edgeThreshold) = 0;
    CV_WRAP virtual double getEdgeThreshold() const = 0;

    CV_WRAP virtual void setSigma(double sigma) = 0;
    CV_WRAP virtual double getSigma() const = 0;
};

typedef SIFT SiftFeatureDetector;
typedef SIFT SiftDescriptorExtractor;


/** @brief Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor

described in @cite RRKB11 . The algorithm uses FAST in pyramids to detect stable keypoints, selects
the strongest features using FAST or Harris response, finds their orientation using first-order
moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or
k-tuples) are rotated according to the measured orientation).
 */
class CV_EXPORTS_W ORB : public Feature2D
{
public:
    enum ScoreType { HARRIS_SCORE=0, FAST_SCORE=1 };
    static const int kBytes = 32;

    /** @brief The ORB constructor

    @param nfeatures The maximum number of features to retain.
    @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
    pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
    will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
    will mean that to cover certain scale range you will need more pyramid levels and so the speed
    will suffer.
    @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
    input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
    @param edgeThreshold This is size of the border where the features are not detected. It should
    roughly match the patchSize parameter.
    @param firstLevel The level of pyramid to put source image to. Previous layers are filled
    with upscaled source image.
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
    @param fastThreshold the fast threshold
     */
    CV_WRAP static Ptr<ORB> create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31,
        int firstLevel=0, int WTA_K=2, ORB::ScoreType scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20);

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

    CV_WRAP virtual void setScoreType(ORB::ScoreType scoreType) = 0;
    CV_WRAP virtual ORB::ScoreType getScoreType() const = 0;

    CV_WRAP virtual void setPatchSize(int patchSize) = 0;
    CV_WRAP virtual int getPatchSize() const = 0;

    CV_WRAP virtual void setFastThreshold(int fastThreshold) = 0;
    CV_WRAP virtual int getFastThreshold() const = 0;
    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;
};

/** @example samples/python/snippets/mser.py
An example using Maximally stable extremal region(MSER) extractor in python
*/

/** @brief Maximally stable extremal region extractor

The class encapsulates all the parameters of the %MSER extraction algorithm (see [wiki
article](http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions)).

- there are two different implementation of %MSER: one for grey image, one for color image

- the grey image algorithm is taken from: @cite nister2008linear ;  the paper claims to be faster
than union-find method; it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.

- the color image algorithm is taken from: @cite forssen2007maximally ; it should be much slower
than grey image method ( 3~4 times )

- (Python) A complete example showing the use of the %MSER detector can be found at samples/python/mser.py
*/
class CV_EXPORTS_W MSER : public Feature2D
{
public:
    /** @brief Full constructor for %MSER detector

    @param delta it compares \f$(size_{i}-size_{i-delta})/size_{i-delta}\f$
    @param min_area prune the area which smaller than minArea
    @param max_area prune the area which bigger than maxArea
    @param max_variation prune the area have similar size to its children
    @param min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
    @param max_evolution  for color image, the evolution steps
    @param area_threshold for color image, the area threshold to cause re-initialize
    @param min_margin for color image, ignore too small margin
    @param edge_blur_size for color image, the aperture size for edge blur
     */
    CV_WRAP static Ptr<MSER> create( int delta=5, int min_area=60, int max_area=14400,
          double max_variation=0.25, double min_diversity=.2,
          int max_evolution=200, double area_threshold=1.01,
          double min_margin=0.003, int edge_blur_size=5 );

    /** @brief Detect %MSER regions

    @param image input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
    @param msers resulting list of point sets
    @param bboxes resulting bounding boxes
    */
    CV_WRAP virtual void detectRegions( InputArray image,
                                        CV_OUT std::vector<std::vector<Point> >& msers,
                                        CV_OUT std::vector<Rect>& bboxes ) = 0;

    CV_WRAP virtual void setDelta(int delta) = 0;
    CV_WRAP virtual int getDelta() const = 0;

    CV_WRAP virtual void setMinArea(int minArea) = 0;
    CV_WRAP virtual int getMinArea() const = 0;

    CV_WRAP virtual void setMaxArea(int maxArea) = 0;
    CV_WRAP virtual int getMaxArea() const = 0;

    CV_WRAP virtual void setMaxVariation(double maxVariation) = 0;
    CV_WRAP virtual double getMaxVariation() const = 0;

    CV_WRAP virtual void setMinDiversity(double minDiversity) = 0;
    CV_WRAP virtual double getMinDiversity() const = 0;

    CV_WRAP virtual void setMaxEvolution(int maxEvolution) = 0;
    CV_WRAP virtual int getMaxEvolution() const = 0;

    CV_WRAP virtual void setAreaThreshold(double areaThreshold) = 0;
    CV_WRAP virtual double getAreaThreshold() const = 0;

    CV_WRAP virtual void setMinMargin(double min_margin) = 0;
    CV_WRAP virtual double getMinMargin() const = 0;

    CV_WRAP virtual void setEdgeBlurSize(int edge_blur_size) = 0;
    CV_WRAP virtual int getEdgeBlurSize() const = 0;

    CV_WRAP virtual void setPass2Only(bool f) = 0;
    CV_WRAP virtual bool getPass2Only() const = 0;

    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;
};


/** @brief Wrapping class for feature detection using the FAST method.

Check @ref tutorial_py_fast "the corresponding tutorial" for more details.
 */
class CV_EXPORTS_W FastFeatureDetector : public Feature2D
{
public:
    enum DetectorType
    {
        TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2
    };
    enum
    {
        THRESHOLD = 10000, NONMAX_SUPPRESSION=10001, FAST_N=10002
    };


    CV_WRAP static Ptr<FastFeatureDetector> create( int threshold=10,
                                                    bool nonmaxSuppression=true,
                                                    FastFeatureDetector::DetectorType type=FastFeatureDetector::TYPE_9_16 );

    CV_WRAP virtual void setThreshold(int threshold) = 0;
    CV_WRAP virtual int getThreshold() const = 0;

    CV_WRAP virtual void setNonmaxSuppression(bool f) = 0;
    CV_WRAP virtual bool getNonmaxSuppression() const = 0;

    CV_WRAP virtual void setType(FastFeatureDetector::DetectorType type) = 0;
    CV_WRAP virtual FastFeatureDetector::DetectorType getType() const = 0;
    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;
};

/** @brief Detects corners using the FAST algorithm

@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected keypoints (corners).
@param type one of the three neighborhoods as defined in the paper:
FastFeatureDetector::TYPE_9_16, FastFeatureDetector::TYPE_7_12,
FastFeatureDetector::TYPE_5_8

Detects corners using the FAST algorithm by @cite Rosten06 .

Check @ref tutorial_py_fast "the corresponding tutorial" for more details.
 */
CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression=true, FastFeatureDetector::DetectorType type=FastFeatureDetector::TYPE_9_16 );


/** @brief Wrapping class for feature detection using the goodFeaturesToTrack function. :
 */
class CV_EXPORTS_W GFTTDetector : public Feature2D
{
public:
    CV_WRAP static Ptr<GFTTDetector> create( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
                                             int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
    CV_WRAP static Ptr<GFTTDetector> create( int maxCorners, double qualityLevel, double minDistance,
                                             int blockSize, int gradiantSize, bool useHarrisDetector=false, double k=0.04 );
    CV_WRAP virtual void setMaxFeatures(int maxFeatures) = 0;
    CV_WRAP virtual int getMaxFeatures() const = 0;

    CV_WRAP virtual void setQualityLevel(double qlevel) = 0;
    CV_WRAP virtual double getQualityLevel() const = 0;

    CV_WRAP virtual void setMinDistance(double minDistance) = 0;
    CV_WRAP virtual double getMinDistance() const = 0;

    CV_WRAP virtual void setBlockSize(int blockSize) = 0;
    CV_WRAP virtual int getBlockSize() const = 0;

    CV_WRAP virtual void setGradientSize(int gradientSize_) = 0;
    CV_WRAP virtual int getGradientSize() = 0;

    CV_WRAP virtual void setHarrisDetector(bool val) = 0;
    CV_WRAP virtual bool getHarrisDetector() const = 0;

    CV_WRAP virtual void setK(double k) = 0;
    CV_WRAP virtual double getK() const = 0;
    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;
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

      CV_PROP_RW bool collectContours;

      void read( const FileNode& fn );
      void write( FileStorage& fs ) const;
  };

  CV_WRAP static Ptr<SimpleBlobDetector>
    create(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());

  CV_WRAP virtual void setParams(const SimpleBlobDetector::Params& params ) = 0;
  CV_WRAP virtual SimpleBlobDetector::Params getParams() const = 0;

  CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;
  CV_WRAP virtual const std::vector<std::vector<cv::Point> >& getBlobContours() const = 0;
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
    static const NormTypes normType = NORM_L2SQR;
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
struct L2
{
    static const NormTypes normType = NORM_L2;
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
struct L1
{
    static const NormTypes normType = NORM_L1;
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return normL1<ValueType, ResultType>(a, b, size);
    }
};

//! @} features_main

/****************************************************************************************\
*                                  DescriptorMatcher                                     *
\****************************************************************************************/

//! @addtogroup features_match
//! @{

/** @brief Abstract base class for matching keypoint descriptors.

It has two groups of match methods: for matching descriptors of an image with another image or with
an image set.
 */
class CV_EXPORTS_W DescriptorMatcher : public Algorithm
{
public:
   enum MatcherType
    {
        FLANNBASED            = 1,
        BRUTEFORCE            = 2,
        BRUTEFORCE_L1         = 3,
        BRUTEFORCE_HAMMING    = 4,
        BRUTEFORCE_HAMMINGLUT = 5,
        BRUTEFORCE_SL2        = 6
    };

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
    CV_WRAP virtual void clear() CV_OVERRIDE;

    /** @brief Returns true if there are no train descriptors in the both collections.
     */
    CV_WRAP virtual bool empty() const CV_OVERRIDE;

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
    CV_WRAP void radiusMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                      CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
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
    CV_WRAP void radiusMatch( InputArray queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      InputArrayOfArrays masks=noArray(), bool compactResult=false );


    CV_WRAP void write( const String& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        write(fs);
    }

    CV_WRAP void read( const String& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        read(fs.root());
    }
    // Reads matcher object from a file node
    // see corresponding cv::Algorithm method
    CV_WRAP virtual void read( const FileNode& ) CV_OVERRIDE;
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const CV_OVERRIDE;

    /** @brief Clones the matcher.

    @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object,
    that is, copies both parameters and train data. If emptyTrainData is true, the method creates an
    object copy with the current parameters but with empty train data.
     */
    CV_WRAP CV_NODISCARD_STD virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

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

    CV_WRAP static Ptr<DescriptorMatcher> create( const DescriptorMatcher::MatcherType& matcherType );


    // see corresponding cv::Algorithm method
    CV_WRAP inline void write(FileStorage& fs, const String& name) const { Algorithm::write(fs, name); }
#if CV_VERSION_MAJOR < 5
    inline void write(const Ptr<FileStorage>& fs, const String& name) const { CV_Assert(fs); Algorithm::write(*fs, name); }
#endif

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
        Mat getDescriptor( int imgIdx, int localDescIdx ) const;
        Mat getDescriptor( int globalDescIdx ) const;
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

    CV_NODISCARD_STD static Mat clone_op( Mat m ) { return m.clone(); }
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
    /** @brief Brute-force matcher constructor (obsolete). Please use BFMatcher.create()
     *
     *
    */
    CV_WRAP BFMatcher( int normType=NORM_L2, bool crossCheck=false );

    virtual ~BFMatcher() {}

    virtual bool isMaskSupported() const CV_OVERRIDE { return true; }

    /** @brief Brute-force matcher create method.
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
    CV_WRAP static Ptr<BFMatcher> create( int normType=NORM_L2, bool crossCheck=false ) ;

    CV_NODISCARD_STD virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const CV_OVERRIDE;
protected:
    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) CV_OVERRIDE;
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) CV_OVERRIDE;

    int normType;
    bool crossCheck;
};

#if defined(HAVE_OPENCV_FLANN) || defined(CV_DOXYGEN)

/** @brief Flann-based descriptor matcher.

This matcher trains cv::flann::Index on a train descriptor collection and calls its nearest search
methods to find the best matches. So, this matcher may be faster when matching a large train
collection than the brute force matcher. FlannBasedMatcher does not support masking permissible
matches of descriptor sets because flann::Index does not support this. :
 */
class CV_EXPORTS_W FlannBasedMatcher : public DescriptorMatcher
{
public:
    CV_WRAP FlannBasedMatcher( const Ptr<flann::IndexParams>& indexParams=makePtr<flann::KDTreeIndexParams>(),
                       const Ptr<flann::SearchParams>& searchParams=makePtr<flann::SearchParams>() );

    virtual void add( InputArrayOfArrays descriptors ) CV_OVERRIDE;
    virtual void clear() CV_OVERRIDE;

    // Reads matcher object from a file node
    virtual void read( const FileNode& ) CV_OVERRIDE;
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const CV_OVERRIDE;

    virtual void train() CV_OVERRIDE;
    virtual bool isMaskSupported() const CV_OVERRIDE;

    CV_WRAP static Ptr<FlannBasedMatcher> create();

    CV_NODISCARD_STD virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const CV_OVERRIDE;
protected:
    static void convertToDMatches( const DescriptorCollection& descriptors,
                                   const Mat& indices, const Mat& distances,
                                   std::vector<std::vector<DMatch> >& matches );

    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) CV_OVERRIDE;
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) CV_OVERRIDE;

    Ptr<flann::IndexParams> indexParams;
    Ptr<flann::SearchParams> searchParams;
    Ptr<flann::Index> flannIndex;

    DescriptorCollection mergedDescriptors;
    int addedDescCount;
};

#endif

//! @} features_match

/****************************************************************************************\
*                                   Drawing functions                                    *
\****************************************************************************************/

//! @addtogroup features_draw
//! @{

enum struct DrawMatchesFlags
{
  DEFAULT = 0, //!< Output image matrix will be created (Mat::create),
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
CV_ENUM_FLAGS(DrawMatchesFlags)

/** @brief Draws keypoints.

@param image Source image.
@param keypoints Keypoints from the source image.
@param outImage Output image. Its content depends on the flags value defining what is drawn in the
output image. See possible flags bit values below.
@param color Color of keypoints.
@param flags Flags setting drawing features. Possible flags bit values are defined by
DrawMatchesFlags. See details above in drawMatches .

@note
For Python API, flags are modified as cv.DRAW_MATCHES_FLAGS_DEFAULT,
cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
 */
CV_EXPORTS_W void drawKeypoints( InputArray image, const std::vector<KeyPoint>& keypoints, InputOutputArray outImage,
                               const Scalar& color=Scalar::all(-1), DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );

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
                             const std::vector<char>& matchesMask=std::vector<char>(), DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );

/** @overload */
CV_EXPORTS_W void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                             const int matchesThickness, const Scalar& matchColor=Scalar::all(-1),
                             const Scalar& singlePointColor=Scalar::all(-1), const std::vector<char>& matchesMask=std::vector<char>(),
                             DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );

CV_EXPORTS_AS(drawMatchesKnn) void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );

//! @} features_draw

/****************************************************************************************\
*   Functions to evaluate the feature detectors and [generic] descriptor extractors      *
\****************************************************************************************/

//! @addtogroup features_main
//! @{

CV_EXPORTS void evaluateFeatureDetector( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                         std::vector<KeyPoint>* keypoints1, std::vector<KeyPoint>* keypoints2,
                                         float& repeatability, int& correspCount,
                                         const Ptr<FeatureDetector>& fdetector=Ptr<FeatureDetector>() );

CV_EXPORTS void computeRecallPrecisionCurve( const std::vector<std::vector<DMatch> >& matches1to2,
                                             const std::vector<std::vector<uchar> >& correctMatches1to2Mask,
                                             std::vector<Point2f>& recallPrecisionCurve );

CV_EXPORTS float getRecall( const std::vector<Point2f>& recallPrecisionCurve, float l_precision );
CV_EXPORTS int getNearestPoint( const std::vector<Point2f>& recallPrecisionCurve, float l_precision );

//! @}


/****************************************************************************************\
*                             Approximate Nearest Neighbors                              *
\****************************************************************************************/

//! @addtogroup features_annoy
//! @{

class CV_EXPORTS_W ANNIndex
{
public:
    /** @brief Metrics used to calculate the distance between two feature vectors.
     */
    enum Distance
    {
        DIST_EUCLIDEAN,
        DIST_MANHATTAN,
        DIST_ANGULAR,
        DIST_HAMMING,
        DIST_DOTPRODUCT
    };

    virtual ~ANNIndex() = default;

    /** @brief Add feature vectors to index.
     *
     * @param features Matrix containing the feature vectors to index. The size of the matrix is
        num_features x feature_dimension.
     */
    CV_WRAP virtual void addItems(InputArray features) = 0;

    /** @brief Build the index.
     *
     *  @param trees Number of trees in the index. If not provided, the number is determined automatically
     *  in a way that at most 2x as much memory as the features vectors take is used.
     */
    CV_WRAP virtual void build(int trees=-1) = 0;

    /** @brief Performs a K-nearest neighbor search for given query vector(s) using the index.
     *
     *  @param query The query vector(s).
     *  @param indices Matrix that will contain the indices of the K-nearest neighbors found, optional.
     *  @param dists Matrix that will contain the distances to the K-nearest neighbors found, optional.
     *  @param knn Number of nearest neighbors to search for.
     *  @param search_k The maximum number of nodes to inspect, which defaults to trees x knn if not provided.
     */
    CV_WRAP virtual void knnSearch(InputArray query, OutputArray indices, OutputArray dists, int knn, int search_k=-1) = 0;

    /** @brief Save the index to disk and loads it. After saving, no more vectors can be added.
     *
     *  @param filename Filename of the index to be saved.
     *  @param prefault If prefault is set to true, it will pre-read the entire file into memory (using mmap
     *  with MAP_POPULATE). Default is false.
     */
    CV_WRAP virtual void save(const String &filename, bool prefault=false) = 0;

    /** @brief Loads (mmaps) an index from disk.
     *
     *  @param filename Filename of the index to be loaded.
     *  @param prefault If prefault is set to true, it will pre-read the entire file into memory (using mmap
     *  with MAP_POPULATE). Default is false.
     */
    CV_WRAP virtual void load(const String &filename, bool prefault=false) = 0;

    /** @brief Return the number of trees in the index.
     */
    CV_WRAP virtual int getTreeNumber() = 0;

    /** @brief Return the number of feature vectors in the index.
     */
    CV_WRAP virtual int getItemNumber() = 0;

    /** @brief  Prepare to build the index in the specified file instead of RAM (execute before adding
     * items, no need to save after build)
     *
     *  @param filename Filename of the index to be built.
     */
    CV_WRAP virtual bool setOnDiskBuild(const String &filename) = 0;

    /** @brief Initialize the random number generator with the given seed. Only necessary to pass this
     *  before adding the items. Will have no effect after calling build() or load().
     *
     *  @param seed The given seed of the random number generator. Its value should be within the range of uint32_t.
     */
    CV_WRAP virtual void setSeed(int seed) = 0;

    /** @brief Creates an instance of annoy index class with given parameters
     *
     *  @param dim The dimension of the feature vector.
     *  @param distType Metric to calculate the distance between two feature vectors, can be DIST_EUCLIDEAN,
        DIST_MANHATTAN, DIST_ANGULAR, DIST_HAMMING, or DIST_DOTPRODUCT.
     */
    CV_WRAP static Ptr<ANNIndex> create(int dim, ANNIndex::Distance distType=ANNIndex::DIST_EUCLIDEAN);
};

//! @} features_annoy

} /* namespace cv */

#endif
