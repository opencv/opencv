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

#ifndef __OPENCV_CUDAFEATURES2D_HPP__
#define __OPENCV_CUDAFEATURES2D_HPP__

#ifndef __cplusplus
#  error cudafeatures2d.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/cudafilters.hpp"

/**
  @addtogroup cuda
  @{
    @defgroup cudafeatures2d Feature Detection and Description
  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudafeatures2d
//! @{

/** @brief Brute-force descriptor matcher.

For each descriptor in the first set, this matcher finds the closest descriptor in the second set
by trying each one. This descriptor matcher supports masking permissible matches between descriptor
sets.

The class BFMatcher_CUDA has an interface similar to the class DescriptorMatcher. It has two groups
of match methods: for matching descriptors of one image with another image or with an image set.
Also, all functions have an alternative to save results either to the GPU memory or to the CPU
memory.

@sa DescriptorMatcher, BFMatcher
 */
class CV_EXPORTS BFMatcher_CUDA
{
public:
    explicit BFMatcher_CUDA(int norm = cv::NORM_L2);

    //! Add descriptors to train descriptor collection
    void add(const std::vector<GpuMat>& descCollection);

    //! Get train descriptors collection
    const std::vector<GpuMat>& getTrainDescriptors() const;

    //! Clear train descriptors collection
    void clear();

    //! Return true if there are not train descriptors in collection
    bool empty() const;

    //! Return true if the matcher supports mask in match methods
    bool isMaskSupported() const;

    //! Find one best match for each query descriptor
    void matchSingle(const GpuMat& query, const GpuMat& train,
        GpuMat& trainIdx, GpuMat& distance,
        const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

    //! Download trainIdx and distance and convert it to CPU vector with DMatch
    static void matchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector<DMatch>& matches);
    //! Convert trainIdx and distance to vector with DMatch
    static void matchConvert(const Mat& trainIdx, const Mat& distance, std::vector<DMatch>& matches);

    //! Find one best match for each query descriptor
    void match(const GpuMat& query, const GpuMat& train, std::vector<DMatch>& matches, const GpuMat& mask = GpuMat());

    //! Make gpu collection of trains and masks in suitable format for matchCollection function
    void makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection, const std::vector<GpuMat>& masks = std::vector<GpuMat>());

    //! Find one best match from train collection for each query descriptor
    void matchCollection(const GpuMat& query, const GpuMat& trainCollection,
        GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
        const GpuMat& masks = GpuMat(), Stream& stream = Stream::Null());

    //! Download trainIdx, imgIdx and distance and convert it to vector with DMatch
    static void matchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, std::vector<DMatch>& matches);
    //! Convert trainIdx, imgIdx and distance to vector with DMatch
    static void matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>& matches);

    //! Find one best match from train collection for each query descriptor.
    void match(const GpuMat& query, std::vector<DMatch>& matches, const std::vector<GpuMat>& masks = std::vector<GpuMat>());

    //! Find k best matches for each query descriptor (in increasing order of distances)
    void knnMatchSingle(const GpuMat& query, const GpuMat& train,
        GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k,
        const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

    //! Download trainIdx and distance and convert it to vector with DMatch
    //! compactResult is used when mask is not empty. If compactResult is false matches
    //! vector will have the same size as queryDescriptors rows. If compactResult is true
    //! matches vector will not contain matches for fully masked out query descriptors.
    static void knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    //! Convert trainIdx and distance to vector with DMatch
    static void knnMatchConvert(const Mat& trainIdx, const Mat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    //! Find k best matches for each query descriptor (in increasing order of distances).
    //! compactResult is used when mask is not empty. If compactResult is false matches
    //! vector will have the same size as queryDescriptors rows. If compactResult is true
    //! matches vector will not contain matches for fully masked out query descriptors.
    void knnMatch(const GpuMat& query, const GpuMat& train,
        std::vector< std::vector<DMatch> >& matches, int k, const GpuMat& mask = GpuMat(),
        bool compactResult = false);

    //! Find k best matches from train collection for each query descriptor (in increasing order of distances)
    void knnMatch2Collection(const GpuMat& query, const GpuMat& trainCollection,
        GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
        const GpuMat& maskCollection = GpuMat(), Stream& stream = Stream::Null());

    //! Download trainIdx and distance and convert it to vector with DMatch
    //! compactResult is used when mask is not empty. If compactResult is false matches
    //! vector will have the same size as queryDescriptors rows. If compactResult is true
    //! matches vector will not contain matches for fully masked out query descriptors.
    //! @see BFMatcher_CUDA::knnMatchDownload
    static void knnMatch2Download(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    //! Convert trainIdx and distance to vector with DMatch
    //! @see BFMatcher_CUDA::knnMatchConvert
    static void knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    //! Find k best matches  for each query descriptor (in increasing order of distances).
    //! compactResult is used when mask is not empty. If compactResult is false matches
    //! vector will have the same size as queryDescriptors rows. If compactResult is true
    //! matches vector will not contain matches for fully masked out query descriptors.
    void knnMatch(const GpuMat& query, std::vector< std::vector<DMatch> >& matches, int k,
        const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false);

    //! Find best matches for each query descriptor which have distance less than maxDistance.
    //! nMatches.at<int>(0, queryIdx) will contain matches count for queryIdx.
    //! carefully nMatches can be greater than trainIdx.cols - it means that matcher didn't find all matches,
    //! because it didn't have enough memory.
    //! If trainIdx is empty, then trainIdx and distance will be created with size nQuery x max((nTrain / 100), 10),
    //! otherwize user can pass own allocated trainIdx and distance with size nQuery x nMaxMatches
    //! Matches doesn't sorted.
    void radiusMatchSingle(const GpuMat& query, const GpuMat& train,
        GpuMat& trainIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance,
        const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

    //! Download trainIdx, nMatches and distance and convert it to vector with DMatch.
    //! matches will be sorted in increasing order of distances.
    //! compactResult is used when mask is not empty. If compactResult is false matches
    //! vector will have the same size as queryDescriptors rows. If compactResult is true
    //! matches vector will not contain matches for fully masked out query descriptors.
    static void radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, const GpuMat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    //! Convert trainIdx, nMatches and distance to vector with DMatch.
    static void radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    //! Find best matches for each query descriptor which have distance less than maxDistance
    //! in increasing order of distances).
    void radiusMatch(const GpuMat& query, const GpuMat& train,
        std::vector< std::vector<DMatch> >& matches, float maxDistance,
        const GpuMat& mask = GpuMat(), bool compactResult = false);

    //! Find best matches for each query descriptor which have distance less than maxDistance.
    //! If trainIdx is empty, then trainIdx and distance will be created with size nQuery x max((nQuery / 100), 10),
    //! otherwize user can pass own allocated trainIdx and distance with size nQuery x nMaxMatches
    //! Matches doesn't sorted.
    void radiusMatchCollection(const GpuMat& query, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance,
        const std::vector<GpuMat>& masks = std::vector<GpuMat>(), Stream& stream = Stream::Null());

    //! Download trainIdx, imgIdx, nMatches and distance and convert it to vector with DMatch.
    //! matches will be sorted in increasing order of distances.
    //! compactResult is used when mask is not empty. If compactResult is false matches
    //! vector will have the same size as queryDescriptors rows. If compactResult is true
    //! matches vector will not contain matches for fully masked out query descriptors.
    static void radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, const GpuMat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
    //! Convert trainIdx, nMatches and distance to vector with DMatch.
    static void radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches,
        std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

    //! Find best matches from train collection for each query descriptor which have distance less than
    //! maxDistance (in increasing order of distances).
    void radiusMatch(const GpuMat& query, std::vector< std::vector<DMatch> >& matches, float maxDistance,
        const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false);

    int norm;

private:
    std::vector<GpuMat> trainDescCollection;
};

//
// Feature2DAsync
//

/** @brief Abstract base class for 2D image feature detectors and descriptor extractors.
 */
class CV_EXPORTS Feature2DAsync
{
public:
    virtual ~Feature2DAsync() {}

    virtual void detectAsync(InputArray image, OutputArray keypoints,
                             InputArray mask = noArray(),
                             Stream& stream = Stream::Null()) = 0;

    virtual void convert(InputArray gpu_keypoints, std::vector<KeyPoint>& keypoints) = 0;
};

//
// FastFeatureDetector
//

/** @brief Wrapping class for feature detection using the FAST method.
 */
class CV_EXPORTS FastFeatureDetector : public cv::FastFeatureDetector, public Feature2DAsync
{
public:
    enum
    {
        LOCATION_ROW = 0,
        RESPONSE_ROW,
        ROWS_COUNT,

        FEATURE_SIZE = 7
    };

    static Ptr<FastFeatureDetector> create(int threshold=10,
                                           bool nonmaxSuppression=true,
                                           int type=FastFeatureDetector::TYPE_9_16,
                                           int max_npoints = 5000);

    virtual void setMaxNumPoints(int max_npoints) = 0;
    virtual int getMaxNumPoints() const = 0;
};

/** @brief Class for extracting ORB features and descriptors from an image. :
 */
class CV_EXPORTS ORB_CUDA
{
public:
    enum
    {
        X_ROW = 0,
        Y_ROW,
        RESPONSE_ROW,
        ANGLE_ROW,
        OCTAVE_ROW,
        SIZE_ROW,
        ROWS_COUNT
    };

    enum
    {
        DEFAULT_FAST_THRESHOLD = 20
    };

    /** @brief Constructor.

    @param nFeatures The number of desired features.
    @param scaleFactor Coefficient by which we divide the dimensions from one scale pyramid level to
    the next.
    @param nLevels The number of levels in the scale pyramid.
    @param edgeThreshold How far from the boundary the points should be.
    @param firstLevel The level at which the image is given. If 1, that means we will also look at the
    image scaleFactor times bigger.
    @param WTA_K
    @param scoreType
    @param patchSize
     */
    explicit ORB_CUDA(int nFeatures = 500, float scaleFactor = 1.2f, int nLevels = 8, int edgeThreshold = 31,
                     int firstLevel = 0, int WTA_K = 2, int scoreType = 0, int patchSize = 31);

    /** @overload */
    void operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints);
    /** @overload */
    void operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints);

    /** @brief Detects keypoints and computes descriptors for them.

    @param image Input 8-bit grayscale image.
    @param mask Optional input mask that marks the regions where we should detect features.
    @param keypoints The input/output vector of keypoints. Can be stored both in CPU and GPU memory.
    For GPU memory:
    -   keypoints.ptr\<float\>(X_ROW)[i] contains x coordinate of the i'th feature.
    -   keypoints.ptr\<float\>(Y_ROW)[i] contains y coordinate of the i'th feature.
    -   keypoints.ptr\<float\>(RESPONSE_ROW)[i] contains the response of the i'th feature.
    -   keypoints.ptr\<float\>(ANGLE_ROW)[i] contains orientation of the i'th feature.
    -   keypoints.ptr\<float\>(OCTAVE_ROW)[i] contains the octave of the i'th feature.
    -   keypoints.ptr\<float\>(SIZE_ROW)[i] contains the size of the i'th feature.
    @param descriptors Computed descriptors. if blurForDescriptor is true, image will be blurred
    before descriptors calculation.
     */
    void operator()(const GpuMat& image, const GpuMat& mask, std::vector<KeyPoint>& keypoints, GpuMat& descriptors);
    /** @overload */
    void operator()(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors);

    /** @brief Download keypoints from GPU to CPU memory.
    */
    static void downloadKeyPoints(const GpuMat& d_keypoints, std::vector<KeyPoint>& keypoints);
    /** @brief Converts keypoints from CUDA representation to vector of KeyPoint.
    */
    static void convertKeyPoints(const Mat& d_keypoints, std::vector<KeyPoint>& keypoints);

    //! returns the descriptor size in bytes
    inline int descriptorSize() const { return kBytes; }

    inline void setFastParams(int threshold, bool nonmaxSuppression = true)
    {
        fastDetector_->setThreshold(threshold);
        fastDetector_->setNonmaxSuppression(nonmaxSuppression);
    }

    /** @brief Releases inner buffer memory.
    */
    void release();

    //! if true, image will be blurred before descriptors calculation
    bool blurForDescriptor;

private:
    enum { kBytes = 32 };

    void buildScalePyramids(const GpuMat& image, const GpuMat& mask);

    void computeKeyPointsPyramid();

    void computeDescriptors(GpuMat& descriptors);

    void mergeKeyPoints(GpuMat& keypoints);

    int nFeatures_;
    float scaleFactor_;
    int nLevels_;
    int edgeThreshold_;
    int firstLevel_;
    int WTA_K_;
    int scoreType_;
    int patchSize_;

    //! The number of desired features per scale
    std::vector<size_t> n_features_per_level_;

    //! Points to compute BRIEF descriptors from
    GpuMat pattern_;

    std::vector<GpuMat> imagePyr_;
    std::vector<GpuMat> maskPyr_;

    GpuMat buf_;

    std::vector<GpuMat> keyPointsPyr_;
    std::vector<int> keyPointsCount_;

    Ptr<cv::cuda::FastFeatureDetector> fastDetector_;

    Ptr<cuda::Filter> blurFilter;

    GpuMat d_keypoints_;
};

//! @}

}} // namespace cv { namespace cuda {

#endif /* __OPENCV_CUDAFEATURES2D_HPP__ */
