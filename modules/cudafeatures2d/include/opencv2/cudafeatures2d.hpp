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

#ifndef OPENCV_CUDAFEATURES2D_HPP
#define OPENCV_CUDAFEATURES2D_HPP

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

//
// DescriptorMatcher
//

/** @brief Abstract base class for matching keypoint descriptors.

It has two groups of match methods: for matching descriptors of an image with another image or with
an image set.
 */
class CV_EXPORTS DescriptorMatcher : public cv::Algorithm
{
public:
    //
    // Factories
    //

    /** @brief Brute-force descriptor matcher.

    For each descriptor in the first set, this matcher finds the closest descriptor in the second set
    by trying each one. This descriptor matcher supports masking permissible matches of descriptor
    sets.

    @param normType One of NORM_L1, NORM_L2, NORM_HAMMING. L1 and L2 norms are
    preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
    BRIEF).
     */
    static Ptr<DescriptorMatcher> createBFMatcher(int normType = cv::NORM_L2);

    //
    // Utility
    //

    /** @brief Returns true if the descriptor matcher supports masking permissible matches.
     */
    virtual bool isMaskSupported() const = 0;

    //
    // Descriptor collection
    //

    /** @brief Adds descriptors to train a descriptor collection.

    If the collection is not empty, the new descriptors are added to existing train descriptors.

    @param descriptors Descriptors to add. Each descriptors[i] is a set of descriptors from the same
    train image.
     */
    virtual void add(const std::vector<GpuMat>& descriptors) = 0;

    /** @brief Returns a constant link to the train descriptor collection.
     */
    virtual const std::vector<GpuMat>& getTrainDescriptors() const = 0;

    /** @brief Clears the train descriptor collection.
     */
    virtual void clear() = 0;

    /** @brief Returns true if there are no train descriptors in the collection.
     */
    virtual bool empty() const = 0;

    /** @brief Trains a descriptor matcher.

    Trains a descriptor matcher (for example, the flann index). In all methods to match, the method
    train() is run every time before matching.
     */
    virtual void train() = 0;

    //
    // 1 to 1 match
    //

    /** @brief Finds the best match for each descriptor from a query set (blocking version).

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
    virtual void match(InputArray queryDescriptors, InputArray trainDescriptors,
                       std::vector<DMatch>& matches,
                       InputArray mask = noArray()) = 0;

    /** @overload
     */
    virtual void match(InputArray queryDescriptors,
                       std::vector<DMatch>& matches,
                       const std::vector<GpuMat>& masks = std::vector<GpuMat>()) = 0;

    /** @brief Finds the best match for each descriptor from a query set (asynchronous version).

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Matches array stored in GPU memory. Internal representation is not defined.
    Use DescriptorMatcher::matchConvert method to retrieve results in standard representation.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param stream CUDA stream.

    In the first variant of this method, the train descriptors are passed as an input argument. In the
    second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is
    used. Optional mask (or masks) can be passed to specify which query and training descriptors can be
    matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if
    mask.at\<uchar\>(i,j) is non-zero.
     */
    virtual void matchAsync(InputArray queryDescriptors, InputArray trainDescriptors,
                            OutputArray matches,
                            InputArray mask = noArray(),
                            Stream& stream = Stream::Null()) = 0;

    /** @overload
     */
    virtual void matchAsync(InputArray queryDescriptors,
                            OutputArray matches,
                            const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                            Stream& stream = Stream::Null()) = 0;

    /** @brief Converts matches array from internal representation to standard matches vector.

    The method is supposed to be used with DescriptorMatcher::matchAsync to get final result.
    Call this method only after DescriptorMatcher::matchAsync is completed (ie. after synchronization).

    @param gpu_matches Matches, returned from DescriptorMatcher::matchAsync.
    @param matches Vector of DMatch objects.
     */
    virtual void matchConvert(InputArray gpu_matches,
                              std::vector<DMatch>& matches) = 0;

    //
    // knn match
    //

    /** @brief Finds the k best matches for each descriptor from a query set (blocking version).

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.

    These extended variants of DescriptorMatcher::match methods find several best matches for each query
    descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match
    for the details about query and train descriptors.
     */
    virtual void knnMatch(InputArray queryDescriptors, InputArray trainDescriptors,
                          std::vector<std::vector<DMatch> >& matches,
                          int k,
                          InputArray mask = noArray(),
                          bool compactResult = false) = 0;

    /** @overload
     */
    virtual void knnMatch(InputArray queryDescriptors,
                          std::vector<std::vector<DMatch> >& matches,
                          int k,
                          const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                          bool compactResult = false) = 0;

    /** @brief Finds the k best matches for each descriptor from a query set (asynchronous version).

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Matches array stored in GPU memory. Internal representation is not defined.
    Use DescriptorMatcher::knnMatchConvert method to retrieve results in standard representation.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param stream CUDA stream.

    These extended variants of DescriptorMatcher::matchAsync methods find several best matches for each query
    descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::matchAsync
    for the details about query and train descriptors.
     */
    virtual void knnMatchAsync(InputArray queryDescriptors, InputArray trainDescriptors,
                               OutputArray matches,
                               int k,
                               InputArray mask = noArray(),
                               Stream& stream = Stream::Null()) = 0;

    /** @overload
     */
    virtual void knnMatchAsync(InputArray queryDescriptors,
                               OutputArray matches,
                               int k,
                               const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                               Stream& stream = Stream::Null()) = 0;

    /** @brief Converts matches array from internal representation to standard matches vector.

    The method is supposed to be used with DescriptorMatcher::knnMatchAsync to get final result.
    Call this method only after DescriptorMatcher::knnMatchAsync is completed (ie. after synchronization).

    @param gpu_matches Matches, returned from DescriptorMatcher::knnMatchAsync.
    @param matches Vector of DMatch objects.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
     */
    virtual void knnMatchConvert(InputArray gpu_matches,
                                 std::vector< std::vector<DMatch> >& matches,
                                 bool compactResult = false) = 0;

    //
    // radius match
    //

    /** @brief For each query descriptor, finds the training descriptors not farther than the specified distance (blocking version).

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Found matches.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.

    For each query descriptor, the methods find such training descriptors that the distance between the
    query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
    returned in the distance increasing order.
     */
    virtual void radiusMatch(InputArray queryDescriptors, InputArray trainDescriptors,
                             std::vector<std::vector<DMatch> >& matches,
                             float maxDistance,
                             InputArray mask = noArray(),
                             bool compactResult = false) = 0;

    /** @overload
     */
    virtual void radiusMatch(InputArray queryDescriptors,
                             std::vector<std::vector<DMatch> >& matches,
                             float maxDistance,
                             const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                             bool compactResult = false) = 0;

    /** @brief For each query descriptor, finds the training descriptors not farther than the specified distance (asynchronous version).

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Matches array stored in GPU memory. Internal representation is not defined.
    Use DescriptorMatcher::radiusMatchConvert method to retrieve results in standard representation.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param stream CUDA stream.

    For each query descriptor, the methods find such training descriptors that the distance between the
    query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
    returned in the distance increasing order.
     */
    virtual void radiusMatchAsync(InputArray queryDescriptors, InputArray trainDescriptors,
                                  OutputArray matches,
                                  float maxDistance,
                                  InputArray mask = noArray(),
                                  Stream& stream = Stream::Null()) = 0;

    /** @overload
     */
    virtual void radiusMatchAsync(InputArray queryDescriptors,
                                  OutputArray matches,
                                  float maxDistance,
                                  const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                                  Stream& stream = Stream::Null()) = 0;

    /** @brief Converts matches array from internal representation to standard matches vector.

    The method is supposed to be used with DescriptorMatcher::radiusMatchAsync to get final result.
    Call this method only after DescriptorMatcher::radiusMatchAsync is completed (ie. after synchronization).

    @param gpu_matches Matches, returned from DescriptorMatcher::radiusMatchAsync.
    @param matches Vector of DMatch objects.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
     */
    virtual void radiusMatchConvert(InputArray gpu_matches,
                                    std::vector< std::vector<DMatch> >& matches,
                                    bool compactResult = false) = 0;
};

//
// Feature2DAsync
//

/** @brief Abstract base class for CUDA asynchronous 2D image feature detectors and descriptor extractors.
 */
class CV_EXPORTS Feature2DAsync
{
public:
    virtual ~Feature2DAsync();

    /** @brief Detects keypoints in an image.

    @param image Image.
    @param keypoints The detected keypoints.
    @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
    matrix with non-zero values in the region of interest.
    @param stream CUDA stream.
     */
    virtual void detectAsync(InputArray image,
                             OutputArray keypoints,
                             InputArray mask = noArray(),
                             Stream& stream = Stream::Null());

    /** @brief Computes the descriptors for a set of keypoints detected in an image.

    @param image Image.
    @param keypoints Input collection of keypoints.
    @param descriptors Computed descriptors. Row j is the descriptor for j-th keypoint.
    @param stream CUDA stream.
     */
    virtual void computeAsync(InputArray image,
                              OutputArray keypoints,
                              OutputArray descriptors,
                              Stream& stream = Stream::Null());

    /** Detects keypoints and computes the descriptors. */
    virtual void detectAndComputeAsync(InputArray image,
                                       InputArray mask,
                                       OutputArray keypoints,
                                       OutputArray descriptors,
                                       bool useProvidedKeypoints = false,
                                       Stream& stream = Stream::Null());

    /** Converts keypoints array from internal representation to standard vector. */
    virtual void convert(InputArray gpu_keypoints,
                         std::vector<KeyPoint>& keypoints) = 0;
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

//
// ORB
//

/** @brief Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor
 *
 * @sa cv::ORB
 */
class CV_EXPORTS ORB : public cv::ORB, public Feature2DAsync
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

    static Ptr<ORB> create(int nfeatures=500,
                           float scaleFactor=1.2f,
                           int nlevels=8,
                           int edgeThreshold=31,
                           int firstLevel=0,
                           int WTA_K=2,
                           int scoreType=ORB::HARRIS_SCORE,
                           int patchSize=31,
                           int fastThreshold=20,
                           bool blurForDescriptor=false);

    //! if true, image will be blurred before descriptors calculation
    virtual void setBlurForDescriptor(bool blurForDescriptor) = 0;
    virtual bool getBlurForDescriptor() const = 0;
};

//! @}

}} // namespace cv { namespace cuda {

#endif /* OPENCV_CUDAFEATURES2D_HPP */
