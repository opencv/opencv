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

#ifndef __OPENCV_CUDA_HPP__
#define __OPENCV_CUDA_HPP__

#ifndef __cplusplus
#  error cuda.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"

/**
    @addtogroup cuda
    @{
        @defgroup cuda_calib3d Camera Calibration and 3D Reconstruction
        @defgroup cuda_objdetect Object Detection
    @}
 */

namespace cv { namespace cuda {

//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

//! @addtogroup cuda_objdetect
//! @{

struct CV_EXPORTS HOGConfidence
{
   double scale;
   std::vector<Point> locations;
   std::vector<double> confidences;
   std::vector<double> part_scores[4];
};

/** @brief The class implements Histogram of Oriented Gradients (@cite Dalal2005) object detector.

Interfaces of all methods are kept similar to the CPU HOG descriptor and detector analogues as much
as possible.

@note
   -   An example applying the HOG descriptor for people detection can be found at
        opencv_source_code/samples/cpp/peopledetect.cpp
    -   A CUDA example applying the HOG descriptor for people detection can be found at
        opencv_source_code/samples/gpu/hog.cpp
    -   (Python) An example applying the HOG descriptor for people detection can be found at
        opencv_source_code/samples/python2/peopledetect.py
 */
struct CV_EXPORTS HOGDescriptor
{
    enum { DEFAULT_WIN_SIGMA = -1 };
    enum { DEFAULT_NLEVELS = 64 };
    enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };

    /** @brief Creates the HOG descriptor and detector.

    @param win_size Detection window size. Align to block size and block stride.
    @param block_size Block size in pixels. Align to cell size. Only (16,16) is supported for now.
    @param block_stride Block stride. It must be a multiple of cell size.
    @param cell_size Cell size. Only (8, 8) is supported for now.
    @param nbins Number of bins. Only 9 bins per cell are supported for now.
    @param win_sigma Gaussian smoothing window parameter.
    @param threshold_L2hys L2-Hys normalization method shrinkage.
    @param gamma_correction Flag to specify whether the gamma correction preprocessing is required or
    not.
    @param nlevels Maximum number of detection window increases.
     */
    HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
                  Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
                  int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
                  double threshold_L2hys=0.2, bool gamma_correction=true,
                  int nlevels=DEFAULT_NLEVELS);

    /** @brief Returns the number of coefficients required for the classification.
     */
    size_t getDescriptorSize() const;
    /** @brief Returns the block histogram size.
    */
    size_t getBlockHistogramSize() const;

    /** @brief Sets coefficients for the linear SVM classifier.
    */
    void setSVMDetector(const std::vector<float>& detector);

    /** @brief Returns coefficients of the classifier trained for people detection (for default window size).
    */
    static std::vector<float> getDefaultPeopleDetector();
    /** @brief Returns coefficients of the classifier trained for people detection (for 48x96 windows).
    */
    static std::vector<float> getPeopleDetector48x96();
    /** @brief Returns coefficients of the classifier trained for people detection (for 64x128 windows).
    */
    static std::vector<float> getPeopleDetector64x128();

    /** @brief Performs object detection without a multi-scale window.

    @param img Source image. CV_8UC1 and CV_8UC4 types are supported for now.
    @param found_locations Left-top corner points of detected objects boundaries.
    @param hit_threshold Threshold for the distance between features and SVM classifying plane.
    Usually it is 0 and should be specfied in the detector coefficients (as the last free
    coefficient). But if the free coefficient is omitted (which is allowed), you can specify it
    manually here.
    @param win_stride Window stride. It must be a multiple of block stride.
    @param padding Mock parameter to keep the CPU interface compatibility. It must be (0,0).
     */
    void detect(const GpuMat& img, std::vector<Point>& found_locations,
                double hit_threshold=0, Size win_stride=Size(),
                Size padding=Size());

    /** @brief Performs object detection with a multi-scale window.

    @param img Source image. See cuda::HOGDescriptor::detect for type limitations.
    @param found_locations Detected objects boundaries.
    @param hit_threshold Threshold for the distance between features and SVM classifying plane. See
    cuda::HOGDescriptor::detect for details.
    @param win_stride Window stride. It must be a multiple of block stride.
    @param padding Mock parameter to keep the CPU interface compatibility. It must be (0,0).
    @param scale0 Coefficient of the detection window increase.
    @param group_threshold Coefficient to regulate the similarity threshold. When detected, some
    objects can be covered by many rectangles. 0 means not to perform grouping. See groupRectangles .
     */
    void detectMultiScale(const GpuMat& img, std::vector<Rect>& found_locations,
                          double hit_threshold=0, Size win_stride=Size(),
                          Size padding=Size(), double scale0=1.05,
                          int group_threshold=2);

    void computeConfidence(const GpuMat& img, std::vector<Point>& hits, double hit_threshold,
                                                Size win_stride, Size padding, std::vector<Point>& locations, std::vector<double>& confidences);

    void computeConfidenceMultiScale(const GpuMat& img, std::vector<Rect>& found_locations,
                                                                    double hit_threshold, Size win_stride, Size padding,
                                                                    std::vector<HOGConfidence> &conf_out, int group_threshold);

    /** @brief Returns block descriptors computed for the whole image.

    @param img Source image. See cuda::HOGDescriptor::detect for type limitations.
    @param win_stride Window stride. It must be a multiple of block stride.
    @param descriptors 2D array of descriptors.
    @param descr_format Descriptor storage format:
    -   **DESCR_FORMAT_ROW_BY_ROW** - Row-major order.
    -   **DESCR_FORMAT_COL_BY_COL** - Column-major order.

    The function is mainly used to learn the classifier.
     */
    void getDescriptors(const GpuMat& img, Size win_stride,
                        GpuMat& descriptors,
                        int descr_format=DESCR_FORMAT_COL_BY_COL);

    Size win_size;
    Size block_size;
    Size block_stride;
    Size cell_size;
    int nbins;
    double win_sigma;
    double threshold_L2hys;
    bool gamma_correction;
    int nlevels;

protected:
    void computeBlockHistograms(const GpuMat& img);
    void computeGradient(const GpuMat& img, GpuMat& grad, GpuMat& qangle);

    double getWinSigma() const;
    bool checkDetectorSize() const;

    static int numPartsWithin(int size, int part_size, int stride);
    static Size numPartsWithin(Size size, Size part_size, Size stride);

    // Coefficients of the separating plane
    float free_coef;
    GpuMat detector;

    // Results of the last classification step
    GpuMat labels, labels_buf;
    Mat labels_host;

    // Results of the last histogram evaluation step
    GpuMat block_hists, block_hists_buf;

    // Gradients conputation results
    GpuMat grad, qangle, grad_buf, qangle_buf;

    // returns subbuffer with required size, reallocates buffer if nessesary.
    static GpuMat getBuffer(const Size& sz, int type, GpuMat& buf);
    static GpuMat getBuffer(int rows, int cols, int type, GpuMat& buf);

    std::vector<GpuMat> image_scales;
};

//////////////////////////// CascadeClassifier ////////////////////////////

/** @brief Cascade classifier class used for object detection. Supports HAAR and LBP cascades. :

@note
   -   A cascade classifier example can be found at
        opencv_source_code/samples/gpu/cascadeclassifier.cpp
    -   A Nvidea API specific cascade classifier example can be found at
        opencv_source_code/samples/gpu/cascadeclassifier_nvidia_api.cpp
 */
class CV_EXPORTS CascadeClassifier_CUDA
{
public:
    CascadeClassifier_CUDA();
    /** @brief Loads the classifier from a file. Cascade type is detected automatically by constructor parameter.

    @param filename Name of the file from which the classifier is loaded. Only the old haar classifier
    (trained by the haar training application) and NVIDIA's nvbin are supported for HAAR and only new
    type of OpenCV XML cascade supported for LBP.
     */
    CascadeClassifier_CUDA(const String& filename);
    ~CascadeClassifier_CUDA();

    /** @brief Checks whether the classifier is loaded or not.
    */
    bool empty() const;
    /** @brief Loads the classifier from a file. The previous content is destroyed.

    @param filename Name of the file from which the classifier is loaded. Only the old haar classifier
    (trained by the haar training application) and NVIDIA's nvbin are supported for HAAR and only new
    type of OpenCV XML cascade supported for LBP.
     */
    bool load(const String& filename);
    /** @brief Destroys the loaded classifier.
    */
    void release();

    /** @overload */
    int detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, double scaleFactor = 1.2, int minNeighbors = 4, Size minSize = Size());
    /** @brief Detects objects of different sizes in the input image.

    @param image Matrix of type CV_8U containing an image where objects should be detected.
    @param objectsBuf Buffer to store detected objects (rectangles). If it is empty, it is allocated
    with the default size. If not empty, the function searches not more than N objects, where
    N = sizeof(objectsBufer's data)/sizeof(cv::Rect).
    @param maxObjectSize Maximum possible object size. Objects larger than that are ignored. Used for
    second signature and supported only for LBP cascades.
    @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    to retain it.
    @param minSize Minimum possible object size. Objects smaller than that are ignored.

    The detected objects are returned as a list of rectangles.

    The function returns the number of detected objects, so you can retrieve them as in the following
    example:
    @code
        cuda::CascadeClassifier_CUDA cascade_gpu(...);

        Mat image_cpu = imread(...)
        GpuMat image_gpu(image_cpu);

        GpuMat objbuf;
        int detections_number = cascade_gpu.detectMultiScale( image_gpu,
                  objbuf, 1.2, minNeighbors);

        Mat obj_host;
        // download only detected number of rectangles
        objbuf.colRange(0, detections_number).download(obj_host);

        Rect* faces = obj_host.ptr<Rect>();
        for(int i = 0; i < detections_num; ++i)
           cv::rectangle(image_cpu, faces[i], Scalar(255));

        imshow("Faces", image_cpu);
    @endcode
    @sa CascadeClassifier::detectMultiScale
     */
    int detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, Size maxObjectSize, Size minSize = Size(), double scaleFactor = 1.1, int minNeighbors = 4);

    bool findLargestObject;
    bool visualizeInPlace;

    Size getClassifierSize() const;

private:
    struct CascadeClassifierImpl;
    CascadeClassifierImpl* impl;
    struct HaarCascade;
    struct LbpCascade;
    friend class CascadeClassifier_CUDA_LBP;
};

//! @} cuda_objdetect

//////////////////////////// Labeling ////////////////////////////

//! @addtogroup cuda
//! @{

//!performs labeling via graph cuts of a 2D regular 4-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//!performs labeling via graph cuts of a 2D regular 8-connected graph.
CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& topLeft, GpuMat& topRight,
                         GpuMat& bottom, GpuMat& bottomLeft, GpuMat& bottomRight,
                         GpuMat& labels,
                         GpuMat& buf, Stream& stream = Stream::Null());

//! compute mask for Generalized Flood fill componetns labeling.
CV_EXPORTS void connectivityMask(const GpuMat& image, GpuMat& mask, const cv::Scalar& lo, const cv::Scalar& hi, Stream& stream = Stream::Null());

//! performs connected componnents labeling.
CV_EXPORTS void labelComponents(const GpuMat& mask, GpuMat& components, int flags = 0, Stream& stream = Stream::Null());

//! @}

//////////////////////////// Calib3d ////////////////////////////

//! @addtogroup cuda_calib3d
//! @{

CV_EXPORTS void transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                                GpuMat& dst, Stream& stream = Stream::Null());

CV_EXPORTS void projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                              const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst,
                              Stream& stream = Stream::Null());

/** @brief Finds the object pose from 3D-2D point correspondences.

@param object Single-row matrix of object points.
@param image Single-row matrix of image points.
@param camera_mat 3x3 matrix of intrinsic camera parameters.
@param dist_coef Distortion coefficients. See undistortPoints for details.
@param rvec Output 3D rotation vector.
@param tvec Output 3D translation vector.
@param use_extrinsic_guess Flag to indicate that the function must use rvec and tvec as an
initial transformation guess. It is not supported for now.
@param num_iters Maximum number of RANSAC iterations.
@param max_dist Euclidean distance threshold to detect whether point is inlier or not.
@param min_inlier_count Flag to indicate that the function must stop if greater or equal number
of inliers is achieved. It is not supported for now.
@param inliers Output vector of inlier indices.
 */
CV_EXPORTS void solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat,
                               const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false,
                               int num_iters=100, float max_dist=8.0, int min_inlier_count=100,
                               std::vector<int>* inliers=NULL);

//! @}

//////////////////////////// VStab ////////////////////////////

//! @addtogroup cuda
//! @{

//! removes points (CV_32FC2, single row matrix) with zero mask value
CV_EXPORTS void compactPoints(GpuMat &points0, GpuMat &points1, const GpuMat &mask);

CV_EXPORTS void calcWobbleSuppressionMaps(
        int left, int idx, int right, Size size, const Mat &ml, const Mat &mr,
        GpuMat &mapx, GpuMat &mapy);

//! @}

}} // namespace cv { namespace cuda {

#endif /* __OPENCV_CUDA_HPP__ */
