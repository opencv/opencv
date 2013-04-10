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

#ifndef __OPENCV_GPU_HPP__
#define __OPENCV_GPU_HPP__

#ifndef SKIP_INCLUDES
#include <vector>
#include <memory>
#include <iosfwd>
#endif

#include "opencv2/core/gpumat.hpp"
#include "opencv2/gpuarithm.hpp"
#include "opencv2/gpufilters.hpp"
#include "opencv2/gpuimgproc.hpp"
#include "opencv2/gpufeatures2d.hpp"
#include "opencv2/gpuvideo.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"

namespace cv { namespace gpu {
////////////////////////////// Image processing //////////////////////////////




///////////////////////////// Calibration 3D //////////////////////////////////

CV_EXPORTS void transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                                GpuMat& dst, Stream& stream = Stream::Null());

CV_EXPORTS void projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                              const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst,
                              Stream& stream = Stream::Null());

CV_EXPORTS void solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat,
                               const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false,
                               int num_iters=100, float max_dist=8.0, int min_inlier_count=100,
                               std::vector<int>* inliers=NULL);

//////////////////////////////// Image Labeling ////////////////////////////////



////////////////////////////////// Histograms //////////////////////////////////



//////////////////////////////// StereoBM_GPU ////////////////////////////////

class CV_EXPORTS StereoBM_GPU
{
public:
    enum { BASIC_PRESET = 0, PREFILTER_XSOBEL = 1 };

    enum { DEFAULT_NDISP = 64, DEFAULT_WINSZ = 19 };

    //! the default constructor
    StereoBM_GPU();
    //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size. ndisparities must be multiple of 8.
    StereoBM_GPU(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ);

    //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
    //! Output disparity has CV_8U type.
    void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());

    //! Some heuristics that tries to estmate
    // if current GPU will be faster than CPU in this algorithm.
    // It queries current active device.
    static bool checkIfGpuCallReasonable();

    int preset;
    int ndisp;
    int winSize;

    // If avergeTexThreshold  == 0 => post procesing is disabled
    // If avergeTexThreshold != 0 then disparity is set 0 in each point (x,y) where for left image
    // SumOfHorizontalGradiensInWindow(x, y, winSize) < (winSize * winSize) * avergeTexThreshold
    // i.e. input left image is low textured.
    float avergeTexThreshold;

private:
    GpuMat minSSD, leBuf, riBuf;
};

////////////////////////// StereoBeliefPropagation ///////////////////////////
// "Efficient Belief Propagation for Early Vision"
// P.Felzenszwalb

class CV_EXPORTS StereoBeliefPropagation
{
public:
    enum { DEFAULT_NDISP  = 64 };
    enum { DEFAULT_ITERS  = 5  };
    enum { DEFAULT_LEVELS = 5  };

    static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels);

    //! the default constructor
    explicit StereoBeliefPropagation(int ndisp  = DEFAULT_NDISP,
                                     int iters  = DEFAULT_ITERS,
                                     int levels = DEFAULT_LEVELS,
                                     int msg_type = CV_32F);

    //! the full constructor taking the number of disparities, number of BP iterations on each level,
    //! number of levels, truncation of data cost, data weight,
    //! truncation of discontinuity cost and discontinuity single jump
    //! DataTerm = data_weight * min(fabs(I2-I1), max_data_term)
    //! DiscTerm = min(disc_single_jump * fabs(f1-f2), max_disc_term)
    //! please see paper for more details
    StereoBeliefPropagation(int ndisp, int iters, int levels,
        float max_data_term, float data_weight,
        float max_disc_term, float disc_single_jump,
        int msg_type = CV_32F);

    //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
    //! if disparity is empty output type will be CV_16S else output type will be disparity.type().
    void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());


    //! version for user specified data term
    void operator()(const GpuMat& data, GpuMat& disparity, Stream& stream = Stream::Null());

    int ndisp;

    int iters;
    int levels;

    float max_data_term;
    float data_weight;
    float max_disc_term;
    float disc_single_jump;

    int msg_type;
private:
    GpuMat u, d, l, r, u2, d2, l2, r2;
    std::vector<GpuMat> datas;
    GpuMat out;
};

/////////////////////////// StereoConstantSpaceBP ///////////////////////////
// "A Constant-Space Belief Propagation Algorithm for Stereo Matching"
// Qingxiong Yang, Liang Wang, Narendra Ahuja
// http://vision.ai.uiuc.edu/~qyang6/

class CV_EXPORTS StereoConstantSpaceBP
{
public:
    enum { DEFAULT_NDISP    = 128 };
    enum { DEFAULT_ITERS    = 8   };
    enum { DEFAULT_LEVELS   = 4   };
    enum { DEFAULT_NR_PLANE = 4   };

    static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane);

    //! the default constructor
    explicit StereoConstantSpaceBP(int ndisp    = DEFAULT_NDISP,
                                   int iters    = DEFAULT_ITERS,
                                   int levels   = DEFAULT_LEVELS,
                                   int nr_plane = DEFAULT_NR_PLANE,
                                   int msg_type = CV_32F);

    //! the full constructor taking the number of disparities, number of BP iterations on each level,
    //! number of levels, number of active disparity on the first level, truncation of data cost, data weight,
    //! truncation of discontinuity cost, discontinuity single jump and minimum disparity threshold
    StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane,
        float max_data_term, float data_weight, float max_disc_term, float disc_single_jump,
        int min_disp_th = 0,
        int msg_type = CV_32F);

    //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
    //! if disparity is empty output type will be CV_16S else output type will be disparity.type().
    void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());

    int ndisp;

    int iters;
    int levels;

    int nr_plane;

    float max_data_term;
    float data_weight;
    float max_disc_term;
    float disc_single_jump;

    int min_disp_th;

    int msg_type;

    bool use_local_init_data_cost;
private:
    GpuMat messages_buffers;

    GpuMat temp;
    GpuMat out;
};

/////////////////////////// DisparityBilateralFilter ///////////////////////////
// Disparity map refinement using joint bilateral filtering given a single color image.
// Qingxiong Yang, Liang Wang, Narendra Ahuja
// http://vision.ai.uiuc.edu/~qyang6/

class CV_EXPORTS DisparityBilateralFilter
{
public:
    enum { DEFAULT_NDISP  = 64 };
    enum { DEFAULT_RADIUS = 3 };
    enum { DEFAULT_ITERS  = 1 };

    //! the default constructor
    explicit DisparityBilateralFilter(int ndisp = DEFAULT_NDISP, int radius = DEFAULT_RADIUS, int iters = DEFAULT_ITERS);

    //! the full constructor taking the number of disparities, filter radius,
    //! number of iterations, truncation of data continuity, truncation of disparity continuity
    //! and filter range sigma
    DisparityBilateralFilter(int ndisp, int radius, int iters, float edge_threshold, float max_disc_threshold, float sigma_range);

    //! the disparity map refinement operator. Refine disparity map using joint bilateral filtering given a single color image.
    //! disparity must have CV_8U or CV_16S type, image must have CV_8UC1 or CV_8UC3 type.
    void operator()(const GpuMat& disparity, const GpuMat& image, GpuMat& dst, Stream& stream = Stream::Null());

private:
    int ndisp;
    int radius;
    int iters;

    float edge_threshold;
    float max_disc_threshold;
    float sigma_range;

    GpuMat table_color;
    GpuMat table_space;
};


//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////
struct CV_EXPORTS HOGConfidence
{
   double scale;
   std::vector<Point> locations;
   std::vector<double> confidences;
   std::vector<double> part_scores[4];
};

struct CV_EXPORTS HOGDescriptor
{
    enum { DEFAULT_WIN_SIGMA = -1 };
    enum { DEFAULT_NLEVELS = 64 };
    enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };

    HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
                  Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
                  int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
                  double threshold_L2hys=0.2, bool gamma_correction=true,
                  int nlevels=DEFAULT_NLEVELS);

    size_t getDescriptorSize() const;
    size_t getBlockHistogramSize() const;

    void setSVMDetector(const std::vector<float>& detector);

    static std::vector<float> getDefaultPeopleDetector();
    static std::vector<float> getPeopleDetector48x96();
    static std::vector<float> getPeopleDetector64x128();

    void detect(const GpuMat& img, std::vector<Point>& found_locations,
                double hit_threshold=0, Size win_stride=Size(),
                Size padding=Size());

    void detectMultiScale(const GpuMat& img, std::vector<Rect>& found_locations,
                          double hit_threshold=0, Size win_stride=Size(),
                          Size padding=Size(), double scale0=1.05,
                          int group_threshold=2);

    void computeConfidence(const GpuMat& img, std::vector<Point>& hits, double hit_threshold,
                                                Size win_stride, Size padding, std::vector<Point>& locations, std::vector<double>& confidences);

    void computeConfidenceMultiScale(const GpuMat& img, std::vector<Rect>& found_locations,
                                                                    double hit_threshold, Size win_stride, Size padding,
                                                                    std::vector<HOGConfidence> &conf_out, int group_threshold);

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


////////////////////////////////// BruteForceMatcher //////////////////////////////////



template <class Distance>
class CV_EXPORTS BruteForceMatcher_GPU;

template <typename T>
class CV_EXPORTS BruteForceMatcher_GPU< L1<T> > : public BFMatcher_GPU
{
public:
    explicit BruteForceMatcher_GPU() : BFMatcher_GPU(NORM_L1) {}
    explicit BruteForceMatcher_GPU(L1<T> /*d*/) : BFMatcher_GPU(NORM_L1) {}
};
template <typename T>
class CV_EXPORTS BruteForceMatcher_GPU< L2<T> > : public BFMatcher_GPU
{
public:
    explicit BruteForceMatcher_GPU() : BFMatcher_GPU(NORM_L2) {}
    explicit BruteForceMatcher_GPU(L2<T> /*d*/) : BFMatcher_GPU(NORM_L2) {}
};
template <> class CV_EXPORTS BruteForceMatcher_GPU< Hamming > : public BFMatcher_GPU
{
public:
    explicit BruteForceMatcher_GPU() : BFMatcher_GPU(NORM_HAMMING) {}
    explicit BruteForceMatcher_GPU(Hamming /*d*/) : BFMatcher_GPU(NORM_HAMMING) {}
};

////////////////////////////////// CascadeClassifier_GPU //////////////////////////////////////////
// The cascade classifier class for object detection: supports old haar and new lbp xlm formats and nvbin for haar cascades olny.
class CV_EXPORTS CascadeClassifier_GPU
{
public:
    CascadeClassifier_GPU();
    CascadeClassifier_GPU(const String& filename);
    ~CascadeClassifier_GPU();

    bool empty() const;
    bool load(const String& filename);
    void release();

    /* returns number of detected objects */
    int detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, double scaleFactor = 1.2, int minNeighbors = 4, Size minSize = Size());
    int detectMultiScale(const GpuMat& image, GpuMat& objectsBuf, Size maxObjectSize, Size minSize = Size(), double scaleFactor = 1.1, int minNeighbors = 4);

    bool findLargestObject;
    bool visualizeInPlace;

    Size getClassifierSize() const;

private:
    struct CascadeClassifierImpl;
    CascadeClassifierImpl* impl;
    struct HaarCascade;
    struct LbpCascade;
    friend class CascadeClassifier_GPU_LBP;
};

////////////////////////////////// FAST //////////////////////////////////////////



////////////////////////////////// ORB //////////////////////////////////////////





















//! removes points (CV_32FC2, single row matrix) with zero mask value
CV_EXPORTS void compactPoints(GpuMat &points0, GpuMat &points1, const GpuMat &mask);

CV_EXPORTS void calcWobbleSuppressionMaps(
        int left, int idx, int right, Size size, const Mat &ml, const Mat &mr,
        GpuMat &mapx, GpuMat &mapy);

} // namespace gpu

} // namespace cv

#endif /* __OPENCV_GPU_HPP__ */
