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

#ifndef __OPENCV_GPUSTEREO_HPP__
#define __OPENCV_GPUSTEREO_HPP__

#ifndef __cplusplus
#  error gpustereo.hpp header must be compiled as C++
#endif

#include "opencv2/core/gpumat.hpp"

namespace cv { namespace gpu {

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

//! Reprojects disparity image to 3D space.
//! Supports CV_8U and CV_16S types of input disparity.
//! The output is a 3- or 4-channel floating-point matrix.
//! Each element of this matrix will contain the 3D coordinates of the point (x,y,z,1), computed from the disparity map.
//! Q is the 4x4 perspective transformation matrix that can be obtained with cvStereoRectify.
CV_EXPORTS void reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, int dst_cn = 4, Stream& stream = Stream::Null());

//! Does coloring of disparity image: [0..ndisp) -> [0..240, 1, 1] in HSV.
//! Supported types of input disparity: CV_8U, CV_16S.
//! Output disparity has CV_8UC4 type in BGRA format (alpha = 255).
CV_EXPORTS void drawColorDisp(const GpuMat& src_disp, GpuMat& dst_disp, int ndisp, Stream& stream = Stream::Null());

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPUSTEREO_HPP__ */
