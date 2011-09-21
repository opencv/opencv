 /*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

#ifndef __OPENCV_STITCHING_WARPERS_HPP__
#define __OPENCV_STITCHING_WARPERS_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#ifndef ANDROID
# include "opencv2/gpu/gpu.hpp"
#endif

namespace cv {
namespace detail {

class CV_EXPORTS Warper
{
public:
    virtual ~Warper() {}
    virtual Point warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
                       int interp_mode, int border_mode) = 0;
    virtual Point warp(const Mat &src, const Mat &K, const Mat &R, const Mat &T, Mat &dst,
                       int interp_mode, int border_mode) = 0;
    virtual Rect warpRoi(const Size &sz, const Mat &K, const Mat &R) = 0;
    virtual Rect warpRoi(const Size &sz, const Mat &K, const Mat &R, const Mat &T) = 0;
};


struct CV_EXPORTS ProjectorBase
{
    void setCameraParams(const Mat &K = Mat::eye(3, 3, CV_32F), 
                         const Mat &R = Mat::eye(3, 3, CV_32F), 
                         const Mat &T = Mat::zeros(3, 1, CV_32F));

    float scale;
    float k[9];
    float rinv[9];
    float r_kinv[9];
    float k_rinv[9];
    float t[3];
};


template <class P>
class CV_EXPORTS WarperBase : public Warper
{   
public:
    Point warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
               int interp_mode, int border_mode);
    Point warp(const Mat &src, const Mat &K, const Mat &R, const Mat &T, Mat &dst,
               int interp_mode, int border_mode);
    Rect warpRoi(const Size &sz, const Mat &K, const Mat &R);
    Rect warpRoi(const Size &sz, const Mat &K, const Mat &R, const Mat &T);

protected:
    // Detects ROI of the destination image. It's correct for any projection.
    virtual void detectResultRoi(Point &dst_tl, Point &dst_br);

    // Detects ROI of the destination image by walking over image border.
    // Correctness for any projection isn't guaranteed.
    void detectResultRoiByBorder(Point &dst_tl, Point &dst_br);

    Size src_size_;
    P projector_;
};


struct CV_EXPORTS PlaneProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto z = plane_dist plane
class CV_EXPORTS PlaneWarper : public WarperBase<PlaneProjector>
{
public:
    PlaneWarper(float scale = 1.f) { projector_.scale = scale; }
    void setScale(float scale) { projector_.scale = scale; }
    Point warp(const Mat &src, const Mat &K, const Mat &R, const Mat &T, Mat &dst,
               int interp_mode, int border_mode);
    Rect warpRoi(const Size &sz, const Mat &K, const Mat &R, const Mat &T);

protected:
    void detectResultRoi(Point &dst_tl, Point &dst_br);
};

#ifndef ANDROID
class CV_EXPORTS PlaneWarperGpu : public PlaneWarper
{
public:
    PlaneWarperGpu(float scale = 1.f) : PlaneWarper(scale) {}
    Point warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
               int interp_mode, int border_mode);
    Point warp(const Mat &src, const Mat &K, const Mat &R, const Mat &T, Mat &dst,
               int interp_mode, int border_mode);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_dst_, d_src_;
};
#endif


struct CV_EXPORTS SphericalProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto unit sphere with origin at (0, 0, 0).
// Poles are located at (0, -1, 0) and (0, 1, 0) points.
class CV_EXPORTS SphericalWarper : public WarperBase<SphericalProjector>
{
public:
    SphericalWarper(float scale) { projector_.scale = scale; }

protected:
    void detectResultRoi(Point &dst_tl, Point &dst_br);
};


#ifndef ANDROID
class CV_EXPORTS SphericalWarperGpu : public SphericalWarper
{
public:
    SphericalWarperGpu(float scale) : SphericalWarper(scale) {}
    Point warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
               int interp_mode, int border_mode);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_dst_, d_src_;
};
#endif


struct CV_EXPORTS CylindricalProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto x * x + z * z = 1 cylinder
class CV_EXPORTS CylindricalWarper : public WarperBase<CylindricalProjector>
{
public:
    CylindricalWarper(float scale) { projector_.scale = scale; }

protected:
    void detectResultRoi(Point &dst_tl, Point &dst_br)
        { WarperBase<CylindricalProjector>::detectResultRoiByBorder(dst_tl, dst_br); }
};


#ifndef ANDROID
class CV_EXPORTS CylindricalWarperGpu : public CylindricalWarper
{
public:
    CylindricalWarperGpu(float scale) : CylindricalWarper(scale) {}
    Point warp(const Mat &src, const Mat &K, const Mat &R, Mat &dst,
               int interp_mode, int border_mode);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_dst_, d_src_;
};
#endif

} // namespace detail
} // namespace cv

#include "warpers_inl.hpp"

#endif // __OPENCV_STITCHING_WARPERS_HPP__
