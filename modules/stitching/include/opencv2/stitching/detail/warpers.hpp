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
#include "opencv2/gpu/gpu.hpp"

namespace cv {
namespace detail {

class CV_EXPORTS Warper
{
public:
    enum { PLANE, CYLINDRICAL, SPHERICAL };
    static Ptr<Warper> createByCameraFocal(float focal, int type, bool try_gpu = false);

    virtual ~Warper() {}
    virtual Point warp(const Mat &src, float focal, const Mat& R, Mat &dst,
                           int interp_mode = INTER_LINEAR, int border_mode = BORDER_REFLECT) = 0;
    virtual Rect warpRoi(const Size &sz, float focal, const Mat &R) = 0;
};


struct CV_EXPORTS ProjectorBase
{
    void setTransformation(const Mat& R);

    Size size;
    float focal;
    float r[9];
    float rinv[9];
    float scale;
};


template <class P>
class CV_EXPORTS WarperBase : public Warper
{   
public:
    virtual Point warp(const Mat &src, float focal, const Mat &R, Mat &dst,
                           int interp_mode, int border_mode);

    virtual Rect warpRoi(const Size &sz, float focal, const Mat &R);

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
    float plane_dist;
};


// Projects image onto z = plane_dist plane
class CV_EXPORTS PlaneWarper : public WarperBase<PlaneProjector>
{
public:
    PlaneWarper(float plane_dist = 1.f, float scale = 1.f)
    {
        projector_.plane_dist = plane_dist;
        projector_.scale = scale;
    }

protected:
    void detectResultRoi(Point &dst_tl, Point &dst_br);
};


class CV_EXPORTS PlaneWarperGpu : public PlaneWarper
{
public:
    PlaneWarperGpu(float plane_dist = 1.f, float scale = 1.f) : PlaneWarper(plane_dist, scale) {}
    Point warp(const Mat &src, float focal, const Mat &R, Mat &dst,
                   int interp_mode, int border_mode);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_dst_, d_src_;
};


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
    SphericalWarper(float scale = 300.f) { projector_.scale = scale; }

protected:
    void detectResultRoi(Point &dst_tl, Point &dst_br);
};


class CV_EXPORTS SphericalWarperGpu : public SphericalWarper
{
public:
    SphericalWarperGpu(float scale = 300.f) : SphericalWarper(scale) {}
    Point warp(const Mat &src, float focal, const Mat &R, Mat &dst,
                   int interp_mode, int border_mode);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_dst_, d_src_;
};


struct CV_EXPORTS CylindricalProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto x * x + z * z = 1 cylinder
class CV_EXPORTS CylindricalWarper : public WarperBase<CylindricalProjector>
{
public:
    CylindricalWarper(float scale = 300.f) { projector_.scale = scale; }

protected:
    void detectResultRoi(Point &dst_tl, Point &dst_br)
    {
        WarperBase<CylindricalProjector>::detectResultRoiByBorder(dst_tl, dst_br);
    }
};


class CV_EXPORTS CylindricalWarperGpu : public CylindricalWarper
{
public:
    CylindricalWarperGpu(float scale = 300.f) : CylindricalWarper(scale) {}
    Point warp(const Mat &src, float focal, const Mat &R, Mat &dst,
                   int interp_mode, int border_mode);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_dst_, d_src_;
};

} // namespace detail
} // namespace cv

#include "warpers_inl.hpp"

#endif // __OPENCV_STITCHING_WARPERS_HPP__
