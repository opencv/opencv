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
#include "opencv2/core/gpumat.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace cv {
namespace detail {

class CV_EXPORTS RotationWarper
{
public:
    virtual ~RotationWarper() {}

    virtual Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R) = 0;

    virtual Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap) = 0;

    virtual Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                       Mat &dst) = 0;

    virtual void warpBackward(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                              Size dst_size, Mat &dst) = 0;

    virtual Rect warpRoi(Size src_size, const Mat &K, const Mat &R) = 0;

    float getScale() const { return 1.f; }
    void setScale(float) {}
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
class CV_EXPORTS RotationWarperBase : public RotationWarper
{
public:
    Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R);

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap);

    Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
               Mat &dst);

    void warpBackward(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                      Size dst_size, Mat &dst);

    Rect warpRoi(Size src_size, const Mat &K, const Mat &R);

    float getScale() const { return projector_.scale; }
    void setScale(float val) { projector_.scale = val; }

protected:

    // Detects ROI of the destination image. It's correct for any projection.
    virtual void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);

    // Detects ROI of the destination image by walking over image border.
    // Correctness for any projection isn't guaranteed.
    void detectResultRoiByBorder(Size src_size, Point &dst_tl, Point &dst_br);

    P projector_;
};


struct CV_EXPORTS PlaneProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS PlaneWarper : public RotationWarperBase<PlaneProjector>
{
public:
    PlaneWarper(float scale = 1.f) { projector_.scale = scale; }

    void setScale(float scale) { projector_.scale = scale; }

    Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R, const Mat &T);

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, const Mat &T, Mat &xmap, Mat &ymap);

    Point warp(const Mat &src, const Mat &K, const Mat &R, const Mat &T, int interp_mode, int border_mode,
               Mat &dst);

    Rect warpRoi(Size src_size, const Mat &K, const Mat &R, const Mat &T);

protected:
    void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);
};


struct CV_EXPORTS SphericalProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto unit sphere with origin at (0, 0, 0).
// Poles are located at (0, -1, 0) and (0, 1, 0) points.
class CV_EXPORTS SphericalWarper : public RotationWarperBase<SphericalProjector>
{
public:
    SphericalWarper(float scale) { projector_.scale = scale; }

protected:
    void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);
};


struct CV_EXPORTS CylindricalProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto x * x + z * z = 1 cylinder
class CV_EXPORTS CylindricalWarper : public RotationWarperBase<CylindricalProjector>
{
public:
    CylindricalWarper(float scale) { projector_.scale = scale; }

protected:
    void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
    {
        RotationWarperBase<CylindricalProjector>::detectResultRoiByBorder(src_size, dst_tl, dst_br);
    }
};


struct CV_EXPORTS FisheyeProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS FisheyeWarper : public RotationWarperBase<FisheyeProjector>
{
public:
    FisheyeWarper(float scale) { projector_.scale = scale; }
};


struct CV_EXPORTS StereographicProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS StereographicWarper : public RotationWarperBase<StereographicProjector>
{
public:
    StereographicWarper(float scale) { projector_.scale = scale; }
};


struct CV_EXPORTS CompressedRectilinearProjector : ProjectorBase
{
    float a, b;

    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS CompressedRectilinearWarper : public RotationWarperBase<CompressedRectilinearProjector>
{
public:
    CompressedRectilinearWarper(float scale, float A = 1, float B = 1)
    {
        projector_.a = A;
        projector_.b = B;
        projector_.scale = scale;
    }
};


struct CV_EXPORTS CompressedRectilinearPortraitProjector : ProjectorBase
{
    float a, b;

    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS CompressedRectilinearPortraitWarper : public RotationWarperBase<CompressedRectilinearPortraitProjector>
{
public:
   CompressedRectilinearPortraitWarper(float scale, float A = 1, float B = 1)
   {
       projector_.a = A;
       projector_.b = B;
       projector_.scale = scale;
   }
};


struct CV_EXPORTS PaniniProjector : ProjectorBase
{
    float a, b;

    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS PaniniWarper : public RotationWarperBase<PaniniProjector>
{
public:
   PaniniWarper(float scale, float A = 1, float B = 1)
   {
       projector_.a = A;
       projector_.b = B;
       projector_.scale = scale;
   }
};


struct CV_EXPORTS PaniniPortraitProjector : ProjectorBase
{
    float a, b;

    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS PaniniPortraitWarper : public RotationWarperBase<PaniniPortraitProjector>
{
public:
   PaniniPortraitWarper(float scale, float A = 1, float B = 1)
   {
       projector_.a = A;
       projector_.b = B;
       projector_.scale = scale;
   }

};


struct CV_EXPORTS MercatorProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS MercatorWarper : public RotationWarperBase<MercatorProjector>
{
public:
    MercatorWarper(float scale) { projector_.scale = scale; }
};


struct CV_EXPORTS TransverseMercatorProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS TransverseMercatorWarper : public RotationWarperBase<TransverseMercatorProjector>
{
public:
    TransverseMercatorWarper(float scale) { projector_.scale = scale; }
};


class CV_EXPORTS PlaneWarperGpu : public PlaneWarper
{
public:
    PlaneWarperGpu(float scale = 1.f) : PlaneWarper(scale) {}

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)
    {
        Rect result = buildMaps(src_size, K, R, d_xmap_, d_ymap_);
        d_xmap_.download(xmap);
        d_ymap_.download(ymap);
        return result;
    }

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, const Mat &T, Mat &xmap, Mat &ymap)
    {
        Rect result = buildMaps(src_size, K, R, T, d_xmap_, d_ymap_);
        d_xmap_.download(xmap);
        d_ymap_.download(ymap);
        return result;
    }

    Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
               Mat &dst)
    {
        d_src_.upload(src);
        Point result = warp(d_src_, K, R, interp_mode, border_mode, d_dst_);
        d_dst_.download(dst);
        return result;
    }

    Point warp(const Mat &src, const Mat &K, const Mat &R, const Mat &T, int interp_mode, int border_mode,
               Mat &dst)
    {
        d_src_.upload(src);
        Point result = warp(d_src_, K, R, T, interp_mode, border_mode, d_dst_);
        d_dst_.download(dst);
        return result;
    }

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, gpu::GpuMat &xmap, gpu::GpuMat &ymap);

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, const Mat &T, gpu::GpuMat &xmap, gpu::GpuMat &ymap);

    Point warp(const gpu::GpuMat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
               gpu::GpuMat &dst);

    Point warp(const gpu::GpuMat &src, const Mat &K, const Mat &R, const Mat &T, int interp_mode, int border_mode,
               gpu::GpuMat &dst);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_src_, d_dst_;
};


class CV_EXPORTS SphericalWarperGpu : public SphericalWarper
{
public:
    SphericalWarperGpu(float scale) : SphericalWarper(scale) {}

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)
    {
        Rect result = buildMaps(src_size, K, R, d_xmap_, d_ymap_);
        d_xmap_.download(xmap);
        d_ymap_.download(ymap);
        return result;
    }

    Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
               Mat &dst)
    {
        d_src_.upload(src);
        Point result = warp(d_src_, K, R, interp_mode, border_mode, d_dst_);
        d_dst_.download(dst);
        return result;
    }

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, gpu::GpuMat &xmap, gpu::GpuMat &ymap);

    Point warp(const gpu::GpuMat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
               gpu::GpuMat &dst);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_src_, d_dst_;
};


class CV_EXPORTS CylindricalWarperGpu : public CylindricalWarper
{
public:
    CylindricalWarperGpu(float scale) : CylindricalWarper(scale) {}

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)
    {
        Rect result = buildMaps(src_size, K, R, d_xmap_, d_ymap_);
        d_xmap_.download(xmap);
        d_ymap_.download(ymap);
        return result;
    }

    Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
               Mat &dst)
    {
        d_src_.upload(src);
        Point result = warp(d_src_, K, R, interp_mode, border_mode, d_dst_);
        d_dst_.download(dst);
        return result;
    }

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, gpu::GpuMat &xmap, gpu::GpuMat &ymap);

    Point warp(const gpu::GpuMat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
               gpu::GpuMat &dst);

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_src_, d_dst_;
};


struct SphericalPortraitProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


// Projects image onto unit sphere with origin at (0, 0, 0).
// Poles are located NOT at (0, -1, 0) and (0, 1, 0) points, BUT at (1, 0, 0) and (-1, 0, 0) points.
class CV_EXPORTS SphericalPortraitWarper : public RotationWarperBase<SphericalPortraitProjector>
{
public:
    SphericalPortraitWarper(float scale) { projector_.scale = scale; }

protected:
    void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);
};

struct CylindricalPortraitProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS CylindricalPortraitWarper : public RotationWarperBase<CylindricalPortraitProjector>
{
public:
    CylindricalPortraitWarper(float scale) { projector_.scale = scale; }

protected:
    void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
    {
        RotationWarperBase<CylindricalPortraitProjector>::detectResultRoiByBorder(src_size, dst_tl, dst_br);
    }
};

struct PlanePortraitProjector : ProjectorBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};


class CV_EXPORTS PlanePortraitWarper : public RotationWarperBase<PlanePortraitProjector>
{
public:
    PlanePortraitWarper(float scale) { projector_.scale = scale; }

protected:
    void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
    {
        RotationWarperBase<PlanePortraitProjector>::detectResultRoiByBorder(src_size, dst_tl, dst_br);
    }
};

} // namespace detail
} // namespace cv

#include "warpers_inl.hpp"

#endif // __OPENCV_STITCHING_WARPERS_HPP__
