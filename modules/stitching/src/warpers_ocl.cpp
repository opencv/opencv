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

#include "precomp.hpp"
#include "opencl_kernels.hpp"

namespace cv {
namespace detail {

/////////////////////////////////////////// PlaneWarperOcl ////////////////////////////////////////////

Rect PlaneWarperOcl::buildMaps(Size src_size, InputArray K, InputArray R, InputArray T, OutputArray xmap, OutputArray ymap)
{
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    if (ocl::useOpenCL())
    {
        ocl::Kernel k("buildWarpPlaneMaps", ocl::stitching::warpers_oclsrc);
        if (!k.empty())
        {
            Size dsize(dst_br.x - dst_tl.x + 1, dst_br.y - dst_tl.y + 1);
            xmap.create(dsize, CV_32FC1);
            ymap.create(dsize, CV_32FC1);

            Mat r_kinv(1, 9, CV_32FC1, projector_.r_kinv), t(1, 3, CV_32FC1, projector_.t);
            UMat uxmap = xmap.getUMat(), uymap = ymap.getUMat(),
                    ur_kinv = r_kinv.getUMat(ACCESS_READ), ut = t.getUMat(ACCESS_READ);

            k.args(ocl::KernelArg::WriteOnlyNoSize(uxmap), ocl::KernelArg::WriteOnly(uymap),
                   ocl::KernelArg::PtrReadOnly(ur_kinv), ocl::KernelArg::PtrReadOnly(ut),
                   dst_tl.x, dst_tl.y, projector_.scale);

            size_t globalsize[2] = { dsize.width, dsize.height };
            if (k.run(2, globalsize, NULL, true))
                return Rect(dst_tl, dst_br);
        }
    }

    return PlaneWarper::buildMaps(src_size, K, R, T, xmap, ymap);
}

Point PlaneWarperOcl::warp(InputArray src, InputArray K, InputArray R, InputArray T, int interp_mode, int border_mode, OutputArray dst)
{
    UMat uxmap, uymap;
    Rect dst_roi = buildMaps(src.size(), K, R, T, uxmap, uymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    UMat udst = dst.getUMat();
    remap(src, udst, uxmap, uymap, interp_mode, border_mode);

    return dst_roi.tl();
}

/////////////////////////////////////////// SphericalWarperOcl ////////////////////////////////////////

Rect SphericalWarperOcl::buildMaps(Size src_size, InputArray K, InputArray R, OutputArray xmap, OutputArray ymap)
{
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    if (ocl::useOpenCL())
    {
        ocl::Kernel k("buildWarpSphericalMaps", ocl::stitching::warpers_oclsrc);
        if (!k.empty())
        {
            Size dsize(dst_br.x - dst_tl.x + 1, dst_br.y - dst_tl.y + 1);
            xmap.create(dsize, CV_32FC1);
            ymap.create(dsize, CV_32FC1);

            Mat r_kinv(1, 9, CV_32FC1, projector_.r_kinv);
            UMat uxmap = xmap.getUMat(), uymap = ymap.getUMat(), ur_kinv = r_kinv.getUMat(ACCESS_READ);

            k.args(ocl::KernelArg::WriteOnlyNoSize(uxmap), ocl::KernelArg::WriteOnly(uymap),
                   ocl::KernelArg::PtrReadOnly(ur_kinv), dst_tl.x, dst_tl.y, projector_.scale);

            size_t globalsize[2] = { dsize.width, dsize.height };
            if (k.run(2, globalsize, NULL, true))
                return Rect(dst_tl, dst_br);
        }
    }

    return SphericalWarper::buildMaps(src_size, K, R, xmap, ymap);
}

Point SphericalWarperOcl::warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode, OutputArray dst)
{
    UMat uxmap, uymap;
    Rect dst_roi = buildMaps(src.size(), K, R, uxmap, uymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    UMat udst = dst.getUMat();
    remap(src, udst, uxmap, uymap, interp_mode, border_mode);

    return dst_roi.tl();
}

/////////////////////////////////////////// CylindricalWarperOcl ////////////////////////////////////////

Rect CylindricalWarperOcl::buildMaps(Size src_size, InputArray K, InputArray R, OutputArray xmap, OutputArray ymap)
{
    projector_.setCameraParams(K, R);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    if (ocl::useOpenCL())
    {
        ocl::Kernel k("buildWarpCylindricalMaps", ocl::stitching::warpers_oclsrc);
        if (!k.empty())
        {
            Size dsize(dst_br.x - dst_tl.x + 1, dst_br.y - dst_tl.y + 1);
            xmap.create(dsize, CV_32FC1);
            ymap.create(dsize, CV_32FC1);

            Mat r_kinv(1, 9, CV_32FC1, projector_.r_kinv);
            UMat uxmap = xmap.getUMat(), uymap = ymap.getUMat(), ur_kinv = r_kinv.getUMat(ACCESS_READ);

            k.args(ocl::KernelArg::WriteOnlyNoSize(uxmap), ocl::KernelArg::WriteOnly(uymap),
                   ocl::KernelArg::PtrReadOnly(ur_kinv), dst_tl.x, dst_tl.y, projector_.scale);

            size_t globalsize[2] = { dsize.width, dsize.height };
            if (k.run(2, globalsize, NULL, true))
                return Rect(dst_tl, dst_br);
        }
    }

    return CylindricalWarper::buildMaps(src_size, K, R, xmap, ymap);
}

Point CylindricalWarperOcl::warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode, OutputArray dst)
{
    UMat uxmap, uymap;
    Rect dst_roi = buildMaps(src.size(), K, R, uxmap, uymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    UMat udst = dst.getUMat();
    remap(src, udst, uxmap, uymap, interp_mode, border_mode);

    return dst_roi.tl();
}

} // namespace detail
} // namespace cv
