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

#ifndef OPENCV_STITCHING_WARPER_CREATORS_HPP
#define OPENCV_STITCHING_WARPER_CREATORS_HPP

#include "opencv2/stitching/detail/warpers.hpp"
#include <string>

namespace cv {
    class CV_EXPORTS_W PyRotationWarper
    {
        Ptr<detail::RotationWarper> rw;

    public:
        CV_WRAP PyRotationWarper(String type, float scale);
        CV_WRAP PyRotationWarper() {}
        ~PyRotationWarper() {}

        /** @brief Projects the image point.

        @param pt Source point
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @return Projected point
        */
        CV_WRAP Point2f warpPoint(const Point2f &pt, InputArray K, InputArray R);

        /** @brief Projects the image point backward.

        @param pt Projected point
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @return Backward-projected point
        */
        CV_WRAP Point2f warpPointBackward(const Point2f &pt, InputArray K, InputArray R);

        /** @brief Builds the projection maps according to the given camera data.

        @param src_size Source image size
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param xmap Projection map for the x axis
        @param ymap Projection map for the y axis
        @return Projected image minimum bounding box
        */
        CV_WRAP Rect buildMaps(Size src_size, InputArray K, InputArray R, OutputArray xmap, OutputArray ymap);

        /** @brief Projects the image.

        @param src Source image
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param interp_mode Interpolation mode
        @param border_mode Border extrapolation mode
        @param dst Projected image
        @return Project image top-left corner
        */
        CV_WRAP Point warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode,
            CV_OUT OutputArray dst);

        /** @brief Projects the image backward.

        @param src Projected image
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param interp_mode Interpolation mode
        @param border_mode Border extrapolation mode
        @param dst_size Backward-projected image size
        @param dst Backward-projected image
        */
        CV_WRAP void warpBackward(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode,
            Size dst_size, CV_OUT OutputArray dst);

        /**
        @param src_size Source image bounding box
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @return Projected image minimum bounding box
        */
        CV_WRAP Rect warpRoi(Size src_size, InputArray K, InputArray R);

        CV_WRAP float getScale() const { return 1.f; }
        CV_WRAP void setScale(float) {}
    };

//! @addtogroup stitching_warp
//! @{

/** @brief Image warper factories base class.
 */

class CV_EXPORTS_W WarperCreator
{
public:
    CV_WRAP virtual ~WarperCreator() {}
    virtual Ptr<detail::RotationWarper> create(float scale) const = 0;
};


/** @brief Plane warper factory class.
  @sa detail::PlaneWarper
 */
class CV_EXPORTS  PlaneWarper : public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::PlaneWarper>(scale); }
};

/** @brief Affine warper factory class.
  @sa detail::AffineWarper
 */
class CV_EXPORTS  AffineWarper : public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::AffineWarper>(scale); }
};

/** @brief Cylindrical warper factory class.
@sa detail::CylindricalWarper
*/
class CV_EXPORTS CylindricalWarper: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::CylindricalWarper>(scale); }
};

/** @brief Spherical warper factory class */
class CV_EXPORTS SphericalWarper: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::SphericalWarper>(scale); }
};

class CV_EXPORTS FisheyeWarper : public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::FisheyeWarper>(scale); }
};

class CV_EXPORTS StereographicWarper: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::StereographicWarper>(scale); }
};

class CV_EXPORTS CompressedRectilinearWarper: public WarperCreator
{
    float a, b;
public:
    CompressedRectilinearWarper(float A = 1, float B = 1)
    {
        a = A; b = B;
    }
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::CompressedRectilinearWarper>(scale, a, b); }
};

class CV_EXPORTS CompressedRectilinearPortraitWarper: public WarperCreator
{
    float a, b;
public:
    CompressedRectilinearPortraitWarper(float A = 1, float B = 1)
    {
        a = A; b = B;
    }
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::CompressedRectilinearPortraitWarper>(scale, a, b); }
};

class CV_EXPORTS PaniniWarper: public WarperCreator
{
    float a, b;
public:
    PaniniWarper(float A = 1, float B = 1)
    {
        a = A; b = B;
    }
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::PaniniWarper>(scale, a, b); }
};

class CV_EXPORTS PaniniPortraitWarper: public WarperCreator
{
    float a, b;
public:
    PaniniPortraitWarper(float A = 1, float B = 1)
    {
        a = A; b = B;
    }
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::PaniniPortraitWarper>(scale, a, b); }
};

class CV_EXPORTS MercatorWarper: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::MercatorWarper>(scale); }
};

class CV_EXPORTS TransverseMercatorWarper: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::TransverseMercatorWarper>(scale); }
};



#ifdef HAVE_OPENCV_CUDAWARPING
class PlaneWarperGpu: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::PlaneWarperGpu>(scale); }
};


class CylindricalWarperGpu: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::CylindricalWarperGpu>(scale); }
};


class SphericalWarperGpu: public WarperCreator
{
public:
    Ptr<detail::RotationWarper> create(float scale) const CV_OVERRIDE { return makePtr<detail::SphericalWarperGpu>(scale); }
};
#endif

//! @} stitching_warp

} // namespace cv

#endif // OPENCV_STITCHING_WARPER_CREATORS_HPP
