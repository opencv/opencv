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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../perf_precomp.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////////////////// Stitching Warpers ///////////////////////////

enum
{
    SphericalWarperType = 0,
    CylindricalWarperType = 1,
    PlaneWarperType = 2
};

class WarperBase
{
public:
    explicit WarperBase(int type, Size srcSize)
    {
        Ptr<WarperCreator> creator;
        if (type == SphericalWarperType)
            creator = makePtr<SphericalWarper>();
        else if (type == CylindricalWarperType)
            creator = makePtr<CylindricalWarper>();
        else if (type == PlaneWarperType)
            creator = makePtr<PlaneWarper>();
        CV_Assert(!creator.empty());

        K = Mat::eye(3, 3, CV_32FC1);
        K.at<float>(0,0) = (float)srcSize.width;
        K.at<float>(0,2) = (float)srcSize.width/2;
        K.at<float>(1,1) = (float)srcSize.height;
        K.at<float>(1,2) = (float)srcSize.height/2;
        K.at<float>(2,2) = 1.0f;
        R = Mat::eye(3, 3, CV_32FC1);
        float scale = (float)srcSize.width;

        warper = creator->create(scale);
    }

    Rect buildMaps(Size src_size, OutputArray xmap, OutputArray ymap) const
    {
        return warper->buildMaps(src_size, K, R, xmap, ymap);
    }

    Point warp(InputArray src, int interp_mode, int border_mode, OutputArray dst) const
    {
        return warper->warp(src, K, R, interp_mode, border_mode, dst);
    }

private:
    Ptr<detail::RotationWarper> warper;
    Mat K, R;
};

CV_ENUM(WarperType, SphericalWarperType, CylindricalWarperType, PlaneWarperType)

typedef tuple<Size, WarperType> StitchingWarpersParams;
typedef TestBaseWithParam<StitchingWarpersParams> StitchingWarpersFixture;

static void prepareWarperSrc(InputOutputArray src, Size srcSize)
{
    src.create(srcSize, CV_8UC1);
    src.setTo(Scalar::all(64));
    ellipse(src, Point(srcSize.width/2, srcSize.height/2), Size(srcSize.width/2, srcSize.height/2),
            360, 0, 360, Scalar::all(255), 2);
    ellipse(src, Point(srcSize.width/2, srcSize.height/2), Size(srcSize.width/3, srcSize.height/3),
            360, 0, 360, Scalar::all(128), 2);
    rectangle(src, Point(10, 10), Point(srcSize.width - 10, srcSize.height - 10), Scalar::all(128), 2);
}

OCL_PERF_TEST_P(StitchingWarpersFixture, StitchingWarpers_BuildMaps,
                ::testing::Combine(OCL_TEST_SIZES, WarperType::all()))
{
    const StitchingWarpersParams params = GetParam();
    const Size srcSize = get<0>(params);
    const WarperBase warper(get<1>(params), srcSize);

    UMat xmap, ymap;

    OCL_TEST_CYCLE() warper.buildMaps(srcSize, xmap, ymap);

    SANITY_CHECK(xmap, 1e-3);
    SANITY_CHECK(ymap, 1e-3);
}

OCL_PERF_TEST_P(StitchingWarpersFixture, StitchingWarpers_Warp,
                ::testing::Combine(OCL_TEST_SIZES, WarperType::all()))
{
    const StitchingWarpersParams params = GetParam();
    const Size srcSize = get<0>(params);
    const WarperBase warper(get<1>(params), srcSize);

    UMat src, dst;
    prepareWarperSrc(src, srcSize);
    declare.in(src, WARMUP_READ);

    OCL_TEST_CYCLE() warper.warp(src, INTER_LINEAR, BORDER_REPLICATE, dst);

#if 0
    namedWindow("src", WINDOW_NORMAL);
    namedWindow("dst", WINDOW_NORMAL);
    imshow("src", src);
    imshow("dst", dst);
    std::cout << dst.size() << " " << dst.size().area() << std::endl;
    cv::waitKey();
#endif

    SANITY_CHECK(dst, 1e-5);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
