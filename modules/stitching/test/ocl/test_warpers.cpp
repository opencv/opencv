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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"
#include "opencv2/stitching/warpers.hpp"

#ifdef HAVE_OPENCL

/*
Total:
OCL_SphericalWarperTest:   0.0001 vs 0.000751495(M2), 0.000751495(M1), 0.000396729(macos-x64), 0.000473022(win64), 0.000488281(default)
OCL_CylindricalWarperTest: 0.0001 vs 0.000640869(M2), 0.000640869(M1), 0.000320435(macos-x64), 0.000328064(win64), 0.000312805(default)
OCL_PlaneWarperTest:       0.00015 vs 0.000640869(macos-x64), 0.000656128(win64), 0.000656128(default)
OCL_AffineWarperTest:      0.00015 vs 0.000442505(macos-x64), 0.001297(win64), 0.001297(default)
*/

namespace opencv_test {
namespace ocl {

struct WarperTestBase :
        public Test, public TestUtils
{
    Mat src, dst, xmap, ymap;
    UMat usrc, udst, uxmap, uymap;
    Mat K, R;

    void generateTestData()
    {
        Size size = randomSize(1, MAX_VALUE);

        src = randomMat(size, CV_32FC1, -500, 500);
        src.copyTo(usrc);

        K = Mat::eye(3, 3, CV_32FC1);
        float angle = (float)(30.0 * CV_PI / 180.0);
        float rotationMatrix[9] = {
                (float)cos(angle), (float)sin(angle), 0,
                (float)-sin(angle), (float)cos(angle), 0,
                0, 0, 1
        };
        Mat(3, 3, CV_32FC1, rotationMatrix).copyTo(R);
    }

    void Near(double threshold = 0.)
    {
        EXPECT_MAT_NEAR(xmap, uxmap, threshold);
        EXPECT_MAT_NEAR(ymap, uymap, threshold);
        EXPECT_MAT_NEAR(dst, udst, threshold);
    }
};

typedef WarperTestBase SphericalWarperTest;

OCL_TEST_F(SphericalWarperTest, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Ptr<WarperCreator> creator = makePtr<SphericalWarper>();
        Ptr<detail::RotationWarper> warper = creator->create(2.0);

        OCL_OFF(warper->buildMaps(src.size(), K, R, xmap, ymap));
        OCL_ON(warper->buildMaps(usrc.size(), K, R, uxmap, uymap));

        OCL_OFF(warper->warp(src, K, R, INTER_LINEAR, BORDER_REPLICATE, dst));
        OCL_ON(warper->warp(usrc, K, R, INTER_LINEAR, BORDER_REPLICATE, udst));

        Near(7.56e-4);
    }
}

typedef WarperTestBase CylindricalWarperTest;

OCL_TEST_F(CylindricalWarperTest, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Ptr<WarperCreator> creator = makePtr<CylindricalWarper>();
        Ptr<detail::RotationWarper> warper = creator->create(2.0);

        OCL_OFF(warper->buildMaps(src.size(), K, R, xmap, ymap));
        OCL_ON(warper->buildMaps(usrc.size(), K, R, uxmap, uymap));

        OCL_OFF(warper->warp(src, K, R, INTER_LINEAR, BORDER_REPLICATE, dst));
        OCL_ON(warper->warp(usrc, K, R, INTER_LINEAR, BORDER_REPLICATE, udst));

        Near(6.5e-4);
    }
}

typedef WarperTestBase PlaneWarperTest;

OCL_TEST_F(PlaneWarperTest, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Ptr<WarperCreator> creator = makePtr<PlaneWarper>();
        Ptr<detail::RotationWarper> warper = creator->create(2.0);

        OCL_OFF(warper->buildMaps(src.size(), K, R, xmap, ymap));
        OCL_ON(warper->buildMaps(usrc.size(), K, R, uxmap, uymap));

        OCL_OFF(warper->warp(src, K, R, INTER_LINEAR, BORDER_REPLICATE, dst));
        OCL_ON(warper->warp(usrc, K, R, INTER_LINEAR, BORDER_REPLICATE, udst));

        Near(6.6e-4);
    }
}

typedef WarperTestBase AffineWarperTest;

OCL_TEST_F(AffineWarperTest, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Ptr<WarperCreator> creator = makePtr<AffineWarper>();
        Ptr<detail::RotationWarper> warper = creator->create(1.0);

        OCL_OFF(warper->buildMaps(src.size(), K, R, xmap, ymap));
        OCL_ON(warper->buildMaps(usrc.size(), K, R, uxmap, uymap));

        OCL_OFF(warper->warp(src, K, R, INTER_LINEAR, BORDER_REPLICATE, dst));
        OCL_ON(warper->warp(usrc, K, R, INTER_LINEAR, BORDER_REPLICATE, udst));

        Near(1.3e-3);
    }
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
