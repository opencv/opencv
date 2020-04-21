///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan,jlyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Zailong Wu, bullet@yeah.net
//    Yao Wang, bitwangyaoyao@gmail.com
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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////////////// HOG /////////////////////////////
PARAM_TEST_CASE(HOG, Size, MatType)
{
    Size winSize;
    int type;
    Mat img;
    UMat uimg;
    virtual void SetUp()
    {
        winSize = GET_PARAM(0);
        type = GET_PARAM(1);
        img = readImage("cascadeandhog/images/image_00000000_0.png", IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty());
        img.copyTo(uimg);
    }
};

OCL_TEST_P(HOG, GetDescriptors)
{
    HOGDescriptor hog;
    hog.gammaCorrection = true;

    hog.setSVMDetector(hog.getDefaultPeopleDetector());

    std::vector<float> cpu_descriptors;
    std::vector<float> gpu_descriptors;

    OCL_OFF(hog.compute(img, cpu_descriptors, hog.winSize));
    OCL_ON(hog.compute(uimg, gpu_descriptors, hog.winSize));

    Mat cpu_desc(cpu_descriptors), gpu_desc(gpu_descriptors);

    EXPECT_MAT_SIMILAR(cpu_desc, gpu_desc, 1e-1);
}

OCL_TEST_P(HOG, Detect)
{
    HOGDescriptor hog;
    hog.winSize = winSize;
    hog.gammaCorrection = true;

    if (winSize.width == 48 && winSize.height == 96)
        hog.setSVMDetector(hog.getDaimlerPeopleDetector());
    else
        hog.setSVMDetector(hog.getDefaultPeopleDetector());

    std::vector<Rect> cpu_found;
    std::vector<Rect> gpu_found;

    OCL_OFF(hog.detectMultiScale(img, cpu_found, 0, Size(8, 8), Size(0, 0), 1.05, 6));
    OCL_ON(hog.detectMultiScale(uimg, gpu_found, 0, Size(8, 8), Size(0, 0), 1.05, 6));

    EXPECT_LT(checkRectSimilarity(img.size(), cpu_found, gpu_found), 0.05);
}

INSTANTIATE_TEST_CASE_P(OCL_ObjDetect, HOG, testing::Combine(
                            testing::Values(Size(64, 128), Size(48, 96)),
                            testing::Values( MatType(CV_8UC1) ) ) );

}} // namespace
#endif
