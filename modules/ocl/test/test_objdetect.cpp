/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//		Yao Wang, bitwangyaoyao@gmail.com
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "test_precomp.hpp"
#include "opencv2/objdetect.hpp"

using namespace cv;
using namespace testing;

///////////////////// HOG /////////////////////////////
PARAM_TEST_CASE(HOG, Size, int)
{
    Size winSize;
    int type;
    Mat img_rgb;
    virtual void SetUp()
    {
        winSize = GET_PARAM(0);
        type = GET_PARAM(1);
        img_rgb = readImage("gpu/hog/road.png");
        ASSERT_FALSE(img_rgb.empty());
    }
};

OCL_TEST_P(HOG, GetDescriptors)
{
    // Convert image
    Mat img;
    switch (type)
    {
    case CV_8UC1:
        cvtColor(img_rgb, img, COLOR_BGR2GRAY);
        break;
    case CV_8UC4:
    default:
        cvtColor(img_rgb, img, COLOR_BGR2BGRA);
        break;
    }
    ocl::oclMat d_img(img);

    // HOGs
    ocl::HOGDescriptor ocl_hog;
    ocl_hog.gamma_correction = true;
    HOGDescriptor hog;
    hog.gammaCorrection = true;

    // Compute descriptor
    ocl::oclMat d_descriptors;
    ocl_hog.getDescriptors(d_img, ocl_hog.win_size, d_descriptors, ocl_hog.DESCR_FORMAT_COL_BY_COL);
    Mat down_descriptors;
    d_descriptors.download(down_descriptors);
    down_descriptors = down_descriptors.reshape(0, down_descriptors.cols * down_descriptors.rows);

    hog.setSVMDetector(hog.getDefaultPeopleDetector());
    std::vector<float> descriptors;
    switch (type)
    {
    case CV_8UC1:
        hog.compute(img, descriptors, ocl_hog.win_size);
        break;
    case CV_8UC4:
    default:
        hog.compute(img_rgb, descriptors, ocl_hog.win_size);
        break;
    }
    Mat cpu_descriptors(descriptors);

    EXPECT_MAT_SIMILAR(down_descriptors, cpu_descriptors, 1e-2);
}

OCL_TEST_P(HOG, Detect)
{
    // Convert image
    Mat img;
    switch (type)
    {
    case CV_8UC1:
        cvtColor(img_rgb, img, COLOR_BGR2GRAY);
        break;
    case CV_8UC4:
    default:
        cvtColor(img_rgb, img, COLOR_BGR2BGRA);
        break;
    }
    ocl::oclMat d_img(img);

    // HOGs
    if ((winSize != Size(48, 96)) && (winSize != Size(64, 128)))
        winSize = Size(64, 128);
    ocl::HOGDescriptor ocl_hog(winSize);
    ocl_hog.gamma_correction = true;

    HOGDescriptor hog;
    hog.winSize = winSize;
    hog.gammaCorrection = true;

    if (winSize.width == 48 && winSize.height == 96)
    {
        // daimler's base
        ocl_hog.setSVMDetector(hog.getDaimlerPeopleDetector());
        hog.setSVMDetector(hog.getDaimlerPeopleDetector());
    }
    else if (winSize.width == 64 && winSize.height == 128)
    {
        ocl_hog.setSVMDetector(hog.getDefaultPeopleDetector());
        hog.setSVMDetector(hog.getDefaultPeopleDetector());
    }
    else
    {
        ocl_hog.setSVMDetector(hog.getDefaultPeopleDetector());
        hog.setSVMDetector(hog.getDefaultPeopleDetector());
    }

    // OpenCL detection
    std::vector<Rect> d_found;
    ocl_hog.detectMultiScale(d_img, d_found, 0, Size(8, 8), Size(0, 0), 1.05, 6);

    // CPU detection
    std::vector<Rect> found;
    switch (type)
    {
    case CV_8UC1:
        hog.detectMultiScale(img, found, 0, Size(8, 8), Size(0, 0), 1.05, 6);
        break;
    case CV_8UC4:
    default:
        hog.detectMultiScale(img_rgb, found, 0, Size(8, 8), Size(0, 0), 1.05, 6);
        break;
    }

    EXPECT_LT(checkRectSimilarity(img.size(), found, d_found), 1.0);
}


INSTANTIATE_TEST_CASE_P(OCL_ObjDetect, HOG, testing::Combine(
                            testing::Values(Size(64, 128), Size(48, 96)),
                            testing::Values(MatType(CV_8UC1), MatType(CV_8UC4))));


///////////////////////////// Haar //////////////////////////////
IMPLEMENT_PARAM_CLASS(CascadeName, std::string);
CascadeName cascade_frontalface_alt(std::string("haarcascade_frontalface_alt.xml"));
CascadeName cascade_frontalface_alt2(std::string("haarcascade_frontalface_alt2.xml"));

PARAM_TEST_CASE(Haar, int, CascadeName)
{
    ocl::OclCascadeClassifier cascade, nestedCascade;
    CascadeClassifier cpucascade, cpunestedCascade;

    int flags;
    std::string cascadeName;
    std::vector<Rect> faces, oclfaces;
    Mat img;
    ocl::oclMat d_img;

    virtual void SetUp()
    {
        flags = GET_PARAM(0);
        cascadeName = (std::string(cvtest::TS::ptr()->get_data_path()) + "cv/cascadeandhog/cascades/").append(GET_PARAM(1));
        ASSERT_TRUE(cascade.load( cascadeName ));
        ASSERT_TRUE(cpucascade.load(cascadeName));
        img = readImage("cv/shared/lena.png", IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty());
        equalizeHist(img, img);
        d_img.upload(img);
    }
};

OCL_TEST_P(Haar, FaceDetect)
{
    cascade.detectMultiScale(d_img, oclfaces,  1.1, 3,
                                flags,
                                Size(30, 30), Size(0, 0));

    cpucascade.detectMultiScale(img, faces,  1.1, 3,
                                flags,
                                Size(30, 30), Size(0, 0));

    EXPECT_LT(checkRectSimilarity(img.size(), faces, oclfaces), 1.0);
}

INSTANTIATE_TEST_CASE_P(OCL_ObjDetect, Haar,
    Combine(Values((int)CASCADE_SCALE_IMAGE, 0),
            Values(cascade_frontalface_alt, cascade_frontalface_alt2)));
