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

#include "test_precomp.hpp"
#include "opencv2/photo/denoising.hpp"
#include <string>

using namespace cv;
using namespace std;

class CV_DenoisingGrayscaleTest : public cvtest::BaseTest
{
public:
    CV_DenoisingGrayscaleTest();
    ~CV_DenoisingGrayscaleTest();
protected:
    void run(int);
};

CV_DenoisingGrayscaleTest::CV_DenoisingGrayscaleTest() {}
CV_DenoisingGrayscaleTest::~CV_DenoisingGrayscaleTest() {}

void CV_DenoisingGrayscaleTest::run( int )
{
    string folder = string(ts->get_data_path()) + "denoising/";
    Mat orig = imread(folder + "lena_noised_gaussian_sigma=10.png", 0);
    Mat exp = imread(folder + "lena_noised_denoised_grayscale_tw=7_sw=21_h=10.png", 0);

    if (orig.empty() || exp.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat res;
    fastNlMeansDenoising(orig, res, 7, 21, 10);

    if (norm(res - exp) > 0) {        
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
    } else {
        ts->set_failed_test_info(cvtest::TS::OK);
    }
}

class CV_DenoisingColoredTest : public cvtest::BaseTest
{
public:
    CV_DenoisingColoredTest();
    ~CV_DenoisingColoredTest();
protected:
    void run(int);
};

CV_DenoisingColoredTest::CV_DenoisingColoredTest() {}
CV_DenoisingColoredTest::~CV_DenoisingColoredTest() {}

void CV_DenoisingColoredTest::run( int )
{
    string folder = string(ts->get_data_path()) + "denoising/";
    Mat orig = imread(folder + "lena_noised_gaussian_sigma=10.png", 1);
    Mat exp = imread(folder + "lena_noised_denoised_lab12_tw=7_sw=21_h=10_h2=10.png", 1);

    if (orig.empty() || exp.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat res;
    fastNlMeansDenoisingColored(orig, res, 7, 21, 10, 10);

    if (norm(res - exp) > 0) {        
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
    } else {
        ts->set_failed_test_info(cvtest::TS::OK);
    }
}

class CV_DenoisingGrayscaleMultiTest : public cvtest::BaseTest
{
public:
    CV_DenoisingGrayscaleMultiTest();
    ~CV_DenoisingGrayscaleMultiTest();
protected:
    void run(int);
};

CV_DenoisingGrayscaleMultiTest::CV_DenoisingGrayscaleMultiTest() {}
CV_DenoisingGrayscaleMultiTest::~CV_DenoisingGrayscaleMultiTest() {}

void CV_DenoisingGrayscaleMultiTest::run( int )
{        
    string folder = string(ts->get_data_path()) + "denoising/";

    const int imgs_count = 3;
    vector<Mat> src_imgs(imgs_count);
    src_imgs[0] = imread(folder + "lena_noised_gaussian_sigma=20_multi_0.png", 0);
    src_imgs[1] = imread(folder + "lena_noised_gaussian_sigma=20_multi_1.png", 0);
    src_imgs[2] = imread(folder + "lena_noised_gaussian_sigma=20_multi_2.png", 0);
    
    Mat exp = imread(folder + "lena_noised_denoised_multi_tw=7_sw=21_h=15.png", 0);

    bool have_empty_src = false;
    for (int i = 0; i < imgs_count; i++) {
        have_empty_src |= src_imgs[i].empty();
    }

    if (have_empty_src || exp.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat res;
    fastNlMeansDenoisingMulti(src_imgs, imgs_count / 2, imgs_count, res, 7, 21, 15);

    if (norm(res - exp) > 0) {        
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
    } else {
        ts->set_failed_test_info(cvtest::TS::OK);
    }
}

class CV_DenoisingColoredMultiTest : public cvtest::BaseTest
{
public:
    CV_DenoisingColoredMultiTest();
    ~CV_DenoisingColoredMultiTest();
protected:
    void run(int);
};

CV_DenoisingColoredMultiTest::CV_DenoisingColoredMultiTest() {}
CV_DenoisingColoredMultiTest::~CV_DenoisingColoredMultiTest() {}

void CV_DenoisingColoredMultiTest::run( int )
{        
    string folder = string(ts->get_data_path()) + "denoising/";

    const int imgs_count = 3;
    vector<Mat> src_imgs(imgs_count);
    src_imgs[0] = imread(folder + "lena_noised_gaussian_sigma=20_multi_0.png", 1);
    src_imgs[1] = imread(folder + "lena_noised_gaussian_sigma=20_multi_1.png", 1);
    src_imgs[2] = imread(folder + "lena_noised_gaussian_sigma=20_multi_2.png", 1);
    
    Mat exp = imread(folder + "lena_noised_denoised_multi_lab12_tw=7_sw=21_h=10_h2=15.png", 1);
    
    bool have_empty_src = false;
    for (int i = 0; i < imgs_count; i++) {
        have_empty_src |= src_imgs[i].empty();
    }

    if (have_empty_src || exp.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat res;
    fastNlMeansDenoisingColoredMulti(src_imgs, imgs_count / 2, imgs_count, res, 7, 21, 10, 15);

    if (norm(res - exp) > 0) {        
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
    } else {
        ts->set_failed_test_info(cvtest::TS::OK);
    }
}


TEST(Imgproc_DenoisingGrayscale, regression) { CV_DenoisingGrayscaleTest test; test.safe_run(); }
TEST(Imgproc_DenoisingColored, regression) { CV_DenoisingColoredTest test; test.safe_run(); }
TEST(Imgproc_DenoisingGrayscaleMulti, regression) { CV_DenoisingGrayscaleMultiTest test; test.safe_run(); }
TEST(Imgproc_DenoisingColoredMulti, regression) { CV_DenoisingColoredMultiTest test; test.safe_run(); }

