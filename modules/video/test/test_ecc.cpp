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

namespace opencv_test { namespace {

class CV_ECC_BaseTest : public cvtest::BaseTest
{
public:
    CV_ECC_BaseTest();

protected:

    double computeRMS(const Mat& mat1, const Mat& mat2);
    bool isMapCorrect(const Mat& mat);


    double MAX_RMS_ECC;//upper bound for RMS error
    int ntests;//number of tests per motion type
    int ECC_iterations;//number of iterations for ECC
    double ECC_epsilon; //we choose a negative value, so that
    // ECC_iterations are always executed
};

CV_ECC_BaseTest::CV_ECC_BaseTest()
{
    MAX_RMS_ECC=0.1;
    ntests = 3;
    ECC_iterations = 50;
    ECC_epsilon = -1; //-> negative value means that ECC_Iterations will be executed
}


bool CV_ECC_BaseTest::isMapCorrect(const Mat& map)
{
    bool tr = true;
    float mapVal;
    for(int i =0; i<map.rows; i++)
        for(int j=0; j<map.cols; j++){
            mapVal = map.at<float>(i, j);
            tr = tr & (!cvIsNaN(mapVal) && (fabs(mapVal) < 1e9));
        }

    return tr;
}

double CV_ECC_BaseTest::computeRMS(const Mat& mat1, const Mat& mat2){

    CV_Assert(mat1.rows == mat2.rows);
    CV_Assert(mat1.cols == mat2.cols);

    Mat errorMat;
    subtract(mat1, mat2, errorMat);

    return sqrt(errorMat.dot(errorMat)/(mat1.rows*mat1.cols));
}


class CV_ECC_Test_Translation : public CV_ECC_BaseTest
{
public:
    CV_ECC_Test_Translation();
protected:
    void run(int);

    bool testTranslation(int);
};

CV_ECC_Test_Translation::CV_ECC_Test_Translation(){}

bool CV_ECC_Test_Translation::testTranslation(int from)
{
    Mat img = imread( string(ts->get_data_path()) + "shared/fruits.png", 0);


    if (img.empty())
    {
        ts->printf( ts->LOG, "test image can not be read");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return false;
    }
    Mat testImg;
    resize(img, testImg, Size(216, 216), 0, 0, INTER_LINEAR_EXACT);

    cv::RNG rng = ts->get_rng();

    int progress=0;

    for (int k=from; k<ntests; k++){

        ts->update_context( this, k, true );
        progress = update_progress(progress, k, ntests, 0);

        Mat translationGround = (Mat_<float>(2,3) << 1, 0, (rng.uniform(10.f, 20.f)),
            0, 1, (rng.uniform(10.f, 20.f)));

        Mat warpedImage;

        warpAffine(testImg, warpedImage, translationGround,
            Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);

        Mat mapTranslation = (Mat_<float>(2,3) << 1, 0, 0, 0, 1, 0);

        findTransformECC(warpedImage, testImg, mapTranslation, 0,
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, ECC_iterations, ECC_epsilon));

        if (!isMapCorrect(mapTranslation)){
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return false;
        }

        if (computeRMS(mapTranslation, translationGround)>MAX_RMS_ECC){
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf( ts->LOG, "RMS = %f",
                computeRMS(mapTranslation, translationGround));
            return false;
        }

    }
    return true;
}

void CV_ECC_Test_Translation::run(int from)
{

    if (!testTranslation(from))
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}



class CV_ECC_Test_Euclidean : public CV_ECC_BaseTest
{
public:
    CV_ECC_Test_Euclidean();
protected:
    void run(int);

    bool testEuclidean(int);
};

CV_ECC_Test_Euclidean::CV_ECC_Test_Euclidean() { }

bool CV_ECC_Test_Euclidean::testEuclidean(int from)
{
    Mat img = imread( string(ts->get_data_path()) + "shared/fruits.png", 0);


    if (img.empty())
    {
        ts->printf( ts->LOG, "test image can not be read");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return false;
    }
    Mat testImg;
    resize(img, testImg, Size(216, 216), 0, 0, INTER_LINEAR_EXACT);

    cv::RNG rng = ts->get_rng();

    int progress = 0;
    for (int k=from; k<ntests; k++){
        ts->update_context( this, k, true );
        progress = update_progress(progress, k, ntests, 0);

        double angle = CV_PI/30 + CV_PI*rng.uniform((double)-2.f, (double)2.f)/180;

        Mat euclideanGround = (Mat_<float>(2,3) << cos(angle), -sin(angle), (rng.uniform(10.f, 20.f)),
            sin(angle), cos(angle), (rng.uniform(10.f, 20.f)));

        Mat warpedImage;

        warpAffine(testImg, warpedImage, euclideanGround,
            Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);

        Mat mapEuclidean = (Mat_<float>(2,3) << 1, 0, 0, 0, 1, 0);

        findTransformECC(warpedImage, testImg, mapEuclidean, 1,
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, ECC_iterations, ECC_epsilon));

        if (!isMapCorrect(mapEuclidean)){
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return false;
        }

        if (computeRMS(mapEuclidean, euclideanGround)>MAX_RMS_ECC){
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf( ts->LOG, "RMS = %f",
                computeRMS(mapEuclidean, euclideanGround));
            return false;
        }

    }
    return true;
}


void CV_ECC_Test_Euclidean::run(int from)
{

    if (!testEuclidean(from))
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

class CV_ECC_Test_Affine : public CV_ECC_BaseTest
{
public:
    CV_ECC_Test_Affine();
protected:
    void run(int);

    bool testAffine(int);
};

CV_ECC_Test_Affine::CV_ECC_Test_Affine(){}


bool CV_ECC_Test_Affine::testAffine(int from)
{
    Mat img = imread( string(ts->get_data_path()) + "shared/fruits.png", 0);

    if (img.empty())
    {
        ts->printf( ts->LOG, "test image can not be read");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return false;
    }
    Mat testImg;
    resize(img, testImg, Size(216, 216), 0, 0, INTER_LINEAR_EXACT);

    cv::RNG rng = ts->get_rng();

    int progress = 0;
    for (int k=from; k<ntests; k++){
        ts->update_context( this, k, true );
        progress = update_progress(progress, k, ntests, 0);


        Mat affineGround = (Mat_<float>(2,3) << (1-rng.uniform(-0.05f, 0.05f)),
            (rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.f, 20.f)),
            (rng.uniform(-0.03f, 0.03f)), (1-rng.uniform(-0.05f, 0.05f)),
            (rng.uniform(10.f, 20.f)));

        Mat warpedImage;

        warpAffine(testImg, warpedImage, affineGround,
            Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);

        Mat mapAffine = (Mat_<float>(2,3) << 1, 0, 0, 0, 1, 0);

        findTransformECC(warpedImage, testImg, mapAffine, 2,
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, ECC_iterations, ECC_epsilon));

        if (!isMapCorrect(mapAffine)){
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return false;
        }

        if (computeRMS(mapAffine, affineGround)>MAX_RMS_ECC){
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf( ts->LOG, "RMS = %f",
                computeRMS(mapAffine, affineGround));
            return false;
        }

    }

    return true;
}


void CV_ECC_Test_Affine::run(int from)
{

    if (!testAffine(from))
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

class CV_ECC_Test_Homography : public CV_ECC_BaseTest
{
public:
    CV_ECC_Test_Homography();
protected:
    void run(int);

    bool testHomography(int);
};

CV_ECC_Test_Homography::CV_ECC_Test_Homography(){}

bool CV_ECC_Test_Homography::testHomography(int from)
{
    Mat img = imread( string(ts->get_data_path()) + "shared/fruits.png", 0);


    if (img.empty())
    {
        ts->printf( ts->LOG, "test image can not be read");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return false;
    }
    Mat testImg;
    resize(img, testImg, Size(216, 216), 0, 0, INTER_LINEAR_EXACT);

    cv::RNG rng = ts->get_rng();

    int progress = 0;
    for (int k=from; k<ntests; k++){
        ts->update_context( this, k, true );
        progress = update_progress(progress, k, ntests, 0);

        Mat homoGround = (Mat_<float>(3,3) << (1-rng.uniform(-0.05f, 0.05f)),
            (rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.f, 20.f)),
            (rng.uniform(-0.03f, 0.03f)), (1-rng.uniform(-0.05f, 0.05f)),(rng.uniform(10.f, 20.f)),
            (rng.uniform(0.0001f, 0.0003f)), (rng.uniform(0.0001f, 0.0003f)), 1.f);

        Mat warpedImage;

        warpPerspective(testImg, warpedImage, homoGround,
            Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);

        Mat mapHomography = Mat::eye(3, 3, CV_32F);

        findTransformECC(warpedImage, testImg, mapHomography, 3,
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, ECC_iterations, ECC_epsilon));

        if (!isMapCorrect(mapHomography)){
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return false;
        }

        if (computeRMS(mapHomography, homoGround)>MAX_RMS_ECC){
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf( ts->LOG, "RMS = %f",
                computeRMS(mapHomography, homoGround));
            return false;
        }

    }
    return true;
}

void CV_ECC_Test_Homography::run(int from)
{
    if (!testHomography(from))
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

class CV_ECC_Test_Mask : public CV_ECC_BaseTest
{
public:
    CV_ECC_Test_Mask();
protected:
    void run(int);

    bool testMask(int);
};

CV_ECC_Test_Mask::CV_ECC_Test_Mask(){}

bool CV_ECC_Test_Mask::testMask(int from)
{
    Mat img = imread( string(ts->get_data_path()) + "shared/fruits.png", 0);


    if (img.empty())
    {
        ts->printf( ts->LOG, "test image can not be read");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return false;
    }
    Mat scaledImage;
    resize(img, scaledImage, Size(216, 216), 0, 0, INTER_LINEAR_EXACT );

    Mat_<float> testImg;
    scaledImage.convertTo(testImg, testImg.type());

    cv::RNG rng = ts->get_rng();

    int progress=0;

    for (int k=from; k<ntests; k++){

        ts->update_context( this, k, true );
        progress = update_progress(progress, k, ntests, 0);

        Mat translationGround = (Mat_<float>(2,3) << 1, 0, (rng.uniform(10.f, 20.f)),
            0, 1, (rng.uniform(10.f, 20.f)));

        Mat warpedImage;

        warpAffine(testImg, warpedImage, translationGround,
            Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);

        Mat mapTranslation = (Mat_<float>(2,3) << 1, 0, 0, 0, 1, 0);

        Mat_<unsigned char> mask = Mat_<unsigned char>::ones(testImg.rows, testImg.cols);
        for (int i=testImg.rows*2/3; i<testImg.rows; i++) {
          for (int j=testImg.cols*2/3; j<testImg.cols; j++) {
            testImg(i, j) = 0;
            mask(i, j) = 0;
          }
        }

        findTransformECC(warpedImage, testImg, mapTranslation, 0,
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, ECC_iterations, ECC_epsilon), mask);

        if (!isMapCorrect(mapTranslation)){
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return false;
        }

        if (computeRMS(mapTranslation, translationGround)>MAX_RMS_ECC){
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf( ts->LOG, "RMS = %f",
                computeRMS(mapTranslation, translationGround));
            return false;
        }

    }
    return true;
}

void CV_ECC_Test_Mask::run(int from)
{
    if (!testMask(from))
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Video_ECC_Translation, accuracy) { CV_ECC_Test_Translation test; test.safe_run();}
TEST(Video_ECC_Euclidean, accuracy) { CV_ECC_Test_Euclidean test; test.safe_run(); }
TEST(Video_ECC_Affine, accuracy) { CV_ECC_Test_Affine test; test.safe_run(); }
TEST(Video_ECC_Homography, accuracy) { CV_ECC_Test_Homography test; test.safe_run(); }
TEST(Video_ECC_Mask, accuracy) { CV_ECC_Test_Mask test; test.safe_run(); }

}} // namespace
