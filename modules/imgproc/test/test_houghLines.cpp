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
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
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

using namespace cv;
using namespace std;

template<typename T>
struct SimilarWith
{
    T value;
    float theta_eps;
    float rho_eps;
    SimilarWith<T>(T val, float e, float r_e): value(val), theta_eps(e), rho_eps(r_e) { };
    bool operator()(T other);
};

template<>
bool SimilarWith<Vec2f>::operator()(Vec2f other)
{
    return abs(other[0] - value[0]) < rho_eps && abs(other[1] - value[1]) < theta_eps;
}

template<>
bool SimilarWith<Vec4i>::operator()(Vec4i other)
{
    return norm(value, other) < theta_eps;
}

template <typename T>
int countMatIntersection(Mat expect, Mat actual, float eps, float rho_eps)
{
    int count = 0;
    if (!expect.empty() && !actual.empty())
    {
        for (MatIterator_<T> it=expect.begin<T>(); it!=expect.end<T>(); it++)
        {
            MatIterator_<T> f = std::find_if(actual.begin<T>(), actual.end<T>(), SimilarWith<T>(*it, eps, rho_eps));
            if (f != actual.end<T>())
                count++;
        }
    }
    return count;
}

String getTestCaseName(String filename)
{
    string temp(filename);
    size_t pos = temp.find_first_of("\\/.");
    while ( pos != string::npos ) {
       temp.replace( pos, 1, "_" );
       pos = temp.find_first_of("\\/.");
    }
    return String(temp);
}

class BaseHoughLineTest
{
public:
    enum {STANDART = 0, PROBABILISTIC};
protected:
    void run_test(int type);

    string picture_name;
    double rhoStep;
    double thetaStep;
    int threshold;
    int minLineLength;
    int maxGap;
};

typedef std::tr1::tuple<string, double, double, int> Image_RhoStep_ThetaStep_Threshold_t;
class StandartHoughLinesTest : public BaseHoughLineTest, public testing::TestWithParam<Image_RhoStep_ThetaStep_Threshold_t>
{
public:
    StandartHoughLinesTest()
    {
        picture_name = std::tr1::get<0>(GetParam());
        rhoStep = std::tr1::get<1>(GetParam());
        thetaStep = std::tr1::get<2>(GetParam());
        threshold = std::tr1::get<3>(GetParam());
        minLineLength = 0;
        maxGap = 0;
    }
};

typedef std::tr1::tuple<string, double, double, int, int, int> Image_RhoStep_ThetaStep_Threshold_MinLine_MaxGap_t;
class ProbabilisticHoughLinesTest : public BaseHoughLineTest, public testing::TestWithParam<Image_RhoStep_ThetaStep_Threshold_MinLine_MaxGap_t>
{
public:
    ProbabilisticHoughLinesTest()
    {
        picture_name = std::tr1::get<0>(GetParam());
        rhoStep = std::tr1::get<1>(GetParam());
        thetaStep = std::tr1::get<2>(GetParam());
        threshold = std::tr1::get<3>(GetParam());
        minLineLength = std::tr1::get<4>(GetParam());
        maxGap = std::tr1::get<5>(GetParam());
    }
};

typedef std::tr1::tuple<double, double, double, double> HoughLinesUsingSetOfPointsInput_t;
class HoughLinesUsingSetOfPointsTest : public testing::TestWithParam<HoughLinesUsingSetOfPointsInput_t>
{
protected:
    void run_test();
    double Rho;
    double Theta;
    HoughDetectParam paramRho, paramTheta;
public:
    HoughLinesUsingSetOfPointsTest()
    {
        paramRho.min = std::tr1::get<0>(GetParam());
        paramRho.max = std::tr1::get<1>(GetParam());
        paramRho.step = (paramRho.max - paramRho.min) / 180.0f;
        paramTheta.min = std::tr1::get<2>(GetParam());
        paramTheta.max = std::tr1::get<3>(GetParam());
        paramTheta.step = CV_PI / 360.0f;
        Rho = 320.0f;
        Theta = 1.047198f;
    }
};

void BaseHoughLineTest::run_test(int type)
{
    string filename = cvtest::TS::ptr()->get_data_path() + picture_name;
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    EXPECT_FALSE(src.empty()) << "Invalid test image: " << filename;

    string xml;
    if (type == STANDART)
        xml = string(cvtest::TS::ptr()->get_data_path()) + "imgproc/HoughLines.xml";
    else if (type == PROBABILISTIC)
        xml = string(cvtest::TS::ptr()->get_data_path()) + "imgproc/HoughLinesP.xml";

    Mat dst;
    Canny(src, dst, 100, 150, 3);
    EXPECT_FALSE(dst.empty()) << "Failed Canny edge detector";

    Mat lines;
    if (type == STANDART)
        HoughLines(dst, lines, rhoStep, thetaStep, threshold, 0, 0);
    else if (type == PROBABILISTIC)
        HoughLinesP(dst, lines, rhoStep, thetaStep, threshold, minLineLength, maxGap);

    String test_case_name = format("lines_%s_%.0f_%.2f_%d_%d_%d", picture_name.c_str(), rhoStep, thetaStep,
                                    threshold, minLineLength, maxGap);
    test_case_name = getTestCaseName(test_case_name);

    FileStorage fs(xml, FileStorage::READ);
    FileNode node = fs[test_case_name];
    if (node.empty())
    {
        fs.release();
        fs.open(xml, FileStorage::APPEND);
        EXPECT_TRUE(fs.isOpened()) << "Cannot open sanity data file: " << xml;
        fs << test_case_name << lines;
        fs.release();
        fs.open(xml, FileStorage::READ);
        EXPECT_TRUE(fs.isOpened()) << "Cannot open sanity data file: " << xml;
    }

    Mat exp_lines;
    read( fs[test_case_name], exp_lines, Mat() );
    fs.release();

    int count = -1;
    if (type == STANDART)
        count = countMatIntersection<Vec2f>(exp_lines, lines, (float) thetaStep + FLT_EPSILON, (float) rhoStep + FLT_EPSILON);
    else if (type == PROBABILISTIC)
        count = countMatIntersection<Vec4i>(exp_lines, lines, 1e-4f, 0.f);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
    EXPECT_GE( count, (int) (exp_lines.total() * 0.8) );
#else
    EXPECT_EQ( count, (int) exp_lines.total());
#endif
}

void HoughLinesUsingSetOfPointsTest::run_test(void)
{
    HoughLinePolar houghpolar[20];
    static const float Points[20][2] = {
    { 0.0f,   369.50417f }, { 10.0f,  363.73067f }, { 20.0f,  357.95717f }, { 30.0f,  352.18366f },
    { 40.0f,  346.41016f }, { 50.0f,  340.63666f }, { 60.0f,  334.86316f }, { 70.0f,  329.08965f },
    { 80.0f,  323.31615f }, { 90.0f,  317.54265f }, { 100.0f, 311.76915f }, { 110.0f, 305.99564f },
    { 120.0f, 300.22214f }, { 130.0f, 294.44864f }, { 140.0f, 288.67514f }, { 150.0f, 282.90163f },
    { 160.0f, 277.12813f }, { 170.0f, 271.35463f }, { 180.0f, 265.58112f }, { 190.0f, 259.80762f }
    };

    Point2f point[20];
    int polar_index = 0;
    for (int i = 0; i < 20; i++)
    {
        point[i].x = Points[i][0];
        point[i].y = Points[i][1];
    }

    polar_index = HoughLinesUsingSetOfPoints(20, point, &paramRho, &paramTheta, 20, houghpolar);

    EXPECT_EQ((int) ((houghpolar + polar_index)->rho * 10.0f), (int) (Rho * 10.0f));
    EXPECT_EQ((int) ((houghpolar + polar_index)->angle * 100000.0f), (int) (Theta * 100000.0f));
}

TEST_P(StandartHoughLinesTest, regression)
{
    run_test(STANDART);
}

TEST_P(ProbabilisticHoughLinesTest, regression)
{
    run_test(PROBABILISTIC);
}

TEST_P(HoughLinesUsingSetOfPointsTest, regression)
{
    run_test();
}

INSTANTIATE_TEST_CASE_P( ImgProc, StandartHoughLinesTest, testing::Combine(testing::Values( "shared/pic5.png", "../stitching/a1.png" ),
                                                                           testing::Values( 1, 10 ),
                                                                           testing::Values( 0.05, 0.1 ),
                                                                           testing::Values( 80, 150 )
                                                                           ));

INSTANTIATE_TEST_CASE_P( ImgProc, ProbabilisticHoughLinesTest, testing::Combine(testing::Values( "shared/pic5.png", "shared/pic1.png" ),
                                                                                testing::Values( 5, 10 ),
                                                                                testing::Values( 0.05, 0.1 ),
                                                                                testing::Values( 75, 150 ),
                                                                                testing::Values( 0, 10 ),
                                                                                testing::Values( 0, 4 )
                                                                                ));

INSTANTIATE_TEST_CASE_P( Imgproc, HoughLinesUsingSetOfPointsTest, testing::Combine(testing::Values( 0.0f, 120.0f ),             // rhoMin
                                                                                   testing::Values( 320.0f, 320.0f ),         // rhoMax
                                                                                   testing::Values( 0, (CV_PI / 6.0f) ),                   // thetaMin
                                                                                   testing::Values( (CV_PI / 2.0f), (CV_PI / 2.0f) )   // thetaMax
                                                                                   ));
