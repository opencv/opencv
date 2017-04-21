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

TEST_P(StandartHoughLinesTest, regression)
{
    run_test(STANDART);
}

TEST_P(ProbabilisticHoughLinesTest, regression)
{
    run_test(PROBABILISTIC);
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
