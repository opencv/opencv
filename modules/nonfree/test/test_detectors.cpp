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

#include <string>
#include <iostream>
#include <iterator>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <iterator>

using namespace cv;
using namespace std;

class CV_DetectorsTest : public cvtest::BaseTest
{
public:
    CV_DetectorsTest();
    ~CV_DetectorsTest();
protected:
    void run(int);
    template <class T> bool testDetector(const Mat& img, const T& detector, vector<KeyPoint>& expected);

    void LoadExpected(const string& file, vector<KeyPoint>& out);
};

CV_DetectorsTest::CV_DetectorsTest()
{
}
CV_DetectorsTest::~CV_DetectorsTest() {}

void getRotation(const Mat& img, Mat& aff, Mat& out)
{
    Point center(img.cols/2, img.rows/2);
    aff = getRotationMatrix2D(center, 30, 1);
    warpAffine( img, out, aff, img.size());
}

void getZoom(const Mat& img, Mat& aff, Mat& out)
{
    const double mult = 1.2;

    aff.create(2, 3, CV_64F);
    double *data = aff.ptr<double>();
    data[0] = mult; data[1] =    0; data[2] = 0;
    data[3] =    0; data[4] = mult; data[5] = 0;

    warpAffine( img, out, aff, img.size());
}

void getBlur(const Mat& img, Mat& aff, Mat& out)
{
    aff.create(2, 3, CV_64F);
    double *data = aff.ptr<double>();
    data[0] = 1; data[1] = 0; data[2] = 0;
    data[3] = 0; data[4] = 1; data[5] = 0;

    GaussianBlur(img, out, Size(5, 5), 2);
}

void getBrightness(const Mat& img, Mat& aff, Mat& out)
{
    aff.create(2, 3, CV_64F);
    double *data = aff.ptr<double>();
    data[0] = 1; data[1] = 0; data[2] = 0;
    data[3] = 0; data[4] = 1; data[5] = 0;

    add(img, Mat(img.size(), img.type(), Scalar(15)), out);
}

void showOrig(const Mat& img, const vector<KeyPoint>& orig_pts)
{

    Mat img_color;
    cvtColor(img, img_color, COLOR_GRAY2BGR);

    for(size_t i = 0; i < orig_pts.size(); ++i)
        circle(img_color, orig_pts[i].pt, (int)orig_pts[i].size/2, Scalar(0, 255, 0));

    namedWindow("O"); imshow("O", img_color);
}

void show(const string& name, const Mat& new_img, const vector<KeyPoint>& new_pts, const vector<KeyPoint>& transf_pts)
{

    Mat new_img_color;
    cvtColor(new_img, new_img_color, COLOR_GRAY2BGR);

    for(size_t i = 0; i < transf_pts.size(); ++i)
        circle(new_img_color, transf_pts[i].pt, (int)transf_pts[i].size/2, Scalar(255, 0, 0));

    for(size_t i = 0; i < new_pts.size(); ++i)
        circle(new_img_color, new_pts[i].pt, (int)new_pts[i].size/2, Scalar(0, 0, 255));

    namedWindow(name + "_T"); imshow(name + "_T", new_img_color);
}

struct WrapPoint
{
    const double* R;
    WrapPoint(const Mat& rmat) : R(rmat.ptr<double>()) { };

    KeyPoint operator()(const KeyPoint& kp) const
    {
        KeyPoint res = kp;
        res.pt.x = static_cast<float>(kp.pt.x * R[0] + kp.pt.y * R[1] + R[2]);
        res.pt.y = static_cast<float>(kp.pt.x * R[3] + kp.pt.y * R[4] + R[5]);
        return res;
    }
};

struct sortByR { bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) { return norm(kp1.pt) < norm(kp2.pt); } };

template <class T> bool CV_DetectorsTest::testDetector(const Mat& img, const T& detector, vector<KeyPoint>& exp)
{
    vector<KeyPoint> orig_kpts;
    detector(img, orig_kpts);

    typedef void (*TransfFunc )(const Mat&, Mat&, Mat& FransfFunc);
    const TransfFunc transfFunc[] = { getRotation, getZoom, getBlur, getBrightness };
    //const string names[] =  { "Rotation", "Zoom", "Blur", "Brightness" };
    const size_t case_num = sizeof(transfFunc)/sizeof(transfFunc[0]);

    vector<Mat> affs(case_num);
    vector<Mat> new_imgs(case_num);

    vector< vector<KeyPoint> > new_kpts(case_num);
    vector< vector<KeyPoint> > transf_kpts(case_num);

    //showOrig(img, orig_kpts);
    for(size_t i = 0; i < case_num; ++i)
    {
        transfFunc[i](img, affs[i], new_imgs[i]);
        detector(new_imgs[i], new_kpts[i]);
        transform(orig_kpts.begin(), orig_kpts.end(), back_inserter(transf_kpts[i]), WrapPoint(affs[i]));
        //show(names[i], new_imgs[i], new_kpts[i], transf_kpts[i]);
    }

    const float thres = 3;
    const float nthres = 3;

    vector<KeyPoint> result;
    for(size_t i = 0; i < orig_kpts.size(); ++i)
    {
        const KeyPoint& okp = orig_kpts[i];
        int foundCounter = 0;
        for(size_t j = 0; j < case_num; ++j)
        {
            const KeyPoint& tkp = transf_kpts[j][i];

            size_t k = 0;

            for(; k < new_kpts[j].size(); ++k)
                if (norm(new_kpts[j][k].pt - tkp.pt) < nthres && fabs(new_kpts[j][k].size - tkp.size) < thres)
                    break;

            if (k != new_kpts[j].size())
                ++foundCounter;

        }
        if (foundCounter == (int)case_num)
            result.push_back(okp);
    }

    sort(result.begin(), result.end(), sortByR());
    sort(exp.begin(), exp.end(), sortByR());

    if (result.size() != exp.size())
    {
      ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
      return false;
    }

    int foundCounter1 = 0;
    for(size_t i = 0; i < exp.size(); ++i)
    {
        const KeyPoint& e = exp[i];
        size_t j = 0;
        for(; j < result.size(); ++j)
        {
            const KeyPoint& r = result[i];
            if (norm(r.pt-e.pt) < nthres && fabs(r.size - e.size) < thres)
                break;
        }
        if (j != result.size())
            ++foundCounter1;
    }

    int foundCounter2 = 0;
    for(size_t i = 0; i < result.size(); ++i)
    {
        const KeyPoint& r = result[i];
        size_t j = 0;
        for(; j < exp.size(); ++j)
        {
            const KeyPoint& e = exp[i];
            if (norm(r.pt-e.pt) < nthres && fabs(r.size - e.size) < thres)
                break;
        }
        if (j != exp.size())
            ++foundCounter2;
    }
    //showOrig(img, result); waitKey();

    const float errorRate = 0.9f;
    if (float(foundCounter1)/exp.size() < errorRate || float(foundCounter2)/result.size() < errorRate)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

struct SurfNoMaskWrap
{
    const SURF& detector;
    SurfNoMaskWrap(const SURF& surf) : detector(surf) {}
    SurfNoMaskWrap& operator=(const SurfNoMaskWrap&);
    void operator()(const Mat& img, vector<KeyPoint>& kpts) const { detector(img, Mat(), kpts); }
};

void CV_DetectorsTest::LoadExpected(const string& file, vector<KeyPoint>& out)
{
    Mat mat_exp;
    FileStorage fs(file, FileStorage::READ);
    if (fs.isOpened())
    {
        read( fs["ResultVectorData"], mat_exp, Mat() );
        out.resize(mat_exp.cols / sizeof(KeyPoint));
        copy(mat_exp.ptr<KeyPoint>(), mat_exp.ptr<KeyPoint>() + out.size(), out.begin());
    }
    else
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA);
        out.clear();
    }
}

void CV_DetectorsTest::run( int /*start_from*/ )
{
    Mat img = imread(string(ts->get_data_path()) + "shared/graffiti.png", 0);

    if (img.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat to_test(img.size() * 2, img.type(), Scalar(0));
    Mat roi = to_test(Rect(img.rows/2, img.cols/2, img.cols, img.rows));
    img.copyTo(roi);
    GaussianBlur(to_test, to_test, Size(3, 3), 1.5);

    vector<KeyPoint> exp;
    LoadExpected(string(ts->get_data_path()) + "detectors/surf.xml", exp);
    if (exp.empty())
        return;

    if (!testDetector(to_test, SurfNoMaskWrap(SURF(1536+512+512, 2)), exp))
        return;

    LoadExpected(string(ts->get_data_path()) + "detectors/star.xml", exp);
    if (exp.empty())
        return;

    if (!testDetector(to_test, StarDetector(45, 30, 10, 8, 5), exp))
        return;

    ts->set_failed_test_info( cvtest::TS::OK);
}


TEST(Features2d_Detectors, regression) { CV_DetectorsTest test; test.safe_run(); }
