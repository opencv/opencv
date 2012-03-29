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
#include <time.h>

#define IMGPROC_BOUNDINGRECT_ERROR_DIFF 1

#define MESSAGE_ERROR_DIFF "Bounding rectangle found by boundingRect function is incorrect."

using namespace cv;
using namespace std;

class CV_BoundingRectTest: public cvtest::ArrayTest
{
public:
	CV_BoundingRectTest();
	~CV_BoundingRectTest();

protected:
	void run (int);
	
private:
    template <typename T> void generate_src_points(vector <Point_<T> >& src, int n);
    template <typename T> cv::Rect get_bounding_rect(const vector <Point_<T> > src);
    template <typename T> bool checking_function_work(vector <Point_<T> >& src, int type);
};

CV_BoundingRectTest::CV_BoundingRectTest() {}
CV_BoundingRectTest::~CV_BoundingRectTest() {}

template <typename T> void CV_BoundingRectTest::generate_src_points(vector <Point_<T> >& src, int n)
{
    src.clear();
    for (int i = 0; i < n; ++i)
        src.push_back(Point_<T>(cv::randu<T>(), cv::randu<T>()));
}

template <typename T> cv::Rect CV_BoundingRectTest::get_bounding_rect(const vector <Point_<T> > src)
{
    int n = (int)src.size();
    T min_w = std::numeric_limits<T>::max(), max_w = std::numeric_limits<T>::min();
    T min_h = min_w, max_h = max_w;

    for (int i = 0; i < n; ++i)
    {
        min_w = std::min<T>(src.at(i).x, min_w);
        max_w = std::max<T>(src.at(i).x, max_w);
        min_h = std::min<T>(src.at(i).y, min_h);
        max_h = std::max<T>(src.at(i).y, max_h);
    }

    return Rect((int)min_w, (int)min_h, (int)max_w-(int)min_w + 1, (int)max_h-(int)min_h + 1);
}

template <typename T> bool CV_BoundingRectTest::checking_function_work(vector <Point_<T> >& src, int type)
{
    const int MAX_COUNT_OF_POINTS = 1000;
    const int N = 10000;

    for (int k = 0; k < N; ++k)
    {

        RNG& rng = ts->get_rng();

        int n = rng.next()%MAX_COUNT_OF_POINTS + 1;

        generate_src_points <T> (src, n);

        cv::Rect right = get_bounding_rect <T> (src);

        cv::Rect rect[2] = { boundingRect(src), boundingRect(Mat(src)) };

        for (int i = 0; i < 2; ++i) if (rect[i] != right)
        {
            cout << endl; cout << "Checking for the work of boundingRect function..." << endl;
            cout << "Type of src points: ";
            switch (type)
            {
            case 0: {cout << "INT"; break;}
            case 1: {cout << "FLOAT"; break;}
            default: break;
            }
            cout << endl;
            cout << "Src points are stored as "; if (i == 0) cout << "VECTOR" << endl; else cout << "MAT" << endl;
            cout << "Number of points: " << n << endl;
            cout << "Right rect (x, y, w, h): [" << right.x << ", " << right.y << ", " << right.width << ", " << right.height << "]" << endl;
            cout << "Result rect (x, y, w, h): [" << rect[i].x << ", " << rect[i].y << ", " << rect[i].width << ", " << rect[i].height << "]" << endl;
            cout << endl;
            CV_Error(IMGPROC_BOUNDINGRECT_ERROR_DIFF, MESSAGE_ERROR_DIFF);
            return false;
        }

    }

    return true;
}

void CV_BoundingRectTest::run(int)
{
    vector <Point> src_veci; if (!checking_function_work(src_veci, 0)) return;
    vector <Point2f> src_vecf; checking_function_work(src_vecf, 1);
}

TEST (Imgproc_BoundingRect, accuracy) { CV_BoundingRectTest test; test.safe_run(); }
