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
#include "opencv2/imgproc/imgproc_c.h"

#include <vector>
#include <string>
using namespace std;
using namespace cv;

class CV_MserTest : public cvtest::BaseTest
{
public:
    CV_MserTest();
protected:
    void run(int);
    int LoadBoxes(const char* path, vector<CvBox2D>& boxes);
    int SaveBoxes(const char* path, const vector<CvBox2D>& boxes);
    int CompareBoxes(const vector<CvBox2D>& boxes1,const vector<CvBox2D>& boxes2, float max_rel_diff = 0.01f);
};

CV_MserTest::CV_MserTest()
{
}

int CV_MserTest::LoadBoxes(const char* path, vector<CvBox2D>& boxes)
{
    boxes.clear();
    FILE* f = fopen(path,"r");

    if (f==NULL)
    {
        return 0;
    }

    while (!feof(f))
    {
        CvBox2D box;
        int values_read = fscanf(f,"%f,%f,%f,%f,%f\n",&box.angle,&box.center.x,&box.center.y,&box.size.width,&box.size.height);
        CV_Assert(values_read == 5);
        boxes.push_back(box);
    }
    fclose(f);
    return 1;
}

int CV_MserTest::SaveBoxes(const char* path, const vector<CvBox2D>& boxes)
{
    FILE* f = fopen(path,"w");
    if (f==NULL)
    {
        return 0;
    }
    for (int i=0;i<(int)boxes.size();i++)
    {
        fprintf(f,"%f,%f,%f,%f,%f\n",boxes[i].angle,boxes[i].center.x,boxes[i].center.y,boxes[i].size.width,boxes[i].size.height);
    }
    fclose(f);
    return 1;
}

int CV_MserTest::CompareBoxes(const vector<CvBox2D>& boxes1,const vector<CvBox2D>& boxes2, float max_rel_diff)
{
    if (boxes1.size() != boxes2.size())
        return 0;

    for (int i=0; i<(int)boxes1.size();i++)
    {
        float rel_diff;
        if (!((boxes1[i].angle == 0.0f) && (abs(boxes2[i].angle) < max_rel_diff)))
        {
            float angle_diff = (float)fmod(boxes1[i].angle - boxes2[i].angle, 180);
            // for angular correctness, it makes no sense to use a "relative" error.
            // a 1-degree error around 5 degrees is equally bas as around 250 degrees.
            // in correct cases, angle_diff can now be a bit above 0 or a bit below 180
            if (angle_diff > 90.0f)
            {
                    angle_diff -= 180.0f;
            }
            rel_diff = (float)fabs(angle_diff);
            if (rel_diff > max_rel_diff)
                return i;
        }

        if (!((boxes1[i].center.x == 0.0f) && (abs(boxes2[i].center.x) < max_rel_diff)))
        {
            rel_diff = abs(boxes1[i].center.x-boxes2[i].center.x)/abs(boxes1[i].center.x);
            if (rel_diff > max_rel_diff)
                return i;
        }

        if (!((boxes1[i].center.y == 0.0f) && (abs(boxes2[i].center.y) < max_rel_diff)))
        {
            rel_diff = abs(boxes1[i].center.y-boxes2[i].center.y)/abs(boxes1[i].center.y);
            if (rel_diff > max_rel_diff)
                return i;
        }
        if (!((boxes1[i].size.width == 0.0f) && (abs(boxes2[i].size.width) < max_rel_diff)))
        {
            rel_diff = abs(boxes1[i].size.width-boxes2[i].size.width)/abs(boxes1[i].size.width);
            if (rel_diff > max_rel_diff)
            return i;
        }

        if (!((boxes1[i].size.height == 0.0f) && (abs(boxes2[i].size.height) < max_rel_diff)))
        {
            rel_diff = abs(boxes1[i].size.height-boxes2[i].size.height)/abs(boxes1[i].size.height);
            if (rel_diff > max_rel_diff)
                return i;
        }
    }

    return -1;
}

void CV_MserTest::run(int)
{
    string image_path = string(ts->get_data_path()) + "mser/puzzle.png";

    Mat img = imread( image_path );
    if (img.empty())
    {
        ts->printf( cvtest::TS::LOG, "Unable to open image mser/puzzle.png\n");
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        return;
    }

    Mat yuv;
    cvtColor(img, yuv, COLOR_BGR2YCrCb);
    vector<vector<Point> > msers;
    MSER()(yuv, msers);

    vector<CvBox2D> boxes;
    vector<CvBox2D> boxes_orig;
    for ( size_t i = 0; i < msers.size(); i++ )
    {
        RotatedRect box = fitEllipse(msers[i]);
        box.angle=(float)CV_PI/2-box.angle;
        boxes.push_back(box);
    }

    string boxes_path = string(ts->get_data_path()) + "mser/boxes.txt";
    string calc_boxes_path = string(ts->get_data_path()) + "mser/boxes.calc.txt";

    if (!LoadBoxes(boxes_path.c_str(),boxes_orig))
    {
        SaveBoxes(boxes_path.c_str(),boxes);
        ts->printf( cvtest::TS::LOG, "Unable to open data file mser/boxes.txt\n");
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        return;
    }

    const float dissimularity = 0.01f;
    int n_box = CompareBoxes(boxes_orig,boxes,dissimularity);
    if (n_box < 0)
    {
        ts->set_failed_test_info(cvtest::TS::OK);
    }
    else
    {
        SaveBoxes(calc_boxes_path.c_str(), boxes);
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        ts->printf( cvtest::TS::LOG, "Incorrect correspondence in box %d\n",n_box);
    }
}

TEST(Features2d_MSER, DISABLED_regression) { CV_MserTest test; test.safe_run(); }
