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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
#include <fstream>

using namespace std;
using namespace cv;
using namespace cvtest;

//#define DUMP

namespace
{
    // first four bytes, should be the same in little endian
    const float FLO_TAG_FLOAT = 202021.25f;  // check for this when READING the file
    const char FLO_TAG_STRING[] = "PIEH";    // use this when WRITING the file

    // binary file format for flow data specified here:
    // http://vision.middlebury.edu/flow/data/
    void writeOpticalFlowToFile(const Mat_<Point2f>& flow, const string& fileName)
    {
        ofstream file(fileName.c_str(), ios_base::binary);

        file << FLO_TAG_STRING;

        file.write((const char*) &flow.cols, sizeof(int));
        file.write((const char*) &flow.rows, sizeof(int));

        for (int i = 0; i < flow.rows; ++i)
        {
            for (int j = 0; j < flow.cols; ++j)
            {
                const Point2f u = flow(i, j);

                file.write((const char*) &u.x, sizeof(float));
                file.write((const char*) &u.y, sizeof(float));
            }
        }
    }

    // binary file format for flow data specified here:
    // http://vision.middlebury.edu/flow/data/
    void readOpticalFlowFromFile(Mat_<Point2f>& flow, const string& fileName)
    {
        ifstream file(fileName.c_str(), ios_base::binary);

        float tag;
        file.read((char*) &tag, sizeof(float));
        CV_Assert( tag == FLO_TAG_FLOAT );

        Size size;

        file.read((char*) &size.width, sizeof(int));
        file.read((char*) &size.height, sizeof(int));

        flow.create(size);

        for (int i = 0; i < flow.rows; ++i)
        {
            for (int j = 0; j < flow.cols; ++j)
            {
                Point2f u;

                file.read((char*) &u.x, sizeof(float));
                file.read((char*) &u.y, sizeof(float));

                flow(i, j) = u;
            }
        }
    }

    bool isFlowCorrect(Point2f u)
    {
        return !cvIsNaN(u.x) && !cvIsNaN(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
    }

    double calcRMSE(const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2)
    {
        double sum = 0.0;
        int counter = 0;

        for (int i = 0; i < flow1.rows; ++i)
        {
            for (int j = 0; j < flow1.cols; ++j)
            {
                const Point2f u1 = flow1(i, j);
                const Point2f u2 = flow2(i, j);

                if (isFlowCorrect(u1) && isFlowCorrect(u2))
                {
                    const Point2f diff = u1 - u2;
                    sum += diff.ddot(diff);
                    ++counter;
                }
            }
        }
        return sqrt(sum / (1e-9 + counter));
    }
}

TEST(Video_calcOpticalFlowDual_TVL1, Regression)
{
    const double MAX_RMSE = 0.03;

    const string frame1_path = TS::ptr()->get_data_path() + "optflow/RubberWhale1.png";
    const string frame2_path = TS::ptr()->get_data_path() + "optflow/RubberWhale2.png";
    const string gold_flow_path = TS::ptr()->get_data_path() + "optflow/tvl1_flow.flo";

    Mat frame1 = imread(frame1_path, IMREAD_GRAYSCALE);
    Mat frame2 = imread(frame2_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());
    ASSERT_FALSE(frame2.empty());

    Mat_<Point2f> flow;
    Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();

    tvl1->calc(frame1, frame2, flow);

#ifdef DUMP
    writeOpticalFlowToFile(flow, gold_flow_path);
#else
    Mat_<Point2f> gold;
    readOpticalFlowFromFile(gold, gold_flow_path);

    ASSERT_EQ(gold.rows, flow.rows);
    ASSERT_EQ(gold.cols, flow.cols);

    const double err = calcRMSE(gold, flow);
    EXPECT_LE(err, MAX_RMSE);
#endif
}
