// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

namespace opencv_test
{

TEST(GAPI, Mat_Create_NoLink)
{
    cv::Mat m1;
    cv::Mat m2 = m1;
    m2.create(32, 32, CV_8U);

    EXPECT_NE(m1.rows, m2.rows);
    EXPECT_NE(m1.cols, m2.cols);
    EXPECT_NE(m1.data, m2.data);
}

TEST(GAPI, Mat_Recreate)
{
    cv::Mat m1 = cv::Mat::zeros(480, 640, CV_8U);
    m1.at<uchar>(0, 0) = 128;
    cv::Mat m2 = m1;

    EXPECT_EQ(m1.rows, m2.rows);
    EXPECT_EQ(m1.cols, m2.cols);
    EXPECT_EQ(m1.data, m2.data);
    EXPECT_EQ(m1.at<uchar>(0, 0), m2.at<uchar>(0, 0));

    // Calling "create" with the same meta is NOOP - both m1 and m2 are the same
    m1.create(480, 640, CV_8U);
    EXPECT_EQ(m1.rows, m2.rows);
    EXPECT_EQ(m1.cols, m2.cols);
    EXPECT_EQ(m1.data, m2.data);
    EXPECT_EQ(m1.at<uchar>(0, 0), m2.at<uchar>(0, 0));

    // Calling "create" on m2 with different meta doesn't update original m1
    // Now m1 and m2 are distinct
    m2.create(720, 1280, CV_8U);
    m2.at<uchar>(0, 0) = 64; // Initialize 0,0 element since m2 is a new buffer
    EXPECT_NE(m1.rows, m2.rows);
    EXPECT_NE(m1.cols, m2.cols);
    EXPECT_NE(m1.data, m2.data);
    EXPECT_NE(m1.at<uchar>(0, 0), m2.at<uchar>(0, 0));

    // What if a Mat is created from handle?
    uchar data[] = {
        32, 0, 0,
         0, 0, 0,
         0, 0, 0
    };
    cv::Mat m3(3, 3, CV_8U, data);
    cv::Mat m4 = m3;
    EXPECT_EQ(m3.rows, m4.rows);
    EXPECT_EQ(m3.cols, m4.cols);
    EXPECT_EQ(m3.data, m4.data);
    EXPECT_EQ(data, m3.data);
    EXPECT_EQ(data, m4.data);
    EXPECT_EQ(m3.at<uchar>(0, 0), m4.at<uchar>(0, 0));

    // cv::Mat::create must be NOOP if we don't change the meta,
    // even if the origianl mat is created from handle.
    m4.create(3, 3, CV_8U);
    EXPECT_EQ(m3.rows, m4.rows);
    EXPECT_EQ(m3.cols, m4.cols);
    EXPECT_EQ(m3.data, m4.data);
    EXPECT_EQ(data, m3.data);
    EXPECT_EQ(data, m4.data);
    EXPECT_EQ(m3.at<uchar>(0, 0), m4.at<uchar>(0, 0));
}

TEST(GAPI, EmptyOutMat)
{
    cv::Mat in_mat = cv::Mat(480, 640, CV_8U, cv::Scalar(64));

    cv::GComputation cc([]()
    {
        cv::GMat in;
        cv::GMat out = in + in;
        return cv::GComputation(in, out);
    });

    cv::Mat out;
    cc.apply(in_mat, out);

    EXPECT_EQ(640, out.cols);
    EXPECT_EQ(480, out.rows);
    EXPECT_EQ(CV_8U, out.type());
    EXPECT_EQ(0, cv::countNonZero(out - (in_mat+in_mat)));
}

}
