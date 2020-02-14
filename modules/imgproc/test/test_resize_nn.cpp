// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

        TEST(Resize_NN, accuracy)
        {
            int cols = 12, rows = 12;
            const int N = 10;
            double scale_factor = 0.9;
            int i1 = 5, j1 = 5; // position of pixel
            for (int i = 2; i < N; ++i)
            {
                cv::Mat test = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC(i)), dst;
                for (int j = 0; j < i; ++j)
                    //test.at<Vec<uchar, N>>(i1, j1)[j] = 1;
                    test.col(i1).row(j1).data[j] = 1;

                cv::resize(test, dst, Size(0, 0), scale_factor, scale_factor, cv::INTER_NEAREST);
                dst = dst.reshape(1, dst.cols * dst.rows * dst.channels());
                double sum = cv::sum(dst)[0];
                EXPECT_EQ(i, sum) << "Resize NN TEST mat from " << cols << "x" << rows << "x" << i << " failed with sum " << sum << " should be " << i;
            }
        }
    }} // namespace
