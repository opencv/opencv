// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ML_NBAYES, regression_5911)
{
    int N=12;
    Ptr<ml::NormalBayesClassifier> nb = cv::ml::NormalBayesClassifier::create();

    // data:
    Mat_<float> X(N,4);
    X << 1,2,3,4,  1,2,3,4,   1,2,3,4,    1,2,3,4,
         5,5,5,5,  5,5,5,5,   5,5,5,5,    5,5,5,5,
         4,3,2,1,  4,3,2,1,   4,3,2,1,    4,3,2,1;

    // labels:
    Mat_<int> Y(N,1);
    Y << 0,0,0,0, 1,1,1,1, 2,2,2,2;
    nb->train(X, ml::ROW_SAMPLE, Y);

    // single prediction:
    Mat R1,P1;
    for (int i=0; i<N; i++)
    {
        Mat r,p;
        nb->predictProb(X.row(i), r, p);
        R1.push_back(r);
        P1.push_back(p);
    }

    // bulk prediction (continuous memory):
    Mat R2,P2;
    nb->predictProb(X, R2, P2);

    EXPECT_EQ(sum(R1 == R2)[0], 255 * R2.total());
    EXPECT_EQ(sum(P1 == P2)[0], 255 * P2.total());

    // bulk prediction, with non-continuous memory storage
    Mat R3_(N, 1+1, CV_32S),
        P3_(N, 3+1, CV_32F);
    nb->predictProb(X, R3_.col(0), P3_.colRange(0,3));
    Mat R3 = R3_.col(0).clone(),
        P3 = P3_.colRange(0,3).clone();

    EXPECT_EQ(sum(R1 == R3)[0], 255 * R3.total());
    EXPECT_EQ(sum(P1 == P3)[0], 255 * P3.total());
}

}} // namespace
