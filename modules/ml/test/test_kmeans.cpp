// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ML_KMeans, accuracy)
{
    const int iters = 100;
    int sizesArr[] = { 5000, 7000, 8000 };
    int pointsCount = sizesArr[0]+ sizesArr[1] + sizesArr[2];

    Mat data( pointsCount, 2, CV_32FC1 ), labels;
    vector<int> sizes( sizesArr, sizesArr + sizeof(sizesArr) / sizeof(sizesArr[0]) );
    Mat means;
    vector<Mat> covs;
    defaultDistribs( means, covs );
    generateData( data, labels, sizes, means, covs, CV_32FC1, CV_32SC1 );
    TermCriteria termCriteria( TermCriteria::COUNT, iters, 0.0);

    {
        SCOPED_TRACE("KMEANS_PP_CENTERS");
        float err = 1000;
        Mat bestLabels;
        kmeans( data, 3, bestLabels, termCriteria, 0, KMEANS_PP_CENTERS, noArray() );
        EXPECT_TRUE(calcErr( bestLabels, labels, sizes, err , false ));
        EXPECT_LE(err, 0.01f);
    }
    {
        SCOPED_TRACE("KMEANS_RANDOM_CENTERS");
        float err = 1000;
        Mat bestLabels;
        kmeans( data, 3, bestLabels, termCriteria, 0, KMEANS_RANDOM_CENTERS, noArray() );
        EXPECT_TRUE(calcErr( bestLabels, labels, sizes, err, false ));
        EXPECT_LE(err, 0.01f);
    }
    {
        SCOPED_TRACE("KMEANS_USE_INITIAL_LABELS");
        float err = 1000;
        Mat bestLabels;
        labels.copyTo( bestLabels );
        RNG &rng = cv::theRNG();
        for( int i = 0; i < 0.5f * pointsCount; i++ )
        bestLabels.at<int>( rng.next() % pointsCount, 0 ) = rng.next() % 3;
        kmeans( data, 3, bestLabels, termCriteria, 0, KMEANS_USE_INITIAL_LABELS, noArray() );
        EXPECT_TRUE(calcErr( bestLabels, labels, sizes, err, false ));
        EXPECT_LE(err, 0.01f);
    }
}

}} // namespace
