// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Every dataset row is in its own bucket, so it is found at distance 0 whatever bits the tables use.
static void expectRowsFound(flann::Index& index, const Mat& features)
{
    Mat indices, dists;
    index.knnSearch(features, indices, dists, 1, flann::SearchParams());
    ASSERT_EQ(features.rows, indices.rows);
    for (int i = 0; i < features.rows; i++)
    {
        EXPECT_EQ(i, indices.at<int>(i, 0));
        EXPECT_EQ(0, dists.at<int>(i, 0));
    }
}

TEST(Flann_LSH, load_with_small_key_size)
{
    Mat features(500, 32, CV_8UC1);
    randu(features, 0, 256);
    const std::string path = cv::tempfile(".flann");

    {
        // parameters from the matcher tutorial, with a key size below the default 20
        flann::Index built(features, flann::LshIndexParams(6, 12, 1), cvflann::FLANN_DIST_HAMMING);
        built.save(path);
    }

    flann::Index loaded;
    ASSERT_TRUE(loaded.load(features, path));
    expectRowsFound(loaded, features);
    EXPECT_EQ(0, remove(path.c_str()));
}

TEST(Flann_LSH, load_searches_the_given_features)
{
    Mat features(500, 32, CV_8UC1), other(500, 32, CV_8UC1);
    randu(features, 0, 256);
    randu(other, 0, 256);
    const std::string path = cv::tempfile(".flann");

    {
        flann::Index built(features, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
        built.save(path);
    }

    flann::Index loaded;
    ASSERT_TRUE(loaded.load(other, path));
    expectRowsFound(loaded, other);
    EXPECT_EQ(0, remove(path.c_str()));
}

}} // namespace
