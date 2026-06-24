// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

// KNN brute-force findNearest: dominated by the per-sample L2 distance reduction.
typedef TestBaseWithParam< tuple<int, int, int> > KNNFindNearest;  // (train samples, dims, K)

PERF_TEST_P(KNNFindNearest, brute_force, testing::Values(
            make_tuple(5000, 128, 5),
            make_tuple(10000, 64, 10)))
{
    const int nsamples = get<0>(GetParam());
    const int dims     = get<1>(GetParam());
    const int K        = get<2>(GetParam());
    const int nquery   = 2000;

    Mat train(nsamples, dims, CV_32F), responses(nsamples, 1, CV_32F), query(nquery, dims, CV_32F);
    RNG& rng = theRNG();
    rng.fill(train, RNG::UNIFORM, 0.f, 1.f);
    rng.fill(query, RNG::UNIFORM, 0.f, 1.f);
    for (int i = 0; i < nsamples; i++)
        responses.at<float>(i) = (float)(i & 1);

    Ptr<KNearest> knn = KNearest::create();
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);
    knn->setDefaultK(K);
    knn->train(train, ROW_SAMPLE, responses);

    Mat results, neighbors, dists;
    TEST_CYCLE() knn->findNearest(query, K, results, neighbors, dists);

    SANITY_CHECK_NOTHING();
}

}} // namespace
