#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

namespace {

typedef perf::TestBaseWithParam<size_t> VectorLength;

PERF_TEST_P(VectorLength, phase32f, testing::Values(128, 1000, 128*1024, 512*1024, 1024*1024))
{
    size_t length = GetParam();
    vector<float> X(length);
    vector<float> Y(length);
    vector<float> angle(length);

    declare.in(X, Y, WARMUP_RNG).out(angle);

    TEST_CYCLE_N(200) cv::phase(X, Y, angle, true);

    SANITY_CHECK(angle, 5e-5);
}

PERF_TEST_P(VectorLength, phase64f, testing::Values(128, 1000, 128*1024, 512*1024, 1024*1024))
{
    size_t length = GetParam();
    vector<double> X(length);
    vector<double> Y(length);
    vector<double> angle(length);

    declare.in(X, Y, WARMUP_RNG).out(angle);

    TEST_CYCLE_N(200) cv::phase(X, Y, angle, true);

    SANITY_CHECK(angle, 5e-5);
}

typedef perf::TestBaseWithParam< testing::tuple<int, int, int> > KMeans;

PERF_TEST_P_(KMeans, single_iter)
{
    RNG& rng = theRNG();
    const int K = testing::get<0>(GetParam());
    const int dims = testing::get<1>(GetParam());
    const int N = testing::get<2>(GetParam());
    const int attempts = 5;

    Mat data(N, dims, CV_32F);
    rng.fill(data, RNG::UNIFORM, -0.1, 0.1);

    const int N0 = K;
    Mat data0(N0, dims, CV_32F);
    rng.fill(data0, RNG::UNIFORM, -1, 1);

    for (int i = 0; i < N; i++)
    {
        int base = rng.uniform(0, N0);
        cv::add(data0.row(base), data.row(i), data.row(i));
    }

    declare.in(data);

    Mat labels, centers;

    TEST_CYCLE()
    {
        kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1, 0),
               attempts, KMEANS_PP_CENTERS, centers);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(KMeans, good)
{
    RNG& rng = theRNG();
    const int K = testing::get<0>(GetParam());
    const int dims = testing::get<1>(GetParam());
    const int N = testing::get<2>(GetParam());
    const int attempts = 5;

    Mat data(N, dims, CV_32F);
    rng.fill(data, RNG::UNIFORM, -0.1, 0.1);

    const int N0 = K;
    Mat data0(N0, dims, CV_32F);
    rng.fill(data0, RNG::UNIFORM, -1, 1);

    for (int i = 0; i < N; i++)
    {
        int base = rng.uniform(0, N0);
        cv::add(data0.row(base), data.row(i), data.row(i));
    }

    declare.in(data);

    Mat labels, centers;

    TEST_CYCLE()
    {
        kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0),
               attempts, KMEANS_PP_CENTERS, centers);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(KMeans, with_duplicates)
{
    RNG& rng = theRNG();
    const int K = testing::get<0>(GetParam());
    const int dims = testing::get<1>(GetParam());
    const int N = testing::get<2>(GetParam());
    const int attempts = 5;

    Mat data(N, dims, CV_32F, Scalar::all(0));

    const int N0 = std::max(2, K * 2 / 3);
    Mat data0(N0, dims, CV_32F);
    rng.fill(data0, RNG::UNIFORM, -1, 1);

    for (int i = 0; i < N; i++)
    {
        int base = rng.uniform(0, N0);
        data0.row(base).copyTo(data.row(i));
    }

    declare.in(data);

    Mat labels, centers;

    TEST_CYCLE()
    {
        kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0),
               attempts, KMEANS_PP_CENTERS, centers);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , KMeans,
    testing::Values(
        // K clusters, dims, N points
        testing::make_tuple(2, 3, 100000),
        testing::make_tuple(4, 3, 500),
        testing::make_tuple(4, 3, 1000),
        testing::make_tuple(4, 3, 10000),
        testing::make_tuple(8, 3, 1000),
        testing::make_tuple(8, 16, 1000),
        testing::make_tuple(8, 64, 1000),
        testing::make_tuple(16, 16, 1000),
        testing::make_tuple(16, 32, 1000),
        testing::make_tuple(32, 16, 1000),
        testing::make_tuple(32, 32, 1000),
        testing::make_tuple(100, 2, 1000)
    )
);

}

} // namespace
