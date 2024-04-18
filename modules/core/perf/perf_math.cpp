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

Mat randomOrtho(int m, int n)
{
    //TODO: fix
}

typedef perf::TestBaseWithParam<std::tuple<std::tuple<int, int>, int, int, int, bool>> SolveTest;

PERF_TEST_P(SolveTest, randomMat, ::testing::Combine(
    ::testing::Values(std::make_tuple(5, 5), std::make_tuple(10, 10), std::make_tuple(100, 100)),
    ::testing::Values(1, 50, 99, 100),
    ::testing::Values(CV_32F, CV_64F),
    ::testing::Values(DECOMP_LU, DECOMP_SVD, DECOMP_EIG, DECOMP_CHOLESKY, DECOMP_QR),
    ::testing::Bool()
    ))
{
    auto t = GetParam();
    auto rc = std::get<0>(t);
    int rows = std::get<0>(rc);
    int cols = std::get<1>(rc);
    int rankValue = std::get<2>(t);
    int mtype = std::get<2>(t);
    int method = std::get<3>(t);
    bool normal = std::get<4>(t);
    if (normal)
    {
        method |= DECOMP_NORMAL;
    }
    
    RNG& rng = theRNG();
    while (next())
    {
        Mat a = randomOrtho(rows, cols);
        Mat b(rows, 1, mtype);
        rng.fill(b, RNG::UNIFORM, Scalar(-1), Scalar(1));
        Mat x;

        startTimer();
        cv::solve(a, b, x, method);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<std::tuple<std::tuple<int, int>, int, int>> SvdTest;

PERF_TEST_P(SvdTest, decompose, ::testing::Combine(
    ::testing::Values(std::make_tuple(5, 5), std::make_tuple(10, 10), std::make_tuple(100, 100)),
    ::testing::Values(1, 50, 99, 100),
    ::testing::Values(CV_32F, CV_64F)
    ))
{
    auto t = GetParam();
    auto rc = std::get<0>(t);
    int rows = std::get<0>(rc);
    int cols = std::get<1>(rc);
    int rankValue = std::get<2>(t);
    int mtype = std::get<2>(t);

    while (next())
    {
        Mat a = randomOrtho(rows, cols);

        startTimer();
        cv::SVD svd(a);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(SvdTest, backSubst, ::testing::Combine(
    ::testing::Values(std::make_tuple(5, 5), std::make_tuple(10, 10), std::make_tuple(100, 100)),
    ::testing::Values(1, 50, 99, 100),
    ::testing::Values(CV_32F, CV_64F)
    ))
{
    auto t = GetParam();
    auto rc = std::get<0>(t);
    int rows = std::get<0>(rc);
    int cols = std::get<1>(rc);
    int rankValue = std::get<2>(t);
    int mtype = std::get<2>(t);

    while (next())
    {
        Mat a = randomOrtho(rows, cols);
        cv::SVD svd(a);

        startTimer();
        
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
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
