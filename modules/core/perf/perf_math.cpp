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

///////////// Magnitude /////////////

typedef Size_MatType MagnitudeFixture;

PERF_TEST_P(MagnitudeFixture, Magnitude,
    testing::Combine(testing::Values(TYPICAL_MAT_SIZES), testing::Values(CV_32F, CV_64F)))
{
    cv::Size size = std::get<0>(GetParam());
    int type = std::get<1>(GetParam());

    cv::Mat x(size, type);
    cv::Mat y(size, type);
    cv::Mat magnitude(size, type);

    declare.in(x, y, WARMUP_RNG).out(magnitude);

    TEST_CYCLE() cv::magnitude(x, y, magnitude);

    SANITY_CHECK_NOTHING();
}

// generates random vectors, performs Gram-Schmidt orthogonalization on them
Mat randomOrtho(int rows, int ftype, RNG& rng)
{
    Mat result(rows, rows, ftype);
    rng.fill(result, RNG::UNIFORM, cv::Scalar(-1), cv::Scalar(1));

    for (int i = 0; i < rows; i++)
    {
        Mat v = result.row(i);

        for (int j = 0; j < i; j++)
        {
            Mat p = result.row(j);
            v -= p.dot(v) * p;
        }

        v = v * (1. / cv::norm(v));
    }

    return result;
}

template<typename FType>
Mat buildRandomMat(int rows, int cols, RNG& rng, int rank, bool symmetrical)
{
    int mtype = cv::traits::Depth<FType>::value;
    Mat u = randomOrtho(rows, mtype, rng);
    Mat v = randomOrtho(cols, mtype, rng);
    Mat s(rows, cols, mtype, Scalar(0));

    std::vector<FType> singVals(rank);
    rng.fill(singVals, RNG::UNIFORM, Scalar(0), Scalar(10));
    std::sort(singVals.begin(), singVals.end());
    auto singIter = singVals.rbegin();
    for (int i = 0; i < rank; i++)
    {
        s.at<FType>(i, i) = *singIter++;
    }

    if (symmetrical)
        return u * s * u.t();
    else
        return u * s * v.t();
}

Mat buildRandomMat(int rows, int cols, int mtype, RNG& rng, int rank, bool symmetrical)
{
    if (mtype == CV_32F)
    {
        return buildRandomMat<float>(rows, cols, rng, rank, symmetrical);
    }
    else if (mtype == CV_64F)
    {
        return buildRandomMat<double>(rows, cols, rng, rank, symmetrical);
    }
    else
    {
        CV_Error(cv::Error::StsBadArg, "This type is not supported");
    }
}

CV_ENUM(SolveDecompEnum, DECOMP_LU, DECOMP_SVD, DECOMP_EIG, DECOMP_CHOLESKY, DECOMP_QR)

enum RankMatrixOptions
{
    RANK_HALF, RANK_MINUS_1, RANK_FULL
};

CV_ENUM(RankEnum, RANK_HALF, RANK_MINUS_1, RANK_FULL)

enum SolutionsOptions
{
    NO_SOLUTIONS, ONE_SOLUTION, MANY_SOLUTIONS
};

CV_ENUM(SolutionsEnum, NO_SOLUTIONS, ONE_SOLUTION, MANY_SOLUTIONS)

typedef perf::TestBaseWithParam<std::tuple<int, RankEnum, MatDepth, SolveDecompEnum, bool, SolutionsEnum>> SolveTest;

PERF_TEST_P(SolveTest, randomMat, ::testing::Combine(
    ::testing::Values(31, 64, 100),
    ::testing::Values(RANK_HALF, RANK_MINUS_1, RANK_FULL),
    ::testing::Values(CV_32F, CV_64F),
    ::testing::Values(DECOMP_LU, DECOMP_SVD, DECOMP_EIG, DECOMP_CHOLESKY, DECOMP_QR),
    ::testing::Bool(), // normal
    ::testing::Values(NO_SOLUTIONS, ONE_SOLUTION, MANY_SOLUTIONS)
    ))
{
    auto t = GetParam();
    int size       = std::get<0>(t);
    auto rankEnum  = std::get<1>(t);
    int mtype      = std::get<2>(t);
    int method     = std::get<3>(t);
    bool normal    = std::get<4>(t);
    auto solutions = std::get<5>(t);

    bool symmetrical = (method == DECOMP_CHOLESKY || method == DECOMP_LU);

    if (normal)
    {
        method |= DECOMP_NORMAL;
    }

    int rank = size;
    switch (rankEnum)
    {
        case RANK_HALF:    rank /= 2; break;
        case RANK_MINUS_1: rank -= 1; break;
        default: break;
    }

    RNG& rng = theRNG();
    Mat A = buildRandomMat(size, size, mtype, rng, rank, symmetrical);
    Mat x(size, 1, mtype);
    Mat b(size, 1, mtype);

    switch (solutions)
    {
        // no solutions, let's make b random
        case NO_SOLUTIONS:
        {
            rng.fill(b, RNG::UNIFORM, Scalar(-1), Scalar(1));
        }
        break;
        // exactly 1 solution, let's combine b from A and x
        case ONE_SOLUTION:
        {
            rng.fill(x, RNG::UNIFORM, Scalar(-10), Scalar(10));
            b = A * x;
        }
        break;
        // infinitely many solutions, let's make b zero
        default:
        {
            b = 0;
        }
        break;
    }

    TEST_CYCLE() cv::solve(A, b, x, method);

    SANITY_CHECK_NOTHING();
}

typedef perf::TestBaseWithParam<std::tuple<std::tuple<int, int>, RankEnum, MatDepth, bool, bool>> SvdTest;

PERF_TEST_P(SvdTest, decompose, ::testing::Combine(
    ::testing::Values(std::make_tuple(5, 15), std::make_tuple(32, 32), std::make_tuple(100, 100)),
    ::testing::Values(RANK_HALF, RANK_MINUS_1, RANK_FULL),
    ::testing::Values(CV_32F, CV_64F),
    ::testing::Bool(), // symmetrical
    ::testing::Bool() // needUV
    ))
{
    auto t = GetParam();
    auto rc          = std::get<0>(t);
    auto rankEnum    = std::get<1>(t);
    int mtype        = std::get<2>(t);
    bool symmetrical = std::get<3>(t);
    bool needUV      = std::get<4>(t);

    int rows = std::get<0>(rc);
    int cols = std::get<1>(rc);

    if (symmetrical)
    {
        rows = max(rows, cols);
        cols = rows;
    }

    int rank = std::min(rows, cols);
    switch (rankEnum)
    {
    case RANK_HALF:    rank /= 2; break;
    case RANK_MINUS_1: rank -= 1; break;
    default: break;
    }

    int flags = needUV ? 0 : SVD::NO_UV;

    RNG& rng = theRNG();
    Mat A = buildRandomMat(rows, cols, mtype, rng, rank, symmetrical);
    TEST_CYCLE() cv::SVD svd(A, flags);

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P(SvdTest, backSubst, ::testing::Combine(
    ::testing::Values(std::make_tuple(5, 15), std::make_tuple(32, 32), std::make_tuple(100, 100)),
    ::testing::Values(RANK_HALF, RANK_MINUS_1, RANK_FULL),
    ::testing::Values(CV_32F, CV_64F),
    // back substitution works the same regardless of source matrix properties
    ::testing::Values(true),
    // back substitution has no sense without u and v
    ::testing::Values(true) // needUV
    ))
{
    auto t = GetParam();
    auto rc       = std::get<0>(t);
    auto rankEnum = std::get<1>(t);
    int mtype     = std::get<2>(t);

    int rows = std::get<0>(rc);
    int cols = std::get<1>(rc);

    int rank = std::min(rows, cols);
    switch (rankEnum)
    {
    case RANK_HALF:    rank /= 2; break;
    case RANK_MINUS_1: rank -= 1; break;
    default: break;
    }

    RNG& rng = theRNG();
    Mat A = buildRandomMat(rows, cols, mtype, rng, rank, /* symmetrical */ false);
    cv::SVD svd(A);
    // preallocate to not spend time on it during backSubst()
    Mat dst(cols, 1, mtype);
    Mat rhs(rows, 1, mtype);
    rng.fill(rhs, RNG::UNIFORM, Scalar(-10), Scalar(10));

    TEST_CYCLE() svd.backSubst(rhs, dst);

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
