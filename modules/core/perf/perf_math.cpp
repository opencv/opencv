#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<size_t> VectorLength;

typedef std::tr1::tuple<int, int> MaxDim_MaxPoints_t;
typedef perf::TestBaseWithParam<MaxDim_MaxPoints_t> MaxDim_MaxPoints;

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

PERF_TEST_P( MaxDim_MaxPoints, kmeans,
             testing::Combine( testing::Values( 16, 32, 64 ),
                               testing::Values( 300, 400, 500) ) )
{
    RNG& rng = theRNG();
    const int MAX_DIM = get<0>(GetParam());
    const int MAX_POINTS = get<1>(GetParam());
    const int attempts = 5;

    Mat labels, centers;
    int i,  N = 0, N0 = 0, K = 0, dims = 0;
    dims = rng.uniform(1, MAX_DIM+1);
    N = rng.uniform(1, MAX_POINTS+1);
    N0 = rng.uniform(1, MAX(N/10, 2));
    K = rng.uniform(1, N+1);

    Mat data0(N0, dims, CV_32F);
    rng.fill(data0, RNG::UNIFORM, -1, 1);

    Mat data(N, dims, CV_32F);
    for( i = 0; i < N; i++ )
        data0.row(rng.uniform(0, N0)).copyTo(data.row(i));

    declare.in(data);

    TEST_CYCLE()
    {
        kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0),
               attempts, KMEANS_PP_CENTERS, centers);
    }

    Mat clusterPointsNumber = Mat::zeros(1, K, CV_32S);

    for( i = 0; i < labels.rows; i++ )
    {
        int clusterIdx = labels.at<int>(i);
        clusterPointsNumber.at<int>(clusterIdx)++;
    }

    Mat sortedClusterPointsNumber;
    cv::sort(clusterPointsNumber, sortedClusterPointsNumber, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);

    SANITY_CHECK(sortedClusterPointsNumber);
}
