#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef tr1::tuple<int, int> MatType_Length_t;
typedef TestBaseWithParam<MatType_Length_t> MatType_Length;

PERF_TEST_P( MatType_Length, dot,
             testing::Combine(
                testing::Values( CV_8UC1, CV_32SC1, CV_32FC1 ),
                testing::Values( 32, 64, 128, 256, 512, 1024 )
            ))
{
    unsigned int type = std::tr1::get<0>(GetParam());
    unsigned int size = std::tr1::get<1>(GetParam());
    Mat a(size, size, type);
    Mat b(size, size, type);

    declare.in(a, WARMUP_RNG);
    declare.in(b, WARMUP_RNG);

    double product;

	TEST_CYCLE(100)
	{
        product = a.dot(b);
	}

    SANITY_CHECK(product, 1e-5);
}
