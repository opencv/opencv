#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

PERF_TEST_P(Size_MatType, Mat_Eye,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat diagonalMatrix(size.height, size.width, type);

    declare.out(diagonalMatrix);

    int runs = (size.width <= 640) ? 15 : 5;
    TEST_CYCLE_MULTIRUN(runs)
    {
        diagonalMatrix = Mat::eye(size, type);
    }

    SANITY_CHECK(diagonalMatrix, 1);
}

PERF_TEST_P(Size_MatType, Mat_Zeros,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES, CV_32FC3))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat zeroMatrix(size.height, size.width, type);

    declare.out(zeroMatrix);

    int runs = (size.width <= 640) ? 15 : 5;
    TEST_CYCLE_MULTIRUN(runs)
    {
        zeroMatrix = Mat::zeros(size, type);
    }

    SANITY_CHECK(zeroMatrix, 1);
}

PERF_TEST_P(Size_MatType, Mat_Clone,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat source(size.height, size.width, type);
    Mat destination(size.height, size.width, type);;

    declare.in(source, WARMUP_RNG).out(destination);

    TEST_CYCLE()
    {
        source.clone();
    }
    destination = source.clone();

    SANITY_CHECK(destination, 1);
}

PERF_TEST_P(Size_MatType, Mat_Clone_Roi,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(TYPICAL_MAT_TYPES))

             )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());

    unsigned int width = size.width;
    unsigned int height = size.height;
    Mat source(height, width, type);
    Mat destination(size.height/2, size.width/2, type);

    declare.in(source, WARMUP_RNG).out(destination);

    Mat roi(source, Rect(width/4, height/4, 3*width/4, 3*height/4));

    TEST_CYCLE()
    {
        roi.clone();
    }
    destination = roi.clone();

    SANITY_CHECK(destination, 1);
}
