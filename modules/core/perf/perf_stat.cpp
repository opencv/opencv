#include "perf_precomp.hpp"
#include "opencv2/core/core_c.h"

using namespace std;
using namespace cv;
using namespace perf;


/*
// Scalar sum(InputArray arr)
*/
PERF_TEST_P( Size_MatType, sum, TYPICAL_MATS )
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    Mat arr(sz, type);
    Scalar s;

    declare.in(arr, WARMUP_RNG);

    TEST_CYCLE(100) { s = sum(arr); }

    SANITY_CHECK(s);
}


/*
// Scalar mean(InputArray src)
*/
PERF_TEST_P( Size_MatType, mean, TYPICAL_MATS )
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    Mat src(sz, type);
    Scalar s;
    
    declare.in(src, WARMUP_RNG);
    
    TEST_CYCLE(100) { s = mean(src); }
    
    SANITY_CHECK(s);
}


/*
// Scalar mean(InputArray src, InputArray mask=noArray())
*/
PERF_TEST_P( Size_MatType, mean_mask, TYPICAL_MATS )
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    Mat src(sz, type);
    Mat mask = Mat::ones(src.size(), CV_8U);
    Scalar s;
    
    declare.in(src, WARMUP_RNG).in(mask);
    
    TEST_CYCLE(100) { s = mean(src, mask); }
    
    SANITY_CHECK(s);
}

CV_FLAGS(NormType, NORM_INF, NORM_L1, NORM_L2, NORM_TYPE_MASK, NORM_RELATIVE, NORM_MINMAX)
typedef std::tr1::tuple<Size, MatType, NormType> Size_MatType_NormType_t;
typedef perf::TestBaseWithParam<Size_MatType_NormType_t> Size_MatType_NormType;

/*
// double norm(InputArray src1, int normType=NORM_L2)
*/
PERF_TEST_P( Size_MatType_NormType, norm, 
    testing::Combine(
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( (int)NORM_INF, (int)NORM_L1, (int)NORM_L2 )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int normType = std::tr1::get<2>(GetParam());

    Mat src1(sz, matType);
    double n;
    
    declare.in(src1, WARMUP_RNG);

    TEST_CYCLE(100) { n = norm(src1, normType); }
    
    SANITY_CHECK(n);
}


/*
// double norm(InputArray src1, int normType=NORM_L2, InputArray mask=noArray())
*/
PERF_TEST_P( Size_MatType_NormType, norm_mask, 
    testing::Combine(
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( (int)NORM_INF, (int)NORM_L1, (int)NORM_L2 )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int normType = std::tr1::get<2>(GetParam());

    Mat src1(sz, matType);
    Mat mask = Mat::ones(sz, CV_8U);
    double n;
    
    declare.in(src1, WARMUP_RNG).in(mask);
    
    TEST_CYCLE(100) { n = norm(src1, normType, mask); }
    
    SANITY_CHECK(n);
}


/*
// double norm(InputArray src1, InputArray src2, int normType)
*/
PERF_TEST_P( Size_MatType_NormType, norm2, 
    testing::Combine(
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( (int)NORM_INF, (int)NORM_L1, (int)NORM_L2, (int)(NORM_RELATIVE+NORM_INF), (int)(NORM_RELATIVE+NORM_L1), (int)(NORM_RELATIVE+NORM_L2) )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int normType = std::tr1::get<2>(GetParam());

    Mat src1(sz, matType);
    Mat src2(sz, matType);
    double n;
    
    declare.in(src1, src2, WARMUP_RNG);
    
    TEST_CYCLE(100) { n = norm(src1, src2, normType); }
    
    SANITY_CHECK(n);
}


/*
// double norm(InputArray src1, InputArray src2, int normType, InputArray mask=noArray())
*/
PERF_TEST_P( Size_MatType_NormType, norm2_mask,
    testing::Combine(
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( (int)NORM_INF, (int)NORM_L1, (int)NORM_L2, (int)(NORM_RELATIVE+NORM_INF), (int)(NORM_RELATIVE+NORM_L1), (int)(NORM_RELATIVE+NORM_L2) )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int normType = std::tr1::get<2>(GetParam());

    Mat src1(sz, matType);
    Mat src2(sz, matType);
    Mat mask = Mat::ones(sz, CV_8U);
    double n;
    
    declare.in(src1, src2, WARMUP_RNG).in(mask);
    
    TEST_CYCLE(100) { n = norm(src1, src2, normType, mask); }
    
    SANITY_CHECK(n);
}


/*
// void normalize(const InputArray src, OutputArray dst, double alpha=1, double beta=0, int normType=NORM_L2)
*/
PERF_TEST_P( Size_MatType_NormType, normalize, 
    testing::Combine(
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( (int)NORM_INF, (int)NORM_L1, (int)NORM_L2 )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int normType = std::tr1::get<2>(GetParam());

    Mat src(sz, matType);
    Mat dst(sz, matType);
    double alpha = 100.;
    if(normType==NORM_L1) alpha = (double)src.total() * src.channels();
    if(normType==NORM_L2) alpha = (double)src.total()/10;
    
    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE(100) { normalize(src, dst, alpha, 0., normType);  }
    
    SANITY_CHECK(dst);
}


/*
// void normalize(const InputArray src, OutputArray dst, double alpha=1, double beta=0, int normType=NORM_L2, int rtype=-1, InputArray mask=noArray())
*/
PERF_TEST_P( Size_MatType_NormType, normalize_mask, 
    testing::Combine(
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( (int)NORM_INF, (int)NORM_L1, (int)NORM_L2 )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int normType = std::tr1::get<2>(GetParam());

    Mat src(sz, matType);
    Mat dst(sz, matType);
    Mat mask = Mat::ones(sz, CV_8U);
    double alpha = 100.;
    if(normType==NORM_L1) alpha = (double)src.total() * src.channels();
    if(normType==NORM_L2) alpha = (double)src.total()/10;
    
    declare.in(src, WARMUP_RNG).in(mask).out(dst);
    
    TEST_CYCLE(100) { normalize(src, dst, alpha, 0., normType, -1, mask);  }
    
    SANITY_CHECK(dst);
}


/*
// void normalize(const InputArray src, OutputArray dst, double alpha=1, double beta=0, int normType=NORM_L2, int rtype=-1)
*/
PERF_TEST_P( Size_MatType_NormType, normalize_32f, 
    testing::Combine(
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( (int)NORM_INF, (int)NORM_L1, (int)NORM_L2 )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int normType = std::tr1::get<2>(GetParam());

    Mat src(sz, matType);
    Mat dst(sz, matType);
    double alpha = 100.;
    if(normType==NORM_L1) alpha = (double)src.total() * src.channels();
    if(normType==NORM_L2) alpha = (double)src.total()/10;
    
    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE(100) { normalize(src, dst, alpha, 0., normType, CV_32F);  }
    
    SANITY_CHECK(dst);
}


/*
// void normalize(const InputArray src, OutputArray dst, double alpha=1, double beta=0, int normType=NORM_L2)
*/
PERF_TEST_P( Size_MatType, normalize_minmax, TYPICAL_MATS )
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());

    Mat src(sz, matType);
    randu(src, 0, 256);
    Mat dst(sz, matType);
    
    declare.in(src).out(dst);
    
    TEST_CYCLE(100) { normalize(src, dst, 20., 100., NORM_MINMAX);  }
    
    SANITY_CHECK(dst);
}


/*
// void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev)
*/
PERF_TEST_P( Size_MatType, meanStdDev, TYPICAL_MATS )
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());

    Mat src(sz, matType);
    Mat mean, dev;

    declare.in(src, WARMUP_RNG);

    TEST_CYCLE(100) { meanStdDev(src, mean, dev);  }
    
    SANITY_CHECK(mean);
    SANITY_CHECK(dev);
}


/*
// void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev, InputArray mask=noArray())
*/
PERF_TEST_P( Size_MatType, meanStdDev_mask, TYPICAL_MATS )
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());

    Mat src(sz, matType);
    Mat mask = Mat::ones(sz, CV_8U);
    Mat mean, dev;

    declare.in(src, WARMUP_RNG).in(mask);
    
    TEST_CYCLE(100) { meanStdDev(src, mean, dev, mask);  }
    
    SANITY_CHECK(mean);
    SANITY_CHECK(dev);
}


/*
// int countNonZero(InputArray mtx)
*/
PERF_TEST_P( Size_MatType, countNonZero, TYPICAL_MATS_C1 )
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());

    Mat src(sz, matType);
    int cnt = 0;

    declare.in(src, WARMUP_RNG);
    
    TEST_CYCLE(100) { cnt = countNonZero(src);  }
    
    SANITY_CHECK(cnt);
}

/*
// void minMaxLoc(InputArray src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())
*/
PERF_TEST_P( Size_MatType, minMaxLoc, testing::Combine(
                 testing::Values( TYPICAL_MAT_SIZES ),
                 testing::Values( CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1,  CV_32FC1, CV_64FC1 ) ) )
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());

    Mat src(sz, matType);
    double minVal, maxVal;
    Point minLoc, maxLoc;

    // avoid early exit on 1 byte data
    if (matType == CV_8U)
        randu(src, 1, 254);
    else if (matType == CV_8S)
        randu(src, -127, 126);
    else
        warmup(src, WARMUP_RNG);

    declare.in(src);
    
    TEST_CYCLE(100) { minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);  }
    
    SANITY_CHECK(minVal);
    SANITY_CHECK(maxVal);
}



CV_ENUM(ROp, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)
typedef std::tr1::tuple<Size, MatType, ROp> Size_MatType_ROp_t;
typedef perf::TestBaseWithParam<Size_MatType_ROp_t> Size_MatType_ROp;


/*
// void reduce(InputArray mtx, OutputArray vec, int dim, int reduceOp, int dtype=-1)
*/
PERF_TEST_P( Size_MatType_ROp, reduceR, 
    testing::Combine( 
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int reduceOp = std::tr1::get<2>(GetParam());

    int ddepth = -1;
    if( CV_MAT_DEPTH(matType)< CV_32S && (reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG) )
        ddepth = CV_32S;
    Mat src(sz, matType);
    Mat vec;

    declare.in(src, WARMUP_RNG);
    
    TEST_CYCLE(100) { reduce(src, vec, 0, reduceOp, ddepth);  }
    
    SANITY_CHECK(vec);
}

/*
// void reduce(InputArray mtx, OutputArray vec, int dim, int reduceOp, int dtype=-1)
*/
PERF_TEST_P( Size_MatType_ROp, reduceC, 
    testing::Combine( 
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( TYPICAL_MAT_TYPES ),
        testing::Values( CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int matType = std::tr1::get<1>(GetParam());
    int reduceOp = std::tr1::get<2>(GetParam());

    int ddepth = -1;
    if( CV_MAT_DEPTH(matType)< CV_32S && (reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG) )
        ddepth = CV_32S;
    Mat src(sz, matType);
    Mat vec;

    declare.in(src, WARMUP_RNG);
    
    TEST_CYCLE(100) { reduce(src, vec, 1, reduceOp, ddepth);  }
    
    SANITY_CHECK(vec);
}
