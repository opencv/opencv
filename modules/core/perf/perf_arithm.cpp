#include "perf_precomp.hpp"
#include "opencv2\core\core_c.h"
using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

#define TYPICAL_MAT_SIZES_CORE_ARITHM   TYPICAL_MAT_SIZES 
#define TYPICAL_MAT_TYPES_CORE_ARITHM   CV_8UC1, CV_8SC1, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_8UC4, CV_32SC4, CV_32FC4
#define TYPICAL_MATS_CORE_ARITHM        testing::Combine( testing::Values( TYPICAL_MAT_SIZES_CORE_ARITHM ), testing::Values( TYPICAL_MAT_TYPES_CORE_ARITHM ) )

#ifdef ANDROID
PERF_TEST(convert, cvRound)
{
    double number = theRNG().uniform(-100, 100);

    int result = 0;

    TEST_CYCLE_N(1000)
    {
        for (int i = 0; i < 500000; ++i)
            result += cvRound(number);
    }

    SANITY_CHECK(result);
}
#endif

PERF_TEST_P(Size_MatType, min, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() min(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, minScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() min(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, max, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() max(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, maxScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() max(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, absdiff, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() absdiff(a, b, c);

    //see ticket 1529: absdiff can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}
/*
PERF_TEST_P(Size_MatType, absdiffs32, TYPICAL_MATS_CORE_ARITHM )
{

    TEST_CYCLE()
    {
        absdiff(src, src, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffu8, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_8UC1)

    TEST_CYCLE()
    {
        absdiff(src, src, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdifff32, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_32FC1)   

    TEST_CYCLE()
    {
        absdiff(src, src, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVu8, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_8UC1)
    uchar value = 64;

    TEST_CYCLE()
    {
        absdiff(src, value, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVs32, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_32SC1);
    int value = 64;

    TEST_CYCLE()
    {
        absdiff(src, value, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVf32, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_32FC1);
    float value = 64;

    TEST_CYCLE()
    {
        absdiff(src, value, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVc4s32, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_32SC4);
    int r, g, b, alpha;
    r = g = b = 64;
    alpha = 255;
    Scalar color(r, g, b, alpha);

    TEST_CYCLE()
    {
        absdiff(src, color, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVc4f32, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_32FC4);
    float r, g, b, alpha;
    r = g = b = 64;
    alpha = 255;
    Scalar color(r, g, b, alpha);

    TEST_CYCLE()
    {
        absdiff(src, color, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVc3u8, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_8UC3);
    unsigned int r, g, b;
    r = g = b = 64;
    Scalar color(r, g, b);

    TEST_CYCLE()
    {
        absdiff(src, color, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVc3s32, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_32SC3);
    int r, g, b;
    r = g = b = 64;
    Scalar color(r, g, b);

    TEST_CYCLE()
    {
        absdiff(src, color, dst);
    }

    dumpImage(dst);
}

PERF_TEST_P( ImageSize, absdiffVc3f32, SIZE_LIST )
{
    DECLARATIONS_OCV_SIZE(CV_32FC3);
    float r, g, b;
    r = g = b = 64;
    Scalar color(r, g, b);

    TEST_CYCLE()
    {
        absdiff(src, color, dst);
    }

    dumpImage(dst);
}
*/
PERF_TEST_P( Size_MatType, absdiffVc4u8, TYPICAL_MATS_CORE_ARITHM )
{
//    DECLARATIONS_OCV_SIZE(CV_8UC4);
    unsigned int r, g, b, alpha;
    r = g = b = 64;
    alpha = 255;
    Scalar color(r, g, b, alpha);

    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
//    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, WARMUP_RNG).out(c);

    int64 startTicks1, endTicks1;
    int kap=0;
    startTicks1 = cvGetTickCount();

    TEST_CYCLE()
    {
        absdiff(a, color, c);
        kap++;
    }
    endTicks1 = cvGetTickCount();
    cv::Mat src = a;
    cv::Mat dst = c;
    printf(" 4 %f, ", double(endTicks1 - startTicks1)/kap/double(cvGetTickFrequency()));

		for(int j=0 ; j < 4*src.rows; j++ )
		for(int i=0 ; i < src.cols; i +=16 ){			
//            if(*((uchar*)src.data+j*src.cols+i)==115)
//                printf("\n i=%i j=%i", i,j);
            if(i==src.cols-16 &&(j==src.rows-1 || j==src.rows-6|| j==src.rows-7|| j==src.rows-8||j==src.rows-2 || j==src.rows-3|| j==src.rows-4|| j==src.rows-5)) {
				printf("\n j=%i i=%i dst=%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i ",j ,i
					, *((uchar*)c.data+ j*src.cols+ i+0), *((uchar*)c.data+j*src.cols+i+1), *((uchar*)c.data+j*src.cols+i+2), *((uchar*)c.data+j*src.cols+i+3),
					*((uchar*)c.data+j*src.cols+i+4), *((uchar*)c.data+j*src.cols+i+5), *((uchar*)c.data+j*src.cols+i+6), *((uchar*)c.data+i+j*src.cols+7),
					*((uchar*)c.data+j*src.cols+i+8), *((uchar*)c.data+j*src.cols+i+9), *((uchar*)c.data+j*src.cols+i+10), *((uchar*)c.data+i+j*src.cols+11),
					*((uchar*)c.data+j*src.cols+i+12), *((uchar*)c.data+j*src.cols+i+13), *((uchar*)c.data+j*src.cols+i+14), *((uchar*)c.data+i+j*src.cols+15));
				printf("\n src=%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i "
					, *((uchar*)src.data+j*src.cols+i), *((uchar*)src.data+j*src.cols+i+1), *((uchar*)src.data+j*src.cols+i+2), *((uchar*)src.data+j*src.cols+i+3),
					*((uchar*)src.data+j*src.cols+i+4), *((uchar*)src.data+j*src.cols+i+5), *((uchar*)src.data+j*src.cols+i+6), *((uchar*)src.data+j*src.cols+i+7), *((uchar*)src.data+j*src.cols+i+8),
					*((uchar*)src.data+j*src.cols+i+9), *((uchar*)src.data+j*src.cols+i+10), *((uchar*)src.data+j*src.cols+i+11), *((uchar*)src.data+j*src.cols+i+12), *((uchar*)src.data+j*src.cols+i+13),
					*((uchar*)src.data+j*src.cols+i+14), *((uchar*)src.data+j*src.cols+i+15), *((uchar*)src.data+j*src.cols+i+16));
			}
		}

//    dumpImage(c);
}

PERF_TEST_P(Size_MatType, absdiffScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() absdiff(a, b, c);

    //see ticket 1529: absdiff can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, add, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() add(a, b, c);

    //see ticket 1529: add can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, addScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() add(a, b, c);

    //see ticket 1529: add can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, subtract, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() subtract(a, b, c);

    //see ticket 1529: subtract can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, subtractScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() subtract(a, b, c);

    //see ticket 1529: subtract can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}
