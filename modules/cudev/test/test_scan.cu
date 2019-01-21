
#include "test_precomp.hpp"

using namespace cv;
using namespace cv::cudev;
using namespace cvtest;

// BlockScanInt

template <int THREADS_NUM>
__global__ void int_kernel(int* data)
{
    uint tid = Block::threadLineId();

#if CV_CUDEV_ARCH >= 300
    const int n_warps = (THREADS_NUM - 1) / WARP_SIZE + 1;
    __shared__ int smem[n_warps];
#else
    __shared__ int smem[THREADS_NUM];
#endif

    data[tid] = blockScanInclusive<THREADS_NUM>(data[tid], smem, tid);
}

#define BLOCK_SCAN_INT_TEST(block_size)                                 \
    TEST(BlockScanInt, BlockSize##block_size)                           \
    {                                                                   \
        Mat src = randomMat(Size(block_size, 1), CV_32SC1, 0, 1024);    \
                                                                        \
        GpuMat d_src;                                                   \
        d_src.upload(src);                                              \
                                                                        \
        for (int col = 1; col < block_size; col++)                      \
            src.at<int>(0, col) += src.at<int>(0, col - 1);             \
                                                                        \
        int_kernel<block_size><<<1, block_size>>>((int*)d_src.data);    \
                                                                        \
        CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());                    \
                                                                        \
        EXPECT_MAT_NEAR(d_src, src, 0);                                 \
    }

BLOCK_SCAN_INT_TEST(29)
BLOCK_SCAN_INT_TEST(30)
BLOCK_SCAN_INT_TEST(32)
BLOCK_SCAN_INT_TEST(40)
BLOCK_SCAN_INT_TEST(41)

BLOCK_SCAN_INT_TEST(59)
BLOCK_SCAN_INT_TEST(60)
BLOCK_SCAN_INT_TEST(64)
BLOCK_SCAN_INT_TEST(70)
BLOCK_SCAN_INT_TEST(71)

BLOCK_SCAN_INT_TEST(109)
BLOCK_SCAN_INT_TEST(110)
BLOCK_SCAN_INT_TEST(128)
BLOCK_SCAN_INT_TEST(130)
BLOCK_SCAN_INT_TEST(131)

BLOCK_SCAN_INT_TEST(189)
BLOCK_SCAN_INT_TEST(200)
BLOCK_SCAN_INT_TEST(256)
BLOCK_SCAN_INT_TEST(300)
BLOCK_SCAN_INT_TEST(311)

BLOCK_SCAN_INT_TEST(489)
BLOCK_SCAN_INT_TEST(500)
BLOCK_SCAN_INT_TEST(512)
BLOCK_SCAN_INT_TEST(600)
BLOCK_SCAN_INT_TEST(611)

BLOCK_SCAN_INT_TEST(1024)

// BlockScanDouble

template <int THREADS_NUM>
__global__ void double_kernel(double* data)
{
    uint tid = Block::threadLineId();

#if CV_CUDEV_ARCH >= 300
    const int n_warps = (THREADS_NUM - 1) / WARP_SIZE + 1;
    __shared__ double smem[n_warps];
#else
    __shared__ double smem[THREADS_NUM];
#endif

    data[tid] = blockScanInclusive<THREADS_NUM>(data[tid], smem, tid);
}

#define BLOCK_SCAN_DOUBLE_TEST(block_size)                                  \
    TEST(BlockScanDouble, BlockSize##block_size)                            \
    {                                                                       \
        Mat src = randomMat(Size(block_size, 1), CV_64FC1, 0.0, 1.0);       \
                                                                            \
        GpuMat d_src;                                                       \
        d_src.upload(src);                                                  \
                                                                            \
        for (int col = 1; col < block_size; col++)                          \
            src.at<double>(0, col) += src.at<double>(0, col - 1);           \
                                                                            \
        double_kernel<block_size><<<1, block_size>>>((double*)d_src.data);  \
                                                                            \
        CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());                        \
                                                                            \
        EXPECT_MAT_NEAR(d_src, src, 1e-10);                                 \
    }

BLOCK_SCAN_DOUBLE_TEST(29)
BLOCK_SCAN_DOUBLE_TEST(30)
BLOCK_SCAN_DOUBLE_TEST(32)
BLOCK_SCAN_DOUBLE_TEST(40)
BLOCK_SCAN_DOUBLE_TEST(41)

BLOCK_SCAN_DOUBLE_TEST(59)
BLOCK_SCAN_DOUBLE_TEST(60)
BLOCK_SCAN_DOUBLE_TEST(64)
BLOCK_SCAN_DOUBLE_TEST(70)
BLOCK_SCAN_DOUBLE_TEST(71)

BLOCK_SCAN_DOUBLE_TEST(109)
BLOCK_SCAN_DOUBLE_TEST(110)
BLOCK_SCAN_DOUBLE_TEST(128)
BLOCK_SCAN_DOUBLE_TEST(130)
BLOCK_SCAN_DOUBLE_TEST(131)

BLOCK_SCAN_DOUBLE_TEST(189)
BLOCK_SCAN_DOUBLE_TEST(200)
BLOCK_SCAN_DOUBLE_TEST(256)
BLOCK_SCAN_DOUBLE_TEST(300)
BLOCK_SCAN_DOUBLE_TEST(311)

BLOCK_SCAN_DOUBLE_TEST(489)
BLOCK_SCAN_DOUBLE_TEST(500)
BLOCK_SCAN_DOUBLE_TEST(512)
BLOCK_SCAN_DOUBLE_TEST(600)
BLOCK_SCAN_DOUBLE_TEST(611)

BLOCK_SCAN_DOUBLE_TEST(1024)
