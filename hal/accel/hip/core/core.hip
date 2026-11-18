// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../common.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

namespace {

const int DEPTH_SIZE[7] = { 1, 1, 2, 2, 4, 4, 8 };

template<typename T>
__global__ void setToKernel(unsigned char* dst, size_t dstep, int rows, int cols, T val)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
        *(T*)(dst + (size_t)y * dstep + (size_t)x * sizeof(T)) = val;
}

// OpenCV only guarantees a row step that is a multiple of elemSize1, so a ROI or user-wrapped
// buffer can be misaligned for a 8- or 16-byte vector type.
template<typename T>
__global__ void setToUnalignedKernel(unsigned char* dst, size_t dstep, int rows, int cols, T val)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
        hipCopyN<(int)sizeof(T)>(dst + (size_t)y * dstep + (size_t)x * sizeof(T),
                                 (const unsigned char*)&val);
}

template<typename T>
hipError_t setToT(unsigned char* dst, size_t dstep, int rows, int cols, const double* v, hipStream_t s)
{
    const T val = HipVecTraits<T>::make(v);
    if (hipAligned<T>(dst, dstep))
        hipLaunchKernelGGL(setToKernel<T>, grid(rows, cols), dim3(BLOCK_X, BLOCK_Y), 0, s,
                           dst, dstep, rows, cols, val);
    else
        hipLaunchKernelGGL(setToUnalignedKernel<T>, grid(rows, cols), dim3(BLOCK_X, BLOCK_Y), 0, s,
                           dst, dstep, rows, cols, val);
    return hipGetLastError();
}

template<typename T, typename D, typename S>
__global__ void convertScaleKernel(const unsigned char* src, size_t sstep,
                                   unsigned char* dst, size_t dstep,
                                   int rows, int cols, S alpha, S beta)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
        *(D*)(dst + (size_t)y * dstep + (size_t)x * sizeof(D)) =
            hip_saturate_cast<D>(alpha * static_cast<S>(*(const T*)(src + (size_t)y * sstep + (size_t)x * sizeof(T))) + beta);
}

template<typename T, typename D>
hipError_t convertScaleT(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                         int rows, int cols, double alpha, double beta, hipStream_t s)
{
    typedef typename HipWorkType<T, D>::type S;
    hipLaunchKernelGGL((convertScaleKernel<T, D, S>), grid(rows, cols), dim3(BLOCK_X, BLOCK_Y), 0, s,
                       src, sstep, dst, dstep, rows, cols,
                       hip_saturate_cast<S>(alpha), hip_saturate_cast<S>(beta));
    return hipGetLastError();
}

} // namespace

hipError_t setTo(unsigned char* dst, size_t dstep, int rows, int cols, int depth, int cn,
                 const double* val, hipStream_t s)
{
    if (!dst || !val || depth < 0 || depth > 6 || cn < 1 || cn > 4)
        return hipErrorInvalidValue;
    if (rows <= 0 || cols <= 0)
        return hipSuccess;

    // hipMemset2D only writes a repeating byte.
    const size_t widthBytes = (size_t)cols * DEPTH_SIZE[depth] * cn;
    if (val[0] == 0.0 && val[1] == 0.0 && val[2] == 0.0 && val[3] == 0.0)
        return hipMemset2DAsync(dst, dstep, 0, widthBytes, rows, s);
    if (depth == 0)
    {
        bool uniform = true;
        for (int i = 1; i < cn; i++)
            uniform = uniform && (val[i] == val[0]);
        if (uniform)
            return hipMemset2DAsync(dst, dstep, hip_saturate_cast<unsigned char>(val[0]),
                                    widthBytes, rows, s);
    }

    typedef hipError_t (*func_t)(unsigned char*, size_t, int, int, const double*, hipStream_t);
    static const func_t funcs[7][4] = {
        {setToT<unsigned char>,  setToT<uchar2>,  setToT<uchar3>,  setToT<uchar4>},
        {setToT<signed char>,    setToT<char2>,   setToT<char3>,   setToT<char4>},
        {setToT<unsigned short>, setToT<ushort2>, setToT<ushort3>, setToT<ushort4>},
        {setToT<short>,          setToT<short2>,  setToT<short3>,  setToT<short4>},
        {setToT<int>,            setToT<int2>,    setToT<int3>,    setToT<int4>},
        {setToT<float>,          setToT<float2>,  setToT<float3>,  setToT<float4>},
        {setToT<double>,         setToT<double2>, setToT<double3>, setToT<double4>}
    };
    return funcs[depth][cn - 1](dst, dstep, rows, cols, val, s);
}

hipError_t convertToScale(const unsigned char* src, size_t sstep, int sdepth,
                          unsigned char* dst, size_t dstep, int ddepth,
                          int rows, int cols, int cn, double alpha, double beta, hipStream_t s)
{
    if (!src || !dst || sdepth < 0 || sdepth > 6 || ddepth < 0 || ddepth > 6 || cn < 1)
        return hipErrorInvalidValue;
    if (rows <= 0 || cols <= 0)
        return hipSuccess;

    typedef hipError_t (*func_t)(const unsigned char*, size_t, unsigned char*, size_t,
                                 int, int, double, double, hipStream_t);
    #define CV_ACCEL_HIP_CVT_ROW(T) \
        {convertScaleT<T, unsigned char>, convertScaleT<T, signed char>, \
         convertScaleT<T, unsigned short>, convertScaleT<T, short>, \
         convertScaleT<T, int>, convertScaleT<T, float>, convertScaleT<T, double>}
    static const func_t funcs[7][7] = {
        CV_ACCEL_HIP_CVT_ROW(unsigned char),
        CV_ACCEL_HIP_CVT_ROW(signed char),
        CV_ACCEL_HIP_CVT_ROW(unsigned short),
        CV_ACCEL_HIP_CVT_ROW(short),
        CV_ACCEL_HIP_CVT_ROW(int),
        CV_ACCEL_HIP_CVT_ROW(float),
        CV_ACCEL_HIP_CVT_ROW(double)
    };
    #undef CV_ACCEL_HIP_CVT_ROW

    // convertTo preserves the channel count; flatten channels into the width.
    return funcs[sdepth][ddepth](src, sstep, dst, dstep, rows, cols * cn, alpha, beta, s);
}

}}}} // namespace cv::accel::hip_impl::kernels
