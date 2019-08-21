/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <stdio.h>
#include <cuda_runtime.h>

#include "opencv2/core/cuda/common.hpp"

#include "opencv2/cudalegacy/NCV.hpp"
#include "opencv2/cudalegacy/NCVPyramid.hpp"

#include "NCVAlg.hpp"
#include "NCVPixelOperations.hpp"

template<typename T, Ncv32u CN> struct __average4_CN {static __host__ __device__ T _average4_CN(const T &p00, const T &p01, const T &p10, const T &p11);};

template<typename T> struct __average4_CN<T, 1> {
static __host__ __device__ T _average4_CN(const T &p00, const T &p01, const T &p10, const T &p11)
{
    T out;
    out.x = ((Ncv32s)p00.x + p01.x + p10.x + p11.x + 2) / 4;
    return out;
}};

template<> struct __average4_CN<float1, 1> {
static __host__ __device__ float1 _average4_CN(const float1 &p00, const float1 &p01, const float1 &p10, const float1 &p11)
{
    float1 out;
    out.x = (p00.x + p01.x + p10.x + p11.x) / 4;
    return out;
}};

template<> struct __average4_CN<double1, 1> {
static __host__ __device__ double1 _average4_CN(const double1 &p00, const double1 &p01, const double1 &p10, const double1 &p11)
{
    double1 out;
    out.x = (p00.x + p01.x + p10.x + p11.x) / 4;
    return out;
}};

template<typename T> struct __average4_CN<T, 3> {
static __host__ __device__ T _average4_CN(const T &p00, const T &p01, const T &p10, const T &p11)
{
    T out;
    out.x = ((Ncv32s)p00.x + p01.x + p10.x + p11.x + 2) / 4;
    out.y = ((Ncv32s)p00.y + p01.y + p10.y + p11.y + 2) / 4;
    out.z = ((Ncv32s)p00.z + p01.z + p10.z + p11.z + 2) / 4;
    return out;
}};

template<> struct __average4_CN<float3, 3> {
static __host__ __device__ float3 _average4_CN(const float3 &p00, const float3 &p01, const float3 &p10, const float3 &p11)
{
    float3 out;
    out.x = (p00.x + p01.x + p10.x + p11.x) / 4;
    out.y = (p00.y + p01.y + p10.y + p11.y) / 4;
    out.z = (p00.z + p01.z + p10.z + p11.z) / 4;
    return out;
}};

template<> struct __average4_CN<double3, 3> {
static __host__ __device__ double3 _average4_CN(const double3 &p00, const double3 &p01, const double3 &p10, const double3 &p11)
{
    double3 out;
    out.x = (p00.x + p01.x + p10.x + p11.x) / 4;
    out.y = (p00.y + p01.y + p10.y + p11.y) / 4;
    out.z = (p00.z + p01.z + p10.z + p11.z) / 4;
    return out;
}};

template<typename T> struct __average4_CN<T, 4> {
static __host__ __device__ T _average4_CN(const T &p00, const T &p01, const T &p10, const T &p11)
{
    T out;
    out.x = ((Ncv32s)p00.x + p01.x + p10.x + p11.x + 2) / 4;
    out.y = ((Ncv32s)p00.y + p01.y + p10.y + p11.y + 2) / 4;
    out.z = ((Ncv32s)p00.z + p01.z + p10.z + p11.z + 2) / 4;
    out.w = ((Ncv32s)p00.w + p01.w + p10.w + p11.w + 2) / 4;
    return out;
}};

template<> struct __average4_CN<float4, 4> {
static __host__ __device__ float4 _average4_CN(const float4 &p00, const float4 &p01, const float4 &p10, const float4 &p11)
{
    float4 out;
    out.x = (p00.x + p01.x + p10.x + p11.x) / 4;
    out.y = (p00.y + p01.y + p10.y + p11.y) / 4;
    out.z = (p00.z + p01.z + p10.z + p11.z) / 4;
    out.w = (p00.w + p01.w + p10.w + p11.w) / 4;
    return out;
}};

template<> struct __average4_CN<double4, 4> {
static __host__ __device__ double4 _average4_CN(const double4 &p00, const double4 &p01, const double4 &p10, const double4 &p11)
{
    double4 out;
    out.x = (p00.x + p01.x + p10.x + p11.x) / 4;
    out.y = (p00.y + p01.y + p10.y + p11.y) / 4;
    out.z = (p00.z + p01.z + p10.z + p11.z) / 4;
    out.w = (p00.w + p01.w + p10.w + p11.w) / 4;
    return out;
}};

template<typename T> static __host__ __device__ T _average4(const T &p00, const T &p01, const T &p10, const T &p11)
{
    return __average4_CN<T, NC(T)>::_average4_CN(p00, p01, p10, p11);
}


template<typename Tin, typename Tout, Ncv32u CN> struct __lerp_CN {static __host__ __device__ Tout _lerp_CN(const Tin &a, const Tin &b, Ncv32f d);};

template<typename Tin, typename Tout> struct __lerp_CN<Tin, Tout, 1> {
static __host__ __device__ Tout _lerp_CN(const Tin &a, const Tin &b, Ncv32f d)
{
    typedef typename TConvVec2Base<Tout>::TBase TB;
    return _pixMake(TB(b.x * d + a.x * (1 - d)));
}};

template<typename Tin, typename Tout> struct __lerp_CN<Tin, Tout, 3> {
static __host__ __device__ Tout _lerp_CN(const Tin &a, const Tin &b, Ncv32f d)
{
    typedef typename TConvVec2Base<Tout>::TBase TB;
    return _pixMake(TB(b.x * d + a.x * (1 - d)),
                    TB(b.y * d + a.y * (1 - d)),
                    TB(b.z * d + a.z * (1 - d)));
}};

template<typename Tin, typename Tout> struct __lerp_CN<Tin, Tout, 4> {
static __host__ __device__ Tout _lerp_CN(const Tin &a, const Tin &b, Ncv32f d)
{
    typedef typename TConvVec2Base<Tout>::TBase TB;
    return _pixMake(TB(b.x * d + a.x * (1 - d)),
                    TB(b.y * d + a.y * (1 - d)),
                    TB(b.z * d + a.z * (1 - d)),
                    TB(b.w * d + a.w * (1 - d)));
}};

template<typename Tin, typename Tout> static __host__ __device__ Tout _lerp(const Tin &a, const Tin &b, Ncv32f d)
{
    return __lerp_CN<Tin, Tout, NC(Tin)>::_lerp_CN(a, b, d);
}


template<typename T>
__global__ void kernelDownsampleX2(T *d_src,
                                   Ncv32u srcPitch,
                                   T *d_dst,
                                   Ncv32u dstPitch,
                                   NcvSize32u dstRoi)
{
    Ncv32u i = blockIdx.y * blockDim.y + threadIdx.y;
    Ncv32u j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dstRoi.height && j < dstRoi.width)
    {
        T *d_src_line1 = (T *)((Ncv8u *)d_src + (2 * i + 0) * srcPitch);
        T *d_src_line2 = (T *)((Ncv8u *)d_src + (2 * i + 1) * srcPitch);
        T *d_dst_line = (T *)((Ncv8u *)d_dst + i * dstPitch);

        T p00 = d_src_line1[2*j+0];
        T p01 = d_src_line1[2*j+1];
        T p10 = d_src_line2[2*j+0];
        T p11 = d_src_line2[2*j+1];

        d_dst_line[j] = _average4(p00, p01, p10, p11);
    }
}

namespace cv { namespace cuda { namespace device
{
    namespace pyramid
    {
        template <typename T> void kernelDownsampleX2_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
        {
            dim3 bDim(16, 8);
            dim3 gDim(divUp(src.cols, bDim.x), divUp(src.rows, bDim.y));

            kernelDownsampleX2<<<gDim, bDim, 0, stream>>>((T*)src.data, static_cast<Ncv32u>(src.step),
                (T*)dst.data, static_cast<Ncv32u>(dst.step), NcvSize32u(dst.cols, dst.rows));

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void downsampleX2(PtrStepSzb src, PtrStepSzb dst, int depth, int cn, cudaStream_t stream)
        {
            typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

            static const func_t funcs[6][4] =
            {
                {kernelDownsampleX2_gpu<uchar1>       , 0 /*kernelDownsampleX2_gpu<uchar2>*/ , kernelDownsampleX2_gpu<uchar3>      , kernelDownsampleX2_gpu<uchar4>      },
                {0 /*kernelDownsampleX2_gpu<char1>*/  , 0 /*kernelDownsampleX2_gpu<char2>*/  , 0 /*kernelDownsampleX2_gpu<char3>*/ , 0 /*kernelDownsampleX2_gpu<char4>*/ },
                {kernelDownsampleX2_gpu<ushort1>      , 0 /*kernelDownsampleX2_gpu<ushort2>*/, kernelDownsampleX2_gpu<ushort3>     , kernelDownsampleX2_gpu<ushort4>     },
                {0 /*kernelDownsampleX2_gpu<short1>*/ , 0 /*kernelDownsampleX2_gpu<short2>*/ , 0 /*kernelDownsampleX2_gpu<short3>*/, 0 /*kernelDownsampleX2_gpu<short4>*/},
                {0 /*kernelDownsampleX2_gpu<int1>*/   , 0 /*kernelDownsampleX2_gpu<int2>*/   , 0 /*kernelDownsampleX2_gpu<int3>*/  , 0 /*kernelDownsampleX2_gpu<int4>*/  },
                {kernelDownsampleX2_gpu<float1>       , 0 /*kernelDownsampleX2_gpu<float2>*/ , kernelDownsampleX2_gpu<float3>      , kernelDownsampleX2_gpu<float4>      }
            };

            const func_t func = funcs[depth][cn - 1];
            CV_Assert(func != 0);

            func(src, dst, stream);
        }
    }
}}}




template<typename T>
__global__ void kernelInterpolateFrom1(T *d_srcTop,
                                       Ncv32u srcTopPitch,
                                       NcvSize32u szTopRoi,
                                       T *d_dst,
                                       Ncv32u dstPitch,
                                       NcvSize32u dstRoi)
{
    Ncv32u i = blockIdx.y * blockDim.y + threadIdx.y;
    Ncv32u j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dstRoi.height && j < dstRoi.width)
    {
        Ncv32f ptTopX = 1.0f * (szTopRoi.width - 1) * j / (dstRoi.width - 1);
        Ncv32f ptTopY = 1.0f * (szTopRoi.height - 1) * i / (dstRoi.height - 1);
        Ncv32u xl = (Ncv32u)ptTopX;
        Ncv32u xh = xl+1;
        Ncv32f dx = ptTopX - xl;
        Ncv32u yl = (Ncv32u)ptTopY;
        Ncv32u yh = yl+1;
        Ncv32f dy = ptTopY - yl;

        T *d_src_line1 = (T *)((Ncv8u *)d_srcTop + yl * srcTopPitch);
        T *d_src_line2 = (T *)((Ncv8u *)d_srcTop + yh * srcTopPitch);
        T *d_dst_line = (T *)((Ncv8u *)d_dst + i * dstPitch);

        T p00, p01, p10, p11;
        p00 = d_src_line1[xl];
        p01 = xh < szTopRoi.width ? d_src_line1[xh] : p00;
        p10 = yh < szTopRoi.height ? d_src_line2[xl] : p00;
        p11 = (xh < szTopRoi.width && yh < szTopRoi.height) ? d_src_line2[xh] : p00;
        typedef typename TConvBase2Vec<Ncv32f, NC(T)>::TVec TVFlt;
        TVFlt m_00_01 = _lerp<T, TVFlt>(p00, p01, dx);
        TVFlt m_10_11 = _lerp<T, TVFlt>(p10, p11, dx);
        TVFlt mixture = _lerp<TVFlt, TVFlt>(m_00_01, m_10_11, dy);
        T outPix = _pixDemoteClampZ<TVFlt, T>(mixture);

        d_dst_line[j] = outPix;
    }
}
namespace cv { namespace cuda { namespace device
{
    namespace pyramid
    {
        template <typename T> void kernelInterpolateFrom1_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
        {
            dim3 bDim(16, 8);
            dim3 gDim(divUp(dst.cols, bDim.x), divUp(dst.rows, bDim.y));

            kernelInterpolateFrom1<<<gDim, bDim, 0, stream>>>((T*) src.data, static_cast<Ncv32u>(src.step), NcvSize32u(src.cols, src.rows),
                (T*) dst.data, static_cast<Ncv32u>(dst.step), NcvSize32u(dst.cols, dst.rows));

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void interpolateFrom1(PtrStepSzb src, PtrStepSzb dst, int depth, int cn, cudaStream_t stream)
        {
            typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

            static const func_t funcs[6][4] =
            {
                {kernelInterpolateFrom1_gpu<uchar1>      , 0 /*kernelInterpolateFrom1_gpu<uchar2>*/ , kernelInterpolateFrom1_gpu<uchar3>      , kernelInterpolateFrom1_gpu<uchar4>      },
                {0 /*kernelInterpolateFrom1_gpu<char1>*/ , 0 /*kernelInterpolateFrom1_gpu<char2>*/  , 0 /*kernelInterpolateFrom1_gpu<char3>*/ , 0 /*kernelInterpolateFrom1_gpu<char4>*/ },
                {kernelInterpolateFrom1_gpu<ushort1>     , 0 /*kernelInterpolateFrom1_gpu<ushort2>*/, kernelInterpolateFrom1_gpu<ushort3>     , kernelInterpolateFrom1_gpu<ushort4>     },
                {0 /*kernelInterpolateFrom1_gpu<short1>*/, 0 /*kernelInterpolateFrom1_gpu<short2>*/ , 0 /*kernelInterpolateFrom1_gpu<short3>*/, 0 /*kernelInterpolateFrom1_gpu<short4>*/},
                {0 /*kernelInterpolateFrom1_gpu<int1>*/  , 0 /*kernelInterpolateFrom1_gpu<int2>*/   , 0 /*kernelInterpolateFrom1_gpu<int3>*/  , 0 /*kernelInterpolateFrom1_gpu<int4>*/  },
                {kernelInterpolateFrom1_gpu<float1>      , 0 /*kernelInterpolateFrom1_gpu<float2>*/ , kernelInterpolateFrom1_gpu<float3>      , kernelInterpolateFrom1_gpu<float4>      }
            };

            const func_t func = funcs[depth][cn - 1];
            CV_Assert(func != 0);

            func(src, dst, stream);
        }
    }
}}}


#if 0 //def _WIN32

template<typename T>
static T _interpLinear(const T &a, const T &b, Ncv32f d)
{
    typedef typename TConvBase2Vec<Ncv32f, NC(T)>::TVec TVFlt;
    TVFlt tmp = _lerp<T, TVFlt>(a, b, d);
    return _pixDemoteClampZ<TVFlt, T>(tmp);
}


template<typename T>
static T _interpBilinear(const NCVMatrix<T> &refLayer, Ncv32f x, Ncv32f y)
{
    Ncv32u xl = (Ncv32u)x;
    Ncv32u xh = xl+1;
    Ncv32f dx = x - xl;
    Ncv32u yl = (Ncv32u)y;
    Ncv32u yh = yl+1;
    Ncv32f dy = y - yl;
    T p00, p01, p10, p11;
    p00 = refLayer.at(xl, yl);
    p01 = xh < refLayer.width() ? refLayer.at(xh, yl) : p00;
    p10 = yh < refLayer.height() ? refLayer.at(xl, yh) : p00;
    p11 = (xh < refLayer.width() && yh < refLayer.height()) ? refLayer.at(xh, yh) : p00;
    typedef typename TConvBase2Vec<Ncv32f, NC(T)>::TVec TVFlt;
    TVFlt m_00_01 = _lerp<T, TVFlt>(p00, p01, dx);
    TVFlt m_10_11 = _lerp<T, TVFlt>(p10, p11, dx);
    TVFlt mixture = _lerp<TVFlt, TVFlt>(m_00_01, m_10_11, dy);
    return _pixDemoteClampZ<TVFlt, T>(mixture);
}

template <class T>
NCVImagePyramid<T>::NCVImagePyramid(const NCVMatrix<T> &img,
                                    Ncv8u numLayers,
                                    INCVMemAllocator &alloc,
                                    cudaStream_t cuStream)
{
    this->_isInitialized = false;
    ncvAssertPrintReturn(img.memType() == alloc.memType(), "NCVImagePyramid::ctor error", );

    this->layer0 = &img;
    NcvSize32u szLastLayer(img.width(), img.height());
    this->nLayers = 1;

    NCV_SET_SKIP_COND(alloc.isCounting());
    NcvBool bDeviceCode = alloc.memType() == NCVMemoryTypeDevice;

    if (numLayers == 0)
    {
        numLayers = 255; //it will cut-off when any of the dimensions goes 1
    }

#ifdef SELF_CHECK_GPU
    NCVMemNativeAllocator allocCPU(NCVMemoryTypeHostPinned, 512);
#endif

    for (Ncv32u i=0; i<(Ncv32u)numLayers-1; i++)
    {
        NcvSize32u szCurLayer(szLastLayer.width / 2, szLastLayer.height / 2);
        if (szCurLayer.width == 0 || szCurLayer.height == 0)
        {
            break;
        }

        this->pyramid.push_back(new NCVMatrixAlloc<T>(alloc, szCurLayer.width, szCurLayer.height));
        ncvAssertPrintReturn(((NCVMatrixAlloc<T> *)(this->pyramid[i]))->isMemAllocated(), "NCVImagePyramid::ctor error", );
        this->nLayers++;

        //fill in the layer
        NCV_SKIP_COND_BEGIN

        const NCVMatrix<T> *prevLayer = i == 0 ? this->layer0 : this->pyramid[i-1];
        NCVMatrix<T> *curLayer = this->pyramid[i];

        if (bDeviceCode)
        {
            dim3 bDim(16, 8);
            dim3 gDim(divUp(szCurLayer.width, bDim.x), divUp(szCurLayer.height, bDim.y));
            kernelDownsampleX2<<<gDim, bDim, 0, cuStream>>>(prevLayer->ptr(),
                                                            prevLayer->pitch(),
                                                            curLayer->ptr(),
                                                            curLayer->pitch(),
                                                            szCurLayer);
            ncvAssertPrintReturn(cudaSuccess == cudaGetLastError(), "NCVImagePyramid::ctor error", );

#ifdef SELF_CHECK_GPU
            NCVMatrixAlloc<T> h_prevLayer(allocCPU, prevLayer->width(), prevLayer->height());
            ncvAssertPrintReturn(h_prevLayer.isMemAllocated(), "Validation failure in NCVImagePyramid::ctor", );
            NCVMatrixAlloc<T> h_curLayer(allocCPU, curLayer->width(), curLayer->height());
            ncvAssertPrintReturn(h_curLayer.isMemAllocated(), "Validation failure in NCVImagePyramid::ctor", );
            ncvAssertPrintReturn(NCV_SUCCESS == prevLayer->copy2D(h_prevLayer, prevLayer->size(), cuStream), "Validation failure in NCVImagePyramid::ctor", );
            ncvAssertPrintReturn(NCV_SUCCESS == curLayer->copy2D(h_curLayer, curLayer->size(), cuStream), "Validation failure in NCVImagePyramid::ctor", );
            ncvAssertPrintReturn(cudaSuccess == cudaStreamSynchronize(cuStream), "Validation failure in NCVImagePyramid::ctor", );
            for (Ncv32u i=0; i<szCurLayer.height; i++)
            {
                for (Ncv32u j=0; j<szCurLayer.width; j++)
                {
                    T p00 = h_prevLayer.at(2*j+0, 2*i+0);
                    T p01 = h_prevLayer.at(2*j+1, 2*i+0);
                    T p10 = h_prevLayer.at(2*j+0, 2*i+1);
                    T p11 = h_prevLayer.at(2*j+1, 2*i+1);
                    T outGold = _average4(p00, p01, p10, p11);
                    T outGPU = h_curLayer.at(j, i);
                    ncvAssertPrintReturn(0 == memcmp(&outGold, &outGPU, sizeof(T)), "Validation failure in NCVImagePyramid::ctor with kernelDownsampleX2", );
                }
            }
#endif
        }
        else
        {
            for (Ncv32u i=0; i<szCurLayer.height; i++)
            {
                for (Ncv32u j=0; j<szCurLayer.width; j++)
                {
                    T p00 = prevLayer->at(2*j+0, 2*i+0);
                    T p01 = prevLayer->at(2*j+1, 2*i+0);
                    T p10 = prevLayer->at(2*j+0, 2*i+1);
                    T p11 = prevLayer->at(2*j+1, 2*i+1);
                    curLayer->at(j, i) = _average4(p00, p01, p10, p11);
                }
            }
        }

        NCV_SKIP_COND_END

        szLastLayer = szCurLayer;
    }

    this->_isInitialized = true;
}


template <class T>
NCVImagePyramid<T>::~NCVImagePyramid()
{
}


template <class T>
NcvBool NCVImagePyramid<T>::isInitialized() const
{
    return this->_isInitialized;
}


template <class T>
NCVStatus NCVImagePyramid<T>::getLayer(NCVMatrix<T> &outImg,
                                       NcvSize32u outRoi,
                                       NcvBool bTrilinear,
                                       cudaStream_t cuStream) const
{
    ncvAssertReturn(this->isInitialized(), NCV_UNKNOWN_ERROR);
    ncvAssertReturn(outImg.memType() == this->layer0->memType(), NCV_MEM_RESIDENCE_ERROR);
    ncvAssertReturn(outRoi.width <= this->layer0->width() && outRoi.height <= this->layer0->height() &&
                    outRoi.width > 0 && outRoi.height > 0, NCV_DIMENSIONS_INVALID);

    if (outRoi.width == this->layer0->width() && outRoi.height == this->layer0->height())
    {
        ncvAssertReturnNcvStat(this->layer0->copy2D(outImg, NcvSize32u(this->layer0->width(), this->layer0->height()), cuStream));
        return NCV_SUCCESS;
    }

    Ncv32f lastScale = 1.0f;
    Ncv32f curScale;
    const NCVMatrix<T> *lastLayer = this->layer0;
    const NCVMatrix<T> *curLayer = NULL;
    NcvBool bUse2Refs = false;

    for (Ncv32u i=0; i<this->nLayers-1; i++)
    {
        curScale = lastScale * 0.5f;
        curLayer = this->pyramid[i];

        if (outRoi.width == curLayer->width() && outRoi.height == curLayer->height())
        {
            ncvAssertReturnNcvStat(this->pyramid[i]->copy2D(outImg, NcvSize32u(this->pyramid[i]->width(), this->pyramid[i]->height()), cuStream));
            return NCV_SUCCESS;
        }

        if (outRoi.width >= curLayer->width() && outRoi.height >= curLayer->height())
        {
            if (outRoi.width < lastLayer->width() && outRoi.height < lastLayer->height())
            {
                bUse2Refs = true;
            }
            break;
        }

        lastScale = curScale;
        lastLayer = curLayer;
    }

    bUse2Refs = bUse2Refs && bTrilinear;

    NCV_SET_SKIP_COND(outImg.memType() == NCVMemoryTypeNone);
    NcvBool bDeviceCode = this->layer0->memType() == NCVMemoryTypeDevice;

#ifdef SELF_CHECK_GPU
    NCVMemNativeAllocator allocCPU(NCVMemoryTypeHostPinned, 512);
#endif

    NCV_SKIP_COND_BEGIN

    if (bDeviceCode)
    {
        ncvAssertReturn(bUse2Refs == false, NCV_NOT_IMPLEMENTED);

        dim3 bDim(16, 8);
        dim3 gDim(divUp(outRoi.width, bDim.x), divUp(outRoi.height, bDim.y));
        kernelInterpolateFrom1<<<gDim, bDim, 0, cuStream>>>(lastLayer->ptr(),
                                                            lastLayer->pitch(),
                                                            lastLayer->size(),
                                                            outImg.ptr(),
                                                            outImg.pitch(),
                                                            outRoi);
        ncvAssertCUDAReturn(cudaGetLastError(), NCV_CUDA_ERROR);

#ifdef SELF_CHECK_GPU
        ncvSafeMatAlloc(h_lastLayer, T, allocCPU, lastLayer->width(), lastLayer->height(), NCV_ALLOCATOR_BAD_ALLOC);
        ncvSafeMatAlloc(h_outImg, T, allocCPU, outImg.width(), outImg.height(), NCV_ALLOCATOR_BAD_ALLOC);
        ncvAssertReturnNcvStat(lastLayer->copy2D(h_lastLayer, lastLayer->size(), cuStream));
        ncvAssertReturnNcvStat(outImg.copy2D(h_outImg, outRoi, cuStream));
        ncvAssertCUDAReturn(cudaStreamSynchronize(cuStream), NCV_CUDA_ERROR);

        for (Ncv32u i=0; i<outRoi.height; i++)
        {
            for (Ncv32u j=0; j<outRoi.width; j++)
            {
                NcvSize32u szTopLayer(lastLayer->width(), lastLayer->height());
                Ncv32f ptTopX = 1.0f * (szTopLayer.width - 1) * j / (outRoi.width - 1);
                Ncv32f ptTopY = 1.0f * (szTopLayer.height - 1) * i / (outRoi.height - 1);
                T outGold = _interpBilinear(h_lastLayer, ptTopX, ptTopY);
                ncvAssertPrintReturn(0 == memcmp(&outGold, &h_outImg.at(j,i), sizeof(T)), "Validation failure in NCVImagePyramid::ctor with kernelInterpolateFrom1", NCV_UNKNOWN_ERROR);
            }
        }
#endif
    }
    else
    {
        for (Ncv32u i=0; i<outRoi.height; i++)
        {
            for (Ncv32u j=0; j<outRoi.width; j++)
            {
                //top layer pixel (always exists)
                NcvSize32u szTopLayer(lastLayer->width(), lastLayer->height());
                Ncv32f ptTopX = 1.0f * (szTopLayer.width - 1) * j / (outRoi.width - 1);
                Ncv32f ptTopY = 1.0f * (szTopLayer.height - 1) * i / (outRoi.height - 1);
                T topPix = _interpBilinear(*lastLayer, ptTopX, ptTopY);
                T trilinearPix = topPix;

                if (bUse2Refs)
                {
                    //bottom layer pixel (exists only if the requested scale is greater than the smallest layer scale)
                    NcvSize32u szBottomLayer(curLayer->width(), curLayer->height());
                    Ncv32f ptBottomX = 1.0f * (szBottomLayer.width - 1) * j / (outRoi.width - 1);
                    Ncv32f ptBottomY = 1.0f * (szBottomLayer.height - 1) * i / (outRoi.height - 1);
                    T bottomPix = _interpBilinear(*curLayer, ptBottomX, ptBottomY);

                    Ncv32f scale = (1.0f * outRoi.width / layer0->width() + 1.0f * outRoi.height / layer0->height()) / 2;
                    Ncv32f dl = (scale - curScale) / (lastScale - curScale);
                    dl = CLAMP(dl, 0.0f, 1.0f);
                    trilinearPix = _interpLinear(bottomPix, topPix, dl);
                }

                outImg.at(j, i) = trilinearPix;
            }
        }
    }

    NCV_SKIP_COND_END

    return NCV_SUCCESS;
}


template class NCVImagePyramid<uchar1>;
template class NCVImagePyramid<uchar3>;
template class NCVImagePyramid<uchar4>;
template class NCVImagePyramid<ushort1>;
template class NCVImagePyramid<ushort3>;
template class NCVImagePyramid<ushort4>;
template class NCVImagePyramid<uint1>;
template class NCVImagePyramid<uint3>;
template class NCVImagePyramid<uint4>;
template class NCVImagePyramid<float1>;
template class NCVImagePyramid<float3>;
template class NCVImagePyramid<float4>;

#endif //_WIN32
