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

#if !defined CUDA_DISABLER

#include <vector>
#include <cuda_runtime.h>
#include "NPP_staging.hpp"
#include "opencv2/gpu/device/warp.hpp"
#include "opencv2/gpu/device/warp_shuffle.hpp"


texture<Ncv8u,  1, cudaReadModeElementType> tex8u;
texture<Ncv32u, 1, cudaReadModeElementType> tex32u;
texture<uint2,  1, cudaReadModeElementType> tex64u;


//==============================================================================
//
// CUDA streams handling
//
//==============================================================================


static cudaStream_t nppStream = 0;


cudaStream_t nppStGetActiveCUDAstream(void)
{
    return nppStream;
}



cudaStream_t nppStSetActiveCUDAstream(cudaStream_t cudaStream)
{
    cudaStream_t tmp = nppStream;
    nppStream = cudaStream;
    return tmp;
}


//==============================================================================
//
// BlockScan.cuh
//
//==============================================================================


NCV_CT_ASSERT(K_WARP_SIZE == 32); //this is required for the manual unroll of the loop in warpScanInclusive


//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE and size is power of 2
template <class T>
inline __device__ T warpScanInclusive(T idata, volatile T *s_Data)
{
#if __CUDA_ARCH__ >= 300
    const unsigned int laneId = cv::gpu::device::Warp::laneId();

    // scan on shuffl functions
    #pragma unroll
    for (int i = 1; i <= (K_WARP_SIZE / 2); i *= 2)
    {
        const T n = cv::gpu::device::shfl_up(idata, i);
        if (laneId >= i)
              idata += n;
    }

    return idata;
#else
    Ncv32u pos = 2 * threadIdx.x - (threadIdx.x & (K_WARP_SIZE - 1));
    s_Data[pos] = 0;
    pos += K_WARP_SIZE;
    s_Data[pos] = idata;

    s_Data[pos] += s_Data[pos - 1];
    s_Data[pos] += s_Data[pos - 2];
    s_Data[pos] += s_Data[pos - 4];
    s_Data[pos] += s_Data[pos - 8];
    s_Data[pos] += s_Data[pos - 16];

    return s_Data[pos];
#endif
}
inline __device__ Ncv64u warpScanInclusive(Ncv64u idata, volatile Ncv64u *s_Data)
{
    Ncv32u pos = 2 * threadIdx.x - (threadIdx.x & (K_WARP_SIZE - 1));
    s_Data[pos] = 0;
    pos += K_WARP_SIZE;
    s_Data[pos] = idata;

    s_Data[pos] += s_Data[pos - 1];
    s_Data[pos] += s_Data[pos - 2];
    s_Data[pos] += s_Data[pos - 4];
    s_Data[pos] += s_Data[pos - 8];
    s_Data[pos] += s_Data[pos - 16];

    return s_Data[pos];
}


template <class T>
inline __device__ T warpScanExclusive(T idata, volatile T *s_Data)
{
    return warpScanInclusive(idata, s_Data) - idata;
}


template <class T, Ncv32u tiNumScanThreads>
inline __device__ T blockScanInclusive(T idata, volatile T *s_Data)
{
    if (tiNumScanThreads > K_WARP_SIZE)
    {
        //Bottom-level inclusive warp scan
        T warpResult = warpScanInclusive(idata, s_Data);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if( (threadIdx.x & (K_WARP_SIZE - 1)) == (K_WARP_SIZE - 1) )
        {
            s_Data[threadIdx.x >> K_LOG2_WARP_SIZE] = warpResult;
        }

        //wait for warp scans to complete
        __syncthreads();

        if( threadIdx.x < (tiNumScanThreads / K_WARP_SIZE) )
        {
            //grab top warp elements
            T val = s_Data[threadIdx.x];
            //calculate exclusive scan and write back to shared memory
            s_Data[threadIdx.x] = warpScanExclusive(val, s_Data);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        return warpResult + s_Data[threadIdx.x >> K_LOG2_WARP_SIZE];
    }
    else
    {
        return warpScanInclusive(idata, s_Data);
    }
}


//==============================================================================
//
// IntegralImage.cu
//
//==============================================================================


const Ncv32u NUM_SCAN_THREADS = 256;
const Ncv32u LOG2_NUM_SCAN_THREADS = 8;


template<class T_in, class T_out>
struct _scanElemOp
{
    template<bool tbDoSqr>
    static inline __host__ __device__ T_out scanElemOp(T_in elem)
    {
        return scanElemOp( elem, Int2Type<(int)tbDoSqr>() );
    }

private:

    template <int v> struct Int2Type { enum { value = v }; };

    static inline __host__ __device__ T_out scanElemOp(T_in elem, Int2Type<0>)
    {
        return (T_out)elem;
    }

    static inline __host__ __device__ T_out scanElemOp(T_in elem, Int2Type<1>)
    {
        return (T_out)(elem*elem);
    }
};


template<class T>
inline __device__ T readElem(T *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs);


template<>
inline __device__ Ncv8u readElem<Ncv8u>(Ncv8u *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs)
{
    return tex1Dfetch(tex8u, texOffs + srcStride * blockIdx.x + curElemOffs);
}


template<>
inline __device__ Ncv32u readElem<Ncv32u>(Ncv32u *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs)
{
    return d_src[curElemOffs];
}


template<>
inline __device__ Ncv32f readElem<Ncv32f>(Ncv32f *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs)
{
    return d_src[curElemOffs];
}


/**
* \brief Segmented scan kernel
*
* Calculates per-row prefix scans of the input image.
* Out-of-bounds safe: reads 'size' elements, writes 'size+1' elements
*
* \tparam T_in      Type of input image elements
* \tparam T_out     Type of output image elements
* \tparam T_op      Defines an operation to be performed on the input image pixels
*
* \param d_src      [IN] Source image pointer
* \param srcWidth   [IN] Source image width
* \param srcStride  [IN] Source image stride
* \param d_II       [OUT] Output image pointer
* \param IIstride   [IN] Output image stride
*
* \return None
*/
template <class T_in, class T_out, bool tbDoSqr>
__global__ void scanRows(T_in *d_src, Ncv32u texOffs, Ncv32u srcWidth, Ncv32u srcStride,
                         T_out *d_II, Ncv32u IIstride)
{
    //advance pointers to the current line
    if (sizeof(T_in) != 1)
    {
        d_src += srcStride * blockIdx.x;
    }
    //for initial image 8bit source we use texref tex8u
    d_II += IIstride * blockIdx.x;

    Ncv32u numBuckets = (srcWidth + NUM_SCAN_THREADS - 1) >> LOG2_NUM_SCAN_THREADS;
    Ncv32u offsetX = 0;

    __shared__ T_out shmem[NUM_SCAN_THREADS * 2];
    __shared__ T_out carryElem;
    if (threadIdx.x == 0)
        carryElem = 0;
    __syncthreads();

    while (numBuckets--)
    {
        Ncv32u curElemOffs = offsetX + threadIdx.x;
        T_out curScanElem;

        T_in curElem = 0;
        T_out curElemMod;

        if (curElemOffs < srcWidth)
        {
            //load elements
            curElem = readElem<T_in>(d_src, texOffs, srcStride, curElemOffs);
        }
        curElemMod = _scanElemOp<T_in, T_out>::scanElemOp<tbDoSqr>(curElem);

        //inclusive scan
        curScanElem = blockScanInclusive<T_out, NUM_SCAN_THREADS>(curElemMod, shmem);

        if (curElemOffs <= srcWidth)
        {
            //make scan exclusive and write the bucket to the output buffer
            d_II[curElemOffs] = carryElem + curScanElem - curElemMod;
            offsetX += NUM_SCAN_THREADS;
        }

        //remember last element for subsequent buckets adjustment
        __syncthreads();
        if (threadIdx.x == NUM_SCAN_THREADS-1)
        {
            carryElem += curScanElem;
        }
        __syncthreads();
    }

    if (offsetX == srcWidth && !threadIdx.x)
    {
        d_II[offsetX] = carryElem;
    }
}


template <bool tbDoSqr, class T_in, class T_out>
NCVStatus scanRowsWrapperDevice(T_in *d_src, Ncv32u srcStride,
                                T_out *d_dst, Ncv32u dstStride, NcvSize32u roi)
{
    cudaChannelFormatDesc cfdTex;
    size_t alignmentOffset = 0;
    if (sizeof(T_in) == 1)
    {
        cfdTex = cudaCreateChannelDesc<Ncv8u>();
        ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex8u, d_src, cfdTex, roi.height * srcStride), NPPST_TEXTURE_BIND_ERROR);
        if (alignmentOffset > 0)
        {
            ncvAssertCUDAReturn(cudaUnbindTexture(tex8u), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex8u, d_src, cfdTex, alignmentOffset + roi.height * srcStride), NPPST_TEXTURE_BIND_ERROR);
        }
    }
    scanRows
        <T_in, T_out, tbDoSqr>
        <<<roi.height, NUM_SCAN_THREADS, 0, nppStGetActiveCUDAstream()>>>
        (d_src, (Ncv32u)alignmentOffset, roi.width, srcStride, d_dst, dstStride);

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


static Ncv32u getPaddedDimension(Ncv32u dim, Ncv32u elemTypeSize, Ncv32u allocatorAlignment)
{
    Ncv32u alignMask = allocatorAlignment-1;
    Ncv32u inverseAlignMask = ~alignMask;
    Ncv32u dimBytes = dim * elemTypeSize;
    Ncv32u pitch = (dimBytes + alignMask) & inverseAlignMask;
    Ncv32u PaddedDim = pitch / elemTypeSize;
    return PaddedDim;
}


template <class T_in, class T_out>
NCVStatus ncvIntegralImage_device(T_in *d_src, Ncv32u srcStep,
                                  T_out *d_dst, Ncv32u dstStep, NcvSize32u roi,
                                  INCVMemAllocator &gpuAllocator)
{
    ncvAssertReturn(sizeof(T_out) == sizeof(Ncv32u), NPPST_MEM_INTERNAL_ERROR);
    ncvAssertReturn(gpuAllocator.memType() == NCVMemoryTypeDevice ||
                      gpuAllocator.memType() == NCVMemoryTypeNone, NPPST_MEM_RESIDENCE_ERROR);
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);
    ncvAssertReturn((d_src != NULL && d_dst != NULL) || gpuAllocator.isCounting(), NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roi.width > 0 && roi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStep >= roi.width * sizeof(T_in) &&
                      dstStep >= (roi.width + 1) * sizeof(T_out) &&
                      srcStep % sizeof(T_in) == 0 &&
                      dstStep % sizeof(T_out) == 0, NPPST_INVALID_STEP);
    srcStep /= sizeof(T_in);
    dstStep /= sizeof(T_out);

    Ncv32u WidthII = roi.width + 1;
    Ncv32u HeightII = roi.height + 1;
    Ncv32u PaddedWidthII32 = getPaddedDimension(WidthII, sizeof(Ncv32u), gpuAllocator.alignment());
    Ncv32u PaddedHeightII32 = getPaddedDimension(HeightII, sizeof(Ncv32u), gpuAllocator.alignment());

    NCVMatrixAlloc<T_out> Tmp32_1(gpuAllocator, PaddedWidthII32, PaddedHeightII32);
    ncvAssertReturn(gpuAllocator.isCounting() || Tmp32_1.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);
    NCVMatrixAlloc<T_out> Tmp32_2(gpuAllocator, PaddedHeightII32, PaddedWidthII32);
    ncvAssertReturn(gpuAllocator.isCounting() || Tmp32_2.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);
    ncvAssertReturn(Tmp32_1.pitch() * Tmp32_1.height() == Tmp32_2.pitch() * Tmp32_2.height(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat;
    NCV_SET_SKIP_COND(gpuAllocator.isCounting());

    NCV_SKIP_COND_BEGIN

    ncvStat = scanRowsWrapperDevice
        <false>
        (d_src, srcStep, Tmp32_1.ptr(), PaddedWidthII32, roi);
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = nppiStTranspose_32u_C1R((Ncv32u *)Tmp32_1.ptr(), PaddedWidthII32*sizeof(Ncv32u),
                                      (Ncv32u *)Tmp32_2.ptr(), PaddedHeightII32*sizeof(Ncv32u), NcvSize32u(WidthII, roi.height));
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = scanRowsWrapperDevice
        <false>
        (Tmp32_2.ptr(), PaddedHeightII32, Tmp32_1.ptr(), PaddedHeightII32, NcvSize32u(roi.height, WidthII));
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = nppiStTranspose_32u_C1R((Ncv32u *)Tmp32_1.ptr(), PaddedHeightII32*sizeof(Ncv32u),
                                      (Ncv32u *)d_dst, dstStep*sizeof(Ncv32u), NcvSize32u(HeightII, WidthII));
    ncvAssertReturnNcvStat(ncvStat);

    NCV_SKIP_COND_END

    return NPPST_SUCCESS;
}


NCVStatus ncvSquaredIntegralImage_device(Ncv8u *d_src, Ncv32u srcStep,
                                         Ncv64u *d_dst, Ncv32u dstStep, NcvSize32u roi,
                                         INCVMemAllocator &gpuAllocator)
{
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);
    ncvAssertReturn(gpuAllocator.memType() == NCVMemoryTypeDevice ||
                      gpuAllocator.memType() == NCVMemoryTypeNone, NPPST_MEM_RESIDENCE_ERROR);
    ncvAssertReturn((d_src != NULL && d_dst != NULL) || gpuAllocator.isCounting(), NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roi.width > 0 && roi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStep >= roi.width &&
                      dstStep >= (roi.width + 1) * sizeof(Ncv64u) &&
                      dstStep % sizeof(Ncv64u) == 0, NPPST_INVALID_STEP);
    dstStep /= sizeof(Ncv64u);

    Ncv32u WidthII = roi.width + 1;
    Ncv32u HeightII = roi.height + 1;
    Ncv32u PaddedWidthII32 = getPaddedDimension(WidthII, sizeof(Ncv32u), gpuAllocator.alignment());
    Ncv32u PaddedHeightII32 = getPaddedDimension(HeightII, sizeof(Ncv32u), gpuAllocator.alignment());
    Ncv32u PaddedWidthII64 = getPaddedDimension(WidthII, sizeof(Ncv64u), gpuAllocator.alignment());
    Ncv32u PaddedHeightII64 = getPaddedDimension(HeightII, sizeof(Ncv64u), gpuAllocator.alignment());
    Ncv32u PaddedWidthMax = PaddedWidthII32 > PaddedWidthII64 ? PaddedWidthII32 : PaddedWidthII64;
    Ncv32u PaddedHeightMax = PaddedHeightII32 > PaddedHeightII64 ? PaddedHeightII32 : PaddedHeightII64;

    NCVMatrixAlloc<Ncv32u> Tmp32_1(gpuAllocator, PaddedWidthII32, PaddedHeightII32);
    ncvAssertReturn(Tmp32_1.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);
    NCVMatrixAlloc<Ncv64u> Tmp64(gpuAllocator, PaddedWidthMax, PaddedHeightMax);
    ncvAssertReturn(Tmp64.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);

    NCVMatrixReuse<Ncv32u> Tmp32_2(Tmp64.getSegment(), gpuAllocator.alignment(), PaddedWidthII32, PaddedHeightII32);
    ncvAssertReturn(Tmp32_2.isMemReused(), NPPST_MEM_INTERNAL_ERROR);
    NCVMatrixReuse<Ncv64u> Tmp64_2(Tmp64.getSegment(), gpuAllocator.alignment(), PaddedWidthII64, PaddedHeightII64);
    ncvAssertReturn(Tmp64_2.isMemReused(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat;
    NCV_SET_SKIP_COND(gpuAllocator.isCounting());

    NCV_SKIP_COND_BEGIN

    ncvStat = scanRowsWrapperDevice
        <true, Ncv8u, Ncv32u>
        (d_src, srcStep, Tmp32_2.ptr(), PaddedWidthII32, roi);
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = nppiStTranspose_32u_C1R(Tmp32_2.ptr(), PaddedWidthII32*sizeof(Ncv32u),
                                      Tmp32_1.ptr(), PaddedHeightII32*sizeof(Ncv32u), NcvSize32u(WidthII, roi.height));
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = scanRowsWrapperDevice
        <false, Ncv32u, Ncv64u>
        (Tmp32_1.ptr(), PaddedHeightII32, Tmp64_2.ptr(), PaddedHeightII64, NcvSize32u(roi.height, WidthII));
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = nppiStTranspose_64u_C1R(Tmp64_2.ptr(), PaddedHeightII64*sizeof(Ncv64u),
                                      d_dst, dstStep*sizeof(Ncv64u), NcvSize32u(HeightII, WidthII));
    ncvAssertReturnNcvStat(ncvStat);

    NCV_SKIP_COND_END

    return NPPST_SUCCESS;
}


NCVStatus nppiStIntegralGetSize_8u32u(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp)
{
    ncvAssertReturn(pBufsize != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roiSize.width > 0 && roiSize.height > 0, NPPST_INVALID_ROI);

    NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertReturn(gpuCounter.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvIntegralImage_device((Ncv8u*)NULL, roiSize.width,
                                                  (Ncv32u*)NULL, (roiSize.width+1) * sizeof(Ncv32u),
                                                  roiSize, gpuCounter);
    ncvAssertReturnNcvStat(ncvStat);

    *pBufsize = (Ncv32u)gpuCounter.maxSize();
    return NPPST_SUCCESS;
}


NCVStatus nppiStIntegralGetSize_32f32f(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp)
{
    ncvAssertReturn(pBufsize != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roiSize.width > 0 && roiSize.height > 0, NPPST_INVALID_ROI);

    NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertReturn(gpuCounter.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvIntegralImage_device((Ncv32f*)NULL, roiSize.width * sizeof(Ncv32f),
                                                  (Ncv32f*)NULL, (roiSize.width+1) * sizeof(Ncv32f),
                                                  roiSize, gpuCounter);
    ncvAssertReturnNcvStat(ncvStat);

    *pBufsize = (Ncv32u)gpuCounter.maxSize();
    return NPPST_SUCCESS;
}


NCVStatus nppiStSqrIntegralGetSize_8u64u(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp)
{
    ncvAssertReturn(pBufsize != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roiSize.width > 0 && roiSize.height > 0, NPPST_INVALID_ROI);

    NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertReturn(gpuCounter.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvSquaredIntegralImage_device(NULL, roiSize.width,
                                                         NULL, (roiSize.width+1) * sizeof(Ncv64u),
                                                         roiSize, gpuCounter);
    ncvAssertReturnNcvStat(ncvStat);

    *pBufsize = (Ncv32u)gpuCounter.maxSize();
    return NPPST_SUCCESS;
}


NCVStatus nppiStIntegral_8u32u_C1R(Ncv8u *d_src, Ncv32u srcStep,
                                   Ncv32u *d_dst, Ncv32u dstStep,
                                   NcvSize32u roiSize, Ncv8u *pBuffer,
                                   Ncv32u bufSize, cudaDeviceProp &devProp)
{
    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(devProp.textureAlignment), pBuffer);
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvIntegralImage_device(d_src, srcStep, d_dst, dstStep, roiSize, gpuAllocator);
    ncvAssertReturnNcvStat(ncvStat);

    return NPPST_SUCCESS;
}


NCVStatus nppiStIntegral_32f32f_C1R(Ncv32f *d_src, Ncv32u srcStep,
                                    Ncv32f *d_dst, Ncv32u dstStep,
                                    NcvSize32u roiSize, Ncv8u *pBuffer,
                                    Ncv32u bufSize, cudaDeviceProp &devProp)
{
    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(devProp.textureAlignment), pBuffer);
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvIntegralImage_device(d_src, srcStep, d_dst, dstStep, roiSize, gpuAllocator);
    ncvAssertReturnNcvStat(ncvStat);

    return NPPST_SUCCESS;
}


NCVStatus nppiStSqrIntegral_8u64u_C1R(Ncv8u *d_src, Ncv32u srcStep,
                                      Ncv64u *d_dst, Ncv32u dstStep,
                                      NcvSize32u roiSize, Ncv8u *pBuffer,
                                      Ncv32u bufSize, cudaDeviceProp &devProp)
{
    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(devProp.textureAlignment), pBuffer);
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvSquaredIntegralImage_device(d_src, srcStep, d_dst, dstStep, roiSize, gpuAllocator);
    ncvAssertReturnNcvStat(ncvStat);

    return NPPST_SUCCESS;
}


NCVStatus nppiStIntegral_8u32u_C1R_host(Ncv8u *h_src, Ncv32u srcStep,
                                        Ncv32u *h_dst, Ncv32u dstStep,
                                        NcvSize32u roiSize)
{
    ncvAssertReturn(h_src != NULL && h_dst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roiSize.width > 0 && roiSize.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStep >= roiSize.width &&
                      dstStep >= (roiSize.width + 1) * sizeof(Ncv32u) &&
                      dstStep % sizeof(Ncv32u) == 0, NPPST_INVALID_STEP);
    dstStep /= sizeof(Ncv32u);

    Ncv32u WidthII = roiSize.width + 1;
    Ncv32u HeightII = roiSize.height + 1;

    memset(h_dst, 0, WidthII * sizeof(Ncv32u));
    for (Ncv32u i=1; i<HeightII; i++)
    {
        h_dst[i * dstStep] = 0;
        for (Ncv32u j=1; j<WidthII; j++)
        {
            Ncv32u top = h_dst[(i-1) * dstStep + j];
            Ncv32u left = h_dst[i * dstStep + (j - 1)];
            Ncv32u topleft = h_dst[(i - 1) * dstStep + (j - 1)];
            Ncv32u elem = h_src[(i - 1) * srcStep + (j - 1)];
            h_dst[i * dstStep + j] = elem + left - topleft + top;
        }
    }

    return NPPST_SUCCESS;
}


NCVStatus nppiStIntegral_32f32f_C1R_host(Ncv32f *h_src, Ncv32u srcStep,
                                         Ncv32f *h_dst, Ncv32u dstStep,
                                         NcvSize32u roiSize)
{
    ncvAssertReturn(h_src != NULL && h_dst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roiSize.width > 0 && roiSize.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStep >= roiSize.width * sizeof(Ncv32f) &&
                      dstStep >= (roiSize.width + 1) * sizeof(Ncv32f) &&
                      srcStep % sizeof(Ncv32f) == 0 &&
                      dstStep % sizeof(Ncv32f) == 0, NPPST_INVALID_STEP);
    srcStep /= sizeof(Ncv32f);
    dstStep /= sizeof(Ncv32f);

    Ncv32u WidthII = roiSize.width + 1;
    Ncv32u HeightII = roiSize.height + 1;

    memset(h_dst, 0, WidthII * sizeof(Ncv32u));
    for (Ncv32u i=1; i<HeightII; i++)
    {
        h_dst[i * dstStep] = 0.0f;
        for (Ncv32u j=1; j<WidthII; j++)
        {
            Ncv32f top = h_dst[(i-1) * dstStep + j];
            Ncv32f left = h_dst[i * dstStep + (j - 1)];
            Ncv32f topleft = h_dst[(i - 1) * dstStep + (j - 1)];
            Ncv32f elem = h_src[(i - 1) * srcStep + (j - 1)];
            h_dst[i * dstStep + j] = elem + left - topleft + top;
        }
    }

    return NPPST_SUCCESS;
}


NCVStatus nppiStSqrIntegral_8u64u_C1R_host(Ncv8u *h_src, Ncv32u srcStep,
                                           Ncv64u *h_dst, Ncv32u dstStep,
                                           NcvSize32u roiSize)
{
    ncvAssertReturn(h_src != NULL && h_dst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roiSize.width > 0 && roiSize.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStep >= roiSize.width &&
                      dstStep >= (roiSize.width + 1) * sizeof(Ncv64u) &&
                      dstStep % sizeof(Ncv64u) == 0, NPPST_INVALID_STEP);
    dstStep /= sizeof(Ncv64u);

    Ncv32u WidthII = roiSize.width + 1;
    Ncv32u HeightII = roiSize.height + 1;

    memset(h_dst, 0, WidthII * sizeof(Ncv64u));
    for (Ncv32u i=1; i<HeightII; i++)
    {
        h_dst[i * dstStep] = 0;
        for (Ncv32u j=1; j<WidthII; j++)
        {
            Ncv64u top = h_dst[(i-1) * dstStep + j];
            Ncv64u left = h_dst[i * dstStep + (j - 1)];
            Ncv64u topleft = h_dst[(i - 1) * dstStep + (j - 1)];
            Ncv64u elem = h_src[(i - 1) * srcStep + (j - 1)];
            h_dst[i * dstStep + j] = elem*elem + left - topleft + top;
        }
    }

    return NPPST_SUCCESS;
}


//==============================================================================
//
// Decimate.cu
//
//==============================================================================


const Ncv32u NUM_DOWNSAMPLE_NEAREST_THREADS_X = 32;
const Ncv32u NUM_DOWNSAMPLE_NEAREST_THREADS_Y = 8;


template<class T, NcvBool tbCacheTexture>
__device__ T getElem_Decimate(Ncv32u x, T *d_src);


template<>
__device__ Ncv32u getElem_Decimate<Ncv32u, true>(Ncv32u x, Ncv32u *d_src)
{
    return tex1Dfetch(tex32u, x);
}


template<>
__device__ Ncv32u getElem_Decimate<Ncv32u, false>(Ncv32u x, Ncv32u *d_src)
{
    return d_src[x];
}


template<>
__device__ Ncv64u getElem_Decimate<Ncv64u, true>(Ncv32u x, Ncv64u *d_src)
{
    uint2 tmp = tex1Dfetch(tex64u, x);
    Ncv64u res = (Ncv64u)tmp.y;
    res <<= 32;
    res |= tmp.x;
    return res;
}


template<>
__device__ Ncv64u getElem_Decimate<Ncv64u, false>(Ncv32u x, Ncv64u *d_src)
{
    return d_src[x];
}


template <class T, NcvBool tbCacheTexture>
__global__ void decimate_C1R(T *d_src, Ncv32u srcStep, T *d_dst, Ncv32u dstStep,
                                      NcvSize32u dstRoi, Ncv32u scale)
{
    int curX = blockIdx.x * blockDim.x + threadIdx.x;
    int curY = blockIdx.y * blockDim.y + threadIdx.y;

    if (curX >= dstRoi.width || curY >= dstRoi.height)
    {
        return;
    }

    d_dst[curY * dstStep + curX] = getElem_Decimate<T, tbCacheTexture>((curY * srcStep + curX) * scale, d_src);
}


template <class T>
static NCVStatus decimateWrapperDevice(T *d_src, Ncv32u srcStep,
                                                T *d_dst, Ncv32u dstStep,
                                                NcvSize32u srcRoi, Ncv32u scale,
                                                NcvBool readThruTexture)
{
    ncvAssertReturn(d_src != NULL && d_dst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(srcRoi.width > 0 && srcRoi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(scale != 0, NPPST_INVALID_SCALE);
    ncvAssertReturn(srcStep >= (Ncv32u)(srcRoi.width) * sizeof(T) &&
                      dstStep >= (Ncv32u)(srcRoi.width * sizeof(T) / scale), NPPST_INVALID_STEP);
    srcStep /= sizeof(T);
    dstStep /= sizeof(T);

    NcvSize32u dstRoi;
    dstRoi.width = srcRoi.width / scale;
    dstRoi.height = srcRoi.height / scale;

    dim3 grid((dstRoi.width + NUM_DOWNSAMPLE_NEAREST_THREADS_X - 1) / NUM_DOWNSAMPLE_NEAREST_THREADS_X,
              (dstRoi.height + NUM_DOWNSAMPLE_NEAREST_THREADS_Y - 1) / NUM_DOWNSAMPLE_NEAREST_THREADS_Y);
    dim3 block(NUM_DOWNSAMPLE_NEAREST_THREADS_X, NUM_DOWNSAMPLE_NEAREST_THREADS_Y);

    if (!readThruTexture)
    {
        decimate_C1R
            <T, false>
            <<<grid, block, 0, nppStGetActiveCUDAstream()>>>
            (d_src, srcStep, d_dst, dstStep, dstRoi, scale);
    }
    else
    {
        cudaChannelFormatDesc cfdTexSrc;

        if (sizeof(T) == sizeof(Ncv32u))
        {
            cfdTexSrc = cudaCreateChannelDesc<Ncv32u>();

            size_t alignmentOffset;
            ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex32u, d_src, cfdTexSrc, srcRoi.height * srcStep * sizeof(T)), NPPST_TEXTURE_BIND_ERROR);
            ncvAssertReturn(alignmentOffset==0, NPPST_TEXTURE_BIND_ERROR);
        }
        else
        {
            cfdTexSrc = cudaCreateChannelDesc<uint2>();

            size_t alignmentOffset;
            ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex64u, d_src, cfdTexSrc, srcRoi.height * srcStep * sizeof(T)), NPPST_TEXTURE_BIND_ERROR);
            ncvAssertReturn(alignmentOffset==0, NPPST_TEXTURE_BIND_ERROR);
        }

        decimate_C1R
            <T, true>
            <<<grid, block, 0, nppStGetActiveCUDAstream()>>>
            (d_src, srcStep, d_dst, dstStep, dstRoi, scale);
    }

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


template <class T>
static NCVStatus decimateWrapperHost(T *h_src, Ncv32u srcStep,
                                              T *h_dst, Ncv32u dstStep,
                                              NcvSize32u srcRoi, Ncv32u scale)
{
    ncvAssertReturn(h_src != NULL && h_dst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(srcRoi.width != 0 && srcRoi.height != 0, NPPST_INVALID_ROI);
    ncvAssertReturn(scale != 0, NPPST_INVALID_SCALE);
    ncvAssertReturn(srcStep >= (Ncv32u)(srcRoi.width) * sizeof(T) &&
                      dstStep >= (Ncv32u)(srcRoi.width * sizeof(T) / scale) &&
                      srcStep % sizeof(T) == 0 && dstStep % sizeof(T) == 0, NPPST_INVALID_STEP);
    srcStep /= sizeof(T);
    dstStep /= sizeof(T);

    NcvSize32u dstRoi;
    dstRoi.width = srcRoi.width / scale;
    dstRoi.height = srcRoi.height / scale;

    for (Ncv32u i=0; i<dstRoi.height; i++)
    {
        for (Ncv32u j=0; j<dstRoi.width; j++)
        {
            h_dst[i*dstStep+j] = h_src[i*scale*srcStep + j*scale];
        }
    }

    return NPPST_SUCCESS;
}


#define implementNppDecimate(bit, typ) \
    NCVStatus nppiStDecimate_##bit##typ##_C1R(Ncv##bit##typ *d_src, Ncv32u srcStep, \
                                                     Ncv##bit##typ *d_dst, Ncv32u dstStep, \
                                                     NcvSize32u srcRoi, Ncv32u scale, NcvBool readThruTexture) \
    { \
        return decimateWrapperDevice<Ncv##bit##u>((Ncv##bit##u *)d_src, srcStep, \
                                                           (Ncv##bit##u *)d_dst, dstStep, \
                                                           srcRoi, scale, readThruTexture); \
    }


#define implementNppDecimateHost(bit, typ) \
    NCVStatus nppiStDecimate_##bit##typ##_C1R_host(Ncv##bit##typ *h_src, Ncv32u srcStep, \
                                                          Ncv##bit##typ *h_dst, Ncv32u dstStep, \
                                                          NcvSize32u srcRoi, Ncv32u scale) \
    { \
        return decimateWrapperHost<Ncv##bit##u>((Ncv##bit##u *)h_src, srcStep, \
                                                         (Ncv##bit##u *)h_dst, dstStep, \
                                                         srcRoi, scale); \
    }


implementNppDecimate(32, u)
implementNppDecimate(32, s)
implementNppDecimate(32, f)
implementNppDecimate(64, u)
implementNppDecimate(64, s)
implementNppDecimate(64, f)
implementNppDecimateHost(32, u)
implementNppDecimateHost(32, s)
implementNppDecimateHost(32, f)
implementNppDecimateHost(64, u)
implementNppDecimateHost(64, s)
implementNppDecimateHost(64, f)


//==============================================================================
//
// RectStdDev.cu
//
//==============================================================================


const Ncv32u NUM_RECTSTDDEV_THREADS = 128;


template <NcvBool tbCacheTexture>
__device__ Ncv32u getElemSum(Ncv32u x, Ncv32u *d_sum)
{
    if (tbCacheTexture)
    {
        return tex1Dfetch(tex32u, x);
    }
    else
    {
        return d_sum[x];
    }
}


template <NcvBool tbCacheTexture>
__device__ Ncv64u getElemSqSum(Ncv32u x, Ncv64u *d_sqsum)
{
    if (tbCacheTexture)
    {
        uint2 tmp = tex1Dfetch(tex64u, x);
        Ncv64u res = (Ncv64u)tmp.y;
        res <<= 32;
        res |= tmp.x;
        return res;
    }
    else
    {
        return d_sqsum[x];
    }
}


template <NcvBool tbCacheTexture>
__global__ void rectStdDev_32f_C1R(Ncv32u *d_sum, Ncv32u sumStep,
                                   Ncv64u *d_sqsum, Ncv32u sqsumStep,
                                   Ncv32f *d_norm, Ncv32u normStep,
                                   NcvSize32u roi, NcvRect32u rect, Ncv32f invRectArea)
{
    Ncv32u x_offs = blockIdx.x * NUM_RECTSTDDEV_THREADS + threadIdx.x;
    if (x_offs >= roi.width)
    {
        return;
    }

    Ncv32u sum_offset = blockIdx.y * sumStep + x_offs;
    Ncv32u sqsum_offset = blockIdx.y * sqsumStep + x_offs;

    //OPT: try swapping order (could change cache hit/miss ratio)
    Ncv32u sum_tl = getElemSum<tbCacheTexture>(sum_offset + rect.y * sumStep + rect.x, d_sum);
    Ncv32u sum_bl = getElemSum<tbCacheTexture>(sum_offset + (rect.y + rect.height) * sumStep + rect.x, d_sum);
    Ncv32u sum_tr = getElemSum<tbCacheTexture>(sum_offset + rect.y * sumStep + rect.x + rect.width, d_sum);
    Ncv32u sum_br = getElemSum<tbCacheTexture>(sum_offset + (rect.y + rect.height) * sumStep + rect.x + rect.width, d_sum);
    Ncv32u sum_val = sum_br + sum_tl - sum_tr - sum_bl;

    Ncv64u sqsum_tl, sqsum_bl, sqsum_tr, sqsum_br;
    sqsum_tl = getElemSqSum<tbCacheTexture>(sqsum_offset + rect.y * sqsumStep + rect.x, d_sqsum);
    sqsum_bl = getElemSqSum<tbCacheTexture>(sqsum_offset + (rect.y + rect.height) * sqsumStep + rect.x, d_sqsum);
    sqsum_tr = getElemSqSum<tbCacheTexture>(sqsum_offset + rect.y * sqsumStep + rect.x + rect.width, d_sqsum);
    sqsum_br = getElemSqSum<tbCacheTexture>(sqsum_offset + (rect.y + rect.height) * sqsumStep + rect.x + rect.width, d_sqsum);
    Ncv64u sqsum_val = sqsum_br + sqsum_tl - sqsum_tr - sqsum_bl;

    Ncv32f mean = sum_val * invRectArea;

    //////////////////////////////////////////////////////////////////////////
    // sqsum_val_res = sqsum_val / rectArea
    //////////////////////////////////////////////////////////////////////////

    Ncv32f sqsum_val_1 = __ull2float_rz(sqsum_val);
    Ncv64u sqsum_val_2 = __float2ull_rz(sqsum_val_1);
    Ncv64u sqsum_val_3 = sqsum_val - sqsum_val_2;
    Ncv32f sqsum_val_4 = __ull2float_rn(sqsum_val_3);
    sqsum_val_1 *= invRectArea;
    sqsum_val_4 *= invRectArea;
    Ncv32f sqsum_val_res = sqsum_val_1 + sqsum_val_4;

    //////////////////////////////////////////////////////////////////////////
    // variance = sqsum_val_res - mean * mean
    //////////////////////////////////////////////////////////////////////////

#if defined DISABLE_MAD_SELECTIVELY
    Ncv32f variance = sqsum_val_2 - __fmul_rn(mean, mean);
#else
    Ncv32f variance = sqsum_val_res - mean * mean;
#endif

    //////////////////////////////////////////////////////////////////////////
    // stddev = sqrtf(variance)
    //////////////////////////////////////////////////////////////////////////

    //Ncv32f stddev = sqrtf(variance);
    Ncv32f stddev = __fsqrt_rn(variance);

    d_norm[blockIdx.y * normStep + x_offs] = stddev;
}


NCVStatus nppiStRectStdDev_32f_C1R(Ncv32u *d_sum, Ncv32u sumStep,
                                   Ncv64u *d_sqsum, Ncv32u sqsumStep,
                                   Ncv32f *d_norm, Ncv32u normStep,
                                   NcvSize32u roi, NcvRect32u rect,
                                   Ncv32f scaleArea, NcvBool readThruTexture)
{
    ncvAssertReturn(d_sum != NULL && d_sqsum != NULL && d_norm != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roi.width > 0 && roi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(sumStep >= (Ncv32u)(roi.width + rect.x + rect.width - 1) * sizeof(Ncv32u) &&
                      sqsumStep >= (Ncv32u)(roi.width + rect.x + rect.width - 1) * sizeof(Ncv64u) &&
                      normStep >= (Ncv32u)roi.width * sizeof(Ncv32f) &&
                      sumStep % sizeof(Ncv32u) == 0 &&
                      sqsumStep % sizeof(Ncv64u) == 0 &&
                      normStep % sizeof(Ncv32f) == 0, NPPST_INVALID_STEP);
    ncvAssertReturn(scaleArea >= 1.0f, NPPST_INVALID_SCALE);
    sumStep /= sizeof(Ncv32u);
    sqsumStep /= sizeof(Ncv64u);
    normStep /= sizeof(Ncv32f);

    Ncv32f rectArea = rect.width * rect.height * scaleArea;
    Ncv32f invRectArea = 1.0f / rectArea;

    dim3 grid(((roi.width + NUM_RECTSTDDEV_THREADS - 1) / NUM_RECTSTDDEV_THREADS), roi.height);
    dim3 block(NUM_RECTSTDDEV_THREADS);

    if (!readThruTexture)
    {
        rectStdDev_32f_C1R
            <false>
            <<<grid, block, 0, nppStGetActiveCUDAstream()>>>
            (d_sum, sumStep, d_sqsum, sqsumStep, d_norm, normStep, roi, rect, invRectArea);
    }
    else
    {
        cudaChannelFormatDesc cfdTexSrc;
        cudaChannelFormatDesc cfdTexSqr;
        cfdTexSrc = cudaCreateChannelDesc<Ncv32u>();
        cfdTexSqr = cudaCreateChannelDesc<uint2>();

        size_t alignmentOffset;
        ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex32u, d_sum, cfdTexSrc, (roi.height + rect.y + rect.height) * sumStep * sizeof(Ncv32u)), NPPST_TEXTURE_BIND_ERROR);
        ncvAssertReturn(alignmentOffset==0, NPPST_TEXTURE_BIND_ERROR);
        ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex64u, d_sqsum, cfdTexSqr, (roi.height + rect.y + rect.height) * sqsumStep * sizeof(Ncv64u)), NPPST_TEXTURE_BIND_ERROR);
        ncvAssertReturn(alignmentOffset==0, NPPST_TEXTURE_BIND_ERROR);

        rectStdDev_32f_C1R
            <true>
            <<<grid, block, 0, nppStGetActiveCUDAstream()>>>
            (NULL, sumStep, NULL, sqsumStep, d_norm, normStep, roi, rect, invRectArea);
    }

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


NCVStatus nppiStRectStdDev_32f_C1R_host(Ncv32u *h_sum, Ncv32u sumStep,
                                        Ncv64u *h_sqsum, Ncv32u sqsumStep,
                                        Ncv32f *h_norm, Ncv32u normStep,
                                        NcvSize32u roi, NcvRect32u rect,
                                        Ncv32f scaleArea)
{
    ncvAssertReturn(h_sum != NULL && h_sqsum != NULL && h_norm != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roi.width > 0 && roi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(sumStep >= (Ncv32u)(roi.width + rect.x + rect.width - 1) * sizeof(Ncv32u) &&
                      sqsumStep >= (Ncv32u)(roi.width + rect.x + rect.width - 1) * sizeof(Ncv64u) &&
                      normStep >= (Ncv32u)roi.width * sizeof(Ncv32f) &&
                      sumStep % sizeof(Ncv32u) == 0 &&
                      sqsumStep % sizeof(Ncv64u) == 0 &&
                      normStep % sizeof(Ncv32f) == 0, NPPST_INVALID_STEP);
    ncvAssertReturn(scaleArea >= 1.0f, NPPST_INVALID_SCALE);
    sumStep /= sizeof(Ncv32u);
    sqsumStep /= sizeof(Ncv64u);
    normStep /= sizeof(Ncv32f);

    Ncv32f rectArea = rect.width * rect.height * scaleArea;
    Ncv32f invRectArea = 1.0f / rectArea;

    for (Ncv32u i=0; i<roi.height; i++)
    {
        for (Ncv32u j=0; j<roi.width; j++)
        {
            Ncv32u sum_offset = i * sumStep + j;
            Ncv32u sqsum_offset = i * sqsumStep + j;

            Ncv32u sum_tl = h_sum[sum_offset + rect.y * sumStep + rect.x];
            Ncv32u sum_bl = h_sum[sum_offset + (rect.y + rect.height) * sumStep + rect.x];
            Ncv32u sum_tr = h_sum[sum_offset + rect.y * sumStep + rect.x + rect.width];
            Ncv32u sum_br = h_sum[sum_offset + (rect.y + rect.height) * sumStep + rect.x + rect.width];
            Ncv64f sum_val = sum_br + sum_tl - sum_tr - sum_bl;

            Ncv64u sqsum_tl = h_sqsum[sqsum_offset + rect.y * sqsumStep + rect.x];
            Ncv64u sqsum_bl = h_sqsum[sqsum_offset + (rect.y + rect.height) * sqsumStep + rect.x];
            Ncv64u sqsum_tr = h_sqsum[sqsum_offset + rect.y * sqsumStep + rect.x + rect.width];
            Ncv64u sqsum_br = h_sqsum[sqsum_offset + (rect.y + rect.height) * sqsumStep + rect.x + rect.width];
            Ncv64f sqsum_val = (Ncv64f)(sqsum_br + sqsum_tl - sqsum_tr - sqsum_bl);

            Ncv64f mean = sum_val * invRectArea;
            Ncv64f sqsum_val_2 = sqsum_val / rectArea;
            Ncv64f variance = sqsum_val_2 - mean * mean;

            h_norm[i * normStep + j] = (Ncv32f)sqrt(variance);
        }
    }

    return NPPST_SUCCESS;
}


//==============================================================================
//
// Transpose.cu
//
//==============================================================================


const Ncv32u TRANSPOSE_TILE_DIM   = 16;
const Ncv32u TRANSPOSE_BLOCK_ROWS = 16;


/**
* \brief Matrix transpose kernel
*
* Calculates transpose of the input image
* \see TRANSPOSE_TILE_DIM
*
* \tparam T_in      Type of input image elements
* \tparam T_out     Type of output image elements
*
* \param d_src      [IN] Source image pointer
* \param srcStride  [IN] Source image stride
* \param d_dst      [OUT] Output image pointer
* \param dstStride  [IN] Output image stride
*
* \return None
*/
template <class T>
__global__ void transpose(T *d_src, Ncv32u srcStride,
                          T *d_dst, Ncv32u dstStride, NcvSize32u srcRoi)
{
    __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

    Ncv32u blockIdx_x, blockIdx_y;

    // do diagonal reordering
    if (gridDim.x == gridDim.y)
    {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else
    {
        Ncv32u bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    Ncv32u xIndex = blockIdx_x * TRANSPOSE_TILE_DIM + threadIdx.x;
    Ncv32u yIndex = blockIdx_y * TRANSPOSE_TILE_DIM + threadIdx.y;
    Ncv32u index_gmem = xIndex + yIndex * srcStride;

    if (xIndex < srcRoi.width)
    {
        for (Ncv32u i=0; i<TRANSPOSE_TILE_DIM; i+=TRANSPOSE_BLOCK_ROWS)
        {
            if (yIndex + i < srcRoi.height)
            {
                tile[threadIdx.y+i][threadIdx.x] = d_src[index_gmem+i*srcStride];
            }
        }
    }

    __syncthreads();

    xIndex = blockIdx_y * TRANSPOSE_TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TRANSPOSE_TILE_DIM + threadIdx.y;
    index_gmem = xIndex + yIndex * dstStride;

    if (xIndex < srcRoi.height)
    {
        for (Ncv32u i=0; i<TRANSPOSE_TILE_DIM; i+=TRANSPOSE_BLOCK_ROWS)
        {
            if (yIndex + i < srcRoi.width)
            {
                d_dst[index_gmem+i*dstStride] = tile[threadIdx.x][threadIdx.y+i];
            }
        }
    }
}


template <class T>
NCVStatus transposeWrapperDevice(T *d_src, Ncv32u srcStride,
                                   T *d_dst, Ncv32u dstStride, NcvSize32u srcRoi)
{
    ncvAssertReturn(d_src != NULL && d_dst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(srcRoi.width > 0 && srcRoi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStride >= srcRoi.width * sizeof(T) &&
                      dstStride >= srcRoi.height * sizeof(T) &&
                      srcStride % sizeof(T) == 0 && dstStride % sizeof(T) == 0, NPPST_INVALID_STEP);
    srcStride /= sizeof(T);
    dstStride /= sizeof(T);

    dim3 grid((srcRoi.width + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
              (srcRoi.height + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
    transpose
        <T>
        <<<grid, block, 0, nppStGetActiveCUDAstream()>>>
        (d_src, srcStride, d_dst, dstStride, srcRoi);
    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


template <class T>
static NCVStatus transposeWrapperHost(T *h_src, Ncv32u srcStride,
                                        T *h_dst, Ncv32u dstStride, NcvSize32u srcRoi)
{
    ncvAssertReturn(h_src != NULL && h_dst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(srcRoi.width > 0 && srcRoi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStride >= srcRoi.width * sizeof(T) &&
                      dstStride >= srcRoi.height * sizeof(T) &&
                      srcStride % sizeof(T) == 0 && dstStride % sizeof(T) == 0, NPPST_INVALID_STEP);
    srcStride /= sizeof(T);
    dstStride /= sizeof(T);

    for (Ncv32u i=0; i<srcRoi.height; i++)
    {
        for (Ncv32u j=0; j<srcRoi.width; j++)
        {
            h_dst[j*dstStride+i] = h_src[i*srcStride + j];
        }
    }

    return NPPST_SUCCESS;
}


#define implementNppTranspose(bit, typ) \
    NCVStatus nppiStTranspose_##bit##typ##_C1R(Ncv##bit##typ *d_src, Ncv32u srcStep, \
                                             Ncv##bit##typ *d_dst, Ncv32u dstStep, NcvSize32u srcRoi) \
    { \
        return transposeWrapperDevice<Ncv##bit##u>((Ncv##bit##u *)d_src, srcStep, \
                                                   (Ncv##bit##u *)d_dst, dstStep, srcRoi); \
    }


#define implementNppTransposeHost(bit, typ) \
    NCVStatus nppiStTranspose_##bit##typ##_C1R_host(Ncv##bit##typ *h_src, Ncv32u srcStep, \
                                                  Ncv##bit##typ *h_dst, Ncv32u dstStep, \
                                                  NcvSize32u srcRoi) \
    { \
        return transposeWrapperHost<Ncv##bit##u>((Ncv##bit##u *)h_src, srcStep, \
                                                 (Ncv##bit##u *)h_dst, dstStep, srcRoi); \
    }


implementNppTranspose(32,u)
implementNppTranspose(32,s)
implementNppTranspose(32,f)
implementNppTranspose(64,u)
implementNppTranspose(64,s)
implementNppTranspose(64,f)

implementNppTransposeHost(32,u)
implementNppTransposeHost(32,s)
implementNppTransposeHost(32,f)
implementNppTransposeHost(64,u)
implementNppTransposeHost(64,s)
implementNppTransposeHost(64,f)


NCVStatus nppiStTranspose_128_C1R(void *d_src, Ncv32u srcStep,
                                  void *d_dst, Ncv32u dstStep, NcvSize32u srcRoi)
{
    return transposeWrapperDevice<uint4>((uint4 *)d_src, srcStep, (uint4 *)d_dst, dstStep, srcRoi);
}


NCVStatus nppiStTranspose_128_C1R_host(void *d_src, Ncv32u srcStep,
                                       void *d_dst, Ncv32u dstStep, NcvSize32u srcRoi)
{
    return transposeWrapperHost<uint4>((uint4 *)d_src, srcStep, (uint4 *)d_dst, dstStep, srcRoi);
}


//==============================================================================
//
// Compact.cu
//
//==============================================================================


const Ncv32u NUM_REMOVE_THREADS = 256;


template <bool bRemove, bool bWritePartial>
__global__ void removePass1Scan(Ncv32u *d_src, Ncv32u srcLen,
                                Ncv32u *d_offsets, Ncv32u *d_blockSums,
                                Ncv32u elemRemove)
{
    Ncv32u blockId = blockIdx.y * 65535 + blockIdx.x;
    Ncv32u elemAddrIn = blockId * NUM_REMOVE_THREADS + threadIdx.x;

    if (elemAddrIn > srcLen + blockDim.x)
    {
        return;
    }

    __shared__ Ncv32u shmem[NUM_REMOVE_THREADS * 2];

    Ncv32u scanElem = 0;
    if (elemAddrIn < srcLen)
    {
        if (bRemove)
        {
            scanElem = (d_src[elemAddrIn] != elemRemove) ? 1 : 0;
        }
        else
        {
            scanElem = d_src[elemAddrIn];
        }
    }

    Ncv32u localScanInc = blockScanInclusive<Ncv32u, NUM_REMOVE_THREADS>(scanElem, shmem);
    __syncthreads();

    if (elemAddrIn < srcLen)
    {
        if (threadIdx.x == NUM_REMOVE_THREADS-1 && bWritePartial)
        {
            d_blockSums[blockId] = localScanInc;
        }

        if (bRemove)
        {
            d_offsets[elemAddrIn] = localScanInc - scanElem;
        }
        else
        {
            d_src[elemAddrIn] = localScanInc - scanElem;
        }
    }
}


__global__ void removePass2Adjust(Ncv32u *d_offsets, Ncv32u srcLen, Ncv32u *d_blockSums)
{
    Ncv32u blockId = blockIdx.y * 65535 + blockIdx.x;
    Ncv32u elemAddrIn = blockId * NUM_REMOVE_THREADS + threadIdx.x;
    if (elemAddrIn >= srcLen)
    {
        return;
    }

    __shared__ Ncv32u valOffs;
    valOffs = d_blockSums[blockId];
    __syncthreads();

    d_offsets[elemAddrIn] += valOffs;
}


__global__ void removePass3Compact(Ncv32u *d_src, Ncv32u srcLen,
                                   Ncv32u *d_offsets, Ncv32u *d_dst,
                                   Ncv32u elemRemove, Ncv32u *dstLenValue)
{
    Ncv32u blockId = blockIdx.y * 65535 + blockIdx.x;
    Ncv32u elemAddrIn = blockId * NUM_REMOVE_THREADS + threadIdx.x;
    if (elemAddrIn >= srcLen)
    {
        return;
    }

    Ncv32u elem = d_src[elemAddrIn];
    Ncv32u elemAddrOut = d_offsets[elemAddrIn];
    if (elem != elemRemove)
    {
        d_dst[elemAddrOut] = elem;
    }

    if (elemAddrIn == srcLen-1)
    {
        if (elem != elemRemove)
        {
            *dstLenValue = elemAddrOut + 1;
        }
        else
        {
            *dstLenValue = elemAddrOut;
        }
    }
}


NCVStatus compactVector_32u_device(Ncv32u *d_src, Ncv32u srcLen,
                                   Ncv32u *d_dst, Ncv32u *dstLenPinned,
                                   Ncv32u elemRemove,
                                   INCVMemAllocator &gpuAllocator)
{
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);
    ncvAssertReturn((d_src != NULL && d_dst != NULL) || gpuAllocator.isCounting(), NPPST_NULL_POINTER_ERROR);

    if (srcLen == 0)
    {
        if (dstLenPinned != NULL)
        {
            *dstLenPinned = 0;
        }
        return NPPST_SUCCESS;
    }

    std::vector<Ncv32u> partSumNums;
    std::vector<Ncv32u> partSumOffsets;
    Ncv32u partSumLastNum = srcLen;
    Ncv32u partSumLastOffs = 0;
    do
    {
        partSumNums.push_back(partSumLastNum);
        partSumOffsets.push_back(partSumLastOffs);

        Ncv32u curPartSumAlignedLength = alignUp(partSumLastNum * sizeof(Ncv32u),
                                                 gpuAllocator.alignment()) / sizeof(Ncv32u);
        partSumLastOffs += curPartSumAlignedLength;

        partSumLastNum = (partSumLastNum + NUM_REMOVE_THREADS - 1) / NUM_REMOVE_THREADS;
    }
    while (partSumLastNum>1);
    partSumNums.push_back(partSumLastNum);
    partSumOffsets.push_back(partSumLastOffs);

    NCVVectorAlloc<Ncv32u> d_hierSums(gpuAllocator, partSumLastOffs+1);
    ncvAssertReturn(gpuAllocator.isCounting() || d_hierSums.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);
    NCVVectorAlloc<Ncv32u> d_numDstElements(gpuAllocator, 1);
    ncvAssertReturn(gpuAllocator.isCounting() || d_numDstElements.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);

    NCV_SET_SKIP_COND(gpuAllocator.isCounting());
    NCV_SKIP_COND_BEGIN

    dim3 block(NUM_REMOVE_THREADS);

    //calculate zero-level partial sums for indices calculation
    if (partSumNums.size() > 2)
    {
        dim3 grid(partSumNums[1]);

        if (grid.x > 65535)
        {
            grid.y = (grid.x + 65534) / 65535;
            grid.x = 65535;
        }
        removePass1Scan
            <true, true>
            <<<grid, block, 0, nppStGetActiveCUDAstream()>>>
            (d_src, srcLen,
             d_hierSums.ptr(),
             d_hierSums.ptr() + partSumOffsets[1],
             elemRemove);

        ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

        //calculate hierarchical partial sums
        for (Ncv32u i=1; i<partSumNums.size()-1; i++)
        {
            dim3 grid_partial(partSumNums[i+1]);
            if (grid_partial.x > 65535)
            {
                grid_partial.y = (grid_partial.x + 65534) / 65535;
                grid_partial.x = 65535;
            }
            if (grid_partial.x != 1)
            {
                removePass1Scan
                    <false, true>
                    <<<grid_partial, block, 0, nppStGetActiveCUDAstream()>>>
                    (d_hierSums.ptr() + partSumOffsets[i],
                     partSumNums[i], NULL,
                     d_hierSums.ptr() + partSumOffsets[i+1],
                     0);
            }
            else
            {
                removePass1Scan
                    <false, false>
                    <<<grid_partial, block, 0, nppStGetActiveCUDAstream()>>>
                    (d_hierSums.ptr() + partSumOffsets[i],
                     partSumNums[i], NULL,
                     NULL,
                     0);
            }

            ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);
        }

        //adjust hierarchical partial sums
        for (Ncv32s i=(Ncv32s)partSumNums.size()-3; i>=0; i--)
        {
            dim3 grid_local(partSumNums[i+1]);
            if (grid_local.x > 65535)
            {
                grid_local.y = (grid_local.x + 65534) / 65535;
                grid_local.x = 65535;
            }
            removePass2Adjust
                <<<grid_local, block, 0, nppStGetActiveCUDAstream()>>>
                (d_hierSums.ptr() + partSumOffsets[i], partSumNums[i],
                 d_hierSums.ptr() + partSumOffsets[i+1]);

            ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);
        }
    }
    else
    {
        dim3 grid_local(partSumNums[1]);
        removePass1Scan
            <true, false>
            <<<grid_local, block, 0, nppStGetActiveCUDAstream()>>>
            (d_src, srcLen,
             d_hierSums.ptr(),
             NULL, elemRemove);

        ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);
    }

    //compact source vector using indices
    dim3 grid(partSumNums[1]);
    if (grid.x > 65535)
    {
        grid.y = (grid.x + 65534) / 65535;
        grid.x = 65535;
    }
    removePass3Compact
        <<<grid, block, 0, nppStGetActiveCUDAstream()>>>
        (d_src, srcLen, d_hierSums.ptr(), d_dst,
         elemRemove, d_numDstElements.ptr());

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    //get number of dst elements
    if (dstLenPinned != NULL)
    {
        ncvAssertCUDAReturn(cudaMemcpyAsync(dstLenPinned, d_numDstElements.ptr(), sizeof(Ncv32u),
                                              cudaMemcpyDeviceToHost, nppStGetActiveCUDAstream()), NPPST_MEM_RESIDENCE_ERROR);
        ncvAssertCUDAReturn(cudaStreamSynchronize(nppStGetActiveCUDAstream()), NPPST_MEM_RESIDENCE_ERROR);
    }

    NCV_SKIP_COND_END

    return NPPST_SUCCESS;
}


NCVStatus nppsStCompactGetSize_32u(Ncv32u srcLen, Ncv32u *pBufsize, cudaDeviceProp &devProp)
{
    ncvAssertReturn(pBufsize != NULL, NPPST_NULL_POINTER_ERROR);

    if (srcLen == 0)
    {
        *pBufsize = 0;
        return NPPST_SUCCESS;
    }

    NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertReturn(gpuCounter.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = compactVector_32u_device(NULL, srcLen, NULL, NULL, 0xC001C0DE,
                                                 gpuCounter);
    ncvAssertReturnNcvStat(ncvStat);

    *pBufsize = (Ncv32u)gpuCounter.maxSize();
    return NPPST_SUCCESS;
}


NCVStatus nppsStCompactGetSize_32s(Ncv32u srcLen, Ncv32u *pBufsize, cudaDeviceProp &devProp)
{
    return nppsStCompactGetSize_32u(srcLen, pBufsize, devProp);
}


NCVStatus nppsStCompactGetSize_32f(Ncv32u srcLen, Ncv32u *pBufsize, cudaDeviceProp &devProp)
{
    return nppsStCompactGetSize_32u(srcLen, pBufsize, devProp);
}


NCVStatus nppsStCompact_32u(Ncv32u *d_src, Ncv32u srcLen,
                            Ncv32u *d_dst, Ncv32u *p_dstLen,
                            Ncv32u elemRemove, Ncv8u *pBuffer,
                            Ncv32u bufSize, cudaDeviceProp &devProp)
{
    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(devProp.textureAlignment), pBuffer);
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = compactVector_32u_device(d_src, srcLen, d_dst, p_dstLen, elemRemove,
                                                 gpuAllocator);
    ncvAssertReturnNcvStat(ncvStat);

    return NPPST_SUCCESS;
}


NCVStatus nppsStCompact_32s(Ncv32s *d_src, Ncv32u srcLen,
                            Ncv32s *d_dst, Ncv32u *p_dstLen,
                            Ncv32s elemRemove, Ncv8u *pBuffer,
                            Ncv32u bufSize, cudaDeviceProp &devProp)
{
    return nppsStCompact_32u((Ncv32u *)d_src, srcLen, (Ncv32u *)d_dst, p_dstLen,
                             *(Ncv32u *)&elemRemove, pBuffer, bufSize, devProp);
}


#if defined __GNUC__ && __GNUC__ > 2 && __GNUC_MINOR__  > 4
typedef Ncv32u __attribute__((__may_alias__)) Ncv32u_a;
#else
typedef Ncv32u Ncv32u_a;
#endif

NCVStatus nppsStCompact_32f(Ncv32f *d_src, Ncv32u srcLen,
                            Ncv32f *d_dst, Ncv32u *p_dstLen,
                            Ncv32f elemRemove, Ncv8u *pBuffer,
                            Ncv32u bufSize, cudaDeviceProp &devProp)
{
    return nppsStCompact_32u((Ncv32u *)d_src, srcLen, (Ncv32u *)d_dst, p_dstLen,
                             *(Ncv32u_a *)&elemRemove, pBuffer, bufSize, devProp);
}

NCVStatus nppsStCompact_32u_host(Ncv32u *h_src, Ncv32u srcLen,
                                 Ncv32u *h_dst, Ncv32u *dstLen, Ncv32u elemRemove)
{
    ncvAssertReturn(h_src != NULL && h_dst != NULL, NPPST_NULL_POINTER_ERROR);

    if (srcLen == 0)
    {
        if (dstLen != NULL)
        {
            *dstLen = 0;
        }
        return NPPST_SUCCESS;
    }

    Ncv32u dstIndex = 0;
    for (Ncv32u srcIndex=0; srcIndex<srcLen; srcIndex++)
    {
        if (h_src[srcIndex] != elemRemove)
        {
            h_dst[dstIndex++] = h_src[srcIndex];
        }
    }

    if (dstLen != NULL)
    {
        *dstLen = dstIndex;
    }

    return NPPST_SUCCESS;
}


NCVStatus nppsStCompact_32s_host(Ncv32s *h_src, Ncv32u srcLen,
                                 Ncv32s *h_dst, Ncv32u *dstLen, Ncv32s elemRemove)
{
    return nppsStCompact_32u_host((Ncv32u *)h_src, srcLen, (Ncv32u *)h_dst, dstLen, *(Ncv32u_a *)&elemRemove);
}


NCVStatus nppsStCompact_32f_host(Ncv32f *h_src, Ncv32u srcLen,
                                 Ncv32f *h_dst, Ncv32u *dstLen, Ncv32f elemRemove)
{
    return nppsStCompact_32u_host((Ncv32u *)h_src, srcLen, (Ncv32u *)h_dst, dstLen, *(Ncv32u_a *)&elemRemove);
}

//==============================================================================
//
// Filter.cu
//
//==============================================================================


texture <float, 1, cudaReadModeElementType> texSrc;
texture <float, 1, cudaReadModeElementType> texKernel;


__forceinline__ __device__ float getValueMirrorRow(const int rowOffset,
                                                   int i,
                                                   int w)
{
    if (i < 0) i = 1 - i;
    if (i >= w) i = w + w - i - 1;
    return tex1Dfetch (texSrc, rowOffset + i);
}


__forceinline__ __device__ float getValueMirrorColumn(const int offset,
                                                      const int rowStep,
                                                      int j,
                                                      int h)
{
    if (j < 0) j = 1 - j;
    if (j >= h) j = h + h - j - 1;
    return tex1Dfetch (texSrc, offset + j * rowStep);
}


__global__ void FilterRowBorderMirror_32f_C1R(Ncv32u srcStep,
                                              Ncv32f *pDst,
                                              NcvSize32u dstSize,
                                              Ncv32u dstStep,
                                              NcvRect32u roi,
                                              Ncv32s nKernelSize,
                                              Ncv32s nAnchor,
                                              Ncv32f multiplier)
{
    // position within ROI
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix >= roi.width || iy >= roi.height)
    {
        return;
    }

    const int p = nKernelSize - nAnchor - 1;

    const int j = roi.y + iy;

    const int rowOffset = j * srcStep + roi.x;

    float sum = 0.0f;
    for (int m = 0; m < nKernelSize; ++m)
    {
        sum += getValueMirrorRow (rowOffset, ix + m - p, roi.width)
            * tex1Dfetch (texKernel, m);
    }

    pDst[iy * dstStep + ix] = sum * multiplier;
}


__global__ void FilterColumnBorderMirror_32f_C1R(Ncv32u srcStep,
                                                 Ncv32f *pDst,
                                                 NcvSize32u dstSize,
                                                 Ncv32u dstStep,
                                                 NcvRect32u roi,
                                                 Ncv32s nKernelSize,
                                                 Ncv32s nAnchor,
                                                 Ncv32f multiplier)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix >= roi.width || iy >= roi.height)
    {
        return;
    }

    const int p = nKernelSize - nAnchor - 1;
    const int i = roi.x + ix;
    const int offset = i + roi.y * srcStep;

    float sum = 0.0f;
    for (int m = 0; m < nKernelSize; ++m)
    {
        sum += getValueMirrorColumn (offset, srcStep, iy + m - p, roi.height)
            * tex1Dfetch (texKernel, m);
    }

    pDst[ix + iy * dstStep] = sum * multiplier;
}


NCVStatus nppiStFilterRowBorder_32f_C1R(const Ncv32f *pSrc,
                                        NcvSize32u srcSize,
                                        Ncv32u nSrcStep,
                                        Ncv32f *pDst,
                                        NcvSize32u dstSize,
                                        Ncv32u nDstStep,
                                        NcvRect32u oROI,
                                        NppStBorderType borderType,
                                        const Ncv32f *pKernel,
                                        Ncv32s nKernelSize,
                                        Ncv32s nAnchor,
                                        Ncv32f multiplier)
{
    ncvAssertReturn (pSrc != NULL &&
        pDst != NULL &&
        pKernel != NULL, NCV_NULL_PTR);

    ncvAssertReturn (oROI.width > 0 && oROI.height > 0, NPPST_INVALID_ROI);

    ncvAssertReturn (srcSize.width * sizeof (Ncv32f) <= nSrcStep &&
        dstSize.width * sizeof (Ncv32f) <= nDstStep &&
        oROI.width * sizeof (Ncv32f) <= nSrcStep &&
        oROI.width * sizeof (Ncv32f) <= nDstStep &&
        nSrcStep % sizeof (Ncv32f) == 0 &&
        nDstStep % sizeof (Ncv32f) == 0, NPPST_INVALID_STEP);

    Ncv32u srcStep = nSrcStep / sizeof (Ncv32f);
    Ncv32u dstStep = nDstStep / sizeof (Ncv32f);

    // adjust ROI size to be within source image
    if (oROI.x + oROI.width > srcSize.width)
    {
        oROI.width = srcSize.width - oROI.x;
    }

    if (oROI.y + oROI.height > srcSize.height)
    {
        oROI.height = srcSize.height - oROI.y;
    }

    cudaChannelFormatDesc floatChannel = cudaCreateChannelDesc <float> ();
    texSrc.normalized    = false;
    texKernel.normalized = false;

    cudaBindTexture (0, texSrc, pSrc, floatChannel, srcSize.height * nSrcStep);
    cudaBindTexture (0, texKernel, pKernel, floatChannel, nKernelSize * sizeof (Ncv32f));

    dim3 ctaSize (32, 6);
    dim3 gridSize ((oROI.width + ctaSize.x - 1) / ctaSize.x,
        (oROI.height + ctaSize.y - 1) / ctaSize.y);

    switch (borderType)
    {
    case nppStBorderNone:
        return NPPST_ERROR;
    case nppStBorderClamp:
        return NPPST_ERROR;
    case nppStBorderWrap:
        return NPPST_ERROR;
    case nppStBorderMirror:
        FilterRowBorderMirror_32f_C1R <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream ()>>>
            (srcStep, pDst, dstSize, dstStep, oROI, nKernelSize, nAnchor, multiplier);
        ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);
        break;
    default:
        return NPPST_ERROR;
    }

    return NPPST_SUCCESS;
}


NCVStatus nppiStFilterColumnBorder_32f_C1R(const Ncv32f *pSrc,
                                           NcvSize32u srcSize,
                                           Ncv32u nSrcStep,
                                           Ncv32f *pDst,
                                           NcvSize32u dstSize,
                                           Ncv32u nDstStep,
                                           NcvRect32u oROI,
                                           NppStBorderType borderType,
                                           const Ncv32f *pKernel,
                                           Ncv32s nKernelSize,
                                           Ncv32s nAnchor,
                                           Ncv32f multiplier)
{
    ncvAssertReturn (pSrc != NULL &&
        pDst != NULL &&
        pKernel != NULL, NCV_NULL_PTR);

    ncvAssertReturn (oROI.width > 0 && oROI.height > 0, NPPST_INVALID_ROI);

    ncvAssertReturn (srcSize.width * sizeof (Ncv32f) <= nSrcStep &&
        dstSize.width * sizeof (Ncv32f) <= nDstStep &&
        oROI.width * sizeof (Ncv32f) <= nSrcStep &&
        oROI.width * sizeof (Ncv32f) <= nDstStep &&
        nSrcStep % sizeof (Ncv32f) == 0 &&
        nDstStep % sizeof (Ncv32f) == 0, NPPST_INVALID_STEP);

    Ncv32u srcStep = nSrcStep / sizeof (Ncv32f);
    Ncv32u dstStep = nDstStep / sizeof (Ncv32f);

    // adjust ROI size to be within source image
    if (oROI.x + oROI.width > srcSize.width)
    {
        oROI.width = srcSize.width - oROI.x;
    }

    if (oROI.y + oROI.height > srcSize.height)
    {
        oROI.height = srcSize.height - oROI.y;
    }

    cudaChannelFormatDesc floatChannel = cudaCreateChannelDesc <float> ();
    texSrc.normalized    = false;
    texKernel.normalized = false;

    cudaBindTexture (0, texSrc, pSrc, floatChannel, srcSize.height * nSrcStep);
    cudaBindTexture (0, texKernel, pKernel, floatChannel, nKernelSize * sizeof (Ncv32f));

    dim3 ctaSize (32, 6);
    dim3 gridSize ((oROI.width + ctaSize.x - 1) / ctaSize.x,
        (oROI.height + ctaSize.y - 1) / ctaSize.y);

    switch (borderType)
    {
    case nppStBorderClamp:
        return NPPST_ERROR;
    case nppStBorderWrap:
        return NPPST_ERROR;
    case nppStBorderMirror:
        FilterColumnBorderMirror_32f_C1R <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream ()>>>
            (srcStep, pDst, dstSize, dstStep, oROI, nKernelSize, nAnchor, multiplier);
        ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);
        break;
    default:
        return NPPST_ERROR;
    }

    return NPPST_SUCCESS;
}


//==============================================================================
//
// FrameInterpolate.cu
//
//==============================================================================


inline Ncv32u iDivUp(Ncv32u num, Ncv32u denom)
{
    return (num + denom - 1)/denom;
}


texture<float, 2, cudaReadModeElementType> tex_src1;
texture<float, 2, cudaReadModeElementType> tex_src0;


__global__ void BlendFramesKernel(const float *u, const float *v,   // forward flow
                                  const float *ur, const float *vr, // backward flow
                                  const float *o0, const float *o1, // coverage masks
                                  int w, int h, int s,
                                  float theta, float *out)
{
    const int ix = threadIdx.x + blockDim.x * blockIdx.x;
    const int iy = threadIdx.y + blockDim.y * blockIdx.y;

    const int pos = ix + s * iy;

    if (ix >= w || iy >= h) return;

    float _u = u[pos];
    float _v = v[pos];

    float _ur = ur[pos];
    float _vr = vr[pos];

    float x = (float)ix + 0.5f;
    float y = (float)iy + 0.5f;
    bool b0 = o0[pos] > 1e-4f;
    bool b1 = o1[pos] > 1e-4f;

    if (b0 && b1)
    {
        // pixel is visible on both frames
        out[pos] = tex2D(tex_src0, x - _u * theta, y - _v * theta) * (1.0f - theta) +
            tex2D(tex_src1, x + _u * (1.0f - theta), y + _v * (1.0f - theta)) * theta;
    }
    else if (b0)
    {
        // visible on the first frame only
        out[pos] = tex2D(tex_src0, x - _u * theta, y - _v * theta);
    }
    else
    {
        // visible on the second frame only
        out[pos] = tex2D(tex_src1, x - _ur * (1.0f - theta), y - _vr * (1.0f - theta));
    }
}


NCVStatus BlendFrames(const Ncv32f *src0,
                      const Ncv32f *src1,
                      const Ncv32f *ufi,
                      const Ncv32f *vfi,
                      const Ncv32f *ubi,
                      const Ncv32f *vbi,
                      const Ncv32f *o1,
                      const Ncv32f *o2,
                      Ncv32u width,
                      Ncv32u height,
                      Ncv32u stride,
                      Ncv32f theta,
                      Ncv32f *out)
{
    tex_src1.addressMode[0] = cudaAddressModeClamp;
    tex_src1.addressMode[1] = cudaAddressModeClamp;
    tex_src1.filterMode = cudaFilterModeLinear;
    tex_src1.normalized = false;

    tex_src0.addressMode[0] = cudaAddressModeClamp;
    tex_src0.addressMode[1] = cudaAddressModeClamp;
    tex_src0.filterMode = cudaFilterModeLinear;
    tex_src0.normalized = false;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc <float> ();
    const Ncv32u pitch = stride * sizeof (float);
    ncvAssertCUDAReturn (cudaBindTexture2D (0, tex_src1, src1, desc, width, height, pitch), NPPST_TEXTURE_BIND_ERROR);
    ncvAssertCUDAReturn (cudaBindTexture2D (0, tex_src0, src0, desc, width, height, pitch), NPPST_TEXTURE_BIND_ERROR);

    dim3 threads (32, 4);
    dim3 blocks (iDivUp (width, threads.x), iDivUp (height, threads.y));

    BlendFramesKernel<<<blocks, threads, 0, nppStGetActiveCUDAstream ()>>>
        (ufi, vfi, ubi, vbi, o1, o2, width, height, stride, theta, out);

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


NCVStatus nppiStGetInterpolationBufferSize(NcvSize32u srcSize,
                                           Ncv32u nStep,
                                           Ncv32u *hpSize)
{
    NCVStatus status = NPPST_ERROR;
    status = nppiStVectorWarpGetBufferSize(srcSize, nStep, hpSize);
    return status;
}


NCVStatus nppiStInterpolateFrames(const NppStInterpolationState *pState)
{
    // check state validity
    ncvAssertReturn (pState->pSrcFrame0 != 0 &&
        pState->pSrcFrame1 != 0 &&
        pState->pFU != 0 &&
        pState->pFV != 0 &&
        pState->pBU != 0 &&
        pState->pBV != 0 &&
        pState->pNewFrame != 0 &&
        pState->ppBuffers[0] != 0 &&
        pState->ppBuffers[1] != 0 &&
        pState->ppBuffers[2] != 0 &&
        pState->ppBuffers[3] != 0 &&
        pState->ppBuffers[4] != 0 &&
        pState->ppBuffers[5] != 0, NPPST_NULL_POINTER_ERROR);

    ncvAssertReturn (pState->size.width  > 0 &&
        pState->size.height > 0, NPPST_ERROR);

    ncvAssertReturn (pState->nStep >= pState->size.width * sizeof (Ncv32f) &&
        pState->nStep > 0 &&
        pState->nStep % sizeof (Ncv32f) == 0,
        NPPST_INVALID_STEP);

    // change notation
    Ncv32f *cov0 = pState->ppBuffers[0];
    Ncv32f *cov1 = pState->ppBuffers[1];
    Ncv32f *fwdU = pState->ppBuffers[2]; // forward u
    Ncv32f *fwdV = pState->ppBuffers[3]; // forward v
    Ncv32f *bwdU = pState->ppBuffers[4]; // backward u
    Ncv32f *bwdV = pState->ppBuffers[5]; // backward v
    // warp flow
    ncvAssertReturnNcvStat (
        nppiStVectorWarp_PSF2x2_32f_C1 (pState->pFU,
        pState->size,
        pState->nStep,
        pState->pFU,
        pState->pFV,
        pState->nStep,
        cov0,
        pState->pos,
        fwdU) );
    ncvAssertReturnNcvStat (
        nppiStVectorWarp_PSF2x2_32f_C1 (pState->pFV,
        pState->size,
        pState->nStep,
        pState->pFU,
        pState->pFV,
        pState->nStep,
        cov0,
        pState->pos,
        fwdV) );
    // warp backward flow
    ncvAssertReturnNcvStat (
        nppiStVectorWarp_PSF2x2_32f_C1 (pState->pBU,
        pState->size,
        pState->nStep,
        pState->pBU,
        pState->pBV,
        pState->nStep,
        cov1,
        1.0f - pState->pos,
        bwdU) );
    ncvAssertReturnNcvStat (
        nppiStVectorWarp_PSF2x2_32f_C1 (pState->pBV,
        pState->size,
        pState->nStep,
        pState->pBU,
        pState->pBV,
        pState->nStep,
        cov1,
        1.0f - pState->pos,
        bwdU) );
    // interpolate frame
    ncvAssertReturnNcvStat (
        BlendFrames (pState->pSrcFrame0,
        pState->pSrcFrame1,
        fwdU,
        fwdV,
        bwdU,
        bwdV,
        cov0,
        cov1,
        pState->size.width,
        pState->size.height,
        pState->nStep / sizeof (Ncv32f),
        pState->pos,
        pState->pNewFrame) );

    return NPPST_SUCCESS;
}


//==============================================================================
//
// VectorWarpFrame.cu
//
//==============================================================================


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)

// FP32 atomic add
static __forceinline__ __device__ float _atomicAdd(float *addr, float val)
{
    float old = *addr, assumed;

    do {
        assumed = old;
        old = int_as_float(__iAtomicCAS((int*)addr,
              float_as_int(assumed),
              float_as_int(val+assumed)));
    } while( assumed!=old );

    return old;
}
#else
#define _atomicAdd atomicAdd
#endif


__global__ void ForwardWarpKernel_PSF2x2(const float *u,
                                         const float *v,
                                         const float *src,
                                         const int w,
                                         const int h,
                                         const int flow_stride,
                                         const int image_stride,
                                         const float time_scale,
                                         float *normalization_factor,
                                         float *dst)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;

    if (i >= h || j >= w) return;

    int flow_row_offset  = i * flow_stride;
    int image_row_offset = i * image_stride;

    //bottom left corner of a target pixel
    float cx = u[flow_row_offset + j] * time_scale + (float)j + 1.0f;
    float cy = v[flow_row_offset + j] * time_scale + (float)i + 1.0f;
    // pixel containing bottom left corner
    float px;
    float py;
    float dx = modff (cx, &px);
    float dy = modff (cy, &py);
    // target pixel integer coords
    int tx;
    int ty;
    tx = (int) px;
    ty = (int) py;
    float value = src[image_row_offset + j];
    float weight;
    // fill pixel containing bottom right corner
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = dx * dy;
        _atomicAdd (dst + ty * image_stride + tx, value * weight);
        _atomicAdd (normalization_factor + ty * image_stride + tx, weight);
    }

    // fill pixel containing bottom left corner
    tx -= 1;
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = (1.0f - dx) * dy;
        _atomicAdd (dst + ty * image_stride + tx, value * weight);
        _atomicAdd (normalization_factor + ty * image_stride + tx, weight);
    }

    // fill pixel containing upper left corner
    ty -= 1;
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = (1.0f - dx) * (1.0f - dy);
        _atomicAdd (dst + ty * image_stride + tx, value * weight);
        _atomicAdd (normalization_factor + ty * image_stride + tx, weight);
    }

    // fill pixel containing upper right corner
    tx += 1;
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = dx * (1.0f - dy);
        _atomicAdd (dst + ty * image_stride + tx, value * weight);
        _atomicAdd (normalization_factor + ty * image_stride + tx, weight);
    }
}


__global__ void ForwardWarpKernel_PSF1x1(const float *u,
                                         const float *v,
                                         const float *src,
                                         const int w,
                                         const int h,
                                         const int flow_stride,
                                         const int image_stride,
                                         const float time_scale,
                                         float *dst)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;

    if (i >= h || j >= w) return;

    int flow_row_offset = i * flow_stride;
    int image_row_offset = i * image_stride;

    float u_ = u[flow_row_offset + j];
    float v_ = v[flow_row_offset + j];

    //bottom left corner of target pixel
    float cx = u_ * time_scale + (float)j + 1.0f;
    float cy = v_ * time_scale + (float)i + 1.0f;
    // pixel containing bottom left corner
    int tx = __float2int_rn (cx);
    int ty = __float2int_rn (cy);

    float value = src[image_row_offset + j];
    // fill pixel
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        _atomicAdd (dst + ty * image_stride + tx, value);
    }
}


__global__ void NormalizeKernel(const float *normalization_factor, int w, int h, int s, float *image)
{
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= h || j >= w) return;

    const int pos = i * s + j;

    float scale = normalization_factor[pos];

    float invScale = (scale == 0.0f) ? 1.0f : (1.0f / scale);

    image[pos] *= invScale;
}


__global__ void MemsetKernel(const float value, int w, int h, float *image)
{
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= h || j >= w) return;

    const int pos = i * w + j;

    image[pos] = value;
}


NCVStatus nppiStVectorWarpGetBufferSize (NcvSize32u srcSize, Ncv32u nSrcStep, Ncv32u *hpSize)
{
    ncvAssertReturn (hpSize != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn (srcSize.width * sizeof (Ncv32f) <= nSrcStep,
        NPPST_INVALID_STEP);

    *hpSize = nSrcStep * srcSize.height;

    return NPPST_SUCCESS;
}


// does not require normalization
NCVStatus nppiStVectorWarp_PSF1x1_32f_C1(const Ncv32f *pSrc,
                                         NcvSize32u srcSize,
                                         Ncv32u nSrcStep,
                                         const Ncv32f *pU,
                                         const Ncv32f *pV,
                                         Ncv32u nVFStep,
                                         Ncv32f timeScale,
                                         Ncv32f *pDst)
{
    ncvAssertReturn (pSrc != NULL &&
        pU   != NULL &&
        pV   != NULL &&
        pDst != NULL, NPPST_NULL_POINTER_ERROR);

    ncvAssertReturn (srcSize.width * sizeof (Ncv32f) <= nSrcStep &&
        srcSize.width * sizeof (Ncv32f) <= nVFStep,
        NPPST_INVALID_STEP);

    Ncv32u srcStep = nSrcStep / sizeof (Ncv32f);
    Ncv32u vfStep  = nVFStep / sizeof (Ncv32f);

    dim3 ctaSize (32, 6);
    dim3 gridSize (iDivUp (srcSize.width, ctaSize.x), iDivUp (srcSize.height, ctaSize.y));

    ForwardWarpKernel_PSF1x1 <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream()>>>
        (pU, pV, pSrc, srcSize.width, srcSize.height, vfStep, srcStep, timeScale, pDst);

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


NCVStatus nppiStVectorWarp_PSF2x2_32f_C1(const Ncv32f *pSrc,
                                         NcvSize32u srcSize,
                                         Ncv32u nSrcStep,
                                         const Ncv32f *pU,
                                         const Ncv32f *pV,
                                         Ncv32u nVFStep,
                                         Ncv32f *pBuffer,
                                         Ncv32f timeScale,
                                         Ncv32f *pDst)
{
    ncvAssertReturn (pSrc != NULL &&
        pU   != NULL &&
        pV   != NULL &&
        pDst != NULL &&
        pBuffer != NULL, NPPST_NULL_POINTER_ERROR);

    ncvAssertReturn (srcSize.width * sizeof (Ncv32f) <= nSrcStep &&
        srcSize.width * sizeof (Ncv32f) <= nVFStep, NPPST_INVALID_STEP);

    Ncv32u srcStep = nSrcStep / sizeof (Ncv32f);
    Ncv32u vfStep = nVFStep / sizeof(Ncv32f);

    dim3 ctaSize(32, 6);
    dim3 gridSize (iDivUp (srcSize.width, ctaSize.x), iDivUp (srcSize.height, ctaSize.y));

    MemsetKernel <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream()>>>
        (0, srcSize.width, srcSize.height, pBuffer);

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    ForwardWarpKernel_PSF2x2 <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream()>>>
        (pU, pV, pSrc, srcSize.width, srcSize.height, vfStep, srcStep, timeScale, pBuffer, pDst);

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    NormalizeKernel <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream()>>>
        (pBuffer, srcSize.width, srcSize.height, srcStep, pDst);

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


//==============================================================================
//
// Resize.cu
//
//==============================================================================


texture <float, 2, cudaReadModeElementType> texSrc2D;


__forceinline__
__device__ float processLine(int spos,
                             float xmin,
                             float xmax,
                             int ixmin,
                             int ixmax,
                             float fxmin,
                             float cxmax)
{
    // first element
    float wsum = 1.0f - xmin + fxmin;
    float sum = tex1Dfetch(texSrc, spos) * (1.0f - xmin + fxmin);
    spos++;
    for (int ix = ixmin + 1; ix < ixmax; ++ix)
    {
        sum += tex1Dfetch(texSrc, spos);
        spos++;
        wsum += 1.0f;
    }
    sum += tex1Dfetch(texSrc, spos) * (cxmax - xmax);
    wsum += cxmax - xmax;
    return sum / wsum;
}


__global__ void resizeSuperSample_32f(NcvSize32u srcSize,
                                      Ncv32u srcStep,
                                      NcvRect32u srcROI,
                                      Ncv32f *dst,
                                      NcvSize32u dstSize,
                                      Ncv32u dstStep,
                                      NcvRect32u dstROI,
                                      Ncv32f scaleX,
                                      Ncv32f scaleY)
{
    // position within dst ROI
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= dstROI.width || iy >= dstROI.height)
    {
        return;
    }

    float rw = (float) srcROI.width;
    float rh = (float) srcROI.height;

    // source position
    float x = scaleX * (float) ix;
    float y = scaleY * (float) iy;

    // x sampling range
    float xBegin = fmax (x - scaleX, 0.0f);
    float xEnd   = fmin (x + scaleX, rw - 1.0f);
    // y sampling range
    float yBegin = fmax (y - scaleY, 0.0f);
    float yEnd   = fmin (y + scaleY, rh - 1.0f);
    // x range of source samples
    float floorXBegin = floorf (xBegin);
    float ceilXEnd    = ceilf (xEnd);
    int iXBegin = srcROI.x + (int) floorXBegin;
    int iXEnd   = srcROI.x + (int) ceilXEnd;
    // y range of source samples
    float floorYBegin = floorf (yBegin);
    float ceilYEnd    = ceilf (yEnd);
    int iYBegin = srcROI.y + (int) floorYBegin;
    int iYEnd   = srcROI.y + (int) ceilYEnd;

    // first row
    int pos = iYBegin * srcStep + iXBegin;

    float wsum = 1.0f - yBegin + floorYBegin;

    float sum = processLine (pos, xBegin, xEnd, iXBegin, iXEnd, floorXBegin,
        ceilXEnd) * (1.0f - yBegin + floorYBegin);
    pos += srcStep;
    for (int iy = iYBegin + 1; iy < iYEnd; ++iy)
    {
        sum += processLine (pos, xBegin, xEnd, iXBegin, iXEnd, floorXBegin,
            ceilXEnd);
        pos += srcStep;
        wsum += 1.0f;
    }

    sum += processLine (pos, xBegin, xEnd, iXBegin, iXEnd, floorXBegin,
        ceilXEnd) * (ceilYEnd - yEnd);
    wsum += ceilYEnd - yEnd;
    sum /= wsum;

    dst[(ix + dstROI.x) + (iy + dstROI.y) * dstStep] = sum;
}


// bicubic interpolation
__forceinline__
__device__ float bicubicCoeff(float x_)
{
    float x = fabsf(x_);
    if (x <= 1.0f)
    {
        return x * x * (1.5f * x - 2.5f) + 1.0f;
    }
    else if (x < 2.0f)
    {
        return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
    }
    else
    {
        return 0.0f;
    }
}


__global__ void resizeBicubic(NcvSize32u srcSize,
                              NcvRect32u srcROI,
                              NcvSize32u dstSize,
                              Ncv32u dstStep,
                              Ncv32f *dst,
                              NcvRect32u dstROI,
                              Ncv32f scaleX,
                              Ncv32f scaleY)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= dstROI.width || iy >= dstROI.height)
    {
        return;
    }

    const float dx = 1.0f / srcROI.width;
    const float dy = 1.0f / srcROI.height;

    float rx = (float) srcROI.x;
    float ry = (float) srcROI.y;

    float rw = (float) srcROI.width;
    float rh = (float) srcROI.height;

    float x = scaleX * (float) ix;
    float y = scaleY * (float) iy;

    // sampling range
    // border mode is clamp
    float xmin = fmax (ceilf (x - 2.0f), 0.0f);
    float xmax = fmin (floorf (x + 2.0f), rw - 1.0f);

    float ymin = fmax (ceilf (y - 2.0f), 0.0f);
    float ymax = fmin (floorf (y + 2.0f), rh - 1.0f);

    // shift data window to match ROI
    rx += 0.5f;
    ry += 0.5f;

    x += rx;
    y += ry;

    xmin += rx;
    xmax += rx;
    ymin += ry;
    ymax += ry;

    float sum  = 0.0f;
    float wsum = 0.0f;

    for (float cy = ymin; cy <= ymax; cy += 1.0f)
    {
        for (float cx = xmin; cx <= xmax; cx += 1.0f)
        {
            float xDist = x - cx;
            float yDist = y - cy;
            float wx = bicubicCoeff (xDist);
            float wy = bicubicCoeff (yDist);
            wx *= wy;
            sum += wx * tex2D (texSrc2D, cx * dx, cy * dy);
            wsum += wx;
        }
    }
    dst[(ix + dstROI.x)+ (iy + dstROI.y) * dstStep] = (!wsum)? 0 : sum / wsum;
}


NCVStatus nppiStResize_32f_C1R(const Ncv32f *pSrc,
                               NcvSize32u srcSize,
                               Ncv32u nSrcStep,
                               NcvRect32u srcROI,
                               Ncv32f *pDst,
                               NcvSize32u dstSize,
                               Ncv32u nDstStep,
                               NcvRect32u dstROI,
                               Ncv32f xFactor,
                               Ncv32f yFactor,
                               NppStInterpMode interpolation)
{
    NCVStatus status = NPPST_SUCCESS;

    ncvAssertReturn (pSrc != NULL && pDst != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn (xFactor != 0.0 && yFactor != 0.0, NPPST_INVALID_SCALE);

    ncvAssertReturn (nSrcStep >= sizeof (Ncv32f) * (Ncv32u) srcSize.width &&
        nDstStep >= sizeof (Ncv32f) * (Ncv32f) dstSize.width,
        NPPST_INVALID_STEP);

    Ncv32u srcStep = nSrcStep / sizeof (Ncv32f);
    Ncv32u dstStep = nDstStep / sizeof (Ncv32f);

    // TODO: preprocess ROI to prevent out of bounds access

    if (interpolation == nppStSupersample)
    {
        // bind texture
        cudaBindTexture (0, texSrc, pSrc, srcSize.height * nSrcStep);
        // invoke kernel
        dim3 ctaSize (32, 6);
        dim3 gridSize ((dstROI.width  + ctaSize.x - 1) / ctaSize.x,
            (dstROI.height + ctaSize.y - 1) / ctaSize.y);

        resizeSuperSample_32f <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream ()>>>
            (srcSize, srcStep, srcROI, pDst, dstSize, dstStep, dstROI, 1.0f / xFactor, 1.0f / yFactor);
    }
    else if (interpolation == nppStBicubic)
    {
        texSrc2D.addressMode[0] = cudaAddressModeMirror;
        texSrc2D.addressMode[1] = cudaAddressModeMirror;
        texSrc2D.normalized = true;

        cudaChannelFormatDesc desc = cudaCreateChannelDesc <float> ();

        cudaBindTexture2D (0, texSrc2D, pSrc, desc, srcSize.width, srcSize.height,
            nSrcStep);

        dim3 ctaSize (32, 6);
        dim3 gridSize ((dstSize.width  + ctaSize.x - 1) / ctaSize.x,
            (dstSize.height + ctaSize.y - 1) / ctaSize.y);

        resizeBicubic <<<gridSize, ctaSize, 0, nppStGetActiveCUDAstream ()>>>
            (srcSize, srcROI, dstSize, dstStep, pDst, dstROI, 1.0f / xFactor, 1.0f / yFactor);
    }
    else
    {
        status = NPPST_ERROR;
    }

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return status;
}

#endif /* CUDA_DISABLER */
