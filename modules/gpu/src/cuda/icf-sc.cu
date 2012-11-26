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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#include <opencv2/gpu/device/common.hpp>

#include <icf.hpp>
#include <float.h>
#include <stdio.h>

namespace cv { namespace gpu { namespace device {
namespace icf {

    // ToDo: use textures or uncached load instruction.
    __global__ void magToHist(const uchar* __restrict__ mag,
                              const float* __restrict__ angle, const int angPitch,
                                    uchar* __restrict__ hog,   const int hogPitch, const int fh)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;

        const int bin = (int)(angle[y * angPitch + x]);
        const uchar val = mag[y * hogPitch + x];
        hog[((fh * bin) + y) * hogPitch + x] = val;
    }

    void fillBins(cv::gpu::PtrStepSzb hogluv, const cv::gpu::PtrStepSzf& nangle,
                  const int fw,  const int fh, const int bins, cudaStream_t stream )
    {
        const uchar* mag = (const uchar*)hogluv.ptr(fh * bins);
        uchar* hog = (uchar*)hogluv.ptr();
        const float* angle = (const float*)nangle.ptr();

        dim3 block(32, 8);
        dim3 grid(fw / 32, fh / 8);

        magToHist<<<grid, block, 0, stream>>>(mag, angle, nangle.step / sizeof(float), hog, hogluv.step, fh);
        if (!stream)
        {
            cudaSafeCall( cudaGetLastError() );
            cudaSafeCall( cudaDeviceSynchronize() );
        }
    }

    __device__ __forceinline__ float overlapArea(const Detection &a, const Detection &b)
    {
        int w = ::min(a.x + a.w, b.x + b.w) - ::max(a.x, b.x);
        int h = ::min(a.y + a.h, b.y + b.h) - ::max(a.y, b.y);

        return (w < 0 || h < 0)? 0.f : (float)(w * h);
    }

    texture<uint4,  cudaTextureType2D, cudaReadModeElementType> tdetections;

    __global__ void overlap(const uint* n, uchar* overlaps)
    {
        const int idx = threadIdx.x;
        const int total = *n;

        for (int i = idx + 1; i < total; i += 192)
        {
            const uint4 _a = tex2D(tdetections, i, 0);
            const Detection& a = *((Detection*)(&_a));
            bool excluded = false;

            for (int j = i + 1; j < total; ++j)
            {
                const uint4 _b = tex2D(tdetections, j, 0);
                const Detection& b = *((Detection*)(&_b));
                float ovl = overlapArea(a, b) / ::min(a.w * a.h, b.w * b.h);

                if (ovl > 0.65f)
                {
                    int suppessed = (a.confidence > b.confidence)? j : i;
                    overlaps[suppessed] = 1;
                    excluded = excluded || (suppessed == i);
                }

                if (__all(excluded)) break;
            }
        }
    }

    __global__ void collect(const uint* n, uchar* overlaps, uint* ctr, uint4* suppressed)
    {
        const int idx = threadIdx.x;
        const int total = *n;

        for (int i = idx; i < total; i += 192)
        {
            if (!overlaps[i])
            {
                int oidx = atomicInc(ctr, 50);
                suppressed[oidx] = tex2D(tdetections, i + 1, 0);
            }
        }
    }

    void suppress(const PtrStepSzb& objects, PtrStepSzb overlaps, PtrStepSzi ndetections,
        PtrStepSzb suppressed, cudaStream_t stream)
    {
        int block = 192;
        int grid = 1;

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint4>();
        size_t offset;
        cudaSafeCall( cudaBindTexture2D(&offset, tdetections, objects.data, desc, objects.cols / sizeof(uint4), objects.rows, objects.step));

        overlap<<<grid, block>>>((uint*)ndetections.ptr(0), (uchar*)overlaps.ptr(0));
        collect<<<grid, block>>>((uint*)ndetections.ptr(0), (uchar*)overlaps.ptr(0), (uint*)suppressed.ptr(0), ((uint4*)suppressed.ptr(0)) + 1);

        if (!stream)
        {
            cudaSafeCall( cudaGetLastError());
            cudaSafeCall( cudaDeviceSynchronize());
        }
    }

    template<typename Policy>
    struct PrefixSum
    {
    __device static void apply(float& impact)
        {
    #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
    #pragma unroll
            // scan on shuffl functions
            for (int i = 1; i < Policy::WARP; i *= 2)
            {
                const float n = __shfl_up(impact, i, Policy::WARP);

                if (threadIdx.x >= i)
                    impact += n;
            }
    #else
            __shared__ volatile float ptr[Policy::STA_X * Policy::STA_Y];

            const int idx = threadIdx.y * Policy::STA_X + threadIdx.x;

            ptr[idx] = impact;

            if ( threadIdx.x >=  1) ptr [idx ] = (ptr [idx -  1] + ptr [idx]);
            if ( threadIdx.x >=  2) ptr [idx ] = (ptr [idx -  2] + ptr [idx]);
            if ( threadIdx.x >=  4) ptr [idx ] = (ptr [idx -  4] + ptr [idx]);
            if ( threadIdx.x >=  8) ptr [idx ] = (ptr [idx -  8] + ptr [idx]);
            if ( threadIdx.x >= 16) ptr [idx ] = (ptr [idx - 16] + ptr [idx]);

            impact = ptr[idx];
    #endif
        }
    };

    texture<int,  cudaTextureType2D, cudaReadModeElementType> thogluv;

    template<bool isUp>
    __device__ __forceinline__ float rescale(const Level& level, Node& node)
    {
        uchar4& scaledRect = node.rect;
        float relScale = level.relScale;
        float farea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);

        // rescale
        scaledRect.x = __float2int_rn(relScale * scaledRect.x);
        scaledRect.y = __float2int_rn(relScale * scaledRect.y);
        scaledRect.z = __float2int_rn(relScale * scaledRect.z);
        scaledRect.w = __float2int_rn(relScale * scaledRect.w);

        float sarea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);

        const float expected_new_area = farea * relScale * relScale;
        float approx = (sarea == 0)? 1: __fdividef(sarea, expected_new_area);

        float rootThreshold = (node.threshold & 0x0FFFFFFFU) * approx * level.scaling[(node.threshold >> 28) > 6];

        return rootThreshold;
    }

    template<>
    __device__ __forceinline__ float rescale<true>(const Level& level, Node& node)
    {
        uchar4& scaledRect = node.rect;
        float relScale = level.relScale;
        float farea = scaledRect.z * scaledRect.w;

        // rescale
        scaledRect.x = __float2int_rn(relScale * scaledRect.x);
        scaledRect.y = __float2int_rn(relScale * scaledRect.y);
        scaledRect.z = __float2int_rn(relScale * scaledRect.z);
        scaledRect.w = __float2int_rn(relScale * scaledRect.w);

        float sarea = scaledRect.z * scaledRect.w;

        const float expected_new_area = farea * relScale * relScale;
        float approx = __fdividef(sarea, expected_new_area);

        float rootThreshold = (node.threshold & 0x0FFFFFFFU) * approx * level.scaling[(node.threshold >> 28) > 6];

        return rootThreshold;
    }

    template<bool isUp>
    __device__ __forceinline__ int get(int x, int y, uchar4 area)
    {
        int a = tex2D(thogluv, x + area.x, y + area.y);
        int b = tex2D(thogluv, x + area.z, y + area.y);
        int c = tex2D(thogluv, x + area.z, y + area.w);
        int d = tex2D(thogluv, x + area.x, y + area.w);

        return (a - b + c - d);
    }

    template<>
    __device__ __forceinline__ int get<true>(int x, int y, uchar4 area)
    {
        x += area.x;
        y += area.y;
        int a = tex2D(thogluv, x, y);
        int b = tex2D(thogluv, x + area.z, y);
        int c = tex2D(thogluv, x + area.z, y + area.w);
        int d = tex2D(thogluv, x, y + area.w);

        return (a - b + c - d);
    }

    texture<float2,  cudaTextureType2D, cudaReadModeElementType> troi;

template<typename Policy>
template<bool isUp>
__device void CascadeInvoker<Policy>::detect(Detection* objects, const uint ndetections, uint* ctr, const int downscales) const
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x;

    // load Lavel
    __shared__ Level level;

    // check POI
    __shared__ volatile char roiCache[Policy::STA_Y];

    if (!threadIdx.y && !threadIdx.x)
        ((float2*)roiCache)[threadIdx.x] = tex2D(troi, blockIdx.y, x);

    __syncthreads();

    if (!roiCache[threadIdx.y]) return;

    if (!threadIdx.x)
        level = levels[downscales + blockIdx.z];

    if(x >= level.workRect.x || y >= level.workRect.y) return;

    int st = level.octave * level.step;
    const int stEnd = st + level.step;

    const int hogluvStep = gridDim.y * Policy::STA_Y;
    float confidence = 0.f;
    for(; st < stEnd; st += Policy::WARP)
    {
        const int nId = (st + threadIdx.x) * 3;

        Node node = nodes[nId];

        float threshold = rescale<isUp>(level, node);
        int sum = get<isUp>(x, y + (node.threshold >> 28) * hogluvStep, node.rect);

        int next = 1 + (int)(sum >= threshold);

        node = nodes[nId + next];
        threshold = rescale<isUp>(level, node);
        sum = get<isUp>(x, y + (node.threshold >> 28) * hogluvStep, node.rect);

        const int lShift = (next - 1) * 2 + (int)(sum >= threshold);
        float impact = leaves[(st + threadIdx.x) * 4 + lShift];

        PrefixSum<Policy>::apply(impact);
        confidence += impact;

        if(__any((confidence <= stages[(st + threadIdx.x)]))) st += 2048;
    }

    if(!threadIdx.x && st == stEnd &&  ((confidence - FLT_EPSILON) >= 0))
    {
        int idx = atomicInc(ctr, ndetections);
        objects[idx] = Detection(__float2int_rn(x * Policy::SHRINKAGE),
            __float2int_rn(y * Policy::SHRINKAGE), level.objSize.x, level.objSize.y, confidence);
    }
}

template<typename Policy, bool isUp>
__global__ void soft_cascade(const CascadeInvoker<Policy> invoker, Detection* objects, const uint n, uint* ctr, const int downs)
{
    invoker.template detect<isUp>(objects, n, ctr, downs);
}

template<typename Policy>
void CascadeInvoker<Policy>::operator()(const PtrStepSzb& roi, const PtrStepSzi& hogluv,
    PtrStepSz<uchar4> objects, const int downscales, const cudaStream_t& stream) const
{
    int fw = roi.rows;
    int fh = roi.cols;

    dim3 grid(fw, fh / Policy::STA_Y, downscales);

    uint* ctr = (uint*)(objects.ptr(0));
    Detection* det = ((Detection*)objects.ptr(0)) + 1;
    uint max_det = objects.cols / sizeof(Detection);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaSafeCall( cudaBindTexture2D(0, thogluv, hogluv.data, desc, hogluv.cols, hogluv.rows, hogluv.step));

    cudaChannelFormatDesc desc_roi = cudaCreateChannelDesc<typename Policy::roi_type>();
    cudaSafeCall( cudaBindTexture2D(0, troi, roi.data, desc_roi, roi.cols / Policy::STA_Y, roi.rows, roi.step));

    const CascadeInvoker<Policy> inv = *this;

    soft_cascade<Policy, false><<<grid, Policy::block(), 0, stream>>>(inv, det, max_det, ctr, 0);
    cudaSafeCall( cudaGetLastError());

    grid = dim3(fw, fh / Policy::STA_Y, scales - downscales);
    soft_cascade<Policy, true><<<grid, Policy::block(), 0, stream>>>(inv, det, max_det, ctr, downscales);

    if (!stream)
    {
        cudaSafeCall( cudaGetLastError());
        cudaSafeCall( cudaDeviceSynchronize());
    }
}

template void CascadeInvoker<GK107PolicyX4>::operator()(const PtrStepSzb& roi, const PtrStepSzi& hogluv,
    PtrStepSz<uchar4> objects, const int downscales, const cudaStream_t& stream) const;

}
}}}