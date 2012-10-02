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
#include <stdio.h>
#include <float.h>

// #define LOG_CUDA_CASCADE

#if defined LOG_CUDA_CASCADE
# define dprintf(format, ...) \
            do { printf(format, __VA_ARGS__); } while (0)
#else
# define dprintf(format, ...)
#endif

namespace cv { namespace gpu { namespace device {
namespace icf {

    // ToDo: use textures or ancached load instruction.
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
                  const int fw,  const int fh, const int bins)
    {
        const uchar* mag = (const uchar*)hogluv.ptr(fh * bins);
        uchar* hog = (uchar*)hogluv.ptr();
        const float* angle = (const float*)nangle.ptr();

        dim3 block(32, 8);
        dim3 grid(fw / 32, fh / 8);

        magToHist<<<grid, block>>>(mag, angle, nangle.step / sizeof(float), hog, hogluv.step, fh);
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    texture<int,  cudaTextureType2D, cudaReadModeElementType> thogluv;
    texture<char,  cudaTextureType2D, cudaReadModeElementType> troi;

    template<bool isUp>
    __device__ __forceinline__ float rescale(const Level& level, Node& node)
    {
        uchar4& scaledRect = node.rect;
        float relScale = level.relScale;
        float farea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);

        dprintf("%d: feature %d box %d %d %d %d\n",threadIdx.x, (node.threshold >> 28), scaledRect.x, scaledRect.y,
            scaledRect.z, scaledRect.w);
        dprintf("%d: rescale: %f [%f %f] selected %f\n",threadIdx.x, level.relScale, level.scaling[0], level.scaling[1],
            level.scaling[(node.threshold >> 28) > 6]);

        // rescale
        scaledRect.x = __float2int_rn(relScale * scaledRect.x);
        scaledRect.y = __float2int_rn(relScale * scaledRect.y);
        scaledRect.z = __float2int_rn(relScale * scaledRect.z);
        scaledRect.w = __float2int_rn(relScale * scaledRect.w);

        float sarea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);

        const float expected_new_area = farea * relScale * relScale;
        float approx = __fdividef(sarea, expected_new_area);

        dprintf("%d: new rect: %d box %d %d %d %d  rel areas %f %f\n",threadIdx.x, (node.threshold >> 28),
        scaledRect.x, scaledRect.y, scaledRect.z, scaledRect.w, farea * relScale * relScale, sarea);

        float rootThreshold = (node.threshold & 0x0FFFFFFFU) * approx;
        rootThreshold *= level.scaling[(node.threshold >> 28) > 6];

        dprintf("%d: approximation %f %d -> %f %f\n",threadIdx.x, approx, (node.threshold & 0x0FFFFFFFU), rootThreshold,
            level.scaling[(node.threshold >> 28) > 6]);

        return rootThreshold;
    }

    template<>
    __device__ __forceinline__ float rescale<true>(const Level& level, Node& node)
    {
        uchar4& scaledRect = node.rect;
        float relScale = level.relScale;
        float farea = scaledRect.z * scaledRect.w;

        dprintf("%d: feature %d box %d %d %d %d\n",threadIdx.x, (node.threshold >> 28), scaledRect.x, scaledRect.y,
            scaledRect.z, scaledRect.w);
        dprintf("%d: rescale: %f [%f %f] selected %f\n",threadIdx.x, level.relScale, level.scaling[0], level.scaling[1],
            level.scaling[(node.threshold >> 28) > 6]);

        // rescale
        scaledRect.x = __float2int_rn(relScale * scaledRect.x);
        scaledRect.y = __float2int_rn(relScale * scaledRect.y);
        scaledRect.z = __float2int_rn(relScale * scaledRect.z);
        scaledRect.w = __float2int_rn(relScale * scaledRect.w);

        float sarea = scaledRect.z * scaledRect.w;

        const float expected_new_area = farea * relScale * relScale;
        float approx = __fdividef(sarea, expected_new_area);

        dprintf("%d: new rect: %d box %d %d %d %d  rel areas %f %f\n",threadIdx.x, (node.threshold >> 28),
        scaledRect.x, scaledRect.y, scaledRect.z, scaledRect.w, farea * relScale * relScale, sarea);

        float rootThreshold = (node.threshold & 0x0FFFFFFFU) * approx;

        rootThreshold *= level.scaling[(node.threshold >> 28) > 6];

        dprintf("%d: approximation %f %d -> %f %f\n",threadIdx.x, approx, (node.threshold & 0x0FFFFFFFU), rootThreshold,
            level.scaling[(node.threshold >> 28) > 6]);

        return rootThreshold;
    }

    template<bool isUp>
    __device__ __forceinline__ int get(int x, int y, uchar4 area)
    {

        dprintf("%d: feature box %d %d %d %d\n",threadIdx.x, area.x, area.y, area.z, area.w);
        dprintf("%d: extract feature for: [%d %d] [%d %d] [%d %d] [%d %d]\n",threadIdx.x,
            x + area.x, y + area.y,  x + area.z, y + area.y,  x + area.z,y + area.w,
            x + area.x, y + area.w);
        dprintf("%d: at point %d %d with offset %d\n", x, y, 0);

        int a = tex2D(thogluv, x + area.x, y + area.y);
        int b = tex2D(thogluv, x + area.z, y + area.y);
        int c = tex2D(thogluv, x + area.z, y + area.w);
        int d = tex2D(thogluv, x + area.x, y + area.w);

        dprintf("%d   retruved integral values: %d %d %d %d\n",threadIdx.x, a, b, c, d);

        return (a - b + c - d);
    }

    template<>
    __device__ __forceinline__ int get<true>(int x, int y, uchar4 area)
    {

        dprintf("%d: feature box %d %d %d %d\n",threadIdx.x, area.x, area.y, area.z, area.w);
        dprintf("%d: extract feature for: [%d %d] [%d %d] [%d %d] [%d %d]\n",threadIdx.x,
            x + area.x, y + area.y,  x + area.z, y + area.y,  x + area.z,y + area.w,
            x + area.x, y + area.w);
        dprintf("%d: at point %d %d with offset %d\n", x, y, 0);

        x += area.x;
        y += area.y;
        int a = tex2D(thogluv, x, y);
        int b = tex2D(thogluv, x + area.z, y);
        int c = tex2D(thogluv, x + area.z, y + area.w);
        int d = tex2D(thogluv, x, y + area.w);

        dprintf("%d   retruved integral values: %d %d %d %d\n",threadIdx.x, a, b, c, d);

        return (a - b + c - d);
    }

#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
    template<bool isUp>
    __global__ void test_kernel_warp(const Level* levels, const Octave* octaves, const float* stages,
        const Node* nodes, const float* leaves, Detection* objects, const uint ndetections, uint* ctr,
        const int downscales)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x;

        Level level = levels[downscales + blockIdx.z];

        if(x >= level.workRect.x || y >= level.workRect.y) return;

        if (!tex2D(troi, x, y)) return;

        Octave octave = octaves[level.octave];
        int st = octave.index * octave.stages;
        const int stEnd = st + 1024;

        float confidence = 0.f;

        for(; st < stEnd; st += 32)
        {

            const int nId = (st + threadIdx.x) * 3;
            dprintf("\n\n%d: stage: %d %d\n",threadIdx.x, st, nId);
            Node node = nodes[nId];

            float threshold = rescale<isUp>(level, node);
            int sum = get<isUp>(x, y + (node.threshold >> 28) * 121, node.rect);

            int next = 1 + (int)(sum >= threshold);
            dprintf("%d: go: %d (%d >= %f)\n\n" ,threadIdx.x, next, sum, threshold);

            node = nodes[nId + next];
            threshold = rescale<isUp>(level, node);
            sum = get<isUp>(x, y + (node.threshold >> 28) * 121, node.rect);

            const int lShift = (next - 1) * 2 + (int)(sum >= threshold);
            float impact = leaves[(st + threadIdx.x) * 4 + lShift];

            dprintf("%d: decided: %d (%d >= %f) %d %f\n\n" ,threadIdx.x, next, sum, threshold, lShift, impact);
            dprintf("%d: extracted stage: %f\n",threadIdx.x, stages[(st + threadIdx.x)]);
            dprintf("%d: computed  score: %f\n",threadIdx.x, impact);
#pragma unroll
            // scan on shuffl functions
            for (int i = 1; i < 32; i *= 2)
            {
                const float n = __shfl_up(impact, i, 32);

                if (threadIdx.x >= i)
                    impact += n;
            }

            dprintf("%d: impact scaned %f\n" ,threadIdx.x, impact);

            confidence += impact;
            if(__any((confidence <= stages[(st + threadIdx.x)]))) st += stEnd;
        }

        if(st == stEnd && !threadIdx.x)
        {
            int idx = atomicInc(ctr, ndetections);
            // store detection
            objects[idx] = Detection(__float2int_rn(x * octave.shrinkage),
                __float2int_rn(y * octave.shrinkage), level.objSize.x, level.objSize.y, confidence);
        }
    }
#else
    template<bool isUp>
    __global__ void test_kernel_warp(const Level* levels, const Octave* octaves, const float* stages,
        const Node* nodes, const float* leaves, Detection* objects, const uint ndetections, uint* ctr,
        const int downscales)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        Level level = levels[blockIdx.z];

        // if (blockIdx.z != 31) return;
        if(x >= level.workRect.x || y >= level.workRect.y) return;

        int roi = tex2D(troi, x, y);
        printf("%d\n", roi);
        if (!roi) return;

        Octave octave = octaves[level.octave];

        int st = octave.index * octave.stages;
        const int stEnd = st + 1000;//octave.stages;

        float confidence = 0.f;

        for(; st < stEnd; ++st)
        {
            dprintf("\n\nstage: %d\n", st);
            const int nId = st * 3;
            Node node = nodes[nId];

            dprintf("Node: [%d %d %d %d] %d %d\n", node.rect.x, node.rect.y, node.rect.z, node.rect.w,
                node.threshold >> 28, node.threshold & 0x0FFFFFFFU);

            float threshold = rescale<isUp>(level, node);
            int sum = get<isUp>(x, y + (node.threshold >> 28) * 121, node.rect);

            dprintf("Node: [%d %d %d %d] %f\n", node.rect.x, node.rect.y, node.rect.z,
                node.rect.w, threshold);

            int next = 1 + (int)(sum >= threshold);
            dprintf("go: %d (%d >= %f)\n\n" ,next, sum, threshold);

            node = nodes[nId + next];
            threshold = rescale<isUp>(level, node);
            sum = get<isUp>(x, y + (node.threshold >> 28) * 121, node.rect);

            const int lShift = (next - 1) * 2 + (int)(sum >= threshold);
            float impact = leaves[st * 4 + lShift];
            confidence += impact;

            if (confidence <= stages[st]) st = stEnd + 10;
            dprintf("decided: %d (%d >= %f) %d %f\n\n" ,next, sum, threshold, lShift, impact);
            dprintf("extracted stage: %f\n", stages[st]);
            dprintf("computed  score: %f\n\n", confidence);
        }

        if(st == stEnd)
        {
            int idx = atomicInc(ctr, ndetections);
            // store detection
            objects[idx] = Detection(__float2int_rn(x * octave.shrinkage),
                __float2int_rn(y * octave.shrinkage), level.objSize.x, level.objSize.y, confidence);
        }
    }
#endif

    void detect(const PtrStepSzb& roi, const PtrStepSzb& levels, const PtrStepSzb& octaves, const PtrStepSzf& stages,
                const PtrStepSzb& nodes,  const PtrStepSzf& leaves,  const PtrStepSzi& hogluv,
                PtrStepSz<uchar4> objects, PtrStepSzi counter, const int downscales)
    {
        int fw = 160;
        int fh = 120;

        dim3 block(32, 8);
        dim3 grid(fw, fh / 8, downscales);

        const Level* l = (const Level*)levels.ptr();
        const Octave* oct = ((const Octave*)octaves.ptr());
        const float* st = (const float*)stages.ptr();
        const Node* nd = (const Node*)nodes.ptr();
        const float* lf = (const float*)leaves.ptr();
        uint* ctr = (uint*)counter.ptr();
        Detection* det = (Detection*)objects.ptr();
        uint max_det = objects.cols / sizeof(Detection);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
        cudaSafeCall( cudaBindTexture2D(0, thogluv, hogluv.data, desc, hogluv.cols, hogluv.rows, hogluv.step));

        cudaChannelFormatDesc desc_roi = cudaCreateChannelDesc<char>();
        cudaSafeCall( cudaBindTexture2D(0, troi, roi.data, desc_roi, roi.cols, roi.rows, roi.step));

        test_kernel_warp<false><<<grid, block>>>(l, oct, st, nd, lf, det, max_det, ctr, 0);
        cudaSafeCall( cudaGetLastError());

        grid = dim3(fw, fh / 8, 47 - downscales);
        test_kernel_warp<true><<<grid, block>>>(l, oct, st, nd, lf, det, max_det, ctr, downscales);
        cudaSafeCall( cudaGetLastError());
        cudaSafeCall( cudaDeviceSynchronize());
    }

    void detectAtScale(const int scale, const PtrStepSzb& roi, const PtrStepSzb& levels, const PtrStepSzb& octaves,
        const PtrStepSzf& stages, const PtrStepSzb& nodes, const PtrStepSzf& leaves, const PtrStepSzi& hogluv,
        PtrStepSz<uchar4> objects, PtrStepSzi counter, const int downscales)
    {
        int fw = 160;
        int fh = 120;

        dim3 block(32, 8);
        dim3 grid(fw, fh / 8, 1);

        const Level* l = (const Level*)levels.ptr();
        const Octave* oct = ((const Octave*)octaves.ptr());
        const float* st = (const float*)stages.ptr();
        const Node* nd = (const Node*)nodes.ptr();
        const float* lf = (const float*)leaves.ptr();
        uint* ctr = (uint*)counter.ptr();
        Detection* det = (Detection*)objects.ptr();
        uint max_det = objects.cols / sizeof(Detection);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
        cudaSafeCall( cudaBindTexture2D(0, thogluv, hogluv.data, desc, hogluv.cols, hogluv.rows, hogluv.step));

        cudaChannelFormatDesc desc_roi = cudaCreateChannelDesc<char>();
        cudaSafeCall( cudaBindTexture2D(0, troi, roi.data, desc_roi, roi.cols, roi.rows, roi.step));

        if (scale >= downscales)
            test_kernel_warp<true><<<grid, block>>>(l, oct, st, nd, lf, det, max_det, ctr, scale);
        else
            test_kernel_warp<false><<<grid, block>>>(l, oct, st, nd, lf, det, max_det, ctr, scale);

        cudaSafeCall( cudaGetLastError());
        cudaSafeCall( cudaDeviceSynchronize());
    }
}
}}}