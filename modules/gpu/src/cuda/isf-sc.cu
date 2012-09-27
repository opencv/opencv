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

//     enum {
//         HOG_BINS = 6,
//         HOG_LUV_BINS = 10,
//         WIDTH = 640,
//         HEIGHT = 480,
//         GREY_OFFSET = HEIGHT * HOG_LUV_BINS
//     };

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
    // ToDo: do it in load time
    // __device__ __forceinline__ float rescale(const Level& level, uchar4& scaledRect, const Node& node)
    // {
    //     scaledRect = node.rect;
    //     return (float)(node.threshold & 0x0FFFFFFFU);
    // }

    __device__ __forceinline__ float rescale(const Level& level, uchar4& scaledRect, const Node& node)
    {
        float relScale = level.relScale;
        float farea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);

        dprintf("feature %d box %d %d %d %d\n", (node.threshold >> 28), scaledRect.x, scaledRect.y,
            scaledRect.z, scaledRect.w);
        dprintf("rescale: %f [%f %f] selected %f\n",level.relScale, level.scaling[0], level.scaling[1],
            level.scaling[(node.threshold >> 28) > 6]);

        // rescale
        scaledRect.x = __float2int_rn(relScale * scaledRect.x);
        scaledRect.y = __float2int_rn(relScale * scaledRect.y);
        scaledRect.z = __float2int_rn(relScale * scaledRect.z);
        scaledRect.w = __float2int_rn(relScale * scaledRect.w);

        float sarea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);

        float approx = 1.f;
        // if (fabs(farea - 0.f) > FLT_EPSILON && fabs(farea - 0.f) > FLT_EPSILON)
        {
            const float expected_new_area = farea * relScale * relScale;
            approx =  sarea / expected_new_area;
        }

        dprintf("new rect: %d box %d %d %d %d  rel areas %f %f\n", (node.threshold >> 28),
        scaledRect.x, scaledRect.y, scaledRect.z, scaledRect.w, farea * relScale * relScale, sarea);


        float rootThreshold = (node.threshold & 0x0FFFFFFFU) * approx;
        rootThreshold *= level.scaling[(node.threshold >> 28) > 6];

        dprintf("approximation %f %d -> %f %f\n", approx, (node.threshold & 0x0FFFFFFFU), rootThreshold,
            level.scaling[(node.threshold >> 28) > 6]);

        return rootThreshold;
    }

    __device__ __forceinline__ int get(const int x, int y, int channel, uchar4 area)
    {

        dprintf("feature box %d %d %d %d ", area.x, area.y, area.z, area.w);
        dprintf("get for channel %d\n", channel);
        dprintf("extract feature for: [%d %d] [%d %d] [%d %d] [%d %d]\n",
            x + area.x, y + area.y,  x + area.z, y + area.y,  x + area.z,y + area.w,
            x + area.x, y + area.w);
        dprintf("at point %d %d with offset %d\n", x, y, 0);

        int offset = channel * 121;
        y += offset;

        int a = tex2D(thogluv, x + area.x, y + area.y);
        int b = tex2D(thogluv, x + area.z, y + area.y);
        int c = tex2D(thogluv, x + area.z, y + area.w);
        int d = tex2D(thogluv, x + area.x, y + area.w);

        dprintf("    retruved integral values: %d %d %d %d\n", a, b, c, d);

        return (a - b + c - d);
    }

    __global__ void test_kernel(const Level* levels, const Octave* octaves, const float* stages,
        const Node* nodes, const float* leaves, PtrStepSz<uchar4> objects)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        Level level = levels[blockIdx.z];

        // if (x > 0 || y > 0 || blockIdx.z > 0) return;
        if(x >= level.workRect.x || y >= level.workRect.y) return;

        Octave octave = octaves[level.octave];

        int st = octave.index * octave.stages;
        const int stEnd = st + 1000;//octave.stages;

        float confidence = 0.f;

// #pragma unroll 8
        for(; st < stEnd; ++st)
        {
            dprintf("\n\nstage: %d\n", st);
            const int nId = st * 3;
            Node node = nodes[nId];

            dprintf("Node: [%d %d %d %d] %d %d\n", node.rect.x, node.rect.y, node.rect.z, node.rect.w,
                node.threshold >> 28, node.threshold & 0x0FFFFFFFU);

            float threshold = rescale(level, node.rect, node);
            int sum = get(x, y, (node.threshold >> 28), node.rect);

            dprintf("Node: [%d %d %d %d] %f\n", node.rect.x, node.rect.y, node.rect.z,
                node.rect.w, threshold);

            int next = 1 + (int)(sum >= threshold);
            dprintf("go: %d (%d >= %f)\n\n" ,next, sum, threshold);

            node = nodes[nId + next];
            threshold = rescale(level, node.rect, node);
            sum = get(x, y, (node.threshold >> 28), node.rect);

            const int lShift = (next - 1) * 2 + (int)(sum >= threshold);
            float impact = leaves[st * 4 + lShift];
            confidence += impact;

            if (confidence <= stages[st]) st = stEnd + 1;
            dprintf("decided: %d (%d >= %f) %d %f\n\n" ,next, sum, threshold, lShift, impact);
            dprintf("extracted stage: %f\n", stages[st]);
            dprintf("computed  score: %f\n\n", confidence);
        }

        // if (st == stEnd)
        //     printf("%d %d %d\n", x, y, st);

        uchar4 val;
        val.x = (int)confidence;
        if (x == y) objects(0, threadIdx.x) = val;

    }

    void detect(const PtrStepSzb& levels, const PtrStepSzb& octaves, const PtrStepSzf& stages,
        const PtrStepSzb& nodes, const PtrStepSzf& leaves, const PtrStepSzi& hogluv, PtrStepSz<uchar4> objects)
    {
        int fw = 160;
        int fh = 120;

        dim3 block(32, 8);
        dim3 grid(fw / 32, fh / 8, 47);

        const Level* l = (const Level*)levels.ptr();
        const Octave* oct = ((const Octave*)octaves.ptr());
        const float* st = (const float*)stages.ptr();
        const Node* nd = (const Node*)nodes.ptr();
        const float* lf = (const float*)leaves.ptr();

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
        cudaSafeCall( cudaBindTexture2D(0, thogluv, hogluv.data, desc, hogluv.cols, hogluv.rows, hogluv.step));

        test_kernel<<<grid, block>>>(l, oct, st, nd, lf, objects);

        cudaSafeCall( cudaGetLastError());
        cudaSafeCall( cudaDeviceSynchronize());
    }
}
}}}