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

#include <icf.hpp>
#include <opencv2/gpu/device/saturate_cast.hpp>
#include <stdio.h>
#include <float.h>

//#define LOG_CUDA_CASCADE

#if defined LOG_CUDA_CASCADE
# define dprintf(format, ...) \
            do { printf(format, __VA_ARGS__); } while (0)
#else
# define dprintf(format, ...)
#endif

namespace cv { namespace gpu { namespace device {

namespace icf {

    enum {
        HOG_BINS = 6,
        HOG_LUV_BINS = 10,
        WIDTH = 640,
        HEIGHT = 480,
        GREY_OFFSET = HEIGHT * HOG_LUV_BINS
    };

    __global__ void magToHist(const uchar* __restrict__ mag,
                              const float* __restrict__ angle, const int angPitch,
                                    uchar* __restrict__ hog,   const int hogPitch)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;

        const int bin = (int)(angle[y * angPitch + x]);
        const uchar val = mag[y * angPitch + x];

        hog[((HEIGHT * bin) + y) * hogPitch + x] = val;
    }

    void fillBins(cv::gpu::PtrStepSzb hogluv, const cv::gpu::PtrStepSzf& nangle)
    {
        const uchar* mag = (const uchar*)hogluv.ptr(HEIGHT * HOG_BINS);
        uchar* hog = (uchar*)hogluv.ptr();
        const float* angle = (const float*)nangle.ptr();

        dim3 block(32, 8);
        dim3 grid(WIDTH / 32, HEIGHT / 8);

        magToHist<<<grid, block>>>(mag, angle, nangle.step / sizeof(float), hog, hogluv.step);
        cudaSafeCall( cudaGetLastError() );
        cudaSafeCall( cudaDeviceSynchronize() );
    }
}

__global__ void detect(const cv::gpu::icf::Cascade cascade, const int* __restrict__ hogluv, const int pitch,
    PtrStepSz<uchar4> objects)
{
    cascade.detectAt(hogluv, pitch, objects);
}

}

float __device icf::Cascade::rescale(const icf::Level& level, uchar4& scaledRect,
                                     const int channel, const float threshold) const
{
    dprintf("feature %d box %d %d %d %d\n", channel, scaledRect.x, scaledRect.y, scaledRect.z, scaledRect.w);
    dprintf("rescale: %f [%f %f]\n",level.relScale, level.scaling[0], level.scaling[1]);

    float relScale = level.relScale;
    float farea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);

    // rescale
    scaledRect.x = __float2int_rn(relScale * scaledRect.x);
    scaledRect.y = __float2int_rn(relScale * scaledRect.y);
    scaledRect.z = __float2int_rn(relScale * scaledRect.z);
    scaledRect.w = __float2int_rn(relScale * scaledRect.w);

    float sarea = (scaledRect.z - scaledRect.x) * (scaledRect.w - scaledRect.y);


    float approx = 1.f;
    if (fabs(farea - 0.f) > FLT_EPSILON && fabs(farea - 0.f) > FLT_EPSILON)
    {
        const float expected_new_area = farea * relScale * relScale;
        approx = expected_new_area / sarea;
    }

    dprintf("new rect: %d box %d %d %d %d  rel areas %f %f\n", channel,
        scaledRect.x, scaledRect.y, scaledRect.z, scaledRect.w, farea * relScale * relScale, sarea);

    // compensation areas rounding
    float rootThreshold = threshold / approx;
    // printf("    approx %f\n", rootThreshold);
    rootThreshold *= level.scaling[(int)(channel > 6)];

    dprintf("approximation %f %f -> %f %f\n", approx, threshold, rootThreshold, level.scaling[(int)(channel > 6)]);

    return rootThreshold;
}

typedef unsigned char uchar;
float __device get(const int* __restrict__ hogluv, const int pitch,
                   const int x, const int y, int channel, uchar4 area)
{
    dprintf("feature box %d %d %d %d ", area.x, area.y, area.z, area.w);
    dprintf("get for channel %d\n", channel);
    dprintf("extract feature for: [%d %d] [%d %d] [%d %d] [%d %d]\n",
        x + area.x, y + area.y,  x + area.z, y + area.y,  x + area.z,y + area.w,
        x + area.x, y + area.w);
    dprintf("at point %d %d with offset %d\n", x, y, 0);

    const int* curr = hogluv + ((channel * 121) + y) * pitch;

    int a = curr[area.y * pitch + x + area.x];
    int b = curr[area.y * pitch + x + area.z];
    int c = curr[area.w * pitch + x + area.z];
    int d = curr[area.w * pitch + x + area.x];

    dprintf("    retruved integral values: %d %d %d %d\n", a, b, c, d);

    return (a - b + c - d);
}


void __device icf::Cascade::detectAt(const int* __restrict__ hogluv, const int pitch,
                                    PtrStepSz<uchar4>& objects) const
{
    const icf::Level* lls = (const icf::Level*)levels.ptr();

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    // if (x > 0 || y > 0) return;

    Level level = lls[blockIdx.z];
    if (x >= level.workRect.x || y >= level.workRect.y) return;

    dprintf("level: %d (%f %f) [%f %f] (%d %d) (%d %d)\n", level.octave, level.relScale, level.shrScale,
        level.scaling[0], level.scaling[1], level.workRect.x, level.workRect.y, level.objSize.x, level.objSize.y);

    const Octave octave = ((const Octave*)octaves.ptr())[level.octave];
    // printf("Octave: %d %d %d (%d %d) %f\n", octave.index, octave.stages,
    //     octave.shrinkage, octave.size.x, octave.size.y, octave.scale);

    const int stBegin = octave.index * octave.stages, stEnd = stBegin + octave.stages;

    float detectionScore = 0.f;

    int st = stBegin;
    for(; st < stEnd; ++st)
    {
        const float stage = stages(0, st);
        dprintf("Stage: %f\n", stage);
        {
            const int nId = st * 3;

            // work with root node
            const Node node = ((const Node*)nodes.ptr())[nId];

            dprintf("Node: %d %f\n", node.feature, node.threshold);

            const Feature feature = ((const Feature*)features.ptr())[node.feature];

            uchar4 scaledRect = feature.rect;
            float threshold = rescale(level, scaledRect, feature.channel, node.threshold);

            float sum = get(hogluv,pitch, x, y, feature.channel, scaledRect);

            dprintf("root feature %d %f\n",feature.channel, sum);

            int next = 1 + (int)(sum >= threshold);

            dprintf("go: %d (%f >= %f)\n\n" ,next, sum, threshold);

            // leaves
            const Node leaf = ((const Node*)nodes.ptr())[nId + next];
            const Feature fLeaf = ((const Feature*)features.ptr())[leaf.feature];

            scaledRect = fLeaf.rect;
            threshold = rescale(level, scaledRect, fLeaf.channel, leaf.threshold);
            sum = get(hogluv, pitch, x, y, fLeaf.channel, scaledRect);

            const int lShift = (next - 1) * 2 + (int)(sum >= threshold);
            float impact = leaves(0, (st * 4) + lShift);

            detectionScore += impact;

            dprintf("decided: %d (%f >= %f) %d %f\n\n" ,next, sum, threshold, lShift, impact);
            dprintf("extracted stage:\n");
            dprintf("ct %f\n", stage);
            dprintf("computed score %f\n\n", detectionScore);
            dprintf("\n\n");
        }

        if (detectionScore <= stage || st - stBegin == 100) break;
    }

    dprintf("x %d y %d: %d\n", x, y, st - stBegin);

    if (st == stEnd)
    {
        uchar4 a;
        a.x = level.workRect.x;
        a.y = level.workRect.y;
        objects(0, threadIdx.x) = a;
    }
}

void icf::Cascade::detect(const cv::gpu::PtrStepSzi& hogluv, PtrStepSz<uchar4> objects, cudaStream_t stream) const
{
    dim3 block(32, 8, 1);
    dim3 grid(ChannelStorage::FRAME_WIDTH / 32, ChannelStorage::FRAME_HEIGHT / 8, 47);
    device::detect<<<grid, block, 0, stream>>>(*this, hogluv, hogluv.step / sizeof(int), objects);
    cudaSafeCall( cudaGetLastError() );
    if (!stream)
        cudaSafeCall( cudaDeviceSynchronize() );
}

}}