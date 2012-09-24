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

enum {
    HOG_BINS = 6,
    HOG_LUV_BINS = 10,
    WIDTH = 640,
    HEIGHT = 480,
    GREY_OFFSET = HEIGHT * HOG_LUV_BINS
};

/* Returns the nearest upper power of two, works only for
the typical GPU thread count (pert block) values */
int power_2up(unsigned int n)
{
    if (n < 1) return 1;
    else if (n < 2) return 2;
    else if (n < 4) return 4;
    else if (n < 8) return 8;
    else if (n < 16) return 16;
    else if (n < 32) return 32;
    else if (n < 64) return 64;
    else if (n < 128) return 128;
    else if (n < 256) return 256;
    else if (n < 512) return 512;
    else if (n < 1024) return 1024;
    return -1; // Input is too big
}


__device__ __forceinline__ uchar grey(const uchar3 rgb)
{
    return saturate_cast<uchar>(rgb.x * 0.114f + rgb.y * 0.587f + rgb.z * 0.299f);
}

__device__ __forceinline__ void luv(const uchar3 rgb, uchar& l, uchar& u, uchar& v)
{

}

__global__ void rgb2grayluv(const uchar3* __restrict__ rgb, uchar* __restrict__ hog,
                            const int rgbPitch, const int hogPitch)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    const uchar3 color = rgb[rgbPitch * y + x];

    uchar l, u, v;
    luv(color, l, u, v);

    hog[hogPitch *  y + x] = l;
    hog[hogPitch * (y + HEIGHT) + x] = u;
    hog[hogPitch * (y + 2 * HEIGHT) + x] = v;
    hog[hogPitch * (y + 3 * HEIGHT) + x] = grey(color);
}

__device__ __forceinline__
int qangle(const float &y, const float &x)
{
    int bin = 0;
//     const float2 &bin_vector_zero = const_angle_bins_vectors[0];
//     float max_dot_product = fabs(x*bin_vector_zero.x + y*bin_vector_zero.y);

//     // let us hope this gets unrolled
// #pragma unroll
//     for(int i=1; i < num_angles_bin; i+=1)
//     {
//         const float2 &bin_vector_i = const_angle_bins_vectors[i];
//         //const float2 bin_vector_i = const_angle_bins_vectors[i];
//         //const float2 &bin_vector_i = angle_bins_vectors[i];
//         const float dot_product = fabs(x*bin_vector_i.x + y*bin_vector_i.y);
//         if(dot_product > max_dot_product)
//         {
//             max_dot_product = dot_product;
//             index = i;
//         }
//     }

    return bin;
}

// texture<uchar, 2, cudaReadModeElementType> tgray;
__global__ void gray2hog(const uchar* __restrict__ gray, uchar* __restrict__ hog, const int pitch, const float norm)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    // derivative
    float dx = gray[y * pitch + x + 1];
    dx -= gray[y * pitch + x - 1];

    float dy = gray[(y + 1) * pitch + x];
    dy -= gray[(y -1) * pitch + x - 1];

    // mag and angle
    const uchar mag =  saturate_cast<uchar>(sqrtf(dy * dy + dx * dx) * norm);
    const int bin = qangle(dx, dy);

}

template <int FACTOR>
__device__ __forceinline__ uchar shrink(const uchar* ptr, const int pitch, const int y, const int x)
{
    int out = 0;
#pragma unroll
    for(int dy = 0; dy < FACTOR; ++dy)
#pragma unroll
        for(int dx = 0; dx < FACTOR; ++dx)
        {
            out += ptr[dy * pitch + dx];
        }

    return saturate_cast<uchar>(out / FACTOR);
}

template<int FACTOR>
__global__ void decimate(const uchar* __restrict__ hogluv, uchar* __restrict__ shrank,
                        const int inPitch, const int outPitch )
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    const uchar* ptr = hogluv + (FACTOR * y) * inPitch + (FACTOR * x);

    shrank[ y * outPitch + x]= shrink<FACTOR>(ptr, inPitch, y, x);
}

__global__ void intRow(const uchar* __restrict__ hogluv, ushort* __restrict__ sum,
                       const int inPitch, const int outPitch)
{

}

__global__ void intCol(ushort* __restrict__ sum, const int pitch)
{

}


__global__ void detect(const cv::gpu::icf::Cascade cascade, const uchar* __restrict__ hogluv, const int pitch,
    PtrStepSz<uchar4> objects)
{
    cascade.detectAt(hogluv, pitch, objects);
}

}

float __device icf::Cascade::rescale(const icf::Level& level, uchar4& scaledRect,
                                     const int channel, const float threshold) const
{
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

    // compensation areas rounding
    float rootThreshold = threshold / approx;
    rootThreshold *= level.scaling[(int)(channel > 6)];

    return rootThreshold;
}

typedef unsigned char uchar;
float __device get(const uchar* __restrict__ hogluv, const int pitch,
                   const int x, const int y, int channel, uchar4 area)
{
    const uchar* curr = hogluv + ((channel * 121) + y) * pitch;

    int a = curr[area.y * pitch + x + area.x];
    int b = curr[area.y * pitch + x + area.z];
    int c = curr[area.w * pitch + x + area.z];
    int d = curr[area.w * pitch + x + area.x];

    return (a - b + c - d);
}


void __device icf::Cascade::detectAt(const uchar* __restrict__ hogluv, const int pitch,
                                    PtrStepSz<uchar4>& objects) const
{
    const icf::Level* lls = (const icf::Level*)levels.ptr();
    Level level = lls[0];

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= level.workRect.x || y >= level.workRect.y) return;

    const Octave octave = ((const Octave*)octaves.ptr())[level.octave];
    const int stBegin = octave.index * octave.stages, stEnd = stBegin + octave.stages;

    float detectionScore = 0.f;

    int st = stBegin;
    for(; st < stEnd; ++st)
    {
        const float stage = stages(0, st);
        {
            const int nId = st * 3;

            // work with root node
            const Node node = ((const Node*)nodes.ptr())[nId];
            const Feature feature = ((const Feature*)features.ptr())[node.feature];

            uchar4 scaledRect = feature.rect;
            float threshold = rescale(level, scaledRect, feature.channel, node.threshold);

            float sum = get(hogluv,pitch, x, y, feature.channel, scaledRect);

            int next = 1 + (int)(sum >= threshold);

            // leaves
            const Node leaf = ((const Node*)nodes.ptr())[nId + next];
            const Feature fLeaf = ((const Feature*)features.ptr())[leaf.feature];

            scaledRect = fLeaf.rect;
            threshold = rescale(level, scaledRect, feature.channel, node.threshold);
            sum = get(hogluv, pitch, x, y, fLeaf.channel, scaledRect);

            const int lShift = (next - 1) * 2 + (int)(sum >= threshold);
            float impact = leaves(0, (st * 4) + lShift);

            detectionScore += impact;
        }

        if (detectionScore <= stage) break;
    }

    // if (!threadIdx.x && !threadIdx.y)// printf("%f %d\n", detectionScore, st);
    //     printf("x %d y %d: %d\n", x, y, st);

    if (st == stEnd)
    {
        // printf("  got %d\n", st);
        uchar4 a;
        a.x = level.workRect.x;
        a.y = level.workRect.y;
        objects(0, threadIdx.x) = a;
    }
}

void icf::Cascade::detect(const cv::gpu::PtrStepSzb& hogluv, PtrStepSz<uchar4> objects,
                          cudaStream_t stream) const
{
    // detection kernel
    dim3 block(32, 8, 1);
    // dim3 grid(32 * ChannelStorage::FRAME_WIDTH / 32, ChannelStorage::FRAME_HEIGHT / 8, 1);
    dim3 grid(ChannelStorage::FRAME_WIDTH / 32, ChannelStorage::FRAME_HEIGHT / 8, 1);
    device::detect<<<grid, block, 0, stream>>>(*this, hogluv, hogluv.step / sizeof(ushort), objects);
    cudaSafeCall( cudaGetLastError() );
    if (!stream)
        cudaSafeCall( cudaDeviceSynchronize() );

}

////////////////////////////////////////////////////



void icf::ChannelStorage::frame(const cv::gpu::PtrStepSz<uchar3>& rgb, cudaStream_t stream)
{
//     // // color convertin kernel
//     // dim3 block(32, 8);
//     // dim3 grid(FRAME_WIDTH / 32, FRAME_HEIGHT / 8);

//     // uchar * channels = (uchar*)dmem.ptr(FRAME_HEIGHT * HOG_BINS);
//     // device::rgb2grayluv<<<grid, block, 0, stream>>>((uchar3*)rgb.ptr(), channels,
//     //                                                 rgb.step / sizeof(uchar3), dmem.step);
//     // cudaSafeCall( cudaGetLastError());

//     // // hog calculation kernel
//     // channels = (uchar*)dmem.ptr(FRAME_HEIGHT * HOG_LUV_BINS);
//     // device::gray2hog<<<grid, block, 0, stream>>>(channels, (uchar*)dmem.ptr(), dmem.step, magnitudeScaling);
//     // cudaSafeCall( cudaGetLastError() );

//     // const int shrWidth  = FRAME_WIDTH / shrinkage;
//     // const int shrHeight = FRAME_HEIGHT / shrinkage;

//     // // decimate kernel
//     // grid = dim3(shrWidth / 32, shrHeight / 8);
//     // device::decimate<4><<<grid, block, 0, stream>>>((uchar*)dmem.ptr(), (uchar*)shrunk.ptr(), dmem.step, shrunk.step);
//     // cudaSafeCall( cudaGetLastError() );

//     // // integrate rows
//     // block = dim3(shrWidth, 1);
//     // grid = dim3(shrHeight * HOG_LUV_BINS, 1);
//     // device::intRow<<<grid, block, 0, stream>>>((uchar*)shrunk.ptr(), (ushort*)hogluv.ptr(),
//     //     shrunk.step, hogluv.step / sizeof(ushort));
//     // cudaSafeCall( cudaGetLastError() );

//     // // integrate cols
//     // block = dim3(128, 1);
//     // grid = dim3(shrWidth * HOG_LUV_BINS, 1);
//     // device::intCol<<<grid, block, 0, stream>>>((ushort*)hogluv.ptr(), hogluv.step / hogluv.step / sizeof(ushort));
//     // cudaSafeCall( cudaGetLastError() );
}

}}