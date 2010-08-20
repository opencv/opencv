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

#include "cuda_shared.hpp"

using namespace cv::gpu;


/////////////////////////////////// Remap ///////////////////////////////////////////////
namespace imgproc
{
    texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex_remap;

    __global__ void kernel_remap(const float *mapx, const float *mapy, size_t map_step, unsigned char* out, size_t out_step, int width, int height)
    {    
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x < width && y < height)
        {
            int idx = y * (map_step >> 2) + x; /* map_step >> 2  <=> map_step / sizeof(float)*/

            float xcoo = mapx[idx];
            float ycoo = mapy[idx];

            out[y * out_step + x] = (unsigned char)(255.f * tex2D(tex_remap, xcoo, ycoo));            
        }
    }

}

namespace cv { namespace gpu { namespace impl 
{
    extern "C" void remap_gpu(const DevMem2D& src, const DevMem2D_<float>& xmap, const DevMem2D_<float>& ymap, DevMem2D dst)
    {
        dim3 block(16, 16, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(dst.cols, block.x);
        grid.y = divUp(dst.rows, block.y);

        imgproc::tex_remap.filterMode = cudaFilterModeLinear;	    
        imgproc::tex_remap.addressMode[0] = imgproc::tex_remap.addressMode[1] = cudaAddressModeWrap;
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
        cudaSafeCall( cudaBindTexture2D(0, imgproc::tex_remap, src.ptr, desc, dst.cols, dst.rows, src.step) );

        imgproc::kernel_remap<<<grid, block>>>(xmap.ptr, ymap.ptr, xmap.step, dst.ptr, dst.step, dst.cols, dst.rows);

        cudaSafeCall( cudaThreadSynchronize() );  
        cudaSafeCall( cudaUnbindTexture(imgproc::tex_remap) );
    }
}}}


/////////////////////////////////// MeanShiftfiltering ///////////////////////////////////////////////

namespace imgproc
{
    texture<uchar4, 2> tex_meanshift;

    extern "C" __global__ void meanshift_kernel( unsigned char* out, int out_step, int cols, int rows, int sp, int sr, int maxIter, float eps )
    {
        int x0 = blockIdx.x * blockDim.x + threadIdx.x;
        int y0 = blockIdx.y * blockDim.y + threadIdx.y;

        if( x0 < cols && y0 < rows )
        {
            int isr2 = sr*sr;
            uchar4 c = tex2D(tex_meanshift, x0, y0 );
            // iterate meanshift procedure
            for( int iter = 0; iter < maxIter; iter++ )
            {
                int count = 0;
                int s0 = 0, s1 = 0, s2 = 0, sx = 0, sy = 0;
                float icount;

                //mean shift: process pixels in window (p-sigmaSp)x(p+sigmaSp)
                int minx = x0-sp;
                int miny = y0-sp;
                int maxx = x0+sp;
                int maxy = y0+sp;

                for( int y = miny; y <= maxy; y++)
                {
                    int rowCount = 0;
                    for( int x = minx; x <= maxx; x++ )
                    {                    
                        uchar4 t = tex2D( tex_meanshift, x, y );

                        int norm2 = (t.x - c.x) * (t.x - c.x) + (t.y - c.y) * (t.y - c.y) + (t.z - c.z) * (t.z - c.z);
                        if( norm2 <= isr2 )
                        {
                            s0 += t.x; s1 += t.y; s2 += t.z;
                            sx += x; rowCount++;
                        }
                    }
                    count += rowCount;
                    sy += y*rowCount;
                }

                if( count == 0 )
                    break;

                icount = 1.f/count;
                int x1 = __float2int_rz(sx*icount);
                int y1 = __float2int_rz(sy*icount);
                s0 = __float2int_rz(s0*icount);
                s1 = __float2int_rz(s1*icount);
                s2 = __float2int_rz(s2*icount);

                int norm2 = (s0 - c.x) * (s0 - c.x) + (s1 - c.y) * (s1 - c.y) + (s2 - c.z) * (s2 - c.z);

                bool stopFlag = (x0 == x1 && y0 == y1) || (abs(x1-x0) + abs(y1-y0) + norm2 <= eps);

                x0 = x1; y0 = y1;
                c.x = s0; c.y = s1; c.z = s2;

                if( stopFlag )
                    break;
            }

            int base = (blockIdx.y * blockDim.y + threadIdx.y) * out_step + (blockIdx.x * blockDim.x + threadIdx.x) * 3 * sizeof(uchar);
            out[base+0] = c.x;
            out[base+1] = c.y;
            out[base+2] = c.z;
        }
    }
}

namespace cv { namespace gpu { namespace impl 
{
    extern "C" void meanShiftFiltering_gpu(const DevMem2D& src, DevMem2D dst, int sp, int sr, int maxIter, float eps)
    {                        
        dim3  grid(1, 1, 1);
        dim3 threads(32, 16, 1);
        grid.x = divUp(src.cols, threads.x);
        grid.y = divUp(src.rows, threads.y);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        cudaSafeCall( cudaBindTexture2D( 0, imgproc::tex_meanshift, src.ptr, desc, src.cols, src.rows, src.step ) );

        imgproc::meanshift_kernel<<< grid, threads >>>( dst.ptr, dst.step, dst.cols, dst.rows, sp, sr, maxIter, eps );
        cudaSafeCall( cudaThreadSynchronize() );
        cudaSafeCall( cudaUnbindTexture( imgproc::tex_meanshift ) );        
    }
}}}

/////////////////////////////////// colorizeDisp ///////////////////////////////////////////////

namespace imgproc
{
    template <typename T>
    __device__ unsigned int cvtPixel(T d, int ndisp, float S = 1, float V = 1)
    {        
        unsigned int H = ((ndisp-d) * 240)/ndisp;

        unsigned int hi = (H/60) % 6;
        float f = H/60.f - H/60;
        float p = V * (1 - S);
        float q = V * (1 - f * S);
        float t = V * (1 - (1 - f) * S);

        float3 res;
        
        if (hi == 0) //R = V,	G = t,	B = p
        {
            res.x = p;
            res.y = t;
            res.z = V;
        }

        if (hi == 1) // R = q,	G = V,	B = p
        {
            res.x = p;
            res.y = V;
            res.z = q;
        }        
        
        if (hi == 2) // R = p,	G = V,	B = t
        {
            res.x = t;
            res.y = V;
            res.z = p;
        }
            
        if (hi == 3) // R = p,	G = q,	B = V
        {
            res.x = V;
            res.y = q;
            res.z = p;
        }

        if (hi == 4) // R = t,	G = p,	B = V
        {
            res.x = V;
            res.y = p;
            res.z = t;
        }

        if (hi == 5) // R = V,	G = p,	B = q
        {
            res.x = q;
            res.y = p;
            res.z = V;
        }
        unsigned int b = (unsigned int)(max(0.f, min (res.x, 1.f)) * 255.f);
        unsigned int g = (unsigned int)(max(0.f, min (res.y, 1.f)) * 255.f);
        unsigned int r = (unsigned int)(max(0.f, min (res.z, 1.f)) * 255.f);

        return (r << 16) + (g << 8) + b;    
    } 

    __global__ void colorizeDisp(uchar* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height) 
        {
            uchar4 d4 = *(uchar4*)(disp + y * disp_step + x);

            uint4 res;
            res.x = cvtPixel(d4.x, ndisp);
            res.y = cvtPixel(d4.y, ndisp);
            res.z = cvtPixel(d4.z, ndisp);
            res.w = cvtPixel(d4.w, ndisp);
                    
            uint4* line = (uint4*)(out_image + y * out_step);
            line[x >> 2] = res;
        }
    }

    __global__ void colorizeDisp(short* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height) 
        {
            short2 d2 = *(short2*)(disp + y * disp_step + x);

            uint2 res;
            res.x = cvtPixel(d2.x, ndisp);            
            res.y = cvtPixel(d2.y, ndisp);

            uint2* line = (uint2*)(out_image + y * out_step);
            line[x >> 1] = res;
        }
    }
}

namespace cv { namespace gpu { namespace impl 
{
    void colorizeDisp_gpu(const DevMem2D& src, const DevMem2D& dst, int ndisp)
    {
        dim3 threads(16, 16, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(src.cols, threads.x << 2);
        grid.y = divUp(src.rows, threads.y);
         
        imgproc::colorizeDisp<<<grid, threads>>>(src.ptr, src.step, dst.ptr, dst.step, src.cols, src.rows, ndisp);
        cudaThreadSynchronize(); 
    }

    void colorizeDisp_gpu(const DevMem2D_<short>& src, const DevMem2D& dst, int ndisp)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(src.cols, threads.x << 1);
        grid.y = divUp(src.rows, threads.y);
         
        imgproc::colorizeDisp<<<grid, threads>>>(src.ptr, src.step / sizeof(short), dst.ptr, dst.step, src.cols, src.rows, ndisp);
        cudaThreadSynchronize();
    }
}}}
