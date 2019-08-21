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

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        texture<uchar4, 2> tex_meanshift;

        __device__ short2 do_mean_shift(int x0, int y0, unsigned char* out,
                                        size_t out_step, int cols, int rows,
                                        int sp, int sr, int maxIter, float eps)
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

                bool stopFlag = (x0 == x1 && y0 == y1) || (::abs(x1-x0) + ::abs(y1-y0) + norm2 <= eps);

                x0 = x1; y0 = y1;
                c.x = s0; c.y = s1; c.z = s2;

                if( stopFlag )
                    break;
            }

            int base = (blockIdx.y * blockDim.y + threadIdx.y) * out_step + (blockIdx.x * blockDim.x + threadIdx.x) * 4 * sizeof(uchar);
            *(uchar4*)(out + base) = c;

            return make_short2((short)x0, (short)y0);
        }

        __global__ void meanshift_kernel(unsigned char* out, size_t out_step, int cols, int rows, int sp, int sr, int maxIter, float eps )
        {
            int x0 = blockIdx.x * blockDim.x + threadIdx.x;
            int y0 = blockIdx.y * blockDim.y + threadIdx.y;

            if( x0 < cols && y0 < rows )
                do_mean_shift(x0, y0, out, out_step, cols, rows, sp, sr, maxIter, eps);
        }

        void meanShiftFiltering_gpu(const PtrStepSzb& src, PtrStepSzb dst, int sp, int sr, int maxIter, float eps, cudaStream_t stream)
        {
            dim3 grid(1, 1, 1);
            dim3 threads(32, 8, 1);
            grid.x = divUp(src.cols, threads.x);
            grid.y = divUp(src.rows, threads.y);

            cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
            cudaSafeCall( cudaBindTexture2D( 0, tex_meanshift, src.data, desc, src.cols, src.rows, src.step ) );

            meanshift_kernel<<< grid, threads, 0, stream >>>( dst.data, dst.step, dst.cols, dst.rows, sp, sr, maxIter, eps );
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void meanshiftproc_kernel(unsigned char* outr, size_t outrstep,
                                             unsigned char* outsp, size_t outspstep,
                                             int cols, int rows,
                                             int sp, int sr, int maxIter, float eps)
        {
            int x0 = blockIdx.x * blockDim.x + threadIdx.x;
            int y0 = blockIdx.y * blockDim.y + threadIdx.y;

            if( x0 < cols && y0 < rows )
            {
                int basesp = (blockIdx.y * blockDim.y + threadIdx.y) * outspstep + (blockIdx.x * blockDim.x + threadIdx.x) * 2 * sizeof(short);
                *(short2*)(outsp + basesp) = do_mean_shift(x0, y0, outr, outrstep, cols, rows, sp, sr, maxIter, eps);
            }
        }

        void meanShiftProc_gpu(const PtrStepSzb& src, PtrStepSzb dstr, PtrStepSzb dstsp, int sp, int sr, int maxIter, float eps, cudaStream_t stream)
        {
            dim3 grid(1, 1, 1);
            dim3 threads(32, 8, 1);
            grid.x = divUp(src.cols, threads.x);
            grid.y = divUp(src.rows, threads.y);

            cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
            cudaSafeCall( cudaBindTexture2D( 0, tex_meanshift, src.data, desc, src.cols, src.rows, src.step ) );

            meanshiftproc_kernel<<< grid, threads, 0, stream >>>( dstr.data, dstr.step, dstsp.data, dstsp.step, dstr.cols, dstr.rows, sp, sr, maxIter, eps );
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
}}}

#endif
