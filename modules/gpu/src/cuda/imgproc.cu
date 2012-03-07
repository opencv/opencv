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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace imgproc 
    {
        /////////////////////////////////// MeanShiftfiltering ///////////////////////////////////////////////

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

        void meanShiftFiltering_gpu(const DevMem2Db& src, DevMem2Db dst, int sp, int sr, int maxIter, float eps, cudaStream_t stream)
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

            //cudaSafeCall( cudaUnbindTexture( tex_meanshift ) );        
        }

        void meanShiftProc_gpu(const DevMem2Db& src, DevMem2Db dstr, DevMem2Db dstsp, int sp, int sr, int maxIter, float eps, cudaStream_t stream) 
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

            //cudaSafeCall( cudaUnbindTexture( tex_meanshift ) );        
        }

        /////////////////////////////////// drawColorDisp ///////////////////////////////////////////////

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
            const unsigned int b = (unsigned int)(::max(0.f, ::min(res.x, 1.f)) * 255.f);
            const unsigned int g = (unsigned int)(::max(0.f, ::min(res.y, 1.f)) * 255.f);
            const unsigned int r = (unsigned int)(::max(0.f, ::min(res.z, 1.f)) * 255.f);
            const unsigned int a = 255U;

            return (a << 24) + (r << 16) + (g << 8) + b;    
        } 

        __global__ void drawColorDisp(uchar* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
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

        __global__ void drawColorDisp(short* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
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


        void drawColorDisp_gpu(const DevMem2Db& src, const DevMem2Db& dst, int ndisp, const cudaStream_t& stream)
        {
            dim3 threads(16, 16, 1);
            dim3 grid(1, 1, 1);
            grid.x = divUp(src.cols, threads.x << 2);
            grid.y = divUp(src.rows, threads.y);
             
            drawColorDisp<<<grid, threads, 0, stream>>>(src.data, src.step, dst.data, dst.step, src.cols, src.rows, ndisp);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() ); 
        }

        void drawColorDisp_gpu(const DevMem2D_<short>& src, const DevMem2Db& dst, int ndisp, const cudaStream_t& stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);
            grid.x = divUp(src.cols, threads.x << 1);
            grid.y = divUp(src.rows, threads.y);
             
            drawColorDisp<<<grid, threads, 0, stream>>>(src.data, src.step / sizeof(short), dst.data, dst.step, src.cols, src.rows, ndisp);
            cudaSafeCall( cudaGetLastError() );
            
            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        /////////////////////////////////// reprojectImageTo3D ///////////////////////////////////////////////

        __constant__ float cq[16];

        template <typename T>
        __global__ void reprojectImageTo3D(const T* disp, size_t disp_step, float* xyzw, size_t xyzw_step, int rows, int cols)
        {        
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < rows && x < cols)
            {

                float qx = cq[1] * y + cq[3], qy = cq[5] * y + cq[7];
                float qz = cq[9] * y + cq[11], qw = cq[13] * y + cq[15];

                qx += x * cq[0]; 
                qy += x * cq[4];
                qz += x * cq[8];
                qw += x * cq[12];

                T d = *(disp + disp_step * y + x);

                float iW = 1.f / (qw + cq[14] * d);
                float4 v;
                v.x = (qx + cq[2] * d) * iW;
                v.y = (qy + cq[6] * d) * iW;
                v.z = (qz + cq[10] * d) * iW;
                v.w = 1.f;

                *(float4*)(xyzw + xyzw_step * y + (x * 4)) = v;
            }
        }

        template <typename T>
        inline void reprojectImageTo3D_caller(const DevMem2D_<T>& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream)
        {
            dim3 threads(32, 8, 1);
            dim3 grid(1, 1, 1);
            grid.x = divUp(disp.cols, threads.x);
            grid.y = divUp(disp.rows, threads.y);

            cudaSafeCall( cudaMemcpyToSymbol(cq, q, 16 * sizeof(float)) );

            reprojectImageTo3D<<<grid, threads, 0, stream>>>(disp.data, disp.step / sizeof(T), xyzw.data, xyzw.step / sizeof(float), disp.rows, disp.cols);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void reprojectImageTo3D_gpu(const DevMem2Db& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream)
        {
            reprojectImageTo3D_caller(disp, xyzw, q, stream);
        }

        void reprojectImageTo3D_gpu(const DevMem2D_<short>& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream)
        {
            reprojectImageTo3D_caller(disp, xyzw, q, stream);
        }

        /////////////////////////////////////////// Corner Harris /////////////////////////////////////////////////

        texture<float, cudaTextureType2D, cudaReadModeElementType> harrisDxTex(0, cudaFilterModePoint, cudaAddressModeClamp);
        texture<float, cudaTextureType2D, cudaReadModeElementType> harrisDyTex(0, cudaFilterModePoint, cudaAddressModeClamp);

        __global__ void cornerHarris_kernel(const int block_size, const float k, DevMem2Df dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    for (int j = jbegin; j < jend; ++j)
                    {
                        float dx = tex2D(harrisDxTex, j, i);
                        float dy = tex2D(harrisDyTex, j, i);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                dst(y, x) = a * c - b * b - k * (a + c) * (a + c);
            }
        }

        template <typename BR, typename BC>
        __global__ void cornerHarris_kernel(const int block_size, const float k, DevMem2Df dst, const BR border_row, const BC border_col)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    const int y = border_col.idx_row(i);

                    for (int j = jbegin; j < jend; ++j)
                    {
                        const int x = border_row.idx_col(j);

                        float dx = tex2D(harrisDxTex, x, y);
                        float dy = tex2D(harrisDyTex, x, y);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                dst(y, x) = a * c - b * b - k * (a + c) * (a + c);
            }
        }

        void cornerHarris_gpu(int block_size, float k, DevMem2Df Dx, DevMem2Df Dy, DevMem2Df dst, int border_type, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(Dx.cols, block.x), divUp(Dx.rows, block.y));

            bindTexture(&harrisDxTex, Dx);
            bindTexture(&harrisDyTex, Dy);

            switch (border_type) 
            {
            case BORDER_REFLECT101_GPU:
                cornerHarris_kernel<<<grid, block, 0, stream>>>(block_size, k, dst, BrdRowReflect101<void>(Dx.cols), BrdColReflect101<void>(Dx.rows));
                break;

            case BORDER_REFLECT_GPU:
                cornerHarris_kernel<<<grid, block, 0, stream>>>(block_size, k, dst, BrdRowReflect<void>(Dx.cols), BrdColReflect<void>(Dx.rows));
                break;

            case BORDER_REPLICATE_GPU:
                cornerHarris_kernel<<<grid, block, 0, stream>>>(block_size, k, dst);
                break;
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        /////////////////////////////////////////// Corner Min Eigen Val /////////////////////////////////////////////////

        texture<float, cudaTextureType2D, cudaReadModeElementType> minEigenValDxTex(0, cudaFilterModePoint, cudaAddressModeClamp);
        texture<float, cudaTextureType2D, cudaReadModeElementType> minEigenValDyTex(0, cudaFilterModePoint, cudaAddressModeClamp);

        __global__ void cornerMinEigenVal_kernel(const int block_size, DevMem2Df dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    for (int j = jbegin; j < jend; ++j)
                    {
                        float dx = tex2D(minEigenValDxTex, j, i);
                        float dy = tex2D(minEigenValDyTex, j, i);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                a *= 0.5f;
                c *= 0.5f;

                dst(y, x) = (a + c) - sqrtf((a - c) * (a - c) + b * b);
            }
        }


        template <typename BR, typename BC>
        __global__ void cornerMinEigenVal_kernel(const int block_size, DevMem2Df dst, const BR border_row, const BC border_col)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                float a = 0.f;
                float b = 0.f;
                float c = 0.f;

                const int ibegin = y - (block_size / 2);
                const int jbegin = x - (block_size / 2);
                const int iend = ibegin + block_size;
                const int jend = jbegin + block_size;

                for (int i = ibegin; i < iend; ++i)
                {
                    int y = border_col.idx_row(i);

                    for (int j = jbegin; j < jend; ++j)
                    {
                        int x = border_row.idx_col(j);

                        float dx = tex2D(minEigenValDxTex, x, y);
                        float dy = tex2D(minEigenValDyTex, x, y);

                        a += dx * dx;
                        b += dx * dy;
                        c += dy * dy;
                    }
                }

                a *= 0.5f;
                c *= 0.5f;

                dst(y, x) = (a + c) - sqrtf((a - c) * (a - c) + b * b);
            }
        }

        void cornerMinEigenVal_gpu(int block_size, DevMem2Df Dx, DevMem2Df Dy, DevMem2Df dst, int border_type, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(Dx.cols, block.x), divUp(Dx.rows, block.y));
            
            bindTexture(&minEigenValDxTex, Dx);
            bindTexture(&minEigenValDyTex, Dy);

            switch (border_type)
            {
            case BORDER_REFLECT101_GPU:
                cornerMinEigenVal_kernel<<<grid, block, 0, stream>>>(block_size, dst, BrdRowReflect101<void>(Dx.cols), BrdColReflect101<void>(Dx.rows));
                break;

            case BORDER_REFLECT_GPU:
                cornerMinEigenVal_kernel<<<grid, block, 0, stream>>>(block_size, dst, BrdRowReflect<void>(Dx.cols), BrdColReflect<void>(Dx.rows));
                break;

            case BORDER_REPLICATE_GPU:
                cornerMinEigenVal_kernel<<<grid, block, 0, stream>>>(block_size, dst);
                break;
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }

        ////////////////////////////// Column Sum //////////////////////////////////////

        __global__ void column_sumKernel_32F(int cols, int rows, const PtrStepb src, const PtrStepb dst)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x < cols)
            {
                const unsigned char* src_data = src.data + x * sizeof(float);
                unsigned char* dst_data = dst.data + x * sizeof(float);

                float sum = 0.f;
                for (int y = 0; y < rows; ++y)
                {
                    sum += *(const float*)src_data;
                    *(float*)dst_data = sum;
                    src_data += src.step;
                    dst_data += dst.step;
                }
            }
        }


        void columnSum_32F(const DevMem2Db src, const DevMem2Db dst)
        {
            dim3 threads(256);
            dim3 grid(divUp(src.cols, threads.x));

            column_sumKernel_32F<<<grid, threads>>>(src.cols, src.rows, src, dst);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }


        //////////////////////////////////////////////////////////////////////////
        // mulSpectrums

        __global__ void mulSpectrumsKernel(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, DevMem2D_<cufftComplex> c)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;    
            const int y = blockIdx.y * blockDim.y + threadIdx.y;    

            if (x < c.cols && y < c.rows) 
            {
                c.ptr(y)[x] = cuCmulf(a.ptr(y)[x], b.ptr(y)[x]);
            }
        }


        void mulSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, DevMem2D_<cufftComplex> c, cudaStream_t stream)
        {
            dim3 threads(256);
            dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

            mulSpectrumsKernel<<<grid, threads, 0, stream>>>(a, b, c);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }


        //////////////////////////////////////////////////////////////////////////
        // mulSpectrums_CONJ

        __global__ void mulSpectrumsKernel_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, DevMem2D_<cufftComplex> c)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;    
            const int y = blockIdx.y * blockDim.y + threadIdx.y;    

            if (x < c.cols && y < c.rows) 
            {
                c.ptr(y)[x] = cuCmulf(a.ptr(y)[x], cuConjf(b.ptr(y)[x]));
            }
        }


        void mulSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, DevMem2D_<cufftComplex> c, cudaStream_t stream)
        {
            dim3 threads(256);
            dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

            mulSpectrumsKernel_CONJ<<<grid, threads, 0, stream>>>(a, b, c);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }


        //////////////////////////////////////////////////////////////////////////
        // mulAndScaleSpectrums

        __global__ void mulAndScaleSpectrumsKernel(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, DevMem2D_<cufftComplex> c)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < c.cols && y < c.rows) 
            {
                cufftComplex v = cuCmulf(a.ptr(y)[x], b.ptr(y)[x]);
                c.ptr(y)[x] = make_cuFloatComplex(cuCrealf(v) * scale, cuCimagf(v) * scale);
            }
        }


        void mulAndScaleSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, DevMem2D_<cufftComplex> c, cudaStream_t stream)
        {
            dim3 threads(256);
            dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

            mulAndScaleSpectrumsKernel<<<grid, threads, 0, stream>>>(a, b, scale, c);
            cudaSafeCall( cudaGetLastError() );

            if (stream)
                cudaSafeCall( cudaDeviceSynchronize() );
        }


        //////////////////////////////////////////////////////////////////////////
        // mulAndScaleSpectrums_CONJ

        __global__ void mulAndScaleSpectrumsKernel_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, DevMem2D_<cufftComplex> c)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < c.cols && y < c.rows) 
            {
                cufftComplex v = cuCmulf(a.ptr(y)[x], cuConjf(b.ptr(y)[x]));
                c.ptr(y)[x] = make_cuFloatComplex(cuCrealf(v) * scale, cuCimagf(v) * scale);
            }
        }


        void mulAndScaleSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, DevMem2D_<cufftComplex> c, cudaStream_t stream)
        {
            dim3 threads(256);
            dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

            mulAndScaleSpectrumsKernel_CONJ<<<grid, threads, 0, stream>>>(a, b, scale, c);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }    

        //////////////////////////////////////////////////////////////////////////
        // buildWarpMaps

        // TODO use intrinsics like __sinf and so on

        namespace build_warp_maps
        {

            __constant__ float ck_rinv[9];
            __constant__ float cr_kinv[9];
            __constant__ float ct[3];
            __constant__ float cscale;
        }


        class PlaneMapper
        {
        public:
            static __device__ __forceinline__ void mapBackward(float u, float v, float &x, float &y)
            {
                using namespace build_warp_maps;

                float x_ = u / cscale - ct[0];
                float y_ = v / cscale - ct[1];

                float z;
                x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * (1 - ct[2]);
                y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * (1 - ct[2]);
                z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * (1 - ct[2]);

                x /= z;
                y /= z;
            }
        };


        class CylindricalMapper
        {
        public:
            static __device__ __forceinline__ void mapBackward(float u, float v, float &x, float &y)
            {
                using namespace build_warp_maps;

                u /= cscale;
                float x_ = ::sinf(u);
                float y_ = v / cscale;
                float z_ = ::cosf(u);

                float z;
                x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * z_;
                y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * z_;
                z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * z_;

                if (z > 0) { x /= z; y /= z; }
                else x = y = -1;
            }
        };


        class SphericalMapper
        {
        public:
            static __device__ __forceinline__ void mapBackward(float u, float v, float &x, float &y)
            {
                using namespace build_warp_maps;

                v /= cscale;
                u /= cscale;

                float sinv = ::sinf(v);
                float x_ = sinv * ::sinf(u);
                float y_ = -::cosf(v);
                float z_ = sinv * ::cosf(u);

                float z;
                x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * z_;
                y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * z_;
                z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * z_;

                if (z > 0) { x /= z; y /= z; }
                else x = y = -1;
            }
        };


        template <typename Mapper>
        __global__ void buildWarpMapsKernel(int tl_u, int tl_v, int cols, int rows,
                                            PtrStepf map_x, PtrStepf map_y)
        {
            int du = blockIdx.x * blockDim.x + threadIdx.x;
            int dv = blockIdx.y * blockDim.y + threadIdx.y;
            if (du < cols && dv < rows)
            {
                float u = tl_u + du;
                float v = tl_v + dv;
                float x, y;
                Mapper::mapBackward(u, v, x, y);
                map_x.ptr(dv)[du] = x;
                map_y.ptr(dv)[du] = y;
            }
        }


        void buildWarpPlaneMaps(int tl_u, int tl_v, DevMem2Df map_x, DevMem2Df map_y,
                                const float k_rinv[9], const float r_kinv[9], const float t[3], 
                                float scale, cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ck_rinv, k_rinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cr_kinv, r_kinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ct, t, 3*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cscale, &scale, sizeof(float)));

            int cols = map_x.cols;
            int rows = map_x.rows;

            dim3 threads(32, 8);
            dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

            buildWarpMapsKernel<PlaneMapper><<<grid,threads>>>(tl_u, tl_v, cols, rows, map_x, map_y);
            cudaSafeCall(cudaGetLastError());
            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        void buildWarpCylindricalMaps(int tl_u, int tl_v, DevMem2Df map_x, DevMem2Df map_y,
                                      const float k_rinv[9], const float r_kinv[9], float scale,
                                      cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ck_rinv, k_rinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cr_kinv, r_kinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cscale, &scale, sizeof(float)));

            int cols = map_x.cols;
            int rows = map_x.rows;

            dim3 threads(32, 8);
            dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

            buildWarpMapsKernel<CylindricalMapper><<<grid,threads>>>(tl_u, tl_v, cols, rows, map_x, map_y);
            cudaSafeCall(cudaGetLastError());
            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        void buildWarpSphericalMaps(int tl_u, int tl_v, DevMem2Df map_x, DevMem2Df map_y,
                                    const float k_rinv[9], const float r_kinv[9], float scale,
                                    cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::ck_rinv, k_rinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cr_kinv, r_kinv, 9*sizeof(float)));
            cudaSafeCall(cudaMemcpyToSymbol(build_warp_maps::cscale, &scale, sizeof(float)));

            int cols = map_x.cols;
            int rows = map_x.rows;

            dim3 threads(32, 8);
            dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

            buildWarpMapsKernel<SphericalMapper><<<grid,threads>>>(tl_u, tl_v, cols, rows, map_x, map_y);
            cudaSafeCall(cudaGetLastError());
            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }

        //////////////////////////////////////////////////////////////////////////
        // filter2D

        #define FILTER2D_MAX_KERNEL_SIZE 16

        __constant__ float c_filter2DKernel[FILTER2D_MAX_KERNEL_SIZE * FILTER2D_MAX_KERNEL_SIZE];

        texture<float, cudaTextureType2D, cudaReadModeElementType> filter2DTex(0, cudaFilterModePoint, cudaAddressModeBorder);

        __global__ void filter2D(int ofsX, int ofsY, DevMem2Df dst, const int kWidth, const int kHeight, const int anchorX, const int anchorY)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= dst.cols || y >= dst.rows)
                return;

            float res = 0;

            const int baseX = ofsX + x - anchorX;
            const int baseY = ofsY + y - anchorY;

            int kInd = 0;

            for (int i = 0; i < kHeight; ++i)
            {
                for (int j = 0; j < kWidth; ++j)
                    res += tex2D(filter2DTex, baseX + j, baseY + i) * c_filter2DKernel[kInd++];
            }

            dst.ptr(y)[x] = res;
        }

        void filter2D_gpu(DevMem2Df src, int ofsX, int ofsY, DevMem2Df dst, int kWidth, int kHeight, int anchorX, int anchorY, float* kernel, cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(c_filter2DKernel, kernel, kWidth * kHeight * sizeof(float), 0, cudaMemcpyDeviceToDevice) );

            const dim3 block(16, 16);
            const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

            bindTexture(&filter2DTex, src);

            filter2D<<<grid, block, 0, stream>>>(ofsX, ofsY, dst, kWidth, kHeight, anchorX, anchorY);
            cudaSafeCall(cudaGetLastError());

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device {
