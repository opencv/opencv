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
#include "internal_shared.hpp"

namespace cv { namespace gpu { namespace cudev
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

            //cudaSafeCall( cudaUnbindTexture( tex_meanshift ) );
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


        void drawColorDisp_gpu(const PtrStepSzb& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream)
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

        void drawColorDisp_gpu(const PtrStepSz<short>& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream)
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

        template <typename T, typename D>
        __global__ void reprojectImageTo3D(const PtrStepSz<T> disp, PtrStep<D> xyz)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y >= disp.rows || x >= disp.cols)
                return;

            const float qx = x * cq[ 0] + y * cq[ 1] + cq[ 3];
            const float qy = x * cq[ 4] + y * cq[ 5] + cq[ 7];
            const float qz = x * cq[ 8] + y * cq[ 9] + cq[11];
            const float qw = x * cq[12] + y * cq[13] + cq[15];

            const T d = disp(y, x);

            const float iW = 1.f / (qw + cq[14] * d);

            D v = VecTraits<D>::all(1.0f);
            v.x = (qx + cq[2] * d) * iW;
            v.y = (qy + cq[6] * d) * iW;
            v.z = (qz + cq[10] * d) * iW;

            xyz(y, x) = v;
        }

        template <typename T, typename D>
        void reprojectImageTo3D_gpu(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(disp.cols, block.x), divUp(disp.rows, block.y));

            cudaSafeCall( cudaMemcpyToSymbol(cq, q, 16 * sizeof(float)) );

            reprojectImageTo3D<T, D><<<grid, block, 0, stream>>>((PtrStepSz<T>)disp, (PtrStepSz<D>)xyz);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void reprojectImageTo3D_gpu<uchar, float3>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
        template void reprojectImageTo3D_gpu<uchar, float4>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
        template void reprojectImageTo3D_gpu<short, float3>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
        template void reprojectImageTo3D_gpu<short, float4>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);

        /////////////////////////////////////////// Corner Harris /////////////////////////////////////////////////

        texture<float, cudaTextureType2D, cudaReadModeElementType> harrisDxTex(0, cudaFilterModePoint, cudaAddressModeClamp);
        texture<float, cudaTextureType2D, cudaReadModeElementType> harrisDyTex(0, cudaFilterModePoint, cudaAddressModeClamp);

        __global__ void cornerHarris_kernel(const int block_size, const float k, PtrStepSzf dst)
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
        __global__ void cornerHarris_kernel(const int block_size, const float k, PtrStepSzf dst, const BR border_row, const BC border_col)
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

        void cornerHarris_gpu(int block_size, float k, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, cudaStream_t stream)
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

        __global__ void cornerMinEigenVal_kernel(const int block_size, PtrStepSzf dst)
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
        __global__ void cornerMinEigenVal_kernel(const int block_size, PtrStepSzf dst, const BR border_row, const BC border_col)
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

        void cornerMinEigenVal_gpu(int block_size, PtrStepSzf Dx, PtrStepSzf Dy, PtrStepSzf dst, int border_type, cudaStream_t stream)
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


        void buildWarpPlaneMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
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


        void buildWarpCylindricalMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
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


        void buildWarpSphericalMaps(int tl_u, int tl_v, PtrStepSzf map_x, PtrStepSzf map_y,
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
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace cudev {


#endif /* CUDA_DISABLER */
