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
// Copyright (c) 2010, Paul Furgale, Chi Hay Tong
//
// The original code was written by Paul Furgale and Chi Hay Tong 
// and later optimized and prepared for integration into OpenCV by Itseez.
//
//M*/

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/limits.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace pyrlk 
    {
        __constant__ int c_cn;
        __constant__ float c_minEigThreshold;
        __constant__ int c_winSize_x;
        __constant__ int c_winSize_y;
        __constant__ int c_winSize_x_cn;
        __constant__ int c_halfWin_x;
        __constant__ int c_halfWin_y;
        __constant__ int c_iters;

        void loadConstants(int cn, float minEigThreshold, int2 winSize, int iters)
        {
            int2 halfWin = make_int2((winSize.x - 1) / 2, (winSize.y - 1) / 2);            
            cudaSafeCall( cudaMemcpyToSymbol(c_cn, &cn, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_minEigThreshold, &minEigThreshold, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_winSize_x, &winSize.x, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_winSize_y, &winSize.y, sizeof(int)) );
            winSize.x *= cn;
            cudaSafeCall( cudaMemcpyToSymbol(c_winSize_x_cn, &winSize.x, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_halfWin_x, &halfWin.x, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_halfWin_y, &halfWin.y, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_iters, &iters, sizeof(int)) );
        }

        __global__ void calcSharrDeriv_vertical(const PtrStepb src, PtrStep<short> dx_buf, PtrStep<short> dy_buf, int rows, int colsn)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y < rows && x < colsn)
            {
                const uchar src_val0 = src(y > 0 ? y - 1 : 1, x);
                const uchar src_val1 = src(y, x);
                const uchar src_val2 = src(y < rows - 1 ? y + 1 : rows - 2, x);
                
                dx_buf(y, x) = (src_val0 + src_val2) * 3 + src_val1 * 10;
                dy_buf(y, x) = src_val2 - src_val0;
            }
        }

        __global__ void calcSharrDeriv_horizontal(const PtrStep<short> dx_buf, const PtrStep<short> dy_buf, PtrStep<short> dIdx, PtrStep<short> dIdy, int rows, int cols)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const int colsn = cols * c_cn;

            if (y < rows && x < colsn)
            {
                const short* dx_buf_row = dx_buf.ptr(y);
                const short* dy_buf_row = dy_buf.ptr(y);

                const int xr = x + c_cn < colsn ? x + c_cn : (cols - 2) * c_cn + x + c_cn - colsn;
                const int xl = x - c_cn >= 0 ? x - c_cn : c_cn + x;

                dIdx(y, x) = dx_buf_row[xr] - dx_buf_row[xl];
                dIdy(y, x) = (dy_buf_row[xr] + dy_buf_row[xl]) * 3 + dy_buf_row[x] * 10;
            }
        }

        void calcSharrDeriv_gpu(DevMem2Db src, DevMem2D_<short> dx_buf, DevMem2D_<short> dy_buf, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy, int cn, 
            cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(src.cols * cn, block.x), divUp(src.rows, block.y));

            calcSharrDeriv_vertical<<<grid, block, 0, stream>>>(src, dx_buf, dy_buf, src.rows, src.cols * cn);
            cudaSafeCall( cudaGetLastError() );

            calcSharrDeriv_horizontal<<<grid, block, 0, stream>>>(dx_buf, dy_buf, dIdx, dIdy, src.rows, src.cols);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        #define W_BITS 14
        #define W_BITS1 14

        #define  CV_DESCALE(x, n)     (((x) + (1 << ((n)-1))) >> (n))

        __device__ int linearFilter(const PtrStepb& src, float2 pt, int x, int y)
        {
            int2 ipt;
            ipt.x = __float2int_rd(pt.x);
            ipt.y = __float2int_rd(pt.y);

            float a = pt.x - ipt.x;
            float b = pt.y - ipt.y;

            int iw00 = __float2int_rn((1.0f - a) * (1.0f - b) * (1 << W_BITS));
            int iw01 = __float2int_rn(a * (1.0f - b) * (1 << W_BITS));
            int iw10 = __float2int_rn((1.0f - a) * b * (1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            const uchar* src_row = src.ptr(ipt.y + y) + ipt.x * c_cn;
            const uchar* src_row1 = src.ptr(ipt.y + y + 1) + ipt.x * c_cn;

            return CV_DESCALE(src_row[x] * iw00 + src_row[x + c_cn] * iw01 + src_row1[x] * iw10 + src_row1[x + c_cn] * iw11, W_BITS1 - 5);
        }

        __device__ int linearFilter(const PtrStep<short>& src, float2 pt, int x, int y)
        {
            int2 ipt;
            ipt.x = __float2int_rd(pt.x);
            ipt.y = __float2int_rd(pt.y);

            float a = pt.x - ipt.x;
            float b = pt.y - ipt.y;

            int iw00 = __float2int_rn((1.0f - a) * (1.0f - b) * (1 << W_BITS));
            int iw01 = __float2int_rn(a * (1.0f - b) * (1 << W_BITS));
            int iw10 = __float2int_rn((1.0f - a) * b * (1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            const short* src_row = src.ptr(ipt.y + y) + ipt.x * c_cn;
            const short* src_row1 = src.ptr(ipt.y + y + 1) + ipt.x * c_cn;

            return CV_DESCALE(src_row[x] * iw00 + src_row[x + c_cn] * iw01 + src_row1[x] * iw10 + src_row1[x + c_cn] * iw11, W_BITS1);
        }

        __device__ void reduce(float& val1, float& val2, float& val3, float* smem1, float* smem2, float* smem3, int tid)
        {
            smem1[tid] = val1;
            smem2[tid] = val2;
            smem3[tid] = val3;
            __syncthreads();

            if (tid < 128) 
            { 
                smem1[tid] = val1 += smem1[tid + 128]; 
                smem2[tid] = val2 += smem2[tid + 128]; 
                smem3[tid] = val3 += smem3[tid + 128]; 
            } 
            __syncthreads();

            if (tid < 64) 
            { 
                smem1[tid] = val1 += smem1[tid + 64]; 
                smem2[tid] = val2 += smem2[tid + 64]; 
                smem3[tid] = val3 += smem3[tid + 64];
            } 
            __syncthreads();

            if (tid < 32)
            {
                volatile float* vmem1 = smem1;
                volatile float* vmem2 = smem2;
                volatile float* vmem3 = smem3;

                vmem1[tid] = val1 += vmem1[tid + 32]; 
                vmem2[tid] = val2 += vmem2[tid + 32]; 
                vmem3[tid] = val3 += vmem3[tid + 32];

                vmem1[tid] = val1 += vmem1[tid + 16]; 
                vmem2[tid] = val2 += vmem2[tid + 16]; 
                vmem3[tid] = val3 += vmem3[tid + 16];

                vmem1[tid] = val1 += vmem1[tid + 8]; 
                vmem2[tid] = val2 += vmem2[tid + 8]; 
                vmem3[tid] = val3 += vmem3[tid + 8];

                vmem1[tid] = val1 += vmem1[tid + 4]; 
                vmem2[tid] = val2 += vmem2[tid + 4]; 
                vmem3[tid] = val3 += vmem3[tid + 4];

                vmem1[tid] = val1 += vmem1[tid + 2]; 
                vmem2[tid] = val2 += vmem2[tid + 2]; 
                vmem3[tid] = val3 += vmem3[tid + 2];

                vmem1[tid] = val1 += vmem1[tid + 1]; 
                vmem2[tid] = val2 += vmem2[tid + 1]; 
                vmem3[tid] = val3 += vmem3[tid + 1];
            }
        }

        __device__ void reduce(float& val1, float& val2, float* smem1, float* smem2, int tid)
        {
            smem1[tid] = val1;
            smem2[tid] = val2;
            __syncthreads();

            if (tid < 128) 
            { 
                smem1[tid] = val1 += smem1[tid + 128]; 
                smem2[tid] = val2 += smem2[tid + 128];  
            } 
            __syncthreads();

            if (tid < 64) 
            { 
                smem1[tid] = val1 += smem1[tid + 64]; 
                smem2[tid] = val2 += smem2[tid + 64]; 
            } 
            __syncthreads();

            if (tid < 32)
            {
                volatile float* vmem1 = smem1;
                volatile float* vmem2 = smem2;

                vmem1[tid] = val1 += vmem1[tid + 32]; 
                vmem2[tid] = val2 += vmem2[tid + 32]; 

                vmem1[tid] = val1 += vmem1[tid + 16]; 
                vmem2[tid] = val2 += vmem2[tid + 16]; 

                vmem1[tid] = val1 += vmem1[tid + 8]; 
                vmem2[tid] = val2 += vmem2[tid + 8]; 

                vmem1[tid] = val1 += vmem1[tid + 4]; 
                vmem2[tid] = val2 += vmem2[tid + 4]; 

                vmem1[tid] = val1 += vmem1[tid + 2]; 
                vmem2[tid] = val2 += vmem2[tid + 2]; 

                vmem1[tid] = val1 += vmem1[tid + 1]; 
                vmem2[tid] = val2 += vmem2[tid + 1]; 
            }
        }

        __device__ void reduce(float& val1, float* smem1, int tid)
        {
            smem1[tid] = val1;
            __syncthreads();

            if (tid < 128) 
            { 
                smem1[tid] = val1 += smem1[tid + 128]; 
            } 
            __syncthreads();

            if (tid < 64) 
            { 
                smem1[tid] = val1 += smem1[tid + 64]; 
            } 
            __syncthreads();

            if (tid < 32)
            {
                volatile float* vmem1 = smem1;

                vmem1[tid] = val1 += vmem1[tid + 32]; 
                vmem1[tid] = val1 += vmem1[tid + 16]; 
                vmem1[tid] = val1 += vmem1[tid + 8]; 
                vmem1[tid] = val1 += vmem1[tid + 4];
                vmem1[tid] = val1 += vmem1[tid + 2]; 
                vmem1[tid] = val1 += vmem1[tid + 1]; 
            }
        }

        #define SCALE (1.0f / (1 << 20))

        template <int PATCH_X, int PATCH_Y, bool calcErr, bool GET_MIN_EIGENVALS>
        __global__ void lkSparse(const PtrStepb I, const PtrStepb J, const PtrStep<short> dIdx, const PtrStep<short> dIdy,
            const float2* prevPts, float2* nextPts, uchar* status, float* err, const int level, const int rows, const int cols)
        {
            __shared__ float smem1[256];
            __shared__ float smem2[256];
            __shared__ float smem3[256];

            const int tid = threadIdx.y * blockDim.x + threadIdx.x;

            float2 prevPt = prevPts[blockIdx.x];
            prevPt.x *= (1.0f / (1 << level));
            prevPt.y *= (1.0f / (1 << level));

            prevPt.x -= c_halfWin_x;
            prevPt.y -= c_halfWin_y;

            if (prevPt.x < -c_winSize_x || prevPt.x >= cols || prevPt.y < -c_winSize_y || prevPt.y >= rows)
            {
                if (level == 0 && tid == 0)
                {
                    status[blockIdx.x] = 0;

                    if (calcErr) 
                        err[blockIdx.x] = 0;
                }

                return;
            }

            // extract the patch from the first image, compute covariation matrix of derivatives
            
            float A11 = 0;
            float A12 = 0;
            float A22 = 0;

            int I_patch[PATCH_Y][PATCH_X];
            int dIdx_patch[PATCH_Y][PATCH_X];
            int dIdy_patch[PATCH_Y][PATCH_X];

            for (int y = threadIdx.y, i = 0; y < c_winSize_y; y += blockDim.y, ++i)
            {                
                for (int x = threadIdx.x, j = 0; x < c_winSize_x_cn; x += blockDim.x, ++j)
                {
                    I_patch[i][j] = linearFilter(I, prevPt, x, y);

                    int ixval = linearFilter(dIdx, prevPt, x, y);
                    int iyval = linearFilter(dIdy, prevPt, x, y);

                    dIdx_patch[i][j] = ixval;
                    dIdy_patch[i][j] = iyval;
                    
                    A11 += ixval * ixval;
                    A12 += ixval * iyval;
                    A22 += iyval * iyval;
                }
            }

            reduce(A11, A12, A22, smem1, smem2, smem3, tid);
            __syncthreads();

            A11 = smem1[0];
            A12 = smem2[0];
            A22 = smem3[0];
            
            A11 *= SCALE;
            A12 *= SCALE;
            A22 *= SCALE;

            {
                float D = A11 * A22 - A12 * A12;
                float minEig = (A22 + A11 - ::sqrtf((A11 - A22) * (A11 - A22) + 4.f * A12 * A12)) / (2 * c_winSize_x * c_winSize_y);
            
                if (calcErr && GET_MIN_EIGENVALS && tid == 0) 
                    err[blockIdx.x] = minEig;

                if (minEig < c_minEigThreshold || D < numeric_limits<float>::epsilon())
                {
                    if (level == 0 && tid == 0)
                        status[blockIdx.x] = 0;

                    return;
                }

                D = 1.f / D;
            
                A11 *= D;
                A12 *= D;
                A22 *= D;
            }

            float2 nextPt = nextPts[blockIdx.x];
            nextPt.x *= 2.f;
            nextPt.y *= 2.f; 
            
            nextPt.x -= c_halfWin_x;
            nextPt.y -= c_halfWin_y;

            bool status_ = true;

            for (int k = 0; k < c_iters; ++k)
            {
                if (nextPt.x < -c_winSize_x || nextPt.x >= cols || nextPt.y < -c_winSize_y || nextPt.y >= rows)
                {
                    status_ = false;
                    break;
                }

                float b1 = 0;
                float b2 = 0;
                
                for (int y = threadIdx.y, i = 0; y < c_winSize_y; y += blockDim.y, ++i)
                {
                    for (int x = threadIdx.x, j = 0; x < c_winSize_x_cn; x += blockDim.x, ++j)
                    {
                        int diff = linearFilter(J, nextPt, x, y) - I_patch[i][j];

                        b1 += diff * dIdx_patch[i][j];
                        b2 += diff * dIdy_patch[i][j];
                    }
                }
                
                reduce(b1, b2, smem1, smem2, tid);
                __syncthreads();

                b1 = smem1[0];
                b2 = smem2[0];

                b1 *= SCALE;
                b2 *= SCALE;
                    
                float2 delta;
                delta.x = A12 * b2 - A22 * b1;
                delta.y = A12 * b1 - A11 * b2;
                    
                nextPt.x += delta.x;
                nextPt.y += delta.y;

                if (::fabs(delta.x) < 0.01f && ::fabs(delta.y) < 0.01f)
                {
                    nextPt.x -= delta.x * 0.5f;
                    nextPt.y -= delta.y * 0.5f;
                    break;
                }
            }

            if (nextPt.x < -c_winSize_x || nextPt.x >= cols || nextPt.y < -c_winSize_y || nextPt.y >= rows)
                status_ = false;

            // TODO : Why do we compute patch error in shifted window?
            nextPt.x += c_halfWin_x;
            nextPt.y += c_halfWin_y;

            float errval = 0.f;
            if (calcErr && !GET_MIN_EIGENVALS && status_)
            {
                for (int y = threadIdx.y, i = 0; y < c_winSize_y; y += blockDim.y, ++i)
                {
                    for (int x = threadIdx.x, j = 0; x < c_winSize_x_cn; x += blockDim.x, ++j)
                    {
                        int diff = linearFilter(J, nextPt, x, y) - I_patch[i][j];
                        errval += ::fabsf((float)diff);
                    }
                }

                reduce(errval, smem1, tid);

                errval /= 32 * c_winSize_x_cn * c_winSize_y;
            }

            if (tid == 0)
            {
                status[blockIdx.x] = status_;
                nextPts[blockIdx.x] = nextPt;

                if (calcErr && !GET_MIN_EIGENVALS)
                    err[blockIdx.x] = errval;
            }
        }

        template <int PATCH_X, int PATCH_Y>
        void lkSparse_caller(DevMem2Db I, DevMem2Db J, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy,
            const float2* prevPts, float2* nextPts, uchar* status, float* err, bool GET_MIN_EIGENVALS, int ptcount, 
            int level, dim3 block, cudaStream_t stream)
        {
            dim3 grid(ptcount);

            if (level == 0 && err)
            {
                if (GET_MIN_EIGENVALS)
                {
                    cudaSafeCall( cudaFuncSetCacheConfig(lkSparse<PATCH_X, PATCH_Y, true, true>, cudaFuncCachePreferL1) );

                    lkSparse<PATCH_X, PATCH_Y, true, true><<<grid, block>>>(I, J, dIdx, dIdy,
                        prevPts, nextPts, status, err, level, I.rows, I.cols);
                }
                else
                {
                    cudaSafeCall( cudaFuncSetCacheConfig(lkSparse<PATCH_X, PATCH_Y, true, false>, cudaFuncCachePreferL1) );

                    lkSparse<PATCH_X, PATCH_Y, true, false><<<grid, block>>>(I, J, dIdx, dIdy,
                        prevPts, nextPts, status, err, level, I.rows, I.cols);
                }
            }
            else
            {
                cudaSafeCall( cudaFuncSetCacheConfig(lkSparse<PATCH_X, PATCH_Y, false, false>, cudaFuncCachePreferL1) );

                lkSparse<PATCH_X, PATCH_Y, false, false><<<grid, block>>>(I, J, dIdx, dIdy,
                        prevPts, nextPts, status, err, level, I.rows, I.cols);
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void lkSparse_gpu(DevMem2Db I, DevMem2Db J, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy,
            const float2* prevPts, float2* nextPts, uchar* status, float* err, bool GET_MIN_EIGENVALS, int ptcount, 
            int level, dim3 block, dim3 patch, cudaStream_t stream)
        {
            typedef void (*func_t)(DevMem2Db I, DevMem2Db J, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy,
                const float2* prevPts, float2* nextPts, uchar* status, float* err, bool GET_MIN_EIGENVALS, int ptcount, 
                int level, dim3 block, cudaStream_t stream);

            static const func_t funcs[5][5] = 
            {
                {lkSparse_caller<1, 1>, lkSparse_caller<2, 1>, lkSparse_caller<3, 1>, lkSparse_caller<4, 1>, lkSparse_caller<5, 1>},
                {lkSparse_caller<1, 2>, lkSparse_caller<2, 2>, lkSparse_caller<3, 2>, lkSparse_caller<4, 2>, lkSparse_caller<5, 2>},
                {lkSparse_caller<1, 3>, lkSparse_caller<2, 3>, lkSparse_caller<3, 3>, lkSparse_caller<4, 3>, lkSparse_caller<5, 3>},
                {lkSparse_caller<1, 4>, lkSparse_caller<2, 4>, lkSparse_caller<3, 4>, lkSparse_caller<4, 4>, lkSparse_caller<5, 4>},
                {lkSparse_caller<1, 5>, lkSparse_caller<2, 5>, lkSparse_caller<3, 5>, lkSparse_caller<4, 5>, lkSparse_caller<5, 5>}
            };            

            funcs[patch.y - 1][patch.x - 1](I, J, dIdx, dIdy,
                prevPts, nextPts, status, err, GET_MIN_EIGENVALS, ptcount, 
                level, block, stream);
        }

        template <bool calcErr, bool GET_MIN_EIGENVALS>
        __global__ void lkDense(const PtrStepb I, const PtrStepb J, const PtrStep<short> dIdx, const PtrStep<short> dIdy,
            PtrStepf u, PtrStepf v, PtrStepf err, const int rows, const int cols)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= cols || y >= rows)
                return;

            // extract the patch from the first image, compute covariation matrix of derivatives
            
            float A11 = 0;
            float A12 = 0;
            float A22 = 0;

            for (int i = 0; i < c_winSize_y; ++i)
            {                
                for (int j = 0; j < c_winSize_x; ++j)
                {
                    int ixval = dIdx(y - c_halfWin_y + i, x - c_halfWin_x + j);
                    int iyval = dIdy(y - c_halfWin_y + i, x - c_halfWin_x + j);

                    A11 += ixval * ixval;
                    A12 += ixval * iyval;
                    A22 += iyval * iyval;
                }
            }
            
            A11 *= SCALE;
            A12 *= SCALE;
            A22 *= SCALE;

            {
                float D = A11 * A22 - A12 * A12;
                float minEig = (A22 + A11 - ::sqrtf((A11 - A22) * (A11 - A22) + 4.f * A12 * A12)) / (2 * c_winSize_x * c_winSize_y);

                if (calcErr && GET_MIN_EIGENVALS)
                    err(y, x) = minEig;
            
                if (minEig < c_minEigThreshold || D < numeric_limits<float>::epsilon())
                    return;

                D = 1.f / D;
            
                A11 *= D;
                A12 *= D;
                A22 *= D;
            }

            float2 nextPt;
            nextPt.x = x - c_halfWin_x + u(y, x);
            nextPt.y = y - c_halfWin_y + v(y, x);

            for (int k = 0; k < c_iters; ++k)
            {
                if (nextPt.x < -c_winSize_x || nextPt.x >= cols || nextPt.y < -c_winSize_y || nextPt.y >= rows)
                    return;

                float b1 = 0;
                float b2 = 0;
                
                for (int i = 0; i < c_winSize_y; ++i)
                {
                    for (int j = 0; j < c_winSize_x; ++j)
                    {
                        int I_val = I(y - c_halfWin_y + i, x - c_halfWin_x + j);

                        int diff = linearFilter(J, nextPt, j, i) - CV_DESCALE(I_val * (1 << W_BITS), W_BITS1 - 5);
                        
                        b1 += diff * dIdx(y - c_halfWin_y + i, x - c_halfWin_x + j);
                        b2 += diff * dIdy(y - c_halfWin_y + i, x - c_halfWin_x + j);
                    }
                }

                b1 *= SCALE;
                b2 *= SCALE;

                float2 delta;
                delta.x = A12 * b2 - A22 * b1;
                delta.y = A12 * b1 - A11 * b2;
                    
                nextPt.x += delta.x;
                nextPt.y += delta.y;

                if (::fabs(delta.x) < 0.01f && ::fabs(delta.y) < 0.01f)
                    break;
            }            

            // TODO : Why do we compute patch error in shifted window?
            nextPt.x += c_halfWin_x;
            nextPt.y += c_halfWin_y;

            u(y, x) = nextPt.x - x;
            v(y, x) = nextPt.y - y;            

            if (calcErr && !GET_MIN_EIGENVALS)
            {
                float errval = 0.0f;

                for (int i = 0; i < c_winSize_y; ++i)
                {
                    for (int j = 0; j < c_winSize_x; ++j)
                    {
                        int I_val = I(y - c_halfWin_y + i, x - c_halfWin_x + j);
                        int diff = linearFilter(J, nextPt, j, i) - CV_DESCALE(I_val * (1 << W_BITS), W_BITS1 - 5);
                        errval += ::fabsf((float)diff);
                    }
                }

                errval /= 32 * c_winSize_x_cn * c_winSize_y;

                err(y, x) = errval;
            }
        }

        void lkDense_gpu(DevMem2Db I, DevMem2Db J, DevMem2D_<short> dIdx, DevMem2D_<short> dIdy, 
            DevMem2Df u, DevMem2Df v, DevMem2Df* err, bool GET_MIN_EIGENVALS, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(I.cols, block.x), divUp(I.rows, block.y));

            if (err)
            {
                if (GET_MIN_EIGENVALS)
                {
                    cudaSafeCall( cudaFuncSetCacheConfig(lkDense<true, true>, cudaFuncCachePreferL1) );

                    lkDense<true, true><<<grid, block, 0, stream>>>(I, J, dIdx, dIdy, u, v, *err, I.rows, I.cols);
                    cudaSafeCall( cudaGetLastError() );
                }
                else
                {
                    cudaSafeCall( cudaFuncSetCacheConfig(lkDense<true, false>, cudaFuncCachePreferL1) );

                    lkDense<true, false><<<grid, block, 0, stream>>>(I, J, dIdx, dIdy, u, v, *err, I.rows, I.cols);
                    cudaSafeCall( cudaGetLastError() );
                }
            }
            else
            {
                cudaSafeCall( cudaFuncSetCacheConfig(lkDense<false, false>, cudaFuncCachePreferL1) );

                lkDense<false, false><<<grid, block, 0, stream>>>(I, J, dIdx, dIdy, u, v, PtrStepf(), I.rows, I.cols);
                cudaSafeCall( cudaGetLastError() );
            }

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
}}}
