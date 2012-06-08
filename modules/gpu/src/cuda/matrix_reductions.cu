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
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/vec_math.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace matrix_reductions 
    {
        // Performs reduction in shared memory
        template <int size, typename T>
        __device__ void sumInSmem(volatile T* data, const uint tid)
        {
            T sum = data[tid];

            if (size >= 512) { if (tid < 256) { data[tid] = sum = sum + data[tid + 256]; } __syncthreads(); }
            if (size >= 256) { if (tid < 128) { data[tid] = sum = sum + data[tid + 128]; } __syncthreads(); }
            if (size >= 128) { if (tid < 64) { data[tid] = sum = sum + data[tid + 64]; } __syncthreads(); }

            if (tid < 32)
            {
                if (size >= 64) data[tid] = sum = sum + data[tid + 32];
                if (size >= 32) data[tid] = sum = sum + data[tid + 16];
                if (size >= 16) data[tid] = sum = sum + data[tid + 8];
                if (size >= 8) data[tid] = sum = sum + data[tid + 4];
                if (size >= 4) data[tid] = sum = sum + data[tid + 2];
                if (size >= 2) data[tid] = sum = sum + data[tid + 1];
            }
        }

        struct Mask8U
        {
            explicit Mask8U(PtrStepb mask): mask(mask) {}

            __device__ __forceinline__ bool operator()(int y, int x) const 
            { 
                return mask.ptr(y)[x]; 
            }

            PtrStepb mask;
        };

        struct MaskTrue 
        { 
            __device__ __forceinline__ bool operator()(int y, int x) const 
            { 
                return true; 
            }
            __device__ __forceinline__ MaskTrue(){}
            __device__ __forceinline__ MaskTrue(const MaskTrue& mask_){}
        };

        //////////////////////////////////////////////////////////////////////////////
        // Min max

        // To avoid shared bank conflicts we convert each value into value of 
        // appropriate type (32 bits minimum)
        template <typename T> struct MinMaxTypeTraits {};
        template <> struct MinMaxTypeTraits<uchar> { typedef int best_type; };
        template <> struct MinMaxTypeTraits<char> { typedef int best_type; };
        template <> struct MinMaxTypeTraits<ushort> { typedef int best_type; };
        template <> struct MinMaxTypeTraits<short> { typedef int best_type; };
        template <> struct MinMaxTypeTraits<int> { typedef int best_type; };
        template <> struct MinMaxTypeTraits<float> { typedef float best_type; };
        template <> struct MinMaxTypeTraits<double> { typedef double best_type; };

        namespace minmax 
        {
            __constant__ int ctwidth;
            __constant__ int ctheight;

            // Global counter of blocks finished its work
            __device__ uint blocks_finished = 0;


            // Estimates good thread configuration
            //  - threads variable satisfies to threads.x * threads.y == 256
            void estimateThreadCfg(int cols, int rows, dim3& threads, dim3& grid)
            {
                threads = dim3(32, 8);
                grid = dim3(divUp(cols, threads.x * 8), divUp(rows, threads.y * 32));
                grid.x = std::min(grid.x, threads.x);
                grid.y = std::min(grid.y, threads.y);
            }


            // Returns required buffer sizes
            void getBufSizeRequired(int cols, int rows, int elem_size, int& bufcols, int& bufrows)
            {
                dim3 threads, grid;
                estimateThreadCfg(cols, rows, threads, grid);
                bufcols = grid.x * grid.y * elem_size; 
                bufrows = 2;
            }


            // Estimates device constants which are used in the kernels using specified thread configuration
            void setKernelConsts(int cols, int rows, const dim3& threads, const dim3& grid)
            {        
                int twidth = divUp(divUp(cols, grid.x), threads.x);
                int theight = divUp(divUp(rows, grid.y), threads.y);
                cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(ctwidth))); 
                cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(ctheight))); 
            }  


            // Does min and max in shared memory
            template <typename T>
            __device__ __forceinline__ void merge(uint tid, uint offset, volatile T* minval, volatile T* maxval)
            {
                minval[tid] = ::min(minval[tid], minval[tid + offset]);
                maxval[tid] = ::max(maxval[tid], maxval[tid + offset]);
            }


            template <int size, typename T>
            __device__ void findMinMaxInSmem(volatile T* minval, volatile T* maxval, const uint tid)
            {
                if (size >= 512) { if (tid < 256) { merge(tid, 256, minval, maxval); } __syncthreads(); }
                if (size >= 256) { if (tid < 128) { merge(tid, 128, minval, maxval); }  __syncthreads(); }
                if (size >= 128) { if (tid < 64) { merge(tid, 64, minval, maxval); } __syncthreads(); }

                if (tid < 32)
                {
                    if (size >= 64) merge(tid, 32, minval, maxval);
                    if (size >= 32) merge(tid, 16, minval, maxval);
                    if (size >= 16) merge(tid, 8, minval, maxval);
                    if (size >= 8) merge(tid, 4, minval, maxval);
                    if (size >= 4) merge(tid, 2, minval, maxval);
                    if (size >= 2) merge(tid, 1, minval, maxval);
                }
            }


            template <int nthreads, typename T, typename Mask>
            __global__ void minMaxKernel(const DevMem2Db src, Mask mask, T* minval, T* maxval)
            {
                typedef typename MinMaxTypeTraits<T>::best_type best_type;
                __shared__ best_type sminval[nthreads];
                __shared__ best_type smaxval[nthreads];

                uint x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
                uint y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
                uint tid = threadIdx.y * blockDim.x + threadIdx.x;

                T mymin = numeric_limits<T>::max();
                T mymax = numeric_limits<T>::is_signed ? -numeric_limits<T>::max() : numeric_limits<T>::min();
                uint y_end = ::min(y0 + (ctheight - 1) * blockDim.y + 1, src.rows);
                uint x_end = ::min(x0 + (ctwidth - 1) * blockDim.x + 1, src.cols);
                for (uint y = y0; y < y_end; y += blockDim.y)
                {
                    const T* src_row = (const T*)src.ptr(y);
                    for (uint x = x0; x < x_end; x += blockDim.x)
                    {
                        T val = src_row[x];
                        if (mask(y, x)) 
                        { 
                            mymin = ::min(mymin, val); 
                            mymax = ::max(mymax, val); 
                        }
                    }
                }

                sminval[tid] = mymin;
                smaxval[tid] = mymax;
                __syncthreads();

                findMinMaxInSmem<nthreads, best_type>(sminval, smaxval, tid);

                if (tid == 0) 
                {
                    minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
                    maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
                }

            #if __CUDA_ARCH__ >= 110
		        __shared__ bool is_last;

		        if (tid == 0)
		        {
			        minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
                    maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
			        __threadfence();

			        uint ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
			        is_last = ticket == gridDim.x * gridDim.y - 1;
		        }

		        __syncthreads();

		        if (is_last)
		        {
                    uint idx = ::min(tid, gridDim.x * gridDim.y - 1);

                    sminval[tid] = minval[idx];
                    smaxval[tid] = maxval[idx];
                    __syncthreads();

			        findMinMaxInSmem<nthreads, best_type>(sminval, smaxval, tid);

                    if (tid == 0) 
                    {
                        minval[0] = (T)sminval[0];
                        maxval[0] = (T)smaxval[0];
                        blocks_finished = 0;
                    }
		        }
            #else
                if (tid == 0) 
                {
                    minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
                    maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
                }
            #endif
            }

   
            template <typename T>
            void minMaxMaskCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)buf.ptr(0);
                T* maxval_buf = (T*)buf.ptr(1);

                minMaxKernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                T minval_, maxval_;
                cudaSafeCall( cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                cudaSafeCall( cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                *minval = minval_;
                *maxval = maxval_;
            }  

            template void minMaxMaskCaller<uchar>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskCaller<char>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskCaller<ushort>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskCaller<short>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskCaller<int>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskCaller<float>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskCaller<double>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);


            template <typename T>
            void minMaxCaller(const DevMem2Db src, double* minval, double* maxval, PtrStepb buf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)buf.ptr(0);
                T* maxval_buf = (T*)buf.ptr(1);

                minMaxKernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                T minval_, maxval_;
                cudaSafeCall( cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                cudaSafeCall( cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                *minval = minval_;
                *maxval = maxval_;
            }  

            template void minMaxCaller<uchar>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxCaller<char>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxCaller<ushort>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxCaller<short>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxCaller<int>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxCaller<float>(const DevMem2Db, double*,double*, PtrStepb);
            template void minMaxCaller<double>(const DevMem2Db, double*, double*, PtrStepb);


            template <int nthreads, typename T>
            __global__ void minMaxPass2Kernel(T* minval, T* maxval, int size)
            {
                typedef typename MinMaxTypeTraits<T>::best_type best_type;
                __shared__ best_type sminval[nthreads];
                __shared__ best_type smaxval[nthreads];
                
                uint tid = threadIdx.y * blockDim.x + threadIdx.x;
                uint idx = ::min(tid, size - 1);

                sminval[tid] = minval[idx];
                smaxval[tid] = maxval[idx];
                __syncthreads();

                findMinMaxInSmem<nthreads, best_type>(sminval, smaxval, tid);

                if (tid == 0) 
                {
                    minval[0] = (T)sminval[0];
                    maxval[0] = (T)smaxval[0];
                }
            }


            template <typename T>
            void minMaxMaskMultipassCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)buf.ptr(0);
                T* maxval_buf = (T*)buf.ptr(1);

                minMaxKernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf);
                cudaSafeCall( cudaGetLastError() );
                minMaxPass2Kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, grid.x * grid.y);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall(cudaDeviceSynchronize());

                T minval_, maxval_;
                cudaSafeCall( cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                cudaSafeCall( cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                *minval = minval_;
                *maxval = maxval_;
            }

            template void minMaxMaskMultipassCaller<uchar>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskMultipassCaller<char>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskMultipassCaller<ushort>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskMultipassCaller<short>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskMultipassCaller<int>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);
            template void minMaxMaskMultipassCaller<float>(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);


            template <typename T>
            void minMaxMultipassCaller(const DevMem2Db src, double* minval, double* maxval, PtrStepb buf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)buf.ptr(0);
                T* maxval_buf = (T*)buf.ptr(1);

                minMaxKernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf);
                cudaSafeCall( cudaGetLastError() );
                minMaxPass2Kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, grid.x * grid.y);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                T minval_, maxval_;
                cudaSafeCall( cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                cudaSafeCall( cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                *minval = minval_;
                *maxval = maxval_;
            }

            template void minMaxMultipassCaller<uchar>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxMultipassCaller<char>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxMultipassCaller<ushort>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxMultipassCaller<short>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxMultipassCaller<int>(const DevMem2Db, double*, double*, PtrStepb);
            template void minMaxMultipassCaller<float>(const DevMem2Db, double*, double*, PtrStepb);
        } // namespace minmax

        ///////////////////////////////////////////////////////////////////////////////
        // minMaxLoc

        namespace minmaxloc 
        {
            __constant__ int ctwidth;
            __constant__ int ctheight;

            // Global counter of blocks finished its work
            __device__ uint blocks_finished = 0;


            // Estimates good thread configuration
            //  - threads variable satisfies to threads.x * threads.y == 256
            void estimateThreadCfg(int cols, int rows, dim3& threads, dim3& grid)
            {
                threads = dim3(32, 8);
                grid = dim3(divUp(cols, threads.x * 8), divUp(rows, threads.y * 32));
                grid.x = std::min(grid.x, threads.x);
                grid.y = std::min(grid.y, threads.y);
            }


            // Returns required buffer sizes
            void getBufSizeRequired(int cols, int rows, int elem_size, int& b1cols, 
                                    int& b1rows, int& b2cols, int& b2rows)
            {
                dim3 threads, grid;
                estimateThreadCfg(cols, rows, threads, grid);
                b1cols = grid.x * grid.y * elem_size; // For values
                b1rows = 2;
                b2cols = grid.x * grid.y * sizeof(int); // For locations
                b2rows = 2;
            }


            // Estimates device constants which are used in the kernels using specified thread configuration
            void setKernelConsts(int cols, int rows, const dim3& threads, const dim3& grid)
            {        
                int twidth = divUp(divUp(cols, grid.x), threads.x);
                int theight = divUp(divUp(rows, grid.y), threads.y);
                cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(ctwidth))); 
                cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(ctheight))); 
            }  


            template <typename T>
            __device__ void merge(uint tid, uint offset, volatile T* minval, volatile T* maxval, 
                                  volatile uint* minloc, volatile uint* maxloc)
            {
                T val = minval[tid + offset];
                if (val < minval[tid])
                {
                    minval[tid] = val;
                    minloc[tid] = minloc[tid + offset];
                }
                val = maxval[tid + offset];
                if (val > maxval[tid])
                {
                    maxval[tid] = val;
                    maxloc[tid] = maxloc[tid + offset];
                }
            }


            template <int size, typename T>
            __device__ void findMinMaxLocInSmem(volatile T* minval, volatile T* maxval, volatile uint* minloc, 
                                                volatile uint* maxloc, const uint tid)
            {
                if (size >= 512) { if (tid < 256) { merge(tid, 256, minval, maxval, minloc, maxloc); } __syncthreads(); }
                if (size >= 256) { if (tid < 128) { merge(tid, 128, minval, maxval, minloc, maxloc); }  __syncthreads(); }
                if (size >= 128) { if (tid < 64) { merge(tid, 64, minval, maxval, minloc, maxloc); } __syncthreads(); }

                if (tid < 32)
                {
                    if (size >= 64) merge(tid, 32, minval, maxval, minloc, maxloc);
                    if (size >= 32) merge(tid, 16, minval, maxval, minloc, maxloc);
                    if (size >= 16) merge(tid, 8, minval, maxval, minloc, maxloc);
                    if (size >= 8) merge(tid, 4, minval, maxval, minloc, maxloc);
                    if (size >= 4) merge(tid, 2, minval, maxval, minloc, maxloc);
                    if (size >= 2) merge(tid, 1, minval, maxval, minloc, maxloc);
                }
            }


            template <int nthreads, typename T, typename Mask>
            __global__ void minMaxLocKernel(const DevMem2Db src, Mask mask, T* minval, T* maxval, 
                                            uint* minloc, uint* maxloc)
            {
                typedef typename MinMaxTypeTraits<T>::best_type best_type;
                __shared__ best_type sminval[nthreads];
                __shared__ best_type smaxval[nthreads];
                __shared__ uint sminloc[nthreads];
                __shared__ uint smaxloc[nthreads];

                uint x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
                uint y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
                uint tid = threadIdx.y * blockDim.x + threadIdx.x;

                T mymin = numeric_limits<T>::max();
                T mymax = numeric_limits<T>::is_signed ? -numeric_limits<T>::max() : numeric_limits<T>::min(); 
                uint myminloc = 0;
                uint mymaxloc = 0;
                uint y_end = ::min(y0 + (ctheight - 1) * blockDim.y + 1, src.rows);
                uint x_end = ::min(x0 + (ctwidth - 1) * blockDim.x + 1, src.cols);

                for (uint y = y0; y < y_end; y += blockDim.y)
                {
                    const T* ptr = (const T*)src.ptr(y);
                    for (uint x = x0; x < x_end; x += blockDim.x)
                    {
                        if (mask(y, x))
                        {
                            T val = ptr[x];
                            if (val <= mymin) { mymin = val; myminloc = y * src.cols + x; }
                            if (val >= mymax) { mymax = val; mymaxloc = y * src.cols + x; }
                        }
                    }
                }

                sminval[tid] = mymin; 
                smaxval[tid] = mymax;
                sminloc[tid] = myminloc;
                smaxloc[tid] = mymaxloc;
                __syncthreads();

                findMinMaxLocInSmem<nthreads, best_type>(sminval, smaxval, sminloc, smaxloc, tid);

            #if __CUDA_ARCH__ >= 110
		        __shared__ bool is_last;

		        if (tid == 0)
		        {
			        minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
                    maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
                    minloc[blockIdx.y * gridDim.x + blockIdx.x] = sminloc[0];
                    maxloc[blockIdx.y * gridDim.x + blockIdx.x] = smaxloc[0];
			        __threadfence();

			        uint ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
			        is_last = ticket == gridDim.x * gridDim.y - 1;
		        }

		        __syncthreads();

		        if (is_last)
		        {
                    uint idx = ::min(tid, gridDim.x * gridDim.y - 1);

                    sminval[tid] = minval[idx];
                    smaxval[tid] = maxval[idx];
                    sminloc[tid] = minloc[idx];
                    smaxloc[tid] = maxloc[idx];
                    __syncthreads();

			        findMinMaxLocInSmem<nthreads, best_type>(sminval, smaxval, sminloc, smaxloc, tid);

                    if (tid == 0) 
                    {
                        minval[0] = (T)sminval[0];
                        maxval[0] = (T)smaxval[0];
                        minloc[0] = sminloc[0];
                        maxloc[0] = smaxloc[0];
                        blocks_finished = 0;
                    }
		        }
            #else
                if (tid == 0) 
                {
                    minval[blockIdx.y * gridDim.x + blockIdx.x] = (T)sminval[0];
                    maxval[blockIdx.y * gridDim.x + blockIdx.x] = (T)smaxval[0];
                    minloc[blockIdx.y * gridDim.x + blockIdx.x] = sminloc[0];
                    maxloc[blockIdx.y * gridDim.x + blockIdx.x] = smaxloc[0];
                }
            #endif
            }


            template <typename T>
            void minMaxLocMaskCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, 
                                     int minloc[2], int maxloc[2], PtrStepb valbuf, PtrStepb locbuf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)valbuf.ptr(0);
                T* maxval_buf = (T*)valbuf.ptr(1);
                uint* minloc_buf = (uint*)locbuf.ptr(0);
                uint* maxloc_buf = (uint*)locbuf.ptr(1);

                minMaxLocKernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf, 
                                                                   minloc_buf, maxloc_buf);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                T minval_, maxval_;
                cudaSafeCall( cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                cudaSafeCall( cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
                *minval = minval_;
                *maxval = maxval_;

                uint minloc_, maxloc_;
                cudaSafeCall( cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost) );
                cudaSafeCall( cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost) );
                minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
                maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
            }

            template void minMaxLocMaskCaller<uchar>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskCaller<char>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskCaller<ushort>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskCaller<short>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskCaller<int>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskCaller<float>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskCaller<double>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);


            template <typename T>
            void minMaxLocCaller(const DevMem2Db src, double* minval, double* maxval, 
                                 int minloc[2], int maxloc[2], PtrStepb valbuf, PtrStepb locbuf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)valbuf.ptr(0);
                T* maxval_buf = (T*)valbuf.ptr(1);
                uint* minloc_buf = (uint*)locbuf.ptr(0);
                uint* maxloc_buf = (uint*)locbuf.ptr(1);

                minMaxLocKernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf, 
                                                                     minloc_buf, maxloc_buf);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                T minval_, maxval_;
                cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
                *minval = minval_;
                *maxval = maxval_;

                uint minloc_, maxloc_;
                cudaSafeCall(cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
                minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
                maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
            }

            template void minMaxLocCaller<uchar>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocCaller<char>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocCaller<ushort>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocCaller<short>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocCaller<int>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocCaller<float>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocCaller<double>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);


            // This kernel will be used only when compute capability is 1.0
            template <int nthreads, typename T>
            __global__ void minMaxLocPass2Kernel(T* minval, T* maxval, uint* minloc, uint* maxloc, int size)
            {
                typedef typename MinMaxTypeTraits<T>::best_type best_type;
                __shared__ best_type sminval[nthreads];
                __shared__ best_type smaxval[nthreads];
                __shared__ uint sminloc[nthreads];
                __shared__ uint smaxloc[nthreads];

                uint tid = threadIdx.y * blockDim.x + threadIdx.x;
                uint idx = ::min(tid, size - 1);

                sminval[tid] = minval[idx];
                smaxval[tid] = maxval[idx];
                sminloc[tid] = minloc[idx];
                smaxloc[tid] = maxloc[idx];
                __syncthreads();

                findMinMaxLocInSmem<nthreads, best_type>(sminval, smaxval, sminloc, smaxloc, tid);

                if (tid == 0) 
                {
                    minval[0] = (T)sminval[0];
                    maxval[0] = (T)smaxval[0];
                    minloc[0] = sminloc[0];
                    maxloc[0] = smaxloc[0];
                }
            }


            template <typename T>
            void minMaxLocMaskMultipassCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, 
                                              int minloc[2], int maxloc[2], PtrStepb valbuf, PtrStepb locbuf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)valbuf.ptr(0);
                T* maxval_buf = (T*)valbuf.ptr(1);
                uint* minloc_buf = (uint*)locbuf.ptr(0);
                uint* maxloc_buf = (uint*)locbuf.ptr(1);

                minMaxLocKernel<256, T, Mask8U><<<grid, threads>>>(src, Mask8U(mask), minval_buf, maxval_buf, 
                                                                   minloc_buf, maxloc_buf);
                cudaSafeCall( cudaGetLastError() );
                minMaxLocPass2Kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, minloc_buf, maxloc_buf, grid.x * grid.y);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                T minval_, maxval_;
                cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
                *minval = minval_;
                *maxval = maxval_;

                uint minloc_, maxloc_;
                cudaSafeCall(cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
                minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
                maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
            }

            template void minMaxLocMaskMultipassCaller<uchar>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskMultipassCaller<char>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskMultipassCaller<ushort>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskMultipassCaller<short>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskMultipassCaller<int>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMaskMultipassCaller<float>(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);


            template <typename T>
            void minMaxLocMultipassCaller(const DevMem2Db src, double* minval, double* maxval, 
                                          int minloc[2], int maxloc[2], PtrStepb valbuf, PtrStepb locbuf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                T* minval_buf = (T*)valbuf.ptr(0);
                T* maxval_buf = (T*)valbuf.ptr(1);
                uint* minloc_buf = (uint*)locbuf.ptr(0);
                uint* maxloc_buf = (uint*)locbuf.ptr(1);

                minMaxLocKernel<256, T, MaskTrue><<<grid, threads>>>(src, MaskTrue(), minval_buf, maxval_buf, 
                                                                     minloc_buf, maxloc_buf);
                cudaSafeCall( cudaGetLastError() );
                minMaxLocPass2Kernel<256, T><<<1, 256>>>(minval_buf, maxval_buf, minloc_buf, maxloc_buf, grid.x * grid.y);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                T minval_, maxval_;
                cudaSafeCall(cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost));
                *minval = minval_;
                *maxval = maxval_;

                uint minloc_, maxloc_;
                cudaSafeCall(cudaMemcpy(&minloc_, minloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(&maxloc_, maxloc_buf, sizeof(int), cudaMemcpyDeviceToHost));
                minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
                maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
            }

            template void minMaxLocMultipassCaller<uchar>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMultipassCaller<char>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMultipassCaller<ushort>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMultipassCaller<short>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMultipassCaller<int>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
            template void minMaxLocMultipassCaller<float>(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
        } // namespace minmaxloc

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        // countNonZero

        namespace countnonzero 
        {
            __constant__ int ctwidth;
            __constant__ int ctheight;

            __device__ uint blocks_finished = 0;

            void estimateThreadCfg(int cols, int rows, dim3& threads, dim3& grid)
            {
                threads = dim3(32, 8);
                grid = dim3(divUp(cols, threads.x * 8), divUp(rows, threads.y * 32));
                grid.x = std::min(grid.x, threads.x);
                grid.y = std::min(grid.y, threads.y);
            }


            void getBufSizeRequired(int cols, int rows, int& bufcols, int& bufrows)
            {
                dim3 threads, grid;
                estimateThreadCfg(cols, rows, threads, grid);
                bufcols = grid.x * grid.y * sizeof(int);
                bufrows = 1;
            }


            void setKernelConsts(int cols, int rows, const dim3& threads, const dim3& grid)
            {        
                int twidth = divUp(divUp(cols, grid.x), threads.x);
                int theight = divUp(divUp(rows, grid.y), threads.y);
                cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(twidth))); 
                cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(theight))); 
            }


            template <int nthreads, typename T>
            __global__ void countNonZeroKernel(const DevMem2Db src, volatile uint* count)
            {
                __shared__ uint scount[nthreads];

                uint x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
                uint y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
                uint tid = threadIdx.y * blockDim.x + threadIdx.x;

		        uint cnt = 0;
                for (uint y = 0; y < ctheight && y0 + y * blockDim.y < src.rows; ++y)
                {
                    const T* ptr = (const T*)src.ptr(y0 + y * blockDim.y);
                    for (uint x = 0; x < ctwidth && x0 + x * blockDim.x < src.cols; ++x)
				        cnt += ptr[x0 + x * blockDim.x] != 0;
		        }

		        scount[tid] = cnt;
		        __syncthreads();

                sumInSmem<nthreads, uint>(scount, tid);

            #if __CUDA_ARCH__ >= 110
		        __shared__ bool is_last;

		        if (tid == 0)
		        {
			        count[blockIdx.y * gridDim.x + blockIdx.x] = scount[0];
			        __threadfence();

			        uint ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
			        is_last = ticket == gridDim.x * gridDim.y - 1;
		        }

		        __syncthreads();

		        if (is_last)
		        {
                    scount[tid] = tid < gridDim.x * gridDim.y ? count[tid] : 0;
                    __syncthreads();

			        sumInSmem<nthreads, uint>(scount, tid);

			        if (tid == 0) 
                    {
                        count[0] = scount[0];
                        blocks_finished = 0;
                    }
		        }
            #else
                if (tid == 0) count[blockIdx.y * gridDim.x + blockIdx.x] = scount[0];
            #endif
            }

           
            template <typename T>
            int countNonZeroCaller(const DevMem2Db src, PtrStepb buf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                uint* count_buf = (uint*)buf.ptr(0);

                countNonZeroKernel<256, T><<<grid, threads>>>(src, count_buf);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                uint count;
                cudaSafeCall(cudaMemcpy(&count, count_buf, sizeof(int), cudaMemcpyDeviceToHost));
                
                return count;
            }  

            template int countNonZeroCaller<uchar>(const DevMem2Db, PtrStepb);
            template int countNonZeroCaller<char>(const DevMem2Db, PtrStepb);
            template int countNonZeroCaller<ushort>(const DevMem2Db, PtrStepb);
            template int countNonZeroCaller<short>(const DevMem2Db, PtrStepb);
            template int countNonZeroCaller<int>(const DevMem2Db, PtrStepb);
            template int countNonZeroCaller<float>(const DevMem2Db, PtrStepb);
            template int countNonZeroCaller<double>(const DevMem2Db, PtrStepb);


            template <int nthreads, typename T>
            __global__ void countNonZeroPass2Kernel(uint* count, int size)
            {
                __shared__ uint scount[nthreads];
                uint tid = threadIdx.y * blockDim.x + threadIdx.x;

                scount[tid] = tid < size ? count[tid] : 0;
                __syncthreads();

                sumInSmem<nthreads, uint>(scount, tid);

                if (tid == 0) 
                    count[0] = scount[0];
            }


            template <typename T>
            int countNonZeroMultipassCaller(const DevMem2Db src, PtrStepb buf)
            {
                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                uint* count_buf = (uint*)buf.ptr(0);

                countNonZeroKernel<256, T><<<grid, threads>>>(src, count_buf);
                cudaSafeCall( cudaGetLastError() );
                countNonZeroPass2Kernel<256, T><<<1, 256>>>(count_buf, grid.x * grid.y);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                uint count;
                cudaSafeCall(cudaMemcpy(&count, count_buf, sizeof(int), cudaMemcpyDeviceToHost));
                
                return count;
            }  

            template int countNonZeroMultipassCaller<uchar>(const DevMem2Db, PtrStepb);
            template int countNonZeroMultipassCaller<char>(const DevMem2Db, PtrStepb);
            template int countNonZeroMultipassCaller<ushort>(const DevMem2Db, PtrStepb);
            template int countNonZeroMultipassCaller<short>(const DevMem2Db, PtrStepb);
            template int countNonZeroMultipassCaller<int>(const DevMem2Db, PtrStepb);
            template int countNonZeroMultipassCaller<float>(const DevMem2Db, PtrStepb);

        } // namespace countnonzero


        //////////////////////////////////////////////////////////////////////////
        // Sum

        namespace sum
        {
            template <typename T> struct SumType {};
            template <> struct SumType<uchar> { typedef uint R; };
            template <> struct SumType<char> { typedef int R; };
            template <> struct SumType<ushort> { typedef uint R; };
            template <> struct SumType<short> { typedef int R; };
            template <> struct SumType<int> { typedef int R; };
            template <> struct SumType<float> { typedef float R; };
            template <> struct SumType<double> { typedef double R; };

            template <typename R> 
            struct IdentityOp { static __device__ __forceinline__ R call(R x) { return x; } };

            template <typename R> 
            struct AbsOp { static __device__ __forceinline__ R call(R x) { return ::abs(x); } };

            template <>
            struct AbsOp<uint> { static __device__ __forceinline__ uint call(uint x) { return x; } };

            template <typename R> 
            struct SqrOp { static __device__ __forceinline__ R call(R x) { return x * x; } };

            __constant__ int ctwidth;
            __constant__ int ctheight;
            __device__ uint blocks_finished = 0;

            const int threads_x = 32;
            const int threads_y = 8;

            void estimateThreadCfg(int cols, int rows, dim3& threads, dim3& grid)
            {
                threads = dim3(threads_x, threads_y);
                grid = dim3(divUp(cols, threads.x * threads.y), 
                            divUp(rows, threads.y * threads.x));
                grid.x = std::min(grid.x, threads.x);
                grid.y = std::min(grid.y, threads.y);
            }


            void getBufSizeRequired(int cols, int rows, int cn, int& bufcols, int& bufrows)
            {
                dim3 threads, grid;
                estimateThreadCfg(cols, rows, threads, grid);
                bufcols = grid.x * grid.y * sizeof(double) * cn;
                bufrows = 1;
            }


            void setKernelConsts(int cols, int rows, const dim3& threads, const dim3& grid)
            {        
                int twidth = divUp(divUp(cols, grid.x), threads.x);
                int theight = divUp(divUp(rows, grid.y), threads.y);
                cudaSafeCall(cudaMemcpyToSymbol(ctwidth, &twidth, sizeof(twidth))); 
                cudaSafeCall(cudaMemcpyToSymbol(ctheight, &theight, sizeof(theight))); 
            }

            template <typename T, typename R, typename Op, int nthreads>
            __global__ void sumKernel(const DevMem2Db src, R* result)
            {
                __shared__ R smem[nthreads];

                const int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
                const int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
                const int tid = threadIdx.y * blockDim.x + threadIdx.x;
                const int bid = blockIdx.y * gridDim.x + blockIdx.x;

                R sum = 0;
                for (int y = 0; y < ctheight && y0 + y * blockDim.y < src.rows; ++y)
                {
                    const T* ptr = (const T*)src.ptr(y0 + y * blockDim.y);
                    for (int x = 0; x < ctwidth && x0 + x * blockDim.x < src.cols; ++x)
                        sum += Op::call(ptr[x0 + x * blockDim.x]);
                }

                smem[tid] = sum;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);

            #if __CUDA_ARCH__ >= 110
                __shared__ bool is_last;

                if (tid == 0)
                {
                    result[bid] = smem[0];
                    __threadfence();

                    uint ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
                    is_last = (ticket == gridDim.x * gridDim.y - 1);
                }

                __syncthreads();

                if (is_last)
                {
                    smem[tid] = tid < gridDim.x * gridDim.y ? result[tid] : 0;
                    __syncthreads();

                    sumInSmem<nthreads, R>(smem, tid);

                    if (tid == 0) 
                    {
                        result[0] = smem[0];
                        blocks_finished = 0;
                    }
                }
            #else
                if (tid == 0) result[bid] = smem[0];
            #endif
            }


            template <typename T, typename R, int nthreads>
            __global__ void sumPass2Kernel(R* result, int size)
            {
                __shared__ R smem[nthreads];
                int tid = threadIdx.y * blockDim.x + threadIdx.x;

                smem[tid] = tid < size ? result[tid] : 0;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);

                if (tid == 0) 
                    result[0] = smem[0];
            }


            template <typename T, typename R, typename Op, int nthreads>
            __global__ void sumKernel_C2(const DevMem2Db src, typename TypeVec<R, 2>::vec_type* result)
            {
                typedef typename TypeVec<T, 2>::vec_type SrcType;
                typedef typename TypeVec<R, 2>::vec_type DstType;

                __shared__ R smem[nthreads * 2];

                const int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
                const int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
                const int tid = threadIdx.y * blockDim.x + threadIdx.x;
                const int bid = blockIdx.y * gridDim.x + blockIdx.x;

                SrcType val;
                DstType sum = VecTraits<DstType>::all(0);
                for (int y = 0; y < ctheight && y0 + y * blockDim.y < src.rows; ++y)
                {
                    const SrcType* ptr = (const SrcType*)src.ptr(y0 + y * blockDim.y);
                    for (int x = 0; x < ctwidth && x0 + x * blockDim.x < src.cols; ++x)
                    {
                        val = ptr[x0 + x * blockDim.x];
                        sum = sum + VecTraits<DstType>::make(Op::call(val.x), Op::call(val.y));
                    }
                }

                smem[tid] = sum.x;
                smem[tid + nthreads] = sum.y;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);
                sumInSmem<nthreads, R>(smem + nthreads, tid);

            #if __CUDA_ARCH__ >= 110
                __shared__ bool is_last;

                if (tid == 0)
                {
                    DstType res;
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    result[bid] = res;
                    __threadfence();

                    uint ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
                    is_last = (ticket == gridDim.x * gridDim.y - 1);
                }

                __syncthreads();

                if (is_last)
                {
                    DstType res = tid < gridDim.x * gridDim.y ? result[tid] : VecTraits<DstType>::all(0);
                    smem[tid] = res.x;
                    smem[tid + nthreads] = res.y;
                    __syncthreads();

                    sumInSmem<nthreads, R>(smem, tid);
                    sumInSmem<nthreads, R>(smem + nthreads, tid);

                    if (tid == 0) 
                    {
                        res.x = smem[0];
                        res.y = smem[nthreads];
                        result[0] = res;
                        blocks_finished = 0;
                    }
                }
            #else
                if (tid == 0) 
                {
                    DstType res;
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    result[bid] = res;
                }
            #endif
            }


            template <typename T, typename R, int nthreads>
            __global__ void sumPass2Kernel_C2(typename TypeVec<R, 2>::vec_type* result, int size)
            {
                typedef typename TypeVec<R, 2>::vec_type DstType;

                __shared__ R smem[nthreads * 2];

                const int tid = threadIdx.y * blockDim.x + threadIdx.x;

                DstType res = tid < size ? result[tid] : VecTraits<DstType>::all(0);
                smem[tid] = res.x;
                smem[tid + nthreads] = res.y;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);
                sumInSmem<nthreads, R>(smem + nthreads, tid);

                if (tid == 0) 
                {
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    result[0] = res;
                }
            }


            template <typename T, typename R, typename Op, int nthreads>
            __global__ void sumKernel_C3(const DevMem2Db src, typename TypeVec<R, 3>::vec_type* result)
            {
                typedef typename TypeVec<T, 3>::vec_type SrcType;
                typedef typename TypeVec<R, 3>::vec_type DstType;

                __shared__ R smem[nthreads * 3];

                const int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
                const int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
                const int tid = threadIdx.y * blockDim.x + threadIdx.x;
                const int bid = blockIdx.y * gridDim.x + blockIdx.x;

                SrcType val;
                DstType sum = VecTraits<DstType>::all(0);
                for (int y = 0; y < ctheight && y0 + y * blockDim.y < src.rows; ++y)
                {
                    const SrcType* ptr = (const SrcType*)src.ptr(y0 + y * blockDim.y);
                    for (int x = 0; x < ctwidth && x0 + x * blockDim.x < src.cols; ++x)
                    {
                        val = ptr[x0 + x * blockDim.x];
                        sum = sum + VecTraits<DstType>::make(Op::call(val.x), Op::call(val.y), Op::call(val.z));
                    }
                }

                smem[tid] = sum.x;
                smem[tid + nthreads] = sum.y;
                smem[tid + 2 * nthreads] = sum.z;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);
                sumInSmem<nthreads, R>(smem + nthreads, tid);
                sumInSmem<nthreads, R>(smem + 2 * nthreads, tid);

            #if __CUDA_ARCH__ >= 110
                __shared__ bool is_last;

                if (tid == 0)
                {
                    DstType res;
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    res.z = smem[2 * nthreads];
                    result[bid] = res;
                    __threadfence();

                    uint ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
                    is_last = (ticket == gridDim.x * gridDim.y - 1);
                }

                __syncthreads();

                if (is_last)
                {
                    DstType res = tid < gridDim.x * gridDim.y ? result[tid] : VecTraits<DstType>::all(0);
                    smem[tid] = res.x;
                    smem[tid + nthreads] = res.y;
                    smem[tid + 2 * nthreads] = res.z;
                    __syncthreads();

                    sumInSmem<nthreads, R>(smem, tid);
                    sumInSmem<nthreads, R>(smem + nthreads, tid);
                    sumInSmem<nthreads, R>(smem + 2 * nthreads, tid);

                    if (tid == 0) 
                    {
                        res.x = smem[0];
                        res.y = smem[nthreads];
                        res.z = smem[2 * nthreads];
                        result[0] = res;
                        blocks_finished = 0;
                    }
                }
            #else
                if (tid == 0) 
                {
                    DstType res;
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    res.z = smem[2 * nthreads];
                    result[bid] = res;
                }
            #endif
            }


            template <typename T, typename R, int nthreads>
            __global__ void sumPass2Kernel_C3(typename TypeVec<R, 3>::vec_type* result, int size)
            {
                typedef typename TypeVec<R, 3>::vec_type DstType;

                __shared__ R smem[nthreads * 3];

                const int tid = threadIdx.y * blockDim.x + threadIdx.x;

                DstType res = tid < size ? result[tid] : VecTraits<DstType>::all(0);
                smem[tid] = res.x;
                smem[tid + nthreads] = res.y;
                smem[tid + 2 * nthreads] = res.z;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);
                sumInSmem<nthreads, R>(smem + nthreads, tid);
                sumInSmem<nthreads, R>(smem + 2 * nthreads, tid);

                if (tid == 0) 
                {
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    res.z = smem[2 * nthreads];
                    result[0] = res;
                }
            }

            template <typename T, typename R, typename Op, int nthreads>
            __global__ void sumKernel_C4(const DevMem2Db src, typename TypeVec<R, 4>::vec_type* result)
            {
                typedef typename TypeVec<T, 4>::vec_type SrcType;
                typedef typename TypeVec<R, 4>::vec_type DstType;

                __shared__ R smem[nthreads * 4];

                const int x0 = blockIdx.x * blockDim.x * ctwidth + threadIdx.x;
                const int y0 = blockIdx.y * blockDim.y * ctheight + threadIdx.y;
                const int tid = threadIdx.y * blockDim.x + threadIdx.x;
                const int bid = blockIdx.y * gridDim.x + blockIdx.x;

                SrcType val;
                DstType sum = VecTraits<DstType>::all(0);
                for (int y = 0; y < ctheight && y0 + y * blockDim.y < src.rows; ++y)
                {
                    const SrcType* ptr = (const SrcType*)src.ptr(y0 + y * blockDim.y);
                    for (int x = 0; x < ctwidth && x0 + x * blockDim.x < src.cols; ++x)
                    {
                        val = ptr[x0 + x * blockDim.x];
                        sum = sum + VecTraits<DstType>::make(Op::call(val.x), Op::call(val.y), 
                                                             Op::call(val.z), Op::call(val.w));
                    }
                }

                smem[tid] = sum.x;
                smem[tid + nthreads] = sum.y;
                smem[tid + 2 * nthreads] = sum.z;
                smem[tid + 3 * nthreads] = sum.w;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);
                sumInSmem<nthreads, R>(smem + nthreads, tid);
                sumInSmem<nthreads, R>(smem + 2 * nthreads, tid);
                sumInSmem<nthreads, R>(smem + 3 * nthreads, tid);

            #if __CUDA_ARCH__ >= 110
                __shared__ bool is_last;

                if (tid == 0)
                {
                    DstType res;
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    res.z = smem[2 * nthreads];
                    res.w = smem[3 * nthreads];
                    result[bid] = res;
                    __threadfence();

                    uint ticket = atomicInc(&blocks_finished, gridDim.x * gridDim.y);
                    is_last = (ticket == gridDim.x * gridDim.y - 1);
                }

                __syncthreads();

                if (is_last)
                {
                    DstType res = tid < gridDim.x * gridDim.y ? result[tid] : VecTraits<DstType>::all(0);
                    smem[tid] = res.x;
                    smem[tid + nthreads] = res.y;
                    smem[tid + 2 * nthreads] = res.z;
                    smem[tid + 3 * nthreads] = res.w;
                    __syncthreads();

                    sumInSmem<nthreads, R>(smem, tid);
                    sumInSmem<nthreads, R>(smem + nthreads, tid);
                    sumInSmem<nthreads, R>(smem + 2 * nthreads, tid);
                    sumInSmem<nthreads, R>(smem + 3 * nthreads, tid);

                    if (tid == 0) 
                    {
                        res.x = smem[0];
                        res.y = smem[nthreads];
                        res.z = smem[2 * nthreads];
                        res.w = smem[3 * nthreads];
                        result[0] = res;
                        blocks_finished = 0;
                    }
                }
            #else
                if (tid == 0) 
                {
                    DstType res;
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    res.z = smem[2 * nthreads];
                    res.w = smem[3 * nthreads];
                    result[bid] = res;
                }
            #endif
            }


            template <typename T, typename R, int nthreads>
            __global__ void sumPass2Kernel_C4(typename TypeVec<R, 4>::vec_type* result, int size)
            {
                typedef typename TypeVec<R, 4>::vec_type DstType;

                __shared__ R smem[nthreads * 4];

                const int tid = threadIdx.y * blockDim.x + threadIdx.x;

                DstType res = tid < size ? result[tid] : VecTraits<DstType>::all(0);
                smem[tid] = res.x;
                smem[tid + nthreads] = res.y;
                smem[tid + 2 * nthreads] = res.z;
                smem[tid + 3 * nthreads] = res.w;
                __syncthreads();

                sumInSmem<nthreads, R>(smem, tid);
                sumInSmem<nthreads, R>(smem + nthreads, tid);
                sumInSmem<nthreads, R>(smem + 2 * nthreads, tid);
                sumInSmem<nthreads, R>(smem + 3 * nthreads, tid);

                if (tid == 0) 
                {
                    res.x = smem[0];
                    res.y = smem[nthreads];
                    res.z = smem[2 * nthreads];
                    res.w = smem[3 * nthreads];
                    result[0] = res;
                }
            }

            template <typename T>
            void sumMultipassCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn)
            {
                typedef typename SumType<T>::R R;

                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                switch (cn)
                {
                case 1:
                    sumKernel<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 1>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 1>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 2:
                    sumKernel_C2<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 2>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C2<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 2>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 3:
                    sumKernel_C3<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 3>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C3<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 3>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 4:
                    sumKernel_C4<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 4>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C4<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 4>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                }
                cudaSafeCall( cudaDeviceSynchronize() );

                R result[4] = {0, 0, 0, 0};
                cudaSafeCall(cudaMemcpy(&result, buf.ptr(0), sizeof(R) * cn, cudaMemcpyDeviceToHost));

                sum[0] = result[0];
                sum[1] = result[1];
                sum[2] = result[2];
                sum[3] = result[3];
            }  

            template void sumMultipassCaller<uchar>(const DevMem2Db, PtrStepb, double*, int);
            template void sumMultipassCaller<char>(const DevMem2Db, PtrStepb, double*, int);
            template void sumMultipassCaller<ushort>(const DevMem2Db, PtrStepb, double*, int);
            template void sumMultipassCaller<short>(const DevMem2Db, PtrStepb, double*, int);
            template void sumMultipassCaller<int>(const DevMem2Db, PtrStepb, double*, int);
            template void sumMultipassCaller<float>(const DevMem2Db, PtrStepb, double*, int);


            template <typename T>
            void sumCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn)
            {
                typedef typename SumType<T>::R R;

                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                switch (cn)
                {
                case 1:
                    sumKernel<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 1>::vec_type*)buf.ptr(0));
                    break;
                case 2:
                    sumKernel_C2<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 2>::vec_type*)buf.ptr(0));
                    break;
                case 3:
                    sumKernel_C3<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 3>::vec_type*)buf.ptr(0));
                    break;
                case 4:
                    sumKernel_C4<T, R, IdentityOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 4>::vec_type*)buf.ptr(0));
                    break;
                }
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                R result[4] = {0, 0, 0, 0};
                cudaSafeCall(cudaMemcpy(&result, buf.ptr(0), sizeof(R) * cn, cudaMemcpyDeviceToHost));

                sum[0] = result[0];
                sum[1] = result[1];
                sum[2] = result[2];
                sum[3] = result[3];
            }  

            template void sumCaller<uchar>(const DevMem2Db, PtrStepb, double*, int);
            template void sumCaller<char>(const DevMem2Db, PtrStepb, double*, int);
            template void sumCaller<ushort>(const DevMem2Db, PtrStepb, double*, int);
            template void sumCaller<short>(const DevMem2Db, PtrStepb, double*, int);
            template void sumCaller<int>(const DevMem2Db, PtrStepb, double*, int);
            template void sumCaller<float>(const DevMem2Db, PtrStepb, double*, int);


            template <typename T>
            void absSumMultipassCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn)
            {
                typedef typename SumType<T>::R R;

                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                switch (cn)
                {
                case 1:
                    sumKernel<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 1>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 1>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 2:
                    sumKernel_C2<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 2>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C2<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 2>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 3:
                    sumKernel_C3<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 3>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C3<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 3>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 4:
                    sumKernel_C4<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 4>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C4<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 4>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                }
                cudaSafeCall( cudaDeviceSynchronize() );

                R result[4] = {0, 0, 0, 0};
                cudaSafeCall(cudaMemcpy(result, buf.ptr(0), sizeof(R) * cn, cudaMemcpyDeviceToHost));

                sum[0] = result[0];
                sum[1] = result[1];
                sum[2] = result[2];
                sum[3] = result[3];
            }  

            template void absSumMultipassCaller<uchar>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumMultipassCaller<char>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumMultipassCaller<ushort>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumMultipassCaller<short>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumMultipassCaller<int>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumMultipassCaller<float>(const DevMem2Db, PtrStepb, double*, int);


            template <typename T>
            void absSumCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn)
            {
                typedef typename SumType<T>::R R;

                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                switch (cn)
                {
                case 1:
                    sumKernel<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 1>::vec_type*)buf.ptr(0));
                    break;
                case 2:
                    sumKernel_C2<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 2>::vec_type*)buf.ptr(0));
                    break;
                case 3:
                    sumKernel_C3<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 3>::vec_type*)buf.ptr(0));
                    break;
                case 4:
                    sumKernel_C4<T, R, AbsOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 4>::vec_type*)buf.ptr(0));
                    break;
                }
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                R result[4] = {0, 0, 0, 0};
                cudaSafeCall(cudaMemcpy(result, buf.ptr(0), sizeof(R) * cn, cudaMemcpyDeviceToHost));

                sum[0] = result[0];
                sum[1] = result[1];
                sum[2] = result[2];
                sum[3] = result[3];
            }

            template void absSumCaller<uchar>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumCaller<char>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumCaller<ushort>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumCaller<short>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumCaller<int>(const DevMem2Db, PtrStepb, double*, int);
            template void absSumCaller<float>(const DevMem2Db, PtrStepb, double*, int);


            template <typename T>
            void sqrSumMultipassCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn)
            {
                typedef typename SumType<T>::R R;

                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                switch (cn)
                {
                case 1:
                    sumKernel<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 1>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 1>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 2:
                    sumKernel_C2<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 2>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C2<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 2>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 3:
                    sumKernel_C3<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 3>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C3<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 3>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                case 4:
                    sumKernel_C4<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 4>::vec_type*)buf.ptr(0));
                    cudaSafeCall( cudaGetLastError() );

                    sumPass2Kernel_C4<T, R, threads_x * threads_y><<<1, threads_x * threads_y>>>(
                            (typename TypeVec<R, 4>::vec_type*)buf.ptr(0), grid.x * grid.y);
                    cudaSafeCall( cudaGetLastError() );

                    break;
                }
                cudaSafeCall( cudaDeviceSynchronize() );

                R result[4] = {0, 0, 0, 0};
                cudaSafeCall(cudaMemcpy(result, buf.ptr(0), sizeof(R) * cn, cudaMemcpyDeviceToHost));

                sum[0] = result[0];
                sum[1] = result[1];
                sum[2] = result[2];
                sum[3] = result[3];
            }  

            template void sqrSumMultipassCaller<uchar>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumMultipassCaller<char>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumMultipassCaller<ushort>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumMultipassCaller<short>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumMultipassCaller<int>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumMultipassCaller<float>(const DevMem2Db, PtrStepb, double*, int);


            template <typename T>
            void sqrSumCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn)
            {
                typedef double R;

                dim3 threads, grid;
                estimateThreadCfg(src.cols, src.rows, threads, grid);
                setKernelConsts(src.cols, src.rows, threads, grid);

                switch (cn)
                {
                case 1:
                    sumKernel<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 1>::vec_type*)buf.ptr(0));
                    break;
                case 2:
                    sumKernel_C2<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 2>::vec_type*)buf.ptr(0));
                    break;
                case 3:
                    sumKernel_C3<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 3>::vec_type*)buf.ptr(0));
                    break;
                case 4:
                    sumKernel_C4<T, R, SqrOp<R>, threads_x * threads_y><<<grid, threads>>>(
                            src, (typename TypeVec<R, 4>::vec_type*)buf.ptr(0));
                    break;
                }
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );

                R result[4] = {0, 0, 0, 0};
                cudaSafeCall(cudaMemcpy(result, buf.ptr(0), sizeof(R) * cn, cudaMemcpyDeviceToHost));

                sum[0] = result[0];
                sum[1] = result[1];
                sum[2] = result[2];
                sum[3] = result[3];
            }

            template void sqrSumCaller<uchar>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumCaller<char>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumCaller<ushort>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumCaller<short>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumCaller<int>(const DevMem2Db, PtrStepb, double*, int);
            template void sqrSumCaller<float>(const DevMem2Db, PtrStepb, double*, int);
        } // namespace sum

        //////////////////////////////////////////////////////////////////////////////
        // reduce

        template <typename S> struct SumReductor
        {
            __device__ __forceinline__ S startValue() const
            {
                return 0;
            }

            __device__ __forceinline__ SumReductor(const SumReductor& other){}
            __device__ __forceinline__ SumReductor(){}

            __device__ __forceinline__ S operator ()(volatile S a, volatile S b) const
            {
                return a + b;
            }

            __device__ __forceinline__ S result(S r, double) const
            {
                return r;
            }
        };

        template <typename S> struct AvgReductor
        {
            __device__ __forceinline__ S startValue() const
            {
                return 0;
            }

            __device__ __forceinline__ AvgReductor(const AvgReductor& other){}
            __device__ __forceinline__ AvgReductor(){}

            __device__ __forceinline__ S operator ()(volatile S a, volatile S b) const
            {
                return a + b;
            }

            __device__ __forceinline__ double result(S r, double sz) const
            {
                return r / sz;
            }
        };

        template <typename S> struct MinReductor
        {
            __device__ __forceinline__ S startValue() const
            {
                return numeric_limits<S>::max();
            }

            __device__ __forceinline__ MinReductor(const MinReductor& other){}
            __device__ __forceinline__ MinReductor(){}

            template <typename T> __device__ __forceinline__ T operator ()(volatile T a, volatile T b) const
            {
                return saturate_cast<T>(::min(a, b));
            }
            __device__ __forceinline__ float operator ()(volatile float a, volatile float b) const
            {
                return ::fmin(a, b);
            }

            __device__ __forceinline__ S result(S r, double) const
            {
                return r;
            }
        };

        template <typename S> struct MaxReductor
        {
            __device__ __forceinline__ S startValue() const
            {
                return numeric_limits<S>::min();
            }

            __device__ __forceinline__ MaxReductor(const MaxReductor& other){}
            __device__ __forceinline__ MaxReductor(){}

            template <typename T> __device__ __forceinline__ int operator ()(volatile T a, volatile T b) const
            {
                return ::max(a, b);
            }
            __device__ __forceinline__ float operator ()(volatile float a, volatile float b) const
            {
                return ::fmax(a, b);
            }

            __device__ __forceinline__ S result(S r, double) const
            {
                return r;
            }
        };

        template <class Op, typename T, typename S, typename D> __global__ void reduceRows(const DevMem2D_<T> src, D* dst, const Op op)
        {
            __shared__ S smem[16 * 16];

            const int x = blockIdx.x * 16 + threadIdx.x;

            S myVal = op.startValue();

            if (x < src.cols)
            {
                for (int y = threadIdx.y; y < src.rows; y += 16)
                    myVal = op(myVal, src.ptr(y)[x]);
            }        

            smem[threadIdx.x * 16 + threadIdx.y] = myVal;
            __syncthreads();

            if (threadIdx.x < 8)
            {
                volatile S* srow = smem + threadIdx.y * 16;
                srow[threadIdx.x] = op(srow[threadIdx.x], srow[threadIdx.x + 8]);
                srow[threadIdx.x] = op(srow[threadIdx.x], srow[threadIdx.x + 4]);
                srow[threadIdx.x] = op(srow[threadIdx.x], srow[threadIdx.x + 2]);
                srow[threadIdx.x] = op(srow[threadIdx.x], srow[threadIdx.x + 1]);
            }
            __syncthreads();

            if (threadIdx.y == 0 && x < src.cols)
                dst[x] = saturate_cast<D>(op.result(smem[threadIdx.x * 16], src.rows));
        }

        template <template <typename> class Op, typename T, typename S, typename D> void reduceRows_caller(const DevMem2D_<T>& src, DevMem2D_<D> dst, cudaStream_t stream)
        {
            const dim3 block(16, 16);
            const dim3 grid(divUp(src.cols, block.x));

            Op<S> op;
            reduceRows<Op<S>, T, S, D><<<grid, block, 0, stream>>>(src, dst.data, op);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );

        }

        template <typename T, typename S, typename D> void reduceRows_gpu(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream)
        {
            typedef void (*caller_t)(const DevMem2D_<T>& src, DevMem2D_<D> dst, cudaStream_t stream);

            static const caller_t callers[] = 
            {
                reduceRows_caller<SumReductor, T, S, D>, 
                reduceRows_caller<AvgReductor, T, S, D>, 
                reduceRows_caller<MaxReductor, T, S, D>, 
                reduceRows_caller<MinReductor, T, S, D>
            };

            callers[reduceOp](static_cast< DevMem2D_<T> >(src), static_cast< DevMem2D_<D> >(dst), stream);
        }

        template void reduceRows_gpu<uchar, int, uchar>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceRows_gpu<uchar, int, int>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceRows_gpu<uchar, int, float>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);  

        template void reduceRows_gpu<ushort, int, ushort>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceRows_gpu<ushort, int, int>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceRows_gpu<ushort, int, float>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream); 

        template void reduceRows_gpu<short, int, short>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceRows_gpu<short, int, int>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceRows_gpu<short, int, float>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream); 

        template void reduceRows_gpu<int, int, int>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceRows_gpu<int, int, float>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);

        template void reduceRows_gpu<float, float, float>(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);



        template <int cn, class Op, typename T, typename S, typename D> __global__ void reduceCols(const DevMem2D_<T> src, D* dst, const Op op)
        {
            __shared__ S smem[256 * cn];

            const int y = blockIdx.x;

            const T* src_row = src.ptr(y);

            S myVal[cn];

            #pragma unroll
            for (int c = 0; c < cn; ++c)
                myVal[c] = op.startValue();

        #if __CUDA_ARCH__ >= 200

            // For cc >= 2.0 prefer L1 cache
            for (int x = threadIdx.x; x < src.cols; x += 256)
            {
                #pragma unroll
                for (int c = 0; c < cn; ++c)
                    myVal[c] = op(myVal[c], src_row[x * cn + c]);
            }

        #else // __CUDA_ARCH__ >= 200

            // For older arch use shared memory for cache
            for (int x = 0; x < src.cols; x += 256)
            {
                #pragma unroll
                for (int c = 0; c < cn; ++c)
                {
                    smem[c * 256 + threadIdx.x] = op.startValue();
                    const int load_x = x * cn + c * 256 + threadIdx.x;
                    if (load_x < src.cols * cn)
                        smem[c * 256 + threadIdx.x] = src_row[load_x];
                }
                __syncthreads();

                #pragma unroll
                for (int c = 0; c < cn; ++c)
                    myVal[c] = op(myVal[c], smem[threadIdx.x * cn + c]);
                __syncthreads();
            }

        #endif // __CUDA_ARCH__ >= 200

            #pragma unroll
            for (int c = 0; c < cn; ++c)
                smem[c * 256 + threadIdx.x] = myVal[c];
            __syncthreads();

            if (threadIdx.x < 128)
            {
                #pragma unroll
                for (int c = 0; c < cn; ++c)
                    smem[c * 256 + threadIdx.x] = op(smem[c * 256 + threadIdx.x], smem[c * 256 + threadIdx.x + 128]);
            }
            __syncthreads();

            if (threadIdx.x < 64)
            {
                #pragma unroll
                for (int c = 0; c < cn; ++c)
                    smem[c * 256 + threadIdx.x] = op(smem[c * 256 + threadIdx.x], smem[c * 256 + threadIdx.x + 64]);
            }
            __syncthreads();

            volatile S* sdata = smem;

            if (threadIdx.x < 32)
            {
                #pragma unroll
                for (int c = 0; c < cn; ++c)
                {
                    sdata[c * 256 + threadIdx.x] = op(sdata[c * 256 + threadIdx.x], sdata[c * 256 + threadIdx.x + 32]);
                    sdata[c * 256 + threadIdx.x] = op(sdata[c * 256 + threadIdx.x], sdata[c * 256 + threadIdx.x + 16]);
                    sdata[c * 256 + threadIdx.x] = op(sdata[c * 256 + threadIdx.x], sdata[c * 256 + threadIdx.x + 8]);
                    sdata[c * 256 + threadIdx.x] = op(sdata[c * 256 + threadIdx.x], sdata[c * 256 + threadIdx.x + 4]);
                    sdata[c * 256 + threadIdx.x] = op(sdata[c * 256 + threadIdx.x], sdata[c * 256 + threadIdx.x + 2]);
                    sdata[c * 256 + threadIdx.x] = op(sdata[c * 256 + threadIdx.x], sdata[c * 256 + threadIdx.x + 1]);
                }
            }
            __syncthreads();

            if (threadIdx.x < cn)
                dst[y * cn + threadIdx.x] = saturate_cast<D>(op.result(smem[threadIdx.x * 256], src.cols));
        }

        template <int cn, template <typename> class Op, typename T, typename S, typename D> void reduceCols_caller(const DevMem2D_<T>& src, DevMem2D_<D> dst, cudaStream_t stream)
        {
            const dim3 block(256);
            const dim3 grid(src.rows);

            Op<S> op;
            reduceCols<cn, Op<S>, T, S, D><<<grid, block, 0, stream>>>(src, dst.data, op);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );

        }

        template <typename T, typename S, typename D> void reduceCols_gpu(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream)
        {
            typedef void (*caller_t)(const DevMem2D_<T>& src, DevMem2D_<D> dst, cudaStream_t stream);

            static const caller_t callers[4][4] = 
            {
                {reduceCols_caller<1, SumReductor, T, S, D>, reduceCols_caller<1, AvgReductor, T, S, D>, reduceCols_caller<1, MaxReductor, T, S, D>, reduceCols_caller<1, MinReductor, T, S, D>},
                {reduceCols_caller<2, SumReductor, T, S, D>, reduceCols_caller<2, AvgReductor, T, S, D>, reduceCols_caller<2, MaxReductor, T, S, D>, reduceCols_caller<2, MinReductor, T, S, D>},
                {reduceCols_caller<3, SumReductor, T, S, D>, reduceCols_caller<3, AvgReductor, T, S, D>, reduceCols_caller<3, MaxReductor, T, S, D>, reduceCols_caller<3, MinReductor, T, S, D>},
                {reduceCols_caller<4, SumReductor, T, S, D>, reduceCols_caller<4, AvgReductor, T, S, D>, reduceCols_caller<4, MaxReductor, T, S, D>, reduceCols_caller<4, MinReductor, T, S, D>},
            };

            callers[cn - 1][reduceOp](static_cast< DevMem2D_<T> >(src), static_cast< DevMem2D_<D> >(dst), stream);
        }

        template void reduceCols_gpu<uchar, int, uchar>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceCols_gpu<uchar, int, int>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template void reduceCols_gpu<uchar, int, float>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);

        template void reduceCols_gpu<ushort, int, ushort>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream); 
        template void reduceCols_gpu<ushort, int, int>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);                  
        template void reduceCols_gpu<ushort, int, float>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);

        template void reduceCols_gpu<short, int, short>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);  
        template void reduceCols_gpu<short, int, int>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);                  
        template void reduceCols_gpu<short, int, float>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);  

        template void reduceCols_gpu<int, int, int>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);                  
        template void reduceCols_gpu<int, int, float>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);

        template void reduceCols_gpu<float, float, float>(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
    } // namespace mattrix_reductions
}}} // namespace cv { namespace gpu { namespace device
