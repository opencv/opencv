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

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/functional.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_GPUARITHM

namespace cv { namespace gpu { namespace cudev
{
    namespace ght
    {
        __device__ int g_counter;

        template <typename T, int PIXELS_PER_THREAD>
        __global__ void buildEdgePointList(const PtrStepSzb edges, const PtrStep<T> dx, const PtrStep<T> dy,
                                           unsigned int* coordList, float* thetaList)
        {
            __shared__ unsigned int s_coordLists[4][32 * PIXELS_PER_THREAD];
            __shared__ float s_thetaLists[4][32 * PIXELS_PER_THREAD];
            __shared__ int s_sizes[4];
            __shared__ int s_globStart[4];

            const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (threadIdx.x == 0)
                s_sizes[threadIdx.y] = 0;
            __syncthreads();

            if (y < edges.rows)
            {
                // fill the queue
                const uchar* edgesRow = edges.ptr(y);
                const T* dxRow = dx.ptr(y);
                const T* dyRow = dy.ptr(y);

                for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < edges.cols; ++i, xx += blockDim.x)
                {
                    const T dxVal = dxRow[xx];
                    const T dyVal = dyRow[xx];

                    if (edgesRow[xx] && (dxVal != 0 || dyVal != 0))
                    {
                        const unsigned int coord = (y << 16) | xx;

                        float theta = ::atan2f(dyVal, dxVal);
                        if (theta < 0)
                            theta += 2.0f * CV_PI_F;

                        const int qidx = Emulation::smem::atomicAdd(&s_sizes[threadIdx.y], 1);

                        s_coordLists[threadIdx.y][qidx] = coord;
                        s_thetaLists[threadIdx.y][qidx] = theta;
                    }
                }
            }

            __syncthreads();

            // let one thread reserve the space required in the global list
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                // find how many items are stored in each list
                int totalSize = 0;
                for (int i = 0; i < blockDim.y; ++i)
                {
                    s_globStart[i] = totalSize;
                    totalSize += s_sizes[i];
                }

                // calculate the offset in the global list
                const int globalOffset = atomicAdd(&g_counter, totalSize);
                for (int i = 0; i < blockDim.y; ++i)
                    s_globStart[i] += globalOffset;
            }

            __syncthreads();

            // copy local queues to global queue
            const int qsize = s_sizes[threadIdx.y];
            int gidx = s_globStart[threadIdx.y] + threadIdx.x;
            for(int i = threadIdx.x; i < qsize; i += blockDim.x, gidx += blockDim.x)
            {
                coordList[gidx] = s_coordLists[threadIdx.y][i];
                thetaList[gidx] = s_thetaLists[threadIdx.y][i];
            }
        }

        template <typename T>
        int buildEdgePointList_gpu(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList)
        {
            const int PIXELS_PER_THREAD = 8;

            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 4);
            const dim3 grid(divUp(edges.cols, block.x * PIXELS_PER_THREAD), divUp(edges.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(buildEdgePointList<T, PIXELS_PER_THREAD>, cudaFuncCachePreferShared) );

            buildEdgePointList<T, PIXELS_PER_THREAD><<<grid, block>>>(edges, (PtrStepSz<T>) dx, (PtrStepSz<T>) dy, coordList, thetaList);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            return totalCount;
        }

        template int buildEdgePointList_gpu<short>(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);
        template int buildEdgePointList_gpu<int>(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);
        template int buildEdgePointList_gpu<float>(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);

        __global__ void buildRTable(const unsigned int* coordList, const float* thetaList, const int pointsCount,
                                    PtrStep<short2> r_table, int* r_sizes, int maxSize,
                                    const short2 templCenter, const float thetaScale)
        {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid >= pointsCount)
                return;

            const unsigned int coord = coordList[tid];
            short2 p;
            p.x = (coord & 0xFFFF);
            p.y = (coord >> 16) & 0xFFFF;

            const float theta = thetaList[tid];
            const int n = __float2int_rn(theta * thetaScale);

            const int ind = ::atomicAdd(r_sizes + n, 1);
            if (ind < maxSize)
                r_table(n, ind) = saturate_cast<short2>(p - templCenter);
        }

        void buildRTable_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                             PtrStepSz<short2> r_table, int* r_sizes,
                             short2 templCenter, int levels)
        {
            const dim3 block(256);
            const dim3 grid(divUp(pointsCount, block.x));

            const float thetaScale = levels / (2.0f * CV_PI_F);

            buildRTable<<<grid, block>>>(coordList, thetaList, pointsCount, r_table, r_sizes, r_table.cols, templCenter, thetaScale);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // Ballard_Pos

        __global__ void Ballard_Pos_calcHist(const unsigned int* coordList, const float* thetaList, const int pointsCount,
                                             const PtrStep<short2> r_table, const int* r_sizes,
                                             PtrStepSzi hist,
                                             const float idp, const float thetaScale)
        {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid >= pointsCount)
                return;

            const unsigned int coord = coordList[tid];
            short2 p;
            p.x = (coord & 0xFFFF);
            p.y = (coord >> 16) & 0xFFFF;

            const float theta = thetaList[tid];
            const int n = __float2int_rn(theta * thetaScale);

            const short2* r_row = r_table.ptr(n);
            const int r_row_size = r_sizes[n];

            for (int j = 0; j < r_row_size; ++j)
            {
                short2 c = saturate_cast<short2>(p - r_row[j]);

                c.x = __float2int_rn(c.x * idp);
                c.y = __float2int_rn(c.y * idp);

                if (c.x >= 0 && c.x < hist.cols - 2 && c.y >= 0 && c.y < hist.rows - 2)
                    ::atomicAdd(hist.ptr(c.y + 1) + c.x + 1, 1);
            }
        }

        void Ballard_Pos_calcHist_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                      PtrStepSz<short2> r_table, const int* r_sizes,
                                      PtrStepSzi hist,
                                      float dp, int levels)
        {
            const dim3 block(256);
            const dim3 grid(divUp(pointsCount, block.x));

            const float idp = 1.0f / dp;
            const float thetaScale = levels / (2.0f * CV_PI_F);

            Ballard_Pos_calcHist<<<grid, block>>>(coordList, thetaList, pointsCount, r_table, r_sizes, hist, idp, thetaScale);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void Ballard_Pos_findPosInHist(const PtrStepSzi hist, float4* out, int3* votes,
                                                  const int maxSize, const float dp, const int threshold)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= hist.cols - 2 || y >= hist.rows - 2)
                return;

            const int curVotes = hist(y + 1, x + 1);

            if (curVotes > threshold &&
                curVotes >  hist(y + 1, x) &&
                curVotes >= hist(y + 1, x + 2) &&
                curVotes >  hist(y, x + 1) &&
                curVotes >= hist(y + 2, x + 1))
            {
                const int ind = ::atomicAdd(&g_counter, 1);

                if (ind < maxSize)
                {
                    out[ind] = make_float4(x * dp, y * dp, 1.0f, 0.0f);
                    votes[ind] = make_int3(curVotes, 0, 0);
                }
            }
        }

        int Ballard_Pos_findPosInHist_gpu(PtrStepSzi hist, float4* out, int3* votes, int maxSize, float dp, int threshold)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(hist.cols - 2, block.x), divUp(hist.rows - 2, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(Ballard_Pos_findPosInHist, cudaFuncCachePreferL1) );

            Ballard_Pos_findPosInHist<<<grid, block>>>(hist, out, votes, maxSize, dp, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            return totalCount;
        }

        ////////////////////////////////////////////////////////////////////////
        // Guil_Full

        struct FeatureTable
        {
            uchar* p1_pos_data;
            size_t p1_pos_step;

            uchar* p1_theta_data;
            size_t p1_theta_step;

            uchar* p2_pos_data;
            size_t p2_pos_step;

            uchar* d12_data;
            size_t d12_step;

            uchar* r1_data;
            size_t r1_step;

            uchar* r2_data;
            size_t r2_step;
        };

        __constant__ FeatureTable c_templFeatures;
        __constant__ FeatureTable c_imageFeatures;

        void Guil_Full_setTemplFeatures(PtrStepb p1_pos, PtrStepb p1_theta, PtrStepb p2_pos, PtrStepb d12, PtrStepb r1, PtrStepb r2)
        {
            FeatureTable tbl;

            tbl.p1_pos_data = p1_pos.data;
            tbl.p1_pos_step = p1_pos.step;

            tbl.p1_theta_data = p1_theta.data;
            tbl.p1_theta_step = p1_theta.step;

            tbl.p2_pos_data = p2_pos.data;
            tbl.p2_pos_step = p2_pos.step;

            tbl.d12_data = d12.data;
            tbl.d12_step = d12.step;

            tbl.r1_data = r1.data;
            tbl.r1_step = r1.step;

            tbl.r2_data = r2.data;
            tbl.r2_step = r2.step;

            cudaSafeCall( cudaMemcpyToSymbol(c_templFeatures, &tbl, sizeof(FeatureTable)) );
        }
        void Guil_Full_setImageFeatures(PtrStepb p1_pos, PtrStepb p1_theta, PtrStepb p2_pos, PtrStepb d12, PtrStepb r1, PtrStepb r2)
        {
            FeatureTable tbl;

            tbl.p1_pos_data = p1_pos.data;
            tbl.p1_pos_step = p1_pos.step;

            tbl.p1_theta_data = p1_theta.data;
            tbl.p1_theta_step = p1_theta.step;

            tbl.p2_pos_data = p2_pos.data;
            tbl.p2_pos_step = p2_pos.step;

            tbl.d12_data = d12.data;
            tbl.d12_step = d12.step;

            tbl.r1_data = r1.data;
            tbl.r1_step = r1.step;

            tbl.r2_data = r2.data;
            tbl.r2_step = r2.step;

            cudaSafeCall( cudaMemcpyToSymbol(c_imageFeatures, &tbl, sizeof(FeatureTable)) );
        }

        struct TemplFeatureTable
        {
            static __device__ float2* p1_pos(int n)
            {
                return (float2*)(c_templFeatures.p1_pos_data + n * c_templFeatures.p1_pos_step);
            }
            static __device__ float* p1_theta(int n)
            {
                return (float*)(c_templFeatures.p1_theta_data + n * c_templFeatures.p1_theta_step);
            }
            static __device__ float2* p2_pos(int n)
            {
                return (float2*)(c_templFeatures.p2_pos_data + n * c_templFeatures.p2_pos_step);
            }

            static __device__ float* d12(int n)
            {
                return (float*)(c_templFeatures.d12_data + n * c_templFeatures.d12_step);
            }

            static __device__ float2* r1(int n)
            {
                return (float2*)(c_templFeatures.r1_data + n * c_templFeatures.r1_step);
            }
            static __device__ float2* r2(int n)
            {
                return (float2*)(c_templFeatures.r2_data + n * c_templFeatures.r2_step);
            }
        };
        struct ImageFeatureTable
        {
            static __device__ float2* p1_pos(int n)
            {
                return (float2*)(c_imageFeatures.p1_pos_data + n * c_imageFeatures.p1_pos_step);
            }
            static __device__ float* p1_theta(int n)
            {
                return (float*)(c_imageFeatures.p1_theta_data + n * c_imageFeatures.p1_theta_step);
            }
            static __device__ float2* p2_pos(int n)
            {
                return (float2*)(c_imageFeatures.p2_pos_data + n * c_imageFeatures.p2_pos_step);
            }

            static __device__ float* d12(int n)
            {
                return (float*)(c_imageFeatures.d12_data + n * c_imageFeatures.d12_step);
            }

            static __device__ float2* r1(int n)
            {
                return (float2*)(c_imageFeatures.r1_data + n * c_imageFeatures.r1_step);
            }
            static __device__ float2* r2(int n)
            {
                return (float2*)(c_imageFeatures.r2_data + n * c_imageFeatures.r2_step);
            }
        };

        __device__ float clampAngle(float a)
        {
            float res = a;

            while (res > 2.0f * CV_PI_F)
                res -= 2.0f * CV_PI_F;
            while (res < 0.0f)
                res += 2.0f * CV_PI_F;

            return res;
        }

        __device__ bool angleEq(float a, float b, float eps)
        {
            return (::fabs(clampAngle(a - b)) <= eps);
        }

        template <class FT, bool isTempl>
        __global__ void Guil_Full_buildFeatureList(const unsigned int* coordList, const float* thetaList, const int pointsCount,
                                                   int* sizes, const int maxSize,
                                                   const float xi, const float angleEpsilon, const float alphaScale,
                                                   const float2 center, const float maxDist)
        {
            const float p1_theta = thetaList[blockIdx.x];
            const unsigned int coord1 = coordList[blockIdx.x];
            float2 p1_pos;
            p1_pos.x = (coord1 & 0xFFFF);
            p1_pos.y = (coord1 >> 16) & 0xFFFF;

            for (int i = threadIdx.x; i < pointsCount; i += blockDim.x)
            {
                const float p2_theta = thetaList[i];
                const unsigned int coord2 = coordList[i];
                float2 p2_pos;
                p2_pos.x = (coord2 & 0xFFFF);
                p2_pos.y = (coord2 >> 16) & 0xFFFF;

                if (angleEq(p1_theta - p2_theta, xi, angleEpsilon))
                {
                    const float2 d = p1_pos - p2_pos;

                    float alpha12 = clampAngle(::atan2(d.y, d.x) - p1_theta);
                    float d12 = ::sqrtf(d.x * d.x + d.y * d.y);

                    if (d12 > maxDist)
                        continue;

                    float2 r1 = p1_pos - center;
                    float2 r2 = p2_pos - center;

                    const int n = __float2int_rn(alpha12 * alphaScale);

                    const int ind = ::atomicAdd(sizes + n, 1);

                    if (ind < maxSize)
                    {
                        if (!isTempl)
                        {
                            FT::p1_pos(n)[ind] = p1_pos;
                            FT::p2_pos(n)[ind] = p2_pos;
                        }

                        FT::p1_theta(n)[ind] = p1_theta;

                        FT::d12(n)[ind] = d12;

                        if (isTempl)
                        {
                            FT::r1(n)[ind] = r1;
                            FT::r2(n)[ind] = r2;
                        }
                    }
                }
            }
        }

        template <class FT, bool isTempl>
        void Guil_Full_buildFeatureList_caller(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                               int* sizes, int maxSize,
                                               float xi, float angleEpsilon, int levels,
                                               float2 center, float maxDist)
        {
            const dim3 block(256);
            const dim3 grid(pointsCount);

            const float alphaScale = levels / (2.0f * CV_PI_F);

            Guil_Full_buildFeatureList<FT, isTempl><<<grid, block>>>(coordList, thetaList, pointsCount,
                                                                     sizes, maxSize,
                                                                     xi * (CV_PI_F / 180.0f), angleEpsilon * (CV_PI_F / 180.0f), alphaScale,
                                                                     center, maxDist);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            thrust::device_ptr<int> sizesPtr(sizes);
            thrust::transform(sizesPtr, sizesPtr + levels + 1, sizesPtr, cudev::bind2nd(cudev::minimum<int>(), maxSize));
        }

        void Guil_Full_buildTemplFeatureList_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                 int* sizes, int maxSize,
                                                 float xi, float angleEpsilon, int levels,
                                                 float2 center, float maxDist)
        {
            Guil_Full_buildFeatureList_caller<TemplFeatureTable, true>(coordList, thetaList, pointsCount,
                                                                       sizes, maxSize,
                                                                       xi, angleEpsilon, levels,
                                                                       center, maxDist);
        }
        void Guil_Full_buildImageFeatureList_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                 int* sizes, int maxSize,
                                                 float xi, float angleEpsilon, int levels,
                                                 float2 center, float maxDist)
        {
            Guil_Full_buildFeatureList_caller<ImageFeatureTable, false>(coordList, thetaList, pointsCount,
                                                                        sizes, maxSize,
                                                                        xi, angleEpsilon, levels,
                                                                        center, maxDist);
        }

        __global__ void Guil_Full_calcOHist(const int* templSizes, const int* imageSizes, int* OHist,
                                            const float minAngle, const float maxAngle, const float iAngleStep, const int angleRange)
        {
            extern __shared__ int s_OHist[];
            for (int i = threadIdx.x; i <= angleRange; i += blockDim.x)
                s_OHist[i] = 0;
            __syncthreads();

            const int tIdx = blockIdx.x;
            const int level = blockIdx.y;

            const int tSize = templSizes[level];

            if (tIdx < tSize)
            {
                const int imSize = imageSizes[level];

                const float t_p1_theta = TemplFeatureTable::p1_theta(level)[tIdx];

                for (int i = threadIdx.x; i < imSize; i += blockDim.x)
                {
                    const float im_p1_theta = ImageFeatureTable::p1_theta(level)[i];

                    const float angle = clampAngle(im_p1_theta - t_p1_theta);

                    if (angle >= minAngle && angle <= maxAngle)
                    {
                        const int n = __float2int_rn((angle - minAngle) * iAngleStep);
                        Emulation::smem::atomicAdd(&s_OHist[n], 1);
                    }
                }
            }
            __syncthreads();

            for (int i = threadIdx.x; i <= angleRange; i += blockDim.x)
                ::atomicAdd(OHist + i, s_OHist[i]);
        }

        void Guil_Full_calcOHist_gpu(const int* templSizes, const int* imageSizes, int* OHist,
                                     float minAngle, float maxAngle, float angleStep, int angleRange,
                                     int levels, int tMaxSize)
        {
            const dim3 block(256);
            const dim3 grid(tMaxSize, levels + 1);

            minAngle *= (CV_PI_F / 180.0f);
            maxAngle *= (CV_PI_F / 180.0f);
            angleStep *= (CV_PI_F / 180.0f);

            const size_t smemSize = (angleRange + 1) * sizeof(float);

            Guil_Full_calcOHist<<<grid, block, smemSize>>>(templSizes, imageSizes, OHist,
                                                           minAngle, maxAngle, 1.0f / angleStep, angleRange);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void Guil_Full_calcSHist(const int* templSizes, const int* imageSizes, int* SHist,
                                            const float angle, const float angleEpsilon,
                                            const float minScale, const float maxScale, const float iScaleStep, const int scaleRange)
        {
            extern __shared__ int s_SHist[];
            for (int i = threadIdx.x; i <= scaleRange; i += blockDim.x)
                s_SHist[i] = 0;
            __syncthreads();

            const int tIdx = blockIdx.x;
            const int level = blockIdx.y;

            const int tSize = templSizes[level];

            if (tIdx < tSize)
            {
                const int imSize = imageSizes[level];

                const float t_p1_theta = TemplFeatureTable::p1_theta(level)[tIdx] + angle;
                const float t_d12 = TemplFeatureTable::d12(level)[tIdx] + angle;

                for (int i = threadIdx.x; i < imSize; i += blockDim.x)
                {
                    const float im_p1_theta = ImageFeatureTable::p1_theta(level)[i];
                    const float im_d12 = ImageFeatureTable::d12(level)[i];

                    if (angleEq(im_p1_theta, t_p1_theta, angleEpsilon))
                    {
                        const float scale = im_d12 / t_d12;

                        if (scale >= minScale && scale <= maxScale)
                        {
                            const int s = __float2int_rn((scale - minScale) * iScaleStep);
                            Emulation::smem::atomicAdd(&s_SHist[s], 1);
                        }
                    }
                }
            }
            __syncthreads();

            for (int i = threadIdx.x; i <= scaleRange; i += blockDim.x)
                ::atomicAdd(SHist + i, s_SHist[i]);
        }

        void Guil_Full_calcSHist_gpu(const int* templSizes, const int* imageSizes, int* SHist,
                                     float angle, float angleEpsilon,
                                     float minScale, float maxScale, float iScaleStep, int scaleRange,
                                     int levels, int tMaxSize)
        {
            const dim3 block(256);
            const dim3 grid(tMaxSize, levels + 1);

            angle *= (CV_PI_F / 180.0f);
            angleEpsilon *= (CV_PI_F / 180.0f);

            const size_t smemSize = (scaleRange + 1) * sizeof(float);

            Guil_Full_calcSHist<<<grid, block, smemSize>>>(templSizes, imageSizes, SHist,
                                                           angle, angleEpsilon,
                                                           minScale, maxScale, iScaleStep, scaleRange);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void Guil_Full_calcPHist(const int* templSizes, const int* imageSizes, PtrStepSzi PHist,
                                            const float angle, const float sinVal, const float cosVal, const float angleEpsilon, const float scale,
                                            const float idp)
        {
            const int tIdx = blockIdx.x;
            const int level = blockIdx.y;

            const int tSize = templSizes[level];

            if (tIdx < tSize)
            {
                const int imSize = imageSizes[level];

                const float t_p1_theta = TemplFeatureTable::p1_theta(level)[tIdx] + angle;

                float2 r1 = TemplFeatureTable::r1(level)[tIdx];
                float2 r2 = TemplFeatureTable::r2(level)[tIdx];

                r1 = r1 * scale;
                r2 = r2 * scale;

                r1 = make_float2(cosVal * r1.x - sinVal * r1.y, sinVal * r1.x + cosVal * r1.y);
                r2 = make_float2(cosVal * r2.x - sinVal * r2.y, sinVal * r2.x + cosVal * r2.y);

                for (int i = threadIdx.x; i < imSize; i += blockDim.x)
                {
                    const float im_p1_theta = ImageFeatureTable::p1_theta(level)[i];

                    const float2 im_p1_pos = ImageFeatureTable::p1_pos(level)[i];
                    const float2 im_p2_pos = ImageFeatureTable::p2_pos(level)[i];

                    if (angleEq(im_p1_theta, t_p1_theta, angleEpsilon))
                    {
                        float2 c1, c2;

                        c1 = im_p1_pos - r1;
                        c1 = c1 * idp;

                        c2 = im_p2_pos - r2;
                        c2 = c2 * idp;

                        if (::fabs(c1.x - c2.x) > 1 || ::fabs(c1.y - c2.y) > 1)
                            continue;

                        if (c1.y >= 0 && c1.y < PHist.rows - 2 && c1.x >= 0 && c1.x < PHist.cols - 2)
                            ::atomicAdd(PHist.ptr(__float2int_rn(c1.y) + 1) + __float2int_rn(c1.x) + 1, 1);
                    }
                }
            }
        }

        void Guil_Full_calcPHist_gpu(const int* templSizes, const int* imageSizes, PtrStepSzi PHist,
                                     float angle, float angleEpsilon, float scale,
                                     float dp,
                                     int levels, int tMaxSize)
        {
            const dim3 block(256);
            const dim3 grid(tMaxSize, levels + 1);

            angle *= (CV_PI_F / 180.0f);
            angleEpsilon *= (CV_PI_F / 180.0f);

            const float sinVal = ::sinf(angle);
            const float cosVal = ::cosf(angle);

            cudaSafeCall( cudaFuncSetCacheConfig(Guil_Full_calcPHist, cudaFuncCachePreferL1) );

            Guil_Full_calcPHist<<<grid, block>>>(templSizes, imageSizes, PHist,
                                                 angle, sinVal, cosVal, angleEpsilon, scale,
                                                 1.0f / dp);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void Guil_Full_findPosInHist(const PtrStepSzi hist, float4* out, int3* votes, const int maxSize,
                                                const float angle, const int angleVotes, const float scale, const int scaleVotes,
                                                const float dp, const int threshold)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= hist.cols - 2 || y >= hist.rows - 2)
                return;

            const int curVotes = hist(y + 1, x + 1);

            if (curVotes > threshold &&
                curVotes >  hist(y + 1, x) &&
                curVotes >= hist(y + 1, x + 2) &&
                curVotes >  hist(y, x + 1) &&
                curVotes >= hist(y + 2, x + 1))
            {
                const int ind = ::atomicAdd(&g_counter, 1);

                if (ind < maxSize)
                {
                    out[ind] = make_float4(x * dp, y * dp, scale, angle);
                    votes[ind] = make_int3(curVotes, scaleVotes, angleVotes);
                }
            }
        }

        int Guil_Full_findPosInHist_gpu(PtrStepSzi hist, float4* out, int3* votes, int curSize, int maxSize,
                                        float angle, int angleVotes, float scale, int scaleVotes,
                                        float dp, int threshold)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemcpy(counterPtr, &curSize, sizeof(int), cudaMemcpyHostToDevice) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(hist.cols - 2, block.x), divUp(hist.rows - 2, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(Guil_Full_findPosInHist, cudaFuncCachePreferL1) );

            Guil_Full_findPosInHist<<<grid, block>>>(hist, out, votes, maxSize,
                                                     angle, angleVotes, scale, scaleVotes,
                                                     dp, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            return totalCount;
        }
    }
}}}

#endif // HAVE_OPENCV_GPUARITHM

#endif /* CUDA_DISABLER */
