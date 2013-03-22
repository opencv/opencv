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

#include "lbp.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace lbp
    {
        struct LBP
        {
            __host__ __device__ __forceinline__ LBP() {}

            __device__ __forceinline__ int operator() (const int* integral, int ty, int fh, int fw, int& shift) const
            {
                int anchors[9];

                anchors[0]  = integral[ty];
                anchors[1]  = integral[ty + fw];
                anchors[0] -= anchors[1];
                anchors[2]  = integral[ty + fw * 2];
                anchors[1] -= anchors[2];
                anchors[2] -= integral[ty + fw * 3];

                ty += fh;
                anchors[3]  = integral[ty];
                anchors[4]  = integral[ty + fw];
                anchors[3] -= anchors[4];
                anchors[5]  = integral[ty + fw * 2];
                anchors[4] -= anchors[5];
                anchors[5] -= integral[ty + fw * 3];

                anchors[0] -= anchors[3];
                anchors[1] -= anchors[4];
                anchors[2] -= anchors[5];
                // 0 - 2 contains s0 - s2

                ty += fh;
                anchors[6]  = integral[ty];
                anchors[7]  = integral[ty + fw];
                anchors[6] -= anchors[7];
                anchors[8]  = integral[ty + fw * 2];
                anchors[7] -= anchors[8];
                anchors[8] -= integral[ty + fw * 3];

                anchors[3] -= anchors[6];
                anchors[4] -= anchors[7];
                anchors[5] -= anchors[8];
                // 3 - 5 contains s3 - s5

                anchors[0] -= anchors[4];
                anchors[1] -= anchors[4];
                anchors[2] -= anchors[4];
                anchors[3] -= anchors[4];
                anchors[5] -= anchors[4];

                int response = (~(anchors[0] >> 31)) & 4;
                response |= (~(anchors[1] >> 31)) & 2;;
                response |= (~(anchors[2] >> 31)) & 1;

                shift = (~(anchors[5] >> 31)) & 16;
                shift |= (~(anchors[3] >> 31)) & 1;

                ty += fh;
                anchors[0]  = integral[ty];
                anchors[1]  = integral[ty + fw];
                anchors[0] -= anchors[1];
                anchors[2]  = integral[ty + fw * 2];
                anchors[1] -= anchors[2];
                anchors[2] -= integral[ty + fw * 3];

                anchors[6] -= anchors[0];
                anchors[7] -= anchors[1];
                anchors[8] -= anchors[2];
                // 0 -2 contains s6 - s8

                anchors[6] -= anchors[4];
                anchors[7] -= anchors[4];
                anchors[8] -= anchors[4];

                shift |= (~(anchors[6] >> 31)) & 2;
                shift |= (~(anchors[7] >> 31)) & 4;
                shift |= (~(anchors[8] >> 31)) & 8;
                return response;
            }
        };

        template<typename Pr>
        __global__ void disjoin(int4* candidates, int4* objects, unsigned int n, int groupThreshold, float grouping_eps, unsigned int* nclasses)
        {
            unsigned int tid = threadIdx.x;
            extern __shared__ int sbuff[];

            int* labels = sbuff;
            int* rrects = sbuff + n;

            Pr predicate(grouping_eps);
            partition(candidates, n, labels, predicate);

            rrects[tid * 4 + 0] = 0;
            rrects[tid * 4 + 1] = 0;
            rrects[tid * 4 + 2] = 0;
            rrects[tid * 4 + 3] = 0;
            __syncthreads();

            int cls = labels[tid];
            Emulation::smem::atomicAdd((rrects + cls * 4 + 0), candidates[tid].x);
            Emulation::smem::atomicAdd((rrects + cls * 4 + 1), candidates[tid].y);
            Emulation::smem::atomicAdd((rrects + cls * 4 + 2), candidates[tid].z);
            Emulation::smem::atomicAdd((rrects + cls * 4 + 3), candidates[tid].w);

            __syncthreads();
            labels[tid] = 0;

            __syncthreads();
            Emulation::smem::atomicInc((unsigned int*)labels + cls, n);

            __syncthreads();
            *nclasses = 0;

            int active = labels[tid];
            if (active)
            {
                int* r1 = rrects + tid * 4;
                float s = 1.f / active;
                r1[0] = saturate_cast<int>(r1[0] * s);
                r1[1] = saturate_cast<int>(r1[1] * s);
                r1[2] = saturate_cast<int>(r1[2] * s);
                r1[3] = saturate_cast<int>(r1[3] * s);
            }
            __syncthreads();

            if (active && active >= groupThreshold)
            {
                int* r1 = rrects + tid * 4;
                int4 r_out = make_int4(r1[0], r1[1], r1[2], r1[3]);

                int aidx = Emulation::smem::atomicInc(nclasses, n);
                objects[aidx] = r_out;
            }
        }

        void connectedConmonents(PtrStepSz<int4> candidates, int ncandidates, PtrStepSz<int4> objects, int groupThreshold, float grouping_eps, unsigned int* nclasses)
        {
            if (!ncandidates) return;
            int block = ncandidates;
            int smem  = block * ( sizeof(int) + sizeof(int4) );
            disjoin<InSameComponint><<<1, block, smem>>>(candidates, objects, ncandidates, groupThreshold, grouping_eps, nclasses);
            cudaSafeCall( cudaGetLastError() );
        }

        struct Cascade
        {
            __host__ __device__ __forceinline__ Cascade(const Stage* _stages, int _nstages, const ClNode* _nodes, const float* _leaves,
                const int* _subsets, const uchar4* _features, int _subsetSize)

            : stages(_stages), nstages(_nstages), nodes(_nodes), leaves(_leaves), subsets(_subsets), features(_features), subsetSize(_subsetSize){}

            __device__ __forceinline__ bool operator() (int y, int x, int* integral, const int pitch) const
            {
                int current_node = 0;
                int current_leave = 0;

                for (int s = 0; s < nstages; ++s)
                {
                    float sum = 0;
                    Stage stage = stages[s];
                    for (int t = 0; t < stage.ntrees; t++)
                    {
                        ClNode node = nodes[current_node];
                        uchar4 feature = features[node.featureIdx];

                        int shift;
                        int c = evaluator(integral, (y + feature.y) * pitch + x + feature.x, feature.w * pitch, feature.z, shift);
                        int idx =  (subsets[ current_node * subsetSize + c] & ( 1 << shift)) ? current_leave : current_leave + 1;
                        sum += leaves[idx];

                        current_node += 1;
                        current_leave += 2;
                    }

                    if (sum < stage.threshold)
                        return false;
                }

                return true;
            }

            const Stage*  stages;
            const int nstages;

            const ClNode* nodes;
            const float* leaves;
            const int* subsets;
            const uchar4* features;

            const int subsetSize;
            const LBP evaluator;
        };

        // stepShift, scale, width_k, sum_prev => y =  sum_prev + tid_k / width_k, x = tid_k - tid_k / width_k
        __global__ void lbp_cascade(const Cascade cascade, int frameW, int frameH, int windowW, int windowH, float scale, const float factor,
            const int total, int* integral, const int pitch, PtrStepSz<int4> objects, unsigned int* classified)
        {
            int ftid = blockIdx.x * blockDim.x + threadIdx.x;
            if (ftid >= total) return;

            int step = (scale <= 2.f);

            int windowsForLine = (__float2int_rn( __fdividef(frameW, scale)) - windowW) >> step;
            int stotal = windowsForLine * ( (__float2int_rn( __fdividef(frameH, scale)) - windowH) >> step);
            int wshift = 0;

            int scaleTid = ftid;

            while (scaleTid >= stotal)
            {
                scaleTid -= stotal;
                wshift += __float2int_rn(__fdividef(frameW, scale)) + 1;
                scale *= factor;
                step = (scale <= 2.f);
                windowsForLine = ( ((__float2int_rn(__fdividef(frameW, scale)) - windowW) >> step));
                stotal = windowsForLine * ( (__float2int_rn(__fdividef(frameH, scale)) - windowH) >> step);
            }

            int y = __fdividef(scaleTid, windowsForLine);
            int x = scaleTid - y * windowsForLine;

            x <<= step;
            y <<= step;

            if (cascade(y, x + wshift, integral, pitch))
            {
                if(x >= __float2int_rn(__fdividef(frameW, scale)) - windowW) return;

                int4 rect;
                rect.x = __float2int_rn(x * scale);
                rect.y = __float2int_rn(y * scale);
                rect.z = __float2int_rn(windowW * scale);
                rect.w = __float2int_rn(windowH * scale);

                int res = atomicInc(classified, (unsigned int)objects.cols);
                objects(0, res) = rect;
            }
        }

        void classifyPyramid(int frameW, int frameH, int windowW, int windowH, float initialScale, float factor, int workAmount,
            const PtrStepSzb& mstages, const int nstages, const PtrStepSzi& mnodes, const PtrStepSzf& mleaves, const PtrStepSzi& msubsets, const PtrStepSzb& mfeatures,
            const int subsetSize, PtrStepSz<int4> objects, unsigned int* classified, PtrStepSzi integral)
        {
            const int block = 128;
            int grid = divUp(workAmount, block);
            cudaFuncSetCacheConfig(lbp_cascade, cudaFuncCachePreferL1);
            Cascade cascade((Stage*)mstages.ptr(), nstages, (ClNode*)mnodes.ptr(), mleaves.ptr(), msubsets.ptr(), (uchar4*)mfeatures.ptr(), subsetSize);
            lbp_cascade<<<grid, block>>>(cascade, frameW, frameH, windowW, windowH, initialScale, factor, workAmount, integral.ptr(), (int)integral.step / sizeof(int), objects, classified);
        }
    }
}}}

#endif /* CUDA_DISABLER */
