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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Wu Xinglong, wxl370@126.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

// Enter your kernel in this window
//#pragma OPENCL EXTENSION cl_amd_printf:enable
#define CV_HAAR_FEATURE_MAX           3
typedef int   sumtype;
typedef float sqsumtype;
typedef struct  __attribute__((aligned(128)))  GpuHidHaarFeature
{
    struct __attribute__((aligned(32)))
{
        int p0 __attribute__((aligned(4)));
        int p1 __attribute__((aligned(4)));
        int p2 __attribute__((aligned(4)));
        int p3 __attribute__((aligned(4)));
        float weight __attribute__((aligned(4)));
}
rect[CV_HAAR_FEATURE_MAX] __attribute__((aligned(32)));
}
GpuHidHaarFeature;
typedef struct __attribute__((aligned(128))) GpuHidHaarTreeNode
{
    int p[CV_HAAR_FEATURE_MAX][4] __attribute__((aligned(64)));
    float weight[CV_HAAR_FEATURE_MAX] /*__attribute__((aligned (16)))*/;
    float threshold /*__attribute__((aligned (4)))*/;
    float alpha[2] __attribute__((aligned(8)));
    int left __attribute__((aligned(4)));
    int right __attribute__((aligned(4)));
}
GpuHidHaarTreeNode;
typedef struct __attribute__((aligned(32))) GpuHidHaarClassifier
{
    int count __attribute__((aligned(4)));
    GpuHidHaarTreeNode *node __attribute__((aligned(8)));
    float *alpha __attribute__((aligned(8)));
}
GpuHidHaarClassifier;
typedef struct __attribute__((aligned(64))) GpuHidHaarStageClassifier
{
    int  count __attribute__((aligned(4)));
    float threshold __attribute__((aligned(4)));
    int two_rects __attribute__((aligned(4)));
    int reserved0 __attribute__((aligned(8)));
    int reserved1 __attribute__((aligned(8)));
    int reserved2 __attribute__((aligned(8)));
    int reserved3 __attribute__((aligned(8)));
}
GpuHidHaarStageClassifier;
typedef struct __attribute__((aligned(64))) GpuHidHaarClassifierCascade
{
    int  count __attribute__((aligned(4)));
    int  is_stump_based __attribute__((aligned(4)));
    int  has_tilted_features __attribute__((aligned(4)));
    int  is_tree __attribute__((aligned(4)));
    int pq0 __attribute__((aligned(4)));
    int pq1 __attribute__((aligned(4)));
    int pq2 __attribute__((aligned(4)));
    int pq3 __attribute__((aligned(4)));
    int p0 __attribute__((aligned(4)));
    int p1 __attribute__((aligned(4)));
    int p2 __attribute__((aligned(4)));
    int p3 __attribute__((aligned(4)));
    float inv_window_area __attribute__((aligned(4)));
} GpuHidHaarClassifierCascade;

__kernel void gpuRunHaarClassifierCascade_scaled2(
    global GpuHidHaarStageClassifier *stagecascadeptr,
    global int4 *info,
    global GpuHidHaarTreeNode *nodeptr,
    global const int *restrict sum,
    global const float   *restrict sqsum,
    global int4 *candidate,
    const int step,
    const int loopcount,
    const int start_stage,
    const int split_stage,
    const int end_stage,
    const int startnode,
    const int splitnode,
    global int4 *p,
    //const int4 * pq,
    global float *correction,
    const int nodecount)
{
        int grpszx = get_local_size(0);
        int grpszy = get_local_size(1);
        int grpnumx = get_num_groups(0);
        int grpidx = get_group_id(0);
        int lclidx = get_local_id(0);
        int lclidy = get_local_id(1);
        int lcl_sz = mul24(grpszx, grpszy);
        int lcl_id = mad24(lclidy, grpszx, lclidx);
        __local int lclshare[1024];
        __local int *glboutindex = lclshare + 0;
        __local int *lclcount = glboutindex + 1;
        __local int *lcloutindex = lclcount + 1;
        __local float *partialsum = (__local float *)(lcloutindex + (lcl_sz << 1));
        glboutindex[0] = 0;
        int outputoff = mul24(grpidx, 256);
        candidate[outputoff + (lcl_id << 2)] = (int4)0;
        candidate[outputoff + (lcl_id << 2) + 1] = (int4)0;
        candidate[outputoff + (lcl_id << 2) + 2] = (int4)0;
        candidate[outputoff + (lcl_id << 2) + 3] = (int4)0;

        for (int scalei = 0; scalei < loopcount; scalei++)
        {
                int4 scaleinfo1;
                scaleinfo1 = info[scalei];
                int width = (scaleinfo1.x & 0xffff0000) >> 16;
                int height = scaleinfo1.x & 0xffff;
                int grpnumperline = (scaleinfo1.y & 0xffff0000) >> 16;
                int totalgrp = scaleinfo1.y & 0xffff;
                float factor = as_float(scaleinfo1.w);
                float correction_t = correction[scalei];
                int ystep = (int)(max(2.0f, factor) + 0.5f);

                for (int grploop = get_group_id(0); grploop < totalgrp; grploop += grpnumx)
                {
                        int4 cascadeinfo = p[scalei];
                        int grpidy = grploop / grpnumperline;
                        int grpidx = grploop - mul24(grpidy, grpnumperline);
                        int ix = mad24(grpidx, grpszx, lclidx);
                        int iy = mad24(grpidy, grpszy, lclidy);
                        int x = ix * ystep;
                        int y = iy * ystep;
                        lcloutindex[lcl_id] = 0;
                        lclcount[0] = 0;
                        int result = 1, nodecounter;
                        float mean, variance_norm_factor;
                        //if((ix < width) && (iy < height))
                        {
                                const int p_offset = mad24(y, step, x);
                                cascadeinfo.x += p_offset;
                                cascadeinfo.z += p_offset;
                                mean = (sum[mad24(cascadeinfo.y, step, cascadeinfo.x)] - sum[mad24(cascadeinfo.y, step, cascadeinfo.z)] -
                                        sum[mad24(cascadeinfo.w, step, cascadeinfo.x)] + sum[mad24(cascadeinfo.w, step, cascadeinfo.z)])
                                       * correction_t;
                                variance_norm_factor = sqsum[mad24(cascadeinfo.y, step, cascadeinfo.x)] - sqsum[mad24(cascadeinfo.y, step, cascadeinfo.z)] -
                                                       sqsum[mad24(cascadeinfo.w, step, cascadeinfo.x)] + sqsum[mad24(cascadeinfo.w, step, cascadeinfo.z)];
                                variance_norm_factor = variance_norm_factor * correction_t - mean * mean;
                                variance_norm_factor = variance_norm_factor >= 0.f ? sqrt(variance_norm_factor) : 1.f;
                                result = 1;
                                nodecounter = startnode + nodecount * scalei;

                                for (int stageloop = start_stage; stageloop < end_stage && result; stageloop++)
                                {
                                        float stage_sum = 0.f;
                                        int4 stageinfo = *(global int4 *)(stagecascadeptr + stageloop);
                                        float stagethreshold = as_float(stageinfo.y);

                                        for (int nodeloop = 0; nodeloop < stageinfo.x; nodeloop++)
                                        {
                                                __global GpuHidHaarTreeNode *currentnodeptr = (nodeptr + nodecounter);
                                                int4 info1 = *(__global int4 *)(&(currentnodeptr->p[0][0]));
                                                int4 info2 = *(__global int4 *)(&(currentnodeptr->p[1][0]));
                                                int4 info3 = *(__global int4 *)(&(currentnodeptr->p[2][0]));
                                                float4 w = *(__global float4 *)(&(currentnodeptr->weight[0]));
                                                float2 alpha2 = *(__global float2 *)(&(currentnodeptr->alpha[0]));
                                                float nodethreshold  = w.w * variance_norm_factor;
                                                info1.x += p_offset;
                                                info1.z += p_offset;
                                                info2.x += p_offset;
                                                info2.z += p_offset;
                                                float classsum = (sum[mad24(info1.y, step, info1.x)] - sum[mad24(info1.y, step, info1.z)] -
                                                                  sum[mad24(info1.w, step, info1.x)] + sum[mad24(info1.w, step, info1.z)]) * w.x;
                                                classsum += (sum[mad24(info2.y, step, info2.x)] - sum[mad24(info2.y, step, info2.z)] -
                                                             sum[mad24(info2.w, step, info2.x)] + sum[mad24(info2.w, step, info2.z)]) * w.y;
                                                info3.x += p_offset;
                                                info3.z += p_offset;
                                                classsum += (sum[mad24(info3.y, step, info3.x)] - sum[mad24(info3.y, step, info3.z)] -
                                                             sum[mad24(info3.w, step, info3.x)] + sum[mad24(info3.w, step, info3.z)]) * w.z;
                                                stage_sum += classsum >= nodethreshold ? alpha2.y : alpha2.x;
                                                nodecounter++;
                                        }

                                        result = (stage_sum >= stagethreshold);
                                }

                                if (result && (ix < width) && (iy < height))
                                {
                                        int queueindex = atomic_inc(lclcount);
                                        lcloutindex[queueindex << 1] = (y << 16) | x;
                                        lcloutindex[(queueindex << 1) + 1] = as_int(variance_norm_factor);
                                }

                                barrier(CLK_LOCAL_MEM_FENCE);
                                int queuecount = lclcount[0];
                                nodecounter = splitnode + nodecount * scalei;

                                if (lcl_id < queuecount)
                                {
                                        int temp = lcloutindex[lcl_id << 1];
                                        int x = temp & 0xffff;
                                        int y = (temp & (int)0xffff0000) >> 16;
                                        temp = glboutindex[0];
                                        int4 candidate_result;
                                        candidate_result.zw = (int2)convert_int_rtn(factor * 20.f);
                                        candidate_result.x = x;
                                        candidate_result.y = y;
                                        atomic_inc(glboutindex);
                                        candidate[outputoff + temp + lcl_id] = candidate_result;
                                }

                                barrier(CLK_LOCAL_MEM_FENCE);
                        }
                }
        }
}
__kernel void gpuscaleclassifier(global GpuHidHaarTreeNode *orinode, global GpuHidHaarTreeNode *newnode, float scale, float weight_scale, int nodenum)
{
        int counter = get_global_id(0);
        int tr_x[3], tr_y[3], tr_h[3], tr_w[3], i = 0;
        GpuHidHaarTreeNode t1 = *(orinode + counter);
#pragma unroll

        for (i = 0; i < 3; i++)
        {
                tr_x[i] = (int)(t1.p[i][0] * scale + 0.5f);
                tr_y[i] = (int)(t1.p[i][1] * scale + 0.5f);
                tr_w[i] = (int)(t1.p[i][2] * scale + 0.5f);
                tr_h[i] = (int)(t1.p[i][3] * scale + 0.5f);
        }

        t1.weight[0] = t1.p[2][0] ? -(t1.weight[1] * tr_h[1] * tr_w[1] + t1.weight[2] * tr_h[2] * tr_w[2]) / (tr_h[0] * tr_w[0]) : -t1.weight[1] * tr_h[1] * tr_w[1] / (tr_h[0] * tr_w[0]);
        counter += nodenum;
#pragma unroll

        for (i = 0; i < 3; i++)
        {
                newnode[counter].p[i][0] = tr_x[i];
                newnode[counter].p[i][1] = tr_y[i];
                newnode[counter].p[i][2] = tr_x[i] + tr_w[i];
                newnode[counter].p[i][3] = tr_y[i] + tr_h[i];
                newnode[counter].weight[i] = t1.weight[i] * weight_scale;
        }

        newnode[counter].left = t1.left;
        newnode[counter].right = t1.right;
        newnode[counter].threshold = t1.threshold;
        newnode[counter].alpha[0] = t1.alpha[0];
        newnode[counter].alpha[1] = t1.alpha[1];
}

