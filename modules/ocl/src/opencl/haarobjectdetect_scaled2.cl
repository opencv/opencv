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
//    Sen Liu, swjtuls1987@126.com
//    Peng Xiao, pengxiao@outlook.com
//    Erping Pang, erping@multicorewareinc.com
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

typedef struct __attribute__((aligned(128))) GpuHidHaarTreeNode
{
    int p[CV_HAAR_FEATURE_MAX][4] __attribute__((aligned(64)));
    float weight[CV_HAAR_FEATURE_MAX] /*__attribute__((aligned (16)))*/;
    float threshold /*__attribute__((aligned (4)))*/;
    float alpha[3] __attribute__((aligned(16)));
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
    const int rows,
    const int cols,
    const int step,
    const int loopcount,
    const int start_stage,
    const int split_stage,
    const int end_stage,
    const int startnode,
    global int4 *p,
    global float *correction,
    const int nodecount)
{
    int grpszx = get_local_size(0);
    int grpszy = get_local_size(1);
    int grpnumx = get_num_groups(0);
    int grpidx = get_group_id(0);
    int lclidx = get_local_id(0);
    int lclidy = get_local_id(1);
    int lcl_id = mad24(lclidy, grpszx, lclidx);
    __local int glboutindex[1];
    __local int lclcount[1];
    __local int lcloutindex[64];
    glboutindex[0] = 0;
    int outputoff = mul24(grpidx, 256);
    candidate[outputoff + (lcl_id << 2)] = (int4)0;
    candidate[outputoff + (lcl_id << 2) + 1] = (int4)0;
    candidate[outputoff + (lcl_id << 2) + 2] = (int4)0;
    candidate[outputoff + (lcl_id << 2) + 3] = (int4)0;
    int max_idx = rows * cols - 1;
    for (int scalei = 0; scalei < loopcount; scalei++)
    {
        int4 scaleinfo1;
        scaleinfo1 = info[scalei];
        int grpnumperline = (scaleinfo1.y & 0xffff0000) >> 16;
        int totalgrp = scaleinfo1.y & 0xffff;
        float factor = as_float(scaleinfo1.w);
        float correction_t = correction[scalei];
        float ystep = max(2.0f, factor);

        for (int grploop = get_group_id(0); grploop < totalgrp; grploop += grpnumx)
        {
            int4 cascadeinfo = p[scalei];
            int grpidy = grploop / grpnumperline;
            int grpidx = grploop - mul24(grpidy, grpnumperline);
            int ix = mad24(grpidx, grpszx, lclidx);
            int iy = mad24(grpidy, grpszy, lclidy);
            int x = round(ix * ystep);
            int y = round(iy * ystep);
            lcloutindex[lcl_id] = 0;
            lclcount[0] = 0;
            int nodecounter;
            float mean, variance_norm_factor;
            //if((ix < width) && (iy < height))
            {
                const int p_offset = mad24(y, step, x);
                cascadeinfo.x += p_offset;
                cascadeinfo.z += p_offset;
                mean = (sum[clamp(mad24(cascadeinfo.y, step, cascadeinfo.x), 0, max_idx)]
                - sum[clamp(mad24(cascadeinfo.y, step, cascadeinfo.z), 0, max_idx)] -
                        sum[clamp(mad24(cascadeinfo.w, step, cascadeinfo.x), 0, max_idx)]
                + sum[clamp(mad24(cascadeinfo.w, step, cascadeinfo.z), 0, max_idx)])
                       * correction_t;
                variance_norm_factor = sqsum[clamp(mad24(cascadeinfo.y, step, cascadeinfo.x), 0, max_idx)]
                - sqsum[clamp(mad24(cascadeinfo.y, step, cascadeinfo.z), 0, max_idx)] -
                                       sqsum[clamp(mad24(cascadeinfo.w, step, cascadeinfo.x), 0, max_idx)]
                + sqsum[clamp(mad24(cascadeinfo.w, step, cascadeinfo.z), 0, max_idx)];
                variance_norm_factor = variance_norm_factor * correction_t - mean * mean;
                variance_norm_factor = variance_norm_factor >= 0.f ? sqrt(variance_norm_factor) : 1.f;
                bool result = true;
                nodecounter = startnode + nodecount * scalei;
                for (int stageloop = start_stage; (stageloop < end_stage) && result; stageloop++)
                {
                    float stage_sum = 0.f;
                    int   stagecount = stagecascadeptr[stageloop].count;
                    for (int nodeloop = 0; nodeloop < stagecount;)
                    {
                        __global GpuHidHaarTreeNode *currentnodeptr = (nodeptr + nodecounter);
                        int4 info1 = *(__global int4 *)(&(currentnodeptr->p[0][0]));
                        int4 info2 = *(__global int4 *)(&(currentnodeptr->p[1][0]));
                        int4 info3 = *(__global int4 *)(&(currentnodeptr->p[2][0]));
                        float4 w = *(__global float4 *)(&(currentnodeptr->weight[0]));
                        float3 alpha3 = *(__global float3 *)(&(currentnodeptr->alpha[0]));
                        float nodethreshold  = w.w * variance_norm_factor;

                        info1.x += p_offset;
                        info1.z += p_offset;
                        info2.x += p_offset;
                        info2.z += p_offset;
                        info3.x += p_offset;
                        info3.z += p_offset;
                        float classsum = (sum[clamp(mad24(info1.y, step, info1.x), 0, max_idx)]
                        - sum[clamp(mad24(info1.y, step, info1.z), 0, max_idx)] -
                                          sum[clamp(mad24(info1.w, step, info1.x), 0, max_idx)]
                        + sum[clamp(mad24(info1.w, step, info1.z), 0, max_idx)]) * w.x;
                        classsum += (sum[clamp(mad24(info2.y, step, info2.x), 0, max_idx)]
                        - sum[clamp(mad24(info2.y, step, info2.z), 0, max_idx)] -
                                     sum[clamp(mad24(info2.w, step, info2.x), 0, max_idx)]
                        + sum[clamp(mad24(info2.w, step, info2.z), 0, max_idx)]) * w.y;
                        classsum += (sum[clamp(mad24(info3.y, step, info3.x), 0, max_idx)]
                        - sum[clamp(mad24(info3.y, step, info3.z), 0, max_idx)] -
                                     sum[clamp(mad24(info3.w, step, info3.x), 0, max_idx)]
                        + sum[clamp(mad24(info3.w, step, info3.z), 0, max_idx)]) * w.z;

                        bool passThres = classsum >= nodethreshold;

#if STUMP_BASED
                        stage_sum += passThres ? alpha3.y : alpha3.x;
                        nodecounter++;
                        nodeloop++;
#else
                        bool isRootNode = (nodecounter & 1) == 0;
                        if(isRootNode)
                        {
                            if( (passThres && currentnodeptr->right) ||
                                (!passThres && currentnodeptr->left))
                            {
                                nodecounter ++;
                            }
                            else
                            {
                                stage_sum += alpha3.x;
                                nodecounter += 2;
                                nodeloop ++;
                            }
                        }
                        else
                        {
                            stage_sum += (passThres ? alpha3.z : alpha3.y);
                            nodecounter ++;
                            nodeloop ++;
                        }
#endif
                    }
                    result = (int)(stage_sum >= stagecascadeptr[stageloop].threshold);
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if (result)
                {
                    int queueindex = atomic_inc(lclcount);
                    lcloutindex[queueindex] = (y << 16) | x;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                int queuecount = lclcount[0];

                if (lcl_id < queuecount)
                {
                    int temp = lcloutindex[lcl_id];
                    int x = temp & 0xffff;
                    int y = (temp & (int)0xffff0000) >> 16;
                    temp = atomic_inc(glboutindex);
                    int4 candidate_result;
                    candidate_result.zw = (int2)convert_int_rte(factor * 20.f);
                    candidate_result.x = x;
                    candidate_result.y = y;

                    int i = outputoff+temp+lcl_id;
                    if(candidate[i].z == 0)
                    {
                        candidate[i] = candidate_result;
                    }
                    else
                    {
                        for(i=i+1;;i++)
                        {
                            if(candidate[i].z == 0)
                            {
                                candidate[i] = candidate_result;
                                break;
                            }
                        }
                    }
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

    t1.weight[0] = -(t1.weight[1] * tr_h[1] * tr_w[1] + t1.weight[2] * tr_h[2] * tr_w[2]) / (tr_h[0] * tr_w[0]);
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
    newnode[counter].alpha[2] = t1.alpha[2];
}
