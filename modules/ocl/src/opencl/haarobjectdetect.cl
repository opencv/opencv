//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Wang Weiyan, wangweiyanster@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Nathan, liujun@multicorewareinc.com
//    Peng Xiao, pengxiao@outlook.com
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
//

#pragma OPENCL EXTENSION cl_amd_printf : enable
#define CV_HAAR_FEATURE_MAX           3

#define calc_sum(rect,offset)        (sum[(rect).p0+offset] - sum[(rect).p1+offset] - sum[(rect).p2+offset] + sum[(rect).p3+offset])
#define calc_sum1(rect,offset,i)     (sum[(rect).p0[i]+offset] - sum[(rect).p1[i]+offset] - sum[(rect).p2[i]+offset] + sum[(rect).p3[i]+offset])

typedef int   sumtype;
typedef float sqsumtype;

#ifndef STUMP_BASED
#define STUMP_BASED 1
#endif

typedef struct __attribute__((aligned (128) )) GpuHidHaarTreeNode
{
    int p[CV_HAAR_FEATURE_MAX][4] __attribute__((aligned (64)));
    float weight[CV_HAAR_FEATURE_MAX];
    float threshold;
    float alpha[3] __attribute__((aligned (16)));
    int left __attribute__((aligned (4)));
    int right __attribute__((aligned (4)));
}
GpuHidHaarTreeNode;


typedef struct __attribute__((aligned (32))) GpuHidHaarClassifier
{
    int count __attribute__((aligned (4)));
    GpuHidHaarTreeNode* node __attribute__((aligned (8)));
    float* alpha __attribute__((aligned (8)));
}
GpuHidHaarClassifier;


typedef struct __attribute__((aligned (64))) GpuHidHaarStageClassifier
{
    int  count __attribute__((aligned (4)));
    float threshold __attribute__((aligned (4)));
    int two_rects __attribute__((aligned (4)));
    int reserved0 __attribute__((aligned (8)));
    int reserved1 __attribute__((aligned (8)));
    int reserved2 __attribute__((aligned (8)));
    int reserved3 __attribute__((aligned (8)));
}
GpuHidHaarStageClassifier;


typedef struct __attribute__((aligned (64))) GpuHidHaarClassifierCascade
{
    int  count __attribute__((aligned (4)));
    int  is_stump_based __attribute__((aligned (4)));
    int  has_tilted_features __attribute__((aligned (4)));
    int  is_tree __attribute__((aligned (4)));
    int pq0 __attribute__((aligned (4)));
    int pq1 __attribute__((aligned (4)));
    int pq2 __attribute__((aligned (4)));
    int pq3 __attribute__((aligned (4)));
    int p0 __attribute__((aligned (4)));
    int p1 __attribute__((aligned (4)));
    int p2 __attribute__((aligned (4)));
    int p3 __attribute__((aligned (4)));
    float inv_window_area __attribute__((aligned (4)));
} GpuHidHaarClassifierCascade;

__kernel void __attribute__((reqd_work_group_size(8,8,1)))gpuRunHaarClassifierCascade(
    global GpuHidHaarStageClassifier * stagecascadeptr,
    global int4 * info,
    global GpuHidHaarTreeNode * nodeptr,
    global const int * restrict sum1,
    global const float * restrict sqsum1,
    global int4 * candidate,
    const int pixelstep,
    const int loopcount,
    const int start_stage,
    const int split_stage,
    const int end_stage,
    const int startnode,
    const int splitnode,
    const int4 p,
    const int4 pq,
    const float correction)
{
    int grpszx = get_local_size(0);
    int grpszy = get_local_size(1);
    int grpnumx = get_num_groups(0);
    int grpidx = get_group_id(0);
    int lclidx = get_local_id(0);
    int lclidy = get_local_id(1);

    int lcl_sz = mul24(grpszx,grpszy);
    int lcl_id = mad24(lclidy,grpszx,lclidx);

    __local int lclshare[1024];
    __local int* lcldata = lclshare;//for save win data
    __local int* glboutindex = lcldata + 28*28;//for save global out index
    __local int* lclcount = glboutindex + 1;//for save the numuber of temp pass pixel
    __local int* lcloutindex = lclcount + 1;//for save info of temp pass pixel
    __local float* partialsum = (__local float*)(lcloutindex + (lcl_sz<<1));
    glboutindex[0]=0;
    int outputoff = mul24(grpidx,256);

    //assume window size is 20X20
#define WINDOWSIZE 20+1
    //make sure readwidth is the multiple of 4
    //ystep =1, from host code
    int readwidth = ((grpszx-1 + WINDOWSIZE+3)>>2)<<2;
    int readheight = grpszy-1+WINDOWSIZE;
    int read_horiz_cnt = readwidth >> 2;//each read int4
    int total_read = mul24(read_horiz_cnt,readheight);
    int read_loop = (total_read + lcl_sz - 1) >> 6;
    candidate[outputoff+(lcl_id<<2)] = (int4)0;
    candidate[outputoff+(lcl_id<<2)+1] = (int4)0;
    candidate[outputoff+(lcl_id<<2)+2] = (int4)0;
    candidate[outputoff+(lcl_id<<2)+3] = (int4)0;
    for(int scalei = 0; scalei <loopcount; scalei++)
    {
        int4 scaleinfo1= info[scalei];
        int width = (scaleinfo1.x & 0xffff0000) >> 16;
        int height = scaleinfo1.x & 0xffff;
        int grpnumperline =(scaleinfo1.y & 0xffff0000) >> 16;
        int totalgrp = scaleinfo1.y & 0xffff;
        int imgoff = scaleinfo1.z;
        float factor = as_float(scaleinfo1.w);

        __global const int * sum = sum1 + imgoff;
        __global const float * sqsum = sqsum1 + imgoff;
        for(int grploop=grpidx; grploop<totalgrp; grploop+=grpnumx)
        {
            int grpidy = grploop / grpnumperline;
            int grpidx = grploop - mul24(grpidy, grpnumperline);
            int x = mad24(grpidx,grpszx,lclidx);
            int y = mad24(grpidy,grpszy,lclidy);
            int grpoffx = x-lclidx;
            int grpoffy = y-lclidy;

            for(int i=0; i<read_loop; i++)
            {
                int pos_id = mad24(i,lcl_sz,lcl_id);
                pos_id = pos_id < total_read ? pos_id : 0;

                int lcl_y = pos_id / read_horiz_cnt;
                int lcl_x = pos_id - mul24(lcl_y, read_horiz_cnt);

                int glb_x = grpoffx + (lcl_x<<2);
                int glb_y = grpoffy + lcl_y;

                int glb_off = mad24(min(glb_y, height - 1),pixelstep,glb_x);
                int4 data = *(__global int4*)&sum[glb_off];
                int lcl_off = mad24(lcl_y, readwidth, lcl_x<<2);

                vstore4(data, 0, &lcldata[lcl_off]);
            }

            lcloutindex[lcl_id] = 0;
            lclcount[0] = 0;
            int result = 1;
            int nodecounter= startnode;
            float mean, variance_norm_factor;
            barrier(CLK_LOCAL_MEM_FENCE);

            int lcl_off = mad24(lclidy,readwidth,lclidx);
            int4 cascadeinfo1, cascadeinfo2;
            cascadeinfo1 = p;
            cascadeinfo2 = pq;

            cascadeinfo1.x +=lcl_off;
            cascadeinfo1.z +=lcl_off;
            mean = (lcldata[mad24(cascadeinfo1.y,readwidth,cascadeinfo1.x)] - lcldata[mad24(cascadeinfo1.y,readwidth,cascadeinfo1.z)] -
                    lcldata[mad24(cascadeinfo1.w,readwidth,cascadeinfo1.x)] + lcldata[mad24(cascadeinfo1.w,readwidth,cascadeinfo1.z)])
                    *correction;

            int p_offset = mad24(y, pixelstep, x);

            cascadeinfo2.x +=p_offset;
            cascadeinfo2.z +=p_offset;
            variance_norm_factor =sqsum[mad24(cascadeinfo2.y, pixelstep, cascadeinfo2.x)] - sqsum[mad24(cascadeinfo2.y, pixelstep, cascadeinfo2.z)] -
                                    sqsum[mad24(cascadeinfo2.w, pixelstep, cascadeinfo2.x)] + sqsum[mad24(cascadeinfo2.w, pixelstep, cascadeinfo2.z)];

            variance_norm_factor = variance_norm_factor * correction - mean * mean;
            variance_norm_factor = variance_norm_factor >=0.f ? sqrt(variance_norm_factor) : 1.f;

            for(int stageloop = start_stage; (stageloop < split_stage)  && result; stageloop++ )
            {
                float stage_sum = 0.f;
                int2 stageinfo = *(global int2*)(stagecascadeptr+stageloop);
                float stagethreshold = as_float(stageinfo.y);
                for(int nodeloop = 0; nodeloop < stageinfo.x; )
                {
                    __global GpuHidHaarTreeNode* currentnodeptr = (nodeptr + nodecounter);

                    int4 info1 = *(__global int4*)(&(currentnodeptr->p[0][0]));
                    int4 info2 = *(__global int4*)(&(currentnodeptr->p[1][0]));
                    int4 info3 = *(__global int4*)(&(currentnodeptr->p[2][0]));
                    float4 w = *(__global float4*)(&(currentnodeptr->weight[0]));
                    float3 alpha3 = *(__global float3*)(&(currentnodeptr->alpha[0]));

                    float nodethreshold  = w.w * variance_norm_factor;

                    info1.x +=lcl_off;
                    info1.z +=lcl_off;
                    info2.x +=lcl_off;
                    info2.z +=lcl_off;

                    float classsum = (lcldata[mad24(info1.y,readwidth,info1.x)] - lcldata[mad24(info1.y,readwidth,info1.z)] -
                                        lcldata[mad24(info1.w,readwidth,info1.x)] + lcldata[mad24(info1.w,readwidth,info1.z)]) * w.x;

                    classsum += (lcldata[mad24(info2.y,readwidth,info2.x)] - lcldata[mad24(info2.y,readwidth,info2.z)] -
                                    lcldata[mad24(info2.w,readwidth,info2.x)] + lcldata[mad24(info2.w,readwidth,info2.z)]) * w.y;

                    info3.x +=lcl_off;
                    info3.z +=lcl_off;
                    classsum += (lcldata[mad24(info3.y,readwidth,info3.x)] - lcldata[mad24(info3.y,readwidth,info3.z)] -
                                    lcldata[mad24(info3.w,readwidth,info3.x)] + lcldata[mad24(info3.w,readwidth,info3.z)]) * w.z;

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
                        stage_sum += passThres ? alpha3.z : alpha3.y;
                        nodecounter ++;
                        nodeloop ++;
                    }
#endif
                }

                result = (stage_sum >= stagethreshold);
            }

            if(result && (x < width) && (y < height))
            {
                int queueindex = atomic_inc(lclcount);
                lcloutindex[queueindex<<1] = (lclidy << 16) | lclidx;
                lcloutindex[(queueindex<<1)+1] = as_int(variance_norm_factor);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            int queuecount  = lclcount[0];
            barrier(CLK_LOCAL_MEM_FENCE);
            nodecounter = splitnode;
            for(int stageloop = split_stage; stageloop< end_stage && queuecount>0; stageloop++)
            {
                lclcount[0]=0;
                barrier(CLK_LOCAL_MEM_FENCE);

                int2 stageinfo = *(global int2*)(stagecascadeptr+stageloop);
                float stagethreshold = as_float(stageinfo.y);

                int perfscale = queuecount > 4 ? 3 : 2;
                int queuecount_loop = (queuecount + (1<<perfscale)-1) >> perfscale;
                int lcl_compute_win = lcl_sz >> perfscale;
                int lcl_compute_win_id = (lcl_id >>(6-perfscale));
                int lcl_loops = (stageinfo.x + lcl_compute_win -1) >> (6-perfscale);
                int lcl_compute_id = lcl_id - (lcl_compute_win_id << (6-perfscale));
                for(int queueloop=0; queueloop<queuecount_loop; queueloop++)
                {
                    float stage_sum = 0.f;
                    int temp_coord = lcloutindex[lcl_compute_win_id<<1];
                    float variance_norm_factor = as_float(lcloutindex[(lcl_compute_win_id<<1)+1]);
                    int queue_pixel = mad24(((temp_coord  & (int)0xffff0000)>>16),readwidth,temp_coord & 0xffff);

                    if(lcl_compute_win_id < queuecount)
                    {
                        int tempnodecounter = lcl_compute_id;
                        float part_sum = 0.f;
                        const int stump_factor = STUMP_BASED ? 1 : 2;
                        int root_offset = 0;
                        for(int lcl_loop=0; lcl_loop<lcl_loops && tempnodecounter<stageinfo.x;)
                        {
                            __global GpuHidHaarTreeNode* currentnodeptr =
                                nodeptr + (nodecounter + tempnodecounter) * stump_factor + root_offset;

                            int4 info1 = *(__global int4*)(&(currentnodeptr->p[0][0]));
                            int4 info2 = *(__global int4*)(&(currentnodeptr->p[1][0]));
                            int4 info3 = *(__global int4*)(&(currentnodeptr->p[2][0]));
                            float4 w = *(__global float4*)(&(currentnodeptr->weight[0]));
                            float3 alpha3 = *(__global float3*)(&(currentnodeptr->alpha[0]));
                            float nodethreshold  = w.w * variance_norm_factor;

                            info1.x +=queue_pixel;
                            info1.z +=queue_pixel;
                            info2.x +=queue_pixel;
                            info2.z +=queue_pixel;

                            float classsum = (lcldata[mad24(info1.y,readwidth,info1.x)] - lcldata[mad24(info1.y,readwidth,info1.z)] -
                                                lcldata[mad24(info1.w,readwidth,info1.x)] + lcldata[mad24(info1.w,readwidth,info1.z)]) * w.x;


                            classsum += (lcldata[mad24(info2.y,readwidth,info2.x)] - lcldata[mad24(info2.y,readwidth,info2.z)] -
                                            lcldata[mad24(info2.w,readwidth,info2.x)] + lcldata[mad24(info2.w,readwidth,info2.z)]) * w.y;

                            info3.x +=queue_pixel;
                            info3.z +=queue_pixel;
                            classsum += (lcldata[mad24(info3.y,readwidth,info3.x)] - lcldata[mad24(info3.y,readwidth,info3.z)] -
                                            lcldata[mad24(info3.w,readwidth,info3.x)] + lcldata[mad24(info3.w,readwidth,info3.z)]) * w.z;

                            bool passThres = classsum >= nodethreshold;
#if STUMP_BASED
                            part_sum += passThres ? alpha3.y : alpha3.x;
                            tempnodecounter += lcl_compute_win;
                            lcl_loop++;
#else
                            if(root_offset == 0)
                            {
                                if( (passThres && currentnodeptr->right) ||
                                    (!passThres && currentnodeptr->left))
                                {
                                    root_offset = 1;
                                }
                                else
                                {
                                    part_sum += alpha3.x;
                                    tempnodecounter += lcl_compute_win;
                                    lcl_loop++;
                                }
                            }
                            else
                            {
                                part_sum += passThres ? alpha3.z : alpha3.y;
                                tempnodecounter += lcl_compute_win;
                                lcl_loop++;
                                root_offset = 0;
                            }
#endif
                        }//end for(int lcl_loop=0;lcl_loop<lcl_loops;lcl_loop++)
                        partialsum[lcl_id]=part_sum;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if(lcl_compute_win_id < queuecount)
                    {
                        for(int i=0; i<lcl_compute_win && (lcl_compute_id==0); i++)
                        {
                            stage_sum += partialsum[lcl_id+i];
                        }
                        if(stage_sum >= stagethreshold && (lcl_compute_id==0))
                        {
                            int queueindex = atomic_inc(lclcount);
                            lcloutindex[queueindex<<1] = temp_coord;
                            lcloutindex[(queueindex<<1)+1] = as_int(variance_norm_factor);
                        }
                        lcl_compute_win_id +=(1<<perfscale);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }//end for(int queueloop=0;queueloop<queuecount_loop;queueloop++)

                queuecount = lclcount[0];
                barrier(CLK_LOCAL_MEM_FENCE);
                nodecounter += stageinfo.x;
            }//end for(int stageloop = splitstage; stageloop< endstage && queuecount>0;stageloop++)

            if(lcl_id<queuecount)
            {
                int temp = lcloutindex[lcl_id<<1];
                int x = mad24(grpidx,grpszx,temp & 0xffff);
                int y = mad24(grpidy,grpszy,((temp & (int)0xffff0000) >> 16));
                temp = glboutindex[0];
                int4 candidate_result;
                candidate_result.zw = (int2)convert_int_rtn(factor*20.f);
                candidate_result.x = convert_int_rtn(x*factor);
                candidate_result.y = convert_int_rtn(y*factor);
                atomic_inc(glboutindex);
                candidate[outputoff+temp+lcl_id] = candidate_result;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }//end for(int grploop=grpidx;grploop<totalgrp;grploop+=grpnumx)
    }//end for(int scalei = 0; scalei <loopcount; scalei++)
}
