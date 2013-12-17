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
//    Erping Pang, erping@multicorewareinc.com
//    Vadim Pisarevsky, vadim.pisarevsky@itseez.com
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

typedef struct __attribute__((aligned(4))) OptFeature
{
    int4 ofs[3] __attribute__((aligned (4)));
    float4 weight __attribute__((aligned (4)));
}
OptFeature;

typedef struct __attribute__((aligned(4))) DTreeNode
{
    int featureIdx __attribute__((aligned (4)));
    float threshold __attribute__((aligned (4))); // for ordered features only
    int left __attribute__((aligned (4)));
    int right __attribute__((aligned (4)));
}
DTreeNode;

typedef struct __attribute__((aligned (4))) DTree
{
    int nodeCount __attribute__((aligned (4)));
}
DTree;

typedef struct __attribute__((aligned (4))) Stage
{
    int first __attribute__((aligned (4)));
    int ntrees __attribute__((aligned (4)));
    float threshold __attribute__((aligned (4)));
}
Stage;

__kernel void runHaarClassifierStump(
    __global const int* sum,
    int sumstep, int sumoffset,
    __global const int* sqsum,
    int sqsumstep, int sqsumoffset,
    __global const OptFeature* optfeatures,

    int nstages,
    __global const Stage* stages,
    __global const DTree* trees,
    __global const DTreeNode* nodes,
    __global const float* leaves,

    volatile __global int* facepos,
    int2 imgsize, int xyscale, float factor,
    int4 normrect, int2 windowsize)
{
    int ix = get_global_id(0)*xyscale;
    int iy = get_global_id(1)*xyscale;
    sumstep /= sizeof(int);
    sqsumstep /= sizeof(int);
    
    if( ix < imgsize.x && iy < imgsize.y )
    {
        int ntrees, nodeOfs = 0, leafOfs = 0;
        int stageIdx, i;
        float s = 0.f;
        __global const DTreeNode* node;
        __global const OptFeature* f;
        
        __global const int* psum = sum + mad24(iy, sumstep, ix);
        __global const int* pnsum = psum + mad24(normrect.y, sumstep, normrect.x);
        int normarea = normrect.z * normrect.w;
        float invarea = 1.f/normarea;
        float sval = (pnsum[0] - pnsum[normrect.z] - pnsum[mul24(normrect.w, sumstep)] +
                      pnsum[mad24(normrect.w, sumstep, normrect.z)])*invarea;
        float sqval = (sqsum[mad24(iy + normrect.y, sqsumstep, ix + normrect.x)])*invarea;
        float nf = (float)normarea * sqrt(max(sqval - sval * sval, 0.f));
        float4 weight;
        int4 ofs;
        nf = nf > 0 ? nf : 1.f;
        
        for( stageIdx = 0; stageIdx < nstages; stageIdx++ )
        {
            ntrees = stages[stageIdx].ntrees;
            s = 0.f;
            for( i = 0; i < ntrees; i++, nodeOfs++, leafOfs += 2 )
            {
                node = nodes + nodeOfs;
                f = optfeatures + node->featureIdx;
                
                weight = f->weight;
                
                ofs = f->ofs[0];
                sval = (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.x;
                ofs = f->ofs[1];
                sval += (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.y;
                if( weight.z > 0 )
                {
                    ofs = f->ofs[2];
                    sval += (psum[ofs.x] - psum[ofs.y] - psum[ofs.z] + psum[ofs.w])*weight.z;
                }
                s += leaves[ sval < node->threshold*nf ? leafOfs : leafOfs + 1 ];
            }
            
            if( s < stages[stageIdx].threshold )
                break;
        }
        
        if( stageIdx == nstages )
        {
            int nfaces = atomic_inc(facepos);
            //printf("detected face #d!!!!\n", nfaces);
            if( nfaces < MAX_FACES )
            {
                volatile __global int* face = facepos + 1 + nfaces*4;
                face[0] = convert_int_rte(ix*factor);
                face[1] = convert_int_rte(iy*factor);
                face[2] = convert_int_rte(windowsize.x*factor);
                face[3] = convert_int_rte(windowsize.y*factor);
            }
        }
    }
}
