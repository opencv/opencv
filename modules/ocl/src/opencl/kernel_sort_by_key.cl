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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@outlook.com
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

#ifndef K_T
#define K_T float
#endif

#ifndef V_T
#define V_T float
#endif

#ifndef IS_GT
#define IS_GT false
#endif

#if IS_GT
#define my_comp(x,y) ((x) > (y))
#else
#define my_comp(x,y) ((x) < (y))
#endif

/////////////////////// Bitonic sort ////////////////////////////
// ported from 
// https://github.com/HSA-Libraries/Bolt/blob/master/include/bolt/cl/sort_by_key_kernels.cl
__kernel
    void bitonicSort
    (
        __global K_T * keys,
        __global V_T * vals,
        int count,
        int stage,
        int passOfStage
    )
{
    const int threadId = get_global_id(0);
    if(threadId >= count / 2)
    {
        return;
    }
    const int pairDistance = 1 << (stage - passOfStage);
    const int blockWidth   = 2 * pairDistance;

    int leftId = min( (threadId % pairDistance) 
                   + (threadId / pairDistance) * blockWidth, count );

    int rightId = min( leftId + pairDistance, count );

    int temp;

    const V_T lval = vals[leftId];
    const V_T rval = vals[rightId]; 

    const K_T lkey = keys[leftId];
    const K_T rkey = keys[rightId];

    int sameDirectionBlockWidth = 1 << stage;

    if((threadId/sameDirectionBlockWidth) % 2 == 1)
    {
        temp = rightId;
        rightId = leftId;
        leftId = temp;
    }

    const bool compareResult = my_comp(lkey, rkey);

    if(compareResult)
    {
        keys[rightId] = rkey;
        keys[leftId]  = lkey;
        vals[rightId] = rval;
        vals[leftId]  = lval;
    }
    else
    {
        keys[rightId] = lkey;
        keys[leftId]  = rkey;
        vals[rightId] = lval;
        vals[leftId]  = rval;
    }
}

/////////////////////// Selection sort ////////////////////////////
//kernel is ported from Bolt library:
//https://github.com/HSA-Libraries/Bolt/blob/master/include/bolt/cl/sort_kernels.cl
__kernel
    void selectionSortLocal
    (
        __global K_T * keys,
        __global V_T * vals,
        const int count,
        __local  K_T * scratch
    )
{
    int          i  = get_local_id(0); // index in workgroup
    int numOfGroups = get_num_groups(0); // index in workgroup
    int groupID     = get_group_id(0);
    int         wg  = get_local_size(0); // workgroup size = block size
    int n; // number of elements to be processed for this work group

    int offset   = groupID * wg;
    int same     = 0;
    
    vals      += offset;
    keys      += offset;
    n = (groupID == (numOfGroups-1))? (count - wg*(numOfGroups-1)) : wg;

    int clamped_i= min(i, n - 1);

    K_T key1 = keys[clamped_i], key2;
    V_T val1 = vals[clamped_i];
    scratch[i] = key1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(i >= n)
    {
        return;
    }

    int pos = 0;
    for (int j=0;j<n;++j)
    {
        key2  = scratch[j];
        if(my_comp(key2, key1)) 
            pos++;//calculate the rank of this element in this work group
        else 
        {
            if(my_comp(key1, key2))
                continue;
            else 
            {
                // key1 and key2 are same
                same++;
            }
        }
    }
    for (int j=0; j< same; j++)
    {
        vals[pos + j] = val1;
        keys[pos + j] = key1;
    }
}
__kernel
    void selectionSortFinal
    (
        __global K_T * keys,
        __global V_T * vals,
        const int count
    )
{
    const int          i  = get_local_id(0); // index in workgroup
    const int numOfGroups = get_num_groups(0); // index in workgroup
    const int groupID     = get_group_id(0);
    const int         wg  = get_local_size(0); // workgroup size = block size
    int pos = 0, same = 0;
    const int offset = get_group_id(0) * wg;
    const int remainder = count - wg*(numOfGroups-1);

    if((offset + i ) >= count)
        return;
    V_T val1 = vals[offset + i];

    K_T key1 = keys[offset + i];
    K_T key2;

    for(int j=0; j<numOfGroups-1; j++ )
    {
        for(int k=0; k<wg; k++)
        {
            key2 = keys[j*wg + k]; 
            if(my_comp(key1, key2))
                break;
            else
            {
                //Increment only if the value is not the same. 
                if(my_comp(key2, key1))
                    pos++;
                else 
                    same++;
            }
        }
    }

    for(int k=0; k<remainder; k++)
    {
        key2 = keys[(numOfGroups-1)*wg + k]; 
        if(my_comp(key1, key2))
            break;
        else
        {
            //Don't increment if the value is the same. 
            if(my_comp(key2, key1))
                pos++;
            else 
                same++;
        }
    }  
    for (int j=0; j< same; j++)
    {
        vals[pos + j] = val1;
        keys[pos + j] = key1;
    }
}
