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

#ifndef WITH_MASK
#define WITH_MASK 0
#endif

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline float ELEM_INT2(image2d_t _eig, int _x, int _y)
{
    return read_imagef(_eig, sampler, (int2)(_x, _y)).x;
}

inline float ELEM_FLT2(image2d_t _eig, float2 pt)
{
    return read_imagef(_eig, sampler, pt).x;
}

__kernel
    void findCorners
    (
        image2d_t eig,
        __global const char * mask,
        __global float2 * corners,
        const int mask_strip,// in pixels
        const float threshold,
        const int rows,
        const int cols,
        const int max_count,
        __global int * g_counter
    )
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1
#if WITH_MASK
        && mask[i * mask_strip + j] != 0
#endif
        )
    {
        const float val = ELEM_INT2(eig, j, i);

        if (val > threshold)
        {
            float maxVal = val;

            maxVal = fmax(ELEM_INT2(eig, j - 1, i - 1), maxVal);
            maxVal = fmax(ELEM_INT2(eig, j    , i - 1), maxVal);
            maxVal = fmax(ELEM_INT2(eig, j + 1, i - 1), maxVal);

            maxVal = fmax(ELEM_INT2(eig, j - 1, i), maxVal);
            maxVal = fmax(ELEM_INT2(eig, j + 1, i), maxVal);

            maxVal = fmax(ELEM_INT2(eig, j - 1, i + 1), maxVal);
            maxVal = fmax(ELEM_INT2(eig, j    , i + 1), maxVal);
            maxVal = fmax(ELEM_INT2(eig, j + 1, i + 1), maxVal);

            if (val == maxVal)
            {
                const int ind = atomic_inc(g_counter);

                if (ind < max_count)
                    corners[ind] = (float2)(j, i);
            }
        }
    }
}

//bitonic sort
__kernel
    void sortCorners_bitonicSort
    (
        image2d_t eig,
        __global float2 * corners,
        const int count,
        const int stage,
        const int passOfStage
    )
{
    const int threadId = get_global_id(0);
    if(threadId >= count / 2)
    {
        return;
    }

    const int sortOrder = (((threadId/(1 << stage)) % 2)) == 1 ? 1 : 0; // 0 is descent

    const int pairDistance = 1 << (stage - passOfStage);
    const int blockWidth   = 2 * pairDistance;

    const int leftId = min( (threadId % pairDistance)
                   + (threadId / pairDistance) * blockWidth, count );

    const int rightId = min( leftId + pairDistance, count );

    const float2 leftPt  = corners[leftId];
    const float2 rightPt = corners[rightId];

    const float leftVal  = ELEM_FLT2(eig, leftPt);
    const float rightVal = ELEM_FLT2(eig, rightPt);

    const bool compareResult = leftVal > rightVal;

    float2 greater = compareResult ? leftPt:rightPt;
    float2 lesser  = compareResult ? rightPt:leftPt;

    corners[leftId]  = sortOrder ? lesser : greater;
    corners[rightId] = sortOrder ? greater : lesser;
}

//selection sort for gfft
//kernel is ported from Bolt library:
//https://github.com/HSA-Libraries/Bolt/blob/master/include/bolt/cl/sort_kernels.cl
//  Local sort will firstly sort elements of each workgroup using selection sort
//  its performance is O(n)
__kernel
    void sortCorners_selectionSortLocal
    (
        image2d_t eig,
        __global float2 * corners,
        const int count,
        __local float2 * scratch
    )
{
    int          i  = get_local_id(0); // index in workgroup
    int numOfGroups = get_num_groups(0); // index in workgroup
    int groupID     = get_group_id(0);
    int         wg  = get_local_size(0); // workgroup size = block size
    int n; // number of elements to be processed for this work group

    int offset   = groupID * wg;
    int same     = 0;
    corners      += offset;
    n = (groupID == (numOfGroups-1))? (count - wg*(numOfGroups-1)) : wg;
    float2 pt1, pt2;

    pt1 = corners[min(i, n)];
    scratch[i] = pt1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(i >= n)
    {
        return;
    }

    float val1 = ELEM_FLT2(eig, pt1);
    float val2;

    int pos = 0;
    for (int j=0;j<n;++j)
    {
        pt2  = scratch[j];
        val2 = ELEM_FLT2(eig, pt2);
        if(val2 > val1)
            pos++;//calculate the rank of this element in this work group
        else
        {
            if(val1 > val2)
                continue;
            else
            {
                // val1 and val2 are same
                same++;
            }
        }
    }
    for (int j=0; j< same; j++)
        corners[pos + j] = pt1;
}
__kernel
    void sortCorners_selectionSortFinal
    (
        image2d_t eig,
        __global float2 * corners,
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
    float2 pt1, pt2;
    pt1 = corners[groupID*wg + i];

    float val1 = ELEM_FLT2(eig, pt1);
    float val2;

    for(int j=0; j<numOfGroups-1; j++ )
    {
        for(int k=0; k<wg; k++)
        {
            pt2  = corners[j*wg + k];
            val2 = ELEM_FLT2(eig, pt2);
            if(val1 > val2)
                break;
            else
            {
                //Increment only if the value is not the same.
                if( val2 > val1 )
                    pos++;
                else
                    same++;
            }
        }
    }

    for(int k=0; k<remainder; k++)
    {
        pt2  = corners[(numOfGroups-1)*wg + k];
        val2 = ELEM_FLT2(eig, pt2);
        if(val1 > val2)
            break;
        else
        {
            //Don't increment if the value is the same.
            //Two elements are same if (*userComp)(jData, iData)  and (*userComp)(iData, jData) are both false
            if(val2 > val1)
                pos++;
            else
                same++;
        }
    }
    for (int j=0; j< same; j++)
        corners[pos + j] = pt1;
}
