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
//macro to read eigenvalue matrix
#define GET_SRC_32F(_x, _y) ((__global const float*)(eig + (_y)*eig_pitch))[_x]

__kernel
    void findCorners
    (
        __global const char*    eig,
        const int               eig_pitch,
        __global const char*    mask,
        __global float2*        corners,
        const int               mask_strip,// in pixels
        __global const float*   pMinMax,
        const float             qualityLevel,
        const int               rows,
        const int               cols,
        const int               max_count,
        __global int*           g_counter
    )
{
    float threshold = qualityLevel*pMinMax[1];
    const int j = get_global_id(0);
    const int i = get_global_id(1);

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1
#if WITH_MASK
        && mask[i * mask_strip + j] != 0
#endif
        )
    {
        const float val = GET_SRC_32F(j, i);

        if (val > threshold)
        {
            float maxVal = val;
            maxVal = fmax(GET_SRC_32F(j - 1, i - 1), maxVal);
            maxVal = fmax(GET_SRC_32F(j    , i - 1), maxVal);
            maxVal = fmax(GET_SRC_32F(j + 1, i - 1), maxVal);

            maxVal = fmax(GET_SRC_32F(j - 1, i), maxVal);
            maxVal = fmax(GET_SRC_32F(j + 1, i), maxVal);

            maxVal = fmax(GET_SRC_32F(j - 1, i + 1), maxVal);
            maxVal = fmax(GET_SRC_32F(j    , i + 1), maxVal);
            maxVal = fmax(GET_SRC_32F(j + 1, i + 1), maxVal);

            if (val == maxVal)
            {
                const int ind = atomic_inc(g_counter);

                if (ind < max_count)
                {// pack and store eigenvalue and its coordinates
                    corners[ind].x = val;
                    corners[ind].y = as_float(j|(i<<16));
                }
            }
        }
    }
}
#undef GET_SRC_32F


//bitonic sort
__kernel
    void sortCorners_bitonicSort
    (
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

    const float leftVal  = leftPt.x;
    const float rightVal = rightPt.x;

    const bool compareResult = leftVal > rightVal;

    float2 greater = compareResult ? leftPt:rightPt;
    float2 lesser  = compareResult ? rightPt:leftPt;

    corners[leftId]  = sortOrder ? lesser : greater;
    corners[rightId] = sortOrder ? greater : lesser;
}

// this is simple short serial kernel that makes some short reduction and initialization work
// it makes HOST like work to avoid additional sync with HOST to do this short work
// data - input/output float2.
//      input data are sevral (min,max) pairs
//      output data is one reduced (min,max) pair
// g_counter - counter that have to be initialized by 0 for next findCorner call.
__kernel void arithm_op_minMax_final(__global float * data, int groupnum,__global int * g_counter)
{
    g_counter[0] = 0;
    float minVal = data[0];
    float maxVal = data[groupnum];
    for(int i=1;i<groupnum;++i)
    {
        minVal = min(minVal,data[i]);
        maxVal = max(maxVal,data[i+groupnum]);
    }
    data[0] = minVal;
    data[1] = maxVal;
}