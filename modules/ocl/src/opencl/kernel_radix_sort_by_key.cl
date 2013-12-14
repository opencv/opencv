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

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifndef N   // number of radices
#define N 4
#endif

#ifndef K_T
#define K_T float
#endif

#ifndef V_T
#define V_T float
#endif

#ifndef IS_GT
#define IS_GT 0
#endif


// from Thrust::b40c, link:
// https://github.com/thrust/thrust/blob/master/thrust/system/cuda/detail/detail/b40c/radixsort_key_conversion.h
__inline uint convertKey(uint converted_key)
{
#ifdef K_FLT
    unsigned int mask = (converted_key & 0x80000000) ? 0xffffffff : 0x80000000;
    converted_key ^= mask;
#elif defined(K_INT)
    const uint SIGN_MASK = 1u << ((sizeof(int) * 8) - 1);
    converted_key ^= SIGN_MASK;
#else

#endif
    return converted_key;
}

//FIXME(pengx17):
// exclusive scan, need to be optimized as this is too naive...
kernel
    void naiveScanAddition(
    __global int * input,
    __global int * output,
    int size
    )
{
    if(get_global_id(0) == 0)
    {
        output[0] = 0;
        for(int i = 1; i < size; i ++)
        {
            output[i] = output[i - 1] + input[i - 1];
        }
    }
}

// following is ported from
// https://github.com/HSA-Libraries/Bolt/blob/master/include/bolt/cl/sort_uint_kernels.cl
kernel
    void histogramRadixN (
    __global K_T* unsortedKeys,
    __global int * buckets,
    uint shiftCount
    )
{
    const int RADIX_T     = N;
    const int RADICES_T   = (1 << RADIX_T);
    const int NUM_OF_ELEMENTS_PER_WORK_ITEM_T = RADICES_T;
    const int MASK_T      = (1 << RADIX_T) - 1;
    int localBuckets[16] = {0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0};
    int globalId    = get_global_id(0);
    int numOfGroups = get_num_groups(0);

    /* Calculate thread-histograms */
    for(int i = 0; i < NUM_OF_ELEMENTS_PER_WORK_ITEM_T; ++i)
    {
        uint value = convertKey(as_uint(unsortedKeys[mad24(globalId, NUM_OF_ELEMENTS_PER_WORK_ITEM_T, i)]));
        value = (value >> shiftCount) & MASK_T;
#if IS_GT
        localBuckets[RADICES_T - value - 1]++;
#else
        localBuckets[value]++;
#endif
    }

    for(int i = 0; i < NUM_OF_ELEMENTS_PER_WORK_ITEM_T; ++i)
    {
        buckets[mad24(i, RADICES_T * numOfGroups, globalId) ] = localBuckets[i];
    }
}

kernel
    void permuteRadixN (
    __global K_T*  unsortedKeys,
    __global V_T*  unsortedVals,
    __global int* scanedBuckets,
    uint shiftCount,
    __global K_T*  sortedKeys,
    __global V_T*  sortedVals
    )
{
    const int RADIX_T     = N;
    const int RADICES_T   = (1 << RADIX_T);
    const int MASK_T = (1<<RADIX_T)  -1;

    int globalId  = get_global_id(0);
    int numOfGroups = get_num_groups(0);
    const int NUM_OF_ELEMENTS_PER_WORK_GROUP_T = numOfGroups << N;
    int  localIndex[16];

    /*Load the index to local memory*/
    for(int i = 0; i < RADICES_T; ++i)
    {
#if IS_GT
        localIndex[i] = scanedBuckets[mad24(RADICES_T - i - 1, NUM_OF_ELEMENTS_PER_WORK_GROUP_T, globalId)];
#else
        localIndex[i] = scanedBuckets[mad24(i, NUM_OF_ELEMENTS_PER_WORK_GROUP_T, globalId)];
#endif
    }
    /* Permute elements to appropriate location */
    for(int i = 0; i < RADICES_T; ++i)
    {
        int old_idx = mad24(globalId, RADICES_T, i);
        K_T  ovalue = unsortedKeys[old_idx];
        uint value = convertKey(as_uint(ovalue));
        uint maskedValue = (value >> shiftCount) & MASK_T;
        uint index = localIndex[maskedValue];
        sortedKeys[index] = ovalue;
        sortedVals[index] = unsortedVals[old_idx];
        localIndex[maskedValue] = index + 1;
    }
}
