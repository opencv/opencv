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

//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is
//  passed as a functor parameter my_comp
//  This function returns an index that is the first index whos value would be equal to the searched value
inline uint lowerBoundBinary( global K_T* data, uint left, uint right, K_T searchVal)
{
    //  The values firstIndex and lastIndex get modified within the loop, narrowing down the potential sequence
    uint firstIndex = left;
    uint lastIndex = right;

    //  This loops through [firstIndex, lastIndex)
    //  Since firstIndex and lastIndex will be different for every thread depending on the nested branch,
    //  this while loop will be divergent within a wavefront
    while( firstIndex < lastIndex )
    {
        //  midIndex is the average of first and last, rounded down
        uint midIndex = ( firstIndex + lastIndex ) / 2;
        K_T midValue = data[ midIndex ];

        //  This branch will create divergent wavefronts
        if( my_comp( midValue, searchVal ) )
        {
            firstIndex = midIndex+1;
            // printf( "lowerBound: lastIndex[ %i ]=%i\n", get_local_id( 0 ), lastIndex );
        }
        else
        {
            lastIndex = midIndex;
            // printf( "lowerBound: firstIndex[ %i ]=%i\n", get_local_id( 0 ), firstIndex );
        }
    }

    return firstIndex;
}

//  This implements a binary search routine to look for an 'insertion point' in a sequence, denoted
//  by a base pointer and left and right index for a particular candidate value.  The comparison operator is
//  passed as a functor parameter my_comp
//  This function returns an index that is the first index whos value would be greater than the searched value
//  If the search value is not found in the sequence, upperbound returns the same result as lowerbound
inline uint upperBoundBinary( global K_T* data, uint left, uint right, K_T searchVal)
{
    uint upperBound = lowerBoundBinary( data, left, right, searchVal );

    // printf( "upperBoundBinary: upperBound[ %i, %i ]= %i\n", left, right, upperBound );
    //  If upperBound == right, then  searchVal was not found in the sequence.  Just return.
    if( upperBound != right )
    {
        //  While the values are equal i.e. !(x < y) && !(y < x) increment the index
        K_T upperValue = data[ upperBound ];
        while( !my_comp( upperValue, searchVal ) && !my_comp( searchVal, upperValue) && (upperBound != right) )
        {
            upperBound++;
            upperValue = data[ upperBound ];
        }
    }

    return upperBound;
}

//  This kernel implements merging of blocks of sorted data.  The input to this kernel most likely is
//  the output of blockInsertionSortTemplate.  It is expected that the source array contains multiple
//  blocks, each block is independently sorted.  The goal is to write into the output buffer half as
//  many blocks, of double the size.  The even and odd blocks are stably merged together to form
//  a new sorted block of twice the size.  The algorithm is out-of-place.
kernel void merge(
    global K_T*   iKey_ptr,
    global V_T*   iValue_ptr,
    global K_T*   oKey_ptr,
    global V_T*   oValue_ptr,
    const uint    srcVecSize,
    const uint    srcLogicalBlockSize,
    local K_T*    key_lds,
    local V_T*    val_lds
)
{
    size_t globalID     = get_global_id( 0 );

    //  Abort threads that are passed the end of the input vector
    if( globalID >= srcVecSize )
        return; // on SI this doesn't mess-up barriers

    //  For an element in sequence A, find the lowerbound index for it in sequence B
    uint srcBlockNum   = globalID / srcLogicalBlockSize;
    uint srcBlockIndex = globalID % srcLogicalBlockSize;

    // printf( "mergeTemplate: srcBlockNum[%i]=%i\n", srcBlockNum, srcBlockIndex );

    //  Pairs of even-odd blocks will be merged together
    //  An even block should search for an insertion point in the next odd block,
    //  and the odd block should look for an insertion point in the corresponding previous even block
    uint dstLogicalBlockSize = srcLogicalBlockSize<<1;
    uint leftBlockIndex = globalID & ~((dstLogicalBlockSize) - 1 );
    leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize;
    leftBlockIndex = min( leftBlockIndex, srcVecSize );
    uint rightBlockIndex = min( leftBlockIndex + srcLogicalBlockSize, srcVecSize );

    // if( localID == 0 )
    // {
    // printf( "mergeTemplate: wavefront[ %i ] logicalBlock[ %i ] logicalIndex[ %i ] leftBlockIndex[ %i ] <=> rightBlockIndex[ %i ]\n", groupID, srcBlockNum, srcBlockIndex, leftBlockIndex, rightBlockIndex );
    // }

    //  For a particular element in the input array, find the lowerbound index for it in the search sequence given by leftBlockIndex & rightBlockIndex
    // uint insertionIndex = lowerBoundLinear( iKey_ptr, leftBlockIndex, rightBlockIndex, iKey_ptr[ globalID ], my_comp ) - leftBlockIndex;
    uint insertionIndex = 0;
    if( (srcBlockNum & 0x1) == 0 )
    {
        insertionIndex = lowerBoundBinary( iKey_ptr, leftBlockIndex, rightBlockIndex, iKey_ptr[ globalID ] ) - leftBlockIndex;
    }
    else
    {
        insertionIndex = upperBoundBinary( iKey_ptr, leftBlockIndex, rightBlockIndex, iKey_ptr[ globalID ] ) - leftBlockIndex;
    }

    //  The index of an element in the result sequence is the summation of it's indixes in the two input
    //  sequences
    uint dstBlockIndex = srcBlockIndex + insertionIndex;
    uint dstBlockNum = srcBlockNum/2;

    // if( (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex == 395 )
    // {
    // printf( "mergeTemplate: (dstBlockNum[ %i ] * dstLogicalBlockSize[ %i ]) + dstBlockIndex[ %i ] = srcBlockIndex[ %i ] + insertionIndex[ %i ]\n", dstBlockNum, dstLogicalBlockSize, dstBlockIndex, srcBlockIndex, insertionIndex );
    // printf( "mergeTemplate: dstBlockIndex[ %i ] = iKey_ptr[ %i ] ( %i )\n", (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex, globalID, iKey_ptr[ globalID ] );
    // }
    oKey_ptr[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = iKey_ptr[ globalID ];
    oValue_ptr[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = iValue_ptr[ globalID ];
    // printf( "mergeTemplate: leftResultIndex[ %i ]=%i + %i\n", leftResultIndex, srcBlockIndex, leftInsertionIndex );
}

kernel void blockInsertionSort(
    global K_T*   key_ptr,
    global V_T*   value_ptr,
    const uint    vecSize,
    local K_T*    key_lds,
    local V_T*    val_lds
)
{
    int gloId    = get_global_id( 0 );
    int groId    = get_group_id( 0 );
    int locId    = get_local_id( 0 );
    int wgSize   = get_local_size( 0 );

    bool in_range = gloId < (int)vecSize;
    K_T key;
    V_T val;
    //  Abort threads that are passed the end of the input vector
    if (in_range)
    {
        //  Make a copy of the entire input array into fast local memory
        key = key_ptr[ gloId ];
        val = value_ptr[ gloId ];
        key_lds[ locId ] = key;
        val_lds[ locId ] = val;
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    //  Sorts a workgroup using a naive insertion sort
    //  The sort uses one thread within a workgroup to sort the entire workgroup
    if( locId == 0 && in_range )
    {
        //  The last workgroup may have an irregular size, so we calculate a per-block endIndex
        //  endIndex is essentially emulating a mod operator with subtraction and multiply
        int endIndex = vecSize - ( groId * wgSize );
        endIndex = min( endIndex, wgSize );

        // printf( "Debug: endIndex[%i]=%i\n", groId, endIndex );

        //  Indices are signed because the while loop will generate a -1 index inside of the max function
        for( int currIndex = 1; currIndex < endIndex; ++currIndex )
        {
            key = key_lds[ currIndex ];
            val = val_lds[ currIndex ];
            int scanIndex = currIndex;
            K_T ldsKey = key_lds[scanIndex - 1];
            while( scanIndex > 0 && my_comp( key, ldsKey ) )
            {
                V_T ldsVal = val_lds[scanIndex - 1];

                //  If the keys are being swapped, make sure the values are swapped identicaly
                key_lds[ scanIndex ] = ldsKey;
                val_lds[ scanIndex ] = ldsVal;

                scanIndex = scanIndex - 1;
                ldsKey = key_lds[ max( 0, scanIndex - 1 ) ];  // scanIndex-1 may be -1
            }
            key_lds[ scanIndex ] = key;
            val_lds[ scanIndex ] = val;
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    if(in_range)
    {
        key = key_lds[ locId ];
        key_ptr[ gloId ] = key;

        val = val_lds[ locId ];
        value_ptr[ gloId ] = val;
    }
}

///////////// Radix sort from b40c library /////////////
