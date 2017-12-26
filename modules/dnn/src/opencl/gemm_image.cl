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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)

// Types used for parameters, offset computations and so on
#define int_tp int
#define uint_tp unsigned int

#define Dtype  float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8

#define as_Dtype  as_float
#define as_Dtype2 as_float2
#define as_Dtype4 as_float4
#define as_Dtype8 as_float8

#define KERNEL_ARG_DTYPE float

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#define TILE_M          32
#define TILE_K          8

// common block to calculate (alpha * AxB + beta * C) and output to destination image.

#define SUBGROUP_BLOCK_READ8( __image, __coord ) intel_sub_group_block_read8( __image, __coord )
#define SHUFFLE_TYPE2(val) val
#define SHUFFLE_TYPE8(val) val
#define READ_IMAGE(__image, __coord) read_imagef(__image, sampler, __coord)
#define SIZE_OF_ELEMENT sizeof(uint)
#define SIMD_SIZE_GEMM 8
#define TILE_N 8

//#define USE_IMAGE_C
#ifdef USE_IMAGE_C
#define BLOCKC_READ8( _C, _coordC ) as_Dtype8( intel_sub_group_block_read8( _C, _coordC ) )
#define BLOCKC_WRITE8( _C, _coordC, _val ) intel_sub_group_block_write8( _C, _coordC, as_uint8( _val ) )
#define MATC_PARAMETER __read_only image2d_t C, __write_only image2d_t dst
#define GEMM_OUTPUT(ALPHA1, BETA_NOT0) GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, C, dst, sizeof(uint))
#else
#define BLOCKC_READ8( _C, _coordC ) \
          (Dtype8) ( (_coordC.x + get_local_id(0) < N && _coordC.y < M) ? _C[ _coordC.y * ldc + _coordC.x + get_local_id(0) ] : 0, \
                     (_coordC.x + get_local_id(0) < N && _coordC.y + 1 < M) ? _C[ ( _coordC.y + 1 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \
                     (_coordC.x + get_local_id(0) < N && _coordC.y + 2 < M) ? _C[ ( _coordC.y + 2 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \
                     (_coordC.x + get_local_id(0) < N && _coordC.y + 3 < M) ? _C[ ( _coordC.y + 3 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \
                     (_coordC.x + get_local_id(0) < N && _coordC.y + 4 < M) ? _C[ ( _coordC.y + 4 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \
                     (_coordC.x + get_local_id(0) < N && _coordC.y + 5 < M) ? _C[ ( _coordC.y + 5 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \
                     (_coordC.x + get_local_id(0) < N && _coordC.y + 6 < M) ? _C[ ( _coordC.y + 6 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \
                     (_coordC.x + get_local_id(0) < N && _coordC.y + 7 < M) ? _C[ ( _coordC.y + 7 ) * ldc + _coordC.x + get_local_id(0) ] : 0)

#define BLOCKC_WRITE8( _C, _coordC, _val) do {\
                     if (_coordC.x + get_local_id(0) < N) { \
                       if (_coordC.y < M) \
                         _C[ _coordC.y * ldc + _coordC.x + get_local_id(0) ] = _val.s0; \
                       if (_coordC.y + 1 < M) \
                         _C[ ( _coordC.y + 1 )* ldc + _coordC.x + get_local_id(0) ] = _val.s1; \
                       if (_coordC.y + 2 < M) \
                         _C[ ( _coordC.y + 2 )* ldc + _coordC.x + get_local_id(0) ] = _val.s2; \
                       if (_coordC.y + 3 < M) \
                         _C[ ( _coordC.y + 3 )* ldc + _coordC.x + get_local_id(0) ] = _val.s3; \
                       if (_coordC.y + 4 < M) \
                         _C[ ( _coordC.y + 4 )* ldc + _coordC.x + get_local_id(0) ] = _val.s4; \
                       if (_coordC.y + 5 < M) \
                         _C[ ( _coordC.y + 5 )* ldc + _coordC.x + get_local_id(0) ] = _val.s5; \
                       if (_coordC.y + 6 < M) \
                         _C[ ( _coordC.y + 6 )* ldc + _coordC.x + get_local_id(0) ] = _val.s6; \
                       if (_coordC.y + 7 < M) \
                         _C[ ( _coordC.y + 7 )* ldc + _coordC.x + get_local_id(0) ] = _val.s7; \
                     }} while(0)
#define MATC_PARAMETER __global Dtype * C, const int offC, const int M, const int N, const int ldc
#define GEMM_OUTPUT(ALPHA1, BETA_NOT0) GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, (C + offC), (C + offC), 1)
#endif

#define GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, _C, _dst, _C_step) \
    int2    coordDst = (int2)( ( group_x * TILE_N ) * _C_step, ( group_y * TILE_M ) ); \
    int2    coordC = coordDst; \
    Dtype8 blockC00; \
    Dtype8 blockC01; \
    Dtype8 blockC02; \
    Dtype8 blockC03; \
    if (BETA_NOT0) { \
        blockC00 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \
        blockC01 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \
        blockC02 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \
        blockC03 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC ); \
        if (!ALPHA1) { \
            blockC00 = mad(blockAxB00, (Dtype8)alpha, blockC00); \
            blockC01 = mad(blockAxB01, (Dtype8)alpha, blockC01); \
            blockC02 = mad(blockAxB02, (Dtype8)alpha, blockC02); \
            blockC03 = mad(blockAxB03, (Dtype8)alpha, blockC03); \
        } else { \
            blockC00 += blockAxB00; \
            blockC01 += blockAxB01; \
            blockC02 += blockAxB02; \
            blockC03 += blockAxB03; \
        } \
    } else { \
        blockC00 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \
        blockC01 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \
        blockC02 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \
        blockC03 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC ); \
        if (!ALPHA1) { \
          blockC00 = mad(blockAxB00, (Dtype8)alpha, blockC00); \
          blockC01 = mad(blockAxB01, (Dtype8)alpha, blockC01); \
          blockC02 = mad(blockAxB02, (Dtype8)alpha, blockC02); \
          blockC03 = mad(blockAxB03, (Dtype8)alpha, blockC03); \
        } else { \
          blockC00 += blockAxB00; \
          blockC01 += blockAxB01; \
          blockC02 += blockAxB02; \
          blockC03 += blockAxB03; \
        } \
    } \
    BLOCKC_WRITE8( _dst, coordDst, blockC00 );    coordDst.y += 8; \
    BLOCKC_WRITE8( _dst, coordDst, blockC01 );    coordDst.y += 8; \
    BLOCKC_WRITE8( _dst, coordDst, blockC02 );    coordDst.y += 8; \
    BLOCKC_WRITE8( _dst, coordDst, blockC03 );

// Get the specified column of the block of the block
#define TRANSPOSE_BLOCK_8( _block, _col )   \
        (Dtype8)( intel_sub_group_shuffle( _block.s0, _col ),   \
                  intel_sub_group_shuffle( _block.s1, _col ),   \
                  intel_sub_group_shuffle( _block.s2, _col ),   \
                  intel_sub_group_shuffle( _block.s3, _col ),   \
                  intel_sub_group_shuffle( _block.s4, _col ),   \
                  intel_sub_group_shuffle( _block.s5, _col ),   \
                  intel_sub_group_shuffle( _block.s6, _col ),   \
                  intel_sub_group_shuffle( _block.s7, _col ) );

// A's column block multiply B 's row block.
#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \
        {   \
            const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );    \
            const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );    \
            const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );    \
            const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );    \
            const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );    \
            const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );    \
            const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );    \
            const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );    \
            _result = mad( (Dtype8)(_blockB.s0), acol0, _result );      \
            _result = mad( (Dtype8)(_blockB.s1), acol1, _result );      \
            _result = mad( (Dtype8)(_blockB.s2), acol2, _result );      \
            _result = mad( (Dtype8)(_blockB.s3), acol3, _result );      \
            _result = mad( (Dtype8)(_blockB.s4), acol4, _result );      \
            _result = mad( (Dtype8)(_blockB.s5), acol5, _result );      \
            _result = mad( (Dtype8)(_blockB.s6), acol6, _result );      \
            _result = mad( (Dtype8)(_blockB.s7), acol7, _result );      \
        }

#define GEMM_NN(ALPHA1, BETA_NOT0) \
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) \
__attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) \
__kernel void TEMPLATE(gemm_32_1_NN_ ##ALPHA1 ##_ ##BETA_NOT0, Dtype)( \
    __read_only image2d_t A, \
    __read_only image2d_t B, \
    MATC_PARAMETER, \
    KERNEL_ARG_DTYPE alpha_in, \
    KERNEL_ARG_DTYPE beta_in, \
    int width0, \
    int isFirstColBlock) \
{ \
    const Dtype alpha = (Dtype)alpha_in; \
    const Dtype beta = (Dtype)beta_in; \
    const int group_x = get_group_id(0); \
    const int group_y = get_group_id(1); \
    Dtype8 blockAxB00 = 0.0f; \
    Dtype8 blockAxB01 = 0.0f; \
    Dtype8 blockAxB02 = 0.0f; \
    Dtype8 blockAxB03 = 0.0f; \
    int2    coordA = (int2)( 0, group_y * TILE_M ); \
    int2    coordB = (int2)( ( group_x * TILE_N ) * SIZE_OF_ELEMENT, 0 ); \
    do \
    {  \
        int2    coordBTemp = coordB; \
        Dtype8  blockB00 = as_Dtype8( SUBGROUP_BLOCK_READ8( B, coordBTemp ) );    coordB.y += TILE_K; \
        int2    coordATemp = coordA; \
        Dtype8  blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8; \
        Dtype8  blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8; \
        Dtype8  blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8; \
        Dtype8  blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.x += TILE_K * SIZE_OF_ELEMENT; \
        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 ); \
    } \
    while( coordB.y < width0 ); \
    GEMM_OUTPUT(ALPHA1, BETA_NOT0); \
}

GEMM_NN(1, 0) // ALPHA == 1, BETA == 0
GEMM_NN(1, 1) // ALPHA == 1, BETA != 0
GEMM_NN(0, 0) // ALPHA != 1, BETA == 0
GEMM_NN(0, 1) // ALPHA != 1, BETA != 0

#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
#undef GEMM_NN

// replicate the first row to column block.
#define TRANSPOSE_BLOCK_8(_vec, _col) \
        (Dtype8)( intel_sub_group_shuffle(_vec, _col + 0), \
                  intel_sub_group_shuffle(_vec, _col + 1), \
                  intel_sub_group_shuffle(_vec, _col + 2), \
                  intel_sub_group_shuffle(_vec, _col + 3), \
                  intel_sub_group_shuffle(_vec, _col + 4), \
                  intel_sub_group_shuffle(_vec, _col + 5), \
                  intel_sub_group_shuffle(_vec, _col + 6), \
                  intel_sub_group_shuffle(_vec, _col + 7) )

#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB, _col )    \
        {   \
            _result = mad( (Dtype8)(_blockB.s0), TRANSPOSE_BLOCK_8(_blockA.s0, _col), _result );      \
            _result = mad( (Dtype8)(_blockB.s1), TRANSPOSE_BLOCK_8(_blockA.s1, _col), _result );      \
            _result = mad( (Dtype8)(_blockB.s2), TRANSPOSE_BLOCK_8(_blockA.s2, _col), _result );      \
            _result = mad( (Dtype8)(_blockB.s3), TRANSPOSE_BLOCK_8(_blockA.s3, _col), _result );      \
            _result = mad( (Dtype8)(_blockB.s4), TRANSPOSE_BLOCK_8(_blockA.s4, _col), _result );      \
            _result = mad( (Dtype8)(_blockB.s5), TRANSPOSE_BLOCK_8(_blockA.s5, _col), _result );      \
            _result = mad( (Dtype8)(_blockB.s6), TRANSPOSE_BLOCK_8(_blockA.s6, _col), _result );      \
            _result = mad( (Dtype8)(_blockB.s7), TRANSPOSE_BLOCK_8(_blockA.s7, _col), _result );      \
        }

#define GEMM_TN(ALPHA1, BETA_NOT0) \
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) \
__attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) \
__kernel void TEMPLATE(gemm_32_1_TN_ ##ALPHA1 ##_ ##BETA_NOT0,Dtype)( \
    __read_only image2d_t A, \
    __read_only image2d_t B, \
    MATC_PARAMETER, \
    KERNEL_ARG_DTYPE alpha_in, \
    KERNEL_ARG_DTYPE beta_in, \
    int width0, \
    int isFirstColBlock) \
{ \
    const Dtype alpha = (Dtype)alpha_in; \
    const Dtype beta = (Dtype)beta_in; \
    const int group_x = get_group_id(0);\
    const int group_y = get_group_id(1);\
    Dtype8 blockAxB00 = 0.0f;\
    Dtype8 blockAxB01 = 0.0f;\
    Dtype8 blockAxB02 = 0.0f;\
    Dtype8 blockAxB03 = 0.0f;\
    int2    coordA = (int2)( group_y * TILE_M * SIZE_OF_ELEMENT, 0 );\
    int2    coordB = (int2)( ( group_x * TILE_N ) * SIZE_OF_ELEMENT, 0 );\
    do\
    {\
        int2    coordBTemp = coordB;\
        Dtype8 blockB00 = as_Dtype8( SUBGROUP_BLOCK_READ8( B, coordBTemp ) );    coordB.y += TILE_K;\
        int2    coordATemp = coordA;\
        Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;\
        Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;\
        Dtype8 blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;\
        Dtype8 blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.y += TILE_K;\
        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00, 0 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00, 0 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00, 0 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00, 0 ); \
    } \
    while( coordB.y < width0 ); \
    GEMM_OUTPUT(ALPHA1, BETA_NOT0); \
}

GEMM_TN(1, 0) // ALPHA == 1, BETA == 0
GEMM_TN(1, 1) // ALPHA == 1, BETA != 0
GEMM_TN(0, 0) // ALPHA != 1, BETA == 0
GEMM_TN(0, 1) // ALPHA != 1, BETA != 0

#undef MULTIPLY_BLOCKS_8x8
#undef TRANSPOSE_BLOCK_8
#undef GEMM_TN

// The same as GEMM_NN
#define TRANSPOSE_BLOCK_8( _block, _col )   \
        (Dtype8)( intel_sub_group_shuffle( _block.s0, _col),   \
                  intel_sub_group_shuffle( _block.s1, _col),   \
                  intel_sub_group_shuffle( _block.s2, _col),   \
                  intel_sub_group_shuffle( _block.s3, _col),   \
                  intel_sub_group_shuffle( _block.s4, _col),   \
                  intel_sub_group_shuffle( _block.s5, _col),   \
                  intel_sub_group_shuffle( _block.s6, _col),   \
                  intel_sub_group_shuffle( _block.s7, _col) )

#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \
        {   \
            const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );    \
            const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );    \
            const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );    \
            const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );    \
            const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );    \
            const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );    \
            const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );    \
            const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );    \
            _result = mad( (Dtype8)_blockB.s0, acol0, _result );      \
            _result = mad( (Dtype8)_blockB.s1, acol1, _result );      \
            _result = mad( (Dtype8)_blockB.s2, acol2, _result );      \
            _result = mad( (Dtype8)_blockB.s3, acol3, _result );      \
            _result = mad( (Dtype8)_blockB.s4, acol4, _result );      \
            _result = mad( (Dtype8)_blockB.s5, acol5, _result );      \
            _result = mad( (Dtype8)_blockB.s6, acol6, _result );      \
            _result = mad( (Dtype8)_blockB.s7, acol7, _result );      \
        }

#define GEMM_NT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) \
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) \
__attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) \
__kernel void TEMPLATE(gemm_32_1_NT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0,Dtype)( \
    __read_only image2d_t A, \
    MATB_PARAMETER, \
    MATC_PARAMETER, \
    KERNEL_ARG_DTYPE alpha_in, \
    KERNEL_ARG_DTYPE beta_in, \
    int padded_k, \
    int k, \
    int isFirstColBlock) \
{ \
    const Dtype alpha = (Dtype)alpha_in; \
    const Dtype beta = (Dtype)beta_in; \
    const int group_x = get_group_id(0); \
    const int group_y = get_group_id(1); \
    Dtype8 blockAxB00 = 0.0f; \
    Dtype8 blockAxB01 = 0.0f; \
    Dtype8 blockAxB02 = 0.0f; \
    Dtype8 blockAxB03 = 0.0f; \
    int2    coordA = (int2)( 0, group_y * TILE_M ); \
    int2    coordB = (int2)( 0, ( group_x * TILE_N )); \
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; \
    do \
    { \
        Dtype8 blockB00;  \
        BLOCKB_READ8(blockB00, B, coordB); \
        int2    coordATemp = coordA; \
        Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8; \
        Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8; \
        Dtype8 blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8; \
        Dtype8 blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.x += TILE_K * SIZE_OF_ELEMENT; \
        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 ); \
    } \
    while( coordB.x < padded_k / VECSIZE ); \
    GEMM_OUTPUT(ALPHA1, BETA_NOT0); \
}

#define BLOCKB_READ8(_blockb, _B, _coordB) \
        int2 _coordBTemp = _coordB; \
        _coordBTemp.y += get_local_id(0); \
        _blockb.s0123 = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s4567 = READ_IMAGE(_B, _coordBTemp); _coordB.x += 2;

#define MATB_PARAMETER __read_only image2d_t B

GEMM_NT(1, 0, VEC4, 4) // ALPHA == 1, BETA == 0
GEMM_NT(1, 1, VEC4, 4) // ALPHA == 1, BETA != 0
GEMM_NT(0, 0, VEC4, 4) // ALPHA != 1, BETA == 0
GEMM_NT(0, 1, VEC4, 4) // ALPHA != 1, BETA != 0
#undef BLOCKB_READ8
#undef MATB_PARAMETER

#define BLOCKB_READ8(_blockb, _B, _coordB) \
        int2 _coordBTemp = _coordB; \
        _coordBTemp.y += get_local_id(0); \
        const __global Dtype *B_read = (__global Dtype *)(_B + (_coordBTemp.y * ldb) + _coordBTemp.x + offB); \
        _blockb = vload8(0, B_read); \
        _coordB.x += TILE_K;

#define MATB_PARAMETER __global Dtype *B, int offB, int ldb

GEMM_NT(1, 0, BUFFER, 1) // ALPHA == 1, BETA == 0
GEMM_NT(1, 1, BUFFER, 1) // ALPHA == 1, BETA != 0
GEMM_NT(0, 0, BUFFER, 1) // ALPHA != 1, BETA == 0
GEMM_NT(0, 1, BUFFER, 1) // ALPHA != 1, BETA != 0
#undef BLOCKB_READ8
#undef MATB_PARAMETER

#define BLOCKB_READ8(_blockb, _B, _coordB) \
        int2 _coordBTemp = _coordB; \
        _coordBTemp.y += get_local_id(0); \
        Dtype4 temp; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s0 = temp.s0; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s1 = temp.s0; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s2 = temp.s0; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s3 = temp.s0; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s4 = temp.s0; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s5 = temp.s0; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s6 = temp.s0; \
        temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s7 = temp.s0; \
        _coordB.x += 8;

#define MATB_PARAMETER __read_only image2d_t B

GEMM_NT(1, 0, SCALAR, 1) // ALPHA == 1, BETA == 0
GEMM_NT(1, 1, SCALAR, 1) // ALPHA == 1, BETA != 0
GEMM_NT(0, 0, SCALAR, 1) // ALPHA != 1, BETA == 0
GEMM_NT(0, 1, SCALAR, 1) // ALPHA != 1, BETA != 0
#undef BLOCKB_READ8
#undef MATB_PARAMETER

#undef MULTIPLY_BLOCKS_8x8
#undef TRANSPOSE_BLOCK_8
#undef GEMM_NT

//The same as GEMM_TN.
#define TRANSPOSE_BLOCK_8(_vec, _col) \
        (Dtype8)( intel_sub_group_shuffle(_vec, _col + 0), \
                  intel_sub_group_shuffle(_vec, _col + 1), \
                  intel_sub_group_shuffle(_vec, _col + 2), \
                  intel_sub_group_shuffle(_vec, _col + 3), \
                  intel_sub_group_shuffle(_vec, _col + 4), \
                  intel_sub_group_shuffle(_vec, _col + 5), \
                  intel_sub_group_shuffle(_vec, _col + 6), \
                  intel_sub_group_shuffle(_vec, _col + 7) );

#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB, _col )    \
        {   \
            const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA.s0, _col );    \
            const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA.s1, _col );    \
            const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA.s2, _col );    \
            const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA.s3, _col );    \
            const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA.s4, _col );    \
            const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA.s5, _col );    \
            const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA.s6, _col );    \
            const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA.s7, _col );    \
            _result = mad( (Dtype8)_blockB.s0, acol0, _result );      \
            _result = mad( (Dtype8)_blockB.s1, acol1, _result );      \
            _result = mad( (Dtype8)_blockB.s2, acol2, _result );      \
            _result = mad( (Dtype8)_blockB.s3, acol3, _result );      \
            _result = mad( (Dtype8)_blockB.s4, acol4, _result );      \
            _result = mad( (Dtype8)_blockB.s5, acol5, _result );      \
            _result = mad( (Dtype8)_blockB.s6, acol6, _result );      \
            _result = mad( (Dtype8)_blockB.s7, acol7, _result );      \
        }

#define GEMM_TT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) \
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) \
__attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) \
__kernel void TEMPLATE(gemm_32_1_TT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0, Dtype)( \
    __read_only image2d_t A, \
    MATB_PARAMETER, \
    MATC_PARAMETER, \
    KERNEL_ARG_DTYPE alpha_in, \
    KERNEL_ARG_DTYPE beta_in, \
    int padded_k, \
    int k, \
    int isFirstColBlock) \
{ \
    const Dtype alpha = (Dtype)alpha_in; \
    const Dtype beta = (Dtype)beta_in; \
    const int group_x = get_group_id(0); \
    const int group_y = get_group_id(1); \
    Dtype8 blockAxB00 = 0.0f; \
    Dtype8 blockAxB01 = 0.0f; \
    Dtype8 blockAxB02 = 0.0f; \
    Dtype8 blockAxB03 = 0.0f; \
    int2    coordA = (int2)( group_y * TILE_M * SIZE_OF_ELEMENT, 0 ); \
    int2    coordB = (int2)( 0, ( group_x * TILE_N )); \
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; \
    do \
    { \
        Dtype8 blockB00;             \
        BLOCKB_READ8(blockB00, B, coordB); \
        int2    coordATemp = coordA; \
        Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT; \
        Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT; \
        Dtype8 blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT; \
        Dtype8 blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.y += TILE_K; \
        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00 , blockB00, 0 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01 , blockB00, 0 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02 , blockB00, 0 ); \
        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03 , blockB00, 0 ); \
    } \
    while( coordB.x < padded_k / VECSIZE ); \
    GEMM_OUTPUT(ALPHA1, BETA_NOT0);\
}

#define BLOCKB_READ8(_blockb, _B, _coordB) \
        int2 _coordBTemp = _coordB; \
        _coordBTemp.y += get_local_id(0); \
        _blockb.s0123 = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s4567 = READ_IMAGE(_B, _coordBTemp); _coordB.x += 2;

#define MATB_PARAMETER __read_only image2d_t B

GEMM_TT(1, 0, VEC4, 4) // ALPHA == 1, BETA == 0
GEMM_TT(1, 1, VEC4, 4) // ALPHA == 1, BETA != 0
GEMM_TT(0, 0, VEC4, 4) // ALPHA != 1, BETA == 0
GEMM_TT(0, 1, VEC4, 4) // ALPHA != 1, BETA != 0
#undef BLOCKB_READ8
#undef MATB_PARAMETER

#define BLOCKB_READ8(_blockb, _B, _coordB) \
        int2 _coordBTemp = _coordB; \
        _coordBTemp.y += get_local_id(0); \
        const __global Dtype *B_read = (__global Dtype *)(_B + (_coordBTemp.y * k) + _coordBTemp.x + offB); \
        _blockb = vload8(0, B_read); \
        _coordB.x += TILE_K;

#define MATB_PARAMETER __global Dtype *B, int offB, int ldb

GEMM_TT(1, 0, BUFFER, 1) // ALPHA == 1, BETA == 0
GEMM_TT(1, 1, BUFFER, 1) // ALPHA == 1, BETA != 0
GEMM_TT(0, 0, BUFFER, 1) // ALPHA != 1, BETA == 0
GEMM_TT(0, 1, BUFFER, 1) // ALPHA != 1, BETA != 0
#undef BLOCKB_READ8
#undef MATB_PARAMETER

#define BLOCKB_READ8(_blockb, _B, _coordB) \
        int2 _coordBTemp = _coordB; \
        _coordBTemp.y += get_local_id(0); \
        Dtype4 temp; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s0 = temp.s0; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s1 = temp.s0; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s2 = temp.s0; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s3 = temp.s0; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s4 = temp.s0; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s5 = temp.s0; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s6 = temp.s0; \
        temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1; \
        _blockb.s7 = temp.s0; \
        _coordB.x += 8;

#define MATB_PARAMETER __read_only image2d_t B

GEMM_TT(1, 0, SCALAR, 1) // ALPHA == 1, BETA == 0
GEMM_TT(1, 1, SCALAR, 1) // ALPHA == 1, BETA != 0
GEMM_TT(0, 0, SCALAR, 1) // ALPHA != 1, BETA == 0
GEMM_TT(0, 1, SCALAR, 1) // ALPHA != 1, BETA != 0
#undef BLOCKB_READ8
#undef MATB_PARAMETER

#undef MULTIPLY_BLOCKS_8x8
#undef TRANSPOSE_BLOCK_8
#undef GEMM_TT

#undef TILE_M
#undef TILE_K
#undef TILE_N
#undef SUBGROUP_BLOCK_READ8
#undef READ_IMAGE
#undef SIZE_OF_ELEMENT

__kernel void TEMPLATE(gemm_buffer_copy_image_transpose,Dtype)(
    __global Dtype* A,
    __write_only image2d_t ImA,
    int offA,
    int width,
    int height,
    int ldA)
{
    const int gidx = get_global_id(0);
    const int gidy = get_global_id(1);
    int2 coord_dst = (int2)(gidx, gidy);
    __global Dtype* A_off = A + offA;
    Dtype srcA = A_off[gidy * ldA + gidx];
    write_imagef(ImA, coord_dst, (Dtype4)srcA);
}

__kernel void TEMPLATE(gemm_buffer_copy_image_no_transpose,Dtype)(
    __global Dtype* A,
    __write_only image2d_t ImA,
    int offA,
    int width,
    int height,
    int ldA)
{
    const int gidx = get_global_id(0);
    const int gidy = get_global_id(1);
    int2 coord_dst = (int2)(gidx, gidy);
    if (gidx >= width || gidy >= height) {
      write_imageui(ImA, coord_dst, (uint4)0);
      return;
    }
    __global Dtype* A_off = A + offA;
    uint4 srcA = convert_uint4(as_uchar4(A_off[gidy * ldA + gidx]));
    write_imageui(ImA, coord_dst, srcA);
}
