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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/hal/intrin.hpp"

namespace cv
{

void spatialGradient( InputArray _src, OutputArray _dx, OutputArray _dy,
                      int ksize, int borderType )
{

    // Prepare InputArray src
    Mat src = _src.getMat();
    CV_Assert( !src.empty() );
    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( borderType == BORDER_DEFAULT || borderType == BORDER_REPLICATE );

    // Prepare OutputArrays dx, dy
    _dx.create( src.size(), CV_16SC1 );
    _dy.create( src.size(), CV_16SC1 );
    Mat dx = _dx.getMat(),
        dy = _dy.getMat();

    // TODO: Allow for other kernel sizes
    CV_Assert(ksize == 3);

    // Get dimensions
    const int H = src.rows,
              W = src.cols;

    // Row, column indices
    int i, j;

    // Store pointers to rows of input/output data
    // Padded by two rows for border handling
    uchar* P_src[H+2];
    short* P_dx [H+2];
    short* P_dy [H+2];

    int i_top    = 0,     // Case for H == 1 && W == 1 && BORDER_REPLICATE
        i_bottom = H - 1,
        j_offl   = 0,     // j offset from 0th   pixel to reach -1st pixel
        j_offr   = 0;     // j offset from W-1th pixel to reach Wth  pixel

    if ( borderType == BORDER_DEFAULT ) // Equiv. to BORDER_REFLECT_101
    {
        if ( H > 1 )
        {
            i_top    = 1;
            i_bottom = H - 2;
        }
        if ( W > 1 )
        {
            j_offl = 1;
            j_offr = -1;
        }
    }

    P_src[0]   = src.ptr<uchar>(i_top); // Mirrored top border
    P_src[H+1] = src.ptr<uchar>(i_bottom); // Mirrored bottom border

    for ( i = 0; i < H; i++ )
    {
        P_src[i+1] = src.ptr<uchar>(i);
        P_dx [i]   =  dx.ptr<short>(i);
        P_dy [i]   =  dy.ptr<short>(i);
    }

    // Pointer to row vectors
    uchar *p_src, *c_src, *n_src; // previous, current, next row
    short *c_dx,  *c_dy;

    int j_start = 0;
/*
#if CV_SIMD128
    // Characters in variable names have the following meanings:
    // u: unsigned char
    // s: signed int
    //
    // [row][column]
    // m: offset -1
    // n: offset  0
    // p: offset  1
    // Example: umn is offset -1 in row and offset 0 in column
    v_uint8x16 v_um, v_un, v_up;
    v_uint16x8 v_um1, v_um2, v_un1, v_un2, v_up1, v_up2;
    v_int16x8 v_smm1, v_smm2, v_smn1, v_smn2, v_smp1, v_smp2,
              v_snm1, v_snm2, v_snn1, v_snn2, v_snp1, v_snp2,
              v_spm1, v_spm2, v_spn1, v_spn2, v_spp1, v_spp2,
              v_two = v_setall_s16(2),
              v_sdx1, v_sdx2, v_sdy1, v_sdy2;

    for ( i = 1; i < H - 1; i++ )
    {
        // 16-column chunks at a time
        for ( j = 1; j < W - 1 - 15; j += 16 )
        {
            // Load top row for 3x3 Sobel filter
            idx = i*W + j;
            v_um = v_load(&p_src[idx - W - 1]);
            v_un = v_load(&p_src[idx - W]);
            v_up = v_load(&p_src[idx - W + 1]);
            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_smm1 = v_reinterpret_as_s16(v_um1);
            v_smm2 = v_reinterpret_as_s16(v_um2);
            v_smn1 = v_reinterpret_as_s16(v_un1);
            v_smn2 = v_reinterpret_as_s16(v_un2);
            v_smp1 = v_reinterpret_as_s16(v_up1);
            v_smp2 = v_reinterpret_as_s16(v_up2);

            // Load second row for 3x3 Sobel filter
            v_um = v_load(&p_src[idx - 1]);
            v_un = v_load(&p_src[idx]);
            v_up = v_load(&p_src[idx + 1]);
            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_snm1 = v_reinterpret_as_s16(v_um1);
            v_snm2 = v_reinterpret_as_s16(v_um2);
            v_snn1 = v_reinterpret_as_s16(v_un1);
            v_snn2 = v_reinterpret_as_s16(v_un2);
            v_snp1 = v_reinterpret_as_s16(v_up1);
            v_snp2 = v_reinterpret_as_s16(v_up2);

            // Load last row for 3x3 Sobel filter
            v_um = v_load(&p_src[idx + W - 1]);
            v_un = v_load(&p_src[idx + W]);
            v_up = v_load(&p_src[idx + W + 1]);
            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_spm1 = v_reinterpret_as_s16(v_um1);
            v_spm2 = v_reinterpret_as_s16(v_um2);
            v_spn1 = v_reinterpret_as_s16(v_un1);
            v_spn2 = v_reinterpret_as_s16(v_un2);
            v_spp1 = v_reinterpret_as_s16(v_up1);
            v_spp2 = v_reinterpret_as_s16(v_up2);

            // dx
            v_sdx1 = (v_smp1 - v_smm1) + v_two*(v_snp1 - v_snm1) + (v_spp1 - v_spm1);
            v_sdx2 = (v_smp2 - v_smm2) + v_two*(v_snp2 - v_snm2) + (v_spp2 - v_spm2);

            // dy
            v_sdy1 = (v_spm1 - v_smm1) + v_two*(v_spn1 - v_smn1) + (v_spp1 - v_smp1);
            v_sdy2 = (v_spm2 - v_smm2) + v_two*(v_spn2 - v_smn2) + (v_spp2 - v_smp2);

            // Store
            v_store(&p_dx[idx],   v_sdx1);
            v_store(&p_dx[idx+8], v_sdx2);
            v_store(&p_dy[idx],   v_sdy1);
            v_store(&p_dy[idx+8], v_sdy2);
        }

        // Cleanup
        for ( ; j < W - 1; j++ )
        {
            idx = i*W + j;
            p_dx[idx] = -(p_src[idx-W-1] + 2*p_src[idx-1] + p_src[idx+W-1]) +
                         (p_src[idx-W+1] + 2*p_src[idx+1] + p_src[idx+W+1]);
            p_dy[idx] = -(p_src[idx-W-1] + 2*p_src[idx-W] + p_src[idx-W+1]) +
                         (p_src[idx+W-1] + 2*p_src[idx+W] + p_src[idx+W+1]);
        }
    }
#else
*/

    /* NOTE:
     *
     * Sobel-x: -1  0  1
     *          -2  0  2
     *          -1  0  1
     *
     * Sobel-y: -1 -2 -1
     *           0  0  0
     *           1  2  1
     */
    int j_p, j_n;
    for ( i = 0; i < H; i++ )
    {
        p_src = P_src[i]; c_src = P_src[i+1]; n_src = P_src[i+2];
        c_dx  = P_dx [i];
        c_dy  = P_dy [i];

        for ( j = j_start; j < W; j++ )
        {
            j_p = j - 1;
            j_n = j + 1;
            if ( j_p <  0 ) j_p = j + j_offl;
            if ( j_n >= W ) j_n = j + j_offr;

            c_dx[j] = -(p_src[j_p] + c_src[j_p] + c_src[j_p] + n_src[j_p]) +
                       (p_src[j_n] + c_src[j_n] + c_src[j_n] + n_src[j_n]);
            c_dy[j] = -(p_src[j_p] + p_src[j]   + p_src[j]   + p_src[j_n]) +
                       (n_src[j_p] + n_src[j]   + n_src[j]   + n_src[j_n]);
        }
    }
//#endif

}

}
