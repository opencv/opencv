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
    std::vector<uchar*> P_src(H+2);
    std::vector<short*> P_dx (H+2);
    std::vector<short*> P_dy (H+2);

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

    int i_start = 0;
    int j_start = 0;
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
    uchar tmp;
    v_uint8x16 v_um, v_un, v_up;
    v_uint16x8 v_um1, v_um2, v_un1, v_un2, v_up1, v_up2;
    v_int16x8 v_s1m1, v_s1m2, v_s1n1, v_s1n2, v_s1p1, v_s1p2,
              v_s2m1, v_s2m2, v_s2n1, v_s2n2, v_s2p1, v_s2p2,
              v_s3m1, v_s3m2, v_s3n1, v_s3n2, v_s3p1, v_s3p2,
              v_tmp,  v_sdx, v_sdy;

    for ( i = 0; i < H - 2; i += 2 )
    {
        p_src = P_src[i]; c_src = P_src[i+1]; n_src = P_src[i+2];
        c_dx = P_dx [i];
        c_dy = P_dy [i];

        // 16-column chunks at a time
        for ( j = 0; j < W - 15; j += 16 )
        {
            bool left = false, right = false;
            if ( j == 0 )      left  = true;
            if ( j == W - 16 ) right = true;

            // Load top row for 3x3 Sobel filter
            if ( left ) { tmp = p_src[j-1]; p_src[j-1] = p_src[j+j_offl]; }
            v_um = v_load(&p_src[j-1]);
            if ( left ) p_src[j-1] = tmp;

            v_un = v_load(&p_src[j]);

            if ( right ) { tmp = p_src[j+16]; p_src[j+16] = p_src[j+15+j_offr]; }
            v_up = v_load(&p_src[j+1]);
            if ( right ) p_src[j+16] = tmp;

            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_s1m1 = v_reinterpret_as_s16(v_um1);
            v_s1m2 = v_reinterpret_as_s16(v_um2);
            v_s1n1 = v_reinterpret_as_s16(v_un1);
            v_s1n2 = v_reinterpret_as_s16(v_un2);
            v_s1p1 = v_reinterpret_as_s16(v_up1);
            v_s1p2 = v_reinterpret_as_s16(v_up2);

            // Load second row for 3x3 Sobel filter
            if ( left ) { tmp = c_src[j-1]; c_src[j-1] = c_src[j+j_offl]; }
            v_um = v_load(&c_src[j-1]);
            if ( left ) c_src[j-1] = tmp;

            v_un = v_load(&c_src[j]);

            if ( right ) { tmp = c_src[j+16]; c_src[j+16] = c_src[j+15+j_offr]; }
            v_up = v_load(&c_src[j+1]);
            if ( right ) c_src[j+16] = tmp;

            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_s2m1 = v_reinterpret_as_s16(v_um1);
            v_s2m2 = v_reinterpret_as_s16(v_um2);
            v_s2n1 = v_reinterpret_as_s16(v_un1);
            v_s2n2 = v_reinterpret_as_s16(v_un2);
            v_s2p1 = v_reinterpret_as_s16(v_up1);
            v_s2p2 = v_reinterpret_as_s16(v_up2);

            // Load third row for 3x3 Sobel filter
            if ( left ) { tmp = n_src[j-1]; n_src[j-1] = n_src[j+j_offl]; }
            v_um = v_load(&n_src[j-1]);
            if ( left ) n_src[j-1] = tmp;

            v_un = v_load(&n_src[j]);

            if ( right ) { tmp = n_src[j+16]; n_src[j+16] = n_src[j+15+j_offr]; }
            v_up = v_load(&n_src[j+1]);
            if ( right ) n_src[j+16] = tmp;

            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_s3m1 = v_reinterpret_as_s16(v_um1);
            v_s3m2 = v_reinterpret_as_s16(v_um2);
            v_s3n1 = v_reinterpret_as_s16(v_un1);
            v_s3n2 = v_reinterpret_as_s16(v_un2);
            v_s3p1 = v_reinterpret_as_s16(v_up1);
            v_s3p2 = v_reinterpret_as_s16(v_up2);

            // dx
            v_tmp = v_s2p1 - v_s2m1;
            v_sdx = (v_s1p1 - v_s1m1) + (v_tmp + v_tmp) + (v_s3p1 - v_s3m1);
            v_tmp = v_s2p2 - v_s2m2;
            v_sdx = (v_s1p2 - v_s1m2) + (v_tmp + v_tmp) + (v_s3p2 - v_s3m2);

            // dy
            v_tmp = v_s3n1 - v_s1n1;
            v_sdy = (v_s3m1 - v_s1m1) + (v_tmp + v_tmp) + (v_s3p1 - v_s1p1);
            v_tmp = v_s3n2 - v_s1n2;
            v_sdy = (v_s3m2 - v_s1m2) + (v_tmp + v_tmp) + (v_s3p2 - v_s1p2);

            // Store
            v_store(&c_dx[j],   v_sdx);
            v_store(&c_dx[j+8], v_sdx);
            v_store(&c_dy[j],   v_sdy);
            v_store(&c_dy[j+8], v_sdy);
        }
    }
    i_start = i;
    j_start = j;
#endif
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
    uchar v00, v01, v02, v10, v11, v12, v20, v21, v22;
    for ( i = 0; i < H; i++ )
    {
        p_src = P_src[i]; c_src = P_src[i+1]; n_src = P_src[i+2];
        c_dx  = P_dx [i];
        c_dy  = P_dy [i];

        // Pre-load 2 columns
        j = i >= i_start ? 0 : j_start;
        j_p = j - 1;
        if ( j_p <  0 ) j_p = j + j_offl;
        v00 = p_src[j_p]; v01 = p_src[j];
        v10 = c_src[j_p]; v11 = c_src[j];
        v20 = n_src[j_p]; v21 = n_src[j];

        for ( ; j < W; j++ )
        {
            j_n = j + 1;
            if ( j_n >= W ) j_n = j + j_offr;

            // Get values for next column
            v02 = p_src[j_n];
            v12 = c_src[j_n];
            v22 = n_src[j_n];

            c_dx[j] = -(v00 + v10 + v10 + v20) + (v02 + v12 + v12 + v22);
            c_dy[j] = -(v00 + v01 + v01 + v02) + (v20 + v21 + v21 + v22);

            // Move values back one column for next iteration
            v00 = v01; v10 = v11; v20 = v21;
            v01 = v02; v11 = v12; v21 = v22;
        }
    }

}

}
