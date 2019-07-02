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
#include "opencv2/core/hal/intrin.hpp"

#include <iostream>
namespace cv
{

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
template <typename T>
static inline void spatialGradientKernel( T& vx, T& vy,
                                          const T& v00, const T& v01, const T& v02,
                                          const T& v10,               const T& v12,
                                          const T& v20, const T& v21, const T& v22 )
{
    // vx = (v22 - v00) + (v02 - v20) + 2 * (v12 - v10)
    // vy = (v22 - v00) + (v20 - v02) + 2 * (v21 - v01)

    T tmp_add = v22 - v00,
      tmp_sub = v02 - v20,
      tmp_x   = v12 - v10,
      tmp_y   = v21 - v01;

    vx = tmp_add + tmp_sub + tmp_x + tmp_x;
    vy = tmp_add - tmp_sub + tmp_y + tmp_y;
}

void spatialGradient( InputArray _src, OutputArray _dx, OutputArray _dy,
                      int ksize, int borderType )
{
    CV_INSTRUMENT_REGION();

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
    int i = 0,
        j = 0;

    // Handle border types
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

    int i_start = 0;
    int j_start = 0;
#if CV_SIMD
    // Characters in variable names have the following meanings:
    // u: unsigned char
    // s: signed int
    //
    // [row][column]
    // m: offset -1
    // n: offset  0
    // p: offset  1
    // Example: umn is offset -1 in row and offset 0 in column
    for ( i = 0; i < H - 1; i += 2 )
    {
        uchar *p_src = src.ptr<uchar>(i == 0 ? i_top : i - 1);
        uchar *c_src = src.ptr<uchar>(i);
        uchar *n_src = src.ptr<uchar>(i+1);
        uchar *m_src = src.ptr<uchar>(i == H - 2 ? i_bottom : i + 2);

        short *c_dx = dx.ptr<short>(i);
        short *c_dy = dy.ptr<short>(i);
        short *n_dx = dx.ptr<short>(i+1);
        short *n_dy = dy.ptr<short>(i+1);

        // Process rest of columns 16-column chunks at a time
        for ( j = 1; j < W - v_uint8::nlanes; j += v_uint8::nlanes)
        {
            // Load top row for 3x3 Sobel filter
            v_uint8 v_um = vx_load(&p_src[j-1]);
            v_uint8 v_un = vx_load(&p_src[j]);
            v_uint8 v_up = vx_load(&p_src[j+1]);
            v_uint16 v_um1, v_um2, v_un1, v_un2, v_up1, v_up2;
            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_int16 v_s1m1 = v_reinterpret_as_s16(v_um1);
            v_int16 v_s1m2 = v_reinterpret_as_s16(v_um2);
            v_int16 v_s1n1 = v_reinterpret_as_s16(v_un1);
            v_int16 v_s1n2 = v_reinterpret_as_s16(v_un2);
            v_int16 v_s1p1 = v_reinterpret_as_s16(v_up1);
            v_int16 v_s1p2 = v_reinterpret_as_s16(v_up2);

            // Load second row for 3x3 Sobel filter
            v_um = vx_load(&c_src[j-1]);
            v_un = vx_load(&c_src[j]);
            v_up = vx_load(&c_src[j+1]);
            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_int16 v_s2m1 = v_reinterpret_as_s16(v_um1);
            v_int16 v_s2m2 = v_reinterpret_as_s16(v_um2);
            v_int16 v_s2n1 = v_reinterpret_as_s16(v_un1);
            v_int16 v_s2n2 = v_reinterpret_as_s16(v_un2);
            v_int16 v_s2p1 = v_reinterpret_as_s16(v_up1);
            v_int16 v_s2p2 = v_reinterpret_as_s16(v_up2);

            // Load third row for 3x3 Sobel filter
            v_um = vx_load(&n_src[j-1]);
            v_un = vx_load(&n_src[j]);
            v_up = vx_load(&n_src[j+1]);
            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_int16 v_s3m1 = v_reinterpret_as_s16(v_um1);
            v_int16 v_s3m2 = v_reinterpret_as_s16(v_um2);
            v_int16 v_s3n1 = v_reinterpret_as_s16(v_un1);
            v_int16 v_s3n2 = v_reinterpret_as_s16(v_un2);
            v_int16 v_s3p1 = v_reinterpret_as_s16(v_up1);
            v_int16 v_s3p2 = v_reinterpret_as_s16(v_up2);

            // dx & dy for rows 1, 2, 3
            v_int16 v_sdx1, v_sdy1;
            spatialGradientKernel<v_int16>( v_sdx1, v_sdy1,
                                              v_s1m1, v_s1n1, v_s1p1,
                                              v_s2m1,         v_s2p1,
                                              v_s3m1, v_s3n1, v_s3p1 );

            v_int16 v_sdx2, v_sdy2;
            spatialGradientKernel<v_int16>( v_sdx2, v_sdy2,
                                              v_s1m2, v_s1n2, v_s1p2,
                                              v_s2m2,         v_s2p2,
                                              v_s3m2, v_s3n2, v_s3p2 );

            // Store
            v_store(&c_dx[j],                 v_sdx1);
            v_store(&c_dx[j+v_int16::nlanes], v_sdx2);
            v_store(&c_dy[j],                 v_sdy1);
            v_store(&c_dy[j+v_int16::nlanes], v_sdy2);

            // Load fourth row for 3x3 Sobel filter
            v_um = vx_load(&m_src[j-1]);
            v_un = vx_load(&m_src[j]);
            v_up = vx_load(&m_src[j+1]);
            v_expand(v_um, v_um1, v_um2);
            v_expand(v_un, v_un1, v_un2);
            v_expand(v_up, v_up1, v_up2);
            v_int16 v_s4m1 = v_reinterpret_as_s16(v_um1);
            v_int16 v_s4m2 = v_reinterpret_as_s16(v_um2);
            v_int16 v_s4n1 = v_reinterpret_as_s16(v_un1);
            v_int16 v_s4n2 = v_reinterpret_as_s16(v_un2);
            v_int16 v_s4p1 = v_reinterpret_as_s16(v_up1);
            v_int16 v_s4p2 = v_reinterpret_as_s16(v_up2);

            // dx & dy for rows 2, 3, 4
            spatialGradientKernel<v_int16>( v_sdx1, v_sdy1,
                                              v_s2m1, v_s2n1, v_s2p1,
                                              v_s3m1,         v_s3p1,
                                              v_s4m1, v_s4n1, v_s4p1 );

            spatialGradientKernel<v_int16>( v_sdx2, v_sdy2,
                                              v_s2m2, v_s2n2, v_s2p2,
                                              v_s3m2,         v_s3p2,
                                              v_s4m2, v_s4n2, v_s4p2 );

            // Store
            v_store(&n_dx[j],                 v_sdx1);
            v_store(&n_dx[j+v_int16::nlanes], v_sdx2);
            v_store(&n_dy[j],                 v_sdy1);
            v_store(&n_dy[j+v_int16::nlanes], v_sdy2);
        }
    }
    i_start = i;
    j_start = j;
#endif
    int j_p, j_n;
    uchar v00, v01, v02, v10, v11, v12, v20, v21, v22;
    for ( i = 0; i < H; i++ )
    {
        uchar *p_src = src.ptr<uchar>(i == 0 ? i_top : i - 1);
        uchar *c_src = src.ptr<uchar>(i);
        uchar *n_src = src.ptr<uchar>(i == H - 1 ? i_bottom : i + 1);

        short *c_dx = dx.ptr<short>(i);
        short *c_dy = dy.ptr<short>(i);

        // Process left-most column
        j = 0;
        j_p = j + j_offl;
        j_n = 1;
        if ( j_n >= W ) j_n = j + j_offr;
        v00 = p_src[j_p]; v01 = p_src[j]; v02 = p_src[j_n];
        v10 = c_src[j_p]; v11 = c_src[j]; v12 = c_src[j_n];
        v20 = n_src[j_p]; v21 = n_src[j]; v22 = n_src[j_n];
        spatialGradientKernel<short>( c_dx[0], c_dy[0], v00, v01, v02, v10,
                                      v12, v20, v21, v22 );
        v00 = v01; v10 = v11; v20 = v21;
        v01 = v02; v11 = v12; v21 = v22;

        // Process middle columns
        j = i >= i_start ? 1 : j_start;
        j_p = j - 1;
        v00 = p_src[j_p]; v01 = p_src[j];
        v10 = c_src[j_p]; v11 = c_src[j];
        v20 = n_src[j_p]; v21 = n_src[j];

        for ( ; j < W - 1; j++ )
        {
            // Get values for next column
            j_n = j + 1; v02 = p_src[j_n]; v12 = c_src[j_n]; v22 = n_src[j_n];
            spatialGradientKernel<short>( c_dx[j], c_dy[j], v00, v01, v02, v10,
                                          v12, v20, v21, v22 );

            // Move values back one column for next iteration
            v00 = v01; v10 = v11; v20 = v21;
            v01 = v02; v11 = v12; v21 = v22;
        }

        // Process right-most column
        if ( j < W )
        {
            j_n = j + j_offr; v02 = p_src[j_n]; v12 = c_src[j_n]; v22 = n_src[j_n];
            spatialGradientKernel<short>( c_dx[j], c_dy[j], v00, v01, v02, v10,
                                          v12, v20, v21, v22 );
        }
    }

}

}
