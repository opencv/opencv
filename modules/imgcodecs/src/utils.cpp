/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "utils.hpp"

namespace cv {

int validateToInt(size_t sz)
{
    int valueInt = (int)sz;
    CV_Assert((size_t)valueInt == sz);
    return valueInt;
}

#define  SCALE  14
#define  cR  (int)(0.299*(1 << SCALE) + 0.5)
#define  cG  (int)(0.587*(1 << SCALE) + 0.5)
#define  cB  ((1 << SCALE) - cR - cG)

void icvCvt_BGR2Gray_8u_C3C1R( const uchar* bgr, int bgr_step,
                               uchar* gray, int gray_step,
                               Size size, int _swap_rb )
{
    int i;
    for( ; size.height--; gray += gray_step )
    {
        short cBGR0 = cB;
        short cBGR2 = cR;
        if (_swap_rb) std::swap(cBGR0, cBGR2);
        for( i = 0; i < size.width; i++, bgr += 3 )
        {
            int t = descale( bgr[0]*cBGR0 + bgr[1]*cG + bgr[2]*cBGR2, SCALE );
            gray[i] = (uchar)t;
        }

        bgr += bgr_step - size.width*3;
    }
}


void icvCvt_BGRA2Gray_16u_CnC1R( const ushort* bgr, int bgr_step,
                                ushort* gray, int gray_step,
                                Size size, int ncn, int _swap_rb )
{
    int i;
    for( ; size.height--; gray += gray_step )
    {
        short cBGR0 = cB;
        short cBGR2 = cR;
        if (_swap_rb) std::swap(cBGR0, cBGR2);
        for( i = 0; i < size.width; i++, bgr += ncn )
        {
            int t = descale( bgr[0]*cBGR0 + bgr[1]*cG + bgr[2]*cBGR2, SCALE );
            gray[i] = (ushort)t;
        }

        bgr += bgr_step - size.width*ncn;
    }
}


void icvCvt_BGRA2Gray_8u_C4C1R( const uchar* bgra, int rgba_step,
                                uchar* gray, int gray_step,
                                Size size, int _swap_rb )
{
    int i;
    for( ; size.height--; gray += gray_step )
    {
        short cBGR0 = cB;
        short cBGR2 = cR;
        if (_swap_rb) std::swap(cBGR0, cBGR2);
        for( i = 0; i < size.width; i++, bgra += 4 )
        {
            int t = descale( bgra[0]*cBGR0 + bgra[1]*cG + bgra[2]*cBGR2, SCALE );
            gray[i] = (uchar)t;
        }

        bgra += rgba_step - size.width*4;
    }
}


void icvCvt_Gray2BGR_8u_C1C3R( const uchar* gray, int gray_step,
                               uchar* bgr, int bgr_step, Size size )
{
    int i;
    for( ; size.height--; gray += gray_step )
    {
        for( i = 0; i < size.width; i++, bgr += 3 )
        {
            bgr[0] = bgr[1] = bgr[2] = gray[i];
        }
        bgr += bgr_step - size.width*3;
    }
}


void icvCvt_Gray2BGR_16u_C1C3R( const ushort* gray, int gray_step,
                              ushort* bgr, int bgr_step, Size size )
{
    int i;
    for( ; size.height--; gray += gray_step/sizeof(gray[0]) )
    {
        for( i = 0; i < size.width; i++, bgr += 3 )
        {
            bgr[0] = bgr[1] = bgr[2] = gray[i];
        }
        bgr += bgr_step/sizeof(bgr[0]) - size.width*3;
    }
}


void icvCvt_BGRA2BGR_8u_C4C3R( const uchar* bgra, int bgra_step,
                               uchar* bgr, int bgr_step,
                               Size size, int _swap_rb )
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, bgr += 3, bgra += 4 )
        {
            uchar t0 = bgra[swap_rb], t1 = bgra[1];
            bgr[0] = t0; bgr[1] = t1;
            t0 = bgra[swap_rb^2]; bgr[2] = t0;
        }
        bgr += bgr_step - size.width*3;
        bgra += bgra_step - size.width*4;
    }
}


void icvCvt_BGRA2BGR_16u_C4C3R( const ushort* bgra, int bgra_step,
                              ushort* bgr, int bgr_step,
                              Size size, int _swap_rb )
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, bgr += 3, bgra += 4 )
        {
            ushort t0 = bgra[swap_rb], t1 = bgra[1];
            bgr[0] = t0; bgr[1] = t1;
            t0 = bgra[swap_rb^2]; bgr[2] = t0;
        }
        bgr += bgr_step/sizeof(bgr[0]) - size.width*3;
        bgra += bgra_step/sizeof(bgra[0]) - size.width*4;
    }
}


void icvCvt_BGRA2RGBA_8u_C4R( const uchar* bgra, int bgra_step,
                              uchar* rgba, int rgba_step, Size size )
{
    int i;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, bgra += 4, rgba += 4 )
        {
            uchar t0 = bgra[0], t1 = bgra[1];
            uchar t2 = bgra[2], t3 = bgra[3];
            rgba[0] = t2; rgba[1] = t1;
            rgba[2] = t0; rgba[3] = t3;
        }
        bgra += bgra_step - size.width*4;
        rgba += rgba_step - size.width*4;
    }
}

void icvCvt_BGRA2RGBA_16u_C4R( const ushort* bgra, int bgra_step,
                               ushort* rgba, int rgba_step, Size size )
{
 int i;
 for( ; size.height--; )
 {
     for( i = 0; i < size.width; i++, bgra += 4, rgba += 4 )
     {
         ushort t0 = bgra[0], t1 = bgra[1];
         ushort t2 = bgra[2], t3 = bgra[3];

         rgba[0] = t2; rgba[1] = t1;
         rgba[2] = t0; rgba[3] = t3;
     }
     bgra += bgra_step/sizeof(bgra[0]) - size.width*4;
     rgba += rgba_step/sizeof(rgba[0]) - size.width*4;
 }
}


void icvCvt_BGR2RGB_8u_C3R( const uchar* bgr, int bgr_step,
                            uchar* rgb, int rgb_step, Size size )
{
    int i;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, bgr += 3, rgb += 3 )
        {
            uchar t0 = bgr[0], t1 = bgr[1], t2 = bgr[2];
            rgb[2] = t0; rgb[1] = t1; rgb[0] = t2;
        }
        bgr += bgr_step - size.width*3;
        rgb += rgb_step - size.width*3;
    }
}


void icvCvt_BGR2RGB_16u_C3R( const ushort* bgr, int bgr_step,
                             ushort* rgb, int rgb_step, Size size )
{
    int i;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, bgr += 3, rgb += 3 )
        {
            ushort t0 = bgr[0], t1 = bgr[1], t2 = bgr[2];
            rgb[2] = t0; rgb[1] = t1; rgb[0] = t2;
        }
        bgr += bgr_step - size.width*3;
        rgb += rgb_step - size.width*3;
    }
}


typedef unsigned short ushort;

void icvCvt_BGR5552Gray_8u_C2C1R( const uchar* bgr555, int bgr555_step,
                                  uchar* gray, int gray_step, Size size )
{
    int i;
    for( ; size.height--; gray += gray_step, bgr555 += bgr555_step )
    {
        for( i = 0; i < size.width; i++ )
        {
            int t = descale( ((((ushort*)bgr555)[i] << 3) & 0xf8)*cB +
                             ((((ushort*)bgr555)[i] >> 2) & 0xf8)*cG +
                             ((((ushort*)bgr555)[i] >> 7) & 0xf8)*cR, SCALE );
            gray[i] = (uchar)t;
        }
    }
}


void icvCvt_BGR5652Gray_8u_C2C1R( const uchar* bgr565, int bgr565_step,
                                  uchar* gray, int gray_step, Size size )
{
    int i;
    for( ; size.height--; gray += gray_step, bgr565 += bgr565_step )
    {
        for( i = 0; i < size.width; i++ )
        {
            int t = descale( ((((ushort*)bgr565)[i] << 3) & 0xf8)*cB +
                             ((((ushort*)bgr565)[i] >> 3) & 0xfc)*cG +
                             ((((ushort*)bgr565)[i] >> 8) & 0xf8)*cR, SCALE );
            gray[i] = (uchar)t;
        }
    }
}


void icvCvt_BGR5552BGR_8u_C2C3R( const uchar* bgr555, int bgr555_step,
                                 uchar* bgr, int bgr_step, Size size )
{
    int i;
    for( ; size.height--; bgr555 += bgr555_step )
    {
        for( i = 0; i < size.width; i++, bgr += 3 )
        {
            int t0 = (((ushort*)bgr555)[i] << 3) & 0xf8;
            int t1 = (((ushort*)bgr555)[i] >> 2) & 0xf8;
            int t2 = (((ushort*)bgr555)[i] >> 7) & 0xf8;
            bgr[0] = (uchar)t0; bgr[1] = (uchar)t1; bgr[2] = (uchar)t2;
        }
        bgr += bgr_step - size.width*3;
    }
}


void icvCvt_BGR5652BGR_8u_C2C3R( const uchar* bgr565, int bgr565_step,
                                 uchar* bgr, int bgr_step, Size size )
{
    int i;
    for( ; size.height--; bgr565 += bgr565_step )
    {
        for( i = 0; i < size.width; i++, bgr += 3 )
        {
            int t0 = (((ushort*)bgr565)[i] << 3) & 0xf8;
            int t1 = (((ushort*)bgr565)[i] >> 3) & 0xfc;
            int t2 = (((ushort*)bgr565)[i] >> 8) & 0xf8;
            bgr[0] = (uchar)t0; bgr[1] = (uchar)t1; bgr[2] = (uchar)t2;
        }
        bgr += bgr_step - size.width*3;
    }
}


void icvCvt_CMYK2BGR_8u_C4C3R( const uchar* cmyk, int cmyk_step,
                               uchar* bgr, int bgr_step, Size size )
{
    int i;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, bgr += 3, cmyk += 4 )
        {
            int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
            c = k - ((255 - c)*k>>8);
            m = k - ((255 - m)*k>>8);
            y = k - ((255 - y)*k>>8);
            bgr[2] = (uchar)c; bgr[1] = (uchar)m; bgr[0] = (uchar)y;
        }
        bgr += bgr_step - size.width*3;
        cmyk += cmyk_step - size.width*4;
    }
}

void icvCvt_CMYK2RGB_8u_C4C3R( const uchar* cmyk, int cmyk_step,
                               uchar* rgb, int rgb_step, Size size )
{
    int i;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, rgb += 3, cmyk += 4 )
        {
            int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
            c = k - ((255 - c)*k>>8);
            m = k - ((255 - m)*k>>8);
            y = k - ((255 - y)*k>>8);
            rgb[0] = (uchar)c; rgb[1] = (uchar)m; rgb[2] = (uchar)y;
        }
        rgb += rgb_step - size.width*3;
        cmyk += cmyk_step - size.width*4;
    }
}


void icvCvt_CMYK2Gray_8u_C4C1R( const uchar* cmyk, int cmyk_step,
                                uchar* gray, int gray_step, Size size )
{
    int i;
    for( ; size.height--; )
    {
        for( i = 0; i < size.width; i++, cmyk += 4 )
        {
            int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
            c = k - ((255 - c)*k>>8);
            m = k - ((255 - m)*k>>8);
            y = k - ((255 - y)*k>>8);
            int t = descale( y*cB + m*cG + c*cR, SCALE );
            gray[i] = (uchar)t;
        }
        gray += gray_step;
        cmyk += cmyk_step - size.width*4;
    }
}


void CvtPaletteToGray( const PaletteEntry* palette, uchar* grayPalette, int entries )
{
    int i;
    for( i = 0; i < entries; i++ )
    {
        icvCvt_BGR2Gray_8u_C3C1R( (uchar*)(palette + i), 0, grayPalette + i, 0, Size(1,1) );
    }
}


void  FillGrayPalette( PaletteEntry* palette, int bpp, bool negative )
{
    int i, length = 1 << bpp;
    int xor_mask = negative ? 255 : 0;

    for( i = 0; i < length; i++ )
    {
        int val = (i * 255/(length - 1)) ^ xor_mask;
        palette[i].b = palette[i].g = palette[i].r = (uchar)val;
        palette[i].a = 0;
    }
}


bool  IsColorPalette( PaletteEntry* palette, int bpp )
{
    int i, length = 1 << bpp;

    for( i = 0; i < length; i++ )
    {
        if( palette[i].b != palette[i].g ||
            palette[i].b != palette[i].r )
            return true;
    }

    return false;
}


uchar* FillUniColor( uchar* data, uchar*& line_end,
                     int step, int width3,
                     int& y, int height,
                     int count3, PaletteEntry clr )
{
    do
    {
        uchar* end = data + count3;

        if( end > line_end )
            end = line_end;

        count3 -= (int)(end - data);

        for( ; data < end; data += 3 )
        {
            WRITE_PIX( data, clr );
        }

        if( data >= line_end )
        {
            line_end += step;
            data = line_end - width3;
            if( ++y >= height  ) break;
        }
    }
    while( count3 > 0 );

    return data;
}


uchar* FillUniGray( uchar* data, uchar*& line_end,
                    int step, int width,
                    int& y, int height,
                    int count, uchar clr )
{
    do
    {
        uchar* end = data + count;

        if( end > line_end )
            end = line_end;

        count -= (int)(end - data);

        for( ; data < end; data++ )
        {
            *data = clr;
        }

        if( data >= line_end )
        {
            line_end += step;
            data = line_end - width;
            if( ++y >= height  ) break;
        }
    }
    while( count > 0 );

    return data;
}


uchar* FillColorRow8( uchar* data, uchar* indices, int len, PaletteEntry* palette )
{
    uchar* end = data + len*3;
    while( (data += 3) < end )
    {
        *((PaletteEntry*)(data-3)) = palette[*indices++];
    }
    PaletteEntry clr = palette[indices[0]];
    WRITE_PIX( data - 3, clr );
    return data;
}


uchar* FillGrayRow8( uchar* data, uchar* indices, int len, uchar* palette )
{
    int i;
    for( i = 0; i < len; i++ )
    {
        data[i] = palette[indices[i]];
    }
    return data + len;
}


uchar* FillColorRow4( uchar* data, uchar* indices, int len, PaletteEntry* palette )
{
    uchar* end = data + len*3;

    while( (data += 6) < end )
    {
        int idx = *indices++;
        *((PaletteEntry*)(data-6)) = palette[idx >> 4];
        *((PaletteEntry*)(data-3)) = palette[idx & 15];
    }

    int idx = indices[0];
    PaletteEntry clr = palette[idx >> 4];
    WRITE_PIX( data - 6, clr );

    if( data == end )
    {
        clr = palette[idx & 15];
        WRITE_PIX( data - 3, clr );
    }
    return end;
}


uchar* FillGrayRow4( uchar* data, uchar* indices, int len, uchar* palette )
{
    uchar* end = data + len;
    while( (data += 2) < end )
    {
        int idx = *indices++;
        data[-2] = palette[idx >> 4];
        data[-1] = palette[idx & 15];
    }

    int idx = indices[0];
    uchar clr = palette[idx >> 4];
    data[-2] = clr;

    if( data == end )
    {
        clr = palette[idx & 15];
        data[-1] = clr;
    }
    return end;
}


uchar* FillColorRow1( uchar* data, uchar* indices, int len, PaletteEntry* palette )
{
    uchar* end = data + len*3;

    const PaletteEntry p0 = palette[0], p1 = palette[1];

    while( (data += 24) < end )
    {
        int idx = *indices++;
        *((PaletteEntry*)(data - 24)) = (idx & 128) ? p1 : p0;
        *((PaletteEntry*)(data - 21)) = (idx & 64) ? p1 : p0;
        *((PaletteEntry*)(data - 18)) = (idx & 32) ? p1 : p0;
        *((PaletteEntry*)(data - 15)) = (idx & 16) ? p1 : p0;
        *((PaletteEntry*)(data - 12)) = (idx & 8) ? p1 : p0;
        *((PaletteEntry*)(data - 9)) = (idx & 4) ? p1 : p0;
        *((PaletteEntry*)(data - 6)) = (idx & 2) ? p1 : p0;
        *((PaletteEntry*)(data - 3)) = (idx & 1) ? p1 : p0;
    }

    int idx = indices[0];
    for( data -= 24; data < end; data += 3, idx += idx )
    {
        const PaletteEntry clr = (idx & 128) ? p1 : p0;
        WRITE_PIX( data, clr );
    }

    return data;
}


uchar* FillGrayRow1( uchar* data, uchar* indices, int len, uchar* palette )
{
    uchar* end = data + len;

    const uchar p0 = palette[0], p1 = palette[1];

    while( (data += 8) < end )
    {
        int idx = *indices++;
        *((uchar*)(data - 8)) = (idx & 128) ? p1 : p0;
        *((uchar*)(data - 7)) = (idx & 64) ? p1 : p0;
        *((uchar*)(data - 6)) = (idx & 32) ? p1 : p0;
        *((uchar*)(data - 5)) = (idx & 16) ? p1 : p0;
        *((uchar*)(data - 4)) = (idx & 8) ? p1 : p0;
        *((uchar*)(data - 3)) = (idx & 4) ? p1 : p0;
        *((uchar*)(data - 2)) = (idx & 2) ? p1 : p0;
        *((uchar*)(data - 1)) = (idx & 1) ? p1 : p0;
    }

    int idx = indices[0];
    for( data -= 8; data < end; data++, idx += idx )
    {
        data[0] = (idx & 128) ? p1 : p0;
    }

    return data;
}

}  // namespace
