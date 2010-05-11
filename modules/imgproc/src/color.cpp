/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
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

/********************************* COPYRIGHT NOTICE *******************************\
  The function for RGB to Lab conversion is based on the MATLAB script
  RGB2Lab.m translated by Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
  See the page [http://vision.stanford.edu/~ruzon/software/rgblab.html]
\**********************************************************************************/

/********************************* COPYRIGHT NOTICE *******************************\
  Original code for Bayer->BGR/RGB conversion is provided by Dirk Schaefer
  from MD-Mathematische Dienste GmbH. Below is the copyright notice:

    IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
    By downloading, copying, installing or using the software you agree
    to this license. If you do not agree to this license, do not download,
    install, copy or use the software.

    Contributors License Agreement:

      Copyright (c) 2002,
      MD-Mathematische Dienste GmbH
      Im Defdahl 5-10
      44141 Dortmund
      Germany
      www.md-it.de
  
    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met: 

    Redistributions of source code must retain
    the above copyright notice, this list of conditions and the following disclaimer. 
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution. 
    The name of Contributor may not be used to endorse or promote products
    derived from this software without specific prior written permission. 

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************/

#include "precomp.hpp"

typedef CvStatus (CV_STDCALL * CvColorCvtFunc0)(
    const void* src, int srcstep, void* dst, int dststep, CvSize size );

typedef CvStatus (CV_STDCALL * CvColorCvtFunc1)(
    const void* src, int srcstep, void* dst, int dststep,
    CvSize size, int param0 );

typedef CvStatus (CV_STDCALL * CvColorCvtFunc2)(
    const void* src, int srcstep, void* dst, int dststep,
    CvSize size, int param0, int param1 );

typedef CvStatus (CV_STDCALL * CvColorCvtFunc3)(
    const void* src, int srcstep, void* dst, int dststep,
    CvSize size, int param0, int param1, int param2 );

/****************************************************************************************\
*                 Various 3/4-channel to 3/4-channel RGB transformations                 *
\****************************************************************************************/

#define CV_IMPL_BGRX2BGR( flavor, arrtype )                             \
static CvStatus CV_STDCALL                                              \
icvBGRx2BGR_##flavor##_CnC3R( const arrtype* src, int srcstep,          \
                              arrtype* dst, int dststep,                \
                              CvSize size, int src_cn, int blue_idx )   \
{                                                                       \
    int i;                                                              \
                                                                        \
    srcstep /= sizeof(src[0]);                                          \
    dststep /= sizeof(dst[0]);                                          \
    srcstep -= size.width*src_cn;                                       \
    size.width *= 3;                                                    \
                                                                        \
    for( ; size.height--; src += srcstep, dst += dststep )              \
    {                                                                   \
        for( i = 0; i < size.width; i += 3, src += src_cn )             \
        {                                                               \
            arrtype t0=src[blue_idx], t1=src[1], t2=src[blue_idx^2];    \
            dst[i] = t0;                                                \
            dst[i+1] = t1;                                              \
            dst[i+2] = t2;                                              \
        }                                                               \
    }                                                                   \
                                                                        \
    return CV_OK;                                                       \
}


#define CV_IMPL_BGR2BGRX( flavor, arrtype )                             \
static CvStatus CV_STDCALL                                              \
icvBGR2BGRx_##flavor##_C3C4R( const arrtype* src, int srcstep,          \
                              arrtype* dst, int dststep,                \
                              CvSize size, int blue_idx )               \
{                                                                       \
    int i;                                                              \
                                                                        \
    srcstep /= sizeof(src[0]);                                          \
    dststep /= sizeof(dst[0]);                                          \
    srcstep -= size.width*3;                                            \
    size.width *= 4;                                                    \
                                                                        \
    for( ; size.height--; src += srcstep, dst += dststep )              \
    {                                                                   \
        for( i = 0; i < size.width; i += 4, src += 3 )                  \
        {                                                               \
            arrtype t0=src[blue_idx], t1=src[1], t2=src[blue_idx^2];    \
            dst[i] = t0;                                                \
            dst[i+1] = t1;                                              \
            dst[i+2] = t2;                                              \
            dst[i+3] = 0;                                               \
        }                                                               \
    }                                                                   \
                                                                        \
    return CV_OK;                                                       \
}


#define CV_IMPL_BGRA2RGBA( flavor, arrtype )                            \
static CvStatus CV_STDCALL                                              \
icvBGRA2RGBA_##flavor##_C4R( const arrtype* src, int srcstep,           \
                             arrtype* dst, int dststep, CvSize size )   \
{                                                                       \
    int i;                                                              \
                                                                        \
    srcstep /= sizeof(src[0]);                                          \
    dststep /= sizeof(dst[0]);                                          \
    size.width *= 4;                                                    \
                                                                        \
    for( ; size.height--; src += srcstep, dst += dststep )              \
    {                                                                   \
        for( i = 0; i < size.width; i += 4 )                            \
        {                                                               \
            arrtype t0 = src[2], t1 = src[1], t2 = src[0], t3 = src[3]; \
            dst[i] = t0;                                                \
            dst[i+1] = t1;                                              \
            dst[i+2] = t2;                                              \
            dst[i+3] = t3;                                              \
        }                                                               \
    }                                                                   \
                                                                        \
    return CV_OK;                                                       \
}


CV_IMPL_BGRX2BGR( 8u, uchar )
CV_IMPL_BGRX2BGR( 16u, ushort )
CV_IMPL_BGRX2BGR( 32f, int )
CV_IMPL_BGR2BGRX( 8u, uchar )
CV_IMPL_BGR2BGRX( 16u, ushort )
CV_IMPL_BGR2BGRX( 32f, int )
CV_IMPL_BGRA2RGBA( 8u, uchar )
CV_IMPL_BGRA2RGBA( 16u, ushort )
CV_IMPL_BGRA2RGBA( 32f, int )


/****************************************************************************************\
*           Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB          *
\****************************************************************************************/

static CvStatus CV_STDCALL
icvBGR5x52BGRx_8u_C2CnR( const uchar* src, int srcstep,
                         uchar* dst, int dststep,
                         CvSize size, int dst_cn,
                         int blue_idx, int green_bits )
{
    int i;
    assert( green_bits == 5 || green_bits == 6 );
    dststep -= size.width*dst_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        if( green_bits == 6 )
            for( i = 0; i < size.width; i++, dst += dst_cn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[blue_idx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 3) & ~3);
                dst[blue_idx ^ 2] = (uchar)((t >> 8) & ~7);
                if( dst_cn == 4 )
                    dst[3] = 0;
            }
        else
            for( i = 0; i < size.width; i++, dst += dst_cn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[blue_idx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 2) & ~7);
                dst[blue_idx ^ 2] = (uchar)((t >> 7) & ~7);
                if( dst_cn == 4 )
                    dst[3] = 0;
            }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvBGRx2BGR5x5_8u_CnC2R( const uchar* src, int srcstep,
                         uchar* dst, int dststep,
                         CvSize size, int src_cn,
                         int blue_idx, int green_bits )
{
    int i;
    srcstep -= size.width*src_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        if( green_bits == 6 )
            for( i = 0; i < size.width; i++, src += src_cn )
            {
                int t = (src[blue_idx] >> 3)|((src[1]&~3) << 3)|((src[blue_idx^2]&~7) << 8);
                ((ushort*)dst)[i] = (ushort)t;
            }
        else
            for( i = 0; i < size.width; i++, src += src_cn )
            {
                int t = (src[blue_idx] >> 3)|((src[1]&~7) << 2)|((src[blue_idx^2]&~7) << 7);
                ((ushort*)dst)[i] = (ushort)t;
            }
    }

    return CV_OK;
}



/////////////////////////// IPP Color Conversion Functions //////////////////////////////

#define CV_IMPL_BGRx2ABC_IPP( flavor, arrtype )                         \
static CvStatus CV_STDCALL                                              \
icvBGRx2ABC_IPP_##flavor##_CnC3R( const arrtype* src, int srcstep,      \
    arrtype* dst, int dststep, CvSize size, int src_cn,                 \
    int blue_idx, CvColorCvtFunc0 ipp_func )                            \
{                                                                       \
    int block_size = MIN(1 << 14, size.width);                          \
    arrtype* buffer;                                                    \
    int i, di, k;                                                       \
    int do_copy = src_cn > 3 || blue_idx != 2 || src == dst;            \
    CvStatus status = CV_OK;                                            \
                                                                        \
    if( !do_copy )                                                      \
        return ipp_func( src, srcstep, dst, dststep, size );            \
                                                                        \
    srcstep /= sizeof(src[0]);                                          \
    dststep /= sizeof(dst[0]);                                          \
                                                                        \
    buffer = (arrtype*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );  \
    srcstep -= size.width*src_cn;                                       \
                                                                        \
    for( ; size.height--; src += srcstep, dst += dststep )              \
    {                                                                   \
        for( i = 0; i < size.width; i += block_size )                   \
        {                                                               \
            arrtype* dst1 = dst + i*3;                                  \
            di = MIN(block_size, size.width - i);                       \
                                                                        \
            for( k = 0; k < di*3; k += 3, src += src_cn )               \
            {                                                           \
                arrtype b = src[blue_idx];                              \
                arrtype g = src[1];                                     \
                arrtype r = src[blue_idx^2];                            \
                buffer[k] = r;                                          \
                buffer[k+1] = g;                                        \
                buffer[k+2] = b;                                        \
            }                                                           \
                                                                        \
            status = ipp_func( buffer, CV_STUB_STEP,                    \
                               dst1, CV_STUB_STEP, cvSize(di,1) );      \
            if( status < 0 )                                            \
                return status;                                          \
        }                                                               \
    }                                                                   \
                                                                        \
    return CV_OK;                                                       \
}

#ifdef HAVE_IPP
static CvStatus CV_STDCALL
icvBGRx2ABC_IPP_8u_CnC3R( const uchar* src, int srcstep,
    uchar* dst, int dststep, CvSize size, int src_cn,
    int blue_idx, CvColorCvtFunc0 ipp_func )
{
    int block_size = MIN(1 << 14, size.width);
    uchar* buffer;
    int i, di, k;
    int do_copy = src_cn > 3 || blue_idx != 2 || src == dst;
    CvStatus status = CV_OK;

    if( !do_copy )
        return ipp_func( src, srcstep, dst, dststep, size );

    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);

    buffer = (uchar*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );
    srcstep -= size.width*src_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += block_size )
        {
            uchar* dst1 = dst + i*3;
            di = MIN(block_size, size.width - i);

            for( k = 0; k < di*3; k += 3, src += src_cn )
            {
                uchar b = src[blue_idx];
                uchar g = src[1];
                uchar r = src[blue_idx^2];
                buffer[k] = r;
                buffer[k+1] = g;
                buffer[k+2] = b;
            }

            status = ipp_func( buffer, CV_STUB_STEP,
                               dst1, CV_STUB_STEP, cvSize(di,1) );
            if( status < 0 )
                return status;
        }
    }

    return CV_OK;
}

//CV_IMPL_BGRx2ABC_IPP( 8u, uchar )
//CV_IMPL_BGRx2ABC_IPP( 16u, ushort )
CV_IMPL_BGRx2ABC_IPP( 32f, float )

#define CV_IMPL_ABC2BGRx_IPP( flavor, arrtype )                         \
static CvStatus CV_STDCALL                                              \
icvABC2BGRx_IPP_##flavor##_C3CnR( const arrtype* src, int srcstep,      \
    arrtype* dst, int dststep, CvSize size, int dst_cn,                 \
    int blue_idx, CvColorCvtFunc0 ipp_func )                            \
{                                                                       \
    int block_size = MIN(1 << 10, size.width);                          \
    arrtype* buffer;                                                    \
    int i, di, k;                                                       \
    int do_copy = dst_cn > 3 || blue_idx != 2 || src == dst;            \
    CvStatus status = CV_OK;                                            \
                                                                        \
    if( !do_copy )                                                      \
        return ipp_func( src, srcstep, dst, dststep, size );            \
                                                                        \
    srcstep /= sizeof(src[0]);                                          \
    dststep /= sizeof(dst[0]);                                          \
                                                                        \
    buffer = (arrtype*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );  \
    dststep -= size.width*dst_cn;                                       \
                                                                        \
    for( ; size.height--; src += srcstep, dst += dststep )              \
    {                                                                   \
        for( i = 0; i < size.width; i += block_size )                   \
        {                                                               \
            const arrtype* src1 = src + i*3;                            \
            di = MIN(block_size, size.width - i);                       \
                                                                        \
            status = ipp_func( src1, CV_STUB_STEP,                      \
                               buffer, CV_STUB_STEP, cvSize(di,1) );    \
            if( status < 0 )                                            \
                return status;                                          \
                                                                        \
            for( k = 0; k < di*3; k += 3, dst += dst_cn )               \
            {                                                           \
                arrtype r = buffer[k];                                  \
                arrtype g = buffer[k+1];                                \
                arrtype b = buffer[k+2];                                \
                dst[blue_idx] = b;                                      \
                dst[1] = g;                                             \
                dst[blue_idx^2] = r;                                    \
                if( dst_cn == 4 )                                       \
                    dst[3] = 0;                                         \
            }                                                           \
        }                                                               \
    }                                                                   \
                                                                        \
    return CV_OK;                                                       \
}

CV_IMPL_ABC2BGRx_IPP( 8u, uchar )
//CV_IMPL_ABC2BGRx_IPP( 16u, ushort )
CV_IMPL_ABC2BGRx_IPP( 32f, float )
#endif

/////////////////////////////////////////////////////////////////////////////////////////


/****************************************************************************************\
*                                 Color to/from Grayscale                                *
\****************************************************************************************/

#define fix(x,n)      (int)((x)*(1 << (n)) + 0.5)
#define descale       CV_DESCALE

#define cscGr_32f  0.299f
#define cscGg_32f  0.587f
#define cscGb_32f  0.114f

/* BGR/RGB -> Gray */
#define csc_shift  14
#define cscGr  fix(cscGr_32f,csc_shift) 
#define cscGg  fix(cscGg_32f,csc_shift)
#define cscGb  /*fix(cscGb_32f,csc_shift)*/ ((1 << csc_shift) - cscGr - cscGg)

#define CV_IMPL_GRAY2BGRX( flavor, arrtype )                    \
static CvStatus CV_STDCALL                                      \
icvGray2BGRx_##flavor##_C1CnR( const arrtype* src, int srcstep, \
                       arrtype* dst, int dststep, CvSize size,  \
                       int dst_cn )                             \
{                                                               \
    int i;                                                      \
    srcstep /= sizeof(src[0]);                                  \
    dststep /= sizeof(src[0]);                                  \
    dststep -= size.width*dst_cn;                               \
                                                                \
    for( ; size.height--; src += srcstep, dst += dststep )      \
    {                                                           \
        if( dst_cn == 3 )                                       \
            for( i = 0; i < size.width; i++, dst += 3 )         \
                dst[0] = dst[1] = dst[2] = src[i];              \
        else                                                    \
            for( i = 0; i < size.width; i++, dst += 4 )         \
            {                                                   \
                dst[0] = dst[1] = dst[2] = src[i];              \
                dst[3] = 0;                                     \
            }                                                   \
    }                                                           \
                                                                \
    return CV_OK;                                               \
}


CV_IMPL_GRAY2BGRX( 8u, uchar )
CV_IMPL_GRAY2BGRX( 16u, ushort )
CV_IMPL_GRAY2BGRX( 32f, float )


static CvStatus CV_STDCALL
icvBGR5x52Gray_8u_C2C1R( const uchar* src, int srcstep,
                         uchar* dst, int dststep,
                         CvSize size, int green_bits )
{
    int i;
    assert( green_bits == 5 || green_bits == 6 );

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        if( green_bits == 6 )
            for( i = 0; i < size.width; i++ )
            {
                int t = ((ushort*)src)[i];
                t = ((t << 3) & 0xf8)*cscGb + ((t >> 3) & 0xfc)*cscGg +
                    ((t >> 8) & 0xf8)*cscGr;
                dst[i] = (uchar)CV_DESCALE(t,csc_shift);
            }
        else
            for( i = 0; i < size.width; i++ )
            {
                int t = ((ushort*)src)[i];
                t = ((t << 3) & 0xf8)*cscGb + ((t >> 2) & 0xf8)*cscGg +
                    ((t >> 7) & 0xf8)*cscGr;
                dst[i] = (uchar)CV_DESCALE(t,csc_shift);
            }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvGray2BGR5x5_8u_C1C2R( const uchar* src, int srcstep,
                         uchar* dst, int dststep,
                         CvSize size, int green_bits )
{
    int i;
    assert( green_bits == 5 || green_bits == 6 );

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        if( green_bits == 6 )
            for( i = 0; i < size.width; i++ )
            {
                int t = src[i];
                ((ushort*)dst)[i] = (ushort)((t >> 3)|((t & ~3) << 3)|((t & ~7) << 8));
            }
        else
            for( i = 0; i < size.width; i++ )
            {
                int t = src[i] >> 3;
                ((ushort*)dst)[i] = (ushort)(t|(t << 5)|(t << 10));
            }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvBGRx2Gray_8u_CnC1R( const uchar* src, int srcstep,
                       uchar* dst, int dststep, CvSize size,
                       int src_cn, int blue_idx )
{
    int i;
    srcstep -= size.width*src_cn;

    if( size.width*size.height >= 1024 )
    {
        int* tab = (int*)cvStackAlloc( 256*3*sizeof(tab[0]) );
        int r = 0, g = 0, b = (1 << (csc_shift-1));
    
        for( i = 0; i < 256; i++ )
        {
            tab[i] = b;
            tab[i+256] = g;
            tab[i+512] = r;
            g += cscGg;
            if( !blue_idx )
                b += cscGb, r += cscGr;
            else
                b += cscGr, r += cscGb;
        }

        for( ; size.height--; src += srcstep, dst += dststep )
        {
            for( i = 0; i < size.width; i++, src += src_cn )
            {
                int t0 = tab[src[0]] + tab[src[1] + 256] + tab[src[2] + 512];
                dst[i] = (uchar)(t0 >> csc_shift);
            }
        }
    }
    else
    {
        for( ; size.height--; src += srcstep, dst += dststep )
        {
            for( i = 0; i < size.width; i++, src += src_cn )
            {
                int t0 = src[blue_idx]*cscGb + src[1]*cscGg + src[blue_idx^2]*cscGr;
                dst[i] = (uchar)CV_DESCALE(t0, csc_shift);
            }
        }
    }
    return CV_OK;
}


static CvStatus CV_STDCALL
icvBGRx2Gray_16u_CnC1R( const ushort* src, int srcstep,
                        ushort* dst, int dststep, CvSize size,
                        int src_cn, int blue_idx )
{
    int i;
    int cb = cscGb, cr = cscGr;
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    srcstep -= size.width*src_cn;

    if( blue_idx )
        cb = cscGr, cr = cscGb;

    for( ; size.height--; src += srcstep, dst += dststep )
        for( i = 0; i < size.width; i++, src += src_cn )
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb +
                    src[1]*cscGg + src[2]*cr), csc_shift);

    return CV_OK;
}


static CvStatus CV_STDCALL
icvBGRx2Gray_32f_CnC1R( const float* src, int srcstep,
                        float* dst, int dststep, CvSize size,
                        int src_cn, int blue_idx )
{
    int i;
    float cb = cscGb_32f, cr = cscGr_32f;
    if( blue_idx )
        cb = cscGr_32f, cr = cscGb_32f;

    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    srcstep -= size.width*src_cn;
    for( ; size.height--; src += srcstep, dst += dststep )
        for( i = 0; i < size.width; i++, src += src_cn )
            dst[i] = src[0]*cb + src[1]*cscGg_32f + src[2]*cr;

    return CV_OK;
}


/****************************************************************************************\
*                                     RGB <-> YCrCb                                      *
\****************************************************************************************/

/* BGR/RGB -> YCrCb */
#define yuvYr_32f cscGr_32f
#define yuvYg_32f cscGg_32f
#define yuvYb_32f cscGb_32f
#define yuvCr_32f 0.713f
#define yuvCb_32f 0.564f

#define yuv_shift 14
#define yuvYr  fix(yuvYr_32f,yuv_shift)
#define yuvYg  fix(yuvYg_32f,yuv_shift)
#define yuvYb  fix(yuvYb_32f,yuv_shift)
#define yuvCr  fix(yuvCr_32f,yuv_shift)
#define yuvCb  fix(yuvCb_32f,yuv_shift)

#define yuv_descale(x)  CV_DESCALE((x), yuv_shift)
#define yuv_prescale(x) ((x) << yuv_shift)

#define  yuvRCr_32f   1.403f
#define  yuvGCr_32f   (-0.714f)
#define  yuvGCb_32f   (-0.344f)
#define  yuvBCb_32f   1.773f

#define  yuvRCr   fix(yuvRCr_32f,yuv_shift)
#define  yuvGCr   (-fix(-yuvGCr_32f,yuv_shift))
#define  yuvGCb   (-fix(-yuvGCb_32f,yuv_shift))
#define  yuvBCb   fix(yuvBCb_32f,yuv_shift)

#define CV_IMPL_BGRx2YCrCb( flavor, arrtype, worktype, scale_macro, cast_macro,     \
                            YUV_YB, YUV_YG, YUV_YR, YUV_CR, YUV_CB, YUV_Cx_BIAS )   \
static CvStatus CV_STDCALL                                                  \
icvBGRx2YCrCb_##flavor##_CnC3R( const arrtype* src, int srcstep,            \
    arrtype* dst, int dststep, CvSize size, int src_cn, int blue_idx )      \
{                                                                           \
    int i;                                                                  \
    srcstep /= sizeof(src[0]);                                              \
    dststep /= sizeof(src[0]);                                              \
    srcstep -= size.width*src_cn;                                           \
    size.width *= 3;                                                        \
                                                                            \
    for( ; size.height--; src += srcstep, dst += dststep )                  \
    {                                                                       \
        for( i = 0; i < size.width; i += 3, src += src_cn )                 \
        {                                                                   \
            worktype b = src[blue_idx], r = src[2^blue_idx], y;             \
            y = scale_macro(b*YUV_YB + src[1]*YUV_YG + r*YUV_YR);           \
            r = scale_macro((r - y)*YUV_CR) + YUV_Cx_BIAS;                  \
            b = scale_macro((b - y)*YUV_CB) + YUV_Cx_BIAS;                  \
            dst[i] = cast_macro(y);                                         \
            dst[i+1] = cast_macro(r);                                       \
            dst[i+2] = cast_macro(b);                                       \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}


CV_IMPL_BGRx2YCrCb( 8u, uchar, int, yuv_descale, CV_CAST_8U,
                    yuvYb, yuvYg, yuvYr, yuvCr, yuvCb, 128 )

CV_IMPL_BGRx2YCrCb( 16u, ushort, int, yuv_descale, CV_CAST_16U,
                    yuvYb, yuvYg, yuvYr, yuvCr, yuvCb, 32768 )

CV_IMPL_BGRx2YCrCb( 32f, float, float, CV_NOP, CV_NOP,
                    yuvYb_32f, yuvYg_32f, yuvYr_32f, yuvCr_32f, yuvCb_32f, 0.5f )


#define CV_IMPL_YCrCb2BGRx( flavor, arrtype, worktype, prescale_macro,      \
    scale_macro, cast_macro, YUV_BCb, YUV_GCr, YUV_GCb, YUV_RCr, YUV_Cx_BIAS)\
static CvStatus CV_STDCALL                                                  \
icvYCrCb2BGRx_##flavor##_C3CnR( const arrtype* src, int srcstep,            \
                                arrtype* dst, int dststep, CvSize size,     \
                                int dst_cn, int blue_idx )                  \
{                                                                           \
    int i;                                                                  \
    srcstep /= sizeof(src[0]);                                              \
    dststep /= sizeof(src[0]);                                              \
    dststep -= size.width*dst_cn;                                           \
    size.width *= 3;                                                        \
                                                                            \
    for( ; size.height--; src += srcstep, dst += dststep )                  \
    {                                                                       \
        for( i = 0; i < size.width; i += 3, dst += dst_cn )                 \
        {                                                                   \
            worktype Y = prescale_macro(src[i]),                            \
                     Cr = src[i+1] - YUV_Cx_BIAS,                           \
                     Cb = src[i+2] - YUV_Cx_BIAS;                           \
            worktype b, g, r;                                               \
            b = scale_macro( Y + YUV_BCb*Cb );                              \
            g = scale_macro( Y + YUV_GCr*Cr + YUV_GCb*Cb );                 \
            r = scale_macro( Y + YUV_RCr*Cr );                              \
                                                                            \
            dst[blue_idx] = cast_macro(b);                                  \
            dst[1] = cast_macro(g);                                         \
            dst[blue_idx^2] = cast_macro(r);                                \
            if( dst_cn == 4 )                                               \
                dst[3] = 0;                                                 \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}


CV_IMPL_YCrCb2BGRx( 8u, uchar, int, yuv_prescale, yuv_descale, CV_CAST_8U,
                    yuvBCb, yuvGCr, yuvGCb, yuvRCr, 128 )

CV_IMPL_YCrCb2BGRx( 16u, ushort, int, yuv_prescale, yuv_descale, CV_CAST_16U,
                    yuvBCb, yuvGCr, yuvGCb, yuvRCr, 32768 )

CV_IMPL_YCrCb2BGRx( 32f, float, float, CV_NOP, CV_NOP, CV_NOP,
                    yuvBCb_32f, yuvGCr_32f, yuvGCb_32f, yuvRCr_32f, 0.5f )


/****************************************************************************************\
*                                      RGB <-> XYZ                                       *
\****************************************************************************************/

#define xyzXr_32f  0.412453f
#define xyzXg_32f  0.357580f
#define xyzXb_32f  0.180423f

#define xyzYr_32f  0.212671f
#define xyzYg_32f  0.715160f
#define xyzYb_32f  0.072169f

#define xyzZr_32f  0.019334f
#define xyzZg_32f  0.119193f
#define xyzZb_32f  0.950227f

#define xyzRx_32f  3.240479f
#define xyzRy_32f  (-1.53715f)
#define xyzRz_32f  (-0.498535f)

#define xyzGx_32f  (-0.969256f)
#define xyzGy_32f  1.875991f
#define xyzGz_32f  0.041556f

#define xyzBx_32f  0.055648f
#define xyzBy_32f  (-0.204043f)
#define xyzBz_32f  1.057311f

#define xyz_shift  10
#define xyzXr_32s  fix(xyzXr_32f, xyz_shift )
#define xyzXg_32s  fix(xyzXg_32f, xyz_shift )
#define xyzXb_32s  fix(xyzXb_32f, xyz_shift )

#define xyzYr_32s  fix(xyzYr_32f, xyz_shift )
#define xyzYg_32s  fix(xyzYg_32f, xyz_shift )
#define xyzYb_32s  fix(xyzYb_32f, xyz_shift )

#define xyzZr_32s  fix(xyzZr_32f, xyz_shift )
#define xyzZg_32s  fix(xyzZg_32f, xyz_shift )
#define xyzZb_32s  fix(xyzZb_32f, xyz_shift )

#define xyzRx_32s  fix(3.240479f, xyz_shift )
#define xyzRy_32s  -fix(1.53715f, xyz_shift )
#define xyzRz_32s  -fix(0.498535f, xyz_shift )

#define xyzGx_32s  -fix(0.969256f, xyz_shift )
#define xyzGy_32s  fix(1.875991f, xyz_shift )
#define xyzGz_32s  fix(0.041556f, xyz_shift )

#define xyzBx_32s  fix(0.055648f, xyz_shift )
#define xyzBy_32s  -fix(0.204043f, xyz_shift )
#define xyzBz_32s  fix(1.057311f, xyz_shift )

#define xyz_descale(x) CV_DESCALE((x),xyz_shift)

#define CV_IMPL_BGRx2XYZ( flavor, arrtype, worktype,                        \
                          scale_macro, cast_macro, suffix )                 \
static CvStatus CV_STDCALL                                                  \
icvBGRx2XYZ_##flavor##_CnC3R( const arrtype* src, int srcstep,              \
                              arrtype* dst, int dststep, CvSize size,       \
                              int src_cn, int blue_idx )                    \
{                                                                           \
    int i;                                                                  \
    worktype t, matrix[] =                                                  \
    {                                                                       \
        xyzXb##suffix, xyzXg##suffix, xyzXr##suffix,                        \
        xyzYb##suffix, xyzYg##suffix, xyzYr##suffix,                        \
        xyzZb##suffix, xyzZg##suffix, xyzZr##suffix                         \
    };                                                                      \
                                                                            \
    srcstep /= sizeof(src[0]);                                              \
    dststep /= sizeof(dst[0]);                                              \
    srcstep -= size.width*src_cn;                                           \
    size.width *= 3;                                                        \
                                                                            \
    if( blue_idx )                                                          \
    {                                                                       \
        CV_SWAP( matrix[0], matrix[2], t );                                 \
        CV_SWAP( matrix[3], matrix[5], t );                                 \
        CV_SWAP( matrix[6], matrix[8], t );                                 \
    }                                                                       \
                                                                            \
    for( ; size.height--; src += srcstep, dst += dststep )                  \
    {                                                                       \
        for( i = 0; i < size.width; i += 3, src += src_cn )                 \
        {                                                                   \
            worktype x = scale_macro(src[0]*matrix[0] +                     \
                    src[1]*matrix[1] + src[2]*matrix[2]);                   \
            worktype y = scale_macro(src[0]*matrix[3] +                     \
                    src[1]*matrix[4] + src[2]*matrix[5]);                   \
            worktype z = scale_macro(src[0]*matrix[6] +                     \
                    src[1]*matrix[7] + src[2]*matrix[8]);                   \
                                                                            \
            dst[i] = (arrtype)(x);                                          \
            dst[i+1] = (arrtype)(y);                                        \
            dst[i+2] = cast_macro(z); /*sum of weights for z > 1*/          \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}


CV_IMPL_BGRx2XYZ( 8u, uchar, int, xyz_descale, CV_CAST_8U, _32s )
CV_IMPL_BGRx2XYZ( 16u, ushort, int, xyz_descale, CV_CAST_16U, _32s )
CV_IMPL_BGRx2XYZ( 32f, float, float, CV_NOP, CV_NOP, _32f )


#define CV_IMPL_XYZ2BGRx( flavor, arrtype, worktype, scale_macro,           \
                          cast_macro, suffix )                              \
static CvStatus CV_STDCALL                                                  \
icvXYZ2BGRx_##flavor##_C3CnR( const arrtype* src, int srcstep,              \
                              arrtype* dst, int dststep, CvSize size,       \
                              int dst_cn, int blue_idx )                    \
{                                                                           \
    int i;                                                                  \
    worktype t, matrix[] =                                                  \
    {                                                                       \
        xyzBx##suffix, xyzBy##suffix, xyzBz##suffix,                        \
        xyzGx##suffix, xyzGy##suffix, xyzGz##suffix,                        \
        xyzRx##suffix, xyzRy##suffix, xyzRz##suffix                         \
    };                                                                      \
                                                                            \
    srcstep /= sizeof(src[0]);                                              \
    dststep /= sizeof(dst[0]);                                              \
    dststep -= size.width*dst_cn;                                           \
    size.width *= 3;                                                        \
                                                                            \
    if( blue_idx )                                                          \
    {                                                                       \
        CV_SWAP( matrix[0], matrix[6], t );                                 \
        CV_SWAP( matrix[1], matrix[7], t );                                 \
        CV_SWAP( matrix[2], matrix[8], t );                                 \
    }                                                                       \
                                                                            \
    for( ; size.height--; src += srcstep, dst += dststep )                  \
    {                                                                       \
        for( i = 0; i < size.width; i += 3, dst += dst_cn )                 \
        {                                                                   \
            worktype b = scale_macro(src[i]*matrix[0] +                     \
                    src[i+1]*matrix[1] + src[i+2]*matrix[2]);               \
            worktype g = scale_macro(src[i]*matrix[3] +                     \
                    src[i+1]*matrix[4] + src[i+2]*matrix[5]);               \
            worktype r = scale_macro(src[i]*matrix[6] +                     \
                    src[i+1]*matrix[7] + src[i+2]*matrix[8]);               \
                                                                            \
            dst[0] = cast_macro(b);                                         \
            dst[1] = cast_macro(g);                                         \
            dst[2] = cast_macro(r);                                         \
                                                                            \
            if( dst_cn == 4 )                                               \
                dst[3] = 0;                                                 \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}

CV_IMPL_XYZ2BGRx( 8u, uchar, int, xyz_descale, CV_CAST_8U, _32s )
CV_IMPL_XYZ2BGRx( 16u, ushort, int, xyz_descale, CV_CAST_16U, _32s )
CV_IMPL_XYZ2BGRx( 32f, float, float, CV_NOP, CV_NOP, _32f )


/****************************************************************************************\
*                          Non-linear Color Space Transformations                        *
\****************************************************************************************/

#ifndef HAVE_IPP
// driver color space conversion function for 8u arrays that uses 32f function
// with appropriate pre- and post-scaling.
static CvStatus CV_STDCALL
icvABC2BGRx_8u_C3CnR( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int dst_cn, int blue_idx, CvColorCvtFunc2 cvtfunc_32f,
                     const float* pre_coeffs, int postscale )
{
    int block_size = MIN(1 << 8, size.width);
    float* buffer = (float*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );
    int i, di, k;
    CvStatus status = CV_OK;

    dststep -= size.width*dst_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += block_size )
        {
            const uchar* src1 = src + i*3;
            di = MIN(block_size, size.width - i);
            
            for( k = 0; k < di*3; k += 3 )
            {
                float a = CV_8TO32F(src1[k])*pre_coeffs[0] + pre_coeffs[1];
                float b = CV_8TO32F(src1[k+1])*pre_coeffs[2] + pre_coeffs[3];
                float c = CV_8TO32F(src1[k+2])*pre_coeffs[4] + pre_coeffs[5];
                buffer[k] = a;
                buffer[k+1] = b;
                buffer[k+2] = c;
            }

            status = cvtfunc_32f( buffer, 0, buffer, 0, cvSize(di,1), 3, blue_idx );
            if( status < 0 )
                return status;
            
            if( postscale )
            {
                for( k = 0; k < di*3; k += 3, dst += dst_cn )
                {
                    int b = cvRound(buffer[k]*255.);
                    int g = cvRound(buffer[k+1]*255.);
                    int r = cvRound(buffer[k+2]*255.);

                    dst[0] = CV_CAST_8U(b);
                    dst[1] = CV_CAST_8U(g);
                    dst[2] = CV_CAST_8U(r);
                    if( dst_cn == 4 )
                        dst[3] = 0;
                }
            }
            else
            {
                for( k = 0; k < di*3; k += 3, dst += dst_cn )
                {
                    int b = cvRound(buffer[k]);
                    int g = cvRound(buffer[k+1]);
                    int r = cvRound(buffer[k+2]);

                    dst[0] = CV_CAST_8U(b);
                    dst[1] = CV_CAST_8U(g);
                    dst[2] = CV_CAST_8U(r);
                    if( dst_cn == 4 )
                        dst[3] = 0;
                }
            }
        }
    }

    return CV_OK;
}


// driver color space conversion function for 8u arrays that uses 32f function
// with appropriate pre- and post-scaling.
static CvStatus CV_STDCALL
icvBGRx2ABC_8u_CnC3R( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int src_cn, int blue_idx, CvColorCvtFunc2 cvtfunc_32f,
                      int prescale, const float* post_coeffs )
{
    int block_size = MIN(1 << 8, size.width);
    float* buffer = (float*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );
    int i, di, k;
    CvStatus status = CV_OK;

    srcstep -= size.width*src_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += block_size )
        {
            uchar* dst1 = dst + i*3;
            di = MIN(block_size, size.width - i);

            if( prescale )
            {
                for( k = 0; k < di*3; k += 3, src += src_cn )
                {
                    float b = CV_8TO32F(src[0])*0.0039215686274509803f;
                    float g = CV_8TO32F(src[1])*0.0039215686274509803f;
                    float r = CV_8TO32F(src[2])*0.0039215686274509803f;

                    buffer[k] = b;
                    buffer[k+1] = g;
                    buffer[k+2] = r;
                }
            }
            else
            {
                for( k = 0; k < di*3; k += 3, src += src_cn )
                {
                    float b = CV_8TO32F(src[0]);
                    float g = CV_8TO32F(src[1]);
                    float r = CV_8TO32F(src[2]);

                    buffer[k] = b;
                    buffer[k+1] = g;
                    buffer[k+2] = r;
                }
            }
            
            status = cvtfunc_32f( buffer, 0, buffer, 0, cvSize(di,1), 3, blue_idx );
            if( status < 0 )
                return status;

            for( k = 0; k < di*3; k += 3 )
            {
                int a = cvRound( buffer[k]*post_coeffs[0] + post_coeffs[1] );
                int b = cvRound( buffer[k+1]*post_coeffs[2] + post_coeffs[3] );
                int c = cvRound( buffer[k+2]*post_coeffs[4] + post_coeffs[5] );
                dst1[k] = CV_CAST_8U(a);
                dst1[k+1] = CV_CAST_8U(b);
                dst1[k+2] = CV_CAST_8U(c);
            }
        }
    }

    return CV_OK;
}
#endif

/****************************************************************************************\
*                                      RGB <-> HSV                                       *
\****************************************************************************************/

static const uchar icvHue255To180[] =
{
      0,   1,   1,   2,   3,   4,   4,   5,   6,   6,   7,   8,   8,   9,  10,  11,
     11,  12,  13,  13,  14,  15,  16,  16,  17,  18,  18,  19,  20,  20,  21,  22,
     23,  23,  24,  25,  25,  26,  27,  28,  28,  29,  30,  30,  31,  32,  32,  33,
     34,  35,  35,  36,  37,  37,  38,  39,  40,  40,  41,  42,  42,  43,  44,  44,
     45,  46,  47,  47,  48,  49,  49,  50,  51,  52,  52,  53,  54,  54,  55,  56,
     56,  57,  58,  59,  59,  60,  61,  61,  62,  63,  64,  64,  65,  66,  66,  67,
     68,  68,  69,  70,  71,  71,  72,  73,  73,  74,  75,  76,  76,  77,  78,  78,
     79,  80,  80,  81,  82,  83,  83,  84,  85,  85,  86,  87,  88,  88,  89,  90,
     90,  91,  92,  92,  93,  94,  95,  95,  96,  97,  97,  98,  99, 100, 100, 101,
    102, 102, 103, 104, 104, 105, 106, 107, 107, 108, 109, 109, 110, 111, 112, 112,
    113, 114, 114, 115, 116, 116, 117, 118, 119, 119, 120, 121, 121, 122, 123, 124,
    124, 125, 126, 126, 127, 128, 128, 129, 130, 131, 131, 132, 133, 133, 134, 135,
    136, 136, 137, 138, 138, 139, 140, 140, 141, 142, 143, 143, 144, 145, 145, 146,
    147, 148, 148, 149, 150, 150, 151, 152, 152, 153, 154, 155, 155, 156, 157, 157,
    158, 159, 160, 160, 161, 162, 162, 163, 164, 164, 165, 166, 167, 167, 168, 169,
    169, 170, 171, 172, 172, 173, 174, 174, 175, 176, 176, 177, 178, 179, 179, 180
};


static const uchar icvHue180To255[] =
{
      0,   1,   3,   4,   6,   7,   9,  10,  11,  13,  14,  16,  17,  18,  20,  21,
     23,  24,  26,  27,  28,  30,  31,  33,  34,  35,  37,  38,  40,  41,  43,  44,
     45,  47,  48,  50,  51,  52,  54,  55,  57,  58,  60,  61,  62,  64,  65,  67,
     68,  69,  71,  72,  74,  75,  77,  78,  79,  81,  82,  84,  85,  86,  88,  89,
     91,  92,  94,  95,  96,  98,  99, 101, 102, 103, 105, 106, 108, 109, 111, 112,
    113, 115, 116, 118, 119, 120, 122, 123, 125, 126, 128, 129, 130, 132, 133, 135,
    136, 137, 139, 140, 142, 143, 145, 146, 147, 149, 150, 152, 153, 154, 156, 157,
    159, 160, 162, 163, 164, 166, 167, 169, 170, 171, 173, 174, 176, 177, 179, 180,
    181, 183, 184, 186, 187, 188, 190, 191, 193, 194, 196, 197, 198, 200, 201, 203,
    204, 205, 207, 208, 210, 211, 213, 214, 215, 217, 218, 220, 221, 222, 224, 225,
    227, 228, 230, 231, 232, 234, 235, 237, 238, 239, 241, 242, 244, 245, 247, 248,
    249, 251, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};


static CvStatus CV_STDCALL
icvBGRx2HSV_8u_CnC3R( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int src_cn, int blue_idx )
{
    int i;

#ifdef HAVE_IPP
    CvStatus status = icvBGRx2ABC_IPP_8u_CnC3R( src, srcstep, dst, dststep, size,
                                            src_cn, blue_idx, (CvColorCvtFunc0)ippiRGBToHSV_8u_C3R );
    if( status >= 0 )
    {
        size.width *= 3;
        for( ; size.height--; dst += dststep )
        {
            for( i = 0; i <= size.width - 12; i += 12 )
            {
                uchar t0 = icvHue255To180[dst[i]], t1 = icvHue255To180[dst[i+3]];
                dst[i] = t0; dst[i+3] = t1;
                t0 = icvHue255To180[dst[i+6]]; t1 = icvHue255To180[dst[i+9]];
                dst[i+6] = t0; dst[i+9] = t1;
            }
            for( ; i < size.width; i += 3 )
                dst[i] = icvHue255To180[dst[i]];
        }
    }
    return status;
#else
    
    const int hsv_shift = 12;

    static const int div_table[] = {
        0, 1044480, 522240, 348160, 261120, 208896, 174080, 149211,
        130560, 116053, 104448, 94953, 87040, 80345, 74606, 69632,
        65280, 61440, 58027, 54973, 52224, 49737, 47476, 45412,
        43520, 41779, 40172, 38684, 37303, 36017, 34816, 33693,
        32640, 31651, 30720, 29842, 29013, 28229, 27486, 26782,
        26112, 25475, 24869, 24290, 23738, 23211, 22706, 22223,
        21760, 21316, 20890, 20480, 20086, 19707, 19342, 18991,
        18651, 18324, 18008, 17703, 17408, 17123, 16846, 16579,
        16320, 16069, 15825, 15589, 15360, 15137, 14921, 14711,
        14507, 14308, 14115, 13926, 13743, 13565, 13391, 13221,
        13056, 12895, 12738, 12584, 12434, 12288, 12145, 12006,
        11869, 11736, 11605, 11478, 11353, 11231, 11111, 10995,
        10880, 10768, 10658, 10550, 10445, 10341, 10240, 10141,
        10043, 9947, 9854, 9761, 9671, 9582, 9495, 9410,
        9326, 9243, 9162, 9082, 9004, 8927, 8852, 8777,
        8704, 8632, 8561, 8492, 8423, 8356, 8290, 8224,
        8160, 8097, 8034, 7973, 7913, 7853, 7795, 7737,
        7680, 7624, 7569, 7514, 7461, 7408, 7355, 7304,
        7253, 7203, 7154, 7105, 7057, 7010, 6963, 6917,
        6872, 6827, 6782, 6739, 6695, 6653, 6611, 6569,
        6528, 6487, 6447, 6408, 6369, 6330, 6292, 6254,
        6217, 6180, 6144, 6108, 6073, 6037, 6003, 5968,
        5935, 5901, 5868, 5835, 5803, 5771, 5739, 5708,
        5677, 5646, 5615, 5585, 5556, 5526, 5497, 5468,
        5440, 5412, 5384, 5356, 5329, 5302, 5275, 5249,
        5222, 5196, 5171, 5145, 5120, 5095, 5070, 5046,
        5022, 4998, 4974, 4950, 4927, 4904, 4881, 4858,
        4836, 4813, 4791, 4769, 4748, 4726, 4705, 4684,
        4663, 4642, 4622, 4601, 4581, 4561, 4541, 4522,
        4502, 4483, 4464, 4445, 4426, 4407, 4389, 4370,
        4352, 4334, 4316, 4298, 4281, 4263, 4246, 4229,
        4212, 4195, 4178, 4161, 4145, 4128, 4112, 4096
    };

    srcstep -= size.width*src_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += 3, src += src_cn )
        {
            int b = (src)[blue_idx], g = (src)[1], r = (src)[2^blue_idx];
            int h, s, v = b;
            int vmin = b, diff;
            int vr, vg;

            CV_CALC_MAX_8U( v, g );
            CV_CALC_MAX_8U( v, r );
            CV_CALC_MIN_8U( vmin, g );
            CV_CALC_MIN_8U( vmin, r );

            diff = v - vmin;
            vr = v == r ? -1 : 0;
            vg = v == g ? -1 : 0;

            s = diff * div_table[v] >> hsv_shift;
            h = (vr & (g - b)) +
                (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = ((h * div_table[diff] * 15 + (1 << (hsv_shift + 6))) >> (7 + hsv_shift))\
                + (h < 0 ? 30*6 : 0);

            dst[i] = (uchar)h;
            dst[i+1] = (uchar)s;
            dst[i+2] = (uchar)v;
        }
    }

    return CV_OK;
#endif
}


static CvStatus CV_STDCALL
icvBGRx2HSV_32f_CnC3R( const float* src, int srcstep,
                       float* dst, int dststep,
                       CvSize size, int src_cn, int blue_idx )
{
    int i;
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    srcstep -= size.width*src_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += 3, src += src_cn )
        {
            float b = src[blue_idx], g = src[1], r = src[2^blue_idx];
            float h, s, v;

            float vmin, diff;

            v = vmin = r;
            if( v < g ) v = g;
            if( v < b ) v = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = v - vmin;
            s = diff/(float)(fabs(v) + FLT_EPSILON);
            diff = (float)(60./(diff + FLT_EPSILON));
            if( v == r )
                h = (g - b)*diff;
            else if( v == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0 ) h += 360.f;

            dst[i] = h;
            dst[i+1] = s;
            dst[i+2] = v;
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvHSV2BGRx_32f_C3CnR( const float* src, int srcstep, float* dst,
                       int dststep, CvSize size, int dst_cn, int blue_idx )
{
    int i;
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    dststep -= size.width*dst_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += 3, dst += dst_cn )
        {
            float h = src[i], s = src[i+1], v = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = v;
            else
            {
                static const int sector_data[][3]=
                    {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;
                h *= 0.016666666666666666f; // h /= 60;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );
                sector = cvFloor(h);
                h -= sector;

                tab[0] = v;
                tab[1] = v*(1.f - s);
                tab[2] = v*(1.f - s*h);
                tab[3] = v*(1.f - s*(1.f - h));
                
                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[blue_idx] = b;
            dst[1] = g;
            dst[blue_idx^2] = r;
            if( dst_cn == 4 )
                dst[3] = 0;
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvHSV2BGRx_8u_C3CnR( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int dst_cn, int blue_idx )
{
    static const float pre_coeffs[] = { 2.f, 0.f, 0.0039215686274509803f, 0.f, 1.f, 0.f };

#ifdef HAVE_IPP
    int block_size = MIN(1 << 14, size.width);
    uchar* buffer;
    int i, di, k;
    CvStatus status = CV_OK;

    buffer = (uchar*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );
    dststep -= size.width*dst_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += block_size )
        {
            const uchar* src1 = src + i*3;
            di = MIN(block_size, size.width - i);
            for( k = 0; k < di*3; k += 3 )
            {
                uchar h = icvHue180To255[src1[k]];
                uchar s = src1[k+1];
                uchar v = src1[k+2];
                buffer[k] = h;
                buffer[k+1] = s;
                buffer[k+2] = v;
            }

            status = (CvStatus)ippiHSVToRGB_8u_C3R( buffer, di*3,
                                    buffer, di*3, ippiSize(di,1) );
            if( status < 0 )
                return status;

            for( k = 0; k < di*3; k += 3, dst += dst_cn )
            {
                uchar r = buffer[k];
                uchar g = buffer[k+1];
                uchar b = buffer[k+2];
                dst[blue_idx] = b;
                dst[1] = g;
                dst[blue_idx^2] = r;
                if( dst_cn == 4 )
                    dst[3] = 0;
            }
        }
    }
    return CV_OK;
#else
    return icvABC2BGRx_8u_C3CnR( src, srcstep, dst, dststep, size, dst_cn, blue_idx,
                                 (CvColorCvtFunc2)icvHSV2BGRx_32f_C3CnR, pre_coeffs, 0 );
#endif
}


/****************************************************************************************\
*                                     RGB <-> HLS                                        *
\****************************************************************************************/

static CvStatus CV_STDCALL
icvBGRx2HLS_32f_CnC3R( const float* src, int srcstep, float* dst, int dststep,
                       CvSize size, int src_cn, int blue_idx )
{
    int i;

#ifdef HAVE_IPP
    CvStatus status = icvBGRx2ABC_IPP_32f_CnC3R( src, srcstep, dst, dststep, size,
                                                 src_cn, blue_idx, (CvColorCvtFunc0)ippiRGBToHLS_32f_C3R );
    if( status >= 0 )
    {
        size.width *= 3;
        dststep /= sizeof(dst[0]);

        for( ; size.height--; dst += dststep )
        {
            for( i = 0; i <= size.width - 12; i += 12 )
            {
                float t0 = dst[i]*360.f, t1 = dst[i+3]*360.f;
                dst[i] = t0; dst[i+3] = t1;
                t0 = dst[i+6]*360.f; t1 = dst[i+9]*360.f;
                dst[i+6] = t0; dst[i+9] = t1;
            }
            for( ; i < size.width; i += 3 )
                dst[i] = dst[i]*360.f;
        }
    }
    return status;
#else
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    srcstep -= size.width*src_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += 3, src += src_cn )
        {
            float b = src[blue_idx], g = src[1], r = src[2^blue_idx];
            float h = 0.f, s = 0.f, l;
            float vmin, vmax, diff;

            vmax = vmin = r;
            if( vmax < g ) vmax = g;
            if( vmax < b ) vmax = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = vmax - vmin;
            l = (vmax + vmin)*0.5f;

            if( diff > FLT_EPSILON )
            {
                s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
                diff = 60.f/diff;

                if( vmax == r )
                    h = (g - b)*diff;
                else if( vmax == g )
                    h = (b - r)*diff + 120.f;
                else
                    h = (r - g)*diff + 240.f;

                if( h < 0.f ) h += 360.f;
            }

            dst[i] = h;
            dst[i+1] = l;
            dst[i+2] = s;
        }
    }

    return CV_OK;
#endif
}


static CvStatus CV_STDCALL
icvHLS2BGRx_32f_C3CnR( const float* src, int srcstep, float* dst, int dststep,
                       CvSize size, int dst_cn, int blue_idx )
{
    int i;
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);

#ifdef HAVE_IPP
    int block_size = MIN(1 << 10, size.width);
    float* buffer;
    int di, k;
    CvStatus status = CV_OK;

    buffer = (float*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );
    dststep -= size.width*dst_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += block_size )
        {
            const float* src1 = src + i*3;
            di = MIN(block_size, size.width - i);
            for( k = 0; k < di*3; k += 3 )
            {
                float h = src1[k]*0.0027777777777777779f; // /360.
                float s = src1[k+1], v = src1[k+2];
                buffer[k] = h; buffer[k+1] = s; buffer[k+2] = v;
            }

            status = (CvStatus)ippiHLSToRGB_32f_C3R( buffer, di*3*sizeof(dst[0]),
                            buffer, di*3*sizeof(dst[0]), ippiSize(di,1) );
            if( status < 0 )
                return status;

            for( k = 0; k < di*3; k += 3, dst += dst_cn )
            {
                float r = buffer[k], g = buffer[k+1], b = buffer[k+2];
                dst[blue_idx] = b; dst[1] = g; dst[blue_idx^2] = r;
                if( dst_cn == 4 )
                    dst[3] = 0;
            }
        }
    }

    return CV_OK;
#else
    
    dststep -= size.width*dst_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += 3, dst += dst_cn )
        {
            float h = src[i], l = src[i+1], s = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = l;
            else
            {
                static const int sector_data[][3]=
                    {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;
                
                float p2 = l <= 0.5f ? l*(1 + s) : l + s - l*s;
                float p1 = 2*l - p2;

                h *= 0.016666666666666666f; // h /= 60;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                assert( 0 <= h && h < 6 );
                sector = cvFloor(h);
                h -= sector;

                tab[0] = p2;
                tab[1] = p1;
                tab[2] = p1 + (p2 - p1)*(1-h);
                tab[3] = p1 + (p2 - p1)*h;

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[blue_idx] = b;
            dst[1] = g;
            dst[blue_idx^2] = r;
            if( dst_cn == 4 )
                dst[3] = 0;
        }
    }

    return CV_OK;
#endif
}

static CvStatus CV_STDCALL
icvBGRx2HLS_8u_CnC3R( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int src_cn, int blue_idx )
{
    static const float post_coeffs[] = { 0.5f, 0.f, 255.f, 0.f, 255.f, 0.f };

#ifdef HAVE_IPP
    CvStatus status = icvBGRx2ABC_IPP_8u_CnC3R( src, srcstep, dst, dststep, size,
                                                src_cn, blue_idx, (CvColorCvtFunc0)ippiRGBToHLS_8u_C3R );
    if( status >= 0 )
    {
        size.width *= 3;
        for( ; size.height--; dst += dststep )
        {
            int i;
            for( i = 0; i <= size.width - 12; i += 12 )
            {
                uchar t0 = icvHue255To180[dst[i]], t1 = icvHue255To180[dst[i+3]];
                dst[i] = t0; dst[i+3] = t1;
                t0 = icvHue255To180[dst[i+6]]; t1 = icvHue255To180[dst[i+9]];
                dst[i+6] = t0; dst[i+9] = t1;
            }
            for( ; i < size.width; i += 3 )
                dst[i] = icvHue255To180[dst[i]];
        }
    }
    return status;
#else
    return icvBGRx2ABC_8u_CnC3R( src, srcstep, dst, dststep, size, src_cn, blue_idx,
                                 (CvColorCvtFunc2)icvBGRx2HLS_32f_CnC3R, 1, post_coeffs );
#endif
}


static CvStatus CV_STDCALL
icvHLS2BGRx_8u_C3CnR( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int dst_cn, int blue_idx )
{
    static const float pre_coeffs[] = { 2.f, 0.f, 0.0039215686274509803f, 0.f,
                                        0.0039215686274509803f, 0.f };

#ifdef HAVE_IPP
    int block_size = MIN(1 << 14, size.width);
    uchar* buffer;
    int i, di, k;
    CvStatus status = CV_OK;

    buffer = (uchar*)cvStackAlloc( block_size*3*sizeof(buffer[0]) );
    dststep -= size.width*dst_cn;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += block_size )
        {
            const uchar* src1 = src + i*3;
            di = MIN(block_size, size.width - i);
            for( k = 0; k < di*3; k += 3 )
            {
                uchar h = icvHue180To255[src1[k]];
                uchar l = src1[k+1];
                uchar s = src1[k+2];
                buffer[k] = h;
                buffer[k+1] = l;
                buffer[k+2] = s;
            }

            status = (CvStatus)ippiHLSToRGB_8u_C3R( buffer, di*3,
                            buffer, di*3, ippiSize(di,1) );
            if( status < 0 )
                return status;

            for( k = 0; k < di*3; k += 3, dst += dst_cn )
            {
                uchar r = buffer[k];
                uchar g = buffer[k+1];
                uchar b = buffer[k+2];
                dst[blue_idx] = b;
                dst[1] = g;
                dst[blue_idx^2] = r;
                if( dst_cn == 4 )
                    dst[3] = 0;
            }
        }
    }

    return CV_OK;
#else
    return icvABC2BGRx_8u_C3CnR( src, srcstep, dst, dststep, size, dst_cn, blue_idx,
                                 (CvColorCvtFunc2)icvHLS2BGRx_32f_C3CnR, pre_coeffs, 1 );
#endif
}


/****************************************************************************************\
*                                     RGB <-> L*a*b*                                     *
\****************************************************************************************/

#define  labXr_32f  0.433953f /* = xyzXr_32f / 0.950456 */
#define  labXg_32f  0.376219f /* = xyzXg_32f / 0.950456 */
#define  labXb_32f  0.189828f /* = xyzXb_32f / 0.950456 */

#define  labYr_32f  0.212671f /* = xyzYr_32f */
#define  labYg_32f  0.715160f /* = xyzYg_32f */ 
#define  labYb_32f  0.072169f /* = xyzYb_32f */ 

#define  labZr_32f  0.017758f /* = xyzZr_32f / 1.088754 */
#define  labZg_32f  0.109477f /* = xyzZg_32f / 1.088754 */
#define  labZb_32f  0.872766f /* = xyzZb_32f / 1.088754 */

#define  labRx_32f  3.0799327f  /* = xyzRx_32f * 0.950456 */
#define  labRy_32f  (-1.53715f) /* = xyzRy_32f */
#define  labRz_32f  (-0.542782f)/* = xyzRz_32f * 1.088754 */

#define  labGx_32f  (-0.921235f)/* = xyzGx_32f * 0.950456 */
#define  labGy_32f  1.875991f   /* = xyzGy_32f */ 
#define  labGz_32f  0.04524426f /* = xyzGz_32f * 1.088754 */

#define  labBx_32f  0.0528909755f /* = xyzBx_32f * 0.950456 */
#define  labBy_32f  (-0.204043f)  /* = xyzBy_32f */
#define  labBz_32f  1.15115158f   /* = xyzBz_32f * 1.088754 */

#define  labT_32f   0.008856f

#define labT   fix(labT_32f*255,lab_shift)

#undef lab_shift
#define lab_shift 10
#define labXr  fix(labXr_32f,lab_shift)
#define labXg  fix(labXg_32f,lab_shift)
#define labXb  fix(labXb_32f,lab_shift)
                            
#define labYr  fix(labYr_32f,lab_shift)
#define labYg  fix(labYg_32f,lab_shift)
#define labYb  fix(labYb_32f,lab_shift)
                            
#define labZr  fix(labZr_32f,lab_shift)
#define labZg  fix(labZg_32f,lab_shift)
#define labZb  fix(labZb_32f,lab_shift)

#define labSmallScale_32f  7.787f
#define labSmallShift_32f  0.13793103448275862f  /* 16/116 */
#define labLScale_32f      116.f
#define labLShift_32f      16.f
#define labLScale2_32f     903.3f

#define labSmallScale fix(31.27 /* labSmallScale_32f*(1<<lab_shift)/255 */,lab_shift)
#define labSmallShift fix(141.24138 /* labSmallScale_32f*(1<<lab) */,lab_shift)
#define labLScale fix(295.8 /* labLScale_32f*255/100 */,lab_shift)
#define labLShift fix(41779.2 /* labLShift_32f*1024*255/100 */,lab_shift)
#define labLScale2 fix(labLScale2_32f*0.01,lab_shift)

/* 1024*(([0..511]./255)**(1./3)) */
static ushort icvLabCubeRootTab[] = {
       0,  161,  203,  232,  256,  276,  293,  308,  322,  335,  347,  359,  369,  379,  389,  398,
     406,  415,  423,  430,  438,  445,  452,  459,  465,  472,  478,  484,  490,  496,  501,  507,
     512,  517,  523,  528,  533,  538,  542,  547,  552,  556,  561,  565,  570,  574,  578,  582,
     586,  590,  594,  598,  602,  606,  610,  614,  617,  621,  625,  628,  632,  635,  639,  642,
     645,  649,  652,  655,  659,  662,  665,  668,  671,  674,  677,  680,  684,  686,  689,  692,
     695,  698,  701,  704,  707,  710,  712,  715,  718,  720,  723,  726,  728,  731,  734,  736,
     739,  741,  744,  747,  749,  752,  754,  756,  759,  761,  764,  766,  769,  771,  773,  776,
     778,  780,  782,  785,  787,  789,  792,  794,  796,  798,  800,  803,  805,  807,  809,  811,
     813,  815,  818,  820,  822,  824,  826,  828,  830,  832,  834,  836,  838,  840,  842,  844,
     846,  848,  850,  852,  854,  856,  857,  859,  861,  863,  865,  867,  869,  871,  872,  874,
     876,  878,  880,  882,  883,  885,  887,  889,  891,  892,  894,  896,  898,  899,  901,  903,
     904,  906,  908,  910,  911,  913,  915,  916,  918,  920,  921,  923,  925,  926,  928,  929,
     931,  933,  934,  936,  938,  939,  941,  942,  944,  945,  947,  949,  950,  952,  953,  955,
     956,  958,  959,  961,  962,  964,  965,  967,  968,  970,  971,  973,  974,  976,  977,  979,
     980,  982,  983,  985,  986,  987,  989,  990,  992,  993,  995,  996,  997,  999, 1000, 1002,
    1003, 1004, 1006, 1007, 1009, 1010, 1011, 1013, 1014, 1015, 1017, 1018, 1019, 1021, 1022, 1024,
    1025, 1026, 1028, 1029, 1030, 1031, 1033, 1034, 1035, 1037, 1038, 1039, 1041, 1042, 1043, 1044,
    1046, 1047, 1048, 1050, 1051, 1052, 1053, 1055, 1056, 1057, 1058, 1060, 1061, 1062, 1063, 1065,
    1066, 1067, 1068, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1081, 1082, 1083, 1084,
    1085, 1086, 1088, 1089, 1090, 1091, 1092, 1094, 1095, 1096, 1097, 1098, 1099, 1101, 1102, 1103,
    1104, 1105, 1106, 1107, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1117, 1118, 1119, 1120, 1121,
    1122, 1123, 1124, 1125, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1138, 1139,
    1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1154, 1155, 1156,
    1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188,
    1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204,
    1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1215, 1216, 1217, 1218, 1219,
    1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1230, 1231, 1232, 1233, 1234,
    1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249,
    1250, 1251, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1259, 1260, 1261, 1262, 1263,
    1264, 1265, 1266, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1273, 1274, 1275, 1276, 1277,
    1278, 1279, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1285, 1286, 1287, 1288, 1289, 1290, 1291
};


static CvStatus CV_STDCALL
icvBGRx2Lab_8u_CnC3R( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int src_cn, int blue_idx )
{
#ifdef HAVE_IPP
    return icvBGRx2ABC_IPP_8u_CnC3R( src, srcstep, dst, dststep, size,
                                     src_cn, blue_idx^2, (CvColorCvtFunc0)ippiBGRToLab_8u_C3R );
#else
    srcstep -= size.width*src_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( int i = 0; i < size.width; i += 3, src += src_cn )
        {
            int b = src[blue_idx], g = src[1], r = src[2^blue_idx];
            int x, y, z, f;
            int L, a;

            x = b*labXb + g*labXg + r*labXr;
            y = b*labYb + g*labYg + r*labYr;
            z = b*labZb + g*labZg + r*labZr;

            f = x > labT;
            x = CV_DESCALE( x, lab_shift );

            if( f )
                assert( (unsigned)x < 512 ), x = icvLabCubeRootTab[x];
            else
                x = CV_DESCALE(x*labSmallScale + labSmallShift,lab_shift);

            f = z > labT;
            z = CV_DESCALE( z, lab_shift );

            if( f )
                assert( (unsigned)z < 512 ), z = icvLabCubeRootTab[z];
            else
                z = CV_DESCALE(z*labSmallScale + labSmallShift,lab_shift);

            f = y > labT;
            y = CV_DESCALE( y, lab_shift );

            if( f )
            {
                assert( (unsigned)y < 512 ), y = icvLabCubeRootTab[y];
                L = CV_DESCALE(y*labLScale - labLShift, 2*lab_shift );
            }
            else
            {
                L = CV_DESCALE(y*labLScale2,lab_shift);
                y = CV_DESCALE(y*labSmallScale + labSmallShift,lab_shift);
            }

            a = CV_DESCALE( 500*(x - y), lab_shift ) + 128;
            b = CV_DESCALE( 200*(y - z), lab_shift ) + 128;

            dst[i] = CV_CAST_8U(L);
            dst[i+1] = CV_CAST_8U(a);
            dst[i+2] = CV_CAST_8U(b);
        }
    }

    return CV_OK;
#endif
}


static CvStatus CV_STDCALL
icvBGRx2Lab_32f_CnC3R( const float* src, int srcstep, float* dst, int dststep,
                       CvSize size, int src_cn, int blue_idx )
{
    int i;
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    srcstep -= size.width*src_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += 3, src += src_cn )
        {
            float b = src[blue_idx], g = src[1], r = src[2^blue_idx];
            float x, y, z;
            float L, a;

            x = b*labXb_32f + g*labXg_32f + r*labXr_32f;
            y = b*labYb_32f + g*labYg_32f + r*labYr_32f;
            z = b*labZb_32f + g*labZg_32f + r*labZr_32f;

            if( x > labT_32f )
                x = cvCbrt(x);
            else
                x = x*labSmallScale_32f + labSmallShift_32f;

            if( z > labT_32f )
                z = cvCbrt(z);
            else
                z = z*labSmallScale_32f + labSmallShift_32f;

            if( y > labT_32f )
            {
                y = cvCbrt(y);
                L = y*labLScale_32f - labLShift_32f;
            }
            else
            {
                L = y*labLScale2_32f;
                y = y*labSmallScale_32f + labSmallShift_32f;
            }

            a = 500.f*(x - y);
            b = 200.f*(y - z);

            dst[i] = L;
            dst[i+1] = a;
            dst[i+2] = b;
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvLab2BGRx_32f_C3CnR( const float* src, int srcstep, float* dst, int dststep,
                       CvSize size, int dst_cn, int blue_idx )
{
    int i;
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    dststep -= size.width*dst_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( i = 0; i < size.width; i += 3, dst += dst_cn )
        {
            float L = src[i], a = src[i+1], b = src[i+2];
            float x, y, z;
            float g, r;

            L = (L + labLShift_32f)*(1.f/labLScale_32f);
            x = (L + a*0.002f);
            z = (L - b*0.005f);
            y = L*L*L;
            x = x*x*x;
            z = z*z*z;

            b = x*labBx_32f + y*labBy_32f + z*labBz_32f;
            g = x*labGx_32f + y*labGy_32f + z*labGz_32f;
            r = x*labRx_32f + y*labRy_32f + z*labRz_32f;

            dst[blue_idx] = b;
            dst[1] = g;
            dst[blue_idx^2] = r;
            if( dst_cn == 4 )
                dst[3] = 0;
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvLab2BGRx_8u_C3CnR( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int dst_cn, int blue_idx )
{
#ifdef HAVE_IPP
    return icvABC2BGRx_IPP_8u_C3CnR( src, srcstep, dst, dststep, size,
                                     dst_cn, blue_idx^2, (CvColorCvtFunc0)ippiLabToBGR_8u_C3R );
#else
    // L: [0..255] -> [0..100]
    // a: [0..255] -> [-128..127]
    // b: [0..255] -> [-128..127]
    static const float pre_coeffs[] = { 0.39215686274509809f, 0.f, 1.f, -128.f, 1.f, -128.f };

    return icvABC2BGRx_8u_C3CnR( src, srcstep, dst, dststep, size, dst_cn, blue_idx,
                                 (CvColorCvtFunc2)icvLab2BGRx_32f_C3CnR, pre_coeffs, 1 );
#endif
}


/****************************************************************************************\
*                                     RGB <-> L*u*v*                                     *
\****************************************************************************************/

#define luvUn_32f  0.19793943f 
#define luvVn_32f  0.46831096f 
#define luvYmin_32f  0.05882353f /* 15/255 */

static CvStatus CV_STDCALL
icvBGRx2Luv_32f_CnC3R( const float* src, int srcstep, float* dst, int dststep,
                       CvSize size, int src_cn, int blue_idx )
{
#ifdef HAVE_IPP
    return icvBGRx2ABC_IPP_32f_CnC3R( src, srcstep, dst, dststep, size,
                                      src_cn, blue_idx, (CvColorCvtFunc0)ippiRGBToLUV_32f_C3R );
#else
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    srcstep -= size.width*src_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( int i = 0; i < size.width; i += 3, src += src_cn )
        {
            float b = src[blue_idx], g = src[1], r = src[2^blue_idx];
            float x, y, z;
            float L, u, v, t;

            x = b*xyzXb_32f + g*xyzXg_32f + r*xyzXr_32f;
            y = b*xyzYb_32f + g*xyzYg_32f + r*xyzYr_32f;
            z = b*xyzZb_32f + g*xyzZg_32f + r*xyzZr_32f;

            if( !x && !y && !z )
                L = u = v = 0.f;
            else
            {
                if( y > labT_32f )
                    L = labLScale_32f * cvCbrt(y) - labLShift_32f;
                else
                    L = labLScale2_32f * y;

                t = 1.f / (x + 15 * y + 3 * z);            
                u = 4.0f * x * t;
                v = 9.0f * y * t;

                u = 13*L*(u - luvUn_32f);
                v = 13*L*(v - luvVn_32f);
            }

            dst[i] = L;
            dst[i+1] = u;
            dst[i+2] = v;
        }
    }

    return CV_OK;
#endif
}


static CvStatus CV_STDCALL
icvLuv2BGRx_32f_C3CnR( const float* src, int srcstep, float* dst, int dststep,
                       CvSize size, int dst_cn, int blue_idx )
{
#ifdef HAVE_IPP
    return icvABC2BGRx_IPP_32f_C3CnR( src, srcstep, dst, dststep, size,
                                      dst_cn, blue_idx, (CvColorCvtFunc0)ippiLUVToRGB_32f_C3R );
#else
    srcstep /= sizeof(src[0]);
    dststep /= sizeof(dst[0]);
    dststep -= size.width*dst_cn;
    size.width *= 3;

    for( ; size.height--; src += srcstep, dst += dststep )
    {
        for( int i = 0; i < size.width; i += 3, dst += dst_cn )
        {
            float L = src[i], u = src[i+1], v = src[i+2];
            float x, y, z, t, u1, v1, b, g, r;

            if( L >= 8 ) 
            {
                t = (L + labLShift_32f) * (1.f/labLScale_32f);
                y = t*t*t;
            }
            else
            {
                y = L * (1.f/labLScale2_32f);
                L = MAX( L, 0.001f );
            }

            t = 1.f/(13.f * L);
            u1 = u*t + luvUn_32f;
            v1 = v*t + luvVn_32f;
            x = 2.25f * u1 * y / v1 ;
            z = (12 - 3 * u1 - 20 * v1) * y / (4 * v1);                
                       
            b = xyzBx_32f*x + xyzBy_32f*y + xyzBz_32f*z;
            g = xyzGx_32f*x + xyzGy_32f*y + xyzGz_32f*z;
            r = xyzRx_32f*x + xyzRy_32f*y + xyzRz_32f*z;

            dst[blue_idx] = b;
            dst[1] = g;
            dst[blue_idx^2] = r;
            if( dst_cn == 4 )
                dst[3] = 0.f;
        }
    }

    return CV_OK;
#endif
}


static CvStatus CV_STDCALL
icvBGRx2Luv_8u_CnC3R( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int src_cn, int blue_idx )
{
#ifdef HAVE_IPP
    return icvBGRx2ABC_IPP_8u_CnC3R( src, srcstep, dst, dststep, size,
                                     src_cn, blue_idx, (CvColorCvtFunc0)ippiRGBToLUV_8u_C3R );
#else
    // L: [0..100] -> [0..255]
    // u: [-134..220] -> [0..255]
    // v: [-140..122] -> [0..255]
    //static const float post_coeffs[] = { 2.55f, 0.f, 1.f, 83.f, 1.f, 140.f };
    static const float post_coeffs[] = { 2.55f, 0.f, 0.72033898305084743f, 96.525423728813564f,
                                         0.99609375f, 139.453125f };
    return icvBGRx2ABC_8u_CnC3R( src, srcstep, dst, dststep, size, src_cn, blue_idx,
                                 (CvColorCvtFunc2)icvBGRx2Luv_32f_CnC3R, 1, post_coeffs );
#endif
}


static CvStatus CV_STDCALL
icvLuv2BGRx_8u_C3CnR( const uchar* src, int srcstep, uchar* dst, int dststep,
                      CvSize size, int dst_cn, int blue_idx )
{
#ifdef HAVE_IPP
    return icvABC2BGRx_IPP_8u_C3CnR( src, srcstep, dst, dststep, size,
                                     dst_cn, blue_idx, (CvColorCvtFunc0)ippiLUVToRGB_8u_C3R );
#else
    // L: [0..255] -> [0..100]
    // u: [0..255] -> [-134..220]
    // v: [0..255] -> [-140..122]
    static const float pre_coeffs[] = { 0.39215686274509809f, 0.f, 1.388235294117647f, -134.f,
                                        1.003921568627451f, -140.f };

    return icvABC2BGRx_8u_C3CnR( src, srcstep, dst, dststep, size, dst_cn, blue_idx,
                                 (CvColorCvtFunc2)icvLuv2BGRx_32f_C3CnR, pre_coeffs, 1 );
#endif
}

/****************************************************************************************\
*                            Bayer Pattern -> RGB conversion                             *
\****************************************************************************************/

static CvStatus CV_STDCALL
icvBayer2BGR_8u_C1C3R( const uchar* bayer0, int bayer_step,
                       uchar *dst0, int dst_step,
                       CvSize size, int code )
{
    int blue = code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ? -1 : 1;
    int start_with_green = code == CV_BayerGB2BGR || code == CV_BayerGR2BGR;

    memset( dst0, 0, size.width*3*sizeof(dst0[0]) );
    memset( dst0 + (size.height - 1)*dst_step, 0, size.width*3*sizeof(dst0[0]) );
    dst0 += dst_step + 3 + 1;
    size.height -= 2;
    size.width -= 2;

    for( ; size.height-- > 0; bayer0 += bayer_step, dst0 += dst_step )
    {
        int t0, t1;
        const uchar* bayer = bayer0;
        uchar* dst = dst0;
        const uchar* bayer_end = bayer + size.width;

        dst[-4] = dst[-3] = dst[-2] = dst[size.width*3-1] =
            dst[size.width*3] = dst[size.width*3+1] = 0;

        if( size.width <= 0 )
            continue;

        if( start_with_green )
        {
            t0 = (bayer[1] + bayer[bayer_step*2+1] + 1) >> 1;
            t1 = (bayer[bayer_step] + bayer[bayer_step+2] + 1) >> 1;
            dst[-blue] = (uchar)t0;
            dst[0] = bayer[bayer_step+1];
            dst[blue] = (uchar)t1;
            bayer++;
            dst += 3;
        }

        if( blue > 0 )
        {
            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 6 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[-1] = (uchar)t0;
                dst[0] = (uchar)t1;
                dst[1] = bayer[bayer_step+1];

                t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                dst[2] = (uchar)t0;
                dst[3] = bayer[bayer_step+2];
                dst[4] = (uchar)t1;
            }
        }
        else
        {
            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 6 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[1] = (uchar)t0;
                dst[0] = (uchar)t1;
                dst[-1] = bayer[bayer_step+1];

                t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                dst[4] = (uchar)t0;
                dst[3] = bayer[bayer_step+2];
                dst[2] = (uchar)t1;
            }
        }

        if( bayer < bayer_end )
        {
            t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                  bayer[bayer_step*2+2] + 2) >> 2;
            t1 = (bayer[1] + bayer[bayer_step] +
                  bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
            dst[-blue] = (uchar)t0;
            dst[0] = (uchar)t1;
            dst[blue] = bayer[bayer_step+1];
            bayer++;
            dst += 3;
        }

        blue = -blue;
        start_with_green = !start_with_green;
    }

    return CV_OK;
}


/****************************************************************************************\
*                                   The main function                                    *
\****************************************************************************************/

CV_IMPL void
cvCvtColor( const CvArr* srcarr, CvArr* dstarr, int code )
{
    CvMat srcstub, *src = (CvMat*)srcarr;
    CvMat dststub, *dst = (CvMat*)dstarr;
    CvSize size;
    int src_step, dst_step;
    int src_cn, dst_cn, depth;
    CvColorCvtFunc0 func0 = 0;
    CvColorCvtFunc1 func1 = 0;
    CvColorCvtFunc2 func2 = 0;
    CvColorCvtFunc3 func3 = 0;
    int param[] = { 0, 0, 0, 0 };
    
    src = cvGetMat( srcarr, &srcstub );
    dst = cvGetMat( dstarr, &dststub );
    
    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( !CV_ARE_DEPTHS_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedFormats, "" );

    depth = CV_MAT_DEPTH(src->type);
    if( depth != CV_8U && depth != CV_16U && depth != CV_32F )
        CV_Error( CV_StsUnsupportedFormat, "" );

    src_cn = CV_MAT_CN( src->type );
    dst_cn = CV_MAT_CN( dst->type );
    size = cvGetMatSize( src );
    src_step = src->step;
    dst_step = dst->step;

    if( CV_IS_MAT_CONT(src->type & dst->type) &&
        code != CV_BayerBG2BGR && code != CV_BayerGB2BGR &&
        code != CV_BayerRG2BGR && code != CV_BayerGR2BGR ) 
    {
        size.width *= size.height;
        size.height = 1;
        src_step = dst_step = CV_STUB_STEP;
    }

    switch( code )
    {
    case CV_BGR2BGRA:
    case CV_RGB2BGRA:
        if( src_cn != 3 || dst_cn != 4 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        func1 = depth == CV_8U ? (CvColorCvtFunc1)icvBGR2BGRx_8u_C3C4R :
                depth == CV_16U ? (CvColorCvtFunc1)icvBGR2BGRx_16u_C3C4R :
                depth == CV_32F ? (CvColorCvtFunc1)icvBGR2BGRx_32f_C3C4R : 0;
        param[0] = code == CV_BGR2BGRA ? 0 : 2; // blue_idx
        break;

    case CV_BGRA2BGR:
    case CV_RGBA2BGR:
    case CV_RGB2BGR:
        if( (src_cn != 3 && src_cn != 4) || dst_cn != 3 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        func2 = depth == CV_8U ? (CvColorCvtFunc2)icvBGRx2BGR_8u_CnC3R :
                depth == CV_16U ? (CvColorCvtFunc2)icvBGRx2BGR_16u_CnC3R :
                depth == CV_32F ? (CvColorCvtFunc2)icvBGRx2BGR_32f_CnC3R : 0;
        param[0] = src_cn;
        param[1] = code == CV_BGRA2BGR ? 0 : 2; // blue_idx
        break;

    case CV_BGRA2RGBA:
        if( src_cn != 4 || dst_cn != 4 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        func0 = depth == CV_8U ? (CvColorCvtFunc0)icvBGRA2RGBA_8u_C4R :
                depth == CV_16U ? (CvColorCvtFunc0)icvBGRA2RGBA_16u_C4R :
                depth == CV_32F ? (CvColorCvtFunc0)icvBGRA2RGBA_32f_C4R : 0;
        break;

    case CV_BGR2BGR565:
    case CV_BGR2BGR555:
    case CV_RGB2BGR565:
    case CV_RGB2BGR555:
    case CV_BGRA2BGR565:
    case CV_BGRA2BGR555:
    case CV_RGBA2BGR565:
    case CV_RGBA2BGR555:
        if( (src_cn != 3 && src_cn != 4) || dst_cn != 2 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        if( depth != CV_8U )
            CV_Error( CV_BadDepth,
            "Conversion to/from 16-bit packed RGB format "
            "is only possible for 8-bit images (8-bit grayscale, 888 BGR/RGB or 8888 BGRA/RGBA)" );

        func3 = (CvColorCvtFunc3)icvBGRx2BGR5x5_8u_CnC2R;
        param[0] = src_cn;
        param[1] = code == CV_BGR2BGR565 || code == CV_BGR2BGR555 ||
                   code == CV_BGRA2BGR565 || code == CV_BGRA2BGR555 ? 0 : 2; // blue_idx
        param[2] = code == CV_BGR2BGR565 || code == CV_RGB2BGR565 ||
                   code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5; // green_bits
        break;

    case CV_BGR5652BGR:
    case CV_BGR5552BGR:
    case CV_BGR5652RGB:
    case CV_BGR5552RGB:
    case CV_BGR5652BGRA:
    case CV_BGR5552BGRA:
    case CV_BGR5652RGBA:
    case CV_BGR5552RGBA:
        if( src_cn != 2 || (dst_cn != 3 && dst_cn != 4))
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        if( depth != CV_8U )
            CV_Error( CV_BadDepth,
            "Conversion to/from 16-bit packed BGR format "
            "is only possible for 8-bit images (8-bit grayscale, 888 BGR/BGR or 8888 BGRA/BGRA)" );

        func3 = (CvColorCvtFunc3)icvBGR5x52BGRx_8u_C2CnR;
        param[0] = dst_cn;
        param[1] = code == CV_BGR5652BGR || code == CV_BGR5552BGR ||
                   code == CV_BGR5652BGRA || code == CV_BGR5552BGRA ? 0 : 2; // blue_idx
        param[2] = code == CV_BGR5652BGR || code == CV_BGR5652RGB ||
                   code == CV_BGR5652BGRA || code == CV_BGR5652RGBA ? 6 : 5; // green_bits
        break;

    case CV_BGR2GRAY:
    case CV_BGRA2GRAY:
    case CV_RGB2GRAY:
    case CV_RGBA2GRAY:
        if( (src_cn != 3 && src_cn != 4) || dst_cn != 1 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        func2 = depth == CV_8U ? (CvColorCvtFunc2)icvBGRx2Gray_8u_CnC1R :
                depth == CV_16U ? (CvColorCvtFunc2)icvBGRx2Gray_16u_CnC1R :
                depth == CV_32F ? (CvColorCvtFunc2)icvBGRx2Gray_32f_CnC1R : 0;
        
        param[0] = src_cn;
        param[1] = code == CV_BGR2GRAY || code == CV_BGRA2GRAY ? 0 : 2;
        break;

    case CV_BGR5652GRAY:
    case CV_BGR5552GRAY:
        if( src_cn != 2 || dst_cn != 1 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        if( depth != CV_8U )
            CV_Error( CV_BadDepth,
            "Conversion to/from 16-bit packed BGR format "
            "is only possible for 8-bit images (888 BGR/BGR or 8888 BGRA/BGRA)" );

        func2 = (CvColorCvtFunc2)icvBGR5x52Gray_8u_C2C1R;
        
        param[0] = code == CV_BGR5652GRAY ? 6 : 5; // green_bits
        break;

    case CV_GRAY2BGR:
    case CV_GRAY2BGRA:
        if( src_cn != 1 || (dst_cn != 3 && dst_cn != 4))
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        func1 = depth == CV_8U ? (CvColorCvtFunc1)icvGray2BGRx_8u_C1CnR :
                depth == CV_16U ? (CvColorCvtFunc1)icvGray2BGRx_16u_C1CnR :
                depth == CV_32F ? (CvColorCvtFunc1)icvGray2BGRx_32f_C1CnR : 0;
        
        param[0] = dst_cn;
        break;

    case CV_GRAY2BGR565:
    case CV_GRAY2BGR555:
        if( src_cn != 1 || dst_cn != 2 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        if( depth != CV_8U )
            CV_Error( CV_BadDepth,
            "Conversion to/from 16-bit packed BGR format "
            "is only possible for 8-bit images (888 BGR/BGR or 8888 BGRA/BGRA)" );

        func2 = (CvColorCvtFunc2)icvGray2BGR5x5_8u_C1C2R;
        param[0] = code == CV_GRAY2BGR565 ? 6 : 5; // green_bits
        break;

    case CV_BGR2YCrCb:
    case CV_RGB2YCrCb:
    case CV_BGR2XYZ:
    case CV_RGB2XYZ:
    case CV_BGR2HSV:
    case CV_RGB2HSV:
    case CV_BGR2Lab:
    case CV_RGB2Lab:
    case CV_BGR2Luv:
    case CV_RGB2Luv:
    case CV_BGR2HLS:
    case CV_RGB2HLS:
        if( (src_cn != 3 && src_cn != 4) || dst_cn != 3 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        if( depth == CV_8U )
            func2 = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? (CvColorCvtFunc2)icvBGRx2YCrCb_8u_CnC3R :
                    code == CV_BGR2XYZ || code == CV_RGB2XYZ ? (CvColorCvtFunc2)icvBGRx2XYZ_8u_CnC3R :
                    code == CV_BGR2HSV || code == CV_RGB2HSV ? (CvColorCvtFunc2)icvBGRx2HSV_8u_CnC3R :
                    code == CV_BGR2Lab || code == CV_RGB2Lab ? (CvColorCvtFunc2)icvBGRx2Lab_8u_CnC3R :
                    code == CV_BGR2Luv || code == CV_RGB2Luv ? (CvColorCvtFunc2)icvBGRx2Luv_8u_CnC3R :
                    code == CV_BGR2HLS || code == CV_RGB2HLS ? (CvColorCvtFunc2)icvBGRx2HLS_8u_CnC3R : 0;
        else if( depth == CV_16U )
            func2 = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? (CvColorCvtFunc2)icvBGRx2YCrCb_16u_CnC3R :
                    code == CV_BGR2XYZ || code == CV_RGB2XYZ ? (CvColorCvtFunc2)icvBGRx2XYZ_16u_CnC3R : 0;
        else if( depth == CV_32F )
            func2 = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? (CvColorCvtFunc2)icvBGRx2YCrCb_32f_CnC3R :
                    code == CV_BGR2XYZ || code == CV_RGB2XYZ ? (CvColorCvtFunc2)icvBGRx2XYZ_32f_CnC3R :
                    code == CV_BGR2HSV || code == CV_RGB2HSV ? (CvColorCvtFunc2)icvBGRx2HSV_32f_CnC3R :
                    code == CV_BGR2Lab || code == CV_RGB2Lab ? (CvColorCvtFunc2)icvBGRx2Lab_32f_CnC3R :
                    code == CV_BGR2Luv || code == CV_RGB2Luv ? (CvColorCvtFunc2)icvBGRx2Luv_32f_CnC3R :
                    code == CV_BGR2HLS || code == CV_RGB2HLS ? (CvColorCvtFunc2)icvBGRx2HLS_32f_CnC3R : 0;
        
        param[0] = src_cn;
        param[1] = code == CV_BGR2XYZ || code == CV_BGR2YCrCb || code == CV_BGR2HSV ||
                   code == CV_BGR2Lab || code == CV_BGR2Luv || code == CV_BGR2HLS ? 0 : 2;
        break;

    case CV_YCrCb2BGR:
    case CV_YCrCb2RGB:
    case CV_XYZ2BGR:
    case CV_XYZ2RGB:
    case CV_HSV2BGR:
    case CV_HSV2RGB:
    case CV_Lab2BGR:
    case CV_Lab2RGB:
    case CV_Luv2BGR:
    case CV_Luv2RGB:
    case CV_HLS2BGR:
    case CV_HLS2RGB:
        if( src_cn != 3 || (dst_cn != 3 && dst_cn != 4) )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );

        if( depth == CV_8U )
            func2 = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? (CvColorCvtFunc2)icvYCrCb2BGRx_8u_C3CnR :
                    code == CV_XYZ2BGR || code == CV_XYZ2RGB ? (CvColorCvtFunc2)icvXYZ2BGRx_8u_C3CnR :
                    code == CV_HSV2BGR || code == CV_HSV2RGB ? (CvColorCvtFunc2)icvHSV2BGRx_8u_C3CnR :
                    code == CV_HLS2BGR || code == CV_HLS2RGB ? (CvColorCvtFunc2)icvHLS2BGRx_8u_C3CnR :
                    code == CV_Lab2BGR || code == CV_Lab2RGB ? (CvColorCvtFunc2)icvLab2BGRx_8u_C3CnR :
                    code == CV_Luv2BGR || code == CV_Luv2RGB ? (CvColorCvtFunc2)icvLuv2BGRx_8u_C3CnR : 0;
        else if( depth == CV_16U )
            func2 = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? (CvColorCvtFunc2)icvYCrCb2BGRx_16u_C3CnR :
                    code == CV_XYZ2BGR || code == CV_XYZ2RGB ? (CvColorCvtFunc2)icvXYZ2BGRx_16u_C3CnR : 0;
        else if( depth == CV_32F )
            func2 = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? (CvColorCvtFunc2)icvYCrCb2BGRx_32f_C3CnR :
                    code == CV_XYZ2BGR || code == CV_XYZ2RGB ? (CvColorCvtFunc2)icvXYZ2BGRx_32f_C3CnR :
                    code == CV_HSV2BGR || code == CV_HSV2RGB ? (CvColorCvtFunc2)icvHSV2BGRx_32f_C3CnR :
                    code == CV_HLS2BGR || code == CV_HLS2RGB ? (CvColorCvtFunc2)icvHLS2BGRx_32f_C3CnR :
                    code == CV_Lab2BGR || code == CV_Lab2RGB ? (CvColorCvtFunc2)icvLab2BGRx_32f_C3CnR :
                    code == CV_Luv2BGR || code == CV_Luv2RGB ? (CvColorCvtFunc2)icvLuv2BGRx_32f_C3CnR : 0;
        
        param[0] = dst_cn;
        param[1] = code == CV_XYZ2BGR || code == CV_YCrCb2BGR || code == CV_HSV2BGR ||
                   code == CV_Lab2BGR || code == CV_Luv2BGR || code == CV_HLS2BGR ? 0 : 2;
        break;

    case CV_BayerBG2BGR:
    case CV_BayerGB2BGR:
    case CV_BayerRG2BGR:
    case CV_BayerGR2BGR:
        if( src_cn != 1 || dst_cn != 3 )
            CV_Error( CV_BadNumChannels,
            "Incorrect number of channels for this conversion code" );
        
        if( depth != CV_8U )
            CV_Error( CV_BadDepth,
            "Bayer pattern can be converted only to 8-bit 3-channel BGR/RGB image" );

        func1 = (CvColorCvtFunc1)icvBayer2BGR_8u_C1C3R;
        param[0] = code; // conversion code
        break;
    default:
        CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
    }

    if( func0 )
    {
        IPPI_CALL( func0( src->data.ptr, src_step, dst->data.ptr, dst_step, size ));
    }
    else if( func1 )
    {
        IPPI_CALL( func1( src->data.ptr, src_step,
            dst->data.ptr, dst_step, size, param[0] ));
    }
    else if( func2 )
    {
        IPPI_CALL( func2( src->data.ptr, src_step,
            dst->data.ptr, dst_step, size, param[0], param[1] ));
    }
    else if( func3 )
    {
        IPPI_CALL( func3( src->data.ptr, src_step,
            dst->data.ptr, dst_step, size, param[0], param[1], param[2] ));
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "The image format is not supported" );
}


void cv::cvtColor( const Mat& src, Mat& dst, int code, int dst_cn )
{
    switch( code )
    {
    case CV_BGR2BGRA:
    case CV_RGB2BGRA:
    case CV_BGRA2RGBA:
    case CV_BGR5652BGRA:
    case CV_BGR5552BGRA:
    case CV_BGR5652RGBA:
    case CV_BGR5552RGBA:
    case CV_GRAY2BGRA:
        dst_cn = 4;
        break;

    case CV_BGR2YCrCb:
    case CV_RGB2YCrCb:
    case CV_BGR2XYZ:
    case CV_RGB2XYZ:
    case CV_BGR2HSV:
    case CV_RGB2HSV:
    case CV_BGR2Lab:
    case CV_RGB2Lab:
    case CV_BGR2Luv:
    case CV_RGB2Luv:
    case CV_BGR2HLS:
    case CV_RGB2HLS:
        dst_cn = 3;
        break;

    case CV_BayerBG2BGR:
    case CV_BayerGB2BGR:
    case CV_BayerRG2BGR:
    case CV_BayerGR2BGR:

    case CV_BGRA2BGR:
    case CV_RGBA2BGR:
    case CV_RGB2BGR:
    case CV_BGR5652BGR:
    case CV_BGR5552BGR:
    case CV_BGR5652RGB:
    case CV_BGR5552RGB:
    case CV_GRAY2BGR:
        
    case CV_YCrCb2BGR:
    case CV_YCrCb2RGB:
    case CV_XYZ2BGR:
    case CV_XYZ2RGB:
    case CV_HSV2BGR:
    case CV_HSV2RGB:
    case CV_Lab2BGR:
    case CV_Lab2RGB:
    case CV_Luv2BGR:
    case CV_Luv2RGB:
    case CV_HLS2BGR:
    case CV_HLS2RGB:
        if( dst_cn != 4 )
            dst_cn = 3;
        break;

    case CV_BGR2BGR565:
    case CV_BGR2BGR555:
    case CV_RGB2BGR565:
    case CV_RGB2BGR555:
    case CV_BGRA2BGR565:
    case CV_BGRA2BGR555:
    case CV_RGBA2BGR565:
    case CV_RGBA2BGR555:
    case CV_GRAY2BGR565:
    case CV_GRAY2BGR555:
        dst_cn = 2;
        break;

    case CV_BGR2GRAY:
    case CV_BGRA2GRAY:
    case CV_RGB2GRAY:
    case CV_RGBA2GRAY:
    case CV_BGR5652GRAY:
    case CV_BGR5552GRAY:
        dst_cn = 1;
        break;
    default:
        CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
    }
    
    dst.create(src.size(), CV_MAKETYPE(src.depth(), dst_cn));
    CvMat _src = src, _dst = dst;
    cvCvtColor( &_src, &_dst, code );
}

/* End of file. */


