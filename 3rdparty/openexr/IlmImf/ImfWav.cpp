///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////



//-----------------------------------------------------------------------------
//
//	16-bit Haar Wavelet encoding and decoding
//
//	The source code in this file is derived from the encoding
//	and decoding routines written by Christian Rouet for his
//	PIZ image file format.
//
//-----------------------------------------------------------------------------


#include <ImfWav.h>

namespace Imf {
namespace {


//
// Wavelet basis functions without modulo arithmetic; they produce
// the best compression ratios when the wavelet-transformed data are
// Huffman-encoded, but the wavelet transform works only for 14-bit
// data (untransformed data values must be less than (1 << 14)).
//

inline void
wenc14 (unsigned short  a, unsigned short  b,
        unsigned short &l, unsigned short &h)
{
    short as = a;
    short bs = b;

    short ms = (as + bs) >> 1;
    short ds = as - bs;

    l = ms;
    h = ds;
}


inline void
wdec14 (unsigned short  l, unsigned short  h,
        unsigned short &a, unsigned short &b)
{
    short ls = l;
    short hs = h;

    int hi = hs;
    int ai = ls + (hi & 1) + (hi >> 1);

    short as = ai;
    short bs = ai - hi;

    a = as;
    b = bs;
}


//
// Wavelet basis functions with modulo arithmetic; they work with full
// 16-bit data, but Huffman-encoding the wavelet-transformed data doesn't
// compress the data quite as well.
//

const int NBITS = 16;
const int A_OFFSET =  1 << (NBITS  - 1);
const int M_OFFSET =  1 << (NBITS  - 1);
const int MOD_MASK = (1 <<  NBITS) - 1;


inline void
wenc16 (unsigned short  a, unsigned short  b,
        unsigned short &l, unsigned short &h)
{
    int ao =  (a + A_OFFSET) & MOD_MASK;
    int m  = ((ao + b) >> 1);
    int d  =   ao - b;

    if (d < 0)
    m = (m + M_OFFSET) & MOD_MASK;

    d &= MOD_MASK;

    l = m;
    h = d;
}


inline void
wdec16 (unsigned short  l, unsigned short  h,
        unsigned short &a, unsigned short &b)
{
    int m = l;
    int d = h;
    int bb = (m - (d >> 1)) & MOD_MASK;
    int aa = (d + bb - A_OFFSET) & MOD_MASK;
    b = bb;
    a = aa;
}

} // namespace


//
// 2D Wavelet encoding:
//

void
wav2Encode
    (unsigned short*	in,	// io: values are transformed in place
     int		nx,	// i : x size
     int		ox,	// i : x offset
     int		ny,	// i : y size
     int		oy,	// i : y offset
     unsigned short	mx)	// i : maximum in[x][y] value
{
    bool w14 = (mx < (1 << 14));
    int	n  = (nx > ny)? ny: nx;
    int	p  = 1;			// == 1 <<  level
    int p2 = 2;			// == 1 << (level+1)

    //
    // Hierachical loop on smaller dimension n
    //

    while (p2 <= n)
    {
    unsigned short *py = in;
    unsigned short *ey = in + oy * (ny - p2);
    int oy1 = oy * p;
    int oy2 = oy * p2;
    int ox1 = ox * p;
    int ox2 = ox * p2;
    unsigned short i00,i01,i10,i11;

    //
    // Y loop
    //

    for (; py <= ey; py += oy2)
    {
        unsigned short *px = py;
        unsigned short *ex = py + ox * (nx - p2);

        //
        // X loop
        //

        for (; px <= ex; px += ox2)
        {
        unsigned short *p01 = px  + ox1;
        unsigned short *p10 = px  + oy1;
        unsigned short *p11 = p10 + ox1;

        //
        // 2D wavelet encoding
        //

        if (w14)
        {
            wenc14 (*px,  *p01, i00, i01);
            wenc14 (*p10, *p11, i10, i11);
            wenc14 (i00, i10, *px,  *p10);
            wenc14 (i01, i11, *p01, *p11);
        }
        else
        {
            wenc16 (*px,  *p01, i00, i01);
            wenc16 (*p10, *p11, i10, i11);
            wenc16 (i00, i10, *px,  *p10);
            wenc16 (i01, i11, *p01, *p11);
        }
        }

        //
        // Encode (1D) odd column (still in Y loop)
        //

        if (nx & p)
        {
        unsigned short *p10 = px + oy1;

        if (w14)
            wenc14 (*px, *p10, i00, *p10);
        else
            wenc16 (*px, *p10, i00, *p10);

        *px= i00;
        }
    }

    //
    // Encode (1D) odd line (must loop in X)
    //

    if (ny & p)
    {
        unsigned short *px = py;
        unsigned short *ex = py + ox * (nx - p2);

        for (; px <= ex; px += ox2)
        {
        unsigned short *p01 = px + ox1;

        if (w14)
            wenc14 (*px, *p01, i00, *p01);
        else
            wenc16 (*px, *p01, i00, *p01);

        *px= i00;
        }
    }

    //
    // Next level
    //

    p = p2;
    p2 <<= 1;
    }
}


//
// 2D Wavelet decoding:
//

void
wav2Decode
    (unsigned short*	in,	// io: values are transformed in place
     int		nx,	// i : x size
     int		ox,	// i : x offset
     int		ny,	// i : y size
     int		oy,	// i : y offset
     unsigned short	mx)	// i : maximum in[x][y] value
{
    bool w14 = (mx < (1 << 14));
    int	n = (nx > ny)? ny: nx;
    int	p = 1;
    int p2;

    //
    // Search max level
    //

    while (p <= n)
    p <<= 1;

    p >>= 1;
    p2 = p;
    p >>= 1;

    //
    // Hierarchical loop on smaller dimension n
    //

    while (p >= 1)
    {
    unsigned short *py = in;
    unsigned short *ey = in + oy * (ny - p2);
    int oy1 = oy * p;
    int oy2 = oy * p2;
    int ox1 = ox * p;
    int ox2 = ox * p2;
    unsigned short i00,i01,i10,i11;

    //
    // Y loop
    //

    for (; py <= ey; py += oy2)
    {
        unsigned short *px = py;
        unsigned short *ex = py + ox * (nx - p2);

        //
        // X loop
        //

        for (; px <= ex; px += ox2)
        {
        unsigned short *p01 = px  + ox1;
        unsigned short *p10 = px  + oy1;
        unsigned short *p11 = p10 + ox1;

        //
        // 2D wavelet decoding
        //

        if (w14)
        {
            wdec14 (*px,  *p10, i00, i10);
            wdec14 (*p01, *p11, i01, i11);
            wdec14 (i00, i01, *px,  *p01);
            wdec14 (i10, i11, *p10, *p11);
        }
        else
        {
            wdec16 (*px,  *p10, i00, i10);
            wdec16 (*p01, *p11, i01, i11);
            wdec16 (i00, i01, *px,  *p01);
            wdec16 (i10, i11, *p10, *p11);
        }
        }

        //
        // Decode (1D) odd column (still in Y loop)
        //

        if (nx & p)
        {
        unsigned short *p10 = px + oy1;

        if (w14)
            wdec14 (*px, *p10, i00, *p10);
        else
            wdec16 (*px, *p10, i00, *p10);

        *px= i00;
        }
    }

    //
    // Decode (1D) odd line (must loop in X)
    //

    if (ny & p)
    {
        unsigned short *px = py;
        unsigned short *ex = py + ox * (nx - p2);

        for (; px <= ex; px += ox2)
        {
        unsigned short *p01 = px + ox1;

        if (w14)
            wdec14 (*px, *p01, i00, *p01);
        else
            wdec16 (*px, *p01, i00, *p01);

        *px= i00;
        }
    }

    //
    // Next level
    //

    p2 = p;
    p >>= 1;
    }
}


} // namespace Imf
