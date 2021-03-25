/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef OPJ_INTMATH_H
#define OPJ_INTMATH_H
/**
@file opj_intmath.h
@brief Implementation of operations on integers (INT)

The functions in OPJ_INTMATH.H have for goal to realize operations on integers.
*/

/** @defgroup OPJ_INTMATH OPJ_INTMATH - Implementation of operations on integers */
/*@{*/

/** @name Exported functions (see also openjpeg.h) */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Get the minimum of two integers
@return Returns a if a < b else b
*/
static INLINE OPJ_INT32 opj_int_min(OPJ_INT32 a, OPJ_INT32 b)
{
    return a < b ? a : b;
}

/**
Get the minimum of two integers
@return Returns a if a < b else b
*/
static INLINE OPJ_UINT32 opj_uint_min(OPJ_UINT32 a, OPJ_UINT32 b)
{
    return a < b ? a : b;
}

/**
Get the maximum of two integers
@return Returns a if a > b else b
*/
static INLINE OPJ_INT32 opj_int_max(OPJ_INT32 a, OPJ_INT32 b)
{
    return (a > b) ? a : b;
}

/**
Get the maximum of two integers
@return Returns a if a > b else b
*/
static INLINE OPJ_UINT32 opj_uint_max(OPJ_UINT32  a, OPJ_UINT32  b)
{
    return (a > b) ? a : b;
}

/**
 Get the saturated sum of two unsigned integers
 @return Returns saturated sum of a+b
 */
static INLINE OPJ_UINT32 opj_uint_adds(OPJ_UINT32 a, OPJ_UINT32 b)
{
    OPJ_UINT64 sum = (OPJ_UINT64)a + (OPJ_UINT64)b;
    return (OPJ_UINT32)(-(OPJ_INT32)(sum >> 32)) | (OPJ_UINT32)sum;
}

/**
 Get the saturated difference of two unsigned integers
 @return Returns saturated sum of a-b
 */
static INLINE OPJ_UINT32 opj_uint_subs(OPJ_UINT32 a, OPJ_UINT32 b)
{
    return (a >= b) ? a - b : 0;
}

/**
Clamp an integer inside an interval
@return
<ul>
<li>Returns a if (min < a < max)
<li>Returns max if (a > max)
<li>Returns min if (a < min)
</ul>
*/
static INLINE OPJ_INT32 opj_int_clamp(OPJ_INT32 a, OPJ_INT32 min,
                                      OPJ_INT32 max)
{
    if (a < min) {
        return min;
    }
    if (a > max) {
        return max;
    }
    return a;
}

/**
Clamp an integer inside an interval
@return
<ul>
<li>Returns a if (min < a < max)
<li>Returns max if (a > max)
<li>Returns min if (a < min)
</ul>
*/
static INLINE OPJ_INT64 opj_int64_clamp(OPJ_INT64 a, OPJ_INT64 min,
                                        OPJ_INT64 max)
{
    if (a < min) {
        return min;
    }
    if (a > max) {
        return max;
    }
    return a;
}

/**
@return Get absolute value of integer
*/
static INLINE OPJ_INT32 opj_int_abs(OPJ_INT32 a)
{
    return a < 0 ? -a : a;
}
/**
Divide an integer and round upwards
@return Returns a divided by b
*/
static INLINE OPJ_INT32 opj_int_ceildiv(OPJ_INT32 a, OPJ_INT32 b)
{
    assert(b);
    return (OPJ_INT32)(((OPJ_INT64)a + b - 1) / b);
}

/**
Divide an integer and round upwards
@return Returns a divided by b
*/
static INLINE OPJ_UINT32  opj_uint_ceildiv(OPJ_UINT32  a, OPJ_UINT32  b)
{
    assert(b);
    return (OPJ_UINT32)(((OPJ_UINT64)a + b - 1) / b);
}

/**
Divide an integer by a power of 2 and round upwards
@return Returns a divided by 2^b
*/
static INLINE OPJ_INT32 opj_int_ceildivpow2(OPJ_INT32 a, OPJ_INT32 b)
{
    return (OPJ_INT32)((a + ((OPJ_INT64)1 << b) - 1) >> b);
}

/**
 Divide a 64bits integer by a power of 2 and round upwards
 @return Returns a divided by 2^b
 */
static INLINE OPJ_INT32 opj_int64_ceildivpow2(OPJ_INT64 a, OPJ_INT32 b)
{
    return (OPJ_INT32)((a + ((OPJ_INT64)1 << b) - 1) >> b);
}

/**
 Divide an integer by a power of 2 and round upwards
 @return Returns a divided by 2^b
 */
static INLINE OPJ_UINT32 opj_uint_ceildivpow2(OPJ_UINT32 a, OPJ_UINT32 b)
{
    return (OPJ_UINT32)((a + ((OPJ_UINT64)1U << b) - 1U) >> b);
}

/**
Divide an integer by a power of 2 and round downwards
@return Returns a divided by 2^b
*/
static INLINE OPJ_INT32 opj_int_floordivpow2(OPJ_INT32 a, OPJ_INT32 b)
{
    return a >> b;
}

/**
Divide an integer by a power of 2 and round downwards
@return Returns a divided by 2^b
*/
static INLINE OPJ_UINT32 opj_uint_floordivpow2(OPJ_UINT32 a, OPJ_UINT32 b)
{
    return a >> b;
}

/**
Get logarithm of an integer and round downwards
@return Returns log2(a)
*/
static INLINE OPJ_INT32 opj_int_floorlog2(OPJ_INT32 a)
{
    OPJ_INT32 l;
    for (l = 0; a > 1; l++) {
        a >>= 1;
    }
    return l;
}
/**
Get logarithm of an integer and round downwards
@return Returns log2(a)
*/
static INLINE OPJ_UINT32  opj_uint_floorlog2(OPJ_UINT32  a)
{
    OPJ_UINT32  l;
    for (l = 0; a > 1; ++l) {
        a >>= 1;
    }
    return l;
}

/**
Multiply two fixed-precision rational numbers.
@param a
@param b
@return Returns a * b
*/
static INLINE OPJ_INT32 opj_int_fix_mul(OPJ_INT32 a, OPJ_INT32 b)
{
#if defined(_MSC_VER) && (_MSC_VER >= 1400) && !defined(__INTEL_COMPILER) && defined(_M_IX86)
    OPJ_INT64 temp = __emul(a, b);
#else
    OPJ_INT64 temp = (OPJ_INT64) a * (OPJ_INT64) b ;
#endif
    temp += 4096;
    assert((temp >> 13) <= (OPJ_INT64)0x7FFFFFFF);
    assert((temp >> 13) >= (-(OPJ_INT64)0x7FFFFFFF - (OPJ_INT64)1));
    return (OPJ_INT32)(temp >> 13);
}

static INLINE OPJ_INT32 opj_int_fix_mul_t1(OPJ_INT32 a, OPJ_INT32 b)
{
#if defined(_MSC_VER) && (_MSC_VER >= 1400) && !defined(__INTEL_COMPILER) && defined(_M_IX86)
    OPJ_INT64 temp = __emul(a, b);
#else
    OPJ_INT64 temp = (OPJ_INT64) a * (OPJ_INT64) b ;
#endif
    temp += 4096;
    assert((temp >> (13 + 11 - T1_NMSEDEC_FRACBITS)) <= (OPJ_INT64)0x7FFFFFFF);
    assert((temp >> (13 + 11 - T1_NMSEDEC_FRACBITS)) >= (-(OPJ_INT64)0x7FFFFFFF -
            (OPJ_INT64)1));
    return (OPJ_INT32)(temp >> (13 + 11 - T1_NMSEDEC_FRACBITS)) ;
}

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_INTMATH_H */
