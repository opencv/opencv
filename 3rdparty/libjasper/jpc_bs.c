/*
 * Copyright (c) 1999-2000, Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2003 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 *
 * JasPer License Version 2.0
 *
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 *
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 *
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 *
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 *
 * __END_OF_JASPER_LICENSE__
 */

/*
 * Bit Stream Class
 *
 * $Id: jpc_bs.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>

#include "jasper/jas_malloc.h"
#include "jasper/jas_math.h"
#include "jasper/jas_debug.h"

#include "jpc_bs.h"

/******************************************************************************\
* Local function prototypes.
\******************************************************************************/

static jpc_bitstream_t *jpc_bitstream_alloc(void);

/******************************************************************************\
* Code for opening and closing bit streams.
\******************************************************************************/

/* Open a bit stream from a stream. */
jpc_bitstream_t *jpc_bitstream_sopen(jas_stream_t *stream, char *mode)
{
    jpc_bitstream_t *bitstream;

    /* Ensure that the open mode is valid. */
#if 1
/* This causes a string literal too long error (with c99 pedantic mode). */
    assert(!strcmp(mode, "r") || !strcmp(mode, "w") || !strcmp(mode, "r+")
      || !strcmp(mode, "w+"));
#endif

    if (!(bitstream = jpc_bitstream_alloc())) {
        return 0;
    }

    /* By default, do not close the underlying (character) stream, upon
      the close of the bit stream. */
    bitstream->flags_ = JPC_BITSTREAM_NOCLOSE;

    bitstream->stream_ = stream;
    bitstream->openmode_ = (mode[0] == 'w') ? JPC_BITSTREAM_WRITE :
      JPC_BITSTREAM_READ;

    /* Mark the data buffer as empty. */
    bitstream->cnt_ = (bitstream->openmode_ == JPC_BITSTREAM_READ) ? 0 : 8;
    bitstream->buf_ = 0;

    return bitstream;
}

/* Close a bit stream. */
int jpc_bitstream_close(jpc_bitstream_t *bitstream)
{
    int ret = 0;

    /* Align to the next byte boundary while considering the effects of
      bit stuffing. */
    if (jpc_bitstream_align(bitstream)) {
        ret = -1;
    }

    /* If necessary, close the underlying (character) stream. */
    if (!(bitstream->flags_ & JPC_BITSTREAM_NOCLOSE) && bitstream->stream_) {
        if (jas_stream_close(bitstream->stream_)) {
            ret = -1;
        }
        bitstream->stream_ = 0;
    }

    jas_free(bitstream);
    return ret;
}

/* Allocate a new bit stream. */
static jpc_bitstream_t *jpc_bitstream_alloc()
{
    jpc_bitstream_t *bitstream;

    /* Allocate memory for the new bit stream object. */
    if (!(bitstream = jas_malloc(sizeof(jpc_bitstream_t)))) {
        return 0;
    }
    /* Initialize all of the data members. */
    bitstream->stream_ = 0;
    bitstream->cnt_ = 0;
    bitstream->flags_ = 0;
    bitstream->openmode_ = 0;

    return bitstream;
}

/******************************************************************************\
* Code for reading/writing from/to bit streams.
\******************************************************************************/

/* Get a bit from a bit stream. */
int jpc_bitstream_getbit_func(jpc_bitstream_t *bitstream)
{
    int ret;
    JAS_DBGLOG(1000, ("jpc_bitstream_getbit_func(%p)\n", bitstream));
    ret = jpc_bitstream_getbit_macro(bitstream);
    JAS_DBGLOG(1000, ("jpc_bitstream_getbit_func -> %d\n", ret));
    return ret;
}

/* Put a bit to a bit stream. */
int jpc_bitstream_putbit_func(jpc_bitstream_t *bitstream, int b)
{
    int ret;
    JAS_DBGLOG(1000, ("jpc_bitstream_putbit_func(%p, %d)\n", bitstream, b));
    ret = jpc_bitstream_putbit_macro(bitstream, b);
    JAS_DBGLOG(1000, ("jpc_bitstream_putbit_func() -> %d\n", ret));
    return ret;
}

/* Get one or more bits from a bit stream. */
long jpc_bitstream_getbits(jpc_bitstream_t *bitstream, int n)
{
    long v;
    int u;

    /* We can reliably get at most 31 bits since ISO/IEC 9899 only
      guarantees that a long can represent values up to 2^31-1. */
    assert(n >= 0 && n < 32);

    /* Get the number of bits requested from the specified bit stream. */
    v = 0;
    while (--n >= 0) {
        if ((u = jpc_bitstream_getbit(bitstream)) < 0) {
            return -1;
        }
        v = (v << 1) | u;
    }
    return v;
}

/* Put one or more bits to a bit stream. */
int jpc_bitstream_putbits(jpc_bitstream_t *bitstream, int n, long v)
{
    int m;

    /* We can reliably put at most 31 bits since ISO/IEC 9899 only
      guarantees that a long can represent values up to 2^31-1. */
    assert(n >= 0 && n < 32);
    /* Ensure that only the bits to be output are nonzero. */
    assert(!(v & (~JAS_ONES(n))));

    /* Put the desired number of bits to the specified bit stream. */
    m = n - 1;
    while (--n >= 0) {
        if (jpc_bitstream_putbit(bitstream, (v >> m) & 1) == EOF) {
            return EOF;
        }
        v <<= 1;
    }
    return 0;
}

/******************************************************************************\
* Code for buffer filling and flushing.
\******************************************************************************/

/* Fill the buffer for a bit stream. */
int jpc_bitstream_fillbuf(jpc_bitstream_t *bitstream)
{
    int c;
    /* Note: The count has already been decremented by the caller. */
    assert(bitstream->openmode_ & JPC_BITSTREAM_READ);
    assert(bitstream->cnt_ <= 0);

    if (bitstream->flags_ & JPC_BITSTREAM_ERR) {
        bitstream->cnt_ = 0;
        return -1;
    }

    if (bitstream->flags_ & JPC_BITSTREAM_EOF) {
        bitstream->buf_ = 0x7f;
        bitstream->cnt_ = 7;
        return 1;
    }

    bitstream->buf_ = (bitstream->buf_ << 8) & 0xffff;
    if ((c = jas_stream_getc((bitstream)->stream_)) == EOF) {
        bitstream->flags_ |= JPC_BITSTREAM_EOF;
        return 1;
    }
    bitstream->cnt_ = (bitstream->buf_ == 0xff00) ? 6 : 7;
    bitstream->buf_ |= c & ((1 << (bitstream->cnt_ + 1)) - 1);
    return (bitstream->buf_ >> bitstream->cnt_) & 1;
}


/******************************************************************************\
* Code related to flushing.
\******************************************************************************/

/* Does the bit stream need to be aligned to a byte boundary (considering
  the effects of bit stuffing)? */
int jpc_bitstream_needalign(jpc_bitstream_t *bitstream)
{
    if (bitstream->openmode_ & JPC_BITSTREAM_READ) {
        /* The bit stream is open for reading. */
        /* If there are any bits buffered for reading, or the
          previous byte forced a stuffed bit, alignment is
          required. */
        if ((bitstream->cnt_ < 8 && bitstream->cnt_ > 0) ||
          ((bitstream->buf_ >> 8) & 0xff) == 0xff) {
            return 1;
        }
    } else if (bitstream->openmode_ & JPC_BITSTREAM_WRITE) {
        /* The bit stream is open for writing. */
        /* If there are any bits buffered for writing, or the
          previous byte forced a stuffed bit, alignment is
          required. */
        if ((bitstream->cnt_ < 8 && bitstream->cnt_ >= 0) ||
          ((bitstream->buf_ >> 8) & 0xff) == 0xff) {
            return 1;
        }
    } else {
        /* This should not happen.  Famous last words, eh? :-) */
        assert(0);
        return -1;
    }
    return 0;
}

/* How many additional bytes would be output if we align the bit stream? */
int jpc_bitstream_pending(jpc_bitstream_t *bitstream)
{
    if (bitstream->openmode_ & JPC_BITSTREAM_WRITE) {
        /* The bit stream is being used for writing. */
#if 1
        /* XXX - Is this really correct?  Check someday... */
        if (bitstream->cnt_ < 8) {
            return 1;
        }
#else
        if (bitstream->cnt_ < 8) {
            if (((bitstream->buf_ >> 8) & 0xff) == 0xff) {
                return 2;
            }
            return 1;
        }
#endif
        return 0;
    } else {
        /* This operation should not be invoked on a bit stream that
          is being used for reading. */
        return -1;
    }
}

/* Align the bit stream to a byte boundary. */
int jpc_bitstream_align(jpc_bitstream_t *bitstream)
{
    int ret;
    if (bitstream->openmode_ & JPC_BITSTREAM_READ) {
        ret = jpc_bitstream_inalign(bitstream, 0, 0);
    } else if (bitstream->openmode_ & JPC_BITSTREAM_WRITE) {
        ret = jpc_bitstream_outalign(bitstream, 0);
    } else {
        abort();
    }
    return ret;
}

/* Align a bit stream in the input case. */
int jpc_bitstream_inalign(jpc_bitstream_t *bitstream, int fillmask,
  int filldata)
{
    int n;
    int v;
    int u;
    int numfill;
    int m;

    numfill = 7;
    m = 0;
    v = 0;
    if (bitstream->cnt_ > 0) {
        n = bitstream->cnt_;
    } else if (!bitstream->cnt_) {
        n = ((bitstream->buf_ & 0xff) == 0xff) ? 7 : 0;
    } else {
        n = 0;
    }
    if (n > 0) {
        if ((u = jpc_bitstream_getbits(bitstream, n)) < 0) {
            return -1;
        }
        m += n;
        v = (v << n) | u;
    }
    if ((bitstream->buf_ & 0xff) == 0xff) {
        if ((u = jpc_bitstream_getbits(bitstream, 7)) < 0) {
            return -1;
        }
        v = (v << 7) | u;
        m += 7;
    }
    if (m > numfill) {
        v >>= m - numfill;
    } else {
        filldata >>= numfill - m;
        fillmask >>= numfill - m;
    }
    if (((~(v ^ filldata)) & fillmask) != fillmask) {
        /* The actual fill pattern does not match the expected one. */
        return 1;
    }

    return 0;
}

/* Align a bit stream in the output case. */
int jpc_bitstream_outalign(jpc_bitstream_t *bitstream, int filldata)
{
    int n;
    int v;

    /* Ensure that this bit stream is open for writing. */
    assert(bitstream->openmode_ & JPC_BITSTREAM_WRITE);

    /* Ensure that the first bit of fill data is zero. */
    /* Note: The first bit of fill data must be zero.  If this were not
      the case, the fill data itself could cause further bit stuffing to
      be required (which would cause numerous complications). */
    assert(!(filldata & (~0x3f)));

    if (!bitstream->cnt_) {
        if ((bitstream->buf_ & 0xff) == 0xff) {
            n = 7;
            v = filldata;
        } else {
            n = 0;
            v = 0;
        }
    } else if (bitstream->cnt_ > 0 && bitstream->cnt_ < 8) {
        n = bitstream->cnt_;
        v = filldata >> (7 - n);
    } else {
        n = 0;
        v = 0;
        return 0;
    }

    /* Write the appropriate fill data to the bit stream. */
    if (n > 0) {
        if (jpc_bitstream_putbits(bitstream, n, v)) {
            return -1;
        }
    }
    if (bitstream->cnt_ < 8) {
        assert(bitstream->cnt_ >= 0 && bitstream->cnt_ < 8);
        assert((bitstream->buf_ & 0xff) != 0xff);
        /* Force the pending byte of output to be written to the
          underlying (character) stream. */
        if (jas_stream_putc(bitstream->stream_, bitstream->buf_ & 0xff) == EOF) {
            return -1;
        }
        bitstream->cnt_ = 8;
        bitstream->buf_ = (bitstream->buf_ << 8) & 0xffff;
    }

    return 0;
}
