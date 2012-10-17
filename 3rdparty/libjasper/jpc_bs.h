/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2002 Michael David Adams.
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
 * $Id: jpc_bs.h,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

#ifndef JPC_BS_H
#define JPC_BS_H

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>

#include "jasper/jas_types.h"
#include "jasper/jas_stream.h"

/******************************************************************************\
* Constants.
\******************************************************************************/

/*
 * Bit stream open mode flags.
 */

/* Bit stream open for reading. */
#define	JPC_BITSTREAM_READ	0x01
/* Bit stream open for writing. */
#define	JPC_BITSTREAM_WRITE	0x02

/*
 * Bit stream flags.
 */

/* Do not close underlying character stream. */
#define	JPC_BITSTREAM_NOCLOSE	0x01
/* End of file has been reached while reading. */
#define	JPC_BITSTREAM_EOF	0x02
/* An I/O error has occured. */
#define	JPC_BITSTREAM_ERR	0x04

/******************************************************************************\
* Types.
\******************************************************************************/

/* Bit stream class. */

typedef struct {

    /* Some miscellaneous flags. */
    int flags_;

    /* The input/output buffer. */
    uint_fast16_t buf_;

    /* The number of bits remaining in the byte being read/written. */
    int cnt_;

    /* The underlying stream associated with this bit stream. */
    jas_stream_t *stream_;

    /* The mode in which this bit stream was opened. */
    int openmode_;

} jpc_bitstream_t;

/******************************************************************************\
* Functions/macros for opening and closing bit streams..
\******************************************************************************/

/* Open a stream as a bit stream. */
jpc_bitstream_t *jpc_bitstream_sopen(jas_stream_t *stream, char *mode);

/* Close a bit stream. */
int jpc_bitstream_close(jpc_bitstream_t *bitstream);

/******************************************************************************\
* Functions/macros for reading from and writing to bit streams..
\******************************************************************************/

/* Read a bit from a bit stream. */
#if defined(DEBUG)
#define	jpc_bitstream_getbit(bitstream) \
    jpc_bitstream_getbit_func(bitstream)
#else
#define jpc_bitstream_getbit(bitstream) \
    jpc_bitstream_getbit_macro(bitstream)
#endif

/* Write a bit to a bit stream. */
#if defined(DEBUG)
#define	jpc_bitstream_putbit(bitstream, v) \
    jpc_bitstream_putbit_func(bitstream, v)
#else
#define	jpc_bitstream_putbit(bitstream, v) \
    jpc_bitstream_putbit_macro(bitstream, v)
#endif

/* Read one or more bits from a bit stream. */
long jpc_bitstream_getbits(jpc_bitstream_t *bitstream, int n);

/* Write one or more bits to a bit stream. */
int jpc_bitstream_putbits(jpc_bitstream_t *bitstream, int n, long v);

/******************************************************************************\
* Functions/macros for flushing and aligning bit streams.
\******************************************************************************/

/* Align the current position within the bit stream to the next byte
  boundary. */
int jpc_bitstream_align(jpc_bitstream_t *bitstream);

/* Align the current position in the bit stream with the next byte boundary,
  ensuring that certain bits consumed in the process match a particular
  pattern. */
int jpc_bitstream_inalign(jpc_bitstream_t *bitstream, int fillmask,
  int filldata);

/* Align the current position in the bit stream with the next byte boundary,
  writing bits from the specified pattern (if necessary) in the process. */
int jpc_bitstream_outalign(jpc_bitstream_t *bitstream, int filldata);

/* Check if a bit stream needs alignment. */
int jpc_bitstream_needalign(jpc_bitstream_t *bitstream);

/* How many additional bytes would be output if the bit stream was aligned? */
int jpc_bitstream_pending(jpc_bitstream_t *bitstream);

/******************************************************************************\
* Functions/macros for querying state information for bit streams.
\******************************************************************************/

/* Has EOF been encountered on a bit stream? */
#define jpc_bitstream_eof(bitstream) \
    ((bitstream)->flags_ & JPC_BITSTREAM_EOF)

/******************************************************************************\
* Internals.
\******************************************************************************/

/* DO NOT DIRECTLY INVOKE ANY OF THE MACROS OR FUNCTIONS BELOW.  THEY ARE
  FOR INTERNAL USE ONLY. */

int jpc_bitstream_getbit_func(jpc_bitstream_t *bitstream);

int jpc_bitstream_putbit_func(jpc_bitstream_t *bitstream, int v);

int jpc_bitstream_fillbuf(jpc_bitstream_t *bitstream);

#define	jpc_bitstream_getbit_macro(bitstream) \
    (assert((bitstream)->openmode_ & JPC_BITSTREAM_READ), \
      (--(bitstream)->cnt_ >= 0) ? \
      ((int)(((bitstream)->buf_ >> (bitstream)->cnt_) & 1)) : \
      jpc_bitstream_fillbuf(bitstream))

#define jpc_bitstream_putbit_macro(bitstream, bit) \
    (assert((bitstream)->openmode_ & JPC_BITSTREAM_WRITE), \
      (--(bitstream)->cnt_ < 0) ? \
      ((bitstream)->buf_ = ((bitstream)->buf_ << 8) & 0xffff, \
      (bitstream)->cnt_ = ((bitstream)->buf_ == 0xff00) ? 6 : 7, \
      (bitstream)->buf_ |= ((bit) & 1) << (bitstream)->cnt_, \
      (jas_stream_putc((bitstream)->stream_, (bitstream)->buf_ >> 8) == EOF) \
      ? (EOF) : ((bit) & 1)) : \
      ((bitstream)->buf_ |= ((bit) & 1) << (bitstream)->cnt_, \
      (bit) & 1))

#endif
