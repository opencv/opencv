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
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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

#include "opj_includes.h"

#include <assert.h>

/** @defgroup MQC MQC - Implementation of an MQ-Coder */
/*@{*/

/** @name Local static functions */
/*@{*/

/**
Fill mqc->c with 1's for flushing
@param mqc MQC handle
*/
static void opj_mqc_setbits(opj_mqc_t *mqc);
/*@}*/

/*@}*/

/* <summary> */
/* This array defines all the possible states for a context. */
/* </summary> */
static const opj_mqc_state_t mqc_states[47 * 2] = {
    {0x5601, 0, &mqc_states[2], &mqc_states[3]},
    {0x5601, 1, &mqc_states[3], &mqc_states[2]},
    {0x3401, 0, &mqc_states[4], &mqc_states[12]},
    {0x3401, 1, &mqc_states[5], &mqc_states[13]},
    {0x1801, 0, &mqc_states[6], &mqc_states[18]},
    {0x1801, 1, &mqc_states[7], &mqc_states[19]},
    {0x0ac1, 0, &mqc_states[8], &mqc_states[24]},
    {0x0ac1, 1, &mqc_states[9], &mqc_states[25]},
    {0x0521, 0, &mqc_states[10], &mqc_states[58]},
    {0x0521, 1, &mqc_states[11], &mqc_states[59]},
    {0x0221, 0, &mqc_states[76], &mqc_states[66]},
    {0x0221, 1, &mqc_states[77], &mqc_states[67]},
    {0x5601, 0, &mqc_states[14], &mqc_states[13]},
    {0x5601, 1, &mqc_states[15], &mqc_states[12]},
    {0x5401, 0, &mqc_states[16], &mqc_states[28]},
    {0x5401, 1, &mqc_states[17], &mqc_states[29]},
    {0x4801, 0, &mqc_states[18], &mqc_states[28]},
    {0x4801, 1, &mqc_states[19], &mqc_states[29]},
    {0x3801, 0, &mqc_states[20], &mqc_states[28]},
    {0x3801, 1, &mqc_states[21], &mqc_states[29]},
    {0x3001, 0, &mqc_states[22], &mqc_states[34]},
    {0x3001, 1, &mqc_states[23], &mqc_states[35]},
    {0x2401, 0, &mqc_states[24], &mqc_states[36]},
    {0x2401, 1, &mqc_states[25], &mqc_states[37]},
    {0x1c01, 0, &mqc_states[26], &mqc_states[40]},
    {0x1c01, 1, &mqc_states[27], &mqc_states[41]},
    {0x1601, 0, &mqc_states[58], &mqc_states[42]},
    {0x1601, 1, &mqc_states[59], &mqc_states[43]},
    {0x5601, 0, &mqc_states[30], &mqc_states[29]},
    {0x5601, 1, &mqc_states[31], &mqc_states[28]},
    {0x5401, 0, &mqc_states[32], &mqc_states[28]},
    {0x5401, 1, &mqc_states[33], &mqc_states[29]},
    {0x5101, 0, &mqc_states[34], &mqc_states[30]},
    {0x5101, 1, &mqc_states[35], &mqc_states[31]},
    {0x4801, 0, &mqc_states[36], &mqc_states[32]},
    {0x4801, 1, &mqc_states[37], &mqc_states[33]},
    {0x3801, 0, &mqc_states[38], &mqc_states[34]},
    {0x3801, 1, &mqc_states[39], &mqc_states[35]},
    {0x3401, 0, &mqc_states[40], &mqc_states[36]},
    {0x3401, 1, &mqc_states[41], &mqc_states[37]},
    {0x3001, 0, &mqc_states[42], &mqc_states[38]},
    {0x3001, 1, &mqc_states[43], &mqc_states[39]},
    {0x2801, 0, &mqc_states[44], &mqc_states[38]},
    {0x2801, 1, &mqc_states[45], &mqc_states[39]},
    {0x2401, 0, &mqc_states[46], &mqc_states[40]},
    {0x2401, 1, &mqc_states[47], &mqc_states[41]},
    {0x2201, 0, &mqc_states[48], &mqc_states[42]},
    {0x2201, 1, &mqc_states[49], &mqc_states[43]},
    {0x1c01, 0, &mqc_states[50], &mqc_states[44]},
    {0x1c01, 1, &mqc_states[51], &mqc_states[45]},
    {0x1801, 0, &mqc_states[52], &mqc_states[46]},
    {0x1801, 1, &mqc_states[53], &mqc_states[47]},
    {0x1601, 0, &mqc_states[54], &mqc_states[48]},
    {0x1601, 1, &mqc_states[55], &mqc_states[49]},
    {0x1401, 0, &mqc_states[56], &mqc_states[50]},
    {0x1401, 1, &mqc_states[57], &mqc_states[51]},
    {0x1201, 0, &mqc_states[58], &mqc_states[52]},
    {0x1201, 1, &mqc_states[59], &mqc_states[53]},
    {0x1101, 0, &mqc_states[60], &mqc_states[54]},
    {0x1101, 1, &mqc_states[61], &mqc_states[55]},
    {0x0ac1, 0, &mqc_states[62], &mqc_states[56]},
    {0x0ac1, 1, &mqc_states[63], &mqc_states[57]},
    {0x09c1, 0, &mqc_states[64], &mqc_states[58]},
    {0x09c1, 1, &mqc_states[65], &mqc_states[59]},
    {0x08a1, 0, &mqc_states[66], &mqc_states[60]},
    {0x08a1, 1, &mqc_states[67], &mqc_states[61]},
    {0x0521, 0, &mqc_states[68], &mqc_states[62]},
    {0x0521, 1, &mqc_states[69], &mqc_states[63]},
    {0x0441, 0, &mqc_states[70], &mqc_states[64]},
    {0x0441, 1, &mqc_states[71], &mqc_states[65]},
    {0x02a1, 0, &mqc_states[72], &mqc_states[66]},
    {0x02a1, 1, &mqc_states[73], &mqc_states[67]},
    {0x0221, 0, &mqc_states[74], &mqc_states[68]},
    {0x0221, 1, &mqc_states[75], &mqc_states[69]},
    {0x0141, 0, &mqc_states[76], &mqc_states[70]},
    {0x0141, 1, &mqc_states[77], &mqc_states[71]},
    {0x0111, 0, &mqc_states[78], &mqc_states[72]},
    {0x0111, 1, &mqc_states[79], &mqc_states[73]},
    {0x0085, 0, &mqc_states[80], &mqc_states[74]},
    {0x0085, 1, &mqc_states[81], &mqc_states[75]},
    {0x0049, 0, &mqc_states[82], &mqc_states[76]},
    {0x0049, 1, &mqc_states[83], &mqc_states[77]},
    {0x0025, 0, &mqc_states[84], &mqc_states[78]},
    {0x0025, 1, &mqc_states[85], &mqc_states[79]},
    {0x0015, 0, &mqc_states[86], &mqc_states[80]},
    {0x0015, 1, &mqc_states[87], &mqc_states[81]},
    {0x0009, 0, &mqc_states[88], &mqc_states[82]},
    {0x0009, 1, &mqc_states[89], &mqc_states[83]},
    {0x0005, 0, &mqc_states[90], &mqc_states[84]},
    {0x0005, 1, &mqc_states[91], &mqc_states[85]},
    {0x0001, 0, &mqc_states[90], &mqc_states[86]},
    {0x0001, 1, &mqc_states[91], &mqc_states[87]},
    {0x5601, 0, &mqc_states[92], &mqc_states[92]},
    {0x5601, 1, &mqc_states[93], &mqc_states[93]},
};

/*
==========================================================
   local functions
==========================================================
*/

static void opj_mqc_setbits(opj_mqc_t *mqc)
{
    OPJ_UINT32 tempc = mqc->c + mqc->a;
    mqc->c |= 0xffff;
    if (mqc->c >= tempc) {
        mqc->c -= 0x8000;
    }
}

/*
==========================================================
   MQ-Coder interface
==========================================================
*/

OPJ_UINT32 opj_mqc_numbytes(opj_mqc_t *mqc)
{
    const ptrdiff_t diff = mqc->bp - mqc->start;
#if 0
    assert(diff <= 0xffffffff && diff >= 0);   /* UINT32_MAX */
#endif
    return (OPJ_UINT32)diff;
}

void opj_mqc_init_enc(opj_mqc_t *mqc, OPJ_BYTE *bp)
{
    /* To avoid the curctx pointer to be dangling, but not strictly */
    /* required as the current context is always set before encoding */
    opj_mqc_setcurctx(mqc, 0);

    /* As specified in Figure C.10 - Initialization of the encoder */
    /* (C.2.8 Initialization of the encoder (INITENC)) */
    mqc->a = 0x8000;
    mqc->c = 0;
    /* Yes, we point before the start of the buffer, but this is safe */
    /* given opj_tcd_code_block_enc_allocate_data() */
    mqc->bp = bp - 1;
    mqc->ct = 12;
    /* At this point we should test *(mqc->bp) against 0xFF, but this is not */
    /* necessary, as this is only used at the beginning of the code block */
    /* and our initial fake byte is set at 0 */
    assert(*(mqc->bp) != 0xff);

    mqc->start = bp;
    mqc->end_of_byte_stream_counter = 0;
}


void opj_mqc_flush(opj_mqc_t *mqc)
{
    /* C.2.9 Termination of coding (FLUSH) */
    /* Figure C.11 – FLUSH procedure */
    opj_mqc_setbits(mqc);
    mqc->c <<= mqc->ct;
    opj_mqc_byteout(mqc);
    mqc->c <<= mqc->ct;
    opj_mqc_byteout(mqc);

    /* It is forbidden that a coding pass ends with 0xff */
    if (*mqc->bp != 0xff) {
        /* Advance pointer so that opj_mqc_numbytes() returns a valid value */
        mqc->bp++;
    }
}

void opj_mqc_bypass_init_enc(opj_mqc_t *mqc)
{
    /* This function is normally called after at least one opj_mqc_flush() */
    /* which will have advance mqc->bp by at least 2 bytes beyond its */
    /* initial position */
    assert(mqc->bp >= mqc->start);
    mqc->c = 0;
    /* in theory we should initialize to 8, but use this special value */
    /* as a hint that opj_mqc_bypass_enc() has never been called, so */
    /* as to avoid the 0xff 0x7f elimination trick in opj_mqc_bypass_flush_enc() */
    /* to trigger when we don't have output any bit during this bypass sequence */
    /* Any value > 8 will do */
    mqc->ct = BYPASS_CT_INIT;
    /* Given that we are called after opj_mqc_flush(), the previous byte */
    /* cannot be 0xff. */
    assert(mqc->bp[-1] != 0xff);
}

void opj_mqc_bypass_enc(opj_mqc_t *mqc, OPJ_UINT32 d)
{
    if (mqc->ct == BYPASS_CT_INIT) {
        mqc->ct = 8;
    }
    mqc->ct--;
    mqc->c = mqc->c + (d << mqc->ct);
    if (mqc->ct == 0) {
        *mqc->bp = (OPJ_BYTE)mqc->c;
        mqc->ct = 8;
        /* If the previous byte was 0xff, make sure that the next msb is 0 */
        if (*mqc->bp == 0xff) {
            mqc->ct = 7;
        }
        mqc->bp++;
        mqc->c = 0;
    }
}

OPJ_UINT32 opj_mqc_bypass_get_extra_bytes(opj_mqc_t *mqc, OPJ_BOOL erterm)
{
    return (mqc->ct < 7 ||
            (mqc->ct == 7 && (erterm || mqc->bp[-1] != 0xff))) ? 1 : 0;
}

void opj_mqc_bypass_flush_enc(opj_mqc_t *mqc, OPJ_BOOL erterm)
{
    /* Is there any bit remaining to be flushed ? */
    /* If the last output byte is 0xff, we can discard it, unless */
    /* erterm is required (I'm not completely sure why in erterm */
    /* we must output 0xff 0x2a if the last byte was 0xff instead of */
    /* discarding it, but Kakadu requires it when decoding */
    /* in -fussy mode) */
    if (mqc->ct < 7 || (mqc->ct == 7 && (erterm || mqc->bp[-1] != 0xff))) {
        OPJ_BYTE bit_value = 0;
        /* If so, fill the remaining lsbs with an alternating sequence of */
        /* 0,1,... */
        /* Note: it seems the standard only requires that for a ERTERM flush */
        /* and doesn't specify what to do for a regular BYPASS flush */
        while (mqc->ct > 0) {
            mqc->ct--;
            mqc->c += (OPJ_UINT32)(bit_value << mqc->ct);
            bit_value = (OPJ_BYTE)(1U - bit_value);
        }
        *mqc->bp = (OPJ_BYTE)mqc->c;
        /* Advance pointer so that opj_mqc_numbytes() returns a valid value */
        mqc->bp++;
    } else if (mqc->ct == 7 && mqc->bp[-1] == 0xff) {
        /* Discard last 0xff */
        assert(!erterm);
        mqc->bp --;
    } else if (mqc->ct == 8 && !erterm &&
               mqc->bp[-1] == 0x7f && mqc->bp[-2] == 0xff) {
        /* Tiny optimization: discard terminating 0xff 0x7f since it is */
        /* interpreted as 0xff 0x7f [0xff 0xff] by the decoder, and given */
        /* the bit stuffing, in fact as 0xff 0xff [0xff ..] */
        /* Happens once on opj_compress -i ../MAPA.tif -o MAPA.j2k  -M 1 */
        mqc->bp -= 2;
    }

    assert(mqc->bp[-1] != 0xff);
}

void opj_mqc_reset_enc(opj_mqc_t *mqc)
{
    opj_mqc_resetstates(mqc);
    opj_mqc_setstate(mqc, T1_CTXNO_UNI, 0, 46);
    opj_mqc_setstate(mqc, T1_CTXNO_AGG, 0, 3);
    opj_mqc_setstate(mqc, T1_CTXNO_ZC, 0, 4);
}

#ifdef notdef
OPJ_UINT32 opj_mqc_restart_enc(opj_mqc_t *mqc)
{
    OPJ_UINT32 correction = 1;

    /* <flush part> */
    OPJ_INT32 n = (OPJ_INT32)(27 - 15 - mqc->ct);
    mqc->c <<= mqc->ct;
    while (n > 0) {
        opj_mqc_byteout(mqc);
        n -= (OPJ_INT32)mqc->ct;
        mqc->c <<= mqc->ct;
    }
    opj_mqc_byteout(mqc);

    return correction;
}
#endif

void opj_mqc_restart_init_enc(opj_mqc_t *mqc)
{
    /* <Re-init part> */

    /* As specified in Figure C.10 - Initialization of the encoder */
    /* (C.2.8 Initialization of the encoder (INITENC)) */
    mqc->a = 0x8000;
    mqc->c = 0;
    mqc->ct = 12;
    /* This function is normally called after at least one opj_mqc_flush() */
    /* which will have advance mqc->bp by at least 2 bytes beyond its */
    /* initial position */
    mqc->bp --;
    assert(mqc->bp >= mqc->start - 1);
    assert(*mqc->bp != 0xff);
    if (*mqc->bp == 0xff) {
        mqc->ct = 13;
    }
}

void opj_mqc_erterm_enc(opj_mqc_t *mqc)
{
    OPJ_INT32 k = (OPJ_INT32)(11 - mqc->ct + 1);

    while (k > 0) {
        mqc->c <<= mqc->ct;
        mqc->ct = 0;
        opj_mqc_byteout(mqc);
        k -= (OPJ_INT32)mqc->ct;
    }

    if (*mqc->bp != 0xff) {
        opj_mqc_byteout(mqc);
    }
}

static INLINE void opj_mqc_renorme(opj_mqc_t *mqc)
{
    opj_mqc_renorme_macro(mqc, mqc->a, mqc->c, mqc->ct);
}

/**
Encode the most probable symbol
@param mqc MQC handle
*/
static INLINE void opj_mqc_codemps(opj_mqc_t *mqc)
{
    opj_mqc_codemps_macro(mqc, mqc->curctx, mqc->a, mqc->c, mqc->ct);
}

/**
Encode the most least symbol
@param mqc MQC handle
*/
static INLINE void opj_mqc_codelps(opj_mqc_t *mqc)
{
    opj_mqc_codelps_macro(mqc, mqc->curctx, mqc->a, mqc->c, mqc->ct);
}

/**
Encode a symbol using the MQ-coder
@param mqc MQC handle
@param d The symbol to be encoded (0 or 1)
*/
static INLINE void opj_mqc_encode(opj_mqc_t *mqc, OPJ_UINT32 d)
{
    if ((*mqc->curctx)->mps == d) {
        opj_mqc_codemps(mqc);
    } else {
        opj_mqc_codelps(mqc);
    }
}

void opj_mqc_segmark_enc(opj_mqc_t *mqc)
{
    OPJ_UINT32 i;
    opj_mqc_setcurctx(mqc, 18);

    for (i = 1; i < 5; i++) {
        opj_mqc_encode(mqc, i % 2);
    }
}

static void opj_mqc_init_dec_common(opj_mqc_t *mqc,
                                    OPJ_BYTE *bp,
                                    OPJ_UINT32 len,
                                    OPJ_UINT32 extra_writable_bytes)
{
    (void)extra_writable_bytes;

    assert(extra_writable_bytes >= OPJ_COMMON_CBLK_DATA_EXTRA);
    mqc->start = bp;
    mqc->end = bp + len;
    /* Insert an artificial 0xFF 0xFF marker at end of the code block */
    /* data so that the bytein routines stop on it. This saves us comparing */
    /* the bp and end pointers */
    /* But before inserting it, backup th bytes we will overwrite */
    memcpy(mqc->backup, mqc->end, OPJ_COMMON_CBLK_DATA_EXTRA);
    mqc->end[0] = 0xFF;
    mqc->end[1] = 0xFF;
    mqc->bp = bp;
}
void opj_mqc_init_dec(opj_mqc_t *mqc, OPJ_BYTE *bp, OPJ_UINT32 len,
                      OPJ_UINT32 extra_writable_bytes)
{
    /* Implements ISO 15444-1 C.3.5 Initialization of the decoder (INITDEC) */
    /* Note: alternate "J.1 - Initialization of the software-conventions */
    /* decoder" has been tried, but does */
    /* not bring any improvement. */
    /* See https://github.com/uclouvain/openjpeg/issues/921 */
    opj_mqc_init_dec_common(mqc, bp, len, extra_writable_bytes);
    opj_mqc_setcurctx(mqc, 0);
    mqc->end_of_byte_stream_counter = 0;
    if (len == 0) {
        mqc->c = 0xff << 16;
    } else {
        mqc->c = (OPJ_UINT32)(*mqc->bp << 16);
    }

    opj_mqc_bytein(mqc);
    mqc->c <<= 7;
    mqc->ct -= 7;
    mqc->a = 0x8000;
}


void opj_mqc_raw_init_dec(opj_mqc_t *mqc, OPJ_BYTE *bp, OPJ_UINT32 len,
                          OPJ_UINT32 extra_writable_bytes)
{
    opj_mqc_init_dec_common(mqc, bp, len, extra_writable_bytes);
    mqc->c = 0;
    mqc->ct = 0;
}


void opq_mqc_finish_dec(opj_mqc_t *mqc)
{
    /* Restore the bytes overwritten by opj_mqc_init_dec_common() */
    memcpy(mqc->end, mqc->backup, OPJ_COMMON_CBLK_DATA_EXTRA);
}

void opj_mqc_resetstates(opj_mqc_t *mqc)
{
    OPJ_UINT32 i;
    for (i = 0; i < MQC_NUMCTXS; i++) {
        mqc->ctxs[i] = mqc_states;
    }
}

void opj_mqc_setstate(opj_mqc_t *mqc, OPJ_UINT32 ctxno, OPJ_UINT32 msb,
                      OPJ_INT32 prob)
{
    mqc->ctxs[ctxno] = &mqc_states[msb + (OPJ_UINT32)(prob << 1)];
}

void opj_mqc_byteout(opj_mqc_t *mqc)
{
    /* bp is initialized to start - 1 in opj_mqc_init_enc() */
    /* but this is safe, see opj_tcd_code_block_enc_allocate_data() */
    assert(mqc->bp >= mqc->start - 1);
    if (*mqc->bp == 0xff) {
        mqc->bp++;
        *mqc->bp = (OPJ_BYTE)(mqc->c >> 20);
        mqc->c &= 0xfffff;
        mqc->ct = 7;
    } else {
        if ((mqc->c & 0x8000000) == 0) {
            mqc->bp++;
            *mqc->bp = (OPJ_BYTE)(mqc->c >> 19);
            mqc->c &= 0x7ffff;
            mqc->ct = 8;
        } else {
            (*mqc->bp)++;
            if (*mqc->bp == 0xff) {
                mqc->c &= 0x7ffffff;
                mqc->bp++;
                *mqc->bp = (OPJ_BYTE)(mqc->c >> 20);
                mqc->c &= 0xfffff;
                mqc->ct = 7;
            } else {
                mqc->bp++;
                *mqc->bp = (OPJ_BYTE)(mqc->c >> 19);
                mqc->c &= 0x7ffff;
                mqc->ct = 8;
            }
        }
    }
}