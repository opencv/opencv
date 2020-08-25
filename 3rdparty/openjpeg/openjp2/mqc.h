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

#ifndef OPJ_MQC_H
#define OPJ_MQC_H

#include "opj_common.h"

/**
@file mqc.h
@brief Implementation of an MQ-Coder (MQC)

The functions in MQC.C have for goal to realize the MQ-coder operations. The functions
in MQC.C are used by some function in T1.C.
*/

/** @defgroup MQC MQC - Implementation of an MQ-Coder */
/*@{*/

/**
This struct defines the state of a context.
*/
typedef struct opj_mqc_state {
    /** the probability of the Least Probable Symbol (0.75->0x8000, 1.5->0xffff) */
    OPJ_UINT32 qeval;
    /** the Most Probable Symbol (0 or 1) */
    OPJ_UINT32 mps;
    /** next state if the next encoded symbol is the MPS */
    const struct opj_mqc_state *nmps;
    /** next state if the next encoded symbol is the LPS */
    const struct opj_mqc_state *nlps;
} opj_mqc_state_t;

#define MQC_NUMCTXS 19

/**
MQ coder
*/
typedef struct opj_mqc {
    /** temporary buffer where bits are coded or decoded */
    OPJ_UINT32 c;
    /** only used by MQ decoder */
    OPJ_UINT32 a;
    /** number of bits already read or free to write */
    OPJ_UINT32 ct;
    /* only used by decoder, to count the number of times a terminating 0xFF >0x8F marker is read */
    OPJ_UINT32 end_of_byte_stream_counter;
    /** pointer to the current position in the buffer */
    OPJ_BYTE *bp;
    /** pointer to the start of the buffer */
    OPJ_BYTE *start;
    /** pointer to the end of the buffer */
    OPJ_BYTE *end;
    /** Array of contexts */
    const opj_mqc_state_t *ctxs[MQC_NUMCTXS];
    /** Active context */
    const opj_mqc_state_t **curctx;
    /* lut_ctxno_zc shifted by (1 << 9) * bandno */
    const OPJ_BYTE* lut_ctxno_zc_orient;
    /** Original value of the 2 bytes at end[0] and end[1] */
    OPJ_BYTE backup[OPJ_COMMON_CBLK_DATA_EXTRA];
} opj_mqc_t;

#include "mqc_inl.h"

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
Return the number of bytes written/read since initialisation
@param mqc MQC handle
@return Returns the number of bytes already encoded
*/
OPJ_UINT32 opj_mqc_numbytes(opj_mqc_t *mqc);
/**
Reset the states of all the context of the coder/decoder
(each context is set to a state where 0 and 1 are more or less equiprobable)
@param mqc MQC handle
*/
void opj_mqc_resetstates(opj_mqc_t *mqc);
/**
Set the state of a particular context
@param mqc MQC handle
@param ctxno Number that identifies the context
@param msb The MSB of the new state of the context
@param prob Number that identifies the probability of the symbols for the new state of the context
*/
void opj_mqc_setstate(opj_mqc_t *mqc, OPJ_UINT32 ctxno, OPJ_UINT32 msb,
                      OPJ_INT32 prob);
/**
Initialize the encoder
@param mqc MQC handle
@param bp Pointer to the start of the buffer where the bytes will be written
*/
void opj_mqc_init_enc(opj_mqc_t *mqc, OPJ_BYTE *bp);
/**
Set the current context used for coding/decoding
@param mqc MQC handle
@param ctxno Number that identifies the context
*/
#define opj_mqc_setcurctx(mqc, ctxno)   (mqc)->curctx = &(mqc)->ctxs[(OPJ_UINT32)(ctxno)]
/**
Encode a symbol using the MQ-coder
@param mqc MQC handle
@param d The symbol to be encoded (0 or 1)
*/
void opj_mqc_encode(opj_mqc_t *mqc, OPJ_UINT32 d);
/**
Flush the encoder, so that all remaining data is written
@param mqc MQC handle
*/
void opj_mqc_flush(opj_mqc_t *mqc);
/**
BYPASS mode switch, initialization operation.
JPEG 2000 p 505.
@param mqc MQC handle
*/
void opj_mqc_bypass_init_enc(opj_mqc_t *mqc);

/** Return number of extra bytes to add to opj_mqc_numbytes() for the²
    size of a non-terminating BYPASS pass
@param mqc MQC handle
@param erterm 1 if ERTERM is enabled, 0 otherwise
*/
OPJ_UINT32 opj_mqc_bypass_get_extra_bytes(opj_mqc_t *mqc, OPJ_BOOL erterm);

/**
BYPASS mode switch, coding operation.
JPEG 2000 p 505.
@param mqc MQC handle
@param d The symbol to be encoded (0 or 1)
*/
void opj_mqc_bypass_enc(opj_mqc_t *mqc, OPJ_UINT32 d);
/**
BYPASS mode switch, flush operation
@param mqc MQC handle
@param erterm 1 if ERTERM is enabled, 0 otherwise
*/
void opj_mqc_bypass_flush_enc(opj_mqc_t *mqc, OPJ_BOOL erterm);
/**
RESET mode switch
@param mqc MQC handle
*/
void opj_mqc_reset_enc(opj_mqc_t *mqc);

#ifdef notdef
/**
RESTART mode switch (TERMALL)
@param mqc MQC handle
@return Returns 1 (always)
*/
OPJ_UINT32 opj_mqc_restart_enc(opj_mqc_t *mqc);
#endif

/**
RESTART mode switch (TERMALL) reinitialisation
@param mqc MQC handle
*/
void opj_mqc_restart_init_enc(opj_mqc_t *mqc);
/**
ERTERM mode switch (PTERM)
@param mqc MQC handle
*/
void opj_mqc_erterm_enc(opj_mqc_t *mqc);
/**
SEGMARK mode switch (SEGSYM)
@param mqc MQC handle
*/
void opj_mqc_segmark_enc(opj_mqc_t *mqc);

/**
Initialize the decoder for MQ decoding.

opj_mqc_finish_dec() must be absolutely called after finishing the decoding
passes, so as to restore the bytes temporarily overwritten.

@param mqc MQC handle
@param bp Pointer to the start of the buffer from which the bytes will be read
          Note that OPJ_COMMON_CBLK_DATA_EXTRA bytes at the end of the buffer
          will be temporarily overwritten with an artificial 0xFF 0xFF marker.
          (they will be backuped in the mqc structure to be restored later)
          So bp must be at least len + OPJ_COMMON_CBLK_DATA_EXTRA large, and
          writable.
@param len Length of the input buffer
@param extra_writable_bytes Indicate how many bytes after len are writable.
                            This is to indicate your consent that bp must be
                            large enough.
*/
void opj_mqc_init_dec(opj_mqc_t *mqc, OPJ_BYTE *bp, OPJ_UINT32 len,
                      OPJ_UINT32 extra_writable_bytes);

/**
Initialize the decoder for RAW decoding.

opj_mqc_finish_dec() must be absolutely called after finishing the decoding
passes, so as to restore the bytes temporarily overwritten.

@param mqc MQC handle
@param bp Pointer to the start of the buffer from which the bytes will be read
          Note that OPJ_COMMON_CBLK_DATA_EXTRA bytes at the end of the buffer
          will be temporarily overwritten with an artificial 0xFF 0xFF marker.
          (they will be backuped in the mqc structure to be restored later)
          So bp must be at least len + OPJ_COMMON_CBLK_DATA_EXTRA large, and
          writable.
@param len Length of the input buffer
@param extra_writable_bytes Indicate how many bytes after len are writable.
                            This is to indicate your consent that bp must be
                            large enough.
*/
void opj_mqc_raw_init_dec(opj_mqc_t *mqc, OPJ_BYTE *bp, OPJ_UINT32 len,
                          OPJ_UINT32 extra_writable_bytes);


/**
Terminate RAW/MQC decoding

This restores the bytes temporarily overwritten by opj_mqc_init_dec()/
opj_mqc_raw_init_dec()

@param mqc MQC handle
*/
void opq_mqc_finish_dec(opj_mqc_t *mqc);

/**
Decode a symbol
@param mqc MQC handle
@return Returns the decoded symbol (0 or 1)
*/
/*static INLINE OPJ_UINT32 opj_mqc_decode(opj_mqc_t * const mqc);*/
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_MQC_H */
