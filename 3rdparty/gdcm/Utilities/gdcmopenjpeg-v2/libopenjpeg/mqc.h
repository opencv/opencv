/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
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

#ifndef __MQC_H
#define __MQC_H
/**
@file mqc.h
@brief Implementation of an MQ-Coder (MQC)

The functions in MQC.C have for goal to realize the MQ-coder operations. The functions
in MQC.C are used by some function in T1.C.
*/
#include "openjpeg.h"
/** @defgroup MQC MQC - Implementation of an MQ-Coder */
/*@{*/

/**
This struct defines the state of a context.
*/
typedef struct opj_mqc_state {
  /** the probability of the Least Probable Symbol (0.75->0x8000, 1.5->0xffff) */
  OPJ_UINT32 qeval;
  /** the Most Probable Symbol (0 or 1) */
  OPJ_INT32 mps;
  /** next state if the next encoded symbol is the MPS */
  struct opj_mqc_state *nmps;
  /** next state if the next encoded symbol is the LPS */
  struct opj_mqc_state *nlps;
} opj_mqc_state_t;

#define MQC_NUMCTXS 32

/**
MQ coder
*/
typedef struct opj_mqc {
  OPJ_UINT32 c;
  OPJ_UINT32 a;
  OPJ_UINT32 ct;
  OPJ_BYTE *bp;
  OPJ_BYTE *start;
  OPJ_BYTE *end;
  opj_mqc_state_t *ctxs[MQC_NUMCTXS];
  opj_mqc_state_t **curctx;
} opj_mqc_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Create a new MQC handle
@return Returns a new MQC handle if successful, returns NULL otherwise
*/
opj_mqc_t* mqc_create(void);
/**
Destroy a previously created MQC handle
@param mqc MQC handle to destroy
*/
void mqc_destroy(opj_mqc_t *mqc);
/**
Return the number of bytes written/read since initialisation
@param mqc MQC handle
@return Returns the number of bytes already encoded
*/
OPJ_UINT32 mqc_numbytes(opj_mqc_t *mqc);
/**
Reset the states of all the context of the coder/decoder
(each context is set to a state where 0 and 1 are more or less equiprobable)
@param mqc MQC handle
*/
void mqc_resetstates(opj_mqc_t *mqc);
/**
Set the state of a particular context
@param mqc MQC handle
@param ctxno Number that identifies the context
@param msb The MSB of the new state of the context
@param prob Number that identifies the probability of the symbols for the new state of the context
*/
void mqc_setstate(opj_mqc_t *mqc, OPJ_UINT32 ctxno, OPJ_UINT32 msb, OPJ_INT32 prob);
/**
Initialize the encoder
@param mqc MQC handle
@param bp Pointer to the start of the buffer where the bytes will be written
*/
void mqc_init_enc(opj_mqc_t *mqc, OPJ_BYTE *bp);
/**
Set the current context used for coding/decoding
@param mqc MQC handle
@param ctxno Number that identifies the context
*/
#define mqc_setcurctx(mqc, ctxno)  (mqc)->curctx = &(mqc)->ctxs[(OPJ_UINT32)(ctxno)]
/**
Encode a symbol using the MQ-coder
@param mqc MQC handle
@param d The symbol to be encoded (0 or 1)
*/
void mqc_encode(opj_mqc_t *mqc, OPJ_UINT32 d);
/**
Flush the encoder, so that all remaining data is written
@param mqc MQC handle
*/
void mqc_flush(opj_mqc_t *mqc);
/**
BYPASS mode switch, initialization operation.
JPEG 2000 p 505.
<h2>Not fully implemented and tested !!</h2>
@param mqc MQC handle
*/
void mqc_bypass_init_enc(opj_mqc_t *mqc);
/**
BYPASS mode switch, coding operation.
JPEG 2000 p 505.
<h2>Not fully implemented and tested !!</h2>
@param mqc MQC handle
@param d The symbol to be encoded (0 or 1)
*/
void mqc_bypass_enc(opj_mqc_t *mqc, OPJ_UINT32 d);
/**
BYPASS mode switch, flush operation
<h2>Not fully implemented and tested !!</h2>
@param mqc MQC handle
@return Returns 1 (always)
*/
OPJ_UINT32 mqc_bypass_flush_enc(opj_mqc_t *mqc);
/**
RESET mode switch
@param mqc MQC handle
*/
void mqc_reset_enc(opj_mqc_t *mqc);
/**
RESTART mode switch (TERMALL)
@param mqc MQC handle
@return Returns 1 (always)
*/
OPJ_UINT32 mqc_restart_enc(opj_mqc_t *mqc);
/**
RESTART mode switch (TERMALL) reinitialisation
@param mqc MQC handle
*/
void mqc_restart_init_enc(opj_mqc_t *mqc);
/**
ERTERM mode switch (PTERM)
@param mqc MQC handle
*/
void mqc_erterm_enc(opj_mqc_t *mqc);
/**
SEGMARK mode switch (SEGSYM)
@param mqc MQC handle
*/
void mqc_segmark_enc(opj_mqc_t *mqc);
/**
Initialize the decoder
@param mqc MQC handle
@param bp Pointer to the start of the buffer from which the bytes will be read
@param len Length of the input buffer
*/
void mqc_init_dec(opj_mqc_t *mqc, OPJ_BYTE *bp, OPJ_UINT32 len);
/**
Decode a symbol
@param mqc MQC handle
@return Returns the decoded symbol (0 or 1)
*/
OPJ_UINT32 mqc_decode(opj_mqc_t *mqc);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __MQC_H */
