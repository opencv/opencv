/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
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
 * MQ Arithmetic Decoder
 *
 * $Id: jpc_mqdec.c,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>

#include "jasper/jas_types.h"
#include "jasper/jas_malloc.h"
#include "jasper/jas_math.h"
#include "jasper/jas_debug.h"

#include "jpc_mqdec.h"

/******************************************************************************\
* Local macros.
\******************************************************************************/

#if defined(DEBUG)
#define	MQDEC_CALL(n, x) \
	((jas_getdbglevel() >= (n)) ? ((void)(x)) : ((void)0))
#else
#define	MQDEC_CALL(n, x)
#endif

/******************************************************************************\
* Local function prototypes.
\******************************************************************************/

static void jpc_mqdec_bytein(jpc_mqdec_t *mqdec);

/******************************************************************************\
* Code for creation and destruction of a MQ decoder.
\******************************************************************************/

/* Create a MQ decoder. */
jpc_mqdec_t *jpc_mqdec_create(int maxctxs, jas_stream_t *in)
{
	jpc_mqdec_t *mqdec;

	/* There must be at least one context. */
	assert(maxctxs > 0);

	/* Allocate memory for the MQ decoder. */
	if (!(mqdec = jas_malloc(sizeof(jpc_mqdec_t)))) {
		goto error;
	}
	mqdec->in = in;
	mqdec->maxctxs = maxctxs;
	/* Allocate memory for the per-context state information. */
	if (!(mqdec->ctxs = jas_malloc(mqdec->maxctxs * sizeof(jpc_mqstate_t *)))) {
		goto error;
	}
	/* Set the current context to the first context. */
	mqdec->curctx = mqdec->ctxs;

	/* If an input stream has been associated with the MQ decoder,
	  initialize the decoder state from the stream. */
	if (mqdec->in) {
		jpc_mqdec_init(mqdec);
	}
	/* Initialize the per-context state information. */
	jpc_mqdec_setctxs(mqdec, 0, 0);

	return mqdec;

error:
	/* Oops...  Something has gone wrong. */
	if (mqdec) {
		jpc_mqdec_destroy(mqdec);
	}
	return 0;
}

/* Destroy a MQ decoder. */
void jpc_mqdec_destroy(jpc_mqdec_t *mqdec)
{
	if (mqdec->ctxs) {
		jas_free(mqdec->ctxs);
	}
	jas_free(mqdec);
}

/******************************************************************************\
* Code for initialization of a MQ decoder.
\******************************************************************************/

/* Initialize the state of a MQ decoder. */

void jpc_mqdec_init(jpc_mqdec_t *mqdec)
{
	int c;

	mqdec->eof = 0;
	mqdec->creg = 0;
	/* Get the next byte from the input stream. */
	if ((c = jas_stream_getc(mqdec->in)) == EOF) {
		/* We have encountered an I/O error or EOF. */
		c = 0xff;
		mqdec->eof = 1;
	}
	mqdec->inbuffer = c;
	mqdec->creg += mqdec->inbuffer << 16;
	jpc_mqdec_bytein(mqdec);
	mqdec->creg <<= 7;
	mqdec->ctreg -= 7;
	mqdec->areg = 0x8000;
}

/* Set the input stream for a MQ decoder. */

void jpc_mqdec_setinput(jpc_mqdec_t *mqdec, jas_stream_t *in)
{
	mqdec->in = in;
}

/* Initialize one or more contexts. */

void jpc_mqdec_setctxs(jpc_mqdec_t *mqdec, int numctxs, jpc_mqctx_t *ctxs)
{
	jpc_mqstate_t **ctx;
	int n;

	ctx = mqdec->ctxs;
	n = JAS_MIN(mqdec->maxctxs, numctxs);
	while (--n >= 0) {
		*ctx = &jpc_mqstates[2 * ctxs->ind + ctxs->mps];
		++ctx;
		++ctxs;
	}
	n = mqdec->maxctxs - numctxs;
	while (--n >= 0) {
		*ctx = &jpc_mqstates[0];
		++ctx;
	}
}

/* Initialize a context. */

void jpc_mqdec_setctx(jpc_mqdec_t *mqdec, int ctxno, jpc_mqctx_t *ctx)
{
	jpc_mqstate_t **ctxi;
	ctxi = &mqdec->ctxs[ctxno];
	*ctxi = &jpc_mqstates[2 * ctx->ind + ctx->mps];
}

/******************************************************************************\
* Code for decoding a bit.
\******************************************************************************/

/* Decode a bit. */

int jpc_mqdec_getbit_func(register jpc_mqdec_t *mqdec)
{
	int bit;
	JAS_DBGLOG(100, ("jpc_mqdec_getbit_func(%p)\n", mqdec));
	MQDEC_CALL(100, jpc_mqdec_dump(mqdec, stderr));
	bit = jpc_mqdec_getbit_macro(mqdec);
	MQDEC_CALL(100, jpc_mqdec_dump(mqdec, stderr));
	JAS_DBGLOG(100, ("ctx = %d, decoded %d\n", mqdec->curctx -
	  mqdec->ctxs, bit));
	return bit;
}

/* Apply MPS_EXCHANGE algorithm (with RENORMD). */
int jpc_mqdec_mpsexchrenormd(register jpc_mqdec_t *mqdec)
{
	int ret;
	register jpc_mqstate_t *state = *mqdec->curctx;
	jpc_mqdec_mpsexchange(mqdec->areg, state->qeval, mqdec->curctx, ret);
	jpc_mqdec_renormd(mqdec->areg, mqdec->creg, mqdec->ctreg, mqdec->in,
	  mqdec->eof, mqdec->inbuffer);
	return ret;
}

/* Apply LPS_EXCHANGE algorithm (with RENORMD). */
int jpc_mqdec_lpsexchrenormd(register jpc_mqdec_t *mqdec)
{
	int ret;
	register jpc_mqstate_t *state = *mqdec->curctx;
	jpc_mqdec_lpsexchange(mqdec->areg, state->qeval, mqdec->curctx, ret);
	jpc_mqdec_renormd(mqdec->areg, mqdec->creg, mqdec->ctreg, mqdec->in,
	  mqdec->eof, mqdec->inbuffer);
	return ret;
}

/******************************************************************************\
* Support code.
\******************************************************************************/

/* Apply the BYTEIN algorithm. */
static void jpc_mqdec_bytein(jpc_mqdec_t *mqdec)
{
	int c;
	unsigned char prevbuf;

	if (!mqdec->eof) {
		if ((c = jas_stream_getc(mqdec->in)) == EOF) {
			mqdec->eof = 1;
			c = 0xff;
		}
		prevbuf = mqdec->inbuffer;
		mqdec->inbuffer = c;
		if (prevbuf == 0xff) {
			if (c > 0x8f) {
				mqdec->creg += 0xff00;
				mqdec->ctreg = 8;
			} else {
				mqdec->creg += c << 9;
				mqdec->ctreg = 7;
			}
		} else {
			mqdec->creg += c << 8;
			mqdec->ctreg = 8;
		}
	} else {
		mqdec->creg += 0xff00;
		mqdec->ctreg = 8;
	}
}

/******************************************************************************\
* Code for debugging.
\******************************************************************************/

/* Dump a MQ decoder to a stream for debugging. */

void jpc_mqdec_dump(jpc_mqdec_t *mqdec, FILE *out)
{
	fprintf(out, "MQDEC A = %08lx, C = %08lx, CT=%08lx, ",
	  (unsigned long) mqdec->areg, (unsigned long) mqdec->creg,
	  (unsigned long) mqdec->ctreg);
	fprintf(out, "CTX = %d, ", (int)(mqdec->curctx - mqdec->ctxs));
	fprintf(out, "IND %d, MPS %d, QEVAL %x\n", (int)(*mqdec->curctx -
	  jpc_mqstates), (int)(*mqdec->curctx)->mps, (int)(*mqdec->curctx)->qeval);
}
