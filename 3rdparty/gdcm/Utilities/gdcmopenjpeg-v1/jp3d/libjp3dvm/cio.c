/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

/* ----------------------------------------------------------------------- */

opj_cio_t* OPJ_CALLCONV opj_cio_open(opj_common_ptr cinfo, unsigned char *buffer, int length) {
	opj_cp_t *cp = NULL;
	opj_cio_t *cio = (opj_cio_t*)opj_malloc(sizeof(opj_cio_t));
	if(!cio) return NULL;
	cio->cinfo = cinfo;
	if(buffer && length) {
		/* wrap a user buffer containing the encoded image */
		cio->openmode = OPJ_STREAM_READ;
		cio->buffer = buffer;
		cio->length = length;
	}
	else if(!buffer && !length && cinfo) {
		/* allocate a buffer for the encoded image */
		cio->openmode = OPJ_STREAM_WRITE;
		switch(cinfo->codec_format) {
			case CODEC_J3D:
			case CODEC_J2K:
				cp = ((opj_j3d_t*)cinfo->j3d_handle)->cp;
				break;
			default:
				opj_free(cio);
				return NULL;
		}
		cio->length = cp->tdx * cp->tdy * cp->tdz * cp->tw * cp->th * cp->tl * 4;
		cio->buffer = (unsigned char *)opj_malloc(cio->length);
		if(!cio->buffer) {
			opj_free(cio);
			return NULL;
		}
	}
	else {
		opj_free(cio);
		return NULL;
	}

	/* Initialize byte IO */
	cio->start = cio->buffer;
	cio->end = cio->buffer + cio->length;
	cio->bp = cio->buffer;

	return cio;
}

void OPJ_CALLCONV opj_cio_close(opj_cio_t *cio) {
	if(cio) {
		if(cio->openmode == OPJ_STREAM_WRITE) {
			/* destroy the allocated buffer */
			opj_free(cio->buffer);
		}
		/* destroy the cio */
		opj_free(cio);
	}
}


/* ----------------------------------------------------------------------- */

/*
 * Get position in byte stream.
 */
int OPJ_CALLCONV cio_tell(opj_cio_t *cio) {
	return cio->bp - cio->start;
}

/*
 * Set position in byte stream.
 *
 * pos : position, in number of bytes, from the beginning of the stream
 */
void OPJ_CALLCONV cio_seek(opj_cio_t *cio, int pos) {
	cio->bp = cio->start + pos;
}

/*
 * Number of bytes left before the end of the stream.
 */
int cio_numbytesleft(opj_cio_t *cio) {
	return cio->end - cio->bp;
}

/*
 * Get pointer to the current position in the stream.
 */
unsigned char *cio_getbp(opj_cio_t *cio) {
	return cio->bp;
}

/*
 * Write a byte.
 */
bool cio_byteout(opj_cio_t *cio, unsigned char v) {
	if (cio->bp >= cio->end) {
		opj_event_msg(cio->cinfo, EVT_ERROR, "write error\n");
		return false;
	}
	*cio->bp++ = v;
	return true;
}

/*
 * Read a byte.
 */
unsigned char cio_bytein(opj_cio_t *cio) {
	if (cio->bp >= cio->end) {
		opj_event_msg(cio->cinfo, EVT_ERROR, "read error\n");
		return 0;
	}
	return *cio->bp++;
}

/*
 * Write some bytes.
 *
 * v : value to write
 * n : number of bytes to write
 */
unsigned int cio_write(opj_cio_t *cio, unsigned int v, int n) {
	int i;
	for (i = n - 1; i >= 0; i--) {
		if( !cio_byteout(cio, (unsigned char) ((v >> (i << 3)) & 0xff)) )
			return 0;
	}
	return n;
}

/*
 * Read some bytes.
 *
 * n : number of bytes to read
 *
 * return : value of the n bytes read
 */
unsigned int cio_read(opj_cio_t *cio, int n) {
	int i;
	unsigned int v;
	v = 0;
	for (i = n - 1; i >= 0; i--) {
		v += cio_bytein(cio) << (i << 3);
	}
	return v;
}

/* 
 * Skip some bytes.
 *
 * n : number of bytes to skip
 */
void cio_skip(opj_cio_t *cio, int n) {
	cio->bp += n;
}

/*
 * Write some bytes.
 *
 * v : value to write
 * n : number of bytes to write
 */
int cio_write_int(opj_cio_t *cio, int v, int n) {
	int i;
	for (i = n - 1; i >= 0; i--) {
		if( !cio_byteout(cio, (char) ((v >> (i << 3)) & 0xff)) )
			return 0;
	}
	return n;
}

/*
 * Read some bytes.
 *
 * n : number of bytes to read
 *
 * return : value of the n bytes read
 */
int cio_read_int(opj_cio_t *cio, int n) {
	int i;
	int v;
	v = 0;
	for (i = n - 1; i >= 0; i--) {
		v += cio_bytein(cio) << (i << 3);
	}
	return v;
}

