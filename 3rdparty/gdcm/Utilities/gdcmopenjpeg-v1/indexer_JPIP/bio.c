/*
 * Copyright (c) 2001-2002, David Janssens
 * Copyright (c) 2003, Yannick Verschueren
 * Copyright (c) 2003,  Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#include "bio.h"
#include <setjmp.h>

static unsigned char *bio_start, *bio_end, *bio_bp;
static unsigned int bio_buf;
static int bio_ct;

extern jmp_buf j2k_error;

/// <summary>
/// Number of bytes written.
/// </summary>
int bio_numbytes() {
    return bio_bp-bio_start;
}

/// <summary>
/// Init decoder.
/// </summary>
/// <param name="bp">Input buffer</param>
/// <param name="len">Input buffer length</param>
void bio_init_dec(unsigned char *bp, int len) {
    bio_start=bp;
    bio_end=bp+len;
    bio_bp=bp;
    bio_buf=0;
    bio_ct=0;
}

int bio_byteout()
{
	bio_buf = (bio_buf << 8) & 0xffff;
	bio_ct = bio_buf == 0xff00 ? 7 : 8;
	if (bio_bp >= bio_end)
		return 1;
	*bio_bp++ = bio_buf >> 8;
	return 0;
}

/// <summary>
/// Read byte. 
/// </summary>
int bio_bytein() {
    bio_buf=(bio_buf<<8)&0xffff;
    bio_ct=bio_buf==0xff00?7:8;
    if (bio_bp>=bio_end) return 1; //longjmp(j2k_error, 1);
    bio_buf|=*bio_bp++;
    return 0;
}

/// <summary>
/// Read bit.
/// </summary>
int bio_getbit() {
    if (bio_ct==0) {
        bio_bytein();
    }
    bio_ct--;
    return (bio_buf>>bio_ct)&1;
}

/// <summary>
/// Read bits.
/// </summary>
/// <param name="n">Number of bits to read</param>
int bio_read(int n) {
    int i, v;
    v=0;
    for (i=n-1; i>=0; i--) {
        v+=bio_getbit()<<i;
    }
    return v;
}

/// <summary>
/// Flush bits.
/// </summary>
int bio_flush() {
    bio_ct=0;
    bio_byteout();
    if (bio_ct==7) {
        bio_ct=0;
       if ( bio_byteout()) return 1;;
    }
    return 0;
}

/// <summary>
/// </summary>
int bio_inalign() {
    bio_ct=0;
    if ((bio_buf&0xff)==0xff) {
       if( bio_bytein()) return 1;
        bio_ct=0;
    }
    return 0;
}
