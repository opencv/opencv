/*
 * Copyright (c) 2001-2002, David Janssens
 * Copyright (c) 2003, Yannick Verschueren
 * Copyright (c) 2003, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#include "cio.h"
#include <setjmp.h>

static unsigned char *cio_start, *cio_end, *cio_bp;

extern jmp_buf j2k_error;

/// <summary>
/// Number of bytes written.
/// </summary>
int cio_numbytes() {
    return cio_bp-cio_start;
}

/// <summary>
/// Get position in byte stream.
/// </summary>
int cio_tell() {
    return cio_bp-cio_start;
}

/// <summary>
/// Set position in byte stream.
/// </summary>
void cio_seek(int pos) {
    cio_bp=cio_start+pos;
}

/// <summary>
/// Number of bytes left before the end of the stream.
/// </summary>
int cio_numbytesleft() {
    return cio_end-cio_bp;
}

/// <summary>
/// Get pointer to the current position in the stream.
/// </summary>
unsigned char *cio_getbp() {
    return cio_bp;
}

/// <summary>
/// Initialize byte IO.
/// </summary>
void cio_init(unsigned char *bp, int len) {
    cio_start=bp;
    cio_end=bp+len;
    cio_bp=bp;
}

/// <summary>
/// Write a byte.
/// </summary>
void cio_byteout(unsigned char v) {
  if (cio_bp>=cio_end) longjmp(j2k_error, 1);
  *cio_bp++=v;
    
}

/// <summary>
/// Read a byte.
/// </summary>
unsigned char cio_bytein() {
    if (cio_bp>=cio_end) longjmp(j2k_error, 1);
    return *cio_bp++;
}

/// <summary>
/// Write a byte.
/// </summary>
//void cio_write(unsigned int v, int n) {
void cio_write(long long v, int n) {
    int i;
    for (i=n-1; i>=0; i--) 
      {
	cio_byteout((unsigned char)((v>>(i<<3))&0xff));
      }
}

/// <summary>
/// Read some bytes.
/// </summary>
/* unsigned int cio_read(int n) { */
long long cio_read(int n) {
    int i;
    /*unsigned int v;*/
    long long v;
    v=0;
    for (i=n-1; i>=0; i--) {
      v+=cio_bytein()<<(i<<3);
    }
    return v;
}

/// <summary>
/// Write some bytes.
/// </summary>
void cio_skip(int n) {
    cio_bp+=n;
}
