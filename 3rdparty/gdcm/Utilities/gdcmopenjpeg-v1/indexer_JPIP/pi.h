/*
 * Copyright (c) 2001-2002, David Janssens
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

#ifndef __PI_H
#define __PI_H

#include "j2k.h"
#include "tcd.h"

typedef struct {
	int pdx, pdy;
	int pw, ph;
} pi_resolution_t;

typedef struct {
	int dx, dy;
	int numresolutions;
	pi_resolution_t *resolutions;
} pi_comp_t;

typedef struct {
	short int *include;
  int step_l, step_r, step_c, step_p; 
	int compno, resno, precno, layno;	/* component, resolution, precinct and layer that indentify the packet */
	int first;
	j2k_poc_t poc;
	int numcomps;
	pi_comp_t *comps;
	int tx0, ty0, tx1, ty1;
	int x, y, dx, dy;
} pi_iterator_t;								/* packet iterator */

/*
 * Create a packet iterator
 * img: raw image for which the packets will be listed
 * cp: coding paremeters
 * tileno: number that identifies the tile for which to list the packets
 * return value: returns a packet iterator that points to the first packet of the tile
 */
pi_iterator_t *pi_create(j2k_image_t * img, j2k_cp_t * cp, int tileno);

/* 
 * Modify the packet iterator to point to the next packet
 * pi: packet iterator to modify
 * return value: returns 0 if pi pointed to the last packet or else returns 1 
 */
int pi_next(pi_iterator_t * pi);

#endif
