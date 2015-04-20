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

#ifndef __TCD_H
#define __TCD_H

#include "j2k.h"
#include "tgt.h"

typedef struct {
    int numpasses;
    int len;
    unsigned char *data;
    int maxpasses;
    int numnewpasses;
    int newlen;
} tcd_seg_t;

typedef struct {
    int rate;
    double distortiondec;
} tcd_pass_t;

typedef struct {
    int numpasses;
    int len;
    unsigned char *data;
} tcd_layer_t;

typedef struct {
    int x0, y0, x1, y1;
    int numbps;
    int numlenbits;
    int len;
    int numpasses;
    int numnewpasses;
    int numsegs;
    tcd_seg_t segs[100];
    unsigned char data[8192];
    int numpassesinlayers;
    tcd_layer_t layers[100];
    int totalpasses;
    tcd_pass_t passes[100];
} tcd_cblk_t;

typedef struct {
    int x0, y0, x1, y1;
    int cw, ch;
    tcd_cblk_t *cblks;
    tgt_tree_t *incltree;
    tgt_tree_t *imsbtree;
} tcd_precinct_t;

typedef struct {
    int x0, y0, x1, y1;
    int bandno;
    tcd_precinct_t *precincts;
    int numbps;
    int stepsize;
} tcd_band_t;

typedef struct {
    int x0, y0, x1, y1;  
  int previous_x0, previous_y0, previous_x1, previous_y1; // usefull for the DWT
  int cas_col, cas_row; // usefull for the DWT
    int pw, ph;
    int numbands;
    tcd_band_t bands[3];
} tcd_resolution_t;

typedef struct {
    int x0, y0, x1, y1;
  int previous_row, previous_col; // usefull for the DWT
    int numresolutions;
    tcd_resolution_t *resolutions;
    int *data;
} tcd_tilecomp_t;

typedef struct {
    int x0, y0, x1, y1;
    int numcomps;
  //int PPT;
  //int len_ppt;
    tcd_tilecomp_t *comps;
} tcd_tile_t;

typedef struct {
    int tw, th;
    tcd_tile_t *tiles;
} tcd_image_t;

/*
 * Initialize the tile coder/decoder
 * img: raw image
 * cp: coding parameters
 * imgg: creation of index file
 */

void tcd_init(j2k_image_t *img, j2k_cp_t *cp, info_image_t *imgg);

void tcd_free(j2k_image_t *img, j2k_cp_t *cp);

/*
 * Decode a tile from a buffer into a raw image
 * src: source buffer
 * len: length of the source buffer
 * tileno: number that identifies the tile that will be decoded
 * imgg : Structure for index file
 */
int tcd_decode_tile(unsigned char *src, int len, int tileno, info_image_t *imgg);

#endif
