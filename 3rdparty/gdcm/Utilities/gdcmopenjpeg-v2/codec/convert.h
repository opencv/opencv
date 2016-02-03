/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
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
#ifndef __J2K_CONVERT_H
#define __J2K_CONVERT_H

/**@name RAW image encoding parameters */
/*@{*/
typedef struct raw_cparameters {
  /** width of the raw image */
  int rawWidth;
  /** height of the raw image */
  int rawHeight;
  /** components of the raw image */
  int rawComp;
  /** bit depth of the raw image */
  int rawBitDepth;
  /** signed/unsigned raw image */
  bool rawSigned;
  /*@}*/
} raw_cparameters_t;

/* TGA conversion */
opj_image_t* tgatoimage(const char *filename, opj_cparameters_t *parameters);
int imagetotga(opj_image_t * image, const char *outfile);

/* BMP conversion */
opj_image_t* bmptoimage(const char *filename, opj_cparameters_t *parameters);
int imagetobmp(opj_image_t *image, const char *outfile);

/* TIFF to image conversion*/
opj_image_t* tiftoimage(const char *filename, opj_cparameters_t *parameters);
int imagetotif(opj_image_t *image, const char *outfile);
/**
Load a single image component encoded in PGX file format
@param filename Name of the PGX file to load
@param parameters *List ?*
@return Returns a greyscale image if successful, returns NULL otherwise
*/
opj_image_t* pgxtoimage(const char *filename, opj_cparameters_t *parameters);
int imagetopgx(opj_image_t *image, const char *outfile);

opj_image_t* pnmtoimage(const char *filename, opj_cparameters_t *parameters);
int imagetopnm(opj_image_t *image, const char *outfile);

/* RAW conversion */
int imagetoraw(opj_image_t * image, const char *outfile);
opj_image_t* rawtoimage(const char *filename, opj_cparameters_t *parameters, raw_cparameters_t *raw_cp);

#endif /* __J2K_CONVERT_H */

