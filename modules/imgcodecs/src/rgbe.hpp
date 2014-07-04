/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _RGBE_HDR_H_
#define _RGBE_HDR_H_

// posted to http://www.graphics.cornell.edu/~bjw/
// written by Bruce Walter  (bjw@graphics.cornell.edu)  5/26/95
// based on code written by Greg Ward

#include <stdio.h>

typedef struct {
  int valid;            /* indicate which fields are valid */
  char programtype[16]; /* listed at beginning of file to identify it
                         * after "#?".  defaults to "RGBE" */
  float gamma;          /* image has already been gamma corrected with
                         * given gamma.  defaults to 1.0 (no correction) */
  float exposure;       /* a value of 1.0 in an image corresponds to
       * <exposure> watts/steradian/m^2.
       * defaults to 1.0 */
} rgbe_header_info;

/* flags indicating which fields in an rgbe_header_info are valid */
#define RGBE_VALID_PROGRAMTYPE 0x01
#define RGBE_VALID_GAMMA       0x02
#define RGBE_VALID_EXPOSURE    0x04

/* return codes for rgbe routines */
#define RGBE_RETURN_SUCCESS 0
#define RGBE_RETURN_FAILURE -1

/* read or write headers */
/* you may set rgbe_header_info to null if you want to */
int RGBE_WriteHeader(FILE *fp, int width, int height, rgbe_header_info *info);
int RGBE_ReadHeader(FILE *fp, int *width, int *height, rgbe_header_info *info);

/* read or write pixels */
/* can read or write pixels in chunks of any size including single pixels*/
int RGBE_WritePixels(FILE *fp, float *data, int numpixels);
int RGBE_ReadPixels(FILE *fp, float *data, int numpixels);

/* read or write run length encoded files */
/* must be called to read or write whole scanlines */
int RGBE_WritePixels_RLE(FILE *fp, float *data, int scanline_width,
       int num_scanlines);
int RGBE_ReadPixels_RLE(FILE *fp, float *data, int scanline_width,
      int num_scanlines);

#endif/*_RGBE_HDR_H_*/
