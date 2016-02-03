/*
* Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
* Copyright (c) 2002-2007, Professor Benoit Macq
* Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
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

#include "../libopenjpeg/opj_includes.h"
#include "mj2.h"

/*  -----------------------	      */
/*				      */
/*				      */
/*  Count the number of frames	      */
/*  in a YUV file		      */
/*				      */
/*  -----------------------	      */

int yuv_num_frames(mj2_tk_t * tk, char *infile)
{
  int numimages, frame_size;
  long end_of_f;
	FILE *f;

  f = fopen(infile,"rb");
  if (!f) {  
    fprintf(stderr, "failed to open %s for reading\n",infile);
    return -1;
  }
	
  frame_size = (int) (tk->w * tk->h * (1.0 + (double) 2 / (double) (tk->CbCr_subsampling_dx * tk->CbCr_subsampling_dy)));	/* Calculate frame size */
	
  fseek(f, 0, SEEK_END);
  end_of_f = ftell(f);		/* Calculate file size */
	
  if (end_of_f < frame_size) {
    fprintf(stderr,
			"YUV does not contains any frame of %d x %d size\n", tk->w,
			tk->h);
    return -1;
  }
	
  numimages = end_of_f / frame_size;	/* Calculate number of images */
	fclose(f);

  return numimages;
}

//  -----------------------
//
//
//  YUV to IMAGE
//
//  -----------------------

opj_image_t *mj2_image_create(mj2_tk_t * tk, opj_cparameters_t *parameters)
{
	opj_image_cmptparm_t cmptparm[3];
	opj_image_t * img;
	int i;
	int numcomps = 3;
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;

	/* initialize image components */
	memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
	for(i = 0; i < numcomps; i++) {
		cmptparm[i].prec = 8;
		cmptparm[i].bpp = 8;
		cmptparm[i].sgnd = 0;		
		cmptparm[i].dx = i ? subsampling_dx * tk->CbCr_subsampling_dx : subsampling_dx;
		cmptparm[i].dy = i ? subsampling_dy * tk->CbCr_subsampling_dy : subsampling_dy;
		cmptparm[i].w = tk->w;
		cmptparm[i].h = tk->h;
	}
	/* create the image */
	img = opj_image_create(numcomps, cmptparm, CLRSPC_SRGB);
	return img;
}

char yuvtoimage(mj2_tk_t * tk, opj_image_t * img, int frame_num, opj_cparameters_t *parameters, char* infile)
{
  int i, compno;
  int offset;
  long end_of_f, position;
	int numcomps = 3;
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;
	FILE *yuvfile;
	
  yuvfile = fopen(infile,"rb");
  if (!yuvfile) {  
    fprintf(stderr, "failed to open %s for readings\n",parameters->infile);
    return 1;
  }

  offset = (int) ((double) (frame_num * tk->w * tk->h) * (1.0 +
		1.0 * (double) 2 / (double) (tk->CbCr_subsampling_dx * tk->CbCr_subsampling_dy)));
  fseek(yuvfile, 0, SEEK_END);
  end_of_f = ftell(yuvfile);
  fseek(yuvfile, sizeof(unsigned char) * offset, SEEK_SET);
  position = ftell(yuvfile);
  if (position >= end_of_f) {
    fprintf(stderr, "Cannot reach frame number %d in yuv file !!\n",
			frame_num);
		fclose(yuvfile);
    return 1;
  }
	
  img->x0 = tk->Dim[0];
  img->y0 = tk->Dim[1];
  img->x1 = !tk->Dim[0] ? (tk->w - 1) * subsampling_dx + 1 : tk->Dim[0] +
    (tk->w - 1) * subsampling_dx + 1;
  img->y1 = !tk->Dim[1] ? (tk->h - 1) * subsampling_dy + 1 : tk->Dim[1] +
    (tk->h - 1) * subsampling_dy + 1;
	
	for(compno = 0; compno < numcomps; compno++) {
		for (i = 0; i < (tk->w * tk->h / (img->comps[compno].dx * img->comps[compno].dy))
			&& !feof(yuvfile); i++) {
			if (!fread(&img->comps[compno].data[i], 1, 1, yuvfile)) {
				fprintf(stderr, "Error reading %s file !!\n", infile);				
				return 1;
			}
		}
	}
	fclose(yuvfile);
	
  return 0;
}



//  -----------------------
//
//
//  IMAGE to YUV
//
//  -----------------------


bool imagetoyuv(opj_image_t * img, char *outfile)
{
  FILE *f;
  int i;
  
  if (img->numcomps == 3) {
    if (img->comps[0].dx != img->comps[1].dx / 2
      || img->comps[1].dx != img->comps[2].dx) {
      fprintf(stderr,
				"Error with the input image components size: cannot create yuv file)\n");
      return false;
    }
  } else if (!(img->numcomps == 1)) {
    fprintf(stderr,
      "Error with the number of image components(must be one or three)\n");
    return false;
  }
  
  f = fopen(outfile, "a+b");
  if (!f) {
    fprintf(stderr, "failed to open %s for writing\n", outfile);
    return false;
  }
  
  
  for (i = 0; i < (img->comps[0].w * img->comps[0].h); i++) {
    unsigned char y;
    y = img->comps[0].data[i];
    fwrite(&y, 1, 1, f);
  }
  
  
  if (img->numcomps == 3) {
    for (i = 0; i < (img->comps[1].w * img->comps[1].h); i++) {
      unsigned char cb;
      cb = img->comps[1].data[i];
      fwrite(&cb, 1, 1, f);
    }
    
    
    for (i = 0; i < (img->comps[2].w * img->comps[2].h); i++) {
      unsigned char cr;
      cr = img->comps[2].data[i];
      fwrite(&cr, 1, 1, f);
    }
  } else if (img->numcomps == 1) {
    for (i = 0; i < (img->comps[0].w * img->comps[0].h * 0.25); i++) {
      unsigned char cb = 125;
      fwrite(&cb, 1, 1, f);
    }
    
    
    for (i = 0; i < (img->comps[0].w * img->comps[0].h * 0.25); i++) {
      unsigned char cr = 125;
      fwrite(&cr, 1, 1, f);
    }
  }  
  fclose(f);
  return true;
}

//  -----------------------
//
//
//  IMAGE to BMP
//
//  -----------------------

int imagetobmp(opj_image_t * img, char *outfile) {
  int w,wr,h,hr,i,pad;
  FILE *f;
  
  if (img->numcomps == 3 && img->comps[0].dx == img->comps[1].dx
    && img->comps[1].dx == img->comps[2].dx
    && img->comps[0].dy == img->comps[1].dy
    && img->comps[1].dy == img->comps[2].dy
    && img->comps[0].prec == img->comps[1].prec
    && img->comps[1].prec == img->comps[2].prec) {
    /* -->> -->> -->> -->>
    
      24 bits color
      
    <<-- <<-- <<-- <<-- */
    
    f = fopen(outfile, "wb");
    if (!f) {
      fprintf(stderr, "failed to open %s for writing\n", outfile);
      return 1;
    }   
    
    w = img->comps[0].w;
    wr = int_ceildivpow2(img->comps[0].w, img->comps[0].factor);
    
    h = img->comps[0].h;
    hr = int_ceildivpow2(img->comps[0].h, img->comps[0].factor);
    
    fprintf(f, "BM");
    
    /* FILE HEADER */
    /* ------------- */
    fprintf(f, "%c%c%c%c",
      (unsigned char) (hr * wr * 3 + 3 * hr * (wr % 2) +
      54) & 0xff,
      (unsigned char) ((hr * wr * 3 + 3 * hr * (wr % 2) + 54)
      >> 8) & 0xff,
      (unsigned char) ((hr * wr * 3 + 3 * hr * (wr % 2) + 54)
      >> 16) & 0xff,
      (unsigned char) ((hr * wr * 3 + 3 * hr * (wr % 2) + 54)
      >> 24) & 0xff);
    fprintf(f, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff,
      ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
    fprintf(f, "%c%c%c%c", (54) & 0xff, ((54) >> 8) & 0xff,
      ((54) >> 16) & 0xff, ((54) >> 24) & 0xff);
    
    /* INFO HEADER   */
    /* ------------- */
    fprintf(f, "%c%c%c%c", (40) & 0xff, ((40) >> 8) & 0xff,
      ((40) >> 16) & 0xff, ((40) >> 24) & 0xff);
    fprintf(f, "%c%c%c%c", (unsigned char) ((wr) & 0xff),
      (unsigned char) ((wr) >> 8) & 0xff,
      (unsigned char) ((wr) >> 16) & 0xff,
      (unsigned char) ((wr) >> 24) & 0xff);
    fprintf(f, "%c%c%c%c", (unsigned char) ((hr) & 0xff),
      (unsigned char) ((hr) >> 8) & 0xff,
      (unsigned char) ((hr) >> 16) & 0xff,
      (unsigned char) ((hr) >> 24) & 0xff);
    fprintf(f, "%c%c", (1) & 0xff, ((1) >> 8) & 0xff);
    fprintf(f, "%c%c", (24) & 0xff, ((24) >> 8) & 0xff);
    fprintf(f, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff,
      ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
    fprintf(f, "%c%c%c%c",
      (unsigned char) (3 * hr * wr +
      3 * hr * (wr % 2)) & 0xff,
      (unsigned char) ((hr * wr * 3 + 3 * hr * (wr % 2)) >>
      8) & 0xff,
      (unsigned char) ((hr * wr * 3 + 3 * hr * (wr % 2)) >>
      16) & 0xff,
      (unsigned char) ((hr * wr * 3 + 3 * hr * (wr % 2)) >>
      24) & 0xff);
    fprintf(f, "%c%c%c%c", (7834) & 0xff, ((7834) >> 8) & 0xff,
      ((7834) >> 16) & 0xff, ((7834) >> 24) & 0xff);
    fprintf(f, "%c%c%c%c", (7834) & 0xff, ((7834) >> 8) & 0xff,
      ((7834) >> 16) & 0xff, ((7834) >> 24) & 0xff);
    fprintf(f, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff,
      ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
    fprintf(f, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff,
      ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
    
    for (i = 0; i < wr * hr; i++) {
      unsigned char R, G, B;
      /* a modifier */
      // R = img->comps[0].data[w * h - ((i) / (w) + 1) * w + (i) % (w)];
      R = img->comps[0].data[w * hr - ((i) / (wr) + 1) * w + (i) % (wr)];
      // G = img->comps[1].data[w * h - ((i) / (w) + 1) * w + (i) % (w)];
      G = img->comps[1].data[w * hr - ((i) / (wr) + 1) * w + (i) % (wr)];
      // B = img->comps[2].data[w * h - ((i) / (w) + 1) * w + (i) % (w)];
      B = img->comps[2].data[w * hr - ((i) / (wr) + 1) * w + (i) % (wr)];
      fprintf(f, "%c%c%c", B, G, R);
      
      if ((i + 1) % wr == 0) {
				for (pad = (3 * wr) % 4 ? 4 - (3 * wr) % 4 : 0; pad > 0; pad--)	/* ADD */
					fprintf(f, "%c", 0);
      }
    }
    fclose(f);
  }
  return 0;
}
