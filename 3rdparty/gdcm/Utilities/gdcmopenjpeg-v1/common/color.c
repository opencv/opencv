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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "opj_config.h"
#include "openjpeg.h"
#include "color.h"

#ifdef HAVE_LIBLCMS2
#include <lcms2.h>
#endif
#ifdef HAVE_LIBLCMS1
#include <lcms.h>
#endif

/*--------------------------------------------------------
Matrix für sYCC, Amendment 1 to IEC 61966-2-1

Y :   0.299   0.587    0.114   :R
Cb:  -0.1687 -0.3312   0.5     :G
Cr:   0.5    -0.4187  -0.0812  :B

Inverse:

R: 1        -3.68213e-05    1.40199      :Y
G: 1.00003  -0.344125      -0.714128     :Cb - 2^(prec - 1)
B: 0.999823  1.77204       -8.04142e-06  :Cr - 2^(prec - 1)

-----------------------------------------------------------*/
static void sycc_to_rgb(int offset, int upb, int y, int cb, int cr,
	int *out_r, int *out_g, int *out_b)
{
	int r, g, b;

	cb -= offset; cr -= offset;
	r = y + (int)(1.402 * (float)cr);
	if(r < 0) r = 0; else if(r > upb) r = upb; *out_r = r;

	g = y - (int)(0.344 * (float)cb + 0.714 * (float)cr);
	if(g < 0) g = 0; else if(g > upb) g = upb; *out_g = g;

	b = y + (int)(1.772 * (float)cb);
	if(b < 0) b = 0; else if(b > upb) b = upb; *out_b = b;
}

static void sycc444_to_rgb(opj_image_t *img)
{
	int *d0, *d1, *d2, *r, *g, *b;
	const int *y, *cb, *cr;
	int maxw, maxh, max, i, offset, upb;

	i = img->comps[0].prec;
	offset = 1<<(i - 1); upb = (1<<i)-1;

	maxw = img->comps[0].w; maxh = img->comps[0].h;
	max = maxw * maxh;

	y = img->comps[0].data;
	cb = img->comps[1].data;
	cr = img->comps[2].data;

	d0 = r = (int*)malloc(sizeof(int) * max);
	d1 = g = (int*)malloc(sizeof(int) * max);
	d2 = b = (int*)malloc(sizeof(int) * max);

	for(i = 0; i < max; ++i)
   {
	sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);	

	++y; ++cb; ++cr; ++r; ++g; ++b;
   }	
	free(img->comps[0].data); img->comps[0].data = d0;
	free(img->comps[1].data); img->comps[1].data = d1;
	free(img->comps[2].data); img->comps[2].data = d2;

}/* sycc444_to_rgb() */

static void sycc422_to_rgb(opj_image_t *img)
{	
	int *d0, *d1, *d2, *r, *g, *b;
	const int *y, *cb, *cr;
	int maxw, maxh, max, offset, upb;
	int i, j;

	i = img->comps[0].prec;
	offset = 1<<(i - 1); upb = (1<<i)-1;

	maxw = img->comps[0].w; maxh = img->comps[0].h;
	max = maxw * maxh;

	y = img->comps[0].data;
	cb = img->comps[1].data;
	cr = img->comps[2].data;

	d0 = r = (int*)malloc(sizeof(int) * max);
	d1 = g = (int*)malloc(sizeof(int) * max);
	d2 = b = (int*)malloc(sizeof(int) * max);

	for(i=0; i < maxh; ++i)
   {
	for(j=0; j < maxw; j += 2)
  {
	sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);

	++y; ++r; ++g; ++b;

	sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);

	++y; ++r; ++g; ++b; ++cb; ++cr;
  }
   }
	free(img->comps[0].data); img->comps[0].data = d0;
	free(img->comps[1].data); img->comps[1].data = d1;
	free(img->comps[2].data); img->comps[2].data = d2;

	img->comps[1].w = maxw; img->comps[1].h = maxh;
	img->comps[2].w = maxw; img->comps[2].h = maxh;
	img->comps[1].dx = img->comps[0].dx;
	img->comps[2].dx = img->comps[0].dx;
	img->comps[1].dy = img->comps[0].dy;
	img->comps[2].dy = img->comps[0].dy;

}/* sycc422_to_rgb() */

static void sycc420_to_rgb(opj_image_t *img)
{
	int *d0, *d1, *d2, *r, *g, *b, *nr, *ng, *nb;
	const int *y, *cb, *cr, *ny;
	int maxw, maxh, max, offset, upb;
	int i, j;

	i = img->comps[0].prec;
	offset = 1<<(i - 1); upb = (1<<i)-1;

	maxw = img->comps[0].w; maxh = img->comps[0].h;
	max = maxw * maxh;

	y = img->comps[0].data;
	cb = img->comps[1].data;
	cr = img->comps[2].data;

	d0 = r = (int*)malloc(sizeof(int) * max);
	d1 = g = (int*)malloc(sizeof(int) * max);
	d2 = b = (int*)malloc(sizeof(int) * max);

	for(i=0; i < maxh; i += 2)
   {
	ny = y + maxw;
	nr = r + maxw; ng = g + maxw; nb = b + maxw;

	for(j=0; j < maxw;  j += 2)
  {
	sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);

	++y; ++r; ++g; ++b;

	sycc_to_rgb(offset, upb, *y, *cb, *cr, r, g, b);

	++y; ++r; ++g; ++b;

	sycc_to_rgb(offset, upb, *ny, *cb, *cr, nr, ng, nb);

	++ny; ++nr; ++ng; ++nb;

	sycc_to_rgb(offset, upb, *ny, *cb, *cr, nr, ng, nb);

	++ny; ++nr; ++ng; ++nb; ++cb; ++cr;
  }
	y += maxw; r += maxw; g += maxw; b += maxw;
   }
	free(img->comps[0].data); img->comps[0].data = d0;
	free(img->comps[1].data); img->comps[1].data = d1;
	free(img->comps[2].data); img->comps[2].data = d2;

	img->comps[1].w = maxw; img->comps[1].h = maxh;
	img->comps[2].w = maxw; img->comps[2].h = maxh;
	img->comps[1].dx = img->comps[0].dx;
	img->comps[2].dx = img->comps[0].dx;
	img->comps[1].dy = img->comps[0].dy;
	img->comps[2].dy = img->comps[0].dy;

}/* sycc420_to_rgb() */

void color_sycc_to_rgb(opj_image_t *img)
{
	if(img->numcomps < 3) 
   {
	img->color_space = CLRSPC_GRAY;
	return;
   }

	if((img->comps[0].dx == 1)
	&& (img->comps[1].dx == 2)
	&& (img->comps[2].dx == 2)
	&& (img->comps[0].dy == 1)
	&& (img->comps[1].dy == 2)
	&& (img->comps[2].dy == 2))/* horizontal and vertical sub-sample */
  {
	sycc420_to_rgb(img);
  }
	else
	if((img->comps[0].dx == 1)
	&& (img->comps[1].dx == 2)
	&& (img->comps[2].dx == 2)
	&& (img->comps[0].dy == 1)
	&& (img->comps[1].dy == 1)
	&& (img->comps[2].dy == 1))/* horizontal sub-sample only */
  {
	sycc422_to_rgb(img);
  }
	else
	if((img->comps[0].dx == 1)
	&& (img->comps[1].dx == 1)
	&& (img->comps[2].dx == 1)
	&& (img->comps[0].dy == 1)
	&& (img->comps[1].dy == 1)
	&& (img->comps[2].dy == 1))/* no sub-sample */
  {
	sycc444_to_rgb(img);
  }
	else
  {
	fprintf(stderr,"%s:%d:color_sycc_to_rgb\n\tCAN NOT CONVERT\n",
	 __FILE__,__LINE__);
	return;
  }
	img->color_space = CLRSPC_SRGB;

}/* color_sycc_to_rgb() */

#if defined(HAVE_LIBLCMS2) || defined(HAVE_LIBLCMS1)
#ifdef HAVE_LIBLCMS1
/* Bob Friesenhahn proposed:*/
#define cmsSigXYZData   icSigXYZData
#define cmsSigLabData   icSigLabData
#define cmsSigCmykData  icSigCmykData
#define cmsSigYCbCrData icSigYCbCrData
#define cmsSigLuvData   icSigLuvData
#define cmsSigGrayData  icSigGrayData
#define cmsSigRgbData   icSigRgbData
#define cmsUInt32Number DWORD

#define cmsColorSpaceSignature icColorSpaceSignature
#define cmsGetHeaderRenderingIntent cmsTakeRenderingIntent

#endif /* HAVE_LIBLCMS1 */

void color_apply_icc_profile(opj_image_t *image)
{
	cmsHPROFILE in_prof, out_prof;
	cmsHTRANSFORM transform;
	cmsColorSpaceSignature in_space, out_space;
	cmsUInt32Number intent, in_type, out_type, nr_samples;
	int *r, *g, *b;
	int prec, i, max, max_w, max_h;
	OPJ_COLOR_SPACE oldspace;

	in_prof = 
	 cmsOpenProfileFromMem(image->icc_profile_buf, image->icc_profile_len);
	in_space = cmsGetPCS(in_prof);
	out_space = cmsGetColorSpace(in_prof);
	intent = cmsGetHeaderRenderingIntent(in_prof);

	
	max_w = image->comps[0].w; max_h = image->comps[0].h;
	prec = image->comps[0].prec;
	oldspace = image->color_space;

	if(out_space == cmsSigRgbData) /* enumCS 16 */
   {
	in_type = TYPE_RGB_16;
	out_type = TYPE_RGB_16;
	out_prof = cmsCreate_sRGBProfile();
	image->color_space = CLRSPC_SRGB;
   }
	else
	if(out_space == cmsSigGrayData) /* enumCS 17 */
   {
	in_type = TYPE_GRAY_8;
	out_type = TYPE_RGB_8;
	out_prof = cmsCreate_sRGBProfile();
	image->color_space = CLRSPC_SRGB;
   }
	else
	if(out_space == cmsSigYCbCrData) /* enumCS 18 */
   {
	in_type = TYPE_YCbCr_16;
	out_type = TYPE_RGB_16;
	out_prof = cmsCreate_sRGBProfile();
	image->color_space = CLRSPC_SRGB;
   }
	else
   {
#ifdef DEBUG_PROFILE
fprintf(stderr,"%s:%d: color_apply_icc_profile\n\tICC Profile has unknown "
"output colorspace(%#x)(%c%c%c%c)\n\tICC Profile ignored.\n",
__FILE__,__LINE__,out_space,
(out_space>>24) & 0xff,(out_space>>16) & 0xff,
(out_space>>8) & 0xff, out_space & 0xff);
#endif
	return;
   }

#ifdef DEBUG_PROFILE
fprintf(stderr,"%s:%d:color_apply_icc_profile\n\tchannels(%d) prec(%d) w(%d) h(%d)"
"\n\tprofile: in(%p) out(%p)\n",__FILE__,__LINE__,image->numcomps,prec,
max_w,max_h, (void*)in_prof,(void*)out_prof);

fprintf(stderr,"\trender_intent (%u)\n\t"
"color_space: in(%#x)(%c%c%c%c)   out:(%#x)(%c%c%c%c)\n\t"
"       type: in(%u)              out:(%u)\n",
intent,
in_space,
(in_space>>24) & 0xff,(in_space>>16) & 0xff,
(in_space>>8) & 0xff, in_space & 0xff,

out_space,
(out_space>>24) & 0xff,(out_space>>16) & 0xff,
(out_space>>8) & 0xff, out_space & 0xff,

in_type,out_type
 );
#endif /* DEBUG_PROFILE */

	transform = cmsCreateTransform(in_prof, in_type,
	 out_prof, out_type, intent, 0);

#ifdef HAVE_LIBLCMS2
/* Possible for: LCMS_VERSION >= 2000 :*/
	cmsCloseProfile(in_prof);
	cmsCloseProfile(out_prof);
#endif

	if(transform == NULL)
   {
#ifdef DEBUG_PROFILE
fprintf(stderr,"%s:%d:color_apply_icc_profile\n\tcmsCreateTransform failed. "
"ICC Profile ignored.\n",__FILE__,__LINE__);
#endif
	image->color_space = oldspace;
#ifdef HAVE_LIBLCMS1
	cmsCloseProfile(in_prof);
	cmsCloseProfile(out_prof);
#endif
	return;
   }

	if(image->numcomps > 2)/* RGB, RGBA */
   {
	unsigned short *inbuf, *outbuf, *in, *out;
	max = max_w * max_h; nr_samples = max * 3 * sizeof(unsigned short);
	in = inbuf = (unsigned short*)malloc(nr_samples);
	out = outbuf = (unsigned short*)malloc(nr_samples);

	r = image->comps[0].data;
	g = image->comps[1].data;
	b = image->comps[2].data;

	for(i = 0; i < max; ++i)
  {
	*in++ = (unsigned short)*r++;
	*in++ = (unsigned short)*g++;
	*in++ = (unsigned short)*b++;
  }

	cmsDoTransform(transform, inbuf, outbuf, max);

	r = image->comps[0].data;
	g = image->comps[1].data;
	b = image->comps[2].data;

	for(i = 0; i < max; ++i)
  {
	*r++ = (int)*out++;
	*g++ = (int)*out++;
	*b++ = (int)*out++;
  }
	free(inbuf); free(outbuf);
   }
	else /* GRAY, GRAYA */
   {
	unsigned char *in, *inbuf, *out, *outbuf;

	max = max_w * max_h; nr_samples = max * 3 * sizeof(unsigned char);
	in = inbuf = (unsigned char*)malloc(nr_samples);
	out = outbuf = (unsigned char*)malloc(nr_samples);

	image->comps = (opj_image_comp_t*)
	 realloc(image->comps, (image->numcomps+2)*sizeof(opj_image_comp_t));

	if(image->numcomps == 2)
	 image->comps[3] = image->comps[1];

	image->comps[1] = image->comps[0];
	image->comps[2] = image->comps[0];

	image->comps[1].data = (int*)calloc(max, sizeof(int));
	image->comps[2].data = (int*)calloc(max, sizeof(int));

	image->numcomps += 2;

	r = image->comps[0].data;

	for(i = 0; i < max; ++i)
  {
	*in++ = (unsigned char)*r++;
  }
	cmsDoTransform(transform, inbuf, outbuf, max);

	r = image->comps[0].data;
	g = image->comps[1].data;
	b = image->comps[2].data;

	for(i = 0; i < max; ++i)
  {
	*r++ = (int)*out++; *g++ = (int)*out++; *b++ = (int)*out++;
  }
	free(inbuf); free(outbuf);

   }/* if(image->numcomps */

	cmsDeleteTransform(transform);

#ifdef HAVE_LIBLCMS1
	cmsCloseProfile(in_prof);
	cmsCloseProfile(out_prof);
#endif
}/* color_apply_icc_profile() */

#endif /* HAVE_LIBLCMS2 || HAVE_LIBLCMS1 */

