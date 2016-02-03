/*
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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
#include "image.h"
#include "openjpeg.h"
#include "opj_malloc.h"
#include "j2k.h"
#include "int.h"

opj_image_t* opj_image_create0(void) {
  opj_image_t *image = (opj_image_t*)opj_malloc(sizeof(opj_image_t));
  memset(image,0,sizeof(opj_image_t));
  return image;
}

opj_image_t* OPJ_CALLCONV opj_image_create(OPJ_UINT32 numcmpts, opj_image_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc) {
  OPJ_UINT32 compno;
  opj_image_t *image = 00;

  image = (opj_image_t*) opj_malloc(sizeof(opj_image_t));
  if
    (image)
  {
    memset(image,0,sizeof(opj_image_t));
    image->color_space = clrspc;
    image->numcomps = numcmpts;
    /* allocate memory for the per-component information */
    image->comps = (opj_image_comp_t*)opj_malloc(image->numcomps * sizeof(opj_image_comp_t));
    if
      (!image->comps)
    {
      opj_image_destroy(image);
      return 00;
    }
    memset(image->comps,0,image->numcomps * sizeof(opj_image_comp_t));
    /* create the individual image components */
    for(compno = 0; compno < numcmpts; compno++) {
      opj_image_comp_t *comp = &image->comps[compno];
      comp->dx = cmptparms[compno].dx;
      comp->dy = cmptparms[compno].dy;
      comp->w = cmptparms[compno].w;
      comp->h = cmptparms[compno].h;
      comp->x0 = cmptparms[compno].x0;
      comp->y0 = cmptparms[compno].y0;
      comp->prec = cmptparms[compno].prec;
      comp->sgnd = cmptparms[compno].sgnd;
      comp->data = (OPJ_INT32*) opj_calloc(comp->w * comp->h, sizeof(OPJ_INT32));
      if
        (!comp->data)
      {
        opj_image_destroy(image);
        return 00;
      }
    }
  }
  return image;
}

opj_image_t* OPJ_CALLCONV opj_image_tile_create(OPJ_UINT32 numcmpts, opj_image_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc) {
  OPJ_UINT32 compno;
  opj_image_t *image = 00;

  image = (opj_image_t*) opj_malloc(sizeof(opj_image_t));
  if
    (image)
  {
    memset(image,0,sizeof(opj_image_t));
    image->color_space = clrspc;
    image->numcomps = numcmpts;
    /* allocate memory for the per-component information */
    image->comps = (opj_image_comp_t*)opj_malloc(image->numcomps * sizeof(opj_image_comp_t));
    if
      (!image->comps)
    {
      opj_image_destroy(image);
      return 00;
    }
    memset(image->comps,0,image->numcomps * sizeof(opj_image_comp_t));
    /* create the individual image components */
    for(compno = 0; compno < numcmpts; compno++) {
      opj_image_comp_t *comp = &image->comps[compno];
      comp->dx = cmptparms[compno].dx;
      comp->dy = cmptparms[compno].dy;
      comp->w = cmptparms[compno].w;
      comp->h = cmptparms[compno].h;
      comp->x0 = cmptparms[compno].x0;
      comp->y0 = cmptparms[compno].y0;
      comp->prec = cmptparms[compno].prec;
      comp->sgnd = cmptparms[compno].sgnd;
      comp->data = 0;
    }
  }
  return image;
}

void OPJ_CALLCONV opj_image_destroy(opj_image_t *image) {
  OPJ_UINT32 i;
  if
    (image)
  {
    if
      (image->comps)
    {
      /* image components */
      for(i = 0; i < image->numcomps; i++) {
        opj_image_comp_t *image_comp = &image->comps[i];
        if(image_comp->data) {
          opj_free(image_comp->data);
        }
      }
      opj_free(image->comps);
    }
    opj_free(image);
  }
}

/**
 * Updates the components of the image from the coding parameters.
 *
 * @param p_image    the image to update.
 * @param p_cp      the coding parameters from which to update the image.
 */
void opj_image_comp_update(opj_image_t * p_image,const opj_cp_t * p_cp)
{
  OPJ_UINT32 i, l_width, l_height;
  OPJ_INT32 l_x0,l_y0,l_x1,l_y1;
  OPJ_INT32 l_comp_x0,l_comp_y0,l_comp_x1,l_comp_y1;
  opj_image_comp_t * l_img_comp = 00;

  l_x0 = int_max(p_cp->tx0 , p_image->x0);
  l_y0 = int_max(p_cp->ty0 , p_image->y0);
  l_x1 = int_min(p_cp->tx0 + p_cp->tw * p_cp->tdx, p_image->x1);
  l_y1 = int_min(p_cp->ty0 + p_cp->th * p_cp->tdy, p_image->y1);

  l_img_comp = p_image->comps;
  for
    (i = 0; i < p_image->numcomps; ++i)
  {
    l_comp_x0 = int_ceildiv(l_x0, l_img_comp->dx);
    l_comp_y0 = int_ceildiv(l_y0, l_img_comp->dy);
    l_comp_x1 = int_ceildiv(l_x1, l_img_comp->dx);
    l_comp_y1 = int_ceildiv(l_y1, l_img_comp->dy);
    l_width = int_ceildivpow2(l_comp_x1 - l_comp_x0, l_img_comp->factor);
    l_height = int_ceildivpow2(l_comp_y1 - l_comp_y0, l_img_comp->factor);
    l_img_comp->w = l_width;
    l_img_comp->h = l_height;
    l_img_comp->x0 = l_x0;
    l_img_comp->y0 = l_y0;
    ++l_img_comp;
  }
}
