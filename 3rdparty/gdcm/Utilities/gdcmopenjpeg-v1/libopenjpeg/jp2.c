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
#include "opj_includes.h"

/** @defgroup JP2 JP2 - JPEG-2000 file format reader/writer */
/*@{*/

/** @name Local static functions */
/*@{*/

/**
Read box headers
@param cinfo Codec context info
@param cio Input stream
@param box
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_boxhdr(opj_common_ptr cinfo, opj_cio_t *cio, opj_jp2_box_t *box);
/*static void jp2_write_url(opj_cio_t *cio, char *Idx_file);*/
/**
Read the IHDR box - Image Header box
@param jp2 JP2 handle
@param cio Input buffer stream
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_ihdr(opj_jp2_t *jp2, opj_cio_t *cio);
static void jp2_write_ihdr(opj_jp2_t *jp2, opj_cio_t *cio);
static void jp2_write_bpcc(opj_jp2_t *jp2, opj_cio_t *cio);
static bool jp2_read_bpcc(opj_jp2_t *jp2, opj_cio_t *cio);
static void jp2_write_colr(opj_jp2_t *jp2, opj_cio_t *cio);
/**
Write the FTYP box - File type box
@param jp2 JP2 handle
@param cio Output buffer stream
*/
static void jp2_write_ftyp(opj_jp2_t *jp2, opj_cio_t *cio);
/**
Read the FTYP box - File type box
@param jp2 JP2 handle
@param cio Input buffer stream
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_ftyp(opj_jp2_t *jp2, opj_cio_t *cio);
static int jp2_write_jp2c(opj_jp2_t *jp2, opj_cio_t *cio, opj_image_t *image, opj_codestream_info_t *cstr_info);
static bool jp2_read_jp2c(opj_jp2_t *jp2, opj_cio_t *cio, unsigned int *j2k_codestream_length, unsigned int *j2k_codestream_offset);
static void jp2_write_jp(opj_cio_t *cio);
/**
Read the JP box - JPEG 2000 signature
@param jp2 JP2 handle
@param cio Input buffer stream
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_jp(opj_jp2_t *jp2, opj_cio_t *cio);
/**
Decode the structure of a JP2 file
@param jp2 JP2 handle
@param cio Input buffer stream
@param color Collector for profile, cdef and pclr data
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_struct(opj_jp2_t *jp2, opj_cio_t *cio,
	opj_jp2_color_t *color);
/**
Apply collected palette data
@param color Collector for profile, cdef and pclr data
@param image 
*/
static void jp2_apply_pclr(opj_jp2_color_t *color, opj_image_t *image);
/**
Collect palette data
@param jp2 JP2 handle
@param cio Input buffer stream
@param box
@param color Collector for profile, cdef and pclr data
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_pclr(opj_jp2_t *jp2, opj_cio_t *cio,
    opj_jp2_box_t *box, opj_jp2_color_t *color);
/**
Collect component mapping data
@param jp2 JP2 handle
@param cio Input buffer stream
@param box
@param color Collector for profile, cdef and pclr data
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_cmap(opj_jp2_t *jp2, opj_cio_t *cio,
    opj_jp2_box_t *box, opj_jp2_color_t *color);
/**
Collect colour specification data
@param jp2 JP2 handle
@param cio Input buffer stream
@param box
@param color Collector for profile, cdef and pclr data
@return Returns true if successful, returns false otherwise
*/
static bool jp2_read_colr(opj_jp2_t *jp2, opj_cio_t *cio,
    opj_jp2_box_t *box, opj_jp2_color_t *color);
/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */

static bool jp2_read_boxhdr(opj_common_ptr cinfo, opj_cio_t *cio, opj_jp2_box_t *box) {
	box->init_pos = cio_tell(cio);
	box->length = cio_read(cio, 4);
	box->type = cio_read(cio, 4);
	if (box->length == 1) {
		if (cio_read(cio, 4) != 0) {
			opj_event_msg(cinfo, EVT_ERROR, "Cannot handle box sizes higher than 2^32\n");
			return false;
		}
		box->length = cio_read(cio, 4);
		if (box->length == 0) 
			box->length = cio_numbytesleft(cio) + 12;
	}
	else if (box->length == 0) {
		box->length = cio_numbytesleft(cio) + 8;
	}
	
	return true;
}

#if 0
static void jp2_write_url(opj_cio_t *cio, char *Idx_file) {
	unsigned int i;
	opj_jp2_box_t box;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_URL, 4);	/* DBTL */
	cio_write(cio, 0, 1);		/* VERS */
	cio_write(cio, 0, 3);		/* FLAG */

	if(Idx_file) {
		for (i = 0; i < strlen(Idx_file); i++) {
			cio_write(cio, Idx_file[i], 1);
		}
	}

	box.length = cio_tell(cio) - box.init_pos;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);
}
#endif

static bool jp2_read_ihdr(opj_jp2_t *jp2, opj_cio_t *cio) {
	opj_jp2_box_t box;

	opj_common_ptr cinfo = jp2->cinfo;

	jp2_read_boxhdr(cinfo, cio, &box);
	if (JP2_IHDR != box.type) {
		opj_event_msg(cinfo, EVT_ERROR, "Expected IHDR Marker\n");
		return false;
	}

	jp2->h = cio_read(cio, 4);			/* HEIGHT */
	jp2->w = cio_read(cio, 4);			/* WIDTH */
	jp2->numcomps = cio_read(cio, 2);	/* NC */
	jp2->comps = (opj_jp2_comps_t*) opj_malloc(jp2->numcomps * sizeof(opj_jp2_comps_t));

	jp2->bpc = cio_read(cio, 1);		/* BPC */

	jp2->C = cio_read(cio, 1);			/* C */
	jp2->UnkC = cio_read(cio, 1);		/* UnkC */
	jp2->IPR = cio_read(cio, 1);		/* IPR */

	if (cio_tell(cio) - box.init_pos != box.length) {
		opj_event_msg(cinfo, EVT_ERROR, "Error with IHDR Box\n");
		return false;
	}

	return true;
}

static void jp2_write_ihdr(opj_jp2_t *jp2, opj_cio_t *cio) {
	opj_jp2_box_t box;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_IHDR, 4);		/* IHDR */

	cio_write(cio, jp2->h, 4);			/* HEIGHT */
	cio_write(cio, jp2->w, 4);			/* WIDTH */
	cio_write(cio, jp2->numcomps, 2);	/* NC */

	cio_write(cio, jp2->bpc, 1);		/* BPC */

	cio_write(cio, jp2->C, 1);			/* C : Always 7 */
	cio_write(cio, jp2->UnkC, 1);		/* UnkC, colorspace unknown */
	cio_write(cio, jp2->IPR, 1);		/* IPR, no intellectual property */

	box.length = cio_tell(cio) - box.init_pos;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);
}

static void jp2_write_bpcc(opj_jp2_t *jp2, opj_cio_t *cio) {
	unsigned int i;
	opj_jp2_box_t box;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_BPCC, 4);	/* BPCC */

	for (i = 0; i < jp2->numcomps; i++) {
		cio_write(cio, jp2->comps[i].bpcc, 1);
	}

	box.length = cio_tell(cio) - box.init_pos;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);
}


static bool jp2_read_bpcc(opj_jp2_t *jp2, opj_cio_t *cio) {
	unsigned int i;
	opj_jp2_box_t box;

	opj_common_ptr cinfo = jp2->cinfo;

	jp2_read_boxhdr(cinfo, cio, &box);
	if (JP2_BPCC != box.type) {
		opj_event_msg(cinfo, EVT_ERROR, "Expected BPCC Marker\n");
		return false;
	}

	for (i = 0; i < jp2->numcomps; i++) {
		jp2->comps[i].bpcc = cio_read(cio, 1);
	}

	if (cio_tell(cio) - box.init_pos != box.length) {
		opj_event_msg(cinfo, EVT_ERROR, "Error with BPCC Box\n");
		return false;
	}

	return true;
}

static void jp2_write_colr(opj_jp2_t *jp2, opj_cio_t *cio) {
	opj_jp2_box_t box;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_COLR, 4);		/* COLR */

	cio_write(cio, jp2->meth, 1);		/* METH */
	cio_write(cio, jp2->precedence, 1);	/* PRECEDENCE */
	cio_write(cio, jp2->approx, 1);		/* APPROX */

	if (jp2->meth == 1) {
		cio_write(cio, jp2->enumcs, 4);	/* EnumCS */
	} else {
		cio_write(cio, 0, 1);			/* PROFILE (??) */
	}

	box.length = cio_tell(cio) - box.init_pos;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);
}

static void jp2_free_pclr(opj_jp2_color_t *color)
{
    opj_free(color->jp2_pclr->channel_sign);
    opj_free(color->jp2_pclr->channel_size);
    opj_free(color->jp2_pclr->entries);

	if(color->jp2_pclr->cmap) opj_free(color->jp2_pclr->cmap);

    opj_free(color->jp2_pclr); color->jp2_pclr = NULL;
}

static void free_color_data(opj_jp2_color_t *color)
{
	if(color->jp2_pclr)
   {
	jp2_free_pclr(color);
   }
	if(color->jp2_cdef) 
   {
	if(color->jp2_cdef->info) opj_free(color->jp2_cdef->info);
	opj_free(color->jp2_cdef);
   }
	if(color->icc_profile_buf) opj_free(color->icc_profile_buf);
}

static void jp2_apply_pclr(opj_jp2_color_t *color, opj_image_t *image)
{
	opj_image_comp_t *old_comps, *new_comps;
	unsigned char *channel_size, *channel_sign;
	unsigned int *entries;
	opj_jp2_cmap_comp_t *cmap;
	int *src, *dst;
	unsigned int j, max;
	unsigned short i, nr_channels, cmp, pcol;
	int k, top_k;

	channel_size = color->jp2_pclr->channel_size;
	channel_sign = color->jp2_pclr->channel_sign;
	entries = color->jp2_pclr->entries;
	cmap = color->jp2_pclr->cmap;
	nr_channels = color->jp2_pclr->nr_channels;

	old_comps = image->comps;
	new_comps = (opj_image_comp_t*)
	 opj_malloc(nr_channels * sizeof(opj_image_comp_t));

	for(i = 0; i < nr_channels; ++i)
   {
	pcol = cmap[i].pcol; cmp = cmap[i].cmp;

	new_comps[pcol] = old_comps[cmp];

	if(cmap[i].mtyp == 0) /* Direct use */
  {
	old_comps[cmp].data = NULL; continue;
  }
/* Palette mapping: */
	new_comps[pcol].data = (int*)
	 opj_malloc(old_comps[cmp].w * old_comps[cmp].h * sizeof(int));
	new_comps[pcol].prec = channel_size[i];
	new_comps[pcol].sgnd = channel_sign[i];
   }
	top_k = color->jp2_pclr->nr_entries - 1;

	for(i = 0; i < nr_channels; ++i)
   {
/* Direct use: */
	if(cmap[i].mtyp == 0) continue;

/* Palette mapping: */
	cmp = cmap[i].cmp; pcol = cmap[i].pcol;
	src = old_comps[cmp].data; 
	dst = new_comps[pcol].data;
	max = new_comps[pcol].w * new_comps[pcol].h;

	for(j = 0; j < max; ++j)
  {
/* The index */
	if((k = src[j]) < 0) k = 0; else if(k > top_k) k = top_k;
/* The colour */
	dst[j] = entries[k * nr_channels + pcol];
  }
   }
	max = image->numcomps;
	for(i = 0; i < max; ++i)
   {
	if(old_comps[i].data) opj_free(old_comps[i].data);
   }
	opj_free(old_comps);
	image->comps = new_comps;
	image->numcomps = nr_channels;

	jp2_free_pclr(color);

}/* apply_pclr() */

static bool jp2_read_pclr(opj_jp2_t *jp2, opj_cio_t *cio,
	opj_jp2_box_t *box, opj_jp2_color_t *color)
{
	opj_jp2_pclr_t *jp2_pclr;
	unsigned char *channel_size, *channel_sign;
	unsigned int *entries;
	unsigned short nr_entries, nr_channels;
	unsigned short i, j;
	unsigned char uc;

/* Part 1, I.5.3.4: 'There shall be at most one Palette box inside
 * a JP2 Header box' :
*/
	if(color->jp2_pclr) return false;

	nr_entries = cio_read(cio, 2); /* NE */
	nr_channels = cio_read(cio, 1);/* NPC */

	entries = (unsigned int*)
	 opj_malloc(nr_channels * nr_entries * sizeof(unsigned int));
	channel_size = (unsigned char*)opj_malloc(nr_channels);
	channel_sign = (unsigned char*)opj_malloc(nr_channels);

	jp2_pclr = (opj_jp2_pclr_t*)opj_malloc(sizeof(opj_jp2_pclr_t));
	jp2_pclr->channel_sign = channel_sign;
	jp2_pclr->channel_size = channel_size;
	jp2_pclr->entries = entries;
	jp2_pclr->nr_entries = nr_entries;
	jp2_pclr->nr_channels = nr_channels;
	jp2_pclr->cmap = NULL;

	color->jp2_pclr = jp2_pclr;

	for(i = 0; i < nr_channels; ++i)
   {
	uc = cio_read(cio, 1); /* Bi */
	channel_size[i] = (uc & 0x7f) + 1;
	channel_sign[i] = (uc & 0x80)?1:0;
   }

	for(j = 0; j < nr_entries; ++j)
   {
	for(i = 0; i < nr_channels; ++i)
  {
/* Cji */
	*entries++ = cio_read(cio, channel_size[i]>>3);
  }
   }

	return true;
}/* jp2_read_pclr() */

static bool jp2_read_cmap(opj_jp2_t *jp2, opj_cio_t *cio,
	opj_jp2_box_t *box, opj_jp2_color_t *color)
{
	opj_jp2_cmap_comp_t *cmap;
	unsigned short i, nr_channels;

/* Need nr_channels: */
	if(color->jp2_pclr == NULL) return false;

/* Part 1, I.5.3.5: 'There shall be at most one Component Mapping box
 * inside a JP2 Header box' :
*/
	if(color->jp2_pclr->cmap) return false;

	nr_channels = color->jp2_pclr->nr_channels;
	cmap = (opj_jp2_cmap_comp_t*)
	 opj_malloc(nr_channels * sizeof(opj_jp2_cmap_comp_t));

	for(i = 0; i < nr_channels; ++i)
   {
	cmap[i].cmp = cio_read(cio, 2);
	cmap[i].mtyp = cio_read(cio, 1);
	cmap[i].pcol = cio_read(cio, 1);

   }
	color->jp2_pclr->cmap = cmap;

	return true;
}/* jp2_read_cmap() */

static void jp2_apply_cdef(opj_image_t *image, opj_jp2_color_t *color)
{
	opj_jp2_cdef_info_t *info;
	int color_space;
	unsigned short i, n, cn, typ, asoc, acn;

	color_space = image->color_space;
	info = color->jp2_cdef->info;
	n = color->jp2_cdef->n;

	for(i = 0; i < n; ++i)
   {
/* WATCH: acn = asoc - 1 ! */
	if((asoc = info[i].asoc) == 0) continue;

	cn = info[i].cn; typ = info[i].typ; acn = asoc - 1;

	if(cn != acn)
  {
	opj_image_comp_t saved;

	memcpy(&saved, &image->comps[cn], sizeof(opj_image_comp_t));
	memcpy(&image->comps[cn], &image->comps[acn], sizeof(opj_image_comp_t));
	memcpy(&image->comps[acn], &saved, sizeof(opj_image_comp_t));

	info[i].asoc = cn + 1;
	info[acn].asoc = info[acn].cn + 1;
  }
   }
	if(color->jp2_cdef->info) opj_free(color->jp2_cdef->info);

	opj_free(color->jp2_cdef); color->jp2_cdef = NULL;

}/* jp2_apply_cdef() */

static bool jp2_read_cdef(opj_jp2_t *jp2, opj_cio_t *cio,
	opj_jp2_box_t *box, opj_jp2_color_t *color)
{
	opj_jp2_cdef_info_t *info;
	unsigned short i, n;

/* Part 1, I.5.3.6: 'The shall be at most one Channel Definition box
 * inside a JP2 Header box.' 
*/
	if(color->jp2_cdef) return false;

	if((n = cio_read(cio, 2)) == 0) return false; /* szukw000: FIXME */

	info = (opj_jp2_cdef_info_t*)
	 opj_malloc(n * sizeof(opj_jp2_cdef_info_t));

	color->jp2_cdef = (opj_jp2_cdef_t*)opj_malloc(sizeof(opj_jp2_cdef_t));
	color->jp2_cdef->info = info;
	color->jp2_cdef->n = n;

	for(i = 0; i < n; ++i)
   {
	info[i].cn = cio_read(cio, 2);
	info[i].typ = cio_read(cio, 2);
	info[i].asoc = cio_read(cio, 2);

   }
	return true;
}/* jp2_read_cdef() */

static bool jp2_read_colr(opj_jp2_t *jp2, opj_cio_t *cio,
	opj_jp2_box_t *box, opj_jp2_color_t *color) 
{
	int skip_len;
    opj_common_ptr cinfo;

/* Part 1, I.5.3.3 : 'A conforming JP2 reader shall ignore all Colour
 * Specification boxes after the first.' 
*/
	if(color->jp2_has_colr) return false;

	cinfo = jp2->cinfo;

	jp2->meth = cio_read(cio, 1);		/* METH */
	jp2->precedence = cio_read(cio, 1);	/* PRECEDENCE */
	jp2->approx = cio_read(cio, 1);		/* APPROX */

	if (jp2->meth == 1) 
   {
	jp2->enumcs = cio_read(cio, 4);	/* EnumCS */
   } 
	else 
   {
/* skip PROFILE */
	skip_len = box->init_pos + box->length - cio_tell(cio);
	if (skip_len < 0) 
  {
	opj_event_msg(cinfo, EVT_ERROR, "Error with COLR box size\n");
	return false;
  }
	if(skip_len > 0)
  {
	unsigned char *start;

	start = cio_getbp(cio);
	color->icc_profile_buf = (unsigned char*)opj_malloc(skip_len);
	color->icc_profile_len = skip_len;

	cio_skip(cio, box->init_pos + box->length - cio_tell(cio));

	memcpy(color->icc_profile_buf, start, skip_len);
  }
   }

	if (cio_tell(cio) - box->init_pos != box->length) 
   {
	opj_event_msg(cinfo, EVT_ERROR, "Error with COLR Box\n");
	return false;
   }
	color->jp2_has_colr = 1;

	return true;
}/* jp2_read_colr() */

bool jp2_read_jp2h(opj_jp2_t *jp2, opj_cio_t *cio, opj_jp2_color_t *color) 
{
	opj_jp2_box_t box;
	unsigned int jp2h_end;

	opj_common_ptr cinfo = jp2->cinfo;

	jp2_read_boxhdr(cinfo, cio, &box);
	do 
   {
	if (JP2_JP2H != box.type) 
  {
	if (box.type == JP2_JP2C) 
 {
	opj_event_msg(cinfo, EVT_ERROR, "Expected JP2H Marker\n");
	return false;
 }
	cio_skip(cio, box.length - 8);

	if(cio->bp >= cio->end) return false;

	jp2_read_boxhdr(cinfo, cio, &box);
  }
   } while(JP2_JP2H != box.type);

	if (!jp2_read_ihdr(jp2, cio))
		return false;
	jp2h_end = box.init_pos + box.length;

	if (jp2->bpc == 255) 
   {
	if (!jp2_read_bpcc(jp2, cio))
		return false;
   }
	jp2_read_boxhdr(cinfo, cio, &box);

	while(cio_tell(cio) < jp2h_end)
   {
	if(box.type == JP2_COLR)
  {
	if( !jp2_read_colr(jp2, cio, &box, color))
 {
    cio_seek(cio, box.init_pos + 8);
    cio_skip(cio, box.length - 8);
 }
    jp2_read_boxhdr(cinfo, cio, &box);
    continue;
  }
    if(box.type == JP2_CDEF)
  {
    if( !jp2_read_cdef(jp2, cio, &box, color))
 {
    cio_seek(cio, box.init_pos + 8);
    cio_skip(cio, box.length - 8);
 }
    jp2_read_boxhdr(cinfo, cio, &box);
    continue;
  }
    if(box.type == JP2_PCLR)
  {
    if( !jp2_read_pclr(jp2, cio, &box, color))
 {
    cio_seek(cio, box.init_pos + 8);
    cio_skip(cio, box.length - 8);
 }
    jp2_read_boxhdr(cinfo, cio, &box);
    continue;
  }
    if(box.type == JP2_CMAP)
  {
    if( !jp2_read_cmap(jp2, cio, &box, color))
 {
    cio_seek(cio, box.init_pos + 8);
    cio_skip(cio, box.length - 8);
 }
    jp2_read_boxhdr(cinfo, cio, &box);
    continue;
  }
	cio_seek(cio, box.init_pos + 8);
	cio_skip(cio, box.length - 8);
	jp2_read_boxhdr(cinfo, cio, &box);

   }/* while(cio_tell(cio) < box_end) */

	cio_seek(cio, jp2h_end);

/* Part 1, I.5.3.3 : 'must contain at least one' */
	return (color->jp2_has_colr == 1);

}/* jp2_read_jp2h() */

opj_image_t* jp2_decode(opj_jp2_t *jp2, opj_cio_t *cio, 
	opj_codestream_info_t *cstr_info) 
{
	opj_common_ptr cinfo;
	opj_image_t *image = NULL;
	opj_jp2_color_t color;

	if(!jp2 || !cio) 
   {
	return NULL;
   }
	memset(&color, 0, sizeof(opj_jp2_color_t));
	cinfo = jp2->cinfo;

/* JP2 decoding */
	if(!jp2_read_struct(jp2, cio, &color)) 
   {
	free_color_data(&color);
	opj_event_msg(cinfo, EVT_ERROR, "Failed to decode jp2 structure\n");
	return NULL;
   }

/* J2K decoding */
	image = j2k_decode(jp2->j2k, cio, cstr_info);

	if(!image) 
   {
	free_color_data(&color);
	opj_event_msg(cinfo, EVT_ERROR, "Failed to decode J2K image\n");
	return NULL;
   }

/* Set Image Color Space */
	if (jp2->enumcs == 16)
		image->color_space = CLRSPC_SRGB;
	else if (jp2->enumcs == 17)
		image->color_space = CLRSPC_GRAY;
	else if (jp2->enumcs == 18)
		image->color_space = CLRSPC_SYCC;
	else
		image->color_space = CLRSPC_UNKNOWN;

	if(color.jp2_cdef)
   {
	jp2_apply_cdef(image, &color);
   }
	if(color.jp2_pclr)
   {
/* Part 1, I.5.3.4: Either both or none : */
	if( !color.jp2_pclr->cmap) 
	 jp2_free_pclr(&color);
	else
	 jp2_apply_pclr(&color, image);
   }
	if(color.icc_profile_buf)
   {
	image->icc_profile_buf = color.icc_profile_buf;
	color.icc_profile_buf = NULL;
	image->icc_profile_len = color.icc_profile_len;
   }
	return image;

}/* jp2_decode() */


void jp2_write_jp2h(opj_jp2_t *jp2, opj_cio_t *cio) {
	opj_jp2_box_t box;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_JP2H, 4);	/* JP2H */

	jp2_write_ihdr(jp2, cio);

	if (jp2->bpc == 255) {
		jp2_write_bpcc(jp2, cio);
	}
	jp2_write_colr(jp2, cio);

	box.length = cio_tell(cio) - box.init_pos;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);
}

static void jp2_write_ftyp(opj_jp2_t *jp2, opj_cio_t *cio) {
	unsigned int i;
	opj_jp2_box_t box;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_FTYP, 4);		/* FTYP */

	cio_write(cio, jp2->brand, 4);		/* BR */
	cio_write(cio, jp2->minversion, 4);	/* MinV */

	for (i = 0; i < jp2->numcl; i++) {
		cio_write(cio, jp2->cl[i], 4);	/* CL */
	}

	box.length = cio_tell(cio) - box.init_pos;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);
}

static bool jp2_read_ftyp(opj_jp2_t *jp2, opj_cio_t *cio) {
	int i;
	opj_jp2_box_t box;

	opj_common_ptr cinfo = jp2->cinfo;

	jp2_read_boxhdr(cinfo, cio, &box);

	if (JP2_FTYP != box.type) {
		opj_event_msg(cinfo, EVT_ERROR, "Expected FTYP Marker\n");
		return false;
	}

	jp2->brand = cio_read(cio, 4);		/* BR */
	jp2->minversion = cio_read(cio, 4);	/* MinV */
	jp2->numcl = (box.length - 16) / 4;
	jp2->cl = (unsigned int *) opj_malloc(jp2->numcl * sizeof(unsigned int));

	for (i = 0; i < (int)jp2->numcl; i++) {
		jp2->cl[i] = cio_read(cio, 4);	/* CLi */
	}

	if (cio_tell(cio) - box.init_pos != box.length) {
		opj_event_msg(cinfo, EVT_ERROR, "Error with FTYP Box\n");
		return false;
	}

	return true;
}

static int jp2_write_jp2c(opj_jp2_t *jp2, opj_cio_t *cio, opj_image_t *image, opj_codestream_info_t *cstr_info) {
	unsigned int j2k_codestream_offset, j2k_codestream_length;
	opj_jp2_box_t box;

	opj_j2k_t *j2k = jp2->j2k;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_JP2C, 4);	/* JP2C */

	/* J2K encoding */
	j2k_codestream_offset = cio_tell(cio);
	if(!j2k_encode(j2k, cio, image, cstr_info)) {
		opj_event_msg(j2k->cinfo, EVT_ERROR, "Failed to encode image\n");
		return 0;
	}
	j2k_codestream_length = cio_tell(cio) - j2k_codestream_offset;

	jp2->j2k_codestream_offset = j2k_codestream_offset;
	jp2->j2k_codestream_length = j2k_codestream_length;

	box.length = 8 + jp2->j2k_codestream_length;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);

	return box.length;
}

static bool jp2_read_jp2c(opj_jp2_t *jp2, opj_cio_t *cio, unsigned int *j2k_codestream_length, unsigned int *j2k_codestream_offset) {
	opj_jp2_box_t box;

	opj_common_ptr cinfo = jp2->cinfo;

	jp2_read_boxhdr(cinfo, cio, &box);
	do {
		if(JP2_JP2C != box.type) {
			cio_skip(cio, box.length - 8);
			jp2_read_boxhdr(cinfo, cio, &box);
		}
	} while(JP2_JP2C != box.type);

	*j2k_codestream_offset = cio_tell(cio);
	*j2k_codestream_length = box.length - 8;

	return true;
}

static void jp2_write_jp(opj_cio_t *cio) {
	opj_jp2_box_t box;

	box.init_pos = cio_tell(cio);
	cio_skip(cio, 4);
	cio_write(cio, JP2_JP, 4);		/* JP2 signature */
	cio_write(cio, 0x0d0a870a, 4);

	box.length = cio_tell(cio) - box.init_pos;
	cio_seek(cio, box.init_pos);
	cio_write(cio, box.length, 4);	/* L */
	cio_seek(cio, box.init_pos + box.length);
}

static bool jp2_read_jp(opj_jp2_t *jp2, opj_cio_t *cio) {
	opj_jp2_box_t box;

	opj_common_ptr cinfo = jp2->cinfo;

	jp2_read_boxhdr(cinfo, cio, &box);
	if (JP2_JP != box.type) {
		opj_event_msg(cinfo, EVT_ERROR, "Expected JP Marker\n");
		return false;
	}
	if (0x0d0a870a != cio_read(cio, 4)) {
		opj_event_msg(cinfo, EVT_ERROR, "Error with JP Marker\n");
		return false;
	}
	if (cio_tell(cio) - box.init_pos != box.length) {
		opj_event_msg(cinfo, EVT_ERROR, "Error with JP Box size\n");
		return false;
	}

	return true;
}


static bool jp2_read_struct(opj_jp2_t *jp2, opj_cio_t *cio,
	opj_jp2_color_t *color) {
	if (!jp2_read_jp(jp2, cio))
		return false;
	if (!jp2_read_ftyp(jp2, cio))
		return false;
	if (!jp2_read_jp2h(jp2, cio, color))
		return false;
	if (!jp2_read_jp2c(jp2, cio, &jp2->j2k_codestream_length, &jp2->j2k_codestream_offset))
		return false;
	
	return true;
}

/* ----------------------------------------------------------------------- */
/* JP2 decoder interface                                             */
/* ----------------------------------------------------------------------- */

opj_jp2_t* jp2_create_decompress(opj_common_ptr cinfo) {
	opj_jp2_t *jp2 = (opj_jp2_t*) opj_calloc(1, sizeof(opj_jp2_t));
	if(jp2) {
		jp2->cinfo = cinfo;
		/* create the J2K codec */
		jp2->j2k = j2k_create_decompress(cinfo);
		if(jp2->j2k == NULL) {
			jp2_destroy_decompress(jp2);
			return NULL;
		}
	}
	return jp2;
}

void jp2_destroy_decompress(opj_jp2_t *jp2) {
	if(jp2) {
		/* destroy the J2K codec */
		j2k_destroy_decompress(jp2->j2k);

		if(jp2->comps) {
			opj_free(jp2->comps);
		}
		if(jp2->cl) {
			opj_free(jp2->cl);
		}
		opj_free(jp2);
	}
}

void jp2_setup_decoder(opj_jp2_t *jp2, opj_dparameters_t *parameters) {
	/* setup the J2K codec */
	j2k_setup_decoder(jp2->j2k, parameters);
	/* further JP2 initializations go here */
}

/* ----------------------------------------------------------------------- */
/* JP2 encoder interface                                             */
/* ----------------------------------------------------------------------- */

opj_jp2_t* jp2_create_compress(opj_common_ptr cinfo) {
	opj_jp2_t *jp2 = (opj_jp2_t*)opj_malloc(sizeof(opj_jp2_t));
	if(jp2) {
		jp2->cinfo = cinfo;
		/* create the J2K codec */
		jp2->j2k = j2k_create_compress(cinfo);
		if(jp2->j2k == NULL) {
			jp2_destroy_compress(jp2);
			return NULL;
		}
	}
	return jp2;
}

void jp2_destroy_compress(opj_jp2_t *jp2) {
	if(jp2) {
		/* destroy the J2K codec */
		j2k_destroy_compress(jp2->j2k);

		if(jp2->comps) {
			opj_free(jp2->comps);
		}
		if(jp2->cl) {
			opj_free(jp2->cl);
		}
		opj_free(jp2);
	}
}

void jp2_setup_encoder(opj_jp2_t *jp2, opj_cparameters_t *parameters, opj_image_t *image) {
	int i;
	int depth_0, sign;

	if(!jp2 || !parameters || !image)
		return;

	/* setup the J2K codec */
	/* ------------------- */

	/* Check if number of components respects standard */
	if (image->numcomps < 1 || image->numcomps > 16384) {
		opj_event_msg(jp2->cinfo, EVT_ERROR, "Invalid number of components specified while setting up JP2 encoder\n");
		return;
	}

	j2k_setup_encoder(jp2->j2k, parameters, image);

	/* setup the JP2 codec */
	/* ------------------- */
	
	/* Profile box */

	jp2->brand = JP2_JP2;	/* BR */
	jp2->minversion = 0;	/* MinV */
	jp2->numcl = 1;
	jp2->cl = (unsigned int*) opj_malloc(jp2->numcl * sizeof(unsigned int));
	jp2->cl[0] = JP2_JP2;	/* CL0 : JP2 */

	/* Image Header box */

	jp2->numcomps = image->numcomps;	/* NC */
	jp2->comps = (opj_jp2_comps_t*) opj_malloc(jp2->numcomps * sizeof(opj_jp2_comps_t));
	jp2->h = image->y1 - image->y0;		/* HEIGHT */
	jp2->w = image->x1 - image->x0;		/* WIDTH */
	/* BPC */
	depth_0 = image->comps[0].prec - 1;
	sign = image->comps[0].sgnd;
	jp2->bpc = depth_0 + (sign << 7);
	for (i = 1; i < image->numcomps; i++) {
		int depth = image->comps[i].prec - 1;
		sign = image->comps[i].sgnd;
		if (depth_0 != depth)
			jp2->bpc = 255;
	}
	jp2->C = 7;			/* C : Always 7 */
	jp2->UnkC = 0;		/* UnkC, colorspace specified in colr box */
	jp2->IPR = 0;		/* IPR, no intellectual property */
	
	/* BitsPerComponent box */

	for (i = 0; i < image->numcomps; i++) {
		jp2->comps[i].bpcc = image->comps[i].prec - 1 + (image->comps[i].sgnd << 7);
	}

	/* Colour Specification box */

	if ((image->numcomps == 1 || image->numcomps == 3) && (jp2->bpc != 255)) {
		jp2->meth = 1;	/* METH: Enumerated colourspace */
	} else {
		jp2->meth = 2;	/* METH: Restricted ICC profile */
	}
	if (jp2->meth == 1) {
		if (image->color_space == 1)
			jp2->enumcs = 16;	/* sRGB as defined by IEC 61966𣇻 */
		else if (image->color_space == 2)
			jp2->enumcs = 17;	/* greyscale */
		else if (image->color_space == 3)
			jp2->enumcs = 18;	/* YUV */
	} else {
		jp2->enumcs = 0;		/* PROFILE (??) */
	}
	jp2->precedence = 0;	/* PRECEDENCE */
	jp2->approx = 0;		/* APPROX */

}

bool jp2_encode(opj_jp2_t *jp2, opj_cio_t *cio, opj_image_t *image, opj_codestream_info_t *cstr_info) {

	/* JP2 encoding */

	/* JPEG 2000 Signature box */
	jp2_write_jp(cio);
	/* File Type box */
	jp2_write_ftyp(jp2, cio);
	/* JP2 Header box */
	jp2_write_jp2h(jp2, cio);

	/* J2K encoding */

	if(!jp2_write_jp2c(jp2, cio, image, cstr_info)) {
		opj_event_msg(jp2->cinfo, EVT_ERROR, "Failed to encode image\n");
		return false;
	}

	return true;
}


