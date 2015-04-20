/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2003-2007, Francois-Olivier Devaux 
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
/*-- UNUSED
static bool jp2_read_boxhdr(opj_common_ptr cinfo, opj_cio_t *cio, opj_jp2_box_t *box);
--*/
/*
* 
* Read box headers
*
*/

int mj2_read_boxhdr(mj2_box_t * box, opj_cio_t *cio)
{
  box->init_pos = cio_tell(cio);
  box->length = cio_read(cio, 4);
  box->type = cio_read(cio, 4);
  if (box->length == 1) {
    if (cio_read(cio, 4) != 0) {
      opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Cannot handle box sizes higher than 2^32\n");
      return 1;
    };
    box->length = cio_read(cio, 4);
    if (box->length == 0) 
      box->length = cio_numbytesleft(cio) + 12;
  }
  else if (box->length == 0) {
    box->length = cio_numbytesleft(cio) + 8;
  }
  return 0;
}

/*
* 
* Initialisation of a Standard Movie, given a simple movie structure defined by the user 
* The movie will have one sample per chunk
* 
* Arguments: opj_mj2_t * movie
* Several variables of "movie" must be defined in order to enable a correct execution of 
* this function:
*   - The number of tracks of each type (movie->num_vtk, movie->num_stk, movie->num_htk)
*   - The memory for each must be allocated (movie->tk)
*   - For each track:
*	  The track type (tk->track_type)
*	  The number of sample (tk->num_samples)
*	  The sample rate (tk->sample_rate)
*
*/

int mj2_init_stdmovie(opj_mj2_t * movie)
{
  int i;
  unsigned int j;
  time_t ltime;
	
  movie->brand = MJ2_MJ2;
  movie->minversion = 0;
  movie->num_cl = 2;
  movie->cl = (unsigned int*) opj_malloc(movie->num_cl * sizeof(unsigned int));

  movie->cl[0] = MJ2_MJ2;
  movie->cl[1] = MJ2_MJ2S;
  time(&ltime);			/* Time since 1/1/70 */
  movie->creation_time = (unsigned int) ltime + 2082844800;	/* Seconds between 1/1/04 and 1/1/70 */
  movie->timescale = 1000;
	
  movie->rate = 1 << 16;		/* Rate to play presentation  (default = 0x00010000)          */
  movie->volume = 1 << 8;		/* Movie volume (default = 0x0100)                            */
  movie->trans_matrix[0] = 0x00010000;	/* Transformation matrix for video                            */
  movie->trans_matrix[1] = 0;	/* Unity is { 0x00010000,0,0,0,0x00010000,0,0,0,0x40000000 }  */
  movie->trans_matrix[2] = 0;
  movie->trans_matrix[3] = 0;
  movie->trans_matrix[4] = 0x00010000;
  movie->trans_matrix[5] = 0;
  movie->trans_matrix[6] = 0;
  movie->trans_matrix[7] = 0;
  movie->trans_matrix[8] = 0x40000000;
  movie->next_tk_id = 1;
	
  for (i = 0; i < movie->num_htk + movie->num_stk + movie->num_vtk; i++) {
    mj2_tk_t *tk = &movie->tk[i];
    movie->next_tk_id++;
    tk->jp2_struct.comps = NULL;
    tk->jp2_struct.cl = NULL;
    
    if (tk->track_type == 0) {
      if (tk->num_samples == 0)
				return 1;
			
      tk->Dim[0] = 0;
      tk->Dim[1] = 0;
			
      tk->timescale = 1000;	/* Timescale = 1 ms                                          */
			
      tk->chunk[0].num_samples = 1;
      tk->chunk[0].sample_descr_idx = 1;
			
      tk->same_sample_size = 0;
			
      tk->num_samplestochunk = 1;	/* One sample per chunk                                      */
		tk->sampletochunk = (mj2_sampletochunk_t*) opj_malloc(tk->num_samplestochunk * sizeof(mj2_sampletochunk_t));
      tk->sampletochunk[0].first_chunk = 1;
      tk->sampletochunk[0].samples_per_chunk = 1;
      tk->sampletochunk[0].sample_descr_idx = 1;
      
      if (tk->sample_rate == 0) {
				opj_event_msg(tk->cinfo, EVT_ERROR,
					"Error while initializing MJ2 movie: Sample rate of track %d must be different from zero\n",
					tk->track_ID);
				return 1;
      }
			
      for (j = 0; j < tk->num_samples; j++) {
				tk->sample[j].sample_delta = tk->timescale / tk->sample_rate;
      }
			
      tk->num_tts = 1;
		tk->tts = (mj2_tts_t*) opj_malloc(tk->num_tts * sizeof(mj2_tts_t));
      tk->tts[0].sample_count = tk->num_samples;
      tk->tts[0].sample_delta = tk->timescale / tk->sample_rate;
			
      tk->horizresolution = 0x00480000;	/* Horizontal resolution (typically 72)                       */
      tk->vertresolution = 0x00480000;	/* Vertical resolution (typically 72)                         */
      tk->compressorname[0] = 0x0f4d6f74;	/* Compressor Name[]: Motion JPEG2000                         */
      tk->compressorname[1] = 0x696f6e20;
      tk->compressorname[2] = 0x4a504547;
      tk->compressorname[3] = 0x32303030;
      tk->compressorname[4] = 0x00120000;
      tk->compressorname[5] = 0;
      tk->compressorname[6] = 0x00000042;
      tk->compressorname[7] = 0x000000DC;
      tk->num_url = 0;		/* Number of URL                                              */
      tk->num_urn = 0;		/* Number of URN                                              */
      tk->graphicsmode = 0;	/* Graphicsmode                                               */
      tk->opcolor[0] = 0;	/* OpColor                                                    */
      tk->opcolor[1] = 0;	/* OpColor                                                    */
      tk->opcolor[2] = 0;	/* OpColor                                                    */
      tk->creation_time = movie->creation_time;	/* Seconds between 1/1/04 and 1/1/70          */
      tk->language = 0;		/* Language (undefined)					      */
      tk->layer = 0;
      tk->volume = 1 << 8;		/* Movie volume (default = 0x0100) */
      tk->trans_matrix[0] = 0x00010000;	/* Transformation matrix for track */
      tk->trans_matrix[1] = 0;	/* Unity is { 0x00010000,0,0,0,0x00010000,0,0,0,0x40000000 }  */
      tk->trans_matrix[2] = 0;
      tk->trans_matrix[3] = 0;
      tk->trans_matrix[4] = 0x00010000;
      tk->trans_matrix[5] = 0;
      tk->trans_matrix[6] = 0;
      tk->trans_matrix[7] = 0;
      tk->trans_matrix[8] = 0x40000000;
      tk->fieldcount = 1;
      tk->fieldorder = 0;
      tk->or_fieldcount = 1;
      tk->or_fieldorder = 0;
      tk->num_br = 2;
		tk->br = (unsigned int*) opj_malloc(tk->num_br * sizeof(unsigned int));
      tk->br[0] = MJ2_JP2;
      tk->br[1] = MJ2_J2P0;
      tk->num_jp2x = 0;
      tk->hsub = 2;		/* 4:2:0                                                      */
      tk->vsub = 2;		/* 4:2:0                                                      */
      tk->hoff = 0;
      tk->voff = 0;
      tk->visual_w = tk->w << 16;
      tk->visual_h = tk->h << 16;
    }
    else {
      tk->num_br = 0;
      tk->jp2xdata = NULL;
    }
  }
  return 0;
}

/*
* Time To Sample box Decompact
*
*/
void mj2_tts_decompact(mj2_tk_t * tk)
{
  int i, j;
  tk->num_samples = 0;
  for (i = 0; i < tk->num_tts; i++) {
    tk->num_samples += tk->tts[i].sample_count;
  }

  tk->sample = (mj2_sample_t*) opj_malloc(tk->num_samples * sizeof(mj2_sample_t));

  for (i = 0; i < tk->num_tts; i++) {
    for (j = 0; j < tk->tts[i].sample_count; j++) {
      tk->sample[j].sample_delta = tk->tts[i].sample_delta;
    }
  }
}

/*
* Sample To Chunk box Decompact
*
*/
void mj2_stsc_decompact(mj2_tk_t * tk)
{
  int j, i;
  unsigned int k;
  int sampleno=0;
  
  if (tk->num_samplestochunk == 1) {
    tk->num_chunks =
      (unsigned int) ceil((double) tk->num_samples /
      (double) tk->sampletochunk[0].samples_per_chunk);
	 tk->chunk = (mj2_chunk_t*) opj_malloc(tk->num_chunks * sizeof(mj2_chunk_t));
    for (k = 0; k < tk->num_chunks; k++) {
      tk->chunk[k].num_samples = tk->sampletochunk[0].samples_per_chunk;
    }
    
  } else {
    tk->chunk = (mj2_chunk_t*) opj_malloc(tk->num_samples * sizeof(mj2_chunk_t));
    tk->num_chunks = 0;
    for (i = 0; i < tk->num_samplestochunk -1 ; i++) {
      for (j = tk->sampletochunk[i].first_chunk - 1;
      j < tk->sampletochunk[i + 1].first_chunk - 1; j++) {
				tk->chunk[j].num_samples = tk->sampletochunk[i].samples_per_chunk;
				tk->num_chunks++;
				sampleno += tk->chunk[j].num_samples;
      }
    }
    tk->num_chunks += (int)(tk->num_samples  - sampleno) / tk->sampletochunk[tk->num_samplestochunk - 1].samples_per_chunk;
    for (k = tk->sampletochunk[tk->num_samplestochunk - 1].first_chunk - 1;
    k < tk->num_chunks; k++) {
      tk->chunk[k].num_samples =
				tk->sampletochunk[tk->num_samplestochunk - 1].samples_per_chunk;
    }
    tk->chunk = (mj2_chunk_t*)
	 opj_realloc(tk->chunk, tk->num_chunks * sizeof(mj2_chunk_t));
  }
  
}


/*
* Chunk offset box Decompact
*
*/
void mj2_stco_decompact(mj2_tk_t * tk)
{
  int j;
  unsigned int i;
  int k = 0;
  int intra_chunk_offset;
	
  for (i = 0; i < tk->num_chunks; i++) {
    intra_chunk_offset = 0;
    for (j = 0; j < tk->chunk[i].num_samples; j++) {
      tk->sample[k].offset = intra_chunk_offset + tk->chunk[i].offset;
      intra_chunk_offset += tk->sample[k].sample_size;
      k++;
    }
  }
}

/*
* Write the JP box
*
* JP Signature box
*
*/
void mj2_write_jp(opj_cio_t *cio)
{
  mj2_box_t box;
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
	
  cio_write(cio, MJ2_JP, 4);		/* JP */
  cio_write(cio, 0x0d0a870a, 4);	/* 0x0d0a870a required in a JP box */
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the JP box
*
* JPEG 2000 signature
*
*/
int mj2_read_jp(opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_JP != box.type) {	/* Check Marker */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected JP Marker\n");
    return 1;
  }
  if (0x0d0a870a != cio_read(cio, 4)) {	/* read the 0x0d0a870a required in a JP box */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with JP Marker\n");
    return 1;
  }
  if (cio_tell(cio) - box.init_pos != box.length) {	/* Check box length */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with JP Box size \n");
    return 1;
  }
  return 0;
	
}

/*
* Write the FTYP box
*
* File type box
*
*/
void mj2_write_ftyp(opj_mj2_t * movie, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
	
  cio_write(cio, MJ2_FTYP, 4);	/* FTYP       */
  cio_write(cio, movie->brand, 4);	/* BR         */
  cio_write(cio, movie->minversion, 4);	/* MinV       */
	
  for (i = 0; i < movie->num_cl; i++)
    cio_write(cio, movie->cl[i], 4);	/* CL         */
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* Length     */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the FTYP box
*
* File type box
*
*/
int mj2_read_ftyp(opj_mj2_t * movie, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);	/* Box Size */
  if (MJ2_FTYP != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected FTYP Marker\n");
    return 1;
  }
	
  movie->brand = cio_read(cio, 4);	/* BR              */
  movie->minversion = cio_read(cio, 4);	/* MinV            */
  movie->num_cl = (box.length - 16) / 4;
  movie->cl = (unsigned int*) opj_malloc(movie->num_cl * sizeof(unsigned int));

  for (i = movie->num_cl - 1; i > -1; i--)
    movie->cl[i] = cio_read(cio, 4);	/* CLi */
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with FTYP Box\n");
    return 1;
  }
  return 0;
}


/*
* Write the STCO box
*
* Chunk Offset Box
*
*/
void mj2_write_stco(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
  unsigned int i;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_STCO, 4);	/* STCO       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, tk->num_chunks, 4);	/* Entry Count */
	
  for (i = 0; i < tk->num_chunks; i++) {
    cio_write(cio, tk->chunk[i].offset, 4);	/* Entry offset */
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the STCO box
*
* Chunk Offset Box
*
*/
int mj2_read_stco(mj2_tk_t * tk, opj_cio_t *cio)
{
  unsigned int i;
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);	/* Box Size */
  if (MJ2_STCO != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected STCO Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in STCO box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in STCO box. Expected flag 0\n");
    return 1;
  }
	
	
  if (cio_read(cio, 4) != tk->num_chunks) {
    opj_event_msg(cio->cinfo, EVT_ERROR, 
			"Error in STCO box: expecting same amount of entry-count as chunks \n");
  } else {
    for (i = 0; i < tk->num_chunks; i++) {
      tk->chunk[i].offset = cio_read(cio, 4);	/* Entry offset */
    }
  }
	
  mj2_stco_decompact(tk);
	
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with STCO Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the STSZ box
*
* Sample size box
*
*/
void mj2_write_stsz(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
  unsigned int i;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_STSZ, 4);	/* STSZ       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  if (tk->same_sample_size == 1) {	/* If they all have the same size */
    cio_write(cio, tk->sample[0].sample_size, 4);	/* Size */
		
    cio_write(cio, 1, 4);		/* Entry count = 1 */
  }
	
  else {
    cio_write(cio, 0, 4);		/* Sample Size = 0 becase they all have different sizes */
		
    cio_write(cio, tk->num_samples, 4);	/* Sample Count */
		
    for (i = 0; i < tk->num_samples; i++) {
      cio_write(cio, tk->sample[i].sample_size, 4);
    }
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the STSZ box
*
* Sample size box
*
*/
int mj2_read_stsz(mj2_tk_t * tk, opj_cio_t *cio)
{
  int sample_size;
  unsigned int i;
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);	/* Box Size */
  if (MJ2_STSZ != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected STSZ Marker\n");
    return 1;
  }
	
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in STSZ box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in STSZ box. Expected flag 0\n");
    return 1;
  }
	
  sample_size = cio_read(cio, 4);
	
  if (sample_size != 0) {	/* Samples do have the same size */
    tk->same_sample_size = 1;
    for (i = 0; i < tk->num_samples; i++) {
      tk->sample[i].sample_size = sample_size;
    }
    cio_skip(cio,4);		/* Sample count = 1 */
  } else {
    tk->same_sample_size = 0;
    if (tk->num_samples != cio_read(cio, 4)) {	/* Sample count */
      opj_event_msg(cio->cinfo, EVT_ERROR,
				"Error in STSZ box. Expected that sample-count is number of samples in track\n");
      return 1;
    }
    for (i = 0; i < tk->num_samples; i++) {
      tk->sample[i].sample_size = cio_read(cio, 4);	/* Sample Size */
    }
		
    if (cio_tell(cio) - box.init_pos != box.length) {
      opj_event_msg(cio->cinfo, EVT_ERROR, "Error with STSZ Box size\n");
      return 1;
    }
  }
  return 0;
	
}

/*
* Write the STSC box
*
* Sample to Chunk
*
*/
void mj2_write_stsc(mj2_tk_t * tk, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_STSC, 4);	/* STSC       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, tk->num_samplestochunk, 4);	/* Entry Count */
	
  for (i = 0; i < tk->num_samplestochunk; i++) {
    cio_write(cio, tk->sampletochunk[i].first_chunk, 4);	/* First Chunk */
    cio_write(cio, tk->sampletochunk[i].samples_per_chunk, 4);	/* Samples per chunk */
    cio_write(cio, tk->sampletochunk[i].sample_descr_idx, 4);	/* Samples description index */
  }
	
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the STSC box
*
* Sample to Chunk
*
*/
int mj2_read_stsc(mj2_tk_t * tk, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);	/* Box Size */
  if (MJ2_STSC != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected STSC Marker\n");
    return 1;
  }
	
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in STSC box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in STSC box. Expected flag 0\n");
    return 1;
  }
	
  tk->num_samplestochunk = cio_read(cio, 4);

  tk->sampletochunk = (mj2_sampletochunk_t*) opj_malloc(tk->num_samplestochunk * sizeof(mj2_sampletochunk_t));

  for (i = 0; i < tk->num_samplestochunk; i++) {
    tk->sampletochunk[i].first_chunk = cio_read(cio, 4);
    tk->sampletochunk[i].samples_per_chunk = cio_read(cio, 4);
    tk->sampletochunk[i].sample_descr_idx = cio_read(cio, 4);
  }
	
  mj2_stsc_decompact(tk);	/* decompact sample to chunk box */
	
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with STSC Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the STTS box
*
* Time to Sample Box
*
*/
void mj2_write_stts(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  int i;
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_STTS, 4);	/* STTS       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, tk->num_tts, 4);	/* entry_count */
  for (i = 0; i < tk->num_tts; i++) {
    cio_write(cio, tk->tts[i].sample_count, 4);	/* Sample-count */
    cio_write(cio, tk->tts[i].sample_delta, 4);	/* Sample-Delta */
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the STTS box
*
* 
*
*/
int mj2_read_stts(mj2_tk_t * tk, opj_cio_t *cio)
{
  int i;
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_STTS != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected STTS Marker\n");
    return 1;
  }
	
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in STTS box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in STTS box. Expected flag 0\n");
    return 1;
  }
	
  tk->num_tts = cio_read(cio, 4);

  tk->tts = (mj2_tts_t*) opj_malloc(tk->num_tts * sizeof(mj2_tts_t));

  for (i = 0; i < tk->num_tts; i++) {
    tk->tts[i].sample_count = cio_read(cio, 4);
    tk->tts[i].sample_delta = cio_read(cio, 4);
  }
	
  mj2_tts_decompact(tk);
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with STTS Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the FIEL box
*
* Field coding Box
*
*/
void mj2_write_fiel(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_FIEL, 4);	/* STTS       */
	
  cio_write(cio, tk->fieldcount, 1);	/* Field count */
  cio_write(cio, tk->fieldorder, 1);	/* Field order */
	
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the FIEL box
*
* Field coding Box
*
*/
int mj2_read_fiel(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_FIEL != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected FIEL Marker\n");
    return 1;
  }
	
	
  tk->fieldcount = cio_read(cio, 1);
  tk->fieldorder = cio_read(cio, 1);
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with FIEL Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the ORFO box
*
* Original Format Box
*
*/
void mj2_write_orfo(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_ORFO, 4);
	
  cio_write(cio, tk->or_fieldcount, 1);	/* Original Field count */
  cio_write(cio, tk->or_fieldorder, 1);	/* Original Field order */
	
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the ORFO box
*
* Original Format Box
*
*/
int mj2_read_orfo(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_ORFO != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected ORFO Marker\n");
    return 1;
  }
	
	
  tk->or_fieldcount = cio_read(cio, 1);
  tk->or_fieldorder = cio_read(cio, 1);
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with ORFO Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the JP2P box
*
* MJP2 Profile Box
*
*/
void mj2_write_jp2p(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  int i;
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_JP2P, 4);
	
  cio_write(cio, 0, 4);		/* Version 0, flags =0 */
	
  for (i = 0; i < tk->num_br; i++) {
    cio_write(cio, tk->br[i], 4);
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the JP2P box
*
* MJP2 Profile Box
*
*/
int mj2_read_jp2p(mj2_tk_t * tk, opj_cio_t *cio)
{
  int i;
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_JP2P != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected JP2P Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in JP2P box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in JP2P box. Expected flag 0\n");
    return 1;
  }
	
	
  tk->num_br = (box.length - 12) / 4;
  tk->br = (unsigned int*) opj_malloc(tk->num_br * sizeof(unsigned int));

  for (i = 0; i < tk->num_br; i++) {
    tk->br[i] = cio_read(cio, 4);
  }
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with JP2P Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the JP2X box
*
* MJP2 Prefix Box
*
*/
void mj2_write_jp2x(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  int i;
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_JP2X, 4);
	
  for (i = 0; i < tk->num_jp2x; i++) {
    cio_write(cio, tk->jp2xdata[i], 1);
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the JP2X box
*
* MJP2 Prefix Box
*
*/
int mj2_read_jp2x(mj2_tk_t * tk, opj_cio_t *cio)
{
  unsigned int i;
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_JP2X != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected JP2X Marker\n");
    return 1;
  }
	
	
  tk->num_jp2x = (box.length - 8);
  tk->jp2xdata = (unsigned char*) opj_malloc(tk->num_jp2x * sizeof(unsigned char));

  for (i = 0; i < tk->num_jp2x; i++) {
    tk->jp2xdata[i] = cio_read(cio, 1);
  }
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with JP2X Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the JSUB box
*
* MJP2 Subsampling Box
*
*/
void mj2_write_jsub(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_JSUB, 4);
	
  cio_write(cio, tk->hsub, 1);
  cio_write(cio, tk->vsub, 1);
  cio_write(cio, tk->hoff, 1);
  cio_write(cio, tk->voff, 1);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the JSUB box
*
* MJP2 Subsampling Box
*
*/
int mj2_read_jsub(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_JSUB != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected JSUB Marker\n");
    return 1;
  }
	
  tk->hsub = cio_read(cio, 1);
  tk->vsub = cio_read(cio, 1);
  tk->hoff = cio_read(cio, 1);;
  tk->voff = cio_read(cio, 1);
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with JSUB Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the SMJ2 box
*
* Visual Sample Entry Description
*
*/
void mj2_write_smj2(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_MJ2, 4);	/* MJ2       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, 1, 4);
	
  cio_write(cio, 0, 2);		/* Pre-defined */
	
  cio_write(cio, 0, 2);		/* Reserved */
	
  cio_write(cio, 0, 4);		/* Pre-defined */
  cio_write(cio, 0, 4);		/* Pre-defined */
  cio_write(cio, 0, 4);		/* Pre-defined */
	
  cio_write(cio, tk->w, 2);		/* Width  */
  cio_write(cio, tk->h, 2);		/* Height */
	
  cio_write(cio, tk->horizresolution, 4);	/* Horizontal resolution */
  cio_write(cio, tk->vertresolution, 4);	/* Vertical resolution   */
	
  cio_write(cio, 0, 4);		/* Reserved */
	
  cio_write(cio, 1, 2);		/* Pre-defined = 1 */
	
  cio_write(cio, tk->compressorname[0], 4);	/* Compressor Name */
  cio_write(cio, tk->compressorname[1], 4);
  cio_write(cio, tk->compressorname[2], 4);
  cio_write(cio, tk->compressorname[3], 4);
  cio_write(cio, tk->compressorname[4], 4);
  cio_write(cio, tk->compressorname[5], 4);
  cio_write(cio, tk->compressorname[6], 4);
  cio_write(cio, tk->compressorname[7], 4);
	
  cio_write(cio, tk->depth, 2);	/* Depth */
	
  cio_write(cio, 0xffff, 2);		/* Pre-defined = -1 */
	
  jp2_write_jp2h(&tk->jp2_struct, cio);
	
  mj2_write_fiel(tk, cio);
	
  if (tk->num_br != 0)
    mj2_write_jp2p(tk, cio);
  if (tk->num_jp2x != 0)
    mj2_write_jp2x(tk, cio);
	
  mj2_write_jsub(tk, cio);
  mj2_write_orfo(tk, cio);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the SMJ2 box
*
* Visual Sample Entry Description
*
*/
int mj2_read_smj2(opj_image_t * img, mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
  mj2_box_t box2;
  int i;
  opj_jp2_color_t color;
	
  mj2_read_boxhdr(&box, cio);
	
  if (MJ2_MJ2 != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error in SMJ2 box: Expected MJ2 Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in MJP2 box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in MJP2 box. Expected flag 0\n");
    return 1;
  }
	
  cio_skip(cio,4);
	
  cio_skip(cio,2);			/* Pre-defined */
	
  cio_skip(cio,2);			/* Reserved */
	
  cio_skip(cio,4);			/* Pre-defined */
  cio_skip(cio,4);			/* Pre-defined */
  cio_skip(cio,4);			/* Pre-defined */
	
  tk->w = cio_read(cio, 2);		/* Width  */
  tk->h = cio_read(cio, 2);		/* Height */
	
  tk->horizresolution = cio_read(cio, 4);	/* Horizontal resolution */
  tk->vertresolution = cio_read(cio, 4);	/* Vertical resolution   */
	
  cio_skip(cio,4);			/* Reserved */
	
  cio_skip(cio,2);			/* Pre-defined = 1 */
	
  tk->compressorname[0] = cio_read(cio, 4);	/* Compressor Name */
  tk->compressorname[1] = cio_read(cio, 4);
  tk->compressorname[2] = cio_read(cio, 4);
  tk->compressorname[3] = cio_read(cio, 4);
  tk->compressorname[4] = cio_read(cio, 4);
  tk->compressorname[5] = cio_read(cio, 4);
  tk->compressorname[6] = cio_read(cio, 4);
  tk->compressorname[7] = cio_read(cio, 4);
	
  tk->depth = cio_read(cio, 2);	/* Depth */
	
  /* Init std value */
  tk->num_jp2x = 0;
  tk->fieldcount = 1;
  tk->fieldorder = 0;
  tk->or_fieldcount = 1;
  tk->or_fieldorder = 0;
	
  cio_skip(cio,2);			/* Pre-defined = -1 */
  memset(&color, 0, sizeof(opj_jp2_color_t));
	
  if (!jp2_read_jp2h(&tk->jp2_struct, cio, &color)) {
		opj_event_msg(tk->cinfo, EVT_ERROR, "Error reading JP2H Box\n");
    return 1;
  }

  tk->jp2_struct.comps = (opj_jp2_comps_t*) opj_malloc(tk->jp2_struct.numcomps * sizeof(opj_jp2_comps_t));
  tk->jp2_struct.cl = (unsigned int*) opj_malloc(sizeof(unsigned int));

  tk->num_br = 0;
  tk->num_jp2x = 0;
	
  for (i = 0; cio_tell(cio) - box.init_pos < box.length; i++) {
    mj2_read_boxhdr(&box2, cio);
    cio_seek(cio, box2.init_pos);
    switch (box2.type) {
    case MJ2_FIEL:
      if (mj2_read_fiel(tk, cio))
				return 1;
      break;
			
    case MJ2_JP2P:
      if (mj2_read_jp2p(tk, cio))
				return 1;
      break;
			
    case MJ2_JP2X:
      if (mj2_read_jp2x(tk, cio))
				return 1;
      break;
			
    case MJ2_JSUB:
      if (mj2_read_jsub(tk, cio))
				return 1;
      break;
			
    case MJ2_ORFO:
      if (mj2_read_orfo(tk, cio))
				return 1;
      break;
			
    default:
      opj_event_msg(cio->cinfo, EVT_ERROR, "Error with MJP2 Box size\n");
      return 1;
      break;
			
    }
  }
  return 0;
}


/*
* Write the STSD box
*
* Sample Description
*
*/
void mj2_write_stsd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_STSD, 4);	/* STSD       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, 1, 4);		/* entry_count = 1 (considering same JP2 headerboxes) */
	
  if (tk->track_type == 0) {
    mj2_write_smj2(tk, cio);
  } else if (tk->track_type == 1) {
    // Not implemented
  }
  if (tk->track_type == 2) {
    // Not implemented
  }
	
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the STSD box
*
* Sample Description
*
*/
int mj2_read_stsd(mj2_tk_t * tk, opj_image_t * img, opj_cio_t *cio)
{
  int i;
  int entry_count, len_2skip;
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
	
  if (MJ2_STSD != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected STSD Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in STSD box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in STSD box. Expected flag 0\n");
    return 1;
  }
	
  entry_count = cio_read(cio, 4);
	
  if (tk->track_type == 0) {
    for (i = 0; i < entry_count; i++) {
      if (mj2_read_smj2(img, tk, cio))
				return 1;
    }
  } else if (tk->track_type == 1) {
    len_2skip = cio_read(cio, 4);	// Not implemented -> skipping box
    cio_skip(cio,len_2skip - 4);
  } else if (tk->track_type == 2) {
    len_2skip = cio_read(cio, 4);	// Not implemented -> skipping box
    cio_skip(cio,len_2skip - 4);
  }
	
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with STSD Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the STBL box
*
* Sample table box box
*
*/
void mj2_write_stbl(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_STBL, 4);	/* STBL       */
	
  mj2_write_stsd(tk, cio);
  mj2_write_stts(tk, cio);
  mj2_write_stsc(tk, cio);
  mj2_write_stsz(tk, cio);
  mj2_write_stco(tk, cio);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the STBL box
*
* Sample table box box
*
*/
int mj2_read_stbl(mj2_tk_t * tk, opj_image_t * img, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_STBL != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected STBL Marker\n");
    return 1;
  }
	
  if (mj2_read_stsd(tk, img, cio))
    return 1;
  if (mj2_read_stts(tk, cio))
    return 1;
  if (mj2_read_stsc(tk, cio))
    return 1;
  if (mj2_read_stsz(tk, cio))
    return 1;
  if (mj2_read_stco(tk, cio))
    return 1;
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with STBL Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the URL box
*
* URL box
*
*/
void mj2_write_url(mj2_tk_t * tk, int url_num, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_URL, 4);	/* URL       */
	
  if (url_num == 0)
    cio_write(cio, 1, 4);		/* Version = 0, flags = 1 because stored in same file */
  else {
    cio_write(cio, 0, 4);		/* Version = 0, flags =  0 */
    cio_write(cio, tk->url[url_num - 1].location[0], 4);
    cio_write(cio, tk->url[url_num - 1].location[1], 4);
    cio_write(cio, tk->url[url_num - 1].location[2], 4);
    cio_write(cio, tk->url[url_num - 1].location[3], 4);
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the URL box
*
* URL box
*
*/
int mj2_read_url(mj2_tk_t * tk, int urn_num, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_URL != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected URL Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in URL box\n");
    return 1;
  }
	
  if (1 != cio_read(cio, 3)) {	/* If flags = 1 --> media data in file */
    tk->url[urn_num].location[0] = cio_read(cio, 4);
    tk->url[urn_num].location[1] = cio_read(cio, 4);
    tk->url[urn_num].location[2] = cio_read(cio, 4);
    tk->url[urn_num].location[3] = cio_read(cio, 4);
  } else {
    tk->num_url--;
  }
	
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with URL Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the URN box
*
* URN box
*
*/
void mj2_write_urn(mj2_tk_t * tk, int urn_num, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_URN, 4);	/* URN       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags =  0 */
	
  cio_write(cio, tk->urn[urn_num].name[0], 4);
  cio_write(cio, tk->urn[urn_num].name[1], 4);
  cio_write(cio, tk->urn[urn_num].name[2], 4);
  cio_write(cio, tk->urn[urn_num].name[3], 4);
  cio_write(cio, tk->urn[urn_num].location[0], 4);
  cio_write(cio, tk->urn[urn_num].location[1], 4);
  cio_write(cio, tk->urn[urn_num].location[2], 4);
  cio_write(cio, tk->urn[urn_num].location[3], 4);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the URN box
*
* URN box
*
*/
int mj2_read_urn(mj2_tk_t * tk, int urn_num, opj_cio_t *cio)
{
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_URN != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected URN Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in URN box\n");
    return 1;
  }
	
  if (1 != cio_read(cio, 3)) {	/* If flags = 1 --> media data in file */
    tk->urn[urn_num].name[0] = cio_read(cio, 4);
    tk->urn[urn_num].name[1] = cio_read(cio, 4);
    tk->urn[urn_num].name[2] = cio_read(cio, 4);
    tk->urn[urn_num].name[3] = cio_read(cio, 4);
    tk->urn[urn_num].location[0] = cio_read(cio, 4);
    tk->urn[urn_num].location[1] = cio_read(cio, 4);
    tk->urn[urn_num].location[2] = cio_read(cio, 4);
    tk->urn[urn_num].location[3] = cio_read(cio, 4);
  }
	
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with URN Box size\n");
    return 1;
  }
  return 0;
}


/*
* Write the DREF box
*
* Data reference box
*
*/
void mj2_write_dref(mj2_tk_t * tk, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_DREF, 4);	/* DREF       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  if (tk->num_url + tk->num_urn == 0) {	/* Media data in same file */
    cio_write(cio, 1, 4);		/* entry_count = 1 */
    mj2_write_url(tk, 0, cio);
  } else {
    cio_write(cio, tk->num_url + tk->num_urn, 4);	/* entry_count */
		
    for (i = 0; i < tk->num_url; i++)
      mj2_write_url(tk, i + 1, cio);
		
    for (i = 0; i < tk->num_urn; i++)
      mj2_write_urn(tk, i, cio);
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the DREF box
*
* Data reference box
*
*/
int mj2_read_dref(mj2_tk_t * tk, opj_cio_t *cio)
{
	
  int i;
  int entry_count, marker;
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_DREF != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected DREF Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in DREF box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in DREF box. Expected flag 0\n");
    return 1;
  }
	
  entry_count = cio_read(cio, 4);
  tk->num_url = 0;
  tk->num_urn = 0;
	
  for (i = 0; i < entry_count; i++) {
    cio_skip(cio,4);
    marker = cio_read(cio, 4);
    if (marker == MJ2_URL) {
      cio_skip(cio,-8);
      tk->num_url++;
      if (mj2_read_url(tk, tk->num_url, cio))
				return 1;
    } else if (marker == MJ2_URN) {
      cio_skip(cio,-8);
      tk->num_urn++;
      if (mj2_read_urn(tk, tk->num_urn, cio))
				return 1;
    } else {
      opj_event_msg(cio->cinfo, EVT_ERROR, "Error with in DREF box. Expected URN or URL box\n");
      return 1;
    }
		
  }
	
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with DREF Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the DINF box
*
* Data information box
*
*/
void mj2_write_dinf(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_DINF, 4);	/* DINF       */
	
  mj2_write_dref(tk, cio);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the DINF box
*
* Data information box
*
*/
int mj2_read_dinf(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_DINF != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected DINF Marker\n");
    return 1;
  }
	
  if (mj2_read_dref(tk, cio))
    return 1;
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with DINF Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the VMHD box
*
* Video Media information box
*
*/
void mj2_write_vmhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_VMHD, 4);	/* VMHD       */
	
  cio_write(cio, 1, 4);		/* Version = 0, flags = 1 */
	
  cio_write(cio, tk->graphicsmode, 2);
  cio_write(cio, tk->opcolor[0], 2);
  cio_write(cio, tk->opcolor[1], 2);
  cio_write(cio, tk->opcolor[2], 2);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the VMHD box
*
* Video Media information box
*
*/
int mj2_read_vmhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_VMHD != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected VMHD Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in VMHD box\n");
    return 1;
  }
	
  if (1 != cio_read(cio, 3)) {	/* Flags = 1  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in VMHD box. Expected flag 1\n");
    return 1;
  }
	
  tk->track_type = 0;
  tk->graphicsmode = cio_read(cio, 2);
  tk->opcolor[0] = cio_read(cio, 2);
  tk->opcolor[1] = cio_read(cio, 2);
  tk->opcolor[2] = cio_read(cio, 2);
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with VMHD Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the SMHD box
*
* Sound Media information box
*
*/
void mj2_write_smhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_SMHD, 4);	/* SMHD       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, tk->balance, 2);
	
  cio_write(cio, 0, 2);		/* Reserved */
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the SMHD box
*
* Sound Media information box
*
*/
int mj2_read_smhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_SMHD != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected SMHD Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in SMHD box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in SMHD box. Expected flag 0\n");
    return 1;
  }
	
  tk->track_type = 1;
  tk->balance = cio_read(cio, 2);
	
  /* Init variables to zero to avoid problems when freeeing memory
  The values will possibly be overidded when decoding the track structure */
  tk->num_br = 0;
  tk->num_url = 0;
  tk->num_urn = 0;
  tk->num_chunks = 0;
  tk->num_tts = 0;
  tk->num_samplestochunk = 0;
  tk->num_samples = 0;
	
  cio_skip(cio,2);			/* Reserved */
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with SMHD Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the HMHD box
*
* Hint Media information box
*
*/
void mj2_write_hmhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_HMHD, 4);	/* HMHD       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, tk->maxPDUsize, 2);
  cio_write(cio, tk->avgPDUsize, 2);
  cio_write(cio, tk->maxbitrate, 4);
  cio_write(cio, tk->avgbitrate, 4);
  cio_write(cio, tk->slidingavgbitrate, 4);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the HMHD box
*
* Hint Media information box
*
*/
int mj2_read_hmhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_HMHD != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected HMHD Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in HMHD box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in HMHD box. Expected flag 0\n");
    return 1;
  }
	
  tk->track_type = 2;
  tk->maxPDUsize = cio_read(cio, 2);
  tk->avgPDUsize = cio_read(cio, 2);
  tk->maxbitrate = cio_read(cio, 4);
  tk->avgbitrate = cio_read(cio, 4);
  tk->slidingavgbitrate = cio_read(cio, 4);
	
  /* Init variables to zero to avoid problems when freeeing memory
  The values will possibly be overidded when decoding the track structure */
  tk->num_br = 0;
  tk->num_url = 0;
  tk->num_urn = 0;
  tk->num_chunks = 0;
  tk->num_tts = 0;
  tk->num_samplestochunk = 0;
  tk->num_samples = 0;
	
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with HMHD Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the MINF box
*
* Media information box
*
*/
void mj2_write_minf(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_MINF, 4);	/* MINF       */
	
  if (tk->track_type == 0) {
    mj2_write_vmhd(tk, cio);
  } else if (tk->track_type == 1) {
    mj2_write_smhd(tk, cio);
  } else if (tk->track_type == 2) {
    mj2_write_hmhd(tk, cio);
  }
	
  mj2_write_dinf(tk, cio);
  mj2_write_stbl(tk, cio);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the MINF box
*
* Media information box
*
*/
int mj2_read_minf(mj2_tk_t * tk, opj_image_t * img, opj_cio_t *cio)
{
	
  unsigned int box_type;
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_MINF != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected MINF Marker\n");
    return 1;
  }
	
  cio_skip(cio,4);
  box_type = cio_read(cio, 4);
  cio_skip(cio,-8);
	
  if (box_type == MJ2_VMHD) {
    if (mj2_read_vmhd(tk, cio))
      return 1;
  } else if (box_type == MJ2_SMHD) {
    if (mj2_read_smhd(tk, cio))
      return 1;
  } else if (box_type == MJ2_HMHD) {
    if (mj2_read_hmhd(tk, cio))
      return 1;
  } else {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error in MINF box expected vmhd, smhd or hmhd\n");
    return 1;
  }
	
  if (mj2_read_dinf(tk, cio))
    return 1;
	
  if (mj2_read_stbl(tk, img, cio))
    return 1;
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with MINF Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the HDLR box
*
* Handler reference box
*
*/
void mj2_write_hdlr(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_HDLR, 4);	/* HDLR       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, 0, 4);		/* Predefine */
	
  tk->name = 0;			/* The track name is immediately determined by the track type */
	
  if (tk->track_type == 0) {
    tk->handler_type = 0x76696465;	/* Handler type: vide */
    cio_write(cio, tk->handler_type, 4);
		
    cio_write(cio, 0, 4);
    cio_write(cio, 0, 4);
    cio_write(cio, 0, 4);		/* Reserved */
		
    cio_write(cio, 0x76696465, 4);
    cio_write(cio, 0x6F206d65, 4);
    cio_write(cio, 0x64696120, 4);
    cio_write(cio, 0x74726163, 4);
    cio_write(cio, 0x6b00, 2);	/* String: video media track */
  } else if (tk->track_type == 1) {
    tk->handler_type = 0x736F756E;	/* Handler type: soun */
    cio_write(cio, tk->handler_type, 4);
		
    cio_write(cio, 0, 4);
    cio_write(cio, 0, 4);
    cio_write(cio, 0, 4);		/* Reserved */
		
    cio_write(cio, 0x536F756E, 4);
    cio_write(cio, 0x6400, 2);	/* String: Sound */
  } else if (tk->track_type == 2) {
    tk->handler_type = 0x68696E74;	/* Handler type: hint */
    cio_write(cio, tk->handler_type, 4);
		
    cio_write(cio, 0, 4);
    cio_write(cio, 0, 4);
    cio_write(cio, 0, 4);		/* Reserved */
		
    cio_write(cio, 0x48696E74, 4);
    cio_write(cio, 0, 2);		/* String: Hint */
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the HDLR box
*
* Handler reference box
*
*/
int mj2_read_hdlr(mj2_tk_t * tk, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_HDLR != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected HDLR Marker\n");
    return 1;
  }
	
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in HDLR box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0  */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in HDLR box. Expected flag 0\n");
    return 1;
  }
	
  cio_skip(cio,4);			/* Reserved */
	
  tk->handler_type = cio_read(cio, 4);
  cio_skip(cio,12);			/* Reserved */
	
  tk->name_size = box.length - 32;

  tk->name = (char*) opj_malloc(tk->name_size * sizeof(char));
  for (i = 0; i < tk->name_size; i++) {
    tk->name[i] = cio_read(cio, 1);	/* Name */
  }
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with HDLR Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the MDHD box
*
* Media Header Box
*
*/
void mj2_write_mdhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
  unsigned int i;
  time_t ltime;
  unsigned int modification_time;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_MDHD, 4);	/* MDHD       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  cio_write(cio, tk->creation_time, 4);	/* Creation Time */
	
  time(&ltime);			/* Time since 1/1/70 */
  modification_time = (unsigned int)ltime + 2082844800;	/* Seoonds between 1/1/04 and 1/1/70 */
	
  cio_write(cio, modification_time, 4);	/* Modification Time */
	
  cio_write(cio, tk->timescale, 4);	/* Timescale */
	
  tk->duration = 0;
	
  for (i = 0; i < tk->num_samples; i++)
    tk->duration += tk->sample[i].sample_delta;
	
  cio_write(cio, tk->duration, 4);	/* Duration */
	
  cio_write(cio, tk->language, 2);	/* Language */
	
  cio_write(cio, 0, 2);		/* Predefined */
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the MDHD box
*
* Media Header Box
*
*/
int mj2_read_mdhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (!(MJ2_MHDR == box.type || MJ2_MDHD == box.type)) {	// Kakadu writes MHDR instead of MDHD
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected MDHD Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in MDHD box\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 3)) {	/* Flags = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with flag in MDHD box. Expected flag 0\n");
    return 1;
  }
	
	
  tk->creation_time = cio_read(cio, 4);	/* Creation Time */
	
  tk->modification_time = cio_read(cio, 4);	/* Modification Time */
	
  tk->timescale = cio_read(cio, 4);	/* Timescale */
	
  tk->duration = cio_read(cio, 4);	/* Duration */
	
  tk->language = cio_read(cio, 2);	/* Language */
	
  cio_skip(cio,2);			/* Predefined */
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with MDHD Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the MDIA box
*
* Media box
*
*/
void mj2_write_mdia(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_MDIA, 4);	/* MDIA       */
	
  mj2_write_mdhd(tk, cio);
  mj2_write_hdlr(tk, cio);
  mj2_write_minf(tk, cio);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the MDIA box
*
* Media box
*
*/
int mj2_read_mdia(mj2_tk_t * tk, opj_image_t * img, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_MDIA != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected MDIA Marker\n");
    return 1;
  }
	
  if (mj2_read_mdhd(tk, cio))
    return 1;
  if (mj2_read_hdlr(tk, cio))
    return 1;
  if (mj2_read_minf(tk, img, cio))
    return 1;
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with MDIA Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the TKHD box
*
* Track Header box
*
*/
void mj2_write_tkhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
  unsigned int i;
  time_t ltime;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
	
  cio_write(cio, MJ2_TKHD, 4);	/* TKHD       */
	
  cio_write(cio, 3, 4);		/* Version=0, flags=3 */
	
  time(&ltime);			/* Time since 1/1/70 */
  tk->modification_time = (unsigned int)ltime + 2082844800;	/* Seoonds between 1/1/04 and 1/1/70 */
	
  cio_write(cio, tk->creation_time, 4);	/* Creation Time */
	
  cio_write(cio, tk->modification_time, 4);	/* Modification Time */
	
  cio_write(cio, tk->track_ID, 4);	/* Track ID */
	
  cio_write(cio, 0, 4);		/* Reserved */
	
  tk->duration = 0;
	
  for (i = 0; i < tk->num_samples; i++)
    tk->duration += tk->sample[i].sample_delta;
	
  cio_write(cio, tk->duration, 4);	/* Duration */
	
  cio_write(cio, 0, 4);		/* Reserved */
  cio_write(cio, 0, 4);		/* Reserved */
	
  cio_write(cio, tk->layer, 2);	/* Layer    */
	
  cio_write(cio, 0, 2);		/* Predefined */
	
  cio_write(cio, tk->volume, 2);	/* Volume       */
	
  cio_write(cio, 0, 2);		/* Reserved */
	
  cio_write(cio, tk->trans_matrix[0], 4);	/* Transformation matrix for track */
  cio_write(cio, tk->trans_matrix[1], 4);
  cio_write(cio, tk->trans_matrix[2], 4);
  cio_write(cio, tk->trans_matrix[3], 4);
  cio_write(cio, tk->trans_matrix[4], 4);
  cio_write(cio, tk->trans_matrix[5], 4);
  cio_write(cio, tk->trans_matrix[6], 4);
  cio_write(cio, tk->trans_matrix[7], 4);
  cio_write(cio, tk->trans_matrix[8], 4);
	
  cio_write(cio, tk->visual_w, 4);	/* Video Visual Width  */
	
  cio_write(cio, tk->visual_h, 4);	/* Video Visual Height */
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the TKHD box
*
* Track Header box
*
*/
int mj2_read_tkhd(mj2_tk_t * tk, opj_cio_t *cio)
{
  int flag;
	
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
	
  if (MJ2_TKHD != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected TKHD Marker\n");
    return 1;
  }
	
  if (0 != cio_read(cio, 1)) {	/* Version = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in TKHD box\n");
    return 1;
  }
	
  flag = cio_read(cio, 3);
	
  if (!(flag == 1 || flag == 2 || flag == 3 || flag == 4)) {	/* Flags = 1,2,3 or 4 */
    opj_event_msg(cio->cinfo, EVT_ERROR,
			"Error with flag in TKHD box: Expected flag 1,2,3 or 4\n");
    return 1;
  }
	
  tk->creation_time = cio_read(cio, 4);	/* Creation Time */
	
  tk->modification_time = cio_read(cio, 4);	/* Modification Time */
	
  tk->track_ID = cio_read(cio, 4);	/* Track ID */
	
  cio_skip(cio,4);			/* Reserved */
	
  tk->duration = cio_read(cio, 4);	/* Duration */
	
  cio_skip(cio,8);			/* Reserved */
	
  tk->layer = cio_read(cio, 2);	/* Layer    */
	
  cio_read(cio, 2);			/* Predefined */
	
  tk->volume = cio_read(cio, 2);	/* Volume       */
	
  cio_skip(cio,2);			/* Reserved */
	
  tk->trans_matrix[0] = cio_read(cio, 4);	/* Transformation matrix for track */
  tk->trans_matrix[1] = cio_read(cio, 4);
  tk->trans_matrix[2] = cio_read(cio, 4);
  tk->trans_matrix[3] = cio_read(cio, 4);
  tk->trans_matrix[4] = cio_read(cio, 4);
  tk->trans_matrix[5] = cio_read(cio, 4);
  tk->trans_matrix[6] = cio_read(cio, 4);
  tk->trans_matrix[7] = cio_read(cio, 4);
  tk->trans_matrix[8] = cio_read(cio, 4);
	
  tk->visual_w = cio_read(cio, 4);	/* Video Visual Width  */
	
  tk->visual_h = cio_read(cio, 4);	/* Video Visual Height */
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with TKHD Box size\n");
    return 1;
  }
  return 0;
}

/*
* Write the TRAK box
*
* Track box
*
*/
void mj2_write_trak(mj2_tk_t * tk, opj_cio_t *cio)
{
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
	
  cio_write(cio, MJ2_TRAK, 4);	/* TRAK       */
	
  mj2_write_tkhd(tk, cio);
  mj2_write_mdia(tk, cio);
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the TRAK box
*
* Track box
*
*/
int mj2_read_trak(mj2_tk_t * tk, opj_image_t * img, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_TRAK != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected TRAK Marker\n");
    return 1;
  }
  if (mj2_read_tkhd(tk, cio))
    return 1;
  if (mj2_read_mdia(tk, img, cio))
    return 1;
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with TRAK Box\n");
    return 1;
  }
  return 0;
}

/*
* Write the MVHD box
*
* Movie header Box
*
*/
void mj2_write_mvhd(opj_mj2_t * movie, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
  unsigned j;
  time_t ltime;
  int max_tk_num = 0;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_MVHD, 4);	/* MVHD       */
	
  cio_write(cio, 0, 4);		/* Version = 0, flags = 0 */
	
  time(&ltime);			/* Time since 1/1/70 */
  movie->modification_time = (unsigned int)ltime + 2082844800;	/* Seoonds between 1/1/04 and 1/1/70 */
	
  cio_write(cio, movie->creation_time, 4);	/* Creation Time */
	
  cio_write(cio, movie->modification_time, 4);	/* Modification Time */
	
  cio_write(cio, movie->timescale, 4);	/* Timescale */
	
  movie->duration = 0;
	
  for (i = 0; i < (movie->num_stk + movie->num_htk + movie->num_vtk); i++) {
    mj2_tk_t *tk = &movie->tk[i];
		
    for (j = 0; j < tk->num_samples; j++) {
      movie->duration += tk->sample[j].sample_delta;
    }
  }
	
  cio_write(cio, movie->duration, 4);
	
  cio_write(cio, movie->rate, 4);	/* Rate to play presentation    */
	
  cio_write(cio, movie->volume, 2);	/* Volume       */
	
  cio_write(cio, 0, 2);		/* Reserved */
  cio_write(cio, 0, 4);		/* Reserved */
  cio_write(cio, 0, 4);		/* Reserved */
	
  cio_write(cio, movie->trans_matrix[0], 4);	/* Transformation matrix for video */
  cio_write(cio, movie->trans_matrix[1], 4);
  cio_write(cio, movie->trans_matrix[2], 4);
  cio_write(cio, movie->trans_matrix[3], 4);
  cio_write(cio, movie->trans_matrix[4], 4);
  cio_write(cio, movie->trans_matrix[5], 4);
  cio_write(cio, movie->trans_matrix[6], 4);
  cio_write(cio, movie->trans_matrix[7], 4);
  cio_write(cio, movie->trans_matrix[8], 4);
	
  cio_write(cio, 0, 4);		/* Pre-defined */
  cio_write(cio, 0, 4);		/* Pre-defined */
  cio_write(cio, 0, 4);		/* Pre-defined */
  cio_write(cio, 0, 4);		/* Pre-defined */
  cio_write(cio, 0, 4);		/* Pre-defined */
  cio_write(cio, 0, 4);		/* Pre-defined */
	
	
  for (i = 0; i < movie->num_htk + movie->num_stk + movie->num_vtk; i++) {
    if (max_tk_num < movie->tk[i].track_ID)
      max_tk_num = movie->tk[i].track_ID;
  }
	
  movie->next_tk_id = max_tk_num + 1;
	
  cio_write(cio, movie->next_tk_id, 4);	/* ID of Next track to be added */
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);
}

/*
* Read the MVHD box
*
* Movie header Box
*
*/
int mj2_read_mvhd(opj_mj2_t * movie, opj_cio_t *cio)
{
  mj2_box_t box;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_MVHD != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected MVHD Marker\n");
    return 1;
  }
	
	
  if (0 != cio_read(cio, 4)) {	/* Version = 0, flags = 0 */
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Only Version 0 handled in MVHD box\n");
  }
	
  movie->creation_time = cio_read(cio, 4);	/* Creation Time */
	
  movie->modification_time = cio_read(cio, 4);	/* Modification Time */
	
  movie->timescale = cio_read(cio, 4);	/* Timescale */
	
  movie->duration = cio_read(cio, 4);	/* Duration */
	
  movie->rate = cio_read(cio, 4);		/* Rate to play presentation    */
	
  movie->volume = cio_read(cio, 2);		/* Volume       */
	
  cio_skip(cio,10);				/* Reserved */
	
  movie->trans_matrix[0] = cio_read(cio, 4);	/* Transformation matrix for video */
  movie->trans_matrix[1] = cio_read(cio, 4);
  movie->trans_matrix[2] = cio_read(cio, 4);
  movie->trans_matrix[3] = cio_read(cio, 4);
  movie->trans_matrix[4] = cio_read(cio, 4);
  movie->trans_matrix[5] = cio_read(cio, 4);
  movie->trans_matrix[6] = cio_read(cio, 4);
  movie->trans_matrix[7] = cio_read(cio, 4);
  movie->trans_matrix[8] = cio_read(cio, 4);
	
  cio_skip(cio,24);			/* Pre-defined */
	
  movie->next_tk_id = cio_read(cio, 4);	/* ID of Next track to be added */
	
  if (cio_tell(cio) - box.init_pos != box.length) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error with MVHD Box Size\n");
    return 1;
  }
  return 0;
}


/*
* Write the MOOV box
*
* Movie Box
*
*/
void mj2_write_moov(opj_mj2_t * movie, opj_cio_t *cio)
{
  int i;
  mj2_box_t box;
	
  box.init_pos = cio_tell(cio);
  cio_skip(cio,4);
  cio_write(cio, MJ2_MOOV, 4);	/* MOOV       */
	
  mj2_write_mvhd(movie, cio);
	
  for (i = 0; i < (movie->num_stk + movie->num_htk + movie->num_vtk); i++) {
    mj2_write_trak(&movie->tk[i], cio);
  }
	
  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);	/* L          */
  cio_seek(cio, box.init_pos + box.length);	
}

/*
* Read the MOOV box
*
* Movie Box
*
*/
int mj2_read_moov(opj_mj2_t * movie, opj_image_t * img, opj_cio_t *cio)
{
  unsigned int i;
  mj2_box_t box;
  mj2_box_t box2;
	
  mj2_read_boxhdr(&box, cio);
  if (MJ2_MOOV != box.type) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "Error: Expected MOOV Marker\n");
    return 1;
  }
	
  if (mj2_read_mvhd(movie, cio))
    return 1;

  movie->tk = (mj2_tk_t*) opj_malloc((movie->next_tk_id - 1) * sizeof(mj2_tk_t));

  for (i = 0; cio_tell(cio) - box.init_pos < box.length; i++) {
		mj2_tk_t *tk = &movie->tk[i];
		tk->cinfo = movie->cinfo;
    mj2_read_boxhdr(&box2, cio);
    if (box2.type == MJ2_TRAK) {
      cio_seek(cio, box2.init_pos);
      if (mj2_read_trak(tk, img, cio))
				return 1;
			
      if (tk->track_type == 0) {
				movie->num_vtk++;
      } else if (tk->track_type == 1) {
				movie->num_stk++;
      } else if (tk->track_type == 2) {
				movie->num_htk++;
      }
    } else if (box2.type == MJ2_MVEX) {
      cio_seek(cio, box2.init_pos);
      cio_skip(cio,box2.length);
      i--;
    } else {
      opj_event_msg(cio->cinfo, EVT_ERROR, "Error with MOOV Box: Expected TRAK or MVEX box\n");
      return 1;
    }
  }
  return 0;
}

int mj2_read_struct(FILE *file, opj_mj2_t *movie) {
  mj2_box_t box;
  opj_image_t img;
  unsigned char * src;
  int fsresult;
  int foffset;
	opj_cio_t *cio;
	
	/* open a byte stream for reading */	
	src = (unsigned char*) opj_malloc(300 * sizeof(unsigned char));	

	/* Assuming that jp and ftyp markers size do
     not exceed 300 bytes */
  fread(src,300,1, file);  
  
  cio = opj_cio_open((opj_common_ptr)movie->cinfo, src, 300);
  
  if (mj2_read_jp(cio))
    return 1;
  if (mj2_read_ftyp(movie, cio))
    return 1;
	
  fsresult = fseek(file,cio_tell(cio),SEEK_SET);
  if( fsresult ) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "End of file reached while trying to read data after FTYP box\n" );
    return 1;
  }
	
  foffset = cio_tell(cio);
  
  box.type = 0;
  
  fread(src,30,1,file);
  cio = opj_cio_open((opj_common_ptr)movie->cinfo, src, 300);
  mj2_read_boxhdr(&box, cio);
  
  while(box.type != MJ2_MOOV) {
    
    switch(box.type)
    {
    case MJ2_MDAT:
      fsresult = fseek(file,foffset+box.length,SEEK_SET);
      if( fsresult ) {
				opj_event_msg(cio->cinfo, EVT_ERROR, "End of file reached while trying to read MDAT box\n" );
				return 1;
      }
      foffset += box.length;
      break;
      
    case MJ2_MOOF:
      fsresult = fseek(file,foffset+box.length,SEEK_SET);
      if( fsresult ) {
				opj_event_msg(cio->cinfo, EVT_ERROR, "End of file reached while trying to read MOOF box\n" );
				return 1;
      }
      foffset += box.length;
      break;      
    case MJ2_FREE:
      fsresult = fseek(file,foffset+box.length,SEEK_SET);
      if( fsresult ) {
				opj_event_msg(cio->cinfo, EVT_ERROR, "End of file reached while trying to read FREE box\n" );
				return 1;
      }
      foffset += box.length;
      break;      
    case MJ2_SKIP:
      fsresult = fseek(file,foffset+box.length,SEEK_SET);
      if( fsresult ) {
				opj_event_msg(cio->cinfo, EVT_ERROR, "End of file reached while trying to read SKIP box\n" );
				return 1;
      }
      foffset += box.length;
      break;      
    default:
      opj_event_msg(cio->cinfo, EVT_ERROR, "Unknown box in MJ2 stream\n");
      fsresult = fseek(file,foffset+box.length,SEEK_SET);
      if( fsresult ) {
				opj_event_msg(cio->cinfo, EVT_ERROR, "End of file reached while trying to read end of unknown box\n"); 
				return 1;
      }      
      foffset += box.length;
      break;
    }
    fsresult = fread(src,8,1,file);
    if (fsresult != 1) {
      opj_event_msg(cio->cinfo, EVT_ERROR, "MOOV box not found in file\n"); 
      return 1;
    }
		cio = opj_cio_open((opj_common_ptr)movie->cinfo, src, 8);    		
    mj2_read_boxhdr(&box, cio);
  }	

  fseek(file,foffset,SEEK_SET);
  src = (unsigned char*)opj_realloc(src,box.length);
  fsresult = fread(src,box.length,1,file);
  if (fsresult != 1) {
    opj_event_msg(cio->cinfo, EVT_ERROR, "End of file reached while trying to read MOOV box\n"); 
    return 1;
  }
	
	cio = opj_cio_open((opj_common_ptr)movie->cinfo, src, box.length);
  
  if (mj2_read_moov(movie, &img, cio))
    return 1;

  opj_free(src);
  return 0;
}

/* ----------------------------------------------------------------------- */
/* MJ2 decoder interface															                     */
/* ----------------------------------------------------------------------- */

opj_dinfo_t* mj2_create_decompress() {
	opj_mj2_t* mj2;
	opj_dinfo_t *dinfo = (opj_dinfo_t*) opj_calloc(1, sizeof(opj_dinfo_t));
	if(!dinfo) return NULL;

	dinfo->is_decompressor = true;	

	mj2 = (opj_mj2_t*) opj_calloc(1, sizeof(opj_mj2_t));
	dinfo->mj2_handle = mj2;
	if(mj2) {
		mj2->cinfo = (opj_common_ptr)dinfo;
	}
	mj2->j2k = j2k_create_decompress((opj_common_ptr)dinfo);
	dinfo->j2k_handle = mj2->j2k;

	return dinfo;
}

void mj2_setup_decoder(opj_mj2_t *movie, mj2_dparameters_t *mj2_parameters) {
	movie->num_vtk=0;
  movie->num_stk=0;
  movie->num_htk=0;	

	/* setup the J2K decoder parameters */
	j2k_setup_decoder((opj_j2k_t*)movie->cinfo->j2k_handle, 
		&mj2_parameters->j2k_parameters);

}

void mj2_destroy_decompress(opj_mj2_t *movie) {
	if(movie) {
		int i;
		mj2_tk_t *tk=NULL;

		if (movie->cinfo->j2k_handle) 
			j2k_destroy_compress(movie->j2k);
		
		if (movie->num_cl != 0)
			opj_free(movie->cl);
		
		for (i = 0; i < movie->num_vtk + movie->num_stk + movie->num_htk; i++) {
			tk = &movie->tk[i];
			if (tk->name_size != 0)
				opj_free(tk->name);
			if (tk->track_type == 0)  {// Video track
				if (tk->jp2_struct.comps != 0)
					opj_free(tk->jp2_struct.comps);
				if (tk->jp2_struct.cl != 0)
					opj_free(tk->jp2_struct.cl);
				if (tk->num_jp2x != 0)
					opj_free(tk->jp2xdata);
				
			}
			if (tk->num_url != 0)
				opj_free(tk->url);
			if (tk->num_urn != 0)
				opj_free(tk->urn);
			if (tk->num_br != 0)
				opj_free(tk->br);
			if (tk->num_tts != 0)
				opj_free(tk->tts);
			if (tk->num_chunks != 0)
				opj_free(tk->chunk);
			if (tk->num_samplestochunk != 0)
				opj_free(tk->sampletochunk);
			if (tk->num_samples != 0)
				opj_free(tk->sample);
		}
		
		opj_free(movie->tk);
	}	
	opj_free(movie);
}

/* ----------------------------------------------------------------------- */
/* MJ2 encoder interface															                     */
/* ----------------------------------------------------------------------- */


opj_cinfo_t* mj2_create_compress() {
	opj_mj2_t* mj2;
	opj_cinfo_t *cinfo = (opj_cinfo_t*) opj_calloc(1, sizeof(opj_cinfo_t));
	if(!cinfo) return NULL;

	mj2 = (opj_mj2_t*) opj_calloc(1, sizeof(opj_mj2_t));
	cinfo->mj2_handle = mj2;
	if(mj2) {
		mj2->cinfo = (opj_common_ptr)cinfo;
	}

	mj2->j2k = j2k_create_compress(mj2->cinfo);
	cinfo->j2k_handle = mj2->j2k;

	return cinfo;
}

void mj2_setup_encoder(opj_mj2_t *movie, mj2_cparameters_t *parameters) {
	if(movie && parameters) {
		opj_jp2_t *jp2_struct;
			
		movie->num_htk = 0;	  // No hint tracks
		movie->num_stk = 0;	  // No sound tracks
		movie->num_vtk = 1;	  // One video track  

		movie->brand = MJ2_MJ2;  // One brand: MJ2
		movie->num_cl = 2;	  // Two compatible brands: MJ2 and MJ2S
		movie->cl = (unsigned int*) opj_malloc(movie->num_cl * sizeof(unsigned int));
		movie->cl[0] = MJ2_MJ2;
		movie->cl[1] = MJ2_MJ2S;
		movie->minversion = 0;	  // Minimum version: 0		

		movie->tk = (mj2_tk_t*) opj_malloc(sizeof(mj2_tk_t)); //Memory allocation for the video track
		movie->tk[0].track_ID = 1;	  // Track ID = 1 
		movie->tk[0].track_type = 0;	  // Video track
		movie->tk[0].Dim[0] = parameters->Dim[0];
		movie->tk[0].Dim[1] = parameters->Dim[1];
		movie->tk[0].w = parameters->w;
		movie->tk[0].h = parameters->h;
		movie->tk[0].CbCr_subsampling_dx = parameters->CbCr_subsampling_dx;
		movie->tk[0].CbCr_subsampling_dy = parameters->CbCr_subsampling_dy;
		movie->tk[0].sample_rate = parameters->frame_rate;
		movie->tk[0].name_size = 0;
		movie->tk[0].chunk = (mj2_chunk_t*) opj_malloc(sizeof(mj2_chunk_t));  
		movie->tk[0].sample = (mj2_sample_t*) opj_malloc(sizeof(mj2_sample_t));

		jp2_struct = &movie->tk[0].jp2_struct;
		jp2_struct->numcomps = 3;	// NC  		
		jp2_struct->comps = (opj_jp2_comps_t*) opj_malloc(jp2_struct->numcomps * sizeof(opj_jp2_comps_t));
		jp2_struct->precedence = 0;   /* PRECEDENCE*/
		jp2_struct->approx = 0;   /* APPROX*/		
		jp2_struct->brand = JP2_JP2;	/* BR         */
		jp2_struct->minversion = 0;	/* MinV       */
		jp2_struct->numcl = 1;
		jp2_struct->cl = (unsigned int*) opj_malloc(jp2_struct->numcl * sizeof(unsigned int));
		jp2_struct->cl[0] = JP2_JP2;	/* CL0 : JP2  */		
		jp2_struct->C = 7;      /* C : Always 7*/
		jp2_struct->UnkC = 0;      /* UnkC, colorspace specified in colr box*/
		jp2_struct->IPR = 0;      /* IPR, no intellectual property*/						
		jp2_struct->w = parameters->w;
		jp2_struct->h = parameters->h;
		jp2_struct->bpc = 7;  
		jp2_struct->meth = 1;
		jp2_struct->enumcs = 18;  // YUV
  }
}

void mj2_destroy_compress(opj_mj2_t *movie) {
	if(movie) {
		int i;
		mj2_tk_t *tk=NULL;

		if (movie->cinfo->j2k_handle) {
			j2k_destroy_compress(movie->j2k);
		}
		
		if (movie->num_cl != 0)
			opj_free(movie->cl);
		
		for (i = 0; i < movie->num_vtk + movie->num_stk + movie->num_htk; i++) {
			tk = &movie->tk[i];
			if (tk->name_size != 0)
				opj_free(tk->name);
			if (tk->track_type == 0)  {// Video track
				if (tk->jp2_struct.comps != 0)
					opj_free(tk->jp2_struct.comps);
				if (tk->jp2_struct.cl != 0)
					opj_free(tk->jp2_struct.cl);
				if (tk->num_jp2x != 0)
					opj_free(tk->jp2xdata);
				
			}
			if (tk->num_url != 0)
				opj_free(tk->url);
			if (tk->num_urn != 0)
				opj_free(tk->urn);
			if (tk->num_br != 0)
				opj_free(tk->br);
			if (tk->num_tts != 0)
				opj_free(tk->tts);
			if (tk->num_chunks != 0)
				opj_free(tk->chunk);
			if (tk->num_samplestochunk != 0)
				opj_free(tk->sampletochunk);
			if (tk->num_samples != 0)
				opj_free(tk->sample);
		}
		
		opj_free(movie->tk);
	}	
	opj_free(movie);
}
