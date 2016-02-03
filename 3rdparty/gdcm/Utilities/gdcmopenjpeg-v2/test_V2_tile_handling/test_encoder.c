/*
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
#define USE_OPJ_DEPRECATED

#include "openjpeg.h"
#include "stdlib.h"

/* set this macro to enable profiling for the given test */
/* warning : in order to be effective, openjpeg must have been built with profiling enabled !! */
//#define _PROFILE

#ifdef WIN32
#include "windows.h" // needed for rand() function
#else
#include <stdlib.h>
#endif


#define NUM_COMPS      3
#define IMAGE_WIDTH      2000
#define IMAGE_HEIGHT    2000
#define TILE_WIDTH      1000
#define TILE_HEIGHT      1000
#define COMP_PREC      8
#define OUTPUT_FILE      "test.j2k"

/* -------------------------------------------------------------------------- */

/**
sample error callback expecting a FILE* client object
*/
void error_callback_file(const char *msg, void *client_data) {
  FILE *stream = (FILE*)client_data;
  fprintf(stream, "[ERROR] %s", msg);
}
/**
sample warning callback expecting a FILE* client object
*/
void warning_callback_file(const char *msg, void *client_data) {
  FILE *stream = (FILE*)client_data;
  fprintf(stream, "[WARNING] %s", msg);
}
/**
sample error debug callback expecting no client object
*/
void error_callback(const char *msg, void *client_data) {
  (void)client_data;
  fprintf(stdout, "[ERROR] %s", msg);
}
/**
sample warning debug callback expecting no client object
*/
void warning_callback(const char *msg, void *client_data) {
  (void)client_data;
  fprintf(stdout, "[WARNING] %s", msg);
}
/**
sample debug callback expecting no client object
*/
void info_callback(const char *msg, void *client_data) {
  (void)client_data;
  fprintf(stdout, "[INFO] %s", msg);
}

/* -------------------------------------------------------------------------- */

int main ()
{
  opj_cparameters_t l_param;
  opj_codec_t * l_codec;
  opj_image_t * l_image;
  opj_image_cmptparm_t l_params [NUM_COMPS];
  FILE * l_file;
  opj_stream_t * l_stream;
  OPJ_UINT32 l_nb_tiles = (IMAGE_WIDTH/TILE_WIDTH) * (IMAGE_HEIGHT/TILE_HEIGHT);
  OPJ_UINT32 l_data_size = TILE_WIDTH * TILE_HEIGHT * NUM_COMPS * (COMP_PREC/8);

#ifdef USING_MCT
  const OPJ_FLOAT32 l_mct [] =
  {
    1 , 0 , 0 ,
    0 , 1 , 0 ,
    0 , 0 , 1
  };

  const OPJ_INT32 l_offsets [] =
  {
    128 , 128 , 128
  };
#endif

  opj_image_cmptparm_t * l_current_param_ptr;
  OPJ_UINT32 i;
  OPJ_BYTE *l_data;

  PROFINIT();
  l_data = (OPJ_BYTE*) malloc(TILE_WIDTH * TILE_HEIGHT * NUM_COMPS * (COMP_PREC/8) * sizeof(OPJ_BYTE));

  fprintf(stdout, "Encoding random values -> keep in mind that this is very hard to compress\n");
  for
    (i=0;i<l_data_size;++i)
  {
    l_data[i] = rand();
  }

  opj_set_default_encoder_parameters(&l_param);
  /** you may here add custom encoding parameters */
  /* rate specifications */
  /** number of quality layers in the stream */
  l_param.tcp_numlayers = 1;
  l_param.cp_fixed_quality = 1;
  l_param.tcp_distoratio[0] = 20;
  /* is using others way of calculation */
  /* l_param.cp_disto_alloc = 1 or l_param.cp_fixed_alloc = 1 */
  /* l_param.tcp_rates[0] = ... */


  /* tile definitions parameters */
  /* position of the tile grid aligned with the image */
  l_param.cp_tx0 = 0;
  l_param.cp_ty0 = 0;
  /* tile size, we are using tile based encoding */
  l_param.tile_size_on = true;
  l_param.cp_tdx = TILE_WIDTH;
  l_param.cp_tdy = TILE_HEIGHT;

  /* use irreversible encoding ?*/
  l_param.irreversible = 1;

  /* do not bother with mct, the rsiz is set when calling opj_set_MCT*/
  /*l_param.cp_rsiz = STD_RSIZ;*/

  /* no cinema */
  /*l_param.cp_cinema = 0;*/

  /* no not bother using SOP or EPH markers, do not use custom size precinct */
  /* number of precincts to specify */
  /* l_param.csty = 0;*/
  /* l_param.res_spec = ... */
  /* l_param.prch_init[i] = .. */
  /* l_param.prcw_init[i] = .. */


  /* do not use progression order changes */
  /*l_param.numpocs = 0;*/
  /* l_param.POC[i].... */

  /* do not restrain the size for a component.*/
  /* l_param.max_comp_size = 0; */

  /** block encoding style for each component, do not use at the moment */
  /** J2K_CCP_CBLKSTY_TERMALL, J2K_CCP_CBLKSTY_LAZY, J2K_CCP_CBLKSTY_VSC, J2K_CCP_CBLKSTY_SEGSYM, J2K_CCP_CBLKSTY_RESET */
  /* l_param.mode = 0;*/

  /** number of resolutions */
  l_param.numresolution = 6;

  /** progression order to use*/
  /** LRCP, RLCP, RPCL, PCRL, CPRL */
  l_param.prog_order = LRCP;

  /** no "region" of interest, more precisally component */
  /* l_param.roi_compno = -1; */
  /* l_param.roi_shift = 0; */

  /* we are not using multiple tile parts for a tile. */
  /* l_param.tp_on = 0; */
  /* l_param.tp_flag = 0; */

  /* if we are using mct */
#ifdef USING_MCT
  opj_set_MCT(&l_param,l_mct,l_offsets,NUM_COMPS);
#endif


  /* image definition */
  l_current_param_ptr = l_params;
  for
    (i=0;i<NUM_COMPS;++i)
  {
    /* do not bother bpp useless */
    /*l_current_param_ptr->bpp = COMP_PREC;*/
    l_current_param_ptr->dx = 1;
    l_current_param_ptr->dy = 1;
    l_current_param_ptr->h = IMAGE_HEIGHT;
    l_current_param_ptr->sgnd = 0;
    l_current_param_ptr->prec = COMP_PREC;
    l_current_param_ptr->w = IMAGE_WIDTH;
    l_current_param_ptr->x0 = 0;
    l_current_param_ptr->y0 = 0;
    ++l_current_param_ptr;
  }

  l_codec = opj_create_compress(CODEC_J2K);
  if
    (! l_codec)
  {
    return 1;
  }

  /* catch events using our callbacks and give a local context */
  opj_set_info_handler(l_codec, info_callback,00);
  opj_set_warning_handler(l_codec, warning_callback,00);
  opj_set_error_handler(l_codec, error_callback,00);

  l_image = opj_image_tile_create(NUM_COMPS,l_params,CLRSPC_SRGB);
  if
    (! l_image)
  {
    opj_destroy_codec(l_codec);
    return 1;
  }
  l_image->x0 = 0;
  l_image->y0 = 0;
  l_image->x1 = IMAGE_WIDTH;
  l_image->y1 = IMAGE_HEIGHT;
  l_image->color_space = CLRSPC_SRGB;

  if
    (! opj_setup_encoder(l_codec,&l_param,l_image))
  {
    opj_destroy_codec(l_codec);
    opj_image_destroy(l_image);
    return 1;
  }

  l_file = fopen(OUTPUT_FILE,"wb");
  if
    (! l_file)
  {
    opj_destroy_codec(l_codec);
    opj_image_destroy(l_image);
    return 1;
  }

  l_stream = opj_stream_create_default_file_stream(l_file,false);

  if
    (! opj_start_compress(l_codec,l_image,l_stream))
  {
    opj_stream_destroy(l_stream);
    fclose(l_file);
    opj_destroy_codec(l_codec);
    opj_image_destroy(l_image);
    return 1;
  }
  for
    (i=0;i<l_nb_tiles;++i)
  {
    if
      (! opj_write_tile(l_codec,i,l_data,l_data_size,l_stream))
    {
      opj_stream_destroy(l_stream);
      fclose(l_file);
      opj_destroy_codec(l_codec);
      opj_image_destroy(l_image);
      return 1;
    }
  }
  if
    (! opj_end_compress(l_codec,l_stream))
  {
    opj_stream_destroy(l_stream);
    fclose(l_file);
    opj_destroy_codec(l_codec);
    opj_image_destroy(l_image);
    return 1;
  }
  opj_stream_destroy(l_stream);
  fclose(l_file);
  opj_destroy_codec(l_codec);
  opj_image_destroy(l_image);

  // Print profiling
  PROFPRINT();

  return 0;
}
