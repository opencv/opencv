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
/* set this macro to enable profiling for the given test */
/* warning : in order to be effective, openjpeg must have been built with profiling enabled !! */
//#define _PROFILE

#include "openjpeg.h"
#include <FreeImage.h>
#include <string.h>
#include <stdlib.h>

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

struct opj_format
{
  const char * m_extension;
  OPJ_CODEC_FORMAT m_format;
};

const struct opj_format c_extensions[] =
{
  {".j2k",CODEC_J2K},
  {".jp2",CODEC_JP2}
};

#define NB_EXTENSIONS 2
OPJ_CODEC_FORMAT get_format (const char * l_file_name)
{
  OPJ_INT32 i;
  const struct opj_format * l_current = c_extensions;
  for
    (i=0;i<NB_EXTENSIONS;++i)
  {
    if
      (! memcmp(l_current->m_extension,l_file_name + strlen(l_file_name)-4,4))
    {
      return l_current->m_format;
    }
    ++l_current;
  }
  return CODEC_UNKNOWN;
}

/* -------------------------------------------------------------------------- */

int main (int argc, char * argv [])
{
  opj_cparameters_t l_param;
  opj_codec_t * l_codec;
  opj_image_t * l_image;
  opj_image_cmptparm_t l_params [3];
  FILE * l_file;
  opj_stream_t * l_stream;
  opj_image_cmptparm_t * l_current_param_ptr;
  OPJ_UINT32 i,j,k,l;
  OPJ_BYTE *l_tile_data,*l_line_ptr,*l_current_ptr;
  OPJ_BYTE * l_tile_ptr [3];
  OPJ_UINT32 l_tile_width,l_image_width,l_image_height,l_chunk_size,l_image_boundary,l_req_x,l_req_y,l_nb_tiles_x,l_nb_tiles_y,l_current_tile_nb,l_data_size;
  OPJ_UINT32 l_offset;

  OPJ_CODEC_FORMAT l_codec_format;
  FIBITMAP * l_bitmap;
  FREE_IMAGE_FORMAT l_input_format;
  unsigned char * l_image_data;
  char * l_input_file,*l_output_file;
  if
    (argc != 6)
  {
    printf("usage \n");
    return 1;
  }

  l_input_file = argv[1];
  l_output_file = argv[2];
  l_tile_width = atoi(argv[3]);

  FreeImage_Initialise(0);

  l_codec_format = get_format(l_output_file);
  if
    (l_codec_format == CODEC_UNKNOWN)
  {
    return 1;
  }

  l_input_format = FreeImage_GetFileType(l_input_file,0);
  if
    (l_input_format == -1)
  {
    return 1;
  }
  l_bitmap = FreeImage_Load(l_input_format,l_input_file,0);
  l_image_data = FreeImage_GetBits(l_bitmap);
  l_image_width = FreeImage_GetWidth(l_bitmap);
  l_image_height = FreeImage_GetHeight(l_bitmap);
  l_chunk_size = FreeImage_GetBPP(l_bitmap);
  l_chunk_size /= 8;

  if
    (l_chunk_size < 3)
  {
    return 1;
  }
  l_image_boundary = l_image_width * l_chunk_size;

  l_req_x = l_image_boundary % 4;
  if
    (l_req_x)
  {
    l_image_boundary += 4 - l_req_x;
  }

  l_tile_data = (OPJ_BYTE*) malloc(l_tile_width * l_tile_width * 3);

  l_nb_tiles_x = l_image_width / l_tile_width;
  l_req_x = l_image_width % l_tile_width;
  l_nb_tiles_y = l_image_height / l_tile_width;
  l_req_y = l_image_height % l_tile_width;

  opj_set_default_encoder_parameters(&l_param);
  /** you may here add custom encoding parameters */
  /* rate specifications */
  /** number of quality layers in the stream */
  l_param.tcp_numlayers = 1;
  l_param.cp_fixed_quality = 1;
  /* is using others way of calculation */
  /* l_param.cp_disto_alloc = 1 or l_param.cp_fixed_alloc = 1 */
  /* l_param.tcp_rates[0] = ... */


  /* tile definitions parameters */
  /* position of the tile grid aligned with the image */
  l_param.cp_tx0 = 0;
  l_param.cp_ty0 = 0;
  /* tile size, we are using tile based encoding */
  l_param.tile_size_on = true;
  l_param.cp_tdx = l_tile_width;
  l_param.cp_tdy = l_tile_width;

  /* use irreversible encoding ?*/
  l_param.irreversible = atoi(argv[5]);

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
  l_param.numresolution = atoi(argv[4]);

  /** progression order to use*/
  /** LRCP, RLCP, RPCL, PCRL, CPRL */
  l_param.prog_order = LRCP;

  /** no "region" of interest, more precisally component */
  /* l_param.roi_compno = -1; */
  /* l_param.roi_shift = 0; */

  /* we are not using multiple tile parts for a tile. */
  /* l_param.tp_on = 0; */
  /* l_param.tp_flag = 0; */

  l_param.tcp_mct = 1;
  /* if we are using mct */
  /* opj_set_MCT(&l_param,l_mct,l_offsets,NUM_COMPS); */


  /* image definition */
  l_current_param_ptr = l_params;
  for
    (i=0;i<3;++i)
  {
    /* do not bother bpp useless */
    /*l_current_param_ptr->bpp = COMP_PREC;*/
    l_current_param_ptr->dx = 1;
    l_current_param_ptr->dy = 1;
    l_current_param_ptr->h = l_image_height;
    l_current_param_ptr->sgnd = 0;
    l_current_param_ptr->prec = 8;
    l_current_param_ptr->w = l_image_width;
    l_current_param_ptr->x0 = 0;
    l_current_param_ptr->y0 = 0;
    ++l_current_param_ptr;
  }

  l_codec = opj_create_compress(l_codec_format);
  if
    (! l_codec)
  {
    return 1;
  }

  /* catch events using our callbacks and give a local context */
  opj_set_info_handler(l_codec, info_callback,00);
  opj_set_warning_handler(l_codec, warning_callback,00);
  opj_set_error_handler(l_codec, error_callback,00);

  l_image = opj_image_tile_create(3,l_params,CLRSPC_SRGB);
  if
    (! l_image)
  {
    opj_destroy_codec(l_codec);
    return 1;
  }
  l_image->x0 = 0;
  l_image->y0 = 0;
  l_image->x1 = l_image_width;
  l_image->y1 = l_image_height;
  l_image->color_space = CLRSPC_SRGB;

  if
    (! opj_setup_encoder(l_codec,&l_param,l_image))
  {
    opj_destroy_codec(l_codec);
    opj_image_destroy(l_image);
    return 1;
  }

  l_file = fopen(l_output_file,"wb");
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

  l_current_tile_nb = 0;

  for
    (i=0;i<l_nb_tiles_y;++i)
  {
    for
      (j=0;j<l_nb_tiles_x;++j)
    {
      l_offset = (l_image_height - i * l_tile_width - 1) * l_image_boundary + l_chunk_size * j * l_tile_width;
      l_line_ptr = l_image_data + l_offset;
      l_tile_ptr[0] = l_tile_data;
      l_tile_ptr[1] = l_tile_data + l_tile_width * l_tile_width;
      l_tile_ptr[2] = l_tile_data + 2 * l_tile_width * l_tile_width;
      for
        (k=0;k<l_tile_width;++k)
      {
        l_current_ptr = l_line_ptr;
        for
          (l=0;l<l_tile_width;++l)
        {
          *(l_tile_ptr[0]++) = *(l_current_ptr+2);
          *(l_tile_ptr[1]++) = *(l_current_ptr+1);
          *(l_tile_ptr[2]++) = *(l_current_ptr);
          l_current_ptr += l_chunk_size;
        }
        l_line_ptr -= l_image_boundary;
      }
      l_data_size = l_tile_width * l_tile_width * 3;
      if
        (! opj_write_tile(l_codec,l_current_tile_nb++,l_tile_data,l_data_size,l_stream))
      {
        opj_stream_destroy(l_stream);
        fclose(l_file);
        opj_destroy_codec(l_codec);
        opj_image_destroy(l_image);
        return 1;
      }
    }
    if
      (l_req_x)
    {
      l_offset = (l_image_height - i * l_tile_width - 1) * l_image_boundary + l_chunk_size * j * l_tile_width;
      l_line_ptr = l_image_data + l_offset;
      l_tile_ptr[0] = l_tile_data;
      l_tile_ptr[1] = l_tile_data + l_tile_width * l_req_x;
      l_tile_ptr[2] = l_tile_data + 2 * l_tile_width * l_req_x;
      for
        (k=0;k<l_tile_width;++k)
      {
        l_current_ptr = l_line_ptr;
        for
          (l=0;l<l_req_x;++l)
        {
          *(l_tile_ptr[0]++) = *(l_current_ptr+2);
          *(l_tile_ptr[1]++) = *(l_current_ptr+1);
          *(l_tile_ptr[2]++) = *(l_current_ptr);
          l_current_ptr += l_chunk_size;
        }
        l_line_ptr -= l_image_boundary;
      }
      l_data_size = l_tile_width * l_req_x * 3;
      if
        (! opj_write_tile(l_codec,l_current_tile_nb++,l_tile_data,l_data_size,l_stream))
      {
        opj_stream_destroy(l_stream);
        fclose(l_file);
        opj_destroy_codec(l_codec);
        opj_image_destroy(l_image);
        return 1;
      }

    }
  }
  if
    (l_req_y)
  {
    for
      (j=0;j<l_nb_tiles_x;++j)
    {
      l_offset = (l_image_height - i * l_tile_width - 1) * l_image_boundary + l_chunk_size * j * l_tile_width;
      l_line_ptr = l_image_data + l_offset;
      l_tile_ptr[0] = l_tile_data;
      l_tile_ptr[1] = l_tile_data + l_tile_width * l_req_y;
      l_tile_ptr[2] = l_tile_data + 2 * l_tile_width * l_req_y;
      for
        (k=0;k<l_req_y;++k)
      {
        l_current_ptr = l_line_ptr;
        for
          (l=0;l<l_tile_width;++l)
        {
          *(l_tile_ptr[0]++) = *(l_current_ptr+2);
          *(l_tile_ptr[1]++) = *(l_current_ptr+1);
          *(l_tile_ptr[2]++) = *(l_current_ptr);
          l_current_ptr += l_chunk_size;
        }
        l_line_ptr -= l_image_boundary;
      }
      l_data_size = l_req_y * l_tile_width * 3;
      if
        (! opj_write_tile(l_codec,l_current_tile_nb++,l_tile_data,l_data_size,l_stream))
      {
        opj_stream_destroy(l_stream);
        fclose(l_file);
        opj_destroy_codec(l_codec);
        opj_image_destroy(l_image);
        return 1;
      }
    }
    if
      (l_req_x)
    {
      l_offset = (l_image_height - i * l_tile_width - 1) * l_image_boundary + l_chunk_size * j * l_tile_width;
      l_line_ptr = l_image_data + l_offset;
      l_tile_ptr[0] = l_tile_data;
      l_tile_ptr[1] = l_tile_data + l_req_x * l_req_y;
      l_tile_ptr[2] = l_tile_data + 2 * l_req_x * l_req_y;
      for
        (k=0;k<l_req_y;++k)
      {
        l_current_ptr = l_line_ptr;
        for
          (l=0;l<l_req_x;++l)
        {
          *(l_tile_ptr[0]++) = *(l_current_ptr+2);
          *(l_tile_ptr[1]++) = *(l_current_ptr+1);
          *(l_tile_ptr[2]++) = *(l_current_ptr);
          l_current_ptr += l_chunk_size;
        }
        l_line_ptr -= l_image_boundary;
      }
      l_data_size = l_req_y * l_req_x * 3;
      if
        (! opj_write_tile(l_codec,l_current_tile_nb++,l_tile_data,l_data_size,l_stream))
      {
        opj_stream_destroy(l_stream);
        fclose(l_file);
        opj_destroy_codec(l_codec);
        opj_image_destroy(l_image);
        return 1;
      }
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


  FreeImage_DeInitialise();

  // Print profiling
  PROFPRINT();

  return 0;
}
