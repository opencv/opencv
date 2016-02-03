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
#include "FreeImage.h"
#ifdef WIN32
#include <malloc.h>
#endif
#include <string.h>
#include <stdlib.h>

#define NB_EXTENSIONS 2
/* -------------------------------------------------------------------------- */
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

int main (int argc,char * argv [])
{
  opj_dparameters_t l_param;
  opj_codec_t * l_codec;
  opj_image_t * l_image;
  FILE * l_file;
  opj_stream_t * l_stream;
  OPJ_UINT32 l_data_size;
  OPJ_UINT32 l_max_data_size = 1000;
  OPJ_UINT32 l_tile_index;
  OPJ_BYTE * l_data = (OPJ_BYTE *) malloc(1000);
  bool l_go_on = true;
  OPJ_INT32 l_tile_x0,l_tile_y0;
  OPJ_UINT32 l_tile_width,l_tile_height,l_nb_tiles_x,l_nb_tiles_y,l_nb_comps;
  OPJ_INT32 l_current_tile_x0,l_current_tile_y0,l_current_tile_x1,l_current_tile_y1;
  char * l_input_file,* l_output_file;
  OPJ_INT32 l_min_x, l_min_y, l_max_x, l_max_y;
  OPJ_CODEC_FORMAT l_codec_format;
  FIBITMAP * l_bitmap;
  unsigned char * l_image_data;
  OPJ_INT32 l_req_x,l_req_y;
  OPJ_UINT32 l_image_width,l_image_height,l_image_boundary,l_offset;
  unsigned char * l_tile_ptr [3];
  unsigned char * l_line_ptr, * l_current_ptr;
  OPJ_UINT32 i,j;

  if
    (argc != 8)
  {
    printf("usage : ... \n");
    return 1;
  }

  PROFINIT();
  FreeImage_Initialise(0);

  l_input_file = argv[1];
  l_output_file = argv[2];
  l_min_x = atoi(argv[3]);
  l_min_y = atoi(argv[4]);
  l_max_x = atoi(argv[5]);
  l_max_y = atoi(argv[6]);

  l_codec_format = get_format(l_input_file);

  if
    ((! l_data) || (l_codec_format == CODEC_UNKNOWN))
  {
    return 1;
  }


  opj_set_default_decoder_parameters(&l_param);

  /** you may here add custom decoding parameters */
  /* do not use layer decoding limitations */
  l_param.cp_layer = 0;

  /* do not use resolutions reductions */
  l_param.cp_reduce = atoi(argv[7]);

  /* to decode only a part of the image data */
  //opj_restrict_decoding(&l_param,0,0,1000,1000);

  l_codec = opj_create_decompress(l_codec_format);
  if
    (! l_codec)
  {
    free(l_data);
    return 1;
  }

  /* catch events using our callbacks and give a local context */
  opj_set_info_handler(l_codec, info_callback,00);
  opj_set_warning_handler(l_codec, warning_callback,00);
  opj_set_error_handler(l_codec, error_callback,00);
  opj_restrict_decoding(&l_param,l_min_x, l_min_y, l_max_x, l_max_y);

  if
    (! opj_setup_decoder(l_codec,&l_param))
  {
    free(l_data);
    opj_destroy_codec(l_codec);
    return 1;
  }

  l_file = fopen(l_input_file,"rb");
  if
    (! l_file)
  {
    fprintf(stdout, "Error opening input file\n");
    free(l_data);
    opj_destroy_codec(l_codec);
    return 1;
  }

  l_stream = opj_stream_create_default_file_stream(l_file,true);

  if
    (! opj_read_header(l_codec,
              &l_image,
              &l_tile_x0,
              &l_tile_y0,
              &l_tile_width,
              &l_tile_height,
              &l_nb_tiles_x,
              &l_nb_tiles_y,
              l_stream))
  {
    free(l_data);
    opj_stream_destroy(l_stream);
    fclose(l_file);
    opj_destroy_codec(l_codec);
    return 1;
  }

  if
    ((l_tile_x0 != 0) || (l_tile_y0 != 0))
  {
    free(l_data);
    opj_stream_destroy(l_stream);
    fclose(l_file);
    opj_destroy_codec(l_codec);
    return 1;
  }

  l_min_x -= l_min_x % l_tile_width;
  l_min_y -= l_min_y % l_tile_height;

  l_req_x = l_max_x % l_tile_width;
  if
    (l_req_x)
  {
    l_max_x += l_tile_width - l_req_x;
  }

  l_req_y = l_max_y % l_tile_height;
  if
    (l_req_y)
  {
    l_max_y += l_tile_height - l_req_y;
  }

  l_min_x = l_min_x < l_image->x0 ? l_image->x0 : l_min_x;
  l_min_y = l_min_y < l_image->y0 ? l_image->y0 : l_min_y;
  l_max_x = l_max_x > l_image->x1 ? l_image->x1 : l_max_x;
  l_max_y = l_max_y > l_image->y1 ? l_image->y1 : l_max_y;

  l_image_width = (l_max_x - l_min_x + (1 << l_param.cp_reduce) - 1) >> l_param.cp_reduce;
  l_image_height = (l_max_y - l_min_y + (1 << l_param.cp_reduce) - 1) >> l_param.cp_reduce;
  l_image_boundary = 3 * l_image_width;
  l_req_x = l_image_boundary % 4;
  if
    (l_req_x)
  {
    l_image_boundary += 4-l_req_x;
  }

  l_bitmap = FreeImage_Allocate(l_image_width, l_image_height, 24, 0, 0, 0);
  l_image_data = FreeImage_GetBits(l_bitmap);

  while
    (l_go_on)
  {
    if
      (! opj_read_tile_header(
            l_codec,
            &l_tile_index,
            &l_data_size,
            &l_current_tile_x0,
            &l_current_tile_y0,
            &l_current_tile_x1,
            &l_current_tile_y1,
            &l_nb_comps,
            &l_go_on,
            l_stream))
    {
      free(l_data);
      opj_stream_destroy(l_stream);
      fclose(l_file);
      opj_destroy_codec(l_codec);
      opj_image_destroy(l_image);
      return 1;
    }
    if
      (l_go_on)
    {
      if
        (l_data_size > l_max_data_size)
      {
        l_data = (OPJ_BYTE *) realloc(l_data,l_data_size);
        if
          (! l_data)
        {
          opj_stream_destroy(l_stream);
          fclose(l_file);
          opj_destroy_codec(l_codec);
          opj_image_destroy(l_image);
          return 1;
        }
        l_max_data_size = l_data_size;
      }

      if
        (! opj_decode_tile_data(l_codec,l_tile_index,l_data,l_data_size,l_stream))
      {
        free(l_data);
        opj_stream_destroy(l_stream);
        fclose(l_file);
        opj_destroy_codec(l_codec);
        opj_image_destroy(l_image);
        return 1;
      }
      /** now should inspect image to know the reduction factor and then how to behave with data */
      l_offset = (((l_max_y - l_current_tile_y0 + (1 << l_param.cp_reduce) - 1)>>l_param.cp_reduce) - 1) * l_image_boundary + ((l_current_tile_x0 - l_min_x + (1 << l_param.cp_reduce) - 1)>> l_param.cp_reduce )* 3;

      l_tile_width = (l_current_tile_x1 - l_current_tile_x0 + (1 << l_param.cp_reduce) - 1)>>l_param.cp_reduce;
      l_tile_height = (l_current_tile_y1 - l_current_tile_y0 + (1 << l_param.cp_reduce) - 1)>>l_param.cp_reduce;
      l_tile_ptr[0] = l_data + 2 * l_tile_width * l_tile_height;
      l_tile_ptr[1] = l_data + l_tile_width * l_tile_height;
      l_tile_ptr[2] = l_data ;
      l_line_ptr = l_image_data + l_offset;
      for
        (i=0;i<l_tile_width;++i)
      {
        l_current_ptr = l_line_ptr;
        for
          (j=0;j<l_tile_height;++j)
        {
          *(l_current_ptr++) = *(l_tile_ptr[0]++);
          *(l_current_ptr++) = *(l_tile_ptr[1]++);
          *(l_current_ptr++) = *(l_tile_ptr[2]++);
        }
        l_line_ptr -= l_image_boundary;
      }
    }
  }
  if
    (! opj_end_decompress(l_codec,l_stream))
  {
    free(l_data);
    opj_stream_destroy(l_stream);
    fclose(l_file);
    opj_destroy_codec(l_codec);
    opj_image_destroy(l_image);
    return 1;
  }
  free(l_data);
  opj_stream_destroy(l_stream);
  fclose(l_file);
  opj_destroy_codec(l_codec);
  opj_image_destroy(l_image);

  // Print profiling
  PROFPRINT();
  FreeImage_Save(0,l_bitmap,l_output_file,0);
  FreeImage_Unload(l_bitmap);

  return 0;
}
