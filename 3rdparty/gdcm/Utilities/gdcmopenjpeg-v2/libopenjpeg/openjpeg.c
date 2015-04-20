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

#ifdef WIN32
#include <windows.h>
#endif /* WIN32 */

#include "openjpeg.h"
#include "opj_malloc.h"
#include "j2k.h"
#include "jp2.h"
#include "event.h"
#include "cio.h"

typedef struct opj_decompression
{
  bool (* opj_read_header) (
    void *p_codec,
    opj_image_t **,
    OPJ_INT32 * p_tile_x0,
    OPJ_INT32 * p_tile_y0,
    OPJ_UINT32 * p_tile_width,
    OPJ_UINT32 * p_tile_height,
    OPJ_UINT32 * p_nb_tiles_x,
    OPJ_UINT32 * p_nb_tiles_y,
    struct opj_stream_private *cio,
    struct opj_event_mgr * p_manager);
  opj_image_t* (* opj_decode) (void * p_codec, struct opj_stream_private *p_cio, struct opj_event_mgr * p_manager);
  bool (*opj_read_tile_header)(
    void * p_codec,
    OPJ_UINT32 * p_tile_index,
    OPJ_UINT32* p_data_size,
    OPJ_INT32 * p_tile_x0,
    OPJ_INT32 * p_tile_y0,
    OPJ_INT32 * p_tile_x1,
    OPJ_INT32 * p_tile_y1,
    OPJ_UINT32 * p_nb_comps,
    bool * p_should_go_on,
    struct opj_stream_private *p_cio,
    struct opj_event_mgr * p_manager);
    bool (*opj_decode_tile_data)(void * p_codec,OPJ_UINT32 p_tile_index,OPJ_BYTE * p_data,OPJ_UINT32 p_data_size,struct opj_stream_private *p_cio,struct opj_event_mgr * p_manager);
  bool (* opj_end_decompress) (void *p_codec,struct opj_stream_private *cio,struct opj_event_mgr * p_manager);
  void (* opj_destroy) (void * p_codec);
  void (*opj_setup_decoder) (void * p_codec,opj_dparameters_t * p_param);
  bool (*opj_set_decode_area) (void * p_codec,OPJ_INT32 p_start_x,OPJ_INT32 p_end_x,OPJ_INT32 p_start_y,OPJ_INT32 p_end_y,struct opj_event_mgr * p_manager);


}opj_decompression_t;

typedef struct opj_compression
{
  bool (* opj_start_compress) (void *p_codec,struct opj_stream_private *cio,struct opj_image * p_image,  struct opj_event_mgr * p_manager);
  bool (* opj_encode) (void * p_codec, struct opj_stream_private *p_cio, struct opj_event_mgr * p_manager);
  bool (* opj_write_tile) (void * p_codec,OPJ_UINT32 p_tile_index,OPJ_BYTE * p_data,OPJ_UINT32 p_data_size,struct opj_stream_private * p_cio,struct opj_event_mgr * p_manager);
  bool (* opj_end_compress) (void * p_codec, struct opj_stream_private *p_cio, struct opj_event_mgr * p_manager);
  void (* opj_destroy) (void * p_codec);
  void (*opj_setup_encoder) (void * p_codec,opj_cparameters_t * p_param,struct opj_image * p_image, struct opj_event_mgr * p_manager);

}opj_compression_t;



typedef struct opj_codec_private
{
  union
  {    /* code-blocks informations */
    opj_decompression_t m_decompression;
    opj_compression_t m_compression;
    } m_codec_data;
  void * m_codec;
  opj_event_mgr_t m_event_mgr;
  unsigned is_decompressor : 1;
}
opj_codec_private_t;



/**
 * Default callback function.
 * Do nothing.
 */
void opj_default_callback (const char *msg, void *client_data)
{
#if 0
  fprintf( stderr, msg );
  assert( 0 );
#endif
}

void set_default_event_handler(opj_event_mgr_t * p_manager)
{
  p_manager->m_error_data = 00;
  p_manager->m_warning_data = 00;
  p_manager->m_info_data = 00;
  p_manager->error_handler = opj_default_callback;
  p_manager->info_handler = opj_default_callback;
  p_manager->warning_handler = opj_default_callback;
}

OPJ_UINT32 opj_read_from_file (void * p_buffer, OPJ_UINT32 p_nb_bytes, FILE * p_file)
{
  OPJ_UINT32 l_nb_read = fread(p_buffer,1,p_nb_bytes,p_file);
  return l_nb_read ? l_nb_read : -1;
}

OPJ_UINT32 opj_write_from_file (void * p_buffer, OPJ_UINT32 p_nb_bytes, FILE * p_file)
{
  return fwrite(p_buffer,1,p_nb_bytes,p_file);
}

OPJ_SIZE_T opj_skip_from_file (OPJ_SIZE_T p_nb_bytes, FILE * p_user_data)
{
  if
    (fseek(p_user_data,p_nb_bytes,SEEK_CUR))
  {
    return -1;
  }
  return p_nb_bytes;
}

bool opj_seek_from_file (OPJ_SIZE_T p_nb_bytes, FILE * p_user_data)
{
  if
    (fseek(p_user_data,p_nb_bytes,SEEK_SET))
  {
    return false;
  }
  return true;
}

/* ---------------------------------------------------------------------- */
#ifdef WIN32
#ifndef OPJ_STATIC
BOOL APIENTRY
DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
  switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH :
      break;
    case DLL_PROCESS_DETACH :
      break;
    case DLL_THREAD_ATTACH :
    case DLL_THREAD_DETACH :
      break;
    }

    return TRUE;
}
#endif /* OPJ_STATIC */
#endif /* WIN32 */

/* ---------------------------------------------------------------------- */


const char* OPJ_CALLCONV opj_version(void) {
    return OPENJPEG_VERSION;
}

opj_codec_t* OPJ_CALLCONV opj_create_decompress(OPJ_CODEC_FORMAT p_format)
{
  opj_codec_private_t *l_info = 00;

  l_info = (opj_codec_private_t*) opj_calloc(1, sizeof(opj_codec_private_t));
  if
    (!l_info)
  {
    return 00;
  }
  memset(l_info, 0, sizeof(opj_codec_private_t));
  l_info->is_decompressor = 1;
  switch
    (p_format)
  {
    case CODEC_J2K:
      l_info->m_codec_data.m_decompression.opj_decode = (opj_image_t* (*) (void *, struct opj_stream_private *, struct opj_event_mgr * ))j2k_decode;
      l_info->m_codec_data.m_decompression.opj_end_decompress =  (bool (*) (void *,struct opj_stream_private *,struct opj_event_mgr *))j2k_end_decompress;
      l_info->m_codec_data.m_decompression.opj_read_header =  (bool (*) (
        void *,
        opj_image_t **,
        OPJ_INT32 * ,
        OPJ_INT32 * ,
        OPJ_UINT32 * ,
        OPJ_UINT32 * ,
        OPJ_UINT32 * ,
        OPJ_UINT32 * ,
        struct opj_stream_private *,
        struct opj_event_mgr * )) j2k_read_header;
      l_info->m_codec_data.m_decompression.opj_destroy = (void (*) (void *))j2k_destroy;
      l_info->m_codec_data.m_decompression.opj_setup_decoder = (void (*) (void * ,opj_dparameters_t * )) j2k_setup_decoder;
      l_info->m_codec_data.m_decompression.opj_read_tile_header = (bool (*) (
        void *,
        OPJ_UINT32*,
        OPJ_UINT32*,
        OPJ_INT32 * ,
        OPJ_INT32 * ,
        OPJ_INT32 * ,
        OPJ_INT32 * ,
        OPJ_UINT32 * ,
        bool *,
        struct opj_stream_private *,
        struct opj_event_mgr * )) j2k_read_tile_header;
        l_info->m_codec_data.m_decompression.opj_decode_tile_data = (bool (*) (void *,OPJ_UINT32,OPJ_BYTE*,OPJ_UINT32,struct opj_stream_private *,  struct opj_event_mgr * )) j2k_decode_tile;
      l_info->m_codec_data.m_decompression.opj_set_decode_area = (bool (*) (void *,OPJ_INT32,OPJ_INT32,OPJ_INT32,OPJ_INT32, struct opj_event_mgr * )) j2k_set_decode_area;
      l_info->m_codec = j2k_create_decompress();
      if
        (! l_info->m_codec)
      {
        opj_free(l_info);
        return 00;
      }
      break;

    case CODEC_JP2:
      /* get a JP2 decoder handle */
      l_info->m_codec_data.m_decompression.opj_decode = (opj_image_t* (*) (void *, struct opj_stream_private *, struct opj_event_mgr * ))jp2_decode;
      l_info->m_codec_data.m_decompression.opj_end_decompress =  (bool (*) (void *,struct opj_stream_private *,struct opj_event_mgr *)) jp2_end_decompress;
      l_info->m_codec_data.m_decompression.opj_read_header =  (bool (*) (
        void *,
        opj_image_t **,

        OPJ_INT32 * ,
        OPJ_INT32 * ,
        OPJ_UINT32 * ,
        OPJ_UINT32 * ,
        OPJ_UINT32 * ,
        OPJ_UINT32 * ,
        struct opj_stream_private *,
        struct opj_event_mgr * )) jp2_read_header;

      l_info->m_codec_data.m_decompression.opj_read_tile_header = (
        bool (*) (
          void *,
          OPJ_UINT32*,
          OPJ_UINT32*,
          OPJ_INT32*,
          OPJ_INT32*,
          OPJ_INT32 * ,
          OPJ_INT32 * ,
          OPJ_UINT32 * ,
          bool *,
          struct opj_stream_private *,
          struct opj_event_mgr * )) jp2_read_tile_header;

      l_info->m_codec_data.m_decompression.opj_decode_tile_data = (bool (*) (void *,OPJ_UINT32,OPJ_BYTE*,OPJ_UINT32,struct opj_stream_private *,  struct opj_event_mgr * )) jp2_decode_tile;

      l_info->m_codec_data.m_decompression.opj_destroy = (void (*) (void *))jp2_destroy;
      l_info->m_codec_data.m_decompression.opj_setup_decoder = (void (*) (void * ,opj_dparameters_t * )) jp2_setup_decoder;
      l_info->m_codec_data.m_decompression.opj_set_decode_area = (bool (*) (void *,OPJ_INT32,OPJ_INT32,OPJ_INT32,OPJ_INT32, struct opj_event_mgr * )) jp2_set_decode_area;


      l_info->m_codec = jp2_create(true);
      if
        (! l_info->m_codec)
      {
        opj_free(l_info);
        return 00;
      }
      break;
    case CODEC_UNKNOWN:
    case CODEC_JPT:
    default:
      opj_free(l_info);
      return 00;
  }
  set_default_event_handler(&(l_info->m_event_mgr));
  return (opj_codec_t*) l_info;
}

void OPJ_CALLCONV opj_destroy_codec(opj_codec_t *p_info)
{
  if
    (p_info)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_info;
    if
      (l_info->is_decompressor)
    {
      l_info->m_codec_data.m_decompression.opj_destroy(l_info->m_codec);
    }
    else
    {
      l_info->m_codec_data.m_compression.opj_destroy(l_info->m_codec);
    }
    l_info->m_codec = 00;
    opj_free(l_info);
  }
}

void OPJ_CALLCONV opj_set_default_decoder_parameters(opj_dparameters_t *parameters) {
  if(parameters) {
    memset(parameters, 0, sizeof(opj_dparameters_t));
    /* default decoding parameters */
    parameters->cp_layer = 0;
    parameters->cp_reduce = 0;

    parameters->decod_format = -1;
    parameters->cod_format = -1;
/* UniPG>> */
#ifdef USE_JPWL
    parameters->jpwl_correct = false;
    parameters->jpwl_exp_comps = JPWL_EXPECTED_COMPONENTS;
    parameters->jpwl_max_tiles = JPWL_MAXIMUM_TILES;
#endif /* USE_JPWL */
/* <<UniPG */
  }
}

bool OPJ_CALLCONV opj_setup_decoder(opj_codec_t *p_info, opj_dparameters_t *parameters) {
  if
    (p_info && parameters)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_info;
    if
      (! l_info->is_decompressor)
    {
      return false;
    }
    l_info->m_codec_data.m_decompression.opj_setup_decoder(l_info->m_codec,parameters);
    return true;
  }
  return false;
}

opj_image_t* OPJ_CALLCONV opj_decode(opj_codec_t *p_info, opj_stream_t *cio)
{
  if
    (p_info && cio)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_info;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) cio;
    if
      (! l_info->is_decompressor)
    {
      return 00;
    }
    return l_info->m_codec_data.m_decompression.opj_decode(l_info->m_codec,l_cio,&(l_info->m_event_mgr));
  }
  return 00;
}

/**
 * Writes a tile with the given data.
 *
 * @param  p_compressor    the jpeg2000 codec.
 * @param  p_tile_index    the index of the tile to write. At the moment, the tiles must be written from 0 to n-1 in sequence.
 * @param  p_data        pointer to the data to write. Data is arranged in sequence, data_comp0, then data_comp1, then ... NO INTERLEAVING should be set.
 * @param  p_data_size      this value os used to make sure the data being written is correct. The size must be equal to the sum for each component of tile_width * tile_height * component_size. component_size can be 1,2 or 4 bytes,
 *                depending on the precision of the given component.
 * @param  p_stream      the stream to write data to.
 */
bool OPJ_CALLCONV opj_write_tile (
           opj_codec_t *p_codec,
           OPJ_UINT32 p_tile_index,
           OPJ_BYTE * p_data,
           OPJ_UINT32 p_data_size,
           opj_stream_t *p_stream
          )
{
  if
    (p_codec && p_stream && p_data)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_codec;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) p_stream;
    if
      (l_info->is_decompressor)
    {
      return false;
    }
    return l_info->m_codec_data.m_compression.opj_write_tile(l_info->m_codec,p_tile_index,p_data,p_data_size,l_cio,&(l_info->m_event_mgr));
  }
  return false;
}

/**
 * Reads a tile header. This function is compulsory and allows one to know the size of the tile thta will be decoded.
 * The user may need to refer to the image got by opj_read_header to understand the size being taken by the tile.
 *
 * @param  p_codec      the jpeg2000 codec.
 * @param  p_tile_index  pointer to a value that will hold the index of the tile being decoded, in case of success.
 * @param  p_data_size    pointer to a value that will hold the maximum size of the decoded data, in case of success. In case
 *              of truncated codestreams, the actual number of bytes decoded may be lower. The computation of the size is the same
 *              as depicted in opj_write_tile.
 * @param  p_tile_x0    pointer to a value that will hold the x0 pos of the tile (in the image).
 * @param  p_tile_y0    pointer to a value that will hold the y0 pos of the tile (in the image).
 * @param  p_tile_x1    pointer to a value that will hold the x1 pos of the tile (in the image).
 * @param  p_tile_y1    pointer to a value that will hold the y1 pos of the tile (in the image).
 * @param  p_nb_comps    pointer to a value that will hold the number of components in the tile.
 * @param  p_should_go_on  pointer to a boolean that will hold the fact that the decoding should go on. In case the
 *              codestream is over at the time of the call, the value will be set to false. The user should then stop
 *              the decoding.
 * @param  p_stream    the stream to decode.
 * @return  true      if the tile header could be decoded. In case the decoding should end, the returned value is still true.
 *              returning false may be the result of a shortage of memory or an internal error.
 */
bool OPJ_CALLCONV opj_read_tile_header(
          opj_codec_t *p_codec,
          OPJ_UINT32 * p_tile_index,
          OPJ_UINT32 * p_data_size,
          OPJ_INT32 * p_tile_x0,
          OPJ_INT32 * p_tile_y0,
          OPJ_INT32 * p_tile_x1,
          OPJ_INT32 * p_tile_y1,
          OPJ_UINT32 * p_nb_comps,
          bool * p_should_go_on,
          opj_stream_t * p_stream)
{
  if
    (p_codec && p_stream && p_data_size && p_tile_index)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_codec;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) p_stream;
    if
      (! l_info->is_decompressor)
    {
      return false;
    }
    return l_info->m_codec_data.m_decompression.opj_read_tile_header(
      l_info->m_codec,
      p_tile_index,
      p_data_size,
      p_tile_x0,
      p_tile_y0,
      p_tile_x1,
      p_tile_y1,
      p_nb_comps,
      p_should_go_on,
      l_cio,&(l_info->m_event_mgr));
  }
  return false;
}

/**
 * Reads a tile data. This function is compulsory and allows one to decode tile data. opj_read_tile_header should be called before.
 * The user may need to refer to the image got by opj_read_header to understand the size being taken by the tile.
 *
 * @param  p_codec      the jpeg2000 codec.
 * @param  p_tile_index  the index of the tile being decoded, this should be the value set by opj_read_tile_header.
 * @param  p_data      pointer to a memory block that will hold the decoded data.
 * @param  p_data_size    size of p_data. p_data_size should be bigger or equal to the value set by opj_read_tile_header.
 * @param  p_stream    the stream to decode.
 *
 * @return  true      if the data could be decoded.
 */
bool OPJ_CALLCONV opj_decode_tile_data(
          opj_codec_t *p_codec,
          OPJ_UINT32 p_tile_index,
          OPJ_BYTE * p_data,
          OPJ_UINT32 p_data_size,
          opj_stream_t *p_stream
          )
{
  if
    (p_codec && p_data && p_stream)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_codec;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) p_stream;
    if
      (! l_info->is_decompressor)
    {
      return false;
    }
    return l_info->m_codec_data.m_decompression.opj_decode_tile_data(l_info->m_codec,p_tile_index,p_data,p_data_size,l_cio,&(l_info->m_event_mgr));
  }
  return false;
}

bool OPJ_CALLCONV opj_read_header (
                   opj_codec_t *p_codec,
                   opj_image_t ** p_image,
                   OPJ_INT32 * p_tile_x0,
                   OPJ_INT32 * p_tile_y0,
                   OPJ_UINT32 * p_tile_width,
                   OPJ_UINT32 * p_tile_height,
                   OPJ_UINT32 * p_nb_tiles_x,
                   OPJ_UINT32 * p_nb_tiles_y,
                   opj_stream_t *p_cio)
{
  if
    (p_codec && p_cio)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_codec;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) p_cio;
    if
      (! l_info->is_decompressor)
    {
      return false;
    }
    return l_info->m_codec_data.m_decompression.opj_read_header(
      l_info->m_codec,
      p_image,
      p_tile_x0,
      p_tile_y0,
      p_tile_width,
      p_tile_height,
      p_nb_tiles_x,
      p_nb_tiles_y,
      l_cio,
      &(l_info->m_event_mgr));
  }
  return false;
}

/**
 * Sets the given area to be decoded. This function should be called right after opj_read_header and before any tile header reading.
 *
 * @param  p_codec      the jpeg2000 codec.
 * @param  p_start_x    the left position of the rectangle to decode (in image coordinates).
 * @param  p_end_x      the right position of the rectangle to decode (in image coordinates).
 * @param  p_start_y    the up position of the rectangle to decode (in image coordinates).
 * @param  p_end_y      the bottom position of the rectangle to decode (in image coordinates).
 *
 * @return  true      if the area could be set.
 */
bool OPJ_CALLCONV opj_set_decode_area(
          opj_codec_t *p_codec,
          OPJ_INT32 p_start_x,
          OPJ_INT32 p_start_y,
          OPJ_INT32 p_end_x,
          OPJ_INT32 p_end_y
          )
{
  if
    (p_codec)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_codec;
    if
      (! l_info->is_decompressor)
    {
      return false;
    }
    return  l_info->m_codec_data.m_decompression.opj_set_decode_area(
        l_info->m_codec,
        p_start_x,
        p_start_y,
        p_end_x,
        p_end_y,
        &(l_info->m_event_mgr));

  }
  return false;

}

bool OPJ_CALLCONV opj_end_decompress (opj_codec_t *p_codec,opj_stream_t *p_cio)
{
  if
    (p_codec && p_cio)
  {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_codec;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) p_cio;
    if
      (! l_info->is_decompressor)
    {
      return false;
    }
    return l_info->m_codec_data.m_decompression.opj_end_decompress(l_info->m_codec,l_cio,&(l_info->m_event_mgr));
  }
  return false;
}


opj_codec_t* OPJ_CALLCONV opj_create_compress(OPJ_CODEC_FORMAT p_format)
{
  opj_codec_private_t *l_info = 00;

  l_info = (opj_codec_private_t*)opj_calloc(1, sizeof(opj_codec_private_t));
  if
    (!l_info)
  {
    return 00;
  }
  memset(l_info, 0, sizeof(opj_codec_private_t));
  l_info->is_decompressor = 0;
  switch
    (p_format)
  {
    case CODEC_J2K:
      l_info->m_codec_data.m_compression.opj_encode = (bool (*) (void *, struct opj_stream_private *, struct opj_event_mgr * )) j2k_encode;
      l_info->m_codec_data.m_compression.opj_end_compress = (bool (*) (void *, struct opj_stream_private *, struct opj_event_mgr *)) j2k_end_compress;
      l_info->m_codec_data.m_compression.opj_start_compress = (bool (*) (void *,struct opj_stream_private *,struct opj_image * ,  struct opj_event_mgr *)) j2k_start_compress;
      l_info->m_codec_data.m_compression.opj_write_tile = (bool (*) (void *,OPJ_UINT32,OPJ_BYTE*,OPJ_UINT32,struct opj_stream_private *,  struct opj_event_mgr *)) j2k_write_tile;
      l_info->m_codec_data.m_compression.opj_destroy = (void (*) (void *)) j2k_destroy;
      l_info->m_codec_data.m_compression.opj_setup_encoder = (void (*) (void *,opj_cparameters_t *,struct opj_image *, struct opj_event_mgr * )) j2k_setup_encoder;

      l_info->m_codec = j2k_create_compress();
      if
        (! l_info->m_codec)
      {
        opj_free(l_info);
        return 00;
      }
      break;

    case CODEC_JP2:
      /* get a JP2 decoder handle */
      l_info->m_codec_data.m_compression.opj_encode = (bool (*) (void *, struct opj_stream_private *, struct opj_event_mgr * )) jp2_encode;
      l_info->m_codec_data.m_compression.opj_end_compress = (bool (*) (void *, struct opj_stream_private *, struct opj_event_mgr *)) jp2_end_compress;
      l_info->m_codec_data.m_compression.opj_start_compress = (bool (*) (void *,struct opj_stream_private *,struct opj_image * ,  struct opj_event_mgr *))  jp2_start_compress;
      l_info->m_codec_data.m_compression.opj_write_tile = (bool (*) (void *,OPJ_UINT32,OPJ_BYTE*,OPJ_UINT32,struct opj_stream_private *,  struct opj_event_mgr *)) jp2_write_tile;
      l_info->m_codec_data.m_compression.opj_destroy = (void (*) (void *)) jp2_destroy;
      l_info->m_codec_data.m_compression.opj_setup_encoder = (void (*) (void *,opj_cparameters_t *,struct opj_image *, struct opj_event_mgr * )) jp2_setup_encoder;

      l_info->m_codec = jp2_create(false);
      if
        (! l_info->m_codec)
      {
        opj_free(l_info);
        return 00;
      }
      break;
    case CODEC_UNKNOWN:
    case CODEC_JPT:
    default:
      opj_free(l_info);
      return 00;
  }
  set_default_event_handler(&(l_info->m_event_mgr));
  return (opj_codec_t*) l_info;
}

void OPJ_CALLCONV opj_set_default_encoder_parameters(opj_cparameters_t *parameters) {
  if(parameters) {
    memset(parameters, 0, sizeof(opj_cparameters_t));
    /* default coding parameters */
    parameters->cp_cinema = OFF;
    parameters->max_comp_size = 0;
    parameters->numresolution = 6;
    parameters->cp_rsiz = STD_RSIZ;
    parameters->cblockw_init = 64;
    parameters->cblockh_init = 64;
    parameters->prog_order = LRCP;
    parameters->roi_compno = -1;    /* no ROI */
    parameters->subsampling_dx = 1;
    parameters->subsampling_dy = 1;
    parameters->tp_on = 0;
    parameters->decod_format = -1;
    parameters->cod_format = -1;
    parameters->tcp_rates[0] = 0;
    parameters->tcp_numlayers = 0;
    parameters->cp_disto_alloc = 0;
    parameters->cp_fixed_alloc = 0;
    parameters->cp_fixed_quality = 0;
/* UniPG>> */
#ifdef USE_JPWL
    parameters->jpwl_epc_on = false;
    parameters->jpwl_hprot_MH = -1; /* -1 means unassigned */
    {
      int i;
      for (i = 0; i < JPWL_MAX_NO_TILESPECS; i++) {
        parameters->jpwl_hprot_TPH_tileno[i] = -1; /* unassigned */
        parameters->jpwl_hprot_TPH[i] = 0; /* absent */
      }
    };
    {
      int i;
      for (i = 0; i < JPWL_MAX_NO_PACKSPECS; i++) {
        parameters->jpwl_pprot_tileno[i] = -1; /* unassigned */
        parameters->jpwl_pprot_packno[i] = -1; /* unassigned */
        parameters->jpwl_pprot[i] = 0; /* absent */
      }
    };
    parameters->jpwl_sens_size = 0; /* 0 means no ESD */
    parameters->jpwl_sens_addr = 0; /* 0 means auto */
    parameters->jpwl_sens_range = 0; /* 0 means packet */
    parameters->jpwl_sens_MH = -1; /* -1 means unassigned */
    {
      int i;
      for (i = 0; i < JPWL_MAX_NO_TILESPECS; i++) {
        parameters->jpwl_sens_TPH_tileno[i] = -1; /* unassigned */
        parameters->jpwl_sens_TPH[i] = -1; /* absent */
      }
    };
#endif /* USE_JPWL */
/* <<UniPG */
  }
}

/**
 * Helper function.
 * Sets the stream to be a file stream. The FILE must have been open previously.
 * @param    p_stream  the stream to modify
 * @param    p_file    handler to an already open file.
*/
opj_stream_t* OPJ_CALLCONV opj_stream_create_default_file_stream (FILE * p_file,bool p_is_read_stream)
{
  return opj_stream_create_file_stream(p_file,J2K_STREAM_CHUNK_SIZE,p_is_read_stream);
}

opj_stream_t* OPJ_CALLCONV opj_stream_create_file_stream (FILE * p_file,OPJ_UINT32 p_size,bool p_is_read_stream)
{
  opj_stream_t* l_stream = 00;
  if
    (! p_file)
  {
    return 00;
  }
  l_stream = opj_stream_create(p_size,p_is_read_stream);
  if
    (! l_stream)
  {
    return 00;
  }
  opj_stream_set_user_data(l_stream,p_file);
  opj_stream_set_read_function(l_stream,(opj_stream_read_fn) opj_read_from_file);
  opj_stream_set_write_function(l_stream, (opj_stream_write_fn) opj_write_from_file);
  opj_stream_set_skip_function(l_stream, (opj_stream_skip_fn) opj_skip_from_file);
  opj_stream_set_seek_function(l_stream, (opj_stream_seek_fn) opj_seek_from_file);
  return l_stream;
}


bool OPJ_CALLCONV opj_setup_encoder(opj_codec_t *p_info, opj_cparameters_t *parameters, opj_image_t *image)
{
  if
    (p_info && parameters && image)
  {
    opj_codec_private_t * l_codec = ((opj_codec_private_t *) p_info);
    if
      (! l_codec->is_decompressor)
    {
      l_codec->m_codec_data.m_compression.opj_setup_encoder(l_codec->m_codec,parameters,image,&(l_codec->m_event_mgr));
      return true;
    }
  }
  return false;
}

bool OPJ_CALLCONV opj_encode(opj_codec_t *p_info, opj_stream_t *cio)
{
  if
    (p_info && cio)
  {
    opj_codec_private_t * l_codec = (opj_codec_private_t *) p_info;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) cio;
    if
      (! l_codec->is_decompressor)
    {
      l_codec->m_codec_data.m_compression.opj_encode(l_codec->m_codec,l_cio,&(l_codec->m_event_mgr));
      return true;
    }
  }
  return false;

}

bool OPJ_CALLCONV opj_start_compress (opj_codec_t *p_codec,opj_image_t * p_image,opj_stream_t *p_cio)
{
  if
    (p_codec && p_cio)
  {
    opj_codec_private_t * l_codec = (opj_codec_private_t *) p_codec;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) p_cio;
    if
      (! l_codec->is_decompressor)
    {
      return l_codec->m_codec_data.m_compression.opj_start_compress(l_codec->m_codec,l_cio,p_image,&(l_codec->m_event_mgr));
    }
  }
  return false;
}

bool OPJ_CALLCONV opj_end_compress (opj_codec_t *p_codec,opj_stream_t *p_cio)
{
  if
    (p_codec && p_cio)
  {
    opj_codec_private_t * l_codec = (opj_codec_private_t *) p_codec;
    opj_stream_private_t * l_cio = (opj_stream_private_t *) p_cio;
    if
      (! l_codec->is_decompressor)
    {
      return l_codec->m_codec_data.m_compression.opj_end_compress(l_codec->m_codec,l_cio,&(l_codec->m_event_mgr));
    }
  }
  return false;

}

bool OPJ_CALLCONV opj_set_info_handler(opj_codec_t * p_codec, opj_msg_callback p_callback,void * p_user_data)
{
  opj_codec_private_t * l_codec = (opj_codec_private_t *) p_codec;
  if
    (! l_codec)
  {
    return false;
  }
  l_codec->m_event_mgr.info_handler = p_callback;
  l_codec->m_event_mgr.m_info_data = p_user_data;
  return true;
}

bool OPJ_CALLCONV opj_set_warning_handler(opj_codec_t * p_codec, opj_msg_callback p_callback,void * p_user_data)
{
  opj_codec_private_t * l_codec = (opj_codec_private_t *) p_codec;
  if
    (! l_codec)
  {
    return false;
  }
  l_codec->m_event_mgr.warning_handler = p_callback;
  l_codec->m_event_mgr.m_warning_data = p_user_data;
  return true;
}

bool OPJ_CALLCONV opj_set_error_handler(opj_codec_t * p_codec, opj_msg_callback p_callback,void * p_user_data)
{
  opj_codec_private_t * l_codec = (opj_codec_private_t *) p_codec;
  if
    (! l_codec)
  {
    return false;
  }
  l_codec->m_event_mgr.error_handler = p_callback;
  l_codec->m_event_mgr.m_error_data = p_user_data;
  return true;
}

/*bool OPJ_CALLCONV opj_encode_with_info(opj_cinfo_t *cinfo, opj_stream_t *cio, opj_image_t *image, opj_codestream_info_t *cstr_info) {
  if(cinfo && cio && image) {
    switch(cinfo->codec_format) {
      case CODEC_J2K:
        return j2k_encode((opj_j2k_t*)cinfo->j2k_handle, (opj_stream_private_t *) cio, image, cstr_info);
      case CODEC_JP2:
        return jp2_encode((opj_jp2_t*)cinfo->jp2_handle, (opj_stream_private_t *) cio, image, cstr_info);
      case CODEC_JPT:
      case CODEC_UNKNOWN:
      default:
        break;
    }
  }
  return false;
}*/

void OPJ_CALLCONV opj_destroy_cstr_info(opj_codestream_info_t *cstr_info) {
  if
    (cstr_info)
  {
    int tileno;
    for (tileno = 0; tileno < cstr_info->tw * cstr_info->th; tileno++) {
      opj_tile_info_t *tile_info = &cstr_info->tile[tileno];
      opj_free(tile_info->thresh);
      opj_free(tile_info->packet);
      opj_free(tile_info->tp);
    }
    opj_free(cstr_info->tile);
    opj_free(cstr_info->marker);
  }
}

bool OPJ_CALLCONV opj_set_MCT(opj_cparameters_t *parameters,OPJ_FLOAT32 * pEncodingMatrix,OPJ_INT32 * p_dc_shift,OPJ_UINT32 pNbComp)
{
  OPJ_UINT32 l_matrix_size = pNbComp * pNbComp * sizeof(OPJ_FLOAT32);
  OPJ_UINT32 l_dc_shift_size = pNbComp * sizeof(OPJ_INT32);
  OPJ_UINT32 l_mct_total_size = l_matrix_size + l_dc_shift_size;
  // add MCT capability
  int rsiz = (int)parameters->cp_rsiz | (int)MCT;
  parameters->cp_rsiz = (OPJ_RSIZ_CAPABILITIES)rsiz;
  parameters->irreversible = 1;
  // use array based MCT
  parameters->tcp_mct = 2;
  parameters->mct_data = opj_malloc(l_mct_total_size);
  if
    (! parameters->mct_data)
  {
    return false;
  }
  memcpy(parameters->mct_data,pEncodingMatrix,l_matrix_size);
  memcpy(((OPJ_BYTE *) parameters->mct_data) +  l_matrix_size,p_dc_shift,l_dc_shift_size);
  return true;
}

/**
 * Restricts the decoding to the given image area.
 *
 * @param  parameters    the parameters to update.
 * @param  p_start_x    the starting x position of the area to decode.
 * @param  p_start_y    the starting y position of the area to decode.
 * @param  p_end_x      the x end position of the area to decode.
 * @param  p_end_x      the y end position of the area to decode.
 */
OPJ_API bool OPJ_CALLCONV opj_restrict_decoding (opj_dparameters_t *parameters,OPJ_INT32 p_start_x,OPJ_INT32 p_start_y,OPJ_INT32 p_end_x,OPJ_INT32 p_end_y)
{
  parameters->m_use_restrict_decode = 1;
  parameters->m_decode_start_x = p_start_x;
  parameters->m_decode_start_y = p_start_y;
  parameters->m_decode_end_x = p_end_x;
  parameters->m_decode_end_y = p_end_y;
  return true;
}

int j2k_get_reversible(
  opj_j2k_t * p_j2k)
{
  opj_cp_t *cp = 00;
  cp = &(p_j2k->m_cp);
  return cp->tcps->tccps->qmfbid;
}
int jp2_get_reversible(
  opj_jp2_t * p_jp2)
{
  return j2k_get_reversible(p_jp2->j2k);
}

int OPJ_CALLCONV opj_get_reversible(opj_codec_t *p_info, opj_dparameters_t *parameters)
{
  int ret = -1;
  if (p_info)
    {
    opj_codec_private_t * l_info = (opj_codec_private_t *) p_info;
    if (l_info->is_decompressor)
      {
      switch(parameters->decod_format)
        {
      case 0: // J2K_CFMT:
        ret = j2k_get_reversible(l_info->m_codec);
        break;
      case 1: // JP2_CFMT:
        ret = jp2_get_reversible(l_info->m_codec);
        break;
        }
      }
    }
  return ret;
}
