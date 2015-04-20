/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
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
#ifndef __JP2_H
#define __JP2_H
/**
@file jp2.h
@brief The JPEG-2000 file format Reader/Writer (JP2)

*/
#include "openjpeg.h"




/**********************************************************************************
 ********************************* FORWARD DECLARATIONS ***************************
 **********************************************************************************/
struct opj_j2k;
struct opj_procedure_list;
struct opj_event_mgr;
struct opj_stream_private;
struct opj_dparameters;
struct opj_cparameters;

/** @defgroup JP2 JP2 - JPEG-2000 file format reader/writer */
/*@{*/

#define JPIP_JPIP 0x6a706970

#define JP2_JP   0x6a502020    /**< JPEG 2000 signature box */
#define JP2_FTYP 0x66747970    /**< File type box */
#define JP2_JP2H 0x6a703268    /**< JP2 header box */
#define JP2_IHDR 0x69686472    /**< Image header box */
#define JP2_COLR 0x636f6c72    /**< Colour specification box */
#define JP2_JP2C 0x6a703263    /**< Contiguous codestream box */
#define JP2_URL  0x75726c20    /**< URL box */
#define JP2_DBTL 0x6474626c    /**< ??? */
#define JP2_BPCC 0x62706363    /**< Bits per component box */
#define JP2_JP2  0x6a703220    /**< File type fields */

/* ----------------------------------------------------------------------- */


typedef enum
{
  JP2_STATE_NONE      = 0x0,
  JP2_STATE_SIGNATURE    = 0x1,
  JP2_STATE_FILE_TYPE    = 0x2,
  JP2_STATE_HEADER    = 0x4,
  JP2_STATE_CODESTREAM  = 0x8,
  JP2_STATE_END_CODESTREAM  = 0x10,
  JP2_STATE_UNKNOWN    = 0x80000000
}
JP2_STATE;

typedef enum
{
  JP2_IMG_STATE_NONE      = 0x0,
  JP2_IMG_STATE_UNKNOWN    = 0x80000000
}
JP2_IMG_STATE;

/**
JP2 component
*/
typedef struct opj_jp2_comps
{
  unsigned int depth;
  int sgnd;
  unsigned int bpcc;
}
opj_jp2_comps_t;

/**
JPEG-2000 file format reader/writer
*/
typedef struct opj_jp2
{
  /** handle to the J2K codec  */
  struct opj_j2k *j2k;
  /** list of validation procedures */
  struct opj_procedure_list * m_validation_list;
  /** list of execution procedures */
  struct opj_procedure_list * m_procedure_list;

  /* width of image */
  unsigned int w;
  /* height of image */
  unsigned int h;
  /* number of components in the image */
  unsigned int numcomps;
  unsigned int bpc;
  unsigned int C;
  unsigned int UnkC;
  unsigned int IPR;
  unsigned int meth;
  unsigned int approx;
  unsigned int enumcs;
  unsigned int precedence;
  unsigned int brand;
  unsigned int minversion;
  unsigned int numcl;
  unsigned int *cl;
  opj_jp2_comps_t *comps;
  unsigned int j2k_codestream_offset;
  unsigned int jp2_state;
  unsigned int jp2_img_state;

}
opj_jp2_t;

/**
JP2 Box
*/
typedef struct opj_jp2_box
{
  unsigned int length;
  unsigned int type;
}
opj_jp2_box_t;

typedef struct opj_jp2_header_handler
{
  /* marker value */
  int id;
  /* action linked to the marker */
  bool (*handler) (opj_jp2_t *jp2,unsigned char * p_header_data, unsigned int p_header_size,struct opj_event_mgr * p_manager);
}
opj_jp2_header_handler_t;


typedef struct opj_jp2_img_header_writer_handler
{
  /* action to perform */
  unsigned char* (*handler) (opj_jp2_t *jp2,  unsigned int * p_data_size);
  /* result of the action : data */
  unsigned char *      m_data;
  /* size of data */
  unsigned int        m_size;
}
opj_jp2_img_header_writer_handler_t;





/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
 * Creates a jpeg2000 file decompressor.
 *
 * @return  an empty jpeg2000 file codec.
 */
opj_jp2_t* jp2_create (bool p_is_decoder);

/**
Destroy a JP2 decompressor handle
@param jp2 JP2 decompressor handle to destroy
*/
void jp2_destroy(opj_jp2_t *jp2);

/**
Setup the decoder decoding parameters using user parameters.
Decoding parameters are returned in jp2->j2k->cp.
@param jp2 JP2 decompressor handle
@param parameters decompression parameters
*/
void jp2_setup_decoder(opj_jp2_t *jp2, struct opj_dparameters *parameters);

/**
 * Decode an image from a JPEG-2000 file stream
 * @param jp2 JP2 decompressor handle
 * @param cio Input buffer stream
 * @param cstr_info Codestream information structure if required, NULL otherwise
 * @return Returns a decoded image if successful, returns NULL otherwise
*/
struct opj_image* jp2_decode(opj_jp2_t *jp2, struct opj_stream_private *cio, struct opj_event_mgr * p_manager);
/**
Setup the encoder parameters using the current image and using user parameters.
Coding parameters are returned in jp2->j2k->cp.
@param jp2 JP2 compressor handle
@param parameters compression parameters
@param image input filled image
*/
void jp2_setup_encoder(opj_jp2_t *jp2, struct opj_cparameters *parameters, struct opj_image *image,struct opj_event_mgr * p_manager);

/**
 * Starts a compression scheme, i.e. validates the codec parameters, writes the header.
 *
 * @param  jp2    the jpeg2000 file codec.
 * @param  cio    the stream object.
 *
 * @return true if the codec is valid.
 */
bool jp2_start_compress(opj_jp2_t *jp2,  struct opj_stream_private *cio,struct opj_image * p_image,struct opj_event_mgr * p_manager);

/**
 * Ends the compression procedures and possibiliy add data to be read after the
 * codestream.
 */
bool jp2_end_compress(opj_jp2_t *jp2, struct opj_stream_private *cio, struct opj_event_mgr * p_manager);

/**
Encode an image into a JPEG-2000 file stream
@param jp2 JP2 compressor handle
@param cio Output buffer stream
@param image Image to encode
@param cstr_info Codestream information structure if required, NULL otherwise
@return Returns true if successful, returns false otherwise
*/
bool jp2_encode(opj_jp2_t *jp2, struct opj_stream_private *cio, struct opj_event_mgr * p_manager);

/**
 * Reads a jpeg2000 file header structure.
 *
 * @param cio the stream to read data from.
 * @param jp2 the jpeg2000 file header structure.
 * @param p_manager the user event manager.
 *
 * @return true if the box is valid.
 */
bool jp2_read_header(
                opj_jp2_t *jp2,
                struct opj_image ** p_image,
                OPJ_INT32 * p_tile_x0,
                OPJ_INT32 * p_tile_y0,
                OPJ_UINT32 * p_tile_width,
                OPJ_UINT32 * p_tile_height,
                OPJ_UINT32 * p_nb_tiles_x,
                OPJ_UINT32 * p_nb_tiles_y,
                struct opj_stream_private *cio,
                struct opj_event_mgr * p_manager
              );
/**
 * Ends the decompression procedures and possibiliy add data to be read after the
 * codestream.
 */
bool jp2_end_decompress(opj_jp2_t *jp2, struct opj_stream_private *cio, struct opj_event_mgr * p_manager);

/**
 * Writes a tile.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
bool jp2_write_tile (
           opj_jp2_t *p_jp2,
           OPJ_UINT32 p_tile_index,
           OPJ_BYTE * p_data,
           OPJ_UINT32 p_data_size,
           struct opj_stream_private *p_stream,
           struct opj_event_mgr * p_manager
          );
/**
 * Decode tile data.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
bool jp2_decode_tile (
          opj_jp2_t * p_jp2,
          OPJ_UINT32 p_tile_index,
          OPJ_BYTE * p_data,
          OPJ_UINT32 p_data_size,
          struct opj_stream_private *p_stream,
          struct opj_event_mgr * p_manager
          );
/**
 * Reads a tile header.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
bool jp2_read_tile_header (
           opj_jp2_t * p_j2k,
           OPJ_UINT32 * p_tile_index,
           OPJ_UINT32 * p_data_size,
           OPJ_INT32 * p_tile_x0,
           OPJ_INT32 * p_tile_y0,
           OPJ_INT32 * p_tile_x1,
           OPJ_INT32 * p_tile_y1,
           OPJ_UINT32 * p_nb_comps,
           bool * p_go_on,
           struct opj_stream_private *p_stream,
           struct opj_event_mgr * p_manager
          );
/**
 * Sets the given area to be decoded. This function should be called right after opj_read_header and before any tile header reading.
 *
 * @param  p_jp2      the jpeg2000 codec.
 * @param  p_end_x      the right position of the rectangle to decode (in image coordinates).
 * @param  p_start_y    the up position of the rectangle to decode (in image coordinates).
 * @param  p_end_y      the bottom position of the rectangle to decode (in image coordinates).
 * @param  p_manager    the user event manager
 *
 * @return  true      if the area could be set.
 */
bool jp2_set_decode_area(
      opj_jp2_t *p_jp2,
      OPJ_INT32 p_start_x,
      OPJ_INT32 p_start_y,
      OPJ_INT32 p_end_x,
      OPJ_INT32 p_end_y,
      struct opj_event_mgr * p_manager
      );

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __JP2_H */
