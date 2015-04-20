/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2002-2003, Yannick Verschueren
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
#ifndef GDCM_JP2_H
#define GDCM_JP2_H
/**
@file jp2.h
@brief The JPEG-2000 file format Reader/Writer (JP2)

*/

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

/**
JP2 component
*/
typedef struct opj_jp2_comps {
  int depth;
  int sgnd;
  int bpcc;
} opj_jp2_comps_t;

/**
JPEG-2000 file format reader/writer
*/
typedef struct opj_jp2 {
  /** codec context */
  opj_common_ptr cinfo;
  /** handle to the J2K codec  */
  opj_j2k_t *j2k;
  unsigned int w;
  unsigned int h;
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
  unsigned int j2k_codestream_length;
} opj_jp2_t;

/**
JP2 Box
*/
typedef struct opj_jp2_box {
  int length;
  int type;
  int init_pos;
} opj_jp2_box_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Write the JP2H box - JP2 Header box (used in MJ2)
@param jp2 JP2 handle
@param cio Output buffer stream
*/
void jp2_write_jp2h(opj_jp2_t *jp2, opj_cio_t *cio);
/**
Read the JP2H box - JP2 Header box (used in MJ2)
@param jp2 JP2 handle
@param cio Input buffer stream
@return Returns true if successful, returns false otherwise
*/
bool jp2_read_jp2h(opj_jp2_t *jp2, opj_cio_t *cio);
/**
Creates a JP2 decompression structure
@param cinfo Codec context info
@return Returns a handle to a JP2 decompressor if successful, returns NULL otherwise
*/
opj_jp2_t* jp2_create_decompress(opj_common_ptr cinfo);
/**
Destroy a JP2 decompressor handle
@param jp2 JP2 decompressor handle to destroy
*/
void jp2_destroy_decompress(opj_jp2_t *jp2);
/**
Setup the decoder decoding parameters using user parameters.
Decoding parameters are returned in jp2->j2k->cp.
@param jp2 JP2 decompressor handle
@param parameters decompression parameters
*/
void jp2_setup_decoder(opj_jp2_t *jp2, opj_dparameters_t *parameters);
/**
Decode an image from a JPEG-2000 file stream
@param jp2 JP2 decompressor handle
@param cio Input buffer stream
@param cstr_info Codestream information structure if required, NULL otherwise
@return Returns a decoded image if successful, returns NULL otherwise
*/
opj_image_t* jp2_decode(opj_jp2_t *jp2, opj_cio_t *cio, opj_codestream_info_t *cstr_info);
/**
Creates a JP2 compression structure
@param cinfo Codec context info
@return Returns a handle to a JP2 compressor if successful, returns NULL otherwise
*/
opj_jp2_t* jp2_create_compress(opj_common_ptr cinfo);
/**
Destroy a JP2 compressor handle
@param jp2 JP2 compressor handle to destroy
*/
void jp2_destroy_compress(opj_jp2_t *jp2);
/**
Setup the encoder parameters using the current image and using user parameters.
Coding parameters are returned in jp2->j2k->cp.
@param jp2 JP2 compressor handle
@param parameters compression parameters
@param image input filled image
*/
void jp2_setup_encoder(opj_jp2_t *jp2, opj_cparameters_t *parameters, opj_image_t *image);
/**
Encode an image into a JPEG-2000 file stream
@param jp2 JP2 compressor handle
@param cio Output buffer stream
@param image Image to encode
@param cstr_info Codestream information structure if required, NULL otherwise
@return Returns true if successful, returns false otherwise
*/
bool jp2_encode(opj_jp2_t *jp2, opj_cio_t *cio, opj_image_t *image, opj_codestream_info_t *cstr_info);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* GDCM_JP2_H */
