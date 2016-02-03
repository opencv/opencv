/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
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
#include "jp2.h"
#include "cio.h"
#include "opj_malloc.h"
#include "event.h"
#include "j2k.h"
#include "function_list.h"
#include "assert.h"

/** @defgroup JP2 JP2 - JPEG-2000 file format reader/writer */
/*@{*/

#define BOX_SIZE  1024



/** @name Local static functions */
/*@{*/



/**
 * Writes the Jpeg2000 file Header box - JP2 Header box (warning, this is a super box).
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  user event manager.
 *
 * @return true if writting was successful.
*/
bool jp2_write_jp2h(
            opj_jp2_t *jp2,
            struct opj_stream_private *cio,
            struct opj_event_mgr * p_manager
          );

/**
 * Skips the Jpeg2000 Codestream Header box - JP2C Header box.
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  user event manager.
 *
 * @return true if writting was successful.
*/
bool jp2_skip_jp2c(
            opj_jp2_t *jp2,
            struct opj_stream_private *cio,
            struct opj_event_mgr * p_manager
          );

/**
 * Reads the Jpeg2000 file Header box - JP2 Header box (warning, this is a super box).
 *
 * @param  p_header_data  the data contained in the file header box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the file header box.
 * @param  p_manager    the user event manager.
 *
 * @return true if the JP2 Header box was successfully reconized.
*/
bool jp2_read_jp2h(
            opj_jp2_t *jp2,
            unsigned char * p_header_data,
            unsigned int p_header_size,
            struct opj_event_mgr * p_manager
          );

/**
 * Writes the Jpeg2000 codestream Header box - JP2C Header box. This function must be called AFTER the coding has been done.
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  user event manager.
 *
 * @return true if writting was successful.
*/
static bool jp2_write_jp2c(
           opj_jp2_t *jp2,
           struct opj_stream_private *cio,
           struct opj_event_mgr * p_manager
           );

/**
 * Reads a box header. The box is the way data is packed inside a jpeg2000 file structure.
 *
 * @param  cio            the input stream to read data from.
 * @param  box            the box structure to fill.
 * @param  p_number_bytes_read    pointer to an int that will store the number of bytes read from the stream (shoul usually be 2).
 * @param  p_manager        user event manager.
 *
 * @return  true if the box is reconized, false otherwise
*/
static bool jp2_read_boxhdr(
                opj_jp2_box_t *box,
                OPJ_UINT32 * p_number_bytes_read,
                struct opj_stream_private *cio,
                struct opj_event_mgr * p_manager
              );

/**
 * Reads a box header. The box is the way data is packed inside a jpeg2000 file structure. Data is read from a character string
 *
 * @param  p_data          the character string to read data from.
 * @param  box            the box structure to fill.
 * @param  p_number_bytes_read    pointer to an int that will store the number of bytes read from the stream (shoul usually be 2).
 * @param  p_box_max_size      the maximum number of bytes in the box.
 *
 * @return  true if the box is reconized, false otherwise
*/
static bool jp2_read_boxhdr_char(
                opj_jp2_box_t *box,
                OPJ_BYTE * p_data,
                OPJ_UINT32 * p_number_bytes_read,
                OPJ_UINT32 p_box_max_size,
                struct opj_event_mgr * p_manager
              );

/**
 * Reads a jpeg2000 file signature box.
 *
 * @param  p_header_data  the data contained in the signature box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the signature box.
 * @param  p_manager    the user event manager.
 *
 * @return true if the file signature box is valid.
 */
static bool jp2_read_jp(
          opj_jp2_t *jp2,
          unsigned char * p_header_data,
          unsigned int p_header_size,
          struct opj_event_mgr * p_manager
         );

/**
 * Writes a jpeg2000 file signature box.
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  the user event manager.
 *
 * @return true if writting was successful.
 */
static bool jp2_write_jp(
              opj_jp2_t *jp2,
              struct opj_stream_private *cio,
              struct opj_event_mgr * p_manager
            );

/**
 * Writes a FTYP box - File type box
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  the user event manager.
 *
 * @return  true if writting was successful.
 */
static bool jp2_write_ftyp(
              opj_jp2_t *jp2,
              struct opj_stream_private *cio,
              struct opj_event_mgr * p_manager
              );

/**
 * Reads a a FTYP box - File type box
 *
 * @param  p_header_data  the data contained in the FTYP box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the FTYP box.
 * @param  p_manager    the user event manager.
 *
 * @return true if the FTYP box is valid.
 */
static bool jp2_read_ftyp(
              opj_jp2_t *jp2,
              unsigned char * p_header_data,
              unsigned int p_header_size,
              struct opj_event_mgr * p_manager
            );

/**
 * Reads a IHDR box - Image Header box
 *
 * @param  p_image_header_data      pointer to actual data (already read from file)
 * @param  jp2              the jpeg2000 file codec.
 * @param  p_image_header_size      the size of the image header
 * @param  p_manager          the user event manager.
 *
 * @return  true if the image header is valid, fale else.
 */
static bool jp2_read_ihdr(
              opj_jp2_t *jp2,
              unsigned char * p_image_header_data,
              unsigned int p_image_header_size,
              struct opj_event_mgr * p_manager
              );

/**
 * Writes the Image Header box - Image Header box.
 *
 * @param jp2          jpeg2000 file codec.
 * @param p_nb_bytes_written  pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
static unsigned char * jp2_write_ihdr(
                opj_jp2_t *jp2,
                unsigned int * p_nb_bytes_written
               );

/**
 * Reads a Bit per Component box.
 *
 * @param  p_bpc_header_data      pointer to actual data (already read from file)
 * @param  jp2              the jpeg2000 file codec.
 * @param  p_bpc_header_size      the size of the bpc header
 * @param  p_manager          the user event manager.
 *
 * @return  true if the bpc header is valid, fale else.
 */
static bool jp2_read_bpcc(
              opj_jp2_t *jp2,
              unsigned char * p_bpc_header_data,
              unsigned int p_bpc_header_size,
              struct opj_event_mgr * p_manager
              );


/**
 * Writes the Bit per Component box.
 *
 * @param  jp2            jpeg2000 file codec.
 * @param  p_nb_bytes_written    pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
static unsigned char * jp2_write_bpcc(
                opj_jp2_t *jp2,
                unsigned int * p_nb_bytes_written
               );

/**
 * Reads the Colour Specification box.
 *
 * @param  p_colr_header_data      pointer to actual data (already read from file)
 * @param  jp2              the jpeg2000 file codec.
 * @param  p_colr_header_size      the size of the color header
 * @param  p_manager          the user event manager.
 *
 * @return  true if the bpc header is valid, fale else.
*/
static bool jp2_read_colr(
              opj_jp2_t *jp2,
              unsigned char * p_colr_header_data,
              unsigned int p_colr_header_size,
              struct opj_event_mgr * p_manager
              );

/**
 * Writes the Colour Specification box.
 *
 * @param jp2          jpeg2000 file codec.
 * @param p_nb_bytes_written  pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
static unsigned char *jp2_write_colr(
                opj_jp2_t *jp2,
                unsigned int * p_nb_bytes_written
              );


/**
 * Reads a jpeg2000 file header structure.
 *
 * @param cio the stream to read data from.
 * @param jp2 the jpeg2000 file header structure.
 * @param p_manager the user event manager.
 *
 * @return true if the box is valid.
 */
bool jp2_read_header_procedure(
                opj_jp2_t *jp2,
                struct opj_stream_private *cio,
                struct opj_event_mgr * p_manager
              );
/**
 * Excutes the given procedures on the given codec.
 *
 * @param  p_procedure_list  the list of procedures to execute
 * @param  jp2          the jpeg2000 file codec to execute the procedures on.
 * @param  cio          the stream to execute the procedures on.
 * @param  p_manager      the user manager.
 *
 * @return  true        if all the procedures were successfully executed.
 */
static bool jp2_exec (
          opj_jp2_t * jp2,
          struct opj_procedure_list * p_procedure_list,
          struct opj_stream_private *cio,
          struct opj_event_mgr * p_manager
          );
/**
 * Finds the execution function related to the given box id.
 *
 * @param  p_id  the id of the handler to fetch.
 *
 * @return  the given handler or NULL if it could not be found.
 */
static const opj_jp2_header_handler_t * jp2_find_handler (int p_id);

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
static void jp2_setup_encoding_validation (opj_jp2_t *jp2);

/**
 * Sets up the procedures to do on writting header. Developpers wanting to extend the library can add their own writting procedures.
 */
static void jp2_setup_header_writting (opj_jp2_t *jp2);

/**
 * The default validation procedure without any extension.
 *
 * @param  jp2        the jpeg2000 codec to validate.
 * @param  cio        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool jp2_default_validation (
                  opj_jp2_t * jp2,
                  struct opj_stream_private *cio,
                  struct opj_event_mgr * p_manager
                    );

/**
 * Finds the execution function related to the given box id.
 *
 * @param  p_id  the id of the handler to fetch.
 *
 * @return  the given handler or NULL if it could not be found.
 */
static const opj_jp2_header_handler_t * jp2_find_handler (
                        int p_id
                        );

/**
 * Finds the image execution function related to the given box id.
 *
 * @param  p_id  the id of the handler to fetch.
 *
 * @return  the given handler or NULL if it could not be found.
 */
static const opj_jp2_header_handler_t * jp2_img_find_handler (
                        int p_id
                        );

/**
 * Sets up the procedures to do on writting header after the codestream.
 * Developpers wanting to extend the library can add their own writting procedures.
 */
static void jp2_setup_end_header_writting (opj_jp2_t *jp2);

/**
 * Sets up the procedures to do on reading header after the codestream.
 * Developpers wanting to extend the library can add their own writting procedures.
 */
static void jp2_setup_end_header_reading (opj_jp2_t *jp2);

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
static void jp2_setup_decoding_validation (opj_jp2_t *jp2);

/**
 * Sets up the procedures to do on reading header.
 * Developpers wanting to extend the library can add their own writting procedures.
 */
static void jp2_setup_header_reading (opj_jp2_t *jp2);
/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */
const opj_jp2_header_handler_t jp2_header [] =
{
  {JP2_JP,jp2_read_jp},
  {JP2_FTYP,jp2_read_ftyp},
  {JP2_JP2H,jp2_read_jp2h}
};

const opj_jp2_header_handler_t jp2_img_header [] =
{
  {JP2_IHDR,jp2_read_ihdr},
  {JP2_COLR,jp2_read_colr},
  {JP2_BPCC,jp2_read_bpcc}
};
/**
 * Finds the execution function related to the given box id.
 *
 * @param  p_id  the id of the handler to fetch.
 *
 * @return  the given handler or 00 if it could not be found.
 */
const opj_jp2_header_handler_t * jp2_find_handler (
                        int p_id
                        )
{
  unsigned int i, l_handler_size = sizeof(jp2_header) / sizeof(opj_jp2_header_handler_t);
  for
    (i=0;i<l_handler_size;++i)
  {
    if
      (jp2_header[i].id == p_id)
    {
      return &jp2_header[i];
    }
  }
  return 00;
}

/**
 * Finds the image execution function related to the given box id.
 *
 * @param  p_id  the id of the handler to fetch.
 *
 * @return  the given handler or 00 if it could not be found.
 */
static const opj_jp2_header_handler_t * jp2_img_find_handler (
                        int p_id
                        )
{
  unsigned int i, l_handler_size = sizeof(jp2_img_header) / sizeof(opj_jp2_header_handler_t);
  for
    (i=0;i<l_handler_size;++i)
  {
    if
      (jp2_img_header[i].id == p_id)
    {
      return &jp2_img_header[i];
    }
  }
  return 00;

}

/**
 * Reads a jpeg2000 file header structure.
 *
 * @param cio the stream to read data from.
 * @param jp2 the jpeg2000 file header structure.
 * @param p_manager the user event manager.
 *
 * @return true if the box is valid.
 */
bool jp2_read_header_procedure(
           opj_jp2_t *jp2,
           opj_stream_private_t *cio,
           opj_event_mgr_t * p_manager)
{
  opj_jp2_box_t box;
  unsigned int l_nb_bytes_read;
  const opj_jp2_header_handler_t * l_current_handler;
  unsigned int l_last_data_size = BOX_SIZE;
  unsigned int l_current_data_size;
  unsigned char * l_current_data = 00;

  // preconditions
  assert(cio != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  l_current_data = (unsigned char*)opj_malloc(l_last_data_size);

  if
    (l_current_data == 00)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to handle jpeg2000 file header\n");
    return false;
  }
  memset(l_current_data, 0 , l_last_data_size);
  while
    (jp2_read_boxhdr(&box,&l_nb_bytes_read,cio,p_manager))
  {
    // is it the codestream box ?
    if
      (box.type == JP2_JP2C)
    {
      if
        (jp2->jp2_state & JP2_STATE_HEADER)
      {
        jp2->jp2_state |= JP2_STATE_CODESTREAM;
        return true;
      }
      else
      {
        opj_event_msg(p_manager, EVT_ERROR, "bad placed jpeg codestream\n");
        return false;
      }
    }
    else if
      (box.length == 0)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of undefined sizes\n");
      return false;
    }

    l_current_handler = jp2_find_handler(box.type);
    l_current_data_size = box.length - l_nb_bytes_read;

    if
      (l_current_handler != 00)
    {
      if
        (l_current_data_size > l_last_data_size)
      {
        l_current_data = (unsigned char*)opj_realloc(l_current_data,l_current_data_size);
        l_last_data_size = l_current_data_size;
      }
      l_nb_bytes_read = opj_stream_read_data(cio,l_current_data,l_current_data_size,p_manager);
      if
        (l_nb_bytes_read != l_current_data_size)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Problem with reading JPEG2000 box, stream error\n");
        return false;
      }
      if
        (! l_current_handler->handler(jp2,l_current_data,l_current_data_size,p_manager))
      {
        return false;
      }
    }
    else
    {
      jp2->jp2_state |= JP2_STATE_UNKNOWN;
      if
        (opj_stream_skip(cio,l_current_data_size,p_manager) != l_current_data_size)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Problem with skipping JPEG2000 box, stream error\n");
        return false;
      }
    }
  }
  return true;
}

/**
 * Reads a box header. The box is the way data is packed inside a jpeg2000 file structure.
 *
 * @param  cio            the input stream to read data from.
 * @param  box            the box structure to fill.
 * @param  p_number_bytes_read    pointer to an int that will store the number of bytes read from the stream (should usually be 8).
 * @param  p_manager        user event manager.
 *
 * @return  true if the box is reconized, false otherwise
*/
bool jp2_read_boxhdr(opj_jp2_box_t *box, OPJ_UINT32 * p_number_bytes_read,opj_stream_private_t *cio, opj_event_mgr_t * p_manager)
{
  /* read header from file */
  unsigned char l_data_header [8];

  // preconditions
  assert(cio != 00);
  assert(box != 00);
  assert(p_number_bytes_read != 00);
  assert(p_manager != 00);

  *p_number_bytes_read = opj_stream_read_data(cio,l_data_header,8,p_manager);
  if
    (*p_number_bytes_read != 8)
  {
    return false;
  }
  /* process read data */
  opj_read_bytes(l_data_header,&(box->length), 4);
  opj_read_bytes(l_data_header+4,&(box->type), 4);

  // do we have a "special very large box ?"
  // read then the XLBox
  if
    (box->length == 1)
  {
    OPJ_UINT32 l_xl_part_size;
    OPJ_UINT32 l_nb_bytes_read = opj_stream_read_data(cio,l_data_header,8,p_manager);
    if
      (l_nb_bytes_read != 8)
    {
      if
        (l_nb_bytes_read > 0)
      {
        *p_number_bytes_read += l_nb_bytes_read;
      }
      return false;
    }
    opj_read_bytes(l_data_header,&l_xl_part_size, 4);
    if
      (l_xl_part_size != 0)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box sizes higher than 2^32\n");
      return false;
    }
    opj_read_bytes(l_data_header,&(box->length), 4);
  }
  return true;
}

/**
 * Reads a box header. The box is the way data is packed inside a jpeg2000 file structure. Data is read from a character string
 *
 * @param  p_data          the character string to read data from.
 * @param  box            the box structure to fill.
 * @param  p_number_bytes_read    pointer to an int that will store the number of bytes read from the stream (shoul usually be 2).
 * @param  p_box_max_size      the maximum number of bytes in the box.
 *
 * @return  true if the box is reconized, false otherwise
*/
static bool jp2_read_boxhdr_char(
                opj_jp2_box_t *box,
                OPJ_BYTE * p_data,
                OPJ_UINT32 * p_number_bytes_read,
                OPJ_UINT32 p_box_max_size,
                opj_event_mgr_t * p_manager
              )
{
  // preconditions
  assert(p_data != 00);
  assert(box != 00);
  assert(p_number_bytes_read != 00);
  assert(p_manager != 00);

  if
    (p_box_max_size < 8)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of less than 8 bytes\n");
    return false;
  }
  /* process read data */
  opj_read_bytes(p_data,&(box->length), 4);
  p_data += 4;
  opj_read_bytes(p_data,&(box->type), 4);
  p_data += 4;
  *p_number_bytes_read = 8;

  // do we have a "special very large box ?"
  // read then the XLBox
  if
    (box->length == 1)
  {
    unsigned int l_xl_part_size;
    if
      (p_box_max_size < 16)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Cannot handle XL box of less than 16 bytes\n");
      return false;
    }

    opj_read_bytes(p_data,&l_xl_part_size, 4);
    p_data += 4;
    *p_number_bytes_read += 4;
    if
      (l_xl_part_size != 0)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box sizes higher than 2^32\n");
      return false;
    }
    opj_read_bytes(p_data,&(box->length), 4);
    *p_number_bytes_read += 4;
    if
      (box->length == 0)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of undefined sizes\n");
      return false;
    }

  }
  else if
    (box->length == 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of undefined sizes\n");
    return false;
  }
  return true;
}


/**
 * Reads a jpeg2000 file signature box.
 *
 * @param  p_header_data  the data contained in the signature box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the signature box.
 * @param  p_manager    the user event manager.
 *
 * @return true if the file signature box is valid.
 */
bool jp2_read_jp(
          opj_jp2_t *jp2,
          unsigned char * p_header_data,
          unsigned int p_header_size,
          opj_event_mgr_t * p_manager
         )
{
  unsigned int l_magic_number;

  // preconditions
  assert(p_header_data != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  if
    (jp2->jp2_state != JP2_STATE_NONE)
  {
    opj_event_msg(p_manager, EVT_ERROR, "The signature box must be the first box in the file.\n");
    return false;
  }


  /* assure length of data is correct (4 -> magic number) */
  if
    (p_header_size != 4)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error with JP signature Box size\n");
    return false;
  }

  // rearrange data
  opj_read_bytes(p_header_data,&l_magic_number,4);
  if
    (l_magic_number != 0x0d0a870a )
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error with JP Signature : bad magic number\n");
    return false;
  }
  jp2->jp2_state |= JP2_STATE_SIGNATURE;
  return true;
}

/**
 * Reads a a FTYP box - File type box
 *
 * @param  p_header_data  the data contained in the FTYP box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the FTYP box.
 * @param  p_manager    the user event manager.
 *
 * @return true if the FTYP box is valid.
 */
bool jp2_read_ftyp(
              opj_jp2_t *jp2,
              unsigned char * p_header_data,
              unsigned int p_header_size,
              opj_event_mgr_t * p_manager
            )
{
  unsigned int i;
  unsigned int l_remaining_bytes;

  // preconditions
  assert(p_header_data != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  if
    (jp2->jp2_state != JP2_STATE_SIGNATURE)
  {
    opj_event_msg(p_manager, EVT_ERROR, "The ftyp box must be the second box in the file.\n");
    return false;
  }

  /* assure length of data is correct */
  if
    (p_header_size < 8)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error with FTYP signature Box size\n");
    return false;
  }

  opj_read_bytes(p_header_data,&jp2->brand,4);    /* BR */
  p_header_data += 4;

  opj_read_bytes(p_header_data,&jp2->minversion,4);    /* MinV */
  p_header_data += 4;

  l_remaining_bytes = p_header_size - 8;

  /* the number of remaining bytes should be a multiple of 4 */
  if
    ((l_remaining_bytes & 0x3) != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error with FTYP signature Box size\n");
    return false;
  }
  /* div by 4 */
  jp2->numcl = l_remaining_bytes >> 2;
  if
    (jp2->numcl)
  {
    jp2->cl = (unsigned int *) opj_malloc(jp2->numcl * sizeof(unsigned int));
    if
      (jp2->cl == 00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Not enough memory with FTYP Box\n");
      return false;
    }
    memset(jp2->cl,0,jp2->numcl * sizeof(unsigned int));
  }


  for
    (i = 0; i < jp2->numcl; ++i)
  {
    opj_read_bytes(p_header_data,&jp2->cl[i],4);    /* CLi */
    p_header_data += 4;

  }
  jp2->jp2_state |= JP2_STATE_FILE_TYPE;
  return true;
}

/**
 * Writes a jpeg2000 file signature box.
 *
 * @param cio the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param p_manager the user event manager.
 *
 * @return true if writting was successful.
 */
bool jp2_write_jp (
          opj_jp2_t *jp2,
          opj_stream_private_t *cio,
          opj_event_mgr_t * p_manager
           )
{
  /* 12 bytes will be read */
  unsigned char l_signature_data [12];

  // preconditions
  assert(cio != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);


  /* write box length */
  opj_write_bytes(l_signature_data,12,4);
  /* writes box type */
  opj_write_bytes(l_signature_data+4,JP2_JP,4);
  /* writes magic number*/
  opj_write_bytes(l_signature_data+8,0x0d0a870a,4);
  if
    (opj_stream_write_data(cio,l_signature_data,12,p_manager) != 12)
  {
    return false;
  }
  return true;
}


/**
 * Writes a FTYP box - File type box
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  the user event manager.
 *
 * @return  true if writting was successful.
 */
bool jp2_write_ftyp(
            opj_jp2_t *jp2,
            opj_stream_private_t *cio,
            opj_event_mgr_t * p_manager
          )
{
  unsigned int i;
  unsigned int l_ftyp_size = 16 + 4 * jp2->numcl;
  unsigned char * l_ftyp_data, * l_current_data_ptr;
  bool l_result;

  // preconditions
  assert(cio != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  l_ftyp_data = (unsigned char *) opj_malloc(l_ftyp_size);

  if
    (l_ftyp_data == 00)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to handle ftyp data\n");
    return false;
  }
  memset(l_ftyp_data,0,l_ftyp_size);

  l_current_data_ptr = l_ftyp_data;

  opj_write_bytes(l_current_data_ptr, l_ftyp_size,4); /* box size */
  l_current_data_ptr += 4;

  opj_write_bytes(l_current_data_ptr, JP2_FTYP,4); /* FTYP */
  l_current_data_ptr += 4;

  opj_write_bytes(l_current_data_ptr, jp2->brand,4); /* BR */
  l_current_data_ptr += 4;

  opj_write_bytes(l_current_data_ptr, jp2->minversion,4); /* MinV */
  l_current_data_ptr += 4;

  for
    (i = 0; i < jp2->numcl; i++)
  {
    opj_write_bytes(l_current_data_ptr, jp2->cl[i],4);  /* CL */
  }

  l_result = (opj_stream_write_data(cio,l_ftyp_data,l_ftyp_size,p_manager) == l_ftyp_size);
  if
    (! l_result)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error while writting ftyp data to stream\n");
  }
  opj_free(l_ftyp_data);
  return l_result;
}

/**
 * Writes the Jpeg2000 file Header box - JP2 Header box (warning, this is a super box).
 *
 * @param cio      the stream to write data to.
 * @param jp2      the jpeg2000 file codec.
 * @param p_manager    user event manager.
 *
 * @return true if writting was successful.
*/
bool jp2_write_jp2h(
            opj_jp2_t *jp2,
            opj_stream_private_t *cio,
            opj_event_mgr_t * p_manager
          )
{
  opj_jp2_img_header_writer_handler_t l_writers [3];
  opj_jp2_img_header_writer_handler_t * l_current_writer;

  int i, l_nb_pass;
  /* size of data for super box*/
  int l_jp2h_size = 8;
  bool l_result = true;

  /* to store the data of the super box */
  unsigned char l_jp2h_data [8];

  // preconditions
  assert(cio != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  memset(l_writers,0,sizeof(l_writers));

  if
    (jp2->bpc == 255)
  {
    l_nb_pass = 3;
    l_writers[0].handler = jp2_write_ihdr;
    l_writers[1].handler = jp2_write_bpcc;
    l_writers[2].handler = jp2_write_colr;
  }
  else
  {
    l_nb_pass = 2;
    l_writers[0].handler = jp2_write_ihdr;
    l_writers[1].handler = jp2_write_colr;
  }

  /* write box header */
  /* write JP2H type */
  opj_write_bytes(l_jp2h_data+4,JP2_JP2H,4);

  l_current_writer = l_writers;
  for
    (i=0;i<l_nb_pass;++i)
  {
    l_current_writer->m_data = l_current_writer->handler(jp2,&(l_current_writer->m_size));
    if
      (l_current_writer->m_data == 00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to hold JP2 Header data\n");
      l_result = false;
      break;
    }
    l_jp2h_size += l_current_writer->m_size;
    ++l_current_writer;
  }

  if
    (! l_result)
  {
    l_current_writer = l_writers;
    for
      (i=0;i<l_nb_pass;++i)
    {
      if
        (l_current_writer->m_data != 00)
      {
        opj_free(l_current_writer->m_data );
      }
      ++l_current_writer;
    }
    return false;
  }

  /* write super box size */
  opj_write_bytes(l_jp2h_data,l_jp2h_size,4);

  /* write super box data on stream */
  if
    (opj_stream_write_data(cio,l_jp2h_data,8,p_manager) != 8)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Stream error while writting JP2 Header box\n");
    l_result = false;
  }

  if
    (l_result)
  {
    l_current_writer = l_writers;
    for
      (i=0;i<l_nb_pass;++i)
    {
      if
        (opj_stream_write_data(cio,l_current_writer->m_data,l_current_writer->m_size,p_manager) != l_current_writer->m_size)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Stream error while writting JP2 Header box\n");
        l_result = false;
        break;
      }
      ++l_current_writer;
    }
  }
  l_current_writer = l_writers;
  /* cleanup */
  for
    (i=0;i<l_nb_pass;++i)
  {
    if
      (l_current_writer->m_data != 00)
    {
      opj_free(l_current_writer->m_data );
    }
    ++l_current_writer;
  }
  return l_result;
}

/**
 * Reads the Jpeg2000 file Header box - JP2 Header box (warning, this is a super box).
 *
 * @param  p_header_data  the data contained in the file header box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the file header box.
 * @param  p_manager    the user event manager.
 *
 * @return true if the JP2 Header box was successfully reconized.
*/
bool jp2_read_jp2h(
            opj_jp2_t *jp2,
            unsigned char * p_header_data,
            unsigned int p_header_size,
            opj_event_mgr_t * p_manager
          )
{
  unsigned int l_box_size=0, l_current_data_size = 0;
  opj_jp2_box_t box;
  const opj_jp2_header_handler_t * l_current_handler;

  // preconditions
  assert(p_header_data != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  /* make sure the box is well placed */
  if
    ((jp2->jp2_state & JP2_STATE_FILE_TYPE) != JP2_STATE_FILE_TYPE )
  {
    opj_event_msg(p_manager, EVT_ERROR, "The  box must be the first box in the file.\n");
    return false;
  }
  jp2->jp2_img_state = JP2_IMG_STATE_NONE;

  /* iterate while remaining data */
  while
    (p_header_size > 0)
  {
    if
      (! jp2_read_boxhdr_char(&box,p_header_data,&l_box_size,p_header_size, p_manager))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Stream error while reading JP2 Header box\n");
      return false;
    }
    if
      (box.length > p_header_size)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Stream error while reading JP2 Header box\n");
      return false;
    }
    l_current_handler = jp2_img_find_handler(box.type);

    l_current_data_size = box.length - l_box_size;
    p_header_data += l_box_size;

    if
      (l_current_handler != 00)
    {
      if
        (! l_current_handler->handler(jp2,p_header_data,l_current_data_size,p_manager))
      {
        return false;
      }
    }
    else
    {
      jp2->jp2_img_state |= JP2_IMG_STATE_UNKNOWN;
    }
    p_header_data += l_current_data_size;
    p_header_size -= box.length;
  }
  jp2->jp2_state |= JP2_STATE_HEADER;
  return true;
}

/**
 * Reads a IHDR box - Image Header box
 *
 * @param  p_image_header_data      pointer to actual data (already read from file)
 * @param  jp2              the jpeg2000 file codec.
 * @param  p_image_header_size      the size of the image header
 * @param  p_image_header_max_size    maximum size of the header, any size bigger than this value should result the function to output false.
 * @param  p_manager          the user event manager.
 *
 * @return  true if the image header is valid, fale else.
 */
bool jp2_read_ihdr(
              opj_jp2_t *jp2,
              unsigned char * p_image_header_data,
              unsigned int p_image_header_size,
              opj_event_mgr_t * p_manager
              )
{
  // preconditions
  assert(p_image_header_data != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  if
    (p_image_header_size != 14)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Bad image header box (bad size)\n");
    return false;
  }
  opj_read_bytes(p_image_header_data,&(jp2->h),4);      /* HEIGHT */
  p_image_header_data += 4;
  opj_read_bytes(p_image_header_data,&(jp2->w),4);      /* WIDTH */
  p_image_header_data += 4;
  opj_read_bytes(p_image_header_data,&(jp2->numcomps),2);      /* NC */
  p_image_header_data += 2;

  /* allocate memory for components */
  jp2->comps = (opj_jp2_comps_t*) opj_malloc(jp2->numcomps * sizeof(opj_jp2_comps_t));
  if
    (jp2->comps == 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to handle image header (ihdr)\n");
    return false;
  }
  memset(jp2->comps,0,jp2->numcomps * sizeof(opj_jp2_comps_t));

  opj_read_bytes(p_image_header_data,&(jp2->bpc),1);      /* BPC */
  ++ p_image_header_data;
  opj_read_bytes(p_image_header_data,&(jp2->C),1);      /* C */
  ++ p_image_header_data;
  opj_read_bytes(p_image_header_data,&(jp2->UnkC),1);      /* UnkC */
  ++ p_image_header_data;
  opj_read_bytes(p_image_header_data,&(jp2->IPR),1);      /* IPR */
  ++ p_image_header_data;
  return true;
}

/**
 * Writes the Image Header box - Image Header box.
 *
 * @param jp2          jpeg2000 file codec.
 * @param p_nb_bytes_written  pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
static unsigned char * jp2_write_ihdr(
                opj_jp2_t *jp2,
                unsigned int * p_nb_bytes_written
               )
{
  unsigned char * l_ihdr_data,* l_current_ihdr_ptr;

  // preconditions
  assert(jp2 != 00);
  assert(p_nb_bytes_written != 00);

  /* default image header is 22 bytes wide */
  l_ihdr_data = (unsigned char *) opj_malloc(22);
  if
    (l_ihdr_data == 00)
  {
    return 00;
  }
  memset(l_ihdr_data,0,22);

  l_current_ihdr_ptr = l_ihdr_data;

  opj_write_bytes(l_current_ihdr_ptr,22,4);        /* write box size */
  l_current_ihdr_ptr+=4;
  opj_write_bytes(l_current_ihdr_ptr,JP2_IHDR, 4);    /* IHDR */
  l_current_ihdr_ptr+=4;
  opj_write_bytes(l_current_ihdr_ptr,jp2->h, 4);    /* HEIGHT */
  l_current_ihdr_ptr+=4;
  opj_write_bytes(l_current_ihdr_ptr, jp2->w, 4);    /* WIDTH */
  l_current_ihdr_ptr+=4;
  opj_write_bytes(l_current_ihdr_ptr, jp2->numcomps, 2);    /* NC */
  l_current_ihdr_ptr+=2;
  opj_write_bytes(l_current_ihdr_ptr, jp2->bpc, 1);    /* BPC */
  ++l_current_ihdr_ptr;
  opj_write_bytes(l_current_ihdr_ptr, jp2->C, 1);    /* C : Always 7 */
  ++l_current_ihdr_ptr;
  opj_write_bytes(l_current_ihdr_ptr, jp2->UnkC, 1);    /* UnkC, colorspace unknown */
  ++l_current_ihdr_ptr;
  opj_write_bytes(l_current_ihdr_ptr, jp2->IPR, 1);    /* IPR, no intellectual property */
  ++l_current_ihdr_ptr;
  *p_nb_bytes_written = 22;
  return l_ihdr_data;
}

/**
 * Writes the Bit per Component box.
 *
 * @param  jp2            jpeg2000 file codec.
 * @param  p_nb_bytes_written    pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
unsigned char * jp2_write_bpcc(
                opj_jp2_t *jp2,
                unsigned int * p_nb_bytes_written
               )
{
  unsigned int i;
  /* room for 8 bytes for box and 1 byte for each component */
  int l_bpcc_size = 8 + jp2->numcomps;
  unsigned char * l_bpcc_data,* l_current_bpcc_ptr;

  // preconditions
  assert(jp2 != 00);
  assert(p_nb_bytes_written != 00);

  l_bpcc_data = (unsigned char *) opj_malloc(l_bpcc_size);
  if
    (l_bpcc_data == 00)
  {
    return 00;
  }
  memset(l_bpcc_data,0,l_bpcc_size);

  l_current_bpcc_ptr = l_bpcc_data;

  opj_write_bytes(l_current_bpcc_ptr,l_bpcc_size,4);        /* write box size */
  l_current_bpcc_ptr += 4;
  opj_write_bytes(l_current_bpcc_ptr,JP2_BPCC,4);          /* BPCC */
  l_current_bpcc_ptr += 4;

  for
    (i = 0; i < jp2->numcomps; ++i)
  {
    opj_write_bytes(l_current_bpcc_ptr, jp2->comps[i].bpcc, 1); /* write each component information */
    ++l_current_bpcc_ptr;
  }
  *p_nb_bytes_written = l_bpcc_size;
  return l_bpcc_data;
}

/**
 * Reads a Bit per Component box.
 *
 * @param  p_bpc_header_data      pointer to actual data (already read from file)
 * @param  jp2              the jpeg2000 file codec.
 * @param  p_bpc_header_size      pointer that will hold the size of the bpc header
 * @param  p_bpc_header_max_size    maximum size of the header, any size bigger than this value should result the function to output false.
 * @param  p_manager          the user event manager.
 *
 * @return  true if the bpc header is valid, fale else.
 */
bool jp2_read_bpcc(
              opj_jp2_t *jp2,
              unsigned char * p_bpc_header_data,
              unsigned int p_bpc_header_size,
              opj_event_mgr_t * p_manager
              )
{
  unsigned int i;

  // preconditions
  assert(p_bpc_header_data != 00);
  assert(jp2 != 00);
  assert(p_manager != 00);

  // and length is relevant
  if
    (p_bpc_header_size != jp2->numcomps)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Bad BPCC header box (bad size)\n");
    return false;
  }

  // read info for each component
  for
    (i = 0; i < jp2->numcomps; ++i)
  {
    opj_read_bytes(p_bpc_header_data,&jp2->comps[i].bpcc ,1);  /* read each BPCC component */
    ++p_bpc_header_data;
  }
  return true;
}

/**
 * Writes the Colour Specification box.
 *
 * @param jp2          jpeg2000 file codec.
 * @param p_nb_bytes_written  pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
unsigned char *jp2_write_colr(
                opj_jp2_t *jp2,
                unsigned int * p_nb_bytes_written
              )
{
  /* room for 8 bytes for box 3 for common data and variable upon profile*/
  unsigned int l_colr_size = 11;
  unsigned char * l_colr_data,* l_current_colr_ptr;

  // preconditions
  assert(jp2 != 00);
  assert(p_nb_bytes_written != 00);

  switch
    (jp2->meth)
  {
    case 1 :
      l_colr_size += 4;
      break;
    case 2 :
      ++l_colr_size;
      break;
    default :
      return 00;
  }

  l_colr_data = (unsigned char *) opj_malloc(l_colr_size);
  if
    (l_colr_data == 00)
  {
    return 00;
  }
  memset(l_colr_data,0,l_colr_size);
  l_current_colr_ptr = l_colr_data;

  opj_write_bytes(l_current_colr_ptr,l_colr_size,4);        /* write box size */
  l_current_colr_ptr += 4;
  opj_write_bytes(l_current_colr_ptr,JP2_COLR,4);          /* BPCC */
  l_current_colr_ptr += 4;

  opj_write_bytes(l_current_colr_ptr, jp2->meth,1);        /* METH */
  ++l_current_colr_ptr;
  opj_write_bytes(l_current_colr_ptr, jp2->precedence,1);      /* PRECEDENCE */
  ++l_current_colr_ptr;
  opj_write_bytes(l_current_colr_ptr, jp2->approx,1);        /* APPROX */
  ++l_current_colr_ptr;

  if
    (jp2->meth == 1)
  {
    opj_write_bytes(l_current_colr_ptr, jp2->enumcs,4);      /* EnumCS */
  }
  else
  {
    opj_write_bytes(l_current_colr_ptr, 0, 1);            /* PROFILE (??) */
  }
  *p_nb_bytes_written = l_colr_size;
  return l_colr_data;
}

/**
 * Reads the Colour Specification box.
 *
 * @param  p_colr_header_data      pointer to actual data (already read from file)
 * @param  jp2              the jpeg2000 file codec.
 * @param  p_colr_header_size      pointer that will hold the size of the color header
 * @param  p_colr_header_max_size    maximum size of the header, any size bigger than this value should result the function to output false.
 * @param  p_manager          the user event manager.
 *
 * @return  true if the bpc header is valid, fale else.
*/
bool jp2_read_colr(
              opj_jp2_t * jp2,
              unsigned char * p_colr_header_data,
              unsigned int p_colr_header_size,
              opj_event_mgr_t * p_manager
              )
{
  // preconditions
  assert(jp2 != 00);
  assert(p_colr_header_data != 00);
  assert(p_manager != 00);

  if
    (p_colr_header_size < 3)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Bad BPCC header box (bad size)\n");
    return false;
  }

  opj_read_bytes(p_colr_header_data,&jp2->meth ,1);      /* METH */
  ++p_colr_header_data;

  opj_read_bytes(p_colr_header_data,&jp2->precedence ,1);      /* PRECEDENCE */
  ++p_colr_header_data;

  opj_read_bytes(p_colr_header_data,&jp2->approx ,1);      /* APPROX */
  ++p_colr_header_data;


  if
    (jp2->meth == 1)
  {
    if
      (p_colr_header_size != 7)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Bad BPCC header box (bad size)\n");
      return false;
    }
    opj_read_bytes(p_colr_header_data,&jp2->enumcs ,4);      /* EnumCS */
  }
  /*else
  {
    // do not care with profiles.
  }*/
  return true;
}

/**
 * Writes the Jpeg2000 codestream Header box - JP2C Header box.
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  user event manager.
 *
 * @return true if writting was successful.
*/
bool jp2_write_jp2c(
           opj_jp2_t *jp2,
           opj_stream_private_t *cio,
           opj_event_mgr_t * p_manager
           )
{
  unsigned int j2k_codestream_exit;
  unsigned char l_data_header [8];

  // preconditions
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);
  assert(opj_stream_has_seek(cio));

  j2k_codestream_exit = opj_stream_tell(cio);
  opj_write_bytes(l_data_header,j2k_codestream_exit - jp2->j2k_codestream_offset,4); /* size of codestream */
  opj_write_bytes(l_data_header + 4,JP2_JP2C,4);                     /* JP2C */

  if
    (! opj_stream_seek(cio,jp2->j2k_codestream_offset,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
    return false;
  }

  if
    (opj_stream_write_data(cio,l_data_header,8,p_manager) != 8)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
    return false;
  }

  if
    (! opj_stream_seek(cio,j2k_codestream_exit,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
    return false;
  }
  return true;
}

/**
 * Destroys a jpeg2000 file decompressor.
 *
 * @param  jp2    a jpeg2000 file decompressor.
 */
void jp2_destroy(opj_jp2_t *jp2)
{
  if
    (jp2)
  {
    /* destroy the J2K codec */
    j2k_destroy(jp2->j2k);
    jp2->j2k = 00;
    if
      (jp2->comps)
    {
      opj_free(jp2->comps);
      jp2->comps = 00;
    }
    if
      (jp2->cl)
    {
      opj_free(jp2->cl);
      jp2->cl = 00;
    }
    if
      (jp2->m_validation_list)
    {
      opj_procedure_list_destroy(jp2->m_validation_list);
      jp2->m_validation_list = 00;
    }
    if
      (jp2->m_procedure_list)
    {
      opj_procedure_list_destroy(jp2->m_procedure_list);
      jp2->m_procedure_list = 00;
    }
    opj_free(jp2);
  }
}





/* ----------------------------------------------------------------------- */
/* JP2 encoder interface                                             */
/* ----------------------------------------------------------------------- */

opj_jp2_t* jp2_create(bool p_is_decoder)
{
  opj_jp2_t *jp2 = (opj_jp2_t*)opj_malloc(sizeof(opj_jp2_t));
  if
    (jp2)
  {
    memset(jp2,0,sizeof(opj_jp2_t));
    /* create the J2K codec */
    if
      (! p_is_decoder)
    {
      jp2->j2k = j2k_create_compress();
    }
    else
    {
      jp2->j2k = j2k_create_decompress();
    }
    if
      (jp2->j2k == 00)
    {
      jp2_destroy(jp2);
      return 00;
    }
    // validation list creation
    jp2->m_validation_list = opj_procedure_list_create();
    if
      (! jp2->m_validation_list)
    {
      jp2_destroy(jp2);
      return 00;
    }

    // execution list creation
    jp2->m_procedure_list = opj_procedure_list_create();
    if
      (! jp2->m_procedure_list)
    {
      jp2_destroy(jp2);
      return 00;
    }
  }
  return jp2;
}

/**
 * Excutes the given procedures on the given codec.
 *
 * @param  p_procedure_list  the list of procedures to execute
 * @param  jp2          the jpeg2000 file codec to execute the procedures on.
 * @param  cio          the stream to execute the procedures on.
 * @param  p_manager      the user manager.
 *
 * @return  true        if all the procedures were successfully executed.
 */
bool jp2_exec (
          opj_jp2_t * jp2,
          opj_procedure_list_t * p_procedure_list,
          opj_stream_private_t *cio,
          opj_event_mgr_t * p_manager
          )
{
  bool (** l_procedure) (opj_jp2_t * jp2,opj_stream_private_t *,opj_event_mgr_t *) = 00;
  bool l_result = true;
  unsigned int l_nb_proc, i;

  // preconditions
  assert(p_procedure_list != 00);
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);

  l_nb_proc = opj_procedure_list_get_nb_procedures(p_procedure_list);
  l_procedure = (bool (**) (opj_jp2_t * jp2,opj_stream_private_t *,opj_event_mgr_t *)) opj_procedure_list_get_first_procedure(p_procedure_list);
  for
    (i=0;i<l_nb_proc;++i)
  {
    l_result = l_result && (*l_procedure) (jp2,cio,p_manager);
    ++l_procedure;
  }
  // and clear the procedure list at the end.
  opj_procedure_list_clear(p_procedure_list);
  return l_result;
}


/**
 * Starts a compression scheme, i.e. validates the codec parameters, writes the header.
 *
 * @param  jp2    the jpeg2000 file codec.
 * @param  cio    the stream object.
 *
 * @return true if the codec is valid.
 */
bool jp2_start_compress(opj_jp2_t *jp2,  struct opj_stream_private *cio,opj_image_t * p_image, struct opj_event_mgr * p_manager)
{
  // preconditions
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);

  /* customization of the validation */
  jp2_setup_encoding_validation (jp2);

  /* validation of the parameters codec */
  if
    (! jp2_exec(jp2,jp2->m_validation_list,cio,p_manager))
  {
    return false;
  }

  /* customization of the encoding */
  jp2_setup_header_writting(jp2);

  /* write header */
  if
    (! jp2_exec (jp2,jp2->m_procedure_list,cio,p_manager))
  {
    return false;
  }
  return j2k_start_compress(jp2->j2k,cio,p_image,p_manager);
}

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
                opj_image_t ** p_image,
                OPJ_INT32 * p_tile_x0,
                OPJ_INT32 * p_tile_y0,
                OPJ_UINT32 * p_tile_width,
                OPJ_UINT32 * p_tile_height,
                OPJ_UINT32 * p_nb_tiles_x,
                OPJ_UINT32 * p_nb_tiles_y,
                struct opj_stream_private *cio,
                struct opj_event_mgr * p_manager
              )
{
  // preconditions
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);

  /* customization of the validation */
  jp2_setup_decoding_validation (jp2);

  /* customization of the encoding */
  jp2_setup_header_reading(jp2);

  /* validation of the parameters codec */
  if
    (! jp2_exec(jp2,jp2->m_validation_list,cio,p_manager))
  {
    return false;
  }

  /* read header */
  if
    (! jp2_exec (jp2,jp2->m_procedure_list,cio,p_manager))
  {
    return false;
  }
  return j2k_read_header(
    jp2->j2k,
    p_image,
    p_tile_x0,
    p_tile_y0,
    p_tile_width,
    p_tile_height,
    p_nb_tiles_x,
    p_nb_tiles_y,
    cio,
    p_manager);
}

/**
 * Ends the decompression procedures and possibiliy add data to be read after the
 * codestream.
 */
bool jp2_end_decompress(opj_jp2_t *jp2, opj_stream_private_t *cio, opj_event_mgr_t * p_manager)
{
  // preconditions
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);

  /* customization of the end encoding */
  jp2_setup_end_header_reading(jp2);

  /* write header */
  if
    (! jp2_exec (jp2,jp2->m_procedure_list,cio,p_manager))
  {
    return false;
  }
  return j2k_end_decompress(jp2->j2k, cio, p_manager);
}


/**
 * Ends the compression procedures and possibiliy add data to be read after the
 * codestream.
 */
bool jp2_end_compress(opj_jp2_t *jp2, opj_stream_private_t *cio, opj_event_mgr_t * p_manager)
{
  // preconditions
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);

  /* customization of the end encoding */
  jp2_setup_end_header_writting(jp2);

  if
    (! j2k_end_compress(jp2->j2k,cio,p_manager))
  {
    return false;
  }
  /* write header */
  return jp2_exec (jp2,jp2->m_procedure_list,cio,p_manager);
}

/**
Encode an image into a JPEG-2000 file stream
@param jp2 JP2 compressor handle
@param cio Output buffer stream
@param image Image to encode
@param cstr_info Codestream information structure if required, NULL otherwise
@return Returns true if successful, returns false otherwise
*/
bool jp2_encode(opj_jp2_t *jp2, struct opj_stream_private *cio, struct opj_event_mgr * p_manager)
{
  return j2k_encode(jp2->j2k,cio,p_manager);
}
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
          )
{
  return j2k_write_tile (p_jp2->j2k,p_tile_index,p_data,p_data_size,p_stream,p_manager);
}

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
          opj_stream_private_t *p_stream,
          opj_event_mgr_t * p_manager
          )
{
  return j2k_decode_tile (p_jp2->j2k,p_tile_index,p_data,p_data_size,p_stream,p_manager);
}
/**
 * Reads a tile header.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
bool jp2_read_tile_header (
           opj_jp2_t * p_jp2,
           OPJ_UINT32 * p_tile_index,
           OPJ_UINT32 * p_data_size,
           OPJ_INT32 * p_tile_x0,
           OPJ_INT32 * p_tile_y0,
           OPJ_INT32 * p_tile_x1,
           OPJ_INT32 * p_tile_y1,
           OPJ_UINT32 * p_nb_comps,
           bool * p_go_on,
           opj_stream_private_t *p_stream,
           opj_event_mgr_t * p_manager
          )
{
  return j2k_read_tile_header (p_jp2->j2k,
                p_tile_index,
                p_data_size,
                p_tile_x0,
                p_tile_y0,
                p_tile_x1,
                p_tile_y1,
                p_nb_comps,
                p_go_on,
                p_stream,
                p_manager);
}

/**
 * Sets up the procedures to do on writting header after the codestream.
 * Developpers wanting to extend the library can add their own writting procedures.
 */
void jp2_setup_end_header_writting (opj_jp2_t *jp2)
{
  // preconditions
  assert(jp2 != 00);

  opj_procedure_list_add_procedure(jp2->m_procedure_list,(void*)jp2_write_jp2c );
  /* DEVELOPER CORNER, add your custom procedures */
}

/**
 * Sets up the procedures to do on reading header.
 * Developpers wanting to extend the library can add their own writting procedures.
 */
void jp2_setup_header_reading (opj_jp2_t *jp2)
{
  // preconditions
  assert(jp2 != 00);

  opj_procedure_list_add_procedure(jp2->m_procedure_list,(void*)jp2_read_header_procedure );
  /* DEVELOPER CORNER, add your custom procedures */
}

/**
 * Sets up the procedures to do on reading header after the codestream.
 * Developpers wanting to extend the library can add their own writting procedures.
 */
void jp2_setup_end_header_reading (opj_jp2_t *jp2)
{
  // preconditions
  assert(jp2 != 00);
  opj_procedure_list_add_procedure(jp2->m_procedure_list,(void*)jp2_read_header_procedure );
  /* DEVELOPER CORNER, add your custom procedures */
}


/**
 * The default validation procedure without any extension.
 *
 * @param  jp2        the jpeg2000 codec to validate.
 * @param  cio        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool jp2_default_validation (
                opj_jp2_t * jp2,
                opj_stream_private_t *cio,
                opj_event_mgr_t * p_manager
              )
{
  bool l_is_valid = true;
  unsigned int i;

  // preconditions
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);
  /* JPEG2000 codec validation */
  /*TODO*/

  /* STATE checking */
  /* make sure the state is at 0 */
  l_is_valid &= (jp2->jp2_state == JP2_STATE_NONE);
  /* make sure not reading a jp2h ???? WEIRD */
  l_is_valid &= (jp2->jp2_img_state == JP2_IMG_STATE_NONE);

  /* POINTER validation */
  /* make sure a j2k codec is present */
  l_is_valid &= (jp2->j2k != 00);
  /* make sure a procedure list is present */
  l_is_valid &= (jp2->m_procedure_list != 00);
  /* make sure a validation list is present */
  l_is_valid &= (jp2->m_validation_list != 00);

  /* PARAMETER VALIDATION */
  /* number of components */
  l_is_valid &= (jp2->numcl > 0);
  /* width */
  l_is_valid &= (jp2->h > 0);
  /* height */
  l_is_valid &= (jp2->w > 0);
  /* precision */
  for
    (i = 0; i < jp2->numcomps; ++i)
  {
    l_is_valid &= (jp2->comps[i].bpcc > 0);
  }
  /* METH */
  l_is_valid &= ((jp2->meth > 0) && (jp2->meth < 3));



  /* stream validation */
  /* back and forth is needed */
  l_is_valid &= opj_stream_has_seek(cio);

  return l_is_valid;

}

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
void jp2_setup_encoding_validation (opj_jp2_t *jp2)
{
  // preconditions
  assert(jp2 != 00);
  opj_procedure_list_add_procedure(jp2->m_validation_list, (void*)jp2_default_validation);
  /* DEVELOPER CORNER, add your custom validation procedure */
}

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
void jp2_setup_decoding_validation (opj_jp2_t *jp2)
{
  // preconditions
  assert(jp2 != 00);
  /* DEVELOPER CORNER, add your custom validation procedure */
}

/**
 * Sets up the procedures to do on writting header. Developpers wanting to extend the library can add their own writting procedures.
 */
void jp2_setup_header_writting (opj_jp2_t *jp2)
{
  // preconditions
  assert(jp2 != 00);
  opj_procedure_list_add_procedure(jp2->m_procedure_list,(void*)jp2_write_jp );
  opj_procedure_list_add_procedure(jp2->m_procedure_list,(void*)jp2_write_ftyp );
  opj_procedure_list_add_procedure(jp2->m_procedure_list,(void*)jp2_write_jp2h );
  opj_procedure_list_add_procedure(jp2->m_procedure_list,(void*)jp2_skip_jp2c );

  /* DEVELOPER CORNER, insert your custom procedures */

}


/**
 * Skips the Jpeg2000 Codestream Header box - JP2C Header box.
 *
 * @param  cio      the stream to write data to.
 * @param  jp2      the jpeg2000 file codec.
 * @param  p_manager  user event manager.
 *
 * @return true if writting was successful.
*/
bool jp2_skip_jp2c(
            opj_jp2_t *jp2,
            struct opj_stream_private *cio,
            struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(jp2 != 00);
  assert(cio != 00);
  assert(p_manager != 00);

  jp2->j2k_codestream_offset = opj_stream_tell(cio);
  if
    (opj_stream_skip(cio,8,p_manager) != 8)
  {
    return false;
  }
  return true;
}

struct opj_image * jp2_decode(opj_jp2_t *jp2, struct opj_stream_private *cio, struct opj_event_mgr * p_manager)
{
  /* J2K decoding */
  struct opj_image * image = j2k_decode(jp2->j2k, cio, p_manager);
  if
    (!image)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Failed to decode J2K image\n");
    return false;
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
  return image;
}

void jp2_setup_encoder(opj_jp2_t *jp2, opj_cparameters_t *parameters, opj_image_t *image,opj_event_mgr_t * p_manager)
{
  unsigned int i;
  int depth_0, sign;

  if(!jp2 || !parameters || !image)
    return;

  /* setup the J2K codec */
  /* ------------------- */

  /* Check if number of components respects standard */
  if (image->numcomps < 1 || image->numcomps > 16384) {
    opj_event_msg(p_manager, EVT_ERROR, "Invalid number of components specified while setting up JP2 encoder\n");
    return;
  }

  j2k_setup_encoder(jp2->j2k, parameters, image,p_manager);

  /* setup the JP2 codec */
  /* ------------------- */

  /* Profile box */

  jp2->brand = JP2_JP2;  /* BR */
  jp2->minversion = 0;  /* MinV */
  jp2->numcl = 1;
  jp2->cl = (unsigned int*) opj_malloc(jp2->numcl * sizeof(unsigned int));
  jp2->cl[0] = JP2_JP2;  /* CL0 : JP2 */

  /* Image Header box */

  jp2->numcomps = image->numcomps;  /* NC */
  jp2->comps = (opj_jp2_comps_t*) opj_malloc(jp2->numcomps * sizeof(opj_jp2_comps_t));
  jp2->h = image->y1 - image->y0;    /* HEIGHT */
  jp2->w = image->x1 - image->x0;    /* WIDTH */
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
  jp2->C = 7;      /* C : Always 7 */
  jp2->UnkC = 0;    /* UnkC, colorspace specified in colr box */
  jp2->IPR = 0;    /* IPR, no intellectual property */

  /* BitsPerComponent box */

  for (i = 0; i < image->numcomps; i++) {
    jp2->comps[i].bpcc = image->comps[i].prec - 1 + (image->comps[i].sgnd << 7);
  }

  /* Colour Specification box */

  if ((image->numcomps == 1 || image->numcomps == 3) && (jp2->bpc != 255)) {
    jp2->meth = 1;  /* METH: Enumerated colourspace */
  } else {
    jp2->meth = 2;  /* METH: Restricted ICC profile */
  }
  if (jp2->meth == 1) {
    if (image->color_space == 1)
      jp2->enumcs = 16;  /* sRGB as defined by IEC 61966𣇻 */
    else if (image->color_space == 2)
      jp2->enumcs = 17;  /* greyscale */
    else if (image->color_space == 3)
      jp2->enumcs = 18;  /* YUV */
  } else {
    jp2->enumcs = 0;    /* PROFILE (??) */
  }
  jp2->precedence = 0;  /* PRECEDENCE */
  jp2->approx = 0;    /* APPROX */

}

void jp2_setup_decoder(opj_jp2_t *jp2, opj_dparameters_t *parameters)
{
  if(!jp2 || !parameters)
    return;

  /* setup the J2K codec */
  /* ------------------- */
  j2k_setup_decoder(jp2->j2k, parameters);
}
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
      )
{
  return j2k_set_decode_area(p_jp2->j2k,p_start_x,p_start_y,p_end_x,p_end_y,p_manager);
}

#if 0





static void jp2_write_url(opj_cio_t *cio, char *Idx_file) {
  unsigned int i;
  opj_jp2_box_t box;

  box.init_pos = cio_tell(cio);
  cio_skip(cio, 4);
  cio_write(cio, JP2_URL, 4);  /* DBTL */
  cio_write(cio, 0, 1);    /* VERS */
  cio_write(cio, 0, 3);    /* FLAG */

  if(Idx_file) {
    for (i = 0; i < strlen(Idx_file); i++) {
      cio_write(cio, Idx_file[i], 1);
    }
  }

  box.length = cio_tell(cio) - box.init_pos;
  cio_seek(cio, box.init_pos);
  cio_write(cio, box.length, 4);  /* L */
  cio_seek(cio, box.init_pos + box.length);
}
#endif
