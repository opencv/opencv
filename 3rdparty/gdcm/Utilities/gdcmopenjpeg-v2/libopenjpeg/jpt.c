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

#include "jpt.h"
#include "openjpeg.h"
#include "cio.h"
#include "event.h"
/*
 * Read the information contains in VBAS [JPP/JPT stream message header]
 * Store information (7 bits) in value
 * @param p_cio the stream to read from.
 * @param p_value the data to update
 * @return the nb of bytes read or -1 if an io error occurred.
 */
bool jpt_read_VBAS_info(opj_stream_private_t * p_cio, OPJ_UINT32 * p_nb_bytes_read, OPJ_UINT32 * p_value, opj_event_mgr_t * p_manager)
{
  OPJ_BYTE l_elmt;
  OPJ_UINT32 l_nb_bytes_read = 0;

  // read data till the MSB of the current byte is 1.
  // concatenate 7 bits of data, last bit is finish flag

  // read data from the stream

  if
    (opj_stream_read_data(p_cio,&l_elmt,1,p_manager) != 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data.\n");
    return false;
  }
  ++l_nb_bytes_read;

  // is the MSB equal to 1 ?
  while
    (l_elmt & 0x80)
  {
    // concatenate 7 bits of data, last bit is finish flag
    *p_value = (*p_value  << 7) | (l_elmt & 0x7f);
    if
      (opj_stream_read_data(p_cio,&l_elmt,1,p_manager) != 1)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data.\n");
      return false;
    }
    ++l_nb_bytes_read;
  }
  // concatenate 7 bits of data, last bit is finish flag
  *p_value = (*p_value  << 7) | (l_elmt & 0x7f);
  * p_nb_bytes_read = l_nb_bytes_read;
  return true;
}

/*
 * Initialize the value of the message header structure
 *
 */
void jpt_init_msg_header(opj_jpt_msg_header_t * header)
{
  header->Id = 0;    /* In-class Identifier    */
  header->last_byte = 0;  /* Last byte information  */
  header->Class_Id = 0;    /* Class Identifier       */
  header->CSn_Id = 0;    /* CSn : index identifier */
  header->Msg_offset = 0;  /* Message offset         */
  header->Msg_length = 0;  /* Message length         */
  header->Layer_nb = 0;    /* Auxiliary for JPP case */
}

/*
 * Re-initialize the value of the message header structure
 *
 * Only parameters always present in message header
 *
 */
void jpt_reinit_msg_header(opj_jpt_msg_header_t * header)
{
  header->Id = 0;    /* In-class Identifier    */
  header->last_byte = 0;  /* Last byte information  */
  header->Msg_offset = 0;  /* Message offset         */
  header->Msg_length = 0;  /* Message length         */
}

/*
 * Read the message header for a JPP/JPT - stream
 *
 */
bool jpt_read_msg_header(opj_stream_private_t *cio, opj_jpt_msg_header_t *header, OPJ_UINT32 * p_nb_bytes_read, opj_event_mgr_t * p_manager)
{
  OPJ_BYTE elmt, Class = 0, CSn = 0;
  OPJ_UINT32 l_nb_bytes_read = 0;
  OPJ_UINT32 l_last_nb_bytes_read;


  jpt_reinit_msg_header(header);

  /* ------------- */
  /* VBAS : Bin-ID */
  /* ------------- */
  if
    (opj_stream_read_data(cio,&elmt,1,p_manager) != 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Forbidden value encounter in message header !!\n");
    return false;
  }
  ++l_nb_bytes_read;

  /* See for Class and CSn */
  switch ((elmt >> 5) & 0x03)
  {
    case 0:
      opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data!!!\n");
      break;
    case 1:
      Class = 0;
      CSn = 0;
      break;
    case 2:
      Class = 1;
      CSn = 0;
      break;
    case 3:
      Class = 1;
      CSn = 1;
      break;
    default:
      break;
  }

  /* see information on bits 'c' [p 10 : A.2.1 general, ISO/IEC FCD 15444-9] */
  if
    (((elmt >> 4) & 0x01) == 1)
  {
    header->last_byte = 1;
  }

  /* In-class identifier */
  header->Id |= (elmt & 0x0f);
  if
    ((elmt >> 7) == 1)
  {
    l_last_nb_bytes_read = 0;
    if
      (! jpt_read_VBAS_info(cio, &l_last_nb_bytes_read, &(header->Id), p_manager))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data!!!\n");
      return false;
    }
    l_nb_bytes_read += l_last_nb_bytes_read;
  }

  /* ------------ */
  /* VBAS : Class */
  /* ------------ */
  if (Class == 1)
  {
    header->Class_Id = 0;
    l_last_nb_bytes_read = 0;
    if
      (! jpt_read_VBAS_info(cio, &l_last_nb_bytes_read, &(header->Class_Id), p_manager))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data!!!\n");
      return false;
    }
    l_nb_bytes_read += l_last_nb_bytes_read;
  }

  /* ---------- */
  /* VBAS : CSn */
  /* ---------- */
  if (CSn == 1)
  {
    header->CSn_Id = 0;
    l_last_nb_bytes_read = 0;
    if
      (! jpt_read_VBAS_info(cio, &l_last_nb_bytes_read, &(header->CSn_Id), p_manager))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data!!!\n");
      return false;
    }
    l_nb_bytes_read += l_last_nb_bytes_read;
  }

  /* ----------------- */
  /* VBAS : Msg_offset */
  /* ----------------- */
  l_last_nb_bytes_read = 0;
  if
    (! jpt_read_VBAS_info(cio, &l_last_nb_bytes_read, &(header->Msg_offset), p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data!!!\n");
    return false;
  }
  l_nb_bytes_read += l_last_nb_bytes_read;

  /* ----------------- */
  /* VBAS : Msg_length */
  /* ----------------- */
  l_last_nb_bytes_read = 0;
  if
    (! jpt_read_VBAS_info(cio, &l_last_nb_bytes_read, &(header->Msg_length), p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data!!!\n");
    return false;
  }
  l_nb_bytes_read += l_last_nb_bytes_read;

  /* ---------- */
  /* VBAS : Aux */
  /* ---------- */
  if ((header->Class_Id & 0x01) == 1)
  {
    header->Layer_nb = 0;
    if
      (! jpt_read_VBAS_info(cio, &l_last_nb_bytes_read, &(header->Layer_nb), p_manager))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error trying to read a byte of data!!!\n");
      return false;
    }
    l_nb_bytes_read += l_last_nb_bytes_read;
  }
  * p_nb_bytes_read = l_nb_bytes_read;
  return true;
}
