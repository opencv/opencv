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

#ifndef __JPT_H
#define __JPT_H
/**
@file jpt.h
@brief JPT-stream reader (JPEG 2000, JPIP)

JPT-stream functions are implemented in J2K.C.
*/
#include "openjpeg.h"
struct opj_stream_private;
struct opj_event_mgr;
/**
Message Header JPT stream structure
*/
typedef struct opj_jpt_msg_header
{
  /** In-class Identifier */
  OPJ_UINT32 Id;
  /** Last byte information */
  OPJ_UINT32 last_byte;
  /** Class Identifier */
  OPJ_UINT32 Class_Id;
  /** CSn : index identifier */
  OPJ_UINT32 CSn_Id;
  /** Message offset */
  OPJ_UINT32 Msg_offset;
  /** Message length */
  OPJ_UINT32 Msg_length;
  /** Auxiliary for JPP case */
  OPJ_UINT32 Layer_nb;
} opj_jpt_msg_header_t;

/* ----------------------------------------------------------------------- */

/**
Initialize the value of the message header structure
@param header Message header structure
*/
void jpt_init_msg_header(opj_jpt_msg_header_t * header);

/**
 * Read the message header for a JPP/JPT - stream
 * @param  p_cio    stream handle
 * @param  header    JPT Message header structure
 * @param  p_manager  user event manager to display nice messages.
*/
bool jpt_read_msg_header(
    struct opj_stream_private * p_cio,
    opj_jpt_msg_header_t * p_header,
    OPJ_UINT32 * p_nb_bytes_read,
    struct opj_event_mgr * p_manager);

#endif
