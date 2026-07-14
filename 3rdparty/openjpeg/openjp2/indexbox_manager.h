/*
 * $Id: indexbox_manager.h 897 2011-08-28 21:43:57Z Kaori.Hagihara@gmail.com $
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2003-2004, Yannick Verschueren
 * Copyright (c) 2010-2011, Kaori Hagihara
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

/*! \file
 *  \brief Modification of jpip.c from 2KAN indexer
 */

#ifndef  INDEXBOX_MANAGER_H_
# define INDEXBOX_MANAGER_H_

#include "openjpeg.h"
#include "j2k.h" /* needed to use jp2.h */
#include "jp2.h"

#define JPIP_CIDX 0x63696478   /* Codestream index                */
#define JPIP_CPTR 0x63707472   /* Codestream Finder Box           */
#define JPIP_MANF 0x6d616e66   /* Manifest Box                    */
#define JPIP_FAIX 0x66616978   /* Fragment array Index box        */
#define JPIP_MHIX 0x6d686978   /* Main Header Index Table         */
#define JPIP_TPIX 0x74706978   /* Tile-part Index Table box       */
#define JPIP_THIX 0x74686978   /* Tile header Index Table box     */
#define JPIP_PPIX 0x70706978   /* Precinct Packet Index Table box */
#define JPIP_PHIX 0x70686978   /* Packet Header index Table       */
#define JPIP_FIDX 0x66696478   /* File Index                      */
#define JPIP_FPTR 0x66707472   /* File Finder                     */
#define JPIP_PRXY 0x70727879   /* Proxy boxes                     */
#define JPIP_IPTR 0x69707472   /* Index finder box                */
#define JPIP_PHLD 0x70686c64   /* Place holder                    */


/*
 * Write tile-part Index table box (superbox)
 *
 * @param[in] coff      offset of j2k codestream
 * @param[in] cstr_info codestream information
 * @param[in] j2klen    length of j2k codestream
 * @param[in] cio       file output handle
 * @return              length of tpix box
 */
int opj_write_tpix(int coff, opj_codestream_info_t cstr_info, int j2klen,
                   opj_stream_private_t *cio,
                   opj_event_mgr_t * p_manager);


/*
 * Write tile header index table box (superbox)
 *
 * @param[in] coff      offset of j2k codestream
 * @param[in] cstr_info codestream information pointer
 * @param[in] cio       file output handle
 * @return              length of thix box
 */
int opj_write_thix(int coff, opj_codestream_info_t cstr_info,
                   opj_stream_private_t *cio, opj_event_mgr_t * p_manager);


/*
 * Write precinct packet index table box (superbox)
 *
 * @param[in] coff      offset of j2k codestream
 * @param[in] cstr_info codestream information
 * @param[in] EPHused   true if EPH option used
 * @param[in] j2klen    length of j2k codestream
 * @param[in] cio       file output handle
 * @return              length of ppix box
 */
int opj_write_ppix(int coff, opj_codestream_info_t cstr_info, OPJ_BOOL EPHused,
                   int j2klen, opj_stream_private_t *cio,
                   opj_event_mgr_t * p_manager);


/*
 * Write packet header index table box (superbox)
 *
 * @param[in] coff      offset of j2k codestream
 * @param[in] cstr_info codestream information
 * @param[in] EPHused   true if EPH option used
 * @param[in] j2klen    length of j2k codestream
 * @param[in] cio       file output handle
 * @return              length of ppix box
 */
int opj_write_phix(int coff, opj_codestream_info_t cstr_info, OPJ_BOOL EPHused,
                   int j2klen, opj_stream_private_t *cio,
                   opj_event_mgr_t * p_manager);

/*
 * Write manifest box (box)
 *
 * @param[in] second number to be visited
 * @param[in] v      number of boxes
 * @param[in] box    box to be manifested
 * @param[in] cio    file output handle
 */

void opj_write_manf(int second,
                    int v,
                    opj_jp2_box_t *box,
                    opj_stream_private_t *cio,
                    opj_event_mgr_t * p_manager);

/*
 * Write main header index table (box)
 *
 * @param[in] coff offset of j2k codestream
 * @param[in] cstr_info codestream information
 * @param[in] cio  file output handle
 * @return         length of mainmhix box
 */
int opj_write_mainmhix(int coff, opj_codestream_info_t cstr_info,
                       opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager);

int opj_write_phixfaix(int coff, int compno, opj_codestream_info_t cstr_info,
                       OPJ_BOOL EPHused, int j2klen, opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager);

int opj_write_ppixfaix(int coff, int compno, opj_codestream_info_t cstr_info,
                       OPJ_BOOL EPHused, int j2klen, opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager);

int opj_write_tilemhix(int coff, opj_codestream_info_t cstr_info, int tileno,
                       opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager);

int opj_write_tpixfaix(int coff, int compno, opj_codestream_info_t cstr_info,
                       int j2klen, opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager);

#endif      /* !INDEXBOX_MANAGER_H_ */
