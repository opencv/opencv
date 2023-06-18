/*
 * $Id: cidx_manager.h 897 2011-08-28 21:43:57Z Kaori.Hagihara@gmail.com $
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
 *  \brief Modification of jpip.h from 2KAN indexer
 */


#ifndef  CIDX_MANAGER_H_
# define CIDX_MANAGER_H_

#include "openjpeg.h"


/*
 * Write Codestream index box (superbox)
 *
 * @param[in] offset    offset of j2k codestream
 * @param[in] cio       file output handle
 * @param[in] image     image data
 * @param[in] cstr_info codestream information
 * @param[in] j2klen    length of j2k codestream
 * @return              length of cidx box
 */
int opj_write_cidx(int offset, opj_stream_private_t *cio,
                   opj_codestream_info_t cstr_info, int j2klen,
                   opj_event_mgr_t * p_manager);

/*
 * Check if EPH option is used
 *
 * @param[in] coff    offset of j2k codestream
 * @param[in] markers marker information
 * @param[in] marknum number of markers
 * @param[in] cio     file output handle
 * @return            true if EPH is used
 */
OPJ_BOOL opj_check_EPHuse(int coff, opj_marker_info_t *markers, int marknum,
                          opj_stream_private_t *cio,
                          opj_event_mgr_t * p_manager);

#endif      /* !CIDX_MANAGER_H_ */
