/*
 * $Id: thix_manager.c 897 2011-08-28 21:43:57Z Kaori.Hagihara@gmail.com $
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

#include "opj_includes.h"



int opj_write_thix(int coff, opj_codestream_info_t cstr_info,
                   opj_stream_private_t *cio,
                   opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [4];
    int i;
    int tileno;
    opj_jp2_box_t *box;
    OPJ_UINT32 len;
    OPJ_OFF_T lenp;

    lenp = 0;
    box = (opj_jp2_box_t *)opj_calloc((size_t)(cstr_info.tw * cstr_info.th),
                                      sizeof(opj_jp2_box_t));
    if (box == NULL) {
        return 0;
    }
    for (i = 0; i < 2 ; i++) {
        if (i) {
            opj_stream_seek(cio, lenp, p_manager);
        }

        lenp = opj_stream_tell(cio);
        opj_stream_skip(cio, 4, p_manager);             /* L [at the end] */
        opj_write_bytes(l_data_header, JPIP_THIX, 4); /* THIX */
        opj_stream_write_data(cio, l_data_header, 4, p_manager);

        opj_write_manf(i, cstr_info.tw * cstr_info.th, box, cio, p_manager);

        for (tileno = 0; tileno < cstr_info.tw * cstr_info.th; tileno++) {
            box[tileno].length = (OPJ_UINT32)opj_write_tilemhix(coff, cstr_info, tileno,
                                 cio, p_manager);
            box[tileno].type = JPIP_MHIX;
        }

        len = (OPJ_UINT32)(opj_stream_tell(cio) - lenp);
        opj_stream_seek(cio, lenp, p_manager);
        opj_write_bytes(l_data_header, len, 4); /* L              */
        opj_stream_write_data(cio, l_data_header, 4, p_manager);
        opj_stream_seek(cio, lenp + len, p_manager);

    }

    opj_free(box);

    return (int)len;
}

/*
 * Write tile-part headers mhix box
 *
 * @param[in] coff      offset of j2k codestream
 * @param[in] cstr_info codestream information
 * @param[in] tileno    tile number
 * @param[in] cio       file output handle
 * @return              length of mhix box
 */
int opj_write_tilemhix(int coff, opj_codestream_info_t cstr_info, int tileno,
                       opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [8];
    int i;
    opj_tile_info_t tile;
    opj_tp_info_t tp;
    opj_marker_info_t *marker;
    OPJ_UINT32 len;
    OPJ_OFF_T lenp;

    lenp = opj_stream_tell(cio);
    opj_stream_skip(cio, 4,
                    p_manager);               /* L [at the end]                    */
    opj_write_bytes(l_data_header, JPIP_MHIX,
                    4);     /* MHIX                              */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);

    tile = cstr_info.tile[tileno];
    tp = tile.tp[0];

    opj_write_bytes(l_data_header,
                    (OPJ_UINT32)(tp.tp_end_header - tp.tp_start_pos + 1),
                    8);        /* TLEN                              */
    opj_stream_write_data(cio, l_data_header, 8, p_manager);

    marker = cstr_info.tile[tileno].marker;

    for (i = 0; i < cstr_info.tile[tileno].marknum;
            i++) {        /* Marker restricted to 1 apparition */
        opj_write_bytes(l_data_header, marker[i].type, 2);
        opj_write_bytes(l_data_header + 2, 0, 2);
        opj_stream_write_data(cio, l_data_header, 4, p_manager);
        opj_write_bytes(l_data_header, (OPJ_UINT32)(marker[i].pos - coff), 8);
        opj_stream_write_data(cio, l_data_header, 8, p_manager);
        opj_write_bytes(l_data_header, (OPJ_UINT32)marker[i].len, 2);
        opj_stream_write_data(cio, l_data_header, 2, p_manager);
    }

    /*  free( marker);*/

    len = (OPJ_UINT32)(opj_stream_tell(cio) - lenp);
    opj_stream_seek(cio, lenp, p_manager);
    opj_write_bytes(l_data_header, len, 4); /* L  */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);
    opj_stream_seek(cio, lenp + len, p_manager);

    return (int)len;
}
