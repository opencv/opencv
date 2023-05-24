/*
 * $Id: cidx_manager.c 897 2011-08-28 21:43:57Z Kaori.Hagihara@gmail.com $
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

#include "opj_includes.h"


/*
 * Write CPTR Codestream finder box
 *
 * @param[in] coff offset of j2k codestream
 * @param[in] clen length of j2k codestream
 * @param[in] cio  file output handle
 */

void opj_write_cptr(int coff, int clen, opj_stream_private_t *cio,
                    opj_event_mgr_t * p_manager);





int opj_write_cidx(int offset, opj_stream_private_t *cio,
                   opj_codestream_info_t cstr_info, int j2klen,
                   opj_event_mgr_t * p_manager)
{
    int i;
    OPJ_OFF_T lenp;
    OPJ_UINT32 len;
    opj_jp2_box_t *box;
    int num_box = 0;
    OPJ_BOOL  EPHused;
    OPJ_BYTE l_data_header [4];

    lenp = -1;
    box = (opj_jp2_box_t *)opj_calloc(32, sizeof(opj_jp2_box_t));
    if (box == NULL) {
        return 0;
    }
    for (i = 0; i < 2; i++) {

        if (i) {
            opj_stream_seek(cio, lenp, p_manager);
        }


        lenp = opj_stream_tell(cio);

        opj_stream_skip(cio, 4, p_manager); /* L [at the end] */

        opj_write_bytes(l_data_header, JPIP_CIDX, 4); /* CIDX */
        opj_stream_write_data(cio, l_data_header, 4, p_manager);

        opj_write_cptr(offset, cstr_info.codestream_size, cio, p_manager);

        opj_write_manf(i, num_box, box, cio, p_manager);

        num_box = 0;
        box[num_box].length = (OPJ_UINT32)opj_write_mainmhix(offset, cstr_info, cio,
                              p_manager);
        box[num_box].type = JPIP_MHIX;
        num_box++;

        box[num_box].length = (OPJ_UINT32)opj_write_tpix(offset, cstr_info, j2klen, cio,
                              p_manager);
        box[num_box].type = JPIP_TPIX;
        num_box++;

        box[num_box].length = (OPJ_UINT32)opj_write_thix(offset, cstr_info, cio,
                              p_manager);
        box[num_box].type = JPIP_THIX;
        num_box++;

        EPHused = opj_check_EPHuse(offset, cstr_info.marker, cstr_info.marknum, cio,
                                   p_manager);

        box[num_box].length = (OPJ_UINT32)opj_write_ppix(offset, cstr_info, EPHused,
                              j2klen, cio, p_manager);
        box[num_box].type = JPIP_PPIX;
        num_box++;

        box[num_box].length = (OPJ_UINT32)opj_write_phix(offset, cstr_info, EPHused,
                              j2klen, cio, p_manager);
        box[num_box].type = JPIP_PHIX;
        num_box++;

        len = (OPJ_UINT32)(opj_stream_tell(cio) - lenp);
        opj_stream_seek(cio, lenp, p_manager);
        opj_write_bytes(l_data_header, len, 4); /* L  */
        opj_stream_write_data(cio, l_data_header, 4, p_manager);
        opj_stream_seek(cio, lenp + len, p_manager);
    }

    opj_free(box);

    return (int)len;
}



void opj_write_cptr(int coff, int clen, opj_stream_private_t *cio,
                    opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [3 * 8];
    OPJ_UINT32 len;
    OPJ_OFF_T lenp;


    lenp = opj_stream_tell(cio);
    opj_stream_skip(cio, 4, p_manager);                /* L [at the end]     */
    opj_write_bytes(l_data_header, JPIP_CPTR, 4);    /* T                  */
    opj_write_bytes(l_data_header + 4, 0, 2);          /* DR  A PRECISER !!  */
    opj_write_bytes(l_data_header + 6, 0, 2);          /* CONT               */
    opj_write_bytes(l_data_header + 8, (OPJ_UINT32)coff,
                    8);   /* COFF A PRECISER !! */
    opj_write_bytes(l_data_header + 16, (OPJ_UINT32)clen,
                    8);   /* CLEN               */
    opj_stream_write_data(cio, l_data_header, 3 * 8, p_manager);

    len = (OPJ_UINT32)(opj_stream_tell(cio) - lenp);
    opj_stream_seek(cio, lenp, p_manager);
    opj_write_bytes(l_data_header, len, 4);         /* L                  */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);
    opj_stream_seek(cio, lenp + len, p_manager);

}



void opj_write_manf(int second,
                    int v,
                    opj_jp2_box_t *box,
                    opj_stream_private_t *cio,
                    opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [4];
    int i;
    OPJ_UINT32 len;
    OPJ_OFF_T lenp;

    lenp = opj_stream_tell(cio);
    opj_stream_skip(cio, 4, p_manager);              /* L [at the end]     */
    opj_write_bytes(l_data_header, JPIP_MANF, 4);    /* T                  */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);

    if (second) {                         /* Write only during the second pass */
        for (i = 0; i < v; i++) {
            opj_write_bytes(l_data_header, box[i].length,
                            4);  /* Box length                     */
            opj_stream_write_data(cio, l_data_header, 4, p_manager);
            opj_write_bytes(l_data_header, box[i].type,
                            4);  /* Box type                       */
            opj_stream_write_data(cio, l_data_header, 4, p_manager);
        }
    }

    len = (OPJ_UINT32)(opj_stream_tell(cio) - lenp);
    opj_stream_seek(cio, lenp, p_manager);
    opj_write_bytes(l_data_header, len, 4);/* L                                 */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);
    opj_stream_seek(cio, lenp + len, p_manager);
}


int opj_write_mainmhix(int coff, opj_codestream_info_t cstr_info,
                       opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [8];
    OPJ_UINT32 i;
    OPJ_UINT32 len;
    OPJ_OFF_T lenp;

    lenp = opj_stream_tell(cio);
    opj_stream_skip(cio, 4,
                    p_manager);               /* L [at the end]                    */
    opj_write_bytes(l_data_header, JPIP_MHIX,
                    4);     /* MHIX                              */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);

    opj_write_bytes(l_data_header,
                    (OPJ_UINT32)(cstr_info.main_head_end - cstr_info.main_head_start + 1),
                    8);        /* TLEN                              */
    opj_stream_write_data(cio, l_data_header, 8, p_manager);

    for (i = 1; i < (OPJ_UINT32)cstr_info.marknum;
            i++) {  /* Marker restricted to 1 apparition, skip SOC marker */
        opj_write_bytes(l_data_header, cstr_info.marker[i].type, 2);
        opj_write_bytes(l_data_header + 2, 0, 2);
        opj_stream_write_data(cio, l_data_header, 4, p_manager);
        opj_write_bytes(l_data_header, (OPJ_UINT32)(cstr_info.marker[i].pos - coff), 8);
        opj_stream_write_data(cio, l_data_header, 8, p_manager);
        opj_write_bytes(l_data_header, (OPJ_UINT32)cstr_info.marker[i].len, 2);
        opj_stream_write_data(cio, l_data_header, 2, p_manager);
    }

    len = (OPJ_UINT32)(opj_stream_tell(cio) - lenp);
    opj_stream_seek(cio, lenp, p_manager);
    opj_write_bytes(l_data_header, len, 4); /* L  */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);
    opj_stream_seek(cio, lenp + len, p_manager);

    return (int)len;
}

OPJ_BOOL opj_check_EPHuse(int coff, opj_marker_info_t *markers, int marknum,
                          opj_stream_private_t *cio,
                          opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [4];
    OPJ_BOOL EPHused = OPJ_FALSE;
    int i = 0;
    OPJ_OFF_T org_pos;
    unsigned int Scod;

    for (i = 0; i < marknum; i++) {
        if (markers[i].type == J2K_MS_COD) {
            org_pos = opj_stream_tell(cio);
            opj_stream_seek(cio, coff + markers[i].pos + 2, p_manager);

            opj_stream_read_data(cio, l_data_header, 1, p_manager);
            opj_read_bytes(l_data_header, &Scod, 1);
            if (((Scod >> 2) & 1)) {
                EPHused = OPJ_TRUE;
            }
            opj_stream_seek(cio, org_pos, p_manager);

            break;
        }
    }
    return EPHused;
}
