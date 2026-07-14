/*
 * $Id: ppix_manager.c 897 2011-08-28 21:43:57Z Kaori.Hagihara@gmail.com $
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

/*
 * Write faix box of ppix
 *
 * @param[in] coff offset of j2k codestream
 * @param[in] compno    component number
 * @param[in] cstr_info codestream information
 * @param[in] EPHused   true if if EPH option used
 * @param[in] j2klen    length of j2k codestream
 * @param[in] cio       file output handle
 * @return              length of faix box
 */


int opj_write_ppix(int coff, opj_codestream_info_t cstr_info, OPJ_BOOL EPHused,
                   int j2klen, opj_stream_private_t *cio,
                   opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [4];
    int compno, i;
    opj_jp2_box_t *box;
    OPJ_OFF_T lenp;
    OPJ_UINT32 len;

    /*  printf("cstr_info.packno %d\n", cstr_info.packno); //NMAX? */

    lenp = -1;
    box = (opj_jp2_box_t *)opj_calloc((size_t)cstr_info.numcomps,
                                      sizeof(opj_jp2_box_t));
    if (box == NULL) {
        return 0;
    }
    for (i = 0; i < 2; i++) {
        if (i)

        {
            opj_stream_seek(cio, lenp, p_manager);
        }

        lenp = (OPJ_UINT32)(opj_stream_tell(cio));
        opj_stream_skip(cio, 4, p_manager);        /* L [at the end] */
        opj_write_bytes(l_data_header, JPIP_PPIX, 4); /* PPIX           */
        opj_stream_write_data(cio, l_data_header, 4, p_manager);

        opj_write_manf(i, cstr_info.numcomps, box, cio, p_manager);

        for (compno = 0; compno < cstr_info.numcomps; compno++) {
            box[compno].length = (OPJ_UINT32)opj_write_ppixfaix(coff, compno, cstr_info,
                                 EPHused, j2klen, cio, p_manager);
            box[compno].type = JPIP_FAIX;
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



int opj_write_ppixfaix(int coff, int compno, opj_codestream_info_t cstr_info,
                       OPJ_BOOL EPHused, int j2klen, opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [8];
    OPJ_UINT32 tileno, version, i, nmax, size_of_coding; /* 4 or 8*/
    OPJ_UINT32 len;
    OPJ_OFF_T lenp;
    opj_tile_info_t *tile_Idx;
    opj_packet_info_t packet;
    int resno, precno, layno;
    OPJ_UINT32 num_packet;
    int numOfres, numOfprec, numOflayers;
    packet.end_pos = packet.end_ph_pos = packet.start_pos = -1;
    (void)EPHused; /* unused ? */

    if (j2klen > pow(2, 32)) {
        size_of_coding =  8;
        version = 1;
    } else {
        size_of_coding = 4;
        version = 0;
    }

    lenp = opj_stream_tell(cio);
    opj_stream_skip(cio, 4, p_manager);         /* L [at the end]      */
    opj_write_bytes(l_data_header, JPIP_FAIX, 4); /* FAIX */
    opj_write_bytes(l_data_header, version, 1);
    opj_stream_write_data(cio, l_data_header, 1,
                          p_manager); /* Version 0 = 4 bytes */

    nmax = 0;
    for (i = 0; i <= (OPJ_UINT32)cstr_info.numdecompos[compno]; i++) {
        nmax += (OPJ_UINT32)(cstr_info.tile[0].ph[i] * cstr_info.tile[0].pw[i] *
                             cstr_info.numlayers);
    }

    opj_write_bytes(l_data_header, nmax, size_of_coding);       /* NMAX           */
    opj_stream_write_data(cio, l_data_header, size_of_coding, p_manager);
    opj_write_bytes(l_data_header, (OPJ_UINT32)(cstr_info.tw * cstr_info.th),
                    size_of_coding);  /* M              */
    opj_stream_write_data(cio, l_data_header, size_of_coding, p_manager);

    for (tileno = 0; tileno < (OPJ_UINT32)(cstr_info.tw * cstr_info.th); tileno++) {
        tile_Idx = &cstr_info.tile[ tileno];

        num_packet = 0;
        numOfres = cstr_info.numdecompos[compno] + 1;

        for (resno = 0; resno < numOfres ; resno++) {
            numOfprec = tile_Idx->pw[resno] * tile_Idx->ph[resno];
            for (precno = 0; precno < numOfprec; precno++) {
                numOflayers = cstr_info.numlayers;
                for (layno = 0; layno < numOflayers; layno++) {

                    switch (cstr_info.prog) {
                    case OPJ_LRCP:
                        packet = tile_Idx->packet[((layno * numOfres + resno) * cstr_info.numcomps +
                                                                               compno) * numOfprec + precno];
                        break;
                    case OPJ_RLCP:
                        packet = tile_Idx->packet[((resno * numOflayers + layno) * cstr_info.numcomps +
                                                                                  compno) * numOfprec + precno];
                        break;
                    case OPJ_RPCL:
                        packet = tile_Idx->packet[((resno * numOfprec + precno) * cstr_info.numcomps +
                                                                                 compno) * numOflayers + layno];
                        break;
                    case OPJ_PCRL:
                        packet = tile_Idx->packet[((precno * cstr_info.numcomps + compno) * numOfres +
                                                                                           resno) * numOflayers + layno];
                        break;
                    case OPJ_CPRL:
                        packet = tile_Idx->packet[((compno * numOfprec + precno) * numOfres + resno) *
                                                                                 numOflayers + layno];
                        break;
                    default:
                        fprintf(stderr, "failed to ppix indexing\n");
                    }

                    opj_write_bytes(l_data_header, (OPJ_UINT32)(packet.start_pos - coff),
                                    size_of_coding);            /* start position */
                    opj_stream_write_data(cio, l_data_header, size_of_coding, p_manager);
                    opj_write_bytes(l_data_header,
                                    (OPJ_UINT32)(packet.end_pos - packet.start_pos + 1),
                                    size_of_coding); /* length         */
                    opj_stream_write_data(cio, l_data_header, size_of_coding, p_manager);

                    num_packet++;
                }
            }
        }

        while (num_packet < nmax) {    /* PADDING */
            opj_write_bytes(l_data_header, 0,
                            size_of_coding); /* start position            */
            opj_stream_write_data(cio, l_data_header, size_of_coding, p_manager);
            opj_write_bytes(l_data_header, 0,
                            size_of_coding); /* length                    */
            opj_stream_write_data(cio, l_data_header, size_of_coding, p_manager);
            num_packet++;
        }
    }

    len = (OPJ_UINT32)(opj_stream_tell(cio) - lenp);
    opj_stream_seek(cio, lenp, p_manager);
    opj_write_bytes(l_data_header, len, 4); /* L  */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);
    opj_stream_seek(cio, lenp + len, p_manager);

    return (int)len;
}
