/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
 * Copyright (c) 2017, IntoPIX SA <support@intopix.com>
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
#include "opj_common.h"


/** @defgroup T2 T2 - Implementation of a tier-2 coding */
/*@{*/

/** @name Local static functions */
/*@{*/

static void opj_t2_putcommacode(opj_bio_t *bio, OPJ_INT32 n);

static OPJ_UINT32 opj_t2_getcommacode(opj_bio_t *bio);
/**
Variable length code for signalling delta Zil (truncation point)
@param bio  Bit Input/Output component
@param n    delta Zil
*/
static void opj_t2_putnumpasses(opj_bio_t *bio, OPJ_UINT32 n);
static OPJ_UINT32 opj_t2_getnumpasses(opj_bio_t *bio);

/**
Encode a packet of a tile to a destination buffer
@param tileno Number of the tile encoded
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@param dest Destination buffer
@param p_data_written   FIXME DOC
@param len Length of the destination buffer
@param cstr_info Codestream information structure
@param p_t2_mode If == THRESH_CALC In Threshold calculation ,If == FINAL_PASS Final pass
@param p_manager the user event manager
@return
*/
static OPJ_BOOL opj_t2_encode_packet(OPJ_UINT32 tileno,
                                     opj_tcd_tile_t *tile,
                                     opj_tcp_t *tcp,
                                     opj_pi_iterator_t *pi,
                                     OPJ_BYTE *dest,
                                     OPJ_UINT32 * p_data_written,
                                     OPJ_UINT32 len,
                                     opj_codestream_info_t *cstr_info,
                                     J2K_T2_MODE p_t2_mode,
                                     opj_event_mgr_t *p_manager);

/**
Decode a packet of a tile from a source buffer
@param t2 T2 handle
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@param src Source buffer
@param data_read   FIXME DOC
@param max_length  FIXME DOC
@param pack_info Packet information
@param p_manager the user event manager

@return  FIXME DOC
*/
static OPJ_BOOL opj_t2_decode_packet(opj_t2_t* t2,
                                     opj_tcd_tile_t *tile,
                                     opj_tcp_t *tcp,
                                     opj_pi_iterator_t *pi,
                                     OPJ_BYTE *src,
                                     OPJ_UINT32 * data_read,
                                     OPJ_UINT32 max_length,
                                     opj_packet_info_t *pack_info,
                                     opj_event_mgr_t *p_manager);

static OPJ_BOOL opj_t2_skip_packet(opj_t2_t* p_t2,
                                   opj_tcd_tile_t *p_tile,
                                   opj_tcp_t *p_tcp,
                                   opj_pi_iterator_t *p_pi,
                                   OPJ_BYTE *p_src,
                                   OPJ_UINT32 * p_data_read,
                                   OPJ_UINT32 p_max_length,
                                   opj_packet_info_t *p_pack_info,
                                   opj_event_mgr_t *p_manager);

static OPJ_BOOL opj_t2_read_packet_header(opj_t2_t* p_t2,
        opj_tcd_tile_t *p_tile,
        opj_tcp_t *p_tcp,
        opj_pi_iterator_t *p_pi,
        OPJ_BOOL * p_is_data_present,
        OPJ_BYTE *p_src_data,
        OPJ_UINT32 * p_data_read,
        OPJ_UINT32 p_max_length,
        opj_packet_info_t *p_pack_info,
        opj_event_mgr_t *p_manager);

static OPJ_BOOL opj_t2_read_packet_data(opj_t2_t* p_t2,
                                        opj_tcd_tile_t *p_tile,
                                        opj_pi_iterator_t *p_pi,
                                        OPJ_BYTE *p_src_data,
                                        OPJ_UINT32 * p_data_read,
                                        OPJ_UINT32 p_max_length,
                                        opj_packet_info_t *pack_info,
                                        opj_event_mgr_t *p_manager);

static OPJ_BOOL opj_t2_skip_packet_data(opj_t2_t* p_t2,
                                        opj_tcd_tile_t *p_tile,
                                        opj_pi_iterator_t *p_pi,
                                        OPJ_UINT32 * p_data_read,
                                        OPJ_UINT32 p_max_length,
                                        opj_packet_info_t *pack_info,
                                        opj_event_mgr_t *p_manager);

/**
@param cblk
@param index
@param cblksty
@param first
*/
static OPJ_BOOL opj_t2_init_seg(opj_tcd_cblk_dec_t* cblk,
                                OPJ_UINT32 index,
                                OPJ_UINT32 cblksty,
                                OPJ_UINT32 first);

/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */

/* #define RESTART 0x04 */
static void opj_t2_putcommacode(opj_bio_t *bio, OPJ_INT32 n)
{
    while (--n >= 0) {
        opj_bio_write(bio, 1, 1);
    }
    opj_bio_write(bio, 0, 1);
}

static OPJ_UINT32 opj_t2_getcommacode(opj_bio_t *bio)
{
    OPJ_UINT32 n = 0;
    while (opj_bio_read(bio, 1)) {
        ++n;
    }
    return n;
}

static void opj_t2_putnumpasses(opj_bio_t *bio, OPJ_UINT32 n)
{
    if (n == 1) {
        opj_bio_write(bio, 0, 1);
    } else if (n == 2) {
        opj_bio_write(bio, 2, 2);
    } else if (n <= 5) {
        opj_bio_write(bio, 0xc | (n - 3), 4);
    } else if (n <= 36) {
        opj_bio_write(bio, 0x1e0 | (n - 6), 9);
    } else if (n <= 164) {
        opj_bio_write(bio, 0xff80 | (n - 37), 16);
    }
}

static OPJ_UINT32 opj_t2_getnumpasses(opj_bio_t *bio)
{
    OPJ_UINT32 n;
    if (!opj_bio_read(bio, 1)) {
        return 1;
    }
    if (!opj_bio_read(bio, 1)) {
        return 2;
    }
    if ((n = opj_bio_read(bio, 2)) != 3) {
        return (3 + n);
    }
    if ((n = opj_bio_read(bio, 5)) != 31) {
        return (6 + n);
    }
    return (37 + opj_bio_read(bio, 7));
}

/* ----------------------------------------------------------------------- */

OPJ_BOOL opj_t2_encode_packets(opj_t2_t* p_t2,
                               OPJ_UINT32 p_tile_no,
                               opj_tcd_tile_t *p_tile,
                               OPJ_UINT32 p_maxlayers,
                               OPJ_BYTE *p_dest,
                               OPJ_UINT32 * p_data_written,
                               OPJ_UINT32 p_max_len,
                               opj_codestream_info_t *cstr_info,
                               OPJ_UINT32 p_tp_num,
                               OPJ_INT32 p_tp_pos,
                               OPJ_UINT32 p_pino,
                               J2K_T2_MODE p_t2_mode,
                               opj_event_mgr_t *p_manager)
{
    OPJ_BYTE *l_current_data = p_dest;
    OPJ_UINT32 l_nb_bytes = 0;
    OPJ_UINT32 compno;
    OPJ_UINT32 poc;
    opj_pi_iterator_t *l_pi = 00;
    opj_pi_iterator_t *l_current_pi = 00;
    opj_image_t *l_image = p_t2->image;
    opj_cp_t *l_cp = p_t2->cp;
    opj_tcp_t *l_tcp = &l_cp->tcps[p_tile_no];
    OPJ_UINT32 pocno = (l_cp->rsiz == OPJ_PROFILE_CINEMA_4K) ? 2 : 1;
    OPJ_UINT32 l_max_comp = l_cp->m_specific_param.m_enc.m_max_comp_size > 0 ?
                            l_image->numcomps : 1;
    OPJ_UINT32 l_nb_pocs = l_tcp->numpocs + 1;

    l_pi = opj_pi_initialise_encode(l_image, l_cp, p_tile_no, p_t2_mode);
    if (!l_pi) {
        return OPJ_FALSE;
    }

    * p_data_written = 0;

    if (p_t2_mode == THRESH_CALC) { /* Calculating threshold */
        l_current_pi = l_pi;

        for (compno = 0; compno < l_max_comp; ++compno) {
            OPJ_UINT32 l_comp_len = 0;
            l_current_pi = l_pi;

            for (poc = 0; poc < pocno ; ++poc) {
                OPJ_UINT32 l_tp_num = compno;

                /* TODO MSD : check why this function cannot fail (cf. v1) */
                opj_pi_create_encode(l_pi, l_cp, p_tile_no, poc, l_tp_num, p_tp_pos, p_t2_mode);

                if (l_current_pi->poc.prg == OPJ_PROG_UNKNOWN) {
                    /* TODO ADE : add an error */
                    opj_pi_destroy(l_pi, l_nb_pocs);
                    return OPJ_FALSE;
                }
                while (opj_pi_next(l_current_pi)) {
                    if (l_current_pi->layno < p_maxlayers) {
                        l_nb_bytes = 0;

                        if (! opj_t2_encode_packet(p_tile_no, p_tile, l_tcp, l_current_pi,
                                                   l_current_data, &l_nb_bytes,
                                                   p_max_len, cstr_info,
                                                   p_t2_mode,
                                                   p_manager)) {
                            opj_pi_destroy(l_pi, l_nb_pocs);
                            return OPJ_FALSE;
                        }

                        l_comp_len += l_nb_bytes;
                        l_current_data += l_nb_bytes;
                        p_max_len -= l_nb_bytes;

                        * p_data_written += l_nb_bytes;
                    }
                }

                if (l_cp->m_specific_param.m_enc.m_max_comp_size) {
                    if (l_comp_len > l_cp->m_specific_param.m_enc.m_max_comp_size) {
                        opj_pi_destroy(l_pi, l_nb_pocs);
                        return OPJ_FALSE;
                    }
                }

                ++l_current_pi;
            }
        }
    } else { /* t2_mode == FINAL_PASS  */
        opj_pi_create_encode(l_pi, l_cp, p_tile_no, p_pino, p_tp_num, p_tp_pos,
                             p_t2_mode);

        l_current_pi = &l_pi[p_pino];
        if (l_current_pi->poc.prg == OPJ_PROG_UNKNOWN) {
            /* TODO ADE : add an error */
            opj_pi_destroy(l_pi, l_nb_pocs);
            return OPJ_FALSE;
        }
        while (opj_pi_next(l_current_pi)) {
            if (l_current_pi->layno < p_maxlayers) {
                l_nb_bytes = 0;

                if (! opj_t2_encode_packet(p_tile_no, p_tile, l_tcp, l_current_pi,
                                           l_current_data, &l_nb_bytes, p_max_len,
                                           cstr_info, p_t2_mode, p_manager)) {
                    opj_pi_destroy(l_pi, l_nb_pocs);
                    return OPJ_FALSE;
                }

                l_current_data += l_nb_bytes;
                p_max_len -= l_nb_bytes;

                * p_data_written += l_nb_bytes;

                /* INDEX >> */
                if (cstr_info) {
                    if (cstr_info->index_write) {
                        opj_tile_info_t *info_TL = &cstr_info->tile[p_tile_no];
                        opj_packet_info_t *info_PK = &info_TL->packet[cstr_info->packno];
                        if (!cstr_info->packno) {
                            info_PK->start_pos = info_TL->end_header + 1;
                        } else {
                            info_PK->start_pos = ((l_cp->m_specific_param.m_enc.m_tp_on | l_tcp->POC) &&
                                                  info_PK->start_pos) ? info_PK->start_pos : info_TL->packet[cstr_info->packno -
                                                                            1].end_pos + 1;
                        }
                        info_PK->end_pos = info_PK->start_pos + l_nb_bytes - 1;
                        info_PK->end_ph_pos += info_PK->start_pos -
                                               1;  /* End of packet header which now only represents the distance
                                                                                                                                                                                                                                                   to start of packet is incremented by value of start of packet*/
                    }

                    cstr_info->packno++;
                }
                /* << INDEX */
                ++p_tile->packno;
            }
        }
    }

    opj_pi_destroy(l_pi, l_nb_pocs);

    return OPJ_TRUE;
}

/* see issue 80 */
#if 0
#define JAS_FPRINTF fprintf
#else
/* issue 290 */
static void opj_null_jas_fprintf(FILE* file, const char * format, ...)
{
    (void)file;
    (void)format;
}
#define JAS_FPRINTF opj_null_jas_fprintf
#endif

OPJ_BOOL opj_t2_decode_packets(opj_tcd_t* tcd,
                               opj_t2_t *p_t2,
                               OPJ_UINT32 p_tile_no,
                               opj_tcd_tile_t *p_tile,
                               OPJ_BYTE *p_src,
                               OPJ_UINT32 * p_data_read,
                               OPJ_UINT32 p_max_len,
                               opj_codestream_index_t *p_cstr_index,
                               opj_event_mgr_t *p_manager)
{
    OPJ_BYTE *l_current_data = p_src;
    opj_pi_iterator_t *l_pi = 00;
    OPJ_UINT32 pino;
    opj_image_t *l_image = p_t2->image;
    opj_cp_t *l_cp = p_t2->cp;
    opj_tcp_t *l_tcp = &(p_t2->cp->tcps[p_tile_no]);
    OPJ_UINT32 l_nb_bytes_read;
    OPJ_UINT32 l_nb_pocs = l_tcp->numpocs + 1;
    opj_pi_iterator_t *l_current_pi = 00;
#ifdef TODO_MSD
    OPJ_UINT32 curtp = 0;
    OPJ_UINT32 tp_start_packno;
#endif
    opj_packet_info_t *l_pack_info = 00;
    opj_image_comp_t* l_img_comp = 00;

    OPJ_ARG_NOT_USED(p_cstr_index);

#ifdef TODO_MSD
    if (p_cstr_index) {
        l_pack_info = p_cstr_index->tile_index[p_tile_no].packet;
    }
#endif

    /* create a packet iterator */
    l_pi = opj_pi_create_decode(l_image, l_cp, p_tile_no);
    if (!l_pi) {
        return OPJ_FALSE;
    }


    l_current_pi = l_pi;

    for (pino = 0; pino <= l_tcp->numpocs; ++pino) {

        /* if the resolution needed is too low, one dim of the tilec could be equal to zero
         * and no packets are used to decode this resolution and
         * l_current_pi->resno is always >= p_tile->comps[l_current_pi->compno].minimum_num_resolutions
         * and no l_img_comp->resno_decoded are computed
         */
        OPJ_BOOL* first_pass_failed = NULL;

        if (l_current_pi->poc.prg == OPJ_PROG_UNKNOWN) {
            /* TODO ADE : add an error */
            opj_pi_destroy(l_pi, l_nb_pocs);
            return OPJ_FALSE;
        }

        first_pass_failed = (OPJ_BOOL*)opj_malloc(l_image->numcomps * sizeof(OPJ_BOOL));
        if (!first_pass_failed) {
            opj_pi_destroy(l_pi, l_nb_pocs);
            return OPJ_FALSE;
        }
        memset(first_pass_failed, OPJ_TRUE, l_image->numcomps * sizeof(OPJ_BOOL));

        while (opj_pi_next(l_current_pi)) {
            OPJ_BOOL skip_packet = OPJ_FALSE;
            JAS_FPRINTF(stderr,
                        "packet offset=00000166 prg=%d cmptno=%02d rlvlno=%02d prcno=%03d lyrno=%02d\n\n",
                        l_current_pi->poc.prg1, l_current_pi->compno, l_current_pi->resno,
                        l_current_pi->precno, l_current_pi->layno);

            /* If the packet layer is greater or equal than the maximum */
            /* number of layers, skip the packet */
            if (l_current_pi->layno >= l_tcp->num_layers_to_decode) {
                skip_packet = OPJ_TRUE;
            }
            /* If the packet resolution number is greater than the minimum */
            /* number of resolution allowed, skip the packet */
            else if (l_current_pi->resno >=
                     p_tile->comps[l_current_pi->compno].minimum_num_resolutions) {
                skip_packet = OPJ_TRUE;
            } else {
                /* If no precincts of any band intersects the area of interest, */
                /* skip the packet */
                OPJ_UINT32 bandno;
                opj_tcd_tilecomp_t *tilec = &p_tile->comps[l_current_pi->compno];
                opj_tcd_resolution_t *res = &tilec->resolutions[l_current_pi->resno];

                skip_packet = OPJ_TRUE;
                for (bandno = 0; bandno < res->numbands; ++bandno) {
                    opj_tcd_band_t* band = &res->bands[bandno];
                    opj_tcd_precinct_t* prec = &band->precincts[l_current_pi->precno];

                    if (opj_tcd_is_subband_area_of_interest(tcd,
                                                            l_current_pi->compno,
                                                            l_current_pi->resno,
                                                            band->bandno,
                                                            (OPJ_UINT32)prec->x0,
                                                            (OPJ_UINT32)prec->y0,
                                                            (OPJ_UINT32)prec->x1,
                                                            (OPJ_UINT32)prec->y1)) {
                        skip_packet = OPJ_FALSE;
                        break;
                    }
                }
                /*
                                printf("packet cmptno=%02d rlvlno=%02d prcno=%03d lyrno=%02d -> %s\n",
                                    l_current_pi->compno, l_current_pi->resno,
                                    l_current_pi->precno, l_current_pi->layno, skip_packet ? "skipped" : "kept");
                */
            }

            if (!skip_packet) {
                l_nb_bytes_read = 0;

                first_pass_failed[l_current_pi->compno] = OPJ_FALSE;

                if (! opj_t2_decode_packet(p_t2, p_tile, l_tcp, l_current_pi, l_current_data,
                                           &l_nb_bytes_read, p_max_len, l_pack_info, p_manager)) {
                    opj_pi_destroy(l_pi, l_nb_pocs);
                    opj_free(first_pass_failed);
                    return OPJ_FALSE;
                }

                l_img_comp = &(l_image->comps[l_current_pi->compno]);
                l_img_comp->resno_decoded = opj_uint_max(l_current_pi->resno,
                                            l_img_comp->resno_decoded);
            } else {
                l_nb_bytes_read = 0;
                if (! opj_t2_skip_packet(p_t2, p_tile, l_tcp, l_current_pi, l_current_data,
                                         &l_nb_bytes_read, p_max_len, l_pack_info, p_manager)) {
                    opj_pi_destroy(l_pi, l_nb_pocs);
                    opj_free(first_pass_failed);
                    return OPJ_FALSE;
                }
            }

            if (first_pass_failed[l_current_pi->compno]) {
                l_img_comp = &(l_image->comps[l_current_pi->compno]);
                if (l_img_comp->resno_decoded == 0) {
                    l_img_comp->resno_decoded =
                        p_tile->comps[l_current_pi->compno].minimum_num_resolutions - 1;
                }
            }

            l_current_data += l_nb_bytes_read;
            p_max_len -= l_nb_bytes_read;

            /* INDEX >> */
#ifdef TODO_MSD
            if (p_cstr_info) {
                opj_tile_info_v2_t *info_TL = &p_cstr_info->tile[p_tile_no];
                opj_packet_info_t *info_PK = &info_TL->packet[p_cstr_info->packno];
                tp_start_packno = 0;
                if (!p_cstr_info->packno) {
                    info_PK->start_pos = info_TL->end_header + 1;
                } else if (info_TL->packet[p_cstr_info->packno - 1].end_pos >=
                           (OPJ_INT32)
                           p_cstr_info->tile[p_tile_no].tp[curtp].tp_end_pos) { /* New tile part */
                    info_TL->tp[curtp].tp_numpacks = p_cstr_info->packno -
                                                     tp_start_packno; /* Number of packets in previous tile-part */
                    tp_start_packno = p_cstr_info->packno;
                    curtp++;
                    info_PK->start_pos = p_cstr_info->tile[p_tile_no].tp[curtp].tp_end_header + 1;
                } else {
                    info_PK->start_pos = (l_cp->m_specific_param.m_enc.m_tp_on &&
                                          info_PK->start_pos) ? info_PK->start_pos : info_TL->packet[p_cstr_info->packno -
                                                                      1].end_pos + 1;
                }
                info_PK->end_pos = info_PK->start_pos + l_nb_bytes_read - 1;
                info_PK->end_ph_pos += info_PK->start_pos -
                                       1;  /* End of packet header which now only represents the distance */
                ++p_cstr_info->packno;
            }
#endif
            /* << INDEX */
        }
        ++l_current_pi;

        opj_free(first_pass_failed);
    }
    /* INDEX >> */
#ifdef TODO_MSD
    if
    (p_cstr_info) {
        p_cstr_info->tile[p_tile_no].tp[curtp].tp_numpacks = p_cstr_info->packno -
                tp_start_packno; /* Number of packets in last tile-part */
    }
#endif
    /* << INDEX */

    /* don't forget to release pi */
    opj_pi_destroy(l_pi, l_nb_pocs);
    *p_data_read = (OPJ_UINT32)(l_current_data - p_src);
    return OPJ_TRUE;
}

/* ----------------------------------------------------------------------- */

/**
 * Creates a Tier 2 handle
 *
 * @param       p_image         Source or destination image
 * @param       p_cp            Image coding parameters.
 * @return              a new T2 handle if successful, NULL otherwise.
*/
opj_t2_t* opj_t2_create(opj_image_t *p_image, opj_cp_t *p_cp)
{
    /* create the t2 structure */
    opj_t2_t *l_t2 = (opj_t2_t*)opj_calloc(1, sizeof(opj_t2_t));
    if (!l_t2) {
        return NULL;
    }

    l_t2->image = p_image;
    l_t2->cp = p_cp;

    return l_t2;
}

void opj_t2_destroy(opj_t2_t *t2)
{
    if (t2) {
        opj_free(t2);
    }
}

static OPJ_BOOL opj_t2_decode_packet(opj_t2_t* p_t2,
                                     opj_tcd_tile_t *p_tile,
                                     opj_tcp_t *p_tcp,
                                     opj_pi_iterator_t *p_pi,
                                     OPJ_BYTE *p_src,
                                     OPJ_UINT32 * p_data_read,
                                     OPJ_UINT32 p_max_length,
                                     opj_packet_info_t *p_pack_info,
                                     opj_event_mgr_t *p_manager)
{
    OPJ_BOOL l_read_data;
    OPJ_UINT32 l_nb_bytes_read = 0;
    OPJ_UINT32 l_nb_total_bytes_read = 0;

    *p_data_read = 0;

    if (! opj_t2_read_packet_header(p_t2, p_tile, p_tcp, p_pi, &l_read_data, p_src,
                                    &l_nb_bytes_read, p_max_length, p_pack_info, p_manager)) {
        return OPJ_FALSE;
    }

    p_src += l_nb_bytes_read;
    l_nb_total_bytes_read += l_nb_bytes_read;
    p_max_length -= l_nb_bytes_read;

    /* we should read data for the packet */
    if (l_read_data) {
        l_nb_bytes_read = 0;

        if (! opj_t2_read_packet_data(p_t2, p_tile, p_pi, p_src, &l_nb_bytes_read,
                                      p_max_length, p_pack_info, p_manager)) {
            return OPJ_FALSE;
        }

        l_nb_total_bytes_read += l_nb_bytes_read;
    }

    *p_data_read = l_nb_total_bytes_read;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_t2_encode_packet(OPJ_UINT32 tileno,
                                     opj_tcd_tile_t * tile,
                                     opj_tcp_t * tcp,
                                     opj_pi_iterator_t *pi,
                                     OPJ_BYTE *dest,
                                     OPJ_UINT32 * p_data_written,
                                     OPJ_UINT32 length,
                                     opj_codestream_info_t *cstr_info,
                                     J2K_T2_MODE p_t2_mode,
                                     opj_event_mgr_t *p_manager)
{
    OPJ_UINT32 bandno, cblkno;
    OPJ_BYTE* c = dest;
    OPJ_UINT32 l_nb_bytes;
    OPJ_UINT32 compno = pi->compno;     /* component value */
    OPJ_UINT32 resno  = pi->resno;      /* resolution level value */
    OPJ_UINT32 precno = pi->precno;     /* precinct value */
    OPJ_UINT32 layno  = pi->layno;      /* quality layer value */
    OPJ_UINT32 l_nb_blocks;
    opj_tcd_band_t *band = 00;
    opj_tcd_cblk_enc_t* cblk = 00;
    opj_tcd_pass_t *pass = 00;

    opj_tcd_tilecomp_t *tilec = &tile->comps[compno];
    opj_tcd_resolution_t *res = &tilec->resolutions[resno];

    opj_bio_t *bio = 00;    /* BIO component */
#ifdef ENABLE_EMPTY_PACKET_OPTIMIZATION
    OPJ_BOOL packet_empty = OPJ_TRUE;
#else
    OPJ_BOOL packet_empty = OPJ_FALSE;
#endif

    /* <SOP 0xff91> */
    if (tcp->csty & J2K_CP_CSTY_SOP) {
        if (length < 6) {
            if (p_t2_mode == FINAL_PASS) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "opj_t2_encode_packet(): only %u bytes remaining in "
                              "output buffer. %u needed.\n",
                              length, 6);
            }
            return OPJ_FALSE;
        }
        c[0] = 255;
        c[1] = 145;
        c[2] = 0;
        c[3] = 4;
#if 0
        c[4] = (tile->packno % 65536) / 256;
        c[5] = (tile->packno % 65536) % 256;
#else
        c[4] = (tile->packno >> 8) & 0xff; /* packno is uint32_t */
        c[5] = tile->packno & 0xff;
#endif
        c += 6;
        length -= 6;
    }
    /* </SOP> */

    if (!layno) {
        band = res->bands;

        for (bandno = 0; bandno < res->numbands; ++bandno, ++band) {
            opj_tcd_precinct_t *prc;

            /* Skip empty bands */
            if (opj_tcd_is_band_empty(band)) {
                continue;
            }

            prc = &band->precincts[precno];
            opj_tgt_reset(prc->incltree);
            opj_tgt_reset(prc->imsbtree);

            l_nb_blocks = prc->cw * prc->ch;
            for (cblkno = 0; cblkno < l_nb_blocks; ++cblkno) {
                cblk = &prc->cblks.enc[cblkno];

                cblk->numpasses = 0;
                opj_tgt_setvalue(prc->imsbtree, cblkno, band->numbps - (OPJ_INT32)cblk->numbps);
            }
        }
    }

    bio = opj_bio_create();
    if (!bio) {
        /* FIXME event manager error callback */
        return OPJ_FALSE;
    }
    opj_bio_init_enc(bio, c, length);

#ifdef ENABLE_EMPTY_PACKET_OPTIMIZATION
    /* WARNING: this code branch is disabled, since it has been reported that */
    /* such packets cause decoding issues with cinema J2K hardware */
    /* decoders: https://groups.google.com/forum/#!topic/openjpeg/M7M_fLX_Bco */

    /* Check if the packet is empty */
    /* Note: we could also skip that step and always write a packet header */
    band = res->bands;
    for (bandno = 0; bandno < res->numbands; ++bandno, ++band) {
        opj_tcd_precinct_t *prc;
        /* Skip empty bands */
        if (opj_tcd_is_band_empty(band)) {
            continue;
        }

        prc = &band->precincts[precno];
        l_nb_blocks = prc->cw * prc->ch;
        cblk = prc->cblks.enc;
        for (cblkno = 0; cblkno < l_nb_blocks; cblkno++, ++cblk) {
            opj_tcd_layer_t *layer = &cblk->layers[layno];

            /* if cblk not included, go to the next cblk  */
            if (!layer->numpasses) {
                continue;
            }
            packet_empty = OPJ_FALSE;
            break;
        }
        if (!packet_empty) {
            break;
        }
    }
#endif
    opj_bio_write(bio, packet_empty ? 0 : 1, 1);           /* Empty header bit */

    /* Writing Packet header */
    band = res->bands;
    for (bandno = 0; !packet_empty &&
            bandno < res->numbands; ++bandno, ++band)      {
        opj_tcd_precinct_t *prc;

        /* Skip empty bands */
        if (opj_tcd_is_band_empty(band)) {
            continue;
        }

        prc = &band->precincts[precno];
        l_nb_blocks = prc->cw * prc->ch;
        cblk = prc->cblks.enc;

        for (cblkno = 0; cblkno < l_nb_blocks; ++cblkno) {
            opj_tcd_layer_t *layer = &cblk->layers[layno];

            if (!cblk->numpasses && layer->numpasses) {
                opj_tgt_setvalue(prc->incltree, cblkno, (OPJ_INT32)layno);
            }

            ++cblk;
        }

        cblk = prc->cblks.enc;
        for (cblkno = 0; cblkno < l_nb_blocks; cblkno++) {
            opj_tcd_layer_t *layer = &cblk->layers[layno];
            OPJ_UINT32 increment = 0;
            OPJ_UINT32 nump = 0;
            OPJ_UINT32 len = 0, passno;
            OPJ_UINT32 l_nb_passes;

            /* cblk inclusion bits */
            if (!cblk->numpasses) {
                opj_tgt_encode(bio, prc->incltree, cblkno, (OPJ_INT32)(layno + 1));
            } else {
                opj_bio_write(bio, layer->numpasses != 0, 1);
            }

            /* if cblk not included, go to the next cblk  */
            if (!layer->numpasses) {
                ++cblk;
                continue;
            }

            /* if first instance of cblk --> zero bit-planes information */
            if (!cblk->numpasses) {
                cblk->numlenbits = 3;
                opj_tgt_encode(bio, prc->imsbtree, cblkno, 999);
            }

            /* number of coding passes included */
            opj_t2_putnumpasses(bio, layer->numpasses);
            l_nb_passes = cblk->numpasses + layer->numpasses;
            pass = cblk->passes +  cblk->numpasses;

            /* computation of the increase of the length indicator and insertion in the header     */
            for (passno = cblk->numpasses; passno < l_nb_passes; ++passno) {
                ++nump;
                len += pass->len;

                if (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1) {
                    increment = (OPJ_UINT32)opj_int_max((OPJ_INT32)increment,
                                                        opj_int_floorlog2((OPJ_INT32)len) + 1
                                                        - ((OPJ_INT32)cblk->numlenbits + opj_int_floorlog2((OPJ_INT32)nump)));
                    len = 0;
                    nump = 0;
                }

                ++pass;
            }
            opj_t2_putcommacode(bio, (OPJ_INT32)increment);

            /* computation of the new Length indicator */
            cblk->numlenbits += increment;

            pass = cblk->passes +  cblk->numpasses;
            /* insertion of the codeword segment length */
            for (passno = cblk->numpasses; passno < l_nb_passes; ++passno) {
                nump++;
                len += pass->len;

                if (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1) {
                    opj_bio_write(bio, (OPJ_UINT32)len,
                                  cblk->numlenbits + (OPJ_UINT32)opj_int_floorlog2((OPJ_INT32)nump));
                    len = 0;
                    nump = 0;
                }
                ++pass;
            }

            ++cblk;
        }
    }

    if (!opj_bio_flush(bio)) {
        opj_bio_destroy(bio);
        return OPJ_FALSE;               /* modified to eliminate longjmp !! */
    }

    l_nb_bytes = (OPJ_UINT32)opj_bio_numbytes(bio);
    c += l_nb_bytes;
    length -= l_nb_bytes;

    opj_bio_destroy(bio);

    /* <EPH 0xff92> */
    if (tcp->csty & J2K_CP_CSTY_EPH) {
        if (length < 2) {
            if (p_t2_mode == FINAL_PASS) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "opj_t2_encode_packet(): only %u bytes remaining in "
                              "output buffer. %u needed.\n",
                              length, 2);
            }
            return OPJ_FALSE;
        }
        c[0] = 255;
        c[1] = 146;
        c += 2;
        length -= 2;
    }
    /* </EPH> */

    /* << INDEX */
    /* End of packet header position. Currently only represents the distance to start of packet
       Will be updated later by incrementing with packet start value*/
    if (cstr_info && cstr_info->index_write) {
        opj_packet_info_t *info_PK = &cstr_info->tile[tileno].packet[cstr_info->packno];
        info_PK->end_ph_pos = (OPJ_INT32)(c - dest);
    }
    /* INDEX >> */

    /* Writing the packet body */
    band = res->bands;
    for (bandno = 0; !packet_empty && bandno < res->numbands; bandno++, ++band) {
        opj_tcd_precinct_t *prc;

        /* Skip empty bands */
        if (opj_tcd_is_band_empty(band)) {
            continue;
        }

        prc = &band->precincts[precno];
        l_nb_blocks = prc->cw * prc->ch;
        cblk = prc->cblks.enc;

        for (cblkno = 0; cblkno < l_nb_blocks; ++cblkno) {
            opj_tcd_layer_t *layer = &cblk->layers[layno];

            if (!layer->numpasses) {
                ++cblk;
                continue;
            }

            if (layer->len > length) {
                if (p_t2_mode == FINAL_PASS) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "opj_t2_encode_packet(): only %u bytes remaining in "
                                  "output buffer. %u needed.\n",
                                  length, layer->len);
                }
                return OPJ_FALSE;
            }

            memcpy(c, layer->data, layer->len);
            cblk->numpasses += layer->numpasses;
            c += layer->len;
            length -= layer->len;

            /* << INDEX */
            if (cstr_info && cstr_info->index_write) {
                opj_packet_info_t *info_PK = &cstr_info->tile[tileno].packet[cstr_info->packno];
                info_PK->disto += layer->disto;
                if (cstr_info->D_max < info_PK->disto) {
                    cstr_info->D_max = info_PK->disto;
                }
            }

            ++cblk;
            /* INDEX >> */
        }
    }

    assert(c >= dest);
    * p_data_written += (OPJ_UINT32)(c - dest);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_t2_skip_packet(opj_t2_t* p_t2,
                                   opj_tcd_tile_t *p_tile,
                                   opj_tcp_t *p_tcp,
                                   opj_pi_iterator_t *p_pi,
                                   OPJ_BYTE *p_src,
                                   OPJ_UINT32 * p_data_read,
                                   OPJ_UINT32 p_max_length,
                                   opj_packet_info_t *p_pack_info,
                                   opj_event_mgr_t *p_manager)
{
    OPJ_BOOL l_read_data;
    OPJ_UINT32 l_nb_bytes_read = 0;
    OPJ_UINT32 l_nb_total_bytes_read = 0;

    *p_data_read = 0;

    if (! opj_t2_read_packet_header(p_t2, p_tile, p_tcp, p_pi, &l_read_data, p_src,
                                    &l_nb_bytes_read, p_max_length, p_pack_info, p_manager)) {
        return OPJ_FALSE;
    }

    p_src += l_nb_bytes_read;
    l_nb_total_bytes_read += l_nb_bytes_read;
    p_max_length -= l_nb_bytes_read;

    /* we should read data for the packet */
    if (l_read_data) {
        l_nb_bytes_read = 0;

        if (! opj_t2_skip_packet_data(p_t2, p_tile, p_pi, &l_nb_bytes_read,
                                      p_max_length, p_pack_info, p_manager)) {
            return OPJ_FALSE;
        }

        l_nb_total_bytes_read += l_nb_bytes_read;
    }
    *p_data_read = l_nb_total_bytes_read;

    return OPJ_TRUE;
}


static OPJ_BOOL opj_t2_read_packet_header(opj_t2_t* p_t2,
        opj_tcd_tile_t *p_tile,
        opj_tcp_t *p_tcp,
        opj_pi_iterator_t *p_pi,
        OPJ_BOOL * p_is_data_present,
        OPJ_BYTE *p_src_data,
        OPJ_UINT32 * p_data_read,
        OPJ_UINT32 p_max_length,
        opj_packet_info_t *p_pack_info,
        opj_event_mgr_t *p_manager)

{
    /* loop */
    OPJ_UINT32 bandno, cblkno;
    OPJ_UINT32 l_nb_code_blocks;
    OPJ_UINT32 l_remaining_length;
    OPJ_UINT32 l_header_length;
    OPJ_UINT32 * l_modified_length_ptr = 00;
    OPJ_BYTE *l_current_data = p_src_data;
    opj_cp_t *l_cp = p_t2->cp;
    opj_bio_t *l_bio = 00;  /* BIO component */
    opj_tcd_band_t *l_band = 00;
    opj_tcd_cblk_dec_t* l_cblk = 00;
    opj_tcd_resolution_t* l_res =
        &p_tile->comps[p_pi->compno].resolutions[p_pi->resno];

    OPJ_BYTE *l_header_data = 00;
    OPJ_BYTE **l_header_data_start = 00;

    OPJ_UINT32 l_present;

    if (p_pi->layno == 0) {
        l_band = l_res->bands;

        /* reset tagtrees */
        for (bandno = 0; bandno < l_res->numbands; ++bandno) {
            if (!opj_tcd_is_band_empty(l_band)) {
                opj_tcd_precinct_t *l_prc = &l_band->precincts[p_pi->precno];
                if (!(p_pi->precno < (l_band->precincts_data_size / sizeof(
                                          opj_tcd_precinct_t)))) {
                    opj_event_msg(p_manager, EVT_ERROR, "Invalid precinct\n");
                    return OPJ_FALSE;
                }


                opj_tgt_reset(l_prc->incltree);
                opj_tgt_reset(l_prc->imsbtree);
                l_cblk = l_prc->cblks.dec;

                l_nb_code_blocks = l_prc->cw * l_prc->ch;
                for (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno) {
                    l_cblk->numsegs = 0;
                    l_cblk->real_num_segs = 0;
                    ++l_cblk;
                }
            }

            ++l_band;
        }
    }

    /* SOP markers */

    if (p_tcp->csty & J2K_CP_CSTY_SOP) {
        if (p_max_length < 6) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Not enough space for expected SOP marker\n");
        } else if ((*l_current_data) != 0xff || (*(l_current_data + 1) != 0x91)) {
            opj_event_msg(p_manager, EVT_WARNING, "Expected SOP marker\n");
        } else {
            l_current_data += 6;
        }

        /** TODO : check the Nsop value */
    }

    /*
    When the marker PPT/PPM is used the packet header are store in PPT/PPM marker
    This part deal with this caracteristic
    step 1: Read packet header in the saved structure
    step 2: Return to codestream for decoding
    */

    l_bio = opj_bio_create();
    if (! l_bio) {
        return OPJ_FALSE;
    }

    if (l_cp->ppm == 1) { /* PPM */
        l_header_data_start = &l_cp->ppm_data;
        l_header_data = *l_header_data_start;
        l_modified_length_ptr = &(l_cp->ppm_len);

    } else if (p_tcp->ppt == 1) { /* PPT */
        l_header_data_start = &(p_tcp->ppt_data);
        l_header_data = *l_header_data_start;
        l_modified_length_ptr = &(p_tcp->ppt_len);
    } else { /* Normal Case */
        l_header_data_start = &(l_current_data);
        l_header_data = *l_header_data_start;
        l_remaining_length = (OPJ_UINT32)(p_src_data + p_max_length - l_header_data);
        l_modified_length_ptr = &(l_remaining_length);
    }

    opj_bio_init_dec(l_bio, l_header_data, *l_modified_length_ptr);

    l_present = opj_bio_read(l_bio, 1);
    JAS_FPRINTF(stderr, "present=%d \n", l_present);
    if (!l_present) {
        /* TODO MSD: no test to control the output of this function*/
        opj_bio_inalign(l_bio);
        l_header_data += opj_bio_numbytes(l_bio);
        opj_bio_destroy(l_bio);

        /* EPH markers */
        if (p_tcp->csty & J2K_CP_CSTY_EPH) {
            if ((*l_modified_length_ptr - (OPJ_UINT32)(l_header_data -
                    *l_header_data_start)) < 2U) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "Not enough space for expected EPH marker\n");
            } else if ((*l_header_data) != 0xff || (*(l_header_data + 1) != 0x92)) {
                opj_event_msg(p_manager, EVT_WARNING, "Expected EPH marker\n");
            } else {
                l_header_data += 2;
            }
        }

        l_header_length = (OPJ_UINT32)(l_header_data - *l_header_data_start);
        *l_modified_length_ptr -= l_header_length;
        *l_header_data_start += l_header_length;

        /* << INDEX */
        /* End of packet header position. Currently only represents the distance to start of packet
           Will be updated later by incrementing with packet start value */
        if (p_pack_info) {
            p_pack_info->end_ph_pos = (OPJ_INT32)(l_current_data - p_src_data);
        }
        /* INDEX >> */

        * p_is_data_present = OPJ_FALSE;
        *p_data_read = (OPJ_UINT32)(l_current_data - p_src_data);
        return OPJ_TRUE;
    }

    l_band = l_res->bands;
    for (bandno = 0; bandno < l_res->numbands; ++bandno, ++l_band) {
        opj_tcd_precinct_t *l_prc = &(l_band->precincts[p_pi->precno]);

        if (opj_tcd_is_band_empty(l_band)) {
            continue;
        }

        l_nb_code_blocks = l_prc->cw * l_prc->ch;
        l_cblk = l_prc->cblks.dec;
        for (cblkno = 0; cblkno < l_nb_code_blocks; cblkno++) {
            OPJ_UINT32 l_included, l_increment, l_segno;
            OPJ_INT32 n;

            /* if cblk not yet included before --> inclusion tagtree */
            if (!l_cblk->numsegs) {
                l_included = opj_tgt_decode(l_bio, l_prc->incltree, cblkno,
                                            (OPJ_INT32)(p_pi->layno + 1));
                /* else one bit */
            } else {
                l_included = opj_bio_read(l_bio, 1);
            }

            /* if cblk not included */
            if (!l_included) {
                l_cblk->numnewpasses = 0;
                ++l_cblk;
                JAS_FPRINTF(stderr, "included=%d \n", l_included);
                continue;
            }

            /* if cblk not yet included --> zero-bitplane tagtree */
            if (!l_cblk->numsegs) {
                OPJ_UINT32 i = 0;

                while (!opj_tgt_decode(l_bio, l_prc->imsbtree, cblkno, (OPJ_INT32)i)) {
                    ++i;
                }

                l_cblk->numbps = (OPJ_UINT32)l_band->numbps + 1 - i;
                l_cblk->numlenbits = 3;
            }

            /* number of coding passes */
            l_cblk->numnewpasses = opj_t2_getnumpasses(l_bio);
            l_increment = opj_t2_getcommacode(l_bio);

            /* length indicator increment */
            l_cblk->numlenbits += l_increment;
            l_segno = 0;

            if (!l_cblk->numsegs) {
                if (! opj_t2_init_seg(l_cblk, l_segno, p_tcp->tccps[p_pi->compno].cblksty, 1)) {
                    opj_bio_destroy(l_bio);
                    return OPJ_FALSE;
                }
            } else {
                l_segno = l_cblk->numsegs - 1;
                if (l_cblk->segs[l_segno].numpasses == l_cblk->segs[l_segno].maxpasses) {
                    ++l_segno;
                    if (! opj_t2_init_seg(l_cblk, l_segno, p_tcp->tccps[p_pi->compno].cblksty, 0)) {
                        opj_bio_destroy(l_bio);
                        return OPJ_FALSE;
                    }
                }
            }
            n = (OPJ_INT32)l_cblk->numnewpasses;

            do {
                OPJ_UINT32 bit_number;
                l_cblk->segs[l_segno].numnewpasses = (OPJ_UINT32)opj_int_min((OPJ_INT32)(
                        l_cblk->segs[l_segno].maxpasses - l_cblk->segs[l_segno].numpasses), n);
                bit_number = l_cblk->numlenbits + opj_uint_floorlog2(
                                 l_cblk->segs[l_segno].numnewpasses);
                if (bit_number > 32) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "Invalid bit number %d in opj_t2_read_packet_header()\n",
                                  bit_number);
                    opj_bio_destroy(l_bio);
                    return OPJ_FALSE;
                }
                l_cblk->segs[l_segno].newlen = opj_bio_read(l_bio, bit_number);
                JAS_FPRINTF(stderr, "included=%d numnewpasses=%d increment=%d len=%d \n",
                            l_included, l_cblk->segs[l_segno].numnewpasses, l_increment,
                            l_cblk->segs[l_segno].newlen);

                n -= (OPJ_INT32)l_cblk->segs[l_segno].numnewpasses;
                if (n > 0) {
                    ++l_segno;

                    if (! opj_t2_init_seg(l_cblk, l_segno, p_tcp->tccps[p_pi->compno].cblksty, 0)) {
                        opj_bio_destroy(l_bio);
                        return OPJ_FALSE;
                    }
                }
            } while (n > 0);

            ++l_cblk;
        }
    }

    if (!opj_bio_inalign(l_bio)) {
        opj_bio_destroy(l_bio);
        return OPJ_FALSE;
    }

    l_header_data += opj_bio_numbytes(l_bio);
    opj_bio_destroy(l_bio);

    /* EPH markers */
    if (p_tcp->csty & J2K_CP_CSTY_EPH) {
        if ((*l_modified_length_ptr - (OPJ_UINT32)(l_header_data -
                *l_header_data_start)) < 2U) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Not enough space for expected EPH marker\n");
        } else if ((*l_header_data) != 0xff || (*(l_header_data + 1) != 0x92)) {
            opj_event_msg(p_manager, EVT_WARNING, "Expected EPH marker\n");
        } else {
            l_header_data += 2;
        }
    }

    l_header_length = (OPJ_UINT32)(l_header_data - *l_header_data_start);
    JAS_FPRINTF(stderr, "hdrlen=%d \n", l_header_length);
    JAS_FPRINTF(stderr, "packet body\n");
    *l_modified_length_ptr -= l_header_length;
    *l_header_data_start += l_header_length;

    /* << INDEX */
    /* End of packet header position. Currently only represents the distance to start of packet
     Will be updated later by incrementing with packet start value */
    if (p_pack_info) {
        p_pack_info->end_ph_pos = (OPJ_INT32)(l_current_data - p_src_data);
    }
    /* INDEX >> */

    *p_is_data_present = OPJ_TRUE;
    *p_data_read = (OPJ_UINT32)(l_current_data - p_src_data);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_t2_read_packet_data(opj_t2_t* p_t2,
                                        opj_tcd_tile_t *p_tile,
                                        opj_pi_iterator_t *p_pi,
                                        OPJ_BYTE *p_src_data,
                                        OPJ_UINT32 * p_data_read,
                                        OPJ_UINT32 p_max_length,
                                        opj_packet_info_t *pack_info,
                                        opj_event_mgr_t* p_manager)
{
    OPJ_UINT32 bandno, cblkno;
    OPJ_UINT32 l_nb_code_blocks;
    OPJ_BYTE *l_current_data = p_src_data;
    opj_tcd_band_t *l_band = 00;
    opj_tcd_cblk_dec_t* l_cblk = 00;
    opj_tcd_resolution_t* l_res =
        &p_tile->comps[p_pi->compno].resolutions[p_pi->resno];

    OPJ_ARG_NOT_USED(p_t2);
    OPJ_ARG_NOT_USED(pack_info);

    l_band = l_res->bands;
    for (bandno = 0; bandno < l_res->numbands; ++bandno) {
        opj_tcd_precinct_t *l_prc = &l_band->precincts[p_pi->precno];

        if ((l_band->x1 - l_band->x0 == 0) || (l_band->y1 - l_band->y0 == 0)) {
            ++l_band;
            continue;
        }

        l_nb_code_blocks = l_prc->cw * l_prc->ch;
        l_cblk = l_prc->cblks.dec;

        for (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno) {
            opj_tcd_seg_t *l_seg = 00;

            if (!l_cblk->numnewpasses) {
                /* nothing to do */
                ++l_cblk;
                continue;
            }

            if (!l_cblk->numsegs) {
                l_seg = l_cblk->segs;
                ++l_cblk->numsegs;
            } else {
                l_seg = &l_cblk->segs[l_cblk->numsegs - 1];

                if (l_seg->numpasses == l_seg->maxpasses) {
                    ++l_seg;
                    ++l_cblk->numsegs;
                }
            }

            do {
                /* Check possible overflow (on l_current_data only, assumes input args already checked) then size */
                if ((((OPJ_SIZE_T)l_current_data + (OPJ_SIZE_T)l_seg->newlen) <
                        (OPJ_SIZE_T)l_current_data) ||
                        (l_current_data + l_seg->newlen > p_src_data + p_max_length)) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "read: segment too long (%d) with max (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                  l_seg->newlen, p_max_length, cblkno, p_pi->precno, bandno, p_pi->resno,
                                  p_pi->compno);
                    return OPJ_FALSE;
                }

#ifdef USE_JPWL
                /* we need here a j2k handle to verify if making a check to
                the validity of cblocks parameters is selected from user (-W) */

                /* let's check that we are not exceeding */
                if ((l_cblk->len + l_seg->newlen) > 8192) {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "JPWL: segment too long (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                  l_seg->newlen, cblkno, p_pi->precno, bandno, p_pi->resno, p_pi->compno);
                    if (!JPWL_ASSUME) {
                        opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                        return OPJ_FALSE;
                    }
                    l_seg->newlen = 8192 - l_cblk->len;
                    opj_event_msg(p_manager, EVT_WARNING, "      - truncating segment to %d\n",
                                  l_seg->newlen);
                    break;
                };

#endif /* USE_JPWL */

                if (l_cblk->numchunks == l_cblk->numchunksalloc) {
                    OPJ_UINT32 l_numchunksalloc = l_cblk->numchunksalloc * 2 + 1;
                    opj_tcd_seg_data_chunk_t* l_chunks =
                        (opj_tcd_seg_data_chunk_t*)opj_realloc(l_cblk->chunks,
                                l_numchunksalloc * sizeof(opj_tcd_seg_data_chunk_t));
                    if (l_chunks == NULL) {
                        opj_event_msg(p_manager, EVT_ERROR,
                                      "cannot allocate opj_tcd_seg_data_chunk_t* array");
                        return OPJ_FALSE;
                    }
                    l_cblk->chunks = l_chunks;
                    l_cblk->numchunksalloc = l_numchunksalloc;
                }

                l_cblk->chunks[l_cblk->numchunks].data = l_current_data;
                l_cblk->chunks[l_cblk->numchunks].len = l_seg->newlen;
                l_cblk->numchunks ++;

                l_current_data += l_seg->newlen;
                l_seg->len += l_seg->newlen;
                l_seg->numpasses += l_seg->numnewpasses;
                l_cblk->numnewpasses -= l_seg->numnewpasses;

                l_seg->real_num_passes = l_seg->numpasses;

                if (l_cblk->numnewpasses > 0) {
                    ++l_seg;
                    ++l_cblk->numsegs;
                }
            } while (l_cblk->numnewpasses > 0);

            l_cblk->real_num_segs = l_cblk->numsegs;
            ++l_cblk;
        } /* next code_block */

        ++l_band;
    }

    *(p_data_read) = (OPJ_UINT32)(l_current_data - p_src_data);


    return OPJ_TRUE;
}

static OPJ_BOOL opj_t2_skip_packet_data(opj_t2_t* p_t2,
                                        opj_tcd_tile_t *p_tile,
                                        opj_pi_iterator_t *p_pi,
                                        OPJ_UINT32 * p_data_read,
                                        OPJ_UINT32 p_max_length,
                                        opj_packet_info_t *pack_info,
                                        opj_event_mgr_t *p_manager)
{
    OPJ_UINT32 bandno, cblkno;
    OPJ_UINT32 l_nb_code_blocks;
    opj_tcd_band_t *l_band = 00;
    opj_tcd_cblk_dec_t* l_cblk = 00;
    opj_tcd_resolution_t* l_res =
        &p_tile->comps[p_pi->compno].resolutions[p_pi->resno];

    OPJ_ARG_NOT_USED(p_t2);
    OPJ_ARG_NOT_USED(pack_info);

    *p_data_read = 0;
    l_band = l_res->bands;

    for (bandno = 0; bandno < l_res->numbands; ++bandno) {
        opj_tcd_precinct_t *l_prc = &l_band->precincts[p_pi->precno];

        if ((l_band->x1 - l_band->x0 == 0) || (l_band->y1 - l_band->y0 == 0)) {
            ++l_band;
            continue;
        }

        l_nb_code_blocks = l_prc->cw * l_prc->ch;
        l_cblk = l_prc->cblks.dec;

        for (cblkno = 0; cblkno < l_nb_code_blocks; ++cblkno) {
            opj_tcd_seg_t *l_seg = 00;

            if (!l_cblk->numnewpasses) {
                /* nothing to do */
                ++l_cblk;
                continue;
            }

            if (!l_cblk->numsegs) {
                l_seg = l_cblk->segs;
                ++l_cblk->numsegs;
            } else {
                l_seg = &l_cblk->segs[l_cblk->numsegs - 1];

                if (l_seg->numpasses == l_seg->maxpasses) {
                    ++l_seg;
                    ++l_cblk->numsegs;
                }
            }

            do {
                /* Check possible overflow then size */
                if (((*p_data_read + l_seg->newlen) < (*p_data_read)) ||
                        ((*p_data_read + l_seg->newlen) > p_max_length)) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "skip: segment too long (%d) with max (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                  l_seg->newlen, p_max_length, cblkno, p_pi->precno, bandno, p_pi->resno,
                                  p_pi->compno);
                    return OPJ_FALSE;
                }

#ifdef USE_JPWL
                /* we need here a j2k handle to verify if making a check to
                the validity of cblocks parameters is selected from user (-W) */

                /* let's check that we are not exceeding */
                if ((l_cblk->len + l_seg->newlen) > 8192) {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "JPWL: segment too long (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                  l_seg->newlen, cblkno, p_pi->precno, bandno, p_pi->resno, p_pi->compno);
                    if (!JPWL_ASSUME) {
                        opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                        return -999;
                    }
                    l_seg->newlen = 8192 - l_cblk->len;
                    opj_event_msg(p_manager, EVT_WARNING, "      - truncating segment to %d\n",
                                  l_seg->newlen);
                    break;
                };

#endif /* USE_JPWL */
                JAS_FPRINTF(stderr, "p_data_read (%d) newlen (%d) \n", *p_data_read,
                            l_seg->newlen);
                *(p_data_read) += l_seg->newlen;

                l_seg->numpasses += l_seg->numnewpasses;
                l_cblk->numnewpasses -= l_seg->numnewpasses;
                if (l_cblk->numnewpasses > 0) {
                    ++l_seg;
                    ++l_cblk->numsegs;
                }
            } while (l_cblk->numnewpasses > 0);

            ++l_cblk;
        }

        ++l_band;
    }

    return OPJ_TRUE;
}


static OPJ_BOOL opj_t2_init_seg(opj_tcd_cblk_dec_t* cblk,
                                OPJ_UINT32 index,
                                OPJ_UINT32 cblksty,
                                OPJ_UINT32 first)
{
    opj_tcd_seg_t* seg = 00;
    OPJ_UINT32 l_nb_segs = index + 1;

    if (l_nb_segs > cblk->m_current_max_segs) {
        opj_tcd_seg_t* new_segs;
        OPJ_UINT32 l_m_current_max_segs = cblk->m_current_max_segs +
                                          OPJ_J2K_DEFAULT_NB_SEGS;

        new_segs = (opj_tcd_seg_t*) opj_realloc(cblk->segs,
                                                l_m_current_max_segs * sizeof(opj_tcd_seg_t));
        if (! new_segs) {
            /* opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to initialize segment %d\n", l_nb_segs); */
            return OPJ_FALSE;
        }
        cblk->segs = new_segs;
        memset(new_segs + cblk->m_current_max_segs,
               0, OPJ_J2K_DEFAULT_NB_SEGS * sizeof(opj_tcd_seg_t));
        cblk->m_current_max_segs = l_m_current_max_segs;
    }

    seg = &cblk->segs[index];
    opj_tcd_reinit_segment(seg);

    if (cblksty & J2K_CCP_CBLKSTY_TERMALL) {
        seg->maxpasses = 1;
    } else if (cblksty & J2K_CCP_CBLKSTY_LAZY) {
        if (first) {
            seg->maxpasses = 10;
        } else {
            seg->maxpasses = (((seg - 1)->maxpasses == 1) ||
                              ((seg - 1)->maxpasses == 10)) ? 2 : 1;
        }
    } else {
        /* See paragraph "B.10.6 Number of coding passes" of the standard.
         * Probably that 109 must be interpreted a (Mb-1)*3 + 1 with Mb=37,
         * Mb being the maximum number of bit-planes available for the
         * representation of coefficients in the sub-band */
        seg->maxpasses = 109;
    }

    return OPJ_TRUE;
}
