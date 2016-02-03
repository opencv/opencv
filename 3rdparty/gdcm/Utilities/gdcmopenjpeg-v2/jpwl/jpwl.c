/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2005-2006, Dept. of Electronic and Information Engineering, Universita' degli Studi di Perugia, Italy
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

#include "../libopenjpeg/opj_includes.h"

#ifdef USE_JPWL

/** @defgroup JPWL JPWL - JPEG-2000 Part11 (JPWL) codestream manager */
/*@{*/

/** @name Local static variables */
/*@{*/

/** number of JPWL prepared markers */
static int jwmarker_num;
/** properties of JPWL markers to insert */
static jpwl_marker_t jwmarker[JPWL_MAX_NO_MARKERS];

/*@}*/

/*@}*/

/** @name Local static functions */
/*@{*/

/** create an EPC marker segment
@param j2k J2K compressor handle
@param esd_on true if ESD is activated
@param red_on true if RED is activated
@param epb_on true if EPB is activated
@param info_on true if informative techniques are activated
@return returns the freshly created EPC
*/
jpwl_epc_ms_t *jpwl_epc_create(opj_j2k_t *j2k, bool esd_on, bool red_on, bool epb_on, bool info_on);

/*@}*/

/** create an EPC marker segment
@param j2k J2K compressor handle
@param comps considered component (-1=average, 0/1/2/...=component no.)
@param addrm addressing mode (0=packet, 1=byte range, 2=packet range, 3=reserved)
@param ad_size size of addresses (2/4 bytes)
@param senst sensitivity type
@param se_size sensitivity values size (1/2 bytes)
@param tileno tile where this ESD lies (-1 means MH)
@param svalnum number of sensitivity values (if 0, they will be automatically filled)
@param sensval pointer to an array of sensitivity values (if NULL, they will be automatically filled)
@return returns the freshly created ESD
*/
jpwl_esd_ms_t *jpwl_esd_create(opj_j2k_t *j2k, int comps, unsigned char addrm, unsigned char ad_size,
                unsigned char senst, int se_size, int tileno,
                unsigned long int svalnum, void *sensval);

/** this function is used to compare two JPWL markers based on
their relevant wishlist position
@param arg1 pointer to first marker
@param arg2 pointer to second marker
@return 1 if arg1>arg2, 0 if arg1=arg2, -1 if arg1<arg2
*/
int jpwl_markcomp(const void *arg1, const void *arg2);

/** write an EPB MS to a buffer
@param j2k J2K compressor handle
@param epbmark pointer to the EPB MS
@param buf pointer to the memory buffer
*/
void jpwl_epb_write(opj_j2k_t *j2k, jpwl_epb_ms_t *epbmark, unsigned char *buf);

/** write an EPC MS to a buffer
@param j2k J2K compressor handle
@param epcmark pointer to the EPC MS
@param buf pointer to the memory buffer
*/
void jpwl_epc_write(opj_j2k_t *j2k, jpwl_epc_ms_t *epcmark, unsigned char *buf);

/** write an ESD MS to a buffer
@param j2k J2K compressor handle
@param esdmark pointer to the ESD MS
@param buf pointer to the memory buffer
*/
void jpwl_esd_write(opj_j2k_t *j2k, jpwl_esd_ms_t *esdmark, unsigned char *buf);

/*-----------------------------------------------------------------*/

void jpwl_encode(opj_j2k_t *j2k, opj_cio_t *cio, opj_image_t *image) {

  int mm;

  /* let's reset some settings */

  /* clear the existing markers */
  for (mm = 0; mm < jwmarker_num; mm++) {

    switch (jwmarker[mm].id) {

    case J2K_MS_EPB:
      opj_free(jwmarker[mm].m.epbmark);
      break;

    case J2K_MS_EPC:
      opj_free(jwmarker[mm].m.epcmark);
      break;

    case J2K_MS_ESD:
      opj_free(jwmarker[mm].m.esdmark);
      break;

    case J2K_MS_RED:
      opj_free(jwmarker[mm].m.redmark);
      break;

    default:
      break;
    }
  }

  /* clear the marker structure array */
  memset(jwmarker, 0, sizeof(jpwl_marker_t) * JPWL_MAX_NO_MARKERS);

  /* no more markers in the list */
  jwmarker_num = 0;

  /* let's begin creating a marker list, according to user wishes */
  jpwl_prepare_marks(j2k, cio, image);

  /* now we dump the JPWL markers on the codestream */
  jpwl_dump_marks(j2k, cio, image);

  /* do not know exactly what is this for,
  but it gets called during index creation */
  j2k->pos_correction = 0;

}

void j2k_add_marker(opj_codestream_info_t *cstr_info, unsigned short int type, int pos, int len) {

  if (!cstr_info)
    return;

  /* expand the list? */
  if ((cstr_info->marknum + 1) > cstr_info->maxmarknum) {
    cstr_info->maxmarknum = 100 + (int) ((float) cstr_info->maxmarknum * 1.0F);
    cstr_info->marker = opj_realloc(cstr_info->marker, cstr_info->maxmarknum);
  }

  /* add the marker */
  cstr_info->marker[cstr_info->marknum].type = type;
  cstr_info->marker[cstr_info->marknum].pos = pos;
  cstr_info->marker[cstr_info->marknum].len = len;
  cstr_info->marknum++;

}

void jpwl_prepare_marks(opj_j2k_t *j2k, opj_cio_t *cio, opj_image_t *image) {

  unsigned short int socsiz_len = 0;
  int ciopos = cio_tell(cio), soc_pos = j2k->cstr_info->main_head_start;
  unsigned char *socp = NULL;

  int tileno, acc_tpno, tpno, tilespec, hprot, sens, pprot, packspec, lastileno, packno;

  jpwl_epb_ms_t *epb_mark;
  jpwl_epc_ms_t *epc_mark;
  jpwl_esd_ms_t *esd_mark;

  /* find (SOC + SIZ) length */
  /* I assume SIZ is always the first marker after SOC */
  cio_seek(cio, soc_pos + 4);
  socsiz_len = (unsigned short int) cio_read(cio, 2) + 4; /* add the 2 marks length itself */
  cio_seek(cio, soc_pos + 0);
  socp = cio_getbp(cio); /* pointer to SOC */

  /*
   EPC MS for Main Header: if we are here it's required
  */
  /* create the EPC */
  if ((epc_mark = jpwl_epc_create(
      j2k,
      j2k->cp->esd_on, /* is ESD present? */
      j2k->cp->red_on, /* is RED present? */
      j2k->cp->epb_on, /* is EPB present? */
      false /* are informative techniques present? */
    ))) {

    /* Add this marker to the 'insertanda' list */
    if (epc_mark) {
      jwmarker[jwmarker_num].id = J2K_MS_EPC; /* its type */
      jwmarker[jwmarker_num].m.epcmark = epc_mark; /* the EPC */
      jwmarker[jwmarker_num].pos = soc_pos + socsiz_len; /* after SIZ */
      jwmarker[jwmarker_num].dpos = (double) jwmarker[jwmarker_num].pos + 0.1; /* not so first */
      jwmarker[jwmarker_num].len = epc_mark->Lepc; /* its length */
      jwmarker[jwmarker_num].len_ready = true; /* ready */
      jwmarker[jwmarker_num].pos_ready = true; /* ready */
      jwmarker[jwmarker_num].parms_ready = false; /* not ready */
      jwmarker[jwmarker_num].data_ready = true; /* ready */
      jwmarker_num++;
    };

    opj_event_msg(j2k->cinfo, EVT_INFO,
      "MH  EPC : setting %s%s%s\n",
      j2k->cp->esd_on ? "ESD, " : "",
      j2k->cp->red_on ? "RED, " : "",
      j2k->cp->epb_on ? "EPB, " : ""
      );

  } else {
    /* ooops, problems */
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create MH EPC\n");
  };

  /*
   ESD MS for Main Header
  */
  /* first of all, must MH have an ESD MS? */
  if (j2k->cp->esd_on && (j2k->cp->sens_MH >= 0)) {

    /* Create the ESD */
    if ((esd_mark = jpwl_esd_create(
      j2k, /* this encoder handle */
      -1, /* we are averaging over all components */
      (unsigned char) j2k->cp->sens_range, /* range method */
      (unsigned char) j2k->cp->sens_addr, /* sensitivity addressing */
      (unsigned char) j2k->cp->sens_MH, /* sensitivity method */
      j2k->cp->sens_size, /* sensitivity size */
      -1, /* this ESD is in main header */
      0 /*j2k->cstr_info->num*/, /* number of packets in codestream */
      NULL /*sensval*/ /* pointer to sensitivity data of packets */
      ))) {

      /* Add this marker to the 'insertanda' list */
      if (jwmarker_num < JPWL_MAX_NO_MARKERS) {
        jwmarker[jwmarker_num].id = J2K_MS_ESD; /* its type */
        jwmarker[jwmarker_num].m.esdmark = esd_mark; /* the EPB */
        jwmarker[jwmarker_num].pos = soc_pos + socsiz_len; /* we choose to place it after SIZ */
        jwmarker[jwmarker_num].dpos = (double) jwmarker[jwmarker_num].pos + 0.2; /* not first at all! */
        jwmarker[jwmarker_num].len = esd_mark->Lesd; /* its length */
        jwmarker[jwmarker_num].len_ready = true; /* not ready, yet */
        jwmarker[jwmarker_num].pos_ready = true; /* ready */
        jwmarker[jwmarker_num].parms_ready = true; /* not ready */
        jwmarker[jwmarker_num].data_ready = false; /* not ready */
        jwmarker_num++;
      }

      opj_event_msg(j2k->cinfo, EVT_INFO,
        "MH  ESDs: method %d\n",
        j2k->cp->sens_MH
        );

    } else {
      /* ooops, problems */
      opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create MH ESD\n");
    };

  }

  /*
   ESD MSs for Tile Part Headers
  */
  /* cycle through tiles */
  sens = -1; /* default spec: no ESD */
  tilespec = 0; /* first tile spec */
  acc_tpno = 0;
  for (tileno = 0; tileno < j2k->cstr_info->tw * j2k->cstr_info->th; tileno++) {

    opj_event_msg(j2k->cinfo, EVT_INFO,
      "Tile %d has %d tile part(s)\n",
      tileno, j2k->cstr_info->tile[tileno].num_tps
      );

    /* for every tile part in the tile */
    for (tpno = 0; tpno < j2k->cstr_info->tile[tileno].num_tps; tpno++, acc_tpno++) {

      int sot_len, Psot, Psotp, mm;
      unsigned long sot_pos, post_sod_pos;

      unsigned long int left_THmarks_len;

      /******* sot_pos = j2k->cstr_info->tile[tileno].start_pos; */
      sot_pos = j2k->cstr_info->tile[tileno].tp[tpno].tp_start_pos;
      cio_seek(cio, sot_pos + 2);
      sot_len = cio_read(cio, 2); /* SOT Len */
      cio_skip(cio, 2);
      Psotp = cio_tell(cio);
      Psot = cio_read(cio, 4); /* tile length */

      /******* post_sod_pos = j2k->cstr_info->tile[tileno].end_header + 1; */
      post_sod_pos = j2k->cstr_info->tile[tileno].tp[tpno].tp_end_header + 1;
      left_THmarks_len = post_sod_pos - sot_pos;

      /* add all the lengths of the markers which are len-ready and stay within SOT and SOD */
      for (mm = 0; mm < jwmarker_num; mm++) {
        if ((jwmarker[mm].pos >= sot_pos) && (jwmarker[mm].pos < post_sod_pos)) {
          if (jwmarker[mm].len_ready)
            left_THmarks_len += jwmarker[mm].len + 2;
          else {
            opj_event_msg(j2k->cinfo, EVT_ERROR, "MS %x in %f is not len-ready: could not set up TH EPB\n",
              jwmarker[mm].id, jwmarker[mm].dpos);
            exit(1);
          }
        }
      }

      /******* if ((tilespec < JPWL_MAX_NO_TILESPECS) && (j2k->cp->sens_TPH_tileno[tilespec] == tileno)) */
      if ((tilespec < JPWL_MAX_NO_TILESPECS) && (j2k->cp->sens_TPH_tileno[tilespec] == acc_tpno))
        /* we got a specification from this tile onwards */
        sens = j2k->cp->sens_TPH[tilespec++];

      /* must this TPH have an ESD MS? */
      if (j2k->cp->esd_on && (sens >= 0)) {

        /* Create the ESD */
        if ((esd_mark = jpwl_esd_create(
          j2k, /* this encoder handle */
          -1, /* we are averaging over all components */
          (unsigned char) j2k->cp->sens_range, /* range method */
          (unsigned char) j2k->cp->sens_addr, /* sensitivity addressing size */
          (unsigned char) sens, /* sensitivity method */
          j2k->cp->sens_size, /* sensitivity value size */
          tileno, /* this ESD is in a tile */
          0, /* number of packets in codestream */
          NULL /* pointer to sensitivity data of packets */
          ))) {

          /* Add this marker to the 'insertanda' list */
          if (jwmarker_num < JPWL_MAX_NO_MARKERS) {
            jwmarker[jwmarker_num].id = J2K_MS_ESD; /* its type */
            jwmarker[jwmarker_num].m.esdmark = esd_mark; /* the EPB */
            /****** jwmarker[jwmarker_num].pos = j2k->cstr_info->tile[tileno].start_pos + sot_len + 2; */ /* after SOT */
            jwmarker[jwmarker_num].pos = j2k->cstr_info->tile[tileno].tp[tpno].tp_start_pos + sot_len + 2; /* after SOT */
            jwmarker[jwmarker_num].dpos = (double) jwmarker[jwmarker_num].pos + 0.2; /* not first at all! */
            jwmarker[jwmarker_num].len = esd_mark->Lesd; /* its length */
            jwmarker[jwmarker_num].len_ready = true; /* ready, yet */
            jwmarker[jwmarker_num].pos_ready = true; /* ready */
            jwmarker[jwmarker_num].parms_ready = true; /* not ready */
            jwmarker[jwmarker_num].data_ready = false; /* ready */
            jwmarker_num++;
          }

          /* update Psot of the tile  */
          cio_seek(cio, Psotp);
          cio_write(cio, Psot + esd_mark->Lesd + 2, 4);

          opj_event_msg(j2k->cinfo, EVT_INFO,
            /******* "TPH ESDs: tile %02d, method %d\n", */
            "TPH ESDs: tile %02d, part %02d, method %d\n",
            /******* tileno, */
            tileno, tpno,
            sens
            );

        } else {
          /* ooops, problems */
          /***** opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create TPH ESD #%d\n", tileno); */
          opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create TPH ESD #%d,%d\n", tileno, tpno);
        };

      }

    }

  };

  /*
   EPB MS for Main Header
  */
  /* first of all, must MH have an EPB MS? */
  if (j2k->cp->epb_on && (j2k->cp->hprot_MH > 0)) {

    int mm;

    /* position of SOT */
    unsigned int sot_pos = j2k->cstr_info->main_head_end + 1;

    /* how much space is there between end of SIZ and beginning of SOT? */
    int left_MHmarks_len = sot_pos - socsiz_len;

    /* add all the lengths of the markers which are len-ready and stay within SOC and SOT */
    for (mm = 0; mm < jwmarker_num; mm++) {
      if ((jwmarker[mm].pos >=0) && (jwmarker[mm].pos < sot_pos)) {
        if (jwmarker[mm].len_ready)
          left_MHmarks_len += jwmarker[mm].len + 2;
        else {
          opj_event_msg(j2k->cinfo, EVT_ERROR, "MS %x in %f is not len-ready: could not set up MH EPB\n",
            jwmarker[mm].id, jwmarker[mm].dpos);
          exit(1);
        }
      }
    }

    /* Create the EPB */
    if ((epb_mark = jpwl_epb_create(
      j2k, /* this encoder handle */
      true, /* is it the latest? */
      true, /* is it packed? not for now */
      -1, /* we are in main header */
      0, /* its index is 0 (first) */
      j2k->cp->hprot_MH, /* protection type parameters of data */
      socsiz_len, /* pre-data: only SOC+SIZ */
      left_MHmarks_len /* post-data: from SOC to SOT, and all JPWL markers within */
      ))) {

      /* Add this marker to the 'insertanda' list */
      if (jwmarker_num < JPWL_MAX_NO_MARKERS) {
        jwmarker[jwmarker_num].id = J2K_MS_EPB; /* its type */
        jwmarker[jwmarker_num].m.epbmark = epb_mark; /* the EPB */
        jwmarker[jwmarker_num].pos = soc_pos + socsiz_len; /* after SIZ */
        jwmarker[jwmarker_num].dpos = (double) jwmarker[jwmarker_num].pos; /* first first first! */
        jwmarker[jwmarker_num].len = epb_mark->Lepb; /* its length */
        jwmarker[jwmarker_num].len_ready = true; /* ready */
        jwmarker[jwmarker_num].pos_ready = true; /* ready */
        jwmarker[jwmarker_num].parms_ready = true; /* ready */
        jwmarker[jwmarker_num].data_ready = false; /* not ready */
        jwmarker_num++;
      }

      opj_event_msg(j2k->cinfo, EVT_INFO,
        "MH  EPB : prot. %d\n",
        j2k->cp->hprot_MH
        );

    } else {
      /* ooops, problems */
      opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create MH EPB\n");
    };
  }

  /*
   EPB MSs for Tile Parts
  */
  /* cycle through TPHs */
  hprot = j2k->cp->hprot_MH; /* default spec */
  tilespec = 0; /* first tile spec */
  lastileno = 0;
  packspec = 0;
  pprot = -1;
  acc_tpno = 0;
  for (tileno = 0; tileno < j2k->cstr_info->tw * j2k->cstr_info->th; tileno++) {

    opj_event_msg(j2k->cinfo, EVT_INFO,
      "Tile %d has %d tile part(s)\n",
      tileno, j2k->cstr_info->tile[tileno].num_tps
      );

    /* for every tile part in the tile */
    for (tpno = 0; tpno < j2k->cstr_info->tile[tileno].num_tps; tpno++, acc_tpno++) {

      int sot_len, Psot, Psotp, mm, epb_index = 0, prot_len = 0;
      unsigned long sot_pos, post_sod_pos;
      unsigned long int left_THmarks_len/*, epbs_len = 0*/;
      int startpack = 0, stoppack = j2k->cstr_info->packno;
      int first_tp_pack, last_tp_pack;
      jpwl_epb_ms_t *tph_epb = NULL;

      /****** sot_pos = j2k->cstr_info->tile[tileno].start_pos; */
      sot_pos = j2k->cstr_info->tile[tileno].tp[tpno].tp_start_pos;
      cio_seek(cio, sot_pos + 2);
      sot_len = cio_read(cio, 2); /* SOT Len */
      cio_skip(cio, 2);
      Psotp = cio_tell(cio);
      Psot = cio_read(cio, 4); /* tile length */

      /* a-priori length of the data dwelling between SOT and SOD */
      /****** post_sod_pos = j2k->cstr_info->tile[tileno].end_header + 1; */
      post_sod_pos = j2k->cstr_info->tile[tileno].tp[tpno].tp_end_header + 1;
      left_THmarks_len = post_sod_pos - (sot_pos + sot_len + 2);

      /* add all the lengths of the JPWL markers which are len-ready and stay within SOT and SOD */
      for (mm = 0; mm < jwmarker_num; mm++) {
        if ((jwmarker[mm].pos >= sot_pos) && (jwmarker[mm].pos < post_sod_pos)) {
          if (jwmarker[mm].len_ready)
            left_THmarks_len += jwmarker[mm].len + 2;
          else {
            opj_event_msg(j2k->cinfo, EVT_ERROR, "MS %x in %f is not len-ready: could not set up TH EPB\n",
              jwmarker[mm].id, jwmarker[mm].dpos);
            exit(1);
          }
        }
      }

      /****** if ((tilespec < JPWL_MAX_NO_TILESPECS) && (j2k->cp->hprot_TPH_tileno[tilespec] == tileno)) */
      if ((tilespec < JPWL_MAX_NO_TILESPECS) && (j2k->cp->hprot_TPH_tileno[tilespec] == acc_tpno))
        /* we got a specification from this tile part onwards */
        hprot = j2k->cp->hprot_TPH[tilespec++];

      /* must this TPH have an EPB MS? */
      if (j2k->cp->epb_on && (hprot > 0)) {

        /* Create the EPB */
        if ((epb_mark = jpwl_epb_create(
          j2k, /* this encoder handle */
          false, /* is it the latest? in TPH, no for now (if huge data size in TPH, we'd need more) */
          true, /* is it packed? yes for now */
          tileno, /* we are in TPH */
          epb_index++, /* its index is 0 (first) */
          hprot, /* protection type parameters of following data */
          sot_len + 2, /* pre-data length: only SOT */
          left_THmarks_len /* post-data length: from SOT end to SOD inclusive */
          ))) {

          /* Add this marker to the 'insertanda' list */
          if (jwmarker_num < JPWL_MAX_NO_MARKERS) {
            jwmarker[jwmarker_num].id = J2K_MS_EPB; /* its type */
            jwmarker[jwmarker_num].m.epbmark = epb_mark; /* the EPB */
            /****** jwmarker[jwmarker_num].pos = j2k->cstr_info->tile[tileno].start_pos + sot_len + 2; */ /* after SOT */
            jwmarker[jwmarker_num].pos = j2k->cstr_info->tile[tileno].tp[tpno].tp_start_pos + sot_len + 2; /* after SOT */
            jwmarker[jwmarker_num].dpos = (double) jwmarker[jwmarker_num].pos; /* first first first! */
            jwmarker[jwmarker_num].len = epb_mark->Lepb; /* its length */
            jwmarker[jwmarker_num].len_ready = true; /* ready */
            jwmarker[jwmarker_num].pos_ready = true; /* ready */
            jwmarker[jwmarker_num].parms_ready = true; /* ready */
            jwmarker[jwmarker_num].data_ready = false; /* not ready */
            jwmarker_num++;
          }

          /* update Psot of the tile  */
          Psot += epb_mark->Lepb + 2;

          opj_event_msg(j2k->cinfo, EVT_INFO,
            /***** "TPH EPB : tile %02d, prot. %d\n", */
            "TPH EPB : tile %02d, part %02d, prot. %d\n",
            /***** tileno, */
            tileno, tpno,
            hprot
            );

          /* save this TPH EPB address */
          tph_epb = epb_mark;

        } else {
          /* ooops, problems */
          /****** opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create TPH EPB #%d\n", tileno);  */
          opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create TPH EPB in #%d,d\n", tileno, tpno);
        };

      }

      startpack = 0;
      /* EPB MSs for UEP packet data protection in Tile Parts */
      /****** for (packno = 0; packno < j2k->cstr_info->num; packno++) { */
      /*first_tp_pack = (tpno > 0) ? (first_tp_pack + j2k->cstr_info->tile[tileno].tp[tpno - 1].tp_numpacks) : 0;*/
      first_tp_pack = j2k->cstr_info->tile[tileno].tp[tpno].tp_start_pack;
      last_tp_pack = first_tp_pack + j2k->cstr_info->tile[tileno].tp[tpno].tp_numpacks - 1;
      for (packno = 0; packno < j2k->cstr_info->tile[tileno].tp[tpno].tp_numpacks; packno++) {

        /******** if ((packspec < JPWL_MAX_NO_PACKSPECS) &&
          (j2k->cp->pprot_tileno[packspec] == tileno) && (j2k->cp->pprot_packno[packspec] == packno)) { */
        if ((packspec < JPWL_MAX_NO_PACKSPECS) &&
          (j2k->cp->pprot_tileno[packspec] == acc_tpno) && (j2k->cp->pprot_packno[packspec] == packno)) {

          /* we got a specification from this tile and packet onwards */
          /* print the previous spec */
          if (packno > 0) {
            stoppack = packno - 1;
            opj_event_msg(j2k->cinfo, EVT_INFO,
              /***** "UEP EPBs: tile %02d, packs. %02d-%02d (B %d-%d), prot. %d\n", */
              "UEP EPBs: tile %02d, part %02d, packs. %02d-%02d (B %d-%d), prot. %d\n",
              /***** tileno, */
              tileno, tpno,
              startpack,
              stoppack,
              /***** j2k->cstr_info->tile[tileno].packet[startpack].start_pos, */
              j2k->cstr_info->tile[tileno].packet[first_tp_pack + startpack].start_pos,
              /***** j2k->cstr_info->tile[tileno].packet[stoppack].end_pos, */
              j2k->cstr_info->tile[tileno].packet[first_tp_pack + stoppack].end_pos,
              pprot);

            /***** prot_len = j2k->cstr_info->tile[tileno].packet[stoppack].end_pos + 1 -
              j2k->cstr_info->tile[tileno].packet[startpack].start_pos; */
            prot_len = j2k->cstr_info->tile[tileno].packet[first_tp_pack + stoppack].end_pos + 1 -
              j2k->cstr_info->tile[tileno].packet[first_tp_pack + startpack].start_pos;

            /*
              particular case: if this is the last header and the last packet,
              then it is better to protect even the EOC marker
            */
            /****** if ((tileno == ((j2k->cstr_info->tw * j2k->cstr_info->th) - 1)) &&
              (stoppack == (j2k->cstr_info->num - 1))) */
            if ((tileno == ((j2k->cstr_info->tw * j2k->cstr_info->th) - 1)) &&
              (tpno == (j2k->cstr_info->tile[tileno].num_tps - 1)) &&
              (stoppack == last_tp_pack))
              /* add the EOC len */
              prot_len += 2;

            /* let's add the EPBs */
            Psot += jpwl_epbs_add(
              j2k, /* J2K handle */
              jwmarker, /* pointer to JPWL markers list */
              &jwmarker_num, /* pointer to the number of current markers */
              false, /* latest */
              true, /* packed */
              false, /* inside MH */
              &epb_index, /* pointer to EPB index */
              pprot, /* protection type */
              /****** (double) (j2k->cstr_info->tile[tileno].start_pos + sot_len + 2) + 0.0001, */ /* position */
              (double) (j2k->cstr_info->tile[tileno].tp[tpno].tp_start_pos + sot_len + 2) + 0.0001, /* position */
              tileno, /* number of tile */
              0, /* length of pre-data */
              prot_len /*4000*/ /* length of post-data */
              );
          }

          startpack = packno;
          pprot = j2k->cp->pprot[packspec++];
        }

        //printf("Tile %02d, pack %02d ==> %d\n", tileno, packno, pprot);

      }

      /* we are at the end: print the remaining spec */
      stoppack = packno - 1;
      if (pprot >= 0) {

        opj_event_msg(j2k->cinfo, EVT_INFO,
          /**** "UEP EPBs: tile %02d, packs. %02d-%02d (B %d-%d), prot. %d\n", */
          "UEP EPBs: tile %02d, part %02d, packs. %02d-%02d (B %d-%d), prot. %d\n",
          /**** tileno, */
          tileno, tpno,
          startpack,
          stoppack,
          /***** j2k->image_info->tile[tileno].packet[startpack].start_pos,
          j2k->image_info->tile[tileno].packet[stoppack].end_pos, */
          j2k->cstr_info->tile[tileno].packet[first_tp_pack + startpack].start_pos,
          j2k->cstr_info->tile[tileno].packet[first_tp_pack + stoppack].end_pos,
          pprot);

        /***** prot_len = j2k->cstr_info->tile[tileno].packet[stoppack].end_pos + 1 -
          j2k->cstr_info->tile[tileno].packet[startpack].start_pos; */
        prot_len = j2k->cstr_info->tile[tileno].packet[first_tp_pack + stoppack].end_pos + 1 -
          j2k->cstr_info->tile[tileno].packet[first_tp_pack + startpack].start_pos;

        /*
          particular case: if this is the last header and the last packet,
          then it is better to protect even the EOC marker
        */
        /***** if ((tileno == ((j2k->cstr_info->tw * j2k->cstr_info->th) - 1)) &&
          (stoppack == (j2k->cstr_info->num - 1))) */
        if ((tileno == ((j2k->cstr_info->tw * j2k->cstr_info->th) - 1)) &&
          (tpno == (j2k->cstr_info->tile[tileno].num_tps - 1)) &&
          (stoppack == last_tp_pack))
          /* add the EOC len */
          prot_len += 2;

        /* let's add the EPBs */
        Psot += jpwl_epbs_add(
              j2k, /* J2K handle */
              jwmarker, /* pointer to JPWL markers list */
              &jwmarker_num, /* pointer to the number of current markers */
              true, /* latest */
              true, /* packed */
              false, /* inside MH */
              &epb_index, /* pointer to EPB index */
              pprot, /* protection type */
              /***** (double) (j2k->cstr_info->tile[tileno].start_pos + sot_len + 2) + 0.0001,*/ /* position */
              (double) (j2k->cstr_info->tile[tileno].tp[tpno].tp_start_pos + sot_len + 2) + 0.0001, /* position */
              tileno, /* number of tile */
              0, /* length of pre-data */
              prot_len /*4000*/ /* length of post-data */
              );
      }

      /* we can now check if the TPH EPB was really the last one */
      if (tph_epb && (epb_index == 1)) {
        /* set the TPH EPB to be the last one in current header */
        tph_epb->Depb |= (unsigned char) ((true & 0x0001) << 6);
        tph_epb = NULL;
      }

      /* write back Psot */
      cio_seek(cio, Psotp);
      cio_write(cio, Psot, 4);

    }

  };

  /* reset the position */
  cio_seek(cio, ciopos);

}

void jpwl_dump_marks(opj_j2k_t *j2k, opj_cio_t *cio, opj_image_t *image) {

  int mm;
  unsigned long int old_size = j2k->cstr_info->codestream_size;
  unsigned long int new_size = old_size;
  int /*ciopos = cio_tell(cio),*/ soc_pos = j2k->cstr_info->main_head_start;
  unsigned char *jpwl_buf, *orig_buf;
  unsigned long int orig_pos;
  double epbcoding_time = 0.0, esdcoding_time = 0.0;

  /* Order JPWL markers according to their wishlist position */
  qsort((void *) jwmarker, (size_t) jwmarker_num, sizeof (jpwl_marker_t), jpwl_markcomp);

  /* compute markers total size */
  for (mm = 0; mm < jwmarker_num; mm++) {
    /*printf("%x, %d, %.10f, %d long\n", jwmarker[mm].id, jwmarker[mm].pos,
      jwmarker[mm].dpos, jwmarker[mm].len);*/
    new_size += jwmarker[mm].len + 2;
  }

  /* allocate a new buffer of proper size */
  if (!(jpwl_buf = (unsigned char *) opj_malloc((size_t) (new_size + soc_pos) * sizeof(unsigned char)))) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not allocate room for JPWL codestream buffer\n");
    exit(1);
  };

  /* copy the jp2 part, if any */
  orig_buf = jpwl_buf;
  memcpy(jpwl_buf, cio->buffer, soc_pos);
  jpwl_buf += soc_pos;

  /* cycle through markers */
  orig_pos = soc_pos + 0; /* start from the beginning */
  cio_seek(cio, soc_pos + 0); /* rewind the original */
  for (mm = 0; mm < jwmarker_num; mm++) {

    /*
    need to copy a piece of the original codestream
    if there is such
    */
    memcpy(jpwl_buf, cio_getbp(cio), jwmarker[mm].pos - orig_pos);
    jpwl_buf += jwmarker[mm].pos - orig_pos;
    orig_pos = jwmarker[mm].pos;
    cio_seek(cio, orig_pos);

    /*
    then write down the marker
    */
    switch (jwmarker[mm].id) {

    case J2K_MS_EPB:
      jpwl_epb_write(j2k, jwmarker[mm].m.epbmark, jpwl_buf);
      break;

    case J2K_MS_EPC:
      jpwl_epc_write(j2k, jwmarker[mm].m.epcmark, jpwl_buf);
      break;

    case J2K_MS_ESD:
      jpwl_esd_write(j2k, jwmarker[mm].m.esdmark, jpwl_buf);
      break;

    case J2K_MS_RED:
      memset(jpwl_buf, 0, jwmarker[mm].len + 2); /* placeholder */
      break;

    default:
      break;
    };

    /* we update the markers struct */
    if (j2k->cstr_info)
      j2k->cstr_info->marker[j2k->cstr_info->marknum - 1].pos = (jpwl_buf - orig_buf);

    /* we set the marker dpos to the new position in the JPWL codestream */
    jwmarker[mm].dpos = (double) (jpwl_buf - orig_buf);

    /* advance JPWL buffer position */
    jpwl_buf += jwmarker[mm].len + 2;

  }

  /* finish remaining original codestream */
  memcpy(jpwl_buf, cio_getbp(cio), old_size - (orig_pos - soc_pos));
  jpwl_buf += old_size - (orig_pos - soc_pos);
  cio_seek(cio, soc_pos + old_size);

  /*
  update info file based on added markers
  */
  if (!jpwl_update_info(j2k, jwmarker, jwmarker_num))
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not update OPJ cstr_info structure\n");

  /* now we need to repass some markers and fill their data fields */

  /* first of all, DL and Pcrc in EPCs */
  for (mm = 0; mm < jwmarker_num; mm++) {

    /* find the EPCs */
    if (jwmarker[mm].id == J2K_MS_EPC) {

      int epc_pos = (int) jwmarker[mm].dpos, pp;
      unsigned short int mycrc = 0x0000;

      /* fix and fill the DL field */
      jwmarker[mm].m.epcmark->DL = new_size;
      orig_buf[epc_pos + 6] = (unsigned char) (jwmarker[mm].m.epcmark->DL >> 24);
      orig_buf[epc_pos + 7] = (unsigned char) (jwmarker[mm].m.epcmark->DL >> 16);
      orig_buf[epc_pos + 8] = (unsigned char) (jwmarker[mm].m.epcmark->DL >> 8);
      orig_buf[epc_pos + 9] = (unsigned char) (jwmarker[mm].m.epcmark->DL >> 0);

      /* compute the CRC field (excluding itself) */
      for (pp = 0; pp < 4; pp++)
        jpwl_updateCRC16(&mycrc, orig_buf[epc_pos + pp]);
      for (pp = 6; pp < (jwmarker[mm].len + 2); pp++)
        jpwl_updateCRC16(&mycrc, orig_buf[epc_pos + pp]);

      /* fix and fill the CRC */
      jwmarker[mm].m.epcmark->Pcrc = mycrc;
      orig_buf[epc_pos + 4] = (unsigned char) (jwmarker[mm].m.epcmark->Pcrc >> 8);
      orig_buf[epc_pos + 5] = (unsigned char) (jwmarker[mm].m.epcmark->Pcrc >> 0);

    }
  }

  /* then, sensitivity data in ESDs */
  esdcoding_time = opj_clock();
  for (mm = 0; mm < jwmarker_num; mm++) {

    /* find the ESDs */
    if (jwmarker[mm].id == J2K_MS_ESD) {

      /* remember that they are now in a new position (dpos) */
      int esd_pos = (int) jwmarker[mm].dpos;

      jpwl_esd_fill(j2k, jwmarker[mm].m.esdmark, &orig_buf[esd_pos]);

    }

  }
  esdcoding_time = opj_clock() - esdcoding_time;
  if (j2k->cp->esd_on)
    opj_event_msg(j2k->cinfo, EVT_INFO, "ESDs sensitivities computed in %f s\n", esdcoding_time);

  /* finally, RS or CRC parity in EPBs */
  epbcoding_time = opj_clock();
  for (mm = 0; mm < jwmarker_num; mm++) {

    /* find the EPBs */
    if (jwmarker[mm].id == J2K_MS_EPB) {

      /* remember that they are now in a new position (dpos) */
      int nn, accum_len;

      /* let's see how many EPBs are following this one, included itself */
      /* for this to work, we suppose that the markers are correctly ordered */
      /* and, overall, that they are in packed mode inside headers */
      accum_len = 0;
      for (nn = mm; (nn < jwmarker_num) && (jwmarker[nn].id == J2K_MS_EPB) &&
        (jwmarker[nn].pos == jwmarker[mm].pos); nn++)
        accum_len += jwmarker[nn].m.epbmark->Lepb + 2;

      /* fill the current (first) EPB with post-data starting from the computed position */
      jpwl_epb_fill(j2k, jwmarker[mm].m.epbmark, &orig_buf[(int) jwmarker[mm].dpos],
        &orig_buf[(int) jwmarker[mm].dpos + accum_len]);

      /* fill the remaining EPBs in the header with post-data starting from the last position */
      for (nn = mm + 1; (nn < jwmarker_num) && (jwmarker[nn].id == J2K_MS_EPB) &&
        (jwmarker[nn].pos == jwmarker[mm].pos); nn++)
        jpwl_epb_fill(j2k, jwmarker[nn].m.epbmark, &orig_buf[(int) jwmarker[nn].dpos], NULL);

      /* skip all the processed EPBs */
      mm = nn - 1;
    }

  }
  epbcoding_time = opj_clock() - epbcoding_time;
  if (j2k->cp->epb_on)
    opj_event_msg(j2k->cinfo, EVT_INFO, "EPBs redundancy computed in %f s\n", epbcoding_time);

  /* free original cio buffer and set it to the JPWL one */
  opj_free(cio->buffer);
  cio->cinfo = cio->cinfo; /* no change */
  cio->openmode = cio->openmode; /* no change */
  cio->buffer = orig_buf;
  cio->length = new_size + soc_pos;
  cio->start = cio->buffer;
  cio->end = cio->buffer + cio->length;
  cio->bp = cio->buffer;
  cio_seek(cio, soc_pos + new_size);

}


void j2k_read_epc(opj_j2k_t *j2k) {
  unsigned long int DL, Lepcp, Pcrcp, l;
  unsigned short int Lepc, Pcrc = 0x0000;
  unsigned char Pepc;
  opj_cio_t *cio = j2k->cio;
  char *ans1;

  /* Simply read the EPC parameters */
  Lepcp = cio_tell(cio);
  Lepc = cio_read(cio, 2);
  Pcrcp = cio_tell(cio);
  cio_skip(cio, 2); /* Pcrc */
  DL = cio_read(cio, 4);
  Pepc = cio_read(cio, 1);

  /* compute Pcrc */
  cio_seek(cio, Lepcp - 2);

    /* Marker */
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));

    /* Length */
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));

    /* skip Pcrc */
    cio_skip(cio, 2);

    /* read all remaining */
    for (l = 4; l < Lepc; l++)
      jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));

    /* check Pcrc with the result */
    cio_seek(cio, Pcrcp);
    ans1 = (Pcrc == (unsigned short int) cio_read(cio, 2)) ? "crc-ok" : "crc-ko";

  /* now we write them to screen */
  opj_event_msg(j2k->cinfo, EVT_INFO,
    "EPC(%u,%d): %s, DL=%d%s %s %s\n",
    Lepcp - 2,
    Lepc,
    ans1,
    DL, /* data length this EPC is referring to */
    (Pepc & 0x10) ? ", esd" : "", /* ESD is present */
    (Pepc & 0x20) ? ", red" : "", /* RED is present */
    (Pepc & 0x40) ? ", epb" : ""); /* EPB is present */

  cio_seek(cio, Lepcp + Lepc);
}

void j2k_write_epc(opj_j2k_t *j2k) {

  unsigned long int DL, Lepcp, Pcrcp, l;
  unsigned short int Lepc, Pcrc;
  unsigned char Pepc;

  opj_cio_t *cio = j2k->cio;

  cio_write(cio, J2K_MS_EPC, 2);  /* EPC */
  Lepcp = cio_tell(cio);
  cio_skip(cio, 2);

  /* CRC-16 word of the EPC */
  Pcrc = 0x0000; /* initialize */
  Pcrcp = cio_tell(cio);
  cio_write(cio, Pcrc, 2); /* Pcrc placeholder*/

  /* data length of the EPC protection domain */
  DL = 0x00000000; /* we leave this set to 0, as if the information is not available */
  cio_write(cio, DL, 4);   /* DL */

  /* jpwl capabilities */
  Pepc = 0x00;
  cio_write(cio, Pepc, 1); /* Pepc */

  /* ID section */
  /* no ID's, as of now */

  Lepc = (unsigned short) (cio_tell(cio) - Lepcp);
  cio_seek(cio, Lepcp);
  cio_write(cio, Lepc, 2); /* Lepc */

  /* compute Pcrc */
  cio_seek(cio, Lepcp - 2);

    /* Marker */
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));

    /* Length */
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));
    jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));

    /* skip Pcrc */
    cio_skip(cio, 2);

    /* read all remaining */
    for (l = 4; l < Lepc; l++)
      jpwl_updateCRC16(&Pcrc, (unsigned char) cio_read(cio, 1));

    /* fill Pcrc with the result */
    cio_seek(cio, Pcrcp);
    cio_write(cio, Pcrc, 2);

  cio_seek(cio, Lepcp + Lepc);

  /* marker struct update */
  j2k_add_marker(j2k->cstr_info, J2K_MS_EPC, Lepcp - 2, Lepc + 2);

}

void j2k_read_epb(opj_j2k_t *j2k) {
  unsigned long int LDPepb, Pepb;
  unsigned short int Lepb;
  unsigned char Depb;
  char str1[25] = "";
  bool status;
  static bool first_in_tph = true;
  int type, pre_len, post_len;
  static unsigned char *redund = NULL;

  opj_cio_t *cio = j2k->cio;

  /* B/W = 45, RGB = 51 */
  /*           SIZ   SIZ_FIELDS     SIZ_COMPS               FOLLOWING_MARKER */
  int skipnum = 2  +     38     + 3 * j2k->cp->exp_comps  +         2;

  if (j2k->cp->correct) {

    /* go back to EPB marker value */
    cio_seek(cio, cio_tell(cio) - 2);

    /* we need to understand where we are */
    if (j2k->state == J2K_STATE_MH) {
      /* we are in MH */
      type = 0; /* MH */
      pre_len = skipnum; /* SOC+SIZ */
      post_len = -1; /* auto */

    } else if ((j2k->state == J2K_STATE_TPH) && first_in_tph) {
      /* we are in TPH */
      type = 1; /* TPH */
      pre_len = 12; /* SOC+SIZ */
      first_in_tph = false;
      post_len = -1; /* auto */

    } else {
      /* we are elsewhere */
      type = 2; /* other */
      pre_len = 0; /* nada */
      post_len = -1; /* auto */

    }

    /* call EPB corrector */
    /*printf("before %x, ", redund);*/
    status = jpwl_epb_correct(j2k,      /* J2K decompressor handle */
                  cio->bp,  /* pointer to EPB in codestream buffer */
                  type,     /* EPB type: MH */
                  pre_len,  /* length of pre-data */
                  post_len, /* length of post-data: -1 means auto */
                  NULL,     /* do everything auto */
                  &redund
                 );
    /*printf("after %x\n", redund);*/

    /* Read the (possibly corrected) EPB parameters */
    cio_skip(cio, 2);
    Lepb = cio_read(cio, 2);
    Depb = cio_read(cio, 1);
    LDPepb = cio_read(cio, 4);
    Pepb = cio_read(cio, 4);

    if (!status) {

      opj_event_msg(j2k->cinfo, EVT_ERROR, "JPWL correction could not be performed\n");

      /* advance to EPB endpoint */
      cio_skip(cio, Lepb + 2);

      return;
    }

    /* last in current header? */
    if (Depb & 0x40) {
      redund = NULL; /* reset the pointer to L4 buffer */
      first_in_tph = true;
    }

    /* advance to EPB endpoint */
    cio_skip(cio, Lepb - 11);

  } else {

    /* Simply read the EPB parameters */
    Lepb = cio_read(cio, 2);
    Depb = cio_read(cio, 1);
    LDPepb = cio_read(cio, 4);
    Pepb = cio_read(cio, 4);

    /* What does Pepb tells us about the protection method? */
    if (((Pepb & 0xF0000000) >> 28) == 0)
      sprintf(str1, "pred"); /* predefined */
    else if (((Pepb & 0xF0000000) >> 28) == 1)
      sprintf(str1, "crc-%lu", 16 * ((Pepb & 0x00000001) + 1)); /* CRC mode */
    else if (((Pepb & 0xF0000000) >> 28) == 2)
      sprintf(str1, "rs(%lu,32)", (Pepb & 0x0000FF00) >> 8); /* RS mode */
    else if (Pepb == 0xFFFFFFFF)
      sprintf(str1, "nometh"); /* RS mode */
    else
      sprintf(str1, "unknown"); /* unknown */

    /* Now we write them to screen */
    opj_event_msg(j2k->cinfo, EVT_INFO,
      "EPB(%d): (%sl, %sp, %u), %lu, %s\n",
      cio_tell(cio) - 13,
      (Depb & 0x40) ? "" : "n", /* latest EPB or not? */
      (Depb & 0x80) ? "" : "n", /* packed or unpacked EPB? */
      (Depb & 0x3F), /* EPB index value */
      LDPepb, /*length of the data protected by the EPB */
      str1); /* protection method */

    cio_skip(cio, Lepb - 11);
  }
}

void j2k_write_epb(opj_j2k_t *j2k) {
  unsigned long int LDPepb, Pepb, Lepbp;
  unsigned short int Lepb;
  unsigned char Depb;

  opj_cio_t *cio = j2k->cio;

  cio_write(cio, J2K_MS_EPB, 2);  /* EPB */
  Lepbp = cio_tell(cio);
  cio_skip(cio, 2);

  /* EPB style */
  Depb = 0x00; /* test */
  cio_write(cio, Depb, 1);   /* Depb */

  /* length of the data to be protected by this EPB */
  LDPepb = 0x00000000; /* test */
  cio_write(cio, LDPepb, 4);   /* LDPepb */

  /* next error correction tool */
  Pepb = 0x00000000; /* test */
  cio_write(cio, Pepb, 4);   /* Pepb */

  /* EPB data */
  /* no data, as of now */

  Lepb = (unsigned short) (cio_tell(cio) - Lepbp);
  cio_seek(cio, Lepbp);
  cio_write(cio, Lepb, 2);    /* Lepb */

  cio_seek(cio, Lepbp + Lepb);

  /* marker struct update */
  j2k_add_marker(j2k->cstr_info, J2K_MS_EPB, Lepbp - 2, Lepb + 2);
}

void j2k_read_esd(opj_j2k_t *j2k) {
  unsigned short int Lesd, Cesd;
  unsigned char Pesd;

  int cesdsize = (j2k->image->numcomps >= 257) ? 2 : 1;

  char str1[4][4] = {"p", "br", "pr", "res"};
  char str2[8][8] = {"res", "mse", "mse-r", "psnr", "psnr-i", "maxerr", "tse", "res"};

  opj_cio_t *cio = j2k->cio;

  /* Simply read the ESD parameters */
  Lesd = cio_read(cio, 2);
  Cesd = cio_read(cio, cesdsize);
  Pesd = cio_read(cio, 1);

  /* Now we write them to screen */
  opj_event_msg(j2k->cinfo, EVT_INFO,
    "ESD(%d): c%d, %s, %s, %s, %s, %s\n",
    cio_tell(cio) - (5 + cesdsize),
    Cesd, /* component number for this ESD */
    str1[(Pesd & (unsigned char) 0xC0) >> 6], /* addressing mode */
    str2[(Pesd & (unsigned char) 0x38) >> 3], /* sensitivity type */
    ((Pesd & (unsigned char) 0x04) >> 2) ? "2Bs" : "1Bs",
    ((Pesd & (unsigned char) 0x02) >> 1) ? "4Ba" : "2Ba",
    (Pesd & (unsigned char) 0x01) ? "avgc" : "");

  cio_skip(cio, Lesd - (3 + cesdsize));
}

void j2k_read_red(opj_j2k_t *j2k) {
  unsigned short int Lred;
  unsigned char Pred;
  char str1[4][4] = {"p", "br", "pr", "res"};

  opj_cio_t *cio = j2k->cio;

  /* Simply read the RED parameters */
  Lred = cio_read(cio, 2);
  Pred = cio_read(cio, 1);

  /* Now we write them to screen */
  opj_event_msg(j2k->cinfo, EVT_INFO,
    "RED(%d): %s, %dc, %s, %s\n",
    cio_tell(cio) - 5,
    str1[(Pred & (unsigned char) 0xC0) >> 6], /* addressing mode */
    (Pred & (unsigned char) 0x38) >> 3, /* corruption level */
    ((Pred & (unsigned char) 0x02) >> 1) ? "4Ba" : "2Ba", /* address range */
    (Pred & (unsigned char) 0x01) ? "errs" : "free"); /* error free? */

  cio_skip(cio, Lred - 3);
}

bool jpwl_check_tile(opj_j2k_t *j2k, opj_tcd_t *tcd, int tileno) {

#ifdef oerhgierhgvhreit4u
  /*
     we navigate through the tile and find possible invalid parameters:
       this saves a lot of crashes!!!!!
   */
  int compno, resno, precno, /*layno,*/ bandno, blockno;
  int numprecincts, numblocks;

  /* this is the selected tile */
  opj_tcd_tile_t *tile = &(tcd->tcd_image->tiles[tileno]);

  /* will keep the component */
  opj_tcd_tilecomp_t *comp = NULL;

  /* will keep the resolution */
  opj_tcd_resolution_t *res;

  /* will keep the subband */
  opj_tcd_band_t *band;

  /* will keep the precinct */
  opj_tcd_precinct_t *prec;

  /* will keep the codeblock */
  opj_tcd_cblk_t *block;

  /* check all tile components */
  for (compno = 0; compno < tile->numcomps; compno++) {
    comp = &(tile->comps[compno]);

    /* check all component resolutions */
    for (resno = 0; resno < comp->numresolutions; resno++) {
      res = &(comp->resolutions[resno]);
      numprecincts = res->pw * res->ph;

      /* check all the subbands */
      for (bandno = 0; bandno < res->numbands; bandno++) {
        band = &(res->bands[bandno]);

        /* check all the precincts */
        for (precno = 0; precno < numprecincts; precno++) {
          prec = &(band->precincts[precno]);
          numblocks = prec->ch * prec->cw;

          /* check all the codeblocks */
          for (blockno = 0; blockno < numblocks; blockno++) {
            block = &(prec->cblks[blockno]);

            /* x-origin is invalid */
            if ((block->x0 < prec->x0) || (block->x0 > prec->x1)) {
              opj_event_msg(j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
                "JPWL: wrong x-cord of block origin %d => x-prec is (%d, %d)\n",
                block->x0, prec->x0, prec->x1);
              if (!JPWL_ASSUME || JPWL_ASSUME)
                return false;
            };
          }
        }
      }
    }
  }

#endif

  return true;
}

/*@}*/

#endif /* USE_JPWL */


#ifdef USE_JPSEC

/** @defgroup JPSEC JPSEC - JPEG-2000 Part 8 (JPSEC) codestream manager */
/*@{*/


/** @name Local static functions */
/*@{*/

void j2k_read_sec(opj_j2k_t *j2k) {
  unsigned short int Lsec;

  opj_cio_t *cio = j2k->cio;

  /* Simply read the SEC length */
  Lsec = cio_read(cio, 2);

  /* Now we write them to screen */
  opj_event_msg(j2k->cinfo, EVT_INFO,
    "SEC(%d)\n",
    cio_tell(cio) - 2
    );

  cio_skip(cio, Lsec - 2);
}

void j2k_write_sec(opj_j2k_t *j2k) {
  unsigned short int Lsec = 24;
  int i;

  opj_cio_t *cio = j2k->cio;

  cio_write(cio, J2K_MS_SEC, 2);  /* SEC */
  cio_write(cio, Lsec, 2);

  /* write dummy data */
  for (i = 0; i < Lsec - 2; i++)
    cio_write(cio, 0, 1);
}

void j2k_read_insec(opj_j2k_t *j2k) {
  unsigned short int Linsec;

  opj_cio_t *cio = j2k->cio;

  /* Simply read the INSEC length */
  Linsec = cio_read(cio, 2);

  /* Now we write them to screen */
  opj_event_msg(j2k->cinfo, EVT_INFO,
    "INSEC(%d)\n",
    cio_tell(cio) - 2
    );

  cio_skip(cio, Linsec - 2);
}


/*@}*/

/*@}*/

#endif /* USE_JPSEC */
