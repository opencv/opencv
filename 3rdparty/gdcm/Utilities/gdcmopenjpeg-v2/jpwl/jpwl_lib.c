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

#ifdef USE_JPWL

#include "../libopenjpeg/opj_includes.h"
#include <limits.h>

/** Minimum and maximum values for the double->pfp conversion */
#define MIN_V1 0.0
#define MAX_V1 17293822569102704640.0
#define MIN_V2 0.000030517578125
#define MAX_V2 131040.0

/** conversion between a double precision floating point
number and the corresponding pseudo-floating point used
to represent sensitivity values
@param V the double precision value
@param bytes the number of bytes of the representation
@return the pseudo-floating point value (cast accordingly)
*/
unsigned short int jpwl_double_to_pfp(double V, int bytes);

/** conversion between a pseudo-floating point used
to represent sensitivity values and the corresponding
double precision floating point number
@param em the pseudo-floating point value (cast accordingly)
@param bytes the number of bytes of the representation
@return the double precision value
*/
double jpwl_pfp_to_double(unsigned short int em, int bytes);

  /*-------------------------------------------------------------*/

int jpwl_markcomp(const void *arg1, const void *arg2)
{
   /* Compare the two markers' positions */
   double diff = (((jpwl_marker_t *) arg1)->dpos - ((jpwl_marker_t *) arg2)->dpos);

   if (diff == 0.0)
     return (0);
   else if (diff < 0)
     return (-1);
   else
     return (+1);
}

int jpwl_epbs_add(opj_j2k_t *j2k, jpwl_marker_t *jwmarker, int *jwmarker_num,
          bool latest, bool packed, bool insideMH, int *idx, int hprot,
          double place_pos, int tileno,
          unsigned long int pre_len, unsigned long int post_len) {

  jpwl_epb_ms_t *epb_mark = NULL;

  int k_pre, k_post, n_pre, n_post;

  unsigned long int L1, L2, dL4, max_postlen, epbs_len = 0;

  /* We find RS(n,k) for EPB parms and pre-data, if any */
  if (insideMH && (*idx == 0)) {
    /* First EPB in MH */
    k_pre = 64;
    n_pre = 160;
  } else if (!insideMH && (*idx == 0)) {
    /* First EPB in TH */
    k_pre = 25;
    n_pre = 80;
  } else {
    /* Following EPBs in MH or TH */
    k_pre = 13;
    n_pre = 40;
  };

  /* Find lengths, Figs. B3 and B4 */
  /* size of pre data: pre_buf(pre_len) + EPB(2) + Lepb(2) + Depb(1) + LDPepb(4) + Pepb(4) */
  L1 = pre_len + 13;

  /* size of pre-data redundancy */
  /*   (redundancy per codeword)       *     (number of codewords, rounded up)   */
  L2 = (n_pre - k_pre) * (unsigned long int) ceil((double) L1 / (double) k_pre);

  /* Find protection type for post data and its associated redundancy field length*/
  if ((hprot == 16) || (hprot == 32)) {
    /* there is a CRC for post-data */
    k_post = post_len;
    n_post = post_len + (hprot >> 3);
    /*L3 = hprot >> 3;*/ /* 2 (CRC-16) or 4 (CRC-32) bytes */

  } else if ((hprot >= 37) && (hprot <= 128)) {
    /* there is a RS for post-data */
    k_post = 32;
    n_post = hprot;

  } else {
    /* Use predefined codes */
    n_post = n_pre;
    k_post = k_pre;
  };

  /* Create the EPB(s) */
  while (post_len > 0) {

    /* maximum postlen in order to respect EPB size
    (we use JPWL_MAXIMUM_EPB_ROOM instead of 65535 for keeping room for EPB parms)*/
    /*      (message word size)    *            (number of containable parity words)  */
    max_postlen = k_post * (unsigned long int) floor((double) JPWL_MAXIMUM_EPB_ROOM / (double) (n_post - k_post));

    /* maximum postlen in order to respect EPB size */
    if (*idx == 0)
      /* (we use (JPWL_MAXIMUM_EPB_ROOM - L2) instead of 65535 for keeping room for EPB parms + pre-data) */
      /*      (message word size)    *                   (number of containable parity words)  */
      max_postlen = k_post * (unsigned long int) floor((double) (JPWL_MAXIMUM_EPB_ROOM - L2) / (double) (n_post - k_post));

    else
      /* (we use JPWL_MAXIMUM_EPB_ROOM instead of 65535 for keeping room for EPB parms) */
      /*      (message word size)    *            (number of containable parity words)  */
      max_postlen = k_post * (unsigned long int) floor((double) JPWL_MAXIMUM_EPB_ROOM / (double) (n_post - k_post));

    /* null protection case */
    /* the max post length can be as large as the LDPepb field can host */
    if (hprot == 0)
      max_postlen = INT_MAX;

    /* length to use */
    dL4 = min(max_postlen, post_len);

    if ((epb_mark = jpwl_epb_create(
      j2k, /* this encoder handle */
      latest ? (dL4 < max_postlen) : false, /* is it the latest? */
      packed, /* is it packed? */
      tileno, /* we are in TPH */
      *idx, /* its index */
      hprot, /* protection type parameters of following data */
      0, /* pre-data: nothing for now */
      dL4 /* post-data: the stub computed previously */
      ))) {

      /* Add this marker to the 'insertanda' list */
      if (*jwmarker_num < JPWL_MAX_NO_MARKERS) {
        jwmarker[*jwmarker_num].id = J2K_MS_EPB; /* its type */
        jwmarker[*jwmarker_num].m.epbmark = epb_mark; /* the EPB */
        jwmarker[*jwmarker_num].pos = (int) place_pos; /* after SOT */
        jwmarker[*jwmarker_num].dpos = place_pos + 0.0000001 * (double)(*idx); /* not very first! */
        jwmarker[*jwmarker_num].len = epb_mark->Lepb; /* its length */
        jwmarker[*jwmarker_num].len_ready = true; /* ready */
        jwmarker[*jwmarker_num].pos_ready = true; /* ready */
        jwmarker[*jwmarker_num].parms_ready = true; /* ready */
        jwmarker[*jwmarker_num].data_ready = false; /* not ready */
        (*jwmarker_num)++;
      }

      /* increment epb index */
      (*idx)++;

      /* decrease postlen */
      post_len -= dL4;

      /* increase the total length of EPBs */
      epbs_len += epb_mark->Lepb + 2;

    } else {
      /* ooops, problems */
      opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not create TPH EPB for UEP in tile %d\n", tileno);
    };
  }

  return epbs_len;
}


jpwl_epb_ms_t *jpwl_epb_create(opj_j2k_t *j2k, bool latest, bool packed, int tileno, int idx, int hprot,
              unsigned long int pre_len, unsigned long int post_len) {

  jpwl_epb_ms_t *epb = NULL;
  /*unsigned short int data_len = 0;*/
  unsigned short int L2, L3;
  unsigned long int L1, L4;
  /*unsigned char *predata_in = NULL;*/

  bool insideMH = (tileno == -1);

  /* Alloc space */
  if (!(epb = (jpwl_epb_ms_t *) opj_malloc((size_t) 1 * sizeof (jpwl_epb_ms_t)))) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not allocate room for one EPB MS\n");
    return NULL;
  };

  /* We set RS(n,k) for EPB parms and pre-data, if any */
  if (insideMH && (idx == 0)) {
    /* First EPB in MH */
    epb->k_pre = 64;
    epb->n_pre = 160;
  } else if (!insideMH && (idx == 0)) {
    /* First EPB in TH */
    epb->k_pre = 25;
    epb->n_pre = 80;
  } else {
    /* Following EPBs in MH or TH */
    epb->k_pre = 13;
    epb->n_pre = 40;
  };

  /* Find lengths, Figs. B3 and B4 */
  /* size of pre data: pre_buf(pre_len) + EPB(2) + Lepb(2) + Depb(1) + LDPepb(4) + Pepb(4) */
  L1 = pre_len + 13;
  epb->pre_len = pre_len;

  /* size of pre-data redundancy */
  /*   (redundancy per codeword)       *               (number of codewords, rounded up)   */
  L2 = (epb->n_pre - epb->k_pre) * (unsigned short int) ceil((double) L1 / (double) epb->k_pre);

  /* length of post-data */
  L4 = post_len;
  epb->post_len = post_len;

  /* Find protection type for post data and its associated redundancy field length*/
  if ((hprot == 16) || (hprot == 32)) {
    /* there is a CRC for post-data */
    epb->Pepb = 0x10000000 | ((unsigned long int) hprot >> 5); /* 0=CRC-16, 1=CRC-32 */
    epb->k_post = post_len;
    epb->n_post = post_len + (hprot >> 3);
    /*L3 = hprot >> 3;*/ /* 2 (CRC-16) or 4 (CRC-32) bytes */

  } else if ((hprot >= 37) && (hprot <= 128)) {
    /* there is a RS for post-data */
    epb->Pepb = 0x20000020 | (((unsigned long int) hprot & 0x000000FF) << 8);
    epb->k_post = 32;
    epb->n_post = hprot;

  } else if (hprot == 1) {
    /* Use predefined codes */
    epb->Pepb = (unsigned long int) 0x00000000;
    epb->n_post = epb->n_pre;
    epb->k_post = epb->k_pre;

  } else if (hprot == 0) {
    /* Placeholder EPB: only protects its parameters, no protection method */
    epb->Pepb = (unsigned long int) 0xFFFFFFFF;
    epb->n_post = 1;
    epb->k_post = 1;

  } else {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Invalid protection value for EPB h = %d\n", hprot);
    return NULL;
  }

  epb->hprot = hprot;

  /*   (redundancy per codeword)          *                (number of codewords, rounded up) */
  L3 = (epb->n_post - epb->k_post) * (unsigned short int) ceil((double) L4 / (double) epb->k_post);

  /* private fields */
  epb->tileno = tileno;

  /* Fill some fields of the EPB */

  /* total length of the EPB MS (less the EPB marker itself): */
  /* Lepb(2) + Depb(1) + LDPepb(4) + Pepb(4) + pre_redundancy + post-redundancy */
  epb->Lepb = 11 + L2 + L3;

  /* EPB style */
  epb->Depb = ((packed & 0x0001) << 7) | ((latest & 0x0001) << 6) | (idx & 0x003F);

  /* length of data protected by EPB: */
  epb->LDPepb = L1 + L4;

  return epb;
}

void jpwl_epb_write(opj_j2k_t *j2k, jpwl_epb_ms_t *epb, unsigned char *buf) {

  /* Marker */
  *(buf++) = (unsigned char) (J2K_MS_EPB >> 8);
  *(buf++) = (unsigned char) (J2K_MS_EPB >> 0);

  /* Lepb */
  *(buf++) = (unsigned char) (epb->Lepb >> 8);
  *(buf++) = (unsigned char) (epb->Lepb >> 0);

  /* Depb */
  *(buf++) = (unsigned char) (epb->Depb >> 0);

  /* LDPepb */
  *(buf++) = (unsigned char) (epb->LDPepb >> 24);
  *(buf++) = (unsigned char) (epb->LDPepb >> 16);
  *(buf++) = (unsigned char) (epb->LDPepb >> 8);
  *(buf++) = (unsigned char) (epb->LDPepb >> 0);

  /* Pepb */
  *(buf++) = (unsigned char) (epb->Pepb >> 24);
  *(buf++) = (unsigned char) (epb->Pepb >> 16);
  *(buf++) = (unsigned char) (epb->Pepb >> 8);
  *(buf++) = (unsigned char) (epb->Pepb >> 0);

  /* Data */
  /*memcpy(buf, epb->data, (size_t) epb->Lepb - 11);*/
  memset(buf, 0, (size_t) epb->Lepb - 11);

  /* update markers struct */
  j2k_add_marker(j2k->cstr_info, J2K_MS_EPB, -1, epb->Lepb + 2);

};


jpwl_epc_ms_t *jpwl_epc_create(opj_j2k_t *j2k, bool esd_on, bool red_on, bool epb_on, bool info_on) {

  jpwl_epc_ms_t *epc = NULL;

  /* Alloc space */
  if (!(epc = (jpwl_epc_ms_t *) opj_malloc((size_t) 1 * sizeof (jpwl_epc_ms_t)))) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not allocate room for EPC MS\n");
    return NULL;
  };

  /* Set the EPC parameters */
  epc->esd_on = esd_on;
  epc->epb_on = epb_on;
  epc->red_on = red_on;
  epc->info_on = info_on;

  /* Fill the EPC fields with default values */
  epc->Lepc = 9;
  epc->Pcrc = 0x0000;
  epc->DL = 0x00000000;
  epc->Pepc = ((j2k->cp->esd_on & 0x0001) << 4) | ((j2k->cp->red_on & 0x0001) << 5) |
    ((j2k->cp->epb_on & 0x0001) << 6) | ((j2k->cp->info_on & 0x0001) << 7);

  return (epc);
}

bool jpwl_epb_fill(opj_j2k_t *j2k, jpwl_epb_ms_t *epb, unsigned char *buf, unsigned char *post_buf) {

  unsigned long int L1, L2, L3, L4;
  int remaining;
  unsigned long int P, NN_P;

  /* Operating buffer */
  static unsigned char codeword[NN], *parityword;

  unsigned char *L1_buf, *L2_buf;
  /* these ones are static, since we need to keep memory of
   the exact place from one call to the other */
  static unsigned char *L3_buf, *L4_buf;

  /* some consistency check */
  if (!buf) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "There is no operating buffer for EPBs\n");
    return false;
  }

  if (!post_buf && !L4_buf) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "There is no operating buffer for EPBs data\n");
    return false;
  }

  /*
   * Compute parity bytes on pre-data, ALWAYS present (at least only for EPB parms)
   */

  /* Initialize RS structures */
  P = epb->n_pre - epb->k_pre;
  NN_P = NN - P;
  memset(codeword, 0, NN);
  parityword = codeword + NN_P;
  init_rs(NN_P);

  /* pre-data begins pre_len bytes before of EPB buf */
  L1_buf = buf - epb->pre_len;
  L1 = epb->pre_len + 13;

  /* redundancy for pre-data begins immediately after EPB parms */
  L2_buf = buf + 13;
  L2 = (epb->n_pre - epb->k_pre) * (unsigned short int) ceil((double) L1 / (double) epb->k_pre);

  /* post-data
     the position of L4 buffer can be:
       1) passed as a parameter: in that case use it
       2) null: in that case use the previous (static) one
  */
  if (post_buf)
    L4_buf = post_buf;
  L4 = epb->post_len;

  /* post-data redundancy begins immediately after pre-data redundancy */
  L3_buf = L2_buf + L2;
  L3 = (epb->n_post - epb->k_post) * (unsigned short int) ceil((double) L4 / (double) epb->k_post);

  /* let's check whether EPB length is sufficient to contain all these data */
  if (epb->Lepb < (11 + L2 + L3))
    opj_event_msg(j2k->cinfo, EVT_ERROR, "There is no room in EPB data field for writing redundancy data\n");
  /*printf("Env. %d, nec. %d (%d + %d)\n", epb->Lepb - 11, L2 + L3, L2, L3);*/

  /* Compute redundancy of pre-data message words */
  remaining = L1;
  while (remaining) {

    /* copy message data into codeword buffer */
    if (remaining < epb->k_pre) {
      /* the last message word is zero-padded */
      memset(codeword, 0, NN);
      memcpy(codeword, L1_buf, remaining);
      L1_buf += remaining;
      remaining = 0;

    } else {
      memcpy(codeword, L1_buf, epb->k_pre);
      L1_buf += epb->k_pre;
      remaining -= epb->k_pre;

    }

    /* Encode the buffer and obtain parity bytes */
    if (encode_rs(codeword, parityword))
      opj_event_msg(j2k->cinfo, EVT_WARNING,
        "Possible encoding error in codeword @ position #%d\n", (L1_buf - buf) / epb->k_pre);

    /* copy parity bytes only in redundancy buffer */
    memcpy(L2_buf, parityword, P);

    /* advance parity buffer */
    L2_buf += P;
  }

  /*
   * Compute parity bytes on post-data, may be absent if there are no data
   */
  /*printf("Hprot is %d (tileno=%d, k_pre=%d, n_pre=%d, k_post=%d, n_post=%d, pre_len=%d, post_len=%d)\n",
    epb->hprot, epb->tileno, epb->k_pre, epb->n_pre, epb->k_post, epb->n_post, epb->pre_len,
    epb->post_len);*/
  if (epb->hprot < 0) {

    /* there should be no EPB */

  } else if (epb->hprot == 0) {

    /* no protection for the data */
    /* advance anyway */
    L4_buf += epb->post_len;

  } else if (epb->hprot == 16) {

    /* CRC-16 */
    unsigned short int mycrc = 0x0000;

    /* compute the CRC field (excluding itself) */
    remaining = L4;
    while (remaining--)
      jpwl_updateCRC16(&mycrc, *(L4_buf++));

    /* write the CRC field */
    *(L3_buf++) = (unsigned char) (mycrc >> 8);
    *(L3_buf++) = (unsigned char) (mycrc >> 0);

  } else if (epb->hprot == 32) {

    /* CRC-32 */
    unsigned long int mycrc = 0x00000000;

    /* compute the CRC field (excluding itself) */
    remaining = L4;
    while (remaining--)
      jpwl_updateCRC32(&mycrc, *(L4_buf++));

    /* write the CRC field */
    *(L3_buf++) = (unsigned char) (mycrc >> 24);
    *(L3_buf++) = (unsigned char) (mycrc >> 16);
    *(L3_buf++) = (unsigned char) (mycrc >> 8);
    *(L3_buf++) = (unsigned char) (mycrc >> 0);

  } else {

    /* RS */

    /* Initialize RS structures */
    P = epb->n_post - epb->k_post;
    NN_P = NN - P;
    memset(codeword, 0, NN);
    parityword = codeword + NN_P;
    init_rs(NN_P);

    /* Compute redundancy of post-data message words */
    remaining = L4;
    while (remaining) {

      /* copy message data into codeword buffer */
      if (remaining < epb->k_post) {
        /* the last message word is zero-padded */
        memset(codeword, 0, NN);
        memcpy(codeword, L4_buf, remaining);
        L4_buf += remaining;
        remaining = 0;

      } else {
        memcpy(codeword, L4_buf, epb->k_post);
        L4_buf += epb->k_post;
        remaining -= epb->k_post;

      }

      /* Encode the buffer and obtain parity bytes */
      if (encode_rs(codeword, parityword))
        opj_event_msg(j2k->cinfo, EVT_WARNING,
          "Possible encoding error in codeword @ position #%d\n", (L4_buf - buf) / epb->k_post);

      /* copy parity bytes only in redundancy buffer */
      memcpy(L3_buf, parityword, P);

      /* advance parity buffer */
      L3_buf += P;
    }

  }

  return true;
}


bool jpwl_correct(opj_j2k_t *j2k) {

  opj_cio_t *cio = j2k->cio;
  bool status;
  static bool mh_done = false;
  int mark_pos, id, len, skips, sot_pos;
  unsigned long int Psot = 0;

  /* go back to marker position */
  mark_pos = cio_tell(cio) - 2;
  cio_seek(cio, mark_pos);

  if ((j2k->state == J2K_STATE_MHSOC) && !mh_done) {

    int mark_val = 0, skipnum = 0;

    /*
      COLOR IMAGE
      first thing to do, if we are here, is to look whether
      51 (skipnum) positions ahead there is an EPB, in case of MH
    */
    /*
      B/W IMAGE
      first thing to do, if we are here, is to look whether
      45 (skipnum) positions ahead there is an EPB, in case of MH
    */
    /*       SIZ   SIZ_FIELDS     SIZ_COMPS               FOLLOWING_MARKER */
    skipnum = 2  +     38     + 3 * j2k->cp->exp_comps  +         2;
    if ((cio->bp + skipnum) < cio->end) {

      cio_skip(cio, skipnum);

      /* check that you are not going beyond the end of codestream */

      /* call EPB corrector */
      status = jpwl_epb_correct(j2k,     /* J2K decompressor handle */
                    cio->bp, /* pointer to EPB in codestream buffer */
                    0,       /* EPB type: MH */
                    skipnum,      /* length of pre-data */
                    -1,      /* length of post-data: -1 means auto */
                    NULL,
                    NULL
                   );

      /* read the marker value */
      mark_val = (*(cio->bp) << 8) | *(cio->bp + 1);

      if (status && (mark_val == J2K_MS_EPB)) {
        /* we found it! */
        mh_done = true;
        return true;
      }

      /* Disable correction in case of missing or bad head EPB */
      /* We can't do better! */
      /* PATCHED: 2008-01-25 */
      /* MOVED UP: 2008-02-01 */
      if (!status) {
        j2k->cp->correct = false;
        opj_event_msg(j2k->cinfo, EVT_WARNING, "Couldn't find the MH EPB: disabling JPWL\n");
      }

    }

  }

  if (true /*(j2k->state == J2K_STATE_TPHSOT) || (j2k->state == J2K_STATE_TPH)*/) {
    /* else, look if 12 positions ahead there is an EPB, in case of TPH */
    cio_seek(cio, mark_pos);
    if ((cio->bp + 12) < cio->end) {

      cio_skip(cio, 12);

      /* call EPB corrector */
      status = jpwl_epb_correct(j2k,     /* J2K decompressor handle */
                    cio->bp, /* pointer to EPB in codestream buffer */
                    1,       /* EPB type: TPH */
                    12,      /* length of pre-data */
                    -1,      /* length of post-data: -1 means auto */
                    NULL,
                    NULL
                   );
      if (status)
        /* we found it! */
        return true;
    }
  }

  return false;

  /* for now, don't use this code */

  /* else, look if here is an EPB, in case of other */
  if (mark_pos > 64) {
    /* it cannot stay before the first MH EPB */
    cio_seek(cio, mark_pos);
    cio_skip(cio, 0);

    /* call EPB corrector */
    status = jpwl_epb_correct(j2k,     /* J2K decompressor handle */
                  cio->bp, /* pointer to EPB in codestream buffer */
                  2,       /* EPB type: TPH */
                  0,       /* length of pre-data */
                  -1,      /* length of post-data: -1 means auto */
                  NULL,
                  NULL
                 );
    if (status)
      /* we found it! */
      return true;
  }

  /* nope, no EPBs probably, or they are so damaged that we can give up */
  return false;

  return true;

  /* AN ATTEMPT OF PARSER */
  /* NOT USED ACTUALLY    */

  /* go to the beginning of the file */
  cio_seek(cio, 0);

  /* let's begin */
  j2k->state = J2K_STATE_MHSOC;

  /* cycle all over the markers */
  while (cio_tell(cio) < cio->length) {

    /* read the marker */
    mark_pos = cio_tell(cio);
    id = cio_read(cio, 2);

    /* details */
    printf("Marker@%d: %X\n", cio_tell(cio) - 2, id);

    /* do an action in response to the read marker */
    switch (id) {

    /* short markers */

      /* SOC */
    case J2K_MS_SOC:
      j2k->state = J2K_STATE_MHSIZ;
      len = 0;
      skips = 0;
      break;

      /* EOC */
    case J2K_MS_EOC:
      j2k->state = J2K_STATE_MT;
      len = 0;
      skips = 0;
      break;

      /* particular case of SOD */
    case J2K_MS_SOD:
      len = Psot - (mark_pos - sot_pos) - 2;
      skips = len;
      break;

    /* long markers */

      /* SOT */
    case J2K_MS_SOT:
      j2k->state = J2K_STATE_TPH;
      sot_pos = mark_pos; /* position of SOT */
      len = cio_read(cio, 2); /* read the length field */
      cio_skip(cio, 2); /* this field is unnecessary */
      Psot = cio_read(cio, 4); /* tile length */
      skips = len - 8;
      break;

      /* remaining */
    case J2K_MS_SIZ:
      j2k->state = J2K_STATE_MH;
      /* read the length field */
      len = cio_read(cio, 2);
      skips = len - 2;
      break;

      /* remaining */
    default:
      /* read the length field */
      len = cio_read(cio, 2);
      skips = len - 2;
      break;

    }

    /* skip to marker's end */
    cio_skip(cio, skips);

  }


}

bool jpwl_epb_correct(opj_j2k_t *j2k, unsigned char *buffer, int type, int pre_len, int post_len, int *conn,
            unsigned char **L4_bufp) {

  /* Operating buffer */
  unsigned char codeword[NN], *parityword;

  unsigned long int P, NN_P;
  unsigned long int L1, L4;
  int remaining, n_pre, k_pre, n_post, k_post;

  int status, tt;

  int orig_pos = cio_tell(j2k->cio);

  unsigned char *L1_buf, *L2_buf;
  unsigned char *L3_buf, *L4_buf;

  unsigned long int LDPepb, Pepb;
  unsigned short int Lepb;
  unsigned char Depb;
  char str1[25] = "";
  int myconn, errnum = 0;
  bool errflag = false;

  opj_cio_t *cio = j2k->cio;

  /* check for common errors */
  if (!buffer) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "The EPB pointer is a NULL buffer\n");
    return false;
  }

  /* set bignesses */
  L1 = pre_len + 13;

  /* pre-data correction */
  switch (type) {

  case 0:
    /* MH EPB */
    k_pre = 64;
    n_pre = 160;
    break;

  case 1:
    /* TPH EPB */
    k_pre = 25;
    n_pre = 80;
    break;

  case 2:
    /* other EPBs */
    k_pre = 13;
    n_pre = 40;
    break;

  case 3:
    /* automatic setup */
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Auto. setup not yet implemented\n");
    return false;
    break;

  default:
    /* unknown type */
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Unknown expected EPB type\n");
    return false;
    break;

  }

  /* Initialize RS structures */
  P = n_pre - k_pre;
  NN_P = NN - P;
  tt = (int) floor((float) P / 2.0F); /* correction capability of the code */
  memset(codeword, 0, NN);
  parityword = codeword + NN_P;
  init_rs(NN_P);

  /* Correct pre-data message words */
  L1_buf = buffer - pre_len;
  L2_buf = buffer + 13;
  remaining = L1;
  while (remaining) {

    /* always zero-pad codewords */
    /* (this is required, since after decoding the zeros in the long codeword
        could change, and keep unchanged in subsequent calls) */
    memset(codeword, 0, NN);

    /* copy codeword buffer into message bytes */
    if (remaining < k_pre)
      memcpy(codeword, L1_buf, remaining);
    else
      memcpy(codeword, L1_buf, k_pre);

    /* copy redundancy buffer in parity bytes */
    memcpy(parityword, L2_buf, P);

    /* Decode the buffer and possibly obtain corrected bytes */
    status = eras_dec_rs(codeword, NULL, 0);
    if (status == -1) {
      /*if (conn == NULL)
        opj_event_msg(j2k->cinfo, EVT_WARNING,
          "Possible decoding error in codeword @ position #%d\n", (L1_buf - buffer) / k_pre);*/
      errflag = true;
      /* we can try to safely get out from the function:
        if we are here, either this is not an EPB or the first codeword
        is too damaged to be helpful */
      /*return false;*/

    } else if (status == 0) {
      /*if (conn == NULL)
        opj_event_msg(j2k->cinfo, EVT_INFO, "codeword is correctly decoded\n");*/

    } else if (status <= tt) {
      /* it has corrected 0 <= errs <= tt */
      /*if (conn == NULL)
        opj_event_msg(j2k->cinfo, EVT_WARNING, "%d errors corrected in codeword\n", status);*/
      errnum += status;

    } else {
      /*if (conn == NULL)
        opj_event_msg(j2k->cinfo, EVT_WARNING, "EPB correction capability exceeded\n");
      return false;*/
      errflag = true;
    }


    /* advance parity buffer */
    if ((status >= 0) && (status <= tt))
      /* copy back corrected parity only if all is OK */
      memcpy(L2_buf, parityword, P);
    L2_buf += P;

    /* advance message buffer */
    if (remaining < k_pre) {
      if ((status >= 0) && (status <= tt))
        /* copy back corrected data only if all is OK */
        memcpy(L1_buf, codeword, remaining);
      L1_buf += remaining;
      remaining = 0;

    } else {
      if ((status >= 0) && (status <= tt))
        /* copy back corrected data only if all is OK */
        memcpy(L1_buf, codeword, k_pre);
      L1_buf += k_pre;
      remaining -= k_pre;

    }
  }

  /* print summary */
  if (!conn) {

    /*if (errnum)
      opj_event_msg(j2k->cinfo, EVT_INFO, "+ %d symbol errors corrected (Ps=%.1e)\n", errnum,
        (float) errnum / ((float) n_pre * (float) L1 / (float) k_pre));*/
    if (errflag) {
      /*opj_event_msg(j2k->cinfo, EVT_INFO, "+ there were unrecoverable errors\n");*/
      return false;
    }

  }

  /* presumably, now, EPB parameters are correct */
  /* let's get them */

  /* Simply read the EPB parameters */
  if (conn)
    cio->bp = buffer;
  cio_skip(cio, 2); /* the marker */
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
  if (!conn && post_len)
    opj_event_msg(j2k->cinfo, EVT_INFO,
      "EPB(%d): (%sl, %sp, %u), %lu, %s\n",
      cio_tell(cio) - 13,
      (Depb & 0x40) ? "" : "n", /* latest EPB or not? */
      (Depb & 0x80) ? "" : "n", /* packed or unpacked EPB? */
      (Depb & 0x3F), /* EPB index value */
      LDPepb, /*length of the data protected by the EPB */
      str1); /* protection method */


  /* well, we need to investigate how long is the connected length of packed EPBs */
  myconn = Lepb + 2;
  if ((Depb & 0x40) == 0) /* not latest in header */
    jpwl_epb_correct(j2k,      /* J2K decompressor handle */
               buffer + Lepb + 2,   /* pointer to next EPB in codestream buffer */
               2,     /* EPB type: should be of other type */
               0,  /* only EPB fields */
               0, /* do not look after */
             &myconn,
             NULL
               );
  if (conn)
    *conn += myconn;

  /*if (!conn)
    printf("connected = %d\n", myconn);*/

  /*cio_seek(j2k->cio, orig_pos);
  return true;*/

  /* post-data
     the position of L4 buffer is at the end of currently connected EPBs
  */
  if (!(L4_bufp))
    L4_buf = buffer + myconn;
  else if (!(*L4_bufp))
    L4_buf = buffer + myconn;
  else
    L4_buf = *L4_bufp;
  if (post_len == -1)
    L4 = LDPepb - pre_len - 13;
  else if (post_len == 0)
    L4 = 0;
  else
    L4 = post_len;

  L3_buf = L2_buf;

  /* Do a further check here on the read parameters */
  if (L4 > (unsigned long) cio_numbytesleft(j2k->cio))
    /* overflow */
    return false;

  /* we are ready for decoding the remaining data */
  if (((Pepb & 0xF0000000) >> 28) == 1) {
    /* CRC here */
    if ((16 * ((Pepb & 0x00000001) + 1)) == 16) {

      /* CRC-16 */
      unsigned short int mycrc = 0x0000, filecrc = 0x0000;

      /* compute the CRC field */
      remaining = L4;
      while (remaining--)
        jpwl_updateCRC16(&mycrc, *(L4_buf++));

      /* read the CRC field */
      filecrc = *(L3_buf++) << 8;
      filecrc |= *(L3_buf++);

      /* check the CRC field */
      if (mycrc == filecrc) {
        if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_INFO, "- CRC is OK\n");
      } else {
        if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_WARNING, "- CRC is KO (r=%d, c=%d)\n", filecrc, mycrc);
        errflag = true;
      }
    }

    if ((16 * ((Pepb & 0x00000001) + 1)) == 32) {

      /* CRC-32 */
      unsigned long int mycrc = 0x00000000, filecrc = 0x00000000;

      /* compute the CRC field */
      remaining = L4;
      while (remaining--)
        jpwl_updateCRC32(&mycrc, *(L4_buf++));

      /* read the CRC field */
      filecrc = *(L3_buf++) << 24;
      filecrc |= *(L3_buf++) << 16;
      filecrc |= *(L3_buf++) << 8;
      filecrc |= *(L3_buf++);

      /* check the CRC field */
      if (mycrc == filecrc) {
        if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_INFO, "- CRC is OK\n");
      } else {
        if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_WARNING, "- CRC is KO (r=%d, c=%d)\n", filecrc, mycrc);
        errflag = true;
      }
    }

  } else if (Pepb == 0xFFFFFFFF) {
    /* no method */

    /* advance without doing anything */
    remaining = L4;
    while (remaining--)
      L4_buf++;

  } else if ((((Pepb & 0xF0000000) >> 28) == 2) || (((Pepb & 0xF0000000) >> 28) == 0)) {
    /* RS coding here */

    if (((Pepb & 0xF0000000) >> 28) == 0) {

      k_post = k_pre;
      n_post = n_pre;

    } else {

      k_post = 32;
      n_post = (Pepb & 0x0000FF00) >> 8;
    }

    /* Initialize RS structures */
    P = n_post - k_post;
    NN_P = NN - P;
    tt = (int) floor((float) P / 2.0F); /* again, correction capability */
    memset(codeword, 0, NN);
    parityword = codeword + NN_P;
    init_rs(NN_P);

    /* Correct post-data message words */
    /*L4_buf = buffer + Lepb + 2;*/
    L3_buf = L2_buf;
    remaining = L4;
    while (remaining) {

      /* always zero-pad codewords */
      /* (this is required, since after decoding the zeros in the long codeword
        could change, and keep unchanged in subsequent calls) */
      memset(codeword, 0, NN);

      /* copy codeword buffer into message bytes */
      if (remaining < k_post)
        memcpy(codeword, L4_buf, remaining);
      else
        memcpy(codeword, L4_buf, k_post);

      /* copy redundancy buffer in parity bytes */
      memcpy(parityword, L3_buf, P);

      /* Decode the buffer and possibly obtain corrected bytes */
      status = eras_dec_rs(codeword, NULL, 0);
      if (status == -1) {
        /*if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_WARNING,
            "Possible decoding error in codeword @ position #%d\n", (L4_buf - (buffer + Lepb + 2)) / k_post);*/
        errflag = true;

      } else if (status == 0) {
        /*if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_INFO, "codeword is correctly decoded\n");*/

      } else if (status <= tt) {
        /*if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_WARNING, "%d errors corrected in codeword\n", status);*/
        errnum += status;

      } else {
        /*if (conn == NULL)
          opj_event_msg(j2k->cinfo, EVT_WARNING, "EPB correction capability exceeded\n");
        return false;*/
        errflag = true;
      }


      /* advance parity buffer */
      if ((status >= 0) && (status <= tt))
        /* copy back corrected data only if all is OK */
        memcpy(L3_buf, parityword, P);
      L3_buf += P;

      /* advance message buffer */
      if (remaining < k_post) {
        if ((status >= 0) && (status <= tt))
          /* copy back corrected data only if all is OK */
          memcpy(L4_buf, codeword, remaining);
        L4_buf += remaining;
        remaining = 0;

      } else {
        if ((status >= 0) && (status <= tt))
          /* copy back corrected data only if all is OK */
          memcpy(L4_buf, codeword, k_post);
        L4_buf += k_post;
        remaining -= k_post;

      }
    }
  }

  /* give back the L4_buf address */
  if (L4_bufp)
    *L4_bufp = L4_buf;

  /* print summary */
  if (!conn) {

    if (errnum)
      opj_event_msg(j2k->cinfo, EVT_INFO, "- %d symbol errors corrected (Ps=%.1e)\n", errnum,
        (float) errnum / (float) LDPepb);
    if (errflag)
      opj_event_msg(j2k->cinfo, EVT_INFO, "- there were unrecoverable errors\n");

  }

  cio_seek(j2k->cio, orig_pos);

  return true;
}

void jpwl_epc_write(opj_j2k_t *j2k, jpwl_epc_ms_t *epc, unsigned char *buf) {

  /* Marker */
  *(buf++) = (unsigned char) (J2K_MS_EPC >> 8);
  *(buf++) = (unsigned char) (J2K_MS_EPC >> 0);

  /* Lepc */
  *(buf++) = (unsigned char) (epc->Lepc >> 8);
  *(buf++) = (unsigned char) (epc->Lepc >> 0);

  /* Pcrc */
  *(buf++) = (unsigned char) (epc->Pcrc >> 8);
  *(buf++) = (unsigned char) (epc->Pcrc >> 0);

  /* DL */
  *(buf++) = (unsigned char) (epc->DL >> 24);
  *(buf++) = (unsigned char) (epc->DL >> 16);
  *(buf++) = (unsigned char) (epc->DL >> 8);
  *(buf++) = (unsigned char) (epc->DL >> 0);

  /* Pepc */
  *(buf++) = (unsigned char) (epc->Pepc >> 0);

  /* Data */
  /*memcpy(buf, epc->data, (size_t) epc->Lepc - 9);*/
  memset(buf, 0, (size_t) epc->Lepc - 9);

  /* update markers struct */
  j2k_add_marker(j2k->cstr_info, J2K_MS_EPC, -1, epc->Lepc + 2);

};

int jpwl_esds_add(opj_j2k_t *j2k, jpwl_marker_t *jwmarker, int *jwmarker_num,
          int comps, unsigned char addrm, unsigned char ad_size,
          unsigned char senst, unsigned char se_size,
          double place_pos, int tileno) {

  return 0;
}

jpwl_esd_ms_t *jpwl_esd_create(opj_j2k_t *j2k, int comp, unsigned char addrm, unsigned char ad_size,
                unsigned char senst, unsigned char se_size, int tileno,
                unsigned long int svalnum, void *sensval) {

  jpwl_esd_ms_t *esd = NULL;

  /* Alloc space */
  if (!(esd = (jpwl_esd_ms_t *) opj_malloc((size_t) 1 * sizeof (jpwl_esd_ms_t)))) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Could not allocate room for ESD MS\n");
    return NULL;
  };

  /* if relative sensitivity, activate byte range mode */
  if (senst == 0)
    addrm = 1;

  /* size of sensval's ... */
  if ((ad_size != 0) && (ad_size != 2) && (ad_size != 4)) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Address size %d for ESD MS is forbidden\n", ad_size);
    return NULL;
  }
  if ((se_size != 1) && (se_size != 2)) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Sensitivity size %d for ESD MS is forbidden\n", se_size);
    return NULL;
  }

  /* ... depends on the addressing mode */
  switch (addrm) {

  /* packet mode */
  case (0):
    ad_size = 0; /* as per the standard */
    esd->sensval_size = se_size;
    break;

  /* byte range */
  case (1):
    /* auto sense address size */
    if (ad_size == 0)
      /* if there are more than 66% of (2^16 - 1) bytes, switch to 4 bytes
       (we keep space for possible EPBs being inserted) */
      ad_size = (j2k->cstr_info->codestream_size > (1 * 65535 / 3)) ? 4 : 2;
    esd->sensval_size = ad_size + ad_size + se_size;
    break;

  /* packet range */
  case (2):
    /* auto sense address size */
    if (ad_size == 0)
      /* if there are more than 2^16 - 1 packets, switch to 4 bytes */
      ad_size = (j2k->cstr_info->packno > 65535) ? 4 : 2;
    esd->sensval_size = ad_size + ad_size + se_size;
    break;

  case (3):
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Address mode %d for ESD MS is unimplemented\n", addrm);
    return NULL;

  default:
    opj_event_msg(j2k->cinfo, EVT_ERROR, "Address mode %d for ESD MS is forbidden\n", addrm);
    return NULL;
  }

  /* set or unset sensitivity values */
  if (svalnum <= 0) {

    switch (senst) {

    /* just based on the portions of a codestream */
    case (0):
      /* MH + no. of THs + no. of packets */
      svalnum = 1 + (j2k->cstr_info->tw * j2k->cstr_info->th) * (1 + j2k->cstr_info->packno);
      break;

    /* all the ones that are based on the packets */
    default:
      if (tileno < 0)
        /* MH: all the packets and all the tiles info is written */
        svalnum = j2k->cstr_info->tw * j2k->cstr_info->th * j2k->cstr_info->packno;
      else
        /* TPH: only that tile info is written */
        svalnum = j2k->cstr_info->packno;
      break;

    }
  }

  /* fill private fields */
  esd->senst = senst;
  esd->ad_size = ad_size;
  esd->se_size = se_size;
  esd->addrm = addrm;
  esd->svalnum = svalnum;
  esd->numcomps = j2k->image->numcomps;
  esd->tileno = tileno;

  /* Set the ESD parameters */
  /* length, excluding data field */
  if (esd->numcomps < 257)
    esd->Lesd = 4 + (unsigned short int) (esd->svalnum * esd->sensval_size);
  else
    esd->Lesd = 5 + (unsigned short int) (esd->svalnum * esd->sensval_size);

  /* component data field */
  if (comp >= 0)
    esd->Cesd = comp;
  else
    /* we are averaging */
    esd->Cesd = 0;

  /* Pesd field */
  esd->Pesd = 0x00;
  esd->Pesd |= (esd->addrm & 0x03) << 6; /* addressing mode */
  esd->Pesd |= (esd->senst & 0x07) << 3; /* sensitivity type */
  esd->Pesd |= ((esd->se_size >> 1) & 0x01) << 2; /* sensitivity size */
  esd->Pesd |= ((esd->ad_size >> 2) & 0x01) << 1; /* addressing size */
  esd->Pesd |= (comp < 0) ? 0x01 : 0x00; /* averaging components */

  /* if pointer to sensval is NULL, we can fill data field by ourselves */
  if (!sensval) {

    /* old code moved to jpwl_esd_fill() */
    esd->data = NULL;

  } else {
      /* we set the data field as the sensitivity values poinnter passed to the function */
      esd->data = (unsigned char *) sensval;
  }

  return (esd);
}

bool jpwl_esd_fill(opj_j2k_t *j2k, jpwl_esd_ms_t *esd, unsigned char *buf) {

  int i;
  unsigned long int vv;
  unsigned long int addr1 = 0L, addr2 = 0L;
  double dvalue = 0.0, Omax2, tmp, TSE = 0.0, MSE, oldMSE = 0.0, PSNR, oldPSNR = 0.0;
  unsigned short int pfpvalue;
  unsigned long int addrmask = 0x00000000;
  bool doneMH = false, doneTPH = false;

  /* sensitivity values in image info are as follows:
    - for each tile, distotile is the starting distortion for that tile, sum of all components
    - for each packet in a tile, disto is the distortion reduction caused by that packet to that tile
    - the TSE for a single tile should be given by   distotile - sum(disto)  , for all components
    - the MSE for a single tile is given by     TSE / nbpix    , for all components
    - the PSNR for a single tile is given by   10*log10( Omax^2 / MSE)    , for all components
      (Omax is given by    2^bpp - 1    for unsigned images and by    2^(bpp - 1) - 1    for signed images
  */

  /* browse all components and find Omax */
  Omax2 = 0.0;
  for (i = 0; i < j2k->image->numcomps; i++) {
    tmp = pow(2.0, (double) (j2k->image->comps[i].sgnd ?
      (j2k->image->comps[i].bpp - 1) : (j2k->image->comps[i].bpp))) - 1;
    if (tmp > Omax2)
      Omax2 = tmp;
  }
  Omax2 = Omax2 * Omax2;

  /* if pointer of esd->data is not null, simply write down all the values byte by byte */
  if (esd->data) {
    for (i = 0; i < (int) esd->svalnum; i++)
      *(buf++) = esd->data[i];
    return true;
  }

  /* addressing mask */
  if (esd->ad_size == 2)
    addrmask = 0x0000FFFF; /* two bytes */
  else
    addrmask = 0xFFFFFFFF; /* four bytes */

  /* set on precise point where sensitivity starts */
  if (esd->numcomps < 257)
    buf += 6;
  else
    buf += 7;

  /* let's fill the data fields */
  for (vv = (esd->tileno < 0) ? 0 : (j2k->cstr_info->packno * esd->tileno); vv < esd->svalnum; vv++) {

    int thistile = vv / j2k->cstr_info->packno, thispacket = vv % j2k->cstr_info->packno;

    /* skip for the hack some lines below */
    if (thistile == j2k->cstr_info->tw * j2k->cstr_info->th)
      break;

    /* starting tile distortion */
    if (thispacket == 0) {
      TSE = j2k->cstr_info->tile[thistile].distotile;
      oldMSE = TSE / j2k->cstr_info->tile[thistile].numpix;
      oldPSNR = 10.0 * log10(Omax2 / oldMSE);
    }

    /* TSE */
    TSE -= j2k->cstr_info->tile[thistile].packet[thispacket].disto;

    /* MSE */
    MSE = TSE / j2k->cstr_info->tile[thistile].numpix;

    /* PSNR */
    PSNR = 10.0 * log10(Omax2 / MSE);

    /* fill the address range */
    switch (esd->addrm) {

    /* packet mode */
    case (0):
      /* nothing, there is none */
      break;

    /* byte range */
    case (1):
      /* start address of packet */
      addr1 = (j2k->cstr_info->tile[thistile].packet[thispacket].start_pos) & addrmask;
      /* end address of packet */
      addr2 = (j2k->cstr_info->tile[thistile].packet[thispacket].end_pos) & addrmask;
      break;

    /* packet range */
    case (2):
      /* not implemented here */
      opj_event_msg(j2k->cinfo, EVT_WARNING, "Addressing mode packet_range is not implemented\n");
      break;

    /* unknown addressing method */
    default:
      /* not implemented here */
      opj_event_msg(j2k->cinfo, EVT_WARNING, "Unknown addressing mode\n");
      break;

    }

    /* hack for writing relative sensitivity of MH and TPHs */
    if ((esd->senst == 0) && (thispacket == 0)) {

      /* possible MH */
      if ((thistile == 0) && !doneMH) {
        /* we have to manage MH addresses */
        addr1 = 0; /* start of MH */
        addr2 = j2k->cstr_info->main_head_end; /* end of MH */
        /* set special dvalue for this MH */
        dvalue = -10.0;
        doneMH = true; /* don't come here anymore */
        vv--; /* wrap back loop counter */

      } else if (!doneTPH) {
        /* we have to manage TPH addresses */
        addr1 = j2k->cstr_info->tile[thistile].start_pos;
        addr2 = j2k->cstr_info->tile[thistile].end_header;
        /* set special dvalue for this TPH */
        dvalue = -1.0;
        doneTPH = true; /* don't come here till the next tile */
        vv--; /* wrap back loop counter */
      }

    } else
      doneTPH = false; /* reset TPH counter */

    /* write the addresses to the buffer */
    switch (esd->ad_size) {

    case (0):
      /* do nothing */
      break;

    case (2):
      /* two bytes */
      *(buf++) = (unsigned char) (addr1 >> 8);
      *(buf++) = (unsigned char) (addr1 >> 0);
      *(buf++) = (unsigned char) (addr2 >> 8);
      *(buf++) = (unsigned char) (addr2 >> 0);
      break;

    case (4):
      /* four bytes */
      *(buf++) = (unsigned char) (addr1 >> 24);
      *(buf++) = (unsigned char) (addr1 >> 16);
      *(buf++) = (unsigned char) (addr1 >> 8);
      *(buf++) = (unsigned char) (addr1 >> 0);
      *(buf++) = (unsigned char) (addr2 >> 24);
      *(buf++) = (unsigned char) (addr2 >> 16);
      *(buf++) = (unsigned char) (addr2 >> 8);
      *(buf++) = (unsigned char) (addr2 >> 0);
      break;

    default:
      /* do nothing */
      break;
    }


    /* let's fill the value field */
    switch (esd->senst) {

    /* relative sensitivity */
    case (0):
      /* we just write down the packet ordering */
      if (dvalue == -10)
        /* MH */
        dvalue = MAX_V1 + 1000.0; /* this will cause pfpvalue set to 0xFFFF */
      else if (dvalue == -1)
        /* TPH */
        dvalue = MAX_V1 + 1000.0; /* this will cause pfpvalue set to 0xFFFF */
      else
        /* packet: first is most important, and then in decreasing order
        down to the last, which counts for 1 */
        dvalue = jpwl_pfp_to_double((unsigned short) (j2k->cstr_info->packno - thispacket), esd->se_size);
      break;

    /* MSE */
    case (1):
      /* !!! WRONG: let's put here disto field of packets !!! */
      dvalue = MSE;
      break;

    /* MSE reduction */
    case (2):
      dvalue = oldMSE - MSE;
      oldMSE = MSE;
      break;

    /* PSNR */
    case (3):
      dvalue = PSNR;
      break;

    /* PSNR increase */
    case (4):
      dvalue = PSNR - oldPSNR;
      oldPSNR = PSNR;
      break;

    /* MAXERR */
    case (5):
      dvalue = 0.0;
      opj_event_msg(j2k->cinfo, EVT_WARNING, "MAXERR sensitivity mode is not implemented\n");
      break;

    /* TSE */
    case (6):
      dvalue = TSE;
      break;

    /* reserved */
    case (7):
      dvalue = 0.0;
      opj_event_msg(j2k->cinfo, EVT_WARNING, "Reserved sensitivity mode is not implemented\n");
      break;

    default:
      dvalue = 0.0;
      break;
    }

    /* compute the pseudo-floating point value */
    pfpvalue = jpwl_double_to_pfp(dvalue, esd->se_size);

    /* write the pfp value to the buffer */
    switch (esd->se_size) {

    case (1):
      /* one byte */
      *(buf++) = (unsigned char) (pfpvalue >> 0);
      break;

    case (2):
      /* two bytes */
      *(buf++) = (unsigned char) (pfpvalue >> 8);
      *(buf++) = (unsigned char) (pfpvalue >> 0);
      break;
    }

  }

  return true;
}

void jpwl_esd_write(opj_j2k_t *j2k, jpwl_esd_ms_t *esd, unsigned char *buf) {

  /* Marker */
  *(buf++) = (unsigned char) (J2K_MS_ESD >> 8);
  *(buf++) = (unsigned char) (J2K_MS_ESD >> 0);

  /* Lesd */
  *(buf++) = (unsigned char) (esd->Lesd >> 8);
  *(buf++) = (unsigned char) (esd->Lesd >> 0);

  /* Cesd */
  if (esd->numcomps >= 257)
    *(buf++) = (unsigned char) (esd->Cesd >> 8);
  *(buf++) = (unsigned char) (esd->Cesd >> 0);

  /* Pesd */
  *(buf++) = (unsigned char) (esd->Pesd >> 0);

  /* Data */
  if (esd->numcomps < 257)
    memset(buf, 0xAA, (size_t) esd->Lesd - 4);
    /*memcpy(buf, esd->data, (size_t) esd->Lesd - 4);*/
  else
    memset(buf, 0xAA, (size_t) esd->Lesd - 5);
    /*memcpy(buf, esd->data, (size_t) esd->Lesd - 5);*/

  /* update markers struct */
  j2k_add_marker(j2k->cstr_info, J2K_MS_ESD, -1, esd->Lesd + 2);

}

unsigned short int jpwl_double_to_pfp(double V, int bytes) {

  unsigned short int em, e, m;

  switch (bytes) {

  case (1):

    if (V < MIN_V1) {
      e = 0x0000;
      m = 0x0000;
    } else if (V > MAX_V1) {
      e = 0x000F;
      m = 0x000F;
    } else {
      e = (unsigned short int) (floor(log(V) * 1.44269504088896) / 4.0);
      m = (unsigned short int) (0.5 + (V / (pow(2.0, (double) (4 * e)))));
    }
    em = ((e & 0x000F) << 4) + (m & 0x000F);
    break;

  case (2):

    if (V < MIN_V2) {
      e = 0x0000;
      m = 0x0000;
    } else if (V > MAX_V2) {
      e = 0x001F;
      m = 0x07FF;
    } else {
      e = (unsigned short int) floor(log(V) * 1.44269504088896) + 15;
      m = (unsigned short int) (0.5 + 2048.0 * ((V / (pow(2.0, (double) e - 15.0))) - 1.0));
    }
    em = ((e & 0x001F) << 11) + (m & 0x07FF);
    break;

  default:

    em = 0x0000;
    break;
  };

  return em;
}

double jpwl_pfp_to_double(unsigned short int em, int bytes) {

  double V;

  switch (bytes) {

  case 1:
    V = (double) (em & 0x0F) * pow(2.0, (double) (em & 0xF0));
    break;

  case 2:

    V = pow(2.0, (double) ((em & 0xF800) >> 11) - 15.0) * (1.0 + (double) (em & 0x07FF) / 2048.0);
    break;

  default:
    V = 0.0;
    break;

  }

  return V;

}

bool jpwl_update_info(opj_j2k_t *j2k, jpwl_marker_t *jwmarker, int jwmarker_num) {

  int mm;
  unsigned long int addlen;

  opj_codestream_info_t *info = j2k->cstr_info;
  int tileno, tpno, packno, numtiles = info->th * info->tw, numpacks = info->packno;

  if (!j2k || !jwmarker ) {
    opj_event_msg(j2k->cinfo, EVT_ERROR, "J2K handle or JPWL markers list badly allocated\n");
    return false;
  }

  /* main_head_end: how many markers are there before? */
  addlen = 0;
  for (mm = 0; mm < jwmarker_num; mm++)
    if (jwmarker[mm].pos < (unsigned long int) info->main_head_end)
      addlen += jwmarker[mm].len + 2;
  info->main_head_end += addlen;

  /* codestream_size: always increment with all markers */
  addlen = 0;
  for (mm = 0; mm < jwmarker_num; mm++)
    addlen += jwmarker[mm].len + 2;
  info->codestream_size += addlen;

  /* navigate through all the tiles */
  for (tileno = 0; tileno < numtiles; tileno++) {

    /* start_pos: increment with markers before SOT */
    addlen = 0;
    for (mm = 0; mm < jwmarker_num; mm++)
      if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].start_pos)
        addlen += jwmarker[mm].len + 2;
    info->tile[tileno].start_pos += addlen;

    /* end_header: increment with markers before of it */
    addlen = 0;
    for (mm = 0; mm < jwmarker_num; mm++)
      if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].end_header)
        addlen += jwmarker[mm].len + 2;
    info->tile[tileno].end_header += addlen;

    /* end_pos: increment with markers before the end of this tile */
    /* code is disabled, since according to JPWL no markers can be beyond TPH */
    addlen = 0;
    for (mm = 0; mm < jwmarker_num; mm++)
      if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].end_pos)
        addlen += jwmarker[mm].len + 2;
    info->tile[tileno].end_pos += addlen;

    /* navigate through all the tile parts */
    for (tpno = 0; tpno < info->tile[tileno].num_tps; tpno++) {

      /* start_pos: increment with markers before SOT */
      addlen = 0;
      for (mm = 0; mm < jwmarker_num; mm++)
        if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].tp[tpno].tp_start_pos)
          addlen += jwmarker[mm].len + 2;
      info->tile[tileno].tp[tpno].tp_start_pos += addlen;

      /* end_header: increment with markers before of it */
      addlen = 0;
      for (mm = 0; mm < jwmarker_num; mm++)
        if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].tp[tpno].tp_end_header)
          addlen += jwmarker[mm].len + 2;
      info->tile[tileno].tp[tpno].tp_end_header += addlen;

      /* end_pos: increment with markers before the end of this tile part */
      addlen = 0;
      for (mm = 0; mm < jwmarker_num; mm++)
        if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].tp[tpno].tp_end_pos)
          addlen += jwmarker[mm].len + 2;
      info->tile[tileno].tp[tpno].tp_end_pos += addlen;

    }

    /* navigate through all the packets in this tile */
    for (packno = 0; packno < numpacks; packno++) {

      /* start_pos: increment with markers before the packet */
      /* disabled for the same reason as before */
      addlen = 0;
      for (mm = 0; mm < jwmarker_num; mm++)
        if (jwmarker[mm].pos <= (unsigned long int) info->tile[tileno].packet[packno].start_pos)
          addlen += jwmarker[mm].len + 2;
      info->tile[tileno].packet[packno].start_pos += addlen;

      /* end_ph_pos: increment with markers before the packet */
      /* disabled for the same reason as before */
      /*addlen = 0;
      for (mm = 0; mm < jwmarker_num; mm++)
        if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].packet[packno].end_ph_pos)
          addlen += jwmarker[mm].len + 2;*/
      info->tile[tileno].packet[packno].end_ph_pos += addlen;

      /* end_pos: increment if marker is before the end of packet */
      /* disabled for the same reason as before */
      /*addlen = 0;
      for (mm = 0; mm < jwmarker_num; mm++)
        if (jwmarker[mm].pos < (unsigned long int) info->tile[tileno].packet[packno].end_pos)
          addlen += jwmarker[mm].len + 2;*/
      info->tile[tileno].packet[packno].end_pos += addlen;

    }
  }

  /* reorder the markers list */

  return true;
}

#endif /* USE_JPWL */
