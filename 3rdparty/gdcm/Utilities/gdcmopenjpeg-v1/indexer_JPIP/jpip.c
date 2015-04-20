/*
 * Copyright (c) 2003-2004, Yannick Verschueren
 * Copyright (c) 2003-2004, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <math.h>

#include "j2k.h"
#include "cio.h"
#include "tcd.h"
#include "int.h"

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

#define JP2C      0x6a703263

//static info_marker_t marker_jpip[32], marker_local_jpip[32];  /* SIZE to precise ! */
//static int num_marker_jpip, num_marker_local_jpip;

/* 
 * Write the CPTR box
 *
 * Codestream finder box (box)
 *
 */
void jpip_write_cptr(int offset, info_image_t img)
{
  int len, lenp;

  lenp=cio_tell(); 
  cio_skip(4);                       /* L [at the end]     */
  cio_write(JPIP_CPTR,4);            /* T                  */
  cio_write(0,2);                    /* DR  A PRECISER !!  */
  cio_write(0,2);                    /* CONT               */
  cio_write(offset,8);               /* COFF A PRECISER !! */
  cio_write(img.codestream_size,8);  /* CLEN               */
  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);                 /* L                  */
  cio_seek(lenp+len);
}

/* 
 * Read the CPTR box
 *
 * Codestream finder box (box)
 *
 */
void jpip_read_cptr()
{
  int DR, CONT;
  long long Coff, codestream_size;

  DR = cio_read(2);               /* DR   */
  CONT = cio_read(2);             /* CONT */
  Coff = cio_read(8);             /* COFF */
  codestream_size = cio_read(8);  /* CLEN */
}

/* 
 * Write the MANF box
 *
 * Manifest box (box)
 *
 */
void jpip_write_manf(int second, int v, info_marker_t *marker)
{
  int len, lenp, i;
  lenp=cio_tell(); 
  cio_skip(4);                         /* L [at the end]                    */
  cio_write(JPIP_MANF,4);              /* T                                 */

  if (second)                          /* Write only during the second pass */
    {
      for(i=0;i<v;i++)
	{
	  cio_write(marker[i].len,4);  /* Marker length                     */ 
	  cio_write(marker[i].type,4); /* Marker type                       */
	}
    }

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);                   /* L                                 */
  cio_seek(lenp+len);
}

/* 
 * Read the MANF box
 *
 * Manifest box (box)
 *
 */
void jpip_read_manf(int len)
{
  int i, v, marker_len, marker_type;
  
  v = (len - 8)/ 8;
  
  for(i=0;i<v;i++)
    {
      marker_len = cio_read(4);       /* Marker length */ 
      marker_type = cio_read(4);      /* Marker type   */
    }
}

/* 
 * Write the MHIX box
 *
 * Main Header Index Table (box)
 *
 */
int jpip_write_mhix(info_image_t img, int status, int tileno)
{
  int len, lenp, i;
  info_tile_t *tile;
  lenp=cio_tell();
  cio_skip(4);                               /* L [at the end]                    */
  cio_write(JPIP_MHIX, 4);                   /* MHIX                              */

  if (status) /* MAIN HEADER */
    {
      cio_write(img.Main_head_end,8);        /* TLEN                              */
      
      for(i = 0; i < img.num_marker; i++)    /* Marker restricted to 1 apparition */
	{
	  cio_write(img.marker[i].type, 2);
	  cio_write(0, 2);
	  cio_write(img.marker[i].start_pos, 8);
	  cio_write(img.marker[i].len, 2);
	}
      
      /* Marker NOT restricted to 1 apparition */
      for(i = img.marker_mul.num_COC - 1; i >= 0; i--) /* COC */
	{
	  cio_write(img.marker_mul.COC[i].type, 2);
	  cio_write(i, 2);
	  cio_write(img.marker_mul.COC[i].start_pos, 8);
	  cio_write(img.marker_mul.COC[i].len, 2);
	}
      
      for(i = img.marker_mul.num_RGN - 1; i >= 0; i--) /* RGN */
	{
	  cio_write(img.marker_mul.RGN[i].type, 2);
	  cio_write(i, 2);
	  cio_write(img.marker_mul.RGN[i].start_pos, 8);
	  cio_write(img.marker_mul.RGN[i].len, 2);
	}
      
      for(i = img.marker_mul.num_QCC - 1; i >= 0; i--) /* QCC */
	{
	  cio_write(img.marker_mul.QCC[i].type, 2);
	  cio_write(i, 2);
	  cio_write(img.marker_mul.QCC[i].start_pos, 8);
	  cio_write(img.marker_mul.QCC[i].len, 2);
	}
      
      for(i = img.marker_mul.num_TLM - 1; i >= 0; i--) /* TLM */
	{
	  cio_write(img.marker_mul.TLM[i].type, 2);
	  cio_write(i, 2);
	  cio_write(img.marker_mul.TLM[i].start_pos, 8);
	  cio_write(img.marker_mul.TLM[i].len, 2);
	}
      
      for(i = img.marker_mul.num_PLM - 1; i >= 0; i--) /* PLM */
	{
	  cio_write(img.marker_mul.PLM[i].type, 2);
	  cio_write(i, 2);
	  cio_write(img.marker_mul.PLM[i].start_pos, 8);
	  cio_write(img.marker_mul.PLM[i].len, 2);
	}
      
      for(i = img.marker_mul.num_PPM - 1; i >= 0; i--) /* PPM */
	{
	  cio_write(img.marker_mul.PPM[i].type, 2);
	  cio_write(i, 2);
	  cio_write(img.marker_mul.PPM[i].start_pos, 8);
	  cio_write(img.marker_mul.PPM[i].len, 2);
	}

      for(i = img.marker_mul.num_COM - 1; i >= 0; i--) /* COM */
	{
	  cio_write(img.marker_mul.COM[i].type, 2);
	  cio_write(i, 2);
	  cio_write(img.marker_mul.COM[i].start_pos, 8);
	  cio_write(img.marker_mul.COM[i].len, 2);
	}
    } 
  else /* TILE HEADER */
    {
      tile = &img.tile[tileno];
      cio_write(tile->tile_parts[0].length_header, 8);  /* TLEN                              */ 
      
      for(i = 0; i < tile->num_marker; i++)             /* Marker restricted to 1 apparition */
	{
	  cio_write(tile->marker[i].type, 2);
	  cio_write(0, 2);
	  cio_write(tile->marker[i].start_pos, 8);
	  cio_write(tile->marker[i].len, 2);
	}
      
      /* Marker NOT restricted to 1 apparition */
      for(i = tile->marker_mul.num_COC - 1; i >= 0; i--) /* COC */
	{
	  cio_write(tile->marker_mul.COC[i].type, 2);
	  cio_write(i, 2);
	  cio_write(tile->marker_mul.COC[i].start_pos, 8);
	  cio_write(tile->marker_mul.COC[i].len, 2);
	}
      
      for(i = tile->marker_mul.num_RGN - 1; i >= 0; i--) /* RGN */
	{
	  cio_write(tile->marker_mul.RGN[i].type, 2);
	  cio_write(i, 2);
	  cio_write(tile->marker_mul.RGN[i].start_pos, 8);
	  cio_write(tile->marker_mul.RGN[i].len, 2);
	}
      
      for(i = tile->marker_mul.num_QCC - 1; i >= 0; i--) /* QCC */
	{
	  cio_write(tile->marker_mul.QCC[i].type, 2);
	  cio_write(i, 2);
	  cio_write(tile->marker_mul.QCC[i].start_pos, 8);
	  cio_write(tile->marker_mul.QCC[i].len, 2);
	}
      
      for(i = tile->marker_mul.num_PLT - 1; i >= 0; i--) /* PLT */
	{
	  cio_write(tile->marker_mul.PLT[i].type,2);
	  cio_write(i,2);
	  cio_write(tile->marker_mul.PLT[i].start_pos,8);
	  cio_write(tile->marker_mul.PLT[i].len,2);
	}
      
      for(i = tile->marker_mul.num_PPT - 1; i >= 0; i--) /* PPT */
	{
	  cio_write(tile->marker_mul.PPT[i].type, 2);
	  cio_write(i, 2);
	  cio_write(tile->marker_mul.PPT[i].start_pos, 8);
	  cio_write(tile->marker_mul.PPT[i].len, 2);
	}
      
      for(i = tile->marker_mul.num_COM - 1; i >= 0; i--) /* COM */
	{
	  cio_write(tile->marker_mul.COM[i].type, 2);
	  cio_write(i, 2);
	  cio_write(tile->marker_mul.COM[i].start_pos, 8);
	  cio_write(tile->marker_mul.COM[i].len, 2);
	} 
    }      
  
  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);        /* L           */
  cio_seek(lenp+len);
  
  return len;
}

/* 
 * Read the MHIX box
 *
 * Main Header Index Table (box)
 *
 */
void jpip_read_mhix(int len)
{
  int i, v, marker_type, marker_start_pos, marker_len, marker_remains;

  v = (len - 8) / 14;

  for (i=0; i<v ; i++)
    {
      marker_type = cio_read(2);       /* Type of the marker               */ 
      marker_remains = cio_read(2);    /* Number of same markers following */
      marker_start_pos = cio_read(2);  /* Start position of the marker     */
      marker_len = cio_read(2);        /* Length of the marker             */
    }
}

/* 
 * Write the FAIX box
 *
 * Fragment array Index box (box)
 *
 */
int jpip_write_faix(int v, int compno, info_image_t img, j2k_cp_t *j2k_cp, int version)
{
  int len, lenp, i, j;
  /*int version = 0;*/
  int tileno, resno, precno, layno, num_packet=0;

  lenp=cio_tell();
  cio_skip(4);              /* L [at the end]      */
  cio_write(JPIP_FAIX, 4);  /* FAIX                */ 
  cio_write(version,1);     /* Version 0 = 4 bytes */
  
  switch(v)
    {
    case 0:   /* TPIX */
      cio_write(img.num_max_tile_parts,(version & 0x01)?8:4);                      /* NMAX           */
      cio_write(img.tw*img.th,(version & 0x01)?8:4);                               /* M              */
      for (i = 0; i < img.tw*img.th; i++)
	{
	  for (j = 0; j < img.tile[i].numparts ; j++)
	    {
	      cio_write(img.tile[i].tile_parts[j].start_pos,(version & 0x01)?8:4); /* start position */
	      cio_write(img.tile[i].tile_parts[j].length,(version & 0x01)?8:4);    /* length         */
	      if (version & 0x02)
		cio_write(img.tile[i].tile_parts[j].num_reso_AUX,4); /* Aux_i,j : Auxiliary value */
	      //cio_write(0,4);
	    }
	  /* PADDING */
	  while (j < img.num_max_tile_parts)
	    {
	      cio_write(0,(version & 0x01)?8:4); /* start position            */
	      cio_write(0,(version & 0x01)?8:4); /* length                    */
	      if (version & 0x02)
		cio_write(0,4);                  /* Aux_i,j : Auxiliary value */
	      j++;
	    }
	}
      break;
      
      /*   case 1: */   /* THIX */
      /* cio_write(1,(version & 0x01)?8:4);  */           /* NMAX */
      /* cio_write(img.tw*img.th,(version & 0x01)?8:4); */ /* M    */
      /* for (i=0;i<img.tw*img.th;i++) */
      /* { */
      /* cio_write(img.tile[i].start_pos,(version & 0x01)?8:4); */                         /* start position */
      /* cio_write(img.tile[i].end_header-img.tile[i].start_pos,(version & 0x01)?8:4); */  /* length         */
      /* if (version & 0x02)*/
      /* cio_write(0,4); */ /* Aux_i,j : Auxiliary value */
      /* } */
      /* break; */

    case 2:  /* PPIX  NOT FINISHED !! */
      cio_write(img.num_packet_max,(version & 0x01)?8:4); /* NMAX */
      cio_write(img.tw*img.th,(version & 0x01)?8:4);      /* M    */
      for(tileno=0;tileno<img.tw*img.th;tileno++)
	{
	  info_tile_t *tile_Idx = &img.tile[tileno];
	  info_compo_t *compo_Idx = &tile_Idx->compo[compno];
	  int correction;
	  
	  num_packet=0;
	  
	  if(j2k_cp->tcps[tileno].csty&J2K_CP_CSTY_EPH)
	    correction=3;
	  else
	    correction=1;
	  for(resno=0;resno<img.Decomposition+1;resno++)
	    {
	      info_reso_t *reso_Idx = &compo_Idx->reso[resno];
	      for (precno=0;precno<img.tile[tileno].pw*img.tile[tileno].ph;precno++)
		{
		  info_prec_t *prec_Idx = &reso_Idx->prec[precno];
		  for(layno=0;layno<img.Layer;layno++)
		    {
		      info_layer_t *layer_Idx = &prec_Idx->layer[layno];
		      cio_write(layer_Idx->offset,(version & 0x01)?8:4);                                   /* start position */
		      cio_write((layer_Idx->len_header-correction)?0:layer_Idx->len,(version & 0x01)?8:4); /* length         */
		      if (version & 0x02)
			cio_write(0,4); /* Aux_i,j : Auxiliary value */
		      num_packet++;
		    }
		}
	    }
	  /* PADDING */
	  while (num_packet < img.num_packet_max)
	    {
	      cio_write(0,(version & 0x01)?8:4); /* start position            */
	      cio_write(0,(version & 0x01)?8:4); /* length                    */
	      if (version & 0x02)
		cio_write(0,4);                  /* Aux_i,j : Auxiliary value */
	      num_packet++;
	    }
	}
      
      break;
      
    case 3:  /* PHIX NOT FINISHED !! */
      cio_write(img.num_packet_max,(version & 0x01)?8:4); /* NMAX */
      cio_write(img.tw*img.th,(version & 0x01)?8:4);      /* M    */
      for(tileno=0;tileno<img.tw*img.th;tileno++)
	{
	  info_tile_t *tile_Idx = &img.tile[tileno];
	  info_compo_t *compo_Idx = &tile_Idx->compo[compno];
	  int correction;

	  num_packet = 0;
	  if(j2k_cp->tcps[tileno].csty&J2K_CP_CSTY_EPH)
	    correction=3;
	  else
	    correction=1;
	  for(resno=0;resno<img.Decomposition+1;resno++)
	    {
	      info_reso_t *reso_Idx = &compo_Idx->reso[resno];
	      for (precno=0;precno<img.tile[tileno].pw*img.tile[tileno].ph;precno++)
		{
		  info_prec_t *prec_Idx = &reso_Idx->prec[precno];
		  for(layno=0;layno<img.Layer;layno++)
		    {
		      info_layer_t *layer_Idx = &prec_Idx->layer[layno];
		      cio_write(layer_Idx->offset_header,(version & 0x01)?8:4);                                   /* start position */
		      cio_write((layer_Idx->len_header-correction)?0:layer_Idx->len_header,(version & 0x01)?8:4); /* length         */
		      if (version & 0x02)
			cio_write(0,4); /* Aux_i,j : Auxiliary value */
		      num_packet++;
		    }
		}
	    }
	  /* PADDING */
	  while (num_packet<img.num_packet_max)
	    {
	      cio_write(0,(version & 0x01)?8:4); /* start position            */
	      cio_write(0,(version & 0x01)?8:4); /* length                    */
	      if (version & 0x02)
		cio_write(0,4);                  /* Aux_i,j : Auxiliary value */
	      num_packet++;
	    }
	}
      break;
    }
  
  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);        /* L  */
  cio_seek(lenp+len);

  return len;
}

/* 
 * Write the TPIX box
 *
 * Tile-part Index table box (superbox)
 *
 */
int jpip_write_tpix(info_image_t img, j2k_cp_t *j2k_cp, int version)
{
  int len, lenp;
  lenp=cio_tell();
  cio_skip(4);              /* L [at the end] */
  cio_write(JPIP_TPIX, 4);  /* TPIX           */
  
  jpip_write_faix(0,0,img, j2k_cp, version);

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);        /* L              */
  cio_seek(lenp+len);

  return len;
}

/* 
 * Write the THIX box
 *
 * Tile header Index table box (superbox)
 *
 */
//int jpip_write_thix(info_image_t img, j2k_cp_t *j2k_cp)
//  {
//  int len, lenp;
//  lenp=cio_tell();
//  cio_skip(4);              /* L [at the end] */
//  cio_write(JPIP_THIX, 4);  /* THIX           */
  
//  jpip_write_faix(1,0,img, j2k_cp);

//  len=cio_tell()-lenp;
//  cio_seek(lenp);
//  cio_write(len, 4);        /* L              */
//  cio_seek(lenp+len);

//  return len;
//}

int jpip_write_thix(info_image_t img, j2k_cp_t *j2k_cp)
{
  int len, lenp, i;
  int tileno;
  info_marker_t *marker;
  int num_marker_local_jpip;

  marker = (info_marker_t*)calloc(sizeof(info_marker_t), j2k_cp->tw*j2k_cp->th);

  for ( i = 0; i < 2 ; i++ )
    {
      if (i) cio_seek(lenp);
      
      lenp = cio_tell();
      cio_skip(4);              /* L [at the end] */
      cio_write(JPIP_THIX, 4);  /* THIX           */
      jpip_write_manf(i, j2k_cp->tw*j2k_cp->th, marker);
      num_marker_local_jpip=img.Comp;
      
      for (tileno = 0; tileno < j2k_cp->tw*j2k_cp->th; tileno++)
	{
	  marker[tileno].len = jpip_write_mhix(img, 1, tileno);
	  marker[tileno].type = JPIP_MHIX;
	}
      
      len=cio_tell()-lenp;
      cio_seek(lenp);
      cio_write(len, 4);        /* L              */
      cio_seek(lenp+len);
    }

  free(marker);

  return len;
}
/* 
 * Write the PPIX box
 *
 * Precinct Packet Index table box (superbox)
 *
 */
int jpip_write_ppix(info_image_t img,j2k_cp_t *j2k_cp)
{
  int len, lenp, compno, i;
  info_marker_t *marker;
  int num_marker_local_jpip;
  marker = (info_marker_t*)calloc(sizeof(info_marker_t), img.Comp);
  
  for (i=0;i<2;i++)
    {
      if (i) cio_seek(lenp);
      
      lenp=cio_tell();
      cio_skip(4);              /* L [at the end] */
      cio_write(JPIP_PPIX, 4);  /* PPIX           */
      jpip_write_manf(i,img.Comp,marker);
      num_marker_local_jpip=img.Comp;
      
      for (compno=0; compno<img.Comp; compno++)
	{
	  marker[compno].len=jpip_write_faix(2,compno,img, j2k_cp, 0);
	  marker[compno].type=JPIP_FAIX;
	}
   
      len=cio_tell()-lenp;
      cio_seek(lenp);
      cio_write(len, 4);        /* L              */
      cio_seek(lenp+len);
    }
  
  free(marker);

  return len;
}

/* 
 * Write the PHIX box
 *
 * Packet Header Index table box (superbox)
 *
 */
int jpip_write_phix(info_image_t img, j2k_cp_t *j2k_cp)
{
  int len, lenp=0, compno, i;
  info_marker_t *marker;

  marker = (info_marker_t*)calloc(sizeof(info_marker_t), img.Comp);

  for (i=0;i<2;i++)
    {
      if (i) cio_seek(lenp);
      
      lenp=cio_tell();
      cio_skip(4);              /* L [at the end] */
      cio_write(JPIP_PHIX, 4);  /* PHIX           */
      
      jpip_write_manf(i,img.Comp,marker);

      for (compno=0; compno<img.Comp; compno++)
	{	
	  marker[compno].len=jpip_write_faix(3,compno,img, j2k_cp, 0);
	  marker[compno].type=JPIP_FAIX;
	}

      len=cio_tell()-lenp;
      cio_seek(lenp);
      cio_write(len, 4);        /* L              */
      cio_seek(lenp+len);
    }

  free(marker);

  return len;
}

/* 
 * Write the CIDX box
 *
 * Codestream Index box (superbox)
 *
 */
int jpip_write_cidx(int offset, info_image_t img, j2k_cp_t *j2k_cp, int version)
{
  int len, lenp = 0, i;
  info_marker_t *marker_jpip;
  int num_marker_jpip = 0;

  marker_jpip = (info_marker_t*)calloc(sizeof(info_marker_t), 32);

  for (i=0;i<2;i++)
    {
      if(i)
	cio_seek(lenp);

      lenp=cio_tell();

      cio_skip(4);              /* L [at the end] */
      cio_write(JPIP_CIDX, 4);  /* CIDX           */
      jpip_write_cptr(offset, img);

      jpip_write_manf(i,num_marker_jpip, marker_jpip);

      num_marker_jpip=0;
      marker_jpip[num_marker_jpip].len=jpip_write_mhix(img, 0, 0);
      marker_jpip[num_marker_jpip].type=JPIP_MHIX;
      num_marker_jpip++;

      marker_jpip[num_marker_jpip].len=jpip_write_tpix(img, j2k_cp, version);
      marker_jpip[num_marker_jpip].type=JPIP_TPIX;
      num_marker_jpip++;

      marker_jpip[num_marker_jpip].len=jpip_write_thix(img, j2k_cp);
      marker_jpip[num_marker_jpip].type=JPIP_THIX;
      num_marker_jpip++;

      marker_jpip[num_marker_jpip].len=jpip_write_ppix(img, j2k_cp);
      marker_jpip[num_marker_jpip].type=JPIP_PPIX;
      num_marker_jpip++;

      marker_jpip[num_marker_jpip].len=jpip_write_phix(img, j2k_cp);
      marker_jpip[num_marker_jpip].type=JPIP_PHIX;
      num_marker_jpip++;

      len=cio_tell()-lenp;
      cio_seek(lenp);
      cio_write(len, 4);        /* L             */
      cio_seek(lenp+len);
    }

  free(marker_jpip);

  return len;

}

/* 
 * Write the IPTR box
 *
 * Index Finder box
 *
 */
void jpip_write_iptr(int offset, int length)
{
  int len, lenp;
  lenp=cio_tell();
  cio_skip(4);              /* L [at the end] */
  cio_write(JPIP_IPTR, 4);  /* IPTR           */
  
  cio_write(offset,8);
  cio_write(length,8);

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);        /* L             */
  cio_seek(lenp+len);
}

/* 
 * Write the PRXY box
 *
 * proxy (box)
 *
 */
void jpip_write_prxy(int offset_jp2c, int length_jp2c, int offset_idx, int length_idx)
{
  int len, lenp;
  lenp=cio_tell();
  cio_skip(4);              /* L [at the end] */
  cio_write(JPIP_PRXY, 4);  /* IPTR           */
  
  cio_write(offset_jp2c,8); /* OOFF           */
  cio_write(length_jp2c,4); /* OBH part 1     */
  cio_write(JP2C,4);        /* OBH part 2     */
  
  cio_write(1,1);           /* NI             */

  cio_write(offset_idx,8);  /* IOFF           */
  cio_write(length_idx,4);  /* IBH part 1     */
  cio_write(JPIP_CIDX,4);   /* IBH part 2     */

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);        /* L              */
  cio_seek(lenp+len);
}


/* 
 * Write the FIDX box
 *
 * File Index (superbox)
 *
 */
int jpip_write_fidx(int offset_jp2c, int length_jp2c, int offset_idx, int length_idx)
{
  int len, lenp;
  lenp=cio_tell();
  cio_skip(4);              /* L [at the end] */
  cio_write(JPIP_FIDX, 4);  /* IPTR           */
  
  jpip_write_prxy(offset_jp2c, length_jp2c, offset_idx, offset_jp2c);

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len, 4);        /* L              */
  cio_seek(lenp+len);

  return len;
}
