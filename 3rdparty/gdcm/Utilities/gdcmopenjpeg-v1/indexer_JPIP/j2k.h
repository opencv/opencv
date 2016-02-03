/*
 * Copyright (c) 2001-2002, David Janssens
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

#define VERSION "0.0.8"

#ifdef _WIN32
#ifdef LIBJ2K_EXPORTS
#define LIBJ2K_API __declspec(dllexport)
#else
#define LIBJ2K_API __declspec(dllimport)
#endif
#else
#define LIBJ2K_API
#endif

#ifndef __J2K_H
#define __J2K_H

#define J2K_MAXRLVLS 33
#define J2K_MAXBANDS (3*J2K_MAXRLVLS+1)

#define J2K_CP_CSTY_PRT 0x01
#define J2K_CP_CSTY_SOP 0x02
#define J2K_CP_CSTY_EPH 0x04
#define J2K_CCP_CSTY_PRT 0x01
#define J2K_CCP_CBLKSTY_LAZY 0x01
#define J2K_CCP_CBLKSTY_RESET 0x02
#define J2K_CCP_CBLKSTY_TERMALL 0x04
#define J2K_CCP_CBLKSTY_VSC 0x08
#define J2K_CCP_CBLKSTY_PTERM 0x10
#define J2K_CCP_CBLKSTY_SEGSYM 0x20
#define J2K_CCP_QNTSTY_NOQNT 0
#define J2K_CCP_QNTSTY_SIQNT 1
#define J2K_CCP_QNTSTY_SEQNT 2

typedef struct 
{
  int dx, dy;   /* XRsiz, YRsiz            */
  int prec;     /* precision               */
  int bpp;      /* deapth of image in bits */
  int sgnd;     /* signed                  */
  int *data;    /* image-component data    */
} j2k_comp_t;

typedef struct {
  int version;
  int x0, y0;          /* XOsiz, YOsiz              */
  int x1, y1;          /* Xsiz, Ysiz                */ 
  int numcomps;        /* number of components      */
  int index_on;        /* 0 = no index || 1 = index */
  j2k_comp_t *comps;   /* image-components          */
} j2k_image_t;

typedef struct {
  int expn;     /* exponent */
  int mant;     /* mantissa */
} j2k_stepsize_t;

typedef struct {
  int csty;                                /* coding style                          */
  int numresolutions;                      /* number of resolutions                 */
  int cblkw;                               /* width of code-blocks                  */
  int cblkh;                               /* height of code-blocks                 */
  int cblksty;                             /* code-block coding style               */
  int qmfbid;                              /* discrete wavelet transform identifier */
  int qntsty;                              /* quantisation style                    */
  j2k_stepsize_t stepsizes[J2K_MAXBANDS];  /* stepsizes used for quantisation       */
  int numgbits;                            /* number of guard bits                  */
  int roishift;                            /* Region of Interest shift              */
  int prcw[J2K_MAXRLVLS];                  /* Precinct width                        */
  int prch[J2K_MAXRLVLS];                  /* Precinct height                       */
} j2k_tccp_t;

typedef struct {
    int resno0, compno0; 
    int layno1, resno1, compno1;
    int prg;
    int tile;
    char progorder[4];
} j2k_poc_t;

typedef struct {
  int csty;                  /* coding style                                                            */  
  int prg;                   /* progression order                                                       */
  int numlayers;             /* number of layers                                                        */
  int mct;                   /* multi-component transform identifier                                    */
  int rates[100];            /* rates of layers                                                         */
  int numpocs;               /* number of progression order changes                                     */
  int POC;                   /* Precise if a POC marker has been used O:NO, 1:YES                       */
  j2k_poc_t pocs[32];        /* progression order changes                                               */
  unsigned char *ppt_data;   /* packet header store there for futur use in t2_decode_packet             */
  int ppt;                   /* If ppt == 1 --> there was a PPT marker for the present tile             */
  int ppt_store;             /* Use in case of multiple marker PPT (number of info already store)       */
  j2k_tccp_t *tccps;         /* tile-component coding parameters                                        */
} j2k_tcp_t;

typedef struct {
  int tx0, ty0;              /* XTOsiz, YTOsiz                                                          */
  int tdx, tdy;              /* XTsiz, YTsiz                                                            */
  int tw, th;
  unsigned char *ppm_data;   /* packet header store there for futur use in t2_decode_packet             */
  int ppm;                   /* If ppm == 1 --> there was a PPM marker for the present tile             */
  int ppm_store;             /* Use in case of multiple marker PPM (number of info already store)       */
  int ppm_previous;          /* Use in case of multiple marker PPM (case on non-finished previous info) */
  j2k_tcp_t *tcps;           /* tile coding parameters                                                  */
} j2k_cp_t;





/* Packet information : Layer level */
typedef struct {  
  int len;               /* Length of the body of the packet   */
  int len_header;        /* Length of the header of the packet */
  int offset;            /* Offset of the body of the packet   */
  int offset_header;     /* Offset of the header of the packet */
} info_layer_t;


/* Access to packet information : precinct level */
typedef struct {  
  info_layer_t *layer;
} info_prec_t;


/* Access to packet information : resolution level */
typedef struct {  
  info_prec_t *prec;
} info_reso_t;


/* Access to packet information : component level */
typedef struct {  
  info_reso_t *reso;
} info_compo_t;


/* Information about the marker */
typedef struct {
  int type;       /* type of marker [SIZ, QCD, POC, PPM, CRG, COD] appearing only once */
  int start_pos;  /* Start position of the marker                                      */
  int len;        /* Length of the marker                                              */
} info_marker_t;


/* Multiple marker in tile header */
typedef struct{
  info_marker_t *COC;    /* COC markers                    */
  int num_COC;           /* Number of COC marker           */
  int CzCOC;             /* Current size of the vector COC */
  
  info_marker_t *RGN;    /* RGN markers                    */
  int num_RGN;           /* Number of RGN marker           */
  int CzRGN;             /* Current size of the vector RGN */
  
  info_marker_t *QCC;    /* QCC markers                    */
  int num_QCC;           /* Number of QCC marker           */
  int CzQCC;             /* Current size of the vector QCC */
  
  info_marker_t *PLT;    /* PLT markers                    */
  int num_PLT;           /* Number of PLT marker           */
  int CzPLT;             /* Current size of the vector PLT */
  
  info_marker_t *PPT;    /* PPT markers                    */
  int num_PPT;           /* Number of PPT marker           */
  int CzPPT;             /* Current size of the vector PPT */
  
  info_marker_t *COM;    /* COM markers                    */
  int num_COM;           /* Number of COM marker           */
  int CzCOM;             /* Current size of the vector COC */
} info_marker_mul_tile_t; 


/* Information about each tile_part for a particulary tile */
typedef struct{
  int start_pos;                      /* Start position of the tile_part       */ 
  int length;                         /* Length of the tile_part header + body */
  int length_header;                  /* Length of the header                  */
  int end_pos;                        /* End position of the tile part         */
  int end_header;                     /* End position of the tile part header  */

  int num_reso_AUX;                   /* Number of resolution level completed  */
} info_tile_part_t;


/* Information about each tile */
typedef struct {
  int num_tile;                       /* Number of Tile                                                    */
  int pw, ph;                         /* number of precinct by tile                                        */
  int num_packet;                     /* number of packet in the tile                                      */
  info_compo_t *compo;                /* component [packet]                                                */
  
  info_marker_t *marker;              /* information concerning markers inside image [only one apparition] */
  info_marker_mul_tile_t marker_mul;  /* information concerning markers inside image [multiple apparition] */ 
  int num_marker;                     /* number of marker                                                  */
  
  int numparts;                       /* number of tile_part for this tile                                 */
  info_tile_part_t *tile_parts;       /* Information about each tile_part                                  */
  int Cztile_parts;                   /* Current size of the tile_parts vector                             */
} info_tile_t;                        /* index struct                                                      */


/* Multiple marker in main header */
typedef struct{
  info_marker_t *COC;    /* COC markers                    */
  int num_COC;           /* Number of COC marker           */
  int CzCOC;             /* Current size of the vector COC */
 
  info_marker_t *RGN;    /* RGN markers                    */
  int num_RGN;           /* Number of RGN marker           */
  int CzRGN;             /* Current size of the vector RGN */
  
  info_marker_t *QCC;    /* QCC markers                    */
  int num_QCC;           /* Number of QCC marker           */
  int CzQCC;             /* Current size of the vector QCC */
  
  info_marker_t *TLM;    /* TLM markers                    */
  int num_TLM;           /* Number of TLM marker           */
  int CzTLM;             /* Current size of the vector TLM */
  
  info_marker_t *PLM;    /* PLM markers                    */
  int num_PLM;           /* Number of PLM marker           */
  int CzPLM;             /* Current size of the vector PLM */
  
  info_marker_t *PPM;    /* PPM markers                    */
  int num_PPM;           /* Number of PPM marker           */
  int CzPPM;             /* Current size of the vector PPM */
  
  info_marker_t *COM;    /* COM markers                    */
  int num_COM;           /* Number of COM marker           */
  int CzCOM;             /* Current size of the vector COM */
} info_marker_mul_t; /* index struct */


/* Information about image */
typedef struct {
  int Im_w, Im_h;                /* Image width and Height                                            */
  int Tile_x, Tile_y;            /* Number of Tile in X and Y                                         */
  int tw, th;
  int pw, ph;                    /* nombre precinct in X and Y                                        */
  int pdx, pdy;                  /* size of precinct in X and Y                                       */

  int Prog;                      /* progression order                                                 */
  int Comp;                      /* Component numbers                                                 */
  int Layer;                     /* number of layer                                                   */
  int Decomposition;             /* number of decomposition                                           */

  int Main_head_end;             /* Main header position                                              */
  int codestream_size;           /* codestream's size                                                 */

  info_marker_t *marker;         /* information concerning markers inside image [only one apparition] */
  info_marker_mul_t marker_mul;  /* information concerning markers inside image [multiple apparition] */ 
  int num_marker;                /* number of marker                                                  */

  int num_packet_max;            /* Maximum number of packet                                          */

  int num_max_tile_parts;        /* Maximum number of tile-part                                       */
  info_tile_t *tile;             /* information concerning tiles inside image                         */
} info_image_t; /* index struct */


#endif
