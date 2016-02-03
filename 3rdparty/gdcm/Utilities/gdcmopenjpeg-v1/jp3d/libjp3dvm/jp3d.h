/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2006, Mónica Díez García, Image Processing Laboratory, University of Valladolid, Spain
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
#ifndef __J3D_H
#define __J3D_H
/**
@file j3d.h
@brief The JPEG-2000 Codestream Reader/Writer (J3D)

The functions in J3D.C have for goal to read/write the several parts of the codestream: markers and data.
*/

/** @defgroup J3D J3D - JPEG-2000 codestream reader/writer */
/*@{*/

#define J3D_CP_CSTY_PRT 0x01
#define J3D_CP_CSTY_SOP 0x02
#define J3D_CP_CSTY_EPH 0x04
#define J3D_CCP_CSTY_PRT 0x01
/** Table A-8 */
#define J3D_CCP_CBLKSTY_LAZY 0x01	 /* Selective arithmetic coding bypass */
#define J3D_CCP_CBLKSTY_RESET 0x02   /* Reset context probabilities on coding pass boundaries */
#define J3D_CCP_CBLKSTY_TERMALL 0x04 /* Termination on each coding pass */
#define J3D_CCP_CBLKSTY_VSC 0x08	 /* Vertically causal context, add also hook for switching off and on 3D context models */	
#define J3D_CCP_CBLKSTY_PTERM 0x10	 /* Predictable termination */
#define J3D_CCP_CBLKSTY_SEGSYM 0x20	 /* Segmentation symbols are used */
#define J3D_CCP_CBLKSTY_3DCTXT 0x40  /* 3D context models (3D-EBCOT) vs 2D context models */

#define J3D_CCP_QNTSTY_NOQNT 0	/* Quantization style : no quantization */
#define J3D_CCP_QNTSTY_SIQNT 1	/* Quantization style : scalar derived (values signalled only in LLL subband) */
#define J3D_CCP_QNTSTY_SEQNT 2	/* Quantization style : scalar expounded (values signalled for each subband) */

/* ----------------------------------------------------------------------- */

#define J3D_MS_SOC 0xff4f	/**< SOC marker value */
#define J3D_MS_SOT 0xff90	/**< SOT marker value */
#define J3D_MS_SOD 0xff93	/**< SOD marker value */
#define J3D_MS_EOC 0xffd9	/**< EOC marker value */
#define J3D_MS_CAP 0xff50	/**< CAP marker value */
#define J3D_MS_SIZ 0xff51	/**< SIZ marker value */
#define J3D_MS_ZSI 0xff54	/**< ZSI marker value */
#define J3D_MS_COD 0xff52	/**< COD marker value */
#define J3D_MS_COC 0xff53	/**< COC marker value */
#define J3D_MS_RGN 0xff5e	/**< RGN marker value */
#define J3D_MS_QCD 0xff5c	/**< QCD marker value */
#define J3D_MS_QCC 0xff5d	/**< QCC marker value */
#define J3D_MS_POC 0xff5f	/**< POC marker value */
#define J3D_MS_TLM 0xff55	/**< TLM marker value */
#define J3D_MS_PLM 0xff57	/**< PLM marker value */
#define J3D_MS_PLT 0xff58	/**< PLT marker value */
#define J3D_MS_PPM 0xff60	/**< PPM marker value */
#define J3D_MS_PPT 0xff61	/**< PPT marker value */
#define J3D_MS_SOP 0xff91	/**< SOP marker value */
#define J3D_MS_EPH 0xff92	/**< EPH marker value */
#define J3D_MS_CRG 0xff63	/**< CRG marker value */
#define J3D_MS_COM 0xff64	/**< COM marker value */
//15444-2
#define J3D_MS_DCO 0xff70	/**< DCO marker value */
#define J3D_MS_VMS 0xff71   /**< VMS marker value */
#define J3D_MS_DFS 0xff72	/**< DFS marker value */
#define J3D_MS_ADS 0xff73	/**< ADS marker value */
#define J3D_MS_ATK 0xff79	/**< ATK marker value */
#define J3D_MS_CBD 0xff78	/**< CBD marker value */
#define J3D_MS_MCT 0xff74	/**< MCT marker value */
#define J3D_MS_MCC 0xff75	/**< MCC marker value */
#define J3D_MS_MCO 0xff77	/**< MCO marker value */
#define J3D_MS_NLT 0xff76	/**< NLT marker value */
#define J3D_MS_QPD 0xff5a	/**< QPD marker value */
#define J3D_MS_QPC 0xff5b	/**< QPC marker value */

/* ----------------------------------------------------------------------- */
/* Capability RSIZ parameter, extended */
#define J3D_RSIZ_BASIC 0x0000

#define J3D_RSIZ_DCO   0x8001 /* Required */
#define J3D_RSIZ_VSQNT 0x8002
#define J3D_RSIZ_TCQNT 0x8004
#define J3D_RSIZ_VMASK 0x8008
#define J3D_RSIZ_SSOVL 0x8010
#define J3D_RSIZ_ADECS 0x8020
#define J3D_RSIZ_ATK   0x8040 /*Required*/
#define J3D_RSIZ_SSYMK 0x8080
#define J3D_RSIZ_MCT   0x8100 /*Not compatible with DCO*/
#define J3D_RSIZ_NLT   0x8200 /*Required*/
#define J3D_RSIZ_ASHAP 0x8400
#define J3D_RSIZ_PRQNT 0x8800

#define J3D_CAP_10		0x00400000
/* Arbitrary transformation kernel, 15444-2 */
#define J3D_ATK_IRR 0
#define J3D_ATK_REV 1
#define J3D_ATK_ARB 0
#define J3D_ATK_WS 1
#define J3D_ATK_CON 0
/* ----------------------------------------------------------------------- */

/**
Values that specify the status of the decoding process when decoding the main header. 
These values may be combined with a | operator. 
*/
typedef enum J3D_STATUS {
	/**< a SOC marker is expected */
	J3D_STATE_MHSOC  = 0x0001, 
	/**< a SIZ marker is expected */
	J3D_STATE_MHSIZ  = 0x0002, 
	/**< the decoding process is in the main header */
	J3D_STATE_MH     = 0x0004, 
	/**< the decoding process is in a tile part header and expects a SOT marker */
	J3D_STATE_TPHSOT = 0x0008, 
	/**< the decoding process is in a tile part header */
	J3D_STATE_TPH    = 0x0010, 
	/**< the EOC marker has just been read */
	J3D_STATE_MT     = 0x0020, 
	/**< the decoding process must not expect a EOC marker because the codestream is truncated */
	J3D_STATE_NEOC   = 0x0040  
} J3D_STATUS;



/**
Arbitrary transformation kernel
*/
typedef struct opj_atk {
/** index of wavelet kernel */
	int index;
/** Numerical type of scaling factor and lifting step parameters */
	int coeff_typ;		
/** Wavelet filter category */
	int filt_cat;		
/** Wavelet transformation type (REV/IRR) */
	int wt_typ;			
/** Initial odd/even subsequence */
	int minit;			
/** Boundary extension method (constant CON / whole-sample symmetric WS) */
	int exten;			
/** Scaling factor. Only for wt_typ=IRR */
	double Katk;			
/** Number of lifting steps */
	int Natk;			
/** Offset for lifting step s. Only for filt_cat=ARB */
	int Oatk[256];		
/** Base 2 scaling exponent for lifting step s. Only for wt_typ=REV */
	int Eatk[256];		
/** Additive residue for lifting step s. Only for wt_typ=REV */
	int Batk[256];		
/** Number of lifting coefficients signaled for lifting step s  */
	int LCatk[256];	
/** Lifting coefficient k for lifting step s */
	double Aatk[256][256];	
} opj_atk_t;


/**
Quantization stepsize
*/
typedef struct opj_stepsize {
/** exponent */
	int expn;	
/** mantissa */
	int mant;	
} opj_stepsize_t;

/**
Tile-component coding parameters
*/
typedef struct opj_tccp {
	/** coding style */
	int csty;							
	/** number of resolutions of x, y and z-axis */
	int numresolution[3];	
	/** code-blocks width height & depth*/
	int cblk[3];			
	/** code-block coding style */
	int cblksty;			
	/** 0: no ATK (only 9-7 or 5-3) 1: ATK defined WT*/
	int atk_wt[3];				
	/** Arbitrary transformation kernel (15444-2)*/
	opj_atk_t *atk;			
	/** DWT identifier for x, y and z-axis (0:WT9-7 1:WT5-3	>1:WT-atk->index) */
	int dwtid[3];
	/** reversible/irreversible wavelet transfomation (0:irrev 1:reversible)*/	
	int reversible; 		
	/** quantisation style */
	int qntsty;				
	/** stepsizes used for quantization */
	opj_stepsize_t stepsizes[J3D_MAXBANDS];	
	/** number of guard bits. Table A28 de 15444-1*/
	int numgbits;			
	/** Region Of Interest shift */
	int roishift;			
	/** precinct width heigth & depth*/
	int prctsiz[3][J3D_MAXRLVLS];		
} opj_tccp_t;

/**
Tile coding parameters : coding/decoding parameters common to all tiles 
(information like COD, COC in main header)
*/
typedef struct opj_tcp {
/** 1 : first part-tile of a tile */
	int first;				
	/** coding style */
	int csty;				
	/** progression order */
	OPJ_PROG_ORDER prg;		
	/** number of layers */
	int numlayers;			
	/** multi-component transform identifier */
	int mct;				
	/** rates of layers */
	float rates[100];			
	/** number of progression order changes */
	int numpocs;			
	/** indicates if a POC marker has been used O:NO, 1:YES */
	int POC;				
	/** progression order changes */
	opj_poc_t pocs[J3D_MAXRLVLS - 1];
	/** add fixed_quality */
	float distoratio[100];
	/** tile-component coding parameters */
	opj_tccp_t *tccps;		
/** packet header store there for futur use in t2_decode_packet */
	unsigned char *ppt_data;		
	/** pointer remaining on the first byte of the first header if ppt is used */
	unsigned char *ppt_data_first;	
	/** If ppt == 1 --> there was a PPT marker for the present tile */
	int ppt;		
	/** used in case of multiple marker PPT (number of info already stored) */
	int ppt_store;	
	int ppt_len;	
} opj_tcp_t;

/**
Coding parameters
*/
typedef struct opj_cp {
/** transform format 0: 2DWT, 1: 2D1P, 2: 3DWT, 3: 3RLS */
	OPJ_TRANSFORM transform_format;		
	/** entropy coding format 0: 2EB, 1: 3EB, 2: 2GR, 3: 3GR, 4: GRI*/
	OPJ_ENTROPY_CODING encoding_format;	
	/** allocation by rate/distortion */
	int disto_alloc;	
	/** allocation by fixed layer */
	int fixed_alloc;	
	/** add fixed_quality */
	int fixed_quality;	
	/** Rsiz: capabilities */
	int rsiz;			
	/** if != 0, then original dimension divided by 2^(reduce); if == 0 or not used, volume is decoded to the full resolution */
	int reduce[3];		
	/** if != 0, then only the first "layer" layers are decoded; if == 0 or not used, all the quality layers are decoded */
	int layer;			
	/** 0 = no index || 1 = index */
	int index_on;		
	/** Big-Endian/Little-endian order */
	int bigendian;
	/** XTOsiz */
	int tx0;	
	/** YTOsiz */
	int ty0;		
	/** ZTOsiz */
	int tz0;	
	/** XTsiz */
	int tdx;	
	/** YTsiz */
	int tdy;	
	/** ZTsiz */
	int tdz;	
	/** comment for coding */
	char *comment;		
	/** number of tiles in width, heigth and depth */
	int tw;		
	int th;
	int tl;
	/** ID number of the tiles present in the codestream */
	int *tileno;	
	/** size of the vector tileno */
	int tileno_size;
	/** tile coding parameters */
	opj_tcp_t *tcps;
	/** fixed layer */
	int *matrice;		

	/** packet header store there for futur use in t2_decode_packet */
	unsigned char *ppm_data;		
	/** pointer remaining on the first byte of the first header if ppm is used */
	unsigned char *ppm_data_first;	
	/** if ppm == 1 --> there was a PPM marker for the present tile */
	int ppm;			
	/** use in case of multiple marker PPM (number of info already store) */
	int ppm_store;		
	/** use in case of multiple marker PPM (case on non-finished previous info) */
	int ppm_previous;	
	int ppm_len;		
} opj_cp_t;

/**
Information concerning a packet inside tile
*/
typedef struct opj_packet_info {
	/** start position */
	int start_pos;	
	/** end position */
	int end_pos;	
	/** distorsion introduced */
	double disto;	
} opj_packet_info_t;

/**
Index structure : information regarding tiles inside volume
*/
typedef struct opj_tile_info {
	/** value of thresh for each layer by tile cfr. Marcela   */
	double *thresh;		
	/** number of tile */
	int num_tile;		
	/** start position */
	int start_pos;		
	/** end position of the header */
	int end_header;		
	/** end position */
	int end_pos;		
	/** precinct number for each resolution level (width, heigth and depth) */
	int prctno[3][J3D_MAXRLVLS];	
	/** precinct size (in power of 2), in X for each resolution level */
	int prctsiz[3][J3D_MAXRLVLS];	
	/** information concerning packets inside tile */
	opj_packet_info_t *packet;		
	
	/** add fixed_quality */
	int nbpix;			
	/** add fixed_quality */
	double distotile;	
} opj_tile_info_t;

/**
Index structure
*/
typedef struct opj_volume_info {
	
	/** transform format 0: 2DWT, 1: 2D1P, 2: 3DWT, 3: 3RLS */
	OPJ_TRANSFORM transform_format;		
	/** output file format 0: 2EB, 1: 3EB, 2: 2GR, 3: 3GR, 4: GRI*/
	OPJ_ENTROPY_CODING encoding_format;	/** 0 = no index || 1 = index */
	int index_on;	
	/** 0 = wt 9-7 || 1 = wt 5-3 || >=2 wt atk defined */
	int dwtid[3];	
	/** maximum distortion reduction on the whole volume (add for Marcela) */
	double D_max;	
	/** packet number */
	int num;		
	/** writing the packet in the index with t2_encode_packets */
	int index_write;	
	/** volume width, height and depth */
	int volume_w;	
	int volume_h;
	int volume_l;
	/** progression order */
	OPJ_PROG_ORDER prog;	
	/** tile size in x, y and z */
	int tile_x;		
	int tile_y;
	int tile_z;
	/** tile origin in x, y and z */
	int tile_Ox;	
	int tile_Oy;
	int tile_Oz;
	/** number of tiles in X, Y and Z */
	int tw;			
	int th;
	int tl;
	/** component numbers */
	int comp;				
	/** number of layer */
	int layer;				
	/** number of decomposition in X, Y and Z*/
	int decomposition[3];	
	/** DC offset (15444-2) */
	int dcoffset;	
	/** main header position */
	int main_head_end;		
	/** codestream's size */
	int codestream_size;	
	/** information regarding tiles inside volume */
	opj_tile_info_t *tile;	
} opj_volume_info_t;

/**
JPEG-2000 codestream reader/writer
*/
typedef struct opj_j3d {
	/** codec context */
	opj_common_ptr cinfo;	
	/** locate in which part of the codestream the decoder is (main header, tile header, end) */
	int state;				
	/** number of the tile curently concern by coding/decoding */
	int curtileno;			
	/** locate the position of the end of the tile in the codestream, used to detect a truncated codestream (in j3d_read_sod)	*/
	unsigned char *eot;	
	/**	locate the start position of the SOT marker of the current coded tile:  */
	int sot_start;		
	/*  after encoding the tile, a jump (in j3d_write_sod) is done to the SOT marker to store the value of its length. */
	int sod_start;		
	/**	as the J3D-file is written in several parts during encoding, it enables to make the right correction in position return by cio_tell	*/
	int pos_correction;	
	/** array used to store the data of each tile */
	unsigned char **tile_data;	
	/** array used to store the length of each tile */
	int *tile_len;				

	/** decompression only : store decoding parameters common to all tiles */
	opj_tcp_t *default_tcp;		
	/** pointer to the encoded / decoded volume */
	opj_volume_t *volume;		
	/** pointer to the coding parameters */
	opj_cp_t *cp;				
	/** helper used to write the index file */
	opj_volume_info_t *volume_info;	
	/** pointer to the byte i/o stream */
    opj_cio_t *cio;						
} opj_j3d_t;

/** @name Funciones generales */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Creates a J3D decompression structure
@param cinfo Codec context info
@return Returns a handle to a J3D decompressor if successful, returns NULL otherwise
*/
opj_j3d_t* j3d_create_decompress(opj_common_ptr cinfo);
/**
Destroy a J3D decompressor handle
@param j3d J3D decompressor handle to destroy
*/
void j3d_destroy_decompress(opj_j3d_t *j3d);
/**
Setup the decoder decoding parameters using user parameters.
Decoding parameters are returned in j3d->cp. 
@param j3d J3D decompressor handle
@param parameters decompression parameters
*/
void j3d_setup_decoder(opj_j3d_t *j3d, opj_dparameters_t *parameters);
/**
Decode an volume from a JPEG-2000 codestream
@param j3d J3D decompressor handle
@param cio Input buffer stream
@return Returns a decoded volume if successful, returns NULL otherwise
*/
opj_volume_t* j3d_decode(opj_j3d_t *j3d, opj_cio_t *cio);
/**
Decode an volume form a JPT-stream (JPEG 2000, JPIP)
@param j3d J3D decompressor handle
@param cio Input buffer stream
@return Returns a decoded volume if successful, returns NULL otherwise
*/
opj_volume_t* j3d_decode_jpt_stream(opj_j3d_t *j3d, opj_cio_t *cio);
/**
Creates a J3D compression structure
@param cinfo Codec context info
@return Returns a handle to a J3D compressor if successful, returns NULL otherwise
*/
opj_j3d_t* j3d_create_compress(opj_common_ptr cinfo);
/**
Destroy a J3D compressor handle
@param j3d J3D compressor handle to destroy
*/
void j3d_destroy_compress(opj_j3d_t *j3d);
/**
Setup the encoder parameters using the current volume and using user parameters. 
Coding parameters are returned in j3d->cp. 
@param j3d J3D compressor handle
@param parameters compression parameters
@param volume input filled volume
*/
void j3d_setup_encoder(opj_j3d_t *j3d, opj_cparameters_t *parameters, opj_volume_t *volume);
/**
Encode an volume into a JPEG-2000 codestream
@param j3d J3D compressor handle
@param cio Output buffer stream
@param volume Volume to encode
@param index Name of the index file if required, NULL otherwise
@return Returns true if successful, returns false otherwise
*/
bool j3d_encode(opj_j3d_t *j3d, opj_cio_t *cio, opj_volume_t *volume, char *index);
/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __J3D_H */
