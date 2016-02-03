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
#ifndef __TCD_H
#define __TCD_H
/**
@file tcd.h
@brief Implementation of a tile coder/decoder (TCD)

The functions in TCD.C have for goal to encode or decode each tile independently from
each other. The functions in TCD.C are used by some function in JP3D.C.
*/

/** @defgroup TCD TCD - Implementation of a tile coder/decoder */
/*@{*/

/**
Tile coder/decoder: segment instance
*/
typedef struct opj_tcd_seg {
/** Number of passes in the segment */
	int numpasses;			
/** Length of information */
    int len;					
/** Data */
	unsigned char *data;		
/** Number of passes posible for the segment */
	int maxpasses;			
/** Number of passes added to the segment */
	int numnewpasses;		    
/** New length after inclusion of segments */
	int newlen;
} opj_tcd_seg_t;

/**
Tile coder/decoder: pass instance
*/
typedef struct opj_tcd_pass {
/** Rate obtained in the pass*/
  int rate;					
/** Distorsion obtained in the pass*/
  double distortiondec;		
  int term;
/** Length of information */
  int len;					
} opj_tcd_pass_t;

/**
Tile coder/decoder: layer instance
*/
typedef struct opj_tcd_layer {
/** Number of passes in the layer */
	int numpasses;			
/** Length of information */
  int len;					
/** Distortion within layer */
  double disto;				/* add for index (Cfr. Marcela) */
  unsigned char *data;		/* data */
} opj_tcd_layer_t;

/**
Tile coder/decoder: codeblock instance
*/
typedef struct opj_tcd_cblk {
/** Dimension of the code-blocks : left upper corner (x0, y0, z0) */
  int x0, y0, z0;
/** Dimension of the code-blocks : right low corner (x1,y1,z1) */
  int x1, y1, z1;		
/** Number of bits per simbol in codeblock */
  int numbps;
  int numlenbits;
  int len;						/* length */
/** Number of pass already done for the code-blocks */
  int numpasses;				
/** number of pass added to the code-blocks */
  int numnewpasses;				
/** Number of segments */
  int numsegs;					
/** Segments informations */
  opj_tcd_seg_t segs[100];		
/** Number of passes in the layer */
  int numpassesinlayers;		
/** Layer information */
  opj_tcd_layer_t layers[100];	
/** Total number of passes */
  int totalpasses;				
/** Information about the passes */
  opj_tcd_pass_t passes[100];	
/* Data */
  unsigned char data[524288];		
  //unsigned char *data;
} opj_tcd_cblk_t;

/**
Tile coder/decoder: precint instance
*/
typedef struct opj_tcd_precinct {
/** Dimension of the precint : left upper corner (x0, y0, z0) */
  int x0, y0, z0;
/** Dimension of the precint : right low corner (x1,y1,z1) */
  int x1, y1, z1;
/** Number of codeblocks in precinct in width and heigth and length*/
  int cblkno[3];				
/** Information about the codeblocks */
  opj_tcd_cblk_t *cblks;		
/** Inclusion tree */
  opj_tgt_tree_t *incltree;		
/** Missing MSBs tree */
  opj_tgt_tree_t *imsbtree;		
} opj_tcd_precinct_t;

/**
Tile coder/decoder: subband instance
*/
typedef struct opj_tcd_band {
/** Dimension of the subband : left upper corner (x0, y0, z0) */
  int x0, y0, z0;
/** Dimension of the subband : right low corner (x1,y1,z1) */
  int x1, y1, z1;
/** Information about the precints */
  opj_tcd_precinct_t *precincts;	/* precinct information */
/** Number of bits per symbol in band */
  int numbps;
/** Quantization stepsize associated */
  float stepsize;
/** Band orientation (O->LLL,...,7->HHH) */
  int bandno;
} opj_tcd_band_t;

/**
Tile coder/decoder: resolution instance
*/
typedef struct opj_tcd_resolution {
/** Dimension of the resolution level : left upper corner (x0, y0, z0) */
  int x0, y0, z0;
/** Dimension of the resolution level : right low corner (x1,y1,z1) */
  int x1, y1, z1;
/** Number of precints in each dimension for the resolution level */
  int prctno[3];				
/** Number of subbands for the resolution level */
  int numbands;					
/** Subband information */
  opj_tcd_band_t *bands;		
} opj_tcd_resolution_t;

/**
Tile coder/decoder: component instance
*/
typedef struct opj_tcd_tilecomp {
/** Dimension of the component : left upper corner (x0, y0, z0) */
  int x0, y0, z0;
/** Dimension of the component : right low corner (x1,y1,z1) */
  int x1, y1, z1;
/** Number of resolutions level if DWT transform*/
  int numresolution[3];					
/** Resolution information */
  opj_tcd_resolution_t *resolutions;	
/** Data of the component */
  int *data;					
/** Fixed_quality related */
  int nbpix;				
/** Number of bits per voxel in component */
  int bpp;
} opj_tcd_tilecomp_t;

/**
Tile coder/decoder: tile instance
*/
typedef struct opj_tcd_tile {
/** Dimension of the tile : left upper corner (x0, y0, z0) */
  int x0, y0, z0;
/** Dimension of the tile : right low corner (x1,y1,z1) */
  int x1, y1, z1;
/** Number of components in tile */
  int numcomps;					
/** Components information */
  opj_tcd_tilecomp_t *comps;	
/** Fixed_quality related : no of bytes of data*/
  int nbpix;					
/** Fixed_quality related : distortion achieved in tile */
  double distotile;				
/** Fixed_quality related : distortion achieved in each layer */
  double distolayer[100];		
} opj_tcd_tile_t;

/**
Tile coder/decoder: volume instance
*/
typedef struct opj_tcd_volume {
/** Number of tiles in width and heigth and length */
	int tw, th, tl;				
/** Tiles information */
  opj_tcd_tile_t *tiles;		
} opj_tcd_volume_t;

/**
Tile coder/decoder
*/
typedef struct opj_tcd {
/** Codec context */	
	opj_common_ptr cinfo;			
/** Volume information */	
	opj_volume_t *volume;			
/** Coding parameters */	
	opj_cp_t *cp;					
/** Coding/decoding parameters common to all tiles */	
	opj_tcp_t *tcp;					
/** Info on each volume tile */
	opj_tcd_volume_t *tcd_volume;	
/** Pointer to the current encoded/decoded tile */
	opj_tcd_tile_t *tcd_tile;		
/** Current encoded/decoded tile */
	int tcd_tileno;					

	/**@name working variables */
	/*@{*/
	opj_tcd_tile_t *tile;
	opj_tcd_tilecomp_t *tilec;
	opj_tcd_resolution_t *res;
	opj_tcd_band_t *band;
	opj_tcd_precinct_t *prc;
	opj_tcd_cblk_t *cblk;
	/*@}*/
} opj_tcd_t;

/** @name Funciones generales */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
Dump the content of a tcd structure
*/
void tcd_dump(FILE *fd, opj_tcd_t *tcd, opj_tcd_volume_t *img);
/**
Create a new TCD handle
@param cinfo Codec context info
@return Returns a new TCD handle if successful returns NULL otherwise
*/
opj_tcd_t* tcd_create(opj_common_ptr cinfo);
/**
Destroy a previously created TCD handle
@param tcd TCD handle to destroy
*/
void tcd_destroy(opj_tcd_t *tcd);
/**
Initialize the tile coder (allocate the memory) 
@param tcd TCD handle
@param volume Raw volume
@param cp Coding parameters
@param curtileno Number that identifies the tile that will be encoded
*/
void tcd_malloc_encode(opj_tcd_t *tcd, opj_volume_t * volume, opj_cp_t * cp, int curtileno);
/**
Initialize the tile coder (reuses the memory allocated by tcd_malloc_encode)(for 3D-DWT)
@param tcd TCD handle
@param volume Raw volume
@param cp Coding parameters
@param curtileno Number that identifies the tile that will be encoded
*/
void tcd_init_encode(opj_tcd_t *tcd, opj_volume_t * volume, opj_cp_t * cp, int curtileno);
/**
Free the memory allocated for encoding
@param tcd TCD handle
*/
void tcd_free_encode(opj_tcd_t *tcd);
/**
Initialize the tile decoder
@param tcd TCD handle
@param volume Raw volume
@param cp Coding parameters
*/
void tcd_malloc_decode(opj_tcd_t *tcd, opj_volume_t * volume, opj_cp_t * cp);

void tcd_makelayer_fixed(opj_tcd_t *tcd, int layno, int final);
void tcd_rateallocate_fixed(opj_tcd_t *tcd);
void tcd_makelayer(opj_tcd_t *tcd, int layno, double thresh, int final);
bool tcd_rateallocate(opj_tcd_t *tcd, unsigned char *dest, int len, opj_volume_info_t * volume_info);
/**
Encode a tile from the raw volume into a buffer
@param tcd TCD handle
@param tileno Number that identifies one of the tiles to be encoded
@param dest Destination buffer
@param len Length of destination buffer
@param volume_info Creation of index file
@return 
*/
int tcd_encode_tile(opj_tcd_t *tcd, int tileno, unsigned char *dest, int len, opj_volume_info_t * volume_info);
/**
Decode a tile from a buffer into a raw volume
@param tcd TCD handle
@param src Source buffer
@param len Length of source buffer
@param tileno Number that identifies one of the tiles to be decoded
*/
bool tcd_decode_tile(opj_tcd_t *tcd, unsigned char *src, int len, int tileno);
/**
Free the memory allocated for decoding
@param tcd TCD handle
*/
void tcd_free_decode(opj_tcd_t *tcd);

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __TCD_H */
