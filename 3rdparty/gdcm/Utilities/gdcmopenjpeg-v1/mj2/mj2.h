/*
* Copyright (c) 2003-2004, François-Olivier Devaux
* Copyright (c) 2003-2004,  Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#ifndef __MJ2_H
#define __MJ2_H
/**
@file mj2.h
@brief The Motion JPEG 2000 file format Reader/Writer (MJ22)

*/

/** @defgroup MJ2 MJ2 - Motion JPEG 2000 file format reader/writer */
/*@{*/

#define MJ2_JP    0x6a502020
#define MJ2_FTYP  0x66747970
#define MJ2_MJ2   0x6d6a7032
#define MJ2_MJ2S  0x6d6a3273
#define MJ2_MDAT  0x6d646174
#define MJ2_MOOV  0x6d6f6f76
#define MJ2_MVHD  0x6d766864
#define MJ2_TRAK  0x7472616b
#define MJ2_TKHD  0x746b6864
#define MJ2_MDIA  0x6d646961
#define MJ2_MDHD  0x6d646864
#define MJ2_MHDR  0x6d686472
#define MJ2_HDLR  0x68646C72
#define MJ2_MINF  0x6d696e66
#define MJ2_VMHD  0x766d6864
#define MJ2_SMHD  0x736d6864
#define MJ2_HMHD  0x686d6864
#define MJ2_DINF  0x64696e66
#define MJ2_DREF  0x64726566
#define MJ2_URL   0x75726c20
#define MJ2_URN   0x75726e20
#define MJ2_STBL  0x7374626c
#define MJ2_STSD  0x73747364
#define MJ2_STTS  0x73747473
#define MJ2_STSC  0x73747363
#define MJ2_STSZ  0x7374737a
#define MJ2_STCO  0x7374636f
#define MJ2_MOOF  0x6d6f6f66
#define MJ2_FREE  0x66726565
#define MJ2_SKIP  0x736b6970
#define MJ2_JP2C  0x6a703263
#define MJ2_FIEL  0x6669656c
#define MJ2_JP2P  0x6a703270
#define MJ2_JP2X  0x6a703278
#define MJ2_JSUB  0x6a737562
#define MJ2_ORFO  0x6f72666f
#define MJ2_MVEX  0x6d766578
#define MJ2_JP2   0x6a703220
#define MJ2_J2P0  0x4a325030

/**
Decompressed format used in parameters
YUV = 0
*/
#define YUV_DFMT 1 

/**
Compressed format used in parameters
MJ2 = 0
*/
#define MJ2_CFMT 2


/* ----------------------------------------------------------------------- */

/**
Time To Sample
*/
typedef struct mj2_tts {
  int sample_count;
  int sample_delta;
} mj2_tts_t;

/**
Chunk
*/
typedef struct mj2_chunk {		
  int num_samples;
  int sample_descr_idx;
  int offset;
} mj2_chunk_t;

/**
Sample to chunk
*/
typedef struct mj2_sampletochunk {		
  int first_chunk;
  int samples_per_chunk;
  int sample_descr_idx;
} mj2_sampletochunk_t;

/**
Sample
*/
typedef struct mj2_sample {		
  unsigned int sample_size;
  unsigned int offset;
  unsigned int sample_delta;
} mj2_sample_t;

/**
URL
*/
typedef struct mj2_url {
  int location[4];
} mj2_url_t;

/**
URN
*/
typedef struct mj2_urn {		
  int name[2];
  int location[4];
} mj2_urn_t;

/**
Video Track Parameters
*/
typedef struct mj2_tk {
	/** codec context */
	opj_common_ptr cinfo;
  int track_ID;
  int track_type;
  unsigned int creation_time;
  unsigned int modification_time;
  int duration;
  int timescale;
  int layer;
  int volume;
  int language;
  int balance;
  int maxPDUsize;
  int avgPDUsize;
  int maxbitrate;
  int avgbitrate;
  int slidingavgbitrate;
  int graphicsmode;
  int opcolor[3];
  int num_url;
  mj2_url_t *url;
  int num_urn;
  mj2_urn_t *urn;
  int Dim[2];
  int w;
  int h;
  int visual_w;
  int visual_h;
  int CbCr_subsampling_dx;
  int CbCr_subsampling_dy;
  int sample_rate;
  int sample_description;
  int horizresolution;
  int vertresolution;
  int compressorname[8];
  int depth;
  unsigned char fieldcount;
  unsigned char fieldorder;
  unsigned char or_fieldcount;
  unsigned char or_fieldorder;
  int num_br;
  unsigned int *br;
  unsigned char num_jp2x;
  unsigned char *jp2xdata;
  unsigned char hsub;
  unsigned char vsub;
  unsigned char hoff;
  unsigned char voff;
  int trans_matrix[9];
	/** Number of samples */
  unsigned int num_samples;	
  int transorm;
  int handler_type;
  int name_size;
  unsigned char same_sample_size;
  int num_tts;
	/** Time to sample    */
  mj2_tts_t *tts;		
  unsigned int num_chunks;
  mj2_chunk_t *chunk;
  int num_samplestochunk;
  mj2_sampletochunk_t *sampletochunk;
  char *name;
  opj_jp2_t jp2_struct;
	/** Sample parameters */
  mj2_sample_t *sample;		
} mj2_tk_t;			

/**
MJ2 box
*/
typedef struct mj2_box {
  int length;
  int type;
  int init_pos;
} mj2_box_t;

/**
MJ2 Movie
*/
typedef struct opj_mj2 {		
	/** codec context */
	opj_common_ptr cinfo;
	/** handle to the J2K codec  */
	opj_j2k_t *j2k;
  unsigned int brand;
  unsigned int minversion;
  int num_cl;
  unsigned int *cl;
  unsigned int creation_time;
  unsigned int modification_time;
  int timescale;
  unsigned int duration;
  int rate;
  int num_vtk;
  int num_stk;
  int num_htk;
  int volume;
  int trans_matrix[9];
  int next_tk_id;
	/** Track Parameters  */
  mj2_tk_t *tk;			
} opj_mj2_t;

/**
Decompression parameters
*/
typedef struct mj2_dparameters {
	/**@name command line encoder parameters (not used inside the library) */
	/*@{*/
	/** input file name */
	char infile[OPJ_PATH_LEN];
	/** output file name */
	char outfile[OPJ_PATH_LEN];	
	/** J2K decompression parameters */
	opj_dparameters_t j2k_parameters;	
} mj2_dparameters_t;

/**
Compression parameters
*/
typedef struct mj2_cparameters {
	/**@name command line encoder parameters (not used inside the library) */
	/*@{*/
	/** J2K compression parameters */
	opj_cparameters_t j2k_parameters;	
	/** input file name */
	char infile[OPJ_PATH_LEN];
	/** output file name */
	char outfile[OPJ_PATH_LEN];	
	/** input file format 0:MJ2 */
	int decod_format;
	/** output file format 0:YUV */
	int cod_format;
	/** Portion of the image coded */
	int Dim[2];
	/** YUV Frame width */
	int w;
	/** YUV Frame height */
	int h;
	/*   Sample rate of YUV 4:4:4, 4:2:2 or 4:2:0 */
	int CbCr_subsampling_dx;	
	/*   Sample rate of YUV 4:4:4, 4:2:2 or 4:2:0 */
  int CbCr_subsampling_dy;	
	/*   Video Frame Rate  */
  int frame_rate;		
	/*   In YUV files, numcomps always considered as 3 */
  int numcomps;			
	/*   In YUV files, precision always considered as 8 */
  int prec;		
} mj2_cparameters_t;


/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
Write the JP box 
*/
void mj2_write_jp(opj_cio_t *cio);
/**
Write the FTYP box
@param movie MJ2 movie
@param cio Output buffer stream
*/
void mj2_write_ftyp(opj_mj2_t *movie, opj_cio_t *cio);
/**
Creates an MJ2 decompression structure
@return Returns a handle to a MJ2 decompressor if successful, returns NULL otherwise
*/
opj_dinfo_t* mj2_create_decompress();
/**
Destroy a MJ2 decompressor handle
@param movie MJ2 decompressor handle to destroy
*/
void mj2_destroy_decompress(opj_mj2_t *movie);
/**
Setup the decoder decoding parameters using user parameters.
Decoding parameters are returned in mj2->j2k->cp. 
@param movie MJ2 decompressor handle
@param parameters decompression parameters
*/
void mj2_setup_decoder(opj_mj2_t *movie, mj2_dparameters_t *mj2_parameters);
/**
Decode an image from a JPEG-2000 file stream
@param movie MJ2 decompressor handle
@param cio Input buffer stream
@return Returns a decoded image if successful, returns NULL otherwise
*/
opj_image_t* mj2_decode(opj_mj2_t *movie, opj_cio_t *cio);
/**
Creates a MJ2 compression structure
@return Returns a handle to a MJ2 compressor if successful, returns NULL otherwise
*/
opj_cinfo_t* mj2_create_compress();
/**
Destroy a MJ2 compressor handle
@param movie MJ2 compressor handle to destroy
*/
void mj2_destroy_compress(opj_mj2_t *movie);
/**
Setup the encoder parameters using the current image and using user parameters. 
Coding parameters are returned in mj2->j2k->cp. 
@param movie MJ2 compressor handle
@param parameters compression parameters
*/
void mj2_setup_encoder(opj_mj2_t *movie, mj2_cparameters_t *parameters);
/**
Encode an image into a JPEG-2000 file stream
@param movie MJ2 compressor handle
@param cio Output buffer stream
@param image Image to encode
@param index Name of the index file if required, NULL otherwise
@return Returns true if successful, returns false otherwise
*/
bool mj2_encode(opj_mj2_t *movie, opj_cio_t *cio, opj_image_t *image, char *index);

/**
Init a Standard MJ2 movie
@param movie MJ2 Movie
@return Returns 0 if successful, returns 1 otherwise
*/
int mj2_init_stdmovie(opj_mj2_t *movie);
/**
Read the structure of an MJ2 file
@param File MJ2 input File
@param movie J2 movie structure 
@return Returns 0 if successful, returns 1 otherwise
*/
int mj2_read_struct(FILE *file, opj_mj2_t *mj2);
/**
Write the the MOOV box to an output buffer stream
@param movie MJ2 movie structure 
@param cio Output buffer stream
*/
void mj2_write_moov(opj_mj2_t *movie, opj_cio_t *cio);


/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* __MJ2_H */
