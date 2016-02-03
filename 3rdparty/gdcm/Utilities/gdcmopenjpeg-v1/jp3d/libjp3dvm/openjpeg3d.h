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
#ifndef OPENJPEG_H
#define OPENJPEG_H

/* 
==========================================================
   Compiler directives
==========================================================
*/

#if defined(OPJ_STATIC) || !(defined(WIN32) || defined(__WIN32__))
#define OPJ_API
#define OPJ_CALLCONV
#else
#define OPJ_CALLCONV __stdcall
/*
The following ifdef block is the standard way of creating macros which make exporting 
from a DLL simpler. All files within this DLL are compiled with the OPJ_EXPORTS
symbol defined on the command line. this symbol should not be defined on any project
that uses this DLL. This way any other project whose source files include this file see 
OPJ_API functions as being imported from a DLL, wheras this DLL sees symbols
defined with this macro as being exported.
*/
#ifdef OPJ_EXPORTS
#define OPJ_API __declspec(dllexport)
#else
#define OPJ_API __declspec(dllimport)
#endif /* OPJ_EXPORTS */
#endif /* !OPJ_STATIC || !WIN32 */

#ifndef __cplusplus
#if defined(HAVE_STDBOOL_H)
/*
The C language implementation does correctly provide the standard header
file "stdbool.h".
 */
#include <stdbool.h>
#else
/*
The C language implementation does not provide the standard header file
"stdbool.h" as required by ISO/IEC 9899:1999.  Try to compensate for this
braindamage below.
*/
#if !defined(bool)
#define	bool	int
#endif
#if !defined(true)
#define true	1
#endif
#if !defined(false)
#define	false	0
#endif
#endif
#endif /* __cplusplus */

/* 
==========================================================
   Useful constant definitions
==========================================================
*/
#ifndef MAX_SLICES
#define MAX_SLICES 300	/**< Maximum allowed size for slices */
#endif /* MAX_PATH */

#ifndef MAX_PATH
#define MAX_PATH 260	/**< Maximum allowed size for filenames */
#endif /* MAX_PATH */

#define J3D_MAXRLVLS 32					/**< Number of maximum resolution level authorized */
#define J3D_MAXBANDS (7*J3D_MAXRLVLS + 1)	/**< Number of maximum sub-band linked to number of resolution level */

#define TINY 1.0E-20
/* 
==========================================================
   enum definitions
==========================================================
*/

#define J2K_CFMT 0
#define J3D_CFMT 1
#define LSE_CFMT 2

#define BIN_DFMT 0
#define PGX_DFMT 1
#define IMG_DFMT 2
/* ----------------------------------------------------------------------- */

/** Progression order */
typedef enum PROG_ORDER {
/**< place-holder */	
	PROG_UNKNOWN = -1,	
/**< layer-resolution-component-precinct order */	
	LRCP = 0,		
/**< resolution-layer-component-precinct order */	
	RLCP = 1,		
/**< resolution-precinct-component-layer order */	
	RPCL = 2,		
/**< precinct-component-resolution-layer order */	
	PCRL = 3,		
/**< component-precinct-resolution-layer order */	
	CPRL = 4		
} OPJ_PROG_ORDER;

/**
Supported volume color spaces
*/
typedef enum COLOR_SPACE {
/**< place-holder */	
	CLRSPC_UNKNOWN = -1,	
/**< sRGB */	
	CLRSPC_SRGB = 1,		
/**< grayscale */	
	CLRSPC_GRAY = 2,		
/**< YUV */	
	CLRSPC_SYCC = 3			
} OPJ_COLOR_SPACE;

/**
Supported codec
*/
typedef enum CODEC_FORMAT {
	/**< place-holder */
	CODEC_UNKNOWN = -1,
/**< JPEG-2000 codestream : read/write */
	CODEC_J2K = 0,		
/**< JPEG-2000 Part 10 file format : read/write */
	CODEC_J3D = 1		
} OPJ_CODEC_FORMAT;

/**
Supported entropy coding algorithms
*/
typedef enum ENTROPY_CODING {
/**< place-holder */
	ENCOD_UNKNOWN = -1,
/**< 2D EBCOT encoding */
	ENCOD_2EB = 0,		
/**< 3D EBCOT encoding */
	ENCOD_3EB = 1,		
/**< Golomb-Rice coding with 2D context */
	ENCOD_2GR = 2,		
/**< Golomb-Rice coding with 3D context  */
	ENCOD_3GR = 3		
} OPJ_ENTROPY_CODING;

/**
Supported transforms
*/
typedef enum TRANSFORM {
/**< place-holder */
	TRF_UNKNOWN = -1,	
/**< 2D DWT, no transform in axial dim */
	TRF_2D_DWT = 0,		
/**< 3D DWT */
	TRF_3D_DWT = 1,		
/**< 3D prediction*/
	TRF_3D_RLS = 2,		
	TRF_3D_LSE = 3
} OPJ_TRANSFORM;
/* 
==========================================================
   event manager typedef definitions
==========================================================
*/

/**
Callback function prototype for events
@param msg Event message
@param client_data 
*/
typedef void (*opj_msg_callback) (const char *msg, void *client_data);

/**
Message handler object
used for 
<ul>
<li>Error messages
<li>Warning messages
<li>Debugging messages
</ul>
*/
typedef struct opj_event_mgr {
	/** Error message callback if available, NULL otherwise */
	opj_msg_callback error_handler;
	/** Warning message callback if available, NULL otherwise */
	opj_msg_callback warning_handler;
	/** Debug message callback if available, NULL otherwise */
	opj_msg_callback info_handler;
} opj_event_mgr_t;


/* 
==========================================================
   codec typedef definitions
==========================================================
*/

/**
Progression order changes
*/
typedef struct opj_poc {
  int resno0, compno0;
  int layno1, resno1, compno1;
  OPJ_PROG_ORDER prg;
  int tile;
  char progorder[4];
} opj_poc_t;


/**
Compression parameters
*/
typedef struct opj_cparameters {
/** size of tile: tile_size_on = false (not in argument) or = true (in argument) */
	bool tile_size_on;	
/** XTOsiz */
	int cp_tx0;
/** YTOsiz */
	int cp_ty0;	
/** ZTOsiz */
	int cp_tz0;		

/** XTsiz */
	int cp_tdx;	
/** YTsiz */
	int cp_tdy;	
/** ZTsiz */
	int cp_tdz;		
							
/** allocation by rate/distortion */
	int cp_disto_alloc;		
/** allocation by fixed layer */		
	int cp_fixed_alloc;		
/** add fixed_quality */	
	int cp_fixed_quality;	
/** fixed layer */
	int *cp_matrice;
/** number of layers */
	int tcp_numlayers;
/** rates for successive layers */
	float tcp_rates[100];
/** psnr's for successive layers */
	float tcp_distoratio[100];	
/** comment for coding */
	char *cp_comment;
/** csty : coding style */
	int csty;					
/** DC offset (DCO) */
	int dcoffset;
/** progression order (default LRCP) */
	OPJ_PROG_ORDER prog_order;	
/** progression order changes */
	opj_poc_t POC[J3D_MAXRLVLS-1];	
/** number of progression order changes (POC), default to 0 */
	int numpocs;				

/** number of resolutions */
	int numresolution[3];		
/** initial code block width, height and depth, default to 64 */
 	int cblock_init[3];			
/** mode switch (1=BYPASS(LAZY) 2=RESET 4=RESTART(TERMALL) 8=VSC 16=ERTERM(SEGTERM) 32=SEGMARK(SEGSYM)) */
	int mode;					

/** 1 : use the irreversible DWT 9-7, 0 : use lossless compression (default) */
	int irreversible;			
/** WT from ATK, default to 0 (false), no of atk used */
	int atk_wt[3];
/** region of interest: affected component in [0..3], -1 means no ROI */
	int roi_compno;				
/** region of interest: upshift value */
	int roi_shift;				

/* number of precinct size specifications */
	int res_spec;				
/** initial precinct width */
	int prct_init[3][J3D_MAXRLVLS];	

/** transform format 0: 0: 2DWT, 1: 2D1P, 2: 3DWT, 3: 3RLS */
	OPJ_TRANSFORM transform_format;	
/** output file format 0: 2EB, 1: 3EB, 2: 2GR, 3: 3GR, 4: GRI */
	OPJ_ENTROPY_CODING encoding_format;	

	/**@name command line encoder parameters (not used inside the library) */
	/*@{*/
	char infile[MAX_PATH];		/** input file name */
	char outfile[MAX_PATH];		/** output file name */
	char imgfile[MAX_PATH];		/** IMG file name for BIN volumes*/
	int index_on;				/** creation of an index file, default to 0 (false) */
	char index[MAX_PATH];		/** index file name */
	
	int volume_offset_x0;		/** subvolume encoding: origin volume offset in x, y and z direction */
	int volume_offset_y0;
	int volume_offset_z0;
	
	int subsampling_dx;			/** subsampling value for dx */
	int subsampling_dy;
	int subsampling_dz;
	
	int decod_format;			/** input file format 0: BIN, 1: PGX */
	int cod_format;				/** output file format 0: JP3D */
	/*@}*/
} opj_cparameters_t;

/**
Decompression parameters
*/
typedef struct opj_dparameters {
/** Set the number of highest resolution levels to be discarded. if != 0, then original dimension divided by 2^(reduce); if == 0 or not used, volume is decoded to the full resolution */
	int cp_reduce[3];		
/** Set the maximum number of quality layers to decode. if != 0, then only the first "layer" layers are decoded; if == 0 or not used, all the quality layers are decoded 	*/
	int cp_layer;	
	int bigendian;
	
	/**@name command line encoder parameters (not used inside the library) */
	/*@{*/
/** input file name */
	char infile[MAX_PATH];	
/** output file name */
	char outfile[MAX_PATH];	
/** IMG file name for BIN volumes*/
	char imgfile[MAX_PATH];
/** Original file name for PSNR measures*/
	char original[MAX_PATH];
/** input file format 0: J2K, 1: JP3D */
	int decod_format;		
/** input file format 0: BIN, 1: PGM */
	int cod_format;
/** original file format 0: BIN, 1: PGM */
	int orig_format;	
	/*@}*/
} opj_dparameters_t;

/** Common fields between JPEG-2000 compression and decompression master structs. */
#define opj_common_fields \
	opj_event_mgr_t *event_mgr;	/**< pointer to the event manager */\
	void * client_data;			/**< Available for use by application */\
	bool is_decompressor;		/**< So common code can tell which is which */\
	OPJ_CODEC_FORMAT codec_format;		/**< selected codec */\
	OPJ_ENTROPY_CODING encoding_format;	/**< selected entropy coding */\
	OPJ_TRANSFORM transform_format;		/**< selected transform */\
	void *j3d_handle			/**< pointer to the J3D codec */
	
/* Routines that are to be used by both halves of the library are declared
 * to receive a pointer to this structure.  There are no actual instances of
 * opj_common_struct_t, only of opj_cinfo_t and opj_dinfo_t.
 */
typedef struct opj_common_struct {
  opj_common_fields;		/* Fields common to both master struct types */
  /* Additional fields follow in an actual opj_cinfo_t or
   * opj_dinfo_t.  All three structs must agree on these
   * initial fields!  (This would be a lot cleaner in C++.)
   */
} opj_common_struct_t;

typedef opj_common_struct_t * opj_common_ptr;

/**
Compression context info
*/
typedef struct opj_cinfo {
	/** Fields shared with opj_dinfo_t */
	opj_common_fields;	
	/* other specific fields go here */
} opj_cinfo_t;

/**
Decompression context info
*/
typedef struct opj_dinfo {
	/** Fields shared with opj_cinfo_t */
	opj_common_fields;	
	/* other specific fields go here */
} opj_dinfo_t;

/* 
==========================================================
   I/O stream typedef definitions
==========================================================
*/

/*
 * Stream open flags.
 */
/** The stream was opened for reading. */
#define OPJ_STREAM_READ	0x0001
/** The stream was opened for writing. */
#define OPJ_STREAM_WRITE 0x0002

/**
Byte input-output stream (CIO)
*/
typedef struct opj_cio {
/** codec context */
	opj_common_ptr cinfo;	
/** open mode (read/write) either OPJ_STREAM_READ or OPJ_STREAM_WRITE */
	int openmode;			
/** pointer to the start of the buffer */
	unsigned char *buffer;	
/** buffer size in bytes */
	int length;				
/** pointer to the start of the stream */
	unsigned char *start;		
/** pointer to the end of the stream */
	unsigned char *end;			
/** pointer to the current position */
	unsigned char *bp;			
} opj_cio_t;

/* 
==========================================================
   volume typedef definitions
==========================================================
*/

/**
Defines a single volume component
*/
typedef struct opj_volume_comp {
/** XRsiz: horizontal separation of a sample of ith component with respect to the reference grid */
	int dx;	
/** YRsiz: vertical separation of a sample of ith component with respect to the reference grid */
	int dy;	
/** ZRsiz: vertical separation of a sample of ith component with respect to the reference grid */
	int dz;
/** data width */
	int w;	
	/** data height */
	int h;	
	/** data length : no of slices */
	int l;	
	/** x component offset compared to the whole volume */
	int x0;	
	/** y component offset compared to the whole volume */
	int y0;	
	/** z component offset compared to the whole volume */
	int z0;	
	/** precision */
	int prec;	
	/** volume depth in bits */
	int bpp;	
	/** DC offset (15444-2) */
	int dcoffset;
	/** signed (1) / unsigned (0) */
	int sgnd;		
	/** BE byte order (1) / LE byte order  (0) */
	int bigendian;
	/** number of decoded resolution */
	int resno_decoded[3];	
	/** number of division by 2 of the out volume compared to the original size of volume */
	int factor[3];	
	/** volume component data */
	int *data;				
} opj_volume_comp_t;

/** 
Defines volume data and characteristics
*/
typedef struct opj_volume {
/** XOsiz: horizontal offset from the origin of the reference grid to the left side of the volume area */
	int x0;	
/** YOsiz: vertical offset from the origin of the reference grid to the top side of the volume area */
	int y0;	
/** ZOsiz: vertical offset from the origin of the reference grid to the top side of the volume area */
	int z0;	
/** Xsiz: width of the reference grid */
	int x1;	
/** Ysiz: height of the reference grid */
	int y1;	
/** Zsiz: length of the reference grid */
	int z1;	
/** number of components in the volume */
	int numcomps;	
/** number of slices in the volume */
	int numslices;	
/** color space: sRGB, Greyscale or YUV */
	OPJ_COLOR_SPACE color_space;
/** volume components */
	opj_volume_comp_t *comps;	
} opj_volume_t;

/**
Component parameters structure used by the opj_volume_create function
*/
typedef struct opj_volume_comptparm {
	/** XRsiz: horizontal separation of a sample of ith component with respect to the reference grid */
	int dx;	
	/** YRsiz: vertical separation of a sample of ith component with respect to the reference grid */
	int dy;	
	/** ZRsiz: axial separation of a sample of ith component with respect to the reference grid */
	int dz;	
	/** data width */
	int w;	
	/** data height */
	int h;	
	/** data length */
	int l;	
	/** x component offset compared to the whole volume */
	int x0;	
	/** y component offset compared to the whole volume */
	int y0;
	/** z component offset compared to the whole volume */
	int z0;	
	/** precision */
	int prec;
	/** volume depth in bits */
	int bpp;	
	/** signed (1) / unsigned (0) */
	int sgnd;
	/** DC offset*/
	int dcoffset;
	/** BE byte order (1) / LE byte order  (0) */
	int bigendian;
} opj_volume_cmptparm_t;

#ifdef __cplusplus
extern "C" {
#endif


/* 
==========================================================
   openjpeg version
==========================================================
*/

OPJ_API const char * OPJ_CALLCONV opj_version();

/* 
==========================================================
   volume functions definitions
==========================================================
*/

/**
Create an volume
@param numcmpts number of components
@param cmptparms components parameters
@param clrspc volume color space
@return returns a new volume structure if successful, returns NULL otherwise
*/
OPJ_API opj_volume_t* OPJ_CALLCONV opj_volume_create(int numcmpts, opj_volume_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc);

/**
Deallocate any resources associated with an volume
@param volume volume to be destroyed
*/
OPJ_API void OPJ_CALLCONV opj_volume_destroy(opj_volume_t *volume);

/* 
==========================================================
   stream functions definitions
==========================================================
*/

/**
Open and allocate a memory stream for read / write. 
On reading, the user must provide a buffer containing encoded data. The buffer will be 
wrapped by the returned CIO handle. 
On writing, buffer parameters must be set to 0: a buffer will be allocated by the library 
to contain encoded data. 
@param cinfo Codec context info
@param buffer Reading: buffer address. Writing: NULL
@param length Reading: buffer length. Writing: 0
@return Returns a CIO handle if successful, returns NULL otherwise
*/
OPJ_API opj_cio_t* OPJ_CALLCONV opj_cio_open(opj_common_ptr cinfo, unsigned char *buffer, int length);

/**
Close and free a CIO handle
@param cio CIO handle to free
*/
OPJ_API void OPJ_CALLCONV opj_cio_close(opj_cio_t *cio);

/**
Get position in byte stream
@param cio CIO handle
@return Returns the position in bytes
*/
OPJ_API int OPJ_CALLCONV cio_tell(opj_cio_t *cio);
/**
Set position in byte stream
@param cio CIO handle
@param pos Position, in number of bytes, from the beginning of the stream
*/
OPJ_API void OPJ_CALLCONV cio_seek(opj_cio_t *cio, int pos);

/* 
==========================================================
   event manager functions definitions
==========================================================
*/

OPJ_API opj_event_mgr_t* OPJ_CALLCONV opj_set_event_mgr(opj_common_ptr cinfo, opj_event_mgr_t *event_mgr, void *context);

/* 
==========================================================
   codec functions definitions
==========================================================
*/
/**
Creates a J3D decompression structure
@param format Decoder to select
@return Returns a handle to a decompressor if successful, returns NULL otherwise
*/
OPJ_API opj_dinfo_t* OPJ_CALLCONV opj_create_decompress(OPJ_CODEC_FORMAT format);
/**
Destroy a decompressor handle
@param dinfo decompressor handle to destroy
*/
OPJ_API void OPJ_CALLCONV opj_destroy_decompress(opj_dinfo_t *dinfo);
/**
Set decoding parameters to default values
@param parameters Decompression parameters
*/
OPJ_API void OPJ_CALLCONV opj_set_default_decoder_parameters(opj_dparameters_t *parameters);
/**
Setup the decoder decoding parameters using user parameters.
Decoding parameters are returned in j3d->cp. 
@param dinfo decompressor handle
@param parameters decompression parameters
*/
OPJ_API void OPJ_CALLCONV opj_setup_decoder(opj_dinfo_t *dinfo, opj_dparameters_t *parameters);
/**
Decode an volume from a JPEG-2000 codestream
@param dinfo decompressor handle
@param cio Input buffer stream
@return Returns a decoded volume if successful, returns NULL otherwise
*/
OPJ_API opj_volume_t* OPJ_CALLCONV opj_decode(opj_dinfo_t *dinfo, opj_cio_t *cio);
/**
Creates a J3D/JP2 compression structure
@param format Coder to select
@return Returns a handle to a compressor if successful, returns NULL otherwise
*/
OPJ_API opj_cinfo_t* OPJ_CALLCONV opj_create_compress(OPJ_CODEC_FORMAT format);
/**
Destroy a compressor handle
@param cinfo compressor handle to destroy
*/
OPJ_API void OPJ_CALLCONV opj_destroy_compress(opj_cinfo_t *cinfo);
/**
Set encoding parameters to default values, that means : 
<ul>
<li>Lossless
<li>1 tile
<li>Size of precinct : 2^15 x 2^15 (means 1 precinct)
<li>Size of code-block : 64 x 64
<li>Number of resolutions: 6
<li>No SOP marker in the codestream
<li>No EPH marker in the codestream
<li>No sub-sampling in x or y direction
<li>No mode switch activated
<li>Progression order: LRCP
<li>No index file
<li>No ROI upshifted
<li>No offset of the origin of the volume
<li>No offset of the origin of the tiles
<li>Reversible DWT 5-3
</ul>
@param parameters Compression parameters
*/
OPJ_API void OPJ_CALLCONV opj_set_default_encoder_parameters(opj_cparameters_t *parameters);
/**
Setup the encoder parameters using the current volume and using user parameters. 
@param cinfo compressor handle
@param parameters compression parameters
@param volume input filled volume
*/
OPJ_API void OPJ_CALLCONV opj_setup_encoder(opj_cinfo_t *cinfo, opj_cparameters_t *parameters, opj_volume_t *volume);
/**
Encode an volume into a JPEG-2000 codestream
@param cinfo compressor handle
@param cio Output buffer stream
@param volume Volume to encode
@param index Name of the index file if required, NULL otherwise
@return Returns true if successful, returns false otherwise
*/
OPJ_API bool OPJ_CALLCONV opj_encode(opj_cinfo_t *cinfo, opj_cio_t *cio, opj_volume_t *volume, char *index);

#ifdef __cplusplus
}
#endif

#endif /* OPENJPEG_H */
