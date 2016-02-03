 /*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2006-2007, Parvatha Elangovan
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

#if defined(OPJ_STATIC) || !defined(_WIN32)
#if __GNUC__ >= 4
#define OPJ_API    __attribute__ ((visibility ("default")))
#else
#define OPJ_API
#endif
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
#if defined(OPJ_EXPORTS) || defined(DLL_EXPORT)
#define OPJ_API __declspec(dllexport)
#else
#define OPJ_API __declspec(dllimport)
#endif /* OPJ_EXPORTS */
#endif /* !OPJ_STATIC || !_WIN32 */

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

#define OPJ_PATH_LEN 4096 /**< Maximum allowed size for filenames */

#define J2K_MAXRLVLS 33					/**< Number of maximum resolution level authorized */
#define J2K_MAXBANDS (3*J2K_MAXRLVLS-2)	/**< Number of maximum sub-band linked to number of resolution level */

/* UniPG>> */
#define JPWL_MAX_NO_TILESPECS	16 /**< Maximum number of tile parts expected by JPWL: increase at your will */
#define JPWL_MAX_NO_PACKSPECS	16 /**< Maximum number of packet parts expected by JPWL: increase at your will */
#define JPWL_MAX_NO_MARKERS	512 /**< Maximum number of JPWL markers: increase at your will */
#define JPWL_PRIVATEINDEX_NAME "jpwl_index_privatefilename" /**< index file name used when JPWL is on */
#define JPWL_EXPECTED_COMPONENTS 3 /**< Expect this number of components, so you'll find better the first EPB */
#define JPWL_MAXIMUM_TILES 8192 /**< Expect this maximum number of tiles, to avoid some crashes */
#define JPWL_MAXIMUM_HAMMING 2 /**< Expect this maximum number of bit errors in marker id's */
#define JPWL_MAXIMUM_EPB_ROOM 65450 /**< Expect this maximum number of bytes for composition of EPBs */
/* <<UniPG */

/* 
==========================================================
   enum definitions
==========================================================
*/
/** 
Rsiz Capabilities
*/
typedef enum RSIZ_CAPABILITIES {
	STD_RSIZ = 0,		/** Standard JPEG2000 profile*/
	CINEMA2K = 3,		/** Profile name for a 2K image*/
	CINEMA4K = 4		/** Profile name for a 4K image*/
} OPJ_RSIZ_CAPABILITIES;

/** 
Digital cinema operation mode 
*/
typedef enum CINEMA_MODE {
	OFF = 0,					/** Not Digital Cinema*/
	CINEMA2K_24 = 1,	/** 2K Digital Cinema at 24 fps*/
	CINEMA2K_48 = 2,	/** 2K Digital Cinema at 48 fps*/
	CINEMA4K_24 = 3		/** 4K Digital Cinema at 24 fps*/
}OPJ_CINEMA_MODE;

/** 
Progression order 
*/
typedef enum PROG_ORDER {
	PROG_UNKNOWN = -1,	/**< place-holder */
	LRCP = 0,		/**< layer-resolution-component-precinct order */
	RLCP = 1,		/**< resolution-layer-component-precinct order */
	RPCL = 2,		/**< resolution-precinct-component-layer order */
	PCRL = 3,		/**< precinct-component-resolution-layer order */
	CPRL = 4		/**< component-precinct-resolution-layer order */
} OPJ_PROG_ORDER;

/**
Supported image color spaces
*/
typedef enum COLOR_SPACE {
	CLRSPC_UNKNOWN = -1,	/**< not supported by the library */
	CLRSPC_UNSPECIFIED = 0, /**< not specified in the codestream */ 
	CLRSPC_SRGB = 1,		/**< sRGB */
	CLRSPC_GRAY = 2,		/**< grayscale */
	CLRSPC_SYCC = 3			/**< YUV */
} OPJ_COLOR_SPACE;

/**
Supported codec
*/
typedef enum CODEC_FORMAT {
	CODEC_UNKNOWN = -1,	/**< place-holder */
	CODEC_J2K = 0,		/**< JPEG-2000 codestream : read/write */
	CODEC_JPT = 1,		/**< JPT-stream (JPEG 2000, JPIP) : read only */
	CODEC_JP2 = 2		/**< JPEG-2000 file format : read/write */
} OPJ_CODEC_FORMAT;

/** 
Limit decoding to certain portions of the codestream. 
*/
typedef enum LIMIT_DECODING {
	NO_LIMITATION = 0,				  /**< No limitation for the decoding. The entire codestream will de decoded */
	LIMIT_TO_MAIN_HEADER = 1,		/**< The decoding is limited to the Main Header */
	DECODE_ALL_BUT_PACKETS = 2	/**< Decode everything except the JPEG 2000 packets */
} OPJ_LIMIT_DECODING;

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
	/** Resolution num start, Component num start, given by POC */
	int resno0, compno0;
	/** Layer num end,Resolution num end, Component num end, given by POC */
	int layno1, resno1, compno1;
	/** Layer num start,Precinct num start, Precinct num end */
	int layno0, precno0, precno1;
	/** Progression order enum*/
	OPJ_PROG_ORDER prg1,prg;
	/** Progression order string*/
	char progorder[5];
	/** Tile number */
	int tile;
	/** Start and end values for Tile width and height*/
	int tx0,tx1,ty0,ty1;
	/** Start value, initialised in pi_initialise_encode*/
	int layS, resS, compS, prcS;
	/** End value, initialised in pi_initialise_encode */
	int layE, resE, compE, prcE;
	/** Start and end values of Tile width and height, initialised in pi_initialise_encode*/
	int txS,txE,tyS,tyE,dx,dy;
	/** Temporary values for Tile parts, initialised in pi_create_encode */
	int lay_t, res_t, comp_t, prc_t,tx0_t,ty0_t;
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
	/** XTsiz */
	int cp_tdx;
	/** YTsiz */
	int cp_tdy;
	/** allocation by rate/distortion */
	int cp_disto_alloc;
	/** allocation by fixed layer */
	int cp_fixed_alloc;
	/** add fixed_quality */
	int cp_fixed_quality;
	/** fixed layer */
	int *cp_matrice;
	/** comment for coding */
	char *cp_comment;
	/** csty : coding style */
	int csty;
	/** progression order (default LRCP) */
	OPJ_PROG_ORDER prog_order;
	/** progression order changes */
	opj_poc_t POC[32];
	/** number of progression order changes (POC), default to 0 */
	int numpocs;
	/** number of layers */
	int tcp_numlayers;
	/** rates of layers */
	float tcp_rates[100];
	/** different psnr for successive layers */
	float tcp_distoratio[100];
	/** number of resolutions */
	int numresolution;
	/** initial code block width, default to 64 */
 	int cblockw_init;
	/** initial code block height, default to 64 */
	int cblockh_init;
	/** mode switch (cblk_style) */
	int mode;
	/** 1 : use the irreversible DWT 9-7, 0 : use lossless compression (default) */
	int irreversible;
	/** region of interest: affected component in [0..3], -1 means no ROI */
	int roi_compno;
	/** region of interest: upshift value */
	int roi_shift;
	/* number of precinct size specifications */
	int res_spec;
	/** initial precinct width */
	int prcw_init[J2K_MAXRLVLS];
	/** initial precinct height */
	int prch_init[J2K_MAXRLVLS];

	/**@name command line encoder parameters (not used inside the library) */
	/*@{*/
	/** input file name */
	char infile[OPJ_PATH_LEN];
	/** output file name */
	char outfile[OPJ_PATH_LEN];
	/** DEPRECATED. Index generation is now handeld with the opj_encode_with_info() function. Set to NULL */
	int index_on;
	/** DEPRECATED. Index generation is now handeld with the opj_encode_with_info() function. Set to NULL */
	char index[OPJ_PATH_LEN];
	/** subimage encoding: origin image offset in x direction */
	int image_offset_x0;
	/** subimage encoding: origin image offset in y direction */
	int image_offset_y0;
	/** subsampling value for dx */
	int subsampling_dx;
	/** subsampling value for dy */
	int subsampling_dy;
	/** input file format 0: PGX, 1: PxM, 2: BMP 3:TIF*/
	int decod_format;
	/** output file format 0: J2K, 1: JP2, 2: JPT */
	int cod_format;
	/*@}*/

/* UniPG>> */
	/**@name JPWL encoding parameters */
	/*@{*/
	/** enables writing of EPC in MH, thus activating JPWL */
	bool jpwl_epc_on;
	/** error protection method for MH (0,1,16,32,37-128) */
	int jpwl_hprot_MH;
	/** tile number of header protection specification (>=0) */
	int jpwl_hprot_TPH_tileno[JPWL_MAX_NO_TILESPECS];
	/** error protection methods for TPHs (0,1,16,32,37-128) */
	int jpwl_hprot_TPH[JPWL_MAX_NO_TILESPECS];
	/** tile number of packet protection specification (>=0) */
	int jpwl_pprot_tileno[JPWL_MAX_NO_PACKSPECS];
	/** packet number of packet protection specification (>=0) */
	int jpwl_pprot_packno[JPWL_MAX_NO_PACKSPECS];
	/** error protection methods for packets (0,1,16,32,37-128) */
	int jpwl_pprot[JPWL_MAX_NO_PACKSPECS];
	/** enables writing of ESD, (0=no/1/2 bytes) */
	int jpwl_sens_size;
	/** sensitivity addressing size (0=auto/2/4 bytes) */
	int jpwl_sens_addr;
	/** sensitivity range (0-3) */
	int jpwl_sens_range;
	/** sensitivity method for MH (-1=no,0-7) */
	int jpwl_sens_MH;
	/** tile number of sensitivity specification (>=0) */
	int jpwl_sens_TPH_tileno[JPWL_MAX_NO_TILESPECS];
	/** sensitivity methods for TPHs (-1=no,0-7) */
	int jpwl_sens_TPH[JPWL_MAX_NO_TILESPECS];
	/*@}*/
/* <<UniPG */

	/** Digital Cinema compliance 0-not compliant, 1-compliant*/
	OPJ_CINEMA_MODE cp_cinema;
	/** Maximum rate for each component. If == 0, component size limitation is not considered */
	int max_comp_size;
	/** Profile name*/
	OPJ_RSIZ_CAPABILITIES cp_rsiz;
	/** Tile part generation*/
	char tp_on;
	/** Flag for Tile part generation*/
	char tp_flag;
	/** MCT (multiple component transform) */
	char tcp_mct;
} opj_cparameters_t;

/**
Decompression parameters
*/
typedef struct opj_dparameters {
	/** 
	Set the number of highest resolution levels to be discarded. 
	The image resolution is effectively divided by 2 to the power of the number of discarded levels. 
	The reduce factor is limited by the smallest total number of decomposition levels among tiles.
	if != 0, then original dimension divided by 2^(reduce); 
	if == 0 or not used, image is decoded to the full resolution 
	*/
	int cp_reduce;
	/** 
	Set the maximum number of quality layers to decode. 
	If there are less quality layers than the specified number, all the quality layers are decoded.
	if != 0, then only the first "layer" layers are decoded; 
	if == 0 or not used, all the quality layers are decoded 
	*/
	int cp_layer;

	/**@name command line encoder parameters (not used inside the library) */
	/*@{*/
	/** input file name */
	char infile[OPJ_PATH_LEN];
	/** output file name */
	char outfile[OPJ_PATH_LEN];
	/** input file format 0: J2K, 1: JP2, 2: JPT */
	int decod_format;
	/** output file format 0: PGX, 1: PxM, 2: BMP */
	int cod_format;
	/*@}*/

/* UniPG>> */
	/**@name JPWL decoding parameters */
	/*@{*/
	/** activates the JPWL correction capabilities */
	bool jpwl_correct;
	/** expected number of components */
	int jpwl_exp_comps;
	/** maximum number of tiles */
	int jpwl_max_tiles;
	/*@}*/
/* <<UniPG */

	/** 
	Specify whether the decoding should be done on the entire codestream, or be limited to the main header
	Limiting the decoding to the main header makes it possible to extract the characteristics of the codestream
	if == NO_LIMITATION, the entire codestream is decoded; 
	if == LIMIT_TO_MAIN_HEADER, only the main header is decoded; 
	*/
	OPJ_LIMIT_DECODING cp_limit_decoding;

} opj_dparameters_t;

/** Common fields between JPEG-2000 compression and decompression master structs. */

#define opj_common_fields \
	opj_event_mgr_t *event_mgr;	/**< pointer to the event manager */\
	void * client_data;			/**< Available for use by application */\
	bool is_decompressor;		/**< So common code can tell which is which */\
	OPJ_CODEC_FORMAT codec_format;	/**< selected codec */\
	void *j2k_handle;			/**< pointer to the J2K codec */\
	void *jp2_handle;			/**< pointer to the JP2 codec */\
	void *mj2_handle			/**< pointer to the MJ2 codec */
	
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
   image typedef definitions
==========================================================
*/

/**
Defines a single image component
*/
typedef struct opj_image_comp {
	/** XRsiz: horizontal separation of a sample of ith component with respect to the reference grid */
	int dx;
	/** YRsiz: vertical separation of a sample of ith component with respect to the reference grid */
	int dy;
	/** data width */
	int w;
	/** data height */
	int h;
	/** x component offset compared to the whole image */
	int x0;
	/** y component offset compared to the whole image */
	int y0;
	/** precision */
	int prec;
	/** image depth in bits */
	int bpp;
	/** signed (1) / unsigned (0) */
	int sgnd;
	/** number of decoded resolution */
	int resno_decoded;
	/** number of division by 2 of the out image compared to the original size of image */
	int factor;
	/** image component data */
	int *data;
} opj_image_comp_t;

/** 
Defines image data and characteristics
*/
typedef struct opj_image {
	/** XOsiz: horizontal offset from the origin of the reference grid to the left side of the image area */
	int x0;
	/** YOsiz: vertical offset from the origin of the reference grid to the top side of the image area */
	int y0;
	/** Xsiz: width of the reference grid */
	int x1;
	/** Ysiz: height of the reference grid */
	int y1;
	/** number of components in the image */
	int numcomps;
	/** color space: sRGB, Greyscale or YUV */
	OPJ_COLOR_SPACE color_space;
	/** image components */
	opj_image_comp_t *comps;
	/** 'restricted' ICC profile */
	unsigned char *icc_profile_buf;
	/** size of ICC profile */
	int icc_profile_len;
} opj_image_t;

/**
Component parameters structure used by the opj_image_create function
*/
typedef struct opj_image_comptparm {
	/** XRsiz: horizontal separation of a sample of ith component with respect to the reference grid */
	int dx;
	/** YRsiz: vertical separation of a sample of ith component with respect to the reference grid */
	int dy;
	/** data width */
	int w;
	/** data height */
	int h;
	/** x component offset compared to the whole image */
	int x0;
	/** y component offset compared to the whole image */
	int y0;
	/** precision */
	int prec;
	/** image depth in bits */
	int bpp;
	/** signed (1) / unsigned (0) */
	int sgnd;
} opj_image_cmptparm_t;

/* 
==========================================================
   Information on the JPEG 2000 codestream
==========================================================
*/

/**
Index structure : Information concerning a packet inside tile
*/
typedef struct opj_packet_info {
	/** packet start position (including SOP marker if it exists) */
	int start_pos;
	/** end of packet header position (including EPH marker if it exists)*/
	int end_ph_pos;
	/** packet end position */
	int end_pos;
	/** packet distorsion */
	double disto;
} opj_packet_info_t;

/**
Index structure : Information concerning tile-parts
*/
typedef struct opj_tp_info {
	/** start position of tile part */
	int tp_start_pos;
	/** end position of tile part header */
	int tp_end_header;
	/** end position of tile part */
	int tp_end_pos;
	/** start packet of tile part */
	int tp_start_pack;
	/** number of packets of tile part */
	int tp_numpacks;
} opj_tp_info_t;

/**
Index structure : information regarding tiles 
*/
typedef struct opj_tile_info {
	/** value of thresh for each layer by tile cfr. Marcela   */
	double *thresh;
	/** number of tile */
	int tileno;
	/** start position */
	int start_pos;
	/** end position of the header */
	int end_header;
	/** end position */
	int end_pos;
	/** precinct number for each resolution level (width) */
	int pw[33];
	/** precinct number for each resolution level (height) */
	int ph[33];
	/** precinct size (in power of 2), in X for each resolution level */
	int pdx[33];
	/** precinct size (in power of 2), in Y for each resolution level */
	int pdy[33];
	/** information concerning packets inside tile */
	opj_packet_info_t *packet;
	/** add fixed_quality */
	int numpix;
	/** add fixed_quality */
	double distotile;
	/** number of tile parts */
	int num_tps;
	/** information concerning tile parts */
	opj_tp_info_t *tp;
} opj_tile_info_t;

/* UniPG>> */
/**
Marker structure
*/
typedef struct opj_marker_info_t {
	/** marker type */
	unsigned short int type;
	/** position in codestream */
	int pos;
	/** length, marker val included */
	int len;
} opj_marker_info_t;
/* <<UniPG */

/**
Index structure of the codestream
*/
typedef struct opj_codestream_info {
	/** maximum distortion reduction on the whole image (add for Marcela) */
	double D_max;
	/** packet number */
	int packno;
	/** writing the packet in the index with t2_encode_packets */
	int index_write;
	/** image width */
	int image_w;
	/** image height */
	int image_h;
	/** progression order */
	OPJ_PROG_ORDER prog;
	/** tile size in x */
	int tile_x;
	/** tile size in y */
	int tile_y;
	/** */
	int tile_Ox;
	/** */
	int tile_Oy;
	/** number of tiles in X */
	int tw;
	/** number of tiles in Y */
	int th;
	/** component numbers */
	int numcomps;
	/** number of layer */
	int numlayers;
	/** number of decomposition for each component */
	int *numdecompos;
/* UniPG>> */
	/** number of markers */
	int marknum;
	/** list of markers */
	opj_marker_info_t *marker;
	/** actual size of markers array */
	int maxmarknum;
/* <<UniPG */
	/** main header position */
	int main_head_start;
	/** main header position */
	int main_head_end;
	/** codestream's size */
	int codestream_size;
	/** information regarding tiles inside image */
	opj_tile_info_t *tile;
} opj_codestream_info_t;

#ifdef __cplusplus
extern "C" {
#endif


/* 
==========================================================
   openjpeg version
==========================================================
*/

OPJ_API const char * OPJ_CALLCONV opj_version(void);

/* 
==========================================================
   image functions definitions
==========================================================
*/

/**
Create an image
@param numcmpts number of components
@param cmptparms components parameters
@param clrspc image color space
@return returns a new image structure if successful, returns NULL otherwise
*/
OPJ_API opj_image_t* OPJ_CALLCONV opj_image_create(int numcmpts, opj_image_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc);

/**
Deallocate any resources associated with an image
@param image image to be destroyed
*/
OPJ_API void OPJ_CALLCONV opj_image_destroy(opj_image_t *image);

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
Creates a J2K/JPT/JP2 decompression structure
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
Decoding parameters are returned in j2k->cp. 
@param dinfo decompressor handle
@param parameters decompression parameters
*/
OPJ_API void OPJ_CALLCONV opj_setup_decoder(opj_dinfo_t *dinfo, opj_dparameters_t *parameters);
/**
Decode an image from a JPEG-2000 codestream 
@param dinfo decompressor handle
@param cio Input buffer stream
@return Returns a decoded image if successful, returns NULL otherwise
*/
OPJ_API opj_image_t* OPJ_CALLCONV opj_decode(opj_dinfo_t *dinfo, opj_cio_t *cio);

/**
Decode an image from a JPEG-2000 codestream and extract the codestream information
@param dinfo decompressor handle
@param cio Input buffer stream
@param cstr_info Codestream information structure if needed afterwards, NULL otherwise
@return Returns a decoded image if successful, returns NULL otherwise
*/
OPJ_API opj_image_t* OPJ_CALLCONV opj_decode_with_info(opj_dinfo_t *dinfo, opj_cio_t *cio, opj_codestream_info_t *cstr_info);
/**
Creates a J2K/JP2 compression structure
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
<li>No offset of the origin of the image
<li>No offset of the origin of the tiles
<li>Reversible DWT 5-3
</ul>
@param parameters Compression parameters
*/
OPJ_API void OPJ_CALLCONV opj_set_default_encoder_parameters(opj_cparameters_t *parameters);
/**
Setup the encoder parameters using the current image and using user parameters. 
@param cinfo Compressor handle
@param parameters Compression parameters
@param image Input filled image
*/
OPJ_API void OPJ_CALLCONV opj_setup_encoder(opj_cinfo_t *cinfo, opj_cparameters_t *parameters, opj_image_t *image);
/**
Encode an image into a JPEG-2000 codestream
@param cinfo compressor handle
@param cio Output buffer stream
@param image Image to encode
@param index Depreacted -> Set to NULL. To extract index, used opj_encode_wci()
@return Returns true if successful, returns false otherwise
*/
OPJ_API bool OPJ_CALLCONV opj_encode(opj_cinfo_t *cinfo, opj_cio_t *cio, opj_image_t *image, char *index);
/**
Encode an image into a JPEG-2000 codestream and extract the codestream information
@param cinfo compressor handle
@param cio Output buffer stream
@param image Image to encode
@param cstr_info Codestream information structure if needed afterwards, NULL otherwise
@return Returns true if successful, returns false otherwise
*/
OPJ_API bool OPJ_CALLCONV opj_encode_with_info(opj_cinfo_t *cinfo, opj_cio_t *cio, opj_image_t *image, opj_codestream_info_t *cstr_info);
/**
Destroy Codestream information after compression or decompression
@param cstr_info Codestream information structure
*/
OPJ_API void OPJ_CALLCONV opj_destroy_cstr_info(opj_codestream_info_t *cstr_info);

#ifdef __cplusplus
}
#endif

#endif /* OPENJPEG_H */
