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
* Copyright (c) 2006-2007, Parvatha Elangovan
* Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
* Copyright (c) 2010-2011, Kaori Hagihara
* Copyright (c) 2011-2012, Centre National d'Etudes Spatiales (CNES), France
* Copyright (c) 2012, CS Systemes d'Information, France
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

/*
The inline keyword is supported by C99 but not by C90.
Most compilers implement their own version of this keyword ...
*/
#ifndef INLINE
#if defined(_MSC_VER)
#define INLINE __forceinline
#elif defined(__GNUC__)
#define INLINE __inline__
#elif defined(__MWERKS__)
#define INLINE inline
#else
/* add other compilers here ... */
#define INLINE
#endif /* defined(<Compiler>) */
#endif /* INLINE */

/* deprecated attribute */
#ifdef __GNUC__
#define OPJ_DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define OPJ_DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define OPJ_DEPRECATED(func) func
#endif

#if defined(OPJ_STATIC) || !defined(_WIN32)
/* http://gcc.gnu.org/wiki/Visibility */
#   if !defined(_WIN32) && __GNUC__ >= 4
#       if defined(OPJ_STATIC) /* static library uses "hidden" */
#           define OPJ_API    __attribute__ ((visibility ("hidden")))
#       else
#           define OPJ_API    __attribute__ ((visibility ("default")))
#       endif
#       define OPJ_LOCAL  __attribute__ ((visibility ("hidden")))
#   else
#       define OPJ_API
#       define OPJ_LOCAL
#   endif
#   define OPJ_CALLCONV
#else
#   define OPJ_CALLCONV __stdcall
/*
The following ifdef block is the standard way of creating macros which make exporting
from a DLL simpler. All files within this DLL are compiled with the OPJ_EXPORTS
symbol defined on the command line. this symbol should not be defined on any project
that uses this DLL. This way any other project whose source files include this file see
OPJ_API functions as being imported from a DLL, whereas this DLL sees symbols
defined with this macro as being exported.
*/
#   if defined(OPJ_EXPORTS) || defined(DLL_EXPORT)
#       define OPJ_API __declspec(dllexport)
#   else
#       define OPJ_API __declspec(dllimport)
#   endif /* OPJ_EXPORTS */
#endif /* !OPJ_STATIC || !_WIN32 */

typedef int OPJ_BOOL;
#define OPJ_TRUE 1
#define OPJ_FALSE 0

typedef char          OPJ_CHAR;
typedef float         OPJ_FLOAT32;
typedef double        OPJ_FLOAT64;
typedef unsigned char OPJ_BYTE;

#include "opj_stdint.h"

typedef int8_t   OPJ_INT8;
typedef uint8_t  OPJ_UINT8;
typedef int16_t  OPJ_INT16;
typedef uint16_t OPJ_UINT16;
typedef int32_t  OPJ_INT32;
typedef uint32_t OPJ_UINT32;
typedef int64_t  OPJ_INT64;
typedef uint64_t OPJ_UINT64;

typedef int64_t  OPJ_OFF_T; /* 64-bit file offset type */

#include <stdio.h>
typedef size_t   OPJ_SIZE_T;

/* Avoid compile-time warning because parameter is not used */
#define OPJ_ARG_NOT_USED(x) (void)(x)

/*
==========================================================
   Useful constant definitions
==========================================================
*/

#define OPJ_PATH_LEN 4096 /**< Maximum allowed size for filenames */

#define OPJ_J2K_MAXRLVLS 33                 /**< Number of maximum resolution level authorized */
#define OPJ_J2K_MAXBANDS (3*OPJ_J2K_MAXRLVLS-2) /**< Number of maximum sub-band linked to number of resolution level */

#define OPJ_J2K_DEFAULT_NB_SEGS             10
#define OPJ_J2K_STREAM_CHUNK_SIZE           0x100000 /** 1 mega by default */
#define OPJ_J2K_DEFAULT_HEADER_SIZE         1000
#define OPJ_J2K_MCC_DEFAULT_NB_RECORDS      10
#define OPJ_J2K_MCT_DEFAULT_NB_RECORDS      10

/* UniPG>> */ /* NOT YET USED IN THE V2 VERSION OF OPENJPEG */
#define JPWL_MAX_NO_TILESPECS   16 /**< Maximum number of tile parts expected by JPWL: increase at your will */
#define JPWL_MAX_NO_PACKSPECS   16 /**< Maximum number of packet parts expected by JPWL: increase at your will */
#define JPWL_MAX_NO_MARKERS 512 /**< Maximum number of JPWL markers: increase at your will */
#define JPWL_PRIVATEINDEX_NAME "jpwl_index_privatefilename" /**< index file name used when JPWL is on */
#define JPWL_EXPECTED_COMPONENTS 3 /**< Expect this number of components, so you'll find better the first EPB */
#define JPWL_MAXIMUM_TILES 8192 /**< Expect this maximum number of tiles, to avoid some crashes */
#define JPWL_MAXIMUM_HAMMING 2 /**< Expect this maximum number of bit errors in marker id's */
#define JPWL_MAXIMUM_EPB_ROOM 65450 /**< Expect this maximum number of bytes for composition of EPBs */
/* <<UniPG */

/**
 * EXPERIMENTAL FOR THE MOMENT
 * Supported options about file information used only in j2k_dump
*/
#define OPJ_IMG_INFO        1   /**< Basic image information provided to the user */
#define OPJ_J2K_MH_INFO     2   /**< Codestream information based only on the main header */
#define OPJ_J2K_TH_INFO     4   /**< Tile information based on the current tile header */
#define OPJ_J2K_TCH_INFO    8   /**< Tile/Component information of all tiles */
#define OPJ_J2K_MH_IND      16  /**< Codestream index based only on the main header */
#define OPJ_J2K_TH_IND      32  /**< Tile index based on the current tile */
/*FIXME #define OPJ_J2K_CSTR_IND    48*/    /**<  */
#define OPJ_JP2_INFO        128 /**< JP2 file information */
#define OPJ_JP2_IND         256 /**< JP2 file index */

/**
 * JPEG 2000 Profiles, see Table A.10 from 15444-1 (updated in various AMD)
 * These values help choosing the RSIZ value for the J2K codestream.
 * The RSIZ value triggers various encoding options, as detailed in Table A.10.
 * If OPJ_PROFILE_PART2 is chosen, it has to be combined with one or more extensions
 * described hereunder.
 *   Example: rsiz = OPJ_PROFILE_PART2 | OPJ_EXTENSION_MCT;
 * For broadcast profiles, the OPJ_PROFILE value has to be combined with the targeted
 * mainlevel (3-0 LSB, value between 0 and 11):
 *   Example: rsiz = OPJ_PROFILE_BC_MULTI | 0x0005; (here mainlevel 5)
 * For IMF profiles, the OPJ_PROFILE value has to be combined with the targeted mainlevel
 * (3-0 LSB, value between 0 and 11) and sublevel (7-4 LSB, value between 0 and 9):
 *   Example: rsiz = OPJ_PROFILE_IMF_2K | 0x0040 | 0x0005; (here main 5 and sublevel 4)
 * */
#define OPJ_PROFILE_NONE        0x0000 /** no profile, conform to 15444-1 */
#define OPJ_PROFILE_0           0x0001 /** Profile 0 as described in 15444-1,Table A.45 */
#define OPJ_PROFILE_1           0x0002 /** Profile 1 as described in 15444-1,Table A.45 */
#define OPJ_PROFILE_PART2       0x8000 /** At least 1 extension defined in 15444-2 (Part-2) */
#define OPJ_PROFILE_CINEMA_2K   0x0003 /** 2K cinema profile defined in 15444-1 AMD1 */
#define OPJ_PROFILE_CINEMA_4K   0x0004 /** 4K cinema profile defined in 15444-1 AMD1 */
#define OPJ_PROFILE_CINEMA_S2K  0x0005 /** Scalable 2K cinema profile defined in 15444-1 AMD2 */
#define OPJ_PROFILE_CINEMA_S4K  0x0006 /** Scalable 4K cinema profile defined in 15444-1 AMD2 */
#define OPJ_PROFILE_CINEMA_LTS  0x0007 /** Long term storage cinema profile defined in 15444-1 AMD2 */
#define OPJ_PROFILE_BC_SINGLE   0x0100 /** Single Tile Broadcast profile defined in 15444-1 AMD3 */
#define OPJ_PROFILE_BC_MULTI    0x0200 /** Multi Tile Broadcast profile defined in 15444-1 AMD3 */
#define OPJ_PROFILE_BC_MULTI_R  0x0300 /** Multi Tile Reversible Broadcast profile defined in 15444-1 AMD3 */
#define OPJ_PROFILE_IMF_2K      0x0400 /** 2K Single Tile Lossy IMF profile defined in 15444-1 AMD 8 */
#define OPJ_PROFILE_IMF_4K      0x0500 /** 4K Single Tile Lossy IMF profile defined in 15444-1 AMD 8 */
#define OPJ_PROFILE_IMF_8K      0x0600 /** 8K Single Tile Lossy IMF profile defined in 15444-1 AMD 8 */
#define OPJ_PROFILE_IMF_2K_R    0x0700 /** 2K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8 */
#define OPJ_PROFILE_IMF_4K_R    0x0800 /** 4K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8 */
#define OPJ_PROFILE_IMF_8K_R    0x0900 /** 8K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8 */

/**
 * JPEG 2000 Part-2 extensions
 * */
#define OPJ_EXTENSION_NONE      0x0000 /** No Part-2 extension */
#define OPJ_EXTENSION_MCT       0x0100  /** Custom MCT support */

/**
 * JPEG 2000 profile macros
 * */
#define OPJ_IS_CINEMA(v)     (((v) >= OPJ_PROFILE_CINEMA_2K)&&((v) <= OPJ_PROFILE_CINEMA_S4K))
#define OPJ_IS_STORAGE(v)    ((v) == OPJ_PROFILE_CINEMA_LTS)
#define OPJ_IS_BROADCAST(v)  (((v) >= OPJ_PROFILE_BC_SINGLE)&&((v) <= ((OPJ_PROFILE_BC_MULTI_R) | (0x000b))))
#define OPJ_IS_IMF(v)        (((v) >= OPJ_PROFILE_IMF_2K)&&((v) <= ((OPJ_PROFILE_IMF_8K_R) | (0x009b))))
#define OPJ_IS_PART2(v)      ((v) & OPJ_PROFILE_PART2)

#define OPJ_GET_IMF_PROFILE(v)   ((v) & 0xff00)      /** Extract IMF profile without mainlevel/sublevel */
#define OPJ_GET_IMF_MAINLEVEL(v) ((v) & 0xf)         /** Extract IMF main level */
#define OPJ_GET_IMF_SUBLEVEL(v)  (((v) >> 4) & 0xf)  /** Extract IMF sub level */

#define OPJ_IMF_MAINLEVEL_MAX    11   /** Maximum main level */

/** Max. Components Sampling Rate (MSamples/sec) per IMF main level */
#define OPJ_IMF_MAINLEVEL_1_MSAMPLESEC   65      /** MSamples/sec for IMF main level 1 */
#define OPJ_IMF_MAINLEVEL_2_MSAMPLESEC   130     /** MSamples/sec for IMF main level 2 */
#define OPJ_IMF_MAINLEVEL_3_MSAMPLESEC   195     /** MSamples/sec for IMF main level 3 */
#define OPJ_IMF_MAINLEVEL_4_MSAMPLESEC   260     /** MSamples/sec for IMF main level 4 */
#define OPJ_IMF_MAINLEVEL_5_MSAMPLESEC   520     /** MSamples/sec for IMF main level 5 */
#define OPJ_IMF_MAINLEVEL_6_MSAMPLESEC   1200    /** MSamples/sec for IMF main level 6 */
#define OPJ_IMF_MAINLEVEL_7_MSAMPLESEC   2400    /** MSamples/sec for IMF main level 7 */
#define OPJ_IMF_MAINLEVEL_8_MSAMPLESEC   4800    /** MSamples/sec for IMF main level 8 */
#define OPJ_IMF_MAINLEVEL_9_MSAMPLESEC   9600    /** MSamples/sec for IMF main level 9 */
#define OPJ_IMF_MAINLEVEL_10_MSAMPLESEC  19200   /** MSamples/sec for IMF main level 10 */
#define OPJ_IMF_MAINLEVEL_11_MSAMPLESEC  38400   /** MSamples/sec for IMF main level 11 */

/** Max. compressed Bit Rate (Mbits/s) per IMF sub level */
#define OPJ_IMF_SUBLEVEL_1_MBITSSEC      200     /** Mbits/s for IMF sub level 1 */
#define OPJ_IMF_SUBLEVEL_2_MBITSSEC      400     /** Mbits/s for IMF sub level 2 */
#define OPJ_IMF_SUBLEVEL_3_MBITSSEC      800     /** Mbits/s for IMF sub level 3 */
#define OPJ_IMF_SUBLEVEL_4_MBITSSEC     1600     /** Mbits/s for IMF sub level 4 */
#define OPJ_IMF_SUBLEVEL_5_MBITSSEC     3200     /** Mbits/s for IMF sub level 5 */
#define OPJ_IMF_SUBLEVEL_6_MBITSSEC     6400     /** Mbits/s for IMF sub level 6 */
#define OPJ_IMF_SUBLEVEL_7_MBITSSEC    12800     /** Mbits/s for IMF sub level 7 */
#define OPJ_IMF_SUBLEVEL_8_MBITSSEC    25600     /** Mbits/s for IMF sub level 8 */
#define OPJ_IMF_SUBLEVEL_9_MBITSSEC    51200     /** Mbits/s for IMF sub level 9 */

/**
 * JPEG 2000 codestream and component size limits in cinema profiles
 * */
#define OPJ_CINEMA_24_CS     1302083    /** Maximum codestream length for 24fps */
#define OPJ_CINEMA_48_CS     651041     /** Maximum codestream length for 48fps */
#define OPJ_CINEMA_24_COMP   1041666    /** Maximum size per color component for 2K & 4K @ 24fps */
#define OPJ_CINEMA_48_COMP   520833     /** Maximum size per color component for 2K @ 48fps */

/*
==========================================================
   enum definitions
==========================================================
*/

/**
 * DEPRECATED: use RSIZ, OPJ_PROFILE_* and OPJ_EXTENSION_* instead
 * Rsiz Capabilities
 * */
typedef enum RSIZ_CAPABILITIES {
    OPJ_STD_RSIZ = 0,       /** Standard JPEG2000 profile*/
    OPJ_CINEMA2K = 3,       /** Profile name for a 2K image*/
    OPJ_CINEMA4K = 4,       /** Profile name for a 4K image*/
    OPJ_MCT = 0x8100
} OPJ_RSIZ_CAPABILITIES;

/**
 * DEPRECATED: use RSIZ, OPJ_PROFILE_* and OPJ_EXTENSION_* instead
 * Digital cinema operation mode
 * */
typedef enum CINEMA_MODE {
    OPJ_OFF = 0,            /** Not Digital Cinema*/
    OPJ_CINEMA2K_24 = 1,    /** 2K Digital Cinema at 24 fps*/
    OPJ_CINEMA2K_48 = 2,    /** 2K Digital Cinema at 48 fps*/
    OPJ_CINEMA4K_24 = 3     /** 4K Digital Cinema at 24 fps*/
} OPJ_CINEMA_MODE;

/**
 * Progression order
 * */
typedef enum PROG_ORDER {
    OPJ_PROG_UNKNOWN = -1,  /**< place-holder */
    OPJ_LRCP = 0,           /**< layer-resolution-component-precinct order */
    OPJ_RLCP = 1,           /**< resolution-layer-component-precinct order */
    OPJ_RPCL = 2,           /**< resolution-precinct-component-layer order */
    OPJ_PCRL = 3,           /**< precinct-component-resolution-layer order */
    OPJ_CPRL = 4            /**< component-precinct-resolution-layer order */
} OPJ_PROG_ORDER;

/**
 * Supported image color spaces
*/
typedef enum COLOR_SPACE {
    OPJ_CLRSPC_UNKNOWN = -1,    /**< not supported by the library */
    OPJ_CLRSPC_UNSPECIFIED = 0, /**< not specified in the codestream */
    OPJ_CLRSPC_SRGB = 1,        /**< sRGB */
    OPJ_CLRSPC_GRAY = 2,        /**< grayscale */
    OPJ_CLRSPC_SYCC = 3,        /**< YUV */
    OPJ_CLRSPC_EYCC = 4,        /**< e-YCC */
    OPJ_CLRSPC_CMYK = 5         /**< CMYK */
} OPJ_COLOR_SPACE;

/**
 * Supported codec
*/
typedef enum CODEC_FORMAT {
    OPJ_CODEC_UNKNOWN = -1, /**< place-holder */
    OPJ_CODEC_J2K  = 0,     /**< JPEG-2000 codestream : read/write */
    OPJ_CODEC_JPT  = 1,     /**< JPT-stream (JPEG 2000, JPIP) : read only */
    OPJ_CODEC_JP2  = 2,     /**< JP2 file format : read/write */
    OPJ_CODEC_JPP  = 3,     /**< JPP-stream (JPEG 2000, JPIP) : to be coded */
    OPJ_CODEC_JPX  = 4      /**< JPX file format (JPEG 2000 Part-2) : to be coded */
} OPJ_CODEC_FORMAT;


/*
==========================================================
   event manager typedef definitions
==========================================================
*/

/**
 * Callback function prototype for events
 * @param msg               Event message
 * @param client_data       Client object where will be return the event message
 * */
typedef void (*opj_msg_callback)(const char *msg, void *client_data);

/*
==========================================================
   codec typedef definitions
==========================================================
*/

#ifndef OPJ_UINT32_SEMANTICALLY_BUT_INT32
#define OPJ_UINT32_SEMANTICALLY_BUT_INT32 OPJ_INT32
#endif

/**
 * Progression order changes
 *
 */
typedef struct opj_poc {
    /** Resolution num start, Component num start, given by POC */
    OPJ_UINT32 resno0, compno0;
    /** Layer num end,Resolution num end, Component num end, given by POC */
    OPJ_UINT32 layno1, resno1, compno1;
    /** Layer num start,Precinct num start, Precinct num end */
    OPJ_UINT32 layno0, precno0, precno1;
    /** Progression order enum*/
    OPJ_PROG_ORDER prg1, prg;
    /** Progression order string*/
    OPJ_CHAR progorder[5];
    /** Tile number (starting at 1) */
    OPJ_UINT32 tile;
    /** Start and end values for Tile width and height*/
    OPJ_UINT32_SEMANTICALLY_BUT_INT32 tx0, tx1, ty0, ty1;
    /** Start value, initialised in pi_initialise_encode*/
    OPJ_UINT32 layS, resS, compS, prcS;
    /** End value, initialised in pi_initialise_encode */
    OPJ_UINT32 layE, resE, compE, prcE;
    /** Start and end values of Tile width and height, initialised in pi_initialise_encode*/
    OPJ_UINT32 txS, txE, tyS, tyE, dx, dy;
    /** Temporary values for Tile parts, initialised in pi_create_encode */
    OPJ_UINT32 lay_t, res_t, comp_t, prc_t, tx0_t, ty0_t;
} opj_poc_t;

/**
 * Compression parameters
 * */
typedef struct opj_cparameters {
    /** size of tile: tile_size_on = false (not in argument) or = true (in argument) */
    OPJ_BOOL tile_size_on;
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
    /** progression order (default OPJ_LRCP) */
    OPJ_PROG_ORDER prog_order;
    /** progression order changes */
    opj_poc_t POC[32];
    /** number of progression order changes (POC), default to 0 */
    OPJ_UINT32 numpocs;
    /** number of layers */
    int tcp_numlayers;
    /** rates of layers - might be subsequently limited by the max_cs_size field.
     * Should be decreasing. 1 can be
     * used as last value to indicate the last layer is lossless. */
    float tcp_rates[100];
    /** different psnr for successive layers. Should be increasing. 0 can be
     * used as last value to indicate the last layer is lossless. */
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
    int prcw_init[OPJ_J2K_MAXRLVLS];
    /** initial precinct height */
    int prch_init[OPJ_J2K_MAXRLVLS];

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

    /* UniPG>> */ /* NOT YET USED IN THE V2 VERSION OF OPENJPEG */
    /**@name JPWL encoding parameters */
    /*@{*/
    /** enables writing of EPC in MH, thus activating JPWL */
    OPJ_BOOL jpwl_epc_on;
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

    /**
     * DEPRECATED: use RSIZ, OPJ_PROFILE_* and MAX_COMP_SIZE instead
     * Digital Cinema compliance 0-not compliant, 1-compliant
     * */
    OPJ_CINEMA_MODE cp_cinema;
    /**
     * Maximum size (in bytes) for each component.
     * If == 0, component size limitation is not considered
     * */
    int max_comp_size;
    /**
     * DEPRECATED: use RSIZ, OPJ_PROFILE_* and OPJ_EXTENSION_* instead
     * Profile name
     * */
    OPJ_RSIZ_CAPABILITIES cp_rsiz;
    /** Tile part generation*/
    char tp_on;
    /** Flag for Tile part generation*/
    char tp_flag;
    /** MCT (multiple component transform) */
    char tcp_mct;
    /** Enable JPIP indexing*/
    OPJ_BOOL jpip_on;
    /** Naive implementation of MCT restricted to a single reversible array based
        encoding without offset concerning all the components. */
    void * mct_data;
    /**
     * Maximum size (in bytes) for the whole codestream.
     * If == 0, codestream size limitation is not considered
     * If it does not comply with tcp_rates, max_cs_size prevails
     * and a warning is issued.
     * */
    int max_cs_size;
    /** RSIZ value
        To be used to combine OPJ_PROFILE_*, OPJ_EXTENSION_* and (sub)levels values. */
    OPJ_UINT16 rsiz;
} opj_cparameters_t;

#define OPJ_DPARAMETERS_IGNORE_PCLR_CMAP_CDEF_FLAG  0x0001
#define OPJ_DPARAMETERS_DUMP_FLAG 0x0002

/**
 * Decompression parameters
 * */
typedef struct opj_dparameters {
    /**
    Set the number of highest resolution levels to be discarded.
    The image resolution is effectively divided by 2 to the power of the number of discarded levels.
    The reduce factor is limited by the smallest total number of decomposition levels among tiles.
    if != 0, then original dimension divided by 2^(reduce);
    if == 0 or not used, image is decoded to the full resolution
    */
    OPJ_UINT32 cp_reduce;
    /**
    Set the maximum number of quality layers to decode.
    If there are less quality layers than the specified number, all the quality layers are decoded.
    if != 0, then only the first "layer" layers are decoded;
    if == 0 or not used, all the quality layers are decoded
    */
    OPJ_UINT32 cp_layer;

    /**@name command line decoder parameters (not used inside the library) */
    /*@{*/
    /** input file name */
    char infile[OPJ_PATH_LEN];
    /** output file name */
    char outfile[OPJ_PATH_LEN];
    /** input file format 0: J2K, 1: JP2, 2: JPT */
    int decod_format;
    /** output file format 0: PGX, 1: PxM, 2: BMP */
    int cod_format;

    /** Decoding area left boundary */
    OPJ_UINT32 DA_x0;
    /** Decoding area right boundary */
    OPJ_UINT32 DA_x1;
    /** Decoding area up boundary */
    OPJ_UINT32 DA_y0;
    /** Decoding area bottom boundary */
    OPJ_UINT32 DA_y1;
    /** Verbose mode */
    OPJ_BOOL m_verbose;

    /** tile number of the decoded tile */
    OPJ_UINT32 tile_index;
    /** Nb of tile to decode */
    OPJ_UINT32 nb_tile_to_decode;

    /*@}*/

    /* UniPG>> */ /* NOT YET USED IN THE V2 VERSION OF OPENJPEG */
    /**@name JPWL decoding parameters */
    /*@{*/
    /** activates the JPWL correction capabilities */
    OPJ_BOOL jpwl_correct;
    /** expected number of components */
    int jpwl_exp_comps;
    /** maximum number of tiles */
    int jpwl_max_tiles;
    /*@}*/
    /* <<UniPG */

    unsigned int flags;

} opj_dparameters_t;


/**
 * JPEG2000 codec V2.
 * */
typedef void * opj_codec_t;

/*
==========================================================
   I/O stream typedef definitions
==========================================================
*/

/**
 * Stream open flags.
 * */
/** The stream was opened for reading. */
#define OPJ_STREAM_READ OPJ_TRUE
/** The stream was opened for writing. */
#define OPJ_STREAM_WRITE OPJ_FALSE

/*
 * Callback function prototype for read function
 */
typedef OPJ_SIZE_T(* opj_stream_read_fn)(void * p_buffer, OPJ_SIZE_T p_nb_bytes,
        void * p_user_data) ;

/*
 * Callback function prototype for write function
 */
typedef OPJ_SIZE_T(* opj_stream_write_fn)(void * p_buffer,
        OPJ_SIZE_T p_nb_bytes, void * p_user_data) ;

/*
 * Callback function prototype for skip function
 */
typedef OPJ_OFF_T(* opj_stream_skip_fn)(OPJ_OFF_T p_nb_bytes,
                                        void * p_user_data) ;

/*
 * Callback function prototype for seek function
 */
typedef OPJ_BOOL(* opj_stream_seek_fn)(OPJ_OFF_T p_nb_bytes,
                                       void * p_user_data) ;

/*
 * Callback function prototype for free user data function
 */
typedef void (* opj_stream_free_user_data_fn)(void * p_user_data) ;

/*
 * JPEG2000 Stream.
 */
typedef void * opj_stream_t;

/*
==========================================================
   image typedef definitions
==========================================================
*/

/**
 * Defines a single image component
 * */
typedef struct opj_image_comp {
    /** XRsiz: horizontal separation of a sample of ith component with respect to the reference grid */
    OPJ_UINT32 dx;
    /** YRsiz: vertical separation of a sample of ith component with respect to the reference grid */
    OPJ_UINT32 dy;
    /** data width */
    OPJ_UINT32 w;
    /** data height */
    OPJ_UINT32 h;
    /** x component offset compared to the whole image */
    OPJ_UINT32 x0;
    /** y component offset compared to the whole image */
    OPJ_UINT32 y0;
    /** precision */
    OPJ_UINT32 prec;
    /** image depth in bits */
    OPJ_UINT32 bpp;
    /** signed (1) / unsigned (0) */
    OPJ_UINT32 sgnd;
    /** number of decoded resolution */
    OPJ_UINT32 resno_decoded;
    /** number of division by 2 of the out image compared to the original size of image */
    OPJ_UINT32 factor;
    /** image component data */
    OPJ_INT32 *data;
    /** alpha channel */
    OPJ_UINT16 alpha;
} opj_image_comp_t;

/**
 * Defines image data and characteristics
 * */
typedef struct opj_image {
    /** XOsiz: horizontal offset from the origin of the reference grid to the left side of the image area */
    OPJ_UINT32 x0;
    /** YOsiz: vertical offset from the origin of the reference grid to the top side of the image area */
    OPJ_UINT32 y0;
    /** Xsiz: width of the reference grid */
    OPJ_UINT32 x1;
    /** Ysiz: height of the reference grid */
    OPJ_UINT32 y1;
    /** number of components in the image */
    OPJ_UINT32 numcomps;
    /** color space: sRGB, Greyscale or YUV */
    OPJ_COLOR_SPACE color_space;
    /** image components */
    opj_image_comp_t *comps;
    /** 'restricted' ICC profile */
    OPJ_BYTE *icc_profile_buf;
    /** size of ICC profile */
    OPJ_UINT32 icc_profile_len;
} opj_image_t;


/**
 * Component parameters structure used by the opj_image_create function
 * */
typedef struct opj_image_comptparm {
    /** XRsiz: horizontal separation of a sample of ith component with respect to the reference grid */
    OPJ_UINT32 dx;
    /** YRsiz: vertical separation of a sample of ith component with respect to the reference grid */
    OPJ_UINT32 dy;
    /** data width */
    OPJ_UINT32 w;
    /** data height */
    OPJ_UINT32 h;
    /** x component offset compared to the whole image */
    OPJ_UINT32 x0;
    /** y component offset compared to the whole image */
    OPJ_UINT32 y0;
    /** precision */
    OPJ_UINT32 prec;
    /** image depth in bits */
    OPJ_UINT32 bpp;
    /** signed (1) / unsigned (0) */
    OPJ_UINT32 sgnd;
} opj_image_cmptparm_t;


/*
==========================================================
   Information on the JPEG 2000 codestream
==========================================================
*/
/* QUITE EXPERIMENTAL FOR THE MOMENT */

/**
 * Index structure : Information concerning a packet inside tile
 * */
typedef struct opj_packet_info {
    /** packet start position (including SOP marker if it exists) */
    OPJ_OFF_T start_pos;
    /** end of packet header position (including EPH marker if it exists)*/
    OPJ_OFF_T end_ph_pos;
    /** packet end position */
    OPJ_OFF_T end_pos;
    /** packet distorsion */
    double disto;
} opj_packet_info_t;


/* UniPG>> */
/**
 * Marker structure
 * */
typedef struct opj_marker_info {
    /** marker type */
    unsigned short int type;
    /** position in codestream */
    OPJ_OFF_T pos;
    /** length, marker val included */
    int len;
} opj_marker_info_t;
/* <<UniPG */

/**
 * Index structure : Information concerning tile-parts
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
 * Index structure : information regarding tiles
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
    /** number of markers */
    int marknum;
    /** list of markers */
    opj_marker_info_t *marker;
    /** actual size of markers array */
    int maxmarknum;
    /** number of tile parts */
    int num_tps;
    /** information concerning tile parts */
    opj_tp_info_t *tp;
} opj_tile_info_t;

/**
 * Index structure of the codestream
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

/* <----------------------------------------------------------- */
/* new output management of the codestream information and index */

/**
 * Tile-component coding parameters information
 */
typedef struct opj_tccp_info {
    /** component index */
    OPJ_UINT32 compno;
    /** coding style */
    OPJ_UINT32 csty;
    /** number of resolutions */
    OPJ_UINT32 numresolutions;
    /** log2 of code-blocks width */
    OPJ_UINT32 cblkw;
    /** log2 of code-blocks height */
    OPJ_UINT32 cblkh;
    /** code-block coding style */
    OPJ_UINT32 cblksty;
    /** discrete wavelet transform identifier: 0 = 9-7 irreversible, 1 = 5-3 reversible */
    OPJ_UINT32 qmfbid;
    /** quantisation style */
    OPJ_UINT32 qntsty;
    /** stepsizes used for quantization */
    OPJ_UINT32 stepsizes_mant[OPJ_J2K_MAXBANDS];
    /** stepsizes used for quantization */
    OPJ_UINT32 stepsizes_expn[OPJ_J2K_MAXBANDS];
    /** number of guard bits */
    OPJ_UINT32 numgbits;
    /** Region Of Interest shift */
    OPJ_INT32 roishift;
    /** precinct width */
    OPJ_UINT32 prcw[OPJ_J2K_MAXRLVLS];
    /** precinct height */
    OPJ_UINT32 prch[OPJ_J2K_MAXRLVLS];
}
opj_tccp_info_t;

/**
 * Tile coding parameters information
 */
typedef struct opj_tile_v2_info {

    /** number (index) of tile */
    int tileno;
    /** coding style */
    OPJ_UINT32 csty;
    /** progression order */
    OPJ_PROG_ORDER prg;
    /** number of layers */
    OPJ_UINT32 numlayers;
    /** multi-component transform identifier */
    OPJ_UINT32 mct;

    /** information concerning tile component parameters*/
    opj_tccp_info_t *tccp_info;

} opj_tile_info_v2_t;

/**
 * Information structure about the codestream (FIXME should be expand and enhance)
 */
typedef struct opj_codestream_info_v2 {
    /* Tile info */
    /** tile origin in x = XTOsiz */
    OPJ_UINT32 tx0;
    /** tile origin in y = YTOsiz */
    OPJ_UINT32 ty0;
    /** tile size in x = XTsiz */
    OPJ_UINT32 tdx;
    /** tile size in y = YTsiz */
    OPJ_UINT32 tdy;
    /** number of tiles in X */
    OPJ_UINT32 tw;
    /** number of tiles in Y */
    OPJ_UINT32 th;

    /** number of components*/
    OPJ_UINT32 nbcomps;

    /** Default information regarding tiles inside image */
    opj_tile_info_v2_t m_default_tile_info;

    /** information regarding tiles inside image */
    opj_tile_info_v2_t *tile_info; /* FIXME not used for the moment */

} opj_codestream_info_v2_t;


/**
 * Index structure about a tile part
 */
typedef struct opj_tp_index {
    /** start position */
    OPJ_OFF_T start_pos;
    /** end position of the header */
    OPJ_OFF_T end_header;
    /** end position */
    OPJ_OFF_T end_pos;

} opj_tp_index_t;

/**
 * Index structure about a tile
 */
typedef struct opj_tile_index {
    /** tile index */
    OPJ_UINT32 tileno;

    /** number of tile parts */
    OPJ_UINT32 nb_tps;
    /** current nb of tile part (allocated)*/
    OPJ_UINT32 current_nb_tps;
    /** current tile-part index */
    OPJ_UINT32 current_tpsno;
    /** information concerning tile parts */
    opj_tp_index_t *tp_index;

    /* UniPG>> */ /* NOT USED FOR THE MOMENT IN THE V2 VERSION */
    /** number of markers */
    OPJ_UINT32 marknum;
    /** list of markers */
    opj_marker_info_t *marker;
    /** actual size of markers array */
    OPJ_UINT32 maxmarknum;
    /* <<UniPG */

    /** packet number */
    OPJ_UINT32 nb_packet;
    /** information concerning packets inside tile */
    opj_packet_info_t *packet_index;

} opj_tile_index_t;

/**
 * Index structure of the codestream (FIXME should be expand and enhance)
 */
typedef struct opj_codestream_index {
    /** main header start position (SOC position) */
    OPJ_OFF_T main_head_start;
    /** main header end position (first SOT position) */
    OPJ_OFF_T main_head_end;

    /** codestream's size */
    OPJ_UINT64 codestream_size;

    /* UniPG>> */ /* NOT USED FOR THE MOMENT IN THE V2 VERSION */
    /** number of markers */
    OPJ_UINT32 marknum;
    /** list of markers */
    opj_marker_info_t *marker;
    /** actual size of markers array */
    OPJ_UINT32 maxmarknum;
    /* <<UniPG */

    /** */
    OPJ_UINT32 nb_of_tiles;
    /** */
    opj_tile_index_t *tile_index; /* FIXME not used for the moment */

} opj_codestream_index_t;
/* -----------------------------------------------------------> */

/*
==========================================================
   Metadata from the JP2file
==========================================================
*/

/**
 * Info structure of the JP2 file
 * EXPERIMENTAL FOR THE MOMENT
 */
typedef struct opj_jp2_metadata {
    /** */
    OPJ_INT32   not_used;

} opj_jp2_metadata_t;

/**
 * Index structure of the JP2 file
 * EXPERIMENTAL FOR THE MOMENT
 */
typedef struct opj_jp2_index {
    /** */
    OPJ_INT32   not_used;

} opj_jp2_index_t;


#ifdef __cplusplus
extern "C" {
#endif


/*
==========================================================
   openjpeg version
==========================================================
*/

/* Get the version of the openjpeg library*/
OPJ_API const char * OPJ_CALLCONV opj_version(void);

/*
==========================================================
   image functions definitions
==========================================================
*/

/**
 * Create an image
 *
 * @param numcmpts      number of components
 * @param cmptparms     components parameters
 * @param clrspc        image color space
 * @return returns      a new image structure if successful, returns NULL otherwise
 * */
OPJ_API opj_image_t* OPJ_CALLCONV opj_image_create(OPJ_UINT32 numcmpts,
        opj_image_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc);

/**
 * Deallocate any resources associated with an image
 *
 * @param image         image to be destroyed
 */
OPJ_API void OPJ_CALLCONV opj_image_destroy(opj_image_t *image);

/**
 * Creates an image without allocating memory for the image (used in the new version of the library).
 *
 * @param   numcmpts    the number of components
 * @param   cmptparms   the components parameters
 * @param   clrspc      the image color space
 *
 * @return  a new image structure if successful, NULL otherwise.
*/
OPJ_API opj_image_t* OPJ_CALLCONV opj_image_tile_create(OPJ_UINT32 numcmpts,
        opj_image_cmptparm_t *cmptparms, OPJ_COLOR_SPACE clrspc);

/**
 * Allocator for opj_image_t->comps[].data
 * To be paired with opj_image_data_free.
 *
 * @param   size    number of bytes to allocate
 *
 * @return  a new pointer if successful, NULL otherwise.
 * @since 2.2.0
*/
OPJ_API void* OPJ_CALLCONV opj_image_data_alloc(OPJ_SIZE_T size);

/**
 * Destructor for opj_image_t->comps[].data
 * To be paired with opj_image_data_alloc.
 *
 * @param   ptr    Pointer to free
 *
 * @since 2.2.0
*/
OPJ_API void OPJ_CALLCONV opj_image_data_free(void* ptr);

/*
==========================================================
   stream functions definitions
==========================================================
*/

/**
 * Creates an abstract stream. This function does nothing except allocating memory and initializing the abstract stream.
 *
 * @param   p_is_input      if set to true then the stream will be an input stream, an output stream else.
 *
 * @return  a stream object.
*/
OPJ_API opj_stream_t* OPJ_CALLCONV opj_stream_default_create(
    OPJ_BOOL p_is_input);

/**
 * Creates an abstract stream. This function does nothing except allocating memory and initializing the abstract stream.
 *
 * @param   p_buffer_size  FIXME DOC
 * @param   p_is_input      if set to true then the stream will be an input stream, an output stream else.
 *
 * @return  a stream object.
*/
OPJ_API opj_stream_t* OPJ_CALLCONV opj_stream_create(OPJ_SIZE_T p_buffer_size,
        OPJ_BOOL p_is_input);

/**
 * Destroys a stream created by opj_create_stream. This function does NOT close the abstract stream. If needed the user must
 * close its own implementation of the stream.
 *
 * @param   p_stream    the stream to destroy.
 */
OPJ_API void OPJ_CALLCONV opj_stream_destroy(opj_stream_t* p_stream);

/**
 * Sets the given function to be used as a read function.
 * @param       p_stream    the stream to modify
 * @param       p_function  the function to use a read function.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_read_function(opj_stream_t* p_stream,
        opj_stream_read_fn p_function);

/**
 * Sets the given function to be used as a write function.
 * @param       p_stream    the stream to modify
 * @param       p_function  the function to use a write function.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_write_function(opj_stream_t* p_stream,
        opj_stream_write_fn p_function);

/**
 * Sets the given function to be used as a skip function.
 * @param       p_stream    the stream to modify
 * @param       p_function  the function to use a skip function.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_skip_function(opj_stream_t* p_stream,
        opj_stream_skip_fn p_function);

/**
 * Sets the given function to be used as a seek function, the stream is then seekable,
 * using SEEK_SET behavior.
 * @param       p_stream    the stream to modify
 * @param       p_function  the function to use a skip function.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_seek_function(opj_stream_t* p_stream,
        opj_stream_seek_fn p_function);

/**
 * Sets the given data to be used as a user data for the stream.
 * @param       p_stream    the stream to modify
 * @param       p_data      the data to set.
 * @param       p_function  the function to free p_data when opj_stream_destroy() is called.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_user_data(opj_stream_t* p_stream,
        void * p_data, opj_stream_free_user_data_fn p_function);

/**
 * Sets the length of the user data for the stream.
 *
 * @param p_stream    the stream to modify
 * @param data_length length of the user_data.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_user_data_length(
    opj_stream_t* p_stream, OPJ_UINT64 data_length);

/**
 * Create a stream from a file identified with its filename with default parameters (helper function)
 * @param fname             the filename of the file to stream
 * @param p_is_read_stream  whether the stream is a read stream (true) or not (false)
*/
OPJ_API opj_stream_t* OPJ_CALLCONV opj_stream_create_default_file_stream(
    const char *fname, OPJ_BOOL p_is_read_stream);

/** Create a stream from a file identified with its filename with a specific buffer size
 * @param fname             the filename of the file to stream
 * @param p_buffer_size     size of the chunk used to stream
 * @param p_is_read_stream  whether the stream is a read stream (true) or not (false)
*/
OPJ_API opj_stream_t* OPJ_CALLCONV opj_stream_create_file_stream(
    const char *fname,
    OPJ_SIZE_T p_buffer_size,
    OPJ_BOOL p_is_read_stream);

/*
==========================================================
   event manager functions definitions
==========================================================
*/
/**
 * Set the info handler use by openjpeg.
 * @param p_codec       the codec previously initialise
 * @param p_callback    the callback function which will be used
 * @param p_user_data   client object where will be returned the message
*/
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_set_info_handler(opj_codec_t * p_codec,
        opj_msg_callback p_callback,
        void * p_user_data);
/**
 * Set the warning handler use by openjpeg.
 * @param p_codec       the codec previously initialise
 * @param p_callback    the callback function which will be used
 * @param p_user_data   client object where will be returned the message
*/
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_set_warning_handler(opj_codec_t * p_codec,
        opj_msg_callback p_callback,
        void * p_user_data);
/**
 * Set the error handler use by openjpeg.
 * @param p_codec       the codec previously initialise
 * @param p_callback    the callback function which will be used
 * @param p_user_data   client object where will be returned the message
*/
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_set_error_handler(opj_codec_t * p_codec,
        opj_msg_callback p_callback,
        void * p_user_data);

/*
==========================================================
   codec functions definitions
==========================================================
*/

/**
 * Creates a J2K/JP2 decompression structure
 * @param format        Decoder to select
 *
 * @return Returns a handle to a decompressor if successful, returns NULL otherwise
 * */
OPJ_API opj_codec_t* OPJ_CALLCONV opj_create_decompress(
    OPJ_CODEC_FORMAT format);

/**
 * Destroy a decompressor handle
 *
 * @param   p_codec         decompressor handle to destroy
 */
OPJ_API void OPJ_CALLCONV opj_destroy_codec(opj_codec_t * p_codec);

/**
 * Read after the codestream if necessary
 * @param   p_codec         the JPEG2000 codec to read.
 * @param   p_stream        the JPEG2000 stream.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_end_decompress(opj_codec_t *p_codec,
        opj_stream_t *p_stream);


/**
 * Set decoding parameters to default values
 * @param parameters Decompression parameters
 */
OPJ_API void OPJ_CALLCONV opj_set_default_decoder_parameters(
    opj_dparameters_t *parameters);

/**
 * Setup the decoder with decompression parameters provided by the user and with the message handler
 * provided by the user.
 *
 * @param p_codec       decompressor handler
 * @param parameters    decompression parameters
 *
 * @return true         if the decoder is correctly set
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_setup_decoder(opj_codec_t *p_codec,
        opj_dparameters_t *parameters);

/**
 * Allocates worker threads for the compressor/decompressor.
 *
 * By default, only the main thread is used. If this function is not used,
 * but the OPJ_NUM_THREADS environment variable is set, its value will be
 * used to initialize the number of threads. The value can be either an integer
 * number, or "ALL_CPUS". If OPJ_NUM_THREADS is set and this function is called,
 * this function will override the behaviour of the environment variable.
 *
 * This function must be called after opj_setup_decoder() and
 * before opj_read_header() for the decoding side, or after opj_setup_encoder()
 * and before opj_start_compress() for the encoding side.
 *
 * @param p_codec       decompressor or compressor handler
 * @param num_threads   number of threads.
 *
 * @return OPJ_TRUE     if the function is successful.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_codec_set_threads(opj_codec_t *p_codec,
        int num_threads);

/**
 * Decodes an image header.
 *
 * @param   p_stream        the jpeg2000 stream.
 * @param   p_codec         the jpeg2000 codec to read.
 * @param   p_image         the image structure initialized with the characteristics of encoded image.
 *
 * @return true             if the main header of the codestream and the JP2 header is correctly read.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_read_header(opj_stream_t *p_stream,
        opj_codec_t *p_codec,
        opj_image_t **p_image);


/** Restrict the number of components to decode.
 *
 * This function should be called after opj_read_header().
 *
 * This function enables to restrict the set of decoded components to the
 * specified indices.
 * Note that the current implementation (apply_color_transforms == OPJ_FALSE)
 * is such that neither the multi-component transform at codestream level,
 * nor JP2 channel transformations will be applied.
 * Consequently the indices are relative to the codestream.
 *
 * Note: opj_decode_tile_data() should not be used together with opj_set_decoded_components().
 *
 * @param   p_codec         the jpeg2000 codec to read.
 * @param   numcomps        Size of the comps_indices array.
 * @param   comps_indices   Array of numcomps values representing the indices
 *                          of the components to decode (relative to the
 *                          codestream, starting at 0)
 * @param   apply_color_transforms Whether multi-component transform at codestream level
 *                                 or JP2 channel transformations should be applied.
 *                                 Currently this parameter should be set to OPJ_FALSE.
 *                                 Setting it to OPJ_TRUE will result in an error.
 *
 * @return OPJ_TRUE         in case of success.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_set_decoded_components(opj_codec_t *p_codec,
        OPJ_UINT32 numcomps,
        const OPJ_UINT32* comps_indices,
        OPJ_BOOL apply_color_transforms);

/**
 * Sets the given area to be decoded. This function should be called right after opj_read_header and before any tile header reading.
 *
 * The coordinates passed to this function should be expressed in the reference grid,
 * that is to say at the highest resolution level, even if requesting the image at lower
 * resolution levels.
 *
 * Generally opj_set_decode_area() should be followed by opj_decode(), and the
 * codec cannot be re-used.
 * In the particular case of an image made of a single tile, several sequences of
 * calls to opoj_set_decode_area() and opj_decode() are allowed, and will bring
 * performance improvements when reading an image by chunks.
 *
 * @param   p_codec         the jpeg2000 codec.
 * @param   p_image         the decoded image previously set by opj_read_header
 * @param   p_start_x       the left position of the rectangle to decode (in image coordinates).
 * @param   p_end_x         the right position of the rectangle to decode (in image coordinates).
 * @param   p_start_y       the up position of the rectangle to decode (in image coordinates).
 * @param   p_end_y         the bottom position of the rectangle to decode (in image coordinates).
 *
 * @return  true            if the area could be set.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_set_decode_area(opj_codec_t *p_codec,
        opj_image_t* p_image,
        OPJ_INT32 p_start_x, OPJ_INT32 p_start_y,
        OPJ_INT32 p_end_x, OPJ_INT32 p_end_y);

/**
 * Decode an image from a JPEG-2000 codestream
 *
 * @param p_decompressor    decompressor handle
 * @param p_stream          Input buffer stream
 * @param p_image           the decoded image
 * @return                  true if success, otherwise false
 * */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_decode(opj_codec_t *p_decompressor,
        opj_stream_t *p_stream,
        opj_image_t *p_image);

/**
 * Get the decoded tile from the codec
 *
 * @param   p_codec         the jpeg2000 codec.
 * @param   p_stream        input streamm
 * @param   p_image         output image
 * @param   tile_index      index of the tile which will be decode
 *
 * @return                  true if success, otherwise false
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_get_decoded_tile(opj_codec_t *p_codec,
        opj_stream_t *p_stream,
        opj_image_t *p_image,
        OPJ_UINT32 tile_index);

/**
 * Set the resolution factor of the decoded image
 * @param   p_codec         the jpeg2000 codec.
 * @param   res_factor      resolution factor to set
 *
 * @return                  true if success, otherwise false
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_set_decoded_resolution_factor(
    opj_codec_t *p_codec, OPJ_UINT32 res_factor);

/**
 * Writes a tile with the given data.
 *
 * @param   p_codec             the jpeg2000 codec.
 * @param   p_tile_index        the index of the tile to write. At the moment, the tiles must be written from 0 to n-1 in sequence.
 * @param   p_data              pointer to the data to write. Data is arranged in sequence, data_comp0, then data_comp1, then ... NO INTERLEAVING should be set.
 * @param   p_data_size         this value os used to make sure the data being written is correct. The size must be equal to the sum for each component of
 *                              tile_width * tile_height * component_size. component_size can be 1,2 or 4 bytes, depending on the precision of the given component.
 * @param   p_stream            the stream to write data to.
 *
 * @return  true if the data could be written.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_write_tile(opj_codec_t *p_codec,
        OPJ_UINT32 p_tile_index,
        OPJ_BYTE * p_data,
        OPJ_UINT32 p_data_size,
        opj_stream_t *p_stream);

/**
 * Reads a tile header. This function is compulsory and allows one to know the size of the tile that will be decoded.
 * The user may need to refer to the image got by opj_read_header to understand the size being taken by the tile.
 *
 * @param   p_codec         the jpeg2000 codec.
 * @param   p_tile_index    pointer to a value that will hold the index of the tile being decoded, in case of success.
 * @param   p_data_size     pointer to a value that will hold the maximum size of the decoded data, in case of success. In case
 *                          of truncated codestreams, the actual number of bytes decoded may be lower. The computation of the size is the same
 *                          as depicted in opj_write_tile.
 * @param   p_tile_x0       pointer to a value that will hold the x0 pos of the tile (in the image).
 * @param   p_tile_y0       pointer to a value that will hold the y0 pos of the tile (in the image).
 * @param   p_tile_x1       pointer to a value that will hold the x1 pos of the tile (in the image).
 * @param   p_tile_y1       pointer to a value that will hold the y1 pos of the tile (in the image).
 * @param   p_nb_comps      pointer to a value that will hold the number of components in the tile.
 * @param   p_should_go_on  pointer to a boolean that will hold the fact that the decoding should go on. In case the
 *                          codestream is over at the time of the call, the value will be set to false. The user should then stop
 *                          the decoding.
 * @param   p_stream        the stream to decode.
 * @return  true            if the tile header could be decoded. In case the decoding should end, the returned value is still true.
 *                          returning false may be the result of a shortage of memory or an internal error.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_read_tile_header(opj_codec_t *p_codec,
        opj_stream_t * p_stream,
        OPJ_UINT32 * p_tile_index,
        OPJ_UINT32 * p_data_size,
        OPJ_INT32 * p_tile_x0, OPJ_INT32 * p_tile_y0,
        OPJ_INT32 * p_tile_x1, OPJ_INT32 * p_tile_y1,
        OPJ_UINT32 * p_nb_comps,
        OPJ_BOOL * p_should_go_on);

/**
 * Reads a tile data. This function is compulsory and allows one to decode tile data. opj_read_tile_header should be called before.
 * The user may need to refer to the image got by opj_read_header to understand the size being taken by the tile.
 *
 * Note: opj_decode_tile_data() should not be used together with opj_set_decoded_components().
 *
 * @param   p_codec         the jpeg2000 codec.
 * @param   p_tile_index    the index of the tile being decoded, this should be the value set by opj_read_tile_header.
 * @param   p_data          pointer to a memory block that will hold the decoded data.
 * @param   p_data_size     size of p_data. p_data_size should be bigger or equal to the value set by opj_read_tile_header.
 * @param   p_stream        the stream to decode.
 *
 * @return  true            if the data could be decoded.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_decode_tile_data(opj_codec_t *p_codec,
        OPJ_UINT32 p_tile_index,
        OPJ_BYTE * p_data,
        OPJ_UINT32 p_data_size,
        opj_stream_t *p_stream);

/* COMPRESSION FUNCTIONS*/

/**
 * Creates a J2K/JP2 compression structure
 * @param   format      Coder to select
 * @return              Returns a handle to a compressor if successful, returns NULL otherwise
 */
OPJ_API opj_codec_t* OPJ_CALLCONV opj_create_compress(OPJ_CODEC_FORMAT format);

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
OPJ_API void OPJ_CALLCONV opj_set_default_encoder_parameters(
    opj_cparameters_t *parameters);

/**
 * Setup the encoder parameters using the current image and using user parameters.
 * @param p_codec       Compressor handle
 * @param parameters    Compression parameters
 * @param image         Input filled image
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_setup_encoder(opj_codec_t *p_codec,
        opj_cparameters_t *parameters,
        opj_image_t *image);


/**
 * Specify extra options for the encoder.
 *
 * This may be called after opj_setup_encoder() and before opj_start_compress()
 *
 * This is the way to add new options in a fully ABI compatible way, without
 * extending the opj_cparameters_t structure.
 *
 * Currently supported options are:
 * <ul>
 * <li>PLT=YES/NO. Defaults to NO. If set to YES, PLT marker segments,
 *     indicating the length of each packet in the tile-part header, will be
 *     written. Since 2.3.2</li>
 * </ul>
 *
 * @param p_codec       Compressor handle
 * @param p_options     Compression options. This should be a NULL terminated
 *                      array of strings. Each string is of the form KEY=VALUE.
 *
 * @return OPJ_TRUE in case of success.
 * @since 2.3.2
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_encoder_set_extra_options(
    opj_codec_t *p_codec,
    const char* const* p_options);

/**
 * Start to compress the current image.
 * @param p_codec       Compressor handle
 * @param p_image       Input filled image
 * @param p_stream      Input stgream
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_start_compress(opj_codec_t *p_codec,
        opj_image_t * p_image,
        opj_stream_t *p_stream);

/**
 * End to compress the current image.
 * @param p_codec       Compressor handle
 * @param p_stream      Input stgream
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_end_compress(opj_codec_t *p_codec,
        opj_stream_t *p_stream);

/**
 * Encode an image into a JPEG-2000 codestream
 * @param p_codec       compressor handle
 * @param p_stream      Output buffer stream
 *
 * @return              Returns true if successful, returns false otherwise
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_encode(opj_codec_t *p_codec,
        opj_stream_t *p_stream);
/*
==========================================================
   codec output functions definitions
==========================================================
*/
/* EXPERIMENTAL FUNCTIONS FOR NOW, USED ONLY IN J2K_DUMP*/

/**
Destroy Codestream information after compression or decompression
@param cstr_info Codestream information structure
*/
OPJ_API void OPJ_CALLCONV opj_destroy_cstr_info(opj_codestream_info_v2_t
        **cstr_info);


/**
 * Dump the codec information into the output stream
 *
 * @param   p_codec         the jpeg2000 codec.
 * @param   info_flag       type of information dump.
 * @param   output_stream   output stream where dump the information gotten from the codec.
 *
 */
OPJ_API void OPJ_CALLCONV opj_dump_codec(opj_codec_t *p_codec,
        OPJ_INT32 info_flag,
        FILE* output_stream);

/**
 * Get the codestream information from the codec
 *
 * @param   p_codec         the jpeg2000 codec.
 *
 * @return                  a pointer to a codestream information structure.
 *
 */
OPJ_API opj_codestream_info_v2_t* OPJ_CALLCONV opj_get_cstr_info(
    opj_codec_t *p_codec);

/**
 * Get the codestream index from the codec
 *
 * @param   p_codec         the jpeg2000 codec.
 *
 * @return                  a pointer to a codestream index structure.
 *
 */
OPJ_API opj_codestream_index_t * OPJ_CALLCONV opj_get_cstr_index(
    opj_codec_t *p_codec);

OPJ_API void OPJ_CALLCONV opj_destroy_cstr_index(opj_codestream_index_t
        **p_cstr_index);


/**
 * Get the JP2 file information from the codec FIXME
 *
 * @param   p_codec         the jpeg2000 codec.
 *
 * @return                  a pointer to a JP2 metadata structure.
 *
 */
OPJ_API opj_jp2_metadata_t* OPJ_CALLCONV opj_get_jp2_metadata(
    opj_codec_t *p_codec);

/**
 * Get the JP2 file index from the codec FIXME
 *
 * @param   p_codec         the jpeg2000 codec.
 *
 * @return                  a pointer to a JP2 index structure.
 *
 */
OPJ_API opj_jp2_index_t* OPJ_CALLCONV opj_get_jp2_index(opj_codec_t *p_codec);


/*
==========================================================
   MCT functions
==========================================================
*/

/**
 * Sets the MCT matrix to use.
 *
 * @param   parameters      the parameters to change.
 * @param   pEncodingMatrix the encoding matrix.
 * @param   p_dc_shift      the dc shift coefficients to use.
 * @param   pNbComp         the number of components of the image.
 *
 * @return  true if the parameters could be set.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_set_MCT(opj_cparameters_t *parameters,
        OPJ_FLOAT32 * pEncodingMatrix,
        OPJ_INT32 * p_dc_shift,
        OPJ_UINT32 pNbComp);

/*
==========================================================
   Thread functions
==========================================================
*/

/** Returns if the library is built with thread support.
 * OPJ_TRUE if mutex, condition, thread, thread pool are available.
 */
OPJ_API OPJ_BOOL OPJ_CALLCONV opj_has_thread_support(void);

/** Return the number of virtual CPUs */
OPJ_API int OPJ_CALLCONV opj_get_num_cpus(void);


#ifdef __cplusplus
}
#endif

#endif /* OPENJPEG_H */
