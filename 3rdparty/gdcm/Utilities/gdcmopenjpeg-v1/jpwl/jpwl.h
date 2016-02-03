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
#ifndef __JPWL_H
#define __JPWL_H

#ifdef USE_JPWL

#include "crc.h"
#include "rs.h"

/**
@file jpwl.h
@brief The JPEG-2000 Part11 (JPWL) marker segments manager

The functions in JPWL.C have for goal to read/write the markers added by JPWL.
*/

/** @defgroup JPWL JPWL - JPEG-2000 Part11 (JPWL) codestream manager */
/*@{*/

/**
Assume a basic codestream structure, so you can resort better from uncorrected errors
*/
#define JPWL_ASSUME true

/**
EPB (Error Protection Block) Marker segment
*/
typedef struct jpwl_epb_ms {
	/**@name Private fields set by epb_create */
	/*@{*/
	/** is the latest in header? */
	bool latest;
	/** is it in packed mode? */
	bool packed;
	/** TH where this marker has been placed (-1 means MH) */
	int tileno;
	/** index in current header (0-63) */
	unsigned char index;
	/** error protection method	[-1=absent 0=none 1=predefined 16=CRC-16 32=CRC-32 37-128=RS] */
	int hprot;
	/** message word length of pre-data */
	int k_pre;
	/** code word length of pre-data */
	int n_pre;
	/** length of pre-data */
	int pre_len;
	/** message word length of post-data */
	int k_post;
	/** code word length of post-data */
	int n_post;
	/** length of post-data */
	int post_len;
	/*@}*/
	/**@name Marker segment fields */
	/*@{*/
	/** two bytes for the length of EPB MS, exluding the marker itself (11 to 65535 bytes) */
	unsigned short int Lepb;
	/** single byte for the style */
	unsigned char Depb; 
	/** four bytes, from 0 to 2^31-1 */
	unsigned long int LDPepb;
	/** four bytes, next error management method */
	unsigned long int Pepb;
	/** EPB data, variable size */
	unsigned char *data;   
	/*@}*/
}	jpwl_epb_ms_t;

/**
EPC (Error Protection Capability) Marker segment
*/
typedef struct jpwl_epc_ms {
	/** is ESD active? */
	bool esd_on;
	/** is RED active? */
	bool red_on;
	/** is EPB active? */
	bool epb_on;
	/** are informative techniques active? */
	bool info_on;
	/**@name Marker segment fields */
	/*@{*/
	/** two bytes for the length of EPC MS, exluding the marker itself (9 to 65535 bytes) */
	unsigned short int Lepc;   
	/** two bytes, CRC for the EPC, excluding Pcrc itself */
	unsigned short int Pcrc;   
	/** four bytes, the codestream length from SOC to EOC */
	unsigned long int DL;     
	/** one byte, signals JPWL techniques adoption */
	unsigned char Pepc;	
	/** EPC data, variable length */
	unsigned char *data;	
	/*@}*/
}	jpwl_epc_ms_t;

/**
ESD (Error Sensitivity Descriptor) Marker segment
*/
typedef struct jpwl_esd_ms {
	/** codestream addressing mode [0=packet, 1=byte range, 2=packet range, 3=reserved] */
	unsigned char addrm;
	/** size of codestream addresses [2/4 bytes] */
	unsigned char ad_size;
	/** type of sensitivity
	[0=relative error, 1=MSE, 2=MSE reduction, 3=PSNR, 4=PSNR increment,
	5=MAXERR (absolute peak error), 6=TSE (total squared error), 7=reserved */
	unsigned char senst;
	/** size of sensitivity data (1/2 bytes) */
	unsigned char se_size;
	/**@name Marker segment fields */
	/*@{*/
	/** two bytes for the length of ESD MS, exluding the marker itself (4 to 65535 bytes) */
	unsigned short int Lesd;   
	/** two bytes, component of error sensitivity */
	unsigned short int Cesd;
	/** one byte, signals JPWL techniques adoption */
	unsigned char Pesd;	
	/** ESD data, variable length */
	unsigned char *data;	
	/*@}*/
	/**@name Fields set by esd_create (only internal use) */
	/*@{*/
	/** number of components in the image */
	int numcomps;
	/** tile where this marker has been placed (-1 means MH) */
	int tileno;
	/** number of sensitivity values */
	unsigned long int svalnum;
	/** size of a single sensitivity pair (address+value) */
	size_t sensval_size;
	/*@}*/
}	jpwl_esd_ms_t;

/**
RED (Residual Error Descriptor) Marker segment
*/
typedef struct jpwl_red_ms {
	/** two bytes for the length of RED MS, exluding the marker itself (3 to 65535 bytes) */
	unsigned short int Lred;
	/** one byte, signals JPWL techniques adoption */
	unsigned char Pred;	
	/** RED data, variable length */
	unsigned char *data;	
}	jpwl_red_ms_t;

/**
Structure used to store JPWL markers temporary position and readyness
*/
typedef struct jpwl_marker {
	/** marker value (J2K_MS_EPC, etc.) */
	int id;
	/** union keeping the pointer to the real marker struct */
	union jpwl_marks {
		/** pointer to EPB marker */
		jpwl_epb_ms_t *epbmark;
		/** pointer to EPC marker */
		jpwl_epc_ms_t *epcmark;
		/** pointer to ESD marker */
		jpwl_esd_ms_t *esdmark;
		/** pointer to RED marker */
		jpwl_red_ms_t *redmark;
	} m;
	/** position where the marker should go, in the pre-JPWL codestream */ 
	unsigned long int pos;
	/** same as before, only written as a double, so we can sort it better */
	double dpos;
	/** length of the marker segment (marker excluded) */
	unsigned short int len;
	/** the marker length is ready or not? */
	bool len_ready;
	/** the marker position is ready or not? */
	bool pos_ready;
	/** the marker parameters are ready or not? */
	bool parms_ready;
	/** are the written data ready or not */
	bool data_ready;
}	jpwl_marker_t;

/**
Encode according to JPWL specs
@param j2k J2K handle
@param cio codestream handle
@param image image handle
*/
void jpwl_encode(opj_j2k_t *j2k, opj_cio_t *cio, opj_image_t *image);

/**
Prepare the list of JPWL markers, after the Part 1 codestream
has been finalized (index struct is full)
@param j2k J2K handle
@param cio codestream handle
@param image image handle
*/
void jpwl_prepare_marks(opj_j2k_t *j2k, opj_cio_t *cio, opj_image_t *image);

/**
Dump the list of JPWL markers, after it has been prepared
@param j2k J2K handle
@param cio codestream handle
@param image image handle
*/
void jpwl_dump_marks(opj_j2k_t *j2k, opj_cio_t *cio, opj_image_t *image);

/**
Read the EPC marker (Error Protection Capability)
@param j2k J2K handle
*/
void j2k_read_epc(opj_j2k_t *j2k);

/**
Write the EPC marker (Error Protection Capability), BUT the DL field is always set to 0
(this simplifies the management of EPBs and it is openly stated in the standard
as a possible value, mening that the information is not available) and the informative techniques
are not yet implemented
@param j2k J2K handle
*/
void j2k_write_epc(opj_j2k_t *j2k);

/**
Read the EPB marker (Error Protection Block)
@param j2k J2K handle
*/
void j2k_read_epb(opj_j2k_t *j2k);

/**
Write the EPB marker (Error Protection Block)
@param j2k J2K handle
*/
void j2k_write_epb(opj_j2k_t *j2k);

/**
Read the ESD marker (Error Sensitivity Descriptor)
@param j2k J2K handle
*/
void j2k_read_esd(opj_j2k_t *j2k);

/**
Read the RED marker (Residual Error Descriptor)
@param j2k J2K handle
*/
void j2k_read_red(opj_j2k_t *j2k);

/** create an EPB marker segment
@param j2k J2K compressor handle
@param latest it is the latest EPB in the header
@param packed EPB is in packed style
@param tileno tile number where the marker has been placed (-1 means MH)
@param idx current EPB running index
@param hprot applied protection type (-1/0,1,16,32,37-128)
@param pre_len length of pre-protected data
@param post_len length of post-protected data
@return returns the freshly created EPB
*/
jpwl_epb_ms_t *jpwl_epb_create(opj_j2k_t *j2k, bool latest, bool packed, int tileno, int idx, int hprot,
							   unsigned long int pre_len, unsigned long int post_len);

/** add a number of EPB marker segments
@param j2k J2K compressor handle
@param jwmarker pointer to the JPWL markers list
@param jwmarker_num pointer to the number of JPWL markers (gets updated)
@param latest it is the latest group of EPBs in the header
@param packed EPBs are in packed style
@param insideMH it is in the MH
@param idx pointer to the starting EPB running index (gets updated)
@param hprot applied protection type (-1/0,1,16,32,37-128)
@param place_pos place in original codestream where EPBs should go
@param tileno tile number of these EPBs
@param pre_len length of pre-protected data
@param post_len length of post-protected data
@return returns the length of all added markers
*/
int jpwl_epbs_add(opj_j2k_t *j2k, jpwl_marker_t *jwmarker, int *jwmarker_num,
				  bool latest, bool packed, bool insideMH, int *idx, int hprot,
				  double place_pos, int tileno,
				  unsigned long int pre_len, unsigned long int post_len);

/** add a number of ESD marker segments
@param j2k J2K compressor handle
@param jwmarker pointer to the JPWL markers list
@param jwmarker_num pointer to the number of JPWL markers (gets updated)
@param comps considered component (-1=average, 0/1/2/...=component no.)
@param addrm addressing mode (0=packet, 1=byte range, 2=packet range, 3=reserved)
@param ad_size size of addresses (2/4 bytes)
@param senst sensitivity type
@param se_size sensitivity values size (1/2 bytes)
@param place_pos place in original codestream where EPBs should go
@param tileno tile number of these EPBs
@return returns the length of all added markers
*/
int jpwl_esds_add(opj_j2k_t *j2k, jpwl_marker_t *jwmarker, int *jwmarker_num,
				  int comps, unsigned char addrm, unsigned char ad_size,
				  unsigned char senst, unsigned char se_size,
				  double place_pos, int tileno);
	
/** updates the information structure by modifying the positions and lengths
@param j2k J2K compressor handle
@param jwmarker pointer to JPWL markers list
@param jwmarker_num number of JPWL markers
@return returns true in case of success
*/			  
bool jpwl_update_info(opj_j2k_t *j2k, jpwl_marker_t *jwmarker, int jwmarker_num);


bool jpwl_esd_fill(opj_j2k_t *j2k, jpwl_esd_ms_t *esdmark, unsigned char *buf);

bool jpwl_epb_fill(opj_j2k_t *j2k, jpwl_epb_ms_t *epbmark, unsigned char *buf, unsigned char *post_buf);

void j2k_add_marker(opj_codestream_info_t *cstr_info, unsigned short int type, int pos, int len);

/** corrects the data in the JPWL codestream
@param j2k J2K compressor handle
@return true if correction is performed correctly
*/
bool jpwl_correct(opj_j2k_t *j2k);

/** corrects the data protected by an EPB
@param j2k J2K compressor handle
@param buffer pointer to the EPB position
@param type type of EPB: 0=MH, 1=TPH, 2=other, 3=auto
@param pre_len length of pre-data
@param post_len length of post_data
@param conn is a pointer to the length of all connected (packed) EPBs
@param L4_bufp is a pointer to the buffer pointer of redundancy data
@return returns true if correction could be succesfully performed
*/
bool jpwl_epb_correct(opj_j2k_t *j2k, unsigned char *buffer, int type, int pre_len, int post_len, int *conn,
					  unsigned char **L4_bufp);

/** check that a tile and its children have valid data
@param j2k J2K decompressor handle
@param tcd Tile decompressor handle
@param tileno number of the tile to check
*/
bool jpwl_check_tile(opj_j2k_t *j2k, opj_tcd_t *tcd, int tileno);

/** Macro functions for CRC computation */

/**
Computes the CRC-16, as stated in JPWL specs
@param CRC two bytes containing the CRC value (must be initialized with 0x0000)
@param DATA byte for which the CRC is computed; call this on every byte of the sequence
and get the CRC at the end
*/
#define jpwl_updateCRC16(CRC, DATA) updateCRC16(CRC, DATA)

/**
Computes the CRC-32, as stated in JPWL specs
@param CRC four bytes containing the CRC value (must be initialized with 0x00000000)
@param DATA byte for which the CRC is computed; call this on every byte of the sequence
and get the CRC at the end
*/
#define jpwl_updateCRC32(CRC, DATA) updateCRC32(CRC, DATA)

/**
Computes the minimum between two integers
@param a first integer to compare
@param b second integer to compare
@return returns the minimum integer between a and b
*/
#ifndef min
#define min(a,b)    (((a) < (b)) ? (a) : (b))
#endif /* min */

/*@}*/

#endif /* USE_JPWL */

#ifdef USE_JPSEC

/** @defgroup JPSEC JPSEC - JPEG-2000 Part 8 (JPSEC) codestream manager */
/*@{*/

/**
Read the SEC marker (SEcured Codestream)
@param j2k J2K handle
*/
void j2k_read_sec(opj_j2k_t *j2k);

/**
Write the SEC marker (SEcured Codestream)
@param j2k J2K handle
*/
void j2k_write_sec(opj_j2k_t *j2k);

/**
Read the INSEC marker (SEcured Codestream)
@param j2k J2K handle
*/
void j2k_read_insec(opj_j2k_t *j2k);

/*@}*/

#endif /* USE_JPSEC */

#endif /* __JPWL_H */

