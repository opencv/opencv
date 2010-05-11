/* $Id: tif_dir.h,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

/*
 * Copyright (c) 1988-1997 Sam Leffler
 * Copyright (c) 1991-1997 Silicon Graphics, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and 
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 * 
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY 
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  
 * 
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF 
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
 * OF THIS SOFTWARE.
 */

#ifndef _TIFFDIR_
#define	_TIFFDIR_
/*
 * ``Library-private'' Directory-related Definitions.
 */

/*
 * Internal format of a TIFF directory entry.
 */
typedef	struct {
#define	FIELD_SETLONGS	4
	/* bit vector of fields that are set */
	unsigned long	td_fieldsset[FIELD_SETLONGS];

	uint32	td_imagewidth, td_imagelength, td_imagedepth;
	uint32	td_tilewidth, td_tilelength, td_tiledepth;
	uint32	td_subfiletype;
	uint16	td_bitspersample;
	uint16	td_sampleformat;
	uint16	td_compression;
	uint16	td_photometric;
	uint16	td_threshholding;
	uint16	td_fillorder;
	uint16	td_orientation;
	uint16	td_samplesperpixel;
	uint32	td_rowsperstrip;
	uint16	td_minsamplevalue, td_maxsamplevalue;
	double	td_sminsamplevalue, td_smaxsamplevalue;
	float	td_xresolution, td_yresolution;
	uint16	td_resolutionunit;
	uint16	td_planarconfig;
	float	td_xposition, td_yposition;
	uint16	td_pagenumber[2];
	uint16*	td_colormap[3];
	uint16	td_halftonehints[2];
	uint16	td_extrasamples;
	uint16*	td_sampleinfo;
	double	td_stonits;
	char*	td_documentname;
	char*	td_artist;
	char*	td_datetime;
	char*	td_hostcomputer;
	char*	td_imagedescription;
	char*	td_make;
	char*	td_model;
        char*   td_copyright;
	char*	td_pagename;
	tstrip_t td_stripsperimage;
	tstrip_t td_nstrips;		/* size of offset & bytecount arrays */
	uint32*	td_stripoffset;
	uint32*	td_stripbytecount;
	int	td_stripbytecountsorted; /* is the bytecount array sorted ascending? */
	uint16	td_nsubifd;
	uint32*	td_subifd;
	/* YCbCr parameters */
	float*	td_ycbcrcoeffs;
	uint16	td_ycbcrsubsampling[2];
	uint16	td_ycbcrpositioning;
	/* Colorimetry parameters */
	float*	td_whitepoint;
	float*	td_primarychromas;
	float*	td_refblackwhite;
	uint16*	td_transferfunction[3];
	/* CMYK parameters */
	uint16	td_inkset;
	uint16	td_ninks;
	uint16	td_dotrange[2];
	int	td_inknameslen;
	char*	td_inknames;
	char*	td_targetprinter;
	/* ICC parameters */
	uint32	td_profileLength;
	void	*td_profileData;
	/* Adobe Photoshop tag handling */
	uint32	td_photoshopLength;
	void	*td_photoshopData;
	/* IPTC parameters */
	uint32	td_richtiffiptcLength;
	void	*td_richtiffiptcData;
        /* Begin Pixar Tag values. */
        uint32	td_imagefullwidth, td_imagefulllength;
 	char*	td_textureformat;
 	char*	td_wrapmodes;
 	float	td_fovcot;
 	float*	td_matrixWorldToScreen;
 	float*	td_matrixWorldToCamera;
 	/* End Pixar Tag Values. */
	uint32	td_xmlpacketLength;
	void	*td_xmlpacketData;
	int     td_customValueCount;
        TIFFTagValue *td_customValues;
} TIFFDirectory;

/*
 * Field flags used to indicate fields that have
 * been set in a directory, and to reference fields
 * when manipulating a directory.
 */

/*
 * FIELD_IGNORE is used to signify tags that are to
 * be processed but otherwise ignored.  This permits
 * antiquated tags to be quietly read and discarded.
 * Note that a bit *is* allocated for ignored tags;
 * this is understood by the directory reading logic
 * which uses this fact to avoid special-case handling
 */ 
#define	FIELD_IGNORE			0

/* multi-item fields */
#define	FIELD_IMAGEDIMENSIONS		1
#define FIELD_TILEDIMENSIONS		2
#define	FIELD_RESOLUTION		3
#define	FIELD_POSITION			4

/* single-item fields */
#define	FIELD_SUBFILETYPE		5
#define	FIELD_BITSPERSAMPLE		6
#define	FIELD_COMPRESSION		7
#define	FIELD_PHOTOMETRIC		8
#define	FIELD_THRESHHOLDING		9
#define	FIELD_FILLORDER			10
#define	FIELD_DOCUMENTNAME		11
#define	FIELD_IMAGEDESCRIPTION		12
#define	FIELD_MAKE			13
#define	FIELD_MODEL			14
#define	FIELD_ORIENTATION		15
#define	FIELD_SAMPLESPERPIXEL		16
#define	FIELD_ROWSPERSTRIP		17
#define	FIELD_MINSAMPLEVALUE		18
#define	FIELD_MAXSAMPLEVALUE		19
#define	FIELD_PLANARCONFIG		20
#define	FIELD_PAGENAME			21
#define	FIELD_RESOLUTIONUNIT		22
#define	FIELD_PAGENUMBER		23
#define	FIELD_STRIPBYTECOUNTS		24
#define	FIELD_STRIPOFFSETS		25
#define	FIELD_COLORMAP			26
#define FIELD_ARTIST			27
#define FIELD_DATETIME			28
#define FIELD_HOSTCOMPUTER		29
/* unused - was FIELD_SOFTWARE          30 */
#define	FIELD_EXTRASAMPLES		31
#define FIELD_SAMPLEFORMAT		32
#define	FIELD_SMINSAMPLEVALUE		33
#define	FIELD_SMAXSAMPLEVALUE		34
#define FIELD_IMAGEDEPTH		35
#define FIELD_TILEDEPTH			36
#define	FIELD_HALFTONEHINTS		37
#define FIELD_YCBCRCOEFFICIENTS		38
#define FIELD_YCBCRSUBSAMPLING		39
#define FIELD_YCBCRPOSITIONING		40
#define	FIELD_REFBLACKWHITE		41
#define	FIELD_WHITEPOINT		42
#define	FIELD_PRIMARYCHROMAS		43
#define	FIELD_TRANSFERFUNCTION		44
#define	FIELD_INKSET			45
#define	FIELD_INKNAMES			46
#define	FIELD_DOTRANGE			47
#define	FIELD_TARGETPRINTER		48
#define	FIELD_SUBIFD			49
#define	FIELD_NUMBEROFINKS		50
#define FIELD_ICCPROFILE		51
#define FIELD_PHOTOSHOP			52
#define FIELD_RICHTIFFIPTC		53
#define FIELD_STONITS			54
/* Begin PIXAR */
#define	FIELD_IMAGEFULLWIDTH		55
#define	FIELD_IMAGEFULLLENGTH		56
#define FIELD_TEXTUREFORMAT		57
#define FIELD_WRAPMODES			58
#define FIELD_FOVCOT			59
#define FIELD_MATRIX_WORLDTOSCREEN	60
#define FIELD_MATRIX_WORLDTOCAMERA	61
#define FIELD_COPYRIGHT			62
#define FIELD_XMLPACKET			63
/*      FIELD_CUSTOM (see tiffio.h)     65 */
/* end of support for well-known tags; codec-private tags follow */
#define	FIELD_CODEC			66	/* base of codec-private tags */


/*
 * Pseudo-tags don't normally need field bits since they
 * are not written to an output file (by definition).
 * The library also has express logic to always query a
 * codec for a pseudo-tag so allocating a field bit for
 * one is a waste.   If codec wants to promote the notion
 * of a pseudo-tag being ``set'' or ``unset'' then it can
 * do using internal state flags without polluting the
 * field bit space defined for real tags.
 */
#define	FIELD_PSEUDO			0

#define	FIELD_LAST			(32*FIELD_SETLONGS-1)

#define	TIFFExtractData(tif, type, v) \
    ((uint32) ((tif)->tif_header.tiff_magic == TIFF_BIGENDIAN ? \
        ((v) >> (tif)->tif_typeshift[type]) & (tif)->tif_typemask[type] : \
	(v) & (tif)->tif_typemask[type]))
#define	TIFFInsertData(tif, type, v) \
    ((uint32) ((tif)->tif_header.tiff_magic == TIFF_BIGENDIAN ? \
        ((v) & (tif)->tif_typemask[type]) << (tif)->tif_typeshift[type] : \
	(v) & (tif)->tif_typemask[type]))


#define BITn(n)				(((unsigned long)1L)<<((n)&0x1f)) 
#define BITFIELDn(tif, n)		((tif)->tif_dir.td_fieldsset[(n)/32]) 
#define TIFFFieldSet(tif, field)	(BITFIELDn(tif, field) & BITn(field)) 
#define TIFFSetFieldBit(tif, field)	(BITFIELDn(tif, field) |= BITn(field))
#define TIFFClrFieldBit(tif, field)	(BITFIELDn(tif, field) &= ~BITn(field))

#define	FieldSet(fields, f)		(fields[(f)/32] & BITn(f))
#define	ResetFieldBit(fields, f)	(fields[(f)/32] &= ~BITn(f))

#if defined(__cplusplus)
extern "C" {
#endif
extern	void _TIFFSetupFieldInfo(TIFF*);
extern	void _TIFFPrintFieldInfo(TIFF*, FILE*);
extern	TIFFDataType _TIFFSampleToTagType(TIFF*);
extern  const TIFFFieldInfo* _TIFFFindOrRegisterFieldInfo( TIFF *tif,
							   ttag_t tag,
							   TIFFDataType dt );
extern  TIFFFieldInfo* _TIFFCreateAnonFieldInfo( TIFF *tif, ttag_t tag,
                                                 TIFFDataType dt );

#define _TIFFMergeFieldInfo	    TIFFMergeFieldInfo
#define _TIFFFindFieldInfo	    TIFFFindFieldInfo
#define _TIFFFindFieldInfoByName    TIFFFindFieldInfoByName
#define _TIFFFieldWithTag	    TIFFFieldWithTag
#define _TIFFFieldWithName	    TIFFFieldWithName

#if defined(__cplusplus)
}
#endif
#endif /* _TIFFDIR_ */

/* vim: set ts=8 sts=8 sw=8 noet: */
