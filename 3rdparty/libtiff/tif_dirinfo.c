/* $Id: tif_dirinfo.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

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

/*
 * TIFF Library.
 *
 * Core Directory Tag Support.
 */
#include "tiffiop.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * NB: NB: THIS ARRAY IS ASSUMED TO BE SORTED BY TAG.
 *     If a tag can have both LONG and SHORT types
 *     then the LONG must be placed before the SHORT for
 *     writing to work properly.
 *
 * NOTE: The second field (field_readcount) and third field (field_writecount)
 *       sometimes use the values TIFF_VARIABLE (-1), TIFF_VARIABLE2 (-3)
 *       and TIFFTAG_SPP (-2). The macros should be used but would throw off 
 *       the formatting of the code, so please interprete the -1, -2 and -3 
 *       values accordingly.
 */
#ifndef VMS
static 
#endif
const TIFFFieldInfo tiffFieldInfo[] = {
    { TIFFTAG_SUBFILETYPE,	 1, 1, TIFF_LONG,	FIELD_SUBFILETYPE,
      TRUE,	FALSE,	"SubfileType" },
/* XXX SHORT for compatibility w/ old versions of the library */
    { TIFFTAG_SUBFILETYPE,	 1, 1, TIFF_SHORT,	FIELD_SUBFILETYPE,
      TRUE,	FALSE,	"SubfileType" },
    { TIFFTAG_OSUBFILETYPE,	 1, 1, TIFF_SHORT,	FIELD_SUBFILETYPE,
      TRUE,	FALSE,	"OldSubfileType" },
    { TIFFTAG_IMAGEWIDTH,	 1, 1, TIFF_LONG,	FIELD_IMAGEDIMENSIONS,
      FALSE,	FALSE,	"ImageWidth" },
    { TIFFTAG_IMAGEWIDTH,	 1, 1, TIFF_SHORT,	FIELD_IMAGEDIMENSIONS,
      FALSE,	FALSE,	"ImageWidth" },
    { TIFFTAG_IMAGELENGTH,	 1, 1, TIFF_LONG,	FIELD_IMAGEDIMENSIONS,
      TRUE,	FALSE,	"ImageLength" },
    { TIFFTAG_IMAGELENGTH,	 1, 1, TIFF_SHORT,	FIELD_IMAGEDIMENSIONS,
      TRUE,	FALSE,	"ImageLength" },
    { TIFFTAG_BITSPERSAMPLE,	-1,-1, TIFF_SHORT,	FIELD_BITSPERSAMPLE,
      FALSE,	FALSE,	"BitsPerSample" },
/* XXX LONG for compatibility with some broken TIFF writers */
    { TIFFTAG_BITSPERSAMPLE,	-1,-1, TIFF_LONG,	FIELD_BITSPERSAMPLE,
      FALSE,	FALSE,	"BitsPerSample" },
    { TIFFTAG_COMPRESSION,	-1, 1, TIFF_SHORT,	FIELD_COMPRESSION,
      FALSE,	FALSE,	"Compression" },
/* XXX LONG for compatibility with some broken TIFF writers */
    { TIFFTAG_COMPRESSION,	-1, 1, TIFF_LONG,	FIELD_COMPRESSION,
      FALSE,	FALSE,	"Compression" },
    { TIFFTAG_PHOTOMETRIC,	 1, 1, TIFF_SHORT,	FIELD_PHOTOMETRIC,
      FALSE,	FALSE,	"PhotometricInterpretation" },
/* XXX LONG for compatibility with some broken TIFF writers */
    { TIFFTAG_PHOTOMETRIC,	 1, 1, TIFF_LONG,	FIELD_PHOTOMETRIC,
      FALSE,	FALSE,	"PhotometricInterpretation" },
    { TIFFTAG_THRESHHOLDING,	 1, 1, TIFF_SHORT,	FIELD_THRESHHOLDING,
      TRUE,	FALSE,	"Threshholding" },
    { TIFFTAG_CELLWIDTH,	 1, 1, TIFF_SHORT,	FIELD_IGNORE,
      TRUE,	FALSE,	"CellWidth" },
    { TIFFTAG_CELLLENGTH,	 1, 1, TIFF_SHORT,	FIELD_IGNORE,
      TRUE,	FALSE,	"CellLength" },
    { TIFFTAG_FILLORDER,	 1, 1, TIFF_SHORT,	FIELD_FILLORDER,
      FALSE,	FALSE,	"FillOrder" },
    { TIFFTAG_DOCUMENTNAME,	-1,-1, TIFF_ASCII,	FIELD_DOCUMENTNAME,
      TRUE,	FALSE,	"DocumentName" },
    { TIFFTAG_IMAGEDESCRIPTION,	-1,-1, TIFF_ASCII,	FIELD_IMAGEDESCRIPTION,
      TRUE,	FALSE,	"ImageDescription" },
    { TIFFTAG_MAKE,		-1,-1, TIFF_ASCII,	FIELD_MAKE,
      TRUE,	FALSE,	"Make" },
    { TIFFTAG_MODEL,		-1,-1, TIFF_ASCII,	FIELD_MODEL,
      TRUE,	FALSE,	"Model" },
    { TIFFTAG_STRIPOFFSETS,	-1,-1, TIFF_LONG,	FIELD_STRIPOFFSETS,
      FALSE,	FALSE,	"StripOffsets" },
    { TIFFTAG_STRIPOFFSETS,	-1,-1, TIFF_SHORT,	FIELD_STRIPOFFSETS,
      FALSE,	FALSE,	"StripOffsets" },
    { TIFFTAG_ORIENTATION,	 1, 1, TIFF_SHORT,	FIELD_ORIENTATION,
      FALSE,	FALSE,	"Orientation" },
    { TIFFTAG_SAMPLESPERPIXEL,	 1, 1, TIFF_SHORT,	FIELD_SAMPLESPERPIXEL,
      FALSE,	FALSE,	"SamplesPerPixel" },
    { TIFFTAG_ROWSPERSTRIP,	 1, 1, TIFF_LONG,	FIELD_ROWSPERSTRIP,
      FALSE,	FALSE,	"RowsPerStrip" },
    { TIFFTAG_ROWSPERSTRIP,	 1, 1, TIFF_SHORT,	FIELD_ROWSPERSTRIP,
      FALSE,	FALSE,	"RowsPerStrip" },
    { TIFFTAG_STRIPBYTECOUNTS,	-1,-1, TIFF_LONG,	FIELD_STRIPBYTECOUNTS,
      FALSE,	FALSE,	"StripByteCounts" },
    { TIFFTAG_STRIPBYTECOUNTS,	-1,-1, TIFF_SHORT,	FIELD_STRIPBYTECOUNTS,
      FALSE,	FALSE,	"StripByteCounts" },
    { TIFFTAG_MINSAMPLEVALUE,	-2,-1, TIFF_SHORT,	FIELD_MINSAMPLEVALUE,
      TRUE,	FALSE,	"MinSampleValue" },
    { TIFFTAG_MAXSAMPLEVALUE,	-2,-1, TIFF_SHORT,	FIELD_MAXSAMPLEVALUE,
      TRUE,	FALSE,	"MaxSampleValue" },
    { TIFFTAG_XRESOLUTION,	 1, 1, TIFF_RATIONAL,	FIELD_RESOLUTION,
      FALSE,	FALSE,	"XResolution" },
    { TIFFTAG_YRESOLUTION,	 1, 1, TIFF_RATIONAL,	FIELD_RESOLUTION,
      FALSE,	FALSE,	"YResolution" },
    { TIFFTAG_PLANARCONFIG,	 1, 1, TIFF_SHORT,	FIELD_PLANARCONFIG,
      FALSE,	FALSE,	"PlanarConfiguration" },
    { TIFFTAG_PAGENAME,		-1,-1, TIFF_ASCII,	FIELD_PAGENAME,
      TRUE,	FALSE,	"PageName" },
    { TIFFTAG_XPOSITION,	 1, 1, TIFF_RATIONAL,	FIELD_POSITION,
      TRUE,	FALSE,	"XPosition" },
    { TIFFTAG_YPOSITION,	 1, 1, TIFF_RATIONAL,	FIELD_POSITION,
      TRUE,	FALSE,	"YPosition" },
    { TIFFTAG_FREEOFFSETS,	-1,-1, TIFF_LONG,	FIELD_IGNORE,
      FALSE,	FALSE,	"FreeOffsets" },
    { TIFFTAG_FREEBYTECOUNTS,	-1,-1, TIFF_LONG,	FIELD_IGNORE,
      FALSE,	FALSE,	"FreeByteCounts" },
    { TIFFTAG_GRAYRESPONSEUNIT,	 1, 1, TIFF_SHORT,	FIELD_IGNORE,
      TRUE,	FALSE,	"GrayResponseUnit" },
    { TIFFTAG_GRAYRESPONSECURVE,-1,-1, TIFF_SHORT,	FIELD_IGNORE,
      TRUE,	FALSE,	"GrayResponseCurve" },
    { TIFFTAG_RESOLUTIONUNIT,	 1, 1, TIFF_SHORT,	FIELD_RESOLUTIONUNIT,
      FALSE,	FALSE,	"ResolutionUnit" },
    { TIFFTAG_PAGENUMBER,	 2, 2, TIFF_SHORT,	FIELD_PAGENUMBER,
      TRUE,	FALSE,	"PageNumber" },
    { TIFFTAG_COLORRESPONSEUNIT, 1, 1, TIFF_SHORT,	FIELD_IGNORE,
      TRUE,	FALSE,	"ColorResponseUnit" },
    { TIFFTAG_TRANSFERFUNCTION,	-1,-1, TIFF_SHORT,	FIELD_TRANSFERFUNCTION,
      TRUE,	FALSE,	"TransferFunction" },
    { TIFFTAG_SOFTWARE,		-1,-1, TIFF_ASCII,	FIELD_CUSTOM,
      TRUE,	FALSE,	"Software" },
    { TIFFTAG_DATETIME,		20,20, TIFF_ASCII,	FIELD_DATETIME,
      TRUE,	FALSE,	"DateTime" },
    { TIFFTAG_ARTIST,		-1,-1, TIFF_ASCII,	FIELD_ARTIST,
      TRUE,	FALSE,	"Artist" },
    { TIFFTAG_HOSTCOMPUTER,	-1,-1, TIFF_ASCII,	FIELD_HOSTCOMPUTER,
      TRUE,	FALSE,	"HostComputer" },
    { TIFFTAG_WHITEPOINT,	 2, 2, TIFF_RATIONAL,FIELD_WHITEPOINT,
      TRUE,	FALSE,	"WhitePoint" },
    { TIFFTAG_PRIMARYCHROMATICITIES,6,6,TIFF_RATIONAL,FIELD_PRIMARYCHROMAS,
      TRUE,	FALSE,	"PrimaryChromaticities" },
    { TIFFTAG_COLORMAP,		-1,-1, TIFF_SHORT,	FIELD_COLORMAP,
      TRUE,	FALSE,	"ColorMap" },
    { TIFFTAG_HALFTONEHINTS,	 2, 2, TIFF_SHORT,	FIELD_HALFTONEHINTS,
      TRUE,	FALSE,	"HalftoneHints" },
    { TIFFTAG_TILEWIDTH,	 1, 1, TIFF_LONG,	FIELD_TILEDIMENSIONS,
      FALSE,	FALSE,	"TileWidth" },
    { TIFFTAG_TILEWIDTH,	 1, 1, TIFF_SHORT,	FIELD_TILEDIMENSIONS,
      FALSE,	FALSE,	"TileWidth" },
    { TIFFTAG_TILELENGTH,	 1, 1, TIFF_LONG,	FIELD_TILEDIMENSIONS,
      FALSE,	FALSE,	"TileLength" },
    { TIFFTAG_TILELENGTH,	 1, 1, TIFF_SHORT,	FIELD_TILEDIMENSIONS,
      FALSE,	FALSE,	"TileLength" },
    { TIFFTAG_TILEOFFSETS,	-1, 1, TIFF_LONG,	FIELD_STRIPOFFSETS,
      FALSE,	FALSE,	"TileOffsets" },
    { TIFFTAG_TILEBYTECOUNTS,	-1, 1, TIFF_LONG,	FIELD_STRIPBYTECOUNTS,
      FALSE,	FALSE,	"TileByteCounts" },
    { TIFFTAG_TILEBYTECOUNTS,	-1, 1, TIFF_SHORT,	FIELD_STRIPBYTECOUNTS,
      FALSE,	FALSE,	"TileByteCounts" },
    { TIFFTAG_SUBIFD,		-1,-1, TIFF_IFD,	FIELD_SUBIFD,
      TRUE,	TRUE,	"SubIFD" },
    { TIFFTAG_SUBIFD,		-1,-1, TIFF_LONG,	FIELD_SUBIFD,
      TRUE,	TRUE,	"SubIFD" },
    { TIFFTAG_INKSET,		 1, 1, TIFF_SHORT,	FIELD_INKSET,
      FALSE,	FALSE,	"InkSet" },
    { TIFFTAG_INKNAMES,		-1,-1, TIFF_ASCII,	FIELD_INKNAMES,
      TRUE,	TRUE,	"InkNames" },
    { TIFFTAG_NUMBEROFINKS,	 1, 1, TIFF_SHORT,	FIELD_NUMBEROFINKS,
      TRUE,	FALSE,	"NumberOfInks" },
    { TIFFTAG_DOTRANGE,		 2, 2, TIFF_SHORT,	FIELD_DOTRANGE,
      FALSE,	FALSE,	"DotRange" },
    { TIFFTAG_DOTRANGE,		 2, 2, TIFF_BYTE,	FIELD_DOTRANGE,
      FALSE,	FALSE,	"DotRange" },
    { TIFFTAG_TARGETPRINTER,	-1,-1, TIFF_ASCII,	FIELD_TARGETPRINTER,
      TRUE,	FALSE,	"TargetPrinter" },
    { TIFFTAG_EXTRASAMPLES,	-1,-1, TIFF_SHORT,	FIELD_EXTRASAMPLES,
      FALSE,	TRUE,	"ExtraSamples" },
/* XXX for bogus Adobe Photoshop v2.5 files */
    { TIFFTAG_EXTRASAMPLES,	-1,-1, TIFF_BYTE,	FIELD_EXTRASAMPLES,
      FALSE,	TRUE,	"ExtraSamples" },
    { TIFFTAG_SAMPLEFORMAT,	-1,-1, TIFF_SHORT,	FIELD_SAMPLEFORMAT,
      FALSE,	FALSE,	"SampleFormat" },
    { TIFFTAG_SMINSAMPLEVALUE,	-2,-1, TIFF_ANY,	FIELD_SMINSAMPLEVALUE,
      TRUE,	FALSE,	"SMinSampleValue" },
    { TIFFTAG_SMAXSAMPLEVALUE,	-2,-1, TIFF_ANY,	FIELD_SMAXSAMPLEVALUE,
      TRUE,	FALSE,	"SMaxSampleValue" },
    { TIFFTAG_YCBCRCOEFFICIENTS, 3, 3, TIFF_RATIONAL,	FIELD_YCBCRCOEFFICIENTS,
      FALSE,	FALSE,	"YCbCrCoefficients" },
    { TIFFTAG_YCBCRSUBSAMPLING,	 2, 2, TIFF_SHORT,	FIELD_YCBCRSUBSAMPLING,
      FALSE,	FALSE,	"YCbCrSubsampling" },
    { TIFFTAG_YCBCRPOSITIONING,	 1, 1, TIFF_SHORT,	FIELD_YCBCRPOSITIONING,
      FALSE,	FALSE,	"YCbCrPositioning" },
    { TIFFTAG_REFERENCEBLACKWHITE,6,6,TIFF_RATIONAL,	FIELD_REFBLACKWHITE,
      TRUE,	FALSE,	"ReferenceBlackWhite" },
/* XXX temporarily accept LONG for backwards compatibility */
    { TIFFTAG_REFERENCEBLACKWHITE,6,6,TIFF_LONG,	FIELD_REFBLACKWHITE,
      TRUE,	FALSE,	"ReferenceBlackWhite" },
    { TIFFTAG_XMLPACKET,	-1,-3, TIFF_BYTE,	FIELD_XMLPACKET,
      FALSE,	TRUE,	"XMLPacket" },
/* begin SGI tags */
    { TIFFTAG_MATTEING,		 1, 1, TIFF_SHORT,	FIELD_EXTRASAMPLES,
      FALSE,	FALSE,	"Matteing" },
    { TIFFTAG_DATATYPE,		-2,-1, TIFF_SHORT,	FIELD_SAMPLEFORMAT,
      FALSE,	FALSE,	"DataType" },
    { TIFFTAG_IMAGEDEPTH,	 1, 1, TIFF_LONG,	FIELD_IMAGEDEPTH,
      FALSE,	FALSE,	"ImageDepth" },
    { TIFFTAG_IMAGEDEPTH,	 1, 1, TIFF_SHORT,	FIELD_IMAGEDEPTH,
      FALSE,	FALSE,	"ImageDepth" },
    { TIFFTAG_TILEDEPTH,	 1, 1, TIFF_LONG,	FIELD_TILEDEPTH,
      FALSE,	FALSE,	"TileDepth" },
    { TIFFTAG_TILEDEPTH,	 1, 1, TIFF_SHORT,	FIELD_TILEDEPTH,
      FALSE,	FALSE,	"TileDepth" },
/* end SGI tags */
/* begin Pixar tags */
    { TIFFTAG_PIXAR_IMAGEFULLWIDTH,  1, 1, TIFF_LONG,	FIELD_IMAGEFULLWIDTH,
      TRUE,	FALSE,	"ImageFullWidth" },
    { TIFFTAG_PIXAR_IMAGEFULLLENGTH, 1, 1, TIFF_LONG,	FIELD_IMAGEFULLLENGTH,
      TRUE,	FALSE,	"ImageFullLength" },
    { TIFFTAG_PIXAR_TEXTUREFORMAT,  -1,-1, TIFF_ASCII,	FIELD_TEXTUREFORMAT,
      TRUE,	FALSE,	"TextureFormat" },
    { TIFFTAG_PIXAR_WRAPMODES,	    -1,-1, TIFF_ASCII,	FIELD_WRAPMODES,
      TRUE,	FALSE,	"TextureWrapModes" },
    { TIFFTAG_PIXAR_FOVCOT,	     1, 1, TIFF_FLOAT,	FIELD_FOVCOT,
      TRUE,	FALSE,	"FieldOfViewCotan" },
    { TIFFTAG_PIXAR_MATRIX_WORLDTOSCREEN,	16,16,	TIFF_FLOAT,
      FIELD_MATRIX_WORLDTOSCREEN,	TRUE,	FALSE,	"MatrixWorldToScreen" },
    { TIFFTAG_PIXAR_MATRIX_WORLDTOCAMERA,	16,16,	TIFF_FLOAT,
       FIELD_MATRIX_WORLDTOCAMERA,	TRUE,	FALSE,	"MatrixWorldToCamera" },
    { TIFFTAG_COPYRIGHT,	-1,-1, TIFF_ASCII,	FIELD_COPYRIGHT,
      TRUE,	FALSE,	"Copyright" },
/* end Pixar tags */
    { TIFFTAG_RICHTIFFIPTC, -1,-3, TIFF_LONG,   FIELD_RICHTIFFIPTC, 
      FALSE,    TRUE,   "RichTIFFIPTC" },
    { TIFFTAG_PHOTOSHOP,    -1,-3, TIFF_BYTE,   FIELD_PHOTOSHOP, 
      FALSE,    TRUE,   "Photoshop" },
    { TIFFTAG_ICCPROFILE,	-1,-3, TIFF_UNDEFINED,	FIELD_ICCPROFILE,
      FALSE,	TRUE,	"ICC Profile" },
    { TIFFTAG_STONITS,		 1, 1, TIFF_DOUBLE,	FIELD_STONITS,
      FALSE,	FALSE,	"StoNits" },
};
#define	N(a)	(sizeof (a) / sizeof (a[0]))

void
_TIFFSetupFieldInfo(TIFF* tif)
{
	if (tif->tif_fieldinfo) {
		int  i;

		for (i = 0; i < tif->tif_nfields; i++) 
		{
			TIFFFieldInfo *fld = tif->tif_fieldinfo[i];
			if (fld->field_bit == FIELD_CUSTOM && 
				strncmp("Tag ", fld->field_name, 4) == 0) 
				{
				_TIFFfree(fld->field_name);
				_TIFFfree(fld);
				}
		}   
      
		_TIFFfree(tif->tif_fieldinfo);
		tif->tif_nfields = 0;
	}
	_TIFFMergeFieldInfo(tif, tiffFieldInfo, N(tiffFieldInfo));
}

static int
tagCompare(const void* a, const void* b)
{
	const TIFFFieldInfo* ta = *(const TIFFFieldInfo**) a;
	const TIFFFieldInfo* tb = *(const TIFFFieldInfo**) b;
	/* NB: be careful of return values for 16-bit platforms */
	if (ta->field_tag != tb->field_tag)
		return (ta->field_tag < tb->field_tag ? -1 : 1);
	else
		return ((int)tb->field_type - (int)ta->field_type);
}

static int
tagNameCompare(const void* a, const void* b)
{
	const TIFFFieldInfo* ta = *(const TIFFFieldInfo**) a;
	const TIFFFieldInfo* tb = *(const TIFFFieldInfo**) b;

        return strcmp(ta->field_name, tb->field_name);
}

void
_TIFFMergeFieldInfo(TIFF* tif, const TIFFFieldInfo info[], int n)
{
	TIFFFieldInfo** tp;
	int i;

        tif->tif_foundfield = NULL;

	if (tif->tif_nfields > 0) {
		tif->tif_fieldinfo = (TIFFFieldInfo**)
		    _TIFFrealloc(tif->tif_fieldinfo,
			(tif->tif_nfields+n) * sizeof (TIFFFieldInfo*));
	} else {
		tif->tif_fieldinfo = (TIFFFieldInfo**)
		    _TIFFmalloc(n * sizeof (TIFFFieldInfo*));
	}
	assert(tif->tif_fieldinfo != NULL);
	tp = &tif->tif_fieldinfo[tif->tif_nfields];
	for (i = 0; i < n; i++)
		tp[i] = (TIFFFieldInfo*) &info[i];	/* XXX */

        /* Sort the field info by tag number */
        qsort(tif->tif_fieldinfo, (size_t) (tif->tif_nfields += n),
              sizeof (TIFFFieldInfo*), tagCompare);
}

void
_TIFFPrintFieldInfo(TIFF* tif, FILE* fd)
{
	int i;

	fprintf(fd, "%s: \n", tif->tif_name);
	for (i = 0; i < tif->tif_nfields; i++) {
		const TIFFFieldInfo* fip = tif->tif_fieldinfo[i];
		fprintf(fd, "field[%2d] %5lu, %2d, %2d, %d, %2d, %5s, %5s, %s\n"
			, i
			, (unsigned long) fip->field_tag
			, fip->field_readcount, fip->field_writecount
			, fip->field_type
			, fip->field_bit
			, fip->field_oktochange ? "TRUE" : "FALSE"
			, fip->field_passcount ? "TRUE" : "FALSE"
			, fip->field_name
		);
	}
}

/*
 * Return size of TIFFDataType in bytes
 */
int
TIFFDataWidth(TIFFDataType type)
{
	switch(type)
	{
	case 0:  /* nothing */
	case 1:  /* TIFF_BYTE */
	case 2:  /* TIFF_ASCII */
	case 6:  /* TIFF_SBYTE */
	case 7:  /* TIFF_UNDEFINED */
		return 1;
	case 3:  /* TIFF_SHORT */
	case 8:  /* TIFF_SSHORT */
		return 2;
	case 4:  /* TIFF_LONG */
	case 9:  /* TIFF_SLONG */
	case 11: /* TIFF_FLOAT */
        case 13: /* TIFF_IFD */
		return 4;
	case 5:  /* TIFF_RATIONAL */
	case 10: /* TIFF_SRATIONAL */
	case 12: /* TIFF_DOUBLE */
		return 8;
	default:
		return 0; /* will return 0 for unknown types */
	}
}

/*
 * Return nearest TIFFDataType to the sample type of an image.
 */
TIFFDataType
_TIFFSampleToTagType(TIFF* tif)
{
	uint32 bps = TIFFhowmany8(tif->tif_dir.td_bitspersample);

	switch (tif->tif_dir.td_sampleformat) {
	case SAMPLEFORMAT_IEEEFP:
		return (bps == 4 ? TIFF_FLOAT : TIFF_DOUBLE);
	case SAMPLEFORMAT_INT:
		return (bps <= 1 ? TIFF_SBYTE :
		    bps <= 2 ? TIFF_SSHORT : TIFF_SLONG);
	case SAMPLEFORMAT_UINT:
		return (bps <= 1 ? TIFF_BYTE :
		    bps <= 2 ? TIFF_SHORT : TIFF_LONG);
	case SAMPLEFORMAT_VOID:
		return (TIFF_UNDEFINED);
	}
	/*NOTREACHED*/
	return (TIFF_UNDEFINED);
}

const TIFFFieldInfo*
_TIFFFindFieldInfo(TIFF* tif, ttag_t tag, TIFFDataType dt)
{
	int i, n;

	if (tif->tif_foundfield && tif->tif_foundfield->field_tag == tag &&
	    (dt == TIFF_ANY || dt == tif->tif_foundfield->field_type))
		return (tif->tif_foundfield);
	/* NB: use sorted search (e.g. binary search) */
	if(dt != TIFF_ANY) {
            TIFFFieldInfo key = {0, 0, 0, 0, 0, 0, 0, 0};
            key.field_tag = tag;
            key.field_type = dt;
            return((const TIFFFieldInfo *) bsearch(&key, 
						   tif->tif_fieldinfo, 
						   tif->tif_nfields,
						   sizeof(TIFFFieldInfo), 
						   tagCompare));
        } else for (i = 0, n = tif->tif_nfields; i < n; i++) {
		const TIFFFieldInfo* fip = tif->tif_fieldinfo[i];
		if (fip->field_tag == tag &&
		    (dt == TIFF_ANY || fip->field_type == dt))
			return (tif->tif_foundfield = fip);
	}
	return ((const TIFFFieldInfo *)0);
}

const TIFFFieldInfo*
_TIFFFindFieldInfoByName(TIFF* tif, const char *field_name, TIFFDataType dt)
{
	int i, n;

	if (tif->tif_foundfield
	    && streq(tif->tif_foundfield->field_name, field_name)
	    && (dt == TIFF_ANY || dt == tif->tif_foundfield->field_type))
		return (tif->tif_foundfield);
	/* NB: use sorted search (e.g. binary search) */
	if(dt != TIFF_ANY) {
            TIFFFieldInfo key = {0, 0, 0, 0, 0, 0, 0, 0};
            key.field_name = (char *)field_name;
            key.field_type = dt;
            return((const TIFFFieldInfo *) bsearch(&key, 
						   tif->tif_fieldinfo, 
						   tif->tif_nfields,
						   sizeof(TIFFFieldInfo), 
						   tagNameCompare));
        } else for (i = 0, n = tif->tif_nfields; i < n; i++) {
		const TIFFFieldInfo* fip = tif->tif_fieldinfo[i];
		if (streq(fip->field_name, field_name) &&
		    (dt == TIFF_ANY || fip->field_type == dt))
			return (tif->tif_foundfield = fip);
	}
	return ((const TIFFFieldInfo *)0);
}

const TIFFFieldInfo*
_TIFFFieldWithTag(TIFF* tif, ttag_t tag)
{
	const TIFFFieldInfo* fip = _TIFFFindFieldInfo(tif, tag, TIFF_ANY);
	if (!fip) {
		TIFFError("TIFFFieldWithTag",
			  "Internal error, unknown tag 0x%x",
                          (unsigned int) tag);
		assert(fip != NULL);
		/*NOTREACHED*/
	}
	return (fip);
}

const TIFFFieldInfo*
_TIFFFieldWithName(TIFF* tif, const char *field_name)
{
	const TIFFFieldInfo* fip =
		_TIFFFindFieldInfoByName(tif, field_name, TIFF_ANY);
	if (!fip) {
		TIFFError("TIFFFieldWithName",
			  "Internal error, unknown tag %s", field_name);
		assert(fip != NULL);
		/*NOTREACHED*/
	}
	return (fip);
}

const TIFFFieldInfo*
_TIFFFindOrRegisterFieldInfo( TIFF *tif, ttag_t tag, TIFFDataType dt )

{
    const TIFFFieldInfo *fld;

    fld = _TIFFFindFieldInfo( tif, tag, dt );
    if( fld == NULL )
    {
        fld = _TIFFCreateAnonFieldInfo( tif, tag, dt );
        _TIFFMergeFieldInfo( tif, fld, 1 );
    }

    return fld;
}

TIFFFieldInfo*
_TIFFCreateAnonFieldInfo(TIFF *tif, ttag_t tag, TIFFDataType field_type)
{
    TIFFFieldInfo *fld;

    fld = (TIFFFieldInfo *) _TIFFmalloc(sizeof (TIFFFieldInfo));
    if (fld == NULL)
	return NULL;
    _TIFFmemset( fld, 0, sizeof(TIFFFieldInfo) );

    fld->field_tag = tag;
    fld->field_readcount = TIFF_VARIABLE;
    fld->field_writecount = TIFF_VARIABLE;
    fld->field_type = field_type;
    fld->field_bit = FIELD_CUSTOM;
    fld->field_oktochange = TRUE;
    fld->field_passcount = TRUE;
    fld->field_name = (char *) _TIFFmalloc(32);
    if (fld->field_name == NULL) {
	_TIFFfree(fld);
	return NULL;
    }

    /* note that this name is a special sign to TIFFClose() and
     * _TIFFSetupFieldInfo() to free the field
     */
    sprintf(fld->field_name, "Tag %d", (int) tag);

    return fld;    
}

/* vim: set ts=8 sts=8 sw=8 noet: */
