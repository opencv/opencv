/* $Id: tif_dirinfo.c,v 1.65.2.9 2010-06-09 21:15:27 bfriesen Exp $ */

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
#include <string.h>

/*
 * NB: NB: THIS ARRAY IS ASSUMED TO BE SORTED BY TAG.
 *       If a tag can have both LONG and SHORT types then the LONG must be
 *       placed before the SHORT for writing to work properly.
 *
 * NOTE: The second field (field_readcount) and third field (field_writecount)
 *       sometimes use the values TIFF_VARIABLE (-1), TIFF_VARIABLE2 (-3)
 *       and TIFFTAG_SPP (-2). The macros should be used but would throw off 
 *       the formatting of the code, so please interprete the -1, -2 and -3 
 *       values accordingly.
 */
static const TIFFFieldInfo
tiffFieldInfo[] = {
    { TIFFTAG_SUBFILETYPE,	 1, 1,	TIFF_LONG,	FIELD_SUBFILETYPE,
      1,	0,	"SubfileType" },
/* XXX SHORT for compatibility w/ old versions of the library */
    { TIFFTAG_SUBFILETYPE,	 1, 1,	TIFF_SHORT,	FIELD_SUBFILETYPE,
      1,	0,	"SubfileType" },
    { TIFFTAG_OSUBFILETYPE,	 1, 1,	TIFF_SHORT,	FIELD_SUBFILETYPE,
      1,	0,	"OldSubfileType" },
    { TIFFTAG_IMAGEWIDTH,	 1, 1,	TIFF_LONG,	FIELD_IMAGEDIMENSIONS,
      0,	0,	"ImageWidth" },
    { TIFFTAG_IMAGEWIDTH,	 1, 1,	TIFF_SHORT,	FIELD_IMAGEDIMENSIONS,
      0,	0,	"ImageWidth" },
    { TIFFTAG_IMAGELENGTH,	 1, 1,	TIFF_LONG,	FIELD_IMAGEDIMENSIONS,
      1,	0,	"ImageLength" },
    { TIFFTAG_IMAGELENGTH,	 1, 1,	TIFF_SHORT,	FIELD_IMAGEDIMENSIONS,
      1,	0,	"ImageLength" },
    { TIFFTAG_BITSPERSAMPLE,	-1,-1,	TIFF_SHORT,	FIELD_BITSPERSAMPLE,
      0,	0,	"BitsPerSample" },
/* XXX LONG for compatibility with some broken TIFF writers */
    { TIFFTAG_BITSPERSAMPLE,	-1,-1,	TIFF_LONG,	FIELD_BITSPERSAMPLE,
      0,	0,	"BitsPerSample" },
    { TIFFTAG_COMPRESSION,	-1, 1,	TIFF_SHORT,	FIELD_COMPRESSION,
      0,	0,	"Compression" },
/* XXX LONG for compatibility with some broken TIFF writers */
    { TIFFTAG_COMPRESSION,	-1, 1,	TIFF_LONG,	FIELD_COMPRESSION,
      0,	0,	"Compression" },
    { TIFFTAG_PHOTOMETRIC,	 1, 1,	TIFF_SHORT,	FIELD_PHOTOMETRIC,
      0,	0,	"PhotometricInterpretation" },
/* XXX LONG for compatibility with some broken TIFF writers */
    { TIFFTAG_PHOTOMETRIC,	 1, 1,	TIFF_LONG,	FIELD_PHOTOMETRIC,
      0,	0,	"PhotometricInterpretation" },
    { TIFFTAG_THRESHHOLDING,	 1, 1,	TIFF_SHORT,	FIELD_THRESHHOLDING,
      1,	0,	"Threshholding" },
    { TIFFTAG_CELLWIDTH,	 1, 1,	TIFF_SHORT,	FIELD_IGNORE,
      1,	0,	"CellWidth" },
    { TIFFTAG_CELLLENGTH,	 1, 1,	TIFF_SHORT,	FIELD_IGNORE,
      1,	0,	"CellLength" },
    { TIFFTAG_FILLORDER,	 1, 1,	TIFF_SHORT,	FIELD_FILLORDER,
      0,	0,	"FillOrder" },
    { TIFFTAG_DOCUMENTNAME,	-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"DocumentName" },
    { TIFFTAG_IMAGEDESCRIPTION,	-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"ImageDescription" },
    { TIFFTAG_MAKE,		-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"Make" },
    { TIFFTAG_MODEL,		-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"Model" },
    { TIFFTAG_STRIPOFFSETS,	-1,-1,	TIFF_LONG,	FIELD_STRIPOFFSETS,
      0,	0,	"StripOffsets" },
    { TIFFTAG_STRIPOFFSETS,	-1,-1,	TIFF_SHORT,	FIELD_STRIPOFFSETS,
      0,	0,	"StripOffsets" },
    { TIFFTAG_ORIENTATION,	 1, 1,	TIFF_SHORT,	FIELD_ORIENTATION,
      0,	0,	"Orientation" },
    { TIFFTAG_SAMPLESPERPIXEL,	 1, 1,	TIFF_SHORT,	FIELD_SAMPLESPERPIXEL,
      0,	0,	"SamplesPerPixel" },
    { TIFFTAG_ROWSPERSTRIP,	 1, 1,	TIFF_LONG,	FIELD_ROWSPERSTRIP,
      0,	0,	"RowsPerStrip" },
    { TIFFTAG_ROWSPERSTRIP,	 1, 1,	TIFF_SHORT,	FIELD_ROWSPERSTRIP,
      0,	0,	"RowsPerStrip" },
    { TIFFTAG_STRIPBYTECOUNTS,	-1,-1,	TIFF_LONG,	FIELD_STRIPBYTECOUNTS,
      0,	0,	"StripByteCounts" },
    { TIFFTAG_STRIPBYTECOUNTS,	-1,-1,	TIFF_SHORT,	FIELD_STRIPBYTECOUNTS,
      0,	0,	"StripByteCounts" },
    { TIFFTAG_MINSAMPLEVALUE,	-2,-1,	TIFF_SHORT,	FIELD_MINSAMPLEVALUE,
      1,	0,	"MinSampleValue" },
    { TIFFTAG_MAXSAMPLEVALUE,	-2,-1,	TIFF_SHORT,	FIELD_MAXSAMPLEVALUE,
      1,	0,	"MaxSampleValue" },
    { TIFFTAG_XRESOLUTION,	 1, 1,	TIFF_RATIONAL,	FIELD_RESOLUTION,
      1,	0,	"XResolution" },
    { TIFFTAG_YRESOLUTION,	 1, 1,	TIFF_RATIONAL,	FIELD_RESOLUTION,
      1,	0,	"YResolution" },
    { TIFFTAG_PLANARCONFIG,	 1, 1,	TIFF_SHORT,	FIELD_PLANARCONFIG,
      0,	0,	"PlanarConfiguration" },
    { TIFFTAG_PAGENAME,		-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"PageName" },
    { TIFFTAG_XPOSITION,	 1, 1,	TIFF_RATIONAL,	FIELD_POSITION,
      1,	0,	"XPosition" },
    { TIFFTAG_YPOSITION,	 1, 1,	TIFF_RATIONAL,	FIELD_POSITION,
      1,	0,	"YPosition" },
    { TIFFTAG_FREEOFFSETS,	-1,-1,	TIFF_LONG,	FIELD_IGNORE,
      0,	0,	"FreeOffsets" },
    { TIFFTAG_FREEBYTECOUNTS,	-1,-1,	TIFF_LONG,	FIELD_IGNORE,
      0,	0,	"FreeByteCounts" },
    { TIFFTAG_GRAYRESPONSEUNIT,	 1, 1,	TIFF_SHORT,	FIELD_IGNORE,
      1,	0,	"GrayResponseUnit" },
    { TIFFTAG_GRAYRESPONSECURVE,-1,-1,	TIFF_SHORT,	FIELD_IGNORE,
      1,	0,	"GrayResponseCurve" },
    { TIFFTAG_RESOLUTIONUNIT,	 1, 1,	TIFF_SHORT,	FIELD_RESOLUTIONUNIT,
      1,	0,	"ResolutionUnit" },
    { TIFFTAG_PAGENUMBER,	 2, 2,	TIFF_SHORT,	FIELD_PAGENUMBER,
      1,	0,	"PageNumber" },
    { TIFFTAG_COLORRESPONSEUNIT, 1, 1,	TIFF_SHORT,	FIELD_IGNORE,
      1,	0,	"ColorResponseUnit" },
    { TIFFTAG_TRANSFERFUNCTION,	-1,-1,	TIFF_SHORT,	FIELD_TRANSFERFUNCTION,
      1,	0,	"TransferFunction" },
    { TIFFTAG_SOFTWARE,		-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"Software" },
    { TIFFTAG_DATETIME,		20,20,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"DateTime" },
    { TIFFTAG_ARTIST,		-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"Artist" },
    { TIFFTAG_HOSTCOMPUTER,	-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"HostComputer" },
    { TIFFTAG_WHITEPOINT,	 2, 2,	TIFF_RATIONAL,	FIELD_CUSTOM,
      1,	0,	"WhitePoint" },
    { TIFFTAG_PRIMARYCHROMATICITIES,6,6,TIFF_RATIONAL,	FIELD_CUSTOM,
      1,	0,	"PrimaryChromaticities" },
    { TIFFTAG_COLORMAP,		-1,-1,	TIFF_SHORT,	FIELD_COLORMAP,
      1,	0,	"ColorMap" },
    { TIFFTAG_HALFTONEHINTS,	 2, 2,	TIFF_SHORT,	FIELD_HALFTONEHINTS,
      1,	0,	"HalftoneHints" },
    { TIFFTAG_TILEWIDTH,	 1, 1,	TIFF_LONG,	FIELD_TILEDIMENSIONS,
      0,	0,	"TileWidth" },
    { TIFFTAG_TILEWIDTH,	 1, 1,	TIFF_SHORT,	FIELD_TILEDIMENSIONS,
      0,	0,	"TileWidth" },
    { TIFFTAG_TILELENGTH,	 1, 1,	TIFF_LONG,	FIELD_TILEDIMENSIONS,
      0,	0,	"TileLength" },
    { TIFFTAG_TILELENGTH,	 1, 1,	TIFF_SHORT,	FIELD_TILEDIMENSIONS,
      0,	0,	"TileLength" },
    { TIFFTAG_TILEOFFSETS,	-1, 1,	TIFF_LONG,	FIELD_STRIPOFFSETS,
      0,	0,	"TileOffsets" },
    { TIFFTAG_TILEBYTECOUNTS,	-1, 1,	TIFF_LONG,	FIELD_STRIPBYTECOUNTS,
      0,	0,	"TileByteCounts" },
    { TIFFTAG_TILEBYTECOUNTS,	-1, 1,	TIFF_SHORT,	FIELD_STRIPBYTECOUNTS,
      0,	0,	"TileByteCounts" },
    { TIFFTAG_SUBIFD,		-1,-1,	TIFF_IFD,	FIELD_SUBIFD,
      1,	1,	"SubIFD" },
    { TIFFTAG_SUBIFD,		-1,-1,	TIFF_LONG,	FIELD_SUBIFD,
      1,	1,	"SubIFD" },
    { TIFFTAG_INKSET,		 1, 1,	TIFF_SHORT,	FIELD_CUSTOM,
      0,	0,	"InkSet" },
    { TIFFTAG_INKNAMES,		-1,-1,	TIFF_ASCII,	FIELD_INKNAMES,
      1,	1,	"InkNames" },
    { TIFFTAG_NUMBEROFINKS,	 1, 1,	TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"NumberOfInks" },
    { TIFFTAG_DOTRANGE,		 2, 2,	TIFF_SHORT,	FIELD_CUSTOM,
      0,	0,	"DotRange" },
    { TIFFTAG_DOTRANGE,		 2, 2,	TIFF_BYTE,	FIELD_CUSTOM,
      0,	0,	"DotRange" },
    { TIFFTAG_TARGETPRINTER,	-1,-1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"TargetPrinter" },
    { TIFFTAG_EXTRASAMPLES,	-1,-1,	TIFF_SHORT,	FIELD_EXTRASAMPLES,
      0,	1,	"ExtraSamples" },
/* XXX for bogus Adobe Photoshop v2.5 files */
    { TIFFTAG_EXTRASAMPLES,	-1,-1,	TIFF_BYTE,	FIELD_EXTRASAMPLES,
      0,	1,	"ExtraSamples" },
    { TIFFTAG_SAMPLEFORMAT,	-1,-1,	TIFF_SHORT,	FIELD_SAMPLEFORMAT,
      0,	0,	"SampleFormat" },
    { TIFFTAG_SMINSAMPLEVALUE,	-2,-1,	TIFF_ANY,	FIELD_SMINSAMPLEVALUE,
      1,	0,	"SMinSampleValue" },
    { TIFFTAG_SMAXSAMPLEVALUE,	-2,-1,	TIFF_ANY,	FIELD_SMAXSAMPLEVALUE,
      1,	0,	"SMaxSampleValue" },
    { TIFFTAG_CLIPPATH,		-1, -3, TIFF_BYTE,	FIELD_CUSTOM,
      0,	1,	"ClipPath" },
    { TIFFTAG_XCLIPPATHUNITS,	 1, 1,	TIFF_SLONG,	FIELD_CUSTOM,
      0,	0,	"XClipPathUnits" },
    { TIFFTAG_XCLIPPATHUNITS,	 1, 1,	TIFF_SSHORT,	FIELD_CUSTOM,
      0,	0,	"XClipPathUnits" },
    { TIFFTAG_XCLIPPATHUNITS,	 1, 1,	TIFF_SBYTE,	FIELD_CUSTOM,
      0,	0,	"XClipPathUnits" },
    { TIFFTAG_YCLIPPATHUNITS,	 1, 1,	TIFF_SLONG,	FIELD_CUSTOM,
      0,	0,	"YClipPathUnits" },
    { TIFFTAG_YCLIPPATHUNITS,	 1, 1,	TIFF_SSHORT,	FIELD_CUSTOM,
      0,	0,	"YClipPathUnits" },
    { TIFFTAG_YCLIPPATHUNITS,	 1, 1,	TIFF_SBYTE,	FIELD_CUSTOM,
      0,	0,	"YClipPathUnits" },
    { TIFFTAG_YCBCRCOEFFICIENTS, 3, 3,	TIFF_RATIONAL,	FIELD_CUSTOM,
      0,	0,	"YCbCrCoefficients" },
    { TIFFTAG_YCBCRSUBSAMPLING,	 2, 2,	TIFF_SHORT,	FIELD_YCBCRSUBSAMPLING,
      0,	0,	"YCbCrSubsampling" },
    { TIFFTAG_YCBCRPOSITIONING,	 1, 1,	TIFF_SHORT,	FIELD_YCBCRPOSITIONING,
      0,	0,	"YCbCrPositioning" },
    { TIFFTAG_REFERENCEBLACKWHITE, 6, 6, TIFF_RATIONAL,	FIELD_REFBLACKWHITE,
      1,	0,	"ReferenceBlackWhite" },
/* XXX temporarily accept LONG for backwards compatibility */
    { TIFFTAG_REFERENCEBLACKWHITE, 6, 6, TIFF_LONG,	FIELD_REFBLACKWHITE,
      1,	0,	"ReferenceBlackWhite" },
    { TIFFTAG_XMLPACKET,	-3,-3,	TIFF_BYTE,	FIELD_CUSTOM,
      0,	1,	"XMLPacket" },
/* begin SGI tags */
    { TIFFTAG_MATTEING,		 1, 1,	TIFF_SHORT,	FIELD_EXTRASAMPLES,
      0,	0,	"Matteing" },
    { TIFFTAG_DATATYPE,		-2,-1,	TIFF_SHORT,	FIELD_SAMPLEFORMAT,
      0,	0,	"DataType" },
    { TIFFTAG_IMAGEDEPTH,	 1, 1,	TIFF_LONG,	FIELD_IMAGEDEPTH,
      0,	0,	"ImageDepth" },
    { TIFFTAG_IMAGEDEPTH,	 1, 1,	TIFF_SHORT,	FIELD_IMAGEDEPTH,
      0,	0,	"ImageDepth" },
    { TIFFTAG_TILEDEPTH,	 1, 1,	TIFF_LONG,	FIELD_TILEDEPTH,
      0,	0,	"TileDepth" },
    { TIFFTAG_TILEDEPTH,	 1, 1,	TIFF_SHORT,	FIELD_TILEDEPTH,
      0,	0,	"TileDepth" },
/* end SGI tags */
/* begin Pixar tags */
    { TIFFTAG_PIXAR_IMAGEFULLWIDTH,  1, 1, TIFF_LONG,	FIELD_CUSTOM,
      1,	0,	"ImageFullWidth" },
    { TIFFTAG_PIXAR_IMAGEFULLLENGTH, 1, 1, TIFF_LONG,	FIELD_CUSTOM,
      1,	0,	"ImageFullLength" },
    { TIFFTAG_PIXAR_TEXTUREFORMAT,  -1, -1, TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"TextureFormat" },
    { TIFFTAG_PIXAR_WRAPMODES,	    -1, -1, TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"TextureWrapModes" },
    { TIFFTAG_PIXAR_FOVCOT,	     1, 1, TIFF_FLOAT,	FIELD_CUSTOM,
      1,	0,	"FieldOfViewCotangent" },
    { TIFFTAG_PIXAR_MATRIX_WORLDTOSCREEN,	16,16,	TIFF_FLOAT,
      FIELD_CUSTOM,	1,	0,	"MatrixWorldToScreen" },
    { TIFFTAG_PIXAR_MATRIX_WORLDTOCAMERA,	16,16,	TIFF_FLOAT,
       FIELD_CUSTOM,	1,	0,	"MatrixWorldToCamera" },
    { TIFFTAG_COPYRIGHT,	-1, -1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"Copyright" },
/* end Pixar tags */
    { TIFFTAG_RICHTIFFIPTC, -3, -3,	TIFF_LONG,	FIELD_CUSTOM, 
      0,    1,   "RichTIFFIPTC" },
    { TIFFTAG_PHOTOSHOP,    -3, -3,	TIFF_BYTE,	FIELD_CUSTOM, 
      0,    1,   "Photoshop" },
    { TIFFTAG_EXIFIFD,		1, 1,	TIFF_LONG,	FIELD_CUSTOM,
      0,	0,	"EXIFIFDOffset" },
    { TIFFTAG_ICCPROFILE,	-3, -3,	TIFF_UNDEFINED,	FIELD_CUSTOM,
      0,	1,	"ICC Profile" },
    { TIFFTAG_GPSIFD,		1, 1,	TIFF_LONG,	FIELD_CUSTOM,
      0,	0,	"GPSIFDOffset" },
    { TIFFTAG_STONITS,		 1, 1,	TIFF_DOUBLE,	FIELD_CUSTOM,
      0,	0,	"StoNits" },
    { TIFFTAG_INTEROPERABILITYIFD, 1, 1, TIFF_LONG,	FIELD_CUSTOM,
      0,	0,	"InteroperabilityIFDOffset" },
/* begin DNG tags */
    { TIFFTAG_DNGVERSION,	4, 4,	TIFF_BYTE,	FIELD_CUSTOM, 
      0,	0,	"DNGVersion" },
    { TIFFTAG_DNGBACKWARDVERSION, 4, 4,	TIFF_BYTE,	FIELD_CUSTOM, 
      0,	0,	"DNGBackwardVersion" },
    { TIFFTAG_UNIQUECAMERAMODEL,    -1, -1, TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"UniqueCameraModel" },
    { TIFFTAG_LOCALIZEDCAMERAMODEL, -1, -1, TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"LocalizedCameraModel" },
    { TIFFTAG_LOCALIZEDCAMERAMODEL, -1, -1, TIFF_BYTE,	FIELD_CUSTOM,
      1,	1,	"LocalizedCameraModel" },
    { TIFFTAG_CFAPLANECOLOR,	-1, -1,	TIFF_BYTE,	FIELD_CUSTOM, 
      0,	1,	"CFAPlaneColor" },
    { TIFFTAG_CFALAYOUT,	1, 1,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"CFALayout" },
    { TIFFTAG_LINEARIZATIONTABLE, -1, -1, TIFF_SHORT,	FIELD_CUSTOM, 
      0,	1,	"LinearizationTable" },
    { TIFFTAG_BLACKLEVELREPEATDIM, 2, 2, TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"BlackLevelRepeatDim" },
    { TIFFTAG_BLACKLEVEL,	-1, -1,	TIFF_LONG,	FIELD_CUSTOM, 
      0,	1,	"BlackLevel" },
    { TIFFTAG_BLACKLEVEL,	-1, -1,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	1,	"BlackLevel" },
    { TIFFTAG_BLACKLEVEL,	-1, -1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	1,	"BlackLevel" },
    { TIFFTAG_BLACKLEVELDELTAH,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"BlackLevelDeltaH" },
    { TIFFTAG_BLACKLEVELDELTAV,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"BlackLevelDeltaV" },
    { TIFFTAG_WHITELEVEL,	-2, -2,	TIFF_LONG,	FIELD_CUSTOM, 
      0,	0,	"WhiteLevel" },
    { TIFFTAG_WHITELEVEL,	-2, -2,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"WhiteLevel" },
    { TIFFTAG_DEFAULTSCALE,	2, 2,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"DefaultScale" },
    { TIFFTAG_BESTQUALITYSCALE,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"BestQualityScale" },
    { TIFFTAG_DEFAULTCROPORIGIN,	2, 2,	TIFF_LONG,	FIELD_CUSTOM, 
      0,	0,	"DefaultCropOrigin" },
    { TIFFTAG_DEFAULTCROPORIGIN,	2, 2,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"DefaultCropOrigin" },
    { TIFFTAG_DEFAULTCROPORIGIN,	2, 2,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"DefaultCropOrigin" },
    { TIFFTAG_DEFAULTCROPSIZE,	2, 2,	TIFF_LONG,	FIELD_CUSTOM, 
      0,	0,	"DefaultCropSize" },
    { TIFFTAG_DEFAULTCROPSIZE,	2, 2,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"DefaultCropSize" },
    { TIFFTAG_DEFAULTCROPSIZE,	2, 2,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"DefaultCropSize" },
    { TIFFTAG_COLORMATRIX1,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"ColorMatrix1" },
    { TIFFTAG_COLORMATRIX2,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"ColorMatrix2" },
    { TIFFTAG_CAMERACALIBRATION1,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"CameraCalibration1" },
    { TIFFTAG_CAMERACALIBRATION2,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"CameraCalibration2" },
    { TIFFTAG_REDUCTIONMATRIX1,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"ReductionMatrix1" },
    { TIFFTAG_REDUCTIONMATRIX2,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"ReductionMatrix2" },
    { TIFFTAG_ANALOGBALANCE,	-1, -1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	1,	"AnalogBalance" },
    { TIFFTAG_ASSHOTNEUTRAL,	-1, -1,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	1,	"AsShotNeutral" },
    { TIFFTAG_ASSHOTNEUTRAL,	-1, -1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	1,	"AsShotNeutral" },
    { TIFFTAG_ASSHOTWHITEXY,	2, 2,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"AsShotWhiteXY" },
    { TIFFTAG_BASELINEEXPOSURE,	1, 1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	0,	"BaselineExposure" },
    { TIFFTAG_BASELINENOISE,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"BaselineNoise" },
    { TIFFTAG_BASELINESHARPNESS,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"BaselineSharpness" },
    { TIFFTAG_BAYERGREENSPLIT,	1, 1,	TIFF_LONG,	FIELD_CUSTOM, 
      0,	0,	"BayerGreenSplit" },
    { TIFFTAG_LINEARRESPONSELIMIT,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"LinearResponseLimit" },
    { TIFFTAG_CAMERASERIALNUMBER,    -1, -1, TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"CameraSerialNumber" },
    { TIFFTAG_LENSINFO,	4, 4,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"LensInfo" },
    { TIFFTAG_CHROMABLURRADIUS,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"ChromaBlurRadius" },
    { TIFFTAG_ANTIALIASSTRENGTH,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"AntiAliasStrength" },
    { TIFFTAG_SHADOWSCALE,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      0,	0,	"ShadowScale" },
    { TIFFTAG_DNGPRIVATEDATA,    -1, -1, TIFF_BYTE,	FIELD_CUSTOM,
      0,	1,	"DNGPrivateData" },
    { TIFFTAG_MAKERNOTESAFETY,	1, 1,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"MakerNoteSafety" },
    { TIFFTAG_CALIBRATIONILLUMINANT1,	1, 1,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"CalibrationIlluminant1" },
    { TIFFTAG_CALIBRATIONILLUMINANT2,	1, 1,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"CalibrationIlluminant2" },
    { TIFFTAG_RAWDATAUNIQUEID,	16, 16,	TIFF_BYTE,	FIELD_CUSTOM, 
      0,	0,	"RawDataUniqueID" },
    { TIFFTAG_ORIGINALRAWFILENAME,    -1, -1, TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"OriginalRawFileName" },
    { TIFFTAG_ORIGINALRAWFILENAME,    -1, -1, TIFF_BYTE,	FIELD_CUSTOM,
      1,	1,	"OriginalRawFileName" },
    { TIFFTAG_ORIGINALRAWFILEDATA,    -1, -1, TIFF_UNDEFINED,	FIELD_CUSTOM,
      0,	1,	"OriginalRawFileData" },
    { TIFFTAG_ACTIVEAREA,	4, 4,	TIFF_LONG,	FIELD_CUSTOM, 
      0,	0,	"ActiveArea" },
    { TIFFTAG_ACTIVEAREA,	4, 4,	TIFF_SHORT,	FIELD_CUSTOM, 
      0,	0,	"ActiveArea" },
    { TIFFTAG_MASKEDAREAS,	-1, -1,	TIFF_LONG,	FIELD_CUSTOM, 
      0,	1,	"MaskedAreas" },
    { TIFFTAG_ASSHOTICCPROFILE,    -1, -1, TIFF_UNDEFINED,	FIELD_CUSTOM,
      0,	1,	"AsShotICCProfile" },
    { TIFFTAG_ASSHOTPREPROFILEMATRIX,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"AsShotPreProfileMatrix" },
    { TIFFTAG_CURRENTICCPROFILE,    -1, -1, TIFF_UNDEFINED,	FIELD_CUSTOM,
      0,	1,	"CurrentICCProfile" },
    { TIFFTAG_CURRENTPREPROFILEMATRIX,	-1, -1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      0,	1,	"CurrentPreProfileMatrix" },
/* end DNG tags */
};

static const TIFFFieldInfo
exifFieldInfo[] = {
    { EXIFTAG_EXPOSURETIME,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"ExposureTime" },
    { EXIFTAG_FNUMBER,		1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"FNumber" },
    { EXIFTAG_EXPOSUREPROGRAM,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"ExposureProgram" },
    { EXIFTAG_SPECTRALSENSITIVITY,    -1, -1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"SpectralSensitivity" },
    { EXIFTAG_ISOSPEEDRATINGS,  -1, -1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	1,	"ISOSpeedRatings" },
    { EXIFTAG_OECF,	-1, -1,			TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	1,	"OptoelectricConversionFactor" },
    { EXIFTAG_EXIFVERSION,	4, 4,		TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	0,	"ExifVersion" },
    { EXIFTAG_DATETIMEORIGINAL,	20, 20,		TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"DateTimeOriginal" },
    { EXIFTAG_DATETIMEDIGITIZED, 20, 20,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"DateTimeDigitized" },
    { EXIFTAG_COMPONENTSCONFIGURATION,	 4, 4,	TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	0,	"ComponentsConfiguration" },
    { EXIFTAG_COMPRESSEDBITSPERPIXEL,	 1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM,
      1,	0,	"CompressedBitsPerPixel" },
    { EXIFTAG_SHUTTERSPEEDVALUE,	1, 1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      1,	0,	"ShutterSpeedValue" },
    { EXIFTAG_APERTUREVALUE,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"ApertureValue" },
    { EXIFTAG_BRIGHTNESSVALUE,	1, 1,		TIFF_SRATIONAL,	FIELD_CUSTOM, 
      1,	0,	"BrightnessValue" },
    { EXIFTAG_EXPOSUREBIASVALUE,	1, 1,	TIFF_SRATIONAL,	FIELD_CUSTOM, 
      1,	0,	"ExposureBiasValue" },
    { EXIFTAG_MAXAPERTUREVALUE,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"MaxApertureValue" },
    { EXIFTAG_SUBJECTDISTANCE,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"SubjectDistance" },
    { EXIFTAG_METERINGMODE,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"MeteringMode" },
    { EXIFTAG_LIGHTSOURCE,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"LightSource" },
    { EXIFTAG_FLASH,	1, 1,			TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"Flash" },
    { EXIFTAG_FOCALLENGTH,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"FocalLength" },
    { EXIFTAG_SUBJECTAREA,	-1, -1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	1,	"SubjectArea" },
    { EXIFTAG_MAKERNOTE,	-1, -1,		TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	1,	"MakerNote" },
    { EXIFTAG_USERCOMMENT,	-1, -1,		TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	1,	"UserComment" },
    { EXIFTAG_SUBSECTIME,    -1, -1,		TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"SubSecTime" },
    { EXIFTAG_SUBSECTIMEORIGINAL, -1, -1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"SubSecTimeOriginal" },
    { EXIFTAG_SUBSECTIMEDIGITIZED,-1, -1,	TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"SubSecTimeDigitized" },
    { EXIFTAG_FLASHPIXVERSION,	4, 4,		TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	0,	"FlashpixVersion" },
    { EXIFTAG_COLORSPACE,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"ColorSpace" },
    { EXIFTAG_PIXELXDIMENSION,	1, 1,		TIFF_LONG,	FIELD_CUSTOM,
      1,	0,	"PixelXDimension" },
    { EXIFTAG_PIXELXDIMENSION,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"PixelXDimension" },
    { EXIFTAG_PIXELYDIMENSION,	1, 1,		TIFF_LONG,	FIELD_CUSTOM,
      1,	0,	"PixelYDimension" },
    { EXIFTAG_PIXELYDIMENSION,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"PixelYDimension" },
    { EXIFTAG_RELATEDSOUNDFILE,	13, 13,		TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"RelatedSoundFile" },
    { EXIFTAG_FLASHENERGY,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"FlashEnergy" },
    { EXIFTAG_SPATIALFREQUENCYRESPONSE,	-1, -1,	TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	1,	"SpatialFrequencyResponse" },
    { EXIFTAG_FOCALPLANEXRESOLUTION,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"FocalPlaneXResolution" },
    { EXIFTAG_FOCALPLANEYRESOLUTION,	1, 1,	TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"FocalPlaneYResolution" },
    { EXIFTAG_FOCALPLANERESOLUTIONUNIT,	1, 1,	TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"FocalPlaneResolutionUnit" },
    { EXIFTAG_SUBJECTLOCATION,	2, 2,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"SubjectLocation" },
    { EXIFTAG_EXPOSUREINDEX,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"ExposureIndex" },
    { EXIFTAG_SENSINGMETHOD,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"SensingMethod" },
    { EXIFTAG_FILESOURCE,	1, 1,		TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	0,	"FileSource" },
    { EXIFTAG_SCENETYPE,	1, 1,		TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	0,	"SceneType" },
    { EXIFTAG_CFAPATTERN,	-1, -1,		TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	1,	"CFAPattern" },
    { EXIFTAG_CUSTOMRENDERED,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"CustomRendered" },
    { EXIFTAG_EXPOSUREMODE,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"ExposureMode" },
    { EXIFTAG_WHITEBALANCE,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"WhiteBalance" },
    { EXIFTAG_DIGITALZOOMRATIO,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"DigitalZoomRatio" },
    { EXIFTAG_FOCALLENGTHIN35MMFILM, 1, 1,	TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"FocalLengthIn35mmFilm" },
    { EXIFTAG_SCENECAPTURETYPE,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"SceneCaptureType" },
    { EXIFTAG_GAINCONTROL,	1, 1,		TIFF_RATIONAL,	FIELD_CUSTOM, 
      1,	0,	"GainControl" },
    { EXIFTAG_CONTRAST,		1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"Contrast" },
    { EXIFTAG_SATURATION,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"Saturation" },
    { EXIFTAG_SHARPNESS,	1, 1,		TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"Sharpness" },
    { EXIFTAG_DEVICESETTINGDESCRIPTION,	-1, -1,	TIFF_UNDEFINED,	FIELD_CUSTOM,
      1,	1,	"DeviceSettingDescription" },
    { EXIFTAG_SUBJECTDISTANCERANGE, 1, 1,	TIFF_SHORT,	FIELD_CUSTOM,
      1,	0,	"SubjectDistanceRange" },
    { EXIFTAG_IMAGEUNIQUEID,	33, 33,		TIFF_ASCII,	FIELD_CUSTOM,
      1,	0,	"ImageUniqueID" }
};

const TIFFFieldInfo *
_TIFFGetFieldInfo(size_t *size)
{
	*size = TIFFArrayCount(tiffFieldInfo);
	return tiffFieldInfo;
}

const TIFFFieldInfo *
_TIFFGetExifFieldInfo(size_t *size)
{
	*size = TIFFArrayCount(exifFieldInfo);
	return exifFieldInfo;
}

void
_TIFFSetupFieldInfo(TIFF* tif, const TIFFFieldInfo info[], size_t n)
{
	if (tif->tif_fieldinfo) {
		size_t  i;

		for (i = 0; i < tif->tif_nfields; i++) 
		{
			TIFFFieldInfo *fld = tif->tif_fieldinfo[i];
			if (fld->field_bit == FIELD_CUSTOM && 
				strncmp("Tag ", fld->field_name, 4) == 0) {
					_TIFFfree(fld->field_name);
					_TIFFfree(fld);
				}
		}   
      
		_TIFFfree(tif->tif_fieldinfo);
		tif->tif_nfields = 0;
	}
	if (!_TIFFMergeFieldInfo(tif, info, n))
	{
		TIFFErrorExt(tif->tif_clientdata, "_TIFFSetupFieldInfo",
			     "Setting up field info failed");
	}
}

static int
tagCompare(const void* a, const void* b)
{
	const TIFFFieldInfo* ta = *(const TIFFFieldInfo**) a;
	const TIFFFieldInfo* tb = *(const TIFFFieldInfo**) b;
	/* NB: be careful of return values for 16-bit platforms */
	if (ta->field_tag != tb->field_tag)
		return (int)ta->field_tag - (int)tb->field_tag;
	else
		return (ta->field_type == TIFF_ANY) ?
			0 : ((int)tb->field_type - (int)ta->field_type);
}

static int
tagNameCompare(const void* a, const void* b)
{
	const TIFFFieldInfo* ta = *(const TIFFFieldInfo**) a;
	const TIFFFieldInfo* tb = *(const TIFFFieldInfo**) b;
	int ret = strcmp(ta->field_name, tb->field_name);

	if (ret)
		return ret;
	else
		return (ta->field_type == TIFF_ANY) ?
			0 : ((int)tb->field_type - (int)ta->field_type);
}

void
TIFFMergeFieldInfo(TIFF* tif, const TIFFFieldInfo info[], int n)
{
	if (_TIFFMergeFieldInfo(tif, info, n) < 0)
	{
		TIFFErrorExt(tif->tif_clientdata, "TIFFMergeFieldInfo",
			     "Merging block of %d fields failed", n);
	}
}

int
_TIFFMergeFieldInfo(TIFF* tif, const TIFFFieldInfo info[], int n)
{
	static const char module[] = "_TIFFMergeFieldInfo";
	static const char reason[] = "for field info array";
	TIFFFieldInfo** tp;
	int i;

        tif->tif_foundfield = NULL;

	if (tif->tif_nfields > 0) {
		tif->tif_fieldinfo = (TIFFFieldInfo**)
			_TIFFCheckRealloc(tif, tif->tif_fieldinfo,
					  (tif->tif_nfields + n),
					  sizeof (TIFFFieldInfo*), reason);
	} else {
		tif->tif_fieldinfo = (TIFFFieldInfo**)
			_TIFFCheckMalloc(tif, n, sizeof (TIFFFieldInfo*),
					 reason);
	}
	if (!tif->tif_fieldinfo) {
		TIFFErrorExt(tif->tif_clientdata, module,
			     "Failed to allocate field info array");
		return 0;
	}
	tp = tif->tif_fieldinfo + tif->tif_nfields;
	for (i = 0; i < n; i++)
        {
            const TIFFFieldInfo *fip =
                _TIFFFindFieldInfo(tif, info[i].field_tag, info[i].field_type);

            /* only add definitions that aren't already present */
            if (!fip) {
                *tp++ = (TIFFFieldInfo*) (info + i);
                tif->tif_nfields++;
            }
        }

        /* Sort the field info by tag number */
        qsort(tif->tif_fieldinfo, tif->tif_nfields,
	      sizeof (TIFFFieldInfo*), tagCompare);

	return n;
}

void
_TIFFPrintFieldInfo(TIFF* tif, FILE* fd)
{
	size_t i;

	fprintf(fd, "%s: \n", tif->tif_name);
	for (i = 0; i < tif->tif_nfields; i++) {
		const TIFFFieldInfo* fip = tif->tif_fieldinfo[i];
		fprintf(fd, "field[%2d] %5lu, %2d, %2d, %d, %2d, %5s, %5s, %s\n"
			, (int)i
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
 * Return size of TIFFDataType in bytes.
 *
 * XXX: We need a separate function to determine the space needed
 * to store the value. For TIFF_RATIONAL values TIFFDataWidth() returns 8,
 * but we use 4-byte float to represent rationals.
 */
int
_TIFFDataSize(TIFFDataType type)
{
	switch (type) {
		case TIFF_BYTE:
		case TIFF_SBYTE:
		case TIFF_ASCII:
		case TIFF_UNDEFINED:
		    return 1;
		case TIFF_SHORT:
		case TIFF_SSHORT:
		    return 2;
		case TIFF_LONG:
		case TIFF_SLONG:
		case TIFF_FLOAT:
		case TIFF_IFD:
		case TIFF_RATIONAL:
		case TIFF_SRATIONAL:
		    return 4;
		case TIFF_DOUBLE:
		    return 8;
		default:
		    return 0;
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
        TIFFFieldInfo key = {0, 0, 0, TIFF_NOTYPE, 0, 0, 0, 0};
	TIFFFieldInfo* pkey = &key;
	const TIFFFieldInfo **ret;

	if (tif->tif_foundfield && tif->tif_foundfield->field_tag == tag &&
	    (dt == TIFF_ANY || dt == tif->tif_foundfield->field_type))
		return tif->tif_foundfield;

	/* If we are invoked with no field information, then just return. */
	if ( !tif->tif_fieldinfo ) {
		return NULL;
	}

	/* NB: use sorted search (e.g. binary search) */
	key.field_tag = tag;
        key.field_type = dt;

	ret = (const TIFFFieldInfo **) bsearch(&pkey,
					       tif->tif_fieldinfo, 
					       tif->tif_nfields,
					       sizeof(TIFFFieldInfo *), 
					       tagCompare);
	return tif->tif_foundfield = (ret ? *ret : NULL);
}

const TIFFFieldInfo*
_TIFFFindFieldInfoByName(TIFF* tif, const char *field_name, TIFFDataType dt)
{
        TIFFFieldInfo key = {0, 0, 0, TIFF_NOTYPE, 0, 0, 0, 0};
	TIFFFieldInfo* pkey = &key;
	const TIFFFieldInfo **ret;

	if (tif->tif_foundfield
	    && streq(tif->tif_foundfield->field_name, field_name)
	    && (dt == TIFF_ANY || dt == tif->tif_foundfield->field_type))
		return (tif->tif_foundfield);

	/* If we are invoked with no field information, then just return. */
	if ( !tif->tif_fieldinfo ) {
		return NULL;
	}

	/* NB: use sorted search (e.g. binary search) */
        key.field_name = (char *)field_name;
        key.field_type = dt;

        ret = (const TIFFFieldInfo **) lfind(&pkey,
					     tif->tif_fieldinfo, 
					     &tif->tif_nfields,
					     sizeof(TIFFFieldInfo *),
					     tagNameCompare);
	return tif->tif_foundfield = (ret ? *ret : NULL);
}

const TIFFFieldInfo*
_TIFFFieldWithTag(TIFF* tif, ttag_t tag)
{
	const TIFFFieldInfo* fip = _TIFFFindFieldInfo(tif, tag, TIFF_ANY);
	if (!fip) {
		TIFFErrorExt(tif->tif_clientdata, "TIFFFieldWithTag",
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
		TIFFErrorExt(tif->tif_clientdata, "TIFFFieldWithName",
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
        if (!_TIFFMergeFieldInfo(tif, fld, 1))
		return NULL;
    }

    return fld;
}

TIFFFieldInfo*
_TIFFCreateAnonFieldInfo(TIFF *tif, ttag_t tag, TIFFDataType field_type)
{
	TIFFFieldInfo *fld;
	(void) tif;

	fld = (TIFFFieldInfo *) _TIFFmalloc(sizeof (TIFFFieldInfo));
	if (fld == NULL)
	    return NULL;
	_TIFFmemset( fld, 0, sizeof(TIFFFieldInfo) );

	fld->field_tag = tag;
	fld->field_readcount = TIFF_VARIABLE2;
	fld->field_writecount = TIFF_VARIABLE2;
	fld->field_type = field_type;
	fld->field_bit = FIELD_CUSTOM;
	fld->field_oktochange = TRUE;
	fld->field_passcount = TRUE;
	fld->field_name = (char *) _TIFFmalloc(32);
	if (fld->field_name == NULL) {
	    _TIFFfree(fld);
	    return NULL;
	}

	/* 
	 * note that this name is a special sign to TIFFClose() and
	 * _TIFFSetupFieldInfo() to free the field
	 */
	sprintf(fld->field_name, "Tag %d", (int) tag);

	return fld;    
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
