/* $Id: tif_dirinfo.c,v 1.114 2011-05-17 00:21:17 fwarmerdam Exp $ */

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

/*
 * NOTE: THIS ARRAY IS ASSUMED TO BE SORTED BY TAG.
 *
 * NOTE: The second field (field_readcount) and third field (field_writecount)
 *       sometimes use the values TIFF_VARIABLE (-1), TIFF_VARIABLE2 (-3)
 *       and TIFFTAG_SPP (-2). The macros should be used but would throw off
 *       the formatting of the code, so please interprete the -1, -2 and -3
 *       values accordingly.
 */

static TIFFFieldArray tiffFieldArray;
static TIFFFieldArray exifFieldArray;

static TIFFField
tiffFields[] = {
    { TIFFTAG_SUBFILETYPE, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_SUBFILETYPE, 1, 0, "SubfileType", NULL },
    { TIFFTAG_OSUBFILETYPE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_SUBFILETYPE, 1, 0, "OldSubfileType", NULL },
    { TIFFTAG_IMAGEWIDTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_IMAGEDIMENSIONS, 0, 0, "ImageWidth", NULL },
    { TIFFTAG_IMAGELENGTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_IMAGEDIMENSIONS, 1, 0, "ImageLength", NULL },
    { TIFFTAG_BITSPERSAMPLE, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_BITSPERSAMPLE, 0, 0, "BitsPerSample", NULL },
    { TIFFTAG_COMPRESSION, -1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_COMPRESSION, 0, 0, "Compression", NULL },
    { TIFFTAG_PHOTOMETRIC, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_PHOTOMETRIC, 0, 0, "PhotometricInterpretation", NULL },
    { TIFFTAG_THRESHHOLDING, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_THRESHHOLDING, 1, 0, "Threshholding", NULL },
    { TIFFTAG_CELLWIDTH, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_IGNORE, 1, 0, "CellWidth", NULL },
    { TIFFTAG_CELLLENGTH, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_IGNORE, 1, 0, "CellLength", NULL },
    { TIFFTAG_FILLORDER, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_FILLORDER, 0, 0, "FillOrder", NULL },
    { TIFFTAG_DOCUMENTNAME, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "DocumentName", NULL },
    { TIFFTAG_IMAGEDESCRIPTION, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ImageDescription", NULL },
    { TIFFTAG_MAKE, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Make", NULL },
    { TIFFTAG_MODEL, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Model", NULL },
    { TIFFTAG_STRIPOFFSETS, -1, -1, TIFF_LONG8, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_STRIPOFFSETS, 0, 0, "StripOffsets", NULL },
    { TIFFTAG_ORIENTATION, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_ORIENTATION, 0, 0, "Orientation", NULL },
    { TIFFTAG_SAMPLESPERPIXEL, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_SAMPLESPERPIXEL, 0, 0, "SamplesPerPixel", NULL },
    { TIFFTAG_ROWSPERSTRIP, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_ROWSPERSTRIP, 0, 0, "RowsPerStrip", NULL },
    { TIFFTAG_STRIPBYTECOUNTS, -1, -1, TIFF_LONG8, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_STRIPBYTECOUNTS, 0, 0, "StripByteCounts", NULL },
    { TIFFTAG_MINSAMPLEVALUE, -2, -1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_MINSAMPLEVALUE, 1, 0, "MinSampleValue", NULL },
    { TIFFTAG_MAXSAMPLEVALUE, -2, -1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_MAXSAMPLEVALUE, 1, 0, "MaxSampleValue", NULL },
    { TIFFTAG_XRESOLUTION, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_RESOLUTION, 1, 0, "XResolution", NULL },
    { TIFFTAG_YRESOLUTION, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_RESOLUTION, 1, 0, "YResolution", NULL },
    { TIFFTAG_PLANARCONFIG, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_PLANARCONFIG, 0, 0, "PlanarConfiguration", NULL },
    { TIFFTAG_PAGENAME, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "PageName", NULL },
    { TIFFTAG_XPOSITION, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_POSITION, 1, 0, "XPosition", NULL },
    { TIFFTAG_YPOSITION, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_POSITION, 1, 0, "YPosition", NULL },
    { TIFFTAG_FREEOFFSETS, -1, -1, TIFF_LONG8, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_IGNORE, 0, 0, "FreeOffsets", NULL },
    { TIFFTAG_FREEBYTECOUNTS, -1, -1, TIFF_LONG8, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_IGNORE, 0, 0, "FreeByteCounts", NULL },
    { TIFFTAG_GRAYRESPONSEUNIT, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_IGNORE, 1, 0, "GrayResponseUnit", NULL },
    { TIFFTAG_GRAYRESPONSECURVE, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_IGNORE, 1, 0, "GrayResponseCurve", NULL },
    { TIFFTAG_RESOLUTIONUNIT, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_RESOLUTIONUNIT, 1, 0, "ResolutionUnit", NULL },
    { TIFFTAG_PAGENUMBER, 2, 2, TIFF_SHORT, 0, TIFF_SETGET_UINT16_PAIR, TIFF_SETGET_UNDEFINED, FIELD_PAGENUMBER, 1, 0, "PageNumber", NULL },
    { TIFFTAG_COLORRESPONSEUNIT, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_IGNORE, 1, 0, "ColorResponseUnit", NULL },
    { TIFFTAG_TRANSFERFUNCTION, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_OTHER, TIFF_SETGET_UNDEFINED, FIELD_TRANSFERFUNCTION, 1, 0, "TransferFunction", NULL },
    { TIFFTAG_SOFTWARE, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Software", NULL },
    { TIFFTAG_DATETIME, 20, 20, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "DateTime", NULL },
    { TIFFTAG_ARTIST, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Artist", NULL },
    { TIFFTAG_HOSTCOMPUTER, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "HostComputer", NULL },
    { TIFFTAG_WHITEPOINT, 2, 2, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "WhitePoint", NULL },
    { TIFFTAG_PRIMARYCHROMATICITIES, 6, 6, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "PrimaryChromaticities", NULL },
    { TIFFTAG_COLORMAP, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_OTHER, TIFF_SETGET_UNDEFINED, FIELD_COLORMAP, 1, 0, "ColorMap", NULL },
    { TIFFTAG_HALFTONEHINTS, 2, 2, TIFF_SHORT, 0, TIFF_SETGET_UINT16_PAIR, TIFF_SETGET_UNDEFINED, FIELD_HALFTONEHINTS, 1, 0, "HalftoneHints", NULL },
    { TIFFTAG_TILEWIDTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_TILEDIMENSIONS, 0, 0, "TileWidth", NULL },
    { TIFFTAG_TILELENGTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_TILEDIMENSIONS, 0, 0, "TileLength", NULL },
    { TIFFTAG_TILEOFFSETS, -1, 1, TIFF_LONG8, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_STRIPOFFSETS, 0, 0, "TileOffsets", NULL },
    { TIFFTAG_TILEBYTECOUNTS, -1, 1, TIFF_LONG8, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_STRIPBYTECOUNTS, 0, 0, "TileByteCounts", NULL },
    { TIFFTAG_SUBIFD, -1, -1, TIFF_IFD8, 0, TIFF_SETGET_C16_IFD8, TIFF_SETGET_UNDEFINED, FIELD_SUBIFD, 1, 1, "SubIFD", &tiffFieldArray },
    { TIFFTAG_INKSET, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "InkSet", NULL },
    { TIFFTAG_INKNAMES, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_C16_ASCII, TIFF_SETGET_UNDEFINED, FIELD_INKNAMES, 1, 1, "InkNames", NULL },
    { TIFFTAG_NUMBEROFINKS, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "NumberOfInks", NULL },
    { TIFFTAG_DOTRANGE, 2, 2, TIFF_SHORT, 0, TIFF_SETGET_UINT16_PAIR, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "DotRange", NULL },
    { TIFFTAG_TARGETPRINTER, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "TargetPrinter", NULL },
    { TIFFTAG_EXTRASAMPLES, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_C16_UINT16, TIFF_SETGET_UNDEFINED, FIELD_EXTRASAMPLES, 0, 1, "ExtraSamples", NULL },
    { TIFFTAG_SAMPLEFORMAT, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_SAMPLEFORMAT, 0, 0, "SampleFormat", NULL },
    { TIFFTAG_SMINSAMPLEVALUE, -2, -1, TIFF_ANY, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_SMINSAMPLEVALUE, 1, 0, "SMinSampleValue", NULL },
    { TIFFTAG_SMAXSAMPLEVALUE, -2, -1, TIFF_ANY, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_SMAXSAMPLEVALUE, 1, 0, "SMaxSampleValue", NULL },
    { TIFFTAG_CLIPPATH, -1, -3, TIFF_BYTE, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "ClipPath", NULL },
    { TIFFTAG_XCLIPPATHUNITS, 1, 1, TIFF_SLONG, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "XClipPathUnits", NULL },
    { TIFFTAG_XCLIPPATHUNITS, 1, 1, TIFF_SBYTE, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "XClipPathUnits", NULL },
    { TIFFTAG_YCLIPPATHUNITS, 1, 1, TIFF_SLONG, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "YClipPathUnits", NULL },
    { TIFFTAG_YCBCRCOEFFICIENTS, 3, 3, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "YCbCrCoefficients", NULL },
    { TIFFTAG_YCBCRSUBSAMPLING, 2, 2, TIFF_SHORT, 0, TIFF_SETGET_UINT16_PAIR, TIFF_SETGET_UNDEFINED, FIELD_YCBCRSUBSAMPLING, 0, 0, "YCbCrSubsampling", NULL },
    { TIFFTAG_YCBCRPOSITIONING, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_YCBCRPOSITIONING, 0, 0, "YCbCrPositioning", NULL },
    { TIFFTAG_REFERENCEBLACKWHITE, 6, 6, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_REFBLACKWHITE, 1, 0, "ReferenceBlackWhite", NULL },
    { TIFFTAG_XMLPACKET, -3, -3, TIFF_BYTE, 0, TIFF_SETGET_C32_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "XMLPacket", NULL },
    /* begin SGI tags */
    { TIFFTAG_MATTEING, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_EXTRASAMPLES, 0, 0, "Matteing", NULL },
    { TIFFTAG_DATATYPE, -2, -1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_SAMPLEFORMAT, 0, 0, "DataType", NULL },
    { TIFFTAG_IMAGEDEPTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_IMAGEDEPTH, 0, 0, "ImageDepth", NULL },
    { TIFFTAG_TILEDEPTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_TILEDEPTH, 0, 0, "TileDepth", NULL },
    /* end SGI tags */
    /* begin Pixar tags */
    { TIFFTAG_PIXAR_IMAGEFULLWIDTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ImageFullWidth", NULL },
    { TIFFTAG_PIXAR_IMAGEFULLLENGTH, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ImageFullLength", NULL },
    { TIFFTAG_PIXAR_TEXTUREFORMAT, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "TextureFormat", NULL },
    { TIFFTAG_PIXAR_WRAPMODES, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "TextureWrapModes", NULL },
    { TIFFTAG_PIXAR_FOVCOT, 1, 1, TIFF_FLOAT, 0, TIFF_SETGET_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FieldOfViewCotangent", NULL },
    { TIFFTAG_PIXAR_MATRIX_WORLDTOSCREEN, 16, 16, TIFF_FLOAT, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "MatrixWorldToScreen", NULL },
    { TIFFTAG_PIXAR_MATRIX_WORLDTOCAMERA, 16, 16, TIFF_FLOAT, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "MatrixWorldToCamera", NULL },
    { TIFFTAG_COPYRIGHT, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Copyright", NULL },
    /* end Pixar tags */
    { TIFFTAG_RICHTIFFIPTC, -3, -3, TIFF_LONG, 0, TIFF_SETGET_C32_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "RichTIFFIPTC", NULL },
    { TIFFTAG_PHOTOSHOP, -3, -3, TIFF_BYTE, 0, TIFF_SETGET_C32_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "Photoshop", NULL },
    { TIFFTAG_EXIFIFD, 1, 1, TIFF_IFD8, 0, TIFF_SETGET_IFD8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "EXIFIFDOffset", &exifFieldArray },
    { TIFFTAG_ICCPROFILE, -3, -3, TIFF_UNDEFINED, 0, TIFF_SETGET_C32_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "ICC Profile", NULL },
    { TIFFTAG_GPSIFD, 1, 1, TIFF_IFD8, 0, TIFF_SETGET_IFD8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "GPSIFDOffset", NULL },
    { TIFFTAG_FAXRECVPARAMS, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UINT32, FIELD_CUSTOM, TRUE, FALSE, "FaxRecvParams", NULL },
    { TIFFTAG_FAXSUBADDRESS, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_ASCII, FIELD_CUSTOM, TRUE, FALSE, "FaxSubAddress", NULL },
    { TIFFTAG_FAXRECVTIME, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UINT32, FIELD_CUSTOM, TRUE, FALSE, "FaxRecvTime", NULL },
    { TIFFTAG_FAXDCS, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_ASCII, FIELD_CUSTOM, TRUE, FALSE, "FaxDcs", NULL },
    { TIFFTAG_STONITS, 1, 1, TIFF_DOUBLE, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "StoNits", NULL },
    { TIFFTAG_INTEROPERABILITYIFD, 1, 1, TIFF_IFD8, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "InteroperabilityIFDOffset", NULL },
    /* begin DNG tags */
    { TIFFTAG_DNGVERSION, 4, 4, TIFF_BYTE, 0, TIFF_SETGET_C0_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "DNGVersion", NULL },
    { TIFFTAG_DNGBACKWARDVERSION, 4, 4, TIFF_BYTE, 0, TIFF_SETGET_C0_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "DNGBackwardVersion", NULL },
    { TIFFTAG_UNIQUECAMERAMODEL, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "UniqueCameraModel", NULL },
    { TIFFTAG_LOCALIZEDCAMERAMODEL, -1, -1, TIFF_BYTE, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "LocalizedCameraModel", NULL },
    { TIFFTAG_CFAPLANECOLOR, -1, -1, TIFF_BYTE, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "CFAPlaneColor", NULL },
    { TIFFTAG_CFALAYOUT, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "CFALayout", NULL },
    { TIFFTAG_LINEARIZATIONTABLE, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_C16_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "LinearizationTable", NULL },
    { TIFFTAG_BLACKLEVELREPEATDIM, 2, 2, TIFF_SHORT, 0, TIFF_SETGET_C0_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "BlackLevelRepeatDim", NULL },
    { TIFFTAG_BLACKLEVEL, -1, -1, TIFF_RATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "BlackLevel", NULL },
    { TIFFTAG_BLACKLEVELDELTAH, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "BlackLevelDeltaH", NULL },
    { TIFFTAG_BLACKLEVELDELTAV, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "BlackLevelDeltaV", NULL },
    { TIFFTAG_WHITELEVEL, -1, -1, TIFF_LONG, 0, TIFF_SETGET_C16_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "WhiteLevel", NULL },
    { TIFFTAG_DEFAULTSCALE, 2, 2, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "DefaultScale", NULL },
    { TIFFTAG_BESTQUALITYSCALE, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "BestQualityScale", NULL },
    { TIFFTAG_DEFAULTCROPORIGIN, 2, 2, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "DefaultCropOrigin", NULL },
    { TIFFTAG_DEFAULTCROPSIZE, 2, 2, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "DefaultCropSize", NULL },
    { TIFFTAG_COLORMATRIX1, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "ColorMatrix1", NULL },
    { TIFFTAG_COLORMATRIX2, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "ColorMatrix2", NULL },
    { TIFFTAG_CAMERACALIBRATION1, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "CameraCalibration1", NULL },
    { TIFFTAG_CAMERACALIBRATION2, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "CameraCalibration2", NULL },
    { TIFFTAG_REDUCTIONMATRIX1, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "ReductionMatrix1", NULL },
    { TIFFTAG_REDUCTIONMATRIX2, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "ReductionMatrix2", NULL },
    { TIFFTAG_ANALOGBALANCE, -1, -1, TIFF_RATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "AnalogBalance", NULL },
    { TIFFTAG_ASSHOTNEUTRAL, -1, -1, TIFF_RATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "AsShotNeutral", NULL },
    { TIFFTAG_ASSHOTWHITEXY, 2, 2, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "AsShotWhiteXY", NULL },
    { TIFFTAG_BASELINEEXPOSURE, 1, 1, TIFF_SRATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "BaselineExposure", NULL },
    { TIFFTAG_BASELINENOISE, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "BaselineNoise", NULL },
    { TIFFTAG_BASELINESHARPNESS, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "BaselineSharpness", NULL },
    { TIFFTAG_BAYERGREENSPLIT, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "BayerGreenSplit", NULL },
    { TIFFTAG_LINEARRESPONSELIMIT, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "LinearResponseLimit", NULL },
    { TIFFTAG_CAMERASERIALNUMBER, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "CameraSerialNumber", NULL },
    { TIFFTAG_LENSINFO, 4, 4, TIFF_RATIONAL, 0, TIFF_SETGET_C0_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "LensInfo", NULL },
    { TIFFTAG_CHROMABLURRADIUS, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "ChromaBlurRadius", NULL },
    { TIFFTAG_ANTIALIASSTRENGTH, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "AntiAliasStrength", NULL },
    { TIFFTAG_SHADOWSCALE, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "ShadowScale", NULL },
    { TIFFTAG_DNGPRIVATEDATA, -1, -1, TIFF_BYTE, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "DNGPrivateData", NULL },
    { TIFFTAG_MAKERNOTESAFETY, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "MakerNoteSafety", NULL },
    { TIFFTAG_CALIBRATIONILLUMINANT1, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "CalibrationIlluminant1", NULL },
    { TIFFTAG_CALIBRATIONILLUMINANT2, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "CalibrationIlluminant2", NULL },
    { TIFFTAG_RAWDATAUNIQUEID, 16, 16, TIFF_BYTE, 0, TIFF_SETGET_C0_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "RawDataUniqueID", NULL },
    { TIFFTAG_ORIGINALRAWFILENAME, -1, -1, TIFF_BYTE, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "OriginalRawFileName", NULL },
    { TIFFTAG_ORIGINALRAWFILEDATA, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "OriginalRawFileData", NULL },
    { TIFFTAG_ACTIVEAREA, 4, 4, TIFF_LONG, 0, TIFF_SETGET_C0_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 0, "ActiveArea", NULL },
    { TIFFTAG_MASKEDAREAS, -1, -1, TIFF_LONG, 0, TIFF_SETGET_C16_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "MaskedAreas", NULL },
    { TIFFTAG_ASSHOTICCPROFILE, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "AsShotICCProfile", NULL },
    { TIFFTAG_ASSHOTPREPROFILEMATRIX, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "AsShotPreProfileMatrix", NULL },
    { TIFFTAG_CURRENTICCPROFILE, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "CurrentICCProfile", NULL },
    { TIFFTAG_CURRENTPREPROFILEMATRIX, -1, -1, TIFF_SRATIONAL, 0, TIFF_SETGET_C16_FLOAT, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 0, 1, "CurrentPreProfileMatrix", NULL },
    /* end DNG tags */
    /* begin pseudo tags */
    { TIFFTAG_PERSAMPLE, 0, 0, TIFF_SHORT, 0, TIFF_SETGET_UNDEFINED, TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, TRUE, FALSE, "PerSample", NULL},
};

static TIFFField
exifFields[] = {
    { EXIFTAG_EXPOSURETIME, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ExposureTime", NULL },
    { EXIFTAG_FNUMBER, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FNumber", NULL },
    { EXIFTAG_EXPOSUREPROGRAM, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ExposureProgram", NULL },
    { EXIFTAG_SPECTRALSENSITIVITY, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SpectralSensitivity", NULL },
    { EXIFTAG_ISOSPEEDRATINGS, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_C16_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "ISOSpeedRatings", NULL },
    { EXIFTAG_OECF, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "OptoelectricConversionFactor", NULL },
    { EXIFTAG_EXIFVERSION, 4, 4, TIFF_UNDEFINED, 0, TIFF_SETGET_C0_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ExifVersion", NULL },
    { EXIFTAG_DATETIMEORIGINAL, 20, 20, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "DateTimeOriginal", NULL },
    { EXIFTAG_DATETIMEDIGITIZED, 20, 20, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "DateTimeDigitized", NULL },
    { EXIFTAG_COMPONENTSCONFIGURATION, 4, 4, TIFF_UNDEFINED, 0, TIFF_SETGET_C0_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ComponentsConfiguration", NULL },
    { EXIFTAG_COMPRESSEDBITSPERPIXEL, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "CompressedBitsPerPixel", NULL },
    { EXIFTAG_SHUTTERSPEEDVALUE, 1, 1, TIFF_SRATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ShutterSpeedValue", NULL },
    { EXIFTAG_APERTUREVALUE, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ApertureValue", NULL },
    { EXIFTAG_BRIGHTNESSVALUE, 1, 1, TIFF_SRATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "BrightnessValue", NULL },
    { EXIFTAG_EXPOSUREBIASVALUE, 1, 1, TIFF_SRATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ExposureBiasValue", NULL },
    { EXIFTAG_MAXAPERTUREVALUE, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "MaxApertureValue", NULL },
    { EXIFTAG_SUBJECTDISTANCE, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SubjectDistance", NULL },
    { EXIFTAG_METERINGMODE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "MeteringMode", NULL },
    { EXIFTAG_LIGHTSOURCE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "LightSource", NULL },
    { EXIFTAG_FLASH, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Flash", NULL },
    { EXIFTAG_FOCALLENGTH, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FocalLength", NULL },
    { EXIFTAG_SUBJECTAREA, -1, -1, TIFF_SHORT, 0, TIFF_SETGET_C16_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "SubjectArea", NULL },
    { EXIFTAG_MAKERNOTE, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "MakerNote", NULL },
    { EXIFTAG_USERCOMMENT, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "UserComment", NULL },
    { EXIFTAG_SUBSECTIME, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SubSecTime", NULL },
    { EXIFTAG_SUBSECTIMEORIGINAL, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SubSecTimeOriginal", NULL },
    { EXIFTAG_SUBSECTIMEDIGITIZED, -1, -1, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SubSecTimeDigitized", NULL },
    { EXIFTAG_FLASHPIXVERSION, 4, 4, TIFF_UNDEFINED, 0, TIFF_SETGET_C0_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FlashpixVersion", NULL },
    { EXIFTAG_COLORSPACE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ColorSpace", NULL },
    { EXIFTAG_PIXELXDIMENSION, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "PixelXDimension", NULL },
    { EXIFTAG_PIXELYDIMENSION, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "PixelYDimension", NULL },
    { EXIFTAG_RELATEDSOUNDFILE, 13, 13, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "RelatedSoundFile", NULL },
    { EXIFTAG_FLASHENERGY, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FlashEnergy", NULL },
    { EXIFTAG_SPATIALFREQUENCYRESPONSE, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "SpatialFrequencyResponse", NULL },
    { EXIFTAG_FOCALPLANEXRESOLUTION, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FocalPlaneXResolution", NULL },
    { EXIFTAG_FOCALPLANEYRESOLUTION, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FocalPlaneYResolution", NULL },
    { EXIFTAG_FOCALPLANERESOLUTIONUNIT, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FocalPlaneResolutionUnit", NULL },
    { EXIFTAG_SUBJECTLOCATION, 2, 2, TIFF_SHORT, 0, TIFF_SETGET_C0_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SubjectLocation", NULL },
    { EXIFTAG_EXPOSUREINDEX, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ExposureIndex", NULL },
    { EXIFTAG_SENSINGMETHOD, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SensingMethod", NULL },
    { EXIFTAG_FILESOURCE, 1, 1, TIFF_UNDEFINED, 0, TIFF_SETGET_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FileSource", NULL },
    { EXIFTAG_SCENETYPE, 1, 1, TIFF_UNDEFINED, 0, TIFF_SETGET_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SceneType", NULL },
    { EXIFTAG_CFAPATTERN, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "CFAPattern", NULL },
    { EXIFTAG_CUSTOMRENDERED, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "CustomRendered", NULL },
    { EXIFTAG_EXPOSUREMODE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ExposureMode", NULL },
    { EXIFTAG_WHITEBALANCE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "WhiteBalance", NULL },
    { EXIFTAG_DIGITALZOOMRATIO, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "DigitalZoomRatio", NULL },
    { EXIFTAG_FOCALLENGTHIN35MMFILM, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "FocalLengthIn35mmFilm", NULL },
    { EXIFTAG_SCENECAPTURETYPE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SceneCaptureType", NULL },
    { EXIFTAG_GAINCONTROL, 1, 1, TIFF_RATIONAL, 0, TIFF_SETGET_DOUBLE, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "GainControl", NULL },
    { EXIFTAG_CONTRAST, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Contrast", NULL },
    { EXIFTAG_SATURATION, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Saturation", NULL },
    { EXIFTAG_SHARPNESS, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "Sharpness", NULL },
    { EXIFTAG_DEVICESETTINGDESCRIPTION, -1, -1, TIFF_UNDEFINED, 0, TIFF_SETGET_C16_UINT8, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 1, "DeviceSettingDescription", NULL },
    { EXIFTAG_SUBJECTDISTANCERANGE, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "SubjectDistanceRange", NULL },
    { EXIFTAG_IMAGEUNIQUEID, 33, 33, TIFF_ASCII, 0, TIFF_SETGET_ASCII, TIFF_SETGET_UNDEFINED, FIELD_CUSTOM, 1, 0, "ImageUniqueID", NULL }
};

static TIFFFieldArray
tiffFieldArray = { tfiatImage, 0, TIFFArrayCount(tiffFields), tiffFields };
static TIFFFieldArray
exifFieldArray = { tfiatExif, 0, TIFFArrayCount(exifFields), exifFields };

/*
 *  We have our own local lfind() equivelent to avoid subtle differences
 *  in types passed to lfind() on different systems.
 */

static void *
td_lfind(const void *key, const void *base, size_t *nmemb, size_t size,
         int(*compar)(const void *, const void *))
{
    char *element, *end;

    end = (char *)base + *nmemb * size;
    for (element = (char *)base; element < end; element += size)
        if (!compar(key, element))		/* key found */
            return element;

    return NULL;
}

const TIFFFieldArray*
_TIFFGetFields(void)
{
    return(&tiffFieldArray);
}

const TIFFFieldArray*
_TIFFGetExifFields(void)
{
    return(&exifFieldArray);
}

void
_TIFFSetupFields(TIFF* tif, const TIFFFieldArray* fieldarray)
{
    if (tif->tif_fields && tif->tif_nfields > 0) {
        uint32 i;

        for (i = 0; i < tif->tif_nfields; i++) {
            TIFFField *fld = tif->tif_fields[i];
            if (fld->field_bit == FIELD_CUSTOM &&
                strncmp("Tag ", fld->field_name, 4) == 0) {
                    _TIFFfree(fld->field_name);
                    _TIFFfree(fld);
                }
        }

        _TIFFfree(tif->tif_fields);
        tif->tif_fields = NULL;
        tif->tif_nfields = 0;
    }
    if (!_TIFFMergeFields(tif, fieldarray->fields, fieldarray->count)) {
        TIFFErrorExt(tif->tif_clientdata, "_TIFFSetupFields",
                 "Setting up field info failed");
    }
}

static int
tagCompare(const void* a, const void* b)
{
    const TIFFField* ta = *(const TIFFField**) a;
    const TIFFField* tb = *(const TIFFField**) b;
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
    const TIFFField* ta = *(const TIFFField**) a;
    const TIFFField* tb = *(const TIFFField**) b;
    int ret = strcmp(ta->field_name, tb->field_name);

    if (ret)
        return ret;
    else
        return (ta->field_type == TIFF_ANY) ?
            0 : ((int)tb->field_type - (int)ta->field_type);
}

int
_TIFFMergeFields(TIFF* tif, const TIFFField info[], uint32 n)
{
    static const char module[] = "_TIFFMergeFields";
    static const char reason[] = "for fields array";
    TIFFField** tp;
    uint32 i;

        tif->tif_foundfield = NULL;

    if (tif->tif_fields && tif->tif_nfields > 0) {
        tif->tif_fields = (TIFFField**)
            _TIFFCheckRealloc(tif, tif->tif_fields,
                      (tif->tif_nfields + n),
                      sizeof(TIFFField *), reason);
    } else {
        tif->tif_fields = (TIFFField **)
            _TIFFCheckMalloc(tif, n, sizeof(TIFFField *),
                     reason);
    }
    if (!tif->tif_fields) {
        TIFFErrorExt(tif->tif_clientdata, module,
                 "Failed to allocate fields array");
        return 0;
    }

    tp = tif->tif_fields + tif->tif_nfields;
    for (i = 0; i < n; i++) {
        const TIFFField *fip =
            TIFFFindField(tif, info[i].field_tag, TIFF_ANY);

                /* only add definitions that aren't already present */
        if (!fip) {
                        tif->tif_fields[tif->tif_nfields] = (TIFFField *) (info+i);
                        tif->tif_nfields++;
                }
    }

        /* Sort the field info by tag number */
    qsort(tif->tif_fields, tif->tif_nfields,
          sizeof(TIFFField *), tagCompare);

    return n;
}

void
_TIFFPrintFieldInfo(TIFF* tif, FILE* fd)
{
    uint32 i;

    fprintf(fd, "%s: \n", tif->tif_name);
    for (i = 0; i < tif->tif_nfields; i++) {
        const TIFFField* fip = tif->tif_fields[i];
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
        case TIFF_BYTE:
        case TIFF_ASCII:
        case TIFF_SBYTE:
        case TIFF_UNDEFINED:
            return 1;
        case TIFF_SHORT:
        case TIFF_SSHORT:
            return 2;
        case TIFF_LONG:
        case TIFF_SLONG:
        case TIFF_FLOAT:
        case TIFF_IFD:
            return 4;
        case TIFF_RATIONAL:
        case TIFF_SRATIONAL:
        case TIFF_DOUBLE:
        case TIFF_LONG8:
        case TIFF_SLONG8:
        case TIFF_IFD8:
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
    switch (type)
    {
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
        case TIFF_LONG8:
        case TIFF_SLONG8:
        case TIFF_IFD8:
            return 8;
        default:
            return 0;
    }
}

const TIFFField*
TIFFFindField(TIFF* tif, uint32 tag, TIFFDataType dt)
{
    TIFFField key = {0, 0, 0, TIFF_NOTYPE, 0, 0, 0, 0, 0, 0, NULL, NULL};
    TIFFField* pkey = &key;
    const TIFFField **ret;
    if (tif->tif_foundfield && tif->tif_foundfield->field_tag == tag &&
        (dt == TIFF_ANY || dt == tif->tif_foundfield->field_type))
        return tif->tif_foundfield;

    /* If we are invoked with no field information, then just return. */
    if (!tif->tif_fields)
        return NULL;

    /* NB: use sorted search (e.g. binary search) */

    key.field_tag = tag;
    key.field_type = dt;

    ret = (const TIFFField **) bsearch(&pkey, tif->tif_fields,
                       tif->tif_nfields,
                       sizeof(TIFFField *), tagCompare);
    return tif->tif_foundfield = (ret ? *ret : NULL);
}

const TIFFField*
_TIFFFindFieldByName(TIFF* tif, const char *field_name, TIFFDataType dt)
{
    TIFFField key = {0, 0, 0, TIFF_NOTYPE, 0, 0, 0, 0, 0, 0, NULL, NULL};
    TIFFField* pkey = &key;
    const TIFFField **ret;
    if (tif->tif_foundfield
        && streq(tif->tif_foundfield->field_name, field_name)
        && (dt == TIFF_ANY || dt == tif->tif_foundfield->field_type))
        return (tif->tif_foundfield);

    /* If we are invoked with no field information, then just return. */
    if (!tif->tif_fields)
        return NULL;

    /* NB: use linear search since list is sorted by key#, not name */

    key.field_name = (char *)field_name;
    key.field_type = dt;

    ret = (const TIFFField **)
            td_lfind(&pkey, tif->tif_fields, &tif->tif_nfields,
                     sizeof(TIFFField *), tagNameCompare);

    return tif->tif_foundfield = (ret ? *ret : NULL);
}

const TIFFField*
TIFFFieldWithTag(TIFF* tif, uint32 tag)
{
    const TIFFField* fip = TIFFFindField(tif, tag, TIFF_ANY);
    if (!fip) {
        TIFFErrorExt(tif->tif_clientdata, "TIFFFieldWithTag",
                 "Internal error, unknown tag 0x%x",
                 (unsigned int) tag);
    }
    return (fip);
}

const TIFFField*
TIFFFieldWithName(TIFF* tif, const char *field_name)
{
    const TIFFField* fip =
        _TIFFFindFieldByName(tif, field_name, TIFF_ANY);
    if (!fip) {
        TIFFErrorExt(tif->tif_clientdata, "TIFFFieldWithName",
                 "Internal error, unknown tag %s", field_name);
    }
    return (fip);
}

const TIFFField*
_TIFFFindOrRegisterField(TIFF *tif, uint32 tag, TIFFDataType dt)

{
    const TIFFField *fld;

    fld = TIFFFindField(tif, tag, dt);
    if (fld == NULL) {
        fld = _TIFFCreateAnonField(tif, tag, dt);
        if (!_TIFFMergeFields(tif, fld, 1))
            return NULL;
    }

    return fld;
}

TIFFField*
_TIFFCreateAnonField(TIFF *tif, uint32 tag, TIFFDataType field_type)
{
    TIFFField *fld;
    (void) tif;

    fld = (TIFFField *) _TIFFmalloc(sizeof (TIFFField));
    if (fld == NULL)
        return NULL;
    _TIFFmemset(fld, 0, sizeof(TIFFField));

    fld->field_tag = tag;
    fld->field_readcount = TIFF_VARIABLE2;
    fld->field_writecount = TIFF_VARIABLE2;
    fld->field_type = field_type;
    fld->reserved = 0;
    switch (field_type)
    {
        case TIFF_BYTE:
        case TIFF_UNDEFINED:
            fld->set_field_type = TIFF_SETGET_C32_UINT8;
            fld->get_field_type = TIFF_SETGET_C32_UINT8;
            break;
        case TIFF_ASCII:
            fld->set_field_type = TIFF_SETGET_C32_ASCII;
            fld->get_field_type = TIFF_SETGET_C32_ASCII;
            break;
        case TIFF_SHORT:
            fld->set_field_type = TIFF_SETGET_C32_UINT16;
            fld->get_field_type = TIFF_SETGET_C32_UINT16;
            break;
        case TIFF_LONG:
            fld->set_field_type = TIFF_SETGET_C32_UINT32;
            fld->get_field_type = TIFF_SETGET_C32_UINT32;
            break;
        case TIFF_RATIONAL:
        case TIFF_SRATIONAL:
        case TIFF_FLOAT:
            fld->set_field_type = TIFF_SETGET_C32_FLOAT;
            fld->get_field_type = TIFF_SETGET_C32_FLOAT;
            break;
        case TIFF_SBYTE:
            fld->set_field_type = TIFF_SETGET_C32_SINT8;
            fld->get_field_type = TIFF_SETGET_C32_SINT8;
            break;
        case TIFF_SSHORT:
            fld->set_field_type = TIFF_SETGET_C32_SINT16;
            fld->get_field_type = TIFF_SETGET_C32_SINT16;
            break;
        case TIFF_SLONG:
            fld->set_field_type = TIFF_SETGET_C32_SINT32;
            fld->get_field_type = TIFF_SETGET_C32_SINT32;
            break;
        case TIFF_DOUBLE:
            fld->set_field_type = TIFF_SETGET_C32_DOUBLE;
            fld->get_field_type = TIFF_SETGET_C32_DOUBLE;
            break;
        case TIFF_IFD:
        case TIFF_IFD8:
            fld->set_field_type = TIFF_SETGET_C32_IFD8;
            fld->get_field_type = TIFF_SETGET_C32_IFD8;
            break;
        case TIFF_LONG8:
            fld->set_field_type = TIFF_SETGET_C32_UINT64;
            fld->get_field_type = TIFF_SETGET_C32_UINT64;
            break;
        case TIFF_SLONG8:
            fld->set_field_type = TIFF_SETGET_C32_SINT64;
            fld->get_field_type = TIFF_SETGET_C32_SINT64;
            break;
        default:
            fld->set_field_type = TIFF_SETGET_UNDEFINED;
            fld->get_field_type = TIFF_SETGET_UNDEFINED;
            break;
    }
    fld->field_bit = FIELD_CUSTOM;
    fld->field_oktochange = TRUE;
    fld->field_passcount = TRUE;
    fld->field_name = (char *) _TIFFmalloc(32);
    if (fld->field_name == NULL) {
        _TIFFfree(fld);
        return NULL;
    }
    fld->field_subfields = NULL;

    /*
     * note that this name is a special sign to TIFFClose() and
     * _TIFFSetupFields() to free the field
     */
    sprintf(fld->field_name, "Tag %d", (int) tag);

    return fld;
}

/****************************************************************************
 *               O B S O L E T E D    I N T E R F A C E S
 *
 * Don't use this stuff in your applications, it may be removed in the future
 * libtiff versions.
 ****************************************************************************/

static TIFFSetGetFieldType
_TIFFSetGetType(TIFFDataType type, short count, unsigned char passcount)
{
    if (type == TIFF_ASCII && count == TIFF_VARIABLE && passcount == 0)
        return TIFF_SETGET_ASCII;

    else if (count == 1 && passcount == 0) {
        switch (type)
        {
            case TIFF_BYTE:
            case TIFF_UNDEFINED:
                return TIFF_SETGET_UINT8;
            case TIFF_ASCII:
                return TIFF_SETGET_ASCII;
            case TIFF_SHORT:
                return TIFF_SETGET_UINT16;
            case TIFF_LONG:
                return TIFF_SETGET_UINT32;
            case TIFF_RATIONAL:
            case TIFF_SRATIONAL:
            case TIFF_FLOAT:
                return TIFF_SETGET_FLOAT;
            case TIFF_SBYTE:
                return TIFF_SETGET_SINT8;
            case TIFF_SSHORT:
                return TIFF_SETGET_SINT16;
            case TIFF_SLONG:
                return TIFF_SETGET_SINT32;
            case TIFF_DOUBLE:
                return TIFF_SETGET_DOUBLE;
            case TIFF_IFD:
            case TIFF_IFD8:
                return TIFF_SETGET_IFD8;
            case TIFF_LONG8:
                return TIFF_SETGET_UINT64;
            case TIFF_SLONG8:
                return TIFF_SETGET_SINT64;
            default:
                return TIFF_SETGET_UNDEFINED;
        }
    }

    else if (count >= 1 && passcount == 0) {
        switch (type)
        {
            case TIFF_BYTE:
            case TIFF_UNDEFINED:
                return TIFF_SETGET_C0_UINT8;
            case TIFF_ASCII:
                return TIFF_SETGET_C0_ASCII;
            case TIFF_SHORT:
                return TIFF_SETGET_C0_UINT16;
            case TIFF_LONG:
                return TIFF_SETGET_C0_UINT32;
            case TIFF_RATIONAL:
            case TIFF_SRATIONAL:
            case TIFF_FLOAT:
                return TIFF_SETGET_C0_FLOAT;
            case TIFF_SBYTE:
                return TIFF_SETGET_C0_SINT8;
            case TIFF_SSHORT:
                return TIFF_SETGET_C0_SINT16;
            case TIFF_SLONG:
                return TIFF_SETGET_C0_SINT32;
            case TIFF_DOUBLE:
                return TIFF_SETGET_C0_DOUBLE;
            case TIFF_IFD:
            case TIFF_IFD8:
                return TIFF_SETGET_C0_IFD8;
            case TIFF_LONG8:
                return TIFF_SETGET_C0_UINT64;
            case TIFF_SLONG8:
                return TIFF_SETGET_C0_SINT64;
            default:
                return TIFF_SETGET_UNDEFINED;
        }
    }

    else if (count == TIFF_VARIABLE && passcount == 1) {
        switch (type)
        {
            case TIFF_BYTE:
            case TIFF_UNDEFINED:
                return TIFF_SETGET_C16_UINT8;
            case TIFF_ASCII:
                return TIFF_SETGET_C16_ASCII;
            case TIFF_SHORT:
                return TIFF_SETGET_C16_UINT16;
            case TIFF_LONG:
                return TIFF_SETGET_C16_UINT32;
            case TIFF_RATIONAL:
            case TIFF_SRATIONAL:
            case TIFF_FLOAT:
                return TIFF_SETGET_C16_FLOAT;
            case TIFF_SBYTE:
                return TIFF_SETGET_C16_SINT8;
            case TIFF_SSHORT:
                return TIFF_SETGET_C16_SINT16;
            case TIFF_SLONG:
                return TIFF_SETGET_C16_SINT32;
            case TIFF_DOUBLE:
                return TIFF_SETGET_C16_DOUBLE;
            case TIFF_IFD:
            case TIFF_IFD8:
                return TIFF_SETGET_C16_IFD8;
            case TIFF_LONG8:
                return TIFF_SETGET_C16_UINT64;
            case TIFF_SLONG8:
                return TIFF_SETGET_C16_SINT64;
            default:
                return TIFF_SETGET_UNDEFINED;
        }
    }

    else if (count == TIFF_VARIABLE2 && passcount == 1) {
        switch (type)
        {
            case TIFF_BYTE:
            case TIFF_UNDEFINED:
                return TIFF_SETGET_C32_UINT8;
            case TIFF_ASCII:
                return TIFF_SETGET_C32_ASCII;
            case TIFF_SHORT:
                return TIFF_SETGET_C32_UINT16;
            case TIFF_LONG:
                return TIFF_SETGET_C32_UINT32;
            case TIFF_RATIONAL:
            case TIFF_SRATIONAL:
            case TIFF_FLOAT:
                return TIFF_SETGET_C32_FLOAT;
            case TIFF_SBYTE:
                return TIFF_SETGET_C32_SINT8;
            case TIFF_SSHORT:
                return TIFF_SETGET_C32_SINT16;
            case TIFF_SLONG:
                return TIFF_SETGET_C32_SINT32;
            case TIFF_DOUBLE:
                return TIFF_SETGET_C32_DOUBLE;
            case TIFF_IFD:
            case TIFF_IFD8:
                return TIFF_SETGET_C32_IFD8;
            case TIFF_LONG8:
                return TIFF_SETGET_C32_UINT64;
            case TIFF_SLONG8:
                return TIFF_SETGET_C32_SINT64;
            default:
                return TIFF_SETGET_UNDEFINED;
        }
    }

    return TIFF_SETGET_UNDEFINED;
}

int
TIFFMergeFieldInfo(TIFF* tif, const TIFFFieldInfo info[], uint32 n)
{
    static const char module[] = "TIFFMergeFieldInfo";
    static const char reason[] = "for fields array";
    TIFFField *tp;
    size_t nfields;
    uint32 i;

    if (tif->tif_nfieldscompat > 0) {
        tif->tif_fieldscompat = (TIFFFieldArray *)
            _TIFFCheckRealloc(tif, tif->tif_fieldscompat,
                      tif->tif_nfieldscompat + 1,
                      sizeof(TIFFFieldArray), reason);
    } else {
        tif->tif_fieldscompat = (TIFFFieldArray *)
            _TIFFCheckMalloc(tif, 1, sizeof(TIFFFieldArray),
                     reason);
    }
    if (!tif->tif_fieldscompat) {
        TIFFErrorExt(tif->tif_clientdata, module,
                 "Failed to allocate fields array");
        return -1;
    }
    nfields = tif->tif_nfieldscompat++;

    tif->tif_fieldscompat[nfields].type = tfiatOther;
    tif->tif_fieldscompat[nfields].allocated_size = n;
    tif->tif_fieldscompat[nfields].count = n;
    tif->tif_fieldscompat[nfields].fields =
        (TIFFField *)_TIFFCheckMalloc(tif, n, sizeof(TIFFField),
                          reason);
    if (!tif->tif_fieldscompat[nfields].fields) {
        TIFFErrorExt(tif->tif_clientdata, module,
                 "Failed to allocate fields array");
        return -1;
    }

    tp = tif->tif_fieldscompat[nfields].fields;
    for (i = 0; i < n; i++) {
        tp->field_tag = info[i].field_tag;
        tp->field_readcount = info[i].field_readcount;
        tp->field_writecount = info[i].field_writecount;
        tp->field_type = info[i].field_type;
        tp->reserved = 0;
        tp->set_field_type =
             _TIFFSetGetType(info[i].field_type,
                info[i].field_readcount,
                info[i].field_passcount);
        tp->get_field_type =
             _TIFFSetGetType(info[i].field_type,
                info[i].field_readcount,
                info[i].field_passcount);
        tp->field_bit = info[i].field_bit;
        tp->field_oktochange = info[i].field_oktochange;
        tp->field_passcount = info[i].field_passcount;
        tp->field_name = info[i].field_name;
        tp->field_subfields = NULL;
        tp++;
    }

    if (!_TIFFMergeFields(tif, tif->tif_fieldscompat[nfields].fields, n)) {
        TIFFErrorExt(tif->tif_clientdata, module,
                 "Setting up field info failed");
        return -1;
    }

    return 0;
}

/* vim: set ts=8 sts=8 sw=8 noet: */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
