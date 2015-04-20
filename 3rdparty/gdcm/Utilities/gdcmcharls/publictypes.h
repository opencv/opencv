// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 
#ifndef CHARLS_PUBLICTYPES
#define CHARLS_PUBLICTYPES

#include "config.h"

enum JLS_ERROR
{
	OK = 0,
	InvalidJlsParameters,
	ParameterValueNotSupported,
	UncompressedBufferTooSmall,
	CompressedBufferTooSmall,
	InvalidCompressedData,
	TooMuchCompressedData,
	ImageTypeNotSupported,
	UnsupportedBitDepthForTransform,
	UnsupportedColorTransform
};


enum interleavemode
{
	ILV_NONE = 0,
	ILV_LINE = 1,
	ILV_SAMPLE = 2
};


struct JlsCustomParameters
{
	int MAXVAL;
	int T1;
	int T2;
	int T3;
	int RESET;
};


struct JlsRect
{
	int X, Y;
	int Width, Height;
};


struct JfifParameters
{
	int   Ver;
	char  units;
	int   XDensity;
	int   YDensity;
	short Xthumb;
	short Ythumb;
	void* pdataThumbnail; // user must set buffer which size is Xthumb*Ythumb*3(RGB) before JpegLsDecode()
};


struct JlsParameters
{
	int width;
	int height;
	int bitspersample;
	int bytesperline;	// for [source (at encoding)][decoded (at decoding)] pixel image in user buffer
	int components;
	int allowedlossyerror;
	enum interleavemode ilv;
	int colorTransform;
	char outputBgr;
	struct JlsCustomParameters custom;
	struct JfifParameters jfif;
};

#endif
