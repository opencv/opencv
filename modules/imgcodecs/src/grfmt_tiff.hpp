/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _GRFMT_TIFF_H_
#define _GRFMT_TIFF_H_

#include "grfmt_base.hpp"

namespace cv
{

// native simple TIFF codec
enum TiffCompression
{
    TIFF_UNCOMP = 1,
    TIFF_HUFFMAN = 2,
    TIFF_PACKBITS = 32773
};

enum TiffByteOrder
{
    TIFF_ORDER_II = 0x4949,
    TIFF_ORDER_MM = 0x4d4d
};


enum  TiffTag
{
    TIFF_TAG_WIDTH  = 256,
    TIFF_TAG_HEIGHT = 257,
    TIFF_TAG_BITS_PER_SAMPLE = 258,
    TIFF_TAG_COMPRESSION = 259,
    TIFF_TAG_PHOTOMETRIC = 262,
    TIFF_TAG_STRIP_OFFSETS = 273,
    TIFF_TAG_STRIP_COUNTS = 279,
    TIFF_TAG_SAMPLES_PER_PIXEL = 277,
    TIFF_TAG_ROWS_PER_STRIP = 278,
    TIFF_TAG_PLANAR_CONFIG = 284,
    TIFF_TAG_COLOR_MAP = 320
};


enum TiffFieldType
{
    TIFF_TYPE_BYTE = 1,
    TIFF_TYPE_SHORT = 3,
    TIFF_TYPE_LONG = 4
};


#ifdef HAVE_TIFF

// libtiff based TIFF codec

class TiffDecoder : public BaseImageDecoder
{
public:
    TiffDecoder();
    virtual ~TiffDecoder();

    bool  readHeader();
    bool  readData( Mat& img );
    void  close();
    bool  nextPage();

    size_t signatureLength() const;
    bool checkSignature( const String& signature ) const;
    ImageDecoder newDecoder() const;

protected:
    void* m_tif;
    int normalizeChannelsNumber(int channels) const;
    bool readHdrData(Mat& img);
    bool m_hdr;
};

#endif

// ... and writer
class TiffEncoder : public BaseImageEncoder
{
public:
    TiffEncoder();
    virtual ~TiffEncoder();

    bool isFormatSupported( int depth ) const;

    bool  write( const Mat& img, const std::vector<int>& params );
    ImageEncoder newEncoder() const;

protected:
    void  writeTag( WLByteStream& strm, TiffTag tag,
                    TiffFieldType fieldType,
                    int count, int value );

    bool writeLibTiff( const Mat& img, const std::vector<int>& params );
    bool writeHdr( const Mat& img );
};

}

#endif/*_GRFMT_TIFF_H_*/
