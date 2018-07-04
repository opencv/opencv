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

#ifdef HAVE_TIFF

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


// libtiff based TIFF codec
class TiffDecoder CV_FINAL : public BaseImageDecoder
{
public:
    TiffDecoder();
    virtual ~TiffDecoder() CV_OVERRIDE;

    bool  readHeader() CV_OVERRIDE;
    bool  readData( Mat& img ) CV_OVERRIDE;
    void  close();
    bool  nextPage() CV_OVERRIDE;

    size_t signatureLength() const CV_OVERRIDE;
    bool checkSignature( const String& signature ) const CV_OVERRIDE;
    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:
    void* m_tif;
    int normalizeChannelsNumber(int channels) const;
    bool readData_32FC3(Mat& img);
    bool readData_32FC1(Mat& img);
    bool m_hdr;
    size_t m_buf_pos;

private:
    TiffDecoder(const TiffDecoder &); // copy disabled
    TiffDecoder& operator=(const TiffDecoder &); // assign disabled
};

// ... and writer
class TiffEncoder CV_FINAL : public BaseImageEncoder
{
public:
    TiffEncoder();
    virtual ~TiffEncoder() CV_OVERRIDE;

    bool isFormatSupported( int depth ) const CV_OVERRIDE;

    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;

    bool writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE;

protected:
    void  writeTag( WLByteStream& strm, TiffTag tag,
                    TiffFieldType fieldType,
                    int count, int value );

    bool writeLibTiff( const std::vector<Mat>& img_vec, const std::vector<int>& params );
    bool write_32FC3( const Mat& img );
    bool write_32FC1( const Mat& img );

private:
    TiffEncoder(const TiffEncoder &); // copy disabled
    TiffEncoder& operator=(const TiffEncoder &); // assign disabled
};

}

#endif // HAVE_TIFF

#endif/*_GRFMT_TIFF_H_*/
