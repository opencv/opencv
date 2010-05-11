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

/****************************************************************************************\
    A part of the file implements TIFF reader on base of libtiff library
    (see otherlibs/_graphics/readme.txt for copyright notice)
\****************************************************************************************/

#include "precomp.hpp"
#include "grfmt_tiff.hpp"

namespace cv
{
static const char fmtSignTiffII[] = "II\x2a\x00";
static const char fmtSignTiffMM[] = "MM\x00\x2a";

#ifdef HAVE_TIFF

#include "tiff.h"
#include "tiffio.h"

static int grfmt_tiff_err_handler_init = 0;
static void GrFmtSilentTIFFErrorHandler( const char*, const char*, va_list ) {}

TiffDecoder::TiffDecoder()
{
    m_tif = 0;
    if( !grfmt_tiff_err_handler_init )
    {
        grfmt_tiff_err_handler_init = 1;

        TIFFSetErrorHandler( GrFmtSilentTIFFErrorHandler );
        TIFFSetWarningHandler( GrFmtSilentTIFFErrorHandler );
    }
}


void TiffDecoder::close()
{
    if( m_tif )
    {
        TIFF* tif = (TIFF*)m_tif;
        TIFFClose( tif );
        m_tif = 0;
    }
}

TiffDecoder::~TiffDecoder()
{
    close();
}

size_t TiffDecoder::signatureLength() const
{
    return 4;
}

bool TiffDecoder::checkSignature( const string& signature ) const
{
    return signature.size() >= 4 &&
        (memcmp(signature.c_str(), fmtSignTiffII, 4) == 0 ||
        memcmp(signature.c_str(), fmtSignTiffMM, 4) == 0);
}

ImageDecoder TiffDecoder::newDecoder() const
{
    return new TiffDecoder;
}

bool TiffDecoder::readHeader()
{
    char errmsg[1024];
    bool result = false;

    close();
    TIFF* tif = TIFFOpen( m_filename.c_str(), "rb" );

    if( tif )
    {
        int width = 0, height = 0, photometric = 0, compression = 0;
        m_tif = tif;

        if( TIFFRGBAImageOK( tif, errmsg ) &&
            TIFFGetField( tif, TIFFTAG_IMAGEWIDTH, &width ) &&
            TIFFGetField( tif, TIFFTAG_IMAGELENGTH, &height ) &&
            TIFFGetField( tif, TIFFTAG_PHOTOMETRIC, &photometric ) &&
            (!TIFFGetField( tif, TIFFTAG_COMPRESSION, &compression ) ||
            (compression != COMPRESSION_LZW &&
             compression != COMPRESSION_OJPEG)))
        {
            m_width = width;
            m_height = height;
            m_type = photometric > 1 ? CV_8UC3 : CV_8UC1;

            result = true;
        }
    }

    if( !result )
        close();

    return result;
}


bool  TiffDecoder::readData( Mat& img )
{
    bool result = false;
    bool color = img.channels() > 1;
    uchar* data = img.data;
    int step = img.step;

    if( m_tif && m_width && m_height )
    {
        TIFF* tif = (TIFF*)m_tif;
        int tile_width0 = m_width, tile_height0 = 0;
        int x, y, i;
        int is_tiled = TIFFIsTiled(tif);

        if( (!is_tiled &&
            TIFFGetField( tif, TIFFTAG_ROWSPERSTRIP, &tile_height0 )) ||
            (is_tiled &&
            TIFFGetField( tif, TIFFTAG_TILEWIDTH, &tile_width0 ) &&
            TIFFGetField( tif, TIFFTAG_TILELENGTH, &tile_height0 )))
        {
            if( tile_width0 <= 0 )
                tile_width0 = m_width;

            if( tile_height0 <= 0 )
                tile_height0 = m_height;

            AutoBuffer<uchar> _buffer(tile_height0*tile_width0*4);
            uchar* buffer = _buffer;

            for( y = 0; y < m_height; y += tile_height0, data += step*tile_height0 )
            {
                int tile_height = tile_height0;

                if( y + tile_height > m_height )
                    tile_height = m_height - y;

                for( x = 0; x < m_width; x += tile_width0 )
                {
                    int tile_width = tile_width0, ok;

                    if( x + tile_width > m_width )
                        tile_width = m_width - x;

                    if( !is_tiled )
                        ok = TIFFReadRGBAStrip( tif, y, (uint32*)buffer );
                    else
                        ok = TIFFReadRGBATile( tif, x, y, (uint32*)buffer );

                    if( !ok )
                    {
                        close();
                        return false;
                    }

                    for( i = 0; i < tile_height; i++ )
                        if( color )
                            icvCvt_BGRA2BGR_8u_C4C3R( buffer + i*tile_width*4, 0,
                                          data + x*3 + step*(tile_height - i - 1), 0,
                                          cvSize(tile_width,1), 2 );
                        else
                            icvCvt_BGRA2Gray_8u_C4C1R( buffer + i*tile_width*4, 0,
                                           data + x + step*(tile_height - i - 1), 0,
                                           cvSize(tile_width,1), 2 );
                }
            }

            result = true;
        }
    }

    close();
    return result;
}

#endif

//////////////////////////////////////////////////////////////////////////////////////////

TiffEncoder::TiffEncoder()
{
    m_description = "TIFF Files (*.tiff;*.tif)";
    m_buf_supported = true;
}

TiffEncoder::~TiffEncoder()
{
}

ImageEncoder TiffEncoder::newEncoder() const
{
    return new TiffEncoder;
}

void  TiffEncoder::writeTag( WLByteStream& strm, TiffTag tag,
                             TiffFieldType fieldType,
                             int count, int value )
{
    strm.putWord( tag );
    strm.putWord( fieldType );
    strm.putDWord( count );
    strm.putDWord( value );
}


bool  TiffEncoder::write( const Mat& img, const vector<int>& )
{
    int channels = img.channels();
    int width = img.cols, height = img.rows;
    int fileStep = width*channels;
    WLByteStream strm;

    if( m_buf )
    {
        if( !strm.open(*m_buf) )
            return false;
    }
    else if( !strm.open(m_filename) )
        return false;

    int rowsPerStrip = (1 << 13)/fileStep;

    if( rowsPerStrip < 1 )
        rowsPerStrip = 1;

    if( rowsPerStrip > height )
        rowsPerStrip = height;

    int i, stripCount = (height + rowsPerStrip - 1) / rowsPerStrip;

    if( m_buf )
        m_buf->reserve( alignSize(stripCount*8 + fileStep*height + 256, 256) );

/*#if defined _DEBUG || !defined WIN32
    int uncompressedRowSize = rowsPerStrip * fileStep;
#endif*/
    int directoryOffset = 0;

    AutoBuffer<int,1024> stripOffsets(stripCount);
    AutoBuffer<short,1024> stripCounts(stripCount);
    AutoBuffer<uchar,1024> _buffer(fileStep+32);
    uchar* buffer = _buffer;
    int  stripOffsetsOffset = 0;
    int  stripCountsOffset = 0;
    int  bitsPerSample = 8; // TODO support 16 bit
    int  y = 0;

    strm.putBytes( fmtSignTiffII, 4 );
    strm.putDWord( directoryOffset );

    // write an image data first (the most reasonable way
    // for compressed images)
    for( i = 0; i < stripCount; i++ )
    {
        int limit = y + rowsPerStrip;

        if( limit > height )
            limit = height;

        stripOffsets[i] = strm.getPos();

        for( ; y < limit; y++ )
        {
            if( channels == 3 )
                icvCvt_BGR2RGB_8u_C3R( img.data + img.step*y, 0, buffer, 0, cvSize(width,1) );
            else if( channels == 4 )
                icvCvt_BGRA2RGBA_8u_C4R( img.data + img.step*y, 0, buffer, 0, cvSize(width,1) );

            strm.putBytes( channels > 1 ? buffer : img.data + img.step*y, fileStep );
        }

        stripCounts[i] = (short)(strm.getPos() - stripOffsets[i]);
        /*assert( stripCounts[i] == uncompressedRowSize ||
                stripCounts[i] < uncompressedRowSize &&
                i == stripCount - 1);*/
    }

    if( stripCount > 2 )
    {
        stripOffsetsOffset = strm.getPos();
        for( i = 0; i < stripCount; i++ )
            strm.putDWord( stripOffsets[i] );

        stripCountsOffset = strm.getPos();
        for( i = 0; i < stripCount; i++ )
            strm.putWord( stripCounts[i] );
    }
    else if(stripCount == 2)
    {
        stripOffsetsOffset = strm.getPos();
        for (i = 0; i < stripCount; i++)
        {
            strm.putDWord (stripOffsets [i]);
        }
        stripCountsOffset = stripCounts [0] + (stripCounts [1] << 16);
    }
    else
    {
        stripOffsetsOffset = stripOffsets[0];
        stripCountsOffset = stripCounts[0];
    }

    if( channels > 1 )
    {
        bitsPerSample = strm.getPos();
        strm.putWord(8);
        strm.putWord(8);
        strm.putWord(8);
        if( channels == 4 )
            strm.putWord(8);
    }

    directoryOffset = strm.getPos();

    // write header
    strm.putWord( 9 );

    /* warning: specification 5.0 of Tiff want to have tags in
       ascending order. This is a non-fatal error, but this cause
       warning with some tools. So, keep this in ascending order */

    writeTag( strm, TIFF_TAG_WIDTH, TIFF_TYPE_LONG, 1, width );
    writeTag( strm, TIFF_TAG_HEIGHT, TIFF_TYPE_LONG, 1, height );
    writeTag( strm, TIFF_TAG_BITS_PER_SAMPLE,
              TIFF_TYPE_SHORT, channels, bitsPerSample );
    writeTag( strm, TIFF_TAG_COMPRESSION, TIFF_TYPE_LONG, 1, TIFF_UNCOMP );
    writeTag( strm, TIFF_TAG_PHOTOMETRIC, TIFF_TYPE_SHORT, 1, channels > 1 ? 2 : 1 );

    writeTag( strm, TIFF_TAG_STRIP_OFFSETS, TIFF_TYPE_LONG,
              stripCount, stripOffsetsOffset );

    writeTag( strm, TIFF_TAG_SAMPLES_PER_PIXEL, TIFF_TYPE_SHORT, 1, channels );
    writeTag( strm, TIFF_TAG_ROWS_PER_STRIP, TIFF_TYPE_LONG, 1, rowsPerStrip );

    writeTag( strm, TIFF_TAG_STRIP_COUNTS,
              stripCount > 1 ? TIFF_TYPE_SHORT : TIFF_TYPE_LONG,
              stripCount, stripCountsOffset );

    strm.putDWord(0);
    strm.close();

    if( m_buf )
    {
        (*m_buf)[4] = (uchar)directoryOffset;
        (*m_buf)[5] = (uchar)(directoryOffset >> 8);
        (*m_buf)[6] = (uchar)(directoryOffset >> 16);
        (*m_buf)[7] = (uchar)(directoryOffset >> 24);
    }
    else
    {
        // write directory offset
        FILE* f = fopen( m_filename.c_str(), "r+b" );
        buffer[0] = (uchar)directoryOffset;
        buffer[1] = (uchar)(directoryOffset >> 8);
        buffer[2] = (uchar)(directoryOffset >> 16);
        buffer[3] = (uchar)(directoryOffset >> 24);

        fseek( f, 4, SEEK_SET );
        fwrite( buffer, 1, 4, f );
        fclose(f);
    }

    return true;
}

}
