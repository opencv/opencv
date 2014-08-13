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
#include <opencv2/imgproc.hpp>

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
    m_hdr = false;
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

bool TiffDecoder::checkSignature( const String& signature ) const
{
    return signature.size() >= 4 &&
        (memcmp(signature.c_str(), fmtSignTiffII, 4) == 0 ||
        memcmp(signature.c_str(), fmtSignTiffMM, 4) == 0);
}

int TiffDecoder::normalizeChannelsNumber(int channels) const
{
    return channels > 4 ? 4 : channels;
}

ImageDecoder TiffDecoder::newDecoder() const
{
    return makePtr<TiffDecoder>();
}

bool TiffDecoder::readHeader()
{
    bool result = false;

    close();
    // TIFFOpen() mode flags are different to fopen().  A 'b' in mode "rb" has no effect when reading.
    // http://www.remotesensing.org/libtiff/man/TIFFOpen.3tiff.html
    TIFF* tif = TIFFOpen( m_filename.c_str(), "r" );

    if( tif )
    {
        uint32 wdth = 0, hght = 0;
        uint16 photometric = 0;
        m_tif = tif;

        if( TIFFGetField( tif, TIFFTAG_IMAGEWIDTH, &wdth ) &&
            TIFFGetField( tif, TIFFTAG_IMAGELENGTH, &hght ) &&
            TIFFGetField( tif, TIFFTAG_PHOTOMETRIC, &photometric ))
        {
            uint16 bpp=8, ncn = photometric > 1 ? 3 : 1;
            TIFFGetField( tif, TIFFTAG_BITSPERSAMPLE, &bpp );
            TIFFGetField( tif, TIFFTAG_SAMPLESPERPIXEL, &ncn );

            m_width = wdth;
            m_height = hght;
            if((bpp == 32 && ncn == 3) || photometric == PHOTOMETRIC_LOGLUV)
            {
                m_type = CV_32FC3;
                m_hdr = true;
                return true;
            }
            m_hdr = false;

            if( bpp > 8 &&
               ((photometric != 2 && photometric != 1) ||
                (ncn != 1 && ncn != 3 && ncn != 4)))
                bpp = 8;

            int wanted_channels = normalizeChannelsNumber(ncn);
            switch(bpp)
            {
                case 8:
                    m_type = CV_MAKETYPE(CV_8U, photometric > 1 ? wanted_channels : 1);
                    break;
                case 16:
                    m_type = CV_MAKETYPE(CV_16U, photometric > 1 ? wanted_channels : 1);
                    break;

                case 32:
                    m_type = CV_MAKETYPE(CV_32F, photometric > 1 ? 3 : 1);
                    break;
                case 64:
                    m_type = CV_MAKETYPE(CV_64F, photometric > 1 ? 3 : 1);
                    break;

                default:
                    result = false;
            }
            result = true;
        }
    }

    if( !result )
        close();

    return result;
}


bool  TiffDecoder::readData( Mat& img )
{
    if(m_hdr && img.type() == CV_32FC3)
    {
        return readHdrData(img);
    }
    bool result = false;
    bool color = img.channels() > 1;
    uchar* data = img.ptr();

    if( img.depth() != CV_8U && img.depth() != CV_16U && img.depth() != CV_32F && img.depth() != CV_64F )
        return false;

    if( m_tif && m_width && m_height )
    {
        TIFF* tif = (TIFF*)m_tif;
        uint32 tile_width0 = m_width, tile_height0 = 0;
        int x, y, i;
        int is_tiled = TIFFIsTiled(tif);
        uint16 photometric;
        TIFFGetField( tif, TIFFTAG_PHOTOMETRIC, &photometric );
        uint16 bpp = 8, ncn = photometric > 1 ? 3 : 1;
        TIFFGetField( tif, TIFFTAG_BITSPERSAMPLE, &bpp );
        TIFFGetField( tif, TIFFTAG_SAMPLESPERPIXEL, &ncn );
        const int bitsPerByte = 8;
        int dst_bpp = (int)(img.elemSize1() * bitsPerByte);
        int wanted_channels = normalizeChannelsNumber(img.channels());

        if(dst_bpp == 8)
        {
            char errmsg[1024];
            if(!TIFFRGBAImageOK( tif, errmsg ))
            {
                close();
                return false;
            }
        }

        if( (!is_tiled) ||
            (is_tiled &&
            TIFFGetField( tif, TIFFTAG_TILEWIDTH, &tile_width0 ) &&
            TIFFGetField( tif, TIFFTAG_TILELENGTH, &tile_height0 )))
        {
            if(!is_tiled)
                TIFFGetField( tif, TIFFTAG_ROWSPERSTRIP, &tile_height0 );

            if( tile_width0 <= 0 )
                tile_width0 = m_width;

            if( tile_height0 <= 0 )
                tile_height0 = m_height;

            AutoBuffer<uchar> _buffer( size_t(8) * tile_height0*tile_width0);
            uchar* buffer = _buffer;
            ushort* buffer16 = (ushort*)buffer;
            float* buffer32 = (float*)buffer;
            double* buffer64 = (double*)buffer;
            int tileidx = 0;

            for( y = 0; y < m_height; y += tile_height0, data += img.step*tile_height0 )
            {
                int tile_height = tile_height0;

                if( y + tile_height > m_height )
                    tile_height = m_height - y;

                for( x = 0; x < m_width; x += tile_width0, tileidx++ )
                {
                    int tile_width = tile_width0, ok;

                    if( x + tile_width > m_width )
                        tile_width = m_width - x;

                    switch(dst_bpp)
                    {
                        case 8:
                        {
                            uchar * bstart = buffer;
                            if( !is_tiled )
                                ok = TIFFReadRGBAStrip( tif, y, (uint32*)buffer );
                            else
                            {
                                ok = TIFFReadRGBATile( tif, x, y, (uint32*)buffer );
                                //Tiles fill the buffer from the bottom up
                                bstart += (tile_height0 - tile_height) * tile_width0 * 4;
                            }
                            if( !ok )
                            {
                                close();
                                return false;
                            }

                            for( i = 0; i < tile_height; i++ )
                                if( color )
                                {
                                    if (wanted_channels == 4)
                                    {
                                        icvCvt_BGRA2RGBA_8u_C4R( bstart + i*tile_width0*4, 0,
                                                             data + x*4 + img.step*(tile_height - i - 1), 0,
                                                             cvSize(tile_width,1) );
                                    }
                                    else
                                    {
                                        icvCvt_BGRA2BGR_8u_C4C3R( bstart + i*tile_width0*4, 0,
                                                             data + x*3 + img.step*(tile_height - i - 1), 0,
                                                             cvSize(tile_width,1), 2 );
                                    }
                                }
                                else
                                    icvCvt_BGRA2Gray_8u_C4C1R( bstart + i*tile_width0*4, 0,
                                                              data + x + img.step*(tile_height - i - 1), 0,
                                                              cvSize(tile_width,1), 2 );
                            break;
                        }

                        case 16:
                        {
                            if( !is_tiled )
                                ok = (int)TIFFReadEncodedStrip( tif, tileidx, (uint32*)buffer, (tsize_t)-1 ) >= 0;
                            else
                                ok = (int)TIFFReadEncodedTile( tif, tileidx, (uint32*)buffer, (tsize_t)-1 ) >= 0;

                            if( !ok )
                            {
                                close();
                                return false;
                            }

                            for( i = 0; i < tile_height; i++ )
                            {
                                if( color )
                                {
                                    if( ncn == 1 )
                                    {
                                        icvCvt_Gray2BGR_16u_C1C3R(buffer16 + i*tile_width0*ncn, 0,
                                                                  (ushort*)(data + img.step*i) + x*3, 0,
                                                                  cvSize(tile_width,1) );
                                    }
                                    else if( ncn == 3 )
                                    {
                                        icvCvt_RGB2BGR_16u_C3R(buffer16 + i*tile_width0*ncn, 0,
                                                               (ushort*)(data + img.step*i) + x*3, 0,
                                                               cvSize(tile_width,1) );
                                    }
                                    else if (ncn == 4)
                                    {
                                        if (wanted_channels == 4)
                                        {
                                            icvCvt_BGRA2RGBA_16u_C4R(buffer16 + i*tile_width0*ncn, 0,
                                                (ushort*)(data + img.step*i) + x * 4, 0,
                                                cvSize(tile_width, 1));
                                        }
                                        else
                                        {
                                            icvCvt_BGRA2BGR_16u_C4C3R(buffer16 + i*tile_width0*ncn, 0,
                                                (ushort*)(data + img.step*i) + x * 3, 0,
                                                cvSize(tile_width, 1), 2);
                                        }
                                    }
                                    else
                                    {
                                        icvCvt_BGRA2BGR_16u_C4C3R(buffer16 + i*tile_width0*ncn, 0,
                                                               (ushort*)(data + img.step*i) + x*3, 0,
                                                               cvSize(tile_width,1), 2 );
                                    }
                                }
                                else
                                {
                                    if( ncn == 1 )
                                    {
                                        memcpy((ushort*)(data + img.step*i)+x,
                                               buffer16 + i*tile_width0*ncn,
                                               tile_width*sizeof(buffer16[0]));
                                    }
                                    else
                                    {
                                        icvCvt_BGRA2Gray_16u_CnC1R(buffer16 + i*tile_width0*ncn, 0,
                                                               (ushort*)(data + img.step*i) + x, 0,
                                                               cvSize(tile_width,1), ncn, 2 );
                                    }
                                }
                            }
                            break;
                        }

                        case 32:
                        case 64:
                        {
                            if( !is_tiled )
                                ok = (int)TIFFReadEncodedStrip( tif, tileidx, buffer, (tsize_t)-1 ) >= 0;
                            else
                                ok = (int)TIFFReadEncodedTile( tif, tileidx, buffer, (tsize_t)-1 ) >= 0;

                            if( !ok || ncn != 1 )
                            {
                                close();
                                return false;
                            }

                            for( i = 0; i < tile_height; i++ )
                            {
                                if(dst_bpp == 32)
                                {
                                    memcpy((float*)(data + img.step*i)+x,
                                           buffer32 + i*tile_width0*ncn,
                                           tile_width*sizeof(buffer32[0]));
                                }
                                else
                                {
                                    memcpy((double*)(data + img.step*i)+x,
                                         buffer64 + i*tile_width0*ncn,
                                         tile_width*sizeof(buffer64[0]));
                                }
                            }

                            break;
                        }
                        default:
                        {
                            close();
                            return false;
                        }
                    }
                }
            }

            result = true;
        }
    }

    close();
    return result;
}

bool TiffDecoder::readHdrData(Mat& img)
{
    int rows_per_strip = 0, photometric = 0;
    if(!m_tif)
    {
        return false;
    }
    TIFF *tif = static_cast<TIFF*>(m_tif);
    TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
    TIFFGetField( tif, TIFFTAG_PHOTOMETRIC, &photometric );
    TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_FLOAT);
    int size = 3 * m_width * m_height * sizeof (float);
    tstrip_t strip_size = 3 * m_width * rows_per_strip;
    float *ptr = img.ptr<float>();
    for (tstrip_t i = 0; i < TIFFNumberOfStrips(tif); i++, ptr += strip_size)
    {
        TIFFReadEncodedStrip(tif, i, ptr, size);
        size -= strip_size * sizeof(float);
    }
    close();
    if(photometric == PHOTOMETRIC_LOGLUV)
    {
        cvtColor(img, img, COLOR_XYZ2BGR);
    }
    else
    {
        cvtColor(img, img, COLOR_RGB2BGR);
    }
    return true;
}

#endif

//////////////////////////////////////////////////////////////////////////////////////////

TiffEncoder::TiffEncoder()
{
    m_description = "TIFF Files (*.tiff;*.tif)";
#ifdef HAVE_TIFF
    m_buf_supported = false;
#else
    m_buf_supported = true;
#endif
}

TiffEncoder::~TiffEncoder()
{
}

ImageEncoder TiffEncoder::newEncoder() const
{
    return makePtr<TiffEncoder>();
}

bool TiffEncoder::isFormatSupported( int depth ) const
{
#ifdef HAVE_TIFF
    return depth == CV_8U || depth == CV_16U || depth == CV_32F;
#else
    return depth == CV_8U || depth == CV_16U;
#endif
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

#ifdef HAVE_TIFF

static void readParam(const std::vector<int>& params, int key, int& value)
{
    for(size_t i = 0; i + 1 < params.size(); i += 2)
        if(params[i] == key)
        {
            value = params[i+1];
            break;
        }
}

bool  TiffEncoder::writeLibTiff( const Mat& img, const std::vector<int>& params)
{
    int channels = img.channels();
    int width = img.cols, height = img.rows;
    int depth = img.depth();

    int bitsPerChannel = -1;
    switch (depth)
    {
        case CV_8U:
        {
            bitsPerChannel = 8;
            break;
        }
        case CV_16U:
        {
            bitsPerChannel = 16;
            break;
        }
        default:
        {
            return false;
        }
    }

    const int bitsPerByte = 8;
    size_t fileStep = (width * channels * bitsPerChannel) / bitsPerByte;

    int rowsPerStrip = (int)((1 << 13)/fileStep);
    readParam(params, TIFFTAG_ROWSPERSTRIP, rowsPerStrip);

    if( rowsPerStrip < 1 )
        rowsPerStrip = 1;

    if( rowsPerStrip > height )
        rowsPerStrip = height;


    // do NOT put "wb" as the mode, because the b means "big endian" mode, not "binary" mode.
    // http://www.remotesensing.org/libtiff/man/TIFFOpen.3tiff.html
    TIFF* pTiffHandle = TIFFOpen(m_filename.c_str(), "w");
    if (!pTiffHandle)
    {
        return false;
    }

    // defaults for now, maybe base them on params in the future
    int   compression  = COMPRESSION_LZW;
    int   predictor    = PREDICTOR_HORIZONTAL;

    readParam(params, TIFFTAG_COMPRESSION, compression);
    readParam(params, TIFFTAG_PREDICTOR, predictor);

    int   colorspace = channels > 1 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;

    if ( !TIFFSetField(pTiffHandle, TIFFTAG_IMAGEWIDTH, width)
      || !TIFFSetField(pTiffHandle, TIFFTAG_IMAGELENGTH, height)
      || !TIFFSetField(pTiffHandle, TIFFTAG_BITSPERSAMPLE, bitsPerChannel)
      || !TIFFSetField(pTiffHandle, TIFFTAG_COMPRESSION, compression)
      || !TIFFSetField(pTiffHandle, TIFFTAG_PHOTOMETRIC, colorspace)
      || !TIFFSetField(pTiffHandle, TIFFTAG_SAMPLESPERPIXEL, channels)
      || !TIFFSetField(pTiffHandle, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG)
      || !TIFFSetField(pTiffHandle, TIFFTAG_ROWSPERSTRIP, rowsPerStrip)
      || !TIFFSetField(pTiffHandle, TIFFTAG_PREDICTOR, predictor)
       )
    {
        TIFFClose(pTiffHandle);
        return false;
    }

    // row buffer, because TIFFWriteScanline modifies the original data!
    size_t scanlineSize = TIFFScanlineSize(pTiffHandle);
    AutoBuffer<uchar> _buffer(scanlineSize+32);
    uchar* buffer = _buffer;
    if (!buffer)
    {
        TIFFClose(pTiffHandle);
        return false;
    }

    for (int y = 0; y < height; ++y)
    {
        switch(channels)
        {
            case 1:
            {
                memcpy(buffer, img.ptr(y), scanlineSize);
                break;
            }

            case 3:
            {
                if (depth == CV_8U)
                    icvCvt_BGR2RGB_8u_C3R( img.ptr(y), 0, buffer, 0, cvSize(width,1) );
                else
                    icvCvt_BGR2RGB_16u_C3R( img.ptr<ushort>(y), 0, (ushort*)buffer, 0, cvSize(width,1) );
                break;
            }

            case 4:
            {
                if (depth == CV_8U)
                    icvCvt_BGRA2RGBA_8u_C4R( img.ptr(y), 0, buffer, 0, cvSize(width,1) );
                else
                    icvCvt_BGRA2RGBA_16u_C4R( img.ptr<ushort>(y), 0, (ushort*)buffer, 0, cvSize(width,1) );
                break;
            }

            default:
            {
                TIFFClose(pTiffHandle);
                return false;
            }
        }

        int writeResult = TIFFWriteScanline(pTiffHandle, buffer, y, 0);
        if (writeResult != 1)
        {
            TIFFClose(pTiffHandle);
            return false;
        }
    }

    TIFFClose(pTiffHandle);
    return true;
}

bool TiffEncoder::writeHdr(const Mat& _img)
{
    Mat img;
    cvtColor(_img, img, COLOR_BGR2XYZ);
    TIFF* tif = TIFFOpen(m_filename.c_str(), "w");
    if (!tif)
    {
        return false;
    }
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.cols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.rows);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_SGILOG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_LOGLUV);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_FLOAT);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
    int strip_size = 3 * img.cols;
    float *ptr = const_cast<float*>(img.ptr<float>());
    for (int i = 0; i < img.rows; i++, ptr += strip_size)
    {
        TIFFWriteEncodedStrip(tif, i, ptr, strip_size * sizeof(float));
    }
    TIFFClose(tif);
    return true;
}

#endif

#ifdef HAVE_TIFF
bool  TiffEncoder::write( const Mat& img, const std::vector<int>& params)
#else
bool  TiffEncoder::write( const Mat& img, const std::vector<int>& /*params*/)
#endif
{
    int channels = img.channels();
    int width = img.cols, height = img.rows;
    int depth = img.depth();
#ifdef HAVE_TIFF
    if(img.type() == CV_32FC3)
    {
        return writeHdr(img);
    }
#endif

    if (depth != CV_8U && depth != CV_16U)
        return false;

    int bytesPerChannel = depth == CV_8U ? 1 : 2;
    int fileStep = width * channels * bytesPerChannel;

    WLByteStream strm;

    if( m_buf )
    {
        if( !strm.open(*m_buf) )
            return false;
    }
    else
    {
#ifdef HAVE_TIFF
      return writeLibTiff(img, params);
#else
      if( !strm.open(m_filename) )
          return false;
#endif
    }

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

    AutoBuffer<int> stripOffsets(stripCount);
    AutoBuffer<short> stripCounts(stripCount);
    AutoBuffer<uchar> _buffer(fileStep+32);
    uchar* buffer = _buffer;
    int  stripOffsetsOffset = 0;
    int  stripCountsOffset = 0;
    int  bitsPerSample = 8 * bytesPerChannel;
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
            {
                if (depth == CV_8U)
                    icvCvt_BGR2RGB_8u_C3R( img.ptr(y), 0, buffer, 0, cvSize(width,1) );
                else
                    icvCvt_BGR2RGB_16u_C3R( img.ptr<ushort>(y), 0, (ushort*)buffer, 0, cvSize(width,1) );
            }
            else
            {
              if( channels == 4 )
              {
                if (depth == CV_8U)
                    icvCvt_BGRA2RGBA_8u_C4R( img.ptr(y), 0, buffer, 0, cvSize(width,1) );
                else
                    icvCvt_BGRA2RGBA_16u_C4R( img.ptr<ushort>(y), 0, (ushort*)buffer, 0, cvSize(width,1) );
              }
            }

            strm.putBytes( channels > 1 ? buffer : img.ptr(y), fileStep );
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
        int bitsPerSamplePos = strm.getPos();
        strm.putWord(bitsPerSample);
        strm.putWord(bitsPerSample);
        strm.putWord(bitsPerSample);
        if( channels == 4 )
            strm.putWord(bitsPerSample);
        bitsPerSample = bitsPerSamplePos;
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
