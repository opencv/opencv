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

#ifdef HAVE_TIFF
#include "grfmt_tiff.hpp"
#include <limits>

// TODO FIXIT Conflict declarations for common types like int64/uint64
namespace tiff_dummy_namespace {
#include "tiff.h"
#include "tiffio.h"
}
using namespace tiff_dummy_namespace;

namespace cv
{


static const char fmtSignTiffII[] = "II\x2a\x00";
static const char fmtSignTiffMM[] = "MM\x00\x2a";

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
    m_buf_supported = true;
    m_buf_pos = 0;
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

class TiffDecoderBufHelper
{
    Mat& m_buf;
    size_t& m_buf_pos;
public:
    TiffDecoderBufHelper(Mat& buf, size_t& buf_pos) :
        m_buf(buf), m_buf_pos(buf_pos)
    {}
    static tmsize_t read( thandle_t handle, void* buffer, tmsize_t n )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        const Mat& buf = helper->m_buf;
        const tmsize_t size = buf.cols*buf.rows*buf.elemSize();
        tmsize_t pos = helper->m_buf_pos;
        if ( n > (size - pos) )
        {
            n = size - pos;
        }
        memcpy(buffer, buf.ptr() + pos, n);
        helper->m_buf_pos += n;
        return n;
    }

    static tmsize_t write( thandle_t /*handle*/, void* /*buffer*/, tmsize_t /*n*/ )
    {
        // Not used for decoding.
        return 0;
    }

    static toff_t seek( thandle_t handle, toff_t offset, int whence )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        const Mat& buf = helper->m_buf;
        const toff_t size = buf.cols*buf.rows*buf.elemSize();
        toff_t new_pos = helper->m_buf_pos;
        switch (whence)
        {
            case SEEK_SET:
                new_pos = offset;
                break;
            case SEEK_CUR:
                new_pos += offset;
                break;
            case SEEK_END:
                new_pos = size + offset;
                break;
        }
        new_pos = std::min(new_pos, size);
        helper->m_buf_pos = (size_t)new_pos;
        return new_pos;
    }

    static int map( thandle_t handle, void** base, toff_t* size )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        Mat& buf = helper->m_buf;
        *base = buf.ptr();
        *size = buf.cols*buf.rows*buf.elemSize();
        return 0;
    }

    static toff_t size( thandle_t handle )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        const Mat& buf = helper->m_buf;
        return buf.cols*buf.rows*buf.elemSize();
    }

    static int close( thandle_t handle )
    {
        TiffDecoderBufHelper *helper = reinterpret_cast<TiffDecoderBufHelper*>(handle);
        delete helper;
        return 0;
    }
};

bool TiffDecoder::readHeader()
{
    bool result = false;

    TIFF* tif = static_cast<TIFF*>(m_tif);
    if (!m_tif)
    {
        // TIFFOpen() mode flags are different to fopen().  A 'b' in mode "rb" has no effect when reading.
        // http://www.remotesensing.org/libtiff/man/TIFFOpen.3tiff.html
        if ( !m_buf.empty() )
        {
            m_buf_pos = 0;
            TiffDecoderBufHelper* buf_helper = new TiffDecoderBufHelper(this->m_buf, this->m_buf_pos);
            tif = TIFFClientOpen( "", "r", reinterpret_cast<thandle_t>(buf_helper), &TiffDecoderBufHelper::read,
                                  &TiffDecoderBufHelper::write, &TiffDecoderBufHelper::seek,
                                  &TiffDecoderBufHelper::close, &TiffDecoderBufHelper::size,
                                  &TiffDecoderBufHelper::map, /*unmap=*/0 );
        }
        else
        {
            tif = TIFFOpen(m_filename.c_str(), "r");
        }
    }

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
               ((photometric > 2) ||
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

bool TiffDecoder::nextPage()
{
    // Prepare the next page, if any.
    return m_tif &&
           TIFFReadDirectory(static_cast<TIFF*>(m_tif)) &&
           readHeader();
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

            if( tile_height0 <= 0 ||
               (!is_tiled && tile_height0 == std::numeric_limits<uint32>::max()) )
                tile_height0 = m_height;

            if(dst_bpp == 8) {
                // we will use TIFFReadRGBA* functions, so allocate temporary buffer for 32bit RGBA
                bpp = 8;
                ncn = 4;
            }
            const size_t buffer_size = (bpp/bitsPerByte) * ncn * tile_height0 * tile_width0;
            AutoBuffer<uchar> _buffer( buffer_size );
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
                                ok = (int)TIFFReadEncodedStrip( tif, tileidx, (uint32*)buffer, buffer_size ) >= 0;
                            else
                                ok = (int)TIFFReadEncodedTile( tif, tileidx, (uint32*)buffer, buffer_size ) >= 0;

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
                                ok = (int)TIFFReadEncodedStrip( tif, tileidx, buffer, buffer_size ) >= 0;
                            else
                                ok = (int)TIFFReadEncodedTile( tif, tileidx, buffer, buffer_size ) >= 0;

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
    return makePtr<TiffEncoder>();
}

bool TiffEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U || depth == CV_32F;
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

class TiffEncoderBufHelper
{
public:

    TiffEncoderBufHelper(std::vector<uchar> *buf)
            : m_buf(buf), m_buf_pos(0)
    {}

    TIFF* open ()
    {
        return TIFFClientOpen( "", "w", reinterpret_cast<thandle_t>(this), &TiffEncoderBufHelper::read,
                               &TiffEncoderBufHelper::write, &TiffEncoderBufHelper::seek,
                               &TiffEncoderBufHelper::close, &TiffEncoderBufHelper::size,
                               /*map=*/0, /*unmap=*/0 );
    }

    static tmsize_t read( thandle_t /*handle*/, void* /*buffer*/, tmsize_t /*n*/ )
    {
        // Not used for encoding.
        return 0;
    }

    static tmsize_t write( thandle_t handle, void* buffer, tmsize_t n )
    {
        TiffEncoderBufHelper *helper = reinterpret_cast<TiffEncoderBufHelper*>(handle);
        size_t begin = (size_t)helper->m_buf_pos;
        size_t end = begin + n;
        if ( helper->m_buf->size() < end )
        {
            helper->m_buf->resize(end);
        }
        memcpy(&(*helper->m_buf)[begin], buffer, n);
        helper->m_buf_pos = end;
        return n;
    }

    static toff_t seek( thandle_t handle, toff_t offset, int whence )
    {
        TiffEncoderBufHelper *helper = reinterpret_cast<TiffEncoderBufHelper*>(handle);
        const toff_t size = helper->m_buf->size();
        toff_t new_pos = helper->m_buf_pos;
        switch (whence)
        {
            case SEEK_SET:
                new_pos = offset;
                break;
            case SEEK_CUR:
                new_pos += offset;
                break;
            case SEEK_END:
                new_pos = size + offset;
                break;
        }
        helper->m_buf_pos = new_pos;
        return new_pos;
    }

    static toff_t size( thandle_t handle )
    {
        TiffEncoderBufHelper *helper = reinterpret_cast<TiffEncoderBufHelper*>(handle);
        return helper->m_buf->size();
    }

    static int close( thandle_t /*handle*/ )
    {
        // Do nothing.
        return 0;
    }

private:

    std::vector<uchar>* m_buf;
    toff_t m_buf_pos;
};

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
    TIFF* pTiffHandle;

    TiffEncoderBufHelper buf_helper(m_buf);
    if ( m_buf )
    {
        pTiffHandle = buf_helper.open();
    }
    else
    {
        pTiffHandle = TIFFOpen(m_filename.c_str(), "w");
    }
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
       )
    {
        TIFFClose(pTiffHandle);
        return false;
    }

    if (compression != COMPRESSION_NONE && !TIFFSetField(pTiffHandle, TIFFTAG_PREDICTOR, predictor) )
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

    TIFF* tif;

    TiffEncoderBufHelper buf_helper(m_buf);
    if ( m_buf )
    {
        tif = buf_helper.open();
    }
    else
    {
        tif = TIFFOpen(m_filename.c_str(), "w");
    }

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

bool  TiffEncoder::write( const Mat& img, const std::vector<int>& params)
{
    int depth = img.depth();

    if(img.type() == CV_32FC3)
    {
        return writeHdr(img); // TODO Rename
    }

    CV_Assert(depth == CV_8U || depth == CV_16U);

    return writeLibTiff(img, params);
}

} // namespace

#endif
