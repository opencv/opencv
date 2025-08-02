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

#include "precomp.hpp"

#ifdef HAVE_JASPER
#include <sstream>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "grfmt_jpeg2000.hpp"
#include "opencv2/imgproc.hpp"

#ifdef _WIN32
#define JAS_WIN_MSVC_BUILD 1
#ifdef __GNUC__
#define HAVE_STDINT_H 1
#endif
#endif

#undef VERSION

#include <jasper/jasper.h>
// FIXME bad hack
#undef uchar
#undef ulong

namespace cv
{

struct JasperInitializer
{
    JasperInitializer() { jas_init(); }
    ~JasperInitializer() { jas_cleanup(); }
};

static JasperInitializer& _initJasper()
{
    static JasperInitializer initialize_jasper;
    return initialize_jasper;
}

static bool isJasperEnabled()
{
    static const bool PARAM_ENABLE_JASPER = utils::getConfigurationParameterBool("OPENCV_IO_ENABLE_JASPER",
#ifdef OPENCV_IMGCODECS_FORCE_JASPER
        true
#else
        false
#endif
    );
    return PARAM_ENABLE_JASPER;
}
static JasperInitializer& initJasper()
{
    if (isJasperEnabled())
    {
        return _initJasper();
    }
    else
    {
        const char* message = "imgcodecs: Jasper (JPEG-2000) codec is disabled. You can enable it via 'OPENCV_IO_ENABLE_JASPER' option. Refer for details and cautions here: https://github.com/opencv/opencv/issues/14058";
        CV_LOG_WARNING(NULL, message);
        CV_Error(Error::StsNotImplemented, message);
    }
}


/////////////////////// Jpeg2KDecoder ///////////////////

Jpeg2KDecoder::Jpeg2KDecoder()
{
    static const unsigned char signature_[12] = { 0, 0, 0, 0x0c, 'j', 'P', ' ', ' ', 13, 10, 0x87, 10};
    m_signature = String((const char*)signature_, (const char*)signature_ + sizeof(signature_));
    m_stream = 0;
    m_image = 0;
}


Jpeg2KDecoder::~Jpeg2KDecoder()
{
}

ImageDecoder Jpeg2KDecoder::newDecoder() const
{
    initJasper();
    return makePtr<Jpeg2KDecoder>();
}

void  Jpeg2KDecoder::close()
{
    if( m_stream )
    {
        CV_Assert(isJasperEnabled());
        jas_stream_close( (jas_stream_t*)m_stream );
        m_stream = 0;
    }

    if( m_image )
    {
        CV_Assert(isJasperEnabled());
        jas_image_destroy( (jas_image_t*)m_image );
        m_image = 0;
    }
}


bool  Jpeg2KDecoder::readHeader()
{
    CV_Assert(isJasperEnabled());
    bool result = false;

    close();
    jas_stream_t* stream = jas_stream_fopen( m_filename.c_str(), "rb" );
    m_stream = stream;

    if( stream )
    {
        jas_image_t* image = jas_image_decode( stream, -1, 0 );
        m_image = image;
        if( image ) {
            CV_Assert(0 == (jas_image_tlx(image)) && "not supported");
            CV_Assert(0 == (jas_image_tly(image)) && "not supported");
            m_width = jas_image_width( image );
            m_height = jas_image_height( image );

            int cntcmpts = 0; // count the known components
            int numcmpts = jas_image_numcmpts( image );
            int depth = 0;
            for( int i = 0; i < numcmpts; i++ )
            {
                int depth_i = jas_image_cmptprec( image, i );
                CV_Assert(depth == 0 || depth == depth_i); // component data type mismatch
                depth = MAX(depth, depth_i);
                if( jas_image_cmpttype( image, i ) > 2 )
                    continue;
                int sgnd = jas_image_cmptsgnd(image, i);
                int xstart = jas_image_cmpttlx(image, i);
                int xend = jas_image_cmptbrx(image, i);
                int xstep = jas_image_cmpthstep(image, i);
                int ystart = jas_image_cmpttly(image, i);
                int yend = jas_image_cmptbry(image, i);
                int ystep = jas_image_cmptvstep(image, i);
                CV_Assert(sgnd == 0 && "not supported");
                CV_Assert(xstart == 0 && "not supported");
                CV_Assert(ystart == 0 && "not supported");
                CV_Assert(xstep == 1 && "not supported");
                CV_Assert(ystep == 1 && "not supported");
                CV_Assert(xend == m_width);
                CV_Assert(yend == m_height);
                cntcmpts++;
            }

            if( cntcmpts )
            {
                CV_Assert(depth == 8 || depth == 16);
                CV_Assert(cntcmpts == 1 || cntcmpts == 3);
                m_type = CV_MAKETYPE(depth <= 8 ? CV_8U : CV_16U, cntcmpts > 1 ? 3 : 1);
                result = true;
            }
        }
    }

    if( !result )
        close();

    return result;
}

static void Jpeg2KDecoder_close(Jpeg2KDecoder* ptr)
{
    ptr->close();
}

bool  Jpeg2KDecoder::readData( Mat& img )
{
    CV_Assert(isJasperEnabled());

    Ptr<Jpeg2KDecoder> close_this(this, Jpeg2KDecoder_close);
    bool result = false;
    bool color = img.channels() > 1;
    uchar* data = img.ptr();
    size_t step = img.step;
    jas_stream_t* stream = (jas_stream_t*)m_stream;
    jas_image_t* image = (jas_image_t*)m_image;

#ifndef _WIN32
    // At least on some Linux instances the
    // system libjasper segfaults when
    // converting color to grey.
    // We do this conversion manually at the end.
    Mat clr;
    if (CV_MAT_CN(img.type()) < CV_MAT_CN(this->type()))
    {
        clr.create(img.size().height, img.size().width, this->type());
        color = true;
        data = clr.ptr();
        step = (int)clr.step;
    }
#endif

    if( stream && image )
    {
        bool convert;
        int colorspace;
        if( color )
        {
            convert = (jas_image_clrspc( image ) != JAS_CLRSPC_SRGB);
            colorspace = JAS_CLRSPC_SRGB;
        }
        else
        {
            convert = (jas_clrspc_fam( jas_image_clrspc( image ) ) != JAS_CLRSPC_FAM_GRAY);
            colorspace = JAS_CLRSPC_SGRAY; // TODO GENGRAY or SGRAY? (GENGRAY fails on Win.)
        }

        // convert to the desired colorspace
        if( convert )
        {
            jas_cmprof_t *clrprof = jas_cmprof_createfromclrspc( colorspace );
            if( clrprof )
            {
                jas_image_t *_img = jas_image_chclrspc( image, clrprof, JAS_CMXFORM_INTENT_RELCLR );
                if( _img )
                {
                    jas_image_destroy( image );
                    m_image = image = _img;
                    result = true;
                }
                else
                {
                    jas_cmprof_destroy(clrprof);
                    CV_Error(Error::StsError, "JPEG 2000 LOADER ERROR: cannot convert colorspace");
                }
                jas_cmprof_destroy( clrprof );
            }
            else
            {
                CV_Error(Error::StsError, "JPEG 2000 LOADER ERROR: unable to create colorspace");
            }
        }
        else
            result = true;

        if( result )
        {
            int ncmpts;
            int cmptlut[3];
            int swap_rb = m_use_rgb ? 0 : 2;
            if( color )
            {
                cmptlut[0] = jas_image_getcmptbytype( image, swap_rb );
                cmptlut[1] = jas_image_getcmptbytype( image, 1 );
                cmptlut[2] = jas_image_getcmptbytype( image, swap_rb^2 );
                if( cmptlut[0] < 0 || cmptlut[1] < 0 || cmptlut[2] < 0 )
                    result = false;
                ncmpts = 3;
            }
            else
            {
                cmptlut[0] = jas_image_getcmptbytype( image, JAS_IMAGE_CT_GRAY_Y );
                if( cmptlut[0] < 0 )
                    result = false;
                ncmpts = 1;
            }

            if( result )
            {
                for( int i = 0; i < ncmpts; i++ )
                {
                    int maxval = 1 << jas_image_cmptprec( image, cmptlut[i] );
                    int offset =  jas_image_cmptsgnd( image, cmptlut[i] ) ? maxval / 2 : 0;

                    int yend = jas_image_cmptbry( image, cmptlut[i] );
                    int ystep = jas_image_cmptvstep( image, cmptlut[i] );
                    int xend = jas_image_cmptbrx( image, cmptlut[i] );
                    int xstep = jas_image_cmpthstep( image, cmptlut[i] );

                    jas_matrix_t *buffer = jas_matrix_create( yend / ystep, xend / xstep );
                    if( buffer )
                    {
                        if( !jas_image_readcmpt( image, cmptlut[i], 0, 0, xend / xstep, yend / ystep, buffer ))
                        {
                            if( img.depth() == CV_8U )
                                result = readComponent8u( data + i, buffer, validateToInt(step), cmptlut[i], maxval, offset, ncmpts );
                            else
                                result = readComponent16u( ((unsigned short *)data) + i, buffer, validateToInt(step / 2), cmptlut[i], maxval, offset, ncmpts );
                            if( !result )
                            {
                                jas_matrix_destroy( buffer );
                                CV_Error(Error::StsError, "JPEG2000 LOADER ERROR: failed to read component");
                            }
                        }
                        jas_matrix_destroy( buffer );
                    }
                }
            }
        }
        else
        {
            CV_Error(Error::StsError, "JPEG2000 LOADER ERROR: colorspace conversion failed");
        }
    }

    CV_Assert(result == true);

#ifndef _WIN32
    if (!clr.empty())
    {
        cv::cvtColor(clr, img, COLOR_BGR2GRAY);
    }
#endif

    return result;
}


bool  Jpeg2KDecoder::readComponent8u( uchar *data, void *_buffer,
                                      int step, int cmpt,
                                      int maxval, int offset, int ncmpts )
{
    CV_Assert(isJasperEnabled());

    jas_matrix_t* buffer = (jas_matrix_t*)_buffer;
    jas_image_t* image = (jas_image_t*)m_image;
    int xstart = jas_image_cmpttlx( image, cmpt );
    int xend = jas_image_cmptbrx( image, cmpt );
    int xstep = jas_image_cmpthstep( image, cmpt );
    int xoffset = jas_image_tlx( image );
    int ystart = jas_image_cmpttly( image, cmpt );
    int yend = jas_image_cmptbry( image, cmpt );
    int ystep = jas_image_cmptvstep( image, cmpt );
    int yoffset = jas_image_tly( image );
    int x, y, x1, y1, j;
    int rshift = cvRound(std::log(maxval/256.)/std::log(2.));
    int lshift = MAX(0, -rshift);
    rshift = MAX(0, rshift);
    int delta = (rshift > 0 ? 1 << (rshift - 1) : 0) + offset;

    for( y = 0; y < yend - ystart; )
    {
        jas_seqent_t* pix_row = jas_matrix_getref( buffer, y / ystep, 0 );
        uchar* dst = data + (y - yoffset) * step - xoffset;

        if( xstep == 1 )
        {
            if( maxval == 256 && offset == 0 )
                for( x = 0; x < xend - xstart; x++ )
                {
                    int pix = pix_row[x];
                    dst[x*ncmpts] = cv::saturate_cast<uchar>(pix);
                }
            else
                for( x = 0; x < xend - xstart; x++ )
                {
                    int pix = ((pix_row[x] + delta) >> rshift) << lshift;
                    dst[x*ncmpts] = cv::saturate_cast<uchar>(pix);
                }
        }
        else if( xstep == 2 && offset == 0 )
            for( x = 0, j = 0; x < xend - xstart; x += 2, j++ )
            {
                int pix = ((pix_row[j] + delta) >> rshift) << lshift;
                dst[x*ncmpts] = dst[(x+1)*ncmpts] = cv::saturate_cast<uchar>(pix);
            }
        else
            for( x = 0, j = 0; x < xend - xstart; j++ )
            {
                int pix = ((pix_row[j] + delta) >> rshift) << lshift;
                pix = cv::saturate_cast<uchar>(pix);
                for( x1 = x + xstep; x < x1; x++ )
                    dst[x*ncmpts] = (uchar)pix;
            }
        y1 = y + ystep;
        for( ++y; y < y1; y++, dst += step )
            for( x = 0; x < xend - xstart; x++ )
                dst[x*ncmpts + step] = dst[x*ncmpts];
    }

    return true;
}


bool  Jpeg2KDecoder::readComponent16u( unsigned short *data, void *_buffer,
                                       int step, int cmpt,
                                       int maxval, int offset, int ncmpts )
{
    CV_Assert(isJasperEnabled());

    jas_matrix_t* buffer = (jas_matrix_t*)_buffer;
    jas_image_t* image = (jas_image_t*)m_image;
    int xstart = jas_image_cmpttlx( image, cmpt );
    int xend = jas_image_cmptbrx( image, cmpt );
    int xstep = jas_image_cmpthstep( image, cmpt );
    int xoffset = jas_image_tlx( image );
    int ystart = jas_image_cmpttly( image, cmpt );
    int yend = jas_image_cmptbry( image, cmpt );
    int ystep = jas_image_cmptvstep( image, cmpt );
    int yoffset = jas_image_tly( image );
    int x, y, x1, y1, j;
    int rshift = cvRound(std::log(maxval/65536.)/std::log(2.));
    int lshift = MAX(0, -rshift);
    rshift = MAX(0, rshift);
    int delta = (rshift > 0 ? 1 << (rshift - 1) : 0) + offset;

    for( y = 0; y < yend - ystart; )
    {
        jas_seqent_t* pix_row = jas_matrix_getref( buffer, y / ystep, 0 );
        ushort* dst = data + (y - yoffset) * step - xoffset;

        if( xstep == 1 )
        {
            if( maxval == 65536 && offset == 0 )
                for( x = 0; x < xend - xstart; x++ )
                {
                    int pix = pix_row[x];
                    dst[x*ncmpts] = cv::saturate_cast<ushort>(pix);
                }
            else
                for( x = 0; x < xend - xstart; x++ )
                {
                    int pix = ((pix_row[x] + delta) >> rshift) << lshift;
                    dst[x*ncmpts] = cv::saturate_cast<ushort>(pix);
                }
        }
        else if( xstep == 2 && offset == 0 )
            for( x = 0, j = 0; x < xend - xstart; x += 2, j++ )
            {
                int pix = ((pix_row[j] + delta) >> rshift) << lshift;
                dst[x*ncmpts] = dst[(x+1)*ncmpts] = cv::saturate_cast<ushort>(pix);
            }
        else
            for( x = 0, j = 0; x < xend - xstart; j++ )
            {
                int pix = ((pix_row[j] + delta) >> rshift) << lshift;
                pix = cv::saturate_cast<ushort>(pix);
                for( x1 = x + xstep; x < x1; x++ )
                    dst[x*ncmpts] = (ushort)pix;
            }
        y1 = y + ystep;
        for( ++y; y < y1; y++, dst += step )
            for( x = 0; x < xend - xstart; x++ )
                dst[x*ncmpts + step] = dst[x*ncmpts];
    }

    return true;
}


/////////////////////// Jpeg2KEncoder ///////////////////


Jpeg2KEncoder::Jpeg2KEncoder()
{
    m_description = "JPEG-2000 files (*.jp2)";
}


Jpeg2KEncoder::~Jpeg2KEncoder()
{
}

ImageEncoder Jpeg2KEncoder::newEncoder() const
{
    initJasper();
    return makePtr<Jpeg2KEncoder>();
}

bool  Jpeg2KEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U;
}


bool  Jpeg2KEncoder::write( const Mat& _img, const std::vector<int>& params )
{
    CV_Assert(isJasperEnabled());
    int width = _img.cols, height = _img.rows;
    int depth = _img.depth(), channels = _img.channels();
    depth = depth == CV_8U ? 8 : 16;

    if( channels > 3 || channels < 1 )
        return false;

    CV_Assert(params.size() % 2 == 0);
    double target_compression_rate = 1.0;
    for( size_t i = 0; i < params.size(); i += 2 )
    {
        switch(params[i])
        {
        case cv::IMWRITE_JPEG2000_COMPRESSION_X1000:
            target_compression_rate = std::min(std::max(params[i+1], 0), 1000) / 1000.0;
            break;
        }
    }

    jas_image_cmptparm_t component_info[3];
    for( int i = 0; i < channels; i++ )
    {
        component_info[i].tlx = 0;
        component_info[i].tly = 0;
        component_info[i].hstep = 1;
        component_info[i].vstep = 1;
        component_info[i].width = width;
        component_info[i].height = height;
        component_info[i].prec = depth;
        component_info[i].sgnd = 0;
    }
    jas_image_t *img = jas_image_create( channels, component_info, (channels == 1) ? JAS_CLRSPC_SGRAY : JAS_CLRSPC_SRGB );
    if( !img )
        return false;

    if(channels == 1)
        jas_image_setcmpttype( img, 0, JAS_IMAGE_CT_GRAY_Y );
    else
    {
        jas_image_setcmpttype( img, 0, JAS_IMAGE_CT_RGB_B );
        jas_image_setcmpttype( img, 1, JAS_IMAGE_CT_RGB_G );
        jas_image_setcmpttype( img, 2, JAS_IMAGE_CT_RGB_R );
    }

    bool result;
    if( depth == 8 )
        result = writeComponent8u( img, _img );
    else
        result = writeComponent16u( img, _img );
    if( result )
    {
        jas_stream_t *stream = jas_stream_fopen( m_filename.c_str(), "wb" );
        if( stream )
        {
            std::stringstream options;
            options << "rate=" << target_compression_rate;

            result = !jas_image_encode( img, stream, jas_image_strtofmt( (char*)"jp2" ), (char*)options.str().c_str() );

            jas_stream_close( stream );
        }

    }
    jas_image_destroy( img );

    return result;
}


bool  Jpeg2KEncoder::writeComponent8u( void *__img, const Mat& _img )
{
    CV_Assert(isJasperEnabled());

    jas_image_t* img = (jas_image_t*)__img;
    int w = _img.cols, h = _img.rows, ncmpts = _img.channels();
    jas_matrix_t *row = jas_matrix_create( 1, w );
    if(!row)
        return false;

    for( int y = 0; y < h; y++ )
    {
        const uchar* data = _img.ptr(y);
        for( int i = 0; i < ncmpts; i++ )
        {
            for( int x = 0; x < w; x++)
                jas_matrix_setv( row, x, data[x * ncmpts + i] );
            jas_image_writecmpt( img, i, 0, y, w, 1, row );
        }
    }

    jas_matrix_destroy( row );
    return true;
}


bool  Jpeg2KEncoder::writeComponent16u( void *__img, const Mat& _img )
{
    CV_Assert(isJasperEnabled());

    jas_image_t* img = (jas_image_t*)__img;
    int w = _img.cols, h = _img.rows, ncmpts = _img.channels();
    jas_matrix_t *row = jas_matrix_create( 1, w );
    if(!row)
        return false;

    for( int y = 0; y < h; y++ )
    {
        const ushort* data = _img.ptr<ushort>(y);
        for( int i = 0; i < ncmpts; i++ )
        {
            for( int x = 0; x < w; x++)
                jas_matrix_setv( row, x, data[x * ncmpts + i] );
            jas_image_writecmpt( img, i, 0, y, w, 1, row );
        }
    }

    jas_matrix_destroy( row );

    return true;
}

bool Jpeg2KEncoder::isValidParam(const int key, const int value) const
{
    bool ret = false;
    switch(key)
    {
        case IMWRITE_JPEG2000_COMPRESSION_X1000:
            ret = (0 <= value) && (value <= 1000);
            break;
        default:
            break;
    }
    return ret;
}

}

#endif

/* End of file. */
