/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                            License Agreement
//                 For Open Source Computer Vision Library
//                         (3-clause BSD License)
//
//  Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
//  Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
//  Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
//  Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
//  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
//  Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
//  Third party copyrights are property of their respective owners.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright notice,
//      this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//
//    * Neither the names of the copyright holders nor the names of the contributors
//      may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall copyright holders or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"


#ifdef HAVE_IMGCODEC_PXM

#include <cerrno>

#include "utils.hpp"
#include "grfmt_pam.hpp"

namespace cv {

/* the PAM related fields */
#define MAX_PAM_HEADER_IDENITFIER_LENGTH 8
#define MAX_PAM_HEADER_VALUE_LENGTH 255

/* PAM header fields */
typedef enum {
    PAM_HEADER_NONE,
    PAM_HEADER_COMMENT,
    PAM_HEADER_ENDHDR,
    PAM_HEADER_HEIGHT,
    PAM_HEADER_WIDTH,
    PAM_HEADER_DEPTH,
    PAM_HEADER_MAXVAL,
    PAM_HEADER_TUPLTYPE,
} PamHeaderFieldType;

struct pam_header_field {
    PamHeaderFieldType type;
    char identifier[MAX_PAM_HEADER_IDENITFIER_LENGTH+1];
};

const static struct pam_header_field fields[] = {
    {PAM_HEADER_ENDHDR,   "ENDHDR"},
    {PAM_HEADER_HEIGHT,   "HEIGHT"},
    {PAM_HEADER_WIDTH,    "WIDTH"},
    {PAM_HEADER_DEPTH,    "DEPTH"},
    {PAM_HEADER_MAXVAL,   "MAXVAL"},
    {PAM_HEADER_TUPLTYPE, "TUPLTYPE"},
};
#define PAM_FIELDS_NO (sizeof (fields) / sizeof ((fields)[0]))

typedef bool (*cvtFunc) (void *src, void *target, int width, int target_channels,
    int target_depth, bool use_rgb);

struct channel_layout {
    uint rchan, gchan, bchan, graychan;
};

struct pam_format {
    uint fmt;
    char name[MAX_PAM_HEADER_VALUE_LENGTH+1];
    cvtFunc cvt_func;
    /* the channel layout that should be used when
     * imread_ creates a 3 channel or 1 channel image
     * used when no conversion function is available
     */
    struct channel_layout layout;
};

static bool rgb_convert (void *src, void *target, int width, int target_channels,
    int target_depth, bool use_rgb);

const static struct pam_format formats[] = {
    {IMWRITE_PAM_FORMAT_NULL, "", NULL, {0, 0, 0, 0} },
    {IMWRITE_PAM_FORMAT_BLACKANDWHITE, "BLACKANDWHITE", NULL, {0, 0, 0, 0} },
    {IMWRITE_PAM_FORMAT_GRAYSCALE, "GRAYSCALE", NULL, {0, 0, 0, 0} },
    {IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA, "GRAYSCALE_ALPHA", NULL, {0, 0, 0, 0} },
    {IMWRITE_PAM_FORMAT_RGB, "RGB", rgb_convert, {0, 1, 2, 0} },
    {IMWRITE_PAM_FORMAT_RGB_ALPHA, "RGB_ALPHA", NULL, {0, 1, 2, 0} },
};
#define PAM_FORMATS_NO (sizeof (fields) / sizeof ((fields)[0]))

/*
 * conversion functions
 */

static bool
rgb_convert (void *src, void *target, int width, int target_channels, int target_depth, bool use_rgb)
{
    bool ret = false;
    if (target_channels == 3) {
        switch (target_depth) {
            case CV_8U:
                if (use_rgb)
                    memcpy(target, src, sizeof(uchar) * width);
                else
                    icvCvt_RGB2BGR_8u_C3R( (uchar*) src, 0, (uchar*) target, 0,
                                           Size(width,1) );
                ret = true;
                break;
            case CV_16U:
                if (use_rgb)
                    memcpy(target, src, sizeof(ushort) * width);
                else
                    icvCvt_RGB2BGR_16u_C3R( (ushort *)src, 0, (ushort *)target, 0,
                                            Size(width,1) );
                ret = true;
                break;
            default:
                break;
        }
    } else if (target_channels == 1) {
        switch (target_depth) {
            case CV_8U:
                icvCvt_BGR2Gray_8u_C3C1R( (uchar*) src, 0, (uchar*) target, 0,
                    Size(width,1), 2 );
                ret = true;
                break;
            case CV_16U:
                icvCvt_BGRA2Gray_16u_CnC1R( (ushort *)src, 0, (ushort *)target, 0,
                    Size(width,1), 3, 2 );
                ret = true;
                break;
            default:
                break;
        }
    }
    return ret;
}

/*
 * copy functions used as a fall back for undefined formats
 * or simpler conversion options
 */

static void
basic_conversion (void *src, const struct channel_layout *layout, int src_sampe_size,
    int src_width, void *target, int target_channels, int target_depth, bool use_rgb)
{
    switch (target_depth) {
        case CV_8U:
        {
            uchar *d = (uchar *)target, *s = (uchar *)src,
                *end = ((uchar *)src) + src_width;
            switch (target_channels) {
                case 1:
                    for( ; s < end; d += 3, s += src_sampe_size )
                        d[0] = d[1] = d[2] = s[layout->graychan];
                    break;
                case 3:
                    if (use_rgb)
                        for( ; s < end; d += 3, s += src_sampe_size ) {
                            d[0] = s[layout->rchan];
                            d[1] = s[layout->gchan];
                            d[2] = s[layout->bchan];
                        }
                    else
                        for( ; s < end; d += 3, s += src_sampe_size ) {
                            d[0] = s[layout->bchan];
                            d[1] = s[layout->gchan];
                            d[2] = s[layout->rchan];
                        }
                    break;
                default:
                    CV_Error(Error::StsInternal, "");
            }
            break;
        }
        case CV_16U:
        {
            ushort *d = (ushort *)target, *s = (ushort *)src,
                *end = ((ushort *)src) + src_width;
            switch (target_channels) {
                case 1:
                    for( ; s < end; d += 3, s += src_sampe_size )
                        d[0] = d[1] = d[2] = s[layout->graychan];
                    break;
                case 3:
                    if (use_rgb)
                        for( ; s < end; d += 3, s += src_sampe_size ) {
                            d[0] = s[layout->rchan];
                            d[1] = s[layout->gchan];
                            d[2] = s[layout->bchan];
                        }
                    else
                        for( ; s < end; d += 3, s += src_sampe_size ) {
                            d[0] = s[layout->bchan];
                            d[1] = s[layout->gchan];
                            d[2] = s[layout->rchan];
                        }
                    break;
                default:
                    CV_Error(Error::StsInternal, "");
            }
            break;
        }
        default:
            CV_Error(Error::StsInternal, "");
    }
}


static
bool ReadPAMHeaderLine(
        cv::RLByteStream& strm,
        CV_OUT PamHeaderFieldType &fieldtype,
        CV_OUT char value[MAX_PAM_HEADER_VALUE_LENGTH+1])
{
    int code;
    char ident[MAX_PAM_HEADER_IDENITFIER_LENGTH+1] = {};

    do {
        code = strm.getByte();
    } while ( isspace(code) );

    if (code == '#') {
        /* we are in a comment, eat characters until linebreak */
        do
        {
            code = strm.getByte();
        } while( code != '\n' && code != '\r' );
        fieldtype = PAM_HEADER_COMMENT;
        return true;
    } else if (code == '\n' || code == '\r' ) {
        fieldtype = PAM_HEADER_NONE;
        return true;
    }

    int ident_sz = 0;
    for (; ident_sz < MAX_PAM_HEADER_IDENITFIER_LENGTH; ident_sz++)
    {
        if (isspace(code))
            break;
        ident[ident_sz] = (char)code;
        code = strm.getByte();
    }
    CV_DbgAssert(ident_sz <= MAX_PAM_HEADER_IDENITFIER_LENGTH);
    ident[ident_sz] = 0;

    /* we may have filled the buffer and still have data */
    if (!isspace(code))
        return false;

    bool ident_found = false;
    for (uint i = 0; i < PAM_FIELDS_NO; i++)
    {
        if (0 == strncmp(fields[i].identifier, ident, std::min(ident_sz, MAX_PAM_HEADER_IDENITFIER_LENGTH) + 1))
        {
            fieldtype = fields[i].type;
            ident_found = true;
            break;
        }
    }

    if (!ident_found)
        return false;

    memset(value, 0, sizeof(char) * (MAX_PAM_HEADER_VALUE_LENGTH + 1));
    /* we may have an identifier that has no value */
    if (code == '\n' || code == '\r')
        return true;

    do {
        code = strm.getByte();
    } while (isspace(code));

    /* read identifier value */
    int value_sz = 0;
    for (; value_sz < MAX_PAM_HEADER_VALUE_LENGTH; value_sz++)
    {
        if (code == '\n' || code == '\r')
            break;
        value[value_sz] = (char)code;
        code = strm.getByte();
    }
    CV_DbgAssert(value_sz <= MAX_PAM_HEADER_VALUE_LENGTH);
    value[value_sz] = 0;

    int pos = value_sz;

    /* should be terminated */
    if (code != '\n' && code != '\r')
        return false;

    /* remove trailing white spaces */
    while (--pos >= 0 && isspace(value[pos]))
        value[pos] = 0;

    return true;
}

static int ParseInt(const char *str, int len)
{
    CV_Assert(len > 0);

    int pos = 0;
    bool is_negative = false;
    if (str[0] == '-')
    {
        is_negative = true;
        pos++;
        CV_Assert(isdigit(str[pos]));
    }
    uint64_t number = 0;
    while (pos < len && isdigit(str[pos]))
    {
        char ch = str[pos];
        number = (number * 10) + (uint64_t)((int)ch - (int)'0');
        CV_Assert(number < INT_MAX);
        pos++;
    }
    if (pos < len)
        CV_Assert(str[pos] == 0);
    return (is_negative) ? -(int)number : (int)number;
}



PAMDecoder::PAMDecoder()
{
    m_offset = -1;
    m_buf_supported = true;
    bit_mode = false;
    selected_fmt = IMWRITE_PAM_FORMAT_NULL;
    m_maxval = 0;
    m_channels = 0;
    m_sampledepth = 0;
}


PAMDecoder::~PAMDecoder()
{
    m_strm.close();
}

size_t PAMDecoder::signatureLength() const
{
    return 3;
}

bool PAMDecoder::checkSignature( const String& signature ) const
{
    return signature.size() >= 3 && signature[0] == 'P' &&
           signature[1] == '7' &&
           isspace(signature[2]);
}

ImageDecoder PAMDecoder::newDecoder() const
{
    return makePtr<PAMDecoder>();
}

bool PAMDecoder::readHeader()
{
    PamHeaderFieldType fieldtype = PAM_HEADER_NONE;
    char value[MAX_PAM_HEADER_VALUE_LENGTH+1];
    int byte;

    if( !m_buf.empty() )
    {
        if( !m_strm.open(m_buf) )
            return false;
    }
    else if( !m_strm.open( m_filename ))
        return false;

    try
    {
        byte = m_strm.getByte();
        if( byte != 'P' )
            throw RBS_BAD_HEADER;

        byte = m_strm.getByte();
        if (byte != '7')
            throw RBS_BAD_HEADER;

        byte = m_strm.getByte();
        if (byte != '\n' && byte != '\r')
            throw RBS_BAD_HEADER;

        bool flds_endhdr = false, flds_height = false, flds_width = false, flds_depth = false, flds_maxval = false;

        do {
            if (!ReadPAMHeaderLine(m_strm, fieldtype, value))
                throw RBS_BAD_HEADER;
            switch (fieldtype)
            {
                case PAM_HEADER_NONE:
                case PAM_HEADER_COMMENT:
                    continue;
                case PAM_HEADER_ENDHDR:
                    flds_endhdr = true;
                    break;
                case PAM_HEADER_HEIGHT:
                    if (flds_height)
                        throw RBS_BAD_HEADER;
                    m_height = ParseInt(value, MAX_PAM_HEADER_VALUE_LENGTH);
                    flds_height = true;
                    break;
                case PAM_HEADER_WIDTH:
                    if (flds_width)
                        throw RBS_BAD_HEADER;
                    m_width = ParseInt(value, MAX_PAM_HEADER_VALUE_LENGTH);
                    flds_width = true;
                    break;
                case PAM_HEADER_DEPTH:
                    if (flds_depth)
                        throw RBS_BAD_HEADER;
                    m_channels = ParseInt(value, MAX_PAM_HEADER_VALUE_LENGTH);
                    flds_depth = true;
                    break;
                case PAM_HEADER_MAXVAL:
                    if (flds_maxval)
                        throw RBS_BAD_HEADER;
                    m_maxval = ParseInt(value, MAX_PAM_HEADER_VALUE_LENGTH);
                    if ( m_maxval > 65535 )
                        throw RBS_BAD_HEADER;
                    m_sampledepth = (m_maxval > 255) ? CV_16U : CV_8U;
                    if (m_maxval == 1)
                        bit_mode = true;
                    flds_maxval = true;
                    break;
                case PAM_HEADER_TUPLTYPE:
                {
                    bool format_found = false;
                    for (uint i=0; i<PAM_FORMATS_NO; i++)
                    {
                        if (0 == strncmp(formats[i].name, value, MAX_PAM_HEADER_VALUE_LENGTH+1))
                        {
                            selected_fmt = formats[i].fmt;
                            format_found = true;
                            break;
                        }
                    }
                    CV_Assert(format_found);
                    break;
                }
                default:
                    throw RBS_BAD_HEADER;
            }
        } while (fieldtype != PAM_HEADER_ENDHDR);

        if (flds_endhdr && flds_height && flds_width && flds_depth && flds_maxval)
        {
            if (selected_fmt == IMWRITE_PAM_FORMAT_NULL)
            {
                if (m_channels == 1 && m_maxval == 1)
                    selected_fmt = IMWRITE_PAM_FORMAT_BLACKANDWHITE;
                else if (m_channels == 1 && m_maxval < 256)
                    selected_fmt = IMWRITE_PAM_FORMAT_GRAYSCALE;
                else if (m_channels == 3 && m_maxval < 256)
                    selected_fmt = IMWRITE_PAM_FORMAT_RGB;
                else
                    CV_Error(Error::StsError, "Can't determine selected_fmt (IMWRITE_PAM_FORMAT_NULL)");
            }
            CV_CheckDepth(m_sampledepth, m_sampledepth == CV_8U || m_sampledepth == CV_16U, "");
            CV_Check(m_channels, m_channels >= 1 && m_channels <= 4, "Unsupported number of channels");
            m_type = CV_MAKETYPE(m_sampledepth, m_channels);
            m_offset = m_strm.getPos();

            return true;
        }

        // failed
        m_offset = -1;
        m_width = m_height = -1;
        m_strm.close();
        return false;
    }
    catch (...)
    {
        m_offset = -1;
        m_width = m_height = -1;
        m_strm.close();
        throw;
    }
}


bool PAMDecoder::readData(Mat& img)
{
    uchar* data = img.ptr();
    const int target_channels = img.channels();
    size_t imp_stride = img.step;
    const int sample_depth = CV_ELEM_SIZE1(m_type);
    const int src_elems_per_row = m_width*m_channels;
    const int src_stride = src_elems_per_row*sample_depth;
    PaletteEntry palette[256] = {};
    const struct pam_format *fmt = NULL;
    struct channel_layout layout = { 0, 0, 0, 0 }; // normalized to 1-channel grey format

    /* setting buffer to max data size so scaling up is possible */
    AutoBuffer<uchar> _src(src_elems_per_row * 2);
    uchar* src = _src.data();

    if( m_offset < 0 || !m_strm.isOpened())
        return false;

    if (selected_fmt != IMWRITE_PAM_FORMAT_NULL)
        fmt = &formats[selected_fmt];
    else {
        /* default layout handling */
        if (m_channels >= 3) {
            layout.bchan = 0;
            layout.gchan = 1;
            layout.rchan = 2;
        }
    }

    {
        m_strm.setPos( m_offset );

        /* the case where data fits the opencv matrix */
        if (m_sampledepth == img.depth() && target_channels == m_channels && !bit_mode) {
            /* special case for 16bit images with wrong endianness */
            if (m_sampledepth == CV_16U && !isBigEndian())
            {
                for (int y = 0; y < m_height; y++, data += imp_stride)
                {
                    m_strm.getBytes( src, src_stride );
                    for (int x = 0; x < src_elems_per_row; x++)
                    {
                        uchar v = src[x * 2];
                        data[x * 2] = src[x * 2 + 1];
                        data[x * 2 + 1] = v;
                    }
                }
            }
            else {
                m_strm.getBytes( data, src_stride * m_height );
            }

        }
        else {
            /* black and white mode */
            if (bit_mode) {
                if( target_channels == 1 )
                {
                    uchar gray_palette[2] = {0, 255};
                    for (int y = 0; y < m_height; y++, data += imp_stride)
                    {
                        m_strm.getBytes( src, src_stride );
                        FillGrayRow1( data, src, m_width, gray_palette );
                    }
                } else if ( target_channels == 3 )
                {
                    FillGrayPalette( palette, 1 , false );
                    for (int y = 0; y < m_height; y++, data += imp_stride)
                    {
                        m_strm.getBytes( src, src_stride );
                        FillColorRow1( data, src, m_width, palette );
                    }
                }
                else
                {
                    CV_Error(Error::StsError, cv::format("Unsupported value of target_channels: %d", target_channels));
                }
            } else {
                for (int y = 0; y < m_height; y++, data += imp_stride)
                {
                    m_strm.getBytes( src, src_stride );

                    /* endianness correction */
                    if( m_sampledepth == CV_16U && !isBigEndian() )
                    {
                        for (int x = 0; x < src_elems_per_row; x++)
                        {
                            uchar v = src[x * 2];
                            src[x * 2] = src[x * 2 + 1];
                            src[x * 2 + 1] = v;
                        }
                    }

                    /* scale down */
                    if( img.depth() == CV_8U && m_sampledepth == CV_16U )
                    {
                        for (int x = 0; x < src_elems_per_row; x++)
                        {
                            int v = ((ushort *)src)[x];
                            src[x] = (uchar)(v >> 8);
                        }
                    }

                    /* if we are only scaling up/down then we can then copy the data */
                    if (target_channels == m_channels) {
                        memcpy (data, src, imp_stride);
                    }
                    /* perform correct conversion based on format */
                    else if (fmt) {
                        bool funcout = false;
                        if (fmt->cvt_func)
                            funcout = fmt->cvt_func (src, data, m_width, target_channels,
                                img.depth(), m_use_rgb);
                        /* fall back to default if there is no conversion function or it
                         * can't handle the specified characteristics
                         */
                        if (!funcout)
                            basic_conversion (src, &fmt->layout, m_channels,
                                m_width, data, target_channels, img.depth(), m_use_rgb);

                    /* default to selecting the first available channels */
                    } else {
                        basic_conversion (src, &layout, m_channels,
                            m_width, data, target_channels, img.depth(), m_use_rgb);
                    }
                }
            }
        }
    }
    return true;
}


//////////////////////////////////////////////////////////////////////////////////////////

PAMEncoder::PAMEncoder()
{
    m_description = "Portable arbitrary format (*.pam)";
    m_buf_supported = true;
}


PAMEncoder::~PAMEncoder()
{
}


ImageEncoder PAMEncoder::newEncoder() const
{
    return makePtr<PAMEncoder>();
}


bool PAMEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U || depth == CV_16U;
}


bool PAMEncoder::write( const Mat& img, const std::vector<int>& params )
{

    WLByteStream strm;

    int width = img.cols, height = img.rows;
    int stride = width*(int)img.elemSize();
    const uchar* data = img.ptr();
    const struct pam_format *fmt = NULL;
    int x, y, tmp, bufsize = 256;

    /* parse save file type */
    for( size_t i = 0; i < params.size(); i += 2 )
        if( params[i] == IMWRITE_PAM_TUPLETYPE ) {
            if ( params[i+1] > IMWRITE_PAM_FORMAT_NULL &&
                 params[i+1] < (int) PAM_FORMATS_NO)
                fmt = &formats[params[i+1]];
        }

    if( m_buf )
    {
        if( !strm.open(*m_buf) )
            return false;
        m_buf->reserve( alignSize(256 + stride*height, 256));
    }
    else if( !strm.open(m_filename) )
        return false;

    tmp = width * (int)img.elemSize();

    if (bufsize < tmp)
        bufsize = tmp;

    AutoBuffer<char> _buffer(bufsize);
    char* buffer = _buffer.data();

    /* write header */
    tmp = 0;
    tmp += snprintf( buffer, bufsize, "P7\n");
    tmp += snprintf( buffer + tmp, bufsize - tmp, "WIDTH %d\n", width);
    tmp += snprintf( buffer + tmp, bufsize - tmp, "HEIGHT %d\n", height);
    tmp += snprintf( buffer + tmp, bufsize - tmp, "DEPTH %d\n", img.channels());
    tmp += snprintf( buffer + tmp, bufsize - tmp, "MAXVAL %d\n", (1 << img.elemSize1()*8) - 1);
    if (fmt)
        tmp += snprintf( buffer + tmp, bufsize - tmp, "TUPLTYPE %s\n", fmt->name );
    snprintf( buffer + tmp, bufsize - tmp, "ENDHDR\n" );

    strm.putBytes( buffer, (int)strlen(buffer) );
    /* write data */
    if (img.depth() == CV_8U)
        strm.putBytes( data, stride*height );
    else if (img.depth() == CV_16U) {
        /* fix endianness */
        if (!isBigEndian()) {
            for( y = 0; y < height; y++ ) {
                memcpy( buffer, img.ptr(y), stride );
                for( x = 0; x < stride; x += 2 )
                {
                    uchar v = buffer[x];
                    buffer[x] = buffer[x + 1];
                    buffer[x + 1] = v;
                }
                strm.putBytes( buffer, stride );
            }
        } else
            strm.putBytes( data, stride*height );
    } else
        CV_Error(Error::StsInternal, "");

    strm.close();
    return true;
}

bool PAMEncoder::isValidParam(const int key, const int value) const
{
    bool ret = false;
    switch(key)
    {
        case IMWRITE_PAM_TUPLETYPE:
            {
                switch(value)
                {
                    case IMWRITE_PAM_FORMAT_NULL:
                    case IMWRITE_PAM_FORMAT_BLACKANDWHITE:
                    case IMWRITE_PAM_FORMAT_GRAYSCALE:
                    case IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA:
                    case IMWRITE_PAM_FORMAT_RGB:
                    case IMWRITE_PAM_FORMAT_RGB_ALPHA:
                        ret = true;
                        break;
                    default:
                        break;
                }
            }
            break;
        default:
            break;
    }
    return ret;
}

}

#endif
