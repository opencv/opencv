// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "grfmt_sunras.hpp"

#ifdef HAVE_IMGCODEC_SUNRASTER

namespace cv
{

static const char* fmtSignSunRas = "\x59\xA6\x6A\x95";

/************************ Sun Raster reader *****************************/

SunRasterDecoder::SunRasterDecoder()
{
    m_offset = -1;
    m_signature = fmtSignSunRas;
    m_bpp = 0;
    m_encoding = RAS_STANDARD;
    m_maptype = RMT_NONE;
    m_maplength = 0;
    m_buf_supported = true;
}


SunRasterDecoder::~SunRasterDecoder()
{
}

ImageDecoder SunRasterDecoder::newDecoder() const
{
    return makePtr<SunRasterDecoder>();
}

void  SunRasterDecoder::close()
{
    m_strm.close();
}


bool  SunRasterDecoder::readHeader()
{
    bool result = false;

    if (!m_buf.empty())
        m_strm.open(m_buf);
    else
        m_strm.open(m_filename);

    if( !m_strm.isOpened()) return false;

    try
    {
        m_strm.skip( 4 );
        m_width  = m_strm.getDWord();
        m_height = m_strm.getDWord();
        m_bpp    = m_strm.getDWord();
        int palSize = (m_bpp > 0 && m_bpp <= 8) ? (3*(1 << m_bpp)) : 0;

        m_strm.skip( 4 );
        m_encoding = (SunRasType)m_strm.getDWord();
        m_maptype = (SunRasMapType)m_strm.getDWord();
        m_maplength = m_strm.getDWord();

        if( m_width > 0 && m_height > 0 &&
            (m_bpp == 1 || m_bpp == 8 || m_bpp == 24 || m_bpp == 32) &&
            (m_encoding == RAS_OLD || m_encoding == RAS_STANDARD ||
             (m_type == RAS_BYTE_ENCODED && m_bpp == 8) || m_type == RAS_FORMAT_RGB) &&
            ((m_maptype == RMT_NONE && m_maplength == 0) ||
             (m_maptype == RMT_EQUAL_RGB && m_maplength <= palSize && m_maplength > 0 && m_bpp <= 8)))
        {
            memset( m_palette, 0, sizeof(m_palette));

            if( m_maplength != 0 )
            {
                uchar buffer[256*3];

                if( m_strm.getBytes( buffer, m_maplength ) == m_maplength )
                {
                    int i;
                    palSize = m_maplength/3;

                    for( i = 0; i < palSize; i++ )
                    {
                        m_palette[i].b = buffer[i + 2*palSize];
                        m_palette[i].g = buffer[i + palSize];
                        m_palette[i].r = buffer[i];
                        m_palette[i].a = 0;
                    }

                    m_type = IsColorPalette( m_palette, m_bpp ) ? CV_8UC3 : CV_8UC1;
                    m_offset = m_strm.getPos();

                    CV_Assert(m_offset == 32 + m_maplength);
                    result = true;
                }
            }
            else
            {
                m_type = m_bpp > 8 ? CV_8UC3 : CV_8UC1;

                if( CV_MAT_CN(m_type) == 1 )
                    FillGrayPalette( m_palette, m_bpp );

                m_offset = m_strm.getPos();

                CV_Assert(m_offset == 32 + m_maplength);
                result = true;
            }
        }
    }
    catch(...)
    {
    }

    if( !result )
    {
        m_offset = -1;
        m_width = m_height = -1;
        m_strm.close();
    }
    return result;
}


bool  SunRasterDecoder::readData( Mat& img )
{
    bool color = img.channels() > 1;
    uchar* data = img.ptr();
    size_t step = img.step;
    uchar  gray_palette[256] = {0};
    bool   result = false;
    int  src_pitch = ((m_width*m_bpp + 7)/8 + 1) & -2;
    int  nch = color ? 3 : 1;
    int  width3 = m_width*nch;
    int  y;

    if( m_offset < 0 || !m_strm.isOpened())
        return false;

    AutoBuffer<uchar> _src(src_pitch + 32);
    uchar* src = _src.data();

    if( !color && m_maptype == RMT_EQUAL_RGB )
        CvtPaletteToGray( m_palette, gray_palette, 1 << m_bpp );

    try
    {
        m_strm.setPos( m_offset );

        switch( m_bpp )
        {
        /************************* 1 BPP ************************/
        case 1:
            if( m_type != RAS_BYTE_ENCODED )
            {
                for( y = 0; y < m_height; y++, data += step )
                {
                    m_strm.getBytes( src, src_pitch );
                    if( color )
                        FillColorRow1( data, src, m_width, m_palette );
                    else
                        FillGrayRow1( data, src, m_width, gray_palette );
                }
                result = true;
            }
            else
            {
                uchar* line_end = src + (m_width*m_bpp + 7)/8;
                uchar* tsrc = src;
                y = 0;

                for(;;)
                {
                    int max_count = (int)(line_end - tsrc);
                    int code = 0, len = 0, len1 = 0;

                    do
                    {
                        code = m_strm.getByte();
                        if( code == 0x80 )
                        {
                            len = m_strm.getByte();
                            if( len != 0 ) break;
                        }
                        tsrc[len1] = (uchar)code;
                    }
                    while( ++len1 < max_count );

                    tsrc += len1;

                    if( len > 0 ) // encoded mode
                    {
                        ++len;
                        code = m_strm.getByte();
                        if( len > line_end - tsrc )
                        {
                            CV_Error(Error::StsInternal, "");
                            goto bad_decoding_1bpp;
                        }

                        memset( tsrc, code, len );
                        tsrc += len;
                    }

                    if( tsrc >= line_end )
                    {
                        tsrc = src;
                        if( color )
                            FillColorRow1( data, src, m_width, m_palette );
                        else
                            FillGrayRow1( data, src, m_width, gray_palette );
                        data += step;
                        if( ++y >= m_height ) break;
                    }
                }
                result = true;
bad_decoding_1bpp:
                ;
            }
            break;
        /************************* 8 BPP ************************/
        case 8:
            if( m_type != RAS_BYTE_ENCODED )
            {
                for( y = 0; y < m_height; y++, data += step )
                {
                    m_strm.getBytes( src, src_pitch );
                    if( color )
                        FillColorRow8( data, src, m_width, m_palette );
                    else
                        FillGrayRow8( data, src, m_width, gray_palette );
                }
                result = true;
            }
            else // RLE-encoded
            {
                uchar* line_end = data + width3;
                y = 0;

                for(;;)
                {
                    int max_count = (int)(line_end - data);
                    int code = 0, len = 0, len1;
                    uchar* tsrc = src;

                    do
                    {
                        code = m_strm.getByte();
                        if( code == 0x80 )
                        {
                            len = m_strm.getByte();
                            if( len != 0 ) break;
                        }
                        *tsrc++ = (uchar)code;
                    }
                    while( (max_count -= nch) > 0 );

                    len1 = (int)(tsrc - src);

                    if( len1 > 0 )
                    {
                        if( color )
                            FillColorRow8( data, src, len1, m_palette );
                        else
                            FillGrayRow8( data, src, len1, gray_palette );
                        data += len1*nch;
                    }

                    if( len > 0 ) // encoded mode
                    {
                        len = (len + 1)*nch;
                        code = m_strm.getByte();

                        if( color )
                            data = FillUniColor( data, line_end, validateToInt(step), width3,
                                                 y, m_height, len,
                                                 m_palette[code] );
                        else
                            data = FillUniGray( data, line_end, validateToInt(step), width3,
                                                y, m_height, len,
                                                gray_palette[code] );
                        if( y >= m_height )
                            break;
                    }

                    if( data == line_end )
                    {
                        if( m_strm.getByte() != 0 )
                            goto bad_decoding_end;
                        line_end += step;
                        data = line_end - width3;
                        if( ++y >= m_height ) break;
                    }
                }

                result = true;
bad_decoding_end:
                ;
            }
            break;
        /************************* 24 BPP ************************/
        case 24:
            for( y = 0; y < m_height; y++, data += step )
            {
                m_strm.getBytes(src, src_pitch );

                if( color )
                {
                    if( m_type == RAS_FORMAT_RGB || m_use_rgb)
                        icvCvt_RGB2BGR_8u_C3R(src, 0, data, 0, Size(m_width,1) );
                    else
                        memcpy(data, src, std::min(step, (size_t)src_pitch));
                }
                else
                {
                    icvCvt_BGR2Gray_8u_C3C1R(src, 0, data, 0, Size(m_width,1),
                                              m_type == RAS_FORMAT_RGB ? 2 : 0 );
                }
            }
            result = true;
            break;
        /************************* 32 BPP ************************/
        case 32:
            for( y = 0; y < m_height; y++, data += step )
            {
                /* hack: a0 b0 g0 r0 a1 b1 g1 r1 ... are written to src + 3,
                   so when we look at src + 4, we see b0 g0 r0 x b1 g1 g1 x ... */
                m_strm.getBytes( src + 3, src_pitch );

                if( color )
                    icvCvt_BGRA2BGR_8u_C4C3R( src + 4, 0, data, 0, Size(m_width,1),
                                              (m_type == RAS_FORMAT_RGB || m_use_rgb) ? 2 : 0 );
                else
                    icvCvt_BGRA2Gray_8u_C4C1R( src + 4, 0, data, 0, Size(m_width,1),
                                               m_type == RAS_FORMAT_RGB ? 2 : 0 );
            }
            result = true;
            break;
        default:
            CV_Error(Error::StsInternal, "");
        }
    }
    catch( ... )
    {
    }

    return result;
}


//////////////////////////////////////////////////////////////////////////////////////////

SunRasterEncoder::SunRasterEncoder()
{
    m_description = "Sun raster files (*.sr;*.ras)";
    m_buf_supported = true;
}


ImageEncoder SunRasterEncoder::newEncoder() const
{
    return makePtr<SunRasterEncoder>();
}

SunRasterEncoder::~SunRasterEncoder()
{
}

bool  SunRasterEncoder::write( const Mat& img, const std::vector<int>& )
{
    bool result = false;
    int y, width = img.cols, height = img.rows, channels = img.channels();
    int fileStep = (width*channels + 1) & -2;
    WMByteStream  strm;

    if (m_buf) {
        if (!strm.open(*m_buf)) {
            return false;
        }
        else {
            m_buf->reserve(height * fileStep + 32);
        }
    }
    else
        strm.open(m_filename);

    if( strm.isOpened() )
    {
        CHECK_WRITE(strm.putBytes( fmtSignSunRas, (int)strlen(fmtSignSunRas) ));
        CHECK_WRITE(strm.putDWord( width ));
        CHECK_WRITE(strm.putDWord( height ));
        CHECK_WRITE(strm.putDWord( channels*8 ));
        CHECK_WRITE(strm.putDWord( fileStep*height ));
        CHECK_WRITE(strm.putDWord( RAS_STANDARD ));
        CHECK_WRITE(strm.putDWord( RMT_NONE ));
        CHECK_WRITE(strm.putDWord( 0 ));

        for( y = 0; y < height; y++ )
            CHECK_WRITE(strm.putBytes( img.ptr(y), fileStep ));

        strm.close();
        result = true;
    }
    return result;
}

}

#endif // HAVE_IMGCODEC_SUNRASTER
