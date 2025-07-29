// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "bitstrm.hpp"
#include "utils.hpp"

namespace cv
{

const int BS_DEF_BLOCK_SIZE = 1<<15;

/////////////////////////  RBaseStream ////////////////////////////

bool  RBaseStream::isOpened()
{
    return m_is_opened;
}

void  RBaseStream::allocate()
{
    if( !m_allocated )
    {
        m_start = new uchar[m_block_size];
        m_end = m_start + m_block_size;
        m_current = m_end;
        m_allocated = true;
    }
}


RBaseStream::RBaseStream()
{
    m_start = m_end = m_current = 0;
    m_file = 0;
    m_block_pos = 0;
    m_block_size = BS_DEF_BLOCK_SIZE;
    m_is_opened = false;
    m_allocated = false;
}


RBaseStream::~RBaseStream()
{
    close();    // Close files
    release();  // free  buffers
}


void  RBaseStream::readBlock()
{
    setPos( getPos() ); // normalize position

    if( m_file == 0 )
    {
        if( m_block_pos == 0 && m_current < m_end )
            return;
        throw RBS_THROW_EOS;
    }

    fseek( m_file, m_block_pos, SEEK_SET );
    size_t readed = fread( m_start, 1, m_block_size, m_file );
    m_end = m_start + readed;

    if( readed == 0 || m_current >= m_end )
        throw RBS_THROW_EOS;
}


bool  RBaseStream::open( const String& filename )
{
    close();
    allocate();

    m_file = fopen( filename.c_str(), "rb" );
    if( m_file )
    {
        m_is_opened = true;
        setPos(0);
        readBlock();
    }
    return m_file != 0;
}

bool  RBaseStream::open( const Mat& buf )
{
    close();
    if( buf.empty() )
        return false;
    CV_Assert(buf.isContinuous());
    m_start = buf.data;
    m_end = m_start + buf.cols*buf.rows*buf.elemSize();
    m_allocated = false;
    m_is_opened = true;
    setPos(0);

    return true;
}

void  RBaseStream::close()
{
    if( m_file )
    {
        fclose( m_file );
        m_file = 0;
    }
    m_is_opened = false;
    if( !m_allocated )
        m_start = m_end = m_current = 0;
}


void  RBaseStream::release()
{
    if( m_allocated )
        delete[] m_start;
    m_start = m_end = m_current = 0;
    m_allocated = false;
}


void  RBaseStream::setPos( int pos )
{
    CV_Assert(isOpened() && pos >= 0);

    if( !m_file )
    {
        m_current = m_start + pos;
        m_block_pos = 0;
        return;
    }

    int offset = pos % m_block_size;
    int old_block_pos = m_block_pos;
    m_block_pos = pos - offset;
    m_current = m_start + offset;
    if (old_block_pos != m_block_pos)
        readBlock();
}


int  RBaseStream::getPos()
{
    CV_Assert(isOpened());
    int pos = validateToInt((m_current - m_start) + m_block_pos);
    CV_Assert(pos >= m_block_pos); // overflow check
    CV_Assert(pos >= 0); // overflow check
    return pos;
}

void  RBaseStream::skip( int bytes )
{
    CV_Assert(bytes >= 0);
    uchar* old = m_current;
    m_current += bytes;
    CV_Assert(m_current >= old);  // overflow check
}

/////////////////////////  RLByteStream ////////////////////////////

RLByteStream::~RLByteStream()
{
}

int  RLByteStream::getByte()
{
    uchar *current = m_current;
    int   val;

    if( current >= m_end )
    {
        readBlock();
        current = m_current;
    }

    CV_Assert(current < m_end);

    val = *((uchar*)current);
    m_current = current + 1;
    return val;
}


int RLByteStream::getBytes( void* buffer, int count )
{
    uchar*  data = (uchar*)buffer;
    int readed = 0;
    CV_Assert(count >= 0);

    while( count > 0 )
    {
        int l;

        for(;;)
        {
            l = (int)(m_end - m_current);
            if( l > count ) l = count;
            if( l > 0 ) break;
            readBlock();
        }
        memcpy( data, m_current, l );
        m_current += l;
        data += l;
        count -= l;
        readed += l;
    }
    return readed;
}


////////////  RLByteStream & RMByteStream <Get[d]word>s ////////////////

RMByteStream::~RMByteStream()
{
}


int  RLByteStream::getWord()
{
    uchar *current = m_current;
    int   val;

    if( current+1 < m_end )
    {
        val = current[0] + (current[1] << 8);
        m_current = current + 2;
    }
    else
    {
        val = getByte();
        val|= getByte() << 8;
    }
    return val;
}


int  RLByteStream::getDWord()
{
    uchar *current = m_current;
    int   val;

    if( current+3 < m_end )
    {
        val = current[0] + (current[1] << 8) +
              (current[2] << 16) + (current[3] << 24);
        m_current = current + 4;
    }
    else
    {
        val = getByte();
        val |= getByte() << 8;
        val |= getByte() << 16;
        val |= getByte() << 24;
    }
    return val;
}


int  RMByteStream::getWord()
{
    uchar *current = m_current;
    int   val;

    if( current+1 < m_end )
    {
        val = (current[0] << 8) + current[1];
        m_current = current + 2;
    }
    else
    {
        val = getByte() << 8;
        val|= getByte();
    }
    return val;
}


int  RMByteStream::getDWord()
{
    uchar *current = m_current;
    int   val;

    if( current+3 < m_end )
    {
        val = (current[0] << 24) + (current[1] << 16) +
              (current[2] << 8) + current[3];
        m_current = current + 4;
    }
    else
    {
        val = getByte() << 24;
        val |= getByte() << 16;
        val |= getByte() << 8;
        val |= getByte();
    }
    return val;
}

/////////////////////////// WBaseStream /////////////////////////////////

// WBaseStream - base class for output streams
WBaseStream::WBaseStream()
{
    m_start = m_end = m_current = 0;
    m_file = 0;
    m_block_pos = 0;
    m_block_size = BS_DEF_BLOCK_SIZE;
    m_is_opened = false;
    m_buf = 0;
}


WBaseStream::~WBaseStream()
{
    close();
    release();
}


bool  WBaseStream::isOpened()
{
    return m_is_opened;
}


void  WBaseStream::allocate()
{
    if( !m_start )
        m_start = new uchar[m_block_size];

    m_end = m_start + m_block_size;
    m_current = m_start;
}


bool  WBaseStream::writeBlock()
{
    int size = (int)(m_current - m_start);

    CV_Assert(isOpened());
    if( size == 0 )
        return true;

    if( m_buf )
    {
        size_t sz = m_buf->size();
        m_buf->resize( sz + size );
        memcpy( &(*m_buf)[sz], m_start, size );
        m_current = m_start;
        m_block_pos += size;
        return true;
    }
    else
    {
        size_t written = fwrite( m_start, 1, size, m_file );
        m_current = m_start;
        m_block_pos += size;
        return written == (size_t)size;
    }
}


bool  WBaseStream::open( const String& filename )
{
    close();
    allocate();

    m_file = fopen( filename.c_str(), "wb" );
    if( m_file )
    {
        m_is_opened = true;
        m_block_pos = 0;
        m_current = m_start;
    }
    return m_file != 0;
}

bool  WBaseStream::open( std::vector<uchar>& buf )
{
    close();
    allocate();

    m_buf = &buf;
    m_is_opened = true;
    m_block_pos = 0;
    m_current = m_start;

    return true;
}

void  WBaseStream::close()
{
    if( m_is_opened )
        writeBlock();
    if( m_file )
    {
        fclose( m_file );
        m_file = 0;
    }
    m_buf = 0;
    m_is_opened = false;
}


void  WBaseStream::release()
{
    if( m_start )
        delete[] m_start;
    m_start = m_end = m_current = 0;
}


int  WBaseStream::getPos()
{
    CV_Assert(isOpened());
    return m_block_pos + (int)(m_current - m_start);
}


///////////////////////////// WLByteStream ///////////////////////////////////

WLByteStream::~WLByteStream()
{
}

bool  WLByteStream::putByte( int val )
{
    *m_current++ = (uchar)val;
    if( m_current >= m_end )
        return writeBlock();

    return true;
}


bool  WLByteStream::putBytes( const void* buffer, int count )
{
    uchar* data = (uchar*)buffer;

    CV_Assert(data && m_current && count >= 0);

    while( count )
    {
        int l = (int)(m_end - m_current);

        if( l > count )
            l = count;

        if( l > 0 )
        {
            memcpy( m_current, data, l );
            m_current += l;
            data += l;
            count -= l;
        }
        if( m_current == m_end )
        {
            bool written = writeBlock();
            if (!written)
                return false;
        }
    }

    return true;
}


bool  WLByteStream::putWord( int val )
{
    uchar *current = m_current;

    if( current+1 < m_end )
    {
        current[0] = (uchar)val;
        current[1] = (uchar)(val >> 8);
        m_current = current + 2;
        if( m_current == m_end )
            return writeBlock();
    }
    else
    {
        putByte(val);
        putByte(val >> 8);
    }

    return true;
}


bool  WLByteStream::putDWord( int val )
{
    uchar *current = m_current;

    if( current+3 < m_end )
    {
        current[0] = (uchar)val;
        current[1] = (uchar)(val >> 8);
        current[2] = (uchar)(val >> 16);
        current[3] = (uchar)(val >> 24);
        m_current = current + 4;
        if( m_current == m_end )
            return writeBlock();
    }
    else
    {
        putByte(val);
        putByte(val >> 8);
        putByte(val >> 16);
        putByte(val >> 24);
    }

    return true;
}


///////////////////////////// WMByteStream ///////////////////////////////////

WMByteStream::~WMByteStream()
{
}


bool  WMByteStream::putWord( int val )
{
    uchar *current = m_current;

    if( current+1 < m_end )
    {
        current[0] = (uchar)(val >> 8);
        current[1] = (uchar)val;
        m_current = current + 2;
        if( m_current == m_end )
            return writeBlock();
    }
    else
    {
        putByte(val >> 8);
        putByte(val);
    }

    return true;
}


bool  WMByteStream::putDWord( int val )
{
    uchar *current = m_current;

    if( current+3 < m_end )
    {
        current[0] = (uchar)(val >> 24);
        current[1] = (uchar)(val >> 16);
        current[2] = (uchar)(val >> 8);
        current[3] = (uchar)val;
        m_current = current + 4;
        if( m_current == m_end )
            return writeBlock();
    }
    else
    {
        putByte(val >> 24);
        putByte(val >> 16);
        putByte(val >> 8);
        putByte(val);
    }

    return true;
}

}
