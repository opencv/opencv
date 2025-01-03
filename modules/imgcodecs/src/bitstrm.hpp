// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#ifndef _BITSTRM_H_
#define _BITSTRM_H_

#include <stdio.h>

namespace cv
{

#define DECLARE_RBS_EXCEPTION(name) \
class RBS_ ## name ## _Exception : public cv::Exception \
{ \
public: \
    RBS_ ## name ## _Exception(int code_, const String& err_, const String& func_, const String& file_, int line_) : \
        cv::Exception(code_, err_, func_, file_, line_) \
    {} \
};
DECLARE_RBS_EXCEPTION(THROW_EOS)
#define RBS_THROW_EOS RBS_THROW_EOS_Exception(cv::Error::StsError, "Unexpected end of input stream", CV_Func, __FILE__, __LINE__)
DECLARE_RBS_EXCEPTION(THROW_FORB)
#define RBS_THROW_FORB RBS_THROW_FORB_Exception(cv::Error::StsError, "Forrbidden huffman code", CV_Func, __FILE__, __LINE__)
DECLARE_RBS_EXCEPTION(BAD_HEADER)
#define RBS_BAD_HEADER RBS_BAD_HEADER_Exception(cv::Error::StsError, "Invalid header", CV_Func, __FILE__, __LINE__)

#define CHECK_WRITE(action) \
if (!action) \
{ \
    return false; \
}

typedef unsigned long ulong;

// class RBaseStream - base class for other reading streams.
class RBaseStream
{
public:
    //methods
    RBaseStream();
    virtual ~RBaseStream();

    virtual bool  open( const String& filename );
    virtual bool  open( const Mat& buf );
    virtual void  close();
    bool          isOpened();
    void          setPos( int pos );
    int           getPos();
    void          skip( int bytes );

protected:

    bool    m_allocated;
    uchar*  m_start;
    uchar*  m_end;
    uchar*  m_current;
    FILE*   m_file;
    int     m_block_size;
    int     m_block_pos;
    bool    m_is_opened;

    virtual void  readBlock();
    virtual void  release();
    virtual void  allocate();
};


// class RLByteStream - uchar-oriented stream.
// l in prefix means that the least significant uchar of a multi-uchar value goes first
class RLByteStream : public RBaseStream
{
public:
    virtual ~RLByteStream();

    int     getByte();
    int     getBytes( void* buffer, int count );
    int     getWord();
    int     getDWord();
};

// class RMBitStream - uchar-oriented stream.
// m in prefix means that the most significant uchar of a multi-uchar value go first
class RMByteStream : public RLByteStream
{
public:
    virtual ~RMByteStream();

    int     getWord();
    int     getDWord();
};

// WBaseStream - base class for output streams
class WBaseStream
{
public:
    //methods
    WBaseStream();
    virtual ~WBaseStream();

    virtual bool  open( const String& filename );
    virtual bool  open( std::vector<uchar>& buf );
    virtual void  close();
    bool          isOpened();
    int           getPos();

protected:

    uchar*  m_start;
    uchar*  m_end;
    uchar*  m_current;
    int     m_block_size;
    int     m_block_pos;
    FILE*   m_file;
    bool    m_is_opened;
    std::vector<uchar>* m_buf;

    virtual bool  writeBlock();
    virtual void  release();
    virtual void  allocate();
};


// class WLByteStream - uchar-oriented stream.
// l in prefix means that the least significant uchar of a multi-byte value goes first
class WLByteStream : public WBaseStream
{
public:
    virtual ~WLByteStream();

    bool putByte( int val );
    bool putBytes( const void* buffer, int count );
    bool putWord( int val );
    bool putDWord( int val );
};


// class WLByteStream - uchar-oriented stream.
// m in prefix means that the least significant uchar of a multi-byte value goes last
class WMByteStream : public WLByteStream
{
public:
    virtual ~WMByteStream();
    bool putWord( int val );
    bool putDWord( int val );
};

}

#endif/*_BITSTRM_H_*/
