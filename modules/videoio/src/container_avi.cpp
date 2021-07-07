// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/videoio/container_avi.private.hpp"
#include <fstream>
#include <limits>
#include <typeinfo>

namespace cv
{

// Utility function for safe integer conversions
template <typename D, typename S>
inline D safe_int_cast(S val, const char * msg = 0)
{
    typedef std::numeric_limits<S> st;
    typedef std::numeric_limits<D> dt;
    CV_StaticAssert(st::is_integer && dt::is_integer, "Integer type is expected");
    const bool in_range_r = (double)val <= (double)dt::max();
    const bool in_range_l = (double)val >= (double)dt::min();
    if (!in_range_r || !in_range_l)
    {
        if (!msg)
            CV_Error_(Error::StsOutOfRange, ("Can not convert integer values (%s -> %s), value 0x%llx is out of range", typeid(S).name(), typeid(D).name(), val));
        else
            CV_Error(Error::StsOutOfRange, msg);
    }
    return static_cast<D>(val);
}

const uint32_t RIFF_CC = CV_FOURCC('R','I','F','F');
const uint32_t LIST_CC = CV_FOURCC('L','I','S','T');
const uint32_t HDRL_CC = CV_FOURCC('h','d','r','l');
const uint32_t AVIH_CC = CV_FOURCC('a','v','i','h');
const uint32_t STRL_CC = CV_FOURCC('s','t','r','l');
const uint32_t STRH_CC = CV_FOURCC('s','t','r','h');
const uint32_t STRF_CC = CV_FOURCC('s','t','r','f');
const uint32_t VIDS_CC = CV_FOURCC('v','i','d','s');
const uint32_t MJPG_CC = CV_FOURCC('M','J','P','G');
const uint32_t MOVI_CC = CV_FOURCC('m','o','v','i');
const uint32_t IDX1_CC = CV_FOURCC('i','d','x','1');
const uint32_t AVI_CC  = CV_FOURCC('A','V','I',' ');
const uint32_t AVIX_CC = CV_FOURCC('A','V','I','X');
const uint32_t JUNK_CC = CV_FOURCC('J','U','N','K');
const uint32_t INFO_CC = CV_FOURCC('I','N','F','O');
const uint32_t ODML_CC = CV_FOURCC('o','d','m','l');
const uint32_t DMLH_CC = CV_FOURCC('d','m','l','h');

String fourccToString(uint32_t fourcc);


#pragma pack(push, 1)
struct AviMainHeader
{
    uint32_t dwMicroSecPerFrame;    //  The period between video frames
    uint32_t dwMaxBytesPerSec;      //  Maximum data rate of the file
    uint32_t dwReserved1;           // 0
    uint32_t dwFlags;               //  0x10 AVIF_HASINDEX: The AVI file has an idx1 chunk containing an index at the end of the file.
    uint32_t dwTotalFrames;         // Field of the main header specifies the total number of frames of data in file.
    uint32_t dwInitialFrames;       // Is used for interleaved files
    uint32_t dwStreams;             // Specifies the number of streams in the file.
    uint32_t dwSuggestedBufferSize; // Field specifies the suggested buffer size forreading the file
    uint32_t dwWidth;               // Fields specify the width of the AVIfile in pixels.
    uint32_t dwHeight;              // Fields specify the height of the AVIfile in pixels.
    uint32_t dwReserved[4];         // 0, 0, 0, 0
};

struct AviStreamHeader
{
    uint32_t fccType;              // 'vids', 'auds', 'txts'...
    uint32_t fccHandler;           // "cvid", "DIB "
    uint32_t dwFlags;               // 0
    uint32_t dwPriority;            // 0
    uint32_t dwInitialFrames;       // 0
    uint32_t dwScale;               // 1
    uint32_t dwRate;                // Fps (dwRate - frame rate for video streams)
    uint32_t dwStart;               // 0
    uint32_t dwLength;              // Frames number (playing time of AVI file as defined by scale and rate)
    uint32_t dwSuggestedBufferSize; // For reading the stream
    uint32_t dwQuality;             // -1 (encoding quality. If set to -1, drivers use the default quality value)
    uint32_t dwSampleSize;          // 0 means that each frame is in its own chunk
    struct {
        short int left;
        short int top;
        short int right;
        short int bottom;
    } rcFrame;                // If stream has a different size than dwWidth*dwHeight(unused)
};

struct AviIndex
{
    uint32_t ckid;
    uint32_t dwFlags;
    uint32_t dwChunkOffset;
    uint32_t dwChunkLength;
};

struct BitmapInfoHeader
{
    uint32_t biSize;                // Write header size of BITMAPINFO header structure
    int32_t  biWidth;               // width in pixels
    int32_t  biHeight;              // height in pixels
    uint16_t  biPlanes;              // Number of color planes in which the data is stored
    uint16_t  biBitCount;            // Number of bits per pixel
    uint32_t biCompression;         // Type of compression used (uncompressed: NO_COMPRESSION=0)
    uint32_t biSizeImage;           // Image Buffer. Quicktime needs 3 bytes also for 8-bit png
                                 //   (biCompression==NO_COMPRESSION)?0:xDim*yDim*bytesPerPixel;
    int32_t  biXPelsPerMeter;       // Horizontal resolution in pixels per meter
    int32_t  biYPelsPerMeter;       // Vertical resolution in pixels per meter
    uint32_t biClrUsed;             // 256 (color table size; for 8-bit only)
    uint32_t biClrImportant;        // Specifies that the first x colors of the color table. Are important to the DIB.
};

struct RiffChunk
{
    uint32_t m_four_cc;
    uint32_t m_size;
};

struct RiffList
{
    uint32_t m_riff_or_list_cc;
    uint32_t m_size;
    uint32_t m_list_type_cc;
};
#pragma pack(pop)

class VideoInputStream
{
public:
    VideoInputStream();
    VideoInputStream(const String& filename);
    ~VideoInputStream();
    VideoInputStream& read(char*, uint32_t);
    VideoInputStream& seekg(uint64_t);
    uint64_t tellg();
    bool isOpened() const;
    bool open(const String& filename);
    void close();
    operator bool();

private:
    VideoInputStream(const VideoInputStream&);
    VideoInputStream& operator=(const VideoInputStream&);

private:
    std::ifstream input;
    bool    m_is_valid;
    String  m_fname;
};


inline VideoInputStream& operator >> (VideoInputStream& is, AviMainHeader& avih)
{
    is.read((char*)(&avih), sizeof(AviMainHeader));
    return is;
}
inline VideoInputStream& operator >> (VideoInputStream& is, AviStreamHeader& strh)
{
    is.read((char*)(&strh), sizeof(AviStreamHeader));
    return is;
}
inline VideoInputStream& operator >> (VideoInputStream& is, BitmapInfoHeader& bmph)
{
    is.read((char*)(&bmph), sizeof(BitmapInfoHeader));
    return is;
}
inline VideoInputStream& operator >> (VideoInputStream& is, AviIndex& idx1)
{
    is.read((char*)(&idx1), sizeof(idx1));
    return is;
}

inline VideoInputStream& operator >> (VideoInputStream& is, RiffChunk& riff_chunk)
{
    is.read((char*)(&riff_chunk), sizeof(riff_chunk));
    return is;
}

inline VideoInputStream& operator >> (VideoInputStream& is, RiffList& riff_list)
{
    is.read((char*)(&riff_list), sizeof(riff_list));
    return is;
}

static const int AVIH_STRH_SIZE = 56;
static const int STRF_SIZE = 40;
static const int AVI_DWFLAG = 0x00000910;
static const int AVI_DWSCALE = 1;
static const int AVI_DWQUALITY = -1;
static const int JUNK_SEEK = 4096;
static const int AVIIF_KEYFRAME = 0x10;
static const int MAX_BYTES_PER_SEC = 99999999;
static const int SUG_BUFFER_SIZE = 1048576;

String fourccToString(uint32_t fourcc)
{
    return format("%c%c%c%c", fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255);
}

VideoInputStream::VideoInputStream(): m_is_valid(false)
{
    m_fname = String();
}

VideoInputStream::VideoInputStream(const String& filename): m_is_valid(false)
{
    m_fname = filename;
    open(filename);
}

bool VideoInputStream::isOpened() const
{
    return input.is_open();
}

bool VideoInputStream::open(const String& filename)
{
    close();
    input.open(filename.c_str(), std::ios_base::binary);
    m_is_valid = isOpened();
    return m_is_valid;
}

void VideoInputStream::close()
{
    if(isOpened())
    {
        m_is_valid = false;
        input.close();
    }
}

VideoInputStream& VideoInputStream::read(char* buf, uint32_t count)
{
    if(isOpened())
    {
        input.read(buf, safe_int_cast<std::streamsize>(count, "Failed to read AVI file: requested chunk size is too large"));
        m_is_valid = (input.gcount() == (std::streamsize)count);
    }

    return *this;
}

VideoInputStream& VideoInputStream::seekg(uint64_t pos)
{
    input.clear();
    input.seekg(safe_int_cast<std::streamoff>(pos, "Failed to seek in AVI file: position is out of range"));
    m_is_valid = !input.eof();
    return *this;
}

uint64_t VideoInputStream::tellg()
{
    return input.tellg();
}

VideoInputStream::operator bool()
{
    return m_is_valid;
}

VideoInputStream::~VideoInputStream()
{
    close();
}

AVIReadContainer::AVIReadContainer(): m_stream_id(0), m_movi_start(0), m_movi_end(0), m_width(0), m_height(0), m_fps(0), m_is_indx_present(false)
{
    m_file_stream = makePtr<VideoInputStream>();
}

void AVIReadContainer::initStream(const String &filename)
{
    m_file_stream = makePtr<VideoInputStream>(filename);
}

void AVIReadContainer::initStream(Ptr<VideoInputStream> m_file_stream_)
{
    m_file_stream = m_file_stream_;
}

void AVIReadContainer::close()
{
    m_file_stream->close();
}

bool AVIReadContainer::parseIndex(uint32_t index_size, frame_list& in_frame_list)
{
    uint64_t index_end = m_file_stream->tellg();
    index_end += index_size;
    bool result = false;

    while(m_file_stream && (m_file_stream->tellg() < index_end))
    {
        AviIndex idx1;
        *m_file_stream >> idx1;

        if(idx1.ckid == m_stream_id)
        {
            uint64_t absolute_pos = m_movi_start + idx1.dwChunkOffset;

            if(absolute_pos < m_movi_end)
            {
                in_frame_list.push_back(std::make_pair(absolute_pos, idx1.dwChunkLength));
            }
            else
            {
                //unsupported case
                fprintf(stderr, "Frame offset points outside movi section.\n");
            }
        }

        result = true;
    }

    return result;
}

bool AVIReadContainer::parseStrl(char stream_id, Codecs codec_)
{
    RiffChunk strh;
    *m_file_stream >> strh;

    if(m_file_stream && strh.m_four_cc == STRH_CC)
    {
        AviStreamHeader strm_hdr;
        *m_file_stream >> strm_hdr;

        if (codec_ == MJPEG)
        {
            if(strm_hdr.fccType == VIDS_CC && strm_hdr.fccHandler == MJPG_CC)
            {
                uint8_t first_digit = (stream_id/10) + '0';
                uint8_t second_digit = (stream_id%10) + '0';

                if(m_stream_id == 0)
                {
                    m_stream_id = CV_FOURCC(first_digit, second_digit, 'd', 'c');
                    m_fps = double(strm_hdr.dwRate)/strm_hdr.dwScale;
                }
                else
                {
                    //second mjpeg video stream found which is not supported
                    fprintf(stderr, "More than one video stream found within AVI/AVIX list. Stream %c%cdc would be ignored\n", first_digit, second_digit);
                }

                return true;
            }
        }
    }

    return false;
}

void AVIReadContainer::skipJunk(RiffChunk& chunk)
{
    if(chunk.m_four_cc == JUNK_CC)
    {
        m_file_stream->seekg(m_file_stream->tellg() + chunk.m_size);
        *m_file_stream >> chunk;
    }
}

void AVIReadContainer::skipJunk(RiffList& list)
{
    if(list.m_riff_or_list_cc == JUNK_CC)
    {
        //JUNK chunk is 4 bytes less than LIST
        m_file_stream->seekg(m_file_stream->tellg() + list.m_size - 4);
        *m_file_stream >> list;
    }
}

bool AVIReadContainer::parseHdrlList(Codecs codec_)
{
    bool result = false;

    RiffChunk avih;
    *m_file_stream >> avih;

    if(m_file_stream && avih.m_four_cc == AVIH_CC)
    {
        uint64_t next_strl_list = m_file_stream->tellg();
        next_strl_list += avih.m_size;

        AviMainHeader avi_hdr;
        *m_file_stream >> avi_hdr;

        if(m_file_stream)
        {
            m_is_indx_present = ((avi_hdr.dwFlags & 0x10) != 0);
            uint32_t number_of_streams = avi_hdr.dwStreams;
            CV_Assert(number_of_streams < 0xFF);
            m_width = avi_hdr.dwWidth;
            m_height = avi_hdr.dwHeight;

            //the number of strl lists must be equal to number of streams specified in main avi header
            for(uint32_t i = 0; i < number_of_streams; ++i)
            {
                m_file_stream->seekg(next_strl_list);
                RiffList strl_list;
                *m_file_stream >> strl_list;

                if( m_file_stream && strl_list.m_riff_or_list_cc == LIST_CC && strl_list.m_list_type_cc == STRL_CC )
                {
                    next_strl_list = m_file_stream->tellg();
                    //RiffList::m_size includes fourCC field which we have already read
                    next_strl_list += (strl_list.m_size - 4);

                    result = parseStrl((char)i, codec_);
                }
                else
                {
                    printError(strl_list, STRL_CC);
                }
            }
        }
    }
    else
    {
        printError(avih, AVIH_CC);
    }

    return result;
}

bool AVIReadContainer::parseAviWithFrameList(frame_list& in_frame_list, Codecs codec_)
{
    RiffList hdrl_list;
    *m_file_stream >> hdrl_list;

    if( m_file_stream && hdrl_list.m_riff_or_list_cc == LIST_CC && hdrl_list.m_list_type_cc == HDRL_CC )
    {
        uint64_t next_list = m_file_stream->tellg();
        //RiffList::m_size includes fourCC field which we have already read
        next_list += (hdrl_list.m_size - 4);
        //parseHdrlList sets m_is_indx_present flag which would be used later
        if(parseHdrlList(codec_))
        {
            m_file_stream->seekg(next_list);

            RiffList some_list;
            *m_file_stream >> some_list;

            //an optional section INFO
            if(m_file_stream && some_list.m_riff_or_list_cc == LIST_CC && some_list.m_list_type_cc == INFO_CC)
            {
                next_list = m_file_stream->tellg();
                //RiffList::m_size includes fourCC field which we have already read
                next_list += (some_list.m_size - 4);
                parseInfo();

                m_file_stream->seekg(next_list);
                *m_file_stream >> some_list;
            }

            //an optional section JUNK
            skipJunk(some_list);

            //we are expecting to find here movi list. Must present in avi
            if(m_file_stream && some_list.m_riff_or_list_cc == LIST_CC && some_list.m_list_type_cc == MOVI_CC)
            {
                bool is_index_found = false;

                m_movi_start = m_file_stream->tellg();
                m_movi_start -= 4;

                m_movi_end = m_movi_start + some_list.m_size;
                //if m_is_indx_present is set to true we should find index
                if(m_is_indx_present)
                {
                    //we are expecting to find index section after movi list
                    uint32_t indx_pos = (uint32_t)m_movi_start + 4;
                    indx_pos += (some_list.m_size - 4);
                    m_file_stream->seekg(indx_pos);

                    RiffChunk index_chunk;
                    *m_file_stream >> index_chunk;

                    if(m_file_stream && index_chunk.m_four_cc == IDX1_CC)
                    {
                        is_index_found = parseIndex(index_chunk.m_size, in_frame_list);
                        //we are not going anywhere else
                    }
                    else
                    {
                        printError(index_chunk, IDX1_CC);
                    }
                }
                //index not present or we were not able to find it
                //parsing movi list
                if(!is_index_found)
                {
                    //not implemented
                    parseMovi(in_frame_list);

                    fprintf(stderr, "Failed to parse avi: index was not found\n");
                    //we are not going anywhere else
                }
            }
            else
            {
                printError(some_list, MOVI_CC);
            }
        }
    }
    else
    {
        printError(hdrl_list, HDRL_CC);
    }

    return in_frame_list.size() > 0;
}

std::vector<char> AVIReadContainer::readFrame(frame_iterator it)
{
    m_file_stream->seekg(it->first);

    RiffChunk chunk;
    *(m_file_stream) >> chunk;

    // Assertion added to prevent complaints from static analysis tools
    // as the chunk size is read from a file then used to allocate
    // memory. 64MB was chosen arbitrarily as an upper bound but it may
    // be useful to make it configurable.
    CV_Assert(chunk.m_size <= 67108864);

    std::vector<char> result;

    result.reserve(chunk.m_size);
    result.resize(chunk.m_size);

    m_file_stream->read(&(result[0]), chunk.m_size); // result.data() failed with MSVS2008

    return result;
}

bool AVIReadContainer::parseRiff(frame_list &m_mjpeg_frames_)
{
    bool result = false;
    while(*m_file_stream)
    {
        RiffList riff_list;

        *m_file_stream >> riff_list;

        if( *m_file_stream && riff_list.m_riff_or_list_cc == RIFF_CC &&
            ((riff_list.m_list_type_cc == AVI_CC) | (riff_list.m_list_type_cc == AVIX_CC)) )
        {
            uint64_t next_riff = m_file_stream->tellg();
            //RiffList::m_size includes fourCC field which we have already read
            next_riff += (riff_list.m_size - 4);

            bool is_parsed = parseAvi(m_mjpeg_frames_, MJPEG);
            result = result || is_parsed;
            m_file_stream->seekg(next_riff);
        }
        else
        {
            break;
        }
    }
    return result;
}

void AVIReadContainer::printError(RiffList &list, uint32_t expected_fourcc)
{
    if(!m_file_stream)
    {
        fprintf(stderr, "Unexpected end of file while searching for %s list\n", fourccToString(expected_fourcc).c_str());
    }
    else if(list.m_riff_or_list_cc != LIST_CC)
    {
        fprintf(stderr, "Unexpected element. Expected: %s. Got: %s.\n", fourccToString(LIST_CC).c_str(), fourccToString(list.m_riff_or_list_cc).c_str());
    }
    else
    {
        fprintf(stderr, "Unexpected list type. Expected: %s. Got: %s.\n", fourccToString(expected_fourcc).c_str(), fourccToString(list.m_list_type_cc).c_str());
    }
}

void AVIReadContainer::printError(RiffChunk &chunk, uint32_t expected_fourcc)
{
    if(!m_file_stream)
    {
        fprintf(stderr, "Unexpected end of file while searching for %s chunk\n", fourccToString(expected_fourcc).c_str());
    }
    else
    {
        fprintf(stderr, "Unexpected element. Expected: %s. Got: %s.\n", fourccToString(expected_fourcc).c_str(), fourccToString(chunk.m_four_cc).c_str());
    }
}

class BitStream
{
public:
    BitStream();
    ~BitStream() { close(); }

    bool open(const String& filename);
    bool isOpened() const { return output.is_open(); }
    void close();

    void writeBlock();
    size_t getPos() const;
    void putByte(int val);
    void putBytes(const uchar* buf, int count);

    void putShort(int val);
    void putInt(uint32_t val);
    void jputShort(int val);
    void patchInt(uint32_t val, size_t pos);
    void jput(unsigned currval);
    void jflush(unsigned currval, int bitIdx);

private:
    BitStream(const BitStream &);
    BitStream &operator=(const BitStream&);

protected:
    std::ofstream output;
    std::vector<uchar> m_buf;
    uchar*  m_start;
    uchar*  m_end;
    uchar*  m_current;
    size_t  m_pos;
    bool    m_is_opened;
};

static const size_t DEFAULT_BLOCK_SIZE = (1 << 15);

BitStream::BitStream()
{
    m_buf.resize(DEFAULT_BLOCK_SIZE + 1024);
    m_start = &m_buf[0];
    m_end = m_start + DEFAULT_BLOCK_SIZE;
    m_is_opened = false;
    m_current = 0;
    m_pos = 0;
}

bool BitStream::open(const String& filename)
{
    close();
    output.open(filename.c_str(), std::ios_base::binary);
    m_current = m_start;
    m_pos = 0;
    return true;
}

void BitStream::close()
{
    writeBlock();
    output.close();
}

void BitStream::writeBlock()
{
    ptrdiff_t wsz0 = m_current - m_start;
    if( wsz0 > 0 )
    {
        output.write((char*)m_start, wsz0);
    }
    m_pos += wsz0;
    m_current = m_start;
}

size_t BitStream::getPos() const {
    return safe_int_cast<size_t>(m_current - m_start, "Failed to determine AVI buffer position: value is out of range") + m_pos;
}

void BitStream::putByte(int val)
{
    *m_current++ = (uchar)val;
    if( m_current >= m_end )
        writeBlock();
}

void BitStream::putBytes(const uchar* buf, int count)
{
    uchar* data = (uchar*)buf;
    CV_Assert(data && m_current && count >= 0);
    if( m_current >= m_end )
        writeBlock();

    while( count )
    {
        int l = (int)(m_end - m_current);

        if (l > count)
            l = count;

        if( l > 0 )
        {
            memcpy(m_current, data, l);
            m_current += l;
            data += l;
            count -= l;
        }
        if( m_current >= m_end )
            writeBlock();
    }
}

void BitStream::putShort(int val)
{
    m_current[0] = (uchar)val;
    m_current[1] = (uchar)(val >> 8);
    m_current += 2;
    if( m_current >= m_end )
        writeBlock();
}

void BitStream::putInt(uint32_t val)
{
    m_current[0] = (uchar)val;
    m_current[1] = (uchar)(val >> 8);
    m_current[2] = (uchar)(val >> 16);
    m_current[3] = (uchar)(val >> 24);
    m_current += 4;
    if( m_current >= m_end )
        writeBlock();
}

void BitStream::jputShort(int val)
{
    m_current[0] = (uchar)(val >> 8);
    m_current[1] = (uchar)val;
    m_current += 2;
    if( m_current >= m_end )
        writeBlock();
}

void BitStream::patchInt(uint32_t val, size_t pos)
{
    if( pos >= m_pos )
    {
        ptrdiff_t delta = safe_int_cast<ptrdiff_t>(pos - m_pos, "Failed to seek in AVI buffer: value is out of range");
        CV_Assert( delta < m_current - m_start );
        m_start[delta] = (uchar)val;
        m_start[delta+1] = (uchar)(val >> 8);
        m_start[delta+2] = (uchar)(val >> 16);
        m_start[delta+3] = (uchar)(val >> 24);
    }
    else
    {
        std::streamoff fpos = output.tellp();
        output.seekp(safe_int_cast<std::streamoff>(pos, "Failed to seek in AVI file: value is out of range"));
        uchar buf[] = { (uchar)val, (uchar)(val >> 8), (uchar)(val >> 16), (uchar)(val >> 24) };
        output.write((char *)buf, 4);
        output.seekp(fpos);
    }
}

void BitStream::jput(unsigned currval)
{
    uchar v;
    uchar* ptr = m_current;
    v = (uchar)(currval >> 24);
    *ptr++ = v;
    if( v == 255 )
        *ptr++ = 0;
    v = (uchar)(currval >> 16);
    *ptr++ = v;
    if( v == 255 )
        *ptr++ = 0;
    v = (uchar)(currval >> 8);
    *ptr++ = v;
    if( v == 255 )
        *ptr++ = 0;
    v = (uchar)currval;
    *ptr++ = v;
    if( v == 255 )
        *ptr++ = 0;
    m_current = ptr;
    if( m_current >= m_end )
        writeBlock();
}

void BitStream::jflush(unsigned currval, int bitIdx)
{
    uchar v;
    uchar* ptr = m_current;
    currval |= (1 << bitIdx)-1;
    while( bitIdx < 32 )
    {
        v = (uchar)(currval >> 24);
        *ptr++ = v;
        if( v == 255 )
            *ptr++ = 0;
        currval <<= 8;
        bitIdx += 8;
    }
    m_current = ptr;
    if( m_current >= m_end )
        writeBlock();
}

AVIWriteContainer::AVIWriteContainer() : strm(makePtr<BitStream>())
{
    outfps = 0;
    height = 0;
    width = 0;
    channels = 0;
    moviPointer = 0;
    strm->close();
}

AVIWriteContainer::~AVIWriteContainer() {
    strm->close();
    frameOffset.clear();
    frameSize.clear();
    AVIChunkSizeIndex.clear();
    frameNumIndexes.clear();
}

bool AVIWriteContainer::initContainer(const String& filename, double fps, Size size, bool iscolor)
{
    outfps = cvRound(fps);
    width = size.width;
    height = size.height;
    channels = iscolor ? 3 : 1;
    moviPointer = 0;
    bool result = strm->open(filename);
    return result;
}

void AVIWriteContainer::startWriteAVI(int stream_count)
{
    startWriteChunk(RIFF_CC);

    strm->putInt(AVI_CC);

    startWriteChunk(LIST_CC);

    strm->putInt(HDRL_CC);
    strm->putInt(AVIH_CC);
    strm->putInt(AVIH_STRH_SIZE);
    strm->putInt(cvRound(1e6 / outfps));
    strm->putInt(MAX_BYTES_PER_SEC);
    strm->putInt(0);
    strm->putInt(AVI_DWFLAG);

    frameNumIndexes.push_back(strm->getPos());

    strm->putInt(0);
    strm->putInt(0);
    strm->putInt(stream_count); // number of streams
    strm->putInt(SUG_BUFFER_SIZE);
    strm->putInt(width);
    strm->putInt(height);
    strm->putInt(0);
    strm->putInt(0);
    strm->putInt(0);
    strm->putInt(0);
}

void AVIWriteContainer::writeStreamHeader(Codecs codec_)
{
    // strh
    startWriteChunk(LIST_CC);

    strm->putInt(STRL_CC);
    strm->putInt(STRH_CC);
    strm->putInt(AVIH_STRH_SIZE);
    strm->putInt(VIDS_CC);
    switch (codec_) {
      case MJPEG:
        strm->putInt(MJPG_CC);
      break;
    }
    strm->putInt(0);
    strm->putInt(0);
    strm->putInt(0);
    strm->putInt(AVI_DWSCALE);
    strm->putInt(outfps);
    strm->putInt(0);

    frameNumIndexes.push_back(strm->getPos());

    strm->putInt(0);
    strm->putInt(SUG_BUFFER_SIZE);
    strm->putInt(static_cast<uint32_t>(AVI_DWQUALITY));
    strm->putInt(0);
    strm->putShort(0);
    strm->putShort(0);
    strm->putShort(width);
    strm->putShort(height);

    // strf (use the BITMAPINFOHEADER for video)
    startWriteChunk(STRF_CC);

    strm->putInt(STRF_SIZE);
    strm->putInt(width);
    strm->putInt(height);
    strm->putShort(1); // planes (1 means interleaved data (after decompression))

    strm->putShort(8 * channels); // bits per pixel
    switch (codec_) {
      case MJPEG:
        strm->putInt(MJPG_CC);
      break;
    }
    strm->putInt(width * height * channels);
    strm->putInt(0);
    strm->putInt(0);
    strm->putInt(0);
    strm->putInt(0);
    // Must be indx chunk
    endWriteChunk(); // end strf

    endWriteChunk(); // end strl

    // odml
    startWriteChunk(LIST_CC);
    strm->putInt(ODML_CC);
    startWriteChunk(DMLH_CC);

    frameNumIndexes.push_back(strm->getPos());

    strm->putInt(0);
    strm->putInt(0);

    endWriteChunk(); // end dmlh
    endWriteChunk(); // end odml

    endWriteChunk(); // end hdrl

    // JUNK
    startWriteChunk(JUNK_CC);
    size_t pos = strm->getPos();
    for( ; pos < (size_t)JUNK_SEEK; pos += 4 )
        strm->putInt(0);
    endWriteChunk(); // end JUNK

    // movi
    startWriteChunk(LIST_CC);
    moviPointer = strm->getPos();
    strm->putInt(MOVI_CC);
}

void AVIWriteContainer::startWriteChunk(uint32_t fourcc)
{
    CV_Assert(fourcc != 0);
    strm->putInt(fourcc);

    AVIChunkSizeIndex.push_back(strm->getPos());
    strm->putInt(0);
}

void AVIWriteContainer::endWriteChunk()
{
    if( !AVIChunkSizeIndex.empty() )
    {
        size_t currpos = strm->getPos();
        CV_Assert(currpos > 4);
        currpos -= 4;
        size_t pospos = AVIChunkSizeIndex.back();
        AVIChunkSizeIndex.pop_back();
        CV_Assert(currpos >= pospos);
        uint32_t chunksz = safe_int_cast<uint32_t>(currpos - pospos, "Failed to write AVI file: chunk size is out of bounds");
        strm->patchInt(chunksz, pospos);
    }
}

int AVIWriteContainer::getAVIIndex(int stream_number, StreamType strm_type) {
    char strm_indx[2];
    strm_indx[0] = '0' + static_cast<char>(stream_number / 10);
    strm_indx[1] = '0' + static_cast<char>(stream_number % 10);

    switch (strm_type) {
      case db: return CV_FOURCC(strm_indx[0], strm_indx[1], 'd', 'b');
      case dc: return CV_FOURCC(strm_indx[0], strm_indx[1], 'd', 'c');
      case pc: return CV_FOURCC(strm_indx[0], strm_indx[1], 'p', 'c');
      case wb: return CV_FOURCC(strm_indx[0], strm_indx[1], 'w', 'b');
    }
    return CV_FOURCC(strm_indx[0], strm_indx[1], 'd', 'b');
}

void AVIWriteContainer::writeIndex(int stream_number, StreamType strm_type)
{
    // old style AVI index. Must be Open-DML index
    startWriteChunk(IDX1_CC);
    int nframes = (int)frameOffset.size();
    for( int i = 0; i < nframes; i++ )
    {
        strm->putInt(getAVIIndex(stream_number, strm_type));
        strm->putInt(AVIIF_KEYFRAME);
        strm->putInt((int)frameOffset[i]);
        strm->putInt((int)frameSize[i]);
    }
    endWriteChunk(); // End idx1
}

void AVIWriteContainer::finishWriteAVI()
{
    uint32_t nframes = safe_int_cast<uint32_t>(frameOffset.size(), "Failed to write AVI file: number of frames is too large");
    // Record frames numbers to AVI Header
    while (!frameNumIndexes.empty())
    {
        size_t ppos = frameNumIndexes.back();
        frameNumIndexes.pop_back();
        strm->patchInt(nframes, ppos);
    }
    endWriteChunk(); // end RIFF
}

bool AVIWriteContainer::isOpenedStream() const { return strm->isOpened(); }

size_t AVIWriteContainer::getStreamPos() const { return strm->getPos(); }

void AVIWriteContainer::jputStreamShort(int val) { strm->jputShort(val); }

void AVIWriteContainer::putStreamBytes(const uchar *buf, int count) { strm->putBytes( buf, count ); }

void AVIWriteContainer::putStreamByte(int val) { strm->putByte(val); }

void AVIWriteContainer::jputStream(unsigned currval) { strm->jput(currval); }

void AVIWriteContainer::jflushStream(unsigned currval, int bitIdx) {  strm->jflush(currval, bitIdx); }

}
