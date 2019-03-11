// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef CONTAINER_AVI_HPP
#define CONTAINER_AVI_HPP

#ifndef __OPENCV_BUILD
#  error this is a private header which should not be used from outside of the OpenCV library
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/videoio/videoio_c.h"
#include <deque>

namespace cv
{

/*
AVI struct:

RIFF ('AVI '
      LIST ('hdrl'
            'avih'(<Main AVI Header>)
            LIST ('strl'
                  'strh'(<Stream header>)
                  'strf'(<Stream format>)
                  [ 'strd'(<Additional header data>) ]
                  [ 'strn'(<Stream name>) ]
                  [ 'indx'(<Odml index data>) ]
                  ...
                 )
            [LIST ('strl' ...)]
            [LIST ('strl' ...)]
            ...
            [LIST ('odml'
                  'dmlh'(<ODML header data>)
                  ...
                 )
            ]
            ...
           )
      [LIST ('INFO' ...)]
      [JUNK]
      LIST ('movi'
            {{xxdb|xxdc|xxpc|xxwb}(<Data>) | LIST ('rec '
                              {xxdb|xxdc|xxpc|xxwb}(<Data>)
                              {xxdb|xxdc|xxpc|xxwb}(<Data>)
                              ...
                             )
               ...
            }
            ...
           )
      ['idx1' (<AVI Index>) ]
     )

     {xxdb|xxdc|xxpc|xxwb}
     xx - stream number: 00, 01, 02, ...
     db - uncompressed video frame
     dc - commpressed video frame
     pc - palette change
     wb - audio frame

     JUNK section may pad any data section and must be ignored
*/

typedef std::deque< std::pair<uint64_t, uint32_t> > frame_list;
typedef frame_list::iterator frame_iterator;
struct RiffChunk;
struct RiffList;
class VideoInputStream;
enum Codecs { MJPEG };

//Represents single MJPEG video stream within single AVI/AVIX entry
//Multiple video streams within single AVI/AVIX entry are not supported
//ODML index is not supported
class CV_EXPORTS AVIReadContainer
{
public:
    AVIReadContainer();

    void initStream(const String& filename);
    void initStream(Ptr<VideoInputStream> m_file_stream_);

    void close();
    //stores founded frames in m_frame_list which can be accessed via getFrames
    bool parseAvi(Codecs codec_) { return parseAviWithFrameList(m_frame_list, codec_); }
    //stores founded frames in in_frame_list. getFrames() would return empty list
    bool parseAvi(frame_list& in_frame_list, Codecs codec_) { return parseAviWithFrameList(in_frame_list, codec_); }
    size_t getFramesCount() { return m_frame_list.size(); }
    frame_list& getFrames() { return m_frame_list; }
    unsigned int getWidth() { return m_width; }
    unsigned int getHeight() { return m_height; }
    double getFps() { return m_fps; }
    std::vector<char> readFrame(frame_iterator it);
    bool parseRiff(frame_list &m_mjpeg_frames);

protected:

    bool parseAviWithFrameList(frame_list& in_frame_list, Codecs codec_);
    void skipJunk(RiffChunk& chunk);
    void skipJunk(RiffList& list);
    bool parseHdrlList(Codecs codec_);
    bool parseIndex(unsigned int index_size, frame_list& in_frame_list);
    bool parseMovi(frame_list& in_frame_list)
    {
        //not implemented
        CV_UNUSED(in_frame_list);
        // FIXIT: in_frame_list.empty();
        return true;
    }
    bool parseStrl(char stream_id, Codecs codec_);
    bool parseInfo()
    {
        //not implemented
        return true;
    }

    void printError(RiffList& list, unsigned int expected_fourcc);

    void printError(RiffChunk& chunk, unsigned int expected_fourcc);

    Ptr<VideoInputStream> m_file_stream;
    unsigned int   m_stream_id;
    unsigned long long int   m_movi_start;
    unsigned long long int    m_movi_end;
    frame_list m_frame_list;
    unsigned int   m_width;
    unsigned int   m_height;
    double     m_fps;
    bool       m_is_indx_present;
};

enum { COLORSPACE_GRAY=0, COLORSPACE_RGBA=1, COLORSPACE_BGR=2, COLORSPACE_YUV444P=3 };
enum StreamType { db, dc, pc, wb };
class BitStream;

// {xxdb|xxdc|xxpc|xxwb}
// xx - stream number: 00, 01, 02, ...
// db - uncompressed video frame
// dc - commpressed video frame
// pc - palette change
// wb - audio frame


class CV_EXPORTS AVIWriteContainer
{
public:
    AVIWriteContainer();
    ~AVIWriteContainer();

    bool initContainer(const String& filename, double fps, Size size, bool iscolor);
    void startWriteAVI(int stream_count);
    void writeStreamHeader(Codecs codec_);
    void startWriteChunk(uint32_t fourcc);
    void endWriteChunk();

    int getAVIIndex(int stream_number, StreamType strm_type);
    void writeIndex(int stream_number, StreamType strm_type);
    void finishWriteAVI();

    bool isOpenedStream() const;
    bool isEmptyFrameOffset() const { return frameOffset.empty(); }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
    size_t getMoviPointer() const { return moviPointer; }
    size_t getStreamPos() const;

    void pushFrameOffset(size_t elem) { frameOffset.push_back(elem); }
    void pushFrameSize(size_t elem) { frameSize.push_back(elem); }
    bool isEmptyFrameSize() const { return frameSize.empty(); }
    size_t atFrameSize(size_t i) const { return frameSize[i]; }
    size_t countFrameSize() const { return frameSize.size(); }
    void jputStreamShort(int val);
    void putStreamBytes(const uchar* buf, int count);
    void putStreamByte(int val);
    void jputStream(unsigned currval);
    void jflushStream(unsigned currval, int bitIdx);

private:
    Ptr<BitStream> strm;
    int outfps;
    int width, height, channels;
    size_t moviPointer;
    std::vector<size_t> frameOffset, frameSize, AVIChunkSizeIndex, frameNumIndexes;
};

}

#endif //CONTAINER_AVI_HPP
