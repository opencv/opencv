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
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "opencv2/videoio/container_avi.private.hpp"

#include <vector>
#include <deque>
#include <iostream>
#include <cstdlib>

#if CV_NEON
#define WITH_NEON
#endif

namespace cv
{

static const unsigned bit_mask[] =
{
    0,
    0x00000001, 0x00000003, 0x00000007, 0x0000000F,
    0x0000001F, 0x0000003F, 0x0000007F, 0x000000FF,
    0x000001FF, 0x000003FF, 0x000007FF, 0x00000FFF,
    0x00001FFF, 0x00003FFF, 0x00007FFF, 0x0000FFFF,
    0x0001FFFF, 0x0003FFFF, 0x0007FFFF, 0x000FFFFF,
    0x001FFFFF, 0x003FFFFF, 0x007FFFFF, 0x00FFFFFF,
    0x01FFFFFF, 0x03FFFFFF, 0x07FFFFFF, 0x0FFFFFFF,
    0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF
};

static const uchar huff_val_shift = 20;
static const int huff_code_mask = (1 << huff_val_shift) - 1;

static bool createEncodeHuffmanTable( const int* src, unsigned* table, int max_size )
{
    int  i, k;
    int  min_val = INT_MAX, max_val = INT_MIN;
    int  size;

    /* calc min and max values in the table */
    for( i = 1, k = 1; src[k] >= 0; i++ )
    {
        int code_count = src[k++];

        for( code_count += k; k < code_count; k++ )
        {
            int  val = src[k] >> huff_val_shift;
            if( val < min_val )
                min_val = val;
            if( val > max_val )
                max_val = val;
        }
    }

    size = max_val - min_val + 3;

    if( size > max_size )
    {
        CV_Error(cv::Error::StsOutOfRange, "too big maximum Huffman code size");
    }

    memset( table, 0, size*sizeof(table[0]));

    table[0] = min_val;
    table[1] = size - 2;

    for( i = 1, k = 1; src[k] >= 0; i++ )
    {
        int code_count = src[k++];

        for( code_count += k; k < code_count; k++ )
        {
            int  val = src[k] >> huff_val_shift;
            int  code = src[k] & huff_code_mask;

            table[val - min_val + 2] = (code << 8) | i;
        }
    }
    return true;
}

static int* createSourceHuffmanTable(const uchar* src, int* dst,
                                         int max_bits, int first_bits)
{
    int   i, val_idx, code = 0;
    int*  table = dst;
    *dst++ = first_bits;
    for (i = 1, val_idx = max_bits; i <= max_bits; i++)
    {
        int code_count = src[i - 1];
        dst[0] = code_count;
        code <<= 1;
        for (int k = 0; k < code_count; k++)
        {
            dst[k + 1] = (src[val_idx + k] << huff_val_shift) | (code + k);
        }
        code += code_count;
        dst += code_count + 1;
        val_idx += code_count;
    }
    dst[0] = -1;
    return  table;
}


namespace mjpeg
{

class mjpeg_buffer
{
public:
    mjpeg_buffer()
    {
        reset();
    }

    void resize(int size)
    {
        data.resize(size);
    }

    inline void put_bits(unsigned bits, int len)
    {
        CV_Assert(len >=0 && len < 32);
        if((m_pos == (data.size() - 1) && len > bits_free) || m_pos == data.size())
        {
            resize(int(2*data.size()));
        }

        bits_free -= (len);
        unsigned int tempval = (bits) & bit_mask[(len)];

        if( bits_free <= 0 )
        {
            data[m_pos] |= ((unsigned)tempval >> -bits_free);

            bits_free += 32;
            ++m_pos;
            data[m_pos] = bits_free < 32 ? (tempval << bits_free) : 0;
        }
        else
        {
            data[m_pos] |= (bits_free == 32) ? tempval : (tempval << bits_free);
        }
    }

    inline void put_val(int val, const unsigned * table)
    {
        unsigned code = table[(val) + 2];
        put_bits(code >> 8, (int)(code & 255));
    }

    void finish()
    {
        if(bits_free == 32)
        {
            bits_free = 0;
            m_data_len = m_pos;
        }
        else
        {
            m_data_len = m_pos + 1;
        }
    }

    void reset()
    {
        bits_free = 32;
        m_pos = 0;
        m_data_len = 0;
    }

    void clear()
    {
        //we need to clear only first element, the rest would be overwritten
        data[0] = 0;
    }

    int get_bits_free()
    {
        return bits_free;
    }

    unsigned* get_data()
    {
        return &data[0];
    }

    unsigned get_len()
    {
        return m_data_len;
    }

private:
    std::vector<unsigned> data;
    int bits_free;
    unsigned m_pos;
    unsigned m_data_len;
};


class mjpeg_buffer_keeper
{
public:
    mjpeg_buffer_keeper()
    {
        reset();
    }

    mjpeg_buffer& operator[](int i)
    {
        return m_buffer_list[i];
    }

    void allocate_buffers(int count, int size)
    {
        for(int i = (int)m_buffer_list.size(); i < count; ++i)
        {
            m_buffer_list.push_back(mjpeg_buffer());
            m_buffer_list.back().resize(size);
        }
    }

    unsigned* get_data()
    {
        //if there is only one buffer (single thread) there is no need to stack buffers
        if(m_buffer_list.size() == 1)
        {
            m_buffer_list[0].finish();

            m_data_len = m_buffer_list[0].get_len();
            m_last_bit_len = 32 - m_buffer_list[0].get_bits_free();

            return m_buffer_list[0].get_data();
        }

        allocate_output_buffer();

        int bits = 0;
        unsigned currval = 0;
        m_data_len = 0;

        for(unsigned j = 0; j < m_buffer_list.size(); ++j)
        {
            mjpeg_buffer& buffer = m_buffer_list[j];

            //if no bit shift required we could use memcpy
            if(bits == 0)
            {
                size_t current_pos = m_data_len;

                if(buffer.get_bits_free() == 0)
                {
                    memcpy(&m_output_buffer[current_pos], buffer.get_data(), sizeof(buffer.get_data()[0])*buffer.get_len());
                    m_data_len += buffer.get_len();
                    currval = 0;
                }
                else
                {
                    memcpy(&m_output_buffer[current_pos], buffer.get_data(), sizeof(buffer.get_data()[0])*(buffer.get_len() - 1 ));
                    m_data_len += buffer.get_len() - 1;
                    currval = buffer.get_data()[buffer.get_len() - 1];
                }
            }
            else
            {
                for(unsigned i = 0; i < buffer.get_len() - 1; ++i)
                {
                    currval |= ( (unsigned)buffer.get_data()[i] >> (31 & (-bits)) );

                    m_output_buffer[m_data_len++] = currval;

                    currval = buffer.get_data()[i] << (bits + 32);
                }

                currval |= ( (unsigned)buffer.get_data()[buffer.get_len() - 1] >> (31 & (-bits)) );

                if( buffer.get_bits_free() <= -bits)
                {
                    m_output_buffer[m_data_len++] = currval;

                    currval = buffer.get_data()[buffer.get_len() - 1] << (bits + 32);
                }
            }

            bits += buffer.get_bits_free();

            if(bits > 0)
            {
                bits -= 32;
            }
        }

        //bits == 0 means that last element shouldn't be used.
        if (bits != 0) {
            m_output_buffer[m_data_len++] = currval;
            m_last_bit_len = -bits;
        }
        else
        {
            m_last_bit_len = 32;
        }

        return &m_output_buffer[0];
    }

    int get_last_bit_len()
    {
        return m_last_bit_len;
    }

    int get_data_size()
    {
        return m_data_len;
    }

    void reset()
    {
        m_last_bit_len = 0;
        for(unsigned i = 0; i < m_buffer_list.size(); ++i)
        {
            m_buffer_list[i].reset();
        }

        //there is no need to erase output buffer since it would be overwritten
        m_data_len = 0;
    }

private:

    void allocate_output_buffer()
    {
        unsigned total_size = 0;

        for(unsigned i = 0; i < m_buffer_list.size(); ++i)
        {
            m_buffer_list[i].finish();
            total_size += m_buffer_list[i].get_len();
        }

        if(total_size > m_output_buffer.size())
        {
            m_output_buffer.clear();
            m_output_buffer.resize(total_size);
        }
    }

    std::deque<mjpeg_buffer> m_buffer_list;
    std::vector<unsigned> m_output_buffer;
    int m_data_len;
    int m_last_bit_len;
};

class MotionJpegWriter : public IVideoWriter
{
public:
    MotionJpegWriter()
    {
        rawstream = false;
        nstripes = -1;
        quality = 0;
    }

    MotionJpegWriter(const String& filename, double fps, Size size, bool iscolor)
    {
        rawstream = false;
        open(filename, fps, size, iscolor);
        nstripes = -1;
    }
    ~MotionJpegWriter() { close(); }

    virtual int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_OPENCV_MJPEG; }

    void close()
    {
        if( !container.isOpenedStream() )
            return;

        if( !container.isEmptyFrameOffset() && !rawstream )
        {
            container.endWriteChunk(); // end LIST 'movi'
            container.writeIndex(0, dc);
            container.finishWriteAVI();
        }
    }

    bool open(const String& filename, double fps, Size size, bool iscolor)
    {
        close();

        if( filename.empty() )
            return false;
        const char* ext = strrchr(filename.c_str(), '.');
        if( !ext )
            return false;
        if( strcmp(ext, ".avi") != 0 && strcmp(ext, ".AVI") != 0 && strcmp(ext, ".Avi") != 0 )
            return false;

        if( !container.initContainer(filename, fps, size, iscolor) )
            return false;

        CV_Assert(fps >= 1);
        quality = 75;
        rawstream = false;

        if( !rawstream )
        {
            container.startWriteAVI(1); // count stream
            container.writeStreamHeader(MJPEG);
        }
        //printf("motion jpeg stream %s has been successfully opened\n", filename.c_str());
        return true;
    }

    bool isOpened() const CV_OVERRIDE { return container.isOpenedStream(); }

    void write(InputArray _img) CV_OVERRIDE
    {
        Mat img = _img.getMat();
        size_t chunkPointer = container.getStreamPos();
        int input_channels = img.channels();
        int colorspace = -1;
        int imgWidth = img.cols;
        int frameWidth = container.getWidth();
        int imgHeight = img.rows;
        int frameHeight = container.getHeight();
        int channels = container.getChannels();


        if( input_channels == 1 && channels == 1 )
        {
            CV_Assert( imgWidth == frameWidth && imgHeight == frameHeight );
            colorspace = COLORSPACE_GRAY;
        }
        else if( input_channels == 4 )
        {
            CV_Assert( imgWidth == frameWidth && imgHeight == frameHeight && channels == 3 );
            colorspace = COLORSPACE_RGBA;
        }
        else if( input_channels == 3 )
        {
            CV_Assert( imgWidth == frameWidth && imgHeight == frameHeight && channels == 3 );
            colorspace = COLORSPACE_BGR;
        }
        else if( input_channels == 1 && channels == 3 )
        {
            CV_Assert( imgWidth == frameWidth && imgHeight == frameHeight*3 );
            colorspace = COLORSPACE_YUV444P;
        }
        else
            CV_Error(cv::Error::StsBadArg, "Invalid combination of specified video colorspace and the input image colorspace");

        if( !rawstream ) {
            int avi_index = container.getAVIIndex(0, dc);
            container.startWriteChunk(avi_index);
        }

        writeFrameData(img.data, (int)img.step, colorspace, input_channels);

        if( !rawstream )
        {
            size_t tempChunkPointer = container.getStreamPos();
            size_t moviPointer = container.getMoviPointer();
            container.pushFrameOffset(chunkPointer - moviPointer);
            container.pushFrameSize(tempChunkPointer - chunkPointer - 8);       // Size excludes '00dc' and size field
            container.endWriteChunk(); // end '00dc'
        }
    }

    double getProperty(int propId) const CV_OVERRIDE
    {
        if( propId == VIDEOWRITER_PROP_QUALITY )
            return quality;
        if( propId == VIDEOWRITER_PROP_FRAMEBYTES )
        {
            bool isEmpty = container.isEmptyFrameSize();
            return isEmpty ? 0. : container.atFrameSize(container.countFrameSize() - 1);
        }
        if( propId == VIDEOWRITER_PROP_NSTRIPES )
            return nstripes;
        return 0.;
    }

    bool setProperty(int propId, double value) CV_OVERRIDE
    {
        if( propId == VIDEOWRITER_PROP_QUALITY )
        {
            quality = value;
            return true;
        }

        if( propId == VIDEOWRITER_PROP_NSTRIPES)
        {
            nstripes = value;
            return true;
        }

        return false;
    }

    void writeFrameData( const uchar* data, int step, int colorspace, int input_channels );

protected:
    double quality;
    bool rawstream;
    mjpeg_buffer_keeper buffers_list;
    double nstripes;

    AVIWriteContainer container;
};

#define DCT_DESCALE(x, n) (((x) + (((int)1) << ((n) - 1))) >> (n))
#define fix(x, n)   (int)((x)*(1 << (n)) + .5);

enum
{
    fixb = 14,
    fixc = 12,
    postshift = 14
};

static const int C0_707 = fix(0.707106781f, fixb);
static const int C0_541 = fix(0.541196100f, fixb);
static const int C0_382 = fix(0.382683432f, fixb);
static const int C1_306 = fix(1.306562965f, fixb);

static const int y_r = fix(0.299, fixc);
static const int y_g = fix(0.587, fixc);
static const int y_b = fix(0.114, fixc);

static const int cb_r = -fix(0.1687, fixc);
static const int cb_g = -fix(0.3313, fixc);
static const int cb_b = fix(0.5, fixc);

static const int cr_r = fix(0.5, fixc);
static const int cr_g = -fix(0.4187, fixc);
static const int cr_b = -fix(0.0813, fixc);

// Standard JPEG quantization tables
static const uchar jpegTableK1_T[] =
{
    16, 12, 14, 14,  18,  24,  49,  72,
    11, 12, 13, 17,  22,  35,  64,  92,
    10, 14, 16, 22,  37,  55,  78,  95,
    16, 19, 24, 29,  56,  64,  87,  98,
    24, 26, 40, 51,  68,  81, 103, 112,
    40, 58, 57, 87, 109, 104, 121, 100,
    51, 60, 69, 80, 103, 113, 120, 103,
    61, 55, 56, 62,  77,  92, 101,  99
};

static const uchar jpegTableK2_T[] =
{
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

// Standard Huffman tables

// ... for luma DCs.
static const uchar jpegTableK3[] =
{
    0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
};

// ... for chroma DCs.
static const uchar jpegTableK4[] =
{
    0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
};

// ... for luma ACs.
static const uchar jpegTableK5[] =
{
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

// ... for chroma ACs
static const uchar jpegTableK6[] =
{
    0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119,
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

static const uchar zigzag[] =
{
    0,  8,  1,  2,  9, 16, 24, 17, 10,  3,  4, 11, 18, 25, 32, 40,
    33, 26, 19, 12,  5,  6, 13, 20, 27, 34, 41, 48, 56, 49, 42, 35,
    28, 21, 14,  7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30,
    23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63
};


static const int idct_prescale[] =
{
    16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
    22725, 31521, 29692, 26722, 22725, 17855, 12299,  6270,
    21407, 29692, 27969, 25172, 21407, 16819, 11585,  5906,
    19266, 26722, 25172, 22654, 19266, 15137, 10426,  5315,
    16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
    12873, 17855, 16819, 15137, 12873, 10114,  6967,  3552,
    8867, 12299, 11585, 10426,  8867,  6967,  4799,  2446,
    4520,  6270,  5906,  5315,  4520,  3552,  2446,  1247
};

static const char jpegHeader[] =
"\xFF\xD8"  // SOI  - start of image
"\xFF\xE0"  // APP0 - jfif extension
"\x00\x10"  // 2 bytes: length of APP0 segment
"JFIF\x00"  // JFIF signature
"\x01\x02"  // version of JFIF
"\x00"      // units = pixels ( 1 - inch, 2 - cm )
"\x00\x01\x00\x01" // 2 2-bytes values: x density & y density
"\x00\x00"; // width & height of thumbnail: ( 0x0 means no thumbnail)

#ifdef WITH_NEON
// FDCT with postscaling
static void aan_fdct8x8( const short *src, short *dst,
                        int step, const short *postscale )
{
    // Pass 1: process rows
    int16x8_t x0 = vld1q_s16(src);    int16x8_t x1 = vld1q_s16(src + step*7);
    int16x8_t x2 = vld1q_s16(src + step*3);    int16x8_t x3 = vld1q_s16(src + step*4);

    int16x8_t x4 = vaddq_s16(x0, x1);    x0 = vsubq_s16(x0, x1);
    x1 = vaddq_s16(x2, x3);    x2 = vsubq_s16(x2, x3);

    int16x8_t t1 = x0; int16x8_t t2 = x2;

    x2 = vaddq_s16(x4, x1);    x4 = vsubq_s16(x4, x1);

    x0 = vld1q_s16(src + step);    x3 = vld1q_s16(src + step*6);

    x1 = vaddq_s16(x0, x3);    x0 = vsubq_s16(x0, x3);
    int16x8_t t3 = x0;

    x0 = vld1q_s16(src + step*2);    x3 = vld1q_s16(src + step*5);

    int16x8_t t4 = vsubq_s16(x0, x3);

    x0 = vaddq_s16(x0, x3);
    x3 = vaddq_s16(x0, x1);    x0 = vsubq_s16(x0, x1);
    x1 = vaddq_s16(x2, x3);    x2 = vsubq_s16(x2, x3);

    int16x8_t res0 = x1;
    int16x8_t res4 = x2;
    x0 = vqdmulhq_n_s16(vsubq_s16(x0, x4), (short)(C0_707*2));
    x1 = vaddq_s16(x4, x0);    x4 = vsubq_s16(x4, x0);

    int16x8_t res2 = x4;
    int16x8_t res6 = x1;

    x0 = t2;    x1 = t4;
    x2 = t3;    x3 = t1;
    x0 = vaddq_s16(x0, x1);    x1 = vaddq_s16(x1, x2);    x2 = vaddq_s16(x2, x3);
    x1 =vqdmulhq_n_s16(x1, (short)(C0_707*2));

    x4 = vaddq_s16(x1, x3);    x3 = vsubq_s16(x3, x1);
    x1 = vqdmulhq_n_s16(vsubq_s16(x0, x2), (short)(C0_382*2));
    x0 = vaddq_s16(vqdmulhq_n_s16(x0, (short)(C0_541*2)), x1);
    x2 = vaddq_s16(vshlq_n_s16(vqdmulhq_n_s16(x2, (short)C1_306), 1), x1);

    x1 = vaddq_s16(x0, x3);    x3 = vsubq_s16(x3, x0);
    x0 = vaddq_s16(x4, x2);    x4 = vsubq_s16(x4, x2);

    int16x8_t res1 = x0;
    int16x8_t res3 = x3;
    int16x8_t res5 = x1;
    int16x8_t res7 = x4;

    //transpose a matrix
    /*
     res0 00 01 02 03 04 05 06 07
     res1 10 11 12 13 14 15 16 17
     res2 20 21 22 23 24 25 26 27
     res3 30 31 32 33 34 35 36 37
     res4 40 41 42 43 44 45 46 47
     res5 50 51 52 53 54 55 56 57
     res6 60 61 62 63 64 65 66 67
     res7 70 71 72 73 74 75 76 77
     */

    //transpose elements 00-33
    int16x4_t res0_0 = vget_low_s16(res0);
    int16x4_t res1_0 = vget_low_s16(res1);
    int16x4x2_t tres = vtrn_s16(res0_0, res1_0);
    int32x4_t l0 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    res0_0 = vget_low_s16(res2);
    res1_0 = vget_low_s16(res3);
    tres = vtrn_s16(res0_0, res1_0);
    int32x4_t l1 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    int32x4x2_t tres1 = vtrnq_s32(l0, l1);

    // transpose elements 40-73
    res0_0 = vget_low_s16(res4);
    res1_0 = vget_low_s16(res5);
    tres = vtrn_s16(res0_0, res1_0);
    l0 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    res0_0 = vget_low_s16(res6);
    res1_0 = vget_low_s16(res7);

    tres = vtrn_s16(res0_0, res1_0);
    l1 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    int32x4x2_t tres2 = vtrnq_s32(l0, l1);

    //combine into 0-3
    int16x8_t transp_res0 =  vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(tres1.val[0]), vget_low_s32(tres2.val[0])));
    int16x8_t transp_res1 =  vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(tres1.val[0]), vget_high_s32(tres2.val[0])));
    int16x8_t transp_res2 =  vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(tres1.val[1]), vget_low_s32(tres2.val[1])));
    int16x8_t transp_res3 =  vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(tres1.val[1]), vget_high_s32(tres2.val[1])));

    // transpose elements 04-37
    res0_0 = vget_high_s16(res0);
    res1_0 = vget_high_s16(res1);
    tres = vtrn_s16(res0_0, res1_0);
    l0 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    res0_0 = vget_high_s16(res2);
    res1_0 = vget_high_s16(res3);

    tres = vtrn_s16(res0_0, res1_0);
    l1 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    tres1 = vtrnq_s32(l0, l1);

    // transpose elements 44-77
    res0_0 = vget_high_s16(res4);
    res1_0 = vget_high_s16(res5);
    tres = vtrn_s16(res0_0, res1_0);
    l0 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    res0_0 = vget_high_s16(res6);
    res1_0 = vget_high_s16(res7);

    tres = vtrn_s16(res0_0, res1_0);
    l1 = vcombine_s32(vreinterpret_s32_s16(tres.val[0]),vreinterpret_s32_s16(tres.val[1]));

    tres2 = vtrnq_s32(l0, l1);

    //combine into 4-7
    int16x8_t transp_res4 =  vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(tres1.val[0]), vget_low_s32(tres2.val[0])));
    int16x8_t transp_res5 =  vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(tres1.val[0]), vget_high_s32(tres2.val[0])));
    int16x8_t transp_res6 =  vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(tres1.val[1]), vget_low_s32(tres2.val[1])));
    int16x8_t transp_res7 =  vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(tres1.val[1]), vget_high_s32(tres2.val[1])));

    //special hack for vqdmulhq_s16 command that is producing -1 instead of 0
#define STORE_DESCALED(addr, reg, mul_addr)            postscale_line = vld1q_s16((mul_addr)); \
mask = vreinterpretq_s16_u16(vcltq_s16((reg), z)); \
reg = vabsq_s16(reg); \
reg = vqdmulhq_s16(vqaddq_s16((reg), (reg)), postscale_line); \
reg = vsubq_s16(veorq_s16(reg, mask), mask); \
vst1q_s16((addr), reg);

    int16x8_t z = vdupq_n_s16(0), postscale_line, mask;

    // pass 2: process columns
    x0 = transp_res0;    x1 = transp_res7;
    x2 = transp_res3;    x3 = transp_res4;

    x4 = vaddq_s16(x0, x1);   x0 = vsubq_s16(x0, x1);
    x1 = vaddq_s16(x2, x3);    x2 = vsubq_s16(x2, x3);

    t1 = x0; t2 = x2;

    x2 = vaddq_s16(x4, x1);    x4 = vsubq_s16(x4, x1);

    x0 = transp_res1;
    x3 = transp_res6;

    x1 = vaddq_s16(x0, x3);    x0 = vsubq_s16(x0, x3);

    t3 = x0;

    x0 = transp_res2; x3 = transp_res5;

    t4 = vsubq_s16(x0, x3);

    x0 = vaddq_s16(x0, x3);

    x3 = vaddq_s16(x0, x1);    x0 = vsubq_s16(x0, x1);
    x1 = vaddq_s16(x2, x3);    x2 = vsubq_s16(x2, x3);

    STORE_DESCALED(dst, x1, postscale);
    STORE_DESCALED(dst + 4*8, x2, postscale + 4*8);

    x0 = vqdmulhq_n_s16(vsubq_s16(x0, x4), (short)(C0_707*2));

    x1 = vaddq_s16(x4, x0);    x4 = vsubq_s16(x4, x0);

    STORE_DESCALED(dst + 2*8, x4,postscale + 2*8);
    STORE_DESCALED(dst + 6*8, x1,postscale + 6*8);

    x0 = t2; x1 = t4;
    x2 = t3; x3 = t1;

    x0 = vaddq_s16(x0, x1);    x1 = vaddq_s16(x1, x2);    x2 = vaddq_s16(x2, x3);

    x1 =vqdmulhq_n_s16(x1, (short)(C0_707*2));

    x4 = vaddq_s16(x1, x3);    x3 = vsubq_s16(x3, x1);

    x1 = vqdmulhq_n_s16(vsubq_s16(x0, x2), (short)(C0_382*2));
    x0 = vaddq_s16(vqdmulhq_n_s16(x0, (short)(C0_541*2)), x1);
    x2 = vaddq_s16(vshlq_n_s16(vqdmulhq_n_s16(x2, (short)C1_306), 1), x1);

    x1 = vaddq_s16(x0, x3);    x3 = vsubq_s16(x3, x0);
    x0 = vaddq_s16(x4, x2);    x4 = vsubq_s16(x4, x2);

    STORE_DESCALED(dst + 5*8, x1,postscale + 5*8);
    STORE_DESCALED(dst + 1*8, x0,postscale + 1*8);
    STORE_DESCALED(dst + 7*8, x4,postscale + 7*8);
    STORE_DESCALED(dst + 3*8, x3,postscale + 3*8);
}

#else
// FDCT with postscaling
static void aan_fdct8x8( const short *src, short *dst,
                        int step, const short *postscale )
{
    int workspace[64], *work = workspace;
    int  i;

    // Pass 1: process rows
    for( i = 8; i > 0; i--, src += step, work += 8 )
    {
        int x0 = src[0], x1 = src[7];
        int x2 = src[3], x3 = src[4];

        int x4 = x0 + x1; x0 -= x1;
        x1 = x2 + x3; x2 -= x3;

        work[7] = x0; work[1] = x2;
        x2 = x4 + x1; x4 -= x1;

        x0 = src[1]; x3 = src[6];
        x1 = x0 + x3; x0 -= x3;
        work[5] = x0;

        x0 = src[2]; x3 = src[5];
        work[3] = x0 - x3; x0 += x3;

        x3 = x0 + x1; x0 -= x1;
        x1 = x2 + x3; x2 -= x3;

        work[0] = x1; work[4] = x2;

        x0 = DCT_DESCALE((x0 - x4)*C0_707, fixb);
        x1 = x4 + x0; x4 -= x0;
        work[2] = x4; work[6] = x1;

        x0 = work[1]; x1 = work[3];
        x2 = work[5]; x3 = work[7];

        x0 += x1; x1 += x2; x2 += x3;
        x1 = DCT_DESCALE(x1*C0_707, fixb);

        x4 = x1 + x3; x3 -= x1;
        x1 = (x0 - x2)*C0_382;
        x0 = DCT_DESCALE(x0*C0_541 + x1, fixb);
        x2 = DCT_DESCALE(x2*C1_306 + x1, fixb);

        x1 = x0 + x3; x3 -= x0;
        x0 = x4 + x2; x4 -= x2;

        work[5] = x1; work[1] = x0;
        work[7] = x4; work[3] = x3;
    }

    work = workspace;
    // pass 2: process columns
    for( i = 8; i > 0; i--, work++, postscale += 8, dst += 8 )
    {
        int  x0 = work[8*0], x1 = work[8*7];
        int  x2 = work[8*3], x3 = work[8*4];

        int  x4 = x0 + x1; x0 -= x1;
        x1 = x2 + x3; x2 -= x3;

        work[8*7] = x0; work[8*0] = x2;
        x2 = x4 + x1; x4 -= x1;

        x0 = work[8*1]; x3 = work[8*6];
        x1 = x0 + x3; x0 -= x3;
        work[8*4] = x0;

        x0 = work[8*2]; x3 = work[8*5];
        work[8*3] = x0 - x3; x0 += x3;

        x3 = x0 + x1; x0 -= x1;
        x1 = x2 + x3; x2 -= x3;

        dst[0] = (short)DCT_DESCALE(x1*postscale[0], postshift);
        dst[4] = (short)DCT_DESCALE(x2*postscale[4], postshift);

        x0 = DCT_DESCALE((x0 - x4)*C0_707, fixb);
        x1 = x4 + x0; x4 -= x0;

        dst[2] = (short)DCT_DESCALE(x4*postscale[2], postshift);
        dst[6] = (short)DCT_DESCALE(x1*postscale[6], postshift);

        x0 = work[8*0]; x1 = work[8*3];
        x2 = work[8*4]; x3 = work[8*7];

        x0 += x1; x1 += x2; x2 += x3;
        x1 = DCT_DESCALE(x1*C0_707, fixb);

        x4 = x1 + x3; x3 -= x1;
        x1 = (x0 - x2)*C0_382;
        x0 = DCT_DESCALE(x0*C0_541 + x1, fixb);
        x2 = DCT_DESCALE(x2*C1_306 + x1, fixb);

        x1 = x0 + x3; x3 -= x0;
        x0 = x4 + x2; x4 -= x2;

        dst[5] = (short)DCT_DESCALE(x1*postscale[5], postshift);
        dst[1] = (short)DCT_DESCALE(x0*postscale[1], postshift);
        dst[7] = (short)DCT_DESCALE(x4*postscale[7], postshift);
        dst[3] = (short)DCT_DESCALE(x3*postscale[3], postshift);
    }
}
#endif


inline void convertToYUV(int colorspace, int channels, int input_channels, short* UV_data, short* Y_data, const uchar* pix_data, int y_limit, int x_limit, int step, int u_plane_ofs, int v_plane_ofs)
{
    int i, j;
    const int UV_step = 16;
    int  x_scale = channels > 1 ? 2 : 1, y_scale = x_scale;
    int  Y_step = x_scale*8;

    if( channels > 1 )
    {
        if( colorspace == COLORSPACE_YUV444P && y_limit == 16 && x_limit == 16 )
        {
            for( i = 0; i < y_limit; i += 2, pix_data += step*2, Y_data += Y_step*2, UV_data += UV_step )
            {
#ifdef WITH_NEON
                {
                    uint16x8_t masklo = vdupq_n_u16(255);
                    uint16x8_t lane = vld1q_u16((unsigned short*)(pix_data+v_plane_ofs));
                    uint16x8_t t1 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    lane = vld1q_u16((unsigned short*)(pix_data + v_plane_ofs + step));
                    uint16x8_t t2 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    t1 = vaddq_u16(t1, t2);
                    vst1q_s16(UV_data, vsubq_s16(vreinterpretq_s16_u16(t1), vdupq_n_s16(128*4)));

                    lane = vld1q_u16((unsigned short*)(pix_data+u_plane_ofs));
                    t1 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    lane = vld1q_u16((unsigned short*)(pix_data + u_plane_ofs + step));
                    t2 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    t1 = vaddq_u16(t1, t2);
                    vst1q_s16(UV_data + 8, vsubq_s16(vreinterpretq_s16_u16(t1), vdupq_n_s16(128*4)));
                }

                {
                    int16x8_t lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data)));
                    int16x8_t delta = vdupq_n_s16(128);
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data, lane);

                    lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data+8)));
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data + 8, lane);

                    lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data+step)));
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data+Y_step, lane);

                    lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data + step + 8)));
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data+Y_step + 8, lane);
                }
#else
                for( j = 0; j < x_limit; j += 2, pix_data += 2 )
                {
                    Y_data[j] = pix_data[0] - 128;
                    Y_data[j+1] = pix_data[1] - 128;
                    Y_data[j+Y_step] = pix_data[step] - 128;
                    Y_data[j+Y_step+1] = pix_data[step+1] - 128;

                    UV_data[j>>1] = pix_data[v_plane_ofs] + pix_data[v_plane_ofs+1] +
                        pix_data[v_plane_ofs+step] + pix_data[v_plane_ofs+step+1] - 128*4;
                    UV_data[(j>>1)+8] = pix_data[u_plane_ofs] + pix_data[u_plane_ofs+1] +
                        pix_data[u_plane_ofs+step] + pix_data[u_plane_ofs+step+1] - 128*4;

                }

                pix_data -= x_limit*input_channels;
#endif
            }
        }
        else
        {
            for( i = 0; i < y_limit; i++, pix_data += step, Y_data += Y_step )
            {
                for( j = 0; j < x_limit; j++, pix_data += input_channels )
                {
                    int Y, U, V;

                    if( colorspace == COLORSPACE_BGR )
                    {
                        int r = pix_data[2];
                        int g = pix_data[1];
                        int b = pix_data[0];

                        Y = DCT_DESCALE( r*y_r + g*y_g + b*y_b, fixc) - 128;
                        U = DCT_DESCALE( r*cb_r + g*cb_g + b*cb_b, fixc );
                        V = DCT_DESCALE( r*cr_r + g*cr_g + b*cr_b, fixc );
                    }
                    else if( colorspace == COLORSPACE_RGBA )
                    {
                        int r = pix_data[0];
                        int g = pix_data[1];
                        int b = pix_data[2];

                        Y = DCT_DESCALE( r*y_r + g*y_g + b*y_b, fixc) - 128;
                        U = DCT_DESCALE( r*cb_r + g*cb_g + b*cb_b, fixc );
                        V = DCT_DESCALE( r*cr_r + g*cr_g + b*cr_b, fixc );
                    }
                    else
                    {
                        Y = pix_data[0] - 128;
                        U = pix_data[v_plane_ofs] - 128;
                        V = pix_data[u_plane_ofs] - 128;
                    }

                    int j2 = j >> (x_scale - 1);
                    Y_data[j] = (short)Y;
                    UV_data[j2] = (short)(UV_data[j2] + U);
                    UV_data[j2 + 8] = (short)(UV_data[j2 + 8] + V);
                }

                pix_data -= x_limit*input_channels;
                if( ((i+1) & (y_scale - 1)) == 0 )
                {
                    UV_data += UV_step;
                }
            }
        }

    }
    else
    {
        for( i = 0; i < y_limit; i++, pix_data += step, Y_data += Y_step )
        {
            for( j = 0; j < x_limit; j++ )
                Y_data[j] = (short)(pix_data[j]*4 - 128*4);
        }
    }
}

class MjpegEncoder : public ParallelLoopBody
{
public:
    MjpegEncoder(int _height,
        int _width,
        int _step,
        const uchar* _data,
        int _input_channels,
        int _channels,
        int _colorspace,
        unsigned (&_huff_dc_tab)[2][16],
        unsigned (&_huff_ac_tab)[2][256],
        short (&_fdct_qtab)[2][64],
        uchar* _cat_table,
        mjpeg_buffer_keeper& _buffer_list,
        double nstripes
    ) :
        m_buffer_list(_buffer_list),
        height(_height),
        width(_width),
        step(_step),
        in_data(_data),
        input_channels(_input_channels),
        channels(_channels),
        colorspace(_colorspace),
        huff_dc_tab(_huff_dc_tab),
        huff_ac_tab(_huff_ac_tab),
        fdct_qtab(_fdct_qtab),
        cat_table(_cat_table)
    {
        //empirically found value. if number of pixels is less than that value there is no sense to parallelize it.
        const int min_pixels_count = 96*96;

        stripes_count = 1;

        if(nstripes < 0)
        {
            if(height*width > min_pixels_count)
            {
                const int default_stripes_count = 4;
                stripes_count = default_stripes_count;
            }
        }
        else
        {
            stripes_count = cvCeil(nstripes);
        }

        int y_scale = channels > 1 ? 2 : 1;
        int y_step = y_scale * 8;

        int max_stripes = (height - 1)/y_step + 1;

        stripes_count = std::min(stripes_count, max_stripes);

        m_buffer_list.allocate_buffers(stripes_count, (height*width*2)/stripes_count);
    }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        const int CAT_TAB_SIZE = 4096;

        int x, y;
        int i, j;

        short  buffer[4096];
        int  x_scale = channels > 1 ? 2 : 1, y_scale = x_scale;
        int  dc_pred[] = { 0, 0, 0 };
        int  x_step = x_scale * 8;
        int  y_step = y_scale * 8;
        short  block[6][64];
        int  luma_count = x_scale*y_scale;
        int  block_count = luma_count + channels - 1;
        int u_plane_ofs = step*height;
        int v_plane_ofs = u_plane_ofs + step*height;
        const uchar* data = in_data;
        const uchar* init_data = data;

        int num_steps = (height - 1)/y_step + 1;

        //if this is not first stripe we need to calculate dc_pred from previous step
        if(range.start > 0)
        {
            y = y_step*int(num_steps*range.start/stripes_count - 1);
            data = init_data + y*step;

            for( x = 0; x < width; x += x_step )
            {
                int x_limit = x_step;
                int y_limit = y_step;
                const uchar* pix_data = data + x*input_channels;
                short* Y_data = block[0];
                short* UV_data = block[luma_count];

                if( x + x_limit > width ) x_limit = width - x;
                if( y + y_limit > height ) y_limit = height - y;

                memset( block, 0, block_count*64*sizeof(block[0][0]));

                convertToYUV(colorspace, channels, input_channels, UV_data, Y_data, pix_data, y_limit, x_limit, step, u_plane_ofs, v_plane_ofs);

                for( i = 0; i < block_count; i++ )
                {
                    int is_chroma = i >= luma_count;
                    int src_step = x_scale * 8;
                    const short* src_ptr = block[i & -2] + (i & 1)*8;

                    aan_fdct8x8( src_ptr, buffer, src_step, fdct_qtab[is_chroma] );

                    j = is_chroma + (i > luma_count);
                    dc_pred[j] = buffer[0];
                }
            }
        }

        for(int k = range.start; k < range.end; ++k)
        {
            mjpeg_buffer& output_buffer = m_buffer_list[k];
            output_buffer.clear();

            int y_min = y_step*int(num_steps*k/stripes_count);
            int y_max = y_step*int(num_steps*(k+1)/stripes_count);

            if(k == stripes_count - 1)
            {
                y_max = height;
            }


            data = init_data + y_min*step;

            for( y = y_min; y < y_max; y += y_step, data += y_step*step )
            {
                for( x = 0; x < width; x += x_step )
                {
                    int x_limit = x_step;
                    int y_limit = y_step;
                    const uchar* pix_data = data + x*input_channels;
                    short* Y_data = block[0];
                    short* UV_data = block[luma_count];

                    if( x + x_limit > width ) x_limit = width - x;
                    if( y + y_limit > height ) y_limit = height - y;

                    memset( block, 0, block_count*64*sizeof(block[0][0]));

                    convertToYUV(colorspace, channels, input_channels, UV_data, Y_data, pix_data, y_limit, x_limit, step, u_plane_ofs, v_plane_ofs);

                    for( i = 0; i < block_count; i++ )
                    {
                        int is_chroma = i >= luma_count;
                        int src_step = x_scale * 8;
                        int run = 0, val;
                        const short* src_ptr = block[i & -2] + (i & 1)*8;
                        const unsigned* htable = huff_ac_tab[is_chroma];

                        aan_fdct8x8( src_ptr, buffer, src_step, fdct_qtab[is_chroma] );

                        j = is_chroma + (i > luma_count);
                        val = buffer[0] - dc_pred[j];
                        dc_pred[j] = buffer[0];

                        {
                            int cat = cat_table[val + CAT_TAB_SIZE];

                            //CV_Assert( cat <= 11 );
                            output_buffer.put_val(cat, huff_dc_tab[is_chroma] );
                            output_buffer.put_bits( val - (val < 0 ? 1 : 0), cat );
                        }

                        for( j = 1; j < 64; j++ )
                        {
                            val = buffer[zigzag[j]];

                            if( val == 0 )
                            {
                                run++;
                            }
                            else
                            {
                                while( run >= 16 )
                                {
                                    output_buffer.put_val( 0xF0, htable ); // encode 16 zeros
                                    run -= 16;
                                }

                                {
                                    int cat = cat_table[val + CAT_TAB_SIZE];
                                    //CV_Assert( cat <= 10 );
                                    output_buffer.put_val( cat + run*16, htable );
                                    output_buffer.put_bits( val - (val < 0 ? 1 : 0), cat );
                                }

                                run = 0;
                            }
                        }

                        if( run )
                        {
                            output_buffer.put_val( 0x00, htable ); // encode EOB
                        }
                    }
                }
            }
        }
    }

    cv::Range getRange()
    {
        return cv::Range(0, stripes_count);
    }

    double getNStripes()
    {
        return stripes_count;
    }

    mjpeg_buffer_keeper& m_buffer_list;
private:

    MjpegEncoder& operator=( const MjpegEncoder & ) { return *this; }

    const int height;
    const int width;
    const int step;
    const uchar* in_data;
    const int input_channels;
    const int channels;
    const int colorspace;
    const unsigned (&huff_dc_tab)[2][16];
    const unsigned (&huff_ac_tab)[2][256];
    const short (&fdct_qtab)[2][64];
    const uchar* cat_table;
    int stripes_count;
};

void MotionJpegWriter::writeFrameData( const uchar* data, int step, int colorspace, int input_channels )
{
    //double total_cvt = 0, total_dct = 0;
    static bool init_cat_table = false;
    const int CAT_TAB_SIZE = 4096;
    static uchar cat_table[CAT_TAB_SIZE*2+1];
    if( !init_cat_table )
    {
        for( int i = -CAT_TAB_SIZE; i <= CAT_TAB_SIZE; i++ )
        {
            Cv32suf a;
            a.f = (float)i;
            cat_table[i+CAT_TAB_SIZE] = ((a.i >> 23) & 255) - (126 & (i ? -1 : 0));
        }
        init_cat_table = true;
    }

    //double total_dct = 0, total_cvt = 0;
    int width = container.getWidth();
    int height = container.getHeight();
    int channels = container.getChannels();

    CV_Assert( data && width > 0 && height > 0 );

    // encode the header and tables
    // for each mcu:
    //   convert rgb to yuv with downsampling (if color).
    //   for every block:
    //     calc dct and quantize
    //     encode block.
    int i, j;
    const int max_quality = 12;
    short fdct_qtab[2][64];
    unsigned huff_dc_tab[2][16];
    unsigned huff_ac_tab[2][256];

    int  x_scale = channels > 1 ? 2 : 1, y_scale = x_scale;
    short  buffer[4096];
    int*   hbuffer = (int*)buffer;
    int  luma_count = x_scale*y_scale;
    double _quality = quality*0.01*max_quality;

    if( _quality < 1. ) _quality = 1.;
    if( _quality > max_quality ) _quality = max_quality;

    double inv_quality = 1./_quality;

    // Encode header
    container.putStreamBytes( (const uchar*)jpegHeader, sizeof(jpegHeader) - 1 );

    // Encode quantization tables
    for( i = 0; i < (channels > 1 ? 2 : 1); i++ )
    {
        const uchar* qtable = i == 0 ? jpegTableK1_T : jpegTableK2_T;
        int chroma_scale = i > 0 ? luma_count : 1;

        container.jputStreamShort( 0xffdb );   // DQT marker
        container.jputStreamShort( 2 + 65*1 ); // put single qtable
        container.putStreamByte( 0*16 + i );   // 8-bit table

        // put coefficients
        for( j = 0; j < 64; j++ )
        {
            int idx = zigzag[j];
            int qval = cvRound(qtable[idx]*inv_quality);
            if( qval < 1 )
                qval = 1;
            if( qval > 255 )
                qval = 255;
            fdct_qtab[i][idx] = (short)(cvRound((1 << (postshift + 11)))/
                                (qval*chroma_scale*idct_prescale[idx]));
            container.putStreamByte( qval );
        }
    }

    // Encode huffman tables
    for( i = 0; i < (channels > 1 ? 4 : 2); i++ )
    {
        const uchar* htable = i == 0 ? jpegTableK3 : i == 1 ? jpegTableK5 :
        i == 2 ? jpegTableK4 : jpegTableK6;
        int is_ac_tab = i & 1;
        int idx = i >= 2;
        int tableSize = 16 + (is_ac_tab ? 162 : 12);

        container.jputStreamShort( 0xFFC4 );      // DHT marker
        container.jputStreamShort( 3 + tableSize ); // define one huffman table
        container.putStreamByte( is_ac_tab*16 + idx ); // put DC/AC flag and table index
        container.putStreamBytes( htable, tableSize ); // put table

        createEncodeHuffmanTable(createSourceHuffmanTable( htable, hbuffer, 16, 9 ),
                                 is_ac_tab ? huff_ac_tab[idx] : huff_dc_tab[idx],
                                 is_ac_tab ? 256 : 16 );
    }

    // put frame header
    container.jputStreamShort( 0xFFC0 );          // SOF0 marker
    container.jputStreamShort( 8 + 3*channels );  // length of frame header
    container.putStreamByte( 8 );               // sample precision
    container.jputStreamShort( height );
    container.jputStreamShort( width );
    container.putStreamByte( channels );        // number of components

    for( i = 0; i < channels; i++ )
    {
        container.putStreamByte( i + 1 );  // (i+1)-th component id (Y,U or V)
        if( i == 0 )
            container.putStreamByte(x_scale*16 + y_scale); // chroma scale factors
        else
            container.putStreamByte(1*16 + 1);
        container.putStreamByte( i > 0 ); // quantization table idx
    }

    // put scan header
    container.jputStreamShort( 0xFFDA );          // SOS marker
    container.jputStreamShort( 6 + 2*channels );  // length of scan header
    container.putStreamByte( channels );          // number of components in the scan

    for( i = 0; i < channels; i++ )
    {
        container.putStreamByte( i+1 );             // component id
        container.putStreamByte( (i>0)*16 + (i>0) );// selection of DC & AC tables
    }

    container.jputStreamShort(0*256 + 63); // start and end of spectral selection - for
    // sequential DCT start is 0 and end is 63

    container.putStreamByte( 0 );  // successive approximation bit position
    // high & low - (0,0) for sequential DCT

    buffers_list.reset();

    MjpegEncoder parallel_encoder(height, width, step, data, input_channels, channels, colorspace, huff_dc_tab, huff_ac_tab, fdct_qtab, cat_table, buffers_list, nstripes);

    cv::parallel_for_(parallel_encoder.getRange(), parallel_encoder, parallel_encoder.getNStripes());

    //std::vector<unsigned>& v = parallel_encoder.m_buffer_list.get_data();
    unsigned* v = buffers_list.get_data();
    unsigned last_data_elem = buffers_list.get_data_size() - 1;

    for(unsigned k = 0; k < last_data_elem; ++k)
    {
        container.jputStream(v[k]);
    }
    container.jflushStream(v[last_data_elem], 32 - buffers_list.get_last_bit_len());
    container.jputStreamShort( 0xFFD9 ); // EOI marker
    /*printf("total dct = %.1fms, total cvt = %.1fms\n",
     total_dct*1000./cv::getTickFrequency(),
     total_cvt*1000./cv::getTickFrequency());*/

    size_t pos = container.getStreamPos();
    size_t pos1 = (pos + 3) & ~3;
    for( ; pos < pos1; pos++ )
        container.putStreamByte(0);
}

}

Ptr<IVideoWriter> createMotionJpegWriter(const std::string& filename, int fourcc,
                                         double fps, const Size& frameSize,
                                         const VideoWriterParameters& params)
{
    if (fourcc != CV_FOURCC('M', 'J', 'P', 'G'))
        return Ptr<IVideoWriter>();

    const bool isColor = params.get(VIDEOWRITER_PROP_IS_COLOR, true);
    Ptr<IVideoWriter> iwriter = makePtr<mjpeg::MotionJpegWriter>(filename, fps, frameSize, isColor);
    if( !iwriter->isOpened() )
        iwriter.release();
    return iwriter;
}

}
