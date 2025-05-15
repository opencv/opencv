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

#ifndef _GRFMT_PNG_H_
#define _GRFMT_PNG_H_

#ifdef HAVE_PNG

#include "grfmt_base.hpp"
#include "bitstrm.hpp"
#include <png.h>
#include <zlib.h>
#include <vector>

namespace cv
{

struct Chunk { std::vector<unsigned char> p; };
struct OP { unsigned char* p; uint32_t size; int x, y, w, h, valid, filters; };

typedef struct {
  unsigned char r, g, b;
} rgb;

class APNGFrame {
public:

    APNGFrame();

    // Destructor
    ~APNGFrame();

    bool setMat(const cv::Mat& src, unsigned delayNum = 1, unsigned delayDen = 1000);

    // Getters and Setters
    unsigned char* getPixels() const { return _pixels; }
    void setPixels(unsigned char* pixels);

    unsigned int getWidth() const { return _width; }
    void setWidth(unsigned int width);

    unsigned int getHeight() const { return _height; }
    void setHeight(unsigned int height);

    unsigned char getColorType() const { return _colorType; }
    void setColorType(unsigned char colorType);

    rgb* getPalette() { return _palette; }
    void setPalette(const rgb* palette);

    unsigned char* getTransparency() { return _transparency; }
    void setTransparency(const unsigned char* transparency);

    int getPaletteSize() const { return _paletteSize; }
    void setPaletteSize(int paletteSize);

    int getTransparencySize() const { return _transparencySize; }
    void setTransparencySize(int transparencySize);

    unsigned int getDelayNum() const { return _delayNum; }
    void setDelayNum(unsigned int delayNum);

    unsigned int getDelayDen() const { return _delayDen; }
    void setDelayDen(unsigned int delayDen);

    std::vector<png_bytep>& getRows() { return _rows; }

private:
    unsigned char* _pixels;
    unsigned int _width;
    unsigned int _height;
    unsigned char _colorType;
    rgb _palette[256];
    unsigned char _transparency[256];
    int _paletteSize;
    int _transparencySize;
    unsigned int _delayNum;
    unsigned int _delayDen;
    std::vector<png_bytep> _rows;
};

class PngDecoder CV_FINAL : public BaseImageDecoder
{
public:
    PngDecoder();
    virtual ~PngDecoder();

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    bool  nextPage() CV_OVERRIDE;

    ImageDecoder newDecoder() const CV_OVERRIDE;

private:
    static void readDataFromBuf(void* png_ptr, uchar* dst, size_t size);
    static void info_fn(png_structp png_ptr, png_infop info_ptr);
    static void row_fn(png_structp png_ptr, png_bytep new_row, png_uint_32 row_num, int pass);
    CV_NODISCARD_STD bool processing_start(void* frame_ptr, const Mat& img);
    CV_NODISCARD_STD bool processing_finish();
    void compose_frame(std::vector<png_bytep>& rows_dst, const std::vector<png_bytep>& rows_src, unsigned char bop, uint32_t x, uint32_t y, uint32_t w, uint32_t h, Mat& img);
    /**
     * @brief Reads data from an I/O source into the provided buffer.
     * @param buffer Pointer to the buffer where the data will be stored.
     * @param num_bytes Number of bytes to read into the buffer.
     * @return true if the operation is successful, false otherwise.
     */
    CV_NODISCARD_STD bool readFromStreamOrBuffer(void* buffer, size_t num_bytes);
    uint32_t  read_chunk(Chunk& chunk);
    CV_NODISCARD_STD bool InitPngPtr();
    void ClearPngPtr();

    png_structp m_png_ptr = nullptr; // pointer to decompression structure
    png_infop m_info_ptr = nullptr; // pointer to image information structure
    png_infop m_end_info = nullptr; // pointer to one more image information structure
    int   m_bit_depth;
    FILE* m_f;
    int   m_color_type;
    Chunk m_chunkIHDR;
    int   m_frame_no;
    size_t m_buf_pos;
    std::vector<Chunk> m_chunksInfo;
    APNGFrame frameRaw;
    APNGFrame frameNext;
    APNGFrame frameCur;
    Mat m_mat_raw;
    Mat m_mat_next;
    uint32_t w0;
    uint32_t h0;
    uint32_t x0;
    uint32_t y0;
    uint32_t delay_num;
    uint32_t delay_den;
    uint32_t dop;
    uint32_t bop;
    bool m_is_fcTL_loaded;
    bool m_is_IDAT_loaded;
};


class PngEncoder CV_FINAL : public BaseImageEncoder
{
public:
    PngEncoder();
    virtual ~PngEncoder();

    bool isFormatSupported( int depth ) const CV_OVERRIDE;
    bool write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    bool writeanimation(const Animation& animinfo, const std::vector<int>& params) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE;

protected:
    static void writeDataToBuf(void* png_ptr, unsigned char* src, size_t size);
    static void flushBuf(void* png_ptr);
    /**
    * @brief Writes data to an output destination, either a file stream or an in-memory buffer.
    *
    * This function handles two output scenarios:
    * 1. If a file stream is provided, the data is written to the stream using `fwrite`.
    * 2. If `stream` is null, the data is written to an in-memory buffer (`m_buf`), which is resized as needed.
    *
    * @param buffer Pointer to the data to be written.
    * @param num_bytes The number of bytes to be written.
    * @param stream Pointer to the file stream for writing. If null, the data is written to the in-memory buffer.
    * @return The number of bytes successfully written.
    *         - For file-based writes, this is the number of bytes written to the stream.
    *         - For buffer-based writes, this is the total number of bytes added to the buffer.
    *
    * @throws std::runtime_error If the in-memory buffer (`m_buf`) exceeds its maximum capacity.
    * @note If `num_bytes` is 0 or `buffer` is null, the function returns 0.
    */
    size_t writeToStreamOrBuffer(void const* buffer, size_t  num_bytes, FILE* stream);

private:
    void writeChunk(FILE* f, const char* name, unsigned char* data, uint32_t length);
    void writeIDATs(FILE* f, int frame, unsigned char* data, uint32_t length, uint32_t idat_size);
    void processRect(unsigned char* row, int rowbytes, int bpp, int stride, int h, unsigned char* rows);
    void deflateRectFin(unsigned char* zbuf, uint32_t* zsize, int bpp, int stride, unsigned char* rows, int zbuf_size, int n);
    void deflateRectOp(unsigned char* pdata, int x, int y, int w, int h, int bpp, int stride, int zbuf_size, int n);
    bool getRect(uint32_t w, uint32_t h, unsigned char* pimage1, unsigned char* pimage2, unsigned char* ptemp, uint32_t bpp, uint32_t stride, int zbuf_size, uint32_t has_tcolor, uint32_t tcolor, int n);

    AutoBuffer<unsigned char> op_zbuf1;
    AutoBuffer<unsigned char> op_zbuf2;
    AutoBuffer<unsigned char> row_buf;
    AutoBuffer<unsigned char> sub_row;
    AutoBuffer<unsigned char> up_row;
    AutoBuffer<unsigned char> avg_row;
    AutoBuffer<unsigned char> paeth_row;
    z_stream       op_zstream1;
    z_stream       op_zstream2;
    OP             op[6];
    rgb            palette[256];
    unsigned char  trns[256];
    uint32_t       palsize, trnssize;
    uint32_t       next_seq_num;
    int            m_compression_level;
    int            m_compression_strategy;
    int            m_filter;
    bool           m_isBilevel;
};

}

#endif

#endif/*_GRFMT_PNG_H_*/
