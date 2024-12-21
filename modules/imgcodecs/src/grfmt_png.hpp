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
#include "apngframe.hpp"
#include "bitstrm.hpp"
#include <png.h>
#include <zlib.h>

namespace cv
{

struct CHUNK { unsigned char* p; uint32_t size; };
struct OP { unsigned char* p; uint32_t size; int x, y, w, h, valid, filters; };

class PngDecoder CV_FINAL : public BaseImageDecoder
{
public:

    PngDecoder();
    virtual ~PngDecoder();

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();
    bool  nextPage() CV_OVERRIDE;

    ImageDecoder newDecoder() const CV_OVERRIDE;
    APNGFrame frameCur;

protected:

    static void readDataFromBuf(void* png_ptr, uchar* dst, size_t size);
    static void info_fn(png_structp png_ptr, png_infop info_ptr);
    static void row_fn(png_structp png_ptr, png_bytep new_row, png_uint_32 row_num, int pass);
    bool readAnimation(Mat& img);
    bool processing_start(void* frame_ptr, const Mat& img);
    bool processing_finish();
    void compose_frame(unsigned char** rows_dst, unsigned char** rows_src, unsigned char bop, uint32_t x, uint32_t y, uint32_t w, uint32_t h, int channels);
    size_t read_from_io(void* _Buffer, size_t _ElementSize, size_t _ElementCount);
    uint32_t  read_chunk(CHUNK& pChunk);

    int   m_bit_depth;
    void* m_png_ptr;  // pointer to decompression structure
    void* m_info_ptr; // pointer to image information structure
    void* m_end_info; // pointer to one more image information structure
    FILE* m_f;
    int   m_color_type;
    bool  m_is_animated;
    CHUNK m_chunkIHDR;
    int   m_frame_no;
    size_t m_buf_pos;
    std::vector<CHUNK> m_chunksInfo;
    APNGFrame frameRaw;
    APNGFrame frameNext;
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
    bool writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params) CV_OVERRIDE;
    bool writeanimation(const Animation& animinfo, const std::vector<int>& params) CV_OVERRIDE;
    void optim_dirty(std::vector<APNGFrame>& frames);
    void optim_duplicates(std::vector<APNGFrame>& frames, uint32_t first);

    ImageEncoder newEncoder() const CV_OVERRIDE;

protected:
    static void writeDataToBuf(void* png_ptr, unsigned char* src, size_t size);
    static void flushBuf(void* png_ptr);
    size_t write_to_io(void const* _Buffer, size_t  _ElementSize, size_t _ElementCount, FILE* _Stream);

private:
    void write_chunk(FILE* f, const char* name, unsigned char* data, uint32_t length);
    void write_IDATs(FILE* f, int frame, unsigned char* data, uint32_t length, uint32_t idat_size);
    void process_rect(unsigned char* row, int rowbytes, int bpp, int stride, int h, unsigned char* rows);
    void deflate_rect_fin(int deflate_method, int iter, unsigned char* zbuf, uint32_t* zsize, int bpp, int stride, unsigned char* rows, int zbuf_size, int n);
    void deflate_rect_op(unsigned char* pdata, int x, int y, int w, int h, int bpp, int stride, int zbuf_size, int n);
    void get_rect(uint32_t w, uint32_t h, unsigned char* pimage1, unsigned char* pimage2, unsigned char* ptemp, uint32_t bpp, uint32_t stride, int zbuf_size, uint32_t has_tcolor, uint32_t tcolor, int n);

    unsigned char* op_zbuf1;
    unsigned char* op_zbuf2;
    z_stream       op_zstream1;
    z_stream       op_zstream2;
    unsigned char* row_buf;
    unsigned char* sub_row;
    unsigned char* up_row;
    unsigned char* avg_row;
    unsigned char* paeth_row;
    OP             op[6];
    rgb            palette[256];
    unsigned char  trns[256];
    uint32_t       palsize, trnssize;
    uint32_t       next_seq_num;
};

}

#endif

#endif/*_GRFMT_PNG_H_*/
