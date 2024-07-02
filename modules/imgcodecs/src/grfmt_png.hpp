// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

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

struct CHUNK { uchar* p; uint size; };
struct OP { uchar* p; uint size; int x, y, w, h, valid, filters; };

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

protected:
    int          processing_start(png_structp& png_ptr, png_infop& info_ptr, void* frame_ptr, bool hasInfo, CHUNK& chunkIHDR, std::vector<CHUNK>& chunksInfo);
    int          processing_data(png_structp png_ptr, png_infop info_ptr, uchar* p, uint size);
    int          processing_finish(png_structp png_ptr, png_infop info_ptr);
    void         compose_frame(uchar** rows_dst, uchar** rows_src, uchar bop, uint x, uint y, uint w, uint h);
    int          load_apng(std::string inputFileName, std::vector<APNGFrame>& frames, uint& first, uint& loops);
    static void  readDataFromBuf(void* png_ptr, uchar* dst, size_t size);
    static uint  read_chunk(FILE* f, CHUNK* pChunk);

    int    m_bit_depth;
    void*  m_png_ptr;  // pointer to decompression structure
    void*  m_info_ptr; // pointer to image information structure
    void*  m_end_info; // pointer to one more image information structure
    FILE*  m_f;
    int    m_color_type;
    size_t m_buf_pos;
    bool   m_is_animated;
    int    m_loops;
};


class PngEncoder CV_FINAL : public BaseImageEncoder
{
public:
    PngEncoder();
    virtual ~PngEncoder();

    bool  isFormatSupported( int depth ) const CV_OVERRIDE;
    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;
    bool  writemulti(const std::vector<Mat>& img_vec, const std::vector<int>& params) CV_OVERRIDE;
    size_t save_apng(std::string inputFileName, std::vector<APNGFrame>& frames, uint first, uint loops, uint coltype, int deflate_method, int iter);
    void  optim_dirty(std::vector<APNGFrame>& frames);
    void  optim_duplicates(std::vector<APNGFrame>& frames, uint first);

    ImageEncoder newEncoder() const CV_OVERRIDE;

protected:
    static void writeDataToBuf(void* png_ptr, uchar* src, size_t size);
    static void flushBuf(void* png_ptr);

private:
    void write_chunk(FILE* f, const char* name, uchar* data, uint length);
    void write_IDATs(FILE* f, int frame, uchar* data, uint length, uint idat_size);
    void process_rect(uchar* row, int rowbytes, int bpp, int stride, int h, uchar* rows);
    void deflate_rect_fin(int deflate_method, int iter, uchar* zbuf, uint* zsize, int bpp, int stride, uchar* rows, int zbuf_size, int n);
    void deflate_rect_op(uchar* pdata, int x, int y, int w, int h, int bpp, int stride, int zbuf_size, int n);
    void get_rect(uint w, uint h, uchar* pimage1, uchar* pimage2, uchar* ptemp, uint bpp, uint stride, int zbuf_size, uint has_tcolor, uint tcolor, int n);

    void (*process_callback)(float);
    uchar*         op_zbuf1;
    uchar*         op_zbuf2;
    z_stream       op_zstream1;
    z_stream       op_zstream2;
    uchar*         row_buf;
    uchar*         sub_row;
    uchar*         up_row;
    uchar*         avg_row;
    uchar*         paeth_row;
    OP             op[6];
    rgb            palette[256];
    uchar          trns[256];
    uint   palsize, trnssize;
    uint   next_seq_num;
};

}

#endif

#endif/*_GRFMT_PNG_H_*/
