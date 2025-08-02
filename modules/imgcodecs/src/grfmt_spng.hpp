// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _GRFMT_SPNG_H_
#define _GRFMT_SPNG_H_

#ifdef HAVE_SPNG

#include "grfmt_base.hpp"
#include "bitstrm.hpp"

namespace cv
{

class SPngDecoder CV_FINAL : public BaseImageDecoder
{
public:

    SPngDecoder();
    virtual ~SPngDecoder();

    bool  readData( Mat& img ) CV_OVERRIDE;
    bool  readHeader() CV_OVERRIDE;
    void  close();

    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:

    static int readDataFromBuf(void* sp_ctx, void *user, void* dst, size_t size);

    int   m_bit_depth;
    void* m_ctx;
    FILE* m_f;
    int   m_color_type;
    size_t m_buf_pos;
};


class SPngEncoder CV_FINAL : public BaseImageEncoder
{
public:
    SPngEncoder();
    virtual ~SPngEncoder();

    bool  isFormatSupported( int depth ) const CV_OVERRIDE;
    bool  isValidParam(const int key, const int value) const CV_OVERRIDE;
    bool  write( const Mat& img, const std::vector<int>& params ) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE;

protected:
    static int writeDataToBuf(void *ctx, void *user, void *dst_src, size_t length);
};

}

#endif

#endif/*_GRFMT_PNG_H_*/
