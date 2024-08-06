// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_GRFMT_GIF_HPP
#define OPENCV_GRFMT_GIF_HPP
#ifdef HAVE_IMGCODEC_GIF

#include "grfmt_base.hpp"

namespace cv
{

enum GifOpMode
{
    GRFMT_GIF_Nothing = 0,
    GRFMT_GIF_PreviousImage = 1,
    GRFMT_GIF_Background = 2,
    GRFMT_GIF_Cover = 3
};

//////////////////////////////////////////////////////////////////////
////                        GIF Decoder                           ////
//////////////////////////////////////////////////////////////////////

class GifDecoder CV_FINAL : public BaseImageDecoder
{
public:
    GifDecoder();
    ~GifDecoder() CV_OVERRIDE;

    bool readHeader() CV_OVERRIDE;
    bool readData(Mat& img) CV_OVERRIDE;
    bool nextPage() CV_OVERRIDE;
    void close();

    ImageDecoder newDecoder() const CV_OVERRIDE;

protected:
    RLByteStream        m_strm;

    int                 bgColor;
    int                 depth;
    int                 idx;

    GifOpMode           opMode;
    bool                hasTransparentColor;
    uchar               transparentColor;
    int                 top, left, width, height;

    bool                hasRead;
    AutoBuffer<uchar>   globalColorTable;
    AutoBuffer<uchar>   localColorTable;

    int                 lzwMinCodeSize;
    int                 globalColorTableSize;
    int                 localColorTableSize;

    Mat                 lastImage;
    AutoBuffer<uchar>   imgCodeStream;

    struct lzwNodeD
    {
        int   length;
        uchar suffix;
        std::vector<uchar> prefix;
    };

    void readExtensions();
    void code2pixel(Mat& img, int start, int k);
    bool lzwDecode();
    bool getFrameCount_();
    bool skipHeader();
};
} // namespace cv

#endif // HAVE_IMGCODEC_GIF
#endif //OPENCV_GRFMT_GIF_HPP
