// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_GRFMT_GIF_HPP
#define OPENCV_GRFMT_GIF_HPP
#ifdef HAVE_IMGCODEC_GIF

#include "grfmt_base.hpp"

namespace cv
{

// See https://www.w3.org/Graphics/GIF/spec-gif89a.txt
// 23. Graphic Control Extension.
// <Packed Fields>
//  Reserved               : 3 bits
//  Disposal Method        : 3 bits
//  User Input Flag        : 1 bit
//  Transparent Color Flag : 1 bit
constexpr int GIF_DISPOSE_METHOD_SHIFT = 2;
constexpr int GIF_DISPOSE_METHOD_MASK  = 7; // 0b111
constexpr int GIF_TRANS_COLOR_FLAG_MASK  = 1; // 0b1

enum GifDisposeMethod {
    GIF_DISPOSE_NA                 = 0,
    GIF_DISPOSE_NONE               = 1,
    GIF_DISPOSE_RESTORE_BACKGROUND = 2,
    GIF_DISPOSE_RESTORE_PREVIOUS   = 3,
    // 4-7 are reserved/undefined.

    GIF_DISPOSE_MAX                = GIF_DISPOSE_RESTORE_PREVIOUS,
};

enum GifTransparentColorFlag {
    GIF_TRANSPARENT_INDEX_NOT_GIVEN = 0,
    GIF_TRANSPARENT_INDEX_GIVEN     = 1,

    GIF_TRANSPARENT_INDEX_MAX       = GIF_TRANSPARENT_INDEX_GIVEN,
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

    bool                hasTransparentColor;
    uchar               transparentColor;
    int                 top, left, width, height;

    bool                hasRead;
    std::vector<uchar>  globalColorTable;
    std::vector<uchar>  localColorTable;

    int                 lzwMinCodeSize;
    int                 globalColorTableSize;
    int                 localColorTableSize;

    Mat                 lastImage;
    std::vector<uchar>  imgCodeStream;

    struct lzwNodeD
    {
        int   length;
        uchar suffix;
        std::vector<uchar> prefix;
    };

    GifDisposeMethod readExtensions();
    void code2pixel(Mat& img, int start, int k);
    bool lzwDecode();
    bool getFrameCount_();
    bool skipHeader();
};



//////////////////////////////////////////////////////////////////////
////                        GIF Encoder                           ////
//////////////////////////////////////////////////////////////////////
class GifEncoder CV_FINAL : public BaseImageEncoder {
public:
    GifEncoder();
    ~GifEncoder() CV_OVERRIDE;

    bool writeanimation(const Animation& animation, const std::vector<int>& params) CV_OVERRIDE;

    ImageEncoder newEncoder() const CV_OVERRIDE;

private:
/**  Color Quantization  **/
    class OctreeColorQuant
    {
        struct OctreeNode
        {
            bool  isLeaf;
            std::shared_ptr<OctreeNode> children[8]{};
            int   level;
            uchar index;
            int   leaf;
            int   pixelCount;
            size_t redSum, greenSum, blueSum;

            OctreeNode();
        };

        std::shared_ptr<OctreeNode> root;
        std::vector<std::shared_ptr<OctreeNode>> m_nodeList[8];
        int32_t m_bitLength;
        int32_t m_maxColors;
        int32_t m_leafCount;
        uchar   m_criticalTransparency;
        uchar   r, g, b; // color under transparent color

    public:
        explicit OctreeColorQuant(int maxColors = 256, int bitLength = 8, uchar criticalTransparency = 1);

        int   getPalette(uchar* colorTable);
        uchar getLeaf(uchar red, uchar green, uchar blue);

        void  addMat(const Mat& img);
        void  addMats(const std::vector<Mat>& img_vec);
        void  addColor(int red, int green, int blue);
        void  reduceTree();
        void  recurseReduce(const std::shared_ptr<OctreeNode>& node);
    };

    enum GifDithering // normal dithering level is -1 to 2
    {
        GRFMT_GIF_None = 3,
        GRFMT_GIF_FloydSteinberg = 4
    };

    WLByteStream    strm;
    int             m_width, m_height;

    int             globalColorTableSize;
    int             localColorTableSize;

    uchar           criticalTransparency;
    uchar           transparentColor;
    Vec3b           transparentRGB;
    int             top, left, width, height;

    OctreeColorQuant quantG;
    OctreeColorQuant quantL;

    std::vector<int16_t> lzwTable;
    std::vector<uchar> imgCodeStream;

    std::vector<uchar> globalColorTable;
    std::vector<uchar> localColorTable;

    // params
    int             colorNum;
    int             bitDepth;
    int             dithering;
    int             lzwMinCodeSize, lzwMaxCodeSize;
    bool            fast;

    bool writeFrames(const std::vector<Mat>& img_vec, const std::vector<int>& params);
    bool writeHeader(const std::vector<Mat>& img_vec, const int loopCount);
    bool writeFrame(const Mat& img, const int frameDelay);
    bool pixel2code(const Mat& img);
    void getColorTable(const std::vector<Mat>& img_vec, bool isGlobal);
    static int ditheringKernel(const Mat &img, Mat &img_, int depth, uchar transparency);
    bool lzwEncode();
    void close();
};


} // namespace cv

#endif // HAVE_IMGCODEC_GIF
#endif //OPENCV_GRFMT_GIF_HPP
