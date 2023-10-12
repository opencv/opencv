#include "test_precomp.hpp"

namespace opencv_test { namespace {

#undef RGB
#undef YUV

typedef Vec3b YUV;
typedef Vec3b RGB;

int countOfDifferencies(const Mat& gold, const Mat& result, int maxAllowedDifference = 1)
{
    Mat diff;
    absdiff(gold, result, diff);
    return countNonZero(diff.reshape(1) > maxAllowedDifference);
}

class YUVreader
{
public:
    virtual ~YUVreader() {}
    virtual YUV read(const Mat& yuv, int row, int col) = 0;
    virtual int channels() = 0;
    virtual Size size(Size imgSize) = 0;

    virtual bool requiresEvenHeight() { return true; }
    virtual bool requiresEvenWidth() { return true; }

    static YUVreader* getReader(int code);
};

class RGBreader
{
public:
    virtual ~RGBreader() {}
    virtual RGB read(const Mat& rgb, int row, int col) = 0;
    virtual int channels() = 0;

    static RGBreader* getReader(int code);
};

class RGBwriter
{
public:
    virtual ~RGBwriter() {}

    virtual void write(Mat& rgb, int row, int col, const RGB& val) = 0;
    virtual int channels() = 0;

    static RGBwriter* getWriter(int code);
};

class GRAYwriter
{
public:
    virtual ~GRAYwriter() {}

    virtual void write(Mat& gray, int row, int col, const uchar& val)
    {
        gray.at<uchar>(row, col) = val;
    }

    virtual int channels() { return 1; }

    static GRAYwriter* getWriter(int code);
};

class YUVwriter
{
public:
    virtual ~YUVwriter() {}

    virtual void write(Mat& yuv, int row, int col, const YUV& val) = 0;
    virtual int channels() = 0;
    virtual Size size(Size imgSize) = 0;

    virtual bool requiresEvenHeight() { return true; }
    virtual bool requiresEvenWidth() { return true; }

    static YUVwriter* getWriter(int code);
};

class RGB888Writer : public RGBwriter
{
    void write(Mat& rgb, int row, int col, const RGB& val)
    {
        rgb.at<Vec3b>(row, col) = val;
    }

    int channels() { return 3; }
};

class BGR888Writer : public RGBwriter
{
    void write(Mat& rgb, int row, int col, const RGB& val)
    {
        Vec3b tmp(val[2], val[1], val[0]);
        rgb.at<Vec3b>(row, col) = tmp;
    }

    int channels() { return 3; }
};

class RGBA8888Writer : public RGBwriter
{
    void write(Mat& rgb, int row, int col, const RGB& val)
    {
        Vec4b tmp(val[0], val[1], val[2], 255);
        rgb.at<Vec4b>(row, col) = tmp;
    }

    int channels() { return 4; }
};

class BGRA8888Writer : public RGBwriter
{
    void write(Mat& rgb, int row, int col, const RGB& val)
    {
        Vec4b tmp(val[2], val[1], val[0], 255);
        rgb.at<Vec4b>(row, col) = tmp;
    }

    int channels() { return 4; }
};

class YUV420pWriter: public YUVwriter
{
    int channels() { return 1; }
    Size size(Size imgSize) { return Size(imgSize.width, imgSize.height + imgSize.height/2); }
};

class YV12Writer: public YUV420pWriter
{
    void write(Mat& yuv, int row, int col, const YUV& val)
    {
        int h = yuv.rows * 2 / 3;

        yuv.ptr<uchar>(row)[col] = val[0];
        if( row % 2 == 0 && col % 2 == 0 )
        {
            yuv.ptr<uchar>(h + row/4)[col/2 + ((row/2) % 2) * (yuv.cols/2)] = val[2];
            yuv.ptr<uchar>(h + (row/2 + h/2)/2)[col/2 + ((row/2 + h/2) % 2) * (yuv.cols/2)] = val[1];
        }
    }
};

class I420Writer: public YUV420pWriter
{
    void write(Mat& yuv, int row, int col, const YUV& val)
    {
        int h = yuv.rows * 2 / 3;

        yuv.ptr<uchar>(row)[col] = val[0];
        if( row % 2 == 0 && col % 2 == 0 )
        {
            yuv.ptr<uchar>(h + row/4)[col/2 + ((row/2) % 2) * (yuv.cols/2)] = val[1];
            yuv.ptr<uchar>(h + (row/2 + h/2)/2)[col/2 + ((row/2 + h/2) % 2) * (yuv.cols/2)] = val[2];
        }
    }
};

class YUV422Writer: public YUVwriter
{
    int channels() { return 2; }
    Size size(Size imgSize) { return Size(imgSize.width, imgSize.height); }
};

class UYVYWriter: public YUV422Writer
{
    virtual void write(Mat& yuv, int row, int col, const YUV& val)
    {
        yuv.ptr<Vec2b>(row)[col][1] = val[0];
        yuv.ptr<Vec2b>(row)[(col/2)*2][0] = val[1];
        yuv.ptr<Vec2b>(row)[(col/2)*2 + 1][0] = val[2];
    }
};

class YUY2Writer: public YUV422Writer
{
    virtual void write(Mat& yuv, int row, int col, const YUV& val)
    {
        yuv.ptr<Vec2b>(row)[col][0] = val[0];
        yuv.ptr<Vec2b>(row)[(col/2)*2][1] = val[1];
        yuv.ptr<Vec2b>(row)[(col/2)*2 + 1][1] = val[2];
    }
};

class YVYUWriter: public YUV422Writer
{
    virtual void write(Mat& yuv, int row, int col, const YUV& val)
    {
        yuv.ptr<Vec2b>(row)[col][0] = val[0];
        yuv.ptr<Vec2b>(row)[(col/2)*2 + 1][1] = val[1];
        yuv.ptr<Vec2b>(row)[(col/2)*2][1] = val[2];
    }
};

class YUV420Reader: public YUVreader
{
    int channels() { return 1; }
    Size size(Size imgSize) { return Size(imgSize.width, imgSize.height * 3 / 2); }
};

class YUV422Reader: public YUVreader
{
    int channels() { return 2; }
    Size size(Size imgSize) { return imgSize; }
    bool requiresEvenHeight() { return false; }
};

class NV21Reader: public YUV420Reader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        uchar y = yuv.ptr<uchar>(row)[col];
        uchar u = yuv.ptr<uchar>(yuv.rows * 2 / 3 + row/2)[(col/2)*2 + 1];
        uchar v = yuv.ptr<uchar>(yuv.rows * 2 / 3 + row/2)[(col/2)*2];

        return YUV(y, u, v);
    }
};


struct NV12Reader: public YUV420Reader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        uchar y = yuv.ptr<uchar>(row)[col];
        uchar u = yuv.ptr<uchar>(yuv.rows * 2 / 3 + row/2)[(col/2)*2];
        uchar v = yuv.ptr<uchar>(yuv.rows * 2 / 3 + row/2)[(col/2)*2 + 1];

        return YUV(y, u, v);
    }
};

class YV12Reader: public YUV420Reader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        int h = yuv.rows * 2 / 3;
        uchar y = yuv.ptr<uchar>(row)[col];
        uchar u = yuv.ptr<uchar>(h + (row/2 + h/2)/2)[col/2 + ((row/2 + h/2) % 2) * (yuv.cols/2)];
        uchar v = yuv.ptr<uchar>(h + row/4)[col/2 + ((row/2) % 2) * (yuv.cols/2)];

        return YUV(y, u, v);
    }
};

class IYUVReader: public YUV420Reader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        int h = yuv.rows * 2 / 3;
        uchar y = yuv.ptr<uchar>(row)[col];
        uchar u = yuv.ptr<uchar>(h + row/4)[col/2 + ((row/2) % 2) * (yuv.cols/2)];
        uchar v = yuv.ptr<uchar>(h + (row/2 + h/2)/2)[col/2 + ((row/2 + h/2) % 2) * (yuv.cols/2)];

        return YUV(y, u, v);
    }
};

class UYVYReader: public YUV422Reader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        uchar y = yuv.ptr<Vec2b>(row)[col][1];
        uchar u = yuv.ptr<Vec2b>(row)[(col/2)*2][0];
        uchar v = yuv.ptr<Vec2b>(row)[(col/2)*2 + 1][0];

        return YUV(y, u, v);
    }
};

class YUY2Reader: public YUV422Reader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        uchar y = yuv.ptr<Vec2b>(row)[col][0];
        uchar u = yuv.ptr<Vec2b>(row)[(col/2)*2][1];
        uchar v = yuv.ptr<Vec2b>(row)[(col/2)*2 + 1][1];

        return YUV(y, u, v);
    }
};

class YVYUReader: public YUV422Reader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        uchar y = yuv.ptr<Vec2b>(row)[col][0];
        uchar u = yuv.ptr<Vec2b>(row)[(col/2)*2 + 1][1];
        uchar v = yuv.ptr<Vec2b>(row)[(col/2)*2][1];

        return YUV(y, u, v);
    }
};

class YUV888Reader : public YUVreader
{
    YUV read(const Mat& yuv, int row, int col)
    {
        return yuv.at<YUV>(row, col);
    }

    int channels() { return 3; }
    Size size(Size imgSize) { return imgSize; }
    bool requiresEvenHeight() { return false; }
    bool requiresEvenWidth() { return false; }
};

class RGB888Reader : public RGBreader
{
    RGB read(const Mat& rgb, int row, int col)
    {
        return rgb.at<RGB>(row, col);
    }

    int channels() { return 3; }
};

class BGR888Reader : public RGBreader
{
    RGB read(const Mat& rgb, int row, int col)
    {
        RGB tmp = rgb.at<RGB>(row, col);
        return RGB(tmp[2], tmp[1], tmp[0]);
    }

    int channels() { return 3; }
};

class RGBA8888Reader : public RGBreader
{
    RGB read(const Mat& rgb, int row, int col)
    {
        Vec4b rgba = rgb.at<Vec4b>(row, col);
        return RGB(rgba[0], rgba[1], rgba[2]);
    }

    int channels() { return 4; }
};

class BGRA8888Reader : public RGBreader
{
    RGB read(const Mat& rgb, int row, int col)
    {
        Vec4b rgba = rgb.at<Vec4b>(row, col);
        return RGB(rgba[2], rgba[1], rgba[0]);
    }

    int channels() { return 4; }
};

class YUV2RGB_Converter
{
public:
    RGB convert(YUV yuv)
    {
        int y = std::max(0, yuv[0] - 16);
        int u = yuv[1] - 128;
        int v = yuv[2] - 128;
        uchar r = saturate_cast<uchar>(1.164f * y + 1.596f * v);
        uchar g = saturate_cast<uchar>(1.164f * y - 0.813f * v - 0.391f * u);
        uchar b = saturate_cast<uchar>(1.164f * y + 2.018f * u);

        return RGB(r, g, b);
    }
};

class YUV2GRAY_Converter
{
public:
    uchar convert(YUV yuv)
    {
        return yuv[0];
    }
};

class RGB2YUV_Converter
{
public:
    YUV convert(RGB rgb)
    {
        int r = rgb[0];
        int g = rgb[1];
        int b = rgb[2];

        uchar y = saturate_cast<uchar>((int)( 0.257f*r + 0.504f*g + 0.098f*b + 0.5f) + 16);
        uchar u = saturate_cast<uchar>((int)(-0.148f*r - 0.291f*g + 0.439f*b + 0.5f) + 128);
        uchar v = saturate_cast<uchar>((int)( 0.439f*r - 0.368f*g - 0.071f*b + 0.5f) + 128);

        return YUV(y, u, v);
    }
};

class RGB2YUV422_Converter
{
public:
    YUV convert(RGB rgb1, RGB rgb2, int idx)
    {
        int r1 = rgb1[0];
        int g1 = rgb1[1];
        int b1 = rgb1[2];

        int r2 = rgb2[0];
        int g2 = rgb2[1];
        int b2 = rgb2[2];

        // Coefficients below based on ITU.BT-601, ISBN 1-878707-09-4 (https://fourcc.org/fccyvrgb.php)
        // The conversion coefficients for RGB to YUV422 are based on the ones for RGB to YUV.
        // For both Y components, the coefficients are applied as given in the link to each input RGB pixel
        // separately. For U and V, they are reduced by half to account for two RGB pixels contributing
        // to the same U and V values. In other words, the U and V contributions from the two RGB pixels
        // are averaged. The integer versions are obtained by multiplying the float versions by 16384
        // and rounding to the nearest integer.

        uchar y1 = saturate_cast<uchar>((int)( 0.257f*r1 + 0.504f*g1 + 0.098f*b1 + 16));
        uchar y2 = saturate_cast<uchar>((int)( 0.257f*r2 + 0.504f*g2 + 0.098f*b2 + 16));
        uchar u = saturate_cast<uchar>((int)(-0.074f*(r1+r2) - 0.1455f*(g1+g2) + 0.2195f*(b1+b2) + 128));
        uchar v = saturate_cast<uchar>((int)( 0.2195f*(r1+r2) - 0.184f*(g1+g2) - 0.0355f*(b1+b2) + 128));

        return YUV((idx==0)?y1:y2, u, v);
    }
};

YUVreader* YUVreader::getReader(int code)
{
    switch(code)
    {
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGRA_NV12:
        return new NV12Reader();
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGRA_NV21:
        return new NV21Reader();
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGRA_YV12:
        return new YV12Reader();
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2RGBA_IYUV:
    case COLOR_YUV2BGRA_IYUV:
        return new IYUVReader();
    case COLOR_YUV2RGB_UYVY:
    case COLOR_YUV2BGR_UYVY:
    case COLOR_YUV2RGBA_UYVY:
    case COLOR_YUV2BGRA_UYVY:
        return new UYVYReader();
    //case COLOR_YUV2RGB_VYUY = 109,
    //case COLOR_YUV2BGR_VYUY = 110,
    //case COLOR_YUV2RGBA_VYUY = 113,
    //case COLOR_YUV2BGRA_VYUY = 114,
    //    return ??
    case COLOR_YUV2RGB_YUY2:
    case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2RGBA_YUY2:
    case COLOR_YUV2BGRA_YUY2:
        return new YUY2Reader();
    case COLOR_YUV2RGB_YVYU:
    case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2RGBA_YVYU:
    case COLOR_YUV2BGRA_YVYU:
        return new YVYUReader();
    case COLOR_YUV2GRAY_420:
        return new NV21Reader();
    case COLOR_YUV2GRAY_UYVY:
        return new UYVYReader();
    case COLOR_YUV2GRAY_YUY2:
        return new YUY2Reader();
    case COLOR_YUV2BGR:
    case COLOR_YUV2RGB:
        return new YUV888Reader();
    default:
        return 0;
    }
}

RGBreader* RGBreader::getReader(int code)
{
    switch(code)
    {
    case COLOR_RGB2YUV_YV12:
    case COLOR_RGB2YUV_I420:
    case COLOR_RGB2YUV_UYVY:
    case COLOR_RGB2YUV_YUY2:
    case COLOR_RGB2YUV_YVYU:
        return new RGB888Reader();
    case COLOR_BGR2YUV_YV12:
    case COLOR_BGR2YUV_I420:
    case COLOR_BGR2YUV_UYVY:
    case COLOR_BGR2YUV_YUY2:
    case COLOR_BGR2YUV_YVYU:
        return new BGR888Reader();
    case COLOR_RGBA2YUV_I420:
    case COLOR_RGBA2YUV_YV12:
    case COLOR_RGBA2YUV_UYVY:
    case COLOR_RGBA2YUV_YUY2:
    case COLOR_RGBA2YUV_YVYU:
        return new RGBA8888Reader();
    case COLOR_BGRA2YUV_YV12:
    case COLOR_BGRA2YUV_I420:
    case COLOR_BGRA2YUV_UYVY:
    case COLOR_BGRA2YUV_YUY2:
    case COLOR_BGRA2YUV_YVYU:
        return new BGRA8888Reader();
    default:
        return 0;
    };
}

RGBwriter* RGBwriter::getWriter(int code)
{
    switch(code)
    {
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2RGB_UYVY:
    //case COLOR_YUV2RGB_VYUY:
    case COLOR_YUV2RGB_YUY2:
    case COLOR_YUV2RGB_YVYU:
    case COLOR_YUV2RGB:
        return new RGB888Writer();
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2BGR_UYVY:
    //case COLOR_YUV2BGR_VYUY:
    case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2BGR:
        return new BGR888Writer();
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2RGBA_IYUV:
    case COLOR_YUV2RGBA_UYVY:
    //case COLOR_YUV2RGBA_VYUY:
    case COLOR_YUV2RGBA_YUY2:
    case COLOR_YUV2RGBA_YVYU:
        return new RGBA8888Writer();
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2BGRA_UYVY:
    //case COLOR_YUV2BGRA_VYUY:
    case COLOR_YUV2BGRA_YUY2:
    case COLOR_YUV2BGRA_YVYU:
        return new BGRA8888Writer();
    default:
        return 0;
    };
}

GRAYwriter* GRAYwriter::getWriter(int code)
{
    switch(code)
    {
    case COLOR_YUV2GRAY_420:
    case COLOR_YUV2GRAY_UYVY:
    case COLOR_YUV2GRAY_YUY2:
        return new GRAYwriter();
    default:
        return 0;
    }
}

YUVwriter* YUVwriter::getWriter(int code)
{
    switch(code)
    {
    case COLOR_RGB2YUV_YV12:
    case COLOR_BGR2YUV_YV12:
    case COLOR_RGBA2YUV_YV12:
    case COLOR_BGRA2YUV_YV12:
        return new YV12Writer();
    case COLOR_RGB2YUV_UYVY:
    case COLOR_BGR2YUV_UYVY:
    case COLOR_RGBA2YUV_UYVY:
    case COLOR_BGRA2YUV_UYVY:
        return new UYVYWriter();
    case COLOR_RGB2YUV_YUY2:
    case COLOR_BGR2YUV_YUY2:
    case COLOR_RGBA2YUV_YUY2:
    case COLOR_BGRA2YUV_YUY2:
        return new YUY2Writer();
    case COLOR_RGB2YUV_YVYU:
    case COLOR_BGR2YUV_YVYU:
    case COLOR_RGBA2YUV_YVYU:
    case COLOR_BGRA2YUV_YVYU:
        return new YVYUWriter();
    case COLOR_RGB2YUV_I420:
    case COLOR_BGR2YUV_I420:
    case COLOR_RGBA2YUV_I420:
    case COLOR_BGRA2YUV_I420:
        return new I420Writer();
    default:
        return 0;
    };
}

template<class convertor>
void referenceYUV2RGB(const Mat& yuv, Mat& rgb, YUVreader* yuvReader, RGBwriter* rgbWriter)
{
    convertor cvt;

    for(int row = 0; row < rgb.rows; ++row)
        for(int col = 0; col < rgb.cols; ++col)
            rgbWriter->write(rgb, row, col, cvt.convert(yuvReader->read(yuv, row, col)));
}

template<class convertor>
void referenceYUV2GRAY(const Mat& yuv, Mat& rgb, YUVreader* yuvReader, GRAYwriter* grayWriter)
{
    convertor cvt;

    for(int row = 0; row < rgb.rows; ++row)
        for(int col = 0; col < rgb.cols; ++col)
            grayWriter->write(rgb, row, col, cvt.convert(yuvReader->read(yuv, row, col)));
}

template<class convertor>
void referenceRGB2YUV(const Mat& rgb, Mat& yuv, RGBreader* rgbReader, YUVwriter* yuvWriter)
{
    convertor cvt;

    for(int row = 0; row < rgb.rows; ++row)
        for(int col = 0; col < rgb.cols; ++col)
            yuvWriter->write(yuv, row, col, cvt.convert(rgbReader->read(rgb, row, col)));
}

template<class convertor>
void referenceRGB2YUV422(const Mat& rgb, Mat& yuv, RGBreader* rgbReader, YUVwriter* yuvWriter)
{
    convertor cvt;

    for(int row = 0; row < rgb.rows; ++row)
    {
            for(int col = 0; col < rgb.cols; col+=2)
            {
                yuvWriter->write(yuv, row, col, cvt.convert(rgbReader->read(rgb, row, col), rgbReader->read(rgb, row, col+1), 0));
                yuvWriter->write(yuv, row, col+1, cvt.convert(rgbReader->read(rgb, row, col), rgbReader->read(rgb, row, col+1), 1));
            }
    }
}

struct ConversionYUV
{
    explicit ConversionYUV( const int code )
    {
        yuvReader_  = YUVreader :: getReader(code);
        yuvWriter_  = YUVwriter :: getWriter(code);
        rgbReader_  = RGBreader :: getReader(code);
        rgbWriter_  = RGBwriter :: getWriter(code);
        grayWriter_ = GRAYwriter:: getWriter(code);
    }

    ~ConversionYUV()
    {
        if (yuvReader_)
            delete yuvReader_;

        if (yuvWriter_)
            delete yuvWriter_;

        if (rgbReader_)
            delete rgbReader_;

        if (rgbWriter_)
            delete rgbWriter_;

        if (grayWriter_)
            delete grayWriter_;
    }

    int getDcn()
    {
        return (rgbWriter_ != 0) ? rgbWriter_->channels() : ((grayWriter_ != 0) ? grayWriter_->channels() : yuvWriter_->channels());
    }

    int getScn()
    {
        return (yuvReader_ != 0) ? yuvReader_->channels() : rgbReader_->channels();
    }

    Size getSrcSize( const Size& imgSize )
    {
        return (yuvReader_ != 0) ? yuvReader_->size(imgSize) : imgSize;
    }

    Size getDstSize( const Size& imgSize )
    {
        return (yuvWriter_ != 0) ? yuvWriter_->size(imgSize) : imgSize;
    }

    bool requiresEvenHeight()
    {
        return (yuvReader_ != 0) ? yuvReader_->requiresEvenHeight() : ((yuvWriter_ != 0) ? yuvWriter_->requiresEvenHeight() : false);
    }

    bool requiresEvenWidth()
    {
        return (yuvReader_ != 0) ? yuvReader_->requiresEvenWidth() : ((yuvWriter_ != 0) ? yuvWriter_->requiresEvenWidth() : false);
    }

    YUVreader*  yuvReader_;
    YUVwriter*  yuvWriter_;
    RGBreader*  rgbReader_;
    RGBwriter*  rgbWriter_;
    GRAYwriter* grayWriter_;
};

bool is_rgb2yuv422(int code)
{
    switch (code)
    {
        case COLOR_RGB2YUV_UYVY:
        case COLOR_BGR2YUV_UYVY:
        case COLOR_RGBA2YUV_UYVY:
        case COLOR_BGRA2YUV_UYVY:
        case COLOR_RGB2YUV_YUY2:
        case COLOR_BGR2YUV_YUY2:
        case COLOR_RGBA2YUV_YUY2:
        case COLOR_BGRA2YUV_YUY2:
        case COLOR_RGB2YUV_YVYU:
        case COLOR_BGR2YUV_YVYU:
        case COLOR_RGBA2YUV_YVYU:
        case COLOR_BGRA2YUV_YVYU:
            return true;
        default:
            return false;
    }
}

CV_ENUM(YUVCVTS, COLOR_YUV2RGB_NV12, COLOR_YUV2BGR_NV12, COLOR_YUV2RGB_NV21, COLOR_YUV2BGR_NV21,
                 COLOR_YUV2RGBA_NV12, COLOR_YUV2BGRA_NV12, COLOR_YUV2RGBA_NV21, COLOR_YUV2BGRA_NV21,
                 COLOR_YUV2RGB_YV12, COLOR_YUV2BGR_YV12, COLOR_YUV2RGB_IYUV, COLOR_YUV2BGR_IYUV,
                 COLOR_YUV2RGBA_YV12, COLOR_YUV2BGRA_YV12, COLOR_YUV2RGBA_IYUV, COLOR_YUV2BGRA_IYUV,
                 COLOR_YUV2RGB_UYVY, COLOR_YUV2BGR_UYVY, COLOR_YUV2RGBA_UYVY, COLOR_YUV2BGRA_UYVY,
                 COLOR_YUV2RGB_YUY2, COLOR_YUV2BGR_YUY2, COLOR_YUV2RGB_YVYU, COLOR_YUV2BGR_YVYU,
                 COLOR_YUV2RGBA_YUY2, COLOR_YUV2BGRA_YUY2, COLOR_YUV2RGBA_YVYU, COLOR_YUV2BGRA_YVYU,
                 COLOR_YUV2GRAY_420, COLOR_YUV2GRAY_UYVY, COLOR_YUV2GRAY_YUY2,
                 COLOR_YUV2BGR, COLOR_YUV2RGB, COLOR_RGB2YUV_YV12, COLOR_BGR2YUV_YV12, COLOR_RGBA2YUV_YV12,
                 COLOR_BGRA2YUV_YV12, COLOR_RGB2YUV_I420, COLOR_BGR2YUV_I420, COLOR_RGBA2YUV_I420, COLOR_BGRA2YUV_I420,
                 COLOR_RGB2YUV_UYVY,  COLOR_BGR2YUV_UYVY,  COLOR_RGBA2YUV_UYVY, COLOR_BGRA2YUV_UYVY,
                 COLOR_RGB2YUV_YUY2,  COLOR_BGR2YUV_YUY2,  COLOR_RGB2YUV_YVYU,  COLOR_BGR2YUV_YVYU,
                 COLOR_RGBA2YUV_YUY2, COLOR_BGRA2YUV_YUY2, COLOR_RGBA2YUV_YVYU, COLOR_BGRA2YUV_YVYU)

typedef ::testing::TestWithParam<YUVCVTS> Imgproc_ColorYUV;

TEST_P(Imgproc_ColorYUV, accuracy)
{
    int code = GetParam();
    bool yuv422 = is_rgb2yuv422(code);

    RNG& random = theRNG();

    ConversionYUV cvt(code);

    const int scn = cvt.getScn();
    const int dcn = cvt.getDcn();
    for(int iter = 0; iter < 30; ++iter)
    {
        Size sz(random.uniform(1, 641), random.uniform(1, 481));

        if(cvt.requiresEvenWidth())  sz.width  += sz.width % 2;
        if(cvt.requiresEvenHeight()) sz.height += sz.height % 2;

        Size srcSize = cvt.getSrcSize(sz);
        Mat src = Mat(srcSize.height, srcSize.width * scn, CV_8UC1).reshape(scn);

        Size dstSize = cvt.getDstSize(sz);
        Mat dst = Mat(dstSize.height, dstSize.width * dcn, CV_8UC1).reshape(dcn);
        Mat gold(dstSize, CV_8UC(dcn));

        random.fill(src, RNG::UNIFORM, 0, 256);

        if(cvt.rgbWriter_)
            referenceYUV2RGB<YUV2RGB_Converter>  (src, gold, cvt.yuvReader_, cvt.rgbWriter_);
        else if(cvt.grayWriter_)
            referenceYUV2GRAY<YUV2GRAY_Converter>(src, gold, cvt.yuvReader_, cvt.grayWriter_);
        else if(cvt.yuvWriter_)
        {
            if(!yuv422)
                referenceRGB2YUV<RGB2YUV_Converter>  (src, gold, cvt.rgbReader_, cvt.yuvWriter_);
            else
                referenceRGB2YUV422<RGB2YUV422_Converter>  (src, gold, cvt.rgbReader_, cvt.yuvWriter_);
        }

        cv::cvtColor(src, dst, code, -1);

        EXPECT_EQ(0, countOfDifferencies(gold, dst));
    }
}

TEST_P(Imgproc_ColorYUV, roi_accuracy)
{
    int code = GetParam();
    bool yuv422 = is_rgb2yuv422(code);

    RNG& random = theRNG();

    ConversionYUV cvt(code);

    const int scn = cvt.getScn();
    const int dcn = cvt.getDcn();
    for(int iter = 0; iter < 30; ++iter)
    {
        Size sz(random.uniform(1, 641), random.uniform(1, 481));

        if(cvt.requiresEvenWidth())  sz.width  += sz.width % 2;
        if(cvt.requiresEvenHeight()) sz.height += sz.height % 2;

        int roi_offset_top = random.uniform(0, 6);
        int roi_offset_bottom = random.uniform(0, 6);
        int roi_offset_left = random.uniform(0, 6);
        int roi_offset_right = random.uniform(0, 6);

        Size srcSize = cvt.getSrcSize(sz);
        Mat src_full(srcSize.height + roi_offset_top + roi_offset_bottom, srcSize.width + roi_offset_left + roi_offset_right, CV_8UC(scn));

        Size dstSize = cvt.getDstSize(sz);
        Mat dst_full(dstSize.height  + roi_offset_left + roi_offset_right, dstSize.width + roi_offset_top + roi_offset_bottom, CV_8UC(dcn), Scalar::all(0));
        Mat gold_full(dst_full.size(), CV_8UC(dcn), Scalar::all(0));

        random.fill(src_full, RNG::UNIFORM, 0, 256);

        Mat src = src_full(Range(roi_offset_top, roi_offset_top + srcSize.height), Range(roi_offset_left, roi_offset_left + srcSize.width));
        Mat dst = dst_full(Range(roi_offset_left, roi_offset_left + dstSize.height), Range(roi_offset_top, roi_offset_top + dstSize.width));
        Mat gold = gold_full(Range(roi_offset_left, roi_offset_left + dstSize.height), Range(roi_offset_top, roi_offset_top + dstSize.width));

        if(cvt.rgbWriter_)
            referenceYUV2RGB<YUV2RGB_Converter>  (src, gold, cvt.yuvReader_, cvt.rgbWriter_);
        else if(cvt.grayWriter_)
            referenceYUV2GRAY<YUV2GRAY_Converter>(src, gold, cvt.yuvReader_, cvt.grayWriter_);
        else if(cvt.yuvWriter_)
        {
            if(!yuv422)
                referenceRGB2YUV<RGB2YUV_Converter>  (src, gold, cvt.rgbReader_, cvt.yuvWriter_);
            else
                referenceRGB2YUV422<RGB2YUV422_Converter>  (src, gold, cvt.rgbReader_, cvt.yuvWriter_);
        }

        cv::cvtColor(src, dst, code, -1);

        EXPECT_EQ(0, countOfDifferencies(gold_full, dst_full));
    }
}

INSTANTIATE_TEST_CASE_P(cvt420, Imgproc_ColorYUV,
    ::testing::Values((int)COLOR_YUV2RGB_NV12, (int)COLOR_YUV2BGR_NV12, (int)COLOR_YUV2RGB_NV21, (int)COLOR_YUV2BGR_NV21,
                      (int)COLOR_YUV2RGBA_NV12, (int)COLOR_YUV2BGRA_NV12, (int)COLOR_YUV2RGBA_NV21, (int)COLOR_YUV2BGRA_NV21,
                      (int)COLOR_YUV2RGB_YV12, (int)COLOR_YUV2BGR_YV12, (int)COLOR_YUV2RGB_IYUV, (int)COLOR_YUV2BGR_IYUV,
                      (int)COLOR_YUV2RGBA_YV12, (int)COLOR_YUV2BGRA_YV12, (int)COLOR_YUV2RGBA_IYUV, (int)COLOR_YUV2BGRA_IYUV,
                      (int)COLOR_YUV2GRAY_420, (int)COLOR_RGB2YUV_YV12, (int)COLOR_BGR2YUV_YV12, (int)COLOR_RGBA2YUV_YV12,
                      (int)COLOR_BGRA2YUV_YV12, (int)COLOR_RGB2YUV_I420, (int)COLOR_BGR2YUV_I420, (int)COLOR_RGBA2YUV_I420,
                      (int)COLOR_BGRA2YUV_I420));

INSTANTIATE_TEST_CASE_P(cvt422, Imgproc_ColorYUV,
    ::testing::Values((int)COLOR_YUV2RGB_UYVY, (int)COLOR_YUV2BGR_UYVY, (int)COLOR_YUV2RGBA_UYVY, (int)COLOR_YUV2BGRA_UYVY,
                      (int)COLOR_YUV2RGB_YUY2, (int)COLOR_YUV2BGR_YUY2, (int)COLOR_YUV2RGB_YVYU, (int)COLOR_YUV2BGR_YVYU,
                      (int)COLOR_YUV2RGBA_YUY2, (int)COLOR_YUV2BGRA_YUY2, (int)COLOR_YUV2RGBA_YVYU, (int)COLOR_YUV2BGRA_YVYU,
                      (int)COLOR_YUV2GRAY_UYVY, (int)COLOR_YUV2GRAY_YUY2,
                      (int)COLOR_RGB2YUV_UYVY,  (int)COLOR_BGR2YUV_UYVY,  (int)COLOR_RGBA2YUV_UYVY, (int)COLOR_BGRA2YUV_UYVY,
                      (int)COLOR_RGB2YUV_YUY2,  (int)COLOR_BGR2YUV_YUY2,  (int)COLOR_RGB2YUV_YVYU,  (int)COLOR_BGR2YUV_YVYU,
                      (int)COLOR_RGBA2YUV_YUY2, (int)COLOR_BGRA2YUV_YUY2, (int)COLOR_RGBA2YUV_YVYU, (int)COLOR_BGRA2YUV_YVYU,
                      (int)COLOR_RGB2YUV_YUY2));

}

TEST(cvtColorUYVY, size_issue_21035)
{
    Mat input = Mat::zeros(1, 1, CV_8UC2);
    Mat output;
    EXPECT_THROW(cv::cvtColor(input, output, cv::COLOR_YUV2BGR_UYVY), cv::Exception);
}

} // namespace
