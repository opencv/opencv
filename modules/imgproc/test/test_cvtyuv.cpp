#include "test_precomp.hpp"

using namespace cv;
using namespace std;

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

YUVreader* YUVreader::getReader(int code)
{
    switch(code)
    {
    case CV_YUV2RGB_NV12:
    case CV_YUV2BGR_NV12:
    case CV_YUV2RGBA_NV12:
    case CV_YUV2BGRA_NV12:
        return new NV12Reader();
    case CV_YUV2RGB_NV21:
    case CV_YUV2BGR_NV21:
    case CV_YUV2RGBA_NV21:
    case CV_YUV2BGRA_NV21:
        return new NV21Reader();
    case CV_YUV2RGB_YV12:
    case CV_YUV2BGR_YV12:
    case CV_YUV2RGBA_YV12:
    case CV_YUV2BGRA_YV12:
        return new YV12Reader();
    case CV_YUV2RGB_IYUV:
    case CV_YUV2BGR_IYUV:
    case CV_YUV2RGBA_IYUV:
    case CV_YUV2BGRA_IYUV:
        return new IYUVReader();
    case CV_YUV2RGB_UYVY:
    case CV_YUV2BGR_UYVY:
    case CV_YUV2RGBA_UYVY:
    case CV_YUV2BGRA_UYVY:
        return new UYVYReader();
    //case CV_YUV2RGB_VYUY = 109,
    //case CV_YUV2BGR_VYUY = 110,
    //case CV_YUV2RGBA_VYUY = 113,
    //case CV_YUV2BGRA_VYUY = 114,
    //    return ??
    case CV_YUV2RGB_YUY2:
    case CV_YUV2BGR_YUY2:
    case CV_YUV2RGBA_YUY2:
    case CV_YUV2BGRA_YUY2:
        return new YUY2Reader();
    case CV_YUV2RGB_YVYU:
    case CV_YUV2BGR_YVYU:
    case CV_YUV2RGBA_YVYU:
    case CV_YUV2BGRA_YVYU:
        return new YVYUReader();
    case CV_YUV2GRAY_420:
        return new NV21Reader();
    case CV_YUV2GRAY_UYVY:
        return new UYVYReader();
    case CV_YUV2GRAY_YUY2:
        return new YUY2Reader();
    case CV_YUV2BGR:
    case CV_YUV2RGB:
        return new YUV888Reader();
    default:
        return 0;
    }
}

RGBwriter* RGBwriter::getWriter(int code)
{
    switch(code)
    {
    case CV_YUV2RGB_NV12:
    case CV_YUV2RGB_NV21:
    case CV_YUV2RGB_YV12:
    case CV_YUV2RGB_IYUV:
    case CV_YUV2RGB_UYVY:
    //case CV_YUV2RGB_VYUY:
    case CV_YUV2RGB_YUY2:
    case CV_YUV2RGB_YVYU:
    case CV_YUV2RGB:
        return new RGB888Writer();
    case CV_YUV2BGR_NV12:
    case CV_YUV2BGR_NV21:
    case CV_YUV2BGR_YV12:
    case CV_YUV2BGR_IYUV:
    case CV_YUV2BGR_UYVY:
    //case CV_YUV2BGR_VYUY:
    case CV_YUV2BGR_YUY2:
    case CV_YUV2BGR_YVYU:
    case CV_YUV2BGR:
        return new BGR888Writer();
    case CV_YUV2RGBA_NV12:
    case CV_YUV2RGBA_NV21:
    case CV_YUV2RGBA_YV12:
    case CV_YUV2RGBA_IYUV:
    case CV_YUV2RGBA_UYVY:
    //case CV_YUV2RGBA_VYUY:
    case CV_YUV2RGBA_YUY2:
    case CV_YUV2RGBA_YVYU:
        return new RGBA8888Writer();
    case CV_YUV2BGRA_NV12:
    case CV_YUV2BGRA_NV21:
    case CV_YUV2BGRA_YV12:
    case CV_YUV2BGRA_IYUV:
    case CV_YUV2BGRA_UYVY:
    //case CV_YUV2BGRA_VYUY:
    case CV_YUV2BGRA_YUY2:
    case CV_YUV2BGRA_YVYU:
        return new BGRA8888Writer();
    default:
        return 0;
    };
}

GRAYwriter* GRAYwriter::getWriter(int code)
{
    switch(code)
    {
    case CV_YUV2GRAY_420:
    case CV_YUV2GRAY_UYVY:
    case CV_YUV2GRAY_YUY2:
        return new GRAYwriter();
    default:
        return 0;
    }
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

CV_ENUM(YUVCVTS, CV_YUV2RGB_NV12, CV_YUV2BGR_NV12, CV_YUV2RGB_NV21, CV_YUV2BGR_NV21,
                 CV_YUV2RGBA_NV12, CV_YUV2BGRA_NV12, CV_YUV2RGBA_NV21, CV_YUV2BGRA_NV21,
                 CV_YUV2RGB_YV12, CV_YUV2BGR_YV12, CV_YUV2RGB_IYUV, CV_YUV2BGR_IYUV,
                 CV_YUV2RGBA_YV12, CV_YUV2BGRA_YV12, CV_YUV2RGBA_IYUV, CV_YUV2BGRA_IYUV,
                 CV_YUV2RGB_UYVY, CV_YUV2BGR_UYVY, CV_YUV2RGBA_UYVY, CV_YUV2BGRA_UYVY,
                 CV_YUV2RGB_YUY2, CV_YUV2BGR_YUY2, CV_YUV2RGB_YVYU, CV_YUV2BGR_YVYU,
                 CV_YUV2RGBA_YUY2, CV_YUV2BGRA_YUY2, CV_YUV2RGBA_YVYU, CV_YUV2BGRA_YVYU,
                 CV_YUV2GRAY_420, CV_YUV2GRAY_UYVY, CV_YUV2GRAY_YUY2,
                 CV_YUV2BGR, CV_YUV2RGB);

typedef ::testing::TestWithParam<YUVCVTS> Imgproc_ColorYUV;

TEST_P(Imgproc_ColorYUV, accuracy)
{
    int code = GetParam();
    RNG& random = theRNG();

    YUVreader* yuvReader = YUVreader::getReader(code);
    RGBwriter* rgbWriter = RGBwriter::getWriter(code);
    GRAYwriter* grayWriter = GRAYwriter::getWriter(code);

    int dcn = (rgbWriter == 0) ? grayWriter->channels() : rgbWriter->channels();

    for(int iter = 0; iter < 30; ++iter)
    {
        Size sz(random.uniform(1, 641), random.uniform(1, 481));

        if(yuvReader->requiresEvenWidth()) sz.width += sz.width % 2;
        if(yuvReader->requiresEvenHeight()) sz.height += sz.height % 2;

        Size ysz = yuvReader->size(sz);
        Mat src = Mat(ysz.height, ysz.width * yuvReader->channels(), CV_8UC1).reshape(yuvReader->channels());

        Mat dst = Mat(sz.height, sz.width * dcn, CV_8UC1).reshape(dcn);
        Mat gold(sz, CV_8UC(dcn));

        random.fill(src, RNG::UNIFORM, 0, 256);

        if(rgbWriter)
            referenceYUV2RGB<YUV2RGB_Converter>(src, gold, yuvReader, rgbWriter);
        else
            referenceYUV2GRAY<YUV2GRAY_Converter>(src, gold, yuvReader, grayWriter);

        cv::cvtColor(src, dst, code, -1);

        EXPECT_EQ(0, countOfDifferencies(gold, dst));
    }
}

TEST_P(Imgproc_ColorYUV, roi_accuracy)
{
    int code = GetParam();
    RNG& random = theRNG();

    YUVreader* yuvReader = YUVreader::getReader(code);
    RGBwriter* rgbWriter = RGBwriter::getWriter(code);
    GRAYwriter* grayWriter = GRAYwriter::getWriter(code);

    int dcn = (rgbWriter == 0) ? grayWriter->channels() : rgbWriter->channels();

    for(int iter = 0; iter < 30; ++iter)
    {
        Size sz(random.uniform(1, 641), random.uniform(1, 481));

        if(yuvReader->requiresEvenWidth()) sz.width += sz.width % 2;
        if(yuvReader->requiresEvenHeight()) sz.height += sz.height % 2;

        int roi_offset_top = random.uniform(0, 6);
        int roi_offset_bottom = random.uniform(0, 6);
        int roi_offset_left = random.uniform(0, 6);
        int roi_offset_right = random.uniform(0, 6);

        Size ysz = yuvReader->size(sz);

        Mat src_full(ysz.height + roi_offset_top + roi_offset_bottom, ysz.width + roi_offset_left + roi_offset_right, CV_8UC(yuvReader->channels()));
        Mat dst_full(sz.height  + roi_offset_left + roi_offset_right, sz.width + roi_offset_top + roi_offset_bottom, CV_8UC(dcn), Scalar::all(0));
        Mat gold_full(dst_full.size(), CV_8UC(dcn), Scalar::all(0));

        random.fill(src_full, RNG::UNIFORM, 0, 256);

        Mat src = src_full(Range(roi_offset_top, roi_offset_top + ysz.height), Range(roi_offset_left, roi_offset_left + ysz.width));
        Mat dst = dst_full(Range(roi_offset_left, roi_offset_left + sz.height), Range(roi_offset_top, roi_offset_top + sz.width));
        Mat gold = gold_full(Range(roi_offset_left, roi_offset_left + sz.height), Range(roi_offset_top, roi_offset_top + sz.width));

        if(rgbWriter)
            referenceYUV2RGB<YUV2RGB_Converter>(src, gold, yuvReader, rgbWriter);
        else
            referenceYUV2GRAY<YUV2GRAY_Converter>(src, gold, yuvReader, grayWriter);

        cv::cvtColor(src, dst, code, -1);

        EXPECT_EQ(0, countOfDifferencies(gold_full, dst_full));
    }
}

INSTANTIATE_TEST_CASE_P(cvt420, Imgproc_ColorYUV,
    ::testing::Values((int)CV_YUV2RGB_NV12, (int)CV_YUV2BGR_NV12, (int)CV_YUV2RGB_NV21, (int)CV_YUV2BGR_NV21,
                      (int)CV_YUV2RGBA_NV12, (int)CV_YUV2BGRA_NV12, (int)CV_YUV2RGBA_NV21, (int)CV_YUV2BGRA_NV21,
                      (int)CV_YUV2RGB_YV12, (int)CV_YUV2BGR_YV12, (int)CV_YUV2RGB_IYUV, (int)CV_YUV2BGR_IYUV,
                      (int)CV_YUV2RGBA_YV12, (int)CV_YUV2BGRA_YV12, (int)CV_YUV2RGBA_IYUV, (int)CV_YUV2BGRA_IYUV,
                      (int)CV_YUV2GRAY_420));

INSTANTIATE_TEST_CASE_P(DISABLED_cvt888, Imgproc_ColorYUV,
    ::testing::Values((int)CV_YUV2BGR, (int)CV_YUV2RGB));

INSTANTIATE_TEST_CASE_P(DISABLED_cvt422, Imgproc_ColorYUV,
    ::testing::Values((int)CV_YUV2RGB_UYVY, (int)CV_YUV2BGR_UYVY, (int)CV_YUV2RGBA_UYVY, (int)CV_YUV2BGRA_UYVY,
                      (int)CV_YUV2RGB_YUY2, (int)CV_YUV2BGR_YUY2, (int)CV_YUV2RGB_YVYU, (int)CV_YUV2BGR_YVYU,
                      (int)CV_YUV2RGBA_YUY2, (int)CV_YUV2BGRA_YUY2, (int)CV_YUV2RGBA_YVYU, (int)CV_YUV2BGRA_YVYU,
                      (int)CV_YUV2GRAY_UYVY, (int)CV_YUV2GRAY_YUY2));
