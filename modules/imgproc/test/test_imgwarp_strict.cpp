/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

#include "test_precomp.hpp"

#include <cmath>
#include <vector>
#include <iostream>

using namespace cv;

namespace
{
    void __wrap_printf_func(const char* fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        char buffer[256];
        vsprintf (buffer, fmt, args);
        cvtest::TS::ptr()->printf(cvtest::TS::SUMMARY, buffer);
        va_end(args);
    }

    #define PRINT_TO_LOG __wrap_printf_func
}

#define SHOW_IMAGE
#undef SHOW_IMAGE

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ImageWarpBaseTest
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_ImageWarpBaseTest :
    public cvtest::BaseTest
{
public:
    enum { cell_size = 10 };

    CV_ImageWarpBaseTest();
    virtual ~CV_ImageWarpBaseTest();

    virtual void run(int);
protected:
    virtual void generate_test_data();

    virtual void run_func() = 0;
    virtual void run_reference_func() = 0;
    virtual void validate_results() const;
    virtual void prepare_test_data_for_reference_func();

    Size randSize(RNG& rng) const;

    const char* interpolation_to_string(int inter_type) const;

    int interpolation;
    Mat src;
    Mat dst;
    Mat reference_dst;
};

CV_ImageWarpBaseTest::CV_ImageWarpBaseTest() :
    BaseTest(), interpolation(-1),
    src(), dst(), reference_dst()
{
    test_case_count = 40;
    ts->set_failed_test_info(cvtest::TS::OK);
}

CV_ImageWarpBaseTest::~CV_ImageWarpBaseTest()
{
}

const char* CV_ImageWarpBaseTest::interpolation_to_string(int inter) const
{
    if (inter == INTER_NEAREST)
        return "INTER_NEAREST";
    if (inter == INTER_LINEAR)
        return "INTER_LINEAR";
    if (inter == INTER_AREA)
        return "INTER_AREA";
    if (inter == INTER_CUBIC)
        return "INTER_CUBIC";
    if (inter == INTER_LANCZOS4)
        return "INTER_LANCZOS4";
    if (inter == INTER_LANCZOS4 + 1)
        return "INTER_AREA_FAST";
    return "Unsupported/Unkown interpolation type";
}

Size CV_ImageWarpBaseTest::randSize(RNG& rng) const
{
    Size size;
    size.width = static_cast<int>(std::exp(rng.uniform(1.0f, 7.0f)));
    size.height = static_cast<int>(std::exp(rng.uniform(1.0f, 7.0f)));

    return size;
}

void CV_ImageWarpBaseTest::generate_test_data()
{
    RNG& rng = ts->get_rng();

    // generating the src matrix structure
    Size ssize = randSize(rng), dsize;

    int depth = rng.uniform(0, CV_64F);
    while (depth == CV_8S || depth == CV_32S)
        depth = rng.uniform(0, CV_64F);

    int cn = rng.uniform(1, 4);
    while (cn == 2)
        cn = rng.uniform(1, 4);

    src.create(ssize, CV_MAKE_TYPE(depth, cn));

    // generating the src matrix
    int x, y;
    if (cvtest::randInt(rng) % 2)
    {
        for (y = 0; y < ssize.height; y += cell_size)
            for (x = 0; x < ssize.width; x += cell_size)
                rectangle(src, Point(x, y), Point(x + std::min<int>(cell_size, ssize.width - x), y +
                        std::min<int>(cell_size, ssize.height - y)), Scalar::all((x + y) % 2 ? 255: 0), CV_FILLED);
    }
    else
    {
        src = Scalar::all(255);
        for (y = cell_size; y < src.rows; y += cell_size)
            line(src, Point2i(0, y), Point2i(src.cols, y), Scalar::all(0), 1);
        for (x = cell_size; x < src.cols; x += cell_size)
            line(src, Point2i(x, 0), Point2i(x, src.rows), Scalar::all(0), 1);
    }

    // generating an interpolation type
    interpolation = rng.uniform(0, CV_INTER_LANCZOS4 + 1);

    // generating the dst matrix structure
    double scale_x, scale_y;
    if (interpolation == INTER_AREA)
    {
        bool area_fast = rng.uniform(0., 1.) > 0.5;
        if (area_fast)
        {
            scale_x = rng.uniform(2, 5);
            scale_y = rng.uniform(2, 5);
        }
        else
        {
            scale_x = rng.uniform(1.0, 3.0);
            scale_y = rng.uniform(1.0, 3.0);
        }
    }
    else
    {
        scale_x = rng.uniform(0.4, 4.0);
        scale_y = rng.uniform(0.4, 4.0);
    }
    CV_Assert(scale_x > 0.0f && scale_y > 0.0f);

    dsize.width = saturate_cast<int>((ssize.width + scale_x - 1) / scale_x);
    dsize.height = saturate_cast<int>((ssize.height + scale_y - 1) / scale_y);

    dst = Mat::zeros(dsize, src.type());
    reference_dst = Mat::zeros(dst.size(), CV_MAKE_TYPE(CV_32F, dst.channels()));

    scale_x = src.cols / static_cast<double>(dst.cols);
    scale_y = src.rows / static_cast<double>(dst.rows);

    if (interpolation == INTER_AREA && (scale_x < 1.0 || scale_y < 1.0))
        interpolation = INTER_LINEAR;
}

void CV_ImageWarpBaseTest::run(int)
{
    for (int i = 0; i < test_case_count; ++i)
    {
        generate_test_data();
        run_func();
        run_reference_func();
        if (ts->get_err_code() < 0)
            break;
        validate_results();
        if (ts->get_err_code() < 0)
            break;
        ts->update_context(this, i, true);
    }
    ts->set_gtest_status();
}

void CV_ImageWarpBaseTest::validate_results() const
{
    Mat _dst;
    dst.convertTo(_dst, reference_dst.depth());

    Size dsize = dst.size(), ssize = src.size();
    int cn = _dst.channels();
    dsize.width *= cn;
    float t = 1.0f;
    if (interpolation == INTER_CUBIC)
        t = 1.0f;
    else if (interpolation == INTER_LANCZOS4)
        t = 1.0f;
    else if (interpolation == INTER_NEAREST)
        t = 1.0f;
    else if (interpolation == INTER_AREA)
        t = 2.0f;

    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const float* rD = reference_dst.ptr<float>(dy);
        const float* D = _dst.ptr<float>(dy);

        for (int dx = 0; dx < dsize.width; ++dx)
            if (fabs(rD[dx] - D[dx]) > t &&
//                fabs(rD[dx] - D[dx]) < 250.0f &&
                rD[dx] <= 255.0f && D[dx] <= 255.0f && rD[dx] >= 0.0f && D[dx] >= 0.0f)
            {
                PRINT_TO_LOG("\nNorm of the difference: %lf\n", norm(reference_dst, _dst, NORM_INF));
                PRINT_TO_LOG("Error in (dx, dy): (%d, %d)\n", dx / cn + 1, dy + 1);
                PRINT_TO_LOG("Tuple (rD, D): (%f, %f)\n", rD[dx], D[dx]);
                PRINT_TO_LOG("Dsize: (%d, %d)\n", dsize.width / cn, dsize.height);
                PRINT_TO_LOG("Ssize: (%d, %d)\n", src.cols, src.rows);

                double scale_x = static_cast<double>(ssize.width) / dsize.width;
                double scale_y = static_cast<double>(ssize.height) / dsize.height;
                bool area_fast = interpolation == INTER_AREA &&
                    fabs(scale_x - cvRound(scale_x)) < FLT_EPSILON &&
                    fabs(scale_y - cvRound(scale_y)) < FLT_EPSILON;
                if (area_fast)
                {
                    scale_y = cvRound(scale_y);
                    scale_x = cvRound(scale_x);
                }

                PRINT_TO_LOG("Interpolation: %s\n", interpolation_to_string(area_fast ? INTER_LANCZOS4 + 1 : interpolation));
                PRINT_TO_LOG("Scale (x, y): (%lf, %lf)\n", scale_x, scale_y);
                PRINT_TO_LOG("Elemsize: %d\n", src.elemSize1());
                PRINT_TO_LOG("Channels: %d\n", cn);

#ifdef SHOW_IMAGE
                const std::string w1("OpenCV impl (run func)"), w2("Reference func"), w3("Src image"), w4("Diff");
                namedWindow(w1, CV_WINDOW_KEEPRATIO);
                namedWindow(w2, CV_WINDOW_KEEPRATIO);
                namedWindow(w3, CV_WINDOW_KEEPRATIO);
                namedWindow(w4, CV_WINDOW_KEEPRATIO);

                Mat diff;
                absdiff(reference_dst, _dst, diff);

                imshow(w1, dst);
                imshow(w2, reference_dst);
                imshow(w3, src);
                imshow(w4, diff);

                waitKey();
#endif

                const int radius = 3;
                int rmin = MAX(dy - radius, 0), rmax = MIN(dy + radius, dsize.height);
                int cmin = MAX(dx / cn - radius, 0), cmax = MIN(dx / cn + radius, dsize.width);

                std::cout << "opencv result:\n" << dst(Range(rmin, rmax), Range(cmin, cmax)) << std::endl;
                std::cout << "reference result:\n" << reference_dst(Range(rmin, rmax), Range(cmin, cmax)) << std::endl;

                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
    }
}

void CV_ImageWarpBaseTest::prepare_test_data_for_reference_func()
{
    if (src.depth() != CV_32F)
    {
        Mat tmp;
        src.convertTo(tmp, CV_32F);
        src = tmp;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resize
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_Resize_Test :
    public CV_ImageWarpBaseTest
{
public:
    CV_Resize_Test();
    virtual ~CV_Resize_Test();

protected:
    virtual void generate_test_data();

    virtual void run_func();
    virtual void run_reference_func();

private:
    double scale_x;
    double scale_y;
    bool area_fast;

    void resize_generic();
    void resize_area();
    double getWeight(double a, double b, int x);

    typedef std::vector<std::pair<int, double> > dim;
    void generate_buffer(double scale, dim& _dim);
    void resize_1d(const Mat& _src, Mat& _dst, int dy, const dim& _dim);
};

CV_Resize_Test::CV_Resize_Test() :
    CV_ImageWarpBaseTest(), scale_x(),
    scale_y(), area_fast(false)
{
}

CV_Resize_Test::~CV_Resize_Test()
{
}

namespace
{
    void interpolateLinear(float x, float* coeffs)
    {
        coeffs[0] = 1.f - x;
        coeffs[1] = x;
    }

    void interpolateCubic(float x, float* coeffs)
    {
        const float A = -0.75f;

        coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
        coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
        coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
        coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
    }

    void interpolateLanczos4(float x, float* coeffs)
    {
        static const double s45 = 0.70710678118654752440084436210485;
        static const double cs[][2]=
        {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

        if( x < FLT_EPSILON )
        {
            for( int i = 0; i < 8; i++ )
                coeffs[i] = 0;
            coeffs[3] = 1;
            return;
        }

        float sum = 0;
        double y0=-(x+3)*CV_PI*0.25, s0 = sin(y0), c0=cos(y0);
        for(int i = 0; i < 8; i++ )
        {
            double y = -(x+3-i)*CV_PI*0.25;
            coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
            sum += coeffs[i];
        }

        sum = 1.f/sum;
        for(int i = 0; i < 8; i++ )
            coeffs[i] *= sum;
    }

    typedef void (*interpolate_method)(float x, float* coeffs);
    interpolate_method inter_array[] = { &interpolateLinear, &interpolateCubic, &interpolateLanczos4 };
}

void CV_Resize_Test::generate_test_data()
{
    CV_ImageWarpBaseTest::generate_test_data();

    scale_x = src.cols / static_cast<double>(dst.cols);
    scale_y = src.rows / static_cast<double>(dst.rows);

    area_fast = interpolation == INTER_AREA &&
        fabs(scale_x - cvRound(scale_x)) < FLT_EPSILON &&
        fabs(scale_y - cvRound(scale_y)) < FLT_EPSILON;
    if (area_fast)
    {
        scale_x = cvRound(scale_x);
        scale_y = cvRound(scale_y);
    }
}

void CV_Resize_Test::run_func()
{
    cv::resize(src, dst, dst.size(), 0, 0, interpolation);
}

void CV_Resize_Test::run_reference_func()
{
    CV_ImageWarpBaseTest::prepare_test_data_for_reference_func();

    if (interpolation == INTER_AREA)
        resize_area();
    else
        resize_generic();
}

double CV_Resize_Test::getWeight(double a, double b, int x)
{
    double w = std::min(static_cast<double>(x + 1), b) - std::max(static_cast<double>(x), a);
    CV_Assert(w >= 0);
    return w;
}

void CV_Resize_Test::resize_area()
{
    Size ssize = src.size(), dsize = reference_dst.size();
    CV_Assert(ssize.area() > 0 && dsize.area() > 0);
    int cn = src.channels();

    CV_Assert(scale_x >= 1.0 && scale_y >= 1.0);

    double fsy0 = 0, fsy1 = scale_y;
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        float* yD = reference_dst.ptr<float>(dy);
        int isy0 = cvFloor(fsy0), isy1 = std::min(cvFloor(fsy1), ssize.height - 1);
        CV_Assert(isy1 <= ssize.height && isy0 < ssize.height);

        double fsx0 = 0, fsx1 = scale_x;

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            float* xyD = yD + cn * dx;
            int isx0 = cvFloor(fsx0), isx1 = std::min(ssize.width - 1, cvFloor(fsx1));

            CV_Assert(isx1 <= ssize.width);
            CV_Assert(isx0 < ssize.width);

            // for each pixel of dst
            for (int r = 0; r < cn; ++r)
            {
                xyD[r] = 0.0f;
                double area = 0.0;
                for (int sy = isy0; sy <= isy1; ++sy)
                {
                    const float* yS = src.ptr<float>(sy);
                    for (int sx = isx0; sx <= isx1; ++sx)
                    {
                        double wy = getWeight(fsy0, fsy1, sy);
                        double wx = getWeight(fsx0, fsx1, sx);
                        double w = wx * wy;
                        xyD[r] += static_cast<float>(yS[sx * cn + r] * w);
                        area += w;
                    }
                }

                CV_Assert(area != 0);
                // norming pixel
                xyD[r] = static_cast<float>(xyD[r] / area);
            }
            fsx1 = std::min((fsx0 = fsx1) + scale_x, static_cast<double>(ssize.width));
        }
        fsy1 = std::min((fsy0 = fsy1) + scale_y, static_cast<double>(ssize.height));
    }
}

// for interpolation type : INTER_LINEAR, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
void CV_Resize_Test::resize_1d(const Mat& _src, Mat& _dst, int dy, const dim& _dim)
{
    Size dsize = _dst.size();
    int cn = _dst.channels();
    float* yD = _dst.ptr<float>(dy);

    if (interpolation == INTER_NEAREST)
    {
        const float* yS = _src.ptr<float>(dy);
        for (int dx = 0; dx < dsize.width; ++dx)
        {
            int isx = _dim[dx].first;
            const float* xyS = yS + isx * cn;
            float* xyD = yD + dx * cn;

            for (int r = 0; r < cn; ++r)
                xyD[r] = xyS[r];
        }
    }
    else if (interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_LANCZOS4)
    {
        interpolate_method inter_func = inter_array[interpolation - (interpolation == INTER_LANCZOS4 ? 2 : 1)];
        size_t elemsize = _src.elemSize();

        int ofs = 0, ksize = 2;
        if (interpolation == INTER_CUBIC)
            ofs = 1, ksize = 4;
        else if (interpolation == INTER_LANCZOS4)
            ofs = 3, ksize = 8;

        Mat _extended_src_row(1, _src.cols + ksize * 2, _src.type());
        uchar* srow = _src.data + dy * _src.step;
        memcpy(_extended_src_row.data + elemsize * ksize, srow, _src.step);
        for (int k = 0; k < ksize; ++k)
        {
            memcpy(_extended_src_row.data + k * elemsize, srow, elemsize);
            memcpy(_extended_src_row.data + (ksize + k) * elemsize + _src.step, srow + _src.step - elemsize, elemsize);
        }

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            int isx = _dim[dx].first;
            double fsx = _dim[dx].second;

            float *xyD = yD + dx * cn;
            const float* xyS = _extended_src_row.ptr<float>(0) + (isx + ksize - ofs) * cn;

            float w[8];
            inter_func(static_cast<float>(fsx), w);

            for (int r = 0; r < cn; ++r)
            {
                xyD[r] = 0;
                for (int k = 0; k < ksize; ++k)
                    xyD[r] += w[k] * xyS[k * cn + r];
                xyD[r] = xyD[r];
            }
        }
    }
    else
        CV_Assert(0);
}

void CV_Resize_Test::generate_buffer(double scale, dim& _dim)
{
    size_t length = _dim.size();
    for (size_t dx = 0; dx < length; ++dx)
    {
        double fsx = scale * (dx + 0.5) - 0.5;
        int isx = cvFloor(fsx);
        _dim[dx] = std::make_pair(isx, fsx - isx);
    }
}

void CV_Resize_Test::resize_generic()
{
    Size dsize = reference_dst.size(), ssize = src.size();
    CV_Assert(dsize.area() > 0 && ssize.area() > 0);

    dim dims[] = { dim(dsize.width), dim(dsize.height) };
    if (interpolation == INTER_NEAREST)
    {
        for (int dx = 0; dx < dsize.width; ++dx)
            dims[0][dx].first = std::min(cvFloor(dx * scale_x), ssize.width - 1);
        for (int dy = 0; dy < dsize.height; ++dy)
            dims[1][dy].first = std::min(cvFloor(dy * scale_y), ssize.height - 1);
    }
    else
    {
        generate_buffer(scale_x, dims[0]);
        generate_buffer(scale_y, dims[1]);
    }

    Mat tmp(ssize.height, dsize.width, reference_dst.type());
    for (int dy = 0; dy < tmp.rows; ++dy)
        resize_1d(src, tmp, dy, dims[0]);

    transpose(tmp, tmp);
    transpose(reference_dst, reference_dst);

    for (int dy = 0; dy < tmp.rows; ++dy)
        resize_1d(tmp, reference_dst, dy, dims[1]);
    transpose(reference_dst, reference_dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// remap
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_Remap_Test :
    public CV_ImageWarpBaseTest
{
public:
    CV_Remap_Test();

    virtual ~CV_Remap_Test();

private:
    typedef void (CV_Remap_Test::*remap_func)(const Mat&, Mat&);

protected:
    virtual void generate_test_data();
    virtual void prepare_test_data_for_reference_func();

    virtual void run_func();
    virtual void run_reference_func();

    Mat mapx, mapy;
    int borderType;
    Scalar borderValue;

    remap_func funcs[2];

private:
    void remap_nearest(const Mat&, Mat&);
    void remap_generic(const Mat&, Mat&);

    void convert_maps();
    const char* borderType_to_string() const;
    virtual void validate_results() const;
};

CV_Remap_Test::CV_Remap_Test() :
    CV_ImageWarpBaseTest(), mapx(), mapy(),
    borderType(-1), borderValue()
{
    funcs[0] = &CV_Remap_Test::remap_nearest;
    funcs[1] = &CV_Remap_Test::remap_generic;
}

CV_Remap_Test::~CV_Remap_Test()
{
}

void CV_Remap_Test::generate_test_data()
{
    CV_ImageWarpBaseTest::generate_test_data();

    RNG& rng = ts->get_rng();
    borderType = rng.uniform(1, BORDER_WRAP);
    borderValue = Scalar::all(rng.uniform(0, 255));

    // generating the mapx, mapy matrices
    static const int mapx_types[] = { CV_16SC2, CV_32FC1, CV_32FC2 };
    mapx.create(dst.size(), mapx_types[rng.uniform(0, sizeof(mapx_types) / sizeof(int))]);
    mapy = Mat();

    const int n = std::min(std::min(src.cols, src.rows) / 10 + 1, 2);
    float _n = 0; //static_cast<float>(-n);

    switch (mapx.type())
    {
        case CV_16SC2:
        {
            MatIterator_<Vec2s> begin_x = mapx.begin<Vec2s>(), end_x = mapx.end<Vec2s>();
            for ( ; begin_x != end_x; ++begin_x)
            {
                (*begin_x)[0] = static_cast<short>(rng.uniform(static_cast<int>(_n), std::max(src.cols + n - 1, 0)));
                (*begin_x)[1] = static_cast<short>(rng.uniform(static_cast<int>(_n), std::max(src.rows + n - 1, 0)));
            }

            if (interpolation != INTER_NEAREST)
            {
                static const int mapy_types[] = { CV_16UC1, CV_16SC1 };
                mapy.create(dst.size(), mapy_types[rng.uniform(0, sizeof(mapy_types) / sizeof(int))]);

                switch (mapy.type())
                {
                    case CV_16UC1:
                    {
                        MatIterator_<ushort> begin_y = mapy.begin<ushort>(), end_y = mapy.end<ushort>();
                        for ( ; begin_y != end_y; ++begin_y)
                            begin_y[0] = static_cast<short>(rng.uniform(0, 1024));
                    }
                    break;

                    case CV_16SC1:
                    {
                        MatIterator_<short> begin_y = mapy.begin<short>(), end_y = mapy.end<short>();
                        for ( ; begin_y != end_y; ++begin_y)
                            begin_y[0] = static_cast<short>(rng.uniform(0, 1024));
                    }
                    break;
                }
            }
        }
        break;

        case CV_32FC1:
        {
            mapy.create(dst.size(), CV_32FC1);
            float fscols = static_cast<float>(std::max(src.cols - 1 + n, 0)),
                    fsrows = static_cast<float>(std::max(src.rows - 1 + n, 0));
            MatIterator_<float> begin_x = mapx.begin<float>(), end_x = mapx.end<float>();
            MatIterator_<float> begin_y = mapy.begin<float>();
            for ( ; begin_x != end_x; ++begin_x, ++begin_y)
            {
                begin_x[0] = rng.uniform(_n, fscols);
                begin_y[0] = rng.uniform(_n, fsrows);
            }
        }
        break;

        case CV_32FC2:
        {
            MatIterator_<Vec2f> begin_x = mapx.begin<Vec2f>(), end_x = mapx.end<Vec2f>();
            float fscols = static_cast<float>(std::max(src.cols - 1 + n, 0)),
                    fsrows = static_cast<float>(std::max(src.rows - 1 + n, 0));
            for ( ; begin_x != end_x; ++begin_x)
            {
                begin_x[0] = rng.uniform(_n, fscols);
                begin_x[1] = rng.uniform(_n, fsrows);
            }
        }
        break;

        default:
            assert(0);
        break;
    }
}

void CV_Remap_Test::run_func()
{
    remap(src, dst, mapx, mapy, interpolation, borderType, borderValue);
}

void CV_Remap_Test::convert_maps()
{
    if (mapx.type() != CV_16SC2)
        convertMaps(mapx.clone(), mapy.clone(), mapx, mapy, CV_16SC2, interpolation == INTER_NEAREST);
    else if (interpolation != INTER_NEAREST)
        if (mapy.type() != CV_16UC1)
            mapy.clone().convertTo(mapy, CV_16UC1);

    if (interpolation == INTER_NEAREST)
        mapy = Mat();
    CV_Assert(((interpolation == INTER_NEAREST && !mapy.data) || mapy.type() == CV_16UC1 ||
               mapy.type() == CV_16SC1) && mapx.type() == CV_16SC2);
}

const char* CV_Remap_Test::borderType_to_string() const
{
    if (borderType == BORDER_CONSTANT)
        return "BORDER_CONSTANT";
    if (borderType == BORDER_REPLICATE)
        return "BORDER_REPLICATE";
    if (borderType == BORDER_REFLECT)
        return "BORDER_REFLECT";
    return "Unsupported/Unkown border type";
}

void CV_Remap_Test::prepare_test_data_for_reference_func()
{
    CV_ImageWarpBaseTest::prepare_test_data_for_reference_func();
    convert_maps();
/*
    const int ksize = 3;
    Mat kernel = getStructuringElement(CV_MOP_ERODE, Size(ksize, ksize));
    Mat mask(src.size(), CV_8UC1, Scalar::all(255)), dst_mask;
    cv::erode(src, erode_src, kernel);
    cv::erode(mask, dst_mask, kernel, Point(-1, -1), 1, BORDER_CONSTANT, Scalar::all(0));
    bitwise_not(dst_mask, mask);
    src.copyTo(erode_src, mask);
    dst_mask.release();

    mask = Scalar::all(0);
    kernel = getStructuringElement(CV_MOP_DILATE, kernel.size());
    cv::dilate(src, dilate_src, kernel);
    cv::dilate(mask, dst_mask, kernel, Point(-1, -1), 1, BORDER_CONSTANT, Scalar::all(255));
    src.copyTo(dilate_src, dst_mask);
    dst_mask.release();
*/
}

void CV_Remap_Test::run_reference_func()
{
    prepare_test_data_for_reference_func();

    if (interpolation == INTER_AREA)
        interpolation = INTER_LINEAR;

    int index = interpolation == INTER_NEAREST ? 0 : 1;
    (this->*funcs[index])(src, reference_dst);
}

void CV_Remap_Test::remap_nearest(const Mat& _src, Mat& _dst)
{
    CV_Assert(_src.depth() == CV_32F && _dst.type() == _src.type());
    CV_Assert(mapx.type() == CV_16SC2 && !mapy.data);

    Size ssize = _src.size(), dsize = _dst.size();
    CV_Assert(ssize.area() > 0 && dsize.area() > 0);
    int cn = _src.channels();

    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const short* yM = mapx.ptr<short>(dy);
        float* yD = _dst.ptr<float>(dy);

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            float* xyD = yD + cn * dx;
            int sx = yM[dx * 2], sy = yM[dx * 2 + 1];

            if (sx >= 0 && sx < ssize.width && sy >= 0 && sy < ssize.height)
            {
                const float *xyS = _src.ptr<float>(sy) + sx * cn;

                for (int r = 0; r < cn; ++r)
                    xyD[r] = xyS[r];
            }
            else if (borderType != BORDER_TRANSPARENT)
            {
                if (borderType == BORDER_CONSTANT)
                    for (int r = 0; r < cn; ++r)
                        xyD[r] = saturate_cast<float>(borderValue[r]);
                else
                {
                    sx = borderInterpolate(sx, ssize.width, borderType);
                    sy = borderInterpolate(sy, ssize.height, borderType);
                    CV_Assert(sx >= 0 && sy >= 0 && sx < ssize.width && sy < ssize.height);

                    const float *xyS = _src.ptr<float>(sy) + sx * cn;

                    for (int r = 0; r < cn; ++r)
                        xyD[r] = xyS[r];
                }
            }
        }
    }
}

void CV_Remap_Test::remap_generic(const Mat& _src, Mat& _dst)
{
    CV_Assert(mapx.type() == CV_16SC2 && mapy.type() == CV_16UC1);

    int ksize = 2;
    if (interpolation == INTER_CUBIC)
        ksize = 4;
    else if (interpolation == INTER_LANCZOS4)
        ksize = 8;
    else if (interpolation != INTER_LINEAR)
        assert(0);
    int ofs = (ksize / 2) - 1;

    CV_Assert(_src.depth() == CV_32F && _dst.type() == _src.type());
    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels(), width1 = std::max(ssize.width - ksize + 1, 0),
        height1 = std::max(ssize.height - ksize + 1, 0);

    float ix[8], w[16];
    interpolate_method inter_func = inter_array[interpolation - (interpolation == INTER_LANCZOS4 ? 2 : 1)];

    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const short* yMx = mapx.ptr<short>(dy);
        const ushort* yMy = mapy.ptr<ushort>(dy);

        float* yD = _dst.ptr<float>(dy);

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            float* xyD = yD + dx * cn;
            float sx = yMx[dx * 2], sy = yMx[dx * 2 + 1];
            int isx = cvFloor(sx), isy = cvFloor(sy);

            inter_func((yMy[dx] & (INTER_TAB_SIZE - 1)) / static_cast<float>(INTER_TAB_SIZE), w);
            inter_func(((yMy[dx] >> INTER_BITS) & (INTER_TAB_SIZE - 1)) / static_cast<float>(INTER_TAB_SIZE), w + ksize);

            isx -= ofs;
            isy -= ofs;

            if (isx >= 0 && isx < width1 && isy >= 0 && isy < height1)
            {
                for (int r = 0; r < cn; ++r)
                {
                    for (int y = 0; y < ksize; ++y)
                    {
                        const float* xyS = _src.ptr<float>(isy + y) + isx * cn;

                        ix[y] = 0;
                        for (int i = 0; i < ksize; ++i)
                            ix[y] += w[i] * xyS[i * cn + r];
                    }
                    xyD[r] = 0;
                    for (int i = 0; i < ksize; ++i)
                        xyD[r] += w[ksize + i] * ix[i];
                }
            }
            else if (borderType != BORDER_TRANSPARENT)
            {
                int ar_x[8], ar_y[8];

                for (int k = 0; k < ksize; k++)
                {
                    ar_x[k] = borderInterpolate(isx + k, ssize.width, borderType) * cn;
                    ar_y[k] = borderInterpolate(isy + k, ssize.height, borderType);
                }

                for (int r = 0; r < cn; r++)
                {
                    xyD[r] = 0;
                    for (int i = 0; i < ksize; ++i)
                    {
                        ix[i] = 0;
                        if (ar_y[i] >= 0)
                        {
                            const float* yS = _src.ptr<float>(ar_y[i]);
                            for (int j = 0; j < ksize; ++j)
                                ix[i] += saturate_cast<float>((ar_x[j] >= 0 ? yS[ar_x[j] + r] : borderValue[r]) * w[j]);
                        }
                        else
                            for (int j = 0; j < ksize; ++j)
                                ix[i] += saturate_cast<float>(borderValue[r] * w[j]);
                    }
                    for (int i = 0; i < ksize; ++i)
                        xyD[r] += saturate_cast<float>(w[ksize + i] * ix[i]);
                }
            }
        }
    }
}

void CV_Remap_Test::validate_results() const
{
    CV_ImageWarpBaseTest::validate_results();
    if (cvtest::TS::ptr()->get_err_code() == cvtest::TS::FAIL_BAD_ACCURACY)
    {
        PRINT_TO_LOG("BorderType: %s\n", borderType_to_string());
        PRINT_TO_LOG("BorderValue: (%f, %f, %f, %f)\n",
                     borderValue[0], borderValue[1], borderValue[2], borderValue[3]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// warpAffine
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_WarpAffine_Test :
    public CV_Remap_Test
{
public:
    CV_WarpAffine_Test();

    virtual ~CV_WarpAffine_Test();

protected:
    virtual void generate_test_data();
    virtual void prepare_test_data_for_reference_func();

    virtual void run_func();
    virtual void run_reference_func();

    Mat M;
private:
    void warpAffine(const Mat&, Mat&);
};

CV_WarpAffine_Test::CV_WarpAffine_Test() :
    CV_Remap_Test()
{
}

CV_WarpAffine_Test::~CV_WarpAffine_Test()
{
}

void CV_WarpAffine_Test::generate_test_data()
{
    CV_Remap_Test::generate_test_data();

    RNG& rng = ts->get_rng();

    // generating the M 2x3 matrix
    static const int depths[] = { CV_32FC1, CV_64FC1 };

    // generating 2d matrix
    M = getRotationMatrix2D(Point2f(src.cols / 2.f, src.rows / 2.f),
        rng.uniform(-180.f, 180.f), rng.uniform(0.4f, 2.0f));
    int depth = depths[rng.uniform(0, sizeof(depths) / sizeof(depths[0]))];
    if (M.depth() != depth)
    {
        Mat tmp;
        M.convertTo(tmp, depth);
        M = tmp;
    }

    // warp_matrix is inverse
    if (rng.uniform(0., 1.) > 0)
        interpolation |= CV_WARP_INVERSE_MAP;
}

void CV_WarpAffine_Test::run_func()
{
    cv::warpAffine(src, dst, M, dst.size(), interpolation, borderType, borderValue);
}

void CV_WarpAffine_Test::prepare_test_data_for_reference_func()
{
    CV_ImageWarpBaseTest::prepare_test_data_for_reference_func();
}

void CV_WarpAffine_Test::run_reference_func()
{
    prepare_test_data_for_reference_func();

    warpAffine(src, reference_dst);
}

void CV_WarpAffine_Test::warpAffine(const Mat& _src, Mat& _dst)
{
    Size dsize = _dst.size();

    CV_Assert(_src.size().area() > 0);
    CV_Assert(dsize.area() > 0);
    CV_Assert(_src.type() == _dst.type());

    Mat tM;
    M.convertTo(tM, CV_64F);

    int inter = interpolation & INTER_MAX;
    if (inter == INTER_AREA)
        inter = INTER_LINEAR;

    mapx.create(dsize, CV_16SC2);
    if (inter != INTER_NEAREST)
        mapy.create(dsize, CV_16SC1);
    else
        mapy = Mat();

    if (!(interpolation & CV_WARP_INVERSE_MAP))
        invertAffineTransform(tM.clone(), tM);

    const int AB_BITS = MAX(10, (int)INTER_BITS);
    const int AB_SCALE = 1 << AB_BITS;
    int round_delta = (inter == INTER_NEAREST) ? AB_SCALE / 2 : (AB_SCALE / INTER_TAB_SIZE / 2);

    const double* data_tM = tM.ptr<double>(0);
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        short* yM = mapx.ptr<short>(dy);
        for (int dx = 0; dx < dsize.width; ++dx, yM += 2)
        {
            int v1 = saturate_cast<int>(saturate_cast<int>(data_tM[0] * dx * AB_SCALE) +
                    saturate_cast<int>((data_tM[1] * dy + data_tM[2]) * AB_SCALE) + round_delta),
                   v2 = saturate_cast<int>(saturate_cast<int>(data_tM[3] * dx * AB_SCALE) +
                    saturate_cast<int>((data_tM[4] * dy + data_tM[5]) * AB_SCALE) + round_delta);
            v1 >>= AB_BITS - INTER_BITS;
            v2 >>= AB_BITS - INTER_BITS;

            yM[0] = saturate_cast<short>(v1 >> INTER_BITS);
            yM[1] = saturate_cast<short>(v2 >> INTER_BITS);

            if (inter != INTER_NEAREST)
                mapy.ptr<short>(dy)[dx] = ((v2 & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (v1 & (INTER_TAB_SIZE - 1)));
        }
    }

    CV_Assert(mapx.type() == CV_16SC2 && ((inter == INTER_NEAREST && !mapy.data) || mapy.type() == CV_16SC1));
    cv::remap(_src, _dst, mapx, mapy, inter, borderType, borderValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// warpPerspective
////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_WarpPerspective_Test :
    public CV_WarpAffine_Test
{
public:
    CV_WarpPerspective_Test();

    virtual ~CV_WarpPerspective_Test();

protected:
    virtual void generate_test_data();

    virtual void run_func();
    virtual void run_reference_func();

private:
    void warpPerspective(const Mat&, Mat&);
};

CV_WarpPerspective_Test::CV_WarpPerspective_Test() :
    CV_WarpAffine_Test()
{
}

CV_WarpPerspective_Test::~CV_WarpPerspective_Test()
{
}

void CV_WarpPerspective_Test::generate_test_data()
{
    CV_Remap_Test::generate_test_data();

    // generating the M 3x3 matrix
    RNG& rng = ts->get_rng();

    float cols = static_cast<float>(src.cols), rows = static_cast<float>(src.rows);
    Point2f sp[] = { Point2f(0.0f, 0.0f), Point2f(cols, 0.0f), Point2f(0.0f, rows), Point2f(cols, rows) };
    Point2f dp[] = { Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)),
        Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)),
        Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)),
        Point2f(rng.uniform(0.0f, cols), rng.uniform(0.0f, rows)) };
    M = getPerspectiveTransform(sp, dp);

    static const int depths[] = { CV_32F, CV_64F };
    int depth = depths[rng.uniform(0, 2)];
    M.clone().convertTo(M, depth);
}

void CV_WarpPerspective_Test::run_func()
{
    cv::warpPerspective(src, dst, M, dst.size(), interpolation, borderType, borderValue);
}

void CV_WarpPerspective_Test::run_reference_func()
{
    prepare_test_data_for_reference_func();

    warpPerspective(src, reference_dst);
}

void CV_WarpPerspective_Test::warpPerspective(const Mat& _src, Mat& _dst)
{
    Size ssize = _src.size(), dsize = _dst.size();

    CV_Assert(ssize.area() > 0);
    CV_Assert(dsize.area() > 0);
    CV_Assert(_src.type() == _dst.type());

    if (M.depth() != CV_64F)
    {
        Mat tmp;
        M.convertTo(tmp, CV_64F);
        M = tmp;
    }

    if (!(interpolation & CV_WARP_INVERSE_MAP))
    {
        Mat tmp;
        invert(M, tmp);
        M = tmp;
    }

    int inter = interpolation & INTER_MAX;
    if (inter == INTER_AREA)
        inter = INTER_LINEAR;

    mapx.create(dsize, CV_16SC2);
    if (inter != INTER_NEAREST)
        mapy.create(dsize, CV_16SC1);
    else
        mapy = Mat();

    double* tM = M.ptr<double>(0);
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        short* yMx = mapx.ptr<short>(dy);

        for (int dx = 0; dx < dsize.width; ++dx, yMx += 2)
        {
            double den = tM[6] * dx + tM[7] * dy + tM[8];
            den = den ? 1.0 / den : 0.0;

            if (inter == INTER_NEAREST)
            {
                yMx[0] = saturate_cast<short>((tM[0] * dx + tM[1] * dy + tM[2]) * den);
                yMx[1] = saturate_cast<short>((tM[3] * dx + tM[4] * dy + tM[5]) * den);
                continue;
            }

            den *= INTER_TAB_SIZE;
            int v0 = saturate_cast<int>((tM[0] * dx + tM[1] * dy + tM[2]) * den);
            int v1 = saturate_cast<int>((tM[3] * dx + tM[4] * dy + tM[5]) * den);

            yMx[0] = saturate_cast<short>(v0 >> INTER_BITS);
            yMx[1] = saturate_cast<short>(v1 >> INTER_BITS);
            mapy.ptr<short>(dy)[dx] = saturate_cast<short>((v1 & (INTER_TAB_SIZE - 1)) *
                    INTER_TAB_SIZE + (v0 & (INTER_TAB_SIZE - 1)));
        }
    }

    CV_Assert(mapx.type() == CV_16SC2 && ((inter == INTER_NEAREST && !mapy.data) || mapy.type() == CV_16SC1));
    cv::remap(_src, _dst, mapx, mapy, inter, borderType, borderValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Imgproc_Resize_Test, accuracy) { CV_Resize_Test test; test.safe_run(); }
TEST(Imgproc_Remap_Test, accuracy) { CV_Remap_Test test; test.safe_run(); }
TEST(Imgproc_WarpAffine_Test, accuracy) { CV_WarpAffine_Test test; test.safe_run(); }
TEST(Imgproc_WarpPerspective_Test, accuracy) { CV_WarpPerspective_Test test; test.safe_run(); }
