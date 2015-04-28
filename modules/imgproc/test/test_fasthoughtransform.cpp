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
// Copyright (C) 2015, Smart Engines Ltd, all rights reserved.
// Copyright (C) 2015, Institute for Information Transmission Problems of the Russian Academy of Sciences (Kharkevich Institute), all rights reserved.
// Copyright (C) 2015, Dmitry Nikolaev, Simon Karpenko, Michail Aliev, Elena Kuznetsova, all rights reserved.
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

#include "test_precomp.hpp"
#include "fast_hough_transform.hpp"

using namespace cv;
using namespace std;

class CV_FastHoughTransformTest : public cvtest::BaseTest
{
public:
    CV_FastHoughTransformTest();

private:
    int read_params(CvFileStorage *fs);

    int prepare_test_case(int test_case_idx);
    int prepare_mat_canvas(int mask = cv::DEPTH_MASK_ALL);
    template<typename T>
    pair<Point, double> put_random_point(Mat &img);
    template<typename T>
    int put_random_points();

    void run_func();

    int validate_test_results(int test_case_idx);
    template <typename T>
    int validate_test_results();
    template <typename T>
    int validate_sum(T const &ref_sum, Mat const &fht);
    template <typename T>
    int validate_point(Mat const &fht);
    template <typename T>
    int validate_pair(Point const& f, Point const& s, Mat const& fht);

    int get_test_case_count() { return test_case_count; }

private:
    Mat test_mat;
    Mat fht_mat;

    int fht_anglerange;
    int fht_operatrion;
    int fht_makeskew;

    int test_case_count;

    int max_channels;
    int min_log_array_size;
    int max_log_array_size;

    std::vector<std::vector<pair<Point, double> > > ref_pts;
};

///////////////////////////////////////////////////////////////////////////////
//  random utils
template <typename T> struct Epsilon {
    T operator()() { return saturate_cast<T>(1); }
};
template <> struct Epsilon<float> {
    float operator()() { return 1e-3; }
};
template <> struct Epsilon<double> {
    double operator()() { return 1e-6; }
};

template <typename T> struct Rand {
    T operator ()(T const &min_value, T const &max_value) {
        RNG& rng = cvtest::TS::ptr()->get_rng();
        return saturate_cast<T>(rng.uniform(min_value,
                                            saturate_cast<T>(max_value +
                                                             Epsilon<T>()())));
    }
};
template <> struct Rand<bool> {
    bool operator ()(bool const &/*min_value = false*/,
                     bool const &/*max_value = true*/) {
        RNG& rng = cvtest::TS::ptr()->get_rng();
        return rng.uniform(0, 2) ? true : false;
    }
};

Size rand_size(int min_size_log, int max_size_log) {
    double width_log  = Rand<double>()(min_size_log,
                                       max_size_log);
    double height_log = Rand<double>()(width_log - min_size_log,
                                       max_size_log - width_log);

    if (Rand<bool>()(false, true))
        swap(width_log, height_log);

    return Size(cvRound(exp(width_log)),
                cvRound(exp(height_log)));
}

template <typename T> struct TypDepth {};
#define SPEC_TD(Tp, Td)                                                       \
    template <> struct TypDepth<Tp> { static const int depth = Td; };
SPEC_TD(uchar,  CV_8UC1);
SPEC_TD(schar,  CV_8SC1);
SPEC_TD(ushort, CV_16UC1);
SPEC_TD(short,  CV_16SC1);
SPEC_TD(int,    CV_32SC1);
SPEC_TD(float,  CV_32FC1);
SPEC_TD(double, CV_64FC1);
#undef SPEC_TD

template <typename T> struct MinValue {
    T operator()() { return saturate_cast<T>(Epsilon<T>()()); }
};

template <typename T> struct MaxValue {
    T operator()() { return saturate_cast<T>(
                            cvtest::getMaxVal(TypDepth<T>::depth)); }
};
template <> struct MaxValue<int> {
    int operator()() { return 100000; }
};
template <> struct MaxValue<float> {
    float operator()() { return 10000.; }
};
template <> struct MaxValue<double> {
    double operator()() { return 10000.; }
};

template <typename T> struct Typ {
    static bool eq(const T& first, const T& second) {
        return first > second ?
               first - second < Epsilon<T>()() :
               second - first < Epsilon<T>()();
    }
    static T rand() {
        return Rand<T>()(MinValue<T>()(), MaxValue<T>()());
    }
    static T rand(T max_threshold) {
        return Rand<T>()(MinValue<T>()(), max_threshold);
    }
    static T rand(T min_threshold, T max_threshold) {
        if (min_threshold > max_threshold)
            swap(min_threshold, max_threshold);
        return Rand<T>()(min_threshold, max_threshold);
    }
};

static bool is_failed(int code)
{
    return code < 0;
}

///////////////////////////////////////////////////////////////////////////////
// CV_FastHoughTransformTest
CV_FastHoughTransformTest::CV_FastHoughTransformTest()
:   fht_anglerange(6)//ARO_315_135
,   fht_operatrion(2)//FHT_ADD
,   fht_makeskew(1)//HDO_DESKEW
{

}

int CV_FastHoughTransformTest::read_params(CvFileStorage *fs)
{
    int code = cvtest::BaseTest::read_params(fs);
    if (is_failed(code))
        return code;

    min_log_array_size = cvReadInt(find_param(fs, "min_log_array_size"), 1);
    max_log_array_size = cvReadInt(find_param(fs, "max_log_array_size"), 8);

    min_log_array_size = cvtest::clipInt(min_log_array_size, 1, 8);
    max_log_array_size = cvtest::clipInt(max_log_array_size,
                                         min_log_array_size, 8);

    max_channels = cvReadInt(find_param(fs, "max_channels"), 4);
    max_channels = cvtest::clipInt(max_channels, 1, 4);

    test_case_count = cvReadInt(find_param(fs, "test_case_count"), 5);
    test_case_count = cvRound(test_case_count*ts->get_test_case_count_scale());

    test_case_count = cvtest::clipInt(test_case_count, 0, 500);

    return code;
}
int CV_FastHoughTransformTest::prepare_mat_canvas(int mask)
{
    int code = cvtest::TS::OK;
    int const inputType = cvtest::randomType(cvtest::TS::ptr()->get_rng(),
                                             mask,
                                             1,
                                             max_channels);

    Size const inputSize = rand_size(min_log_array_size, max_log_array_size);
    test_mat = Mat::zeros(inputSize, inputType);

    return code;
}

template<typename T>
pair<Point, double> CV_FastHoughTransformTest::put_random_point(Mat &img)
{
    pair<Point, double> ret;
    ret.first = Point(Typ<int>::rand(0, img.cols - 1),
                      Typ<int>::rand(0, img.rows - 1));
    T value = Typ<T>::rand();
    ret.second = (double)value;

    img.at<T>(ret.first.y, ret.first.x) = value;
    return ret;
}

template<typename T>
int CV_FastHoughTransformTest::put_random_points()
{
    int code = cvtest::TS::OK;

    vector<Mat> test_mat_channels(test_mat.channels());
    split(test_mat, test_mat_channels);
    ref_pts.clear();
    ref_pts.resize(test_mat_channels.size());

    for (size_t c = 0; c < test_mat_channels.size(); ++c)
        for (int pi = 0; pi < Typ<int>::rand(1, 2); ++pi)
            ref_pts[c].push_back(put_random_point<T>(test_mat_channels[c]));

    merge(test_mat_channels, test_mat);

    return code;
}

int CV_FastHoughTransformTest::prepare_test_case(int test_case_idx)
{
    int code = BaseTest::prepare_test_case(test_case_idx);
    if (is_failed(code))
        return code;

    fht_makeskew = Rand<int>()(0, 1);

    code = prepare_mat_canvas();
    if (is_failed(code))
        return code;

    switch (test_mat.depth())
    {
    case CV_8U:
        code = put_random_points<uchar>();
        break;
    case CV_8S:
        code = put_random_points<schar>();
        break;
    case CV_16U:
        code = put_random_points<ushort>();
        break;
    case CV_16S:
        code = put_random_points<short>();
        break;
    case CV_32S:
        code = put_random_points<int>();
        break;
    case CV_32F:
        code = put_random_points<float>();
        break;
    case CV_64F:
        code = put_random_points<double>();
        break;
    default:
        code = cvtest::TS::FAIL_BAD_ARG_CHECK;
        break;
    }
    if (is_failed(code))
        return code;

    return 1;
};


void CV_FastHoughTransformTest::run_func()
{
    FastHoughTransform(test_mat,
                       fht_mat,
                       test_mat.depth(),
                       fht_anglerange,
                       fht_operatrion,
                       fht_makeskew);
}

template <typename T>
int CV_FastHoughTransformTest::validate_sum(T const &ref_sum, Mat const &fht)
{
    int code = cvtest::TS::OK;

    for (int y = 0; y < fht.rows; ++y)
    {
        T exp_sum = saturate_cast<T>(sum(fht.row(y))[0]);
        if (!Typ<T>::eq(exp_sum, ref_sum))
        {
            code = cvtest::TS::FAIL_BAD_ACCURACY;
            ts->printf(cvtest::TS::LOG,
                       "The sum of column #%d in the fast hough transform "
                        "result is inaccurate (=%g, should be =%g)\n",
                       y, static_cast<float>(exp_sum),
                       static_cast<float>(ref_sum));
            break;
        }
    }

    return code;
}

template <typename T>
int CV_FastHoughTransformTest::validate_point(Mat const &fht)
{
    int code = cvtest::TS::OK;

    for (int y = 0; y < fht.rows; ++y)
    {
        int cnt = countNonZero(fht.row(y));
        if (cnt != 1)
        {
            code = cvtest::TS::FAIL_BAD_ACCURACY;
            ts->printf(cvtest::TS::LOG,
                       "Failed to compute fast hough transform for single "
                       "point image (column %d)\n",
                       y);
            break;
        }
    }

    return code;
}

template <typename T>
int CV_FastHoughTransformTest::validate_pair(Point const& f, Point const& s,
                                             Mat const& fht)
{
    int code = cvtest::TS::OK;

    cv::Point max_fht(-1, -1);
    cv::minMaxLoc(fht, 0, 0, 0, &max_fht);
    Vec4i exp_line;
    HoughPoint2Line(exp_line, max_fht, test_mat, fht_anglerange,
                    fht_makeskew, RO_STRICT);

    double a = exp_line[1] - exp_line[3];
    double b = exp_line[2] - exp_line[0];
    double c = - (a * exp_line[0] + b * exp_line[1]);

    double fd = abs(f.x * a + f.y * b + c) / sqrt(a * a + b * b);
    double sd = abs(s.x * a + s.y * b + c) / sqrt(a * a + b * b);
    if (max(fd, sd) > saturate_cast<T>(2.0))
    {
        ts->printf(cvtest::TS::LOG,
                   "Failed to detect line (result is ((%d, %d), (%d, %d)), "
                    "should be ((%d, %d), (%d, %d))\n",
                    exp_line[0], exp_line[1], exp_line[2], exp_line[3],
                    f.x, f.y, s.x, s.y);
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    return code;
}

template<typename T>
int CV_FastHoughTransformTest::validate_test_results()
{
    int code = cvtest::TS::OK;

    vector<Mat> fht_channels(fht_mat.channels());
    split(fht_mat, fht_channels);

    for (int c = 0; c < test_mat.channels(); ++c)
    {
        T sum = 0;
        if (ref_pts[c].size() == 1 ||
            ref_pts[c][0].first == ref_pts[c][1].first)
        {
            sum = saturate_cast<T>(ref_pts[c].back().second);
            code = validate_point<T>(fht_channels[c]);
        }
        else
        {
            sum = saturate_cast<T>(ref_pts[c][0].second +
                                   ref_pts[c][1].second);
            code = validate_pair<T>(ref_pts[c][0].first,
                                    ref_pts[c][1].first,
                                    fht_channels[c]);
        }
        if (is_failed(code))
            break;

        code = validate_sum<T>(sum, fht_channels[c]);
        if (is_failed(code))
            break;
    }
    return code;
}

int CV_FastHoughTransformTest::validate_test_results(int test_case_idx)
{
    int code = BaseTest::validate_test_results(test_case_idx);
    if (is_failed(code))
        return code;

    switch (test_mat.depth())
    {
    case CV_8U:
        code = validate_test_results<uchar>();
        break;
    case CV_8S:
        code = validate_test_results<schar>();
        break;
    case CV_16U:
        code = validate_test_results<ushort>();
        break;
    case CV_16S:
        code = validate_test_results<short>();
        break;
    case CV_32S:
        code = validate_test_results<int>();
        break;
    case CV_32F:
        code = validate_test_results<float>();
        break;
    case CV_64F:
        code = validate_test_results<double>();
        break;
    default:
        code = cvtest::TS::FAIL_BAD_ARG_CHECK;
        break;
    }
    if (is_failed(code))
        ts->set_failed_test_info(code);

    return code;
};

///////////////////////////////////////////////////////////////////////////////

TEST(CV_FastHoughTransformTest, accuracy) {
    CV_FastHoughTransformTest test;
    test.safe_run();
}
