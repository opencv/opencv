// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"
#include "opencv2/imgproc/detail/legacy.hpp"

#define CHECK_OLD 1

namespace opencv_test { namespace {

// debug function
template <typename T>
inline static void print_pts(const T& c)
{
    for (const auto& one_pt : c)
    {
        cout << one_pt << " ";
    }
    cout << endl;
}

// debug function
template <typename T>
inline static void print_pts_2(vector<T>& cs)
{
    int cnt = 0;
    cout << "Contours:" << endl;
    for (const auto& one_c : cs)
    {
        cout << cnt++ << " : ";
        print_pts(one_c);
    }
};

// draw 1-2 px blob with orientation defined by 'kind'
template <typename T>
inline static void drawSmallContour(Mat& img, Point pt, int kind, int color_)
{
    const T color = static_cast<T>(color_);
    img.at<T>(pt) = color;
    switch (kind)
    {
        case 1: img.at<T>(pt + Point(1, 0)) = color; break;
        case 2: img.at<T>(pt + Point(1, -1)) = color; break;
        case 3: img.at<T>(pt + Point(0, -1)) = color; break;
        case 4: img.at<T>(pt + Point(-1, -1)) = color; break;
        case 5: img.at<T>(pt + Point(-1, 0)) = color; break;
        case 6: img.at<T>(pt + Point(-1, 1)) = color; break;
        case 7: img.at<T>(pt + Point(0, 1)) = color; break;
        case 8: img.at<T>(pt + Point(1, 1)) = color; break;
        default: break;
    }
}

inline static void drawContours(Mat& img,
                                const vector<vector<Point>>& contours,
                                const Scalar& color = Scalar::all(255))
{
    for (const auto& contour : contours)
    {
        for (size_t n = 0, end = contour.size(); n < end; ++n)
        {
            size_t m = n + 1;
            if (n == end - 1)
                m = 0;
            line(img, contour[m], contour[n], color, 1, LINE_8);
        }
    }
}

//==================================================================================================

// Test parameters - mode + method
typedef testing::TestWithParam<tuple<int, int>> Imgproc_FindContours_Modes1;


// Draw random rectangle and find contours
//
TEST_P(Imgproc_FindContours_Modes1, rectangle)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());

    const size_t ITER = 100;
    RNG rng = TS::ptr()->get_rng();

    for (size_t i = 0; i < ITER; ++i)
    {
        SCOPED_TRACE(cv::format("i=%zu", i));
        const Size sz(rng.uniform(640, 1920), rng.uniform(480, 1080));
        Mat img(sz, CV_8UC1, Scalar::all(0));
        Mat img32s(sz, CV_32SC1, Scalar::all(0));
        const Rect r(Point(rng.uniform(1, sz.width / 2 - 1), rng.uniform(1, sz.height / 2)),
                     Point(rng.uniform(sz.width / 2 - 1, sz.width - 1),
                           rng.uniform(sz.height / 2 - 1, sz.height - 1)));
        rectangle(img, r, Scalar::all(255));
        rectangle(img32s, r, Scalar::all(255), FILLED);

        const vector<Point> ext_ref {r.tl(),
                                     r.tl() + Point(0, r.height - 1),
                                     r.br() + Point(-1, -1),
                                     r.tl() + Point(r.width - 1, 0)};
        const vector<Point> int_ref {ext_ref[0] + Point(0, 1),
                                     ext_ref[0] + Point(1, 0),
                                     ext_ref[3] + Point(-1, 0),
                                     ext_ref[3] + Point(0, 1),
                                     ext_ref[2] + Point(0, -1),
                                     ext_ref[2] + Point(-1, 0),
                                     ext_ref[1] + Point(1, 0),
                                     ext_ref[1] + Point(0, -1)};
        const size_t ext_perimeter = r.width * 2 + r.height * 2;
        const size_t int_perimeter = ext_perimeter - 4;

        vector<vector<Point>> contours;
        vector<vector<schar>> chains;
        vector<Vec4i> hierarchy;

        // run functionn
        if (mode == RETR_FLOODFILL)
            if (method == 0)
                findContours(img32s, chains, hierarchy, mode, method);
            else
                findContours(img32s, contours, hierarchy, mode, method);
        else if (method == 0)
            findContours(img, chains, hierarchy, mode, method);
        else
            findContours(img, contours, hierarchy, mode, method);

        // verify results
        if (mode == RETR_EXTERNAL)
        {
            if (method == 0)
            {
                ASSERT_EQ(1U, chains.size());
            }
            else
            {
                ASSERT_EQ(1U, contours.size());
                if (method == CHAIN_APPROX_NONE)
                {
                    EXPECT_EQ(int_perimeter, contours[0].size());
                }
                else if (method == CHAIN_APPROX_SIMPLE)
                {
                    EXPECT_MAT_NEAR(Mat(ext_ref), Mat(contours[0]), 0);
                }
            }
        }
        else
        {
            if (method == 0)
            {
                ASSERT_EQ(2U, chains.size());
            }
            else
            {
                ASSERT_EQ(2U, contours.size());
                if (mode == RETR_LIST)
                {
                    if (method == CHAIN_APPROX_NONE)
                    {
                        EXPECT_EQ(int_perimeter - 4, contours[0].size());
                        EXPECT_EQ(int_perimeter, contours[1].size());
                    }
                    else if (method == CHAIN_APPROX_SIMPLE)
                    {
                        EXPECT_MAT_NEAR(Mat(int_ref), Mat(contours[0]), 0);
                        EXPECT_MAT_NEAR(Mat(ext_ref), Mat(contours[1]), 0);
                    }
                }
                else if (mode == RETR_CCOMP || mode == RETR_TREE)
                {
                    if (method == CHAIN_APPROX_NONE)
                    {
                        EXPECT_EQ(int_perimeter, contours[0].size());
                        EXPECT_EQ(int_perimeter - 4, contours[1].size());
                    }
                    else if (method == CHAIN_APPROX_SIMPLE)
                    {
                        EXPECT_MAT_NEAR(Mat(ext_ref), Mat(contours[0]), 0);
                        EXPECT_MAT_NEAR(Mat(int_ref), Mat(contours[1]), 0);
                    }
                }
                else if (mode == RETR_FLOODFILL)
                {
                    if (method == CHAIN_APPROX_NONE)
                    {
                        EXPECT_EQ(int_perimeter + 4, contours[0].size());
                    }
                    else if (method == CHAIN_APPROX_SIMPLE)
                    {
                        EXPECT_EQ(int_ref.size(), contours[0].size());
                        EXPECT_MAT_NEAR(Mat(ext_ref), Mat(contours[1]), 0);
                    }
                }
            }
        }

#if CHECK_OLD
        if (method != 0)  // old doesn't support chain codes
        {
            if (mode != RETR_FLOODFILL)
            {
                vector<vector<Point>> contours_o;
                vector<Vec4i> hierarchy_o;
                findContours_legacy(img, contours_o, hierarchy_o, mode, method);
                ASSERT_EQ(contours.size(), contours_o.size());
                for (size_t j = 0; j < contours.size(); ++j)
                {
                    SCOPED_TRACE(format("contour %zu", j));
                    EXPECT_MAT_NEAR(Mat(contours[j]), Mat(contours_o[j]), 0);
                }
                EXPECT_MAT_NEAR(Mat(hierarchy), Mat(hierarchy_o), 0);
            }
            else
            {
                vector<vector<Point>> contours_o;
                vector<Vec4i> hierarchy_o;
                findContours_legacy(img32s, contours_o, hierarchy_o, mode, method);
                ASSERT_EQ(contours.size(), contours_o.size());
                for (size_t j = 0; j < contours.size(); ++j)
                {
                    SCOPED_TRACE(format("contour %zu", j));
                    EXPECT_MAT_NEAR(Mat(contours[j]), Mat(contours_o[j]), 0);
                }
                EXPECT_MAT_NEAR(Mat(hierarchy), Mat(hierarchy_o), 0);
            }
        }
#endif
    }
}


// Draw many small 1-2px blobs and find contours
//
TEST_P(Imgproc_FindContours_Modes1, small)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());

    const size_t DIM = 1000;
    const Size sz(DIM, DIM);
    const int num = (DIM / 10) * (DIM / 10);  // number of 10x10 squares

    Mat img(sz, CV_8UC1, Scalar::all(0));
    Mat img32s(sz, CV_32SC1, Scalar::all(0));
    vector<Point> pts;
    int extra_contours_32s = 0;
    for (int j = 0; j < num; ++j)
    {
        const int kind = j % 9;
        Point pt {(j % 100) * 10 + 4, (j / 100) * 10 + 4};
        drawSmallContour<uchar>(img, pt, kind, 255);
        drawSmallContour<int>(img32s, pt, kind, j + 1);
        pts.push_back(pt);
        // NOTE: for some reason these small diagonal contours (NW, SE)
        //       result in 2 external contours for FLOODFILL mode
        if (kind == 8 || kind == 4)
            ++extra_contours_32s;
    }
    {
        vector<vector<Point>> contours;
        vector<vector<schar>> chains;
        vector<Vec4i> hierarchy;

        if (mode == RETR_FLOODFILL)
        {
            if (method == 0)
            {
                findContours(img32s, chains, hierarchy, mode, method);
                ASSERT_EQ(pts.size() * 2 + extra_contours_32s, chains.size());
            }
            else
            {
                findContours(img32s, contours, hierarchy, mode, method);
                ASSERT_EQ(pts.size() * 2 + extra_contours_32s, contours.size());
#if CHECK_OLD
                vector<vector<Point>> contours_o;
                vector<Vec4i> hierarchy_o;
                findContours_legacy(img32s, contours_o, hierarchy_o, mode, method);
                ASSERT_EQ(contours.size(), contours_o.size());
                for (size_t i = 0; i < contours.size(); ++i)
                {
                    SCOPED_TRACE(format("contour %zu", i));
                    EXPECT_MAT_NEAR(Mat(contours[i]), Mat(contours_o[i]), 0);
                }
                EXPECT_MAT_NEAR(Mat(hierarchy), Mat(hierarchy_o), 0);
#endif
            }
        }
        else
        {
            if (method == 0)
            {
                findContours(img, chains, hierarchy, mode, method);
                ASSERT_EQ(pts.size(), chains.size());
            }
            else
            {
                findContours(img, contours, hierarchy, mode, method);
                ASSERT_EQ(pts.size(), contours.size());
#if CHECK_OLD
                vector<vector<Point>> contours_o;
                vector<Vec4i> hierarchy_o;
                findContours_legacy(img, contours_o, hierarchy_o, mode, method);
                ASSERT_EQ(contours.size(), contours_o.size());
                for (size_t i = 0; i < contours.size(); ++i)
                {
                    SCOPED_TRACE(format("contour %zu", i));
                    EXPECT_MAT_NEAR(Mat(contours[i]), Mat(contours_o[i]), 0);
                }
                EXPECT_MAT_NEAR(Mat(hierarchy), Mat(hierarchy_o), 0);
#endif
            }
        }
    }
}


// Draw many nested rectangles and find contours
//
TEST_P(Imgproc_FindContours_Modes1, deep)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());

    const size_t DIM = 1000;
    const Size sz(DIM, DIM);
    const size_t NUM = 249U;
    Mat img(sz, CV_8UC1, Scalar::all(0));
    Mat img32s(sz, CV_32SC1, Scalar::all(0));
    Rect rect(1, 1, 998, 998);
    for (size_t i = 0; i < NUM; ++i)
    {
        rectangle(img, rect, Scalar::all(255));
        rectangle(img32s, rect, Scalar::all((double)i + 1), FILLED);
        rect.x += 2;
        rect.y += 2;
        rect.width -= 4;
        rect.height -= 4;
    }
    {
        vector<vector<Point>> contours {{{0, 0}, {1, 1}}};
        vector<vector<schar>> chains {{1, 2, 3}};
        vector<Vec4i> hierarchy;

        if (mode == RETR_FLOODFILL)
        {
            if (method == 0)
            {
                findContours(img32s, chains, hierarchy, mode, method);
                ASSERT_EQ(2 * NUM, chains.size());
            }
            else
            {
                findContours(img32s, contours, hierarchy, mode, method);
                ASSERT_EQ(2 * NUM, contours.size());
#if CHECK_OLD
                vector<vector<Point>> contours_o;
                vector<Vec4i> hierarchy_o;
                findContours_legacy(img32s, contours_o, hierarchy_o, mode, method);
                ASSERT_EQ(contours.size(), contours_o.size());
                for (size_t i = 0; i < contours.size(); ++i)
                {
                    SCOPED_TRACE(format("contour %zu", i));
                    EXPECT_MAT_NEAR(Mat(contours[i]), Mat(contours_o[i]), 0);
                }
                EXPECT_MAT_NEAR(Mat(hierarchy), Mat(hierarchy_o), 0);
#endif
            }
        }
        else
        {
            const size_t expected_count = (mode == RETR_EXTERNAL) ? 1U : 2 * NUM;
            if (method == 0)
            {
                findContours(img, chains, hierarchy, mode, method);
                ASSERT_EQ(expected_count, chains.size());
            }
            else
            {
                findContours(img, contours, hierarchy, mode, method);
                ASSERT_EQ(expected_count, contours.size());
#if CHECK_OLD
                vector<vector<Point>> contours_o;
                vector<Vec4i> hierarchy_o;
                findContours_legacy(img, contours_o, hierarchy_o, mode, method);
                ASSERT_EQ(contours.size(), contours_o.size());
                for (size_t i = 0; i < contours.size(); ++i)
                {
                    SCOPED_TRACE(format("contour %zu", i));
                    EXPECT_MAT_NEAR(Mat(contours[i]), Mat(contours_o[i]), 0);
                }
                EXPECT_MAT_NEAR(Mat(hierarchy), Mat(hierarchy_o), 0);
#endif
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    ,
    Imgproc_FindContours_Modes1,
    testing::Combine(
        testing::Values(RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE, RETR_FLOODFILL),
        testing::Values(0,
                        CHAIN_APPROX_NONE,
                        CHAIN_APPROX_SIMPLE,
                        CHAIN_APPROX_TC89_L1,
                        CHAIN_APPROX_TC89_KCOS)));

//==================================================================================================

typedef testing::TestWithParam<tuple<int, int>> Imgproc_FindContours_Modes2;

// Very approximate backport of an old accuracy test
//
TEST_P(Imgproc_FindContours_Modes2, new_accuracy)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());

    RNG& rng = TS::ptr()->get_rng();
    const int blob_count = rng.uniform(1, 10);
    const Size sz(rng.uniform(640, 1920), rng.uniform(480, 1080));
    const int blob_sz = 50;

    // prepare image
    Mat img(sz, CV_8UC1, Scalar::all(0));
    vector<RotatedRect> rects;
    for (int i = 0; i < blob_count; ++i)
    {
        const Point2f center((float)rng.uniform(blob_sz, sz.width - blob_sz),
                             (float)rng.uniform(blob_sz, sz.height - blob_sz));
        const Size2f rsize((float)rng.uniform(1, blob_sz), (float)rng.uniform(1, blob_sz));
        RotatedRect rect(center, rsize, rng.uniform(0.f, 180.f));
        rects.push_back(rect);
        ellipse(img, rect, Scalar::all(100), FILLED);
    }

    // draw contours manually
    Mat cont_img(sz, CV_8UC1, Scalar::all(0));
    for (int y = 1; y < sz.height - 1; ++y)
    {
        for (int x = 1; x < sz.width - 1; ++x)
        {
            if (img.at<uchar>(y, x) != 0 &&
                ((img.at<uchar>(y - 1, x) == 0) || (img.at<uchar>(y + 1, x) == 0) ||
                 (img.at<uchar>(y, x + 1) == 0) || (img.at<uchar>(y, x - 1) == 0)))
            {
                cont_img.at<uchar>(y, x) = 255;
            }
        }
    }

    // find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, mode, method);

    // 0 < contours <= rects
    EXPECT_GT(contours.size(), 0U);
    EXPECT_GE(rects.size(), contours.size());

    // draw contours
    Mat res_img(sz, CV_8UC1, Scalar::all(0));
    drawContours(res_img, contours);

    // compare resulting drawn contours with manually drawn contours
    const double diff1 = cvtest::norm(cont_img, res_img, NORM_L1) / 255;

    if (method == CHAIN_APPROX_NONE || method == CHAIN_APPROX_SIMPLE)
    {
        EXPECT_EQ(0., diff1);
    }
#if CHECK_OLD
    vector<vector<Point>> contours_o;
    vector<Vec4i> hierarchy_o;
    findContours(img, contours_o, hierarchy_o, mode, method);
    ASSERT_EQ(contours_o.size(), contours.size());
    for (size_t i = 0; i < contours_o.size(); ++i)
    {
        SCOPED_TRACE(format("contour = %zu", i));
        EXPECT_MAT_NEAR(Mat(contours_o[i]), Mat(contours[i]), 0);
    }
    EXPECT_MAT_NEAR(Mat(hierarchy_o), Mat(hierarchy), 0);
#endif
}

TEST_P(Imgproc_FindContours_Modes2, approx)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());

    const Size sz {500, 500};
    Mat img = Mat::zeros(sz, CV_8UC1);

    for (int c = 0; c < 4; ++c)
    {
        if (c != 0)
        {
            // noise + filter + threshold
            RNG& rng = TS::ptr()->get_rng();
            cvtest::randUni(rng, img, 0, 255);

            Mat fimg;
            boxFilter(img, fimg, CV_8U, Size(5, 5));

            Mat timg;
            const int level = 44 + c * 42;
            // 'level' goes through:
            // 86 - some black speckles on white
            // 128 - 50/50 black/white
            // 170 - some white speckles on black
            cv::threshold(fimg, timg, level, 255, THRESH_BINARY);
        }
        else
        {
            // circle with cut
            const Point center {250, 250};
            const int r {20};
            const Point cut {r, r};
            circle(img, center, r, Scalar(255), FILLED);
            rectangle(img, center, center + cut, Scalar(0), FILLED);
        }

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(img, contours, hierarchy, mode, method);

#if CHECK_OLD
        // NOTE: old and new function results might not match when approximation mode is TC89.
        // Currently this test passes, but might fail for other random data.
        // See https://github.com/opencv/opencv/issues/25663 for details.
        vector<vector<Point>> contours_o;
        vector<Vec4i> hierarchy_o;
        findContours_legacy(img, contours_o, hierarchy_o, mode, method);
        ASSERT_EQ(contours_o.size(), contours.size());
        for (size_t i = 0; i < contours_o.size(); ++i)
        {
            SCOPED_TRACE(format("c = %d, contour = %zu", c, i));
            EXPECT_MAT_NEAR(Mat(contours_o[i]), Mat(contours[i]), 0);
        }
        EXPECT_MAT_NEAR(Mat(hierarchy_o), Mat(hierarchy), 0);
#endif
        // TODO: check something
    }
}

// TODO: offset test

// no RETR_FLOODFILL - no CV_32S input images
INSTANTIATE_TEST_CASE_P(
    ,
    Imgproc_FindContours_Modes2,
    testing::Combine(testing::Values(RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE),
                     testing::Values(CHAIN_APPROX_NONE,
                                     CHAIN_APPROX_SIMPLE,
                                     CHAIN_APPROX_TC89_L1,
                                     CHAIN_APPROX_TC89_KCOS)));

TEST(Imgproc_FindContours, link_runs)
{
    const Size sz {500, 500};
    Mat img = Mat::zeros(sz, CV_8UC1);

    // noise + filter + threshold
    RNG& rng = TS::ptr()->get_rng();
    cvtest::randUni(rng, img, 0, 255);

    Mat fimg;
    boxFilter(img, fimg, CV_8U, Size(5, 5));

    const int level = 135;
    cv::threshold(fimg, img, level, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContoursLinkRuns(img, contours, hierarchy);

    if (cvtest::debugLevel >= 10)
    {
        print_pts_2(contours);

        Mat res = Mat::zeros(sz, CV_8UC1);
        drawContours(res, contours);
        imshow("res", res);
        imshow("img", img);
        waitKey(0);
    }

#if CHECK_OLD
    vector<vector<Point>> contours_o;
    vector<Vec4i> hierarchy_o;
    findContours_legacy(img, contours_o, hierarchy_o, 0, 5);  // CV_LINK_RUNS method
    ASSERT_EQ(contours_o.size(), contours.size());
    for (size_t i = 0; i < contours_o.size(); ++i)
    {
        SCOPED_TRACE(format("contour = %zu", i));
        EXPECT_MAT_NEAR(Mat(contours_o[i]), Mat(contours[i]), 0);
    }
    EXPECT_MAT_NEAR(Mat(hierarchy_o), Mat(hierarchy), 0);
#endif
}

}}  // namespace opencv_test
