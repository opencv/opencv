// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

//helps to temporarily change the number of threads and restore it back after the scope
struct CvNThreadScope{
    int nprev;
    CvNThreadScope(int n){
        nprev=cv::getNumThreads();
        cv::setNumThreads(n);
    }
    ~CvNThreadScope(){
        cv::setNumThreads(nprev);
    }
};

static bool trucoPointLess(const Point& a, const Point& b)
{
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

// Find the least cyclic rotation in linear time, including contours that revisit a point.
static size_t trucoContourStart(const vector<Point>& contour)
{
    const size_t n = contour.size();
    size_t i = 0, j = 1, k = 0;
    while (i < n && j < n && k < n)
    {
        const Point& a = contour[(i + k) % n];
        const Point& b = contour[(j + k) % n];
        if (a == b)
        {
            ++k;
            continue;
        }
        if (trucoPointLess(a, b))
        {
            j += k + 1;
            if (j <= i)
                j = i + 1;
        }
        else
        {
            i += k + 1;
            if (i <= j)
                i = j + 1;
        }
        k = 0;
    }
    return std::min(i, j);
}

// Ignore contour order and cyclic starts, but preserve multiplicity and point traversal order.
static bool trucoContoursMatch(const vector<vector<Point>>& cont1, const vector<vector<Point>>& cont2)
{
    if (cont1.size() != cont2.size())
        return false;

    // Sort references and starting indices without copying large contour point buffers.
    typedef std::pair<const vector<Point>*, size_t> ContourReference;
    vector<ContourReference> sorted1, sorted2;
    sorted1.reserve(cont1.size());
    sorted2.reserve(cont2.size());
    for (const auto& contour : cont1)
        sorted1.emplace_back(&contour, trucoContourStart(contour));
    for (const auto& contour : cont2)
        sorted2.emplace_back(&contour, trucoContourStart(contour));

    const auto contourLess = [](const ContourReference& a, const ContourReference& b) {
        const size_t na = a.first->size(), nb = b.first->size();
        for (size_t k = 0; k < std::min(na, nb); ++k)
        {
            const Point& pa = (*a.first)[(a.second + k) % na];
            const Point& pb = (*b.first)[(b.second + k) % nb];
            if (pa != pb)
                return trucoPointLess(pa, pb);
        }
        return na < nb;
    };
    std::sort(sorted1.begin(), sorted1.end(), contourLess);
    std::sort(sorted2.begin(), sorted2.end(), contourLess);
    for (size_t i = 0; i < sorted1.size(); ++i)
    {
        if (contourLess(sorted1[i], sorted2[i]) || contourLess(sorted2[i], sorted1[i]))
            return false;
    }
    return true;
}

TEST(Imgproc_FindTRUContoursComparison, empty_inputs)
{
    const vector<vector<Point>> empty;
    const vector<vector<Point>> contour = {{Point(1, 2)}};
    EXPECT_TRUE(trucoContoursMatch(empty, empty));
    EXPECT_FALSE(trucoContoursMatch(empty, contour));
    EXPECT_FALSE(trucoContoursMatch(contour, empty));
}

TEST(Imgproc_FindTRUContoursComparison, contour_order)
{
    const vector<Point> a = {Point(1, 2), Point(3, 4)};
    const vector<Point> b = {Point(-1, 5)};
    EXPECT_TRUE(trucoContoursMatch({a, b}, {b, a}));
    EXPECT_TRUE(trucoContoursMatch({a, a, b}, {b, a, a}));
}

TEST(Imgproc_FindTRUContoursComparison, missing_or_extra_contours)
{
    const vector<Point> a = {Point(1, 2)};
    const vector<Point> b = {Point(3, 4)};
    EXPECT_FALSE(trucoContoursMatch({a}, {a, b}));
    EXPECT_FALSE(trucoContoursMatch({a, b}, {a}));
}

TEST(Imgproc_FindTRUContoursComparison, multiplicity)
{
    const vector<Point> a = {Point(1, 2)};
    const vector<Point> b = {Point(3, 4)};
    EXPECT_FALSE(trucoContoursMatch({a}, {a, a}));
    EXPECT_FALSE(trucoContoursMatch({a, a}, {a}));
    EXPECT_FALSE(trucoContoursMatch({a, a, b}, {a, b, b}));
    EXPECT_FALSE(trucoContoursMatch({a, b, b}, {a, a, b}));
}

TEST(Imgproc_FindTRUContoursComparison, point_sequences)
{
    const vector<Point> a = {Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)};
    const vector<Point> reversed(a.rbegin(), a.rend());
    const vector<Point> reordered = {a[0], a[2], a[1], a[3]};
    vector<Point> different = a;
    different[1].x += 1;
    EXPECT_TRUE(trucoContoursMatch({a}, {a}));
    EXPECT_FALSE(trucoContoursMatch({a}, {reversed}));
    EXPECT_FALSE(trucoContoursMatch({a}, {reordered}));
    EXPECT_FALSE(trucoContoursMatch({a}, {different}));
    EXPECT_FALSE(trucoContoursMatch({a}, {{Point(1, 2)}}));
}

TEST(Imgproc_FindTRUContoursComparison, cyclic_starts)
{
    const vector<vector<Point>> contours = {
        {}, {Point(1, 2)}, {Point(1, 2), Point(3, 4)},
        {Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)},
        {Point(0, 0), Point(2, 0), Point(0, 0), Point(1, 0)},
        {Point(0, 0), Point(1, 0), Point(0, 0), Point(1, 0)},
        {Point(0, 0), Point(0, 0), Point(0, 0)}
    };
    EXPECT_TRUE(trucoContoursMatch({{}, {}}, {{}, {}}));
    EXPECT_FALSE(trucoContoursMatch({{}}, {{}, {}}));
    for (const auto& contour : contours)
    {
        vector<Point> rotated = contour;
        for (size_t shift = 0; shift < contour.size(); ++shift)
        {
            std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
            EXPECT_TRUE(trucoContoursMatch({contour}, {rotated}));
            EXPECT_TRUE(trucoContoursMatch({contour, contour}, {rotated, contour}));
        }
    }
}

TEST(Imgproc_FindTRUContoursComparison, canonical_start)
{
    // Compare against exhaustive rotations to cover repeated minima and periodic sequences.
    for (size_t n = 1, count = 3; n <= 7; ++n, count *= 3)
    {
        for (size_t code = 0; code < count; ++code)
        {
            vector<Point> contour(n);
            size_t value = code;
            for (size_t i = 0; i < n; ++i, value /= 3)
                contour[i] = Point(static_cast<int>(value % 3), 0);
            vector<Point> expected = contour, rotated = contour;
            for (size_t shift = 1; shift < n; ++shift)
            {
                std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
                if (std::lexicographical_compare(rotated.begin(), rotated.end(),
                                                 expected.begin(), expected.end(), trucoPointLess))
                    expected = rotated;
            }
            const size_t start = trucoContourStart(contour);
            std::rotate(contour.begin(), contour.begin() + start, contour.end());
            ASSERT_EQ(expected, contour);
        }
    }
}

typedef testing::TestWithParam<ContourApproximationModes> Imgproc_FindTRUContours;

TEST_P(Imgproc_FindTRUContours, nthreads_consistency)
{
    ContourApproximationModes method = GetParam();
    const Size sz(1000, 1000);
    RNG& rng = TS::ptr()->get_rng();
    Mat noise(sz, CV_8UC1);
    cvtest::randUni(rng, noise, 0, 255);
    Mat blurred;
    boxFilter(noise, blurred, CV_8U, Size(5, 5));
    Mat img;
    cv::threshold(blurred, img, 128, 255, THRESH_BINARY);

    vector<vector<Point>> ref_contours;
    vector<vector<Point>> ref_contours_m0;
    {
        CvNThreadScope nt(1);
        findContours(img, ref_contours, RETR_LIST, method);
    }

    std::vector<int> thread_counts;
    for(int i=2;i<40;i++) thread_counts.push_back(i);
    for (int t : thread_counts)
    {
        SCOPED_TRACE(cv::format("nthreads=%d method=%d", t, (int)method));
        CvNThreadScope nt(t);
        vector<vector<Point>> contours;
        findContours(img, contours, RETR_LIST, method); //will use TRUCO because NOT using hierarchy AND RETR_LIST
        auto match=trucoContoursMatch(ref_contours, contours);
        EXPECT_TRUE(match);
    }
}

TEST_P(Imgproc_FindTRUContours, circles_vs_standard)
{
    ContourApproximationModes method = GetParam();
    const Size sz(4000, 4000);
    const int ITER = cvtest::debugLevel >= 10?100:10;
    const int NUM_CIRCLES = 250;
    RNG& rng = TS::ptr()->get_rng();

    for (int iter = 0; iter < ITER; ++iter)
    {
        SCOPED_TRACE(cv::format("iter=%d method=%d", iter, (int)method));
        Mat img(sz, CV_8UC1, Scalar::all(0));
        for (int i = 0; i < NUM_CIRCLES; ++i)
        {
            Point center(rng.uniform(50, sz.width  - 50),
                         rng.uniform(50, sz.height - 50));
            int radius = rng.uniform(10, 150);
            circle(img, center, radius, Scalar::all(255), FILLED);
        }
        Mat binary;
        adaptiveThreshold(img, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 0);

        vector<vector<Point>> ref_contours;
        vector<Vec4i> hierarchy;
        findContours(binary, ref_contours, hierarchy, RETR_LIST, method); //will call suzuki abe because using hierarchy
        EXPECT_TRUE(!hierarchy.empty());
        vector<vector<Point>> truco_contours;
        findContours(binary, truco_contours, RETR_LIST, method);
        EXPECT_TRUE(trucoContoursMatch(ref_contours, truco_contours)); //will use TRUCO because NOT using hierarchy AND RETR_LIST
    }
}

TEST_P(Imgproc_FindTRUContours, noise_threshold)
{
    ContourApproximationModes method = GetParam();
    const Size sz(1500, 1500);
    RNG& rng = TS::ptr()->get_rng();
    const int levels[] = {86, 128, 170};
    const int ITER = 2;

    std::vector<int> thread_counts;
    for(int i=2; i<40; i+=3) thread_counts.push_back(i);
    for(int i=0; i<ITER; i++)
    {
        for (int level : levels)
        {
            SCOPED_TRACE(cv::format("level=%d method=%d", level, (int)method));
            Mat noise(sz, CV_8UC1);
            cvtest::randUni(rng, noise, 0, 255);
            Mat blurred;
            boxFilter(noise, blurred, CV_8U, Size(5, 5));
            Mat binary;
            cv::threshold(blurred, binary, level, 255, THRESH_BINARY);

            vector<vector<Point>> ref_contours;
            vector<Vec4i> hierarchy;
            findContours(binary, ref_contours, hierarchy, RETR_LIST, method);//will call suzuki&abe because using hierarchy
            EXPECT_TRUE(!hierarchy.empty());
            for(auto nt: thread_counts){
                CvNThreadScope ts(nt);
                vector<vector<Point>> truco_contours;
                findContours(binary, truco_contours,  RETR_LIST, method);//will call TRUCO abe because NOT using hierarchy
                EXPECT_TRUE(trucoContoursMatch(ref_contours, truco_contours));
            }
        }
    }
}

TEST_P(Imgproc_FindTRUContours, nested_rectangles)
{
    ContourApproximationModes method = GetParam();
    const int DIM = 1500;
    const Size sz(DIM, DIM);
    const int NUM = 25;
    Mat img(sz, CV_8UC1, Scalar::all(0));
    Rect rect(1, 1, DIM - 2, DIM - 2);
    for (int i = 0; i < NUM; ++i)
    {
        rectangle(img, rect, Scalar::all(255));
        rect.x      += 10;
        rect.y      += 10;
        rect.width  -= 20;
        rect.height -= 20;
        if (rect.width <= 0 || rect.height <= 0)
            break;
    }

    vector<vector<Point>> ref_contours;
    vector<Vec4i> hierarchy;
    findContours(img, ref_contours, hierarchy, RETR_LIST, method);//will call suzuki abe because using hierarchy
    EXPECT_TRUE(!hierarchy.empty());

    vector<vector<Point>> truco_contours;
    findContours(img, truco_contours, RETR_LIST, method);//will use TRUCO because NOT using hierarchy AND RETR_LIST

    EXPECT_TRUE(trucoContoursMatch(ref_contours, truco_contours));
}

TEST_P(Imgproc_FindTRUContours, mixed_figures)
{
    ContourApproximationModes method = GetParam();
    const Size sz(1800, 1600);
    RNG& rng = TS::ptr()->get_rng();
    const int ITER = cvtest::debugLevel >= 10?100:10;


    for (int iter = 0; iter < ITER; ++iter)
    {
        SCOPED_TRACE(cv::format("iter=%d method=%d", iter, (int)method));
        Mat img(sz, CV_8UC1, Scalar::all(0));
        for (int i = 0; i < 5; ++i)
        {
            Rect r(rng.uniform(10, sz.width / 2),
                   rng.uniform(10, sz.height / 2),
                   rng.uniform(20, 100),
                   rng.uniform(20, 100));
            r &= Rect(0, 0, sz.width - 1, sz.height - 1);
            rectangle(img, r, Scalar::all(255), FILLED);
        }
        for (int i = 0; i < 5; ++i)
        {
            Point center(rng.uniform(50, sz.width  - 50),
                         rng.uniform(50, sz.height - 50));
            int radius = rng.uniform(10, 50);
            circle(img, center, radius, Scalar::all(255), FILLED);
        }
        for (int i = 0; i < 3; ++i)
        {
            Point pts[3];
            for (auto& p : pts)
                p = Point(rng.uniform(10, sz.width - 10),
                          rng.uniform(10, sz.height - 10));
            const Point* ppts = pts;
            int npts = 3;
            fillPoly(img, &ppts, &npts, 1, Scalar::all(255));
        }
        vector<vector<Point>> ref_contours;
        vector<Vec4i> hierarchy;
        findContours(img, ref_contours, hierarchy, RETR_LIST, method);//will call suzuki abe because using hierarchy
        EXPECT_TRUE(!hierarchy.empty());
        vector<vector<Point>> truco_contours;
        findContours(img, truco_contours, RETR_LIST, method);//will use TRUCO because NOT using hierarchy AND RETR_LIST

        EXPECT_TRUE(trucoContoursMatch(ref_contours, truco_contours));
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_FindTRUContours,
    testing::Values(CHAIN_APPROX_NONE,CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS));

}} // namespace
