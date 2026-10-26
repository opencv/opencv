// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include "opencv2/imgproc/detail/legacy.hpp"

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

// Order-independent comparison of the two contour lists as multisets: every contour must appear
// the same number of times in both (duplicated, missing or extra contours are reported)
static bool trucoContoursMatch(const vector<vector<Point>>& cont1, const vector<vector<Point>>& cont2)
{
        //order senstive hash
    auto Hash=[](const std::vector<cv::Point>& contour) {
        // FNV-1a 64-bit hash constants
        constexpr uint64_t FNV_OFFSET = 1469598103934665603ULL;
        constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

        uint64_t hash = FNV_OFFSET;

        // Mix in the size so that contours with different lengths
        // but same prefix produce different hashes
        uint64_t size = static_cast<uint64_t>(contour.size());
        for (int i = 0; i < 8; ++i) {
            hash ^= (size >> (i * 8)) & 0xFF;
            hash *= FNV_PRIME;
        }

        // Mix in each point's x and y coordinates byte by byte
        for (const cv::Point& p : contour) {
            uint32_t x = static_cast<uint32_t>(p.x);
            uint32_t y = static_cast<uint32_t>(p.y);

            for (int i = 0; i < 4; ++i) {
                hash ^= (x >> (i * 8)) & 0xFF;
                hash *= FNV_PRIME;
            }
            for (int i = 0; i < 4; ++i) {
                hash ^= (y >> (i * 8)) & 0xFF;
                hash *= FNV_PRIME;
            }
        }
        return hash;
    };
    if (cont1.size() != cont2.size()) return false;
    std::vector<uint64> hashes1,hashes2;
    for(auto &contour:cont1){
        hashes1.push_back( Hash(contour));
    }
    for(auto &contour:cont2){
        hashes2.push_back( Hash(contour));
    }
    std::sort(hashes1.begin(), hashes1.end());
    std::sort(hashes2.begin(), hashes2.end());
    return hashes1 == hashes2;
}

// Suzuki-Abe reference (findContours_legacy). Its TC89 may differ from the current one (#25663),
// so for TC89 the reference is findContours with hierarchy, checked in Imgproc_FindTRUContours_Hierarchy.
static void findReferenceContours(const Mat& img, vector<vector<Point>>& contours, vector<Vec4i>& hierarchy, int method)
{
    if (method == CHAIN_APPROX_TC89_L1 || method == CHAIN_APPROX_TC89_KCOS)
        findContours(img, contours, hierarchy, RETR_LIST, method);
    else
        findContours_legacy(img, contours, hierarchy, RETR_LIST, method);
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
        findReferenceContours(binary, ref_contours, hierarchy, method);
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
            findReferenceContours(binary, ref_contours, hierarchy, method);
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
    findReferenceContours(img, ref_contours, hierarchy, method);
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
        findReferenceContours(img, ref_contours, hierarchy, method);
        EXPECT_TRUE(!hierarchy.empty());
        vector<vector<Point>> truco_contours;
        findContours(img, truco_contours, RETR_LIST, method);//will use TRUCO because NOT using hierarchy AND RETR_LIST

        EXPECT_TRUE(trucoContoursMatch(ref_contours, truco_contours));
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_FindTRUContours,
    testing::Values(CHAIN_APPROX_NONE,CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS));

// Exact comparison (same contours in the same order, same hierarchy) of findContours, which uses TRUCO
// for every mode except RETR_FLOODFILL, with the Suzuki-Abe reference, for several thread counts
static void expectSameAsLegacy(const Mat& img, int mode, int method, const std::vector<int>& thread_counts)
{
    vector<vector<Point>> ref_contours;
    vector<Vec4i> ref_hierarchy;
    findContours_legacy(img, ref_contours, ref_hierarchy, mode, method);
    for (int nt : thread_counts)
    {
        SCOPED_TRACE(cv::format("nthreads=%d mode=%d method=%d", nt, mode, method));
        CvNThreadScope ts(nt);
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(img, contours, hierarchy, mode, method);
        ASSERT_EQ(ref_contours.size(), contours.size());
        for (size_t i = 0; i < contours.size(); ++i)
        {
            SCOPED_TRACE(cv::format("contour=%zu", i));
            EXPECT_MAT_NEAR(Mat(ref_contours[i]), Mat(contours[i]), 0);
        }
        if (!ref_hierarchy.empty())
            EXPECT_MAT_NEAR(Mat(ref_hierarchy), Mat(hierarchy), 0);
        else
            EXPECT_TRUE(hierarchy.empty());
    }
}

typedef testing::TestWithParam<tuple<RetrievalModes, ContourApproximationModes>> Imgproc_FindTRUContours_Hierarchy;

TEST_P(Imgproc_FindTRUContours_Hierarchy, noise)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());
    RNG& rng = TS::ptr()->get_rng();
    std::vector<int> thread_counts;
    for (int i = 1; i < 40; i += 3) thread_counts.push_back(i);
    // 86: black speckles on white, 128: 50/50, 170: white speckles on black (many nested holes)
    for (int level : {86, 128, 170})
    {
        SCOPED_TRACE(cv::format("level=%d", level));
        Mat noise(Size(700, 600), CV_8UC1);
        cvtest::randUni(rng, noise, 0, 255);
        Mat blurred, img;
        boxFilter(noise, blurred, CV_8U, Size(5, 5));
        cv::threshold(blurred, img, level, 255, THRESH_BINARY);
        expectSameAsLegacy(img, mode, method, thread_counts);
    }
}

TEST_P(Imgproc_FindTRUContours_Hierarchy, nested)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());
    // concentric rings crossing every stripe, with filled blobs and holes inside some of them
    Mat img(Size(800, 800), CV_8UC1, Scalar::all(0));
    for (int r = 390; r > 10; r -= 12)
        circle(img, Point(400, 400), r, Scalar::all(255), 3);
    for (int i = 0; i < 6; ++i)
    {
        circle(img, Point(400 + 60 * i - 150, 400), 20, Scalar::all(255), FILLED);
        circle(img, Point(400 + 60 * i - 150, 400), 8, Scalar::all(0), FILLED);
    }
    std::vector<int> thread_counts;
    for (int i = 1; i < 40; i += 2) thread_counts.push_back(i);
    expectSameAsLegacy(img, mode, method, thread_counts);
}

TEST_P(Imgproc_FindTRUContours_Hierarchy, small_images)
{
    const int mode = get<0>(GetParam());
    const int method = get<1>(GetParam());
    RNG& rng = TS::ptr()->get_rng();
    const std::vector<int> thread_counts = { 1, 2, 3, 7 };
    for (int iter = 0; iter < 100; ++iter)
    {
        SCOPED_TRACE(cv::format("iter=%d", iter));
        // random noise of varied density on small images, including 1-pixel wide ones
        Mat img(Size(rng.uniform(1, 40), rng.uniform(1, 40)), CV_8UC1);
        const int density = rng.uniform(10, 245);
        cvtest::randUni(rng, img, 0, 255);
        cv::threshold(img, img, density, 255, THRESH_BINARY);
        expectSameAsLegacy(img, mode, method, thread_counts);
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_FindTRUContours_Hierarchy,
    testing::Combine(
        testing::Values(RETR_LIST, RETR_CCOMP, RETR_TREE),
        testing::Values(CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE)));

// Freeman chain code of a closed contour, as returned with CV_CHAIN_CODE (an isolated pixel has none)
static Mat toChainCode(const vector<Point>& contour)
{
    static const schar lut[3][3] = { { 3, 2, 1 }, { 4, -1, 0 }, { 5, 6, 7 } };   // [dy + 1][dx + 1]
    const size_t n = contour.size();
    Mat codes(1, n > 1 ? (int)n : 0, CV_8SC1);
    for (size_t i = 0; n > 1 && i < n; ++i)
    {
        const Point d = contour[(i + 1) % n] - contour[i];
        codes.at<schar>(0, (int)i) = lut[d.y + 1][d.x + 1];
    }
    return codes;
}

// CV_CHAIN_CODE still uses Suzuki-Abe: same contours, order, hierarchy and chain code (so same start
// point) as TRUCO.
static void expectSameAsSuzukiAbe(const Mat& img, int mode, const std::vector<int>& thread_counts)
{
    vector<Mat> chains;
    vector<Vec4i> ref_hierarchy;
    findContours(img, chains, ref_hierarchy, mode, 0 /*CV_CHAIN_CODE*/);
    for (int nt : thread_counts)
    {
        SCOPED_TRACE(cv::format("nthreads=%d mode=%d", nt, mode));
        CvNThreadScope ts(nt);
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(img, contours, hierarchy, mode, CHAIN_APPROX_NONE);
        ASSERT_EQ(chains.size(), contours.size());
        for (size_t i = 0; i < contours.size(); ++i)
        {
            SCOPED_TRACE(cv::format("contour=%zu", i));
            const Mat codes = toChainCode(contours[i]);
            ASSERT_EQ(chains[i].total(), codes.total());
            if (!codes.empty())
            {
                EXPECT_EQ(0, cvtest::norm(chains[i].reshape(1, 1), codes, NORM_INF));
            }
        }
        if (!ref_hierarchy.empty())
            EXPECT_MAT_NEAR(Mat(ref_hierarchy), Mat(hierarchy), 0);
        else
            EXPECT_TRUE(hierarchy.empty());
    }
}

typedef testing::TestWithParam<RetrievalModes> Imgproc_FindTRUContours_SuzukiAbe;

TEST_P(Imgproc_FindTRUContours_SuzukiAbe, noise)
{
    const int mode = GetParam();
    RNG& rng = TS::ptr()->get_rng();
    std::vector<int> thread_counts;
    for (int i = 1; i < 40; i += 3) thread_counts.push_back(i);
    for (int level : {86, 128, 170})
    {
        SCOPED_TRACE(cv::format("level=%d", level));
        Mat noise(Size(700, 600), CV_8UC1);
        cvtest::randUni(rng, noise, 0, 255);
        Mat blurred, img;
        boxFilter(noise, blurred, CV_8U, Size(5, 5));
        cv::threshold(blurred, img, level, 255, THRESH_BINARY);
        expectSameAsSuzukiAbe(img, mode, thread_counts);
    }
}

TEST_P(Imgproc_FindTRUContours_SuzukiAbe, nested_and_shapes)
{
    const int mode = GetParam();
    RNG& rng = TS::ptr()->get_rng();
    std::vector<int> thread_counts;
    for (int i = 1; i < 40; i += 2) thread_counts.push_back(i);
    for (int iter = 0; iter < 5; ++iter)
    {
        SCOPED_TRACE(cv::format("iter=%d", iter));
        // concentric rings crossing every stripe, filled shapes with holes, isolated pixels and thin lines
        Mat img(Size(900, 800), CV_8UC1, Scalar::all(0));
        for (int r = 390; r > 10; r -= rng.uniform(8, 20))
            circle(img, Point(450, 400), r, Scalar::all(255), rng.uniform(1, 4));
        for (int i = 0; i < 20; ++i)
        {
            const Point c(rng.uniform(20, 880), rng.uniform(20, 780));
            circle(img, c, rng.uniform(5, 40), Scalar::all(255), FILLED);
            circle(img, c, rng.uniform(1, 5), Scalar::all(0), FILLED);
        }
        for (int i = 0; i < 200; ++i)
            img.at<uchar>(rng.uniform(0, 800), rng.uniform(0, 900)) = (uchar)rng.uniform(1, 256);
        for (int i = 0; i < 10; ++i)
            line(img, Point(rng.uniform(0, 900), rng.uniform(0, 800)), Point(rng.uniform(0, 900), rng.uniform(0, 800)),
                 Scalar::all(255), 1);
        expectSameAsSuzukiAbe(img, mode, thread_counts);
    }
}

TEST_P(Imgproc_FindTRUContours_SuzukiAbe, small_images)
{
    const int mode = GetParam();
    RNG& rng = TS::ptr()->get_rng();
    const std::vector<int> thread_counts = { 1, 2, 3, 7 };
    for (int iter = 0; iter < 200; ++iter)
    {
        SCOPED_TRACE(cv::format("iter=%d", iter));
        Mat img(Size(rng.uniform(1, 40), rng.uniform(1, 40)), CV_8UC1);
        const int density = rng.uniform(10, 245);
        cvtest::randUni(rng, img, 0, 255);
        cv::threshold(img, img, density, 255, THRESH_BINARY);
        expectSameAsSuzukiAbe(img, mode, thread_counts);
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_FindTRUContours_SuzukiAbe,
    testing::Values(RETR_LIST, RETR_CCOMP, RETR_TREE));

// The TRUCO code paths with an offset and with vector<Mat> output, compared with the Suzuki-Abe reference
// (the offset used to be added only to the first contour of a vector<Mat> output)
TEST(Imgproc_FindTRUContours, offset_and_output_kinds)
{
    RNG& rng = TS::ptr()->get_rng();
    Mat noise(Size(500, 400), CV_8UC1);
    cvtest::randUni(rng, noise, 0, 255);
    Mat blurred, img;
    boxFilter(noise, blurred, CV_8U, Size(5, 5));
    cv::threshold(blurred, img, 140, 255, THRESH_BINARY);
    const Point offset(7, -3);
    for (int mode : {RETR_LIST, RETR_CCOMP, RETR_TREE})
    {
        for (int method : {CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE})
        {
            for (bool withHierarchy : {false, true})
            {
                if (!withHierarchy && mode != RETR_LIST)
                    continue;   // the order of RETR_CCOMP / RETR_TREE is checked with the hierarchy
                SCOPED_TRACE(cv::format("mode=%d method=%d hierarchy=%d", mode, method, (int)withHierarchy));
                vector<vector<Point>> ref;
                vector<Vec4i> ref_hierarchy, hierarchy;
                findContours_legacy(img, ref, ref_hierarchy, mode, method, offset);
                vector<vector<Point>> contours;
                vector<Mat> mats;
                if (withHierarchy)
                {
                    findContours(img, contours, hierarchy, mode, method, offset);
                    EXPECT_MAT_NEAR(Mat(ref_hierarchy), Mat(hierarchy), 0);
                    findContours(img, mats, hierarchy, mode, method, offset);
                    EXPECT_MAT_NEAR(Mat(ref_hierarchy), Mat(hierarchy), 0);
                }
                else
                {
                    findContours(img, contours, mode, method, offset);
                    findContours(img, mats, mode, method, offset);
                }
                ASSERT_EQ(ref.size(), contours.size());
                ASSERT_EQ(ref.size(), mats.size());
                for (size_t i = 0; i < ref.size(); ++i)
                {
                    SCOPED_TRACE(cv::format("contour=%zu", i));
                    EXPECT_MAT_NEAR(Mat(ref[i]), Mat(contours[i]), 0);
                    EXPECT_EQ(CV_32SC2, mats[i].type());
                    EXPECT_MAT_NEAR(Mat(ref[i]), mats[i], 0);
                }
            }
        }
    }
}

// Chain codes are returned as CV_8SC1, with and without hierarchy output
TEST(Imgproc_FindTRUContours, chain_code_list)
{
    Mat img(Size(50, 40), CV_8UC1, Scalar::all(0));
    rectangle(img, Rect(5, 5, 20, 10), Scalar::all(255), FILLED);
    rectangle(img, Rect(30, 20, 10, 15), Scalar::all(255), FILLED);
    vector<Mat> chains, chains_h;
    vector<Vec4i> hierarchy;
    findContours(img, chains, RETR_LIST, 0 /*CV_CHAIN_CODE*/);
    findContours(img, chains_h, hierarchy, RETR_LIST, 0 /*CV_CHAIN_CODE*/);
    ASSERT_EQ(2U, chains.size());
    ASSERT_EQ(chains_h.size(), chains.size());
    for (size_t i = 0; i < chains.size(); ++i)
    {
        EXPECT_EQ(CV_8SC1, chains[i].type());
        EXPECT_MAT_NEAR(chains_h[i], chains[i], 0);
    }
}

// A ROI of a larger image gives the same result as its clone: the pixels around the ROI
// are not part of the input (the parent image is foreground everywhere around the ROI)
TEST(Imgproc_FindTRUContours, roi)
{
    RNG& rng = TS::ptr()->get_rng();
    Mat parent(Size(120, 100), CV_8UC1, Scalar::all(255));
    Mat roi = parent(Rect(10, 20, 90, 60));
    Mat noise(roi.size(), CV_8UC1);
    cvtest::randUni(rng, noise, 0, 255);
    boxFilter(noise, noise, CV_8U, Size(5, 5));
    cv::threshold(noise, roi, 140, 255, THRESH_BINARY);
    roi.row(0).setTo(Scalar::all(255));   // foreground touching the ROI border
    roi.col(roi.cols - 1).setTo(Scalar::all(255));
    const Mat cloned = roi.clone();
    for (int mode : {RETR_LIST, RETR_CCOMP, RETR_TREE})
    {
        for (bool withHierarchy : {false, true})
        {
            SCOPED_TRACE(cv::format("mode=%d hierarchy=%d", mode, (int)withHierarchy));
            vector<vector<Point>> ref, contours;
            vector<Vec4i> ref_hierarchy, hierarchy;
            if (withHierarchy)
            {
                findContours(cloned, ref, ref_hierarchy, mode, CHAIN_APPROX_NONE);
                findContours(roi, contours, hierarchy, mode, CHAIN_APPROX_NONE);
                EXPECT_MAT_NEAR(Mat(ref_hierarchy), Mat(hierarchy), 0);
            }
            else
            {
                if (mode != RETR_LIST)
                    continue;
                findContours(cloned, ref, mode, CHAIN_APPROX_NONE);
                findContours(roi, contours, mode, CHAIN_APPROX_NONE);
            }
            ASSERT_FALSE(ref.empty());
            ASSERT_EQ(ref.size(), contours.size());
            for (size_t i = 0; i < ref.size(); ++i)
                EXPECT_MAT_NEAR(Mat(ref[i]), Mat(contours[i]), 0);
        }
    }
}

// contours with an element type other than Point are rejected, as Suzuki-Abe does (RETR_EXTERNAL)
TEST(Imgproc_FindTRUContours, wrong_output_type)
{
    Mat img = Mat::zeros(Size(64, 48), CV_8UC1);
    rectangle(img, Rect(5, 5, 30, 20), Scalar::all(255), FILLED);
    rectangle(img, Rect(12, 10, 8, 6), Scalar::all(0), FILLED);
    rectangle(img, Rect(40, 20, 15, 15), Scalar::all(255), FILLED);
    for (int mode : {RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE})
    {
        SCOPED_TRACE(cv::format("mode=%d", mode));
        vector<vector<Point2f>> contours_f;
        vector<vector<int>> contours_i;
        vector<Vec4i> hierarchy;
        EXPECT_ANY_THROW(findContours(img, contours_f, mode, CHAIN_APPROX_SIMPLE));
        EXPECT_ANY_THROW(findContours(img, contours_f, hierarchy, mode, CHAIN_APPROX_SIMPLE));
        EXPECT_ANY_THROW(findContours(img, contours_i, mode, CHAIN_APPROX_SIMPLE));
        EXPECT_ANY_THROW(findContours(img, contours_i, hierarchy, mode, CHAIN_APPROX_SIMPLE));
    }
}


}} // namespace
