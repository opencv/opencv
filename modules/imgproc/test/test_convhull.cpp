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

#include "opencv2/core/hal/interface.h"
#include "opencv2/ts.hpp"
#include "opencv2/ts/cuda_test.hpp"
#include "test_precomp.hpp"

namespace opencv_test { namespace {

/****************************************************************************************\
*                                 minEnclosingCircle Test 3                              *
\****************************************************************************************/

TEST(minEnclosingCircle, basic_test)
{
    vector<Point2f> pts;
    pts.push_back(Point2f(0, 0));
    pts.push_back(Point2f(10, 0));
    pts.push_back(Point2f(5, 1));
    const float EPS = 1.0e-3f;
    Point2f center;
    float radius;

    // pts[2] is within the circle with diameter pts[0] - pts[1].
    //        2
    // 0             1
    // NB: The triangle is obtuse, so the only pts[0] and pts[1] are on the circle.
    minEnclosingCircle(pts, center, radius);
    EXPECT_NEAR(center.x, 5, EPS);
    EXPECT_NEAR(center.y, 0, EPS);
    EXPECT_NEAR(5, radius, EPS);

    // pts[2] is on the circle with diameter pts[0] - pts[1].
    //  2
    // 0 1
    pts[2] = Point2f(5, 5);
    minEnclosingCircle(pts, center, radius);
    EXPECT_NEAR(center.x, 5, EPS);
    EXPECT_NEAR(center.y, 0, EPS);
    EXPECT_NEAR(5, radius, EPS);

    // pts[2] is outside the circle with diameter pts[0] - pts[1].
    //   2
    //
    //
    // 0   1
    // NB: The triangle is acute, so all 3 points are on the circle.
    pts[2] = Point2f(5, 10);
    minEnclosingCircle(pts, center, radius);
    EXPECT_NEAR(center.x, 5, EPS);
    EXPECT_NEAR(center.y, 3.75, EPS);
    EXPECT_NEAR(6.25f, radius, EPS);

    // The 3 points are colinear.
    pts[2] = Point2f(3, 0);
    minEnclosingCircle(pts, center, radius);
    EXPECT_NEAR(center.x, 5, EPS);
    EXPECT_NEAR(center.y, 0, EPS);
    EXPECT_NEAR(5, radius, EPS);

    // 2 points are the same.
    pts[2] = pts[1];
    minEnclosingCircle(pts, center, radius);
    EXPECT_NEAR(center.x, 5, EPS);
    EXPECT_NEAR(center.y, 0, EPS);
    EXPECT_NEAR(5, radius, EPS);

    // 3 points are the same.
    pts[0] = pts[1];
    minEnclosingCircle(pts, center, radius);
    EXPECT_NEAR(center.x, 10, EPS);
    EXPECT_NEAR(center.y, 0, EPS);
    EXPECT_NEAR(0, radius, EPS);
}

TEST(Imgproc_minEnclosingCircle, regression_16051) {
    vector<Point2f> pts;
    pts.push_back(Point2f(85, 1415));
    pts.push_back(Point2f(87, 1415));
    pts.push_back(Point2f(89, 1414));
    pts.push_back(Point2f(89, 1414));
    pts.push_back(Point2f(87, 1412));
    Point2f center;
    float radius;
    minEnclosingCircle(pts, center, radius);
    EXPECT_NEAR(center.x, 86.9f, 1e-3);
    EXPECT_NEAR(center.y, 1414.1f, 1e-3);
    EXPECT_NEAR(2.1024551f, radius, 1e-3);
}

PARAM_TEST_CASE(ConvexityDefects_regression_5908, bool, int)
{
public:
    int start_index;
    bool clockwise;

    Mat contour;

    virtual void SetUp()
    {
        clockwise = GET_PARAM(0);
        start_index = GET_PARAM(1);

        const int N = 11;
        const Point2i points[N] = {
            Point2i(154, 408),
            Point2i(45, 223),
            Point2i(115, 275), // inner
            Point2i(104, 166),
            Point2i(154, 256), // inner
            Point2i(169, 144),
            Point2i(185, 256), // inner
            Point2i(235, 170),
            Point2i(240, 320), // inner
            Point2i(330, 287),
            Point2i(224, 390)
        };

        contour = Mat(N, 1, CV_32SC2);
        for (int i = 0; i < N; i++)
        {
            contour.at<Point2i>(i) = (!clockwise) // image and convexHull coordinate systems are different
                    ? points[(start_index + i) % N]
                    : points[N - 1 - ((start_index + i) % N)];
        }
    }
};

TEST_P(ConvexityDefects_regression_5908, simple)
{
    std::vector<int> hull;
    cv::convexHull(contour, hull, clockwise, false);

    std::vector<Vec4i> result;
    cv::convexityDefects(contour, hull, result);

    EXPECT_EQ(4, (int)result.size());
}

INSTANTIATE_TEST_CASE_P(Imgproc, ConvexityDefects_regression_5908,
        testing::Combine(
                testing::Bool(),
                testing::Values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        ));

TEST(Imgproc_FitLine, regression_15083)
{
    int points2i_[] = {
        432, 654,
        370, 656,
        390, 656,
        410, 656,
        348, 658
    };
    Mat points(5, 1, CV_32SC2, points2i_);

    Vec4f lineParam;
    fitLine(points, lineParam, DIST_L1, 0, 0.01, 0.01);
    EXPECT_GE(fabs(lineParam[0]), fabs(lineParam[1]) * 4) << lineParam;
}

TEST(Imgproc_FitLine, regression_4903)
{
    float points2f_[] = {
        1224.0, 576.0,
        1234.0, 683.0,
        1215.0, 471.0,
        1184.0, 137.0,
        1079.0, 377.0,
        1239.0, 788.0,
    };
    Mat points(6, 1, CV_32FC2, points2f_);

    Vec4f lineParam;
    fitLine(points, lineParam, DIST_WELSCH, 0, 0.01, 0.01);
    EXPECT_GE(fabs(lineParam[1]), fabs(lineParam[0]) * 4) << lineParam;
}

#if 0
#define DRAW(x) x
#else
#define DRAW(x)
#endif

// the Python test by @hannarud is converted to C++; see the issue #4539
TEST(Imgproc_ConvexityDefects, ordering_4539)
{
    int contour[][2] =
    {
        {26,  9}, {25, 10}, {24, 10}, {23, 10}, {22, 10}, {21, 10}, {20, 11}, {19, 11}, {18, 11}, {17, 12},
        {17, 13}, {18, 14}, {18, 15}, {18, 16}, {18, 17}, {19, 18}, {19, 19}, {20, 20}, {21, 21}, {21, 22},
        {22, 23}, {22, 24}, {23, 25}, {23, 26}, {24, 27}, {25, 28}, {26, 29}, {27, 30}, {27, 31}, {28, 32},
        {29, 32}, {30, 33}, {31, 34}, {30, 35}, {29, 35}, {30, 35}, {31, 34}, {32, 34}, {33, 34}, {34, 33},
        {35, 32}, {35, 31}, {35, 30}, {36, 29}, {37, 28}, {37, 27}, {38, 26}, {39, 25}, {40, 24}, {40, 23},
        {41, 22}, {42, 21}, {42, 20}, {42, 19}, {43, 18}, {43, 17}, {44, 16}, {45, 15}, {45, 14}, {46, 13},
        {46, 12}, {45, 11}, {44, 11}, {43, 11}, {42, 10}, {41, 10}, {40,  9}, {39,  9}, {38,  9}, {37,  9},
        {36,  9}, {35,  9}, {34,  9}, {33,  9}, {32,  9}, {31,  9}, {30,  9}, {29,  9}, {28,  9}, {27,  9}
    };
    int npoints = (int)(sizeof(contour)/sizeof(contour[0][0])/2);
    Mat contour_(1, npoints, CV_32SC2, contour);
    vector<Point> hull;
    vector<int> hull_ind;
    vector<Vec4i> defects;

    // first, check the original contour as-is, without intermediate fillPoly/drawContours.
    convexHull(contour_, hull_ind, false, false);
    EXPECT_THROW( convexityDefects(contour_, hull_ind, defects), cv::Exception );

    int scale = 20;
    contour_ *= (double)scale;

    Mat canvas_gray(Size(60*scale, 45*scale), CV_8U, Scalar::all(0));
    const Point* ptptr = contour_.ptr<Point>();
    fillPoly(canvas_gray, &ptptr, &npoints, 1, Scalar(255, 255, 255));

    vector<vector<Point> > contours;
    findContours(canvas_gray, contours, noArray(), RETR_LIST, CHAIN_APPROX_SIMPLE);
    convexHull(contours[0], hull_ind, false, false);

    // the original contour contains self-intersections,
    // therefore convexHull does not return a monotonous sequence of points
    // and therefore convexityDefects throws an exception
    EXPECT_THROW( convexityDefects(contours[0], hull_ind, defects), cv::Exception );

#if 1
    // one way to eliminate the contour self-intersection in this particular case is to apply dilate(),
    // so that the self-repeating points are not self-repeating anymore
    dilate(canvas_gray, canvas_gray, Mat());
#else
    // another popular technique to eliminate such thin "hair" is to use morphological "close" operation,
    // which is erode() + dilate()
    erode(canvas_gray, canvas_gray, Mat());
    dilate(canvas_gray, canvas_gray, Mat());
#endif

    // after the "fix", the newly retrieved contour should not have self-intersections,
    // and everything should work well
    findContours(canvas_gray, contours, noArray(), RETR_LIST, CHAIN_APPROX_SIMPLE);
    convexHull(contours[0], hull, false, true);
    convexHull(contours[0], hull_ind, false, false);

    DRAW(Mat canvas(Size(60*scale, 45*scale), CV_8UC3, Scalar::all(0));
        drawContours(canvas, contours, -1, Scalar(255, 255, 255), -1));

    size_t nhull = hull.size();
    ASSERT_EQ( nhull, hull_ind.size() );

    if( nhull > 2 )
    {
        bool initial_lt = hull_ind[0] < hull_ind[1];
        for( size_t i = 0; i < nhull; i++ )
        {
            int ind = hull_ind[i];
            Point pt = contours[0][ind];

            ASSERT_EQ(pt, hull[i]);
            if( i > 0 )
            {
                // check that the convex hull indices are monotone
                if( initial_lt )
                {
                    ASSERT_LT(hull_ind[i-1], hull_ind[i]);
                }
                else
                {
                    ASSERT_GT(hull_ind[i-1], hull_ind[i]);
                }
            }
            DRAW(circle(canvas, pt, 7, Scalar(180, 0, 180), -1, LINE_AA);
                putText(canvas, format("%d (%d)", (int)i, ind), pt+Point(15, 0), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 0, 200), 1, LINE_AA));
            //printf("%d. ind=%d, pt=(%d, %d)\n", (int)i, ind, pt.x, pt.y);
        }
    }

    convexityDefects(contours[0], hull_ind, defects);

    for(size_t i = 0; i < defects.size(); i++ )
    {
        Vec4i d = defects[i];
        //printf("defect %d. start=%d, end=%d, farthest=%d, depth=%d\n", (int)i, d[0], d[1], d[2], d[3]);
        EXPECT_LT(d[0], d[1]);
        EXPECT_LE(d[0], d[2]);
        EXPECT_LE(d[2], d[1]);

        DRAW(Point start = contours[0][d[0]];
             Point end = contours[0][d[1]];
             Point far = contours[0][d[2]];
             line(canvas, start, end, Scalar(255, 255, 128), 3, LINE_AA);
             line(canvas, start, far, Scalar(255, 150, 255), 3, LINE_AA);
             line(canvas, end, far, Scalar(255, 150, 255), 3, LINE_AA);
             circle(canvas, start, 7, Scalar(0, 0, 255), -1, LINE_AA);
             circle(canvas, end, 7, Scalar(0, 0, 255), -1, LINE_AA);
             circle(canvas, far, 7, Scalar(255, 0, 0), -1, LINE_AA));
    }

    DRAW(imshow("defects", canvas);
         waitKey());
}

#undef DRAW

TEST(Imgproc_ConvexHull, overflow)
{
    std::vector<Point> points;
    std::vector<Point2f> pointsf;

    points.push_back(Point(14763, 2890));
    points.push_back(Point(14388, 72088));
    points.push_back(Point(62810, 72274));
    points.push_back(Point(63166, 3945));
    points.push_back(Point(56782, 3945));
    points.push_back(Point(56763, 3077));
    points.push_back(Point(34666, 2965));
    points.push_back(Point(34547, 2953));
    points.push_back(Point(34508, 2866));
    points.push_back(Point(34429, 2965));

    size_t i, n = points.size();
    for( i = 0; i < n; i++ )
        pointsf.push_back(Point2f(points[i]));

    std::vector<int> hull;
    std::vector<int> hullf;

    convexHull(points, hull, false, false);
    convexHull(pointsf, hullf, false, false);

    ASSERT_EQ(hull, hullf);
}

static
bool checkMinAreaRect(const RotatedRect& rr, const Mat& c, double eps = 0.5f)
{
    int N = c.rows;

    Mat rr_pts;
    boxPoints(rr, rr_pts);

    double maxError = 0.0;
    int nfailed = 0;
    for (int i = 0; i < N; i++)
    {
        double d = pointPolygonTest(rr_pts, c.at<Point2f>(i), true);
        maxError = std::max(-d, maxError);
        if (d < -eps)
            nfailed++;
    }

    if (nfailed)
        std::cout << "nfailed=" << nfailed << " (total=" << N << ")   maxError=" << maxError << std::endl;
    return nfailed == 0;
}

TEST(Imgproc_minAreaRect, reproducer_18157)
{
    const int N = 168;
    float pts_[N][2] = {
        { 1903, 266 }, { 1897, 267 }, { 1893, 268 }, { 1890, 269 },
        { 1878, 275 }, { 1875, 277 }, { 1872, 279 }, { 1868, 282 },
        { 1862, 287 }, { 1750, 400 }, { 1748, 402 }, { 1742, 407 },
        { 1742, 408 }, { 1740, 410 }, { 1738, 412 }, { 1593, 558 },
        { 1590, 560 }, { 1588, 562 }, { 1586, 564 }, { 1580, 570 },
        { 1443, 709 }, { 1437, 714 }, { 1435, 716 }, { 1304, 848 },
        { 1302, 850 }, { 1292, 860 }, { 1175, 979 }, { 1172, 981 },
        { 1049, 1105 }, { 936, 1220 }, { 933, 1222 }, { 931, 1224 },
        { 830, 1326 }, { 774, 1383 }, { 769, 1389 }, { 766, 1393 },
        { 764, 1396 }, { 762, 1399 }, { 760, 1402 }, { 757, 1408 },
        { 757, 1410 }, { 755, 1413 }, { 754, 1416 }, { 753, 1420 },
        { 752, 1424 }, { 752, 1442 }, { 753, 1447 }, { 754, 1451 },
        { 755, 1454 }, { 757, 1457 }, { 757, 1459 }, { 761, 1467 },
        { 763, 1470 }, { 765, 1473 }, { 767, 1476 }, { 771, 1481 },
        { 779, 1490 }, { 798, 1510 }, { 843, 1556 }, { 847, 1560 },
        { 851, 1564 }, { 863, 1575 }, { 907, 1620 }, { 909, 1622 },
        { 913, 1626 }, { 1154, 1866 }, { 1156, 1868 }, { 1158, 1870 },
        { 1207, 1918 }, { 1238, 1948 }, { 1252, 1961 }, { 1260, 1968 },
        { 1264, 1971 }, { 1268, 1974 }, { 1271, 1975 }, { 1273, 1977 },
        { 1283, 1982 }, { 1286, 1983 }, { 1289, 1984 }, { 1294, 1985 },
        { 1300, 1986 }, { 1310, 1986 }, { 1316, 1985 }, { 1320, 1984 },
        { 1323, 1983 }, { 1326, 1982 }, { 1338, 1976 }, { 1341, 1974 },
        { 1344, 1972 }, { 1349, 1968 }, { 1358, 1960 }, { 1406, 1911 },
        { 1421, 1897 }, { 1624, 1693 }, { 1788, 1528 }, { 1790, 1526 },
        { 1792, 1524 }, { 1794, 1522 }, { 1796, 1520 }, { 1798, 1518 },
        { 1800, 1516 }, { 1919, 1396 }, { 1921, 1394 }, { 2038, 1275 },
        { 2047, 1267 }, { 2048, 1265 }, { 2145, 1168 }, { 2148, 1165 },
        { 2260, 1052 }, { 2359, 952 }, { 2434, 876 }, { 2446, 863 },
        { 2450, 858 }, { 2453, 854 }, { 2455, 851 }, { 2457, 846 },
        { 2459, 844 }, { 2460, 842 }, { 2460, 840 }, { 2462, 837 },
        { 2463, 834 }, { 2464, 830 }, { 2465, 825 }, { 2465, 809 },
        { 2464, 804 }, { 2463, 800 }, { 2462, 797 }, { 2461, 794 },
        { 2456, 784 }, { 2454, 781 }, { 2452, 778 }, { 2450, 775 },
        { 2446, 770 }, { 2437, 760 }, { 2412, 734 }, { 2410, 732 },
        { 2408, 730 }, { 2382, 704 }, { 2380, 702 }, { 2378, 700 },
        { 2376, 698 }, { 2372, 694 }, { 2370, 692 }, { 2368, 690 },
        { 2366, 688 }, { 2362, 684 }, { 2360, 682 }, { 2252, 576 },
        { 2250, 573 }, { 2168, 492 }, { 2166, 490 }, { 2085, 410 },
        { 2026, 352 }, { 1988, 315 }, { 1968, 296 }, { 1958, 287 },
        { 1953, 283 }, { 1949, 280 }, { 1946, 278 }, { 1943, 276 },
        { 1940, 274 }, { 1936, 272 }, { 1934, 272 }, { 1931, 270 },
        { 1928, 269 }, { 1925, 268 }, { 1921, 267 }, { 1915, 266 }
    };

    Mat contour(N, 1, CV_32FC2, (void*)pts_);

    RotatedRect rr = cv::minAreaRect(contour);

    EXPECT_TRUE(checkMinAreaRect(rr, contour)) << rr.center << " " << rr.size << " " << rr.angle;
}

TEST(Imgproc_minAreaRect, reproducer_19769_lightweight)
{
    const int N = 23;
    float pts_[N][2] = {
            {1325, 732}, {1248, 808}, {582, 1510}, {586, 1524},
            {595, 1541}, {599, 1547}, {789, 1745}, {829, 1786},
            {997, 1958}, {1116, 2074}, {1207, 2066}, {1216, 2058},
            {1231, 2044}, {1265, 2011}, {2036, 1254}, {2100, 1191},
            {2169, 1123}, {2315, 979}, {2395, 900}, {2438, 787},
            {2434, 782}, {2416, 762}, {2266, 610}
    };
    Mat contour(N, 1, CV_32FC2, (void*)pts_);

    RotatedRect rr = cv::minAreaRect(contour);

    EXPECT_TRUE(checkMinAreaRect(rr, contour)) << rr.center << " " << rr.size << " " << rr.angle;
}

TEST(Imgproc_minAreaRect, reproducer_19769)
{
    const int N = 169;
    float pts_[N][2] = {
            {1854, 227}, {1850, 228}, {1847, 229}, {1835, 235},
            {1832, 237}, {1829, 239}, {1825, 242}, {1818, 248},
            {1807, 258}, {1759, 306}, {1712, 351}, {1708, 356},
            {1658, 404}, {1655, 408}, {1602, 459}, {1599, 463},
            {1542, 518}, {1477, 582}, {1402, 656}, {1325, 732},
            {1248, 808}, {1161, 894}, {1157, 898}, {1155, 900},
            {1068, 986}, {1060, 995}, {1058, 997}, {957, 1097},
            {956, 1097}, {814, 1238}, {810, 1242}, {805, 1248},
            {610, 1442}, {603, 1450}, {599, 1455}, {596, 1459},
            {594, 1462}, {592, 1465}, {590, 1470}, {588, 1472},
            {586, 1476}, {586, 1478}, {584, 1481}, {583, 1485},
            {582, 1490}, {582, 1510}, {583, 1515}, {584, 1518},
            {585, 1521}, {586, 1524}, {593, 1538}, {595, 1541},
            {597, 1544}, {599, 1547}, {603, 1552}, {609, 1559},
            {623, 1574}, {645, 1597}, {677, 1630}, {713, 1667},
            {753, 1707}, {789, 1744}, {789, 1745}, {829, 1786},
            {871, 1828}, {909, 1867}, {909, 1868}, {950, 1910},
            {953, 1912}, {997, 1958}, {1047, 2009}, {1094, 2056},
            {1105, 2066}, {1110, 2070}, {1113, 2072}, {1116, 2074},
            {1119, 2076}, {1122, 2077}, {1124, 2079}, {1130, 2082},
            {1133, 2083}, {1136, 2084}, {1139, 2085}, {1142, 2086},
            {1148, 2087}, {1166, 2087}, {1170, 2086}, {1174, 2085},
            {1177, 2084}, {1180, 2083}, {1188, 2079}, {1190, 2077},
            {1193, 2076}, {1196, 2074}, {1199, 2072}, {1202, 2070},
            {1207, 2066}, {1216, 2058}, {1231, 2044}, {1265, 2011},
            {1314, 1962}, {1360, 1917}, {1361, 1917}, {1408, 1871},
            {1457, 1822}, {1508, 1773}, {1512, 1768}, {1560, 1722},
            {1617, 1665}, {1671, 1613}, {1730, 1554}, {1784, 1502},
            {1786, 1500}, {1787, 1498}, {1846, 1440}, {1850, 1437},
            {1908, 1380}, {1974, 1314}, {2034, 1256}, {2036, 1254},
            {2100, 1191}, {2169, 1123}, {2242, 1051}, {2315, 979},
            {2395, 900}, {2426, 869}, {2435, 859}, {2438, 855},
            {2440, 852}, {2442, 849}, {2443, 846}, {2445, 844},
            {2446, 842}, {2446, 840}, {2448, 837}, {2449, 834},
            {2450, 829}, {2450, 814}, {2449, 809}, {2448, 806},
            {2447, 803}, {2442, 793}, {2440, 790}, {2438, 787},
            {2434, 782}, {2428, 775}, {2416, 762}, {2411, 758},
            {2342, 688}, {2340, 686}, {2338, 684}, {2266, 610},
            {2260, 605}, {2170, 513}, {2075, 417}, {2073, 415},
            {2069, 412}, {1955, 297}, {1955, 296}, {1913, 254},
            {1904, 246}, {1897, 240}, {1894, 238}, {1891, 236},
            {1888, 234}, {1880, 230}, {1877, 229}, {1874, 228},
            {1870, 227}
    };
    Mat contour(N, 1, CV_32FC2, (void*)pts_);

    RotatedRect rr = cv::minAreaRect(contour);

    EXPECT_TRUE(checkMinAreaRect(rr, contour)) << rr.center << " " << rr.size << " " << rr.angle;
}

TEST(Imgproc_minEnclosingTriangle, regression_17585)
{
    const int N = 3;
    float pts_[N][2] = { {0, 0}, {0, 1}, {1, 1} };
    cv::Mat points(N, 2, CV_32FC1, static_cast<void*>(pts_));
    vector<Point2f> triangle;

    EXPECT_NO_THROW(minEnclosingTriangle(points, triangle));
}

TEST(Imgproc_minEnclosingTriangle, regression_20890)
{
    vector<Point> points;
    points.push_back(Point(0, 0));
    points.push_back(Point(0, 1));
    points.push_back(Point(1, 1));
    vector<Point2f> triangle;

    EXPECT_NO_THROW(minEnclosingTriangle(points, triangle));
}

TEST(Imgproc_minEnclosingTriangle, regression_mat_with_diff_channels)
{
    const int N = 3;
    float pts_[N][2] = { {0, 0}, {0, 1}, {1, 1} };
    cv::Mat points1xN(1, N, CV_32FC2, static_cast<void*>(pts_));
    cv::Mat pointsNx1(N, 1, CV_32FC2, static_cast<void*>(pts_));
    vector<Point2f> triangle;

    EXPECT_NO_THROW(minEnclosingTriangle(points1xN, triangle));
    EXPECT_NO_THROW(minEnclosingTriangle(pointsNx1, triangle));
}

//==============================================================================

typedef testing::TestWithParam<tuple<int, int>> fitLine_Modes;

TEST_P(fitLine_Modes, accuracy)
{
    const int data_type = get<0>(GetParam());
    const int dist_type = get<1>(GetParam());
    const int CN = CV_MAT_CN(data_type);
    const int res_type = CV_32FC(CN);

    for (int ITER = 0; ITER < 20; ++ITER)
    {
        SCOPED_TRACE(cv::format("iteration %d", ITER));

        Mat v0(1, 1, data_type), v1(1, 1, data_type); // pt = v0 + v1 * t
        Mat v1n;

        RNG& rng = TS::ptr()->get_rng();
        cvtest::randUni(rng, v0, Scalar::all(1), Scalar::all(100));
        cvtest::randUni(rng, v1, Scalar::all(1), Scalar::all(100));
        normalize(v1, v1n, 1, 0, NORM_L2, res_type);
        v0.convertTo(v0, res_type);
        v1.convertTo(v1, res_type);

        const int NUM = rng.uniform(30, 100);
        Mat points(NUM, 1, data_type, Scalar::all(0));
        for (int i = 0; i < NUM; ++i)
        {
            Mat pt = v0 + v1 * i;
            if (CV_MAT_DEPTH(data_type) == CV_32F)
            {
                Mat noise = cvtest::randomMat(rng, Size(1, 1), res_type, -0.01, 0.01, false);
                pt += noise;

            }
            pt.copyTo(points.row(i));
        }

        Mat line_;
        cv::fitLine(points, line_, dist_type, 0, 0.1, 0.01);
        Mat line = line_.reshape(points.channels(), 1);

        // check result type and size
        EXPECT_EQ(res_type, line.type());
        EXPECT_EQ(Size(2, 1), line.size());

        // check result pt1
        const double angle = line.col(0).dot(v1n);
        EXPECT_NEAR(abs(angle), 1, 1e-2);

        // put result pt0 to the original equation (pt = v0 + v1 * t) and find "t"
        Mat diff = line.col(1) - v0;
        cv::divide(diff, v1, diff);
        cv::divide(diff, diff.at<float>(0, 0), diff);
        const Mat unit(1, 1, res_type, Scalar::all(1));
        EXPECT_NEAR(cvtest::norm(diff, unit, NORM_L1), 0, 0.01);
    }
}

INSTANTIATE_TEST_CASE_P(/**/,
    fitLine_Modes,
    testing::Combine(
        testing::Values(CV_32FC2, CV_32FC3, CV_32SC2, CV_32SC3),
        testing::Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR, DIST_WELSCH, DIST_HUBER)));

//==============================================================================

inline float normAngle(float angle_deg)
{
    while (angle_deg < 0.f)
        angle_deg += 180.f;
    while (angle_deg > 180.f)
        angle_deg -= 180.f;
    if (abs(angle_deg - 180.f) < 0.01) // border case
        angle_deg = 0.f;
    return angle_deg;
}

inline float angleToDeg(float angle_rad)
{
    return angle_rad * 180.f / (float)M_PI;
}

inline float angleDiff(float a, float b)
{
    float res = a - b;
    return normAngle(res);
}

typedef testing::TestWithParam<int> fitEllipse_Modes;

TEST_P(fitEllipse_Modes, accuracy)
{
    const int data_type = GetParam();
    const float int_scale = 1000.f;
    const Size sz(1, 2);
    const Matx22f rot {0.f, -1.f, 1.f, 0.f};
    RNG& rng = TS::ptr()->get_rng();

    for (int ITER = 0; ITER < 20; ++ITER)
    {
        SCOPED_TRACE(cv::format("iteration %d", ITER));

        Mat f0(sz, CV_32FC1), f1(sz, CV_32FC1), f2(sz, CV_32FC1);
        cvtest::randUni(rng, f0, Scalar::all(-100), Scalar::all(100));
        cvtest::randUni(rng, f1, Scalar::all(-100), Scalar::all(100));
        if (ITER % 4 == 0)
        {
            // 0/90 degrees case
            f1.at<float>(0, 0) = 0.;
        }
        // f2 is orthogonal to f1 and scaled
        f2 = rot * f1 * cvtest::randomDouble(0.01, 3);

        const Point2f ref_center(f0.at<float>(0), f0.at<float>(1));
        const Size2f ref_size(
            (float)cvtest::norm(f1, NORM_L2) * 2.f,
            (float)cvtest::norm(f2, NORM_L2) * 2.f);
        const float ref_angle1 = angleToDeg(atan(f1.at<float>(1) / f1.at<float>(0)));
        const float ref_angle2 = angleToDeg(atan(f2.at<float>(1) / f2.at<float>(0)));

        const int NUM = rng.uniform(10, 30);
        Mat points(NUM, 1, data_type, Scalar::all(0));
        for (int i = 0; i < NUM; ++i)
        {
            Mat pt = f0 + f1 * sin(i) + f2 * cos(i);
            pt = pt.reshape(2);
            if (data_type == CV_32SC2)
            {
                pt.convertTo(points.row(i), CV_32SC2, int_scale);
            }
            else if (data_type == CV_32FC2)
            {
                pt.copyTo(points.row(i));
            }
            else
            {
                FAIL() << "unsupported data type: " << data_type;
            }
        }

        RotatedRect res = cv::fitEllipse(points);

        if (data_type == CV_32SC2)
        {
            res.center /= int_scale;
            res.size = Size2f(res.size.width / int_scale, res.size.height / int_scale);
        }
        const bool sizeSwap = (res.size.width < res.size.height) != (ref_size.width < ref_size.height);
        if (sizeSwap)
        {
            std::swap(res.size.width, res.size.height);
        }
        EXPECT_FALSE(res.size.empty());
        EXPECT_POINT2_NEAR(res.center, ref_center, 0.01);
        const float sizeDiff = (data_type == CV_32FC2) ? 0.1f : 1.f;
        EXPECT_NEAR(min(res.size.width, res.size.height), min(ref_size.width, ref_size.height), sizeDiff);
        EXPECT_NEAR(max(res.size.width, res.size.height), max(ref_size.width, ref_size.height), sizeDiff);
        if (sizeSwap)
        {
            EXPECT_LE(angleDiff(ref_angle2, res.angle), 0.1);
        }
        else
        {
            EXPECT_LE(angleDiff(ref_angle1, res.angle), 0.1);
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/,
    fitEllipse_Modes,
        testing::Values(CV_32FC2, CV_32SC2));

//==============================================================================

TEST(fitEllipse, small)
{
    Size sz(50, 50);
    vector<vector<Point> > c;
    c.push_back(vector<Point>());
    int scale = 1;
    Point ofs = Point(0,0);//sz.width/2, sz.height/2) - Point(4,4)*scale;
    c[0].push_back(Point(2, 0)*scale+ofs);
    c[0].push_back(Point(0, 2)*scale+ofs);
    c[0].push_back(Point(0, 6)*scale+ofs);
    c[0].push_back(Point(2, 8)*scale+ofs);
    c[0].push_back(Point(6, 8)*scale+ofs);
    c[0].push_back(Point(8, 6)*scale+ofs);
    c[0].push_back(Point(8, 2)*scale+ofs);
    c[0].push_back(Point(6, 0)*scale+ofs);

    RotatedRect e = cv::fitEllipse(c[0]);

    EXPECT_NEAR(e.center.x, 4, 1.f);
    EXPECT_NEAR(e.center.y, 4, 1.f);
    EXPECT_NEAR(e.size.width, 9, 1.);
    EXPECT_NEAR(e.size.height, 9, 1.f);
}

//==============================================================================

// points stored in rows
inline static int findPointInMat(const Mat & data, const Mat & point)
{
    for (int i = 0; i < data.rows; ++i)
        if (cvtest::norm(data.row(i), point, NORM_L1) == 0)
            return i;
    return -1;
}

// > 0 - "pt" is to the right of AB
// < 0 - "pt" is to the left of AB
// points stored in rows
inline static double getSide(const Mat & ptA, const Mat & ptB, const Mat & pt)
{
    Mat d0 = pt - ptA, d1 = ptB - pt, prod;
    vconcat(d0, d1, prod);
    prod = prod.reshape(1);
    if (prod.depth() == CV_32S)
        prod.convertTo(prod, CV_32F);
    return determinant(prod);
}

typedef testing::TestWithParam<perf::MatDepth> convexHull_Modes;

TEST_P(convexHull_Modes, accuracy)
{
    const int data_type = CV_MAKE_TYPE(GetParam(), 2);
    RNG & rng = TS::ptr()->get_rng();

    for (int ITER = 0; ITER < 20; ++ITER)
    {
        SCOPED_TRACE(cv::format("iteration %d", ITER));

        const int NUM = cvtest::randomInt(5, 100);
        Mat points(NUM, 1, data_type, Scalar::all(0));
        cvtest::randUni(rng, points, Scalar(-10), Scalar::all(10));

        Mat hull, c_hull, indexes;
        cv::convexHull(points, hull, false, true); // default parameters
        cv::convexHull(points, c_hull, true, true); // counter-clockwise
        cv::convexHull(points, indexes, false, false); // point indexes

        ASSERT_EQ(hull.size().width, 1);
        ASSERT_GE(hull.size().height, 3);
        ASSERT_EQ(hull.size(), c_hull.size());
        ASSERT_EQ(hull.size(), indexes.size());

        // find shift between hull and counter-clockwise hull
        const int c_diff = findPointInMat(hull, c_hull.row(0));
        ASSERT_NE(c_diff, -1);

        const int sz = (int)hull.total();
        for (int i = 0; i < sz; ++i)
        {
            SCOPED_TRACE(cv::format("vertex %d", i));

            Mat prev = (i == 0) ? hull.row(sz - 1) : hull.row(i - 1);
            Mat cur = hull.row(i);
            Mat next = (i != sz - 1) ? hull.row(i + 1) : hull.row(0);
            // 1. "cur' is one of points
            EXPECT_NE(findPointInMat(points, cur), -1);
            // 2. convexity: "cur" is on right side of "prev - next" edge
            EXPECT_GE(getSide(prev, next, cur), 0);
            // 3. all points are inside polygon - on the left side of "cur - next" edge
            for (int j = 0; j < points.rows; ++j)
            {
                SCOPED_TRACE(cv::format("point %d", j));
                EXPECT_LE(getSide(cur, next, points.row(j)), 0);
            }
            // check counter-clockwise hull
            const int c_idx = (sz - i + c_diff) % sz;
            Mat c_cur = c_hull.row(c_idx);
            EXPECT_MAT_NEAR(cur, c_cur, 0);
            // check indexed hull
            const int pt_index = indexes.at<int>(i);
            EXPECT_MAT_NEAR(cur, points.row(pt_index), 0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/,
    convexHull_Modes,
        testing::Values(CV_32F, CV_32S));


//==============================================================================

typedef testing::TestWithParam<perf::MatDepth> minAreaRect_Modes;

TEST_P(minAreaRect_Modes, accuracy)
{
    const int data_type = CV_MAKE_TYPE(GetParam(), 2);
    RNG & rng = TS::ptr()->get_rng();
    for (int ITER = 0; ITER < 20; ++ITER)
    {
        SCOPED_TRACE(cv::format("iteration %d", ITER));

        const int NUM = cvtest::randomInt(5, 100);
        Mat points(NUM, 1, data_type, Scalar::all(0));
        cvtest::randUni(rng, points, Scalar(-10), Scalar::all(10));

        const RotatedRect res = cv::minAreaRect(points);
        Point2f box_pts[4] {};
        res.points(box_pts);

        // check that the box contains all the points - all on one side
        double common_side = 0.;
        bool edgeHasPoint[4] {0};
        for (int i = 0; i < 4; ++i)
        {
            const int j = (i == 3) ? 0 : i + 1;
            Mat cur(1, 1, CV_32FC2, box_pts + i);
            Mat next(1, 1, CV_32FC2, box_pts + j);
            for (int k = 0; k < points.rows; ++k)
            {
                SCOPED_TRACE(cv::format("point %d", j));
                Mat one_point;
                points.row(k).convertTo(one_point, CV_32FC2);
                const double side = getSide(cur, next, one_point);
                if (abs(side) < 0.01) // point on edge - no need to check
                {
                    edgeHasPoint[i] = true;
                    continue;
                }
                if (common_side == 0.) // initial state
                {
                    common_side = side > 0 ? 1. : -1.; // only sign matters
                }
                else
                {
                    EXPECT_EQ(common_side > 0, side > 0) << common_side << ", " << side;
                }
            }
        }
        EXPECT_TRUE(edgeHasPoint[0] && edgeHasPoint[1] && edgeHasPoint[2] && edgeHasPoint[3]);
    }

}

INSTANTIATE_TEST_CASE_P(/**/,
    minAreaRect_Modes,
        testing::Values(CV_32F, CV_32S));


//==============================================================================

// true if "point" is on one of hull's edges
inline static bool isPointOnHull(const Mat &hull, const Mat &point, const double thresh = 0.01)
{
    const int sz = hull.rows;
    for (int k = 0; k < sz; ++k)
    {
        const double side = getSide(hull.row(k), hull.row(k == sz - 1 ? 0 : k + 1), point);
        if (abs(side) < thresh)
            return true;
    }
    return false;
}

// true if one of hull's edges touches "A-B"
inline static bool isEdgeOnHull(const Mat &hull, const Mat &ptA, const Mat &ptB, const double thresh = 0.01)
{
    const int sz = hull.rows;
    double prev_side = getSide(ptA, ptB, hull.row(sz - 1));
    for (int k = 0; k < sz; ++k)
    {
        Mat cur = hull.row(k);
        const double cur_side = getSide(ptA, ptB, cur);
        if (abs(prev_side) < thresh && abs(cur_side) < thresh)
            return true;
        prev_side = cur_side;
    }
    return false;
}

typedef testing::TestWithParam<perf::MatDepth> minEnclosingTriangle_Modes;

TEST_P(minEnclosingTriangle_Modes, accuracy)
{
    const int data_type = CV_MAKETYPE(GetParam(), 2);
    RNG & rng = TS::ptr()->get_rng();
    for (int ITER = 0; ITER < 20; ++ITER)
    {
        SCOPED_TRACE(cv::format("iteration %d", ITER));

        const int NUM = cvtest::randomInt(5, 100);
        Mat points(NUM, 1, data_type, Scalar::all(0));
        cvtest::randUni(rng, points, Scalar::all(-100), Scalar::all(100));

        Mat triangle;
        const double area = cv::minEnclosingTriangle(points, triangle);

        ASSERT_GT(area, 0.0001);
        ASSERT_EQ(triangle.type(), CV_32FC2);
        triangle = triangle.reshape(2, 1);
        ASSERT_EQ(triangle.size(), Size(3, 1));

        Mat hull;
        cv::convexHull(points, hull);
        hull.convertTo(hull, CV_32FC2);

        // check that all points are enclosed by triangle sides
        double commonSide = 0.;
        bool hasEdgeOnHull = false;
        for (int i = 0; i < 3; ++i)
        {
            SCOPED_TRACE(cv::format("edge %d", i));
            const int j = (i == 2) ? 0 : i + 1;
            Mat cur = triangle.col(i);
            Mat next = triangle.col(j);
            for (int k = 0; k < points.rows; ++k)
            {
                SCOPED_TRACE(cv::format("point %d", k));
                Mat pt;
                points.row(k).convertTo(pt, CV_32FC2);
                const double side = getSide(cur, next, pt);
                if (abs(side) < 0.01) // point on edge - no need to check
                    continue;
                if (commonSide == 0.f) // initial state
                {
                    commonSide = side > 0 ? 1.f : -1.f; // only sign matters
                }
                else
                {
                    // either on the same side or close to zero
                    EXPECT_EQ(commonSide > 0, side > 0) << commonSide << ", side=" << side;
                }
            }

            // triangle mid-points must be on the hull edges
            const Mat midPoint = (cur + next) / 2;
            EXPECT_TRUE(isPointOnHull(hull, midPoint));

            // at least one of hull edges must be on tirangle edge
            hasEdgeOnHull = hasEdgeOnHull || isEdgeOnHull(hull, cur, next);
        }
        EXPECT_TRUE(hasEdgeOnHull);
    }
}

INSTANTIATE_TEST_CASE_P(/**/,
    minEnclosingTriangle_Modes,
        testing::Values(CV_32F, CV_32S));

//==============================================================================

typedef testing::TestWithParam<perf::MatDepth> minEnclosingCircle_Modes;

TEST_P(minEnclosingCircle_Modes, accuracy)
{
    const int data_type = CV_MAKETYPE(GetParam(), 2);
    RNG & rng = TS::ptr()->get_rng();
    for (int ITER = 0; ITER < 20; ++ITER)
    {
        SCOPED_TRACE(cv::format("iteration %d", ITER));

        const int NUM = cvtest::randomInt(5, 100);
        Mat points(NUM, 1, data_type, Scalar::all(0)), fpoints;
        cvtest::randUni(rng, points, Scalar::all(-100), Scalar::all(100));
        points.convertTo(fpoints, CV_32FC2);

        Point2f center {};
        float radius = 0.f;
        cv::minEnclosingCircle(points, center, radius);

        vector<int> boundPts; // indexes
        for (int i = 0; i < NUM; ++i)
        {
            Point2f pt = fpoints.at<Point2f>(i);
            const double dist = cv::norm(pt - center);
            EXPECT_LE(dist, radius);
            if (abs(dist - radius) < 0.01)
                boundPts.push_back(i);
        }
        // 2 points on diameter or at least 3 points on circle
        EXPECT_GE(boundPts.size(), 2llu);

        // 2 points on diameter
        if (boundPts.size() == 2llu)
        {
            const Point2f diff = fpoints.at<Point2f>(boundPts[0]) - fpoints.at<Point2f>(boundPts[1]);
            EXPECT_NEAR(cv::norm(diff), 2 * radius, 0.001);
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/,
    minEnclosingCircle_Modes,
        testing::Values(CV_32F, CV_32S));

//==============================================================================

TEST(minEnclosingCircle, three_points)
{
    RNG & rng = TS::ptr()->get_rng();
    Point2f center = Point2f(rng.uniform(0.0f, 1000.0f), rng.uniform(0.0f, 1000.0f));;
    float radius = rng.uniform(0.0f, 500.0f);
    float angle = (float)rng.uniform(0.0f, (float)(CV_2PI));
    vector<Point2f> pts;
    pts.push_back(center + Point2f(radius * cos(angle), radius * sin(angle)));
    angle += (float)CV_PI;
    pts.push_back(center + Point2f(radius * cos(angle), radius * sin(angle)));
    float radius2 = radius * radius;
    float x = rng.uniform(center.x - radius, center.x + radius);
    float deltaX = x - center.x;
    float upperBoundY = sqrt(radius2 - deltaX * deltaX);
    float y = rng.uniform(center.y - upperBoundY, center.y + upperBoundY);
    pts.push_back(Point2f(x, y));
    // Find the minimum area enclosing circle
    Point2f calcCenter;
    float calcRadius;
    cv::minEnclosingCircle(pts, calcCenter, calcRadius);
    const float delta = (float)cv::norm(calcCenter - center) + abs(calcRadius - radius);
    EXPECT_LE(delta, 1.f);
}

}} // namespace
/* End of file. */
