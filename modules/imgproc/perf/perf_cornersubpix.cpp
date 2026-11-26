// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

static void detectInitialCorners(const Mat& image, int maxCorners, double qualityLevel,
                                 std::vector<Point2f>& corners)
{
    Mat eig, tmp;
    cornerMinEigenVal(image, eig, 3, 3);
    double maxVal = 0;
    minMaxLoc(eig, 0, &maxVal);
    cv::threshold(eig, eig, maxVal * qualityLevel, 0, THRESH_TOZERO);
    dilate(eig, tmp, Mat());

    std::vector<std::pair<float, Point> > candidates;
    for (int y = 1; y < eig.rows - 1; y++)
    {
        const float* e = eig.ptr<float>(y);
        const float* t = tmp.ptr<float>(y);
        for (int x = 1; x < eig.cols - 1; x++)
            if (e[x] != 0 && e[x] == t[x])
                candidates.push_back(std::make_pair(e[x], Point(x, y)));
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const std::pair<float, Point>& a, const std::pair<float, Point>& b) { return a.first > b.first; });

    corners.clear();
    for (size_t i = 0; i < candidates.size() && (int)corners.size() < maxCorners; i++)
        corners.push_back(Point2f((float)candidates[i].second.x, (float)candidates[i].second.y));
}

typedef tuple<string, int, int> Img_NumCorners_WinSize_t;
typedef perf::TestBaseWithParam<Img_NumCorners_WinSize_t> Img_NumCorners_WinSize;

PERF_TEST_P(Img_NumCorners_WinSize, cornerSubPix,
            testing::Combine(
                testing::Values("stitching/a1.png", "cv/shared/pic5.png"),
                testing::Values(100, 1000, 5000),
                testing::Values(5, 11)
                )
          )
{
    string filename = getDataPath(get<0>(GetParam()));
    int numCorners = get<1>(GetParam());
    int winSize = get<2>(GetParam());

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty()) << "Unable to load source image: " << filename;

    Size win(winSize, winSize);
    Size zeroZone(-1, -1);
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

    int marginX = win.width * 2 + 3;
    int marginY = win.height * 2 + 3;
    ASSERT_GT(image.cols, 2 * marginX);
    ASSERT_GT(image.rows, 2 * marginY);

    std::vector<Point2f> detected;
    detectInitialCorners(image, numCorners, 0.001, detected);
    ASSERT_FALSE(detected.empty());
    std::vector<Point2f> valid;
    for (size_t i = 0; i < detected.size(); i++)
    {
        const Point2f& p = detected[i];
        if (p.x >= marginX && p.x < image.cols - marginX &&
            p.y >= marginY && p.y < image.rows - marginY)
            valid.push_back(p);
    }
    ASSERT_FALSE(valid.empty());

    std::vector<Point2f> cornersInit;
    cornersInit.reserve(numCorners);
    for (int i = 0; i < numCorners; i++)
    {
        Point2f p = valid[i % valid.size()];
        float jitter = 0.25f * ((i % 3) - 1);
        p.x = std::min((float)(image.cols - marginX - 1), std::max((float)marginX, p.x + jitter));
        p.y = std::min((float)(image.rows - marginY - 1), std::max((float)marginY, p.y + jitter));
        cornersInit.push_back(p);
    }

    std::vector<Point2f> corners;
    TEST_CYCLE()
    {
        corners = cornersInit;
        cornerSubPix(image, corners, win, zeroZone, criteria);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
