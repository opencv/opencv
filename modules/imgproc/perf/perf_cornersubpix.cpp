// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

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
    goodFeaturesToTrack(image, detected, numCorners, 0.001, 1.0);
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
