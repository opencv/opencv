// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

CV_ENUM(RetrMode, RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE)
CV_ENUM(ApproxMode, CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS)

typedef TestBaseWithParam< tuple<Size, RetrMode, ApproxMode, int> > TestFindContours;

PERF_TEST_P(TestFindContours, findContours,
            Combine(
               Values( szVGA, sz1080p ), // image size
               RetrMode::all(), // retrieval mode
               ApproxMode::all(), // approximation method
               Values( 32, 128 ) // blob count
            )
           )
{
    Size img_size = get<0>(GetParam());
    int retr_mode = get<1>(GetParam());
    int approx_method = get<2>(GetParam());
    int blob_count = get<3>(GetParam());

    RNG rng;
    Mat img = Mat::zeros(img_size, CV_8UC1);
    for(int i = 0; i < blob_count; i++ )
    {
        Point center;
        center.x = (unsigned)rng % (img.cols-2);
        center.y = (unsigned)rng % (img.rows-2);
        Size  axes;
        axes.width = ((unsigned)rng % 49 + 2)/2;
        axes.height = ((unsigned)rng % 49 + 2)/2;
        double angle = (unsigned)rng % 180;
        int brightness = (unsigned)rng % 2;

        // keep the border clear
        ellipse( img(Rect(1,1,img.cols-2,img.rows-2)), Point(center), Size(axes), angle, 0., 360., Scalar(brightness), -1);
    }
    vector< vector<Point> > contours;

    TEST_CYCLE() findContours( img, contours, retr_mode, approx_method );

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam< tuple<Size, ApproxMode, int> > TestFindContoursFF;

PERF_TEST_P(TestFindContoursFF, findContours,
    Combine(
        Values(szVGA, sz1080p), // image size
        ApproxMode::all(), // approximation method
        Values(32, 128) // blob count
    )
)
{
    Size img_size = get<0>(GetParam());
    int approx_method = get<1>(GetParam());
    int blob_count = get<2>(GetParam());

    RNG rng;
    Mat img = Mat::zeros(img_size, CV_32SC1);
    for (int i = 0; i < blob_count; i++)
    {
        Point center;
        center.x = (unsigned)rng % (img.cols - 2);
        center.y = (unsigned)rng % (img.rows - 2);
        Size  axes;
        axes.width = ((unsigned)rng % 49 + 2) / 2;
        axes.height = ((unsigned)rng % 49 + 2) / 2;
        double angle = (unsigned)rng % 180;
        int brightness = (unsigned)rng % 2;

        // keep the border clear
        ellipse(img(Rect(1, 1, img.cols - 2, img.rows - 2)), Point(center), Size(axes), angle, 0., 360., Scalar(brightness), -1);
    }
    vector< vector<Point> > contours;

    TEST_CYCLE() findContours(img, contours, RETR_FLOODFILL, approx_method);

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam< tuple<MatDepth, int> > TestBoundingRect;

PERF_TEST_P(TestBoundingRect, BoundingRect,
    Combine(
        testing::Values(CV_32S, CV_32F), // points type
        Values(400, 511, 1000, 10000, 100000) // points count
    )
)

{
    int ptType = get<0>(GetParam());
    int n = get<1>(GetParam());

    Mat pts(n, 2, ptType);
    declare.in(pts, WARMUP_RNG);

    cv::Rect rect;
    TEST_CYCLE() rect = boundingRect(pts);

    SANITY_CHECK_NOTHING();
}

} } // namespace
