// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file contains wrappers for legacy OpenCV C API

#include "precomp.hpp"
#include "opencv2/calib3d/calib3d_c.h"

using namespace cv;

CV_IMPL void
cvDrawChessboardCorners(CvArr* _image, CvSize pattern_size,
                        CvPoint2D32f* corners, int count, int found)
{
    CV_Assert(corners != NULL); //CV_CheckNULL(corners, "NULL is not allowed for 'corners' parameter");
    Mat image = cvarrToMat(_image);
    CV_StaticAssert(sizeof(CvPoint2D32f) == sizeof(Point2f), "");
    drawChessboardCorners(image, pattern_size, Mat(1, count, traits::Type<Point2f>::value, corners), found != 0);
}

CV_IMPL int
cvFindChessboardCorners(const void* arr, CvSize pattern_size,
                        CvPoint2D32f* out_corners_, int* out_corner_count,
                        int flags)
{
    if (!out_corners_)
        CV_Error( CV_StsNullPtr, "Null pointer to corners" );

    Mat image = cvarrToMat(arr);
    std::vector<Point2f> out_corners;

    if (out_corner_count)
        *out_corner_count = 0;

    bool res = cv::findChessboardCorners(image, pattern_size, out_corners, flags);

    int corner_count = (int)out_corners.size();
    if (out_corner_count)
        *out_corner_count = corner_count;
    CV_CheckLE(corner_count, Size(pattern_size).area(), "Unexpected number of corners");
    for (int i = 0; i < corner_count; ++i)
    {
        out_corners_[i] = cvPoint2D32f(out_corners[i]);
    }
    return res ? 1 : 0;
}
