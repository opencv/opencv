// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGPROC_BINDINGS_HPP
#define OPENCV_IMGPROC_BINDINGS_HPP

// This file contains special overloads for OpenCV bindings
// No need to use these functions in C++ code.

namespace cv {

/** @brief Finds lines in a binary image using the standard Hough transform and get accumulator.
 *
 * @note This function is for bindings use only. Use original function in C++ code
 *
 * @sa HoughLines
 */
CV_WRAP static inline
void HoughLinesWithAccumulator(
        InputArray image, OutputArray lines,
        double rho, double theta, int threshold,
        double srn = 0, double stn = 0,
        double min_theta = 0, double max_theta = CV_PI,
        bool use_edgeval = false
)
{
    std::vector<Vec3f> lines_acc;
    HoughLines(image, lines_acc, rho, theta, threshold, srn, stn, min_theta, max_theta, use_edgeval);
    Mat(lines_acc).copyTo(lines);
}

/** @brief Finds circles in a grayscale image using the Hough transform and get accumulator.
 *
 * @note This function is for bindings use only. Use original function in C++ code
 *
 * @sa HoughCircles
 */
CV_WRAP static inline
void HoughCirclesWithAccumulator(
        InputArray image, OutputArray circles,
        int method, double dp, double minDist,
        double param1 = 100, double param2 = 100,
        int minRadius = 0, int maxRadius = 0
)
{
    std::vector<Vec4f> circles_acc;
    HoughCircles(image, circles_acc, method, dp, minDist, param1, param2, minRadius, maxRadius);
    Mat(1, static_cast<int>(circles_acc.size()), CV_32FC4, &circles_acc.front()).copyTo(circles);
}

}  // namespace

#endif  // OPENCV_IMGPROC_BINDINGS_HPP
