// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_FRACTAL_DETECTOR_HPP
#define OPENCV_OBJDETECT_FRACTAL_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <map>
#include <string>
#include "opencv2/objdetect/fractal_marker.hpp"

namespace cv {
namespace aruco {

/**
 * @brief The FractalMarkerDetector class detects fractal markers in the images passed.
 */
class CV_EXPORTS_W_SIMPLE FractalMarkerDetector {
public:
    /**
     * @brief Set parameters for the fractal marker detector.
     * @param fractal_config Possible values (FRACTAL_2L_6, FRACTAL_3L_6, FRACTAL_4L_6, FRACTAL_5L_6).
     * @param markerSize The size of the marker in meters (optional).
     */
    CV_WRAP void setParams(const std::string& fractal_config, float markerSize = -1);

    /**
     * @brief Detect fractal markers in the given image.
     * @param img Input image.
     * @return A vector of detected FractalMarker objects.
     */
    CV_WRAP std::vector<FractalMarker> detect(const cv::Mat& img);

    /**
     * @brief Detect fractal markers and retrieve 2D/3D correspondences.
     * @param img Input image.
     * @param p3d Output 3D points.
     * @param p2d Output 2D points.
     * @return A vector of detected FractalMarker objects.
     */
    CV_WRAP std::vector<FractalMarker> detect(const cv::Mat& img, std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d);

private:
    CV_PROP_RW FractalMarkerSet fractalMarkerSet;

    /**
     * @brief Sort the points of a marker in clockwise order.
     * @param marker Input marker points.
     * @return Sorted marker points.
     */
    CV_WRAP static std::vector<cv::Point2f> sort(const std::vector<cv::Point2f>& marker);

    /**
     * @brief Get the subpixel value of an image at a given point.
     * @param im_grey Grayscale image.
     * @param p Point to sample.
     * @return Subpixel intensity value.
     */
    CV_WRAP static float getSubpixelValue(const cv::Mat& im_grey, const cv::Point2f& p);

    /**
     * @brief Get the marker ID from the bit matrix.
     * @param bits Bit matrix of the marker.
     * @param nrotations Number of rotations applied to the marker.
     * @param markersId List of marker IDs.
     * @param markerSet Reference to the FractalMarkerSet.
     * @return Marker ID or -1 if not found.
     */
    CV_WRAP static int getMarkerId(const cv::Mat& bits, int& nrotations, const std::vector<int>& markersId, const FractalMarkerSet& markerSet);

    /**
     * @brief Calculate the perimeter of a marker.
     * @param a Marker points.
     * @return Perimeter length.
     */
    CV_WRAP static int perimeter(const std::vector<cv::Point2f>& a);

    /**
     * @brief Filter keypoints.
     * @param kpoints Keypoints to filter.
     */
    CV_WRAP void kfilter(std::vector<cv::KeyPoint>& kpoints);

    /**
     * @brief Assign classes to keypoints.
     * @param im Input image.
     * @param kpoints Keypoints to classify.
     * @param sizeNorm Normalization size.
     * @param wsize Window size.
     */
    CV_WRAP void assignClass(const cv::Mat& im, std::vector<cv::KeyPoint>& kpoints, float sizeNorm = 0.f, int wsize = 5);
};

} // namespace aruco
} // namespace cv

#endif // OPENCV_OBJDETECT_FRACTAL_DETECTOR_HPP