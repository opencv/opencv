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

namespace cv {
namespace aruco {

/**
 * @brief Represents a single fractal marker detected in an image.
 *
 * This class stores the properties and geometry of a fractal marker, including its ID,
 * matrix representation, mask, submarker IDs, and keypoints. It provides methods to draw
 * the marker and is used as the output of the fractal marker detector.
 */
class CV_EXPORTS_W_SIMPLE FractalArucoMarker
{
public:
    /**
     * @brief Default constructor for FractalArucoMarker.
     */
    CV_WRAP FractalArucoMarker();

    /**
     * @brief Constructs a FractalArucoMarker with the given parameters.
     * @param id The marker ID.
     * @param m The marker matrix (bit pattern).
     * @param corners The 3D corners of the marker.
     * @param id_submarkers The IDs of submarkers.
     */
    CV_WRAP FractalArucoMarker(int id, InputArray m, const std::vector<Point3f>& corners, const std::vector<int>& id_submarkers);

    /**
     * @brief Draws detected marker on the given image.
     * @param image The image on which to draw the marker.
     * @param color The color to use for drawing (default is red).
     */
    CV_WRAP void draw(cv::InputOutputArray img, const std::vector<Point2f>& detected_points, const cv::Scalar color = Scalar(0, 0, 255));

    /**
     * @brief Returns marker id in dictionary
     */
    CV_WRAP int getId() const;

protected:
    /**
     * @brief Internal implementation details for FractalMarker (PIMPL idiom).
     */
    struct FractalMarkerImpl;
    Ptr<FractalMarkerImpl> impl;

    friend class FractalMarkerDictionary;
    friend class FractalDetector;
};

enum PredefinedFractalDictionaryType {
    FRACTAL_2L_6,
    FRACTAL_3L_6,
    FRACTAL_4L_6,
    FRACTAL_5L_6,
};

/**
 * The Fractal Marker generation creates a hierarchical structure of nVal x nVal binary
 * matrices, where the central kVal x kVal region is cleared, and outer regions are
 * randomly configured. Each layer optimizes marker distinctiveness by maximizing its
 * distance from others. Pixel size adjusts per layer, and corners are defined in 3D
 * space, ensuring robust and unique markers.
 *
 */
class CV_EXPORTS_W_SIMPLE FractalMarkerDictionary
{
public:
    FractalMarkerDictionary();
    CV_WRAP FractalMarkerDictionary(Mat compressed);
protected:
    struct FractalMarkerDictionaryImpl;
    Ptr<FractalMarkerDictionaryImpl> impl;
    friend class FractalDetector;
};

CV_EXPORTS_W FractalMarkerDictionary getPredefinedFractalArucoDictionary(PredefinedFractalDictionaryType name);

/**
 * @brief Detector for fractal markers in images.
 *
 * This class detects fractal markers in images and provides their geometry and correspondences.
 * It supports setting detection parameters and retrieving 2D/3D correspondences for pose estimation.
 */
class CV_EXPORTS_W FractalDetector : public Algorithm {
public:
    /**
     * @brief Default constructor for FractalDetector.
     */
    CV_WRAP FractalDetector(const FractalMarkerDictionary &dictionary = getPredefinedFractalArucoDictionary(FRACTAL_4L_6));
    /**
     * @brief Detects fractal markers in the input image.
     * @param img Input image (grayscale or color).
     * @param markers Output vector of detected FractalMarker objects.
     * @param p3d Optional output 3D points (object points).
     * @param p2d Optional output 2D points (image points).
     * @return true if detection succeeded, false otherwise.
     */
    CV_WRAP bool detect(InputArray img, CV_OUT std::vector<int>& marker_ids, CV_OUT std::vector<std::vector<cv::Point2f>>& marker_points,
                        OutputArray p3d = noArray(), OutputArray p2d = noArray());

    /**
     * @brief Sets the parameters for fractal marker detection.
     * @param fractal_config The fractal marker configuration string (e.g., "FRACTAL_4L_6").
     * @param minInternalDistSq The minimum squared distance between internal points (default 150).
     * @param markerSize The physical size of the marker in meters (optional, default -1).
     */
    CV_WRAP void setParams(int minInternalDistSq = 150, float markerSize = -1);

protected:
    /**
     * @brief Internal implementation details for FractalDetector (PIMPL idiom).
     */
    struct FractalDetectorImpl;
    Ptr<FractalDetectorImpl> impl;
};

} // namespace aruco
} // namespace cv

#endif // OPENCV_OBJDETECT_FRACTAL_DETECTOR_HPP
