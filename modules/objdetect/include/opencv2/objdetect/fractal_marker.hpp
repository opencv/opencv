// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_FRACTAL_MARKER_HPP
#define OPENCV_OBJDETECT_FRACTAL_MARKER_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <map>

namespace cv {
namespace aruco {

/**
 * @brief Represents a single fractal marker.
 * 
 * This class stores the properties of a fractal marker, such as its ID, matrix representation,
 * mask, submarkers, and keypoints. It also provides methods to manipulate and retrieve marker data.
 */
class CV_EXPORTS_W_SIMPLE FractalMarker : public std::vector<cv::Point2f> {
public:
    /**
     * @brief Constructs a FractalMarker with the given parameters.
     * 
     * @param id The ID of the marker.
     * @param m The matrix representation of the marker.
     * @param corners The 3D corners of the marker.
     * @param id_submarkers The IDs of the submarkers.
     */
    CV_WRAP FractalMarker(int id, cv::Mat m, std::vector<cv::Point3f> corners, std::vector<int> id_submarkers);

    /**
     * @brief Default constructor for FractalMarker.
     */
    CV_WRAP FractalMarker();

    /**
     * @brief Returns the total number of bits in the marker.
     * 
     * @return int The total number of bits.
     */
    CV_WRAP int nBits();

    /**
     * @brief Returns the matrix representation of the marker.
     * 
     * @return cv::Mat The marker matrix.
     */
    CV_WRAP cv::Mat mat();

    /**
     * @brief Returns the mask of the marker.
     * 
     * @return cv::Mat The marker mask.
     */
    CV_WRAP cv::Mat mask();

    /**
     * @brief Returns the IDs of the submarkers.
     * 
     * @return std::vector<int> A vector of submarker IDs.
     */
    CV_WRAP std::vector<int> subMarkers();

    /**
     * @brief Adds a submarker to the current marker.
     * 
     * @param submarker The submarker to add.
     */
    CV_WRAP void addSubFractalMarker(FractalMarker submarker);

    /**
     * @brief Returns the size of the marker.
     * 
     * @return float The size of the marker.
     */
    CV_WRAP float getMarkerSize() const;

    /**
     * @brief Returns the keypoints of the marker.
     * 
     * @return std::vector<cv::KeyPoint> A vector of keypoints.
     */
    CV_WRAP std::vector<cv::KeyPoint> getKeypts();

    /**
     * @brief Draws the marker on the given image.
     * 
     * @param image The image on which to draw the marker.
     * @param color The color of the marker (default is red).
     */
    CV_WRAP void draw(cv::Mat &image, const cv::Scalar color = cv::Scalar(0, 0, 255)) const;

    CV_PROP_RW int id; ///< The ID of the marker.
    CV_PROP_RW std::vector<cv::KeyPoint> keypts; ///< The keypoints of the marker. The first 4 are external corners.

private:
    CV_PROP_RW cv::Mat _M; ///< The matrix representation of the marker.
    CV_PROP_RW cv::Mat _mask; ///< The mask of the marker.
    CV_PROP_RW std::vector<int> _submarkers; ///< The IDs of the submarkers.
};

/**
 * @brief Represents a set of fractal markers and their configurations.
 * 
 * This class manages a collection of fractal markers, including their configurations,
 * relationships, and unit conversions.
 */
class CV_EXPORTS_W_SIMPLE FractalMarkerSet {
public:
    /**
     * @brief Default constructor for FractalMarkerSet.
     */
    CV_WRAP FractalMarkerSet();

    /**
     * @brief Constructs a FractalMarkerSet with the given configuration.
     * 
     * @param config The configuration string (e.g., "FRACTAL_2L_6").
     */
    CV_WRAP FractalMarkerSet(std::string config);

    /**
     * @brief Converts the marker units to meters.
     * 
     * @param size The physical size of the marker in meters.
     * @throws std::runtime_error If the markers are not expressed in pixels or normalized units.
     */
    CV_WRAP void convertToMeters(float size);

    CV_PROP_RW std::map<int, FractalMarker> fractalMarkerCollection; ///< A map of marker IDs to FractalMarker objects.
    CV_PROP_RW std::map<int, std::vector<int>> bits_ids; ///< A map of bit counts to marker IDs.
    CV_PROP_RW int mInfoType; ///< Indicates the unit type (e.g., meters, pixels, or normalized).
    CV_PROP_RW int idExternal; ///< The ID of the external marker.
};

} // namespace aruco
} // namespace cv

#endif // OPENCV_OBJDETECT_FRACTAL_MARKER_HPP