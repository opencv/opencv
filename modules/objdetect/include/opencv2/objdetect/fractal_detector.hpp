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
