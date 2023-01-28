// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef OPENCV_OBJDETECT_ARUCO_DETECTOR_HPP
#define OPENCV_OBJDETECT_ARUCO_DETECTOR_HPP

#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_board.hpp>

namespace cv {
namespace aruco {

//! @addtogroup objdetect_aruco
//! @{

enum CornerRefineMethod{
    CORNER_REFINE_NONE,     ///< Tag and corners detection based on the ArUco approach
    CORNER_REFINE_SUBPIX,   ///< ArUco approach and refine the corners locations using corner subpixel accuracy
    CORNER_REFINE_CONTOUR,  ///< ArUco approach and refine the corners locations using the contour-points line fitting
    CORNER_REFINE_APRILTAG, ///< Tag and corners detection based on the AprilTag 2 approach @cite wang2016iros
};

/** @brief struct DetectorParameters is used by ArucoDetector
 */
struct CV_EXPORTS_W_SIMPLE DetectorParameters {
    CV_WRAP DetectorParameters() {
        adaptiveThreshWinSizeMin = 3;
        adaptiveThreshWinSizeMax = 23;
        adaptiveThreshWinSizeStep = 10;
        adaptiveThreshConstant = 7;
        minMarkerPerimeterRate = 0.03;
        maxMarkerPerimeterRate = 4.;
        polygonalApproxAccuracyRate = 0.03;
        minCornerDistanceRate = 0.05;
        minDistanceToBorder = 3;
        minMarkerDistanceRate = 0.125;
        cornerRefinementMethod = (int)CORNER_REFINE_NONE;
        cornerRefinementWinSize = 5;
        relativeCornerRefinmentWinSize = 0.3f;
        cornerRefinementMaxIterations = 30;
        cornerRefinementMinAccuracy = 0.1;
        markerBorderBits = 1;
        perspectiveRemovePixelPerCell = 4;
        perspectiveRemoveIgnoredMarginPerCell = 0.13;
        maxErroneousBitsInBorderRate = 0.35;
        minOtsuStdDev = 5.0;
        errorCorrectionRate = 0.6;
        aprilTagQuadDecimate = 0.0;
        aprilTagQuadSigma = 0.0;
        aprilTagMinClusterPixels = 5;
        aprilTagMaxNmaxima = 10;
        aprilTagCriticalRad = (float)(10* CV_PI /180);
        aprilTagMaxLineFitMse = 10.0;
        aprilTagMinWhiteBlackDiff = 5;
        aprilTagDeglitch = 0;
        detectInvertedMarker = false;
        useAruco3Detection = false;
        minSideLengthCanonicalImg = 32;
        minMarkerLengthRatioOriginalImg = 0.0;
    }

    /** @brief Read a new set of DetectorParameters from FileNode (use FileStorage.root()).
     */
    CV_WRAP bool readDetectorParameters(const FileNode& fn);

    /** @brief Write a set of DetectorParameters to FileStorage
     */
    CV_WRAP bool writeDetectorParameters(FileStorage& fs, const String& name = String());

    /// minimum window size for adaptive thresholding before finding contours (default 3).
    CV_PROP_RW int adaptiveThreshWinSizeMin;

    /// maximum window size for adaptive thresholding before finding contours (default 23).
    CV_PROP_RW int adaptiveThreshWinSizeMax;

    /// increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding (default 10).
    CV_PROP_RW int adaptiveThreshWinSizeStep;

    /// constant for adaptive thresholding before finding contours (default 7)
    CV_PROP_RW double adaptiveThreshConstant;

    /** @brief determine minimum perimeter for marker contour to be detected.
     *
     * This is defined as a rate respect to the maximum dimension of the input image (default 0.03).
     */
    CV_PROP_RW double minMarkerPerimeterRate;

    /** @brief determine maximum perimeter for marker contour to be detected.
     *
     * This is defined as a rate respect to the maximum dimension of the input image (default 4.0).
     */
    CV_PROP_RW double maxMarkerPerimeterRate;

    /// minimum accuracy during the polygonal approximation process to determine which contours are squares. (default 0.03)
    CV_PROP_RW double polygonalApproxAccuracyRate;

    /// minimum distance between corners for detected markers relative to its perimeter (default 0.05)
    CV_PROP_RW double minCornerDistanceRate;

    /// minimum distance of any corner to the image border for detected markers (in pixels) (default 3)
    CV_PROP_RW int minDistanceToBorder;

    /** @brief minimum average distance between the corners of the two markers to be grouped (default 0.125).
     *
     * The rate is relative to the smaller perimeter of the two markers.
     * Two markers are grouped if average distance between the corners of the two markers is less than
     * min(MarkerPerimeter1, MarkerPerimeter2)*minMarkerDistanceRate.
     *
     * default value is 0.125 because 0.125*MarkerPerimeter = (MarkerPerimeter / 4) * 0.5 = half the side of the marker.
     *
     * @note default value was changed from 0.05 after 4.8.1 release, because the filtering algorithm has been changed.
     * Now a few candidates from the same group can be added to the list of candidates if they are far from each other.
     * @sa minGroupDistance.
     */
    CV_PROP_RW double minMarkerDistanceRate;

    /** @brief minimum average distance between the corners of the two markers in group to add them to the list of candidates
     *
     * The average distance between the corners of the two markers is calculated relative to its module size (default 0.21).
     */
    CV_PROP_RW float minGroupDistance = 0.21f;

    /** @brief default value CORNER_REFINE_NONE */
    CV_PROP_RW int cornerRefinementMethod;

    /** @brief maximum window size for the corner refinement process (in pixels) (default 5).
     *
     * The window size may decrease if the ArUco marker is too small, check relativeCornerRefinmentWinSize.
     * The final window size is calculated as:
     * min(cornerRefinementWinSize, averageArucoModuleSize*relativeCornerRefinmentWinSize),
     * where averageArucoModuleSize is average module size of ArUco marker in pixels.
     * (ArUco marker is composed of black and white modules)
     */
    CV_PROP_RW int cornerRefinementWinSize;

    /** @brief Dynamic window size for corner refinement relative to Aruco module size (default 0.3).
     *
     * The final window size is calculated as:
     * min(cornerRefinementWinSize, averageArucoModuleSize*relativeCornerRefinmentWinSize),
     * where averageArucoModuleSize is average module size of ArUco marker in pixels.
     * (ArUco marker is composed of black and white modules)
     * In the case of markers located far from each other, it may be useful to increase the value of the parameter to 0.4-0.5.
     * In the case of markers located close to each other, it may be useful to decrease the parameter value to 0.1-0.2.
     */
    CV_PROP_RW float relativeCornerRefinmentWinSize;

    /// maximum number of iterations for stop criteria of the corner refinement process (default 30).
    CV_PROP_RW int cornerRefinementMaxIterations;

    /// minimum error for the stop cristeria of the corner refinement process (default: 0.1)
    CV_PROP_RW double cornerRefinementMinAccuracy;

    /// number of bits of the marker border, i.e. marker border width (default 1).
    CV_PROP_RW int markerBorderBits;

    /// number of bits (per dimension) for each cell of the marker when removing the perspective (default 4).
    CV_PROP_RW int perspectiveRemovePixelPerCell;

    /** @brief width of the margin of pixels on each cell not considered for the determination of the cell bit.
     *
     * Represents the rate respect to the total size of the cell, i.e. perspectiveRemovePixelPerCell (default 0.13)
     */
    CV_PROP_RW double perspectiveRemoveIgnoredMarginPerCell;

    /** @brief  maximum number of accepted erroneous bits in the border (i.e. number of allowed white bits in the border).
     *
     * Represented as a rate respect to the total number of bits per marker (default 0.35).
     */
    CV_PROP_RW double maxErroneousBitsInBorderRate;

    /** @brief minimun standard deviation in pixels values during the decodification step to apply Otsu
     * thresholding (otherwise, all the bits are set to 0 or 1 depending on mean higher than 128 or not) (default 5.0)
     */
    CV_PROP_RW double minOtsuStdDev;

    /// error correction rate respect to the maximun error correction capability for each dictionary (default 0.6).
    CV_PROP_RW double errorCorrectionRate;

    /** @brief April :: User-configurable parameters.
     *
     * Detection of quads can be done on a lower-resolution image, improving speed at a cost of
     * pose accuracy and a slight decrease in detection rate. Decoding the binary payload is still
     */
    CV_PROP_RW float aprilTagQuadDecimate;

    /// what Gaussian blur should be applied to the segmented image (used for quad detection?)
    CV_PROP_RW float aprilTagQuadSigma;

    // April :: Internal variables
    /// reject quads containing too few pixels (default 5).
    CV_PROP_RW int aprilTagMinClusterPixels;

    /// how many corner candidates to consider when segmenting a group of pixels into a quad (default 10).
    CV_PROP_RW int aprilTagMaxNmaxima;

    /** @brief reject quads where pairs of edges have angles that are close to straight or close to 180 degrees.
     *
     * Zero means that no quads are rejected. (In radians) (default 10*PI/180)
     */
    CV_PROP_RW float aprilTagCriticalRad;

    /// when fitting lines to the contours, what is the maximum mean squared error
    CV_PROP_RW float aprilTagMaxLineFitMse;

    /** @brief add an extra check that the white model must be (overall) brighter than the black model.
     *
     * When we build our model of black & white pixels, we add an extra check that the white model must be (overall)
     * brighter than the black model. How much brighter? (in pixel values, [0,255]), (default 5)
     */
    CV_PROP_RW int aprilTagMinWhiteBlackDiff;

    /// should the thresholded image be deglitched? Only useful for very noisy images (default 0).
    CV_PROP_RW int aprilTagDeglitch;

    /** @brief to check if there is a white marker.
     *
     * In order to generate a "white" marker just invert a normal marker by using a tilde, ~markerImage. (default false)
     */
    CV_PROP_RW bool detectInvertedMarker;

    /** @brief enable the new and faster Aruco detection strategy.
     *
     * Proposed in the paper:
     * Romero-Ramirez et al: Speeded up detection of squared fiducial markers (2018)
     * https://www.researchgate.net/publication/325787310_Speeded_Up_Detection_of_Squared_Fiducial_Markers
     */
    CV_PROP_RW bool useAruco3Detection;

    /// minimum side length of a marker in the canonical image. Latter is the binarized image in which contours are searched.
    CV_PROP_RW int minSideLengthCanonicalImg;

    /// range [0,1], eq (2) from paper. The parameter tau_i has a direct influence on the processing speed.
    CV_PROP_RW float minMarkerLengthRatioOriginalImg;
};

/** @brief struct RefineParameters is used by ArucoDetector
 */
struct CV_EXPORTS_W_SIMPLE RefineParameters {
    CV_WRAP RefineParameters(float minRepDistance = 10.f, float errorCorrectionRate = 3.f, bool checkAllOrders = true);


    /** @brief Read a new set of RefineParameters from FileNode (use FileStorage.root()).
     */
    CV_WRAP bool readRefineParameters(const FileNode& fn);

    /** @brief Write a set of RefineParameters to FileStorage
     */
    CV_WRAP bool writeRefineParameters(FileStorage& fs, const String& name = String());

    /** @brief minRepDistance minimum distance between the corners of the rejected candidate and the reprojected marker
    in order to consider it as a correspondence.
     */
    CV_PROP_RW float minRepDistance;

    /** @brief errorCorrectionRate rate of allowed erroneous bits respect to the error correction capability of the used dictionary.
     *
     * -1 ignores the error correction step.
     */
    CV_PROP_RW float errorCorrectionRate;

    /** @brief checkAllOrders consider the four posible corner orders in the rejectedCorners array.
     *
     * If it set to false, only the provided corner order is considered (default true).
     */
    CV_PROP_RW bool checkAllOrders;
};

/** @brief The main functionality of ArucoDetector class is detection of markers in an image with detectMarkers() method.
 *
 * After detecting some markers in the image, you can try to find undetected markers from this dictionary with
 * refineDetectedMarkers() method.
 *
 * @see DetectorParameters, RefineParameters
 */
class CV_EXPORTS_W ArucoDetector : public Algorithm
{
public:
    /** @brief Basic ArucoDetector constructor
     *
     * @param dictionary indicates the type of markers that will be searched
     * @param detectorParams marker detection parameters
     * @param refineParams marker refine detection parameters
     */
    CV_WRAP ArucoDetector(const Dictionary &dictionary = getPredefinedDictionary(cv::aruco::DICT_4X4_50),
                          const DetectorParameters &detectorParams = DetectorParameters(),
                          const RefineParameters& refineParams = RefineParameters());

    /** @brief ArucoDetector constructor for multiple dictionaries
     *
     * @param dictionaries indicates the type of markers that will be searched. Empty dictionaries will throw an error.
     * @param detectorParams marker detection parameters
     * @param refineParams marker refine detection parameters
     */
    CV_WRAP ArucoDetector(const std::vector<Dictionary> &dictionaries,
                          const DetectorParameters &detectorParams = DetectorParameters(),
                          const RefineParameters& refineParams = RefineParameters());

    /** @brief Basic marker detection
     *
     * @param image input image
     * @param corners vector of detected marker corners. For each marker, its four corners
     * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
     * the dimensions of this array is Nx4. The order of the corners is clockwise.
     * @param ids vector of identifiers of the detected markers. The identifier is of type int
     * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
     * The identifiers have the same order than the markers in the imgPoints array.
     * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
     * correct codification. Useful for debugging purposes.
     *
     * Performs marker detection in the input image. Only markers included in the first specified dictionary
     * are searched. For each detected marker, it returns the 2D position of its corner in the image
     * and its corresponding identifier.
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponding camera model, if camera parameters are known
     * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
     */
    CV_WRAP void detectMarkers(InputArray image, OutputArrayOfArrays corners, OutputArray ids,
                               OutputArrayOfArrays rejectedImgPoints = noArray()) const;

    /** @brief Marker detection with uncertainty computation
     *
     * @param image input image
     * @param corners vector of detected marker corners. For each marker, its four corners
     * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
     * the dimensions of this array is Nx4. The order of the corners is clockwise.
     * @param ids vector of identifiers of the detected markers. The identifier is of type int
     * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
     * The identifiers have the same order than the markers in the imgPoints array.
     * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
     * correct codification. Useful for debugging purposes.
     * @param markersUnc contains the normalized uncertainty [0;1] of the markers' detection,
     * defined as percentage of incorrect pixel detections, with 0 describing a pixel perfect detection.
     * The uncertainties are of type float (e.g. std::vector<float>)
     *
     * Performs marker detection in the input image. Only markers included in the first specified dictionary
     * are searched. For each detected marker, it returns the 2D position of its corner in the image
     * and its corresponding identifier.
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponding camera model, if camera parameters are known
     * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
     */
    CV_WRAP void detectMarkersWithUnc(InputArray image, OutputArrayOfArrays corners, OutputArray ids,
                               OutputArrayOfArrays rejectedImgPoints = noArray(), OutputArray markersUnc = noArray()) const;

    /** @brief Refine not detected markers based on the already detected and the board layout
     *
     * @param image input image
     * @param board layout of markers in the board.
     * @param detectedCorners vector of already detected marker corners.
     * @param detectedIds vector of already detected marker identifiers.
     * @param rejectedCorners vector of rejected candidates during the marker detection process.
     * @param cameraMatrix optional input 3x3 floating-point camera matrix
     * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
     * @param distCoeffs optional vector of distortion coefficients
     * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
     * @param recoveredIdxs Optional array to returns the indexes of the recovered candidates in the
     * original rejectedCorners array.
     *
     * This function tries to find markers that were not detected in the basic detecMarkers function.
     * First, based on the current detected marker and the board layout, the function interpolates
     * the position of the missing markers. Then it tries to find correspondence between the reprojected
     * markers and the rejected candidates based on the minRepDistance and errorCorrectionRate parameters.
     * If camera parameters and distortion coefficients are provided, missing markers are reprojected
     * using projectPoint function. If not, missing marker projections are interpolated using global
     * homography, and all the marker corners in the board must have the same Z coordinate.
     * @note This function assumes that the board only contains markers from one dictionary, so only the
     * first configured dictionary is used. It has to match the dictionary of the board to work properly.
     */
    CV_WRAP void refineDetectedMarkers(InputArray image, const Board &board,
                                       InputOutputArrayOfArrays detectedCorners,
                                       InputOutputArray detectedIds, InputOutputArrayOfArrays rejectedCorners,
                                       InputArray cameraMatrix = noArray(), InputArray distCoeffs = noArray(),
                                       OutputArray recoveredIdxs = noArray()) const;

    /** @brief Basic marker detection
     *
     * @param image input image
     * @param corners vector of detected marker corners. For each marker, its four corners
     * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
     * the dimensions of this array is Nx4. The order of the corners is clockwise.
     * @param ids vector of identifiers of the detected markers. The identifier is of type int
     * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
     * The identifiers have the same order than the markers in the imgPoints array.
     * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
     * correct codification. Useful for debugging purposes.
     * @param dictIndices vector of dictionary indices for each detected marker. Use getDictionaries() to get the
     * list of corresponding dictionaries.
     *
     * Performs marker detection in the input image. Only markers included in the specific dictionaries
     * are searched. For each detected marker, it returns the 2D position of its corner in the image
     * and its corresponding identifier.
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponding camera model, if camera parameters are known
     * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
     */
    CV_WRAP void detectMarkersMultiDict(InputArray image, OutputArrayOfArrays corners, OutputArray ids,
                               OutputArrayOfArrays rejectedImgPoints = noArray(), OutputArray dictIndices = noArray()) const;

    /** @brief Returns first dictionary from internal list used for marker detection.
     *
     * @return The first dictionary from the configured ArucoDetector.
     */
    CV_WRAP const Dictionary& getDictionary() const;

    /** @brief Sets and replaces the first dictionary in internal list to be used for marker detection.
     *
     * @param dictionary The new dictionary that will replace the first dictionary in the internal list.
     */
    CV_WRAP void setDictionary(const Dictionary& dictionary);

    /** @brief Returns all dictionaries currently used for marker detection as a vector.
     *
     * @return A std::vector<Dictionary> containing all dictionaries used by the ArucoDetector.
     */
    CV_WRAP std::vector<Dictionary> getDictionaries() const;

    /** @brief Sets the entire collection of dictionaries to be used for marker detection, replacing any existing dictionaries.
     *
     * @param dictionaries A std::vector<Dictionary> containing the new set of dictionaries to be used.
     *
     * Configures the ArucoDetector to use the provided vector of dictionaries for marker detection.
     * This method replaces any dictionaries that were previously set.
     * @note Setting an empty vector of dictionaries will throw an error.
     */
    CV_WRAP void setDictionaries(const std::vector<Dictionary>& dictionaries);

    CV_WRAP const DetectorParameters& getDetectorParameters() const;
    CV_WRAP void setDetectorParameters(const DetectorParameters& detectorParameters);

    CV_WRAP const RefineParameters& getRefineParameters() const;
    CV_WRAP void setRefineParameters(const RefineParameters& refineParameters);

    /** @brief Stores algorithm parameters in a file storage
    */
    virtual void write(FileStorage& fs) const override;

    /** @brief simplified API for language bindings
    */
    CV_WRAP inline void write(FileStorage& fs, const String& name) { Algorithm::write(fs, name); }

    /** @brief Reads algorithm parameters from a file storage
    */
    CV_WRAP virtual void read(const FileNode& fn) override;
protected:
    struct ArucoDetectorImpl;
    Ptr<ArucoDetectorImpl> arucoDetectorImpl;
};

/** @brief Draw detected markers in image
 *
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not altered.
 * @param corners positions of marker corners on input image.
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
 * this array should be Nx4. The order of the corners should be clockwise.
 * @param ids vector of identifiers for markers in markersCorners .
 * Optional, if not provided, ids are not painted.
 * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
 * are calculated based on this one to improve visualization.
 *
 * Given an array of detected marker corners and its corresponding ids, this functions draws
 * the markers in the image. The marker borders are painted and the markers identifiers if provided.
 * Useful for debugging purposes.
 */
CV_EXPORTS_W void drawDetectedMarkers(InputOutputArray image, InputArrayOfArrays corners,
                                      InputArray ids = noArray(), Scalar borderColor = Scalar(0, 255, 0));

/** @brief Generate a canonical marker image
 *
 * @param dictionary dictionary of markers indicating the type of markers
 * @param id identifier of the marker that will be returned. It has to be a valid id in the specified dictionary.
 * @param sidePixels size of the image in pixels
 * @param img output image with the marker
 * @param borderBits width of the marker border.
 *
 * This function returns a marker image in its canonical form (i.e. ready to be printed)
 */
CV_EXPORTS_W void generateImageMarker(const Dictionary &dictionary, int id, int sidePixels, OutputArray img,
                                      int borderBits = 1);

//! @}

}
}

#endif
