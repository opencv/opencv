/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OBJDETECT_HPP
#define OPENCV_OBJDETECT_HPP

#include "opencv2/core.hpp"
#include "opencv2/features.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/graphical_code_detector.hpp"
#include "opencv2/objdetect/mcc_checker_detector.hpp"

/**
@defgroup objdetect Object Detection

@{
    @defgroup objdetect_barcode Barcode detection and decoding
    @defgroup objdetect_qrcode QRCode detection and encoding
    @defgroup objdetect_dnn_face DNN-based face detection and recognition

    Check @ref tutorial_dnn_face "the corresponding tutorial" for more details.

    @defgroup objdetect_common Common functions and classes
    @defgroup objdetect_aruco ArUco markers and boards detection for robust camera pose estimation
    @{
        ArUco Marker Detection
        Square fiducial markers (also known as Augmented Reality Markers) are useful for easy,
        fast and robust camera pose estimation.

        The main functionality of ArucoDetector class is detection of markers in an image. If the markers are grouped
        as a board, then you can try to recover the missing markers with ArucoDetector::refineDetectedMarkers().
        ArUco markers can also be used for advanced chessboard corner finding. To do this, group the markers in the
        CharucoBoard and find the corners of the chessboard with the CharucoDetector::detectBoard().

        The implementation is based on the ArUco Library by R. Muñoz-Salinas and S. Garrido-Jurado @cite Aruco2014.

        Markers can also be detected based on the AprilTag 2 @cite wang2016iros fiducial marker detection method.

        @sa @cite Aruco2014
        This code has been originally developed by Sergio Garrido-Jurado as a project
        for Google Summer of Code 2015 (GSoC 15).

        <br>

        @warning In OpenCV, the order of the returned corners locations for the AprilTag family is not aligned with the ArUco one.\n
        Note that this order is also different from the convention adopted by the official [AprilTag library](https://github.com/AprilRobotics/apriltag/).
        ![](pics/AprilTag_corners_comparison_opencv_april.png) { width=80% }

        <br>

        An overview of the supported ArUco markers family is visible in the following image:
        ![](pics/ArUco_family.png) { width=80% }

        <br>

        An overview of the supported AprilTag markers family is visible in the following image:
        ![](pics/AprilTag_family.png) { width=80% }

        @note The generated images (in the above picture) using @ref aruco::generateImageMarker for the AprilTag markers have been
        rotated by 180 degree in order to match the official AprilTag images.
        When using the @ref aruco::generateImageMarker function, it will output by default a different image from the official AprilTag convention,
        see the [AprilRobotics/apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs) repository.
        This is the reason why you see a different corners order between ArUco and AprilTag in the above image.

        <br>

        For the ArUco marker family, the recommended family is the DICT_ARUCO_MIP_36h12 one, [see](https://stackoverflow.com/a/51511558).
        In general, a smaller marker family (e.g. `4x4` vs `6x6`) should give you a better detection rate with respect to the camera distance,
        at the expense of having more probability to have issues with false detection or marker id decoding error.
        The number of marker ids in a family is also something to take into account with respect to the application use case and the ability
        to correct wrong bits during the marker id decoding process.

        You can download some pregenerated MIP_36h12 ArUco marker images from:
          - https://sourceforge.net/projects/aruco/files/
          - or use the `samples/cpp/tutorial_code/objectDetection/create_marker.cpp` sample to generate the marker image for your
          desired marker family (which uses the @ref aruco::generateImageMarker function)

        For the AprilTag family, you can find some pregenerated marker images in the
        [AprilRobotics/apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs) repository.

        @note For accurate corners location extraction, a white border (to have a strong gradient between white and black transition) around the marker is important.
        This is necessary to precisely extract the marker contour in difficult conditions such as bad illumination, confusing color background, etc.

        <br>

        There are multiple parameters which can be tweaked to improve the marker detection rate or to be adapted to your use case (e.g. image resolution).
        Please refer to the:
          - @ref aruco::DetectorParameters
          - "Detector Parameters" section in the @ref tutorial_aruco_detection tutorial or in the @ref tutorial_aruco_faq page
          - [ArUco Library Documentation](https://drive.google.com/file/d/1OiavRVYVJ-WH88sQg1LUsh8CuJZUQyrX) for additional information from the ArUco library

        The corner refinement method can be changed according to the @ref aruco::CornerRefineMethod to improve the corners location accuracy
        at the expense of more computation time.

        <br>

        To estimate the marker pose with respect to the camera frame, we recommend you to look at the following sources of information:
          - @ref tutorial_aruco_detection for a tutorial about ArUco markers detection
          - @ref _3d for some theoretical background about the pinhole camera model and the @ref calib3d_solvePnP page
          - @ref solvePnP, @ref solvePnPGeneric, @ref solveP3P for the relevant pose estimation methods
    @}

@}
 */

namespace cv
{
//! @addtogroup objdetect_qrcode
//! @{

/** @brief QR code encoder. */
class CV_EXPORTS_W QRCodeEncoder {
protected:
    QRCodeEncoder();  // use ::create()
public:
    virtual ~QRCodeEncoder();

    enum EncodeMode {
        MODE_AUTO              = -1,
        MODE_NUMERIC           = 1, // 0b0001
        MODE_ALPHANUMERIC      = 2, // 0b0010
        MODE_BYTE              = 4, // 0b0100
        MODE_ECI               = 7, // 0b0111
        MODE_KANJI             = 8, // 0b1000
        MODE_STRUCTURED_APPEND = 3  // 0b0011
    };

    enum CorrectionLevel {
        CORRECT_LEVEL_L = 0,
        CORRECT_LEVEL_M = 1,
        CORRECT_LEVEL_Q = 2,
        CORRECT_LEVEL_H = 3
    };

    enum ECIEncodings {
        ECI_SHIFT_JIS = 20,
        ECI_UTF8 = 26,
    };

    /** @brief QR code encoder parameters. */
    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();

        //! The optional version of QR code (by default - maximum possible depending on the length of the string).
        CV_PROP_RW int version;

        //! The optional level of error correction (by default - the lowest).
        CV_PROP_RW QRCodeEncoder::CorrectionLevel correction_level;

        //! The optional encoding mode - Numeric, Alphanumeric, Byte, Kanji, ECI or Structured Append.
        CV_PROP_RW QRCodeEncoder::EncodeMode mode;

        //! The optional number of QR codes to generate in Structured Append mode.
        CV_PROP_RW int structure_number;
    };

    /** @brief Constructor
    @param parameters QR code encoder parameters QRCodeEncoder::Params
    */
    static CV_WRAP
    Ptr<QRCodeEncoder> create(const QRCodeEncoder::Params& parameters = QRCodeEncoder::Params());

    /** @brief Generates QR code from input string.
     @param encoded_info Input string to encode.
     @param qrcode Generated QR code.
    */
    CV_WRAP virtual void encode(const String& encoded_info, OutputArray qrcode) = 0;

    /** @brief Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.
     @param encoded_info Input string to encode.
     @param qrcodes Vector of generated QR codes.
    */
    CV_WRAP virtual void encodeStructuredAppend(const String& encoded_info, OutputArrayOfArrays qrcodes) = 0;

};

/** @brief QR code detector. */
class CV_EXPORTS_W_SIMPLE QRCodeDetector : public GraphicalCodeDetector
{
public:
    CV_WRAP QRCodeDetector();

    /** @brief sets the epsilon used during the horizontal scan of QR code stop marker detection.
     @param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
    */
    CV_WRAP QRCodeDetector& setEpsX(double epsX);
    /** @brief sets the epsilon used during the vertical scan of QR code stop marker detection.
     @param epsY Epsilon neighborhood, which allows you to determine the vertical pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
     */
    CV_WRAP QRCodeDetector& setEpsY(double epsY);

    /** @brief use markers to improve the position of the corners of the QR code
     *
     * alignmentMarkers using by default
     */
    CV_WRAP QRCodeDetector& setUseAlignmentMarkers(bool useAlignmentMarkers);

    /** @brief Decodes QR code on a curved surface in image once it's found by the detect() method.

     Returns UTF8-encoded output string or empty string if the code cannot be decoded.
     @param img grayscale or color (BGR) image containing QR code.
     @param points Quadrangle vertices found by detect() method (or some other algorithm).
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP cv::String decodeCurved(InputArray img, InputArray points, OutputArray straight_qrcode = noArray());

    /** @brief Both detects and decodes QR code on a curved surface

     @param img grayscale or color (BGR) image containing QR code.
     @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP std::string detectAndDecodeCurved(InputArray img, OutputArray points=noArray(),
                                              OutputArray straight_qrcode = noArray());

    /** @brief Returns a kind of encoding for the decoded info from the latest @ref decode or @ref detectAndDecode call
    @param codeIdx an index of the previously decoded QR code.
                   When @ref decode or @ref detectAndDecode is used, valid value is zero.
                   For @ref decodeMulti or @ref detectAndDecodeMulti use indices corresponding to the output order.
    */
    CV_WRAP QRCodeEncoder::ECIEncodings getEncoding(int codeIdx = 0);
};

/** @brief QR code detector based on Aruco markers detection code. */
class CV_EXPORTS_W_SIMPLE QRCodeDetectorAruco : public GraphicalCodeDetector {
public:
    CV_WRAP QRCodeDetectorAruco();

    struct CV_EXPORTS_W_SIMPLE Params {
        CV_WRAP Params();

        /** @brief The minimum allowed pixel size of a QR module in the smallest image in the image pyramid, default 4.f */
        CV_PROP_RW float minModuleSizeInPyramid;

        /** @brief The maximum allowed relative rotation for finder patterns in the same QR code, default pi/12 */
        CV_PROP_RW float maxRotation;

        /** @brief The maximum allowed relative mismatch in module sizes for finder patterns in the same QR code, default 1.75f */
        CV_PROP_RW float maxModuleSizeMismatch;

        /** @brief The maximum allowed module relative mismatch for timing pattern module, default 2.f
         *
         * If relative mismatch of timing pattern module more this value, penalty points will be added.
         * If a lot of penalty points are added, QR code will be rejected. */
        CV_PROP_RW float maxTimingPatternMismatch;

        /** @brief The maximum allowed percentage of penalty points out of total pins in timing pattern, default 0.4f */
        CV_PROP_RW float maxPenalties;

        /** @brief The maximum allowed relative color mismatch in the timing pattern, default 0.2f*/
        CV_PROP_RW float maxColorsMismatch;

        /** @brief The algorithm find QR codes with almost minimum timing pattern score and minimum size, default 0.9f
         *
         * The QR code with the minimum "timing pattern score" and minimum "size" is selected as the best QR code.
         * If for the current QR code "timing pattern score" * scaleTimingPatternScore < "previous timing pattern score" and "size" < "previous size", then
         * current QR code set as the best QR code. */
        CV_PROP_RW float scaleTimingPatternScore;
    };

    /** @brief QR code detector constructor for Aruco-based algorithm. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP explicit QRCodeDetectorAruco(const QRCodeDetectorAruco::Params& params);

    /** @brief Detector parameters getter. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP const QRCodeDetectorAruco::Params& getDetectorParameters() const;

    /** @brief Detector parameters setter. See cv::QRCodeDetectorAruco::Params */
    CV_WRAP QRCodeDetectorAruco& setDetectorParameters(const QRCodeDetectorAruco::Params& params);

    /** @brief Aruco detector parameters are used to search for the finder patterns. */
    CV_WRAP const aruco::DetectorParameters& getArucoParameters() const;

    /** @brief Aruco detector parameters are used to search for the finder patterns. */
    CV_WRAP void setArucoParameters(const aruco::DetectorParameters& params);
};

enum { CALIB_CB_ADAPTIVE_THRESH = 1,
       CALIB_CB_NORMALIZE_IMAGE = 2,
       CALIB_CB_FILTER_QUADS    = 4,
       CALIB_CB_FAST_CHECK      = 8,
       CALIB_CB_EXHAUSTIVE      = 16,
       CALIB_CB_ACCURACY        = 32,
       CALIB_CB_LARGER          = 64,
       CALIB_CB_MARKER          = 128,
       CALIB_CB_PLAIN           = 256
     };

enum { CALIB_CB_SYMMETRIC_GRID  = 1,
       CALIB_CB_ASYMMETRIC_GRID = 2,
       CALIB_CB_CLUSTERING      = 4
     };

/** @brief Finds the positions of internal corners of the chessboard.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cv::Size(points_per_row,points_per_column) = cv::Size(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   @ref CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black
and white, rather than a fixed threshold level (computed from the average image brightness).
-   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before
applying fixed or adaptive thresholding.
-   @ref CALIB_CB_FILTER_QUADS Use additional criteria (like contour area, perimeter,
square-like shape) to filter out false quads extracted at the contour retrieval stage.
-   @ref CALIB_CB_FAST_CHECK Run a fast check on the image that looks for chessboard corners,
and shortcut the call if none is found. This can drastically speed up the call in the
degenerate condition when no chessboard is observed.
-   @ref CALIB_CB_PLAIN All other flags are ignored. The input image is taken as is.
No image processing is done to improve to find the checkerboard. This has the effect of speeding up the
execution of the function but could lead to not recognizing the checkerboard if the image
is not previously binarized in the appropriate manner.

@return True if all of the corners are found and placed in a certain order (row by row,
left to right in every row). Otherwise, if the function fails to find all the corners or reorder them,
it returns false.

The function attempts to determine whether the input image is a view of the chessboard pattern and
locate the internal chessboard corners. For example, a regular chessboard has 8 x 8 squares and
7 x 7 internal corners, that is, points where the black squares touch each other. The detected
coordinates are approximate, and to determine their positions more accurately, the function
calls #cornerSubPix. You also may use the function #cornerSubPix with different parameters if
returned coordinates are not accurate enough.

Sample usage of detecting and drawing chessboard corners: :
@code
    Size patternsize(8,6); //interior number of corners
    Mat gray = ....; //source image
    vector<Point2f> corners; //this will be filled by the detected corners

    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    if(patternfound)
      cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments. Otherwise, if there is no
border and the background is dark, the outer black squares cannot be segmented properly and so the
square grouping and ordering algorithm fails.

Use the `generate_pattern.py` Python script (@ref tutorial_camera_calibration_pattern)
to create the desired checkerboard pattern.
 */
CV_EXPORTS_W bool findChessboardCorners( InputArray image, Size patternSize, OutputArray corners,
                                         int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );

/** @brief Checks whether the image contains chessboard of the specific size or not.

@param img Source chessboard view.
@param size Size of the chessboard.

@return Whether a chessboard was found.
*/
CV_EXPORTS_W bool checkChessboard(InputArray img, Size size);

/** @brief Finds the positions of internal corners of the chessboard using a sector based approach.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cv::Size(points_per_row,points_per_column) = cv::Size(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before detection.
-   @ref CALIB_CB_EXHAUSTIVE Run an exhaustive search to improve detection rate.
-   @ref CALIB_CB_ACCURACY Up sample input image to improve sub-pixel accuracy due to aliasing effects.
-   @ref CALIB_CB_LARGER The detected pattern is allowed to be larger than patternSize (see description).
-   @ref CALIB_CB_MARKER The detected pattern must have a marker (see description).
This should be used if an accurate camera calibration is required.
@param meta Optional output array of detected corners (CV_8UC1 and size = cv::Size(columns,rows)).
Each entry stands for one corner of the pattern and can have one of the following values:
-   0 = no meta data attached
-   1 = left-top corner of a black cell
-   2 = left-top corner of a white cell
-   3 = left-top corner of a black cell with a white marker dot
-   4 = left-top corner of a white cell with a black marker dot (pattern origin in case of markers otherwise first corner)

The function is analog to #findChessboardCorners but uses a localized radon
transformation approximated by box filters being more robust to all sort of
noise, faster on larger images and is able to directly return the sub-pixel
position of the internal chessboard corners. The Method is based on the paper
@cite duda2018 "Accurate Detection and Localization of Checkerboard Corners for
Calibration" demonstrating that the returned sub-pixel positions are more
accurate than the one returned by cornerSubPix allowing a precise camera
calibration for demanding applications.

In the case, the flags @ref CALIB_CB_LARGER or @ref CALIB_CB_MARKER are given,
the result can be recovered from the optional meta array. Both flags are
helpful to use calibration patterns exceeding the field of view of the camera.
These oversized patterns allow more accurate calibrations as corners can be
utilized, which are as close as possible to the image borders.  For a
consistent coordinate system across all images, the optional marker (see image
below) can be used to move the origin of the board to the location where the
black circle is located.

@note The function requires a white boarder with roughly the same width as one
of the checkerboard fields around the whole board to improve the detection in
various environments. In addition, because of the localized radon
transformation it is beneficial to use round corners for the field corners
which are located on the outside of the board. The following figure illustrates
a sample checkerboard optimized for the detection. However, any other checkerboard
can be used as well.

Use the `generate_pattern.py` Python script (@ref tutorial_camera_calibration_pattern)
to create the corresponding checkerboard pattern:
\image html pics/checkerboard_radon.png width=60%
 */
CV_EXPORTS_AS(findChessboardCornersSBWithMeta)
bool findChessboardCornersSB(InputArray image,Size patternSize, OutputArray corners,
                             int flags,OutputArray meta);
/** @overload */
CV_EXPORTS_W inline
bool findChessboardCornersSB(InputArray image, Size patternSize, OutputArray corners,
                             int flags = 0)
{
    return findChessboardCornersSB(image, patternSize, corners, flags, noArray());
}

/** @brief Estimates the sharpness of a detected chessboard.

Image sharpness, as well as brightness, are a critical parameter for accuracte
camera calibration. For accessing these parameters for filtering out
problematic calibraiton images, this method calculates edge profiles by traveling from
black to white chessboard cell centers. Based on this, the number of pixels is
calculated required to transit from black to white. This width of the
transition area is a good indication of how sharp the chessboard is imaged
and should be below ~3.0 pixels.

@param image Gray image used to find chessboard corners
@param patternSize Size of a found chessboard pattern
@param corners Corners found by #findChessboardCornersSB
@param rise_distance Rise distance 0.8 means 10% ... 90% of the final signal strength
@param vertical By default edge responses for horizontal lines are calculated
@param sharpness Optional output array with a sharpness value for calculated edge responses (see description)

The optional sharpness array is of type CV_32FC1 and has for each calculated
profile one row with the following five entries:
* 0 = x coordinate of the underlying edge in the image
* 1 = y coordinate of the underlying edge in the image
* 2 = width of the transition area (sharpness)
* 3 = signal strength in the black cell (min brightness)
* 4 = signal strength in the white cell (max brightness)

@return Scalar(average sharpness, average min brightness, average max brightness,0)
*/
CV_EXPORTS_W Scalar estimateChessboardSharpness(InputArray image, Size patternSize, InputArray corners,
                                                float rise_distance=0.8F,bool vertical=false,
                                                OutputArray sharpness=noArray());


//! finds subpixel-accurate positions of the chessboard corners
CV_EXPORTS_W bool find4QuadCornerSubpix( InputArray img, InputOutputArray corners, Size region_size );

/** @brief Renders the detected chessboard corners.

@param image Destination image. It must be an 8-bit color image.
@param patternSize Number of inner corners per a chessboard row and column
(patternSize = cv::Size(points_per_row,points_per_column)).
@param corners Array of detected corners, the output of #findChessboardCorners.
@param patternWasFound Parameter indicating whether the complete board was found or not. The
return value of #findChessboardCorners should be passed here.

The function draws individual chessboard corners detected either as red circles if the board was not
found, or as colored corners connected with lines if the board was found.
 */
CV_EXPORTS_W void drawChessboardCorners( InputOutputArray image, Size patternSize,
                                         InputArray corners, bool patternWasFound );

struct CV_EXPORTS_W_SIMPLE CirclesGridFinderParameters
{
    CV_WRAP CirclesGridFinderParameters();
    CV_PROP_RW cv::Size2f densityNeighborhoodSize;
    CV_PROP_RW float minDensity;
    CV_PROP_RW int kmeansAttempts;
    CV_PROP_RW int minDistanceToAddKeypoint;
    CV_PROP_RW int keypointScale;
    CV_PROP_RW float minGraphConfidence;
    CV_PROP_RW float vertexGain;
    CV_PROP_RW float vertexPenalty;
    CV_PROP_RW float existingVertexGain;
    CV_PROP_RW float edgeGain;
    CV_PROP_RW float edgePenalty;
    CV_PROP_RW float convexHullFactor;
    CV_PROP_RW float minRNGEdgeSwitchDist;

    enum GridType
    {
      SYMMETRIC_GRID, ASYMMETRIC_GRID
    };
    CV_PROP_RW GridType gridType;

    CV_PROP_RW float squareSize; //!< Distance between two adjacent points. Used by CALIB_CB_CLUSTERING.
    CV_PROP_RW float maxRectifiedDistance; //!< Max deviation from prediction. Used by CALIB_CB_CLUSTERING.
};

/** @brief Finds centers in the grid of circles.

@param image grid view of input circles; it must be an 8-bit grayscale or color image.
@param patternSize number of circles per row and column
( patternSize = Size(points_per_row, points_per_column) ).
@param centers output array of detected centers.
@param flags various operation flags that can be one of the following values:
-   @ref CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
-   @ref CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
-   @ref CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to
perspective distortions but much more sensitive to background clutter.
@param blobDetector feature detector that finds blobs like dark circles on light background.
                    If `blobDetector` is NULL then `image` represents Point2f array of candidates.
@param parameters struct for finding circles in a grid pattern.

return True if all of the centers have been found and they have been placed in a certain order
(row by row, left to right in every row). Otherwise, if the function fails to find all the corners
or reorder them, it returns false.

The function attempts to determine whether the input image contains a grid of circles. If it is, the
function locates centers of the circles.

Sample usage of detecting and drawing the centers of circles: :
@code
    Size patternsize(7,7); //number of centers
    Mat gray = ...; //source image
    vector<Point2f> centers; //this will be filled by the detected centers

    bool patternfound = findCirclesGrid(gray, patternsize, centers);

    drawChessboardCorners(img, patternsize, Mat(centers), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments.
 */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags,
                                   const Ptr<FeatureDetector> &blobDetector,
                                   const CirclesGridFinderParameters& parameters);

/** @overload */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags = CALIB_CB_SYMMETRIC_GRID,
                                   const Ptr<FeatureDetector> &blobDetector = cv::SimpleBlobDetector::create());

//! @}
}

#include "opencv2/objdetect/face.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"
#include "opencv2/objdetect/barcode.hpp"

#endif
