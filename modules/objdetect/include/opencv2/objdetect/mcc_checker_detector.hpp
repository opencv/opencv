// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef OPENCV_OBJDETECT_MCC_CHECKER_DETECTOR_HPP
#define OPENCV_OBJDETECT_MCC_CHECKER_DETECTOR_HPP
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

//---------------------------------------------------------------
// #define MCC_DEBUG
//---------------------------------------------------------------

namespace cv
{
namespace mcc
{
//! @addtogroup mcc
//! @{

/** ColorChart
 *
 * @brief enum to hold the type of the checker
 */
enum ColorChart
{
    MCC24 = 0, ///< Standard Macbeth Chart with 24 squares
    SG140,     ///< DigitalSG with 140 squares
    VINYL18,   ///< DKK color chart with 12 squares and 6 rectangle
};

/** CChecker
 *
 * @brief checker object
 *
 *     This class contains the information about the detected checkers,i.e, their
 *     type, the corners of the chart, the color profile, the cost, centers chart,
 *     etc.
 */
class CV_EXPORTS_W CChecker: public Algorithm
{
public:
    CChecker() {}
    virtual ~CChecker() {}
    /** @brief Create a new CChecker object.
     *
     * @return A pointer to the implementation of the CChecker
     */
    CV_WRAP static Ptr<CChecker> create();
public:
    CV_WRAP virtual void setTarget(ColorChart _target)  = 0;
    CV_WRAP virtual void setBox(std::vector<Point2f> _box) = 0;
    CV_WRAP virtual void setChartsRGB(Mat _chartsRGB) = 0;
    CV_WRAP virtual void setChartsYCbCr(Mat _chartsYCbCr) = 0;
    CV_WRAP virtual void setCost(float _cost) = 0;
    CV_WRAP virtual void setCenter(Point2f _center) = 0;

    CV_WRAP virtual ColorChart getTarget() = 0;
    CV_WRAP virtual std::vector<Point2f> getBox() = 0;

    /** @brief Computes and returns the coordinates of the central parts of the charts modules.
     *
     * This method computes transformation matrix from the checkers's coordinates (`CChecker::getBox()`)
     * and find by this the coordinates of the central parts of the charts modules.
     * It is used in `CCheckerDetector::draw()` and in `ChartsRGB` calculation.
     */
    CV_WRAP virtual std::vector<Point2f> getColorCharts() = 0;

    CV_WRAP virtual Mat getChartsRGB(bool getStats = true) = 0;
    CV_WRAP virtual Mat getChartsYCbCr() = 0;
    CV_WRAP virtual float getCost() = 0;
    CV_WRAP virtual Point2f getCenter() = 0;
};

/** @brief struct DetectorParametersMCC is used by CCheckerDetector
 */
struct CV_EXPORTS_W_SIMPLE DetectorParametersMCC
{
    CV_WRAP DetectorParametersMCC(){
        adaptiveThreshWinSizeMin=23;
        adaptiveThreshWinSizeMax=153;
        adaptiveThreshWinSizeStep=16;
        adaptiveThreshConstant=7;
        minContoursAreaRate=0.003;
        minContoursArea=100;
        confidenceThreshold=0.5;
        minContourSolidity=0.9;
        findCandidatesApproxPolyDPEpsMultiplier=0.05;
        borderWidth=0;
        B0factor=1.25f;
        maxError=0.1f;
        minContourPointsAllowed=4;
        minContourLengthAllowed=100;
        minInterContourDistance=100;
        minInterCheckerDistance=10000;
        minImageSize=1000;
        minGroupSize=4;
    }
    /// minimum window size for adaptive thresholding before finding contours (default 23).
    CV_PROP_RW int adaptiveThreshWinSizeMin;

    /// maximum window size for adaptive thresholding before finding contours (default 153).
    CV_PROP_RW int adaptiveThreshWinSizeMax;

    /// increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding (default 16).
    CV_PROP_RW int adaptiveThreshWinSizeStep;

    /// constant for adaptive thresholding before finding contours (default 7)
    CV_PROP_RW double adaptiveThreshConstant;

    /** @brief determine minimum area for marker contour to be detected
     *
     * This is defined as a rate respect to the area of the input image. Used only if neural network is used (default 0.03).
     */
    CV_PROP_RW double minContoursAreaRate;

    /** @brief determine minimum area for marker contour to be detected
     *
     * This is defined as the actual area. Used only if neural network is used (default 100).
     */
    CV_PROP_RW double minContoursArea;

    /// minimum confidence for a bounding box detected by neural network to classify as detection.(default 0.5) (0<=confidenceThreshold<=1)
    CV_PROP_RW double confidenceThreshold;

    /// minimum solidity of a contour for it be detected as a square in the chart. (default 0.9).
    CV_PROP_RW double minContourSolidity;

    /// multipler to be used in ApproxPolyDP function (default 0.05)
    CV_PROP_RW double findCandidatesApproxPolyDPEpsMultiplier;

    /// width of the padding used to pass the inital neural network detection in the succeeding system.(default 0)
    CV_PROP_RW int borderWidth;

    /// distance between two neighboring squares of the same chart as a ratio of the large dimension of a square (default 1.25).
    CV_PROP_RW float B0factor;

    /// maximum allowed error in the detection of a chart (default 0.1).
    CV_PROP_RW float maxError;

    /// minimum points in a detected contour (default 4).
    CV_PROP_RW int minContourPointsAllowed;

    /// minimum length of a contour (default 100).
    CV_PROP_RW int minContourLengthAllowed;

    /// minimum distance between two contours (default 100).
    CV_PROP_RW int minInterContourDistance;

    /// minimum distance between two checkers (default 10000).
    CV_PROP_RW int minInterCheckerDistance;

    /// minimum size of the smaller dimension of the image (default 1000).
    CV_PROP_RW int minImageSize;

    /// minimum number of squares in a chart that must be detected (default 4).
    CV_PROP_RW int minGroupSize;
};

/** @brief A class to find the positions of the ColorCharts in the image.
 */

class CV_EXPORTS_W CCheckerDetector : public Algorithm
{
public:
    /** @brief Find the ColorCharts in the given image.
    *
    * The found charts are not returned but instead stored in the
    * detector, these can be accessed later on using getBestColorChecker()
    * and getListColorChecker()
    * @param image image in color space BGR
    * @param regionsOfInterest regions of image to look for the chart, if
    *                          it is empty, charts are looked for in the
    *                          entire image
    * @param nc number of charts in the image, if you don't know the exact
    *           then keeping this number high helps.
    * @return true if atleast one chart is detected otherwise false
    */

    CV_WRAP_AS(processWithROI) virtual bool
    process(InputArray image, const std::vector<Rect> &regionsOfInterest,
            const int nc = 1) = 0;


    /** @brief Find the ColorCharts in the given image.
    *
    * Differs from the above one only in the arguments.
    *
    * This version searches for the chart in the full image.
    *
    * The found charts are not returned but instead stored in the
    * detector, these can be accessed later on using getBestColorChecker()
    * and getListColorChecker()
    * @param image image in color space BGR
    * @param nc number of charts in the image, if you don't know the exact
    *           then keeping this number high helps.
    * @return true if atleast one chart is detected otherwise false
    */

    CV_WRAP virtual bool
    process(InputArray image, const int nc = 1) = 0;

    /** @brief Get the best color checker. By the best it means the one
    *         detected with the highest confidence.
    * @return checker A single colorchecker, if atleast one colorchecker
    *                 was detected, 'nullptr' otherwise.
    */
    CV_WRAP virtual Ptr<mcc::CChecker> getBestColorChecker() = 0;

    /** @brief Get the list of all detected colorcheckers
    * @return checkers vector of colorcheckers
    */
    CV_WRAP virtual std::vector<Ptr<CChecker>> getListColorChecker() = 0;

    /** @brief Returns the implementation of the CCheckerDetector.
    *
    */
    CV_WRAP static Ptr<CCheckerDetector> create();
     /** @brief Set the net which will be used to find the approximate
    *         bounding boxes for the color charts. And returns the implementation of the CCheckerDetector.
    *
    * It is not necessary to use this, but this usually results in
    * better detection rate.
    *
    * @param net the neural network, if the network in empty, then
    *            the function will return false.
    */
    CV_WRAP static Ptr<CCheckerDetector> create(const dnn::Net &net);

    /** @brief Draws the checker to the given image.
        * @param img image in color space BGR
        * @param checkers The checkers which will be drawn by this object.
        * @param color The color by with which the squares of the checker
        *         will be drawn
        * @param thickness The thickness with which the sqaures will be
        *         drawn
    */

    CV_WRAP virtual void draw(std::vector<Ptr<CChecker>>& checkers, InputOutputArray img, const Scalar color = CV_RGB(0,250,0), const int thickness = 2) = 0;

    /** @brief Gets the reference color for chart.
    */

    CV_WRAP virtual Mat getRefColors() = 0;

    /** @brief Sets the detection paramaters for mcc.
        * @param params DetectorParametersMCC structure containing detection configuration parameters.
    */

    CV_WRAP virtual void setDetectionParams(const DetectorParametersMCC &params) = 0;

    /** @brief Sets the color chart type for MCC detection.
        * @param chartType ColorChart enum specifying the type of color chart to detect.
    */

    CV_WRAP virtual void setColorChartType(ColorChart chartType) = 0;

    /** @brief Enables or disables the use of the neural network for detection.
        * @param useDnn Boolean flag to indicate whether to use neural network (true) or not (false).
    */

    CV_WRAP virtual void setUseDnnModel(bool useDnn) = 0;

    CV_WRAP virtual bool getUseDnnModel() const = 0;

    CV_WRAP virtual const DetectorParametersMCC& getDetectionParams() const = 0;

    CV_WRAP virtual ColorChart getColorChartType() const = 0;
};

//! @} mcc
} // namespace mcc
} // namespace cv

#endif
