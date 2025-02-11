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
#include "checker_model.hpp"
#include <opencv2/dnn.hpp>

//----------To view debugging output-----------------------------
//Read the tutorial on how to use debugging in this module
//It can be found in the documentation of 'mcc' modules,
//Then uncomment the following line to view debugging output
//---------------------------------------------------------------
// #define MCC_DEBUG
//---------------------------------------------------------------

namespace cv
{
namespace mcc
{
//! @addtogroup mcc
//! @{

/**
 * @brief Parameters for the detectMarker process:
 * - int adaptiveThreshWinSizeMin : minimum window size for adaptive
 *                                  thresholding before finding contours
 *                                  (default 23).
 * - int adaptiveThreshWinSizeMax : maximum window size for adaptive
 *                                  thresholding before finding contours
 *                                  (default 153).
 * - int adaptiveThreshWinSizeStep : increments from adaptiveThreshWinSizeMin to
 *                                   adaptiveThreshWinSizeMax during the
 *                                   thresholding (default 16).
 * - double adaptiveThreshConstant : constant for adaptive thresholding before
 *                                   finding contours (default 7)
 * - double minContoursAreaRate : determine minimum area for marker contour to
 *                                be detected. This is defined as a rate respect
 *                                to the area of the input image. Used only if
 *                                neural network is used (default 0.003).
 * - double minContoursArea : determine minimum area for marker contour to be
 *                            detected. This is defined as the actual area. Used
 *                            only if neural network is not used (default 100).
 * - double confidenceThreshold : minimum confidence for a bounding box detected
 *                                by neural network to classify as
 *                                detection.(default 0.5)
 *                                (0<=confidenceThreshold<=1)
 * - double minContourSolidity : minimum solidity of a contour for it be
 *                               detected as a square in the chart. (default
 *                               0.9).
 * - double findCandidatesApproxPolyDPEpsMultiplier : multipler to be used in
 *                                                    cv::ApproxPolyDP function
 *                                                    (default 0.05)
 * - int borderWidth : width of the padding used to pass the inital neural
 *                     network detection in the succeeding system.(default 0)
 * - float B0factor : distance between two neighbours squares of the same chart.
 *                    Defined as the ratio between distance and large dimension
 *                    of square (default 1.25)
 * - float maxError : maximum allowed error in the detection of a chart.
 *                    default(0.1)
 * - int minContourPointsAllowed : minium points in a detected contour.
 *                                 default(4)
 * - int minContourLengthAllowed : minimum length of a countour. default(100)
 * - int minInterContourDistance : minimum distance between two contours.
 *                                 default(100)
 * - int minInterCheckerDistance : minimum distance between two checkers.
 *                                 default(10000)
 * - int minImageSize : minimum size of the smaller dimension of the image.
 *                      default(1000)
 * - unsigned minGroupSize : minimum number of a squared of a chart that must be
 *                           detected. default(4)
 */
struct CV_EXPORTS_W DetectorParametersMCC
{

    DetectorParametersMCC();

    CV_WRAP static Ptr<DetectorParametersMCC> create();

    CV_PROP_RW int adaptiveThreshWinSizeMin;
    CV_PROP_RW int adaptiveThreshWinSizeMax;
    CV_PROP_RW int adaptiveThreshWinSizeStep;
    CV_PROP_RW double adaptiveThreshConstant;
    CV_PROP_RW double minContoursAreaRate;
    CV_PROP_RW double minContoursArea;
    CV_PROP_RW double confidenceThreshold;
    CV_PROP_RW double minContourSolidity;
    CV_PROP_RW double findCandidatesApproxPolyDPEpsMultiplier;
    CV_PROP_RW int borderWidth;
    CV_PROP_RW float B0factor;
    CV_PROP_RW float maxError;
    CV_PROP_RW int minContourPointsAllowed;
    CV_PROP_RW int minContourLengthAllowed;
    CV_PROP_RW int minInterContourDistance;
    CV_PROP_RW int minInterCheckerDistance;
    CV_PROP_RW int minImageSize;
    CV_PROP_RW unsigned minGroupSize;
};

/** @brief A class to find the positions of the ColorCharts in the image.
 */

class CV_EXPORTS_W CCheckerDetector : public Algorithm
{
public:
    /** \brief Set the net which will be used to find the approximate
    *         bounding boxes for the color charts.
    *
    * It is not necessary to use this, but this usually results in
    * better detection rate.
    *
    * \param net the neural network, if the network in empty, then
    *            the function will return false.
    * \return true if it was able to set the detector's network,
    *         false otherwise.
    */

    CV_WRAP virtual bool setNet(dnn::Net net) = 0;

    /** \brief Find the ColorCharts in the given image.
    *
    * The found charts are not returned but instead stored in the
    * detector, these can be accessed later on using getBestColorChecker()
    * and getListColorChecker()
    * \param image image in color space BGR
    * \param chartType type of the chart to detect
    * \param regionsOfInterest regions of image to look for the chart, if
    *                          it is empty, charts are looked for in the
    *                          entire image
    * \param nc number of charts in the image, if you don't know the exact
    *           then keeping this number high helps.
    * \param useNet if it is true the network provided using the setNet()
    *               is used for preliminary search for regions where chart
    *               could be present, inside the regionsOfInterest provied.
    * \param params parameters of the detection system. More information
    *               about them can be found in the struct DetectorParametersMCC.
    * \return true if atleast one chart is detected otherwise false
    */

    CV_WRAP_AS(processWithROI) virtual bool
    process(InputArray image, const TYPECHART chartType,
            const std::vector<Rect> &regionsOfInterest,
            const int nc = 1, bool useNet = false,
            const Ptr<DetectorParametersMCC> &params = DetectorParametersMCC::create()) = 0;


    /** \brief Find the ColorCharts in the given image.
    *
    * Differs from the above one only in the arguments.
    *
    * This version searches for the chart in the full image.
    *
    * The found charts are not returned but instead stored in the
    * detector, these can be accessed later on using getBestColorChecker()
    * and getListColorChecker()
    * \param image image in color space BGR
    * \param chartType type of the chart to detect
    * \param nc number of charts in the image, if you don't know the exact
    *           then keeping this number high helps.
    * \param useNet if it is true the network provided using the setNet()
    *               is used for preliminary search for regions where chart
    *               could be present, inside the regionsOfInterest provied.
    * \param params parameters of the detection system. More information
    *               about them can be found in the struct DetectorParametersMCC.
    * \return true if atleast one chart is detected otherwise false
    */

    CV_WRAP virtual bool
    process(InputArray image, const TYPECHART chartType,
            const int nc = 1, bool useNet = false,
            const Ptr<DetectorParametersMCC> &params = DetectorParametersMCC::create()) = 0;

    /** \brief Get the best color checker. By the best it means the one
    *         detected with the highest confidence.
    * \return checker A single colorchecker, if atleast one colorchecker
    *                 was detected, 'nullptr' otherwise.
    */
    CV_WRAP virtual Ptr<mcc::CChecker> getBestColorChecker() = 0;

    /** \brief Get the list of all detected colorcheckers
    * \return checkers vector of colorcheckers
    */
    CV_WRAP virtual std::vector<Ptr<CChecker>> getListColorChecker() = 0;

    /** \brief Returns the implementation of the CCheckerDetector.
    *
    */
    CV_WRAP static Ptr<CCheckerDetector> create();
};

//! @} mcc

} // namespace mcc
} // namespace cv

#endif
