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

#ifndef _MCC_CHECKER_DETECTOR_HPP
#define _MCC_CHECKER_DETECTOR_HPP

#include "opencv2/objdetect.hpp"
#include "charts.hpp"

namespace cv
{
namespace mcc
{

class CCheckerDetectorImpl : public CCheckerDetector
{

    typedef std::vector<Point> PointsVector;
    typedef std::vector<PointsVector> ContoursVector;

public:
    CCheckerDetectorImpl();
    CCheckerDetectorImpl(const dnn::Net& _net){
        net = _net;
    }
    virtual ~CCheckerDetectorImpl();

    bool process(InputArray image, const std::vector<Rect> &regionsOfInterest,
                 const int nc = 1) CV_OVERRIDE;

    bool process(InputArray image, const int nc = 1) CV_OVERRIDE;

    Ptr<CChecker> getBestColorChecker() CV_OVERRIDE
    {
        if (m_checkers.size())
            return m_checkers[0];
        return nullptr;
    }

    std::vector<Ptr<CChecker>> getListColorChecker() CV_OVERRIDE
    {
        return m_checkers;
    }
    virtual Mat getRefColors() CV_OVERRIDE;

    virtual void setDetectionParams(const DetectorParametersMCC &params) CV_OVERRIDE;

    virtual void setColorChartType(ColorChart chartType) CV_OVERRIDE;

    virtual void setUseNet(bool useNet) CV_OVERRIDE;

    virtual bool getUseNet() const CV_OVERRIDE;

    virtual const DetectorParametersMCC& getDetectionParams() const CV_OVERRIDE;

    virtual ColorChart getColorChartType() const CV_OVERRIDE;

    virtual void draw(std::vector<Ptr<CChecker>>& checkers, InputOutputArray img, const Scalar color = CV_RGB(0,250,0), const int thickness = 2) CV_OVERRIDE;

protected: // methods pipeline
    bool _no_net_process(InputArray image, const int nc, std::vector<Rect> regionsOfInterest);
    /// prepareImage
    /** @brief Prepare Image
      * @param bgrMat image in color space BGR
      * @param grayOut gray scale
      * @param bgrOut rescale image
      * @param aspOut aspect ratio
      * @param max_size rescale factor in max dim
      */
    virtual void
    prepareImage(InputArray bgr, OutputArray grayOut, OutputArray bgrOut, float &aspOut) const;

    /// performThreshold
    /** @brief Adaptative threshold
      * @param grayscaleImg gray scale image
      * @param thresholdImg binary image
      * @param wndx, wndy windows size
      * @param step
      */
    virtual void
    performThreshold(InputArray grayscaleImg, OutputArrayOfArrays thresholdImg) const;

    /// findContours
    /** @brief find contour in the image
    * @param srcImg binary imagen
    * @param contours
    * @param minContourPointsAllowed
    */
    virtual void
    findContours(InputArray srcImg, ContoursVector &contours) const;

    /// findCandidates
    /** @brief find posibel candidates
    * @param contours
    * @param detectedCharts
    * @param minContourLengthAllowed
    */
    virtual void
    findCandidates(const ContoursVector &contours, std::vector<CChart> &detectedCharts);

    /// clustersAnalysis
    /** @brief clusters charts analysis
    * @param detectedCharts
    * @param groups
    */
    virtual void
    clustersAnalysis(const std::vector<CChart> &detectedCharts, std::vector<int> &groups);

    /// checkerRecognize
    /** @brief checker color recognition
    * @param img
    * @param detectedCharts
    * @param G
    * @param colorChartsOut
    */
    virtual void
    checkerRecognize(InputArray img, const std::vector<CChart> &detectedCharts, const std::vector<int> &G,
                     std::vector<std::vector<Point2f>> &colorChartsOut);

    /// checkerAnalysis
    /** @brief evaluate checker
    * @param img
    * @param img_org
    * @param colorCharts
    * @param checker
    * @param asp
    */
    virtual void
    checkerAnalysis(InputArray img_rgb_f, const unsigned int nc,
                    const std::vector<std::vector<Point2f>> &colorCharts,
                    std::vector<Ptr<CChecker>> &checkers, float asp,
                    const Mat &img_rgb_org,
                    const Mat &img_ycbcr_org,
                    std::vector<Mat> &rgb_planes,
                    std::vector<Mat> &ycbcr_planes);

    virtual void
    removeTooCloseDetections();

protected:
    std::vector<Ptr<CChecker>> m_checkers;
    dnn::Net net;
    DetectorParametersMCC m_params = DetectorParametersMCC();
    ColorChart m_chartType;
    bool m_useNet = false;

private: // methods aux
    void get_subbox_chart_physical(
        const std::vector<Point2f> &points,
        std::vector<Point2f> &chartPhy);

    void reduce_array(
        const std::vector<float> &x,
        std::vector<float> &x_new,
        float tol);

    void transform_points_inverse(
        InputArray T,
        const std::vector<Point2f> &X,
        std::vector<Point2f> &Xt);

    void get_profile(
        const std::vector<Point2f> &ibox,
        OutputArray charts_rgb,
        OutputArray charts_ycbcr,
        InputArray im_rgb,
        InputArray im_ycbcr,
        std::vector<Mat> &rgb_planes,
        std::vector<Mat> &ycbcr_planes);

    /** @brief cost function
     *  e(p) = ||f(p)||^2 = \sum_k (mu_{k,p}*r_k')/||mu_{k,p}||||r_k|| + ...
     *                   + \sum_k || \sigma_{k,p} ||^2
     */
    float cost_function(InputArray img, InputOutputArray mask, InputArray lab,
                        const std::vector<Point2f> &ibox);
};

} // namespace mcc
} // namespace cv

#endif //_MCC_CHECKER_DETECTOR_HPP
