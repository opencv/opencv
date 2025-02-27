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

#include "opencv2/objdetect/mcc_checker_detector.hpp"
#include "charts.hpp"

namespace cv
{
namespace mcc
{

class CCheckerDetectorImpl : public CCheckerDetector
{

    typedef std::vector<cv::Point> PointsVector;
    typedef std::vector<PointsVector> ContoursVector;

public:
    CCheckerDetectorImpl();
    CCheckerDetectorImpl(const dnn::Net& _net){
        net = _net;
    }
    virtual ~CCheckerDetectorImpl();

    bool process(InputArray image, const COLORCHART chartType,
                 const std::vector<cv::Rect> &regionsOfInterest,
                 const int nc = 1, bool use_net = false,
                 const DetectorParametersMCC &params = DetectorParametersMCC()) CV_OVERRIDE;

    bool process(InputArray image, const COLORCHART chartType,
                 const int nc = 1, bool use_net = false,
                 const DetectorParametersMCC &params = DetectorParametersMCC()) CV_OVERRIDE;

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
    virtual void getRefColor(const COLORCHART chartType, cv::Mat& output) CV_OVERRIDE;

    virtual void draw(std::vector<Ptr<CChecker>>& checkers, InputOutputArray img, const cv::Scalar color = CV_RGB(0,250,0), const int thickness = 2) CV_OVERRIDE;

protected: // methods pipeline
    bool _no_net_process(InputArray image, const COLORCHART chartType,
                         const int nc,
                         const DetectorParametersMCC &params,
                         std::vector<cv::Rect> regionsOfInterest);
    /// prepareImage
    /** \brief Prepare Image
      * \param[in] bgrMat image in color space BGR
      * \param[out] grayOut gray scale
      * \param[out] bgrOut rescale image
      * \param[out] aspOut aspect ratio
      * \param[in] max_size rescale factor in max dim
      */
    virtual void
    prepareImage(InputArray bgr, OutputArray grayOut, OutputArray bgrOut, float &aspOut, const DetectorParametersMCC &params) const;

    /// performThreshold
    /** \brief Adaptative threshold
      * \param[in] grayscaleImg gray scale image
      * \param[in] thresholdImg binary image
      * \param[in] wndx, wndy windows size
      * \param[in] step
      */
    virtual void
    performThreshold(InputArray grayscaleImg, OutputArrayOfArrays thresholdImg, const DetectorParametersMCC &params) const;

    /// findContours
    /** \brief find contour in the image
    * \param[in] srcImg binary imagen
    * \param[out] contours
    * \param[in] minContourPointsAllowed
    */
    virtual void
    findContours(InputArray srcImg, ContoursVector &contours, const DetectorParametersMCC &params) const;

    /// findCandidates
    /** \brief find posibel candidates
    * \param[in] contours
    * \param[out] detectedCharts
    * \param[in] minContourLengthAllowed
    */
    virtual void
    findCandidates(const ContoursVector &contours, std::vector<CChart> &detectedCharts, const DetectorParametersMCC &params);

    /// clustersAnalysis
    /** \brief clusters charts analysis
    * \param[in] detectedCharts
    * \param[out] groups
    */
    virtual void
    clustersAnalysis(const std::vector<CChart> &detectedCharts, std::vector<int> &groups, const DetectorParametersMCC &params);

    /// checkerRecognize
    /** \brief checker color recognition
    * \param[in] img
    * \param[in] detectedCharts
    * \param[in] G
    * \param[out] colorChartsOut
    */
    virtual void
    checkerRecognize(InputArray img, const std::vector<CChart> &detectedCharts, const std::vector<int> &G,
                     const COLORCHART chartType, std::vector<std::vector<cv::Point2f>> &colorChartsOut,
                     const DetectorParametersMCC &params);

    /// checkerAnalysis
    /** \brief evaluate checker
    * \param[in] img
    * \param[in] img_org
    * \param[in] colorCharts
    * \param[out] checker
    * \param[in] asp
    */
    virtual void
    checkerAnalysis(InputArray img_rgb_f,
                    const COLORCHART chartType, const unsigned int nc,
                    const std::vector<std::vector<cv::Point2f>> &colorCharts,
                    std::vector<Ptr<CChecker>> &checkers, float asp,
                    const DetectorParametersMCC &params,
                    const cv::Mat &img_rgb_org,
                    const cv::Mat &img_ycbcr_org,
                    std::vector<cv::Mat> &rgb_planes,
                    std::vector<cv::Mat> &ycbcr_planes);

    virtual void
    removeTooCloseDetections(const DetectorParametersMCC &params);

protected:
    std::vector<Ptr<CChecker>> m_checkers;
    dnn::Net net;
    bool net_used = false;

private: // methods aux
    void get_subbox_chart_physical(
        const std::vector<cv::Point2f> &points,
        std::vector<cv::Point2f> &chartPhy);

    void reduce_array(
        const std::vector<float> &x,
        std::vector<float> &x_new,
        float tol);

    void transform_points_inverse(
        InputArray T,
        const std::vector<cv::Point2f> &X,
        std::vector<cv::Point2f> &Xt);

    void get_profile(
        const std::vector<cv::Point2f> &ibox,
        const COLORCHART chartType,
        OutputArray charts_rgb,
        OutputArray charts_ycbcr,
        InputArray im_rgb,
        InputArray im_ycbcr,
        std::vector<cv::Mat> &rgb_planes,
        std::vector<cv::Mat> &ycbcr_planes);

    /* \brief cost function
     *  e(p) = ||f(p)||^2 = \sum_k (mu_{k,p}*r_k')/||mu_{k,p}||||r_k|| + ...
     *                   + \sum_k || \sigma_{k,p} ||^2
     */
    float cost_function(InputArray img, InputOutputArray mask, InputArray lab,
                        const std::vector<cv::Point2f> &ibox,
                        const COLORCHART chartType);
};

} // namespace mcc
} // namespace cv

#endif //_MCC_CHECKER_DETECTOR_HPP
