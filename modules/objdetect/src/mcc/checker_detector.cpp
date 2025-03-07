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

#include "precomp.hpp"

#include "checker_detector.hpp"
#include "graph_cluster.hpp"
#include "bound_min.hpp"
#include "wiener_filter.hpp"
#include "checker_model.hpp"
#include "debug.hpp"

namespace cv
{
namespace mcc
{

Ptr<CCheckerDetector> CCheckerDetector::create()
{
    return makePtr<CCheckerDetectorImpl>();
}
Ptr<CCheckerDetector> CCheckerDetector::create(const dnn::Net& net)
{
    return makePtr<CCheckerDetectorImpl>(net);
}

CCheckerDetectorImpl::
    CCheckerDetectorImpl()
{
}

CCheckerDetectorImpl::~CCheckerDetectorImpl()
{
}

bool CCheckerDetectorImpl::
    _no_net_process(InputArray image, const COLORCHART chartType, const int nc,
                    const DetectorParametersMCC &params,
                    std::vector<Rect> regionsOfInterest)
{
    m_checkers.clear();
    this->net_used = false;

    Mat img = image.getMat();
    for (const Rect &region : regionsOfInterest)
    {
        //-------------------------------------------------------------------
        // Run the model to find good regions
        //-------------------------------------------------------------------
        Mat croppedImage = img(region);
#ifdef MCC_DEBUG
        std::string pathOut = "./";
#endif
        //-------------------------------------------------------------------
        // prepare image
        //-------------------------------------------------------------------

        Mat img_bgr, img_gray;
        float asp;
        prepareImage(croppedImage, img_gray, img_bgr, asp, params);

#ifdef MCC_DEBUG
        showAndSave("prepare_image", img_gray, pathOut);
#endif
        //-------------------------------------------------------------------
        // thresholding
        //-------------------------------------------------------------------
        std::vector<Mat> img_bw;
        performThreshold(img_gray, img_bw, params);

        Mat3f img_rgb_f(img_bgr);
        cvtColor(img_rgb_f, img_rgb_f, COLOR_BGR2RGB);
        img_rgb_f /= 255;

        Mat img_rgb_org, img_ycbcr_org;
        std::vector<Mat> rgb_planes(3), ycbcr_planes(3);

        // Convert to RGB and YCbCr space
        cvtColor(croppedImage, img_rgb_org, COLOR_BGR2RGB);
        cvtColor(croppedImage, img_ycbcr_org, COLOR_BGR2YCrCb);

        // Get chanels
        split(img_rgb_org, rgb_planes);
        split(img_ycbcr_org, ycbcr_planes);
        Mutex mtx;
        parallel_for_(
            Range(0, (int)img_bw.size()), [&](const Range &range) {
                const int begin = range.start;
                const int end = range.end;
                for (int i = begin; i < end; i++)
                {

#ifdef MCC_DEBUG
                    showAndSave("threshold_image", img_bw[i], pathOut);
#endif
                    // find contour
                    //-------------------------------------------------------------------
                    ContoursVector contours;
                    findContours(img_bw[i], contours, params);

                    if (contours.empty())
                        continue;
#ifdef MCC_DEBUG
                    Mat im_contour(img_bgr.size(), CV_8UC1);
                    im_contour = Scalar(0);
                    drawContours(im_contour, contours, -1, Scalar(255), 2, LINE_AA);
                    showAndSave("find_contour", im_contour, pathOut);
#endif
                    //-------------------------------------------------------------------
                    // find candidate
                    //-------------------------------------------------------------------

                    std::vector<CChart> detectedCharts;
                    findCandidates(contours, detectedCharts, params);

                    if (detectedCharts.empty())
                        continue;

#ifdef MCC_DEBUG
                    Mat img_chart;
                    img_bgr.copyTo(img_chart);

                    for (size_t ind = 0; ind < detectedCharts.size(); ind++)
                    {

                        CChartDraw chrtdrw((detectedCharts[ind]), img_chart);
                        chrtdrw.drawCenter();
                        chrtdrw.drawContour();
                    }
                    showAndSave("find_candidate", img_chart, pathOut);
#endif
                    //-------------------------------------------------------------------
                    // clusters analysis
                    //-------------------------------------------------------------------

                    std::vector<int> G;
                    clustersAnalysis(detectedCharts, G, params);

                    if (G.empty())
                        continue;

#ifdef MCC_DEBUG
                    Mat im_gru;
                    img_bgr.copyTo(im_gru);
                    RNG rng(0xFFFFFFFF);
                    int radius = 10, thickness = -1;

                    std::vector<int> g;
                    unique(G, g);
                    size_t Nc = g.size();
                    std::vector<Scalar> colors(Nc);
                    for (size_t ind = 0; ind < Nc; ind++)
                        colors[ind] = randomcolor(rng);

                    for (size_t ind = 0; ind < detectedCharts.size(); ind++)
                        circle(im_gru, detectedCharts[ind].center, radius, colors[G[ind]],
                                   thickness);
                    showAndSave("clusters_analysis", im_gru, pathOut);
#endif
                    //-------------------------------------------------------------------
                    // checker color recognize
                    //-------------------------------------------------------------------

                    std::vector<std::vector<Point2f>> colorCharts;
                    checkerRecognize(img_bgr, detectedCharts, G, chartType, colorCharts, params);

                    if (colorCharts.empty())
                        continue;

#ifdef MCC_DEBUG
                    Mat image_box;
                    img_bgr.copyTo(image_box);
                    for (size_t ind = 0; ind < colorCharts.size(); ind++)
                    {
                        std::vector<Point2f> ibox = colorCharts[ind];
                        Scalar color_box = CV_RGB(0, 0, 255);
                        int thickness_box = 2;
                        line(image_box, ibox[0], ibox[1], color_box, thickness_box, LINE_AA);
                        line(image_box, ibox[1], ibox[2], color_box, thickness_box, LINE_AA);
                        line(image_box, ibox[2], ibox[3], color_box, thickness_box, LINE_AA);
                        line(image_box, ibox[3], ibox[0], color_box, thickness_box, LINE_AA);
                    }
                    showAndSave("checker_recognition", image_box, pathOut);
#endif
                    //-------------------------------------------------------------------
                    // checker color analysis
                    //-------------------------------------------------------------------
                    std::vector<Ptr<CChecker>> checkers;
                    checkerAnalysis(img_rgb_f, chartType, nc, colorCharts, checkers, asp, params,
                                    img_rgb_org, img_ycbcr_org, rgb_planes, ycbcr_planes);

#ifdef MCC_DEBUG
                    Mat image_checker;
                    croppedImage.copyTo(image_checker);
                    draw(checkers, image_checker);
                    showAndSave("checker_analysis", image_checker, pathOut);
#endif
                    for (Ptr<CChecker> checker : checkers)
                    {
                        for (Point2f &corner : checker->getBox())
                            corner += static_cast<Point2f>(region.tl());

                        {
                            AutoLock lock(mtx);
                            m_checkers.push_back(checker);
                        }
                    }
                }
#ifdef MCC_DEBUG
            },
            1); //Run only one thread in debug mode
#else
            });
#endif
    }
    //remove too close detections
    removeTooCloseDetections(params);
    m_checkers.resize(min(nc, (int)m_checkers.size()));
    return !m_checkers.empty();
}

bool CCheckerDetectorImpl::
    process(InputArray image, const COLORCHART chartType,const std::vector<Rect> &regionsOfInterest,
            const int nc /*= 1*/, bool useNet /*=false*/, const DetectorParametersMCC &params)
{
    m_checkers.clear();

    if (this->net.empty() || !useNet)
    {
        return _no_net_process(image, chartType, nc, params, regionsOfInterest);
    }
    this->net_used = true;

    Mat img = image.getMat();

    Mat img_rgb_org, img_ycbcr_org;
    std::vector<Mat> rgb_planes(3), ycbcr_planes(3);

    // Convert to RGB and YCbCr space
    cvtColor(img, img_rgb_org, COLOR_BGR2RGB);
    cvtColor(img, img_ycbcr_org, COLOR_BGR2YCrCb);

    // Get chanels
    split(img_rgb_org, rgb_planes);
    split(img_ycbcr_org, ycbcr_planes);

    for (const Rect &region : regionsOfInterest)
    {
        //-------------------------------------------------------------------
        // Run the model to find good regions
        //-------------------------------------------------------------------

        Mat croppedImage = img(region);

        int rows = croppedImage.size[0];
        int cols = croppedImage.size[1];
        net.setInput(dnn::blobFromImage(croppedImage, 1.0, Size(), Scalar(), true));
        Mat output = net.forward();

        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > params.confidenceThreshold)
            {
                float xTopLeft = max(0.0f, detectionMat.at<float>(i, 3) * cols - params.borderWidth);
                float yTopLeft = max(0.0f, detectionMat.at<float>(i, 4) * rows - params.borderWidth);
                float xBottomRight = min((float)cols - 1, detectionMat.at<float>(i, 5) * cols + params.borderWidth);
                float yBottomRight = min((float)rows - 1, detectionMat.at<float>(i, 6) * rows + params.borderWidth);

                Point2f topLeft = {xTopLeft, yTopLeft};
                Point2f bottomRight = {xBottomRight, yBottomRight};

                Rect innerRegion(topLeft, bottomRight);
                Mat innerCroppedImage = croppedImage(innerRegion);

#ifdef MCC_DEBUG
                std::string pathOut = "./";
#endif
                //-------------------------------------------------------------------
                // prepare image
                //-------------------------------------------------------------------

                Mat img_bgr, img_gray;
                float asp;
                prepareImage(innerCroppedImage, img_gray, img_bgr, asp, params);

                //-------------------------------------------------------------------
                // thresholding
                //-------------------------------------------------------------------

                std::vector<Mat> img_bw;
                performThreshold(img_gray, img_bw, params);

                Mat3f img_rgb_f(img_bgr);
                cvtColor(img_rgb_f, img_rgb_f, COLOR_BGR2RGB);
                img_rgb_f /= 255;
                Mutex mtx;
                parallel_for_(
                    Range(0, (int)img_bw.size()), [&](const Range &range) {
                        const int begin = range.start;
                        const int end = range.end;

                        for (int ind = begin; ind < end; ind++)
                        {

#ifdef MCC_DEBUG
                            showAndSave("threshold_image", img_bw[ind], pathOut);
#endif
                            //------------------------------------------------------------------
                            // find contour
                            //-------------------------------------------------------------------
                            ContoursVector contours;
                            findContours(img_bw[ind], contours, params);

                            if (contours.empty())
                                continue;
#ifdef MCC_DEBUG
                            Mat im_contour(img_bgr.size(), CV_8UC1);
                            im_contour = Scalar(0);
                            drawContours(im_contour, contours, -1, Scalar(255), 2, LINE_AA);
                            showAndSave("find_contour", im_contour, pathOut);
#endif
                            //-------------------------------------------------------------------
                            // find candidate
                            //-------------------------------------------------------------------

                            std::vector<CChart> detectedCharts;
                            findCandidates(contours, detectedCharts, params);

                            if (detectedCharts.empty())
                                continue;

#ifdef MCC_DEBUG
                            Mat img_chart;
                            img_bgr.copyTo(img_chart);

                            for (size_t index = 0; index < detectedCharts.size(); index++)
                            {

                                CChartDraw chrtdrw((detectedCharts[index]), img_chart);
                                chrtdrw.drawCenter();
                                chrtdrw.drawContour();
                            }
                            showAndSave("find_candidate", img_chart, pathOut);
#endif
                            //-------------------------------------------------------------------
                            // clusters analysis
                            //-------------------------------------------------------------------

                            std::vector<int> G;
                            clustersAnalysis(detectedCharts, G, params);

                            if (G.empty())
                                continue;
#ifdef MCC_DEBUG
                            Mat im_gru;
                            img_bgr.copyTo(im_gru);
                            RNG rng(0xFFFFFFFF);
                            int radius = 10, thickness = -1;

                            std::vector<int> g;
                            unique(G, g);
                            size_t Nc = g.size();
                            std::vector<Scalar> colors(Nc);
                            for (size_t index = 0; index < Nc; index++)
                                colors[index] = randomcolor(rng);

                            for (size_t index = 0; index < detectedCharts.size(); index++)
                                circle(im_gru, detectedCharts[index].center, radius, colors[G[index]],
                                           thickness);
                            showAndSave("clusters_analysis", im_gru, pathOut);
#endif

                            //-------------------------------------------------------------------
                            // checker color recognize
                            //-------------------------------------------------------------------

                            std::vector<std::vector<Point2f>> colorCharts;
                            checkerRecognize(img_bgr, detectedCharts, G, chartType, colorCharts, params);

                            if (colorCharts.empty())
                                continue;

#ifdef MCC_DEBUG
                            Mat image_box;
                            img_bgr.copyTo(image_box);
                            for (size_t index = 0; index < colorCharts.size(); index++)
                            {
                                std::vector<Point2f> ibox = colorCharts[index];
                                Scalar color_box = CV_RGB(0, 0, 255);
                                int thickness_box = 2;
                                line(image_box, ibox[0], ibox[1], color_box, thickness_box, LINE_AA);
                                line(image_box, ibox[1], ibox[2], color_box, thickness_box, LINE_AA);
                                line(image_box, ibox[2], ibox[3], color_box, thickness_box, LINE_AA);
                                line(image_box, ibox[3], ibox[0], color_box, thickness_box, LINE_AA);
                                //circle(image_box, ibox[0], 10, Scalar(0, 0, 255), 3);
                                //circle(image_box, ibox[1], 10, Scalar(0, 255, 0), 3);
                            }
                            showAndSave("checker_recognition", image_box, pathOut);
#endif
                            //-------------------------------------------------------------------
                            // checker color analysis
                            //-------------------------------------------------------------------
                            std::vector<Ptr<CChecker>> checkers;
                            checkerAnalysis(img_rgb_f, chartType, nc, colorCharts, checkers, asp, params,
                                            img_rgb_org, img_ycbcr_org, rgb_planes, ycbcr_planes);
#ifdef MCC_DEBUG
                            Mat image_checker;
                            innerCroppedImage.copyTo(image_checker);
                            draw(checkers, image_checker);
                            showAndSave("checker_analysis", image_checker, pathOut);
#endif
                            for (Ptr<CChecker> checker : checkers)
                            {
                                for (Point2f &corner : checker->getBox())
                                    corner += static_cast<Point2f>(region.tl() + innerRegion.tl());

                                {
                                    AutoLock lock(mtx);
                                    m_checkers.push_back(checker);
                                }
                            }
                        }
#ifdef MCC_DEBUG
                    },
                    1); //Run only one thread in debug mode
#else
                    });
#endif
            }
        }
    }
    // As a failsafe try the classical method
    if (m_checkers.empty())
    {
        return _no_net_process(image, chartType, nc, params, regionsOfInterest);
    }
    //remove too close detections
    removeTooCloseDetections(params);

    m_checkers.resize(min(nc, (int)m_checkers.size()));

    return !m_checkers.empty();
}


//Overload for the above function
bool CCheckerDetectorImpl::
    process(InputArray image, const COLORCHART chartType,
            const int nc /*= 1*/, bool useNet /*=false*/, const DetectorParametersMCC &params)
{
    return process(image, chartType, std::vector<Rect>(1, Rect(0, 0, image.cols(), image.rows())),
                   nc,useNet, params);
}


void CCheckerDetectorImpl::
    prepareImage(InputArray bgr, OutputArray grayOut,
                 OutputArray bgrOut, float &aspOut,
                 const DetectorParametersMCC &params) const
{

    int min_size;
    Size size = bgr.size();
    aspOut = 1;
    bgr.copyTo(bgrOut);

    // Resize image
    min_size = std::min(size.width, size.height);
    if (params.minImageSize > min_size)
    {
        aspOut = (float)params.minImageSize / min_size;
        resize(bgr, bgrOut, Size(int(size.width * aspOut), int(size.height * aspOut)), INTER_LINEAR_EXACT);
    }

    // Convert to grayscale
    cvtColor(bgrOut, grayOut, COLOR_BGR2GRAY);

    // PDiamel: wiener adaptative methods to minimize the noise effets
    // by illumination

    CWienerFilter filter;
    filter.wiener2(grayOut, grayOut, 5, 5);

    //JLeandro: perform morphological open on the equalized image
    //to minimize the noise effects by CLAHE and to even intensities
    //inside the MCC patches (regions)

    Mat strelbox = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(grayOut, grayOut, MORPH_OPEN, strelbox);
}

void CCheckerDetectorImpl::
    performThreshold(InputArray grayscaleImg,
                     OutputArrayOfArrays thresholdImgs,
                     const DetectorParametersMCC &params) const
{
    // number of window sizes (scales) to apply adaptive thresholding
    int nScales = (params.adaptiveThreshWinSizeMax - params.adaptiveThreshWinSizeMin) / params.adaptiveThreshWinSizeStep + 1;
    thresholdImgs.create(nScales, 1, CV_8U);
    std::vector<Mat> _thresholdImgs(nScales);
    parallel_for_(Range(0, nScales),[&](const Range& range) {
        const int start = range.start;
        const int end = range.end;
        for (int i = start; i < end; i++) {
            int currScale = params.adaptiveThreshWinSizeMin + i * params.adaptiveThreshWinSizeStep;
            Mat tempThresholdImg;
            adaptiveThreshold(grayscaleImg, tempThresholdImg, 255, ADAPTIVE_THRESH_MEAN_C,
                                  THRESH_BINARY_INV, currScale, params.adaptiveThreshConstant);
            _thresholdImgs[i] = tempThresholdImg;
        }
    });

    thresholdImgs.assign(_thresholdImgs);
}

void CCheckerDetectorImpl::
    findContours(
        InputArray srcImg,
        ContoursVector &contours,
        const DetectorParametersMCC &params) const
{
    // contour detected
    // [Suzuki85] Suzuki, S. and Abe, K., Topological Structural Analysis of Digitized
    // Binary Images by Border Following. CVGIP 30 1, pp 32-46 (1985)
    ContoursVector allContours;
    cv::findContours(srcImg, allContours, RETR_LIST, CHAIN_APPROX_NONE);

    //select contours
    contours.clear();

    const long long int srcImgArea = srcImg.rows() * srcImg.cols();
    for (size_t i = 0; i < allContours.size(); i++)
    {

        PointsVector contour;
        contour = allContours[i];

        int contourSize = (int)contour.size();
        if (contourSize <= params.minContourPointsAllowed)
            continue;

        double area = contourArea(contour);

        if (this->net_used && area / srcImgArea < params.minContoursAreaRate)
            continue;

        if (!this->net_used && area < params.minContoursArea)
            continue;

        PointsVector hull;
        convexHull(contour, hull);
        double area_hull = contourArea(hull);
        double S = area / area_hull;
        if (S < params.minContourSolidity)
            continue;

        contours.push_back(allContours[i]);
    }
}

void CCheckerDetectorImpl::
    findCandidates(
        const ContoursVector &contours,
        std::vector<CChart> &detectedCharts,
        const DetectorParametersMCC &params)
{
    std::vector<Point> approxCurve;
    std::vector<CChart> possibleCharts;

    // For each contour, analyze if it is a parallelepiped likely to be the chart
    for (size_t i = 0; i < contours.size(); i++)
    {
        // Approximate to a polygon
        //  It uses the Douglas-Peucker algorithm
        // http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
        double eps = contours[i].size() * params.findCandidatesApproxPolyDPEpsMultiplier;
        approxPolyDP(contours[i], approxCurve, eps, true);

        // We interested only in polygons that contains only four points
        if (approxCurve.size() != 4)
            continue;

        // And they have to be convex
        if (!isContourConvex(approxCurve))
            continue;

        // Ensure that the distance between consecutive points is large enough
        float minDist = INFINITY;

        for (size_t j = 0; j < 4; j++)
        {
            Point side = approxCurve[j] - approxCurve[(j + 1) % 4];
            float squaredSideLength = (float)side.dot(side);
            minDist = std::min(minDist, squaredSideLength);
        }

        // Check that distance is not very small
        if (minDist < params.minContourLengthAllowed)
            continue;

        // All tests are passed. Save chart candidate:
        CChart chart;

        std::vector<Point2f> corners(4);
        for (int j = 0; j < 4; j++)
            corners[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
        chart.setCorners(corners);

        possibleCharts.push_back(chart);
    }

    // Remove these elements which corners are too close to each other.
    // Eliminate overlaps!!!
    // First detect candidates for removal:
    std::vector<std::pair<int, int>> tooNearCandidates;
    for (int i = 0; i < (int)possibleCharts.size(); i++)
    {
        const CChart &m1 = possibleCharts[i];

        //calculate the average distance of each corner to the nearest corner of the other chart candidate
        for (int j = i + 1; j < (int)possibleCharts.size(); j++)
        {
            const CChart &m2 = possibleCharts[j];

            float distSquared = 0;

            for (int c = 0; c < 4; c++)
            {
                Point v = m1.corners[c] - m2.corners[c];
                distSquared += v.dot(v);
            }

            distSquared /= 4;

            if (distSquared < params.minInterContourDistance)
            {
                tooNearCandidates.push_back(std::pair<int, int>(i, j));
            }
        }
    }

    // Mark for removal the element of the pair with smaller perimeter
    std::vector<bool> removalMask(possibleCharts.size(), false);

    for (size_t i = 0; i < tooNearCandidates.size(); i++)
    {
        float p1 = perimeter(possibleCharts[tooNearCandidates[i].first].corners);
        float p2 = perimeter(possibleCharts[tooNearCandidates[i].second].corners);

        size_t removalIndex;
        if (p1 > p2)
            removalIndex = tooNearCandidates[i].second;
        else
            removalIndex = tooNearCandidates[i].first;

        removalMask[removalIndex] = true;
    }

    detectedCharts.clear();
    for (size_t i = 0; i < possibleCharts.size(); i++)
    {
        if (removalMask[i])
            continue;
        detectedCharts.push_back(possibleCharts[i]);
    }
}

void CCheckerDetectorImpl::
    clustersAnalysis(
        const std::vector<CChart> &detectedCharts,
        std::vector<int> &groups,
        const DetectorParametersMCC &params)
{
    size_t N = detectedCharts.size();
    std::vector<Point> X(N);
    std::vector<double> B0(N), W(N);
    std::vector<int> G;

    CChart chart;
    double b0;
    for (size_t i = 0; i < N; i++)
    {
        chart = detectedCharts[i];
        b0 = chart.large_side * params.B0factor;
        X[i] = chart.center;
        W[i] = chart.area;
        B0[i] = b0;
    }

    CB0cluster bocluster;
    bocluster.setVertex(X);
    bocluster.setWeight(W);
    bocluster.setB0(B0);
    bocluster.group();
    bocluster.getGroup(G);
    groups = G;
}

void CCheckerDetectorImpl::
    checkerRecognize(
        InputArray img,
        const std::vector<CChart> &detectedCharts,
        const std::vector<int> &G,
        const COLORCHART chartType,
        std::vector<std::vector<Point2f>> &colorChartsOut,
        const DetectorParametersMCC &params)
{
    std::vector<int> gU;
    unique(G, gU);
    size_t Nc = gU.size();                //numero de grupos
    size_t Ncc = detectedCharts.size(); //numero de charts

    std::vector<std::vector<Point2f>> colorCharts;

    for (size_t g = 0; g < Nc; g++)
    {

        ///-------------------------------------------------
        /// selecionar grupo i-esimo

        std::vector<CChart> chartSub;
        for (size_t i = 0; i < Ncc; i++)
            if (G[i] == (int)g)
                chartSub.push_back(detectedCharts[i]);

        size_t Nsc = chartSub.size();
        if (Nsc < params.minGroupSize)
            continue;

        ///-------------------------------------------------
        /// min box estimation

        CBoundMin bm;
        std::vector<Point2f> points;

        bm.setCharts(chartSub);
        bm.calculate();
        bm.getCorners(points);

        // boundary condition
        if (points.size() == 0)
            continue;

        // sort the points in anti-clockwise order
        polyanticlockwise(points);

        ///-------------------------------------------------
        /// box projective transformation

        // get physical char box model
        std::vector<Point2f> chartPhy;
        get_subbox_chart_physical(points, chartPhy);

        // Find the perspective transformation that brings current chart to rectangular form
        Matx33f ccT = getPerspectiveTransform(points, chartPhy);

        // transformer
        std::vector<Point2f> c(Nsc), ct;
        std::vector<Point2f> ch(4 * Nsc), cht;

        for (size_t i = 0; i < Nsc; i++)
        {

            CChart cc = chartSub[i];
            for (size_t j = 0; j < 4; j++)
                ch[i * 4 + j] = cc.corners[j];
            c[i] = chartSub[i].center;
        }

        transform_points_forward(ccT, c, ct);
        transform_points_forward(ccT, ch, cht);

        float wchart = 0, hchart = 0;
        std::vector<float> cx(Nsc), cy(Nsc);
        for (size_t i = 0, k = 0; i < Nsc; i++)
        {
            k = i * 4;
            Point2f v1 = cht[k + 1] - cht[k + 0];
            Point2f v2 = cht[k + 3] - cht[k + 0];
            wchart += (float)norm(v1);
            hchart += (float)norm(v2);
            cx[i] = ct[i].x;
            cy[i] = ct[i].y;
        }

        wchart /= Nsc;
        hchart /= Nsc;

        ///-------------------------------------------------
        /// centers and color estimate

        float tolx = wchart / 2, toly = hchart / 2;
        std::vector<float> cxr, cyr;
        reduce_array(cx, cxr, tolx);
        reduce_array(cy, cyr, toly);

        if (cxr.size() == 1 || cyr.size() == 1) //no information can be extracted if \
                                                //only one row or columns in present
            continue;
        // color and center rectificate
        Size2i colorSize = Size2i((int)cxr.size(), (int)cyr.size());
        Mat colorMat(colorSize, CV_32FC3);
        std::vector<Point2f> cte(colorSize.area());

        int k = 0;

        for (int i = 0; i < colorSize.height; i++)
        {
            for (int j = 0; j < colorSize.width; j++)
            {
                Point2f vc = Point2f(cxr[j], cyr[i]);
                cte[k] = vc;

                // recovery color
                Point2f cti;
                Matx31f p, xt;

                p(0, 0) = vc.x;
                p(1, 0) = vc.y;
                p(2, 0) = 1;
                xt = ccT.inv() * p;
                cti.x = xt(0, 0) / xt(2, 0);
                cti.y = xt(1, 0) / xt(2, 0);

                // color
                int x, y;
                x = (int)cti.x;
                y = (int)cti.y;
                Vec3f &srgb = colorMat.at<Vec3f>(i, j);
                Vec3b rgb;
                if (0 <= y && y < img.rows() && 0 <= x && x < img.cols())
                    rgb = img.getMat().at<Vec3b>(y, x);

                srgb[0] = (float)rgb[0] / 255;
                srgb[1] = (float)rgb[1] / 255;
                srgb[2] = (float)rgb[2] / 255;

                k++;
            }
        }

        CChartModel::SUBCCMModel scm;
        scm.centers = cte;
        scm.color_size = colorSize;
        colorMat = colorMat.t();
        scm.sub_chart = colorMat.reshape(3, colorSize.area());

        ///-------------------------------------------------

        // color chart model
        CChartModel cccm(chartType);

        int iTheta;     // rotation angle of chart
        int offset;     // offset
        float error; // min error
        if (!cccm.evaluate(scm, offset, iTheta, error))
            continue;
        if (iTheta >= 4)
            cccm.flip();

        for (int i = 0; i < iTheta % 4; i++)
            cccm.rotate90();

        ///-------------------------------------------------
        /// calculate coordanate

        Size2i dim = cccm.size;
        std::vector<Point2f> center = cccm.center;
        std::vector<Point2f> box = cccm.box;
        int cols = dim.height - colorSize.width + 1;

        int x = (offset) / cols;
        int y = (offset) % cols;

        // seleccionar sub grid centers of model
        std::vector<Point2f> ctss(colorSize.area());
        Point2f point_ac = Point2f(0, 0);
        int p = 0;

        for (int i = x; i < (x + colorSize.height); i++)
        {
            for (int j = y; j < (y + colorSize.width); j++)
            {
                int iter = i * dim.height + j;
                ctss[p] = center[iter];
                point_ac += ctss[p];
                p++;
            }
        }
        // is colineal point
        if (point_ac.x == ctss[0].x * p || point_ac.y == ctss[0].y * p)
            continue;
        // Find the perspective transformation
        Matx33f ccTe = findHomography(ctss, cte);

        std::vector<Point2f> tbox, ibox;
        transform_points_forward(ccTe, box, tbox);
        transform_points_inverse(ccT, tbox, ibox);

        // sort the points in anti-clockwise order
        if (iTheta < 4)
            mcc::polyanticlockwise(ibox);
        else
            mcc::polyclockwise(ibox);
        colorCharts.push_back(ibox);
    }

    colorChartsOut = colorCharts;
}
void CCheckerDetectorImpl::draw(std::vector<Ptr<CChecker>>& checkers, InputOutputArray img, Scalar color, int thickness)
{
    for (Ptr<CChecker> checker : checkers)
    {
        std::vector<Point2f> charts = checker->getColorCharts();
        size_t N = charts.size() / 4;
        for (size_t i = 0, k; i < N; i++)
        {
            k = 4 * i;
            for (size_t j = 0; j < 4; j++)
                line(img, charts[k+j], charts[k+((j + 1) % 4)], color, thickness, LINE_AA);
        }
    }
}
void CCheckerDetectorImpl::getRefColor(const COLORCHART chartType, Mat& output) {
    static const double ColorChecker2005_LAB_D50_2[24][3] = { { 37.986, 13.555, 14.059 },
        { 65.711, 18.13, 17.81 },
        { 49.927, -4.88, -21.925 },
        { 43.139, -13.095, 21.905 },
        { 55.112, 8.844, -25.399 },
        { 70.719, -33.397, -0.199 },
        { 62.661, 36.067, 57.096 },
        { 40.02, 10.41, -45.964 },
        { 51.124, 48.239, 16.248 },
        { 30.325, 22.976, -21.587 },
        { 72.532, -23.709, 57.255 },
        { 71.941, 19.363, 67.857 },
        { 28.778, 14.179, -50.297 },
        { 55.261, -38.342, 31.37 },
        { 42.101, 53.378, 28.19 },
        { 81.733, 4.039, 79.819 },
        { 51.935, 49.986, -14.574 },
        { 51.038, -28.631, -28.638 },
        { 96.539, -0.425, 1.186 },
        { 81.257, -0.638, -0.335 },
        { 66.766, -0.734, -0.504 },
        { 50.867, -0.153, -0.27 },
        { 35.656, -0.421, -1.231 },
        { 20.461, -0.079, -0.973 } };

    static const double Vinyl_LAB_D50_2[18][3] = { { 100, 0.00520000001, -0.0104 },
        { 73.0833969, -0.819999993, -2.02099991 },
        { 62.493, 0.425999999, -2.23099995 },
        { 50.4640007, 0.446999997, -2.32399988 },
        { 37.7970009, 0.0359999985, -1.29700005 },
        { 0, 0, 0 },
        { 51.5880013, 73.5179977, 51.5690002 },
        { 93.6989975, -15.7340002, 91.9420013 },
        { 69.4079971, -46.5940018, 50.4869995 },
        { 66.61000060000001, -13.6789999, -43.1720009 },
        { 11.7110004, 16.9799995, -37.1759987 },
        { 51.973999, 81.9440002, -8.40699959 },
        { 40.5489998, 50.4399986, 24.8490009 },
        { 60.8160019, 26.0690002, 49.4420013 },
        { 52.2529984, -19.9500008, -23.9960003 },
        { 51.2859993, 48.4700012, -15.0579996 },
        { 68.70700069999999, 12.2959995, 16.2129993 },
        { 63.6839981, 10.2930002, 16.7639999 } };

    static const double DigitalSG_LAB_D50_2[140][3] = { { 96.55, -0.91, 0.57 },
        { 6.43, -0.06, -0.41 },
        { 49.7, -0.18, 0.03 },
        { 96.5, -0.89, 0.59 },
        { 6.5, -0.06, -0.44 },
        { 49.66, -0.2, 0.01 },
        { 96.52, -0.91, 0.58 },
        { 6.49, -0.02, -0.28 },
        { 49.72, -0.2, 0.04 },
        { 96.43, -0.91, 0.67 },
        { 49.72, -0.19, 0 },
        { 32.6, 51.58, -10.85 },
        { 60.75, 26.22, -18.6 },
        { 28.69, 48.28, -39 },
        { 49.38, -15.43, -48.48 },
        { 60.63, -30.77, -26.23 },
        { 19.29, -26.37, -6.15 },
        { 60.15, -41.77, -12.6 },
        { 21.42, 1.67, 8.79 },
        { 49.69, -0.2, 0.01 },
        { 6.5, -0.03, -0.67 },
        { 21.82, 17.33, -18.35 },
        { 41.53, 18.48, -37.26 },
        { 19.99, -0.16, -36.29 },
        { 60.16, -18.45, -31.42 },
        { 19.94, -17.92, -20.96 },
        { 60.68, -6.05, -32.81 },
        { 50.81, -49.8, -9.63 },
        { 60.65, -39.77, 20.76 },
        { 6.53, -0.03, -0.43 },
        { 96.56, -0.91, 0.59 },
        { 84.19, -1.95, -8.23 },
        { 84.75, 14.55, 0.23 },
        { 84.87, -19.07, -0.82 },
        { 85.15, 13.48, 6.82 },
        { 84.17, -10.45, 26.78 },
        { 61.74, 31.06, 36.42 },
        { 64.37, 20.82, 18.92 },
        { 50.4, -53.22, 14.62 },
        { 96.51, -0.89, 0.65 },
        { 49.74, -0.19, 0.03 },
        { 31.91, 18.62, 21.99 },
        { 60.74, 38.66, 70.97 },
        { 19.35, 22.23, -58.86 },
        { 96.52, -0.91, 0.62 },
        { 6.66, 0, -0.3 },
        { 76.51, 20.81, 22.72 },
        { 72.79, 29.15, 24.18 },
        { 22.33, -20.7, 5.75 },
        { 49.7, -0.19, 0.01 },
        { 6.53, -0.05, -0.61 },
        { 63.42, 20.19, 19.22 },
        { 34.94, 11.64, -50.7 },
        { 52.03, -44.15, 39.04 },
        { 79.43, 0.29, -0.17 },
        { 30.67, -0.14, -0.53 },
        { 63.6, 14.44, 26.07 },
        { 64.37, 14.5, 17.05 },
        { 60.01, -44.33, 8.49 },
        { 6.63, -0.01, -0.47 },
        { 96.56, -0.93, 0.59 },
        { 46.37, -5.09, -24.46 },
        { 47.08, 52.97, 20.49 },
        { 36.04, 64.92, 38.51 },
        { 65.05, 0, -0.32 },
        { 40.14, -0.19, -0.38 },
        { 43.77, 16.46, 27.12 },
        { 64.39, 17, 16.59 },
        { 60.79, -29.74, 41.5 },
        { 96.48, -0.89, 0.64 },
        { 49.75, -0.21, 0.01 },
        { 38.18, -16.99, 30.87 },
        { 21.31, 29.14, -27.51 },
        { 80.57, 3.85, 89.61 },
        { 49.71, -0.2, 0.03 },
        { 60.27, 0.08, -0.41 },
        { 67.34, 14.45, 16.9 },
        { 64.69, 16.95, 18.57 },
        { 51.12, -49.31, 44.41 },
        { 49.7, -0.2, 0.02 },
        { 6.67, -0.05, -0.64 },
        { 51.56, 9.16, -26.88 },
        { 70.83, -24.26, 64.77 },
        { 48.06, 55.33, -15.61 },
        { 35.26, -0.09, -0.24 },
        { 75.16, 0.25, -0.2 },
        { 44.54, 26.27, 38.93 },
        { 35.91, 16.59, 26.46 },
        { 61.49, -52.73, 47.3 },
        { 6.59, -0.05, -0.5 },
        { 96.58, -0.9, 0.61 },
        { 68.93, -34.58, -0.34 },
        { 69.65, 20.09, 78.57 },
        { 47.79, -33.18, -30.21 },
        { 15.94, -0.42, -1.2 },
        { 89.02, -0.36, -0.48 },
        { 63.43, 25.44, 26.25 },
        { 65.75, 22.06, 27.82 },
        { 61.47, 17.1, 50.72 },
        { 96.53, -0.89, 0.66 },
        { 49.79, -0.2, 0.03 },
        { 85.17, 10.89, 17.26 },
        { 89.74, -16.52, 6.19 },
        { 84.55, 5.07, -6.12 },
        { 84.02, -13.87, -8.72 },
        { 70.76, 0.07, -0.35 },
        { 45.59, -0.05, 0.23 },
        { 20.3, 0.07, -0.32 },
        { 61.79, -13.41, 55.42 },
        { 49.72, -0.19, 0.02 },
        { 6.77, -0.05, -0.44 },
        { 21.85, 34.37, 7.83 },
        { 42.66, 67.43, 48.42 },
        { 60.33, 36.56, 3.56 },
        { 61.22, 36.61, 17.32 },
        { 62.07, 52.8, 77.14 },
        { 72.42, -9.82, 89.66 },
        { 62.03, 3.53, 57.01 },
        { 71.95, -27.34, 73.69 },
        { 6.59, -0.04, -0.45 },
        { 49.77, -0.19, 0.04 },
        { 41.84, 62.05, 10.01 },
        { 19.78, 29.16, -7.85 },
        { 39.56, 65.98, 33.71 },
        { 52.39, 68.33, 47.84 },
        { 81.23, 24.12, 87.51 },
        { 81.8, 6.78, 95.75 },
        { 71.72, -16.23, 76.28 },
        { 20.31, 14.45, 16.74 },
        { 49.68, -0.19, 0.05 },
        { 96.48, -0.88, 0.68 },
        { 49.69, -0.18, 0.03 },
        { 6.39, -0.04, -0.33 },
        { 96.54, -0.9, 0.67 },
        { 49.72, -0.18, 0.05 },
        { 6.49, -0.03, -0.41 },
        { 96.51, -0.9, 0.69 },
        { 49.7, -0.19, 0.07 },
        { 6.47, 0, -0.38 },
        { 96.46, -0.89, 0.7 } };

    Mat lab_color(1, 1, CV_32FC3), rgb_color(1, 1, CV_32FC3);

    const double (*colorData)[3] = nullptr;
    int colorCount = 0;

    switch (chartType) {
        case MCC24:
            colorData = ColorChecker2005_LAB_D50_2;
            colorCount = 24;
            break;
        case VINYL18:
            colorData = Vinyl_LAB_D50_2;
            colorCount = 18;
            break;
        case SG140:
            colorData = DigitalSG_LAB_D50_2;
            colorCount = 140;
            break;
        default:
            throw std::runtime_error("Unknown color type requested.");
    }

    Mat out = Mat(colorCount, 1, CV_32SC3);

    for (int i = 0; i < colorCount; i++) {
        lab_color.at<Vec3f>(0, 0) = Vec3f(
            static_cast<float>(colorData[i][0]),
            static_cast<float>(colorData[i][1]),
            static_cast<float>(colorData[i][2])
        );

        cvtColor(lab_color, rgb_color, COLOR_Lab2BGR);

        // Convert to 8-bit RGB values (0-255) but store as int
        Vec3f rgb = rgb_color.at<Vec3f>(0, 0) * 255;
        out.at<Vec3i>(i, 0) = Vec3i(
            std::min(255, std::max(0, static_cast<int>(rgb[0]))),  // Blue
            std::min(255, std::max(0, static_cast<int>(rgb[1]))),  // Green
            std::min(255, std::max(0, static_cast<int>(rgb[2])))   // Red
        );
    }
    Mat(out).copyTo(output);
}

void CCheckerDetectorImpl::
    checkerAnalysis(
        InputArray img_f,
        const COLORCHART chartType,
        const unsigned int nc,
        const std::vector<std::vector<Point2f>> &colorCharts,
        std::vector<Ptr<CChecker>> &checkers,
        float asp,
        const DetectorParametersMCC &params,
        const Mat &img_rgb_org,
        const Mat &img_ycbcr_org,
        std::vector<Mat> &rgb_planes,
        std::vector<Mat> &ycbcr_planes)
{
    size_t N;
    std::vector<Point2f> ibox;

    // color chart classic model
    CChartModel cccm(chartType);
    Mat lab;
    cccm.copyToColorMat(lab, 0);
    lab = lab.reshape(3, lab.size().area());
    lab /= 255;

    Mat mask(img_f.size(), CV_8U);
    mask.setTo(Scalar::all(0));

    N = colorCharts.size();
    std::vector<float> J(N);
    for (size_t i = 0; i < N; i++)
    {
        ibox = colorCharts[i];
        J[i] = cost_function(img_f, mask, lab, ibox, chartType);
    }

    std::vector<int> idx;
    sort(J, idx);
    float invAsp = 1 / asp;
    size_t n = min(nc, (unsigned)N);
    checkers.clear();

    for (size_t i = 0; i < n; i++)
    {
        ibox = colorCharts[idx[i]];

        if (J[i] > params.maxError)
            continue;

        // redimention box
        for (size_t j = 0; j < 4; j++)
            ibox[j] = invAsp * ibox[j];

        Mat charts_rgb, charts_ycbcr;
        get_profile(ibox, chartType, charts_rgb, charts_ycbcr, img_rgb_org,
                    img_ycbcr_org, rgb_planes, ycbcr_planes);

        // result
        Ptr<CChecker> checker = CChecker::create();
        checker->setBox(ibox);
        checker->setTarget(chartType);
        checker->setChartsRGB(charts_rgb);
        checker->setChartsYCbCr(charts_ycbcr);
        checker->setCenter(mace_center(ibox));
        checker->setCost(J[i]);

        checkers.push_back(checker);
    }
}

void CCheckerDetectorImpl::
    removeTooCloseDetections(const DetectorParametersMCC &params)
{
    // Remove these elements which corners are too close to each other.
    // Eliminate overlaps!!!
    // First detect candidates for removal:
    std::vector<std::pair<int, int>> tooNearCandidates;
    for (int i = 0; i < (int)m_checkers.size(); i++)
    {
        const Ptr<CChecker> &m1 = m_checkers[i];

        //calculate the average distance of each corner to the nearest corner of the other chart candidate
        for (int j = i + 1; j < (int)m_checkers.size(); j++)
        {
            const Ptr<CChecker> &m2 = m_checkers[j];

            float distSquared = 0;

            for (int c = 0; c < 4; c++)
            {
                Point v = m1->getBox()[c] - m2->getBox()[c];
                distSquared += v.dot(v);
            }

            distSquared /= 4;

            if (distSquared < params.minInterCheckerDistance)
            {
                tooNearCandidates.push_back(std::pair<int, int>(i, j));
            }
        }
    }

    // Mark for removal the element of the pair with smaller cost
    std::vector<bool> removalMask(m_checkers.size(), false);

    for (size_t i = 0; i < tooNearCandidates.size(); i++)
    {
        float p1 = m_checkers[tooNearCandidates[i].first]->getCost();
        float p2 = m_checkers[tooNearCandidates[i].second]->getCost();

        size_t removalIndex;
        if (p1 < p2)
            removalIndex = tooNearCandidates[i].second;
        else
            removalIndex = tooNearCandidates[i].first;

        removalMask[removalIndex] = true;
    }

    std::vector<Ptr<CChecker>> copy_m_checkers = m_checkers;
    m_checkers.clear();

    for (size_t i = 0; i < copy_m_checkers.size(); i++)
    {
        if (removalMask[i])
            continue;
        m_checkers.push_back(copy_m_checkers[i]);
    }

    sort( m_checkers.begin(), m_checkers.end(),
          [&](const Ptr<CChecker> &a, const Ptr<CChecker> &b)
          {
              return a->getCost() < b->getCost();
          });
}

void CCheckerDetectorImpl::
    get_subbox_chart_physical(const std::vector<Point2f> &points, std::vector<Point2f> &chartPhy)
{
    float w, h;
    Point2f v1 = points[1] - points[0];
    Point2f v2 = points[3] - points[0];
    float asp = (float)(norm(v2) / norm(v1));

    w = 100;
    h = (float)floor(100 * asp + 0.5);

    chartPhy.clear();
    chartPhy.resize(4);
    chartPhy[0] = Point2f(0, 0);
    chartPhy[1] = Point2f(w, 0);
    chartPhy[2] = Point2f(w, h);
    chartPhy[3] = Point2f(0, h);
}

void CCheckerDetectorImpl::
    reduce_array(const std::vector<float> &x, std::vector<float> &x_new, float tol)
{
    size_t n = x.size(), nn;
    std::vector<float> xx = x;
    x_new.clear();

    // sort array
    std::sort(xx.begin(), xx.end());

    // label array
    std::vector<int> label(n);
    for (size_t i = 0; i < n; i++)
        label[i] = abs(xx[(n + i - 1) % n] - xx[i]) > tol;

    // diff array
    for (size_t i = 1; i < n; i++)
        label[i] += label[i - 1];

    // unique array
    std::vector<int> ulabel;
    unique(label, ulabel);

    // mean for group
    nn = ulabel.size();
    x_new.resize(nn);
    for (size_t i = 0; i < nn; i++)
    {
        float mu = 0, s = 0;
        for (size_t j = 0; j < n; j++)
        {
            mu += (label[j] == ulabel[i]) * xx[j];
            s += (label[j] == ulabel[i]);
        }
        x_new[i] = mu / s;
    }

    // diff array
    std::vector<float> dif(nn - 1);
    for (size_t i = 0; i < nn - 1; i++)
        dif[i] = (x_new[(i + 1) % nn] - x_new[i]);

    // max and idx
    float fmax = 0;
    size_t idx = 0;
    for (size_t i = 0; i < nn - 1; i++)
        if (fmax < dif[i])
        {
            fmax = dif[i];
            idx = i;
        }

    // add ... X[i] MAX X[i+] ...
    if (fmax > 4 * tol)
        x_new.insert(x_new.begin() + idx + 1, (x_new[idx] + x_new[idx + 1]) / 2);
}

void CCheckerDetectorImpl::
    transform_points_inverse(InputArray T, const std::vector<Point2f> &X, std::vector<Point2f> &Xt)
{
    Matx33f _T = T.getMat();
    Matx33f Tinv = _T.inv();
    transform_points_forward(Tinv, X, Xt);
}
void CCheckerDetectorImpl::
    get_profile(
        const std::vector<Point2f> &ibox,
        const COLORCHART chartType,
        OutputArray charts_rgb,
        OutputArray charts_ycbcr,
        InputArray im_rgb,
        InputArray im_ycbcr,
        std::vector<Mat> &rgb_planes,
        std::vector<Mat> &ycbcr_planes)
{
    // color chart classic model
    CChartModel cccm(chartType);
    Mat lab;
    size_t N;
    std::vector<Point2f> fbox = cccm.box;
    std::vector<Point2f> cellchart = cccm.cellchart;

    // tranformation
    Matx33f ccT = getPerspectiveTransform(fbox, ibox);

    Mat mask(im_rgb.size(), CV_8U);
    mask.setTo(Scalar::all(0));
    std::vector<Point2f> bch(4), bcht(4);
    N = cellchart.size() / 4;

    // Create table charts information
    //          |p_size|average|stddev|max|min|
    //    RGB   |      |       |      |   |   |
    //  YCbCr   |

    Mat _charts_rgb = Mat(Size(5, 3 * (int)N), CV_64F);
    Mat _charts_ycbcr = Mat(Size(5, 3 * (int)N), CV_64F);

    Scalar mu_rgb, st_rgb, mu_ycb, st_ycb, p_size;
    double max_rgb[3], min_rgb[3], max_ycb[3], min_ycb[3];

    for (int i = 0, k; i < (int)N; i++)
    {
        k = 4 * i;
        bch[0] = cellchart[k + 0];
        bch[1] = cellchart[k + 1];
        bch[2] = cellchart[k + 2];
        bch[3] = cellchart[k + 3];
        polyanticlockwise(bch);
        transform_points_forward(ccT, bch, bcht);

        Point2f c(0, 0);
        for (int j = 0; j < 4; j++)
            c += bcht[j];
        c /= 4;
        for (size_t j = 0; j < 4; j++)
            bcht[j] = ((bcht[j] - c) * 0.50) + c;

        Rect roi = poly2mask(bcht, im_rgb.size(), mask);
        Mat submask = mask(roi);
        p_size = sum(submask);

        // rgb space
        meanStdDev(im_rgb.getMat()(roi), mu_rgb, st_rgb, submask);
        minMaxLoc(rgb_planes[0](roi), &min_rgb[0], &max_rgb[0], NULL, NULL, submask);
        minMaxLoc(rgb_planes[1](roi), &min_rgb[1], &max_rgb[1], NULL, NULL, submask);
        minMaxLoc(rgb_planes[2](roi), &min_rgb[2], &max_rgb[2], NULL, NULL, submask);

        // create tabla
        //|p_size|average|stddev|max|min|
        // raw_r
        _charts_rgb.at<double>(3 * i + 0, 0) = p_size(0);
        _charts_rgb.at<double>(3 * i + 0, 1) = mu_rgb(0);
        _charts_rgb.at<double>(3 * i + 0, 2) = st_rgb(0);
        _charts_rgb.at<double>(3 * i + 0, 3) = min_rgb[0];
        _charts_rgb.at<double>(3 * i + 0, 4) = max_rgb[0];
        // raw_g
        _charts_rgb.at<double>(3 * i + 1, 0) = p_size(0);
        _charts_rgb.at<double>(3 * i + 1, 1) = mu_rgb(1);
        _charts_rgb.at<double>(3 * i + 1, 2) = st_rgb(1);
        _charts_rgb.at<double>(3 * i + 1, 3) = min_rgb[1];
        _charts_rgb.at<double>(3 * i + 1, 4) = max_rgb[1];
        // raw_b
        _charts_rgb.at<double>(3 * i + 2, 0) = p_size(0);
        _charts_rgb.at<double>(3 * i + 2, 1) = mu_rgb(2);
        _charts_rgb.at<double>(3 * i + 2, 2) = st_rgb(2);
        _charts_rgb.at<double>(3 * i + 2, 3) = min_rgb[2];
        _charts_rgb.at<double>(3 * i + 2, 4) = max_rgb[2];

        // YCbCr space
        meanStdDev(im_ycbcr.getMat()(roi), mu_ycb, st_ycb, submask);
        minMaxLoc(ycbcr_planes[0](roi), &min_ycb[0], &max_ycb[0], NULL, NULL, submask);
        minMaxLoc(ycbcr_planes[1](roi), &min_ycb[1], &max_ycb[1], NULL, NULL, submask);
        minMaxLoc(ycbcr_planes[2](roi), &min_ycb[2], &max_ycb[2], NULL, NULL, submask);

        // create tabla
        //|p_size|average|stddev|max|min|
        // raw_Y
        _charts_ycbcr.at<double>(3 * i + 0, 0) = p_size(0);
        _charts_ycbcr.at<double>(3 * i + 0, 1) = mu_ycb(0);
        _charts_ycbcr.at<double>(3 * i + 0, 2) = st_ycb(0);
        _charts_ycbcr.at<double>(3 * i + 0, 3) = min_ycb[0];
        _charts_ycbcr.at<double>(3 * i + 0, 4) = max_ycb[0];
        // raw_Cb
        _charts_ycbcr.at<double>(3 * i + 1, 0) = p_size(0);
        _charts_ycbcr.at<double>(3 * i + 1, 1) = mu_ycb(1);
        _charts_ycbcr.at<double>(3 * i + 1, 2) = st_ycb(1);
        _charts_ycbcr.at<double>(3 * i + 1, 3) = min_ycb[1];
        _charts_ycbcr.at<double>(3 * i + 1, 4) = max_ycb[1];
        // raw_Cr
        _charts_ycbcr.at<double>(3 * i + 2, 0) = p_size(0);
        _charts_ycbcr.at<double>(3 * i + 2, 1) = mu_ycb(2);
        _charts_ycbcr.at<double>(3 * i + 2, 2) = st_ycb(2);
        _charts_ycbcr.at<double>(3 * i + 2, 3) = min_ycb[2];
        _charts_ycbcr.at<double>(3 * i + 2, 4) = max_ycb[2];

        submask.setTo(Scalar::all(0));
    }
    charts_rgb.assign(_charts_rgb);
    charts_ycbcr.assign(_charts_ycbcr);
}

float CCheckerDetectorImpl::
    cost_function(InputArray im_rgb, InputOutputArray mask, InputArray lab,
                  const std::vector<Point2f> &ibox, const COLORCHART chartType)
{
    CChartModel cccm(chartType);
    std::vector<Point2f> fbox = cccm.box;
    std::vector<Point2f> cellchart = cccm.cellchart;

    // tranformation
    Matx33f ccT = getPerspectiveTransform(fbox, ibox);
    std::vector<Point2f> bch(4), bcht(4);

    int N = (int)(cellchart.size() / 4);

    Mat _lab = lab.getMat();
    Mat _im_rgb = im_rgb.getMat();

    float ec = 0, es = 0;
    for (int i = 0, k; i < N; i++)
    {
        Vec3f r = _lab.at<Vec3f>(i);

        k = 4 * i;
        bch[0] = cellchart[k + 0];
        bch[1] = cellchart[k + 1];
        bch[2] = cellchart[k + 2];
        bch[3] = cellchart[k + 3];
        polyanticlockwise(bch);
        transform_points_forward(ccT, bch, bcht);

        Point2f c(0, 0);
        for (int j = 0; j < 4; j++)
            c += bcht[j];
        c /= 4;
        for (int j = 0; j < 4; j++)
            bcht[j] = ((bcht[j] - c) * 0.75) + c;

        Scalar mu, st;
        Rect roi = poly2mask(bcht, _im_rgb.size(), mask);
        if (!roi.empty())
        {
            Mat submask = mask.getMat()(roi);
            meanStdDev(_im_rgb(roi), mu, st, submask);
            submask.setTo(Scalar::all(0));

            // cos error
            float costh;
            costh = (float)(mu.dot(Scalar(r)) / (norm(mu) * norm(r) + FLT_EPSILON));
            ec += (1 - (1 + costh) / 2);

            // standar desviation
            es += (float)st.dot(st);
        }
    }

    // J = arg min ec + es
    float J = ec + es;
    return J / N;
}

} // namespace mcc
} // namespace cv
