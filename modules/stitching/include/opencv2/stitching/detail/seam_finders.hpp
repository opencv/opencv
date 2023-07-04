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

#ifndef OPENCV_STITCHING_SEAM_FINDERS_HPP
#define OPENCV_STITCHING_SEAM_FINDERS_HPP

#include <set>
#include "opencv2/core.hpp"
#include "opencv2/opencv_modules.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching_seam
//! @{

/** @brief Base class for a seam estimator.
 */
class CV_EXPORTS_W SeamFinder
{
public:
    CV_WRAP  virtual ~SeamFinder() {}
    enum { NO, VORONOI_SEAM, DP_SEAM };
    /** @brief Estimates seams.

    @param src Source images
    @param corners Source image top-left corners
    @param masks Source image masks to update
     */
    CV_WRAP virtual void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                      CV_IN_OUT std::vector<UMat> &masks) = 0;
    CV_WRAP static Ptr<SeamFinder> createDefault(int type);
};

/** @brief Stub seam estimator which does nothing.
 */
class CV_EXPORTS_W NoSeamFinder : public SeamFinder
{
public:
    CV_WRAP void find(const std::vector<UMat>&, const std::vector<Point>&, CV_IN_OUT std::vector<UMat>&) CV_OVERRIDE {}
};

/** @brief Base class for all pairwise seam estimators.
 */
class CV_EXPORTS_W PairwiseSeamFinder : public SeamFinder
{
public:
    CV_WRAP virtual void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                      CV_IN_OUT std::vector<UMat> &masks) CV_OVERRIDE;

protected:
    void run();
    /** @brief Resolves masks intersection of two specified images in the given ROI.

    @param first First image index
    @param second Second image index
    @param roi Region of interest
     */
    virtual void findInPair(size_t first, size_t second, Rect roi) = 0;

    std::vector<UMat> images_;
    std::vector<Size> sizes_;
    std::vector<Point> corners_;
    std::vector<UMat> masks_;
};

/** @brief Voronoi diagram-based seam estimator.
 */
class CV_EXPORTS_W VoronoiSeamFinder : public PairwiseSeamFinder
{
public:
    CV_WRAP virtual void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                      CV_IN_OUT std::vector<UMat> &masks) CV_OVERRIDE;
    virtual void find(const std::vector<Size> &size, const std::vector<Point> &corners,
                      std::vector<UMat> &masks);
private:
    void findInPair(size_t first, size_t second, Rect roi) CV_OVERRIDE;
};


class CV_EXPORTS_W DpSeamFinder : public SeamFinder
{
public:
    enum CostFunction { COLOR, COLOR_GRAD };

    DpSeamFinder(CostFunction costFunc = COLOR);
    CV_WRAP DpSeamFinder(String costFunc );

    CostFunction costFunction() const { return costFunc_; }
    void setCostFunction(CostFunction val) { costFunc_ = val; }
    CV_WRAP void setCostFunction(String val);

    virtual void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                      std::vector<UMat> &masks) CV_OVERRIDE;

private:
    enum ComponentState
    {
        FIRST = 1, SECOND = 2, INTERS = 4,
        INTERS_FIRST = INTERS | FIRST,
        INTERS_SECOND = INTERS | SECOND
    };

    class ImagePairLess
    {
    public:
        ImagePairLess(const std::vector<Mat> &images, const std::vector<Point> &corners)
            : src_(&images[0]), corners_(&corners[0]) {}

        bool operator() (const std::pair<size_t, size_t> &l, const std::pair<size_t, size_t> &r) const
        {
            Point c1 = corners_[l.first] + Point(src_[l.first].cols / 2, src_[l.first].rows / 2);
            Point c2 = corners_[l.second] + Point(src_[l.second].cols / 2, src_[l.second].rows / 2);
            int d1 = (c1 - c2).dot(c1 - c2);

            c1 = corners_[r.first] + Point(src_[r.first].cols / 2, src_[r.first].rows / 2);
            c2 = corners_[r.second] + Point(src_[r.second].cols / 2, src_[r.second].rows / 2);
            int d2 = (c1 - c2).dot(c1 - c2);

            return d1 < d2;
        }

    private:
        const Mat *src_;
        const Point *corners_;
    };

    class ClosePoints
    {
    public:
        ClosePoints(int minDist) : minDist_(minDist) {}

        bool operator() (const Point &p1, const Point &p2) const
        {
            int dist2 = (p1.x-p2.x) * (p1.x-p2.x) + (p1.y-p2.y) * (p1.y-p2.y);
            return dist2 < minDist_ * minDist_;
        }

    private:
        int minDist_;
    };

    void process(
            const Mat &image1, const Mat &image2, Point tl1, Point tl2,  Mat &mask1, Mat &mask2);

    void findComponents();

    void findEdges();

    void resolveConflicts(
            const Mat &image1, const Mat &image2, Point tl1, Point tl2, Mat &mask1, Mat &mask2);

    void computeGradients(const Mat &image1, const Mat &image2);

    bool hasOnlyOneNeighbor(int comp);

    bool closeToContour(int y, int x, const Mat_<uchar> &contourMask);

    bool getSeamTips(int comp1, int comp2, Point &p1, Point &p2);

    void computeCosts(
            const Mat &image1, const Mat &image2, Point tl1, Point tl2,
            int comp, Mat_<float> &costV, Mat_<float> &costH);

    bool estimateSeam(
            const Mat &image1, const Mat &image2, Point tl1, Point tl2, int comp,
            Point p1, Point p2, std::vector<Point> &seam, bool &isHorizontal);

    void updateLabelsUsingSeam(
            int comp1, int comp2, const std::vector<Point> &seam, bool isHorizontalSeam);

    CostFunction costFunc_;

    // processing images pair data
    Point unionTl_, unionBr_;
    Size unionSize_;
    Mat_<uchar> mask1_, mask2_;
    Mat_<uchar> contour1mask_, contour2mask_;
    Mat_<float> gradx1_, grady1_;
    Mat_<float> gradx2_, grady2_;

    // components data
    int ncomps_;
    Mat_<int> labels_;
    std::vector<ComponentState> states_;
    std::vector<Point> tls_, brs_;
    std::vector<std::vector<Point> > contours_;
    std::set<std::pair<int, int> > edges_;
};

/** @brief Base class for all minimum graph-cut-based seam estimators.
 */
class CV_EXPORTS GraphCutSeamFinderBase
{
public:
    enum CostType { COST_COLOR, COST_COLOR_GRAD };
};

/** @brief Minimum graph cut-based seam estimator. See details in @cite V03 .
 */
class CV_EXPORTS_W GraphCutSeamFinder : public GraphCutSeamFinderBase, public SeamFinder
{
public:
    GraphCutSeamFinder(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                       float bad_region_penalty = 1000.f);
    CV_WRAP GraphCutSeamFinder(String cost_type,float terminal_cost = 10000.f,
        float bad_region_penalty = 1000.f);

    ~GraphCutSeamFinder();

    CV_WRAP void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                      CV_IN_OUT std::vector<UMat> &masks) CV_OVERRIDE;

private:
    // To avoid GCGraph dependency
    class Impl;
    Ptr<PairwiseSeamFinder> impl_;
};


#ifdef HAVE_OPENCV_CUDALEGACY
class CV_EXPORTS GraphCutSeamFinderGpu : public GraphCutSeamFinderBase, public PairwiseSeamFinder
{
public:
    GraphCutSeamFinderGpu(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                          float bad_region_penalty = 1000.f)
                          : cost_type_(cost_type), terminal_cost_(terminal_cost),
                            bad_region_penalty_(bad_region_penalty) {}

    void find(const std::vector<cv::UMat> &src, const std::vector<cv::Point> &corners,
              std::vector<cv::UMat> &masks) CV_OVERRIDE;
    void findInPair(size_t first, size_t second, Rect roi) CV_OVERRIDE;

private:
    void setGraphWeightsColor(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &mask1, const cv::Mat &mask2,
                              cv::Mat &terminals, cv::Mat &leftT, cv::Mat &rightT, cv::Mat &top, cv::Mat &bottom);
    void setGraphWeightsColorGrad(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &dx1, const cv::Mat &dx2,
                                  const cv::Mat &dy1, const cv::Mat &dy2, const cv::Mat &mask1, const cv::Mat &mask2,
                                  cv::Mat &terminals, cv::Mat &leftT, cv::Mat &rightT, cv::Mat &top, cv::Mat &bottom);
    std::vector<Mat> dx_, dy_;
    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
};
#endif

//! @}

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_SEAM_FINDERS_HPP
