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

#include "precomp.hpp"
#include <map>

namespace cv {
namespace detail {

void PairwiseSeamFinder::find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                              std::vector<Mat> &masks)
{
    LOGLN("Finding seams...");
    if (src.size() == 0)
        return;

#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    images_ = src;
    sizes_.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        sizes_[i] = src[i].size();
    corners_ = corners;
    masks_ = masks;
    run();

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


void PairwiseSeamFinder::run()
{
    for (size_t i = 0; i < sizes_.size() - 1; ++i)
    {
        for (size_t j = i + 1; j < sizes_.size(); ++j)
        {
            Rect roi;
            if (overlapRoi(corners_[i], corners_[j], sizes_[i], sizes_[j], roi))
                findInPair(i, j, roi);
        }
    }
}


void VoronoiSeamFinder::find(const std::vector<Size> &sizes, const std::vector<Point> &corners,
                             std::vector<Mat> &masks)
{
    LOGLN("Finding seams...");
    if (sizes.size() == 0)
        return;

#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    sizes_ = sizes;
    corners_ = corners;
    masks_ = masks;
    run();

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


void VoronoiSeamFinder::findInPair(size_t first, size_t second, Rect roi)
{
    const int gap = 10;
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);

    Size img1 = sizes_[first], img2 = sizes_[second];
    Mat mask1 = masks_[first], mask2 = masks_[second];
    Point tl1 = corners_[first], tl2 = corners_[second];

    // Cut submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.height && x1 < img1.width)
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
            else
                submask1.at<uchar>(y + gap, x + gap) = 0;

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.height && x2 < img2.width)
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
            else
                submask2.at<uchar>(y + gap, x + gap) = 0;
        }
    }

    Mat collision = (submask1 != 0) & (submask2 != 0);
    Mat unique1 = submask1.clone(); unique1.setTo(0, collision);
    Mat unique2 = submask2.clone(); unique2.setTo(0, collision);

    Mat dist1, dist2;
    distanceTransform(unique1 == 0, dist1, DIST_L1, 3);
    distanceTransform(unique2 == 0, dist2, DIST_L1, 3);

    Mat seam = dist1 < dist2;

    for (int y = 0; y < roi.height; ++y)
    {
        for (int x = 0; x < roi.width; ++x)
        {
            if (seam.at<uchar>(y + gap, x + gap))
                mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x) = 0;
            else
                mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x) = 0;
        }
    }
}


DpSeamFinder::DpSeamFinder(CostFunction costFunc) : costFunc_(costFunc) {}


void DpSeamFinder::find(const std::vector<Mat> &src, const std::vector<Point> &corners, std::vector<Mat> &masks)
{
    LOGLN("Finding seams...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    if (src.size() == 0)
        return;

    std::vector<std::pair<size_t, size_t> > pairs;

    for (size_t i = 0; i+1 < src.size(); ++i)
        for (size_t j = i+1; j < src.size(); ++j)
            pairs.push_back(std::make_pair(i, j));

    sort(pairs.begin(), pairs.end(), ImagePairLess(src, corners));
    std::reverse(pairs.begin(), pairs.end());

    for (size_t i = 0; i < pairs.size(); ++i)
    {
        size_t i0 = pairs[i].first, i1 = pairs[i].second;
        process(src[i0], src[i1], corners[i0], corners[i1], masks[i0], masks[i1]);
    }

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


void DpSeamFinder::process(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2,
        Mat &mask1, Mat &mask2)
{
    CV_Assert(image1.size() == mask1.size());
    CV_Assert(image2.size() == mask2.size());

    Point intersectTl(std::max(tl1.x, tl2.x), std::max(tl1.y, tl2.y));

    Point intersectBr(std::min(tl1.x + image1.cols, tl2.x + image2.cols),
                      std::min(tl1.y + image1.rows, tl2.y + image2.rows));

    if (intersectTl.x >= intersectBr.x || intersectTl.y >= intersectBr.y)
        return; // there are no conflicts

    unionTl_ = Point(std::min(tl1.x, tl2.x), std::min(tl1.y, tl2.y));

    unionBr_ = Point(std::max(tl1.x + image1.cols, tl2.x + image2.cols),
                     std::max(tl1.y + image1.rows, tl2.y + image2.rows));

    unionSize_ = Size(unionBr_.x - unionTl_.x, unionBr_.y - unionTl_.y);

    mask1_ = Mat::zeros(unionSize_, CV_8U);
    mask2_ = Mat::zeros(unionSize_, CV_8U);

    Mat tmp = mask1_(Rect(tl1.x - unionTl_.x, tl1.y - unionTl_.y, mask1.cols, mask1.rows));
    mask1.copyTo(tmp);

    tmp = mask2_(Rect(tl2.x - unionTl_.x, tl2.y - unionTl_.y, mask2.cols, mask2.rows));
    mask2.copyTo(tmp);

    // find both images contour masks

    contour1mask_ = Mat::zeros(unionSize_, CV_8U);
    contour2mask_ = Mat::zeros(unionSize_, CV_8U);

    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
            if (mask1_(y, x) &&
                ((x == 0 || !mask1_(y, x-1)) || (x == unionSize_.width-1 || !mask1_(y, x+1)) ||
                 (y == 0 || !mask1_(y-1, x)) || (y == unionSize_.height-1 || !mask1_(y+1, x))))
            {
                contour1mask_(y, x) = 255;
            }

            if (mask2_(y, x) &&
                ((x == 0 || !mask2_(y, x-1)) || (x == unionSize_.width-1 || !mask2_(y, x+1)) ||
                 (y == 0 || !mask2_(y-1, x)) || (y == unionSize_.height-1 || !mask2_(y+1, x))))
            {
                contour2mask_(y, x) = 255;
            }
        }
    }

    findComponents();

    findEdges();

    resolveConflicts(image1, image2, tl1, tl2, mask1, mask2);
}


void DpSeamFinder::findComponents()
{
    // label all connected components and get information about them

    ncomps_ = 0;
    labels_.create(unionSize_);
    states_.clear();
    tls_.clear();
    brs_.clear();
    contours_.clear();

    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
            if (mask1_(y, x) && mask2_(y, x))
                labels_(y, x) = std::numeric_limits<int>::max();
            else if (mask1_(y, x))
                labels_(y, x) = std::numeric_limits<int>::max()-1;
            else if (mask2_(y, x))
                labels_(y, x) = std::numeric_limits<int>::max()-2;
            else
                labels_(y, x) = 0;
        }
    }

    for (int y = 0; y < unionSize_.height; ++y)
    {
        for (int x = 0; x < unionSize_.width; ++x)
        {
            if (labels_(y, x) >= std::numeric_limits<int>::max()-2)
            {
                if (labels_(y, x) == std::numeric_limits<int>::max())
                    states_.push_back(INTERS);
                else if (labels_(y, x) == std::numeric_limits<int>::max()-1)
                    states_.push_back(FIRST);
                else if (labels_(y, x) == std::numeric_limits<int>::max()-2)
                    states_.push_back(SECOND);

                floodFill(labels_, Point(x, y), ++ncomps_);
                tls_.push_back(Point(x, y));
                brs_.push_back(Point(x+1, y+1));
                contours_.push_back(std::vector<Point>());
            }

            if (labels_(y, x))
            {
                int l = labels_(y, x);
                int ci = l-1;

                tls_[ci].x = std::min(tls_[ci].x, x);
                tls_[ci].y = std::min(tls_[ci].y, y);
                brs_[ci].x = std::max(brs_[ci].x, x+1);
                brs_[ci].y = std::max(brs_[ci].y, y+1);

                if ((x == 0 || labels_(y, x-1) != l) || (x == unionSize_.width-1 || labels_(y, x+1) != l) ||
                    (y == 0 || labels_(y-1, x) != l) || (y == unionSize_.height-1 || labels_(y+1, x) != l))
                {
                    contours_[ci].push_back(Point(x, y));
                }
            }
        }
    }
}


void DpSeamFinder::findEdges()
{
    // find edges between components

    std::map<std::pair<int, int>, int> wedges; // weighted edges

    for (int ci = 0; ci < ncomps_-1; ++ci)
    {
        for (int cj = ci+1; cj < ncomps_; ++cj)
        {
            wedges[std::make_pair(ci, cj)] = 0;
            wedges[std::make_pair(cj, ci)] = 0;
        }
    }

    for (int ci = 0; ci < ncomps_; ++ci)
    {
        for (size_t i = 0; i < contours_[ci].size(); ++i)
        {
            int x = contours_[ci][i].x;
            int y = contours_[ci][i].y;
            int l = ci + 1;

            if (x > 0 && labels_(y, x-1) && labels_(y, x-1) != l)
            {
                wedges[std::make_pair(ci, labels_(y, x-1)-1)]++;
                wedges[std::make_pair(labels_(y, x-1)-1, ci)]++;
            }

            if (y > 0 && labels_(y-1, x) && labels_(y-1, x) != l)
            {
                wedges[std::make_pair(ci, labels_(y-1, x)-1)]++;
                wedges[std::make_pair(labels_(y-1, x)-1, ci)]++;
            }

            if (x < unionSize_.width-1 && labels_(y, x+1) && labels_(y, x+1) != l)
            {
                wedges[std::make_pair(ci, labels_(y, x+1)-1)]++;
                wedges[std::make_pair(labels_(y, x+1)-1, ci)]++;
            }

            if (y < unionSize_.height-1 && labels_(y+1, x) && labels_(y+1, x) != l)
            {
                wedges[std::make_pair(ci, labels_(y+1, x)-1)]++;
                wedges[std::make_pair(labels_(y+1, x)-1, ci)]++;
            }
        }
    }

    edges_.clear();

    for (int ci = 0; ci < ncomps_-1; ++ci)
    {
        for (int cj = ci+1; cj < ncomps_; ++cj)
        {
            std::map<std::pair<int, int>, int>::iterator itr = wedges.find(std::make_pair(ci, cj));
            if (itr != wedges.end() && itr->second > 0)
                edges_.insert(itr->first);

            itr = wedges.find(std::make_pair(cj, ci));
            if (itr != wedges.end() && itr->second > 0)
                edges_.insert(itr->first);
        }
    }
}


void DpSeamFinder::resolveConflicts(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2, Mat &mask1, Mat &mask2)
{
    if (costFunc_ == COLOR_GRAD)
        computeGradients(image1, image2);

    // resolve conflicts between components

    bool hasConflict = true;
    while (hasConflict)
    {
        int c1 = 0, c2 = 0;
        hasConflict = false;

        for (std::set<std::pair<int, int> >::iterator itr = edges_.begin(); itr != edges_.end(); ++itr)
        {
            c1 = itr->first;
            c2 = itr->second;

            if ((states_[c1] & INTERS) && (states_[c1] & (~INTERS)) != states_[c2])
            {
                hasConflict = true;
                break;
            }
        }

        if (hasConflict)
        {
            int l1 = c1+1, l2 = c2+1;

            if (hasOnlyOneNeighbor(c1))
            {
                // if the first components has only one adjacent component

                for (int y = tls_[c1].y; y < brs_[c1].y; ++y)
                    for (int x = tls_[c1].x; x < brs_[c1].x; ++x)
                        if (labels_(y, x) == l1)
                            labels_(y, x) = l2;

                states_[c1] = states_[c2] == FIRST ? SECOND : FIRST;
            }
            else
            {
                // if the first component has more than one adjacent component

                Point p1, p2;
                if (getSeamTips(c1, c2, p1, p2))
                {
                    std::vector<Point> seam;
                    bool isHorizontalSeam;

                    if (estimateSeam(image1, image2, tl1, tl2, c1, p1, p2, seam, isHorizontalSeam))
                        updateLabelsUsingSeam(c1, c2, seam, isHorizontalSeam);
                }

                states_[c1] = states_[c2] == FIRST ? INTERS_SECOND : INTERS_FIRST;
            }

            const int c[] = {c1, c2};
            const int l[] = {l1, l2};

            for (int i = 0; i < 2; ++i)
            {
                // update information about the (i+1)-th component

                int x0 = tls_[c[i]].x, x1 = brs_[c[i]].x;
                int y0 = tls_[c[i]].y, y1 = brs_[c[i]].y;

                tls_[c[i]] = Point(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
                brs_[c[i]] = Point(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());
                contours_[c[i]].clear();

                for (int y = y0; y < y1; ++y)
                {
                    for (int x = x0; x < x1; ++x)
                    {
                        if (labels_(y, x) == l[i])
                        {
                            tls_[c[i]].x = std::min(tls_[c[i]].x, x);
                            tls_[c[i]].y = std::min(tls_[c[i]].y, y);
                            brs_[c[i]].x = std::max(brs_[c[i]].x, x+1);
                            brs_[c[i]].y = std::max(brs_[c[i]].y, y+1);

                            if ((x == 0 || labels_(y, x-1) != l[i]) || (x == unionSize_.width-1 || labels_(y, x+1) != l[i]) ||
                                (y == 0 || labels_(y-1, x) != l[i]) || (y == unionSize_.height-1 || labels_(y+1, x) != l[i]))
                            {
                                contours_[c[i]].push_back(Point(x, y));
                            }
                        }
                    }
                }
            }

            // remove edges

            edges_.erase(std::make_pair(c1, c2));
            edges_.erase(std::make_pair(c2, c1));
        }
    }

    // update masks

    int dx1 = unionTl_.x - tl1.x, dy1 = unionTl_.y - tl1.y;
    int dx2 = unionTl_.x - tl2.x, dy2 = unionTl_.y - tl2.y;

    for (int y = 0; y < mask2.rows; ++y)
    {
        for (int x = 0; x < mask2.cols; ++x)
        {
             int l = labels_(y - dy2, x - dx2);
             if (l > 0 && (states_[l-1] & FIRST) && mask1.at<uchar>(y - dy2 + dy1, x - dx2 + dx1))
                mask2.at<uchar>(y, x) = 0;
        }
    }

    for (int y = 0; y < mask1.rows; ++y)
    {
        for (int x = 0; x < mask1.cols; ++x)
        {
             int l = labels_(y - dy1, x - dx1);
             if (l > 0 && (states_[l-1] & SECOND) && mask2.at<uchar>(y - dy1 + dy2, x - dx1 + dx2))
                mask1.at<uchar>(y, x) = 0;
        }
    }
}


void DpSeamFinder::computeGradients(const Mat &image1, const Mat &image2)
{
    CV_Assert(image1.channels() == 3 || image1.channels() == 4);
    CV_Assert(image2.channels() == 3 || image2.channels() == 4);
    CV_Assert(costFunction() == COLOR_GRAD);

    Mat gray;

    if (image1.channels() == 3)
        cvtColor(image1, gray, COLOR_BGR2GRAY);
    else if (image1.channels() == 4)
        cvtColor(image1, gray, COLOR_BGRA2GRAY);

    Sobel(gray, gradx1_, CV_32F, 1, 0);
    Sobel(gray, grady1_, CV_32F, 0, 1);

    if (image2.channels() == 3)
        cvtColor(image2, gray, COLOR_BGR2GRAY);
    else if (image2.channels() == 4)
        cvtColor(image2, gray, COLOR_BGRA2GRAY);

    Sobel(gray, gradx2_, CV_32F, 1, 0);
    Sobel(gray, grady2_, CV_32F, 0, 1);
}


bool DpSeamFinder::hasOnlyOneNeighbor(int comp)
{
    std::set<std::pair<int, int> >::iterator begin, end;
    begin = lower_bound(edges_.begin(), edges_.end(), std::make_pair(comp, std::numeric_limits<int>::min()));
    end = upper_bound(edges_.begin(), edges_.end(), std::make_pair(comp, std::numeric_limits<int>::max()));
    return ++begin == end;
}


bool DpSeamFinder::closeToContour(int y, int x, const Mat_<uchar> &contourMask)
{
    const int rad = 2;

    for (int dy = -rad; dy <= rad; ++dy)
    {
        if (y + dy >= 0 && y + dy < unionSize_.height)
        {
            for (int dx = -rad; dx <= rad; ++dx)
            {
                if (x + dx >= 0 && x + dx < unionSize_.width &&
                    contourMask(y + dy, x + dx))
                {
                    return true;
                }
            }
        }
    }

    return false;
}


bool DpSeamFinder::getSeamTips(int comp1, int comp2, Point &p1, Point &p2)
{
    CV_Assert(states_[comp1] & INTERS);

    // find special points

    std::vector<Point> specialPoints;
    int l2 = comp2+1;

    for (size_t i = 0; i < contours_[comp1].size(); ++i)
    {
        int x = contours_[comp1][i].x;
        int y = contours_[comp1][i].y;

        if (closeToContour(y, x, contour1mask_) &&
            closeToContour(y, x, contour2mask_) &&
            ((x > 0 && labels_(y, x-1) == l2) ||
             (y > 0 && labels_(y-1, x) == l2) ||
             (x < unionSize_.width-1 && labels_(y, x+1) == l2) ||
             (y < unionSize_.height-1 && labels_(y+1, x) == l2)))
        {
            specialPoints.push_back(Point(x, y));
        }
    }

    if (specialPoints.size() < 2)
        return false;

    // find clusters

    std::vector<int> labels;
    cv::partition(specialPoints, labels, ClosePoints(10));

    int nlabels = *std::max_element(labels.begin(), labels.end()) + 1;
    if (nlabels < 2)
        return false;

    std::vector<Point> sum(nlabels);
    std::vector<std::vector<Point> > points(nlabels);

    for (size_t i = 0; i < specialPoints.size(); ++i)
    {
        sum[labels[i]] += specialPoints[i];
        points[labels[i]].push_back(specialPoints[i]);
    }

    // select two most distant clusters

    int idx[2] = {-1,-1};
    double maxDist = -std::numeric_limits<double>::max();

    for (int i = 0; i < nlabels-1; ++i)
    {
        for (int j = i+1; j < nlabels; ++j)
        {
            double size1 = static_cast<double>(points[i].size()), size2 = static_cast<double>(points[j].size());
            double cx1 = cvRound(sum[i].x / size1), cy1 = cvRound(sum[i].y / size1);
            double cx2 = cvRound(sum[j].x / size2), cy2 = cvRound(sum[j].y / size1);

            double dist = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
            if (dist > maxDist)
            {
                maxDist = dist;
                idx[0] = i;
                idx[1] = j;
            }
        }
    }

    // select two points closest to the clusters' centers

    Point p[2];

    for (int i = 0; i < 2; ++i)
    {
        double size = static_cast<double>(points[idx[i]].size());
        double cx = cvRound(sum[idx[i]].x / size);
        double cy = cvRound(sum[idx[i]].y / size);

        size_t closest = points[idx[i]].size();
        double minDist = std::numeric_limits<double>::max();

        for (size_t j = 0; j < points[idx[i]].size(); ++j)
        {
            double dist = (points[idx[i]][j].x - cx) * (points[idx[i]][j].x - cx) +
                          (points[idx[i]][j].y - cy) * (points[idx[i]][j].y - cy);
            if (dist < minDist)
            {
                minDist = dist;
                closest = j;
            }
        }

        p[i] = points[idx[i]][closest];
    }

    p1 = p[0];
    p2 = p[1];
    return true;
}


namespace
{

template <typename T>
float diffL2Square3(const Mat &image1, int y1, int x1, const Mat &image2, int y2, int x2)
{
    const T *r1 = image1.ptr<T>(y1);
    const T *r2 = image2.ptr<T>(y2);
    return static_cast<float>(sqr(r1[3*x1] - r2[3*x2]) + sqr(r1[3*x1+1] - r2[3*x2+1]) +
                              sqr(r1[3*x1+2] - r2[3*x2+2]));
}


template <typename T>
float diffL2Square4(const Mat &image1, int y1, int x1, const Mat &image2, int y2, int x2)
{
    const T *r1 = image1.ptr<T>(y1);
    const T *r2 = image2.ptr<T>(y2);
    return static_cast<float>(sqr(r1[4*x1] - r2[4*x2]) + sqr(r1[4*x1+1] - r2[4*x2+1]) +
                              sqr(r1[4*x1+2] - r2[4*x2+2]));
}

} // namespace


void DpSeamFinder::computeCosts(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2,
        int comp, Mat_<float> &costV, Mat_<float> &costH)
{
    CV_Assert(states_[comp] & INTERS);

    // compute costs

    float (*diff)(const Mat&, int, int, const Mat&, int, int) = 0;
    if (image1.type() == CV_32FC3 && image2.type() == CV_32FC3)
        diff = diffL2Square3<float>;
    else if (image1.type() == CV_8UC3 && image2.type() == CV_8UC3)
        diff = diffL2Square3<uchar>;
    else if (image1.type() == CV_32FC4 && image2.type() == CV_32FC4)
        diff = diffL2Square4<float>;
    else if (image1.type() == CV_8UC4 && image2.type() == CV_8UC4)
        diff = diffL2Square4<uchar>;
    else
        CV_Error(Error::StsBadArg, "both images must have CV_32FC3(4) or CV_8UC3(4) type");

    int l = comp+1;
    Rect roi(tls_[comp], brs_[comp]);

    int dx1 = unionTl_.x - tl1.x, dy1 = unionTl_.y - tl1.y;
    int dx2 = unionTl_.x - tl2.x, dy2 = unionTl_.y - tl2.y;

    const float badRegionCost = normL2(Point3f(255.f, 255.f, 255.f),
                                       Point3f(0.f, 0.f, 0.f));

    costV.create(roi.height, roi.width+1);

    for (int y = roi.y; y < roi.br().y; ++y)
    {
        for (int x = roi.x; x < roi.br().x+1; ++x)
        {
            if (labels_(y, x) == l && x > 0 && labels_(y, x-1) == l)
            {
                float costColor = (diff(image1, y + dy1, x + dx1 - 1, image2, y + dy2, x + dx2) +
                                   diff(image1, y + dy1, x + dx1, image2, y + dy2, x + dx2 - 1)) / 2;
                if (costFunc_ == COLOR)
                    costV(y - roi.y, x - roi.x) = costColor;
                else if (costFunc_ == COLOR_GRAD)
                {
                    float costGrad = std::abs(gradx1_(y + dy1, x + dx1)) + std::abs(gradx1_(y + dy1, x + dx1 - 1)) +
                                     std::abs(gradx2_(y + dy2, x + dx2)) + std::abs(gradx2_(y + dy2, x + dx2 - 1)) + 1.f;
                    costV(y - roi.y, x - roi.x) = costColor / costGrad;
                }
            }
            else
                costV(y - roi.y, x - roi.x) = badRegionCost;
        }
    }

    costH.create(roi.height+1, roi.width);

    for (int y = roi.y; y < roi.br().y+1; ++y)
    {
        for (int x = roi.x; x < roi.br().x; ++x)
        {
            if (labels_(y, x) == l && y > 0 && labels_(y-1, x) == l)
            {
                float costColor = (diff(image1, y + dy1 - 1, x + dx1, image2, y + dy2, x + dx2) +
                                   diff(image1, y + dy1, x + dx1, image2, y + dy2 - 1, x + dx2)) / 2;
                if (costFunc_ == COLOR)
                    costH(y - roi.y, x - roi.x) = costColor;
                else if (costFunc_ == COLOR_GRAD)
                {
                    float costGrad = std::abs(grady1_(y + dy1, x + dx1)) + std::abs(grady1_(y + dy1 - 1, x + dx1)) +
                                     std::abs(grady2_(y + dy2, x + dx2)) + std::abs(grady2_(y + dy2 - 1, x + dx2)) + 1.f;
                    costH(y - roi.y, x - roi.x) = costColor / costGrad;
                }
            }
            else
                costH(y - roi.y, x - roi.x) = badRegionCost;
        }
    }
}


bool DpSeamFinder::estimateSeam(
        const Mat &image1, const Mat &image2, Point tl1, Point tl2, int comp,
        Point p1, Point p2, std::vector<Point> &seam, bool &isHorizontal)
{
    CV_Assert(states_[comp] & INTERS);

    Mat_<float> costV, costH;
    computeCosts(image1, image2, tl1, tl2, comp, costV, costH);

    Rect roi(tls_[comp], brs_[comp]);
    Point src = p1 - roi.tl();
    Point dst = p2 - roi.tl();
    int l = comp+1;

    // estimate seam direction

    bool swapped = false;
    isHorizontal = std::abs(dst.x - src.x) > std::abs(dst.y - src.y);

    if (isHorizontal)
    {
        if (src.x > dst.x)
        {
            std::swap(src, dst);
            swapped = true;
        }
    }
    else if (src.y > dst.y)
    {
        swapped = true;
        std::swap(src, dst);
    }

    // find optimal control

    Mat_<uchar> control = Mat::zeros(roi.size(), CV_8U);
    Mat_<uchar> reachable = Mat::zeros(roi.size(), CV_8U);
    Mat_<float> cost = Mat::zeros(roi.size(), CV_32F);

    reachable(src) = 1;
    cost(src) = 0.f;

    int nsteps;
    std::pair<float, int> steps[3];

    if (isHorizontal)
    {
        for (int x = src.x+1; x <= dst.x; ++x)
        {
            for (int y = 0; y < roi.height; ++y)
            {
                // seam follows along upper side of pixels

                nsteps = 0;

                if (labels_(y + roi.y, x + roi.x) == l)
                {
                    if (reachable(y, x-1))
                        steps[nsteps++] = std::make_pair(cost(y, x-1) + costH(y, x-1), 1);
                    if (y > 0 && reachable(y-1, x-1))
                        steps[nsteps++] = std::make_pair(cost(y-1, x-1) + costH(y-1, x-1) + costV(y-1, x), 2);
                    if (y < roi.height-1 && reachable(y+1, x-1))
                        steps[nsteps++] = std::make_pair(cost(y+1, x-1) + costH(y+1, x-1) + costV(y, x), 3);
                }

                if (nsteps)
                {
                    std::pair<float, int> opt = *min_element(steps, steps + nsteps);
                    cost(y, x) = opt.first;
                    control(y, x) = (uchar)opt.second;
                    reachable(y, x) = 255;
                }
            }
        }
    }
    else
    {
        for (int y = src.y+1; y <= dst.y; ++y)
        {
            for (int x = 0; x < roi.width; ++x)
            {
                // seam follows along left side of pixels

                nsteps = 0;

                if (labels_(y + roi.y, x + roi.x) == l)
                {
                    if (reachable(y-1, x))
                        steps[nsteps++] = std::make_pair(cost(y-1, x) + costV(y-1, x), 1);
                    if (x > 0 && reachable(y-1, x-1))
                        steps[nsteps++] = std::make_pair(cost(y-1, x-1) + costV(y-1, x-1) + costH(y, x-1), 2);
                    if (x < roi.width-1 && reachable(y-1, x+1))
                        steps[nsteps++] = std::make_pair(cost(y-1, x+1) + costV(y-1, x+1) + costH(y, x), 3);
                }

                if (nsteps)
                {
                    std::pair<float, int> opt = *min_element(steps, steps + nsteps);
                    cost(y, x) = opt.first;
                    control(y, x) = (uchar)opt.second;
                    reachable(y, x) = 255;
                }
            }
        }
    }

    if (!reachable(dst))
        return false;

    // restore seam

    Point p = dst;
    seam.clear();
    seam.push_back(p + roi.tl());

    if (isHorizontal)
    {
        for (; p.x != src.x; seam.push_back(p + roi.tl()))
        {
            if (control(p) == 2) p.y--;
            else if (control(p) == 3) p.y++;
            p.x--;
        }
    }
    else
    {
        for (; p.y != src.y; seam.push_back(p + roi.tl()))
        {
            if (control(p) == 2) p.x--;
            else if (control(p) == 3) p.x++;
            p.y--;
        }
    }

    if (!swapped)
        std::reverse(seam.begin(), seam.end());

    CV_Assert(seam.front() == p1);
    CV_Assert(seam.back() == p2);
    return true;
}


void DpSeamFinder::updateLabelsUsingSeam(
        int comp1, int comp2, const std::vector<Point> &seam, bool isHorizontalSeam)
{
    Mat_<int> mask = Mat::zeros(brs_[comp1].y - tls_[comp1].y,
                                brs_[comp1].x - tls_[comp1].x, CV_32S);

    for (size_t i = 0; i < contours_[comp1].size(); ++i)
        mask(contours_[comp1][i] - tls_[comp1]) = 255;

    for (size_t i = 0; i < seam.size(); ++i)
        mask(seam[i] - tls_[comp1]) = 255;

    // find connected components after seam carving

    int l1 = comp1+1, l2 = comp2+1;

    int ncomps = 0;

    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            if (!mask(y, x) && labels_(y + tls_[comp1].y, x + tls_[comp1].x) == l1)
                floodFill(mask, Point(x, y), ++ncomps);

    for (size_t i = 0; i < contours_[comp1].size(); ++i)
    {
        int x = contours_[comp1][i].x - tls_[comp1].x;
        int y = contours_[comp1][i].y - tls_[comp1].y;

        bool ok = false;
        static const int dx[] = {-1, +1, 0, 0, -1, +1, -1, +1};
        static const int dy[] = {0, 0, -1, +1, -1, -1, +1, +1};

        for (int j = 0; j < 8; ++j)
        {
            int c = x + dx[j];
            int r = y + dy[j];

            if (c >= 0 && c < mask.cols && r >= 0 && r < mask.rows &&
                mask(r, c) && mask(r, c) != 255)
            {
                ok = true;
                mask(y, x) = mask(r, c);
            }
        }

        if (!ok)
            mask(y, x) = 0;
    }

    if (isHorizontalSeam)
    {
        for (size_t i = 0; i < seam.size(); ++i)
        {
            int x = seam[i].x - tls_[comp1].x;
            int y = seam[i].y - tls_[comp1].y;

            if (y < mask.rows-1 && mask(y+1, x) && mask(y+1, x) != 255)
                mask(y, x) = mask(y+1, x);
            else
                mask(y, x) = 0;
        }
    }
    else
    {
        for (size_t i = 0; i < seam.size(); ++i)
        {
            int x = seam[i].x - tls_[comp1].x;
            int y = seam[i].y - tls_[comp1].y;

            if (x < mask.cols-1 && mask(y, x+1) && mask(y, x+1) != 255)
                mask(y, x) = mask(y, x+1);
            else
                mask(y, x) = 0;
        }
    }

    // find new components connected with the second component and
    // with other components except the ones we are working with

    std::map<int, int> connect2;
    std::map<int, int> connectOther;

    for (int i = 1; i <= ncomps; ++i)
    {
        connect2.insert(std::make_pair(i, 0));
        connectOther.insert(std::make_pair(i, 0));
    }

    for (size_t i = 0; i < contours_[comp1].size(); ++i)
    {
        int x = contours_[comp1][i].x;
        int y = contours_[comp1][i].y;

        if ((x > 0 && labels_(y, x-1) == l2) ||
            (y > 0 && labels_(y-1, x) == l2) ||
            (x < unionSize_.width-1 && labels_(y, x+1) == l2) ||
            (y < unionSize_.height-1 && labels_(y+1, x) == l2))
        {
            connect2[mask(y - tls_[comp1].y, x - tls_[comp1].x)]++;
        }

        if ((x > 0 && labels_(y, x-1) != l1 && labels_(y, x-1) != l2) ||
            (y > 0 && labels_(y-1, x) != l1 && labels_(y-1, x) != l2) ||
            (x < unionSize_.width-1 && labels_(y, x+1) != l1 && labels_(y, x+1) != l2) ||
            (y < unionSize_.height-1 && labels_(y+1, x) != l1 && labels_(y+1, x) != l2))
        {
            connectOther[mask(y - tls_[comp1].y, x - tls_[comp1].x)]++;
        }
    }

    std::vector<int> isAdjComp(ncomps + 1, 0);

    for (std::map<int, int>::iterator itr = connect2.begin(); itr != connect2.end(); ++itr)
    {
        double len = static_cast<double>(contours_[comp1].size());
        isAdjComp[itr->first] = itr->second / len > 0.05 && connectOther.find(itr->first)->second / len < 0.1;
    }

    // update labels

    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            if (mask(y, x) && isAdjComp[mask(y, x)])
                labels_(y + tls_[comp1].y, x + tls_[comp1].x) = l2;
}


class GraphCutSeamFinder::Impl : public PairwiseSeamFinder
{
public:
    Impl(int cost_type, float terminal_cost, float bad_region_penalty)
        : cost_type_(cost_type), terminal_cost_(terminal_cost), bad_region_penalty_(bad_region_penalty) {}

    ~Impl() {}

    void find(const std::vector<Mat> &src, const std::vector<Point> &corners, std::vector<Mat> &masks);
    void findInPair(size_t first, size_t second, Rect roi);

private:
    void setGraphWeightsColor(const Mat &img1, const Mat &img2,
                              const Mat &mask1, const Mat &mask2, GCGraph<float> &graph);
    void setGraphWeightsColorGrad(const Mat &img1, const Mat &img2, const Mat &dx1, const Mat &dx2,
                                  const Mat &dy1, const Mat &dy2, const Mat &mask1, const Mat &mask2,
                                  GCGraph<float> &graph);

    std::vector<Mat> dx_, dy_;
    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
};


void GraphCutSeamFinder::Impl::find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                                    std::vector<Mat> &masks)
{
    // Compute gradients
    dx_.resize(src.size());
    dy_.resize(src.size());
    Mat dx, dy;
    for (size_t i = 0; i < src.size(); ++i)
    {
        CV_Assert(src[i].channels() == 3);
        Sobel(src[i], dx, CV_32F, 1, 0);
        Sobel(src[i], dy, CV_32F, 0, 1);
        dx_[i].create(src[i].size(), CV_32F);
        dy_[i].create(src[i].size(), CV_32F);
        for (int y = 0; y < src[i].rows; ++y)
        {
            const Point3f* dx_row = dx.ptr<Point3f>(y);
            const Point3f* dy_row = dy.ptr<Point3f>(y);
            float* dx_row_ = dx_[i].ptr<float>(y);
            float* dy_row_ = dy_[i].ptr<float>(y);
            for (int x = 0; x < src[i].cols; ++x)
            {
                dx_row_[x] = normL2(dx_row[x]);
                dy_row_[x] = normL2(dy_row[x]);
            }
        }
    }
    PairwiseSeamFinder::find(src, corners, masks);
}


void GraphCutSeamFinder::Impl::setGraphWeightsColor(const Mat &img1, const Mat &img2,
                                                    const Mat &mask1, const Mat &mask2, GCGraph<float> &graph)
{
    const Size img_size = img1.size();

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f,
                                    mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = y * img_size.width + x;
            if (x < img_size.width - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < img_size.height - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + img_size.width, weight, weight);
            }
        }
    }
}


void GraphCutSeamFinder::Impl::setGraphWeightsColorGrad(
        const Mat &img1, const Mat &img2, const Mat &dx1, const Mat &dx2,
        const Mat &dy1, const Mat &dy2, const Mat &mask1, const Mat &mask2,
        GCGraph<float> &graph)
{
    const Size img_size = img1.size();

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = graph.addVtx();
            graph.addTermWeights(v, mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f,
                                    mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            int v = y * img_size.width + x;
            if (x < img_size.width - 1)
            {
                float grad = dx1.at<float>(y, x) + dx1.at<float>(y, x + 1) +
                             dx2.at<float>(y, x) + dx2.at<float>(y, x + 1) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + 1, weight, weight);
            }
            if (y < img_size.height - 1)
            {
                float grad = dy1.at<float>(y, x) + dy1.at<float>(y + 1, x) +
                             dy2.at<float>(y, x) + dy2.at<float>(y + 1, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                graph.addEdges(v, v + img_size.width, weight, weight);
            }
        }
    }
}


void GraphCutSeamFinder::Impl::findInPair(size_t first, size_t second, Rect roi)
{
    Mat img1 = images_[first], img2 = images_[second];
    Mat dx1 = dx_[first], dx2 = dx_[second];
    Mat dy1 = dy_[first], dy2 = dy_[second];
    Mat mask1 = masks_[first], mask2 = masks_[second];
    Point tl1 = corners_[first], tl2 = corners_[second];

    const int gap = 10;
    Mat subimg1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat subimg2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat subdx1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdx2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);

    // Cut subimages and submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.rows && x1 < img1.cols)
            {
                subimg1.at<Point3f>(y + gap, x + gap) = img1.at<Point3f>(y1, x1);
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
                subdx1.at<float>(y + gap, x + gap) = dx1.at<float>(y1, x1);
                subdy1.at<float>(y + gap, x + gap) = dy1.at<float>(y1, x1);
            }
            else
            {
                subimg1.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask1.at<uchar>(y + gap, x + gap) = 0;
                subdx1.at<float>(y + gap, x + gap) = 0.f;
                subdy1.at<float>(y + gap, x + gap) = 0.f;
            }

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.rows && x2 < img2.cols)
            {
                subimg2.at<Point3f>(y + gap, x + gap) = img2.at<Point3f>(y2, x2);
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
                subdx2.at<float>(y + gap, x + gap) = dx2.at<float>(y2, x2);
                subdy2.at<float>(y + gap, x + gap) = dy2.at<float>(y2, x2);
            }
            else
            {
                subimg2.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask2.at<uchar>(y + gap, x + gap) = 0;
                subdx2.at<float>(y + gap, x + gap) = 0.f;
                subdy2.at<float>(y + gap, x + gap) = 0.f;
            }
        }
    }

    const int vertex_count = (roi.height + 2 * gap) * (roi.width + 2 * gap);
    const int edge_count = (roi.height - 1 + 2 * gap) * (roi.width + 2 * gap) +
                           (roi.width - 1 + 2 * gap) * (roi.height + 2 * gap);
    GCGraph<float> graph(vertex_count, edge_count);

    switch (cost_type_)
    {
    case GraphCutSeamFinder::COST_COLOR:
        setGraphWeightsColor(subimg1, subimg2, submask1, submask2, graph);
        break;
    case GraphCutSeamFinder::COST_COLOR_GRAD:
        setGraphWeightsColorGrad(subimg1, subimg2, subdx1, subdx2, subdy1, subdy2,
                                 submask1, submask2, graph);
        break;
    default:
        CV_Error(Error::StsBadArg, "unsupported pixel similarity measure");
    }

    graph.maxFlow();

    for (int y = 0; y < roi.height; ++y)
    {
        for (int x = 0; x < roi.width; ++x)
        {
            if (graph.inSourceSegment((y + gap) * (roi.width + 2 * gap) + x + gap))
            {
                if (mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x))
                    mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x) = 0;
            }
            else
            {
                if (mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x))
                    mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x) = 0;
            }
        }
    }
}


GraphCutSeamFinder::GraphCutSeamFinder(int cost_type, float terminal_cost, float bad_region_penalty)
    : impl_(new Impl(cost_type, terminal_cost, bad_region_penalty)) {}

GraphCutSeamFinder::~GraphCutSeamFinder() {}


void GraphCutSeamFinder::find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                              std::vector<Mat> &masks)
{
    impl_->find(src, corners, masks);
}


#ifdef HAVE_OPENCV_CUDA
void GraphCutSeamFinderGpu::find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                                 std::vector<Mat> &masks)
{
    // Compute gradients
    dx_.resize(src.size());
    dy_.resize(src.size());
    Mat dx, dy;
    for (size_t i = 0; i < src.size(); ++i)
    {
        CV_Assert(src[i].channels() == 3);
        Sobel(src[i], dx, CV_32F, 1, 0);
        Sobel(src[i], dy, CV_32F, 0, 1);
        dx_[i].create(src[i].size(), CV_32F);
        dy_[i].create(src[i].size(), CV_32F);
        for (int y = 0; y < src[i].rows; ++y)
        {
            const Point3f* dx_row = dx.ptr<Point3f>(y);
            const Point3f* dy_row = dy.ptr<Point3f>(y);
            float* dx_row_ = dx_[i].ptr<float>(y);
            float* dy_row_ = dy_[i].ptr<float>(y);
            for (int x = 0; x < src[i].cols; ++x)
            {
                dx_row_[x] = normL2(dx_row[x]);
                dy_row_[x] = normL2(dy_row[x]);
            }
        }
    }
    PairwiseSeamFinder::find(src, corners, masks);
}


void GraphCutSeamFinderGpu::findInPair(size_t first, size_t second, Rect roi)
{
    Mat img1 = images_[first], img2 = images_[second];
    Mat dx1 = dx_[first], dx2 = dx_[second];
    Mat dy1 = dy_[first], dy2 = dy_[second];
    Mat mask1 = masks_[first], mask2 = masks_[second];
    Point tl1 = corners_[first], tl2 = corners_[second];

    const int gap = 10;
    Mat subimg1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat subimg2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat subdx1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdx2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);
    Mat subdy2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32F);

    // Cut subimages and submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.rows && x1 < img1.cols)
            {
                subimg1.at<Point3f>(y + gap, x + gap) = img1.at<Point3f>(y1, x1);
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
                subdx1.at<float>(y + gap, x + gap) = dx1.at<float>(y1, x1);
                subdy1.at<float>(y + gap, x + gap) = dy1.at<float>(y1, x1);
            }
            else
            {
                subimg1.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask1.at<uchar>(y + gap, x + gap) = 0;
                subdx1.at<float>(y + gap, x + gap) = 0.f;
                subdy1.at<float>(y + gap, x + gap) = 0.f;
            }

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.rows && x2 < img2.cols)
            {
                subimg2.at<Point3f>(y + gap, x + gap) = img2.at<Point3f>(y2, x2);
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
                subdx2.at<float>(y + gap, x + gap) = dx2.at<float>(y2, x2);
                subdy2.at<float>(y + gap, x + gap) = dy2.at<float>(y2, x2);
            }
            else
            {
                subimg2.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask2.at<uchar>(y + gap, x + gap) = 0;
                subdx2.at<float>(y + gap, x + gap) = 0.f;
                subdy2.at<float>(y + gap, x + gap) = 0.f;
            }
        }
    }

    Mat terminals, leftT, rightT, top, bottom;

    switch (cost_type_)
    {
    case GraphCutSeamFinder::COST_COLOR:
        setGraphWeightsColor(subimg1, subimg2, submask1, submask2,
                             terminals, leftT, rightT, top, bottom);
        break;
    case GraphCutSeamFinder::COST_COLOR_GRAD:
        setGraphWeightsColorGrad(subimg1, subimg2, subdx1, subdx2, subdy1, subdy2,
                                 submask1, submask2, terminals, leftT, rightT, top, bottom);
        break;
    default:
        CV_Error(Error::StsBadArg, "unsupported pixel similarity measure");
    }

    cuda::GpuMat terminals_d(terminals);
    cuda::GpuMat leftT_d(leftT);
    cuda::GpuMat rightT_d(rightT);
    cuda::GpuMat top_d(top);
    cuda::GpuMat bottom_d(bottom);
    cuda::GpuMat labels_d, buf_d;

    cuda::graphcut(terminals_d, leftT_d, rightT_d, top_d, bottom_d, labels_d, buf_d);

    Mat_<uchar> labels = (Mat)labels_d;
    for (int y = 0; y < roi.height; ++y)
    {
        for (int x = 0; x < roi.width; ++x)
        {
            if (labels(y + gap, x + gap))
            {
                if (mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x))
                    mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x) = 0;
            }
            else
            {
                if (mask2.at<uchar>(roi.y - tl2.y + y, roi.x - tl2.x + x))
                    mask1.at<uchar>(roi.y - tl1.y + y, roi.x - tl1.x + x) = 0;
            }
        }
    }
}


void GraphCutSeamFinderGpu::setGraphWeightsColor(const Mat &img1, const Mat &img2, const Mat &mask1, const Mat &mask2,
                                                 Mat &terminals, Mat &leftT, Mat &rightT, Mat &top, Mat &bottom)
{
    const Size img_size = img1.size();

    terminals.create(img_size, CV_32S);
    leftT.create(Size(img_size.height, img_size.width), CV_32S);
    rightT.create(Size(img_size.height, img_size.width), CV_32S);
    top.create(img_size, CV_32S);
    bottom.create(img_size, CV_32S);

    Mat_<int> terminals_(terminals);
    Mat_<int> leftT_(leftT);
    Mat_<int> rightT_(rightT);
    Mat_<int> top_(top);
    Mat_<int> bottom_(bottom);

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            float source = mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            float sink = mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            terminals_(y, x) = saturate_cast<int>((source - sink) * 255.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            if (x > 0)
            {
                float weight = normL2(img1.at<Point3f>(y, x - 1), img2.at<Point3f>(y, x - 1)) +
                               normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x - 1) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y, x - 1) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                leftT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                leftT_(x, y) = 0;

            if (x < img_size.width - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                rightT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                rightT_(x, y) = 0;

            if (y > 0)
            {
                float weight = normL2(img1.at<Point3f>(y - 1, x), img2.at<Point3f>(y - 1, x)) +
                               normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y - 1, x) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y - 1, x) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                top_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                top_(y, x) = 0;

            if (y < img_size.height - 1)
            {
                float weight = normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                               normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x)) +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                bottom_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                bottom_(y, x) = 0;
        }
    }
}


void GraphCutSeamFinderGpu::setGraphWeightsColorGrad(
        const Mat &img1, const Mat &img2, const Mat &dx1, const Mat &dx2,
        const Mat &dy1, const Mat &dy2, const Mat &mask1, const Mat &mask2,
        Mat &terminals, Mat &leftT, Mat &rightT, Mat &top, Mat &bottom)
{
    const Size img_size = img1.size();

    terminals.create(img_size, CV_32S);
    leftT.create(Size(img_size.height, img_size.width), CV_32S);
    rightT.create(Size(img_size.height, img_size.width), CV_32S);
    top.create(img_size, CV_32S);
    bottom.create(img_size, CV_32S);

    Mat_<int> terminals_(terminals);
    Mat_<int> leftT_(leftT);
    Mat_<int> rightT_(rightT);
    Mat_<int> top_(top);
    Mat_<int> bottom_(bottom);

    // Set terminal weights
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            float source = mask1.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            float sink = mask2.at<uchar>(y, x) ? terminal_cost_ : 0.f;
            terminals_(y, x) = saturate_cast<int>((source - sink) * 255.f);
        }
    }

    // Set regular edge weights
    const float weight_eps = 1.f;
    for (int y = 0; y < img_size.height; ++y)
    {
        for (int x = 0; x < img_size.width; ++x)
        {
            if (x > 0)
            {
                float grad = dx1.at<float>(y, x - 1) + dx1.at<float>(y, x) +
                             dx2.at<float>(y, x - 1) + dx2.at<float>(y, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x - 1), img2.at<Point3f>(y, x - 1)) +
                                normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x - 1) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y, x - 1) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                leftT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                leftT_(x, y) = 0;

            if (x < img_size.width - 1)
            {
                float grad = dx1.at<float>(y, x) + dx1.at<float>(y, x + 1) +
                             dx2.at<float>(y, x) + dx2.at<float>(y, x + 1) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y, x + 1), img2.at<Point3f>(y, x + 1))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y, x + 1) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y, x + 1))
                    weight += bad_region_penalty_;
                rightT_(x, y) = saturate_cast<int>(weight * 255.f);
            }
            else
                rightT_(x, y) = 0;

            if (y > 0)
            {
                float grad = dy1.at<float>(y - 1, x) + dy1.at<float>(y, x) +
                             dy2.at<float>(y - 1, x) + dy2.at<float>(y, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y - 1, x), img2.at<Point3f>(y - 1, x)) +
                                normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y - 1, x) || !mask1.at<uchar>(y, x) ||
                    !mask2.at<uchar>(y - 1, x) || !mask2.at<uchar>(y, x))
                    weight += bad_region_penalty_;
                top_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                top_(y, x) = 0;

            if (y < img_size.height - 1)
            {
                float grad = dy1.at<float>(y, x) + dy1.at<float>(y + 1, x) +
                             dy2.at<float>(y, x) + dy2.at<float>(y + 1, x) + weight_eps;
                float weight = (normL2(img1.at<Point3f>(y, x), img2.at<Point3f>(y, x)) +
                                normL2(img1.at<Point3f>(y + 1, x), img2.at<Point3f>(y + 1, x))) / grad +
                               weight_eps;
                if (!mask1.at<uchar>(y, x) || !mask1.at<uchar>(y + 1, x) ||
                    !mask2.at<uchar>(y, x) || !mask2.at<uchar>(y + 1, x))
                    weight += bad_region_penalty_;
                bottom_(y, x) = saturate_cast<int>(weight * 255.f);
            }
            else
                bottom_(y, x) = 0;
        }
    }
}
#endif

} // namespace detail
} // namespace cv
