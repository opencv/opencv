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

namespace cv {
namespace detail {

void PairwiseSeamFinder::find(const vector<Mat> &src, const vector<Point> &corners,
                              vector<Mat> &masks)
{
    LOGLN("Finding seams...");
    int64 t = getTickCount();

    if (src.size() == 0)
        return;

    images_ = src;
    corners_ = corners;
    masks_ = masks;

    for (size_t i = 0; i < src.size() - 1; ++i)
    {
        for (size_t j = i + 1; j < src.size(); ++j)
        {
            Rect roi;
            if (overlapRoi(corners[i], corners[j], src[i].size(), src[j].size(), roi))
                findInPair(i, j, roi);
        }
    }

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


void VoronoiSeamFinder::findInPair(size_t first, size_t second, Rect roi)
{
    const int gap = 10;
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);

    Mat img1 = images_[first], img2 = images_[second];
    Mat mask1 = masks_[first], mask2 = masks_[second];
    Point tl1 = corners_[first], tl2 = corners_[second];

    // Cut submasks with some gap
    for (int y = -gap; y < roi.height + gap; ++y)
    {
        for (int x = -gap; x < roi.width + gap; ++x)
        {
            int y1 = roi.y - tl1.y + y;
            int x1 = roi.x - tl1.x + x;
            if (y1 >= 0 && x1 >= 0 && y1 < img1.rows && x1 < img1.cols)
                submask1.at<uchar>(y + gap, x + gap) = mask1.at<uchar>(y1, x1);
            else
                submask1.at<uchar>(y + gap, x + gap) = 0;

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.rows && x2 < img2.cols)
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
            else
                submask2.at<uchar>(y + gap, x + gap) = 0;
        }
    }

    Mat collision = (submask1 != 0) & (submask2 != 0);
    Mat unique1 = submask1.clone(); unique1.setTo(0, collision);
    Mat unique2 = submask2.clone(); unique2.setTo(0, collision);

    Mat dist1, dist2;
    distanceTransform(unique1 == 0, dist1, CV_DIST_L1, 3);
    distanceTransform(unique2 == 0, dist2, CV_DIST_L1, 3);

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


class GraphCutSeamFinder::Impl : public PairwiseSeamFinder
{
public:
    Impl(int cost_type, float terminal_cost, float bad_region_penalty)
        : cost_type_(cost_type), terminal_cost_(terminal_cost), bad_region_penalty_(bad_region_penalty) {}

    void find(const vector<Mat> &src, const vector<Point> &corners, vector<Mat> &masks);
    void findInPair(size_t first, size_t second, Rect roi);

private:
    void setGraphWeightsColor(const Mat &img1, const Mat &img2, 
                              const Mat &mask1, const Mat &mask2, GCGraph<float> &graph);
    void setGraphWeightsColorGrad(const Mat &img1, const Mat &img2, const Mat &dx1, const Mat &dx2, 
                                  const Mat &dy1, const Mat &dy2, const Mat &mask1, const Mat &mask2, 
                                  GCGraph<float> &graph);

    vector<Mat> dx_, dy_;
    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
};


void GraphCutSeamFinder::Impl::find(const vector<Mat> &src, const vector<Point> &corners,
                                    vector<Mat> &masks)
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
        CV_Error(CV_StsBadArg, "unsupported pixel similarity measure");
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


void GraphCutSeamFinder::find(const vector<Mat> &src, const vector<Point> &corners,
                              vector<Mat> &masks)
{
    impl_->find(src, corners, masks);
}


#ifndef ANDROID
void GraphCutSeamFinderGpu::find(const vector<Mat> &src, const vector<Point> &corners,
                                 vector<Mat> &masks)
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
        CV_Error(CV_StsBadArg, "unsupported pixel similarity measure");
    }

    gpu::GpuMat terminals_d(terminals);
    gpu::GpuMat leftT_d(leftT);
    gpu::GpuMat rightT_d(rightT);
    gpu::GpuMat top_d(top);
    gpu::GpuMat bottom_d(bottom);
    gpu::GpuMat labels_d, buf_d;

    gpu::graphcut(terminals_d, leftT_d, rightT_d, top_d, bottom_d, labels_d, buf_d);

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

