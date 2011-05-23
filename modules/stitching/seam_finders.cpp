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
#include "seam_finders.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;


Ptr<SeamFinder> SeamFinder::createDefault(int type)
{
    if (type == NO)
        return new NoSeamFinder();
    if (type == VORONOI)
        return new VoronoiSeamFinder();
    if (type == GRAPH_CUT)
        return new GraphCutSeamFinder();
    CV_Error(CV_StsBadArg, "unsupported seam finding method");
    return NULL;
}


void PairwiseSeamFinder::find(const vector<Mat> &src, const vector<Point> &corners,
                              vector<Mat> &masks)
{
    if (src.size() == 0)
        return;

    for (size_t i = 0; i < src.size() - 1; ++i)
    {
        for (size_t j = i + 1; j < src.size(); ++j)
        {
            int x_min = max(corners[i].x, corners[j].x);
            int x_max = min(corners[i].x + src[i].cols - 1, corners[j].x + src[j].cols - 1);
            int y_min = max(corners[i].y, corners[j].y);
            int y_max = min(corners[i].y + src[i].rows - 1, corners[j].y + src[j].rows - 1);

            if (x_max >= x_min && y_max >= y_min)
                findInPair(src[i], src[j], corners[i], corners[j],
                           Rect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1),
                           masks[i], masks[j]);
        }
    }
}


void VoronoiSeamFinder::findInPair(const Mat &img1, const Mat &img2, Point tl1, Point tl2,
                                   Rect roi, Mat &mask1, Mat &mask2)
{
    const int gap = 10;

    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);

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


class GraphCutSeamFinder::Impl
{
public:
    Impl(int cost_type, float terminal_cost, float bad_region_penalty)
        : cost_type_(cost_type), terminal_cost_(terminal_cost), bad_region_penalty_(bad_region_penalty) {}

    void findInPair(const Mat &img1, const Mat &img2, Point tl1, Point tl2,
                    Rect roi, Mat &mask1, Mat &mask2);

private:
    void setGraphWeightsColor(const Mat &img1, const Mat &img2, const Mat &mask1, const Mat &mask2,
                              GCGraph<float> &graph);

    int cost_type_;
    float terminal_cost_;
    float bad_region_penalty_;
};


void GraphCutSeamFinder::Impl::setGraphWeightsColor(const Mat &img1, const Mat &img2, const Mat &mask1, const Mat &mask2,
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

    const float weight_eps = 1e-3f;

    // Set regular edge weights
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


void GraphCutSeamFinder::Impl::findInPair(const Mat &img1, const Mat &img2, Point tl1, Point tl2,
                                          Rect roi, Mat &mask1, Mat &mask2)
{
    const int gap = 10;

    Mat subimg1(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat subimg2(roi.height + 2 * gap, roi.width + 2 * gap, CV_32FC3);
    Mat submask1(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);
    Mat submask2(roi.height + 2 * gap, roi.width + 2 * gap, CV_8U);

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
            }
            else
            {
                subimg1.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask1.at<uchar>(y + gap, x + gap) = 0;
            }

            int y2 = roi.y - tl2.y + y;
            int x2 = roi.x - tl2.x + x;
            if (y2 >= 0 && x2 >= 0 && y2 < img2.rows && x2 < img2.cols)
            {
                subimg2.at<Point3f>(y + gap, x + gap) = img2.at<Point3f>(y2, x2);
                submask2.at<uchar>(y + gap, x + gap) = mask2.at<uchar>(y2, x2);
            }
            else
            {
                subimg2.at<Point3f>(y + gap, x + gap) = Point3f(0, 0, 0);
                submask2.at<uchar>(y + gap, x + gap) = 0;
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


void GraphCutSeamFinder::findInPair(const Mat &img1, const Mat &img2, Point tl1, Point tl2,
                                    Rect roi, Mat &mask1, Mat &mask2)
{
    impl_->findInPair(img1, img2, tl1, tl2, roi, mask1, mask2);
}
