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
#ifdef HAVE_EIGEN
#include <Eigen/Core>
#include <Eigen/Dense>
#endif

namespace cv {
namespace detail {

Ptr<ExposureCompensator> ExposureCompensator::createDefault(int type)
{
    Ptr<ExposureCompensator> e;
    if (type == NO)
        e = makePtr<NoExposureCompensator>();
    else if (type == GAIN)
        e = makePtr<GainCompensator>();
    if (type == GAIN_BLOCKS)
        e = makePtr<BlocksGainCompensator>();
    if (e.get() != nullptr)
    {
        e->setUpdateGain(true);
        return e;
    }
    CV_Error(Error::StsBadArg, "unsupported exposure compensation method");
}


void ExposureCompensator::feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                               const std::vector<UMat> &masks)
{
    std::vector<std::pair<UMat,uchar> > level_masks;
    for (size_t i = 0; i < masks.size(); ++i)
        level_masks.push_back(std::make_pair(masks[i], (uchar)255));
    feed(corners, images, level_masks);
}


void GainCompensator::feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                           const std::vector<std::pair<UMat,uchar> > &masks)
{
    LOGLN("Exposure compensation...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());
    Mat_<int> N(num_images, num_images); N.setTo(0);
    Mat_<double> I(num_images, num_images); I.setTo(0);
    Mat_<bool> skip(num_images, 1); skip.setTo(true);

    //Rect dst_roi = resultRoi(corners, images);
    Mat subimg1, subimg2;
    Mat_<uchar> submask1, submask2, intersect;

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = i; j < num_images; ++j)
        {
            Rect roi;
            if (overlapRoi(corners[i], corners[j], images[i].size(), images[j].size(), roi))
            {
                subimg1 = images[i](Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                subimg2 = images[j](Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);

                submask1 = masks[i].first(Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                submask2 = masks[j].first(Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);
                intersect = (submask1 == masks[i].second) & (submask2 == masks[j].second);

                int intersect_count = countNonZero(intersect);
                N(i, j) = N(j, i) = std::max(1, intersect_count);

                // Don't compute Isums if subimages do not intersect anyway
                if (intersect_count == 0)
                    continue;

                // Don't skip images that intersect with at least one other image
                if (i != j)
                {
                    skip(i, 0) = false;
                    skip(j, 0) = false;
                }

                double Isum1 = 0, Isum2 = 0;
                for (int y = 0; y < roi.height; ++y)
                {
                    const Point3_<uchar>* r1 = subimg1.ptr<Point3_<uchar> >(y);
                    const Point3_<uchar>* r2 = subimg2.ptr<Point3_<uchar> >(y);
                    for (int x = 0; x < roi.width; ++x)
                    {
                        if (intersect(y, x))
                        {
                            Isum1 += std::sqrt(static_cast<double>(sqr(r1[x].x) + sqr(r1[x].y) + sqr(r1[x].z)));
                            Isum2 += std::sqrt(static_cast<double>(sqr(r2[x].x) + sqr(r2[x].y) + sqr(r2[x].z)));
                        }
                    }
                }
                I(i, j) = Isum1 / N(i, j);
                I(j, i) = Isum2 / N(i, j);
            }
        }
    }
    if (getUpdateGain() || gains_.rows != num_images)
    {
        double alpha = 0.01;
        double beta = 100;
        int num_eq = num_images - countNonZero(skip);

        Mat_<double> A(num_eq, num_eq); A.setTo(0);
        Mat_<double> b(num_eq, 1); b.setTo(0);
        for (int i = 0, ki = 0; i < num_images; ++i)
        {
            if (skip(i, 0))
                continue;

            for (int j = 0, kj = 0; j < num_images; ++j)
            {
                if (skip(j, 0))
                    continue;

                b(ki, 0) += beta * N(i, j);
                A(ki, ki) += beta * N(i, j);
                if (j != i)
                {
                    A(ki, ki) += 2 * alpha * I(i, j) * I(i, j) * N(i, j);
                    A(ki, kj) -= 2 * alpha * I(i, j) * I(j, i) * N(i, j);
                }
                ++kj;
            }
            ++ki;
        }

        Mat_<double> l_gains;

#ifdef HAVE_EIGEN
        Eigen::MatrixXf eigen_A, eigen_b, eigen_x;
        cv2eigen(A, eigen_A);
        cv2eigen(b, eigen_b);

        Eigen::LLT<Eigen::MatrixXf> solver(eigen_A);
#if ENABLE_LOG
        if (solver.info() != Eigen::ComputationInfo::Success)
            LOGLN("Failed to solve exposure compensation system");
#endif
        eigen_x = solver.solve(eigen_b);

        Mat_<float> l_gains_float;
        eigen2cv(eigen_x, l_gains_float);
        l_gains_float.convertTo(l_gains, CV_64FC1);
#else
        solve(A, b, l_gains);
#endif
        CV_CheckTypeEQ(l_gains.type(), CV_64FC1, "");

        gains_.create(num_images, 1);
        for (int i = 0, j = 0; i < num_images; ++i)
        {
            if (skip(i, 0))
                gains_.at<double>(i, 0) = 1;
            else
                gains_.at<double>(i, 0) = l_gains(j++, 0);
        }
    }

    LOGLN("Exposure compensation, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


void GainCompensator::apply(int index, Point /*corner*/, InputOutputArray image, InputArray /*mask*/)
{
    CV_INSTRUMENT_REGION();

    multiply(image, gains_(index, 0), image);
}


std::vector<double> GainCompensator::gains() const
{
    std::vector<double> gains_vec(gains_.rows);
    for (int i = 0; i < gains_.rows; ++i)
        gains_vec[i] = gains_(i, 0);
    return gains_vec;
}

void GainCompensator::getMatGains(std::vector<Mat>& umv)
{
    umv.clear();
    for (int i = 0; i < gains_.rows; ++i)
        umv.push_back(Mat(1,1,CV_64FC1,Scalar(gains_(i, 0))));
}
void GainCompensator::setMatGains(std::vector<Mat>& umv)
{
    gains_=Mat_<double>(static_cast<int>(umv.size()),1);
    for (int i = 0; i < static_cast<int>(umv.size()); i++)
    {
        int type = umv[i].type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
        CV_CheckType(type, depth == CV_64F && cn == 1, "Only double images are supported for gain");
        CV_Assert(umv[i].rows == 1 && umv[i].cols == 1);
        gains_(i, 0) = umv[i].at<double>(0, 0);
    }
}


void BlocksGainCompensator::feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                                     const std::vector<std::pair<UMat,uchar> > &masks)
{
    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());

    std::vector<Size> bl_per_imgs(num_images);
    std::vector<Point> block_corners;
    std::vector<UMat> block_images;
    std::vector<std::pair<UMat,uchar> > block_masks;

    // Construct blocks for gain compensator
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        Size bl_per_img((images[img_idx].cols + bl_width_ - 1) / bl_width_,
                        (images[img_idx].rows + bl_height_ - 1) / bl_height_);
        int bl_width = (images[img_idx].cols + bl_per_img.width - 1) / bl_per_img.width;
        int bl_height = (images[img_idx].rows + bl_per_img.height - 1) / bl_per_img.height;
        bl_per_imgs[img_idx] = bl_per_img;
        for (int by = 0; by < bl_per_img.height; ++by)
        {
            for (int bx = 0; bx < bl_per_img.width; ++bx)
            {
                Point bl_tl(bx * bl_width, by * bl_height);
                Point bl_br(std::min(bl_tl.x + bl_width, images[img_idx].cols),
                            std::min(bl_tl.y + bl_height, images[img_idx].rows));

                block_corners.push_back(corners[img_idx] + bl_tl);
                block_images.push_back(images[img_idx](Rect(bl_tl, bl_br)));
                block_masks.push_back(std::make_pair(masks[img_idx].first(Rect(bl_tl, bl_br)),
                                                masks[img_idx].second));
            }
        }
    }

    if (getUpdateGain())
    {
        GainCompensator compensator;
        compensator.feed(block_corners, block_images, block_masks);
        std::vector<double> gains = compensator.gains();
        gain_maps_.resize(num_images);

        Mat_<float> ker(1, 3);
        ker(0, 0) = 0.25; ker(0, 1) = 0.5; ker(0, 2) = 0.25;

        int bl_idx = 0;
        for (int img_idx = 0; img_idx < num_images; ++img_idx)
        {
            Size bl_per_img = bl_per_imgs[img_idx];
            gain_maps_[img_idx].create(bl_per_img, CV_32F);

            {
                Mat_<float> gain_map = gain_maps_[img_idx].getMat(ACCESS_WRITE);
                for (int by = 0; by < bl_per_img.height; ++by)
                    for (int bx = 0; bx < bl_per_img.width; ++bx, ++bl_idx)
                        gain_map(by, bx) = static_cast<float>(gains[bl_idx]);
            }

            sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
            sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
        }
    }
}


void BlocksGainCompensator::apply(int index, Point /*corner*/, InputOutputArray _image, InputArray /*mask*/)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(_image.type() == CV_8UC3);

    UMat u_gain_map;
    if (gain_maps_[index].size() == _image.size())
        u_gain_map = gain_maps_[index];
    else
        resize(gain_maps_[index], u_gain_map, _image.size(), 0, 0, INTER_LINEAR);

    Mat_<float> gain_map = u_gain_map.getMat(ACCESS_READ);
    Mat image = _image.getMat();
    for (int y = 0; y < image.rows; ++y)
    {
        const float* gain_row = gain_map.ptr<float>(y);
        Point3_<uchar>* row = image.ptr<Point3_<uchar> >(y);
        for (int x = 0; x < image.cols; ++x)
        {
            row[x].x = saturate_cast<uchar>(row[x].x * gain_row[x]);
            row[x].y = saturate_cast<uchar>(row[x].y * gain_row[x]);
            row[x].z = saturate_cast<uchar>(row[x].z * gain_row[x]);
        }
    }
}

void BlocksGainCompensator::getMatGains(std::vector<Mat>& umv)
{
    umv.clear();
    for (int i = 0; i < static_cast<int>(gain_maps_.size()); ++i)
    {
        Mat m;
        gain_maps_[i].copyTo(m);
        umv.push_back(m);
    }
}
void BlocksGainCompensator::setMatGains(std::vector<Mat>& umv)
{
    for (int i = 0; i < static_cast<int>(umv.size()); i++)
    {
        UMat m;
        umv[i].copyTo(m);
        gain_maps_.push_back(m);
    }
}


} // namespace detail
} // namespace cv
