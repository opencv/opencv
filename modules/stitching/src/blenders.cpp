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
#include "opencl_kernels_stitching.hpp"

#ifdef HAVE_CUDA
    namespace cv { namespace cuda { namespace device
    {
        namespace blend
        {
            void addSrcWeightGpu16S(const PtrStep<short> src, const PtrStep<short> src_weight,
                                    PtrStep<short> dst, PtrStep<short> dst_weight, cv::Rect &rc);
            void addSrcWeightGpu32F(const PtrStep<short> src, const PtrStepf src_weight,
                                    PtrStep<short> dst, PtrStepf dst_weight, cv::Rect &rc);
            void normalizeUsingWeightMapGpu16S(const PtrStep<short> weight, PtrStep<short> src,
                                               const int width, const int height);
            void normalizeUsingWeightMapGpu32F(const PtrStepf weight, PtrStep<short> src,
                                               const int width, const int height);
        }
    }}}
#endif

namespace cv {
namespace detail {

static const float WEIGHT_EPS = 1e-5f;

Ptr<Blender> Blender::createDefault(int type, bool try_gpu)
{
    if (type == NO)
        return makePtr<Blender>();
    if (type == FEATHER)
        return makePtr<FeatherBlender>();
    if (type == MULTI_BAND)
        return makePtr<MultiBandBlender>(try_gpu);
    CV_Error(Error::StsBadArg, "unsupported blending method");
    return Ptr<Blender>();
}


void Blender::prepare(const std::vector<Point> &corners, const std::vector<Size> &sizes)
{
    prepare(resultRoi(corners, sizes));
}


void Blender::prepare(Rect dst_roi)
{
    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;
}


void Blender::feed(InputArray _img, InputArray _mask, Point tl)
{
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();
    Mat dst = dst_.getMat(ACCESS_RW);
    Mat dst_mask = dst_mask_.getMat(ACCESS_RW);

    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst.ptr<Point3_<short> >(dy + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask.ptr<uchar>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x])
                dst_row[dx + x] = src_row[x];
            dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}


void Blender::blend(InputOutputArray dst, InputOutputArray dst_mask)
{
    UMat mask;
    compare(dst_mask_, 0, mask, CMP_EQ);
    dst_.setTo(Scalar::all(0), mask);
    dst.assign(dst_);
    dst_mask.assign(dst_mask_);
    dst_.release();
    dst_mask_.release();
}


void FeatherBlender::prepare(Rect dst_roi)
{
    Blender::prepare(dst_roi);
    dst_weight_map_.create(dst_roi.size(), CV_32F);
    dst_weight_map_.setTo(0);
}


void FeatherBlender::feed(InputArray _img, InputArray mask, Point tl)
{
    Mat img = _img.getMat();
    Mat dst = dst_.getMat(ACCESS_RW);

    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    createWeightMap(mask, sharpness_, weight_map_);
    Mat weight_map = weight_map_.getMat(ACCESS_READ);
    Mat dst_weight_map = dst_weight_map_.getMat(ACCESS_RW);

    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short>* src_row = img.ptr<Point3_<short> >(y);
        Point3_<short>* dst_row = dst.ptr<Point3_<short> >(dy + y);
        const float* weight_row = weight_map.ptr<float>(y);
        float* dst_weight_row = dst_weight_map.ptr<float>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
            dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
            dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
            dst_weight_row[dx + x] += weight_row[x];
        }
    }
}


void FeatherBlender::blend(InputOutputArray dst, InputOutputArray dst_mask)
{
    normalizeUsingWeightMap(dst_weight_map_, dst_);
    compare(dst_weight_map_, WEIGHT_EPS, dst_mask_, CMP_GT);
    Blender::blend(dst, dst_mask);
}


Rect FeatherBlender::createWeightMaps(const std::vector<UMat> &masks, const std::vector<Point> &corners,
                                      std::vector<UMat> &weight_maps)
{
    weight_maps.resize(masks.size());
    for (size_t i = 0; i < masks.size(); ++i)
        createWeightMap(masks[i], sharpness_, weight_maps[i]);

    Rect dst_roi = resultRoi(corners, masks);
    Mat weights_sum(dst_roi.size(), CV_32F);
    weights_sum.setTo(0);

    for (size_t i = 0; i < weight_maps.size(); ++i)
    {
        Rect roi(corners[i].x - dst_roi.x, corners[i].y - dst_roi.y,
                 weight_maps[i].cols, weight_maps[i].rows);
        add(weights_sum(roi), weight_maps[i], weights_sum(roi));
    }

    for (size_t i = 0; i < weight_maps.size(); ++i)
    {
        Rect roi(corners[i].x - dst_roi.x, corners[i].y - dst_roi.y,
                 weight_maps[i].cols, weight_maps[i].rows);
        Mat tmp = weights_sum(roi);
        tmp.setTo(1, tmp < std::numeric_limits<float>::epsilon());
        divide(weight_maps[i], tmp, weight_maps[i]);
    }

    return dst_roi;
}


MultiBandBlender::MultiBandBlender(int try_gpu, int num_bands, int weight_type)
{
    setNumBands(num_bands);

#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    can_use_gpu_ = try_gpu && cuda::getCudaEnabledDeviceCount();
#else
    (void) try_gpu;
    can_use_gpu_ = false;
#endif

    CV_Assert(weight_type == CV_32F || weight_type == CV_16S);
    weight_type_ = weight_type;
}


void MultiBandBlender::prepare(Rect dst_roi)
{
    dst_roi_final_ = dst_roi;

    // Crop unnecessary bands
    double max_len = static_cast<double>(std::max(dst_roi.width, dst_roi.height));
    num_bands_ = std::min(actual_num_bands_, static_cast<int>(ceil(std::log(max_len) / std::log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);

    Blender::prepare(dst_roi);

#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    if (can_use_gpu_)
    {
        gpu_dst_pyr_laplace_.resize(num_bands_ + 1);
        gpu_dst_pyr_laplace_[0].create(dst_roi.size(), CV_16SC3);
        gpu_dst_pyr_laplace_[0].setTo(Scalar::all(0));

        gpu_dst_band_weights_.resize(num_bands_ + 1);
        gpu_dst_band_weights_[0].create(dst_roi.size(), weight_type_);
        gpu_dst_band_weights_[0].setTo(0);

        for (int i = 1; i <= num_bands_; ++i)
        {
            gpu_dst_pyr_laplace_[i].create((gpu_dst_pyr_laplace_[i - 1].rows + 1) / 2,
                (gpu_dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
            gpu_dst_band_weights_[i].create((gpu_dst_band_weights_[i - 1].rows + 1) / 2,
                (gpu_dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
            gpu_dst_pyr_laplace_[i].setTo(Scalar::all(0));
            gpu_dst_band_weights_[i].setTo(0);
        }
    }
    else
#endif
    {
        dst_pyr_laplace_.resize(num_bands_ + 1);
        dst_pyr_laplace_[0] = dst_;

        dst_band_weights_.resize(num_bands_ + 1);
        dst_band_weights_[0].create(dst_roi.size(), weight_type_);
        dst_band_weights_[0].setTo(0);

        for (int i = 1; i <= num_bands_; ++i)
        {
            dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2,
                (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
            dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
                (dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
            dst_pyr_laplace_[i].setTo(Scalar::all(0));
            dst_band_weights_[i].setTo(0);
        }
    }
}

#ifdef HAVE_OPENCL
static bool ocl_MultiBandBlender_feed(InputArray _src, InputArray _weight,
        InputOutputArray _dst, InputOutputArray _dst_weight)
{
    String buildOptions = "-D DEFINE_feed";
    ocl::buildOptionsAddMatrixDescription(buildOptions, "src", _src);
    ocl::buildOptionsAddMatrixDescription(buildOptions, "weight", _weight);
    ocl::buildOptionsAddMatrixDescription(buildOptions, "dst", _dst);
    ocl::buildOptionsAddMatrixDescription(buildOptions, "dstWeight", _dst_weight);
    ocl::Kernel k("feed", ocl::stitching::multibandblend_oclsrc, buildOptions);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();

    k.args(ocl::KernelArg::ReadOnly(src),
           ocl::KernelArg::ReadOnly(_weight.getUMat()),
           ocl::KernelArg::ReadWrite(_dst.getUMat()),
           ocl::KernelArg::ReadWrite(_dst_weight.getUMat())
           );

    size_t globalsize[2] = {(size_t)src.cols, (size_t)src.rows };
    return k.run(2, globalsize, NULL, false);
}
#endif

void MultiBandBlender::feed(InputArray _img, InputArray mask, Point tl)
{
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    UMat img = _img.getUMat();
    CV_Assert(img.type() == CV_16SC3 || img.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8U);

    // Keep source image in memory with small border
    int gap = 3 * (1 << num_bands_);
    Point tl_new(std::max(dst_roi_.x, tl.x - gap),
                 std::max(dst_roi_.y, tl.y - gap));
    Point br_new(std::min(dst_roi_.br().x, tl.x + img.cols + gap),
                 std::min(dst_roi_.br().y, tl.y + img.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
    // After that scale between layers is exactly 2.
    //
    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = std::max(br_new.y - dst_roi_.br().y, 0);
    int dx = std::max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - img.rows;
    int right = br_new.x - tl.x - img.cols;

#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    if (can_use_gpu_)
    {
        // Create the source image Laplacian pyramid
        cuda::GpuMat gpu_img;
        gpu_img.upload(img);
        cuda::GpuMat img_with_border;
        cuda::copyMakeBorder(gpu_img, img_with_border, top, bottom, left, right, BORDER_REFLECT);
        std::vector<cuda::GpuMat> gpu_src_pyr_laplace(num_bands_ + 1);
        img_with_border.convertTo(gpu_src_pyr_laplace[0], CV_16S);
        for (int i = 0; i < num_bands_; ++i)
            cuda::pyrDown(gpu_src_pyr_laplace[i], gpu_src_pyr_laplace[i + 1]);
        for (int i = 0; i < num_bands_; ++i)
        {
            cuda::GpuMat up;
            cuda::pyrUp(gpu_src_pyr_laplace[i + 1], up);
            cuda::subtract(gpu_src_pyr_laplace[i], up, gpu_src_pyr_laplace[i]);
        }

        // Create the weight map Gaussian pyramid
        cuda::GpuMat gpu_mask;
        gpu_mask.upload(mask);
        cuda::GpuMat weight_map;
        std::vector<cuda::GpuMat> gpu_weight_pyr_gauss(num_bands_ + 1);

        if (weight_type_ == CV_32F)
        {
            gpu_mask.convertTo(weight_map, CV_32F, 1. / 255.);
        }
        else // weight_type_ == CV_16S
        {
            gpu_mask.convertTo(weight_map, CV_16S);
            cuda::GpuMat add_mask;
            cuda::compare(gpu_mask, 0, add_mask, CMP_NE);
            cuda::add(weight_map, Scalar::all(1), weight_map, add_mask);
        }
        cuda::copyMakeBorder(weight_map, gpu_weight_pyr_gauss[0], top, bottom, left, right, BORDER_CONSTANT);
        for (int i = 0; i < num_bands_; ++i)
            cuda::pyrDown(gpu_weight_pyr_gauss[i], gpu_weight_pyr_gauss[i + 1]);

        int y_tl = tl_new.y - dst_roi_.y;
        int y_br = br_new.y - dst_roi_.y;
        int x_tl = tl_new.x - dst_roi_.x;
        int x_br = br_new.x - dst_roi_.x;

        // Add weighted layer of the source image to the final Laplacian pyramid layer
        for (int i = 0; i <= num_bands_; ++i)
        {
            Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
            cuda::GpuMat &_src_pyr_laplace = gpu_src_pyr_laplace[i];
            cuda::GpuMat _dst_pyr_laplace = gpu_dst_pyr_laplace_[i](rc);
            cuda::GpuMat &_weight_pyr_gauss = gpu_weight_pyr_gauss[i];
            cuda::GpuMat _dst_band_weights = gpu_dst_band_weights_[i](rc);

            using namespace cv::cuda::device::blend;
            if (weight_type_ == CV_32F)
            {
                addSrcWeightGpu32F(_src_pyr_laplace, _weight_pyr_gauss, _dst_pyr_laplace, _dst_band_weights, rc);
            }
            else
            {
                addSrcWeightGpu16S(_src_pyr_laplace, _weight_pyr_gauss, _dst_pyr_laplace, _dst_band_weights, rc);
            }
            x_tl /= 2; y_tl /= 2;
            x_br /= 2; y_br /= 2;
        }
        return;
    }
#endif

    // Create the source image Laplacian pyramid
    UMat img_with_border;
    copyMakeBorder(_img, img_with_border, top, bottom, left, right,
                   BORDER_REFLECT);
    LOGLN("  Add border to the source image, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#if ENABLE_LOG
    t = getTickCount();
#endif

    std::vector<UMat> src_pyr_laplace;
    createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);

    LOGLN("  Create the source image Laplacian pyramid, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#if ENABLE_LOG
    t = getTickCount();
#endif

    // Create the weight map Gaussian pyramid
    UMat weight_map;
    std::vector<UMat> weight_pyr_gauss(num_bands_ + 1);

    if(weight_type_ == CV_32F)
    {
        mask.getUMat().convertTo(weight_map, CV_32F, 1./255.);
    }
    else // weight_type_ == CV_16S
    {
        mask.getUMat().convertTo(weight_map, CV_16S);
        UMat add_mask;
        compare(mask, 0, add_mask, CMP_NE);
        add(weight_map, Scalar::all(1), weight_map, add_mask);
    }

    copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, BORDER_CONSTANT);

    for (int i = 0; i < num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    LOGLN("  Create the weight map Gaussian pyramid, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#if ENABLE_LOG
    t = getTickCount();
#endif

    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    for (int i = 0; i <= num_bands_; ++i)
    {
        Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
#ifdef HAVE_OPENCL
        if ( !cv::ocl::useOpenCL() ||
             !ocl_MultiBandBlender_feed(src_pyr_laplace[i], weight_pyr_gauss[i],
                    dst_pyr_laplace_[i](rc), dst_band_weights_[i](rc)) )
#endif
        {
            Mat _src_pyr_laplace = src_pyr_laplace[i].getMat(ACCESS_READ);
            Mat _dst_pyr_laplace = dst_pyr_laplace_[i](rc).getMat(ACCESS_RW);
            Mat _weight_pyr_gauss = weight_pyr_gauss[i].getMat(ACCESS_READ);
            Mat _dst_band_weights = dst_band_weights_[i](rc).getMat(ACCESS_RW);
            if(weight_type_ == CV_32F)
            {
                for (int y = 0; y < rc.height; ++y)
                {
                    const Point3_<short>* src_row = _src_pyr_laplace.ptr<Point3_<short> >(y);
                    Point3_<short>* dst_row = _dst_pyr_laplace.ptr<Point3_<short> >(y);
                    const float* weight_row = _weight_pyr_gauss.ptr<float>(y);
                    float* dst_weight_row = _dst_band_weights.ptr<float>(y);

                    for (int x = 0; x < rc.width; ++x)
                    {
                        dst_row[x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                        dst_row[x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                        dst_row[x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                        dst_weight_row[x] += weight_row[x];
                    }
                }
            }
            else // weight_type_ == CV_16S
            {
                for (int y = 0; y < y_br - y_tl; ++y)
                {
                    const Point3_<short>* src_row = _src_pyr_laplace.ptr<Point3_<short> >(y);
                    Point3_<short>* dst_row = _dst_pyr_laplace.ptr<Point3_<short> >(y);
                    const short* weight_row = _weight_pyr_gauss.ptr<short>(y);
                    short* dst_weight_row = _dst_band_weights.ptr<short>(y);

                    for (int x = 0; x < x_br - x_tl; ++x)
                    {
                        dst_row[x].x += short((src_row[x].x * weight_row[x]) >> 8);
                        dst_row[x].y += short((src_row[x].y * weight_row[x]) >> 8);
                        dst_row[x].z += short((src_row[x].z * weight_row[x]) >> 8);
                        dst_weight_row[x] += weight_row[x];
                    }
                }
            }
        }
#ifdef HAVE_OPENCL
        else
        {
            CV_IMPL_ADD(CV_IMPL_OCL);
        }
#endif

        x_tl /= 2; y_tl /= 2;
        x_br /= 2; y_br /= 2;
    }

    LOGLN("  Add weighted layer of the source image to the final Laplacian pyramid layer, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


void MultiBandBlender::blend(InputOutputArray dst, InputOutputArray dst_mask)
{
    cv::UMat dst_band_weights_0;
    Rect dst_rc(0, 0, dst_roi_final_.width, dst_roi_final_.height);
#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    if (can_use_gpu_)
    {
        for (int i = 0; i <= num_bands_; ++i)
        {
            cuda::GpuMat dst_i = gpu_dst_pyr_laplace_[i];
            cuda::GpuMat weight_i = gpu_dst_band_weights_[i];

            using namespace ::cv::cuda::device::blend;
            if (weight_type_ == CV_32F)
            {
                normalizeUsingWeightMapGpu32F(weight_i, dst_i, weight_i.cols, weight_i.rows);
            }
            else
            {
                normalizeUsingWeightMapGpu16S(weight_i, dst_i, weight_i.cols, weight_i.rows);
            }
        }

        // Restore image from Laplacian pyramid
        for (size_t i = num_bands_; i > 0; --i)
        {
            cuda::GpuMat up;
            cuda::pyrUp(gpu_dst_pyr_laplace_[i], up);
            cuda::add(up, gpu_dst_pyr_laplace_[i - 1], gpu_dst_pyr_laplace_[i - 1]);
        }

        gpu_dst_pyr_laplace_[0](dst_rc).download(dst_);
        gpu_dst_band_weights_[0].download(dst_band_weights_0);

        gpu_dst_pyr_laplace_.clear();
        gpu_dst_band_weights_.clear();
    }
    else
#endif
    {
        for (int i = 0; i <= num_bands_; ++i)
            normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i]);

        restoreImageFromLaplacePyr(dst_pyr_laplace_);

        dst_ = dst_pyr_laplace_[0](dst_rc);
        dst_band_weights_0 = dst_band_weights_[0];

        dst_pyr_laplace_.clear();
        dst_band_weights_.clear();
    }

    compare(dst_band_weights_0(dst_rc), WEIGHT_EPS, dst_mask_, CMP_GT);

    Blender::blend(dst, dst_mask);
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

#ifdef HAVE_OPENCL
static bool ocl_normalizeUsingWeightMap(InputArray _weight, InputOutputArray _mat)
{
    String buildOptions = "-D DEFINE_normalizeUsingWeightMap";
    ocl::buildOptionsAddMatrixDescription(buildOptions, "mat", _mat);
    ocl::buildOptionsAddMatrixDescription(buildOptions, "weight", _weight);
    ocl::Kernel k("normalizeUsingWeightMap", ocl::stitching::multibandblend_oclsrc, buildOptions);
    if (k.empty())
        return false;

    UMat mat = _mat.getUMat();

    k.args(ocl::KernelArg::ReadWrite(mat),
           ocl::KernelArg::ReadOnly(_weight.getUMat())
           );

    size_t globalsize[2] = {(size_t)mat.cols, (size_t)mat.rows };
    return k.run(2, globalsize, NULL, false);
}
#endif

void normalizeUsingWeightMap(InputArray _weight, InputOutputArray _src)
{
    Mat src;
    Mat weight;
#ifdef HAVE_TEGRA_OPTIMIZATION
    src = _src.getMat();
    weight = _weight.getMat();
    if(tegra::useTegra() && tegra::normalizeUsingWeightMap(weight, src))
        return;
#endif

#ifdef HAVE_OPENCL
    if ( !cv::ocl::useOpenCL() ||
            !ocl_normalizeUsingWeightMap(_weight, _src) )
#endif
    {
        src = _src.getMat();
        weight = _weight.getMat();

        CV_Assert(src.type() == CV_16SC3);

        if (weight.type() == CV_32FC1)
        {
            for (int y = 0; y < src.rows; ++y)
            {
                Point3_<short> *row = src.ptr<Point3_<short> >(y);
                const float *weight_row = weight.ptr<float>(y);

                for (int x = 0; x < src.cols; ++x)
                {
                    row[x].x = static_cast<short>(row[x].x / (weight_row[x] + WEIGHT_EPS));
                    row[x].y = static_cast<short>(row[x].y / (weight_row[x] + WEIGHT_EPS));
                    row[x].z = static_cast<short>(row[x].z / (weight_row[x] + WEIGHT_EPS));
                }
            }
        }
        else
        {
            CV_Assert(weight.type() == CV_16SC1);

            for (int y = 0; y < src.rows; ++y)
            {
                const short *weight_row = weight.ptr<short>(y);
                Point3_<short> *row = src.ptr<Point3_<short> >(y);

                for (int x = 0; x < src.cols; ++x)
                {
                    int w = weight_row[x] + 1;
                    row[x].x = static_cast<short>((row[x].x << 8) / w);
                    row[x].y = static_cast<short>((row[x].y << 8) / w);
                    row[x].z = static_cast<short>((row[x].z << 8) / w);
                }
            }
        }
    }
#ifdef HAVE_OPENCL
    else
    {
        CV_IMPL_ADD(CV_IMPL_OCL);
    }
#endif
}


void createWeightMap(InputArray mask, float sharpness, InputOutputArray weight)
{
    CV_Assert(mask.type() == CV_8U);
    distanceTransform(mask, weight, DIST_L1, 3);
    UMat tmp;
    multiply(weight, sharpness, tmp);
    threshold(tmp, weight, 1.f, 1.f, THRESH_TRUNC);
}


void createLaplacePyr(InputArray img, int num_levels, std::vector<UMat> &pyr)
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    cv::Mat imgMat = img.getMat();
    if(tegra::useTegra() && tegra::createLaplacePyr(imgMat, num_levels, pyr))
        return;
#endif

    pyr.resize(num_levels + 1);

    if(img.depth() == CV_8U)
    {
        if(num_levels == 0)
        {
            img.getUMat().convertTo(pyr[0], CV_16S);
            return;
        }

        UMat downNext;
        UMat current = img.getUMat();
        pyrDown(img, downNext);

        for(int i = 1; i < num_levels; ++i)
        {
            UMat lvl_up;
            UMat lvl_down;

            pyrDown(downNext, lvl_down);
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[i-1], noArray(), CV_16S);

            current = downNext;
            downNext = lvl_down;
        }

        {
            UMat lvl_up;
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[num_levels-1], noArray(), CV_16S);

            downNext.convertTo(pyr[num_levels], CV_16S);
        }
    }
    else
    {
        pyr[0] = img.getUMat();
        for (int i = 0; i < num_levels; ++i)
            pyrDown(pyr[i], pyr[i + 1]);
        UMat tmp;
        for (int i = 0; i < num_levels; ++i)
        {
            pyrUp(pyr[i + 1], tmp, pyr[i].size());
            subtract(pyr[i], tmp, pyr[i]);
        }
    }
}


void createLaplacePyrGpu(InputArray img, int num_levels, std::vector<UMat> &pyr)
{
#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    pyr.resize(num_levels + 1);

    std::vector<cuda::GpuMat> gpu_pyr(num_levels + 1);
    gpu_pyr[0].upload(img);
    for (int i = 0; i < num_levels; ++i)
        cuda::pyrDown(gpu_pyr[i], gpu_pyr[i + 1]);

    cuda::GpuMat tmp;
    for (int i = 0; i < num_levels; ++i)
    {
        cuda::pyrUp(gpu_pyr[i + 1], tmp);
        cuda::subtract(gpu_pyr[i], tmp, gpu_pyr[i]);
        gpu_pyr[i].download(pyr[i]);
    }

    gpu_pyr[num_levels].download(pyr[num_levels]);
#else
    (void)img;
    (void)num_levels;
    (void)pyr;
    CV_Error(Error::StsNotImplemented, "CUDA optimization is unavailable");
#endif
}


void restoreImageFromLaplacePyr(std::vector<UMat> &pyr)
{
    if (pyr.empty())
        return;
    UMat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyrUp(pyr[i], tmp, pyr[i - 1].size());
        add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}


void restoreImageFromLaplacePyrGpu(std::vector<UMat> &pyr)
{
#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    if (pyr.empty())
        return;

    std::vector<cuda::GpuMat> gpu_pyr(pyr.size());
    for (size_t i = 0; i < pyr.size(); ++i)
        gpu_pyr[i].upload(pyr[i]);

    cuda::GpuMat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        cuda::pyrUp(gpu_pyr[i], tmp);
        cuda::add(tmp, gpu_pyr[i - 1], gpu_pyr[i - 1]);
    }

    gpu_pyr[0].download(pyr[0]);
#else
    (void)pyr;
    CV_Error(Error::StsNotImplemented, "CUDA optimization is unavailable");
#endif
}

} // namespace detail
} // namespace cv
