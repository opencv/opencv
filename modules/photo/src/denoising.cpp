/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "fast_nlmeans_denoising_invoker.hpp"
#include "fast_nlmeans_multi_denoising_invoker.hpp"
#include "fast_nlmeans_denoising_opencl.hpp"

template<typename ST, typename IT, typename UIT, typename D>
static void fastNlMeansDenoising_( const Mat& src, Mat& dst, const std::vector<float>& h,
                                   int templateWindowSize, int searchWindowSize)
{
    int hn = (int)h.size();
    double granularity = (double)std::max(1., (double)dst.total()/(1 << 17));

    switch (CV_MAT_CN(src.type())) {
        case 1:
            parallel_for_(cv::Range(0, src.rows),
                          FastNlMeansDenoisingInvoker<ST, IT, UIT, D, int>(
                              src, dst, templateWindowSize, searchWindowSize, &h[0]),
                          granularity);
            break;
        case 2:
            if (hn == 1)
                parallel_for_(cv::Range(0, src.rows),
                              FastNlMeansDenoisingInvoker<Vec<ST, 2>, IT, UIT, D, int>(
                                  src, dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            else
                parallel_for_(cv::Range(0, src.rows),
                              FastNlMeansDenoisingInvoker<Vec<ST, 2>, IT, UIT, D, Vec2i>(
                                  src, dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            break;
        case 3:
            if (hn == 1)
                parallel_for_(cv::Range(0, src.rows),
                              FastNlMeansDenoisingInvoker<Vec<ST, 3>, IT, UIT, D, int>(
                                  src, dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            else
                parallel_for_(cv::Range(0, src.rows),
                              FastNlMeansDenoisingInvoker<Vec<ST, 3>, IT, UIT, D, Vec3i>(
                                  src, dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            break;
        case 4:
            if (hn == 1)
                parallel_for_(cv::Range(0, src.rows),
                              FastNlMeansDenoisingInvoker<Vec<ST, 4>, IT, UIT, D, int>(
                                  src, dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            else
                parallel_for_(cv::Range(0, src.rows),
                              FastNlMeansDenoisingInvoker<Vec<ST, 4>, IT, UIT, D, Vec4i>(
                                  src, dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            break;
        default:
            CV_Error(Error::StsBadArg,
                     "Unsupported number of channels! Only 1, 2, 3, and 4 are supported");
    }
}

void cv::fastNlMeansDenoising( InputArray _src, OutputArray _dst, float h,
                               int templateWindowSize, int searchWindowSize)
{
    fastNlMeansDenoising(_src, _dst, std::vector<float>(1, h),
                         templateWindowSize, searchWindowSize);
}

void cv::fastNlMeansDenoising( InputArray _src, OutputArray _dst, const std::vector<float>& h,
                               int templateWindowSize, int searchWindowSize, int normType)
{
    int hn = (int)h.size(), type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(hn == 1 || hn == cn);

    Size src_size = _src.size();
    CV_OCL_RUN(_src.dims() <= 2 && (_src.isUMat() || _dst.isUMat()) &&
               src_size.width > 5 && src_size.height > 5, // low accuracy on small sizes
               ocl_fastNlMeansDenoising(_src, _dst, &h[0], hn,
                                        templateWindowSize, searchWindowSize, normType))

    Mat src = _src.getMat();
    _dst.create(src_size, src.type());
    Mat dst = _dst.getMat();

    switch (normType) {
        case NORM_L2:
#ifdef HAVE_TEGRA_OPTIMIZATION
            if(hn == 1 && tegra::useTegra() &&
               tegra::fastNlMeansDenoising(src, dst, h[0], templateWindowSize, searchWindowSize))
                return;
#endif
            switch (depth) {
                case CV_8U:
                    fastNlMeansDenoising_<uchar, int, unsigned, DistSquared>(src, dst, h,
                                                                             templateWindowSize,
                                                                             searchWindowSize);
                    break;
                default:
                    CV_Error(Error::StsBadArg,
                             "Unsupported depth! Only CV_8U is supported for NORM_L2");
            }
            break;
        case NORM_L1:
            switch (depth) {
                case CV_8U:
                    fastNlMeansDenoising_<uchar, int, unsigned, DistAbs>(src, dst, h,
                                                                         templateWindowSize,
                                                                         searchWindowSize);
                    break;
                case CV_16U:
                    fastNlMeansDenoising_<ushort, int64, uint64, DistAbs>(src, dst, h,
                                                                          templateWindowSize,
                                                                          searchWindowSize);
                    break;
                default:
                    CV_Error(Error::StsBadArg,
                             "Unsupported depth! Only CV_8U and CV_16U are supported for NORM_L1");
            }
            break;
        default:
            CV_Error(Error::StsBadArg,
                     "Unsupported norm type! Only NORM_L2 and NORM_L1 are supported");
    }
}

void cv::fastNlMeansDenoisingColored( InputArray _src, OutputArray _dst,
                                      float h, float hForColorComponents,
                                      int templateWindowSize, int searchWindowSize)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    Size src_size = _src.size();
    if (type != CV_8UC3 && type != CV_8UC4)
    {
        CV_Error(Error::StsBadArg, "Type of input image should be CV_8UC3 or CV_8UC4!");
        return;
    }

    CV_OCL_RUN(_src.dims() <= 2 && (_dst.isUMat() || _src.isUMat()) &&
                src_size.width > 5 && src_size.height > 5, // low accuracy on small sizes
                ocl_fastNlMeansDenoisingColored(_src, _dst, h, hForColorComponents,
                                                templateWindowSize, searchWindowSize))

    Mat src = _src.getMat();
    _dst.create(src_size, type);
    Mat dst = _dst.getMat();

    Mat src_lab;
    cvtColor(src, src_lab, COLOR_LBGR2Lab);

    Mat l(src_size, CV_MAKE_TYPE(depth, 1));
    Mat ab(src_size, CV_MAKE_TYPE(depth, 2));
    Mat l_ab[] = { l, ab };
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels(&src_lab, 1, l_ab, 2, from_to, 3);

    fastNlMeansDenoising(l, l, h, templateWindowSize, searchWindowSize);
    fastNlMeansDenoising(ab, ab, hForColorComponents, templateWindowSize, searchWindowSize);

    Mat l_ab_denoised[] = { l, ab };
    Mat dst_lab(src_size, CV_MAKE_TYPE(depth, 3));
    mixChannels(l_ab_denoised, 2, &dst_lab, 1, from_to, 3);

    cvtColor(dst_lab, dst, COLOR_Lab2LBGR, cn);
}

static void fastNlMeansDenoisingMultiCheckPreconditions(
                               const std::vector<Mat>& srcImgs,
                               int imgToDenoiseIndex, int temporalWindowSize,
                               int templateWindowSize, int searchWindowSize)
{
    int src_imgs_size = static_cast<int>(srcImgs.size());
    if (src_imgs_size == 0)
    {
        CV_Error(Error::StsBadArg, "Input images vector should not be empty!");
    }

    if (temporalWindowSize % 2 == 0 ||
        searchWindowSize % 2 == 0 ||
        templateWindowSize % 2 == 0) {
        CV_Error(Error::StsBadArg, "All windows sizes should be odd!");
    }

    int temporalWindowHalfSize = temporalWindowSize / 2;
    if (imgToDenoiseIndex - temporalWindowHalfSize < 0 ||
        imgToDenoiseIndex + temporalWindowHalfSize >= src_imgs_size)
    {
        CV_Error(Error::StsBadArg,
            "imgToDenoiseIndex and temporalWindowSize "
            "should be chosen corresponding srcImgs size!");
    }

    for (int i = 1; i < src_imgs_size; i++)
        if (srcImgs[0].size() != srcImgs[i].size() || srcImgs[0].type() != srcImgs[i].type())
        {
            CV_Error(Error::StsBadArg, "Input images should have the same size and type!");
        }
}

template<typename ST, typename IT, typename UIT, typename D>
static void fastNlMeansDenoisingMulti_( const std::vector<Mat>& srcImgs, Mat& dst,
                                        int imgToDenoiseIndex, int temporalWindowSize,
                                        const std::vector<float>& h,
                                        int templateWindowSize, int searchWindowSize)
{
    int hn = (int)h.size();
    double granularity = (double)std::max(1., (double)dst.total()/(1 << 16));

    switch (srcImgs[0].type())
    {
        case CV_8U:
            parallel_for_(cv::Range(0, srcImgs[0].rows),
                          FastNlMeansMultiDenoisingInvoker<uchar, IT, UIT, D, int>(
                              srcImgs, imgToDenoiseIndex, temporalWindowSize,
                              dst, templateWindowSize, searchWindowSize, &h[0]),
                          granularity);
            break;
        case CV_8UC2:
            if (hn == 1)
                parallel_for_(cv::Range(0, srcImgs[0].rows),
                              FastNlMeansMultiDenoisingInvoker<Vec<ST, 2>, IT, UIT, D, int>(
                                  srcImgs, imgToDenoiseIndex, temporalWindowSize,
                                  dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            else
                parallel_for_(cv::Range(0, srcImgs[0].rows),
                              FastNlMeansMultiDenoisingInvoker<Vec<ST, 2>, IT, UIT, D, Vec2i>(
                                  srcImgs, imgToDenoiseIndex, temporalWindowSize,
                                  dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            break;
        case CV_8UC3:
            if (hn == 1)
                parallel_for_(cv::Range(0, srcImgs[0].rows),
                              FastNlMeansMultiDenoisingInvoker<Vec<ST, 3>, IT, UIT, D, int>(
                                  srcImgs, imgToDenoiseIndex, temporalWindowSize,
                                  dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            else
                parallel_for_(cv::Range(0, srcImgs[0].rows),
                              FastNlMeansMultiDenoisingInvoker<Vec<ST, 3>, IT, UIT, D, Vec3i>(
                                  srcImgs, imgToDenoiseIndex, temporalWindowSize,
                                  dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            break;
        case CV_8UC4:
            if (hn == 1)
                parallel_for_(cv::Range(0, srcImgs[0].rows),
                              FastNlMeansMultiDenoisingInvoker<Vec<ST, 4>, IT, UIT, D, int>(
                                  srcImgs, imgToDenoiseIndex, temporalWindowSize,
                                  dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            else
                parallel_for_(cv::Range(0, srcImgs[0].rows),
                              FastNlMeansMultiDenoisingInvoker<Vec<ST, 4>, IT, UIT, D, Vec4i>(
                                  srcImgs, imgToDenoiseIndex, temporalWindowSize,
                                  dst, templateWindowSize, searchWindowSize, &h[0]),
                              granularity);
            break;
        default:
            CV_Error(Error::StsBadArg,
                "Unsupported image format! Only CV_8U, CV_8UC2, CV_8UC3 and CV_8UC4 are supported");
    }
}

void cv::fastNlMeansDenoisingMulti( InputArrayOfArrays _srcImgs, OutputArray _dst,
                                    int imgToDenoiseIndex, int temporalWindowSize,
                                    float h, int templateWindowSize, int searchWindowSize)
{
    fastNlMeansDenoisingMulti(_srcImgs, _dst, imgToDenoiseIndex, temporalWindowSize,
                              std::vector<float>(1, h), templateWindowSize, searchWindowSize);
}

void cv::fastNlMeansDenoisingMulti( InputArrayOfArrays _srcImgs, OutputArray _dst,
                                    int imgToDenoiseIndex, int temporalWindowSize,
                                    const std::vector<float>& h,
                                    int templateWindowSize, int searchWindowSize, int normType)
{
    std::vector<Mat> srcImgs;
    _srcImgs.getMatVector(srcImgs);

    fastNlMeansDenoisingMultiCheckPreconditions(
        srcImgs, imgToDenoiseIndex,
        temporalWindowSize, templateWindowSize, searchWindowSize);

    int hn = (int)h.size();
    int type = srcImgs[0].type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(hn == 1 || hn == cn);

    _dst.create(srcImgs[0].size(), srcImgs[0].type());
    Mat dst = _dst.getMat();

    switch (normType) {
        case NORM_L2:
            switch (depth) {
                case CV_8U:
                    fastNlMeansDenoisingMulti_<uchar, int, unsigned,
                                               DistSquared>(srcImgs, dst,
                                                            imgToDenoiseIndex, temporalWindowSize,
                                                            h,
                                                            templateWindowSize, searchWindowSize);
                    break;
                default:
                    CV_Error(Error::StsBadArg,
                             "Unsupported depth! Only CV_8U is supported for NORM_L2");
            }
            break;
        case NORM_L1:
            switch (depth) {
                case CV_8U:
                    fastNlMeansDenoisingMulti_<uchar, int, unsigned,
                                               DistAbs>(srcImgs, dst,
                                                        imgToDenoiseIndex, temporalWindowSize,
                                                        h,
                                                        templateWindowSize, searchWindowSize);
                    break;
                case CV_16U:
                    fastNlMeansDenoisingMulti_<ushort, int64, uint64,
                                               DistAbs>(srcImgs, dst,
                                                        imgToDenoiseIndex, temporalWindowSize,
                                                        h,
                                                        templateWindowSize, searchWindowSize);
                    break;
                default:
                    CV_Error(Error::StsBadArg,
                             "Unsupported depth! Only CV_8U and CV_16U are supported for NORM_L1");
            }
            break;
        default:
            CV_Error(Error::StsBadArg,
                     "Unsupported norm type! Only NORM_L2 and NORM_L1 are supported");
    }
}

void cv::fastNlMeansDenoisingColoredMulti( InputArrayOfArrays _srcImgs, OutputArray _dst,
                                           int imgToDenoiseIndex, int temporalWindowSize,
                                           float h, float hForColorComponents,
                                           int templateWindowSize, int searchWindowSize)
{
    std::vector<Mat> srcImgs;
    _srcImgs.getMatVector(srcImgs);

    fastNlMeansDenoisingMultiCheckPreconditions(
        srcImgs, imgToDenoiseIndex,
        temporalWindowSize, templateWindowSize, searchWindowSize);

    _dst.create(srcImgs[0].size(), srcImgs[0].type());
    Mat dst = _dst.getMat();

    int type = srcImgs[0].type(), depth = CV_MAT_DEPTH(type);
    int src_imgs_size = static_cast<int>(srcImgs.size());

    if (type != CV_8UC3)
    {
        CV_Error(Error::StsBadArg, "Type of input images should be CV_8UC3!");
        return;
    }

    int from_to[] = { 0,0, 1,1, 2,2 };

    // TODO convert only required images
    std::vector<Mat> src_lab(src_imgs_size);
    std::vector<Mat> l(src_imgs_size);
    std::vector<Mat> ab(src_imgs_size);
    for (int i = 0; i < src_imgs_size; i++)
    {
        src_lab[i] = Mat::zeros(srcImgs[0].size(), type);
        l[i] = Mat::zeros(srcImgs[0].size(), CV_MAKE_TYPE(depth, 1));
        ab[i] = Mat::zeros(srcImgs[0].size(), CV_MAKE_TYPE(depth, 2));
        cvtColor(srcImgs[i], src_lab[i], COLOR_LBGR2Lab);

        Mat l_ab[] = { l[i], ab[i] };
        mixChannels(&src_lab[i], 1, l_ab, 2, from_to, 3);
    }

    Mat dst_l;
    Mat dst_ab;

    fastNlMeansDenoisingMulti(
        l, dst_l, imgToDenoiseIndex, temporalWindowSize,
        h, templateWindowSize, searchWindowSize);

    fastNlMeansDenoisingMulti(
        ab, dst_ab, imgToDenoiseIndex, temporalWindowSize,
        hForColorComponents, templateWindowSize, searchWindowSize);

    Mat l_ab_denoised[] = { dst_l, dst_ab };
    Mat dst_lab(srcImgs[0].size(), srcImgs[0].type());
    mixChannels(l_ab_denoised, 2, &dst_lab, 1, from_to, 3);

    cvtColor(dst_lab, dst, COLOR_Lab2LBGR);
}

void cv::halNlMeansDenoising( InputArray _src, OutputArray _dst, float h )
{
    // Prepare InputArray src
    const Mat src = _src.getMat();
    CV_Assert( !src.empty() );
    CV_Assert( src.type() == CV_8UC3 ); // TODO: Support more types, 8UC1 and C4 for Bayer

    // Prepare OutputArray dst
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    // NOTE: templateWindowSize = 7
    //       searchWindowSize   = 21
    // TODO: Ask @vpisarev, are these defaults fine?
    //       @ref mentions 7, 35 for colour images
    // @ref http://www.ipol.im/pub/art/2011/bcm_nlm/
    const int T  = 7,         // Template window width
              hT = T / 2,     // 3
              S  = 21,        // Search window width
              N  = S - T + 1, // Neighbour search width
              hN = N / 2;

    // OPTIMISATION TODOS:
    // - exploit symmetry in sum_abs(P_a - P_b) == sum_abs(P_b - P_a)

    // STEP 0) Calculate weight lookup table W
    //             Wij = exp( -dij / (3*h*h) )
    //         NOTE: dij = sum_abs(P_i - P_j)
    double max_d = 255*T*T,
           fac = - 1.f / (9*h*h);
    std::vector<float> W((int)max_d);
    for ( size_t i = 0; i < W.size(); i++ )
    {
        float w = std::exp(fac*i);
        W[i] = w;
        if ( w < 1e-3 ) // TODO: Make parameter
        {
            max_d = i;
            W.resize(i);
        }

    }

    int Y = src.rows,
        X = src.cols;

    // Intermediate structures
    Array2d<unsigned int> S_d(Y, X); // Sum of differences
    Mat Z = Mat::zeros(src.size(), CV_32FC1);
    Mat O = Mat::zeros(src.size(), CV_32FC3);
    dst = Scalar(0);

    // For each search window translation
    for ( int sy = -hN; sy <= hN; sy++ )
    for ( int sx = -hN; sx <= hN; sx++ )
    {
        // Build S_d
        for ( int y = 0; y < Y; y++ )
        for ( int x = 0; x < X; x++ )
        {
            // Offset pixel position
            int _y = y + sy,
                _x = x + sx;

            // BORDER_DEFAULT: gfedcb|abcdefgh|gfedcba
            // Handle top/left border
            _y = _y < 0 ? -_y : _y;
            _x = _x < 0 ? -_x : _x;
            // Handle bottom/right border
            _y = _y < Y ? _y : Y - (_y - Y) - 2;
            _x = _x < X ? _x : X - (_x - X) - 2;

            // Calculate distance value
            Vec3b p  = src.at<Vec3b>( y,  x),
                  _p = src.at<Vec3b>(_y, _x);
            unsigned int d = std::abs(p[0] - _p[0]) +
                             std::abs(p[1] - _p[1]) +
                             std::abs(p[2] - _p[2]);

            // Accumulate to form summed area table
            if ( x == 0 && y == 0 )
            {
                S_d.row_ptr(y)[x] = d;
            }
            else if ( x == 0 )
            {
                S_d.row_ptr(y)[x] = d + S_d.row_ptr(y-1)[x];
            }
            else if ( y == 0 )
            {
                S_d.row_ptr(y)[x] = d + S_d.row_ptr(y)[x-1];
            }
            else
            {
                S_d.row_ptr(y)[x] = d + S_d.row_ptr(y-1)[x] -
                    S_d.row_ptr(y-1)[x-1] + S_d.row_ptr(y)[x-1];
            }
        }

        for ( int y = 0; y < Y; y++ )
        for ( int x = 0; x < X; x++ )
        {
            // Compute weights
            int _y = y + sy,
                _x = x + sx;
            if ( _y < 0 || _x < 0 || _y >= Y || _x >= X ) continue;

            int px0 = MAX(x-hT-2, 0), px1 = MIN(x+hT, X-1),
                py0 = MAX(y-hT-2, 0), py1 = MIN(y+hT, Y-1);
            unsigned int d = (S_d.row_ptr(py1)[px1] - S_d.row_ptr(py1)[px0]) +
                             (S_d.row_ptr(py0)[px0] - S_d.row_ptr(py0)[px1]);
            if ( d < max_d )
            {
                float w = W[d];
                Vec3b i = src.at<Vec3b>(_y, _x);
                Vec3f o = O.at<Vec3f>(y, x);
                o[0] += w*i[0];
                o[1] += w*i[1];
                o[2] += w*i[2];
                O.at<Vec3f>(y, x) = o;
                Z.at<float>(y, x) += w;
            }
        }
    }

    // Apply to destination image
    for ( int y = 0; y < Y; y++ )
    for ( int x = 0; x < X; x++ )
    {
        Vec3f o = O.at<Vec3f>(y, x);
        float z = Z.at<float>(y, x);
        dst.at<Vec3b>(y, x) = o / z;
    }
}
