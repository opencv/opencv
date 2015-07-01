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
    const int T  = 7,     // Template window width
              hT = T / 2, // 3
              S  = 21,    // Search window width
              N  = S - T, // Neighbour coordinates width
              hN = N / 2; // 7

    // OPTIMISATION TODOS:
    // - exploit symmetry in sum_abs(P_a - P_b) == sum_abs(P_b - P_a)

    // STEP 0) Calculate weight lookup table W
    //             sigma = h / 0.4
    //             Wij = exp( -max(dij - 2*sigma*sigma, 0) / (h*h) )
    //         NOTE: dij = sum_abs(P_i - P_j)
    //         @ref http://www.ipol.im/pub/art/2011/bcm_nlm/
    //
    // TODO: Ask @vpisarev, is this correct?
    //       OpenCV uses:
    //           double w = std::exp(-dij*dij / (h*h * nchannels));
    double max_d = 255*T*T,
           fac = - 1.f / (3*h*h), w;
    std::vector<float> W((int)max_d);
    for ( size_t i = 0; i < W.size(); i++ )
    {
        w = std::exp(fac*i);
        W[i] = w;
        if ( w < 1e-3 ) // TODO: Make parameter
        {
            max_d = i;
            W.resize(i + 1);
            break;
        }
    }

    int I = src.rows,
        J = src.cols,
        i = 0,
        j = 0;

    const Vec3b *p_src, *p_c, *p_n;
    Vec3b *p_dst, p;

    int x0_c, y0_c, dx_c, dy_c, x0_n, y0_n, x_c, x_n, d;
    Mat D = Mat(src.size(), CV_16UC1);
    double sum_w, sum_pw0, sum_pw1, sum_pw2;

    // For each pixel
    for ( i = 0; i < I; i++ )
    {
        p_src = src.ptr<Vec3b>(i);
        p_dst = dst.ptr<Vec3b>(i);

        y0_c = std::min(std::max(0, i-hT), I-T-1);
        dy_c = i-hT - y0_c;

        for ( j = 0; j < J; j++ )
        {

            // STEP 1) Take 7x7=49/c patch (P_c)
            x0_c = std::min(std::max(0, j-hT), J-T-1);
            dx_c = j-hT - x0_c;

            // Add self
            sum_pw0 = p_src[j][0];
            sum_pw1 = p_src[j][1];
            sum_pw2 = p_src[j][2];
            sum_w = 1.f;

            // For each neighbour patch (P_n) in 21x21=441/c region
            for ( int ni = -hN; ni <= hN; ni++ )
            {
                y0_n = i + ni - hT;
                if ( y0_n < 0 || y0_n >= I-T ) continue;

                for ( int nj = -hN; nj <= hN; nj++ )
                {
                    // Get starting point for P_n, current neighbour patch
                    x0_n = j + nj - hT;
                    if ( x0_n < 0 || x0_n >= J-T || ( ni == 0 && nj == 0 ) )
                        continue;

                    // STEP 2) d_cn = sum_abs(P_c - P_n) = sum_abs(P_n - P_c) = d_nc
                    d = 0;
                    for ( int pi = 0; pi < T; pi++ )
                    {
                        p_c = src.ptr<Vec3b>(y0_c+pi);
                        p_n = src.ptr<Vec3b>(y0_n+ni+pi);
                        x_c = x0_c;
                        x_n = x0_n + nj;
                        for ( int pj = 0; pj < T; pj++ )
                        {
                            d += (
                                     std::abs(p_n[x_n][0] - p_c[x_c][0]) +
                                     std::abs(p_n[x_n][1] - p_c[x_c][1]) +
                                     std::abs(p_n[x_n][2] - p_c[x_c][2])
                                 );
                            x_c++;
                            x_n++;
                        }
                    }

                    // STEP 3) Lookup w_cn in table W
                    d /= 3;
                    w = d > max_d ? 0 : W[d];

                    p = src.at<Vec3b>(y0_n+hT+dy_c, x0_n+hT+dx_c);
                    sum_pw0 += w*p[0]; sum_pw1 += w*p[1]; sum_pw2 += w*p[2];
                    sum_w += w;
                }
            }

            // STEP 4) Calculate denoised pixel sum(p_c*w_cn)/sum(w_cn) for centre pixel p_c
            p_dst[j][0] = (uchar)(sum_pw0 / sum_w);
            p_dst[j][1] = (uchar)(sum_pw1 / sum_w);
            p_dst[j][2] = (uchar)(sum_pw2 / sum_w);
        }
    }
}
