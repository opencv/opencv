// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

static void bilateralFilterProcess(uchar* dst_data, size_t dst_step, uchar* pad_data, size_t pad_step,
    int width, int height, int cn, int radius, int maxk,
    int* space_ofs, float *space_weight, float *color_weight)
{
    int i, j, k;

    for( i = 0; i < height; i++ )
    {
        const uchar* sptr = pad_data + (i + radius) * pad_step + radius * cn;
        uchar* dptr = dst_data + i * dst_step;

        if( cn == 1 )
        {
            std::vector<float> buf(width + width, 0.0);
            float *sum = &buf[0];
            float *wsum = sum + width;
            k = 0;
            for(; k <= maxk-4; k+=4)
            {
                const uchar* ksptr0 = sptr + space_ofs[k];
                const uchar* ksptr1 = sptr + space_ofs[k+1];
                const uchar* ksptr2 = sptr + space_ofs[k+2];
                const uchar* ksptr3 = sptr + space_ofs[k+3];
                j = 0;
                for (; j < width; j++)
                {
                    int rval = sptr[j];

                    int val = ksptr0[j];
                    float w = space_weight[k] * color_weight[std::abs(val - rval)];
                    wsum[j] += w;
                    sum[j] += val * w;

                    val = ksptr1[j];
                    w = space_weight[k+1] * color_weight[std::abs(val - rval)];
                    wsum[j] += w;
                    sum[j] += val * w;

                    val = ksptr2[j];
                    w = space_weight[k+2] * color_weight[std::abs(val - rval)];
                    wsum[j] += w;
                    sum[j] += val * w;

                    val = ksptr3[j];
                    w = space_weight[k+3] * color_weight[std::abs(val - rval)];
                    wsum[j] += w;
                    sum[j] += val * w;
                }
            }
            for(; k < maxk; k++)
            {
                const uchar* ksptr = sptr + space_ofs[k];
                j = 0;
                for (; j < width; j++)
                {
                    int val = ksptr[j];
                    float w = space_weight[k] * color_weight[std::abs(val - sptr[j])];
                    wsum[j] += w;
                    sum[j] += val * w;
                }
            }
            j = 0;
            for (; j < width; j++)
            {
                // overflow is not possible here => there is no need to use cv::saturate_cast
                ndsrvp_assert(fabs(wsum[j]) > 0);
                dptr[j] = (uchar)(sum[j] / wsum[j] + 0.5);
            }
        }
        else
        {
            ndsrvp_assert( cn == 3 );
            std::vector<float> buf(width * 3 + width);
            float *sum_b = &buf[0];
            float *sum_g = sum_b + width;
            float *sum_r = sum_g + width;
            float *wsum = sum_r + width;
            k = 0;
            for(; k <= maxk-4; k+=4)
            {
                const uchar* ksptr0 = sptr + space_ofs[k];
                const uchar* ksptr1 = sptr + space_ofs[k+1];
                const uchar* ksptr2 = sptr + space_ofs[k+2];
                const uchar* ksptr3 = sptr + space_ofs[k+3];
                const uchar* rsptr = sptr;
                j = 0;
                for(; j < width; j++, rsptr += 3, ksptr0 += 3, ksptr1 += 3, ksptr2 += 3, ksptr3 += 3)
                {
                    int rb = rsptr[0], rg = rsptr[1], rr = rsptr[2];

                    int b = ksptr0[0], g = ksptr0[1], r = ksptr0[2];
                    float w = space_weight[k] * color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                    wsum[j] += w;
                    sum_b[j] += b * w; sum_g[j] += g * w; sum_r[j] += r * w;

                    b = ksptr1[0]; g = ksptr1[1]; r = ksptr1[2];
                    w = space_weight[k+1] * color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                    wsum[j] += w;
                    sum_b[j] += b * w; sum_g[j] += g * w; sum_r[j] += r * w;

                    b = ksptr2[0]; g = ksptr2[1]; r = ksptr2[2];
                    w = space_weight[k+2] * color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                    wsum[j] += w;
                    sum_b[j] += b * w; sum_g[j] += g * w; sum_r[j] += r * w;

                    b = ksptr3[0]; g = ksptr3[1]; r = ksptr3[2];
                    w = space_weight[k+3] * color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                    wsum[j] += w;
                    sum_b[j] += b * w; sum_g[j] += g * w; sum_r[j] += r * w;
                }
            }
            for(; k < maxk; k++)
            {
                const uchar* ksptr = sptr + space_ofs[k];
                const uchar* rsptr = sptr;
                j = 0;
                for(; j < width; j++, ksptr += 3, rsptr += 3)
                {
                    int b = ksptr[0], g = ksptr[1], r = ksptr[2];
                    float w = space_weight[k] * color_weight[std::abs(b - rsptr[0]) + std::abs(g - rsptr[1]) + std::abs(r - rsptr[2])];
                    wsum[j] += w;
                    sum_b[j] += b * w; sum_g[j] += g * w; sum_r[j] += r * w;
                }
            }
            j = 0;
            for(; j < width; j++)
            {
                ndsrvp_assert(fabs(wsum[j]) > 0);
                wsum[j] = 1.f / wsum[j];
                *(dptr++) = (uchar)(sum_b[j] * wsum[j] + 0.5);
                *(dptr++) = (uchar)(sum_g[j] * wsum[j] + 0.5);
                *(dptr++) = (uchar)(sum_r[j] * wsum[j] + 0.5);
            }
        }
    }
}

int bilateralFilter(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step, int width, int height, int depth,
    int cn, int d, double sigma_color, double sigma_space, int border_type)
{
    if( depth != CV_8U || !(cn == 1 || cn == 3) || src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int i, j, maxk, radius;

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color * sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space * sigma_space);

    if( d <= 0 )
        radius = (int)(sigma_space * 1.5 + 0.5);
    else
        radius = d / 2;

    radius = MAX(radius, 1);
    d = radius * 2 + 1;

    // no enough submatrix info
    // fetch original image data
    const uchar *ogn_data = src_data;
    int ogn_step = src_step;

    // ROI fully used in the computation
    int cal_width = width + d - 1;
    int cal_height = height + d - 1;
    int cal_x = 0 - radius; // negative if left border exceeded
    int cal_y = 0 - radius; // negative if top border exceeded

    // calculate source border
    std::vector<uchar> padding;
    padding.resize(cal_width * cal_height * cn);
    uchar* pad_data = &padding[0];
    int pad_step = cal_width * cn;

    uchar* pad_ptr;
    const uchar* ogn_ptr;
    std::vector<uchar> vec_zeros(cn, 0);
    for(i = 0; i < cal_height; i++)
    {
        int y = borderInterpolate(i + cal_y, height, border_type);
        if(y < 0) {
            memset(pad_data + i * pad_step, 0, cn * cal_width);
            continue;
        }

        // left border
        j = 0;
        for(; j + cal_x < 0; j++)
        {
            int x = borderInterpolate(j + cal_x, width, border_type);
            if(x < 0) // border constant return value -1
                ogn_ptr = &vec_zeros[0];
            else
                ogn_ptr = ogn_data + y * ogn_step + x * cn;
            pad_ptr = pad_data + i * pad_step + j * cn;
            memcpy(pad_ptr, ogn_ptr, cn);
        }

        // center
        int rborder = MIN(cal_width, width - cal_x);
        ogn_ptr = ogn_data + y * ogn_step + (j + cal_x) * cn;
        pad_ptr = pad_data + i * pad_step + j * cn;
        memcpy(pad_ptr, ogn_ptr, cn * (rborder - j));

        // right border
        j = rborder;
        for(; j < cal_width; j++)
        {
            int x = borderInterpolate(j + cal_x, width, border_type);
            if(x < 0) // border constant return value -1
                ogn_ptr = &vec_zeros[0];
            else
                ogn_ptr = ogn_data + y * ogn_step + x * cn;
            pad_ptr = pad_data + i * pad_step + j * cn;
            memcpy(pad_ptr, ogn_ptr, cn);
        }
    }

    std::vector<float> _color_weight(cn * 256);
    std::vector<float> _space_weight(d * d);
    std::vector<int> _space_ofs(d * d);
    float* color_weight = &_color_weight[0];
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // initialize color-related bilateral filter coefficients

    for( i = 0; i < 256 * cn; i++ )
        color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
    {
        j = -radius;

        for( ; j <= radius; j++ )
        {
            double r = std::sqrt((double)i * i + (double)j * j);
            if( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
            space_ofs[maxk++] = (int)(i * pad_step + j * cn);
        }
    }

    bilateralFilterProcess(dst_data, dst_step, pad_data, pad_step, width, height, cn, radius, maxk, space_ofs, space_weight, color_weight);

    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
