// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

class FilterData
{
public:
    FilterData(uchar *_kernel_data, size_t _kernel_step, int _kernel_type, int _src_type, int _dst_type, int _borderType,
        int _kernel_width, int _kernel_height, int _max_width, int _max_height, double _delta, int _anchor_x, int _anchor_y)
        : kernel_data(_kernel_data), kernel_step(_kernel_step), kernel_type(_kernel_type), src_type(_src_type), dst_type(_dst_type), borderType(_borderType),
        kernel_width(_kernel_width), kernel_height(_kernel_height), max_width(_max_width), max_height(_max_height), delta(_delta), anchor_x(_anchor_x), anchor_y(_anchor_y)
    {
    }

    uchar *kernel_data;
    size_t kernel_step; // bytes between rows(height)
    int kernel_type, src_type, dst_type, borderType;
    int kernel_width, kernel_height;
    int max_width, max_height;
    double delta;
    int anchor_x, anchor_y;
    std::vector<uchar> coords;
    std::vector<float> coeffs;
    int nz;
    std::vector<uchar> padding;
};

static int countNonZero(const FilterData* ctx)
{
    int i, j, nz = 0;
    const uchar* ker_row = ctx->kernel_data;
    for( i = 0; i < ctx->kernel_height; i++, ker_row += ctx->kernel_step )
    {
        for( j = 0; j < ctx->kernel_width; j++ )
        {
            if( ((float*)ker_row)[j] != 0.0 )
                nz++;
        }
    }
    return nz;
}

static void preprocess2DKernel(FilterData* ctx)
{
    int i, j, k, nz = countNonZero(ctx), ktype = ctx->kernel_type;
    if(nz == 0)
        nz = 1; // (0, 0) == 0 by default
    ndsrvp_assert( ktype == CV_32F );

    ctx->coords.resize(nz * 2);
    ctx->coeffs.resize(nz);

    const uchar* ker_row = ctx->kernel_data;
    for( i = k = 0; i < ctx->kernel_height; i++, ker_row += ctx->kernel_step )
    {
        for( j = 0; j < ctx->kernel_width; j++ )
        {
            float val = ((float*)ker_row)[j];
            if( val == 0.0 )
                continue;
            ctx->coords[k * 2] = j;
            ctx->coords[k * 2 + 1] = i;
            ctx->coeffs[k++] = val;
        }
    }

    ctx->nz = k;
}

int filterInit(cvhalFilter2D **context,
    uchar *kernel_data, size_t kernel_step,
    int kernel_type, int kernel_width,
    int kernel_height, int max_width, int max_height,
    int src_type, int dst_type, int borderType,
    double delta, int anchor_x, int anchor_y,
    bool allowSubmatrix, bool allowInplace)
{
    int sdepth = CV_MAT_DEPTH(src_type), ddepth = CV_MAT_DEPTH(dst_type);
    int cn = CV_MAT_CN(src_type), kdepth = kernel_type;

    (void)allowSubmatrix;
    (void)allowInplace;

    if(delta - (int)delta != 0.0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(kdepth != CV_32F || (sdepth != CV_8U && sdepth != CV_16U) || ddepth != sdepth)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    FilterData *ctx = new FilterData(kernel_data, kernel_step, kernel_type, src_type, dst_type, borderType,
        kernel_width, kernel_height, max_width, max_height, delta, anchor_x, anchor_y);

    *context = (cvhalFilter2D*)ctx;

    ndsrvp_assert(cn == CV_MAT_CN(dst_type) && ddepth >= sdepth);

    preprocess2DKernel(ctx);

    return CV_HAL_ERROR_OK;
}

int filter(cvhalFilter2D *context,
    const uchar *src_data, size_t src_step,
    uchar *dst_data, size_t dst_step,
    int width, int height,
    int full_width, int full_height,
    int offset_x, int offset_y)
{
    FilterData *ctx = (FilterData*)context;

    int cn = CV_MAT_CN(ctx->src_type);
    int cnes = CV_ELEM_SIZE(ctx->src_type);
    int ddepth = CV_MAT_DEPTH(ctx->dst_type);
    float delta_sat = (uchar)(ctx->delta);
    if(ddepth == CV_8U)
        delta_sat = (float)saturate_cast<uchar>(ctx->delta);
    else if(ddepth == CV_16U)
        delta_sat = (float)saturate_cast<ushort>(ctx->delta);

    // fetch original image data
    const uchar *ogn_data = src_data - offset_y * src_step - offset_x * cnes;
    int ogn_step = src_step;

    // ROI fully used in the computation
    int cal_width = width + ctx->kernel_width - 1;
    int cal_height = height + ctx->kernel_height - 1;
    int cal_x = offset_x - ctx->anchor_x;
    int cal_y = offset_y - ctx->anchor_y;

    // calculate source border
    ctx->padding.resize(cal_width * cal_height * cnes);
    uchar* pad_data = &ctx->padding[0];
    int pad_step = cal_width * cnes;

    uchar* pad_ptr;
    const uchar* ogn_ptr;
    std::vector<uchar> vec_zeros(cnes, 0);
    for(int i = 0; i < cal_height; i++)
    {
        int y = borderInterpolate(i + cal_y, full_height, ctx->borderType);
        if(y < 0) {
            memset(pad_data + i * pad_step, 0, cnes * cal_width);
            continue;
        }

        // left border
        int j = 0;
        int16x4_t vj = {0, 1, 2, 3};
        vj += saturate_cast<short>(cal_x);
        for(; j + cal_x < -4; j += 4, vj += 4)
        {
            int16x4_t vx = borderInterpolate_vector(vj, full_width, ctx->borderType);
            for(int k = 0; k < 4; k++) {
                if(vx[k] < 0) // border constant return value -1
                    ogn_ptr = &vec_zeros[0];
                else
                    ogn_ptr = ogn_data + y * ogn_step + vx[k] * cnes;
                pad_ptr = pad_data + i * pad_step + (j + k) * cnes;
                memcpy(pad_ptr, ogn_ptr, cnes);
            }
        }
        for(; j + cal_x < 0; j++)
        {
            int x = borderInterpolate(j + cal_x, full_width, ctx->borderType);
            if(x < 0) // border constant return value -1
                ogn_ptr = &vec_zeros[0];
            else
                ogn_ptr = ogn_data + y * ogn_step + x * cnes;
            pad_ptr = pad_data + i * pad_step + j * cnes;
            memcpy(pad_ptr, ogn_ptr, cnes);
        }

        // center
        int rborder = MIN(cal_width, full_width - cal_x);
        ogn_ptr = ogn_data + y * ogn_step + (j + cal_x) * cnes;
        pad_ptr = pad_data + i * pad_step + j * cnes;
        memcpy(pad_ptr, ogn_ptr, cnes * (rborder - j));

        // right border
        j = rborder;
        vj = (int16x4_t){0, 1, 2, 3} + saturate_cast<short>(cal_x + rborder);
        for(; j <= cal_width - 4; j += 4, vj += 4)
        {
            int16x4_t vx = borderInterpolate_vector(vj, full_width, ctx->borderType);
            for(int k = 0; k < 4; k++) {
                if(vx[k] < 0) // border constant return value -1
                    ogn_ptr = &vec_zeros[0];
                else
                    ogn_ptr = ogn_data + y * ogn_step + vx[k] * cnes;
                pad_ptr = pad_data + i * pad_step + (j + k) * cnes;
                memcpy(pad_ptr, ogn_ptr, cnes);
            }
        }
        for(; j < cal_width; j++)
        {
            int x = borderInterpolate(j + cal_x, full_width, ctx->borderType);
            if(x < 0) // border constant return value -1
                ogn_ptr = &vec_zeros[0];
            else
                ogn_ptr = ogn_data + y * ogn_step + x * cnes;
            pad_ptr = pad_data + i * pad_step + j * cnes;
            memcpy(pad_ptr, ogn_ptr, cnes);
        }
    }

    // prepare the pointers
    int i, k, count, nz = ctx->nz;
    const uchar* ker_pts = &ctx->coords[0];
    const float* ker_cfs = &ctx->coeffs[0];

    if( ddepth == CV_8U )
    {
        std::vector<uchar*> src_ptrarr;
        src_ptrarr.resize(nz);
        uchar** src_ptrs = &src_ptrarr[0];
        uchar* dst_row = dst_data;
        uchar* pad_row = pad_data;

        for( count = 0; count < height; count++, dst_row += dst_step, pad_row += pad_step )
        {
            for( k = 0; k < nz; k++ )
                src_ptrs[k] = (uchar*)pad_row + ker_pts[k * 2 + 1] * pad_step + ker_pts[k * 2] * cnes;

            i = 0;
            for( ; i <= width * cnes - 8; i += 8 )
            {
                float vs0[8] = {delta_sat, delta_sat, delta_sat, delta_sat, delta_sat, delta_sat, delta_sat, delta_sat};
                for( k = 0; k < nz; k++ ) {
                    float vker_cfs[8] = {ker_cfs[k], ker_cfs[k], ker_cfs[k], ker_cfs[k], ker_cfs[k], ker_cfs[k], ker_cfs[k], ker_cfs[k]};
                    // experimental code
                    // ndsrvp_f32_u8_mul8(vker_cfs, *(unsigned long*)(src_ptrs[k] + i), vker_cfs);
                    // ndsrvp_f32_add8(vs0, vker_cfs, vs0);
                    vs0[0] += vker_cfs[0] * src_ptrs[k][i];
                    vs0[1] += vker_cfs[1] * src_ptrs[k][i + 1];
                    vs0[2] += vker_cfs[2] * src_ptrs[k][i + 2];
                    vs0[3] += vker_cfs[3] * src_ptrs[k][i + 3];
                    vs0[4] += vker_cfs[4] * src_ptrs[k][i + 4];
                    vs0[5] += vker_cfs[5] * src_ptrs[k][i + 5];
                    vs0[6] += vker_cfs[6] * src_ptrs[k][i + 6];
                    vs0[7] += vker_cfs[7] * src_ptrs[k][i + 7];
                }
                dst_row[i] = saturate_cast<uchar>(vs0[0]);
                dst_row[i + 1] = saturate_cast<uchar>(vs0[1]);
                dst_row[i + 2] = saturate_cast<uchar>(vs0[2]);
                dst_row[i + 3] = saturate_cast<uchar>(vs0[3]);
                dst_row[i + 4] = saturate_cast<uchar>(vs0[4]);
                dst_row[i + 5] = saturate_cast<uchar>(vs0[5]);
                dst_row[i + 6] = saturate_cast<uchar>(vs0[6]);
                dst_row[i + 7] = saturate_cast<uchar>(vs0[7]);
            }
            for( ; i < width * cnes; i++ )
            {
                float s0 = delta_sat;
                for( k = 0; k < nz; k++ ) {
                    s0 += ker_cfs[k] * src_ptrs[k][i];
                }
                dst_row[i] = saturate_cast<uchar>(s0);
            }
        }
    }
    else if( ddepth == CV_16U )
    {
        std::vector<ushort*> src_ptrarr;
        src_ptrarr.resize(nz);
        ushort** src_ptrs = &src_ptrarr[0];
        uchar* dst_row = dst_data;
        uchar* pad_row = pad_data;

        for( count = 0; count < height; count++, dst_row += dst_step, pad_row += pad_step )
        {
            for( k = 0; k < nz; k++ )
                src_ptrs[k] = (ushort*)((uchar*)pad_row + ker_pts[k * 2 + 1] * pad_step + ker_pts[k * 2] * cnes);

            i = 0;
            for( ; i <= width * cn - 4; i += 4 )
            {
                float vs0[8] = {delta_sat, delta_sat, delta_sat, delta_sat};
                for( k = 0; k < nz; k++ ) {
                    float vker_cfs[8] = {ker_cfs[k], ker_cfs[k], ker_cfs[k], ker_cfs[k]};
                    vs0[0] += vker_cfs[0] * src_ptrs[k][i];
                    vs0[1] += vker_cfs[1] * src_ptrs[k][i + 1];
                    vs0[2] += vker_cfs[2] * src_ptrs[k][i + 2];
                    vs0[3] += vker_cfs[3] * src_ptrs[k][i + 3];
                }
                ushort* dst_row_ptr = (ushort*)dst_row;
                dst_row_ptr[i] = saturate_cast<ushort>(vs0[0]);
                dst_row_ptr[i + 1] = saturate_cast<ushort>(vs0[1]);
                dst_row_ptr[i + 2] = saturate_cast<ushort>(vs0[2]);
                dst_row_ptr[i + 3] = saturate_cast<ushort>(vs0[3]);
            }
            for( ; i < width * cn; i++ )
            {
                float s0 = delta_sat;
                for( k = 0; k < nz; k++ ) {
                    s0 += ker_cfs[k] * src_ptrs[k][i];
                }
                ((ushort*)dst_row)[i] = saturate_cast<ushort>(s0);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

int filterFree(cvhalFilter2D *context) {
    FilterData *ctx = (FilterData*)context;
    delete ctx;
    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
