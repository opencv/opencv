// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

int remap32f(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height, float* mapx, size_t mapx_step,
    float* mapy, size_t mapy_step, int interpolation, int border_type, const double border_value[4])
{
    const bool isRelative = ((interpolation & CV_HAL_WARP_RELATIVE_MAP) != 0);
    interpolation &= ~CV_HAL_WARP_RELATIVE_MAP;

    if( interpolation == CV_HAL_INTER_AREA )
        interpolation = CV_HAL_INTER_LINEAR;

    if( interpolation != CV_HAL_INTER_NEAREST )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // only CV_8U
    if( (src_type & CV_MAT_DEPTH_MASK) != CV_8U )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int cn = CV_MAT_CN(src_type);

    src_step /= sizeof(uchar);
    dst_step /= sizeof(uchar);

    // mapping CV_32FC1
    mapx_step /= sizeof(float);
    mapy_step /= sizeof(float);

    // border
    uchar border_const[CV_CN_MAX];
    for( int k = 0; k < CV_CN_MAX; k++ )
        border_const[k] = saturate_cast<uchar>(border_value[k & 3]);

    // divide into blocks
    const int BLOCK_SIZE = 1024;
    int x, y, x1, y1;
    std::array<short, BLOCK_SIZE * BLOCK_SIZE * 2> aXY;
    short* XY = aXY.data();
    size_t XY_step = BLOCK_SIZE * 2;

    // vectorize
    const int32x2_t src_wh = {src_width, src_height};
    const int32x2_t arr_index = {cn, (int)src_step};

    for (y = 0; y < dst_height; y += BLOCK_SIZE)
    {
        int dy = std::min(BLOCK_SIZE, dst_height - y);
        for (x = 0; x < dst_width; x += BLOCK_SIZE)
        {
            const int off_y = isRelative ? y : 0;
            const int off_x = isRelative ? x : 0;
            const int32x2_t voff = {off_x, off_y};

            int dx = std::min(BLOCK_SIZE, dst_width - x);
            // prepare mapping data XY
            for (y1 = 0; y1 < dy; y1++)
            {
                short* rXY = XY + y1 * XY_step;
                const float* sX = mapx + (y + y1) * mapx_step + x;
                const float* sY = mapy + (y + y1) * mapy_step + x;
                for (x1 = 0; x1 < dx; x1++)
                {
                    rXY[x1 * 2] = saturate_cast<short>(sX[x1]);
                    rXY[x1 * 2 + 1] = saturate_cast<short>(sY[x1]);
                }
            }

            // precalulate offset
            if(isRelative)
            {
                int16x8_t voff_x;
                int16x8_t voff_y = {0, 0, 1, 0, 2, 0, 3, 0};
                int16x8_t vones_x = {4, 0, 4, 0, 4, 0, 4, 0};
                int16x8_t vones_y = {0, 1, 0, 1, 0, 1, 0, 1};
                for(y1 = 0; y1 < BLOCK_SIZE; y1++, voff_y += vones_y)
                {
                    int16x8_t* vrXY = (int16x8_t*)(XY + y1 * XY_step);
                    for(x1 = 0, voff_x = voff_y; x1 < BLOCK_SIZE; x1 += 4, vrXY++, voff_x += vones_x)
                    {
                        *vrXY += voff_x;
                    }
                }
            }

            // process the block
            for( y1 = 0; y1 < dy; y1++ )
            {
                uchar* dst_row = dst_data + (y + y1) * dst_step + x * cn;
                const short* rXY = XY + y1 * XY_step;
                if( cn == 1 )
                {
                    for( x1 = 0; x1 < dx; x1++ )
                    {
                        int32x2_t vsxy = (int32x2_t){rXY[x1 * 2], rXY[x1 * 2 + 1]} + voff;
                        if( (long)((uint32x2_t)vsxy < (uint32x2_t)src_wh) == -1 )
                            dst_row[x1] = src_data[__nds__v_smar64(0, vsxy, arr_index)];
                        else
                        {
                            if( border_type == CV_HAL_BORDER_REPLICATE )
                            {
                                vsxy = vclip(vsxy, (int32x2_t){0, 0}, src_wh);
                                dst_row[x1] = src_data[__nds__v_smar64(0, vsxy, arr_index)];
                            }
                            else if( border_type == CV_HAL_BORDER_CONSTANT )
                                dst_row[x1] = border_const[0];
                            else if( border_type != CV_HAL_BORDER_TRANSPARENT )
                            {
                                vsxy[0] = borderInterpolate(vsxy[0], src_width, border_type);
                                vsxy[1] = borderInterpolate(vsxy[1], src_height, border_type);
                                dst_row[x1] = src_data[__nds__v_smar64(0, vsxy, arr_index)];
                            }
                        }
                    }
                }
                else
                {
                    uchar* dst_ptr = dst_row;
                    for(x1 = 0; x1 < dx; x1++, dst_ptr += cn )
                    {
                        int32x2_t vsxy = (int32x2_t){rXY[x1 * 2], rXY[x1 * 2 + 1]} + voff;
                        const uchar *src_ptr;
                        if( (long)((uint32x2_t)vsxy < (uint32x2_t)src_wh) == -1 )
                        {
                            if( cn == 3 )
                            {
                                src_ptr = (uchar*)__nds__v_smar64((long)src_data, vsxy, arr_index);
                                dst_ptr[0] = src_ptr[0]; dst_ptr[1] = src_ptr[1]; dst_ptr[2] = src_ptr[2];
                                // performance loss, commented out
                                // *(unsigned*)dst_ptr = __nds__bpick(*(unsigned*)dst_ptr, *(unsigned*)src_ptr, 0xFF000000);
                            }
                            else if( cn == 4 )
                            {
                                src_ptr = (uchar*)__nds__v_smar64((long)src_data, vsxy, arr_index);
                                *(uint8x4_t*)dst_ptr = *(uint8x4_t*)src_ptr;
                            }
                            else
                            {
                                src_ptr = (uchar*)__nds__v_smar64((long)src_data, vsxy, arr_index);
                                int k = cn;
                                for(; k >= 8; k -= 8, dst_ptr += 8, src_ptr += 8)
                                    *(uint8x8_t*)dst_ptr = *(uint8x8_t*)src_ptr;
                                while( k-- )
                                    dst_ptr[k] = src_ptr[k];
                            }
                        }
                        else if( border_type != CV_HAL_BORDER_TRANSPARENT )
                        {
                            if( border_type == CV_HAL_BORDER_REPLICATE )
                            {
                                vsxy = vclip(vsxy, (int32x2_t){0, 0}, src_wh);
                                src_ptr = (uchar*)__nds__v_smar64((long)src_data, vsxy, arr_index);
                            }
                            else if( border_type == CV_HAL_BORDER_CONSTANT )
                                src_ptr = &border_const[0];
                            else
                            {
                                vsxy[0] = borderInterpolate(vsxy[0], src_width, border_type);
                                vsxy[1] = borderInterpolate(vsxy[1], src_height, border_type);
                                src_ptr = (uchar*)__nds__v_smar64((long)src_data, vsxy, arr_index);
                            }
                            int k = cn;
                            for(; k >= 8; k -= 8, dst_ptr += 8, src_ptr += 8)
                                *(uint8x8_t*)dst_ptr = *(uint8x8_t*)src_ptr;
                            while( k-- )
                                dst_ptr[k] = src_ptr[k];
                        }
                    }
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
