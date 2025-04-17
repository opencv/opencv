// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#undef cv_hal_FAST
#define cv_hal_FAST cv::cv_hal_rvv::FAST

// #undef cv_hal_FAST_dense
// #define cv_hal_FAST_dense cv::cv_hal_rvv::FAST_dense


#include <riscv_vector.h>
#include "../../../modules/features2d/include/opencv2/features2d.hpp"

#include "opencv2/core/utils/buffer_area.private.hpp"

#include "opencv2/core/utils/logger.hpp"


// #include "../../../.vscode/intrinsic_funcs.hpp"
// #include "../../../.vscode/overloaded_intrinsic_funcs.hpp"

#include <cfloat>

namespace cv::cv_hal_rvv {

using RVV_VECTOR_TYPE = vuint8m4_t;


// Since uint16_t range is 0 to 65535, row stride should be less than 65535/6 = 10922
inline void makeOffsets(int16_t pixel[], vuint16m2_t& v_offset, int64_t row_stride, int patternSize)
{
    // set min element (pixel[8] = 0 + row_stride * -3) as the base addr
    uint16_t pixel_u[25];
    pixel_u[0] = row_stride * 6;
    pixel_u[1] = 1 + row_stride * 6;
    pixel_u[2] = 2 + row_stride * 5;
    pixel_u[3] = 3 + row_stride * 4;
    pixel_u[4] = 3 + row_stride * 3;
    pixel_u[5] = 3 + row_stride * 2;
    pixel_u[6] = 2 + row_stride * 1;
    pixel_u[7] = 1 + row_stride * 0;
    pixel_u[8] = 0 + row_stride * 0;
    pixel_u[9] = -1 + row_stride * 0;
    pixel_u[10] = -2 + row_stride * 1;
    pixel_u[11] = -3 + row_stride * 2;
    pixel_u[12] = -3 + row_stride * 3;
    pixel_u[13] = -3 + row_stride * 4;
    pixel_u[14] = -2 + row_stride * 5;
    pixel_u[15] = -1 + row_stride * 6;
    for (int i = 16; i < 25; i++)
    {
        pixel_u[i] = pixel_u[i - 16];
    }
    v_offset = __riscv_vle16_v_u16m2(pixel_u, 25);
    std::string msg = cv::format("pixel = ");
    for (int i = 0; i < 25; i++)
    {
        pixel[i] = pixel_u[i] - 3 * row_stride;
        msg += cv::format("%d ", pixel[i]);
    }
    CV_LOG_INFO(NULL, msg);
}

// Since int16_t range is -32768 to 32767, row stride should be less than 32768/3 = 10922
// vint16m2_t should contains 32 elements
// inline void makeOffsets(int16_t pixel[], vint16m2_t& v_offset, int64_t row_stride, int patternSize)
// {
//     pixel[0] = 0 + row_stride * 3;
//     pixel[1] = 1 + row_stride * 3;
//     pixel[2] = 2 + row_stride * 2;
//     pixel[3] = 3 + row_stride * 1;
//     pixel[4] = 3 + row_stride * 0;
//     pixel[5] = 3 + row_stride * -1;
//     pixel[6] = 2 + row_stride * -2;
//     pixel[7] = 1 + row_stride * -3;
//     pixel[8] = 0 + row_stride * -3;
//     pixel[9] = -1 + row_stride * -3;
//     pixel[10] = -2 + row_stride * -2;
//     pixel[11] = -3 + row_stride * -1;
//     pixel[12] = -3 + row_stride * 0;
//     pixel[13] = -3 + row_stride * 1;
//     pixel[14] = -2 + row_stride * 2;
//     pixel[15] = -1 + row_stride * 3;
//     v_offset = __riscv_vle16_v_i16m2(pixel, 16);
// }

template<typename T> inline T* alignPtr(T* ptr, size_t n=sizeof(T))
{
    return (T*)(((size_t)ptr + n-1) & -n);
}

inline uint8_t cornerScore(const uint8_t* ptr, const vuint16m2_t& v_offset, int64_t row_stride) 
{
    const uint32_t K = 8, N = 16 + K + 1;
    uint32_t k, v = ptr[0];
    
    int vl = __riscv_vsetvl_e16m2(N);

    // use vloxei16_v to indexed ordered load
    vint16m2_t v_c_pixel = __riscv_vmv_v_x_i16m2((int16_t)v, vl);
    // vloxei only support positive offset
    vuint8m1_t v_d_u8 = __riscv_vloxei16(ptr - 3 * row_stride, v_offset, vl);
    vuint16m2_t v_d_u16 = __riscv_vzext_vf2(v_d_u8, vl);
    vint16m2_t d = __riscv_vreinterpret_i16m2(v_d_u16);
    // for( k = 0; k < N; k++ )
    //     d[k] = (uint16_t)(v - ptr[pixel[k]]);
    d = __riscv_vsub_vv_i16m2(v_c_pixel, d, vl);

    vint16m2_t d_slide = __riscv_vmv_v(d, vl);
    
    vint16m2_t q0 = __riscv_vmv_v_x_i16m2((int16_t)(-1000), vl);
    vint16m2_t q1 = __riscv_vmv_v_x_i16m2((int16_t)(1000), vl);

    //k == 0
    vint16m2_t ak0 = __riscv_vmv_v(d, vl);
    vint16m2_t bk0 = __riscv_vmv_v(d, vl);

    for (int i = 0; i < 8; i++)
    {
        d_slide = __riscv_vslide1down(d, (int16_t)0, vl);
        ak0 = __riscv_vmin(ak0, d, vl);
        bk0 = __riscv_vmax(bk0, d, vl);
    }

    q0 = __riscv_vmax(q0, __riscv_vmin(ak0, d, vl), vl);
    q1 = __riscv_vmin(q1, __riscv_vmax(bk0, d, vl), vl);

    d_slide = __riscv_vslide1down(d, (int16_t)0, vl);
    q0 = __riscv_vmax(q0, __riscv_vmin(ak0, d_slide, vl), vl);
    q1 = __riscv_vmin(q1, __riscv_vmax(bk0, d_slide, vl), vl);

    q1 = __riscv_vrsub(q1, (int16_t)0, vl);
    q0 = __riscv_vmax(q0, q1, vl);

    vint16m1_t res = __riscv_vredmax(q0, __riscv_vmv_s_x_i16m1((int16_t)0, vl), vl);

    uint8_t result = (uint8_t)__riscv_vmv_x(res);

    return result;
}


inline int fast_16(const uchar* src_data, size_t src_step, int width, int height, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression) 
{

    const int patternSize = 16;
    const int K = patternSize/2, N = patternSize + K + 1;
    const int quarterPatternSize = patternSize/4;

    std::string msg;
    msg = cv::format("riscv fast_16: patternSize=%d, K=%d, N=%d, quarterPatternSize=%d", patternSize, K, N, quarterPatternSize);
    CV_LOG_INFO(NULL, msg);

    int i, j, k;
    int16_t pixel[25];
    vuint16m2_t v_offset;
    makeOffsets(pixel, v_offset, (int)src_step, patternSize);

    // threshold = std::min(std::max(threshold, 0), 255);
    // uchar threshold_tab[512];
    // for( i = -255; i <= 255; i++ )
    //     threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    // std::vector<uint8_t> _buf((width) * 3 * (sizeof(uint8_t)));
    // uint8_t* buf[3];
    // buf[0] = &_buf[0]; buf[1] = buf[0] + width; buf[2] = buf[1] + width;
    // std::vector<ptrdiff_t> _cpbuf((width + 1) * 3 * (sizeof(ptrdiff_t)));
    // ptrdiff_t* cpbuf[3];
    // cpbuf[0] = &_cpbuf[0]; cpbuf[1] = cpbuf[0] + width + 1; cpbuf[2] = cpbuf[1] + width + 1;
    // memset(buf[0], 0, width*3);

    uchar* buf[3] = { 0 };
    int* cpbuf[3] = { 0 };
    utils::BufferArea area;
    for (unsigned idx = 0; idx < 3; ++idx)
    {
        area.allocate(buf[idx], width);
        area.allocate(cpbuf[idx], width + 1);
    }
    area.commit();

    for (unsigned idx = 0; idx < 3; ++idx)
    {
        memset(buf[idx], 0, width);
    }

    int vlmax = __riscv_vsetvlmax_e8m8();
    vuint8m4_t v_c_delta = __riscv_vmv_v_x_u8m4(0x80, vlmax);
    vuint8m4_t v_c_threshold = __riscv_vmv_v_x_u8m4((char) threshold, vlmax);
    vuint8m4_t v_c_k = __riscv_vmv_v_x_u8m4((char)K, vlmax);
    vint8m4_t v_c_zero = __riscv_vmv_v_x_i8m4(0, vlmax);

    for( i = 3; i < height - 2; i++)
    {

        const uchar* ptr = src_data + i * src_step;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3] + 1;
        memset(curr, 0, width);
        int ncorners = 0;

        if( i < height - 3 )
        {
            j = 3;
            {
                int margin = width - 3;
                int vl = __riscv_vsetvl_e8m4(margin - j);
                for (; j < margin; j += vl, ptr += vl)
                {
                    msg = cv::format("Computing column %d", j);
                    CV_LOG_INFO(NULL, msg);

                    vl = __riscv_vsetvl_e8m4(margin - j);
                    vuint8m4_t v_pixels = __riscv_vle8_v_u8m4(ptr, vl);

                    // pixels add threshold
                    vuint8m4_t v_pat = __riscv_vsaddu(v_pixels, v_c_threshold, vl);
                    // pixels sub threshold
                    vuint8m4_t v_pst = __riscv_vssubu(v_pixels, v_c_threshold, vl);

                    vint8m4_t v0 = __riscv_vreinterpret_i8m4(__riscv_vxor(v_pat, v_c_delta, vl));
                    vint8m4_t v1 = __riscv_vreinterpret_i8m4(__riscv_vxor(v_pst, v_c_delta, vl));


                    v_pixels = __riscv_vle8_v_u8m4(ptr + pixel[0], vl);
                    vint8m4_t x0 = __riscv_vreinterpret_i8m4(__riscv_vsub(v_pixels, v_c_delta, vl));
                    v_pixels = __riscv_vle8_v_u8m4(ptr + pixel[quarterPatternSize], vl);
                    vint8m4_t x1 = __riscv_vreinterpret_i8m4(__riscv_vsub(v_pixels, v_c_delta, vl));
                    v_pixels = __riscv_vle8_v_u8m4(ptr + pixel[2*quarterPatternSize], vl);
                    vint8m4_t x2 = __riscv_vreinterpret_i8m4(__riscv_vsub(v_pixels, v_c_delta, vl));
                    v_pixels = __riscv_vle8_v_u8m4(ptr + pixel[3*quarterPatternSize], vl);
                    vint8m4_t x3 = __riscv_vreinterpret_i8m4(__riscv_vsub(v_pixels, v_c_delta, vl));

                    vbool2_t m0, m1;
                    m0 = __riscv_vmand(__riscv_vmslt(v0, x0, vl), __riscv_vmslt(v0, x1, vl), vl);
                    m1 = __riscv_vmand(__riscv_vmslt(x0, v1, vl), __riscv_vmslt(x1, v1, vl), vl);
                    m0 = __riscv_vmor(m0, __riscv_vmand(__riscv_vmslt(v0, x1, vl), __riscv_vmslt(v0, x2, vl), vl), vl);
                    m1 = __riscv_vmor(m1, __riscv_vmand(__riscv_vmslt(x1, v1, vl), __riscv_vmslt(x2, v1, vl), vl), vl);
                    m0 = __riscv_vmor(m0, __riscv_vmand(__riscv_vmslt(v0, x2, vl), __riscv_vmslt(v0, x3, vl), vl), vl);
                    m1 = __riscv_vmor(m1, __riscv_vmand(__riscv_vmslt(x2, v1, vl), __riscv_vmslt(x3, v1, vl), vl), vl);
                    m0 = __riscv_vmor(m0, __riscv_vmand(__riscv_vmslt(v0, x3, vl), __riscv_vmslt(v0, x0, vl), vl), vl);
                    m1 = __riscv_vmor(m1, __riscv_vmand(__riscv_vmslt(x3, v1, vl), __riscv_vmslt(x0, v1, vl), vl), vl);
                    m0 = __riscv_vmor(m0, m1, vl);

                    unsigned long mask_cnt = __riscv_vcpop(m0, vl);
                    if(!mask_cnt)
                        continue;

                    // TODO: Test if skipping to the first possible key point pixel if faster
                    // Memory access maybe expensive since the data is not aligned
                    // long first_set = __riscv_vfirst(m0, vl);
                    // if( first_set == -1 ) {
                    //     j -= first_set;
                    //     ptr -= first_set;
                    // }

                    vint8m4_t c0 = __riscv_vmv_v_x_i8m4(0, vl);
                    vint8m4_t c1 = __riscv_vmv_v_x_i8m4(0, vl);
                    vuint8m4_t max0 = __riscv_vmv_v_x_u8m4(0, vl);
                    vuint8m4_t max1 = __riscv_vmv_v_x_u8m4(0, vl);

                    for( k = 0; k < N; k++ )
                    {
                        vint8m4_t x = __riscv_vreinterpret_i8m4(__riscv_vxor(__riscv_vle8_v_u8m4(ptr + pixel[k], vl), v_c_delta, vl));
                        
                        m0 = __riscv_vmslt(v0, x, vl);
                        m1 = __riscv_vmslt(x, v1, vl);

                        c0 = __riscv_vadd_mu(m0, c0, c0, (int8_t)1, vl);
                        c1 = __riscv_vadd_mu(m1, c1, c1, (int8_t)1, vl);
                        c0 = __riscv_vmerge(v_c_zero, c0, m0, vl);
                        c1 = __riscv_vmerge(v_c_zero, c1, m1, vl);

                        max0 = __riscv_vmaxu(max0, __riscv_vreinterpret_u8m4(c0), vl);
                        max1 = __riscv_vmaxu(max1, __riscv_vreinterpret_u8m4(c1), vl);
                    }

                    vbool2_t v_comparek = __riscv_vmsltu(v_c_k, __riscv_vmaxu(max0, max1, vl), vl);
                    uint8_t m[64];
                    __riscv_vse8(m, __riscv_vreinterpret_u8m1(v_comparek), vl);

                    msg = cv::format("at row %d, column %d, m = ", i, j);
                    for (int i = 0; i < 16; i++)
                    {
                        msg += cv::format("%d ", m[i]);
                    }
                    CV_LOG_INFO(NULL, msg)

                    for( k = 0; k < vl; k++ )
                    {
                        if( m[k] )
                        {
                            cornerpos[ncorners++] = j + k;
                            if(nonmax_suppression)
                                curr[j + k] = (uchar)cornerScore(ptr + k, v_offset, (int64_t)src_step);
                        }
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3] + 1; // cornerpos[-1] is used to store a value
        ncorners = cornerpos[-1];

        msg = cv::format("ncorners in last row %d = %d", i-1, ncorners);
        CV_LOG_INFO(NULL, msg);

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            msg = cv::format("ncorners[%zu] = %d", k, cornerpos[k]);
            CV_LOG_INFO(NULL, msg);
            int score = prev[j];
            msg = cv::format("score = %d", score);
            CV_LOG_INFO(NULL, msg);

            if(!nonmax_suppression ||
               (score > prev[j+1] && score > prev[j-1] &&
                score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                KeyPoint kp((float)j, (float)(i-1), 7.f, -1, (float)score);
                msg = cv::format("keypoint = (%f, %f, %f, %f, %f)", kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response);
                CV_LOG_INFO(NULL, msg);
                keypoints.push_back(kp);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

inline int FAST(const uchar* src_data, 
    size_t src_step, int width, int height, 
    std::vector<KeyPoint>& keypoints, 
    int threshold, bool nonmax_suppression, 
    cv::FastFeatureDetector::DetectorType type) 
{
    std::string msg;
    msg = cv::format("riscv fast: src_step=%zu, width=%d, height=%d, threshold=%d, nonmax_suppression=%d, type=%d", 
        src_step, width, height, threshold, nonmax_suppression, type);
    CV_LOG_INFO(NULL, msg);
    int res = CV_HAL_ERROR_UNKNOWN;
    switch(type) {
        case FastFeatureDetector::TYPE_5_8:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case FastFeatureDetector::TYPE_7_12:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case FastFeatureDetector::TYPE_9_16:
            return fast_16(src_data, src_step, width, height, keypoints, threshold, nonmax_suppression);
        default:
            return res;
    }
}

// inline int fast_16_dense(const uchar* src_data, size_t src_step, int width, int height, uchar* scores_data, size_t scores_step) 
// {

//     const int patternSize = 16;
//     const int K = patternSize/2, N = patternSize + K + 1;
//     const int quarterPatternSize = patternSize/4;

//     std::string msg;
//     msg = cv::format("riscv fast_16_dense: patternSize=%d, K=%d, N=%d, quarterPatternSize=%d", patternSize, K, N, quarterPatternSize);
//     CV_LOG_INFO(NULL, msg);

//     int i, j, k;
//     uint16_t pixel[25];
//     vuint16m2_t v_offset;
//     makeOffsets(pixel, v_offset, (int)src_step, patternSize);

//     int vlmax = __riscv_vsetvlmax_e8m8();
//     vuint8m4_t v_c_delta = __riscv_vmv_v_x_u8m4(0x80, vlmax);
//     vuint8m4_t v_c_k = __riscv_vmv_v_x_u8m4((char)K, vlmax);
//     vint8m4_t v_c_zero = __riscv_vmv_v_x_i8m4(0, vlmax);

//     for( i = 3; i < height - 2; i++)
//     {
//         msg = cv::format("Calculating FAST for row %d", i);
//         CV_LOG_INFO(NULL, msg);

//         const uchar* ptr = src_data + i * src_step;
//         uchar* score_ptr = scores_data + i * scores_step;

//         if( i < height - 3 )
//         {
//             j = 3;
//             {
//                 int margin = width - 3;
//                 for (; j < margin; j++)
//                 {
//                     score_ptr[j] = (uchar)cornerScore(ptr + j, v_offset, (int64_t)src_step);
//                 }
//             }
//         }
//     }
//     return CV_HAL_ERROR_OK;
// }

// inline int FAST_dense(const uchar* src_data, size_t src_step, uchar* scores_data, size_t scores_step, int width, int height,
//     cv::FastFeatureDetector::DetectorType type) 
// {
//     std::string msg;
//     msg = cv::format("riscv fast_dense: src_step=%zu, width=%d, height=%d type=%d", 
//         src_step, width, height, type);
//     CV_LOG_INFO(NULL, msg);
//     int res = CV_HAL_ERROR_UNKNOWN;
//     switch(type) {
//         case FastFeatureDetector::TYPE_5_8:
//             return CV_HAL_ERROR_NOT_IMPLEMENTED;
//         case FastFeatureDetector::TYPE_7_12:
//             return CV_HAL_ERROR_NOT_IMPLEMENTED;
//         case FastFeatureDetector::TYPE_9_16:
//             return fast_16_dense(src_data, src_step, width, height, scores_data, scores_step);
//         default:
//             return res;
//     }
// }
} // namespace cv::cv_hal_rvv
