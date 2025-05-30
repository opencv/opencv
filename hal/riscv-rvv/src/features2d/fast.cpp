// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once



#include "rvv_hal.hpp"
#include "common.hpp"

#include "include/buffer_area.hpp"
#include "opencv2/core/utils/logger.hpp"

#include <cfloat>


#define CV_HAL_RVV_FAST_DEBUG
#ifdef CV_HAL_RVV_FAST_DEBUG
#include <iostream>
#include <cstdio>
template <typename T>
void printVectorUint8(T vec, int vl, std::string name) {
    uint8_t* data = (uint8_t*)malloc(vl * sizeof(uint8_t));
    __riscv_vse8(data, vec, vl);
    std::cout << name << ": ";
    for (int i = 0; i < vl; i++) {
        std::cout << (int)data[i] << " ";
    }
    std::cout << std::endl;
    free(data);
}
template <typename T>
void printVectorInt8(T vec, int vl, std::string name) {
    int8_t* data = (int8_t*)malloc(vl * sizeof(int8_t));
    __riscv_vse8(data, vec, vl);
    std::cout << name << ": ";
    for (int i = 0; i < vl; i++) {
        std::cout << (int)data[i] << " ";
    }
    std::cout << std::endl;
    free(data);
}
template <typename T>
void printVectorUint16(T vec, int vl, std::string name) {
    uint16_t* data = (uint16_t*)malloc(vl * sizeof(uint16_t));
    __riscv_vse16(data, vec, vl);
    std::cout << name << ": ";
    for (int i = 0; i < vl; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    free(data);
}
template <typename T>
void printVectorInt16(T vec, int vl, std::string name) {
    int16_t* data = (int16_t*)malloc(vl * sizeof(int16_t));
    __riscv_vse16(data, vec, vl);
    std::cout << name << ": ";
    for (int i = 0; i < vl; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    free(data);
}
template <typename T>
void printVectorUint32(T vec, int vl, std::string name) {
    uint32_t* data = (uint32_t*)malloc(vl * sizeof(uint32_t));
    __riscv_vse32(data, vec, vl);
    std::cout << name << ": ";
    for (int i = 0; i < vl; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    free(data);
}
void printVectorBool2(vbool2_t vec, int vl, std::string name) {
    vuint8m1_t tmp = __riscv_vreinterpret_u8m1(vec);
    uint8_t* data = (uint8_t*)malloc(vl * sizeof(uint8_t));
    __riscv_vse8(data, tmp, vl);
    std::cout << name << ": " << std::endl;
    unsigned long mask_cnt = __riscv_vcpop(vec, vl);
    std::cout << "mask_cnt: " << mask_cnt << std::endl;
    int j = 0, k = 0;
    for (int i = 0; i < vl; i++) {
        std::cout << (int)((data[j] >> k) & 1) << " ";
        k++;
        if (k > 7) {
            j++;
            k = 0;
        }
    }
    std::cout << std::endl;
    free(data);    
}

template <typename T>
void printVector(T vec, int vl, std::string name) {
    if constexpr (std::is_same<T, vuint8m1_t>::value) {
        printVectorUint8(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint8m2_t>::value) {
        printVectorUint8(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint8m4_t>::value) {
        printVectorUint8(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint8m8_t>::value) {
        printVectorUint8(vec, vl, name);
    } else if constexpr (std::is_same<T, vint8m1_t>::value) {
        printVectorInt8(vec, vl, name);
    } else if constexpr (std::is_same<T, vint8m2_t>::value) {
        printVectorInt8(vec, vl, name);
    } else if constexpr (std::is_same<T, vint8m4_t>::value) {
        printVectorInt8(vec, vl, name);
    } else if constexpr (std::is_same<T, vint8m8_t>::value) {
        printVectorInt8(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint16m1_t>::value) {
        printVectorUint16(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint16m2_t>::value) {
        printVectorUint16(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint16m4_t>::value) {
        printVectorUint16(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint16m8_t>::value) {
        printVectorUint16(vec, vl, name);
    } else if constexpr (std::is_same<T, vint16m1_t>::value) {
        printVectorInt16(vec, vl, name);
    } else if constexpr (std::is_same<T, vint16m2_t>::value) {
        printVectorInt16(vec, vl, name);
    } else if constexpr (std::is_same<T, vint16m4_t>::value) {
        printVectorInt16(vec, vl, name);
    } else if constexpr (std::is_same<T, vint16m8_t>::value) {
        printVectorInt16(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint32m1_t>::value) {
        printVectorUint32(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint32m2_t>::value) {
        printVectorUint32(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint32m4_t>::value) {
        printVectorUint32(vec, vl, name);
    } else if constexpr (std::is_same<T, vuint32m8_t>::value) {
        printVectorUint32(vec, vl, name);
    }
}
#endif // CV_HAL_RVV_FAST_DEBUG


namespace cv { namespace rvv_hal { namespace features2d {

using RVV_VECTOR_TYPE = vuint8m4_t;


// Since uint16_t range is 0 to 65535, row stride should be less than 65535/6 = 10922
inline void makeOffsets(int16_t pixel[], vuint16m2_t& v_offset, int64_t row_stride, int patternSize)
{
    uint16_t pixel_u[25];

    // set min element (pixel[8] = 0 + row_stride * -3) as the base addr
    // pixel_u[0] = row_stride * 6;
    // pixel_u[1] = 1 + row_stride * 6;
    // pixel_u[2] = 2 + row_stride * 5;
    // pixel_u[3] = 3 + row_stride * 4;
    // pixel_u[4] = 3 + row_stride * 3;
    // pixel_u[5] = 3 + row_stride * 2;
    // pixel_u[6] = 2 + row_stride * 1;
    // pixel_u[7] = 1 + row_stride * 0;
    // pixel_u[8] = 0 + row_stride * 0;
    // pixel_u[9] = -1 + row_stride * 0;
    // pixel_u[10] = -2 + row_stride * 1;
    // pixel_u[11] = -3 + row_stride * 2;
    // pixel_u[12] = -3 + row_stride * 3;
    // pixel_u[13] = -3 + row_stride * 4;
    // pixel_u[14] = -2 + row_stride * 5;
    // pixel_u[15] = -1 + row_stride * 6;

    // set min element (pixel[9] = -1 + row_stride * -3) as the base addr
    pixel_u[0] = 1 + row_stride * 6;
    pixel_u[1] = 2 + row_stride * 6;
    pixel_u[2] = 3 + row_stride * 5;
    pixel_u[3] = 4 + row_stride * 4;
    pixel_u[4] = 4 + row_stride * 3;
    pixel_u[5] = 4 + row_stride * 2;
    pixel_u[6] = 3 + row_stride * 1;
    pixel_u[7] = 2 + row_stride * 0;
    pixel_u[8] = 1 + row_stride * 0;
    pixel_u[9] = 0 + row_stride * 0;
    pixel_u[10] = -1 + row_stride * 1;
    pixel_u[11] = -2 + row_stride * 2;
    pixel_u[12] = -2 + row_stride * 3;
    pixel_u[13] = -2 + row_stride * 4;
    pixel_u[14] = -1 + row_stride * 5;
    pixel_u[15] = 0 + row_stride * 6;

    for (int i = 16; i < 25; i++)
    {
        pixel_u[i] = pixel_u[i - 16];
    }
    v_offset = __riscv_vle16_v_u16m2(pixel_u, 25);
    for (int i = 0; i < 25; i++)
    {
        pixel[i] = pixel_u[i] - 3 * row_stride - 1;
    }
}

template<typename T> inline T* alignPtr(T* ptr, size_t n=sizeof(T))
{
    return (T*)(((size_t)ptr + n-1) & -n);
}

inline uint8_t cornerScore(const uint8_t* ptr, const vuint16m2_t& v_offset, int64_t row_stride, bool debug = false) 
{
    const uint32_t K = 8, N = 16 + K + 1;
    uint32_t k, v = ptr[0];
    
    int vl = __riscv_vsetvl_e16m2(N);
    std::string msg;
    if (debug)
    {
        msg = cv::format("riscv fast_16: vl=%d, N=%d", vl, N);
        CV_LOG_INFO(NULL, msg);
        std::cout<<"vanilla offset loading" << std::endl;
        // 3073 3074 2563 2052 1540 1028 515 2 1 0 511 1022 1534 2046 2559 3072 3073 3074 2563 2052 1540 1028 515 2 1
        uint16_t pixel[25] = {
            3073, 3074, 2563, 2052, 1540, 1028, 515, 2,
            1, 0, 511, 1022, 1534, 2046, 2559, 3072,
            3073, 3074, 2563, 2052, 1540, 1028, 515, 2,
            1
        };
        uint8_t* shift_ptr;
        shift_ptr = ((uint8_t*)ptr) - 3 * row_stride - 1;
        for (int i = 0; i < 25; i++)
        {   
            std::cout << (int)(shift_ptr[pixel[i]]) << " ";
        }
        std::cout << std::endl;
    }
    // use vloxei16_v to indexed ordered load
    vint16m2_t v_c_pixel = __riscv_vmv_v_x_i16m2((int16_t)v, vl);
    // vloxei only support positive offset
    vuint8m1_t v_d_u8 = __riscv_vloxei16(ptr - 3 * row_stride - 1, v_offset, vl);
    vuint16m2_t v_d_u16 = __riscv_vzext_vf2(v_d_u8, vl);
    vint16m2_t d = __riscv_vreinterpret_i16m2(v_d_u16);
    // for( k = 0; k < N; k++ )
    //     d[k] = (uint16_t)(v - ptr[pixel[k]]);
    if (debug)
    {   
        printVector(v_offset, vl, "v_offset");
        printVector(d, vl, "d before sub");
    }
    
    d = __riscv_vsub_vv_i16m2(v_c_pixel, d, vl);
    if (debug) {
        std::cout << "row_stride: " << row_stride << std::endl;
        printVector(d, vl, "d");
    }

    vint16m2_t d_slide = __riscv_vmv_v(d, vl);
    
    vint16m2_t q0 = __riscv_vmv_v_x_i16m2((int16_t)(-1000), vl);
    vint16m2_t q1 = __riscv_vmv_v_x_i16m2((int16_t)(1000), vl);

    //k == 0
    vint16m2_t ak0 = __riscv_vmv_v(d, vl);
    vint16m2_t bk0 = __riscv_vmv_v(d, vl);

    for (int i = 0; i < 8; i++)
    {
        d_slide = __riscv_vslide1down(d_slide, (int16_t)0, vl);
        ak0 = __riscv_vmin(ak0, d_slide, vl);
        bk0 = __riscv_vmax(bk0, d_slide, vl);
    }
    if(debug) {
        printVector(ak0, vl, "ak0");
        printVector(bk0, vl, "bk0");
    }

    q0 = __riscv_vmax(q0, __riscv_vmin(ak0, d, vl), vl);
    q1 = __riscv_vmin(q1, __riscv_vmax(bk0, d, vl), vl);

    if (debug) {
        printVector(q0, vl, "q0");
        printVector(q1, vl, "q1");
    }

    d_slide = __riscv_vslide1down(d_slide, (int16_t)0, vl);
    q0 = __riscv_vmax(q0, __riscv_vmin(ak0, d_slide, vl), vl);
    q1 = __riscv_vmin(q1, __riscv_vmax(bk0, d_slide, vl), vl);

    if (debug) {
        printVector(q0, vl, "q0 after slide");
        printVector(q1, vl, "q1 after slide");
    }

    q1 = __riscv_vrsub(q1, (int16_t)0, vl);
    q0 = __riscv_vmax(q0, q1, vl);

    vint16m1_t res = __riscv_vredmax(q0, __riscv_vmv_s_x_i16m1((int16_t)0, vl), vl);

    if (debug) {
        printVector(res, vl, "res");
    }

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

    int vlmax = __riscv_vsetvlmax_e8m4 ();
    vuint8m4_t v_c_delta = __riscv_vmv_v_x_u8m4(0x80, vlmax);
    vuint8m4_t v_c_threshold = __riscv_vmv_v_x_u8m4((char) threshold, vlmax);
    vuint8m4_t v_c_k = __riscv_vmv_v_x_u8m4((uint8_t)K, vlmax);
    vuint8m4_t v_c_zero = __riscv_vmv_v_x_u8m4(0, vlmax);

    for( i = 3; i < height - 2; i++)
    {

        const uchar* ptr = src_data + i * src_step + 3;
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

                    vuint8m4_t c0 = __riscv_vmv_v_x_u8m4(0, vl);
                    vuint8m4_t c1 = __riscv_vmv_v_x_u8m4(0, vl);
                    vuint8m4_t max0 = __riscv_vmv_v_x_u8m4(0, vl);
                    vuint8m4_t max1 = __riscv_vmv_v_x_u8m4(0, vl);

                    for( k = 0; k < N; k++ )
                    {
                        vint8m4_t x = __riscv_vreinterpret_i8m4(__riscv_vxor(__riscv_vle8_v_u8m4(ptr + pixel[k], vl), v_c_delta, vl));
                        
                        m0 = __riscv_vmslt(v0, x, vl);
                        m1 = __riscv_vmslt(x, v1, vl);

                        c0 = __riscv_vadd_mu(m0, c0, c0, (uint8_t)1, vl);
                        c1 = __riscv_vadd_mu(m1, c1, c1, (uint8_t)1, vl);
                        c0 = __riscv_vmerge(v_c_zero, c0, m0, vl);
                        c1 = __riscv_vmerge(v_c_zero, c1, m1, vl);

                        // printVectorUint8(c0, vl, cv::format("c0 k = %d", k));
                        // printVectorUint8(c1, vl, cv::format("c1 k = %d", k));

                        max0 = __riscv_vmaxu(max0, c0, vl);
                        max1 = __riscv_vmaxu(max1, c1, vl);
                    }

                    vbool2_t v_comparek = __riscv_vmsltu(v_c_k, __riscv_vmaxu(max0, max1, vl), vl);
                    uint8_t m[64];
                    __riscv_vse8(m, __riscv_vreinterpret_u8m1(v_comparek), vl);

                    for( k = 0; k < vl; k++ )
                    {
                        if( (m[k / 8] >> (k % 8)) & 1 )
                        {
                            cornerpos[ncorners++] = j + k;
                            if(nonmax_suppression) {
                                bool debug = false;
                                int debug_x = 15;
                                int debug_y = 357;
                                debug = (debug_x == i && debug_y == j + k);
                                curr[j + k] = (uchar)cornerScore(ptr + k, v_offset, (int64_t)src_step, debug);
                                // msg = cv::format("keypoint = (%d, %d, %f, %f, %d), debug = %d", j + k, i, 7.f, -1.f, curr[j + k], debug);
                                // CV_LOG_INFO(NULL, msg);
                            }
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

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            if(!nonmax_suppression ||
               (score > prev[j+1] && score > prev[j-1] &&
                score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                KeyPoint kp((float)j, (float)(i-1), 7.f, -1, (float)score);
                // msg = cv::format("keypoint = (%f, %f, %f, %f, %f)", kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response);
                // CV_LOG_INFO(NULL, msg);
                keypoints.push_back(kp);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

int FAST(const uchar* src_data, size_t src_step, int width, int height,
          std::vector<KeyPoint>& keypoints,
          int threshold, bool nonmax_suppression, int detector_type)
{
    std::string msg;
    msg = cv::format("riscv fast: src_step=%zu, width=%d, height=%d, threshold=%d, nonmax_suppression=%d, detector_type=%d", 
        src_step, width, height, threshold, nonmax_suppression, detector_type);
    CV_LOG_INFO(NULL, msg);
    int res = CV_HAL_ERROR_UNKNOWN;
    switch(detector_type) {
        case CV_HAL_TYPE_5_8:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_7_12:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_9_16:
            return fast_16(src_data, src_step, width, height, keypoints, threshold, nonmax_suppression);
        default:
            return res;
    }
    std::cout << "In fast.cpp keypoints.size()" << keypoints.size() << std::endl; 
}

}}} // namespace cv::rvv_hal::features2d
