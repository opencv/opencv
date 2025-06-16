#include "rvv_hal.hpp"
#include "common.hpp"
#include <cfloat>

namespace cv { namespace rvv_hal { namespace features2d {

using RVV_VECTOR_TYPE = vuint8m4_t;


// Since uint16_t range is 0 to 65535, row stride should be less than 65535/6 = 10922
inline void makeOffsets(int16_t pixel[], vuint16m2_t& v_offset, int64_t row_stride, int patternSize)
{
    uint16_t pixel_u[25];

    switch(patternSize) {
    case 16:
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
        break;

    default:
        memset(pixel_u, 0, sizeof(uint16_t) * 25);

    }
}


inline uint8_t cornerScore(const uint8_t* ptr, const vuint16m2_t& v_offset, int64_t row_stride)
{
    const uint32_t K = 8, N = 16 + K + 1;
    uint32_t v = ptr[0];

    int vl = __riscv_vsetvl_e16m2(N);
    // use vloxei16_v to indexed ordered load
    vint16m2_t v_c_pixel = __riscv_vmv_v_x_i16m2((int16_t)v, vl);
    // vloxei only support positive offset
    vuint8m1_t v_d_u8 = __riscv_vloxei16(ptr - 3 * row_stride - 1, v_offset, vl);
    vuint16m2_t v_d_u16 = __riscv_vzext_vf2(v_d_u8, vl);
    vint16m2_t d = __riscv_vreinterpret_i16m2(v_d_u16);
    d = __riscv_vsub_vv_i16m2(v_c_pixel, d, vl);
    vint16m2_t d_slide = __riscv_vmv_v(d, vl);

    vint16m2_t q0 = __riscv_vmv_v_x_i16m2((int16_t)(-1000), vl);
    vint16m2_t q1 = __riscv_vmv_v_x_i16m2((int16_t)(1000), vl);

    vint16m2_t ak0 = __riscv_vmv_v(d, vl);
    vint16m2_t bk0 = __riscv_vmv_v(d, vl);

    for (int i = 0; i < 8; i++)
    {
        d_slide = __riscv_vslide1down(d_slide, (int16_t)0, vl);
        ak0 = __riscv_vmin(ak0, d_slide, vl);
        bk0 = __riscv_vmax(bk0, d_slide, vl);
    }

    q0 = __riscv_vmax(q0, __riscv_vmin(ak0, d, vl), vl);
    q1 = __riscv_vmin(q1, __riscv_vmax(bk0, d, vl), vl);

    d_slide = __riscv_vslide1down(d_slide, (int16_t)0, vl);
    q0 = __riscv_vmax(q0, __riscv_vmin(ak0, d_slide, vl), vl);
    q1 = __riscv_vmin(q1, __riscv_vmax(bk0, d_slide, vl), vl);

    q1 = __riscv_vrsub(q1, (int16_t)0, vl);
    q0 = __riscv_vmax(q0, q1, vl);

    vint16m1_t res = __riscv_vredmax(q0, __riscv_vmv_s_x_i16m1((int16_t)0, vl), vl);

    uint8_t result = (uint8_t)__riscv_vmv_x(res);
    return result - 1;
}


inline int fast_16(const uchar* src_data, size_t src_step,
                   int width, int height,
                   uchar* keypoints_data, size_t* keypoints_count,
                   int threshold, bool nonmax_suppression)
{

    const int patternSize = 16;
    const int K = patternSize/2, N = patternSize + K + 1;
    const int quarterPatternSize = patternSize/4;

    KeyPoint* _keypoints_data = (KeyPoint*)keypoints_data;

    int i, j, k;
    int16_t pixel[25];
    vuint16m2_t v_offset;
    makeOffsets(pixel, v_offset, (int)src_step, patternSize);

    std::vector<uchar> _buf((width+16)*3*(sizeof(ptrdiff_t) + sizeof(uchar)) + 128);
    uchar* buf[3];
    buf[0] = &_buf[0]; buf[1] = buf[0] + width; buf[2] = buf[1] + width;
    ptrdiff_t* cpbuf[3];
    cpbuf[0] = (ptrdiff_t*)alignPtr(buf[2] + width, sizeof(ptrdiff_t)) + 1;
    cpbuf[1] = cpbuf[0] + width + 1;
    cpbuf[2] = cpbuf[1] + width + 1;
    memset(buf[0], 0, width*3);

    int vlmax = __riscv_vsetvlmax_e8m4();
    vuint8m4_t v_c_delta = __riscv_vmv_v_x_u8m4(0x80, vlmax);
    vuint8m4_t v_c_threshold = __riscv_vmv_v_x_u8m4((char) threshold, vlmax);
    vuint8m4_t v_c_k = __riscv_vmv_v_x_u8m4((uint8_t)K, vlmax);
    vuint8m4_t v_c_zero = __riscv_vmv_v_x_u8m4(0, vlmax);

    for( i = 3; i < height - 2; i++)
    {

        const uchar* ptr = src_data + i * src_step + 3;
        uchar* curr = buf[(i - 3)%3];
        ptrdiff_t* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, width);
        ptrdiff_t ncorners = 0;

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
                                curr[j + k] = (uchar)cornerScore(ptr + k, v_offset, (int64_t)src_step);
                            }
                        }
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if( i == 3 )            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3]; // cornerpos[-1] is used to store a value
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
                _keypoints_data[*keypoints_count].pt.x = (float)j;
                _keypoints_data[*keypoints_count].pt.y = (float)(i-1);
                _keypoints_data[*keypoints_count].size = 7.f;
                _keypoints_data[*keypoints_count].angle = -1.f;
                _keypoints_data[*keypoints_count].response = (float)score;
                _keypoints_data[*keypoints_count].octave = 0; // Not used in FAST
                _keypoints_data[*keypoints_count].class_id = -1; // Not used in FAST

                (*keypoints_count)++;
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

int FAST(const uchar* src_data, size_t src_step,
         int width, int height, uchar* keypoints_data,
         size_t* keypoints_count, int threshold,
         bool nonmax_suppression, int detector_type)
{
    (*keypoints_count) = 0;
    int res = CV_HAL_ERROR_UNKNOWN;
    switch(detector_type) {
        case CV_HAL_TYPE_5_8:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_7_12:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_9_16:
            return fast_16(src_data, src_step, width, height, keypoints_data, keypoints_count, threshold, nonmax_suppression);
        default:
            return res;
    }
}

}}} // namespace cv::rvv_hal::features2d
