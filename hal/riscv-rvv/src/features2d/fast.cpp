#include "rvv_hal.hpp"
#include "common.hpp"
#include <cfloat>

namespace cv { namespace rvv_hal { namespace features2d {

static inline uint8_t cornerScore(const uint8_t* ptr, const int* pixel)
{
    constexpr int K = 8, N = 16 + K + 1;
    int v = ptr[0];
    int16_t d[32] = {0};
    for (int k = 0; k < N; k++)
        d[k] = (int16_t)(v - ptr[pixel[k]]);
    auto vlenb = __riscv_vlenb();
    switch (vlenb) {
        #define CV_RVV_HAL_FAST_CORNERSOCRE16_CASE(lmul) \
            size_t vl = __riscv_vsetvl_e16m##lmul(N); \
            vint16m##lmul##_t vd = __riscv_vle16_v_i16m##lmul(d, vl); \
            vint16m##lmul##_t q0 = __riscv_vmv_v_x_i16m##lmul((int16_t)(-1000), vl); \
            vint16m##lmul##_t q1 = __riscv_vmv_v_x_i16m##lmul((int16_t)(1000), vl); \
            vint16m##lmul##_t vds = vd, ak0 = vd, bk0 = vd; \
            for (int i = 0; i < 8; i++) { \
                vds = __riscv_vslide1down(vds, 0, vl); \
                ak0 = __riscv_vmin(ak0, vds, vl); \
                bk0 = __riscv_vmax(bk0, vds, vl); \
            } \
            q0 = __riscv_vmax(q0, __riscv_vmin(ak0, vd, vl), vl); \
            q1 = __riscv_vmin(q1, __riscv_vmax(bk0, vd, vl), vl); \
            vds = __riscv_vslide1down(vds, 0, vl); \
            q0 = __riscv_vmax(q0, __riscv_vmin(ak0, vds, vl), vl); \
            q1 = __riscv_vmin(q1, __riscv_vmax(bk0, vds, vl), vl); \
            q0 = __riscv_vmax(q0, __riscv_vrsub(q1, 0, vl), vl); \
            return (uint8_t)(__riscv_vmv_x(__riscv_vredmax(q0, __riscv_vmv_s_x_i16m1(0, vl), vl)) - 1);
        case 16: { // 128-bit
            CV_RVV_HAL_FAST_CORNERSOCRE16_CASE(4)
        } break;
        case 32: { // 256-bit
            CV_RVV_HAL_FAST_CORNERSOCRE16_CASE(2)
        } break;
        default: { // >=512-bit
            CV_RVV_HAL_FAST_CORNERSOCRE16_CASE(1)
        }
    }
}


inline int fast_16(const uchar* src_data, size_t src_step,
                   int width, int height,
                   std::vector<cvhalKeyPoint> &keypoints,
                   int threshold, bool nonmax_suppression)
{

    constexpr int patternSize = 16;
    constexpr int K = patternSize/2, N = patternSize + K + 1;
    constexpr int quarterPatternSize = patternSize/4;

    int i, j, k;
    int pixel[N] = {0};
    pixel[0] = 0 + (int)src_step * 3;
    pixel[1] = 1 + (int)src_step * 3;
    pixel[2] = 2 + (int)src_step * 2;
    pixel[3] = 3 + (int)src_step * 1;
    pixel[4] = 3 + (int)src_step * 0;
    pixel[5] = 3 + (int)src_step * -1;
    pixel[6] = 2 + (int)src_step * -2;
    pixel[7] = 1 + (int)src_step * -3;
    pixel[8] = 0 + (int)src_step * -3;
    pixel[9] = -1 + (int)src_step * -3;
    pixel[10] = -2 + (int)src_step * -2;
    pixel[11] = -3 + (int)src_step * -1;
    pixel[12] = -3 + (int)src_step * 0;
    pixel[13] = -3 + (int)src_step * 1;
    pixel[14] = -2 + (int)src_step * 2;
    pixel[15] = -1 + (int)src_step * 3;
    for (k = 16; k < N; k++)
    {
        pixel[k] = pixel[k - 16];
    }

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
                                curr[j + k] = (uchar)cornerScore(ptr + k, pixel);
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
                cvhalKeyPoint kp;
                kp.x = (float)j;
                kp.y = (float)(i-1);
                kp.size = 7.f;
                kp.angle = -1.f;
                kp.response = (float)score;
                kp.octave = 0; // Not used in FAST
                kp.class_id = -1; // Not used in FAST
                keypoints.push_back(kp);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

int FAST(const uchar* src_data, size_t src_step,
         int width, int height, void** keypoints_data,
         size_t* keypoints_count, int threshold,
         bool nonmax_suppression, int detector_type, void* (*realloc_func)(void*, size_t))
{
    int res = CV_HAL_ERROR_UNKNOWN;
    switch(detector_type) {
        case CV_HAL_TYPE_5_8:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_7_12:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        case CV_HAL_TYPE_9_16: {
            std::vector<cvhalKeyPoint> keypoints;
            res = fast_16(src_data, src_step, width, height, keypoints, threshold, nonmax_suppression);
            if (res == CV_HAL_ERROR_OK) {
                if (keypoints.size() > *keypoints_count) {
                    *keypoints_count = keypoints.size();
                    uchar *tmp = (uchar*)realloc_func(*keypoints_data, sizeof(cvhalKeyPoint)*(*keypoints_count));
                    memcpy(tmp, (uchar*)keypoints.data(), sizeof(cvhalKeyPoint)*(*keypoints_count));
                    *keypoints_data = tmp;
                } else {
                    *keypoints_count = keypoints.size();
                    memcpy(*keypoints_data, (uchar*)keypoints.data(), sizeof(cvhalKeyPoint)*(*keypoints_count));
                }
            }
            return res;
        }
        default:
            return res;
    }
}

}}} // namespace cv::rvv_hal::features2d
