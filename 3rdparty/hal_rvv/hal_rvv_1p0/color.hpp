// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_COLOR_HPP_INCLUDED
#define OPENCV_HAL_RVV_COLOR_HPP_INCLUDED

#include <riscv_vector.h>
#include <thread>
#include <future>

namespace cv { namespace cv_hal_rvv {

namespace color {
    template<typename F, typename... Args>
    inline int parallel_for(int n, F func, Args&&... args)
    {
        int num_threads = std::thread::hardware_concurrency();
        int length = (n + num_threads - 1) / num_threads;
        auto futures = new std::future<int>[num_threads];
        for (int x = 0; x < num_threads; x++)
        {
            futures[x] = std::async(std::launch::async, func, x * length, std::min(n, (x + 1) * length), std::forward<Args>(args)...);
        }
        for (int x = 0; x < num_threads; x++)
        {
            if (futures[x].get() != CV_HAL_ERROR_OK)
            {
                delete[] futures;
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
            }
        }

        delete[] futures;
        return CV_HAL_ERROR_OK;
    }
} // cv::cv_hal_rvv::color

namespace BGRtoBGR {
#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR cv::cv_hal_rvv::BGRtoBGR::cvtBGRtoBGR

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m4(a); }
    static inline vuint8m4_t vlse(const uchar* a, ptrdiff_t b, size_t c) { return __riscv_vlse8_v_u8m4(a, b, c); }
    static inline void vsse(uchar* a, ptrdiff_t b, vuint8m4_t c, size_t d) { return __riscv_vsse8(a, b, c, d); }
    static inline vuint8m4_t vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m4(a, b); }
};
template<> struct rvv<ushort>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m4(a); }
    static inline vuint16m4_t vlse(const ushort* a, ptrdiff_t b, size_t c) { return __riscv_vlse16_v_u16m4(a, b, c); }
    static inline void vsse(ushort* a, ptrdiff_t b, vuint16m4_t c, size_t d) { return __riscv_vsse16(a, b, c, d); }
    static inline vuint16m4_t vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m4(a, b); }
};
template<> struct rvv<float>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vfloat32m4_t vlse(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m4(a, b, c); }
    static inline void vsse(float* a, ptrdiff_t b, vfloat32m4_t c, size_t d) { return __riscv_vsse32(a, b, c, d); }
    static inline vfloat32m4_t vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m4(a, b); }
};

template<typename T>
inline int cvtBGRtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, int dcn, bool swapBlue)
{
    if ((scn != 3 && scn != 4) || (dcn != 3 && dcn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    if (scn == dcn && !swapBlue)
    {
        for (int i = start; i < end; i++)
            memcpy(dst + dst_step * i, src + src_step * i, sizeof(T) * width * scn);
    }
    else
    {
        auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1 : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
        int vl;
        for (int i = start; i < end; i++)
        {
            for (int j = 0; j < width; j += vl)
            {
                vl = rvv<T>::vsetvl(width - j);
                auto vec_src = rvv<T>::vlse(src + src_step * i + j * scn + (swapBlue ? 2 : 0), sizeof(T) * scn, vl);
                rvv<T>::vsse(dst + dst_step * i + j * dcn, sizeof(T) * dcn, vec_src, vl);
                vec_src = rvv<T>::vlse(src + src_step * i + j * scn + 1, sizeof(T) * scn, vl);
                rvv<T>::vsse(dst + dst_step * i + j * dcn + 1, sizeof(T) * dcn, vec_src, vl);
                vec_src = rvv<T>::vlse(src + src_step * i + j * scn + (swapBlue ? 0 : 2), sizeof(T) * scn, vl);
                rvv<T>::vsse(dst + dst_step * i + j * dcn + 2, sizeof(T) * dcn, vec_src, vl);
                if (dcn == 4)
                {
                    vec_src = scn == 3 ? alpha : rvv<T>::vlse(src + src_step * i + j * scn + 3, sizeof(T) * scn, vl);
                    rvv<T>::vsse(dst + dst_step * i + j * dcn + 3, sizeof(T) * dcn, vec_src, vl);
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue)
{
    switch (depth)
    {
    case CV_8U:
        return color::parallel_for(height, cvtBGRtoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_16U:
        return color::parallel_for(height, cvtBGRtoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_32F:
        return color::parallel_for(height, cvtBGRtoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoBGR

namespace GraytoBGR {
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR cv::cv_hal_rvv::GraytoBGR::cvtGraytoBGR

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m4(a); }
    static inline vuint8m4_t vle(const uchar* a, size_t b) { return __riscv_vle8_v_u8m4(a, b); }
    static inline void vsse(uchar* a, ptrdiff_t b, vuint8m4_t c, size_t d) { return __riscv_vsse8(a, b, c, d); }
    static inline vuint8m4_t vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m4(a, b); }
};
template<> struct rvv<ushort>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m4(a); }
    static inline vuint16m4_t vle(const ushort* a, size_t b) { return __riscv_vle16_v_u16m4(a, b); }
    static inline void vsse(ushort* a, ptrdiff_t b, vuint16m4_t c, size_t d) { return __riscv_vsse16(a, b, c, d); }
    static inline vuint16m4_t vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m4(a, b); }
};
template<> struct rvv<float>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vfloat32m4_t vle(const float* a, size_t b) { return __riscv_vle32_v_f32m4(a, b); }
    static inline void vsse(float* a, ptrdiff_t b, vfloat32m4_t c, size_t d) { return __riscv_vsse32(a, b, c, d); }
    static inline vfloat32m4_t vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m4(a, b); }
};

template<typename T>
inline int cvtGraytoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1 : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            auto vec_src = rvv<T>::vle(src + src_step * i + j, vl);
            rvv<T>::vsse(dst + dst_step * i + j * dcn, sizeof(T) * dcn, vec_src, vl);
            rvv<T>::vsse(dst + dst_step * i + j * dcn + 1, sizeof(T) * dcn, vec_src, vl);
            rvv<T>::vsse(dst + dst_step * i + j * dcn + 2, sizeof(T) * dcn, vec_src, vl);
            if (dcn == 4)
            {
                rvv<T>::vsse(dst + dst_step * i + j * dcn + 3, sizeof(T) * dcn, alpha, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtGraytoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn)
{
    switch (depth)
    {
    case CV_8U:
        return color::parallel_for(height, cvtGraytoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn);
    case CV_16U:
        return color::parallel_for(height, cvtGraytoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn);
    case CV_32F:
        return color::parallel_for(height, cvtGraytoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::GraytoBGR

namespace BGRtoGray {
#undef cv_hal_cvtBGRtoGray
#define cv_hal_cvtBGRtoGray cv::cv_hal_rvv::BGRtoGray::cvtBGRtoGray

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static constexpr uint B2Y = 3735, G2Y = 19235, R2Y = 9798;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline vuint8m1_t vlse(const uchar* a, ptrdiff_t b, size_t c) { return __riscv_vlse8_v_u8m1(a, b, c); }
    static inline void vse(uchar* a, vuint8m1_t b, size_t c) { return __riscv_vse8(a, b, c); }
    static inline vuint32m4_t vcvt0(vuint8m1_t a, size_t b) { return __riscv_vzext_vf4(a, b); }
    static inline vuint8m1_t vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vssrl(a, b, __RISCV_VXRM_RNU, c), c), c); }
    static inline vuint32m4_t vmul(vuint32m4_t a, uint b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vuint32m4_t vmadd(vuint32m4_t a, uint b, vuint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
};
template<> struct rvv<ushort>
{
    static constexpr uint B2Y = 3735, G2Y = 19235, R2Y = 9798;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline vuint16m2_t vlse(const ushort* a, ptrdiff_t b, size_t c) { return __riscv_vlse16_v_u16m2(a, b, c); }
    static inline void vse(ushort* a, vuint16m2_t b, size_t c) { return __riscv_vse16(a, b, c); }
    static inline vuint32m4_t vcvt0(vuint16m2_t a, size_t b) { return __riscv_vzext_vf2(a, b); }
    static inline vuint16m2_t vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vncvt_x(__riscv_vssrl(a, b, __RISCV_VXRM_RNU, c), c); }
    static inline vuint32m4_t vmul(vuint32m4_t a, uint b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vuint32m4_t vmadd(vuint32m4_t a, uint b, vuint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
};
template<> struct rvv<float>
{
    static constexpr float B2Y = 0.114f, G2Y = 0.587f, R2Y = 0.299f;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vfloat32m4_t vlse(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m4(a, b, c); }
    static inline void vse(float* a, vfloat32m4_t b, size_t c) { return __riscv_vse32(a, b, c); }
    static inline vfloat32m4_t vcvt0(vfloat32m4_t a, size_t) { return a; }
    static inline vfloat32m4_t vcvt1(vfloat32m4_t a, size_t, size_t) { return a; }
    static inline vfloat32m4_t vmul(vfloat32m4_t a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline vfloat32m4_t vmadd(vfloat32m4_t a, float b, vfloat32m4_t c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
};

template<typename T>
inline int cvtBGRtoGray(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            auto vec_srcB = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * scn + (swapBlue ? 2 : 0), sizeof(T) * scn, vl), vl);
            auto vec_srcG = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * scn + 1, sizeof(T) * scn, vl), vl);
            auto vec_srcR = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * scn + (swapBlue ? 0 : 2), sizeof(T) * scn, vl), vl);
            auto vec_dst = rvv<T>::vmadd(vec_srcB, rvv<T>::B2Y, rvv<T>::vmadd(vec_srcG, rvv<T>::G2Y, rvv<T>::vmul(vec_srcR, rvv<T>::R2Y, vl), vl), vl);
            rvv<T>::vse(dst + dst_step * i + j, rvv<T>::vcvt1(vec_dst, 15, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoGray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue)
{
    switch (depth)
    {
    case CV_8U:
        return color::parallel_for(height, cvtBGRtoGray<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_16U:
        return color::parallel_for(height, cvtBGRtoGray<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_32F:
        return color::parallel_for(height, cvtBGRtoGray<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoGray

namespace YUVtoBGR {
#undef cv_hal_cvtYUVtoBGR
#define cv_hal_cvtYUVtoBGR cv::cv_hal_rvv::YUVtoBGR::cvtYUVtoBGR

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static constexpr int U2B = 33292, U2G = -6472, V2G = -9519, V2R = 18678, CB2B = 29049, CB2G = -5636, CR2G = -11698, CR2R = 22987;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m1(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline vuint8m1_t vlse(const uchar* a, ptrdiff_t b, size_t c) { return __riscv_vlse8_v_u8m1(a, b, c); }
    static inline void vsse(uchar* a, ptrdiff_t b, vuint8m1_t c, size_t d) { return __riscv_vsse8(a, b, c, d); }
    static inline vint32m4_t vcvt0(vuint8m1_t a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(a, b)); }
    static inline vuint8m1_t vcvt1(vint32m4_t a, vint32m4_t b, size_t c, size_t d) { return __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(__riscv_vadd(__riscv_vssra(a, c, __RISCV_VXRM_RNU, d), b, d), 0, d)), 0, __RISCV_VXRM_RNU, d), 0, __RISCV_VXRM_RNU, d); }
    static inline vint32m4_t vsub(vint32m4_t a, int b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline vuint8m1_t vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m1(a, b); }
};
template<> struct rvv<ushort>
{
    static constexpr int U2B = 33292, U2G = -6472, V2G = -9519, V2R = 18678, CB2B = 29049, CB2G = -5636, CR2G = -11698, CR2R = 22987;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline vuint16m2_t vlse(const ushort* a, ptrdiff_t b, size_t c) { return __riscv_vlse16_v_u16m2(a, b, c); }
    static inline void vsse(ushort* a, ptrdiff_t b, vuint16m2_t c, size_t d) { return __riscv_vsse16(a, b, c, d); }
    static inline vint32m4_t vcvt0(vuint16m2_t a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2(a, b)); }
    static inline vuint16m2_t vcvt1(vint32m4_t a, vint32m4_t b, size_t c, size_t d) { return __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(__riscv_vadd(__riscv_vssra(a, c, __RISCV_VXRM_RNU, d), b, d), 0, d)), 0, __RISCV_VXRM_RNU, d); }
    static inline vint32m4_t vsub(vint32m4_t a, int b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline vuint16m2_t vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m2(a, b); }
};
template<> struct rvv<float>
{
    static constexpr float U2B = 2.032f, U2G = -0.395f, V2G = -0.581f, V2R = 1.140f, CB2B = 1.773f, CB2G = -0.344f, CR2G = -0.714f, CR2R = 1.403f;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vfloat32m4_t vlse(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m4(a, b, c); }
    static inline void vsse(float* a, ptrdiff_t b, vfloat32m4_t c, size_t d) { return __riscv_vsse32(a, b, c, d); }
    static inline vfloat32m4_t vcvt0(vfloat32m4_t a, size_t) { return a; }
    static inline vfloat32m4_t vcvt1(vfloat32m4_t a, vfloat32m4_t b, size_t, size_t d) { return __riscv_vfadd(a, b, d); }
    static inline vfloat32m4_t vsub(vfloat32m4_t a, float b, size_t c) { return __riscv_vfsub(a, b, c); }
    static inline vfloat32m4_t vmul(vfloat32m4_t a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline vfloat32m4_t vmadd(vfloat32m4_t a, float b, vfloat32m4_t c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
    static inline vfloat32m4_t vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m4(a, b); }
};

template<typename T>
inline int cvtYUVtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isCbCr)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    decltype(rvv<T>::U2B) delta = typeid(T) == typeid(float) ? 0.5f : std::numeric_limits<T>::max() / 2 + 1;
    auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1 : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            auto vec_srcY = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * 3, sizeof(T) * 3, vl), vl);
            auto vec_srcU = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * 3 + 1, sizeof(T) * 3, vl), vl);
            auto vec_srcV = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * 3 + 2, sizeof(T) * 3, vl), vl);

            auto vec_dst = rvv<T>::vmul(rvv<T>::vsub(vec_srcU, delta, vl), isCbCr ? rvv<T>::CB2B : rvv<T>::U2B, vl);
            rvv<T>::vsse(dst + dst_step * i + j * dcn + (swapBlue ? 2 : 0), sizeof(T) * dcn, rvv<T>::vcvt1(vec_dst, vec_srcY, 14, vl), vl);
            vec_dst = rvv<T>::vmul(rvv<T>::vsub(vec_srcU, delta, vl), isCbCr ? rvv<T>::CB2G : rvv<T>::U2G, vl);
            vec_dst = rvv<T>::vmadd(rvv<T>::vsub(vec_srcV, delta, vl), isCbCr ? rvv<T>::CR2G : rvv<T>::V2G, vec_dst, vl);
            rvv<T>::vsse(dst + dst_step * i + j * dcn + 1, sizeof(T) * dcn, rvv<T>::vcvt1(vec_dst, vec_srcY, 14, vl), vl);
            vec_dst = rvv<T>::vmul(rvv<T>::vsub(vec_srcV, delta, vl), isCbCr ? rvv<T>::CR2R : rvv<T>::V2R, vl);
            rvv<T>::vsse(dst + dst_step * i + j * dcn + (swapBlue ? 0 : 2), sizeof(T) * dcn, rvv<T>::vcvt1(vec_dst, vec_srcY, 14, vl), vl);
            if (dcn == 4)
            {
                rvv<T>::vsse(dst + dst_step * i + j * dcn + 3, sizeof(T) * dcn, alpha, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue, bool isCbCr)
{
    switch (depth)
    {
    case CV_8U:
        return color::parallel_for(height, cvtYUVtoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_16U:
        return color::parallel_for(height, cvtYUVtoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_32F:
        return color::parallel_for(height, cvtYUVtoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::YUVtoBGR

namespace BGRtoYUV {
#undef cv_hal_cvtBGRtoYUV
#define cv_hal_cvtBGRtoYUV cv::cv_hal_rvv::BGRtoYUV::cvtBGRtoYUV

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    static constexpr int B2Y = 1868, G2Y = 9617, R2Y = 4899, B2U = 8061, R2V = 14369, YCB = 9241, YCR = 11682;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m1(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline vuint8m1_t vlse(const uchar* a, ptrdiff_t b, size_t c) { return __riscv_vlse8_v_u8m1(a, b, c); }
    static inline void vsse(uchar* a, ptrdiff_t b, vuint8m1_t c, size_t d) { return __riscv_vsse8(a, b, c, d); }
    static inline vint32m4_t vcvt0(vuint8m1_t a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(a, b)); }
    static inline vuint8m1_t vcvt1(vint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(a, 0, c)), b, __RISCV_VXRM_RNU, c), 0, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vssra(vint32m4_t a, size_t b, size_t c) { return __riscv_vssra(a, b, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vsub(vint32m4_t a, vint32m4_t b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline vint32m4_t vmv_v_x(int a, size_t b) { return __riscv_vmv_v_x_i32m4(a, b); }
};
template<> struct rvv<ushort>
{
    static constexpr int B2Y = 1868, G2Y = 9617, R2Y = 4899, B2U = 8061, R2V = 14369, YCB = 9241, YCR = 11682;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline vuint16m2_t vlse(const ushort* a, ptrdiff_t b, size_t c) { return __riscv_vlse16_v_u16m2(a, b, c); }
    static inline void vsse(ushort* a, ptrdiff_t b, vuint16m2_t c, size_t d) { return __riscv_vsse16(a, b, c, d); }
    static inline vint32m4_t vcvt0(vuint16m2_t a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2(a, b)); }
    static inline vuint16m2_t vcvt1(vint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(a, 0, c)), b, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vssra(vint32m4_t a, size_t b, size_t c) { return __riscv_vssra(a, b, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vsub(vint32m4_t a, vint32m4_t b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline vint32m4_t vmv_v_x(int a, size_t b) { return __riscv_vmv_v_x_i32m4(a, b); }
};
template<> struct rvv<float>
{
    static constexpr float B2Y = 0.114f, G2Y = 0.587f, R2Y = 0.299f, B2U = 0.492f, R2V = 0.877f, YCB = 0.564f, YCR = 0.713f;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vfloat32m4_t vlse(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m4(a, b, c); }
    static inline void vsse(float* a, ptrdiff_t b, vfloat32m4_t c, size_t d) { return __riscv_vsse32(a, b, c, d); }
    static inline vfloat32m4_t vcvt0(vfloat32m4_t a, size_t) { return a; }
    static inline vfloat32m4_t vcvt1(vfloat32m4_t a, size_t, size_t) { return a; }
    static inline vfloat32m4_t vssra(vfloat32m4_t a, size_t, size_t) { return a; }
    static inline vfloat32m4_t vsub(vfloat32m4_t a, vfloat32m4_t b, size_t c) { return __riscv_vfsub(a, b, c); }
    static inline vfloat32m4_t vmul(vfloat32m4_t a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline vfloat32m4_t vmadd(vfloat32m4_t a, float b, vfloat32m4_t c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
    static inline vfloat32m4_t vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m4(a, b); }
};

template<typename T>
inline int cvtBGRtoYUV(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isCbCr)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    auto delta = rvv<T>::vmv_v_x((1 << 14) * (typeid(T) == typeid(float) ? 0.5f : std::numeric_limits<T>::max() / 2 + 1), rvv<T>::vsetvlmax());
    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            auto vec_srcB = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * scn + (swapBlue ? 2 : 0), sizeof(T) * scn, vl), vl);
            auto vec_srcG = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * scn + 1, sizeof(T) * scn, vl), vl);
            auto vec_srcR = rvv<T>::vcvt0(rvv<T>::vlse(src + src_step * i + j * scn + (swapBlue ? 0 : 2), sizeof(T) * scn, vl), vl);
            auto vec_dstY = rvv<T>::vmadd(vec_srcB, rvv<T>::B2Y, rvv<T>::vmadd(vec_srcG, rvv<T>::G2Y, rvv<T>::vmul(vec_srcR, rvv<T>::R2Y, vl), vl), vl);
            rvv<T>::vsse(dst + dst_step * i + j * 3, sizeof(T) * 3, rvv<T>::vcvt1(vec_dstY, 14, vl), vl);

            vec_dstY = rvv<T>::vssra(vec_dstY, 14, vl);
            auto vec_dstC = rvv<T>::vmadd(rvv<T>::vsub(vec_srcB, vec_dstY, vl), isCbCr ? rvv<T>::YCB : rvv<T>::B2U, delta, vl);
            rvv<T>::vsse(dst + dst_step * i + j * 3 + 1, sizeof(T) * 3, rvv<T>::vcvt1(vec_dstC, 14, vl), vl);
            vec_dstC = rvv<T>::vmadd(rvv<T>::vsub(vec_srcR, vec_dstY, vl), isCbCr ? rvv<T>::YCR : rvv<T>::R2V, delta, vl);
            rvv<T>::vsse(dst + dst_step * i + j * 3 + 2, sizeof(T) * 3, rvv<T>::vcvt1(vec_dstC, 14, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isCbCr)
{
    switch (depth)
    {
    case CV_8U:
        return color::parallel_for(height, cvtBGRtoYUV<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_16U:
        return color::parallel_for(height, cvtBGRtoYUV<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_32F:
        return color::parallel_for(height, cvtBGRtoYUV<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoYUV

}}

#endif
