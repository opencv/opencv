// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_RVV_HAL_CORE_HPP
#define OPENCV_RVV_HAL_CORE_HPP

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

/* ############ merge ############ */

int merge8u(const uchar** src, uchar* dst, int len, int cn);
int merge16u(const ushort** src, ushort* dst, int len, int cn);
int merge32s(const int** src, int* dst, int len, int cn);
int merge64s(const int64** src, int64* dst, int len, int cn);

#undef cv_hal_merge8u
#define cv_hal_merge8u cv::rvv_hal::core::merge8u
#undef cv_hal_merge16u
#define cv_hal_merge16u cv::rvv_hal::core::merge16u
#undef cv_hal_merge32s
#define cv_hal_merge32s cv::rvv_hal::core::merge32s
#undef cv_hal_merge64s
#define cv_hal_merge64s cv::rvv_hal::core::merge64s

/* ############ meanStdDev ############ */

int meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
               double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev cv::rvv_hal::core::meanStdDev

/* ############ dft ############ */

int dft(const uchar* src, uchar* dst, int depth, int nf, int *factors, double scale,
        int* itab, void* wave, int tab_size, int n, bool isInverse, bool noPermute);

#undef cv_hal_dft
#define cv_hal_dft cv::rvv_hal::core::dft

/* ############ norm ############ */

int norm(const uchar* src, size_t src_step, const uchar* mask, size_t mask_step,
         int width, int height, int type, int norm_type, double* result);

#undef cv_hal_norm
#define cv_hal_norm cv::rvv_hal::core::norm

/* ############ normDiff ############ */

int normDiff(const uchar* src1, size_t src1_step, const uchar* src2, size_t src2_step,
             const uchar* mask, size_t mask_step, int width, int height, int type,
             int norm_type, double* result);

#undef cv_hal_normDiff
#define cv_hal_normDiff cv::rvv_hal::core::normDiff

/* ############ normHamming ############ */

int normHamming8u(const uchar* a, int n, int cellSize, int* result);
int normHammingDiff8u(const uchar* a, const uchar* b, int n, int cellSize, int* result);

#undef cv_hal_normHamming8u
#define cv_hal_normHamming8u cv::rvv_hal::core::normHamming8u
#undef cv_hal_normHammingDiff8u
#define cv_hal_normHammingDiff8u cv::rvv_hal::core::normHammingDiff8u

/* ############ convertScale ############ */

int convertScale(const uchar* src, size_t src_step, uchar* dst, size_t dst_step,
                 int width, int height, int sdepth, int ddepth, double alpha, double beta);

#undef cv_hal_convertScale
#define cv_hal_convertScale cv::rvv_hal::core::convertScale

/* ############ minMaxIdx ############ */

int minMaxIdx(const uchar* src_data, size_t src_step, int width, int height, int depth,
              double* minVal, double* maxVal, int* minIdx, int* maxIdx, uchar* mask, size_t mask_step = 0);

#undef cv_hal_minMaxIdx
#define cv_hal_minMaxIdx cv::rvv_hal::core::minMaxIdx
#undef cv_hal_minMaxIdxMaskStep
#define cv_hal_minMaxIdxMaskStep cv::rvv_hal::core::minMaxIdx

/* ############ fastAtan ############ */

int fast_atan_32(const float* y, const float* x, float* dst, size_t n, bool angle_in_deg);
int fast_atan_64(const double* y, const double* x, double* dst, size_t n, bool angle_in_deg);

#undef cv_hal_fastAtan32f
#define cv_hal_fastAtan32f cv::rvv_hal::core::fast_atan_32
#undef cv_hal_fastAtan64f
#define cv_hal_fastAtan64f cv::rvv_hal::core::fast_atan_64

/* ############ split ############ */

int split8u(const uchar* src, uchar** dst, int len, int cn);

#undef cv_hal_split8u
#define cv_hal_split8u cv::rvv_hal::core::split8u

/* ############ sqrt ############ */

int sqrt32f(const float* src, float* dst, int _len);
int sqrt64f(const double* src, double* dst, int _len);

#undef cv_hal_sqrt32f
#define cv_hal_sqrt32f cv::rvv_hal::core::sqrt32f
#undef cv_hal_sqrt64f
#define cv_hal_sqrt64f cv::rvv_hal::core::sqrt64f

int invSqrt32f(const float* src, float* dst, int _len);
int invSqrt64f(const double* src, double* dst, int _len);

#undef cv_hal_invSqrt32f
#define cv_hal_invSqrt32f cv::rvv_hal::core::invSqrt32f
#undef cv_hal_invSqrt64f
#define cv_hal_invSqrt64f cv::rvv_hal::core::invSqrt64f

/* ############ magnitude ############ */

int magnitude32f(const float *x, const float *y, float *dst, int len);
int magnitude64f(const double *x, const double  *y, double *dst, int len);

#undef cv_hal_magnitude32f
#define cv_hal_magnitude32f cv::rvv_hal::core::magnitude32f
#undef cv_hal_magnitude64f
#define cv_hal_magnitude64f cv::rvv_hal::core::magnitude64f

/* ############ cartToPolar ############ */

int cartToPolar32f(const float* x, const float* y, float* mag, float* angle, int len, bool angleInDegrees);
int cartToPolar64f(const double* x, const double* y, double* mag, double* angle, int len, bool angleInDegrees);

#undef cv_hal_cartToPolar32f
#define cv_hal_cartToPolar32f cv::rvv_hal::core::cartToPolar32f
#undef cv_hal_cartToPolar64f
#define cv_hal_cartToPolar64f cv::rvv_hal::core::cartToPolar64f

/* ############ polarToCart ############ */

int polarToCart32f(const float* mag, const float* angle, float* x, float* y, int len, bool angleInDegrees);
int polarToCart64f(const double* mag, const double* angle, double* x, double* y, int len, bool angleInDegrees);

#undef cv_hal_polarToCart32f
#define cv_hal_polarToCart32f cv::rvv_hal::core::polarToCart32f
#undef cv_hal_polarToCart64f
#define cv_hal_polarToCart64f cv::rvv_hal::core::polarToCart64f

/* ############ polarToCart ############ */

int flip(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
         uchar* dst_data, size_t dst_step, int flip_mode);

#undef cv_hal_flip
#define cv_hal_flip cv::rvv_hal::core::flip

/* ############ lut ############ */

int lut(const uchar* src_data, size_t src_step, size_t src_type,
        const uchar* lut_data, size_t lut_channel_size, size_t lut_channels,
        uchar* dst_data, size_t dst_step, int width, int height);

#undef cv_hal_lut
#define cv_hal_lut cv::rvv_hal::core::lut

/* ############ exp ############ */

int exp32f(const float* src, float* dst, int _len);
int exp64f(const double* src, double* dst, int _len);

#undef cv_hal_exp32f
#define cv_hal_exp32f cv::rvv_hal::core::exp32f
#undef cv_hal_exp64f
#define cv_hal_exp64f cv::rvv_hal::core::exp64f

/* ############ log ############ */

int log32f(const float* src, float* dst, int _len);
int log64f(const double* src, double* dst, int _len);

#undef cv_hal_log32f
#define cv_hal_log32f cv::rvv_hal::core::log32f
#undef cv_hal_log64f
#define cv_hal_log64f cv::rvv_hal::core::log64f

/* ############ lu ############ */

int LU32f(float* src1, size_t src1_step, int m, float* src2, size_t src2_step, int n, int* info);
int LU64f(double* src1, size_t src1_step, int m, double* src2, size_t src2_step, int n, int* info);

#undef cv_hal_LU32f
#define cv_hal_LU32f cv::rvv_hal::core::LU32f
#undef cv_hal_LU64f
#define cv_hal_LU64f cv::rvv_hal::core::LU64f

/* ############ cholesky ############ */

int Cholesky32f(float* src1, size_t src1_step, int m, float* src2, size_t src2_step, int n, bool* info);
int Cholesky64f(double* src1, size_t src1_step, int m, double* src2, size_t src2_step, int n, bool* info);

#undef cv_hal_Cholesky32f
#define cv_hal_Cholesky32f cv::rvv_hal::core::Cholesky32f
#undef cv_hal_Cholesky64f
#define cv_hal_Cholesky64f cv::rvv_hal::core::Cholesky64f

/* ############ qr ############ */

int QR32f(float* src1, size_t src1_step, int m, int n, int k, float* src2, size_t src2_step, float* dst, int* info);
int QR64f(double* src1, size_t src1_step, int m, int n, int k, double* src2, size_t src2_step, double* dst, int* info);

#undef cv_hal_QR32f
#define cv_hal_QR32f cv::rvv_hal::core::QR32f
#undef cv_hal_QR64f
#define cv_hal_QR64f cv::rvv_hal::core::QR64f

/* ############ SVD ############ */

int SVD32f(float* src, size_t src_step, float* w, float* u, size_t u_step, float* vt, size_t vt_step, int m, int n, int flags);
int SVD64f(double* src, size_t src_step, double* w, double* u, size_t u_step, double* vt, size_t vt_step, int m, int n, int flags);

#undef cv_hal_SVD32f
#define cv_hal_SVD32f cv::rvv_hal::core::SVD32f
#undef cv_hal_SVD64f
#define cv_hal_SVD64f cv::rvv_hal::core::SVD64f

/* ############ copyToMasked ############ */

int copyToMasked(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height,
                 int type, const uchar *mask_data, size_t mask_step, int mask_type);

#undef cv_hal_copyToMasked
#define cv_hal_copyToMasked cv::rvv_hal::core::copyToMasked

/* ############ div, recip ############ */

int div8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, double scale);
int div8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, schar *dst_data, size_t dst_step, int width, int height, double scale);
int div16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, ushort *dst_data, size_t dst_step, int width, int height, double scale);
int div16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, short *dst_data, size_t dst_step, int width, int height, double scale);
int div32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, int *dst_data, size_t dst_step, int width, int height, double scale);
int div32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, float *dst_data, size_t dst_step, int width, int height, double scale);
// int div64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, double *dst_data, size_t dst_step, int width, int height, double scale);

#undef cv_hal_div8u
#define cv_hal_div8u cv::rvv_hal::core::div8u
#undef cv_hal_div8s
#define cv_hal_div8s cv::rvv_hal::core::div8s
#undef cv_hal_div16u
#define cv_hal_div16u cv::rvv_hal::core::div16u
#undef cv_hal_div16s
#define cv_hal_div16s cv::rvv_hal::core::div16s
#undef cv_hal_div32s
#define cv_hal_div32s cv::rvv_hal::core::div32s
#undef cv_hal_div32f
#define cv_hal_div32f cv::rvv_hal::core::div32f
// #undef cv_hal_div64f
// #define cv_hal_div64f cv::rvv_hal::core::div64f

int recip8u(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height, double scale);
int recip8s(const schar *src_data, size_t src_step, schar *dst_data, size_t dst_step, int width, int height, double scale);
int recip16u(const ushort *src_data, size_t src_step, ushort *dst_data, size_t dst_step, int width, int height, double scale);
int recip16s(const short *src_data, size_t src_step, short *dst_data, size_t dst_step, int width, int height, double scale);
int recip32s(const int *src_data, size_t src_step, int *dst_data, size_t dst_step, int width, int height, double scale);
int recip32f(const float *src_data, size_t src_step, float *dst_data, size_t dst_step, int width, int height, double scale);
// int recip64f(const double *src_data, size_t src_step, double *dst_data, size_t dst_step, int width, int height, double scale);

#undef cv_hal_recip8u
#define cv_hal_recip8u cv::rvv_hal::core::recip8u
#undef cv_hal_recip8s
#define cv_hal_recip8s cv::rvv_hal::core::recip8s
#undef cv_hal_recip16u
#define cv_hal_recip16u cv::rvv_hal::core::recip16u
#undef cv_hal_recip16s
#define cv_hal_recip16s cv::rvv_hal::core::recip16s
#undef cv_hal_recip32s
#define cv_hal_recip32s cv::rvv_hal::core::recip32s
#undef cv_hal_recip32f
#define cv_hal_recip32f cv::rvv_hal::core::recip32f
// #undef cv_hal_recip64f
// #define cv_hal_recip64f cv::rvv_hal::core::recip64f

/* ############ dotProduct ############ */

int dotprod(const uchar *a_data, size_t a_step, const uchar *b_data, size_t b_step,
            int width, int height, int type, double *dot_val);

#undef cv_hal_dotProduct
#define cv_hal_dotProduct cv::rvv_hal::core::dotprod

/* ############ compare ############ */

int cmp8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation);
int cmp8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation);
int cmp16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation);
int cmp16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation);
int cmp32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation);
int cmp32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation);
// int cmp64f(const double *src1_data, size_t src1_step, const double *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation);

#undef cv_hal_cmp8u
#define cv_hal_cmp8u cv::rvv_hal::core::cmp8u
#undef cv_hal_cmp8s
#define cv_hal_cmp8s cv::rvv_hal::core::cmp8s
#undef cv_hal_cmp16u
#define cv_hal_cmp16u cv::rvv_hal::core::cmp16u
#undef cv_hal_cmp16s
#define cv_hal_cmp16s cv::rvv_hal::core::cmp16s
#undef cv_hal_cmp32s
#define cv_hal_cmp32s cv::rvv_hal::core::cmp32s
#undef cv_hal_cmp32f
#define cv_hal_cmp32f cv::rvv_hal::core::cmp32f
// #undef cv_hal_cmp64f
// #define cv_hal_cmp64f cv::rvv_hal::core::cmp64f

/* ############ transpose2d ############ */

int transpose2d(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                int src_width, int src_height, int element_size);

#undef cv_hal_transpose2d
#define cv_hal_transpose2d cv::rvv_hal::core::transpose2d

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core

#endif // OPENCV_RVV_HAL_CORE_HPP
