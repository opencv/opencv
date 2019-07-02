#ifndef OPENCV_OPENVX_HAL_HPP_INCLUDED
#define OPENCV_OPENVX_HAL_HPP_INCLUDED

#include "opencv2/core/hal/interface.h"

#include "VX/vx.h"

template <typename T>
int ovx_hal_add(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h);
template <typename T>
int ovx_hal_sub(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h);

template <typename T>
int ovx_hal_absdiff(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h);

template <typename T>
int ovx_hal_and(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h);
template <typename T>
int ovx_hal_or(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h);
template <typename T>
int ovx_hal_xor(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h);
int ovx_hal_not(const uchar *a, size_t astep, uchar *c, size_t cstep, int w, int h);

template <typename T>
int ovx_hal_mul(const T *a, size_t astep, const T *b, size_t bstep, T *c, size_t cstep, int w, int h, double scale);

int ovx_hal_merge8u(const uchar **src_data, uchar *dst_data, int len, int cn);
int ovx_hal_resize(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, double inv_scale_x, double inv_scale_y, int interpolation);
int ovx_hal_warpAffine(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, const double M[6], int interpolation, int borderType, const double borderValue[4]);
int ovx_hal_warpPerspective(int atype, const uchar *a, size_t astep, int aw, int ah, uchar *b, size_t bstep, int bw, int bh, const double M[9], int interpolation, int borderType, const double borderValue[4]);

struct cvhalFilter2D;
int ovx_hal_filterInit(cvhalFilter2D **filter_context, uchar *kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height,
                       int, int, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool allowSubmatrix, bool allowInplace);
int ovx_hal_filterFree(cvhalFilter2D *filter_context);
int ovx_hal_filter(cvhalFilter2D *filter_context, uchar *a, size_t astep, uchar *b, size_t bstep, int w, int h, int, int, int, int);
int ovx_hal_sepFilterInit(cvhalFilter2D **filter_context, int src_type, int dst_type,
                          int kernel_type, uchar *kernelx_data, int kernelx_length, uchar *kernely_data, int kernely_length,
                          int anchor_x, int anchor_y, double delta, int borderType);

#if VX_VERSION > VX_VERSION_1_0
int ovx_hal_morphInit(cvhalFilter2D **filter_context, int operation, int src_type, int dst_type, int , int ,
                      int kernel_type, uchar *kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                      int borderType, const double borderValue[4], int iterations, bool allowSubmatrix, bool allowInplace);
int ovx_hal_morphFree(cvhalFilter2D *filter_context);
int ovx_hal_morph(cvhalFilter2D *filter_context, uchar *a, size_t astep, uchar *b, size_t bstep, int w, int h, int , int , int , int , int , int , int , int );
#endif // 1.0 guard

int ovx_hal_cvtBGRtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int depth, int acn, int bcn, bool swapBlue);
int ovx_hal_cvtGraytoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int depth, int bcn);
int ovx_hal_cvtTwoPlaneYUVtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int bcn, bool swapBlue, int uIdx);
int ovx_hal_cvtThreePlaneYUVtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int bcn, bool swapBlue, int uIdx);
int ovx_hal_cvtBGRtoThreePlaneYUV(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int acn, bool swapBlue, int uIdx);
int ovx_hal_cvtOnePlaneYUVtoBGR(const uchar * a, size_t astep, uchar * b, size_t bstep, int w, int h, int bcn, bool swapBlue, int uIdx, int ycn);
int ovx_hal_integral(int depth, int sdepth, int, const uchar * a, size_t astep, uchar * b, size_t bstep, uchar * c, size_t, uchar * d, size_t, int w, int h, int cn);

//==================================================================================================
// functions redefinition
// ...

#undef cv_hal_add8u
#define cv_hal_add8u ovx_hal_add<uchar>
#undef cv_hal_add16s
#define cv_hal_add16s ovx_hal_add<short>
#undef cv_hal_sub8u
#define cv_hal_sub8u ovx_hal_sub<uchar>
#undef cv_hal_sub16s
#define cv_hal_sub16s ovx_hal_sub<short>

#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u ovx_hal_absdiff<uchar>
#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s ovx_hal_absdiff<short>

#undef cv_hal_and8u
#define cv_hal_and8u ovx_hal_and<uchar>
#undef cv_hal_or8u
#define cv_hal_or8u ovx_hal_or<uchar>
#undef cv_hal_xor8u
#define cv_hal_xor8u ovx_hal_xor<uchar>
#undef cv_hal_not8u
#define cv_hal_not8u ovx_hal_not

#undef cv_hal_mul8u
#define cv_hal_mul8u ovx_hal_mul<uchar>
#undef cv_hal_mul16s
#define cv_hal_mul16s ovx_hal_mul<short>

#undef cv_hal_merge8u
#define cv_hal_merge8u ovx_hal_merge8u

//#undef cv_hal_resize
//#define cv_hal_resize ovx_hal_resize

//OpenVX warps use round to zero policy at least in sample implementation
//while OpenCV require round to nearest
//#undef cv_hal_warpAffine
//#define cv_hal_warpAffine ovx_hal_warpAffine
//#undef cv_hal_warpPerspective
//#define cv_hal_warpPerspective ovx_hal_warpPerspective

#undef cv_hal_filterInit
#define cv_hal_filterInit ovx_hal_filterInit
#undef cv_hal_filter
#define cv_hal_filter ovx_hal_filter
#undef cv_hal_filterFree
#define cv_hal_filterFree ovx_hal_filterFree

//#undef cv_hal_sepFilterInit
//#define cv_hal_sepFilterInit ovx_hal_sepFilterInit
//#undef cv_hal_sepFilter
//#define cv_hal_sepFilter ovx_hal_filter
//#undef cv_hal_sepFilterFree
//#define cv_hal_sepFilterFree ovx_hal_filterFree

#if VX_VERSION > VX_VERSION_1_0

#undef cv_hal_morphInit
#define cv_hal_morphInit ovx_hal_morphInit
#undef cv_hal_morph
#define cv_hal_morph ovx_hal_morph
#undef cv_hal_morphFree
#define cv_hal_morphFree ovx_hal_morphFree

#endif // 1.0 guard

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR ovx_hal_cvtBGRtoBGR
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR ovx_hal_cvtGraytoBGR
#undef cv_hal_cvtTwoPlaneYUVtoBGR
#define cv_hal_cvtTwoPlaneYUVtoBGR ovx_hal_cvtTwoPlaneYUVtoBGR
#undef cv_hal_cvtThreePlaneYUVtoBGR
#define cv_hal_cvtThreePlaneYUVtoBGR ovx_hal_cvtThreePlaneYUVtoBGR
#undef cv_hal_cvtBGRtoThreePlaneYUV
#define cv_hal_cvtBGRtoThreePlaneYUV ovx_hal_cvtBGRtoThreePlaneYUV
#undef cv_hal_cvtOnePlaneYUVtoBGR
#define cv_hal_cvtOnePlaneYUVtoBGR ovx_hal_cvtOnePlaneYUVtoBGR
#undef cv_hal_integral
#define cv_hal_integral ovx_hal_integral

#endif
