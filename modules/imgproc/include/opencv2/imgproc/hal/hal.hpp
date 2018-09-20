#ifndef CV_IMGPROC_HAL_HPP
#define CV_IMGPROC_HAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/hal/interface.h"

namespace cv { namespace hal {

//! @addtogroup imgproc_hal_functions
//! @{

//---------------------------
//! @cond IGNORED

struct CV_EXPORTS Filter2D
{
    CV_DEPRECATED static Ptr<hal::Filter2D> create(uchar * , size_t , int ,
                                     int , int ,
                                     int , int ,
                                     int , int ,
                                     int , double ,
                                     int , int ,
                                     bool , bool );
    virtual void apply(uchar * , size_t ,
                       uchar * , size_t ,
                       int , int ,
                       int , int ,
                       int , int ) = 0;
    virtual ~Filter2D() {}
};

struct CV_EXPORTS SepFilter2D
{
    CV_DEPRECATED static Ptr<hal::SepFilter2D> create(int , int , int ,
                                        uchar * , int ,
                                        uchar * , int ,
                                        int , int ,
                                        double , int );
    virtual void apply(uchar * , size_t ,
                       uchar * , size_t ,
                       int , int ,
                       int , int ,
                       int , int ) = 0;
    virtual ~SepFilter2D() {}
};


struct CV_EXPORTS Morph
{
    CV_DEPRECATED static Ptr<hal::Morph> create(int , int , int , int , int ,
                                    int , uchar * , size_t ,
                                    int , int ,
                                    int , int ,
                                    int , const double *,
                                    int , bool , bool );
    virtual void apply(uchar * , size_t , uchar * , size_t , int , int ,
                       int , int , int , int ,
                       int , int , int , int ) = 0;
    virtual ~Morph() {}
};

//! @endcond
//---------------------------

CV_EXPORTS void filter2D(ElemType stype, ElemType dtype, ElemType kernel_type,
                         uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int full_width, int full_height,
                         int offset_x, int offset_y,
                         uchar * kernel_data, size_t kernel_step,
                         int kernel_width, int kernel_height,
                         int anchor_x, int anchor_y,
                         double delta, int borderType,
                         bool isSubmatrix);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, stype, ElemType, stype) ". Similarly, " CV_DEPRECATED_PARAM(int, kernel_type, ElemType, kernel_type) ". Moreover, " CV_DEPRECATED_PARAM(int, dtype, ElemType, dtype))
#  endif
static inline void filter2D(int stype, int dtype, int kernel_type,
                         uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int full_width, int full_height,
                         int offset_x, int offset_y,
                         uchar * kernel_data, size_t kernel_step,
                         int kernel_width, int kernel_height,
                         int anchor_x, int anchor_y,
                         double delta, int borderType,
                         bool isSubmatrix)
{
    filter2D(static_cast<ElemType>(stype), static_cast<ElemType>(dtype), static_cast<ElemType>(kernel_type), src_data,
                    src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y, kernel_data,
                    kernel_step, kernel_width, kernel_height, anchor_x, anchor_y, delta, borderType, isSubmatrix);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, stype, ElemType, stype) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, kernel_type, ElemType, kernel_type) ". Moreover, " CV_DEPRECATED_PARAM(ElemDepth, dtype, ElemType, dtype))
#  endif
static inline void filter2D(ElemDepth stype, ElemDepth dtype, ElemDepth kernel_type,
                         uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int full_width, int full_height,
                         int offset_x, int offset_y,
                         uchar * kernel_data, size_t kernel_step,
                         int kernel_width, int kernel_height,
                         int anchor_x, int anchor_y,
                         double delta, int borderType,
                         bool isSubmatrix)
{
    filter2D(CV_MAKETYPE(stype, 1), CV_MAKETYPE(dtype, 1), CV_MAKETYPE(kernel_type, 1), src_data,
                    src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y, kernel_data,
                    kernel_step, kernel_width, kernel_height, anchor_x, anchor_y, delta, borderType, isSubmatrix);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void sepFilter2D(ElemType stype, ElemType dtype, ElemType ktype,
                            uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int full_width, int full_height,
                            int offset_x, int offset_y,
                            uchar * kernelx_data, int kernelx_len,
                            uchar * kernely_data, int kernely_len,
                            int anchor_x, int anchor_y,
                            double delta, int borderType);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, ktype, ElemType, ktype) ". Similarly, " CV_DEPRECATED_PARAM(int, stype, ElemType, stype) ". Moreover, " CV_DEPRECATED_PARAM(int, dtype, ElemType, dtype))
#  endif
static inline void sepFilter2D(int stype, int dtype, int ktype,
                            uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int full_width, int full_height,
                            int offset_x, int offset_y,
                            uchar * kernelx_data, int kernelx_len,
                            uchar * kernely_data, int kernely_len,
                            int anchor_x, int anchor_y,
                            double delta, int borderType)
{
    sepFilter2D(static_cast<ElemType>(stype), static_cast<ElemType>(dtype), static_cast<ElemType>(ktype),
                       src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x,
                       offset_y, kernelx_data, kernelx_len, kernely_data, kernely_len, anchor_x, anchor_y, delta, borderType);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, ktype, ElemType, ktype) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, stype, ElemType, stype) ". Moreover, " CV_DEPRECATED_PARAM(ElemDepth, dtype, ElemType, dtype))
#  endif
static inline void sepFilter2D(ElemDepth stype, ElemDepth dtype, ElemDepth ktype,
                            uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int full_width, int full_height,
                            int offset_x, int offset_y,
                            uchar * kernelx_data, int kernelx_len,
                            uchar * kernely_data, int kernely_len,
                            int anchor_x, int anchor_y,
                            double delta, int borderType)
{
    sepFilter2D(CV_MAKETYPE(stype, 1), CV_MAKETYPE(dtype, 1), CV_MAKETYPE(ktype, 1),
                       src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x,
                       offset_y, kernelx_data, kernelx_len, kernely_data, kernely_len, anchor_x, anchor_y, delta, borderType);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void morph(int op, ElemType src_type, ElemType dst_type,
                      uchar * src_data, size_t src_step,
                      uchar * dst_data, size_t dst_step,
                      int width, int height,
                      int roi_width, int roi_height, int roi_x, int roi_y,
                      int roi_width2, int roi_height2, int roi_x2, int roi_y2,
                      int kernel_type, uchar * kernel_data, size_t kernel_step,
                      int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                      int borderType, const double borderValue[4],
                      int iterations, bool isSubmatrix);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, dst_type, ElemType, dst_type) ". Similarly, " CV_DEPRECATED_PARAM(int, src_type, ElemType, src_type))
#  endif
static inline void morph(int op, int src_type, int dst_type,
                      uchar * src_data, size_t src_step,
                      uchar * dst_data, size_t dst_step,
                      int width, int height,
                      int roi_width, int roi_height, int roi_x, int roi_y,
                      int roi_width2, int roi_height2, int roi_x2, int roi_y2,
                      int kernel_type, uchar * kernel_data, size_t kernel_step,
                      int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                      int borderType, const double borderValue[4],
                      int iterations, bool isSubmatrix)
{
    morph(op, static_cast<ElemType>(src_type), static_cast<ElemType>(dst_type), src_data, src_step,
                 dst_data, dst_step, width, height, roi_width, roi_height, roi_x, roi_y, roi_width2, roi_height2,
                 roi_x2, roi_y2, kernel_type, kernel_data, kernel_step, kernel_width, kernel_height, anchor_x,
                 anchor_y, borderType, borderValue, iterations, isSubmatrix);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, dst_type, ElemType, dst_type) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, src_type, ElemType, src_type))
#  endif
static inline void morph(int op, ElemDepth src_type, ElemDepth dst_type,
                      uchar * src_data, size_t src_step,
                      uchar * dst_data, size_t dst_step,
                      int width, int height,
                      int roi_width, int roi_height, int roi_x, int roi_y,
                      int roi_width2, int roi_height2, int roi_x2, int roi_y2,
                      int kernel_type, uchar * kernel_data, size_t kernel_step,
                      int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                      int borderType, const double borderValue[4],
                      int iterations, bool isSubmatrix)
{
    morph(op, CV_MAKETYPE(src_type, 1), CV_MAKETYPE(dst_type, 1), src_data, src_step,
                 dst_data, dst_step, width, height, roi_width, roi_height, roi_x, roi_y, roi_width2, roi_height2,
                 roi_x2, roi_y2, kernel_type, kernel_data, kernel_step, kernel_width, kernel_height, anchor_x,
                 anchor_y, borderType, borderValue, iterations, isSubmatrix);
}
#endif // CV_TYPE_COMPATIBLE_API


CV_EXPORTS void resize(ElemType src_type,
                       const uchar * src_data, size_t src_step, int src_width, int src_height,
                       uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int interpolation);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMTYPE_ATTR(src_type, src_type)
static inline void resize(int src_type,
                       const uchar * src_data, size_t src_step, int src_width, int src_height,
                       uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int interpolation)
{
    resize(static_cast<ElemType>(src_type), src_data, src_step, src_width, src_height, dst_data,
                  dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);
}
CV_DEPRECATED_ELEMDEPTH_TO_ELEMTYPE_ATTR(src_type, src_type)
static inline void resize(ElemDepth src_type,
                       const uchar * src_data, size_t src_step, int src_width, int src_height,
                       uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int interpolation)
{
    resize(CV_MAKETYPE(src_type, 1), src_data, src_step, src_width, src_height, dst_data,
                  dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void warpAffine(ElemType src_type,
                           const uchar * src_data, size_t src_step, int src_width, int src_height,
                           uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                           const double M[6], int interpolation, int borderType, const double borderValue[4]);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMTYPE_ATTR(src_type, src_type)
static inline void warpAffine(int src_type,
                           const uchar * src_data, size_t src_step, int src_width, int src_height,
                           uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                           const double M[6], int interpolation, int borderType, const double borderValue[4])
{
    warpAffine(static_cast<ElemType>(src_type), src_data, src_step, src_width, src_height, dst_data,
                      dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue);
}
CV_DEPRECATED_ELEMDEPTH_TO_ELEMTYPE_ATTR(src_type, src_type)
static inline void warpAffine(ElemDepth src_type,
                           const uchar * src_data, size_t src_step, int src_width, int src_height,
                           uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                           const double M[6], int interpolation, int borderType, const double borderValue[4])
{
    warpAffine(CV_MAKETYPE(src_type, 1), src_data, src_step, src_width, src_height, dst_data,
                      dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void warpPerspective(ElemType src_type,
                               const uchar * src_data, size_t src_step, int src_width, int src_height,
                               uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                               const double M[9], int interpolation, int borderType, const double borderValue[4]);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMTYPE_ATTR(src_type, src_type)
static inline void warpPerspective(int src_type,
                               const uchar * src_data, size_t src_step, int src_width, int src_height,
                               uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                               const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    warpPerspective(static_cast<ElemType>(src_type), src_data, src_step, src_width, src_height, dst_data,
                           dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue);
}
CV_DEPRECATED_ELEMDEPTH_TO_ELEMTYPE_ATTR(src_type, src_type)
static inline void warpPerspective(ElemDepth src_type,
                               const uchar * src_data, size_t src_step, int src_width, int src_height,
                               uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                               const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    warpPerspective(CV_MAKETYPE(src_type, 1), src_data, src_step, src_width, src_height, dst_data,
                           dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int scn, int dcn, bool swapBlue);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, int dcn, bool swapBlue)
{
    cvtBGRtoBGR(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), scn, dcn, swapBlue);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int scn, int dcn, bool swapBlue)
{
    cvtBGRtoBGR(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), scn, dcn, swapBlue);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                               uchar * dst_data, size_t dst_step,
                               int width, int height,
                               int scn, bool swapBlue, int greenBits);

CV_EXPORTS void cvtBGR5x5toBGR(const uchar * src_data, size_t src_step,
                               uchar * dst_data, size_t dst_step,
                               int width, int height,
                               int dcn, bool swapBlue, int greenBits);

CV_EXPORTS void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             ElemDepth depth, int scn, bool swapBlue);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             int depth, int scn, bool swapBlue)
{
    cvtBGRtoGray(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), scn, swapBlue);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             ElemType depth, int scn, bool swapBlue)
{
    cvtBGRtoGray(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), scn, swapBlue);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             ElemDepth depth, int dcn);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             int depth, int dcn)
{
    cvtGraytoBGR(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), dcn);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height,
                             ElemType depth, int dcn)
{
    cvtGraytoBGR(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), dcn);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtBGR5x5toGray(const uchar * src_data, size_t src_step,
                                uchar * dst_data, size_t dst_step,
                                int width, int height,
                                int greenBits);

CV_EXPORTS void cvtGraytoBGR5x5(const uchar * src_data, size_t src_step,
                                uchar * dst_data, size_t dst_step,
                                int width, int height,
                                int greenBits);
CV_EXPORTS void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int scn, bool swapBlue, bool isCbCr);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue, bool isCbCr)
{
    cvtBGRtoYUV(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), scn, swapBlue, isCbCr);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int scn, bool swapBlue, bool isCbCr)
{
    cvtBGRtoYUV(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), scn, swapBlue, isCbCr);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int dcn, bool swapBlue, bool isCbCr);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int dcn, bool swapBlue, bool isCbCr)
{
    cvtYUVtoBGR(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), dcn, swapBlue, isCbCr);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int dcn, bool swapBlue, bool isCbCr)
{
    cvtYUVtoBGR(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), dcn, swapBlue, isCbCr);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtBGRtoXYZ(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int scn, bool swapBlue);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoXYZ(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue)
{
    cvtBGRtoXYZ(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), scn, swapBlue);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoXYZ(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int scn, bool swapBlue)
{
    cvtBGRtoXYZ(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), scn, swapBlue);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtXYZtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int dcn, bool swapBlue);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtXYZtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int dcn, bool swapBlue)
{
    cvtXYZtoBGR(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), dcn, swapBlue);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtXYZtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int dcn, bool swapBlue)
{
    cvtXYZtoBGR(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), dcn, swapBlue);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int scn, bool swapBlue, bool isFullRange, bool isHSV);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    cvtBGRtoHSV(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), scn, swapBlue, isFullRange, isHSV);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    cvtBGRtoHSV(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), scn, swapBlue, isFullRange, isHSV);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    cvtHSVtoBGR(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), dcn, swapBlue, isFullRange, isHSV);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
ElemType depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    cvtHSVtoBGR(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), dcn, swapBlue, isFullRange, isHSV);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtBGRtoLab(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int scn, bool swapBlue, bool isLab, bool srgb);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoLab(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    cvtBGRtoLab(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), scn, swapBlue, isLab, srgb);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtBGRtoLab(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    cvtBGRtoLab(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), scn, swapBlue, isLab, srgb);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtLabtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemDepth depth, int dcn, bool swapBlue, bool isLab, bool srgb);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtLabtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            int depth, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    cvtLabtoBGR(src_data, src_step, dst_data, dst_step, width, height, static_cast<ElemDepth>(depth), dcn, swapBlue, isLab, srgb);
}
CV_DEPRECATED_ELEMTYPE_TO_ELEMDEPTH_ATTR(depth, depth)
static inline void cvtLabtoBGR(const uchar * src_data, size_t src_step,
                            uchar * dst_data, size_t dst_step,
                            int width, int height,
                            ElemType depth, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    cvtLabtoBGR(src_data, src_step, dst_data, dst_step, width, height, CV_MAT_DEPTH(depth), dcn, swapBlue, isLab, srgb);
}
#endif // CV_TYPE_COMPATIBLE_API

CV_EXPORTS void cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                    uchar * dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    int dcn, bool swapBlue, int uIdx);

//! Separate Y and UV planes
CV_EXPORTS void cvtTwoPlaneYUVtoBGR(const uchar * y_data, const uchar * uv_data, size_t src_step,
                                    uchar * dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    int dcn, bool swapBlue, int uIdx);

CV_EXPORTS void cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                      uchar * dst_data, size_t dst_step,
                                      int dst_width, int dst_height,
                                      int dcn, bool swapBlue, int uIdx);

CV_EXPORTS void cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step,
                                      uchar * dst_data, size_t dst_step,
                                      int width, int height,
                                      int scn, bool swapBlue, int uIdx);

//! Separate Y and UV planes
CV_EXPORTS void cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                                    uchar * y_data, uchar * uv_data, size_t dst_step,
                                    int width, int height,
                                    int scn, bool swapBlue, int uIdx);

CV_EXPORTS void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                                    uchar * dst_data, size_t dst_step,
                                    int width, int height,
                                    int dcn, bool swapBlue, int uIdx, int ycn);

CV_EXPORTS void cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step,
                                        uchar * dst_data, size_t dst_step,
                                        int width, int height);

CV_EXPORTS void cvtMultipliedRGBAtoRGBA(const uchar * src_data, size_t src_step,
                                        uchar * dst_data, size_t dst_step,
                                        int width, int height);

CV_EXPORTS void integral(ElemDepth depth, ElemDepth sdepth, ElemDepth sqdepth,
                         const uchar* src, size_t srcstep,
                         uchar* sum, size_t sumstep,
                         uchar* sqsum, size_t sqsumstep,
                         uchar* tilted, size_t tstep,
                         int width, int height, int cn);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, sdepth, ElemDepth, sdepth) ". Similarly, " CV_DEPRECATED_PARAM(int, depth, ElemDepth, depth) ". Moreover, " CV_DEPRECATED_PARAM(int, sqdepth, ElemDepth, sqdepth))
#  endif
static inline void integral(int depth, int sdepth, int sqdepth,
                         const uchar* src, size_t srcstep,
                         uchar* sum, size_t sumstep,
                         uchar* sqsum, size_t sqsumstep,
                         uchar* tilted, size_t tstep,
                         int width, int height, int cn)
{
    integral(static_cast<ElemDepth>(depth), static_cast<ElemDepth>(sdepth), static_cast<ElemDepth>(sqdepth), src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tstep, width, height, cn);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemType, sdepth, ElemDepth, sdepth) ". Similarly, " CV_DEPRECATED_PARAM(ElemType, depth, ElemDepth, depth) ". Moreover, " CV_DEPRECATED_PARAM(ElemType, sqdepth, ElemDepth, sqdepth))
#  endif
static inline void integral(ElemType depth, ElemType sdepth, ElemType sqdepth,
                         const uchar* src, size_t srcstep,
                         uchar* sum, size_t sumstep,
                         uchar* sqsum, size_t sqsumstep,
                         uchar* tilted, size_t tstep,
                         int width, int height, int cn)
{
    integral(CV_MAT_DEPTH(depth), CV_MAT_DEPTH(sdepth), CV_MAT_DEPTH(sqdepth), src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tstep, width, height, cn);
}
#endif // CV_TYPE_COMPATIBLE_API

//! @}

}}

#endif // CV_IMGPROC_HAL_HPP
