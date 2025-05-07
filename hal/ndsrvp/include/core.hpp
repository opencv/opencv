// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_NDSRVP_CORE_HPP
#define OPENCV_NDSRVP_CORE_HPP

namespace cv {

namespace ndsrvp {

template <typename srctype, typename dsttype,
    typename vsrctype, typename vdsttype, int nlane,
    template <typename src, typename dst> typename operators_t,
    typename... params_t>
int elemwise_binop(const srctype* src1_data, size_t src1_step,
    const srctype* src2_data, size_t src2_step,
    dsttype* dst_data, size_t dst_step,
    int width, int height, params_t... params)
{
    src1_step /= sizeof(srctype);
    src2_step /= sizeof(srctype);
    dst_step /= sizeof(dsttype);

    operators_t<srctype, dsttype> operators;

    int i, j;
    for (i = 0; i < height; ++i) {
        const srctype* src1_row = src1_data + (src1_step * i);
        const srctype* src2_row = src2_data + (src2_step * i);
        dsttype* dst_row = dst_data + (dst_step * i);

        j = 0;
        for (; j + nlane <= width; j += nlane) {
            register vsrctype vs1 = *(vsrctype*)(src1_row + j);
            register vsrctype vs2 = *(vsrctype*)(src2_row + j);

            *(vdsttype*)(dst_row + j) = operators.vector(vs1, vs2, params...);
        }
        for (; j < width; j++)
            dst_row[j] = operators.scalar(src1_row[j], src2_row[j], params...);
    }

    return CV_HAL_ERROR_OK;
}

template <typename srctype, typename dsttype,
    typename vsrctype, typename vdsttype, int nlane,
    template <typename src, typename dst> typename operators_t,
    typename... params_t>
int elemwise_unop(const srctype* src_data, size_t src_step,
    dsttype* dst_data, size_t dst_step,
    int width, int height, params_t... params)
{
    src_step /= sizeof(srctype);
    dst_step /= sizeof(dsttype);

    operators_t<srctype, dsttype> operators;

    int i, j;
    for (i = 0; i < height; ++i) {
        const srctype* src_row = src_data + (src_step * i);
        dsttype* dst_row = dst_data + (dst_step * i);

        j = 0;
        for (; j + nlane <= width; j += nlane) {
            register vsrctype vs = *(vsrctype*)(src_row + j);

            *(vdsttype*)(dst_row + j) = operators.vector(vs, params...);
        }
        for (; j < width; j++)
            dst_row[j] = operators.scalar(src_row[j], params...);
    }

    return CV_HAL_ERROR_OK;
}

// ################ add ################

template <typename src, typename dst>
struct operators_add_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return __nds__v_ukadd8(a, b); }
    inline uchar scalar(uchar a, uchar b) { return __nds__ukadd8(a, b); }

    inline int8x8_t vector(int8x8_t a, int8x8_t b) { return __nds__v_kadd8(a, b); }
    inline schar scalar(schar a, schar b) { return __nds__kadd8(a, b); }

    inline uint16x4_t vector(uint16x4_t a, uint16x4_t b) { return __nds__v_ukadd16(a, b); }
    inline ushort scalar(ushort a, ushort b) { return __nds__ukadd16(a, b); }

    inline int16x4_t vector(int16x4_t a, int16x4_t b) { return __nds__v_kadd16(a, b); }
    inline short scalar(short a, short b) { return __nds__kadd16(a, b); }

    inline int32x2_t vector(int32x2_t a, int32x2_t b) { return __nds__v_kadd32(a, b); }
    inline int scalar(int a, int b) { return __nds__kadd32(a, b); }
};

#undef cv_hal_add8u
#define cv_hal_add8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_add_t>)

#undef cv_hal_add8s
#define cv_hal_add8s (cv::ndsrvp::elemwise_binop<schar, schar, int8x8_t, int8x8_t, 8, cv::ndsrvp::operators_add_t>)

#undef cv_hal_add16u
#define cv_hal_add16u (cv::ndsrvp::elemwise_binop<ushort, ushort, uint16x4_t, uint16x4_t, 4, cv::ndsrvp::operators_add_t>)

#undef cv_hal_add16s
#define cv_hal_add16s (cv::ndsrvp::elemwise_binop<short, short, int16x4_t, int16x4_t, 4, cv::ndsrvp::operators_add_t>)

#undef cv_hal_add32s
#define cv_hal_add32s (cv::ndsrvp::elemwise_binop<int, int, int32x2_t, int32x2_t, 2, cv::ndsrvp::operators_add_t>)

// ################ sub ################

template <typename src, typename dst>
struct operators_sub_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return __nds__v_uksub8(a, b); }
    inline uchar scalar(uchar a, uchar b) { return __nds__uksub8(a, b); }

    inline int8x8_t vector(int8x8_t a, int8x8_t b) { return __nds__v_ksub8(a, b); }
    inline schar scalar(schar a, schar b) { return __nds__ksub8(a, b); }

    inline uint16x4_t vector(uint16x4_t a, uint16x4_t b) { return __nds__v_uksub16(a, b); }
    inline ushort scalar(ushort a, ushort b) { return __nds__uksub16(a, b); }

    inline int16x4_t vector(int16x4_t a, int16x4_t b) { return __nds__v_ksub16(a, b); }
    inline short scalar(short a, short b) { return __nds__ksub16(a, b); }

    inline int32x2_t vector(int32x2_t a, int32x2_t b) { return __nds__v_ksub32(a, b); }
    inline int scalar(int a, int b) { return __nds__ksub32(a, b); }
};

#undef cv_hal_sub8u
#define cv_hal_sub8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_sub_t>)

#undef cv_hal_sub8s
#define cv_hal_sub8s (cv::ndsrvp::elemwise_binop<schar, schar, int8x8_t, int8x8_t, 8, cv::ndsrvp::operators_sub_t>)

#undef cv_hal_sub16u
#define cv_hal_sub16u (cv::ndsrvp::elemwise_binop<ushort, ushort, uint16x4_t, uint16x4_t, 4, cv::ndsrvp::operators_sub_t>)

#undef cv_hal_sub16s
#define cv_hal_sub16s (cv::ndsrvp::elemwise_binop<short, short, int16x4_t, int16x4_t, 4, cv::ndsrvp::operators_sub_t>)

#undef cv_hal_sub32s
#define cv_hal_sub32s (cv::ndsrvp::elemwise_binop<int, int, int32x2_t, int32x2_t, 2, cv::ndsrvp::operators_sub_t>)

// ################ max ################

template <typename src, typename dst>
struct operators_max_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return __nds__v_umax8(a, b); }
    inline uchar scalar(uchar a, uchar b) { return __nds__umax8(a, b); }

    inline int8x8_t vector(int8x8_t a, int8x8_t b) { return __nds__v_smax8(a, b); }
    inline schar scalar(schar a, schar b) { return __nds__smax8(a, b); }

    inline uint16x4_t vector(uint16x4_t a, uint16x4_t b) { return __nds__v_umax16(a, b); }
    inline ushort scalar(ushort a, ushort b) { return __nds__umax16(a, b); }

    inline int16x4_t vector(int16x4_t a, int16x4_t b) { return __nds__v_smax16(a, b); }
    inline short scalar(short a, short b) { return __nds__smax16(a, b); }

    inline int32x2_t vector(int32x2_t a, int32x2_t b) { return __nds__v_smax32(a, b); }
    inline int scalar(int a, int b) { return __nds__smax32(a, b); }
};

#undef cv_hal_max8u
#define cv_hal_max8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_max_t>)

#undef cv_hal_max8s
#define cv_hal_max8s (cv::ndsrvp::elemwise_binop<schar, schar, int8x8_t, int8x8_t, 8, cv::ndsrvp::operators_max_t>)

#undef cv_hal_max16u
#define cv_hal_max16u (cv::ndsrvp::elemwise_binop<ushort, ushort, uint16x4_t, uint16x4_t, 4, cv::ndsrvp::operators_max_t>)

#undef cv_hal_max16s
#define cv_hal_max16s (cv::ndsrvp::elemwise_binop<short, short, int16x4_t, int16x4_t, 4, cv::ndsrvp::operators_max_t>)

#undef cv_hal_max32s
#define cv_hal_max32s (cv::ndsrvp::elemwise_binop<int, int, int32x2_t, int32x2_t, 2, cv::ndsrvp::operators_max_t>)

// ################ min ################

template <typename src, typename dst>
struct operators_min_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return __nds__v_umin8(a, b); }
    inline uchar scalar(uchar a, uchar b) { return __nds__umin8(a, b); }

    inline int8x8_t vector(int8x8_t a, int8x8_t b) { return __nds__v_smin8(a, b); }
    inline schar scalar(schar a, schar b) { return __nds__smin8(a, b); }

    inline uint16x4_t vector(uint16x4_t a, uint16x4_t b) { return __nds__v_umin16(a, b); }
    inline ushort scalar(ushort a, ushort b) { return __nds__umin16(a, b); }

    inline int16x4_t vector(int16x4_t a, int16x4_t b) { return __nds__v_smin16(a, b); }
    inline short scalar(short a, short b) { return __nds__smin16(a, b); }

    inline int32x2_t vector(int32x2_t a, int32x2_t b) { return __nds__v_smin32(a, b); }
    inline int scalar(int a, int b) { return __nds__smin32(a, b); }
};

#undef cv_hal_min8u
#define cv_hal_min8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_min_t>)

#undef cv_hal_min8s
#define cv_hal_min8s (cv::ndsrvp::elemwise_binop<schar, schar, int8x8_t, int8x8_t, 8, cv::ndsrvp::operators_min_t>)

#undef cv_hal_min16u
#define cv_hal_min16u (cv::ndsrvp::elemwise_binop<ushort, ushort, uint16x4_t, uint16x4_t, 4, cv::ndsrvp::operators_min_t>)

#undef cv_hal_min16s
#define cv_hal_min16s (cv::ndsrvp::elemwise_binop<short, short, int16x4_t, int16x4_t, 4, cv::ndsrvp::operators_min_t>)

#undef cv_hal_min32s
#define cv_hal_min32s (cv::ndsrvp::elemwise_binop<int, int, int32x2_t, int32x2_t, 2, cv::ndsrvp::operators_min_t>)

// ################ absdiff ################

template <typename src, typename dst>
struct operators_absdiff_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return __nds__v_uksub8(__nds__v_umax8(a, b), __nds__v_umin8(a, b)); }
    inline uchar scalar(uchar a, uchar b) { return __nds__uksub8(__nds__umax8(a, b), __nds__umin8(a, b)); }

    inline int8x8_t vector(int8x8_t a, int8x8_t b) { return __nds__v_ksub8(__nds__v_smax8(a, b), __nds__v_smin8(a, b)); }
    inline schar scalar(schar a, schar b) { return __nds__ksub8(__nds__smax8(a, b), __nds__smin8(a, b)); }

    inline uint16x4_t vector(uint16x4_t a, uint16x4_t b) { return __nds__v_uksub16(__nds__v_umax16(a, b), __nds__v_umin16(a, b)); }
    inline ushort scalar(ushort a, ushort b) { return __nds__uksub16(__nds__umax16(a, b), __nds__umin16(a, b)); }

    inline int16x4_t vector(int16x4_t a, int16x4_t b) { return __nds__v_ksub16(__nds__v_smax16(a, b), __nds__v_smin16(a, b)); }
    inline short scalar(short a, short b) { return __nds__ksub16(__nds__smax16(a, b), __nds__smin16(a, b)); }

    inline int32x2_t vector(int32x2_t a, int32x2_t b) { return __nds__v_ksub32(__nds__v_smax32(a, b), __nds__v_smin32(a, b)); }
    inline int scalar(int a, int b) { return __nds__ksub32(__nds__smax32(a, b), __nds__smin32(a, b)); }
};

#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_absdiff_t>)

#undef cv_hal_absdiff8s
#define cv_hal_absdiff8s (cv::ndsrvp::elemwise_binop<schar, schar, int8x8_t, int8x8_t, 8, cv::ndsrvp::operators_absdiff_t>)

#undef cv_hal_absdiff16u
#define cv_hal_absdiff16u (cv::ndsrvp::elemwise_binop<ushort, ushort, uint16x4_t, uint16x4_t, 4, cv::ndsrvp::operators_absdiff_t>)

#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s (cv::ndsrvp::elemwise_binop<short, short, int16x4_t, int16x4_t, 4, cv::ndsrvp::operators_absdiff_t>)

#undef cv_hal_absdiff32s
#define cv_hal_absdiff32s (cv::ndsrvp::elemwise_binop<int, int, int32x2_t, int32x2_t, 2, cv::ndsrvp::operators_absdiff_t>)

// ################ bitwise ################

template <typename src, typename dst>
struct operators_and_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return a & b; }
    inline uchar scalar(uchar a, uchar b) { return a & b; }
};

#undef cv_hal_and8u
#define cv_hal_and8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_and_t>)

template <typename src, typename dst>
struct operators_or_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return a | b; }
    inline uchar scalar(uchar a, uchar b) { return a | b; }
};

#undef cv_hal_or8u
#define cv_hal_or8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_or_t>)

template <typename src, typename dst>
struct operators_xor_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b) { return a ^ b; }
    inline uchar scalar(uchar a, uchar b) { return a ^ b; }
};

#undef cv_hal_xor8u
#define cv_hal_xor8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_xor_t>)

template <typename src, typename dst>
struct operators_not_t {
    inline uint8x8_t vector(uint8x8_t a) { return ~a; }
    inline uchar scalar(uchar a) { return ~a; }
};

#undef cv_hal_not8u
#define cv_hal_not8u (cv::ndsrvp::elemwise_unop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_not_t>)

// ################ cmp ################

template <typename src, typename dst>
struct operators_cmp_t {
    inline uint8x8_t vector(uint8x8_t a, uint8x8_t b, int operation)
    {
        switch (operation) {
        case CV_HAL_CMP_EQ:
            return __nds__v_ucmpeq8(a, b);
        case CV_HAL_CMP_GT:
            return __nds__v_ucmplt8(b, a);
        case CV_HAL_CMP_GE:
            return __nds__v_ucmple8(b, a);
        case CV_HAL_CMP_LT:
            return __nds__v_ucmplt8(a, b);
        case CV_HAL_CMP_LE:
            return __nds__v_ucmple8(a, b);
        case CV_HAL_CMP_NE:
            return ~__nds__v_ucmpeq8(a, b);
        default:
            return uint8x8_t();
        }
    }
    inline uchar scalar(uchar a, uchar b, int operation)
    {
        switch (operation) {
        case CV_HAL_CMP_EQ:
            return __nds__cmpeq8(a, b);
        case CV_HAL_CMP_GT:
            return __nds__ucmplt8(b, a);
        case CV_HAL_CMP_GE:
            return __nds__ucmple8(b, a);
        case CV_HAL_CMP_LT:
            return __nds__ucmplt8(a, b);
        case CV_HAL_CMP_LE:
            return __nds__ucmple8(a, b);
        case CV_HAL_CMP_NE:
            return ~__nds__cmpeq8(a, b);
        default:
            return 0;
        }
    }

    inline uint8x8_t vector(int8x8_t a, int8x8_t b, int operation)
    {
        switch (operation) {
        case CV_HAL_CMP_EQ:
            return __nds__v_scmpeq8(a, b);
        case CV_HAL_CMP_GT:
            return __nds__v_scmplt8(b, a);
        case CV_HAL_CMP_GE:
            return __nds__v_scmple8(b, a);
        case CV_HAL_CMP_LT:
            return __nds__v_scmplt8(a, b);
        case CV_HAL_CMP_LE:
            return __nds__v_scmple8(a, b);
        case CV_HAL_CMP_NE:
            return ~__nds__v_scmpeq8(a, b);
        default:
            return uint8x8_t();
        }
    }
    inline uchar scalar(schar a, schar b, int operation)
    {
        switch (operation) {
        case CV_HAL_CMP_EQ:
            return __nds__cmpeq8(a, b);
        case CV_HAL_CMP_GT:
            return __nds__scmplt8(b, a);
        case CV_HAL_CMP_GE:
            return __nds__scmple8(b, a);
        case CV_HAL_CMP_LT:
            return __nds__scmplt8(a, b);
        case CV_HAL_CMP_LE:
            return __nds__scmple8(a, b);
        case CV_HAL_CMP_NE:
            return ~__nds__cmpeq8(a, b);
        default:
            return 0;
        }
    }

    inline uint8x4_t vector(uint16x4_t a, uint16x4_t b, int operation)
    {
        register unsigned long cmp;
        switch (operation) {
        case CV_HAL_CMP_EQ:
            cmp = (unsigned long)__nds__v_ucmpeq16(a, b) >> 8;
            break;
        case CV_HAL_CMP_GT:
            cmp = (unsigned long)__nds__v_ucmplt16(b, a) >> 8;
            break;
        case CV_HAL_CMP_GE:
            cmp = (unsigned long)__nds__v_ucmple16(b, a) >> 8;
            break;
        case CV_HAL_CMP_LT:
            cmp = (unsigned long)__nds__v_ucmplt16(a, b) >> 8;
            break;
        case CV_HAL_CMP_LE:
            cmp = (unsigned long)__nds__v_ucmple16(a, b) >> 8;
            break;
        case CV_HAL_CMP_NE:
            cmp = ~(unsigned long)__nds__v_ucmpeq16(a, b) >> 8;
            break;
        default:
            return uint8x4_t();
        }
        return (uint8x4_t)(unsigned int)__nds__pkbb16(cmp >> 32, cmp);
    }
    inline uchar scalar(ushort a, ushort b, int operation)
    {
        switch (operation) {
        case CV_HAL_CMP_EQ:
            return __nds__cmpeq16(a, b);
        case CV_HAL_CMP_GT:
            return __nds__ucmplt16(b, a);
        case CV_HAL_CMP_GE:
            return __nds__ucmple16(b, a);
        case CV_HAL_CMP_LT:
            return __nds__ucmplt16(a, b);
        case CV_HAL_CMP_LE:
            return __nds__ucmple16(a, b);
        case CV_HAL_CMP_NE:
            return ~__nds__cmpeq16(a, b);
        default:
            return 0;
        }
    }

    inline uint8x4_t vector(int16x4_t a, int16x4_t b, int operation)
    {
        register unsigned long cmp;
        switch (operation) {
        case CV_HAL_CMP_EQ:
            cmp = (unsigned long)__nds__v_scmpeq16(a, b) >> 8;
            break;
        case CV_HAL_CMP_GT:
            cmp = (unsigned long)__nds__v_scmplt16(b, a) >> 8;
            break;
        case CV_HAL_CMP_GE:
            cmp = (unsigned long)__nds__v_scmple16(b, a) >> 8;
            break;
        case CV_HAL_CMP_LT:
            cmp = (unsigned long)__nds__v_scmplt16(a, b) >> 8;
            break;
        case CV_HAL_CMP_LE:
            cmp = (unsigned long)__nds__v_scmple16(a, b) >> 8;
            break;
        case CV_HAL_CMP_NE:
            cmp = ~(unsigned long)__nds__v_scmpeq16(a, b) >> 8;
            break;
        default:
            return uint8x4_t();
        }
        return (uint8x4_t)(unsigned int)__nds__pkbb16(cmp >> 32, cmp);
    }
    inline uchar scalar(short a, short b, int operation)
    {
        switch (operation) {
        case CV_HAL_CMP_EQ:
            return __nds__cmpeq16(a, b);
        case CV_HAL_CMP_GT:
            return __nds__scmplt16(b, a);
        case CV_HAL_CMP_GE:
            return __nds__scmple16(b, a);
        case CV_HAL_CMP_LT:
            return __nds__scmplt16(a, b);
        case CV_HAL_CMP_LE:
            return __nds__scmple16(a, b);
        case CV_HAL_CMP_NE:
            return ~__nds__cmpeq16(a, b);
        default:
            return 0;
        }
    }
};

#undef cv_hal_cmp8u
#define cv_hal_cmp8u (cv::ndsrvp::elemwise_binop<uchar, uchar, uint8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_cmp_t>)

#undef cv_hal_cmp8s
#define cv_hal_cmp8s (cv::ndsrvp::elemwise_binop<schar, uchar, int8x8_t, uint8x8_t, 8, cv::ndsrvp::operators_cmp_t>)

#undef cv_hal_cmp16u
#define cv_hal_cmp16u (cv::ndsrvp::elemwise_binop<ushort, uchar, uint16x4_t, uint8x4_t, 4, cv::ndsrvp::operators_cmp_t>)

#undef cv_hal_cmp16s
#define cv_hal_cmp16s (cv::ndsrvp::elemwise_binop<short, uchar, int16x4_t, uint8x4_t, 4, cv::ndsrvp::operators_cmp_t>)

// ################ split ################

/*template <typename srctype, typename vsrctype, int nlane>
int split(const srctype* src_data, srctype** dst_data, int len, int cn)
{
    int i, j;
    for (i = 0; i < len; i++) {
        for (j = 0; j < cn; j++) {
            dst_data[j][i] = src_data[i * cn + j];
        }
    }

    return CV_HAL_ERROR_OK;
}

#undef cv_hal_split8u
#define cv_hal_split8u (cv::ndsrvp::split<uchar, uint8x8_t, 8>)

#undef cv_hal_split16u
#define cv_hal_split16u (cv::ndsrvp::split<ushort, uint16x4_t, 4>)

#undef cv_hal_split32s
#define cv_hal_split32s (cv::ndsrvp::split<int, int32x2_t, 2>)*/

// ################ merge ################

/*template <typename srctype, typename vsrctype, int nlane>
int merge(const srctype** src_data, srctype* dst_data, int len, int cn)
{
    int i, j;
    for (i = 0; i < len; i++) {
        for (j = 0; j < cn; j++) {
            dst_data[i * cn + j] = src_data[j][i];
        }
    }

    return CV_HAL_ERROR_OK;
}

#undef cv_hal_merge8u
#define cv_hal_merge8u (cv::ndsrvp::merge<uchar, uint8x8_t, 8>)

#undef cv_hal_merge16u
#define cv_hal_merge16u (cv::ndsrvp::merge<ushort, uint16x4_t, 4>)

#undef cv_hal_merge32s
#define cv_hal_merge32s (cv::ndsrvp::merge<int, int32x2_t, 2>)*/

} // namespace ndsrvp

} // namespace cv

#endif
