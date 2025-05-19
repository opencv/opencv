// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file has been created for compatibility with older versions of Universal Intrinscs
// Binary operators for vector types has been removed since version 4.11
// Include this file manually after OpenCV headers if you need these operators

#ifndef OPENCV_HAL_INTRIN_LEGACY_OPS_HPP
#define OPENCV_HAL_INTRIN_LEGACY_OPS_HPP

#ifdef __OPENCV_BUILD
#error "Universal Intrinsics operators are deprecated and should not be used in OpenCV library"
#endif

#ifdef __riscv
#warning "Operators might conflict with built-in functions on RISC-V platform"
#endif

#if defined(CV_VERSION) && CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 11
#warning "Older versions of OpenCV (<4.11) already have Universal Intrinscs operators"
#endif


namespace cv { namespace hal {

#define BIN_OP(OP, FUN) \
template <typename R> R operator OP (const R & lhs, const R & rhs) { return FUN(lhs, rhs); }

#define BIN_A_OP(OP, FUN) \
template <typename R> R & operator OP (R & res, const R & val) { res = FUN(res, val); return res; }

#define UN_OP(OP, FUN) \
template <typename R> R operator OP (const R & val) { return FUN(val); }

BIN_OP(+, v_add)
BIN_OP(-, v_sub)
BIN_OP(*, v_mul)
BIN_OP(/, v_div)
BIN_OP(&, v_and)
BIN_OP(|, v_or)
BIN_OP(^, v_xor)

BIN_OP(==, v_eq)
BIN_OP(!=, v_ne)
BIN_OP(<, v_lt)
BIN_OP(>, v_gt)
BIN_OP(<=, v_le)
BIN_OP(>=, v_ge)

BIN_A_OP(+=, v_add)
BIN_A_OP(-=, v_sub)
BIN_A_OP(*=, v_mul)
BIN_A_OP(/=, v_div)
BIN_A_OP(&=, v_and)
BIN_A_OP(|=, v_or)
BIN_A_OP(^=, v_xor)

UN_OP(~, v_not)

// TODO: shift operators?

}} // cv::hal::

//==============================================================================

#ifdef OPENCV_ENABLE_INLINE_INTRIN_OPERATOR_TEST

namespace cv { namespace hal {

inline static void opencv_operator_compile_test()
{
    using namespace cv;
    v_float32 a, b, c;
    uint8_t shift = 1;
    a = b + c;
    a = b - c;
    a = b * c;
    a = b / c;
    a = b & c;
    a = b | c;
    a = b ^ c;
    // a = b >> shift;
    // a = b << shift;

    a = (b == c);
    a = (b != c);
    a = (b < c);}}
    a = (b > c);
    a = (b <= c);
    a = (b >= c);

    a += b;
    a -= b;
    a *= b;
    a /= b;
    a &= b;
    a |= b;
    a ^= b;
    // a <<= shift;
    // a >>= shift;

    a = ~b;
}

}} // cv::hal::

#endif


#endif // OPENCV_HAL_INTRIN_LEGACY_OPS_HPP
