// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_CORE_DETAIL_POINTER_TUPLE_REPLACER_HPP
#define OPENCV_CORE_DETAIL_POINTER_TUPLE_REPLACER_HPP

#include "opencv2/core.hpp"
#include <tuple>
#include <type_traits>
#include "opencv2/core/detail/util.hh"

namespace cv {
namespace detail {

/// We recursively go through the variadic template pack and use overload
/// resolution to either
/// append elements to the tuple that are not inheriting (i.e. are no) openCV
/// iterators and replace the openCV iterators with a pointer to their data.
/// This is part 1) of 2) In another header we check if this is something
/// valid to do

/// Definitions for being globally accessible
//*********************definitions*************************************/

//*********************************************************************/

template <typename T>constexpr  auto replace_cv_it(cv::MatConstIterator_<T> it) -> T * {
  return (T *)it.ptr;
}

template <typename T>constexpr  auto replace_cv_it(cv::MatIterator_<T> it) -> T * {
  return (T *)it.ptr;
}

template <
    typename T,
    enable_if_t<!std::is_base_of<cv::MatConstIterator, T>::value, bool> = true>
auto replace_single_element(T t) -> T {
  return t;
}

template <
    typename T,
    enable_if_t<std::is_base_of<cv::MatConstIterator, T>::value, bool> = true>
constexpr auto replace_single_element(T t) -> decltype(replace_cv_it(t)) {
  return replace_cv_it(t);
}

template <typename T1>
constexpr auto make_tpl_replaced(T1 t1)
    -> decltype(std::make_tuple(replace_single_element(t1))) {
  return std::make_tuple(replace_single_element(t1));
}

template <typename T1, typename T2>
constexpr auto make_tpl_replaced(T1 t1, T2 t2)
    -> decltype(std::make_tuple(replace_single_element(t1),
                                replace_single_element(t2))) {
  return std::make_tuple(replace_single_element(t1),
                         replace_single_element(t2));
}

template <typename T1, typename T2, typename T3>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3)
    -> decltype(std::make_tuple(replace_single_element(t1),
                                replace_single_element(t2),
                                replace_single_element(t3))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3));
}

template <typename T1, typename T2, typename T3, typename T4>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3, T4 t4)
    -> decltype(std::make_tuple(replace_single_element(t1),
                                replace_single_element(t2),
                                replace_single_element(t3),
                                replace_single_element(t4))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3),
                         replace_single_element(t4));
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5)
    -> decltype(std::make_tuple(replace_single_element(t1),
                                replace_single_element(t2),
                                replace_single_element(t3),
                                replace_single_element(t4),
                                replace_single_element(t5))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3), replace_single_element(t4),
                         replace_single_element(t5));
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6) -> decltype(
    std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                    replace_single_element(t3), replace_single_element(t4),
                    replace_single_element(t5), replace_single_element(t6))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3), replace_single_element(t4),
                         replace_single_element(t5),
                         replace_single_element(t6));
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7)
    -> decltype(
        std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                        replace_single_element(t3), replace_single_element(t4),
                        replace_single_element(t5), replace_single_element(t6),
                        replace_single_element(t7))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3), replace_single_element(t4),
                         replace_single_element(t5), replace_single_element(t6),
                         replace_single_element(t7));
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7, typename T8>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8)
    -> decltype(std::make_tuple(
        replace_single_element(t1), replace_single_element(t2),
        replace_single_element(t3), replace_single_element(t4),
        replace_single_element(t5), replace_single_element(t6),
        replace_single_element(t7), replace_single_element(t8))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3), replace_single_element(t4),
                         replace_single_element(t5), replace_single_element(t6),
                         replace_single_element(t7),
                         replace_single_element(t8));
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7, typename T8, typename T9>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8,
                       T9 t9)
    -> decltype(
        std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                        replace_single_element(t3), replace_single_element(t4),
                        replace_single_element(t5), replace_single_element(t6),
                        replace_single_element(t7), replace_single_element(t8),
                        replace_single_element(t9))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3), replace_single_element(t4),
                         replace_single_element(t5), replace_single_element(t6),
                         replace_single_element(t7), replace_single_element(t8),
                         replace_single_element(t9));
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7, typename T8, typename T9, typename T10>
constexpr auto make_tpl_replaced(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8,
                       T9 t9, T10 t10)
    -> decltype(std::make_tuple(
        replace_single_element(t1), replace_single_element(t2),
        replace_single_element(t3), replace_single_element(t4),
        replace_single_element(t5), replace_single_element(t6),
        replace_single_element(t7), replace_single_element(t8),
        replace_single_element(t9), replace_single_element(t10))) {
  return std::make_tuple(replace_single_element(t1), replace_single_element(t2),
                         replace_single_element(t3), replace_single_element(t4),
                         replace_single_element(t5), replace_single_element(t6),
                         replace_single_element(t7), replace_single_element(t8),
                         replace_single_element(t9),
                         replace_single_element(t10));
}


//Thanks to https://stackoverflow.com/questions/34745581/forbids-functions-with-static-assert#comment57237292_34745581
template <typename...>
struct always_false { static constexpr bool value = false; };

template<typename ... Args> void make_tpl_replaced(Args... ){
static_assert(always_false<Args...>::value,
              "Zeor or more than ten arguments are currently not supported for forwarding to stl.");
}


} // namespace detail
} // namespace cv
#endif // OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP
