// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_CORE_DETAIL_POINTER_TUPLE_REPLACER_HPP
#define OPENCV_CORE_DETAIL_POINTER_TUPLE_REPLACER_HPP

#include "opencv2/core.hpp"
#include <tuple>
#include <type_traits>
#include <iterator>
#include "opencv2/core/detail/util.hpp"
#include "opencv2/core/detail/cpp_features.hpp"

//This opencv feature requires certain C++14 features
#ifdef _stl_forward_cpp_features_present

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

template <typename T> constexpr T* replace_cv_it(cv::MatConstIterator_<T> it) {
  return (T *)it.ptr;
}

template <typename T>constexpr T* replace_cv_it(cv::MatIterator_<T> it){
  return (T *)it.ptr;
}

template <typename T,
    enable_if_t<!std::is_base_of<cv::MatConstIterator, T>::value, bool> = true>
constexpr auto replace_single_element(T t){
  return t;
}

template <typename T,
    enable_if_t<std::is_base_of<cv::MatConstIterator, T>::value, bool> = true>
constexpr auto replace_single_element(T t){
  return replace_cv_it(t);
}

//Resolves if an iterator is a reverse iterator or a normal forward iterator
template <typename T,
    enable_if_t<!is_reverse_iterator<T>::value, bool> = true>
constexpr auto replace_single_element_directional(T t){
  return replace_single_element(t);
}

//Resolves if an iterator is a reverse iterator or a normal forward iterator
template <typename T,
    enable_if_t<is_reverse_iterator<T>::value, bool> = true>
constexpr auto replace_single_element_directional(T t){
  return std::make_reverse_iterator(replace_single_element(t.base()));
}

template<typename Arg>
constexpr auto make_tpl_replaced(Arg && arg)
{
    return std::make_tuple(replace_single_element_directional(arg));
}


template<typename Arg, typename ... Args>
auto make_tpl_replaced(Arg && arg, Args&& ... args)
{
    return std::tuple_cat(std::make_tuple(replace_single_element_directional(arg)), make_tpl_replaced(args ...));
}

template<typename Arg>
auto get_replaced_val(Arg && arg)
{
    return std::get<0>(make_tpl_replaced(std::forward<Arg>(arg)));
}

} // namespace detail
} // namespace cv
#endif //_stl_forward_cpp_features_present
#endif // OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP
