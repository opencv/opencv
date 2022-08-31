// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_STL_ALGORITHM_HPP
#define OPENCV_CORE_STL_ALGORITHM_HPP


#include "opencv2/core.hpp"
#include "opencv2/core/detail/pointer-tuple-replacer.hpp"
#include "opencv2/core/detail/variadic-continuous-checker.hpp"
#include "opencv2/core/detail/util.hh"
#include <tuple>

namespace cv {
namespace experimental {


///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto _all_of_impl(std::tuple<Args...> tpl, cv::detail::index_sequence<Is...>){
  return std::all_of(std::get<Is>(tpl)...);
}

///@brief Forwarding for stl algorithm with the same name. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args>
auto all_of(Args&&... args) -> decltype(std::all_of(std::forward<Args>(args)...)){

  if (cv::detail::__it_replacable(std::forward<Args>(args)...)) {
    return _all_of_impl(
        cv::detail::make_tpl_replaced(std::forward<Args>(args)...),
        cv::detail::make_index_sequence_variadic<Args...>());
  } else {
    return std::all_of(std::forward<Args>(args)...);
  }
}


///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto _any_of_impl(std::tuple<Args...> tpl, cv::detail::index_sequence<Is...>) {
  return std::any_of(std::get<Is>(tpl)...);
}

///@brief Forwarding for stl algorithm with the same name. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args>
auto any_of(Args&&... args) -> decltype(std::any_of(std::forward<Args>(args)...)) {

  if (cv::detail::__it_replacable(std::forward<Args>(args)...)) {
    return _any_of_impl(
        cv::detail::make_tpl_replaced(std::forward<Args>(args)...),
        cv::detail::make_index_sequence_variadic<Args...>());
  } else {
    return std::any_of(std::forward<Args>(args)...);
  }
}

///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto _none_of_impl(std::tuple<Args...> tpl, cv::detail::index_sequence<Is...>) {
  return std::none_of(std::get<Is>(tpl)...);
}

///@brief Forwarding for stl algorithm with the same name. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args>
auto none_of(Args&&... args) -> decltype(std::none_of(std::forward<Args>(args)...)){

  if (cv::detail::__it_replacable(std::forward<Args>(args)...)) {
    return _none_of_impl(
        cv::detail::make_tpl_replaced(std::forward<Args>(args)...),
        cv::detail::make_index_sequence_variadic<Args...>());
  } else {
    return std::none_of(std::forward<Args>(args)...);
  }
}

///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto _count_if_impl(std::tuple<Args...> tpl, cv::detail::index_sequence<Is...>){
  return std::count_if(std::get<Is>(tpl)...);
}

///@brief Forwarding for count_if stl algo. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args>
auto count_if(Args&&... args)
    -> decltype(std::count_if(std::forward<Args>(args)...)) {

  if (cv::detail::__it_replacable(std::forward<Args>(args)...)) {
    return _count_if_impl(
        cv::detail::make_tpl_replaced(std::forward<Args>(args)...),
        cv::detail::make_index_sequence_variadic<Args...>());
  } else {
    return std::count_if(std::forward<Args>(args)...);
  }
}

///@brief Forwarding for find stl algo. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload for when we do return an opencv iterator.
/// This means, that we'll use pointer arithmetic to get the offset and then add it to the begin iterator.
template <typename beginIt, typename... Args, std::size_t... Is>
auto _find_impl_calc_diff(beginIt && begin,std::tuple<Args...> tpl, cv::detail::index_sequence<Is...>) {
    auto beginPtr = std::find(std::get<Is>(tpl)...);

    //Offsets to go for iterator
    std::ptrdiff_t diff = beginPtr - (decltype(beginPtr)) begin.ptr;
    return begin + diff;
}


///@brief Forwarding for find stl algo. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices. This is the overload for when we don't return an opencv iterator
template <typename... Args, std::size_t... Is>
auto _find_impl_only_replace(std::tuple<Args...> tpl, cv::detail::index_sequence<Is...>){
    return std::find(std::get<Is>(tpl)...);
}

///@brief Decide if an openCV iterator is returned. When this is not the case no special care needs to be taken
template <typename ReturnType,  typename beginIt, typename... Args,
    cv::detail::enable_if_t<!std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto _find_impl_trampolin(beginIt&&, Args&& ... args)
  -> ReturnType {
    return _find_impl_only_replace(cv::detail::make_tpl_replaced(std::forward<Args>(args)...),
                                   cv::detail::make_index_sequence_variadic<Args...>());
}

///@brief Decide if an openCV iterator is returned. When this is not the case we need to calculate the offset to the cv iterator we want to return
template <typename ReturnType,  typename beginIt, typename... Args,
    cv::detail::enable_if_t<std::is_base_of<cv::MatConstIterator, ReturnType>::value, bool> = true>
auto _find_impl_trampolin(beginIt&& it, Args&& ... args)
  -> ReturnType {
    return _find_impl_calc_diff(std::forward<beginIt>(it),
                                cv::detail::make_tpl_replaced(std::forward<Args>(args)...),
                                cv::detail::make_index_sequence_variadic<Args...>());
}

///@brief Decide if an openCV iterator is returned. When this is not the case we need to calculate the offset to the cv iterator we want to return
template <typename ReturnType,  typename beginIt, typename... Args,
    cv::detail::enable_if_t<!cv::detail::is_reverse_iterator<ReturnType>::value, bool> = true>
auto _find_impl(beginIt&& it, Args&& ... args)
  -> ReturnType {
    return _find_impl_trampolin<ReturnType, beginIt, Args...>(it, std::forward<Args>(args)...);
}


///@brief Decide if an openCV iterator is returned. When this is not the case we need to calculate the offset to the cv iterator we want to return
template <typename ReturnType,  typename beginIt, typename... Args,
    cv::detail::enable_if_t<cv::detail::is_reverse_iterator<ReturnType>::value, bool> = true>
auto _find_impl(beginIt&& it, Args&& ... args)
  -> ReturnType {

    return _find_impl_trampolin<ReturnType, beginIt, Args...>(it, std::forward<Args>(args)...);
}

///@brief Forwarding for find stl algo. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args>
auto find(Args&&... args)
    -> decltype(std::find(std::forward<Args>(args)...)) {

  if (cv::detail::__it_replacable(std::forward<Args>(args)...)) {
      constexpr size_t val = cv::detail::__get_first_cv_it_index<Args...>();
      auto tpl_frwd = std::make_tuple(std::forward<Args>(args)...);

      using ReturnType = decltype(std::find(std::forward<Args>(args)...));
      using beginIt = decltype(std::get<val>(tpl_frwd));

      //Explicitely mention templates to avoid requireing ReturnType being default constructable
      return _find_impl<ReturnType, beginIt, Args...>(std::get<val>(tpl_frwd), std::forward<Args>(args)...);
  } else {
    return std::find(std::forward<Args>(args)...);
  }
}

} // namespace experimental
} // namespace cv
#endif //OPENCV_CORE_STL_ALGORITHM_HPP
