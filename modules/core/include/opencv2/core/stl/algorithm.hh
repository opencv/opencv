// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_STL_ALGORITHM_HPP
#define OPENCV_CORE_STL_ALGORITHM_HPP


#include "opencv2/core.hpp"
#include "opencv2/core/detail/pointer-tuple-replacer.hpp"
#include "opencv2/core/detail/variadic-continuous-checker.hpp"
#include "opencv2/core/detail/util.hh"
#include <algorithm>
#include <tuple>

namespace cv {
namespace experimental {

///@brief overload for forwarding a tuple and index sequence with cv iterators
/// replaced as pointers
template <typename... Args, std::size_t... Is>
auto count_if(std::tuple<Args...> tpl, cv::detail::index_sequence<Is...>)
    -> decltype(std::count_if(std::get<Is>(tpl)...)) {
  return std::count_if(std::get<Is>(tpl)...);
}

///@brief Forwarding for count_if stl algo. Decides at runtime if the iterators are replaced with pointers
/// or kept as cv iterators for non-contiguous matrices.
template <typename... Args>
auto count_if(Args&&... args)
    -> decltype(std::count_if(std::forward<Args>(args)...)) {

  if (cv::detail::__iterators__replaceable(std::forward<Args>(args)...)) {
    return count_if(
        cv::detail::make_tpl_replaced(std::forward<Args>(args)...),
        cv::detail::make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>());
  } else {
    return std::count_if(std::forward<Args>(args)...);
  }
}

} // namespace experimental
} // namespace cv
#endif //OPENCV_CORE_STL_ALGORITHM_HPP
