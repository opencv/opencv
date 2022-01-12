// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_DETAIL_VARIADIC_CONTINUOUS_CHECKER_HPP
#define OPENCV_CORE_DETAIL_VARIADIC_CONTINUOUS_CHECKER_HPP


#include "opencv2/core.hpp"
#include <type_traits>
#include <utility>
#include "opencv2/core/detail/util.hh"

namespace cv {
namespace detail {

//Definitions
template <typename Arg,
          enable_if_t<!std::is_base_of<MatConstIterator, Arg>::value,
                           bool> = true>
std::pair<bool, bool> __iterator__replaceable(Arg &&arg);

template <typename Arg,
          enable_if_t<std::is_base_of<MatConstIterator, Arg>::value,
                           bool> = true>
std::pair<bool, bool> __iterator__replaceable(Arg &&it);

//Reverse iterator
template <typename Arg,
          enable_if_t<std::is_base_of<MatConstIterator, typename Arg::base>::value,
                           bool> = true>
std::pair<bool, bool> __iterator__replaceable(Arg &&it);

template <typename Arg, typename... Args,
          enable_if_t<!std::is_base_of<MatConstIterator, Arg>::value,
                           bool> = true>
std::pair<bool, bool> __iterator__replaceable(Arg &&arg, Args &&... args);

template <typename Arg, typename... Args,
          enable_if_t<std::is_base_of<MatConstIterator, Arg>::value,
                           bool> = true>
std::pair<bool, bool> __iterator__replaceable(Arg &&it, Args &&... args);

//Reverse Iterator
template <typename Arg, typename... Args,
          enable_if_t<std::is_base_of<MatConstIterator, typename Arg::base>::value,
                           bool> = true>
std::pair<bool, bool> __iterator__replaceable(Arg &&it, Args &&... args);

template <typename... Args> bool __iterators__replaceable(Args &&... args);

/// This is how we loop through the tuple.
/// We replace all instances of a MatIterator  with its pointer.
/// Last round in the recursive call

/// These two functions replace a c++17 if constexpr.
/// We return true in the generic case because that's the neutral element of the
/// && operation
template <
    typename Arg,
    enable_if_t<!std::is_base_of<MatConstIterator, Arg>::value, bool>>
std::pair<bool, bool> __iterator__replaceable(Arg && arg) {
    (void)arg;
  return std::make_pair(true, false);
}

/// These two functions replace a c++17 if constexpr.
/// We return if the iterator is contiguous
template <
    typename Arg,
    enable_if_t<std::is_base_of<MatConstIterator, Arg>::value, bool>>
std::pair<bool, bool> __iterator__replaceable(Arg &&it) {
  return std::make_pair(it.m->isContinuous(), true);
}

template <
    typename Arg, typename... Args,
    enable_if_t<!std::is_base_of<MatConstIterator, Arg>::value, bool>>
std::pair<bool, bool> __iterator__replaceable(Arg && arg, Args &&... args) {
(void) arg;
  std::pair<bool, bool> continuousPair =
      __iterator__replaceable(std::forward<Args>(args)...);
  return std::make_pair(true && continuousPair.first,
                        false || continuousPair.second);
}

/// These two functions replace a c++17 if constexpr.
/// We return if the iterator is contiguous
template <
    typename Arg, typename... Args,
    enable_if_t<std::is_base_of<MatConstIterator, Arg>::value, bool>>
std::pair<bool, bool> __iterator__replaceable(Arg &&it, Args &&... args) {

  std::pair<bool, bool> continuousPair =
      __iterator__replaceable(std::forward<Args>(args)...);

  return std::make_pair(it.m->isContinuous() && continuousPair.first, true);
}


/// We loop through the touple and try to find if we can replace iterators by
/// its pointers This is only valid, if there Mat matrices described by the
/// iterators in there, that are all contiguous in memory!
template <typename... Args> bool __iterators__replaceable(Args &&... args) {

  std::pair<bool, bool> continuousPair =
      __iterator__replaceable(std::forward<Args>(args)...);
  return continuousPair.first && continuousPair.second;
}

} // namespace detail
} // namespace cv
#endif // OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP
