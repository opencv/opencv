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

//Find the first index of an openCV iterator in a tuple or return zero.
//Thanks to https://stackoverflow.com/questions/26855322/how-do-i-get-the-index-of-a-type-matching-some-predicate
template <template <class T> class, typename, long = 0>
struct find_if;

template <template <class T> class Pred, typename T, long pos, typename... tail>
struct find_if<Pred, std::tuple<T, tail...>, pos> :
    std::conditional<Pred<T>::value,
                     std::integral_constant<int64_t, pos>,
                     find_if<Pred, std::tuple<tail...>, pos+1>>::type {};

template <template <class T> class Pred>
struct find_if<Pred, std::tuple<>> : std::integral_constant<int64_t, -1> {};

template <template <class, class> class T, class U>
struct bind
{
    template <class X>
    using first  = T<U, X>;
    template <class X>
    using second = T<X, U>;
};


/// Helper function: Return the index of the first cv::MatConstIterator derived type.
/// Extends the tuple with an iterator such that it will always be found, otherwise compilation fails
template<typename ...Args> constexpr size_t __get_first_cv_it_index()
{
    return find_if<bind<std::is_base_of, cv::MatConstIterator>::first, std::tuple<Args...,cv::MatConstIterator>>::value == sizeof ...(Args) ? 0 : find_if<bind<std::is_base_of, cv::MatConstIterator>::first, std::tuple<Args...,cv::MatConstIterator>>::value;
}


} // namespace detail
} // namespace cv
#endif // OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP
