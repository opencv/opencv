// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_DETAIL_VARIADIC_CONTINUOUS_CHECKER_HPP
#define OPENCV_CORE_DETAIL_VARIADIC_CONTINUOUS_CHECKER_HPP


#include "opencv2/core.hpp"
#include <type_traits>
#include <utility>
#include "opencv2/core/detail/util.hpp"

namespace cv {
namespace detail {

/// We loop through the touple and try to find if we can replace iterators by
/// its pointers This is only valid, if
/// 1) There is at least one opencvIt
/// 2) All of the openCV iterators describe conitguous memory
std::pair<bool, bool> ___it_replacable();
std::pair<bool, bool> ___it_replacable()
{
    return {false, true};
}

template <typename Arg, typename... Args,
          enable_if_t<!std::is_base_of<MatConstIterator, Arg>::value, bool> = true>
std::pair<bool, bool> __it_replacable_rev_resolved(Arg &&, Args &&... args);

template <typename Arg, typename... Args,
          enable_if_t<std::is_base_of<MatConstIterator, Arg>::value, bool> = true>
std::pair<bool, bool> __it_replacable_rev_resolved(Arg && it, Args &&... args);

template <typename Arg, typename... Args,
          enable_if_t<!is_reverse_iterator<Arg>::value, bool> = true>
std::pair<bool, bool> ___it_replacable(Arg && arg, Args &&... args) {
    return __it_replacable_rev_resolved(std::forward<Arg>(arg), std::forward<Args>(args)...);
}

template <typename Arg, typename... Args,
          enable_if_t<is_reverse_iterator<Arg>::value, bool> = true>
std::pair<bool, bool> ___it_replacable(Arg && arg, Args &&... args) {
    return __it_replacable_rev_resolved(arg.base(), std::forward<Args>(args)...);
}

//Those two work with already resolved reverse iterators
template <typename Arg, typename... Args,
          enable_if_t<!std::is_base_of<MatConstIterator, Arg>::value, bool>>
std::pair<bool, bool> __it_replacable_rev_resolved(Arg &&, Args &&... args) {
    return ___it_replacable(std::forward<Args>(args)...);
}

template <typename Arg, typename... Args,
          enable_if_t<std::is_base_of<MatConstIterator, Arg>::value, bool>>
std::pair<bool, bool> __it_replacable_rev_resolved(Arg && it, Args &&... args) {
    if(!it.m->isContinuous())
    {
        return {true, false};
    }

     return {true, ___it_replacable(std::forward<Args>(args)...).second};
}

///Convienience function to decide if we can perform substitution with pointers
///  for a given set of arguments
template <typename... Args>
bool __it_replacable(Args &&... args) {
    const auto pair = ___it_replacable(std::forward<Args>(args)...);
    return (pair.first && pair.second);
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

template <template <class, class, class> class T, class U>
struct bind
{
    template <class X>
    using first  = T<U, X, void>;
    template <class X>
    using second = T<X, U, void>;
};

/// Helper function: Return the index of the first cv::MatConstIterator derived type.
/// Extends the tuple with an iterator such that it will always be found, otherwise compilation fails
template<typename ...Args> constexpr size_t __get_first_cv_it_index()
{
    return find_if<bind<is_base_of_reverse, cv::MatConstIterator>::first, std::tuple<Args...,cv::MatConstIterator>>::value == sizeof ...(Args) ? 0 : find_if<bind<is_base_of_reverse, cv::MatConstIterator>::first, std::tuple<Args...,cv::MatConstIterator>>::value;
}



} // namespace detail
} // namespace cv
#endif // OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP
