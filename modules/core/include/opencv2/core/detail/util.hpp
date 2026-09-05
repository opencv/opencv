// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_DETAIL_UTIL_HPP
#define OPENCV_CORE_DETAIL_UTIL_HPP
#include <tuple>

//Forward definition for type matching
namespace  cv{
class MatConstIterator;
}

namespace cv {
namespace detail {

template <size_t... Ints> struct index_sequence {
  using type = index_sequence;
  using value_type = size_t;
  static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
};

// --------------------------------------------------------------

template <class Sequence1, class Sequence2> struct _merge_and_renumber;

template <size_t... I1, size_t... I2>
struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
    : index_sequence<I1..., (sizeof...(I1) + I2)...> {};

// --------------------------------------------------------------


template <size_t N>
struct make_index_sequence
    : _merge_and_renumber<typename make_index_sequence<N / 2>::type,
                          typename make_index_sequence<N - N / 2>::type> {};

template <> struct make_index_sequence<0> : index_sequence<> {};
template <> struct make_index_sequence<1> : index_sequence<0> {};

//Make an index sequence matching a variadic template pack
template<typename ... Args>
struct make_index_sequence_variadic : public make_index_sequence<std::tuple_size<std::tuple<Args...>>::value>
{
};


template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;


//Compile time for loop. Credit to https://stackoverflow.com/questions/37602057/why-isnt-a-for-loop-a-compile-time-expression
template<size_t N>
struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
void for_(F func, index_sequence<Is...>) {
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F> void for_(F func) {
  for_(func, make_index_sequence<N>());
}
template<typename Iter>
struct is_reverse_iterator : std::false_type { };

template<typename Iter>
struct is_reverse_iterator<std::reverse_iterator<Iter>>
: std::integral_constant<bool, !is_reverse_iterator<Iter>::value>
{ };

template<class Base, class Derived, class Enable = void>
struct is_base_of_reverse;

template<class Base, class Derived>
struct is_base_of_reverse<Base, Derived, typename std::enable_if<!is_reverse_iterator<Derived>::value>::type>
        : std::is_base_of<Base,Derived>
{}; // specialization for reverse_iterators

template<class Base, class Derived>
struct is_base_of_reverse<Base, Derived, typename std::enable_if<is_reverse_iterator<Derived>::value>::type>
        : std::is_base_of<Base,decltype(std::declval<Derived&>().base())>
{ }; // specialization for reverse_iterators

//Test if Dummy type is derived from cv::MatConstIterator or is a reverse_iterator thereof.
//Just a convenience class
template<class Dummy>
struct is_base_of_reverse_cvit : is_base_of_reverse<cv::MatConstIterator, Dummy>
{ };



}}
#endif //OPENCV_CORE_STL_ALGORITHM_HPP
