// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_UTIL_HPP
#define OPENCV_GAPI_UTIL_HPP

#include <tuple>

// \cond HIDDEN_SYMBOLS
// This header file contains some generic utility functions which are
// used in other G-API Public API headers.
//
// PLEASE don't put any stuff here if it is NOT used in public API headers!

namespace cv
{
namespace detail
{
    // Recursive integer sequence type, useful for enumerating elements of
    // template parameter packs.
    template<int... I> struct Seq     { using next = Seq<I..., sizeof...(I)>; };
    template<int Sz>   struct MkSeq   { using type = typename MkSeq<Sz-1>::type::next; };
    template<>         struct MkSeq<0>{ using type = Seq<>; };

    // Checks if elements of variadic template satisfy the given Predicate.
    // Implemented via tuple, with an interface to accept plain type lists
    template<template<class> class, typename, typename...> struct all_satisfy;

    template<template<class> class F, typename T, typename... Ts>
    struct all_satisfy<F, std::tuple<T, Ts...> >
    {
        static const constexpr bool value = F<T>::value
            && all_satisfy<F, std::tuple<Ts...> >::value;
    };
    template<template<class> class F, typename T>
    struct all_satisfy<F, std::tuple<T> >
    {
        static const constexpr bool value = F<T>::value;
    };

    template<template<class> class F, typename T, typename... Ts>
    struct all_satisfy: public all_satisfy<F, std::tuple<T, Ts...> > {};

    // Permute given tuple type C with given integer sequence II
    // Sequence may be less than tuple C size.
    template<class, class> struct permute_tuple;

    template<class C, int... IIs>
    struct permute_tuple<C, Seq<IIs...> >
    {
        using type = std::tuple< typename std::tuple_element<IIs, C>::type... >;
    };

    // Given T..., generates a type sequence of sizeof...(T)-1 elements
    // which is T... without its last element
    // Implemented via tuple, with an interface to accept plain type lists
    template<typename T, typename... Ts> struct all_but_last;

    template<typename T, typename... Ts>
    struct all_but_last<std::tuple<T, Ts...> >
    {
        using C    = std::tuple<T, Ts...>;
        using S    = typename MkSeq<std::tuple_size<C>::value - 1>::type;
        using type = typename permute_tuple<C, S>::type;
    };

    template<typename T, typename... Ts>
    struct all_but_last: public all_but_last<std::tuple<T, Ts...> > {};

    template<typename... Ts>
    using all_but_last_t = typename all_but_last<Ts...>::type;

    // NB.: This is here because there's no constexpr std::max in C++11
    template<std::size_t S0, std::size_t... SS> struct max_of_t
    {
        static constexpr const std::size_t rest  = max_of_t<SS...>::value;
        static constexpr const std::size_t value = rest > S0 ? rest : S0;
    };
    template<std::size_t S> struct max_of_t<S>
    {
        static constexpr const std::size_t value = S;
    };

    template <typename...>
    struct contains : std::false_type{};

    template <typename T1, typename T2, typename... Ts>
    struct contains<T1, T2, Ts...> : std::integral_constant<bool, std::is_same<T1, T2>::value ||
                                                                  contains<T1, Ts...>::value> {};
    template<typename T, typename... Types>
    struct contains<T, std::tuple<Types...>> : std::integral_constant<bool, contains<T, Types...>::value> {};

    template <typename...>
    struct all_unique : std::true_type{};

    template <typename T1, typename... Ts>
    struct all_unique<T1, Ts...> : std::integral_constant<bool, !contains<T1, Ts...>::value &&
                                                                 all_unique<Ts...>::value> {};

    template<typename>
    struct tuple_wrap_helper;

    template<typename T> struct tuple_wrap_helper
    {
        using type = std::tuple<T>;
        static type get(T&& obj) { return std::make_tuple(std::move(obj)); }
    };

    template<typename... Objs>
    struct tuple_wrap_helper<std::tuple<Objs...>>
    {
        using type = std::tuple<Objs...>;
        static type get(std::tuple<Objs...>&& objs) { return std::forward<std::tuple<Objs...>>(objs); }
    };
} // namespace detail
} // namespace cv

// \endcond

#endif //  OPENCV_GAPI_UTIL_HPP
