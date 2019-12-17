#ifndef OPENCV_TYPE_TRAITS_HPP
#define OPENCV_TYPE_TRAITS_HPP

#include "opencv2/core.hpp"

#ifdef CV_CXX11
    #include <type_traits>
#endif

namespace stdx {
#ifdef CV_CXX11
using std::enable_if;
using std::is_integral;
using std::is_floating_point;
using std::is_signed;
using std::is_unsigned;
#else
template<class T, T v>
struct integral_constant
{
    typedef T value_type;
    typedef integral_constant type;
    static const value_type value = v;
    operator value_type() const { return value; }
};

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

template<class T, class U>
struct is_same : false_type
{
};

template<class T>
struct is_same<T, T> : true_type
{
};

template<bool Condition, class True, class False>
struct conditional
{
    typedef True type;
};

template<class True, class False>
struct conditional<false, True, False>
{
    typedef False type;
};

template<bool Condition, class TEnable = void>
struct enable_if
{
};

template<class TEnable>
struct enable_if<true, TEnable>
{
    typedef TEnable type;
};

template<class T>
struct remove_const
{
    typedef T type;
};

template<class T>
struct remove_const<const T>
{
    typedef T type;
};

template<class T>
struct remove_volatile
{
    typedef T type;
};

template<class T>
struct remove_volatile<volatile T>
{
    typedef T type;
};

template<class T>
struct remove_cv
{
    typedef typename remove_const<typename remove_volatile<T>::type>::type type;
};

template<class T>
struct is_integral
    : integral_constant<bool, is_same<bool, typename remove_cv<T>::type>::value
                                  || is_same<char, typename remove_cv<T>::type>::value
                                  || is_same<int8_t, typename remove_cv<T>::type>::value
                                  || is_same<uint8_t, typename remove_cv<T>::type>::value
                                  || is_same<int16_t, typename remove_cv<T>::type>::value
                                  || is_same<uint16_t, typename remove_cv<T>::type>::value
                                  || is_same<int32_t, typename remove_cv<T>::type>::value
                                  || is_same<uint32_t, typename remove_cv<T>::type>::value
                                  || is_same<int64_t, typename remove_cv<T>::type>::value
                                  || is_same<uint64_t, typename remove_cv<T>::type>::value
                                  || is_same<size_t, typename remove_cv<T>::type>::value>
{
};

template<class T>
struct is_floating_point
    : integral_constant<bool, is_same<float, typename remove_cv<T>::type>::value
                                  || is_same<double, typename remove_cv<T>::type>::value>
{
};


namespace detail {
template<class T, bool = is_integral<T>::value>
struct is_signed : conditional<T(-1) < T(0), true_type, false_type>::type
{
};

template<class T>
struct is_signed<T, false> : false_type
{
};
} // namespace detail


template<class T>
struct is_signed : conditional<is_floating_point<T>::value, true_type, detail::is_signed<T> >::type
{
};

namespace detail {
template<class T, bool = is_integral<T>::value>
struct is_unsigned : conditional<T(0) < T(-1), true_type, false_type>::type
{
};

template<class T>
struct is_unsigned<T, false> : false_type
{
};
} // namespace detail

template<class T>
struct is_unsigned : detail::is_unsigned<T>::type
{
};

#endif
} // namespace stdx

#endif //OPENCV_TYPE_TRAITS_HPP
