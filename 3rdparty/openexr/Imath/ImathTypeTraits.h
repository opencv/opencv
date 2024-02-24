//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// This file contains type traits related to or used by the Imath library.
//

#ifndef INCLUDED_IMATHTYPETRAITS_H
#define INCLUDED_IMATHTYPETRAITS_H

#include <type_traits>

#include "ImathPlatform.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER


/// Define Imath::enable_if_t to be std for C++14, equivalent for C++11.
#if (IMATH_CPLUSPLUS_VERSION >= 14)
    using std::enable_if_t;    // Use C++14 std::enable_if_t
#else
    // Define enable_if_t for C++11
    template <bool B, class T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;
#endif


/// An enable_if helper to be used in template parameters which results in
/// much shorter symbols.
#define IMATH_ENABLE_IF(...) IMATH_INTERNAL_NAMESPACE::enable_if_t<(__VA_ARGS__), int> = 0


#if IMATH_FOREIGN_VECTOR_INTEROP

/// @{
/// @name Detecting interoperable types.
///
/// In order to construct or assign from external "compatible" types without
/// prior knowledge of their definitions, we have a few helper type traits.
/// The intent of these is to allow custom linear algebra types in an
/// application that have seamless conversion to and from Imath types.
///
/// `has_xy<T,Base>`, `has_xyz<T,Base>`, `has_xyzw<T,Base>` detect if class
/// `T` has elements `.x`, `.y`, and `.z` all of type `Base` and seems to be
/// the right size to hold exactly those members and nothing more.
///
/// `has_subscript<T,Base,N>` detects if class `T` can perform `T[int]`
/// to yield a `Base`, and that it seems to be exactly the right size to
/// hold `N` of those elements.
///
/// This is not exact. It's possible that for a particular user-defined
/// type, this may yield a false negative or false positive. For example:
///   * A class for a 3-vector that contains an extra element of padding
///     so that it will have the right size and alignment to use 4-wide
///     SIMD math ops will appear to be the wrong size.
///   * A `std::vector<T>` is subscriptable and might have N elements at
///     runtime, but the size is dynamic and so would fail this test.
///   * A foreign type may have .x, .y, .z that are not matching our base
///     type but we still want it to work (with appropriate conversions).
///
/// In these cases, user code may declare an exception -- for example,
/// stating that `mytype` should be considered implicitly convertible to
/// an Imath::V3f by subscripting:
///
///     template<>
///     struct Imath::has_subscript<mytype, float, 3> : public std::true_type { };
///
/// And similarly, user code may correct a potential false positive (that
/// is, a `mytype` looks like it should be convertible to a V3f, but you
/// don't want it to ever happen):
///
///     template<typename B, int N>
///     struct Imath::has_subscript<mytype, B, N> : public std::false_type { };
///


/// `has_xy<T,Base>::value` will be true if type `T` has member variables
/// `.x` and `.y`, all of type `Base`, and the size of a `T` is exactly big
/// enough to hold 2 Base values.
template <typename T, typename Base>
struct has_xy {
private:
    typedef char Yes[1];
    typedef char No[2];

    // Valid only if .x, .y exist and are the right type: return a Yes.
    template<typename C,
             IMATH_ENABLE_IF(std::is_same<decltype(C().x), Base>::value),
             IMATH_ENABLE_IF(std::is_same<decltype(C().y), Base>::value)>
    static Yes& test(int);

    // Fallback, default to returning a No.
    template<typename C> static No& test(...);
public:
    enum { value = (sizeof(test<T>(0)) == sizeof(Yes)
                    && sizeof(T) == 2*sizeof(Base))
      };
};


/// `has_xyz<T,Base>::value` will be true if type `T` has member variables
/// `.x`, `.y`, and `.z`, all of type `Base`, and the size of a `T` is
/// exactly big enough to hold 3 Base values.
template <typename T, typename Base>
struct has_xyz {
private:
    typedef char Yes[1];
    typedef char No[2];

    // Valid only if .x, .y, .z exist and are the right type: return a Yes.
    template<typename C,
             IMATH_ENABLE_IF(std::is_same<decltype(C().x), Base>::value),
             IMATH_ENABLE_IF(std::is_same<decltype(C().y), Base>::value),
             IMATH_ENABLE_IF(std::is_same<decltype(C().z), Base>::value)>
    static Yes& test(int);

    // Fallback, default to returning a No.
    template<typename C> static No& test(...);
public:
    enum { value = (sizeof(test<T>(0)) == sizeof(Yes)
                    && sizeof(T) == 3*sizeof(Base))
      };
};


/// `has_xyzw<T,Base>::value` will be true if type `T` has member variables
/// `.x`, `.y`, `.z`, and `.w`, all of type `Base`, and the size of a `T` is
/// exactly big enough to hold 4 Base values.
template <typename T, typename Base>
struct has_xyzw {
private:
    typedef char Yes[1];
    typedef char No[2];

    // Valid only if .x, .y, .z, .w exist and are the right type: return a Yes.
    template<typename C,
             IMATH_ENABLE_IF(std::is_same<decltype(C().x), Base>::value),
             IMATH_ENABLE_IF(std::is_same<decltype(C().y), Base>::value),
             IMATH_ENABLE_IF(std::is_same<decltype(C().z), Base>::value),
             IMATH_ENABLE_IF(std::is_same<decltype(C().w), Base>::value)>
    static Yes& test(int);

    // Fallback, default to returning a No.
    template<typename C> static No& test(...);
public:
    enum { value = (sizeof(test<T>(0)) == sizeof(Yes)
                    && sizeof(T) == 4*sizeof(Base))
      };
};



/// `has_subscript<T,Base,Nelem>::value` will be true if type `T` has
/// subscripting syntax, a `T[int]` returns a `Base`, and the size of a `T`
/// is exactly big enough to hold `Nelem` `Base` values.
template <typename T, typename Base, int Nelem>
struct has_subscript {
private:
    typedef char Yes[1];
    typedef char No[2];

    // Valid only if T[] is possible and is the right type: return a Yes.
    template<typename C,
             IMATH_ENABLE_IF(std::is_same<typename std::decay<decltype(C()[0])>::type, Base>::value)>
    static Yes& test(int);

    // Fallback, default to returning a No.
    template<typename C> static No& test(...);
public:
    enum { value = (sizeof(test<T>(0)) == sizeof(Yes)
                    && sizeof(T) == Nelem*sizeof(Base))
      };
};


/// C arrays of just the right length also are qualified for has_subscript.
template<typename Base, int Nelem>
struct has_subscript<Base[Nelem], Base, Nelem> : public std::true_type { };



/// `has_double_subscript<T,Base,Rows,Cols>::value` will be true if type `T`
/// has 2-level subscripting syntax, a `T[int][int]` returns a `Base`, and
/// the size of a `T` is exactly big enough to hold `R*C` `Base` values.
template <typename T, typename Base, int Rows, int Cols>
struct has_double_subscript {
private:
    typedef char Yes[1];
    typedef char No[2];

    // Valid only if T[][] is possible and is the right type: return a Yes.
    template<typename C,
             IMATH_ENABLE_IF(std::is_same<typename std::decay<decltype(C()[0][0])>::type, Base>::value)>
    static Yes& test(int);

    // Fallback, default to returning a No.
    template<typename C> static No& test(...);
public:
    enum { value = (sizeof(test<T>(0)) == sizeof(Yes)
                    && sizeof(T) == (Rows*Cols)*sizeof(Base))
      };
};


/// C arrays of just the right length also are qualified for has_double_subscript.
template<typename Base, int Rows, int Cols>
struct has_double_subscript<Base[Rows][Cols], Base, Rows, Cols> : public std::true_type { };

/// @}

#endif

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHTYPETRAITS_H
