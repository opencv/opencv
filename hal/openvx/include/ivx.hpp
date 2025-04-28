// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
C++ wrappers over OpenVX 1.x C API
Details: TBD
*/

#pragma once
#ifndef IVX_HPP
#define IVX_HPP

#ifndef __cplusplus
    #error This file has to be compiled with C++ compiler
#endif


#include <VX/vx.h>
#include <VX/vxu.h>

// For OpenVX 1.2 & 1.3
#if (VX_VERSION > VX_VERSION_1_1)
# include <VX/vx_compatibility.h>
#endif


#if (VX_VERSION == VX_VERSION_1_0)
// 1.1 to 1.0 backward compatibility defines

static const vx_enum VX_INTERPOLATION_BILINEAR = VX_INTERPOLATION_TYPE_BILINEAR;
static const vx_enum VX_INTERPOLATION_AREA = VX_INTERPOLATION_TYPE_AREA;
static const vx_enum VX_INTERPOLATION_NEAREST_NEIGHBOR = VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;

static const vx_enum VX_BORDER_CONSTANT = VX_BORDER_MODE_CONSTANT;
static const vx_enum VX_BORDER_REPLICATE = VX_BORDER_MODE_REPLICATE;

#endif

#ifndef IVX_USE_CXX98
    // checking compiler
    #if __cplusplus < 201103L && (!defined(_MSC_VER) || _MSC_VER < 1800)
        #define IVX_USE_CXX98
    #endif
#endif // IVX_USE_CXX98

#if defined(IVX_USE_CXX98) && !defined(IVX_HIDE_INFO_WARNINGS)
    #ifdef _MSC_VER
        #pragma message ("ivx.hpp: The ISO C++ 2011 standard is not enabled, switching to C++98 fallback implementation.")
    #else
        #warning The ISO C++ 2011 standard is not enabled, switching to C++98 fallback implementation.
    #endif
#endif // IVX_USE_CXX98

#ifndef IVX_USE_EXTERNAL_REFCOUNT
    // checking OpenVX version
    #ifndef VX_VERSION_1_1
        #define IVX_USE_EXTERNAL_REFCOUNT
    #endif
#endif // IVX_USE_CXX98

#if defined(IVX_USE_EXTERNAL_REFCOUNT) && !defined(IVX_HIDE_INFO_WARNINGS)
    #ifdef _MSC_VER
        #pragma message ("ivx.hpp: OpenVX version < 1.1, switching to external refcounter implementation.")
    #else
        #warning OpenVX version < 1.1, switching to external refcounter implementation.
    #endif
#endif // IVX_USE_EXTERNAL_REFCOUNT

#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
#include <cstdlib>

#ifndef IVX_USE_CXX98
    #include <type_traits>
    namespace ivx
    {
        using std::is_same;
        using std::is_pointer;
    }
#else
    namespace ivx
    {
    // helpers for compile-time type checking

    template<typename, typename> struct is_same { static const bool value = false; };
    template<typename T> struct is_same<T, T>   { static const bool value = true; };

    template<typename T> struct is_pointer      { static const bool value = false; };
    template<typename T> struct is_pointer<T*>  { static const bool value = true; };
    template<typename T> struct is_pointer<const T*>  { static const bool value = true; };
    }
#endif

#ifdef IVX_USE_OPENCV
    #include "opencv2/core.hpp"
#endif

// disabling false alarm warnings
#if defined(_MSC_VER)
    #pragma warning(push)
    //#pragma warning( disable : 4??? )
#elif defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-local-typedef"
    #pragma clang diagnostic ignored "-Wmissing-prototypes"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-local-typedefs"
    #pragma GCC diagnostic ignored "-Wunused-value"
    #pragma GCC diagnostic ignored "-Wmissing-declarations"
#endif // compiler macro

namespace ivx
{

inline vx_uint16 compiledWithVersion()
{ return VX_VERSION; }

/// Exception class for OpenVX runtime errors
class RuntimeError : public std::runtime_error
{
public:
    /// Constructor
    explicit RuntimeError(vx_status st, const std::string& msg = "")
        : runtime_error(msg), _status(st)
    {}

    /// OpenVX error code
    vx_status status() const
    { return _status; }

private:
    vx_status   _status;
};

/// Exception class for wrappers logic errors
class WrapperError : public std::logic_error
{
public:
    /// Constructor
    explicit WrapperError(const std::string& msg) : logic_error(msg)
    {}
};

inline void checkVxStatus(vx_status status, const std::string& func, const std::string& msg)
{
    if(status != VX_SUCCESS) throw RuntimeError( status, func + "() : " + msg );
}


/// Helper macro for turning a runtime error in the provided code into a \RuntimeError
#define IVX_CHECK_STATUS(code) checkVxStatus(code, __func__, #code)


/// OpenVX enum to type compile-time converter (TODO: add more types)
template<vx_enum E> struct EnumToType {};
template<> struct EnumToType<VX_TYPE_CHAR>     { typedef vx_char type;      static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_INT8>     { typedef vx_int8 type;      static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_UINT8>    { typedef vx_uint8 type;     static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_INT16>    { typedef vx_int16 type;     static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_UINT16>   { typedef vx_uint16 type;    static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_INT32>    { typedef vx_int32 type;     static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_UINT32>   { typedef vx_uint32 type;    static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_INT64>    { typedef vx_int64 type;     static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_UINT64>   { typedef vx_uint64 type;    static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_FLOAT32>  { typedef vx_float32 type;   static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_FLOAT64>  { typedef vx_float64 type;   static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_ENUM>     { typedef vx_enum type;      static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_SIZE>     { typedef vx_size type;      static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_DF_IMAGE> { typedef vx_df_image type;  static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_BOOL>     { typedef vx_bool type;      static const vx_size bytes = sizeof(type); };
template<> struct EnumToType<VX_TYPE_KEYPOINT> { typedef vx_keypoint_t type;static const vx_size bytes = sizeof(type); };
#ifndef IVX_USE_CXX98
template <vx_enum E> using EnumToType_t = typename EnumToType<E>::type;
#endif

/// Gets size in bytes for the provided OpenVX type enum
inline vx_size enumToTypeSize(vx_enum type)
{
    switch (type)
    {
    case VX_TYPE_CHAR:      return EnumToType<VX_TYPE_CHAR>::bytes;
    case VX_TYPE_INT8:      return EnumToType<VX_TYPE_INT8>::bytes;
    case VX_TYPE_UINT8:     return EnumToType<VX_TYPE_UINT8>::bytes;
    case VX_TYPE_INT16:     return EnumToType<VX_TYPE_INT16>::bytes;
    case VX_TYPE_UINT16:    return EnumToType<VX_TYPE_UINT16>::bytes;
    case VX_TYPE_INT32:     return EnumToType<VX_TYPE_INT32>::bytes;
    case VX_TYPE_UINT32:    return EnumToType<VX_TYPE_UINT32>::bytes;
    case VX_TYPE_INT64:     return EnumToType<VX_TYPE_INT64>::bytes;
    case VX_TYPE_UINT64:    return EnumToType<VX_TYPE_UINT64>::bytes;
    case VX_TYPE_FLOAT32:   return EnumToType<VX_TYPE_FLOAT32>::bytes;
    case VX_TYPE_FLOAT64:   return EnumToType<VX_TYPE_FLOAT64>::bytes;
    case VX_TYPE_ENUM:      return EnumToType<VX_TYPE_ENUM>::bytes;
    case VX_TYPE_SIZE:      return EnumToType<VX_TYPE_SIZE>::bytes;
    case VX_TYPE_DF_IMAGE:  return EnumToType<VX_TYPE_DF_IMAGE>::bytes;
    case VX_TYPE_BOOL:      return EnumToType<VX_TYPE_BOOL>::bytes;
    case VX_TYPE_KEYPOINT:  return EnumToType<VX_TYPE_KEYPOINT>::bytes;
    default: throw WrapperError(std::string(__func__) + ": unsupported type enum");
    }
}

/// type to enum compile-time converter (TODO: add more types)
template<typename T> struct TypeToEnum {};
template<> struct TypeToEnum<vx_char>     { static const vx_enum value = VX_TYPE_CHAR; };
template<> struct TypeToEnum<vx_int8>     { static const vx_enum value = VX_TYPE_INT8; };
template<> struct TypeToEnum<vx_uint8>    { static const vx_enum value = VX_TYPE_UINT8, imgType = VX_DF_IMAGE_U8; };
template<> struct TypeToEnum<vx_int16>    { static const vx_enum value = VX_TYPE_INT16, imgType = VX_DF_IMAGE_S16; };
template<> struct TypeToEnum<vx_uint16>   { static const vx_enum value = VX_TYPE_UINT16, imgType = VX_DF_IMAGE_U16; };
template<> struct TypeToEnum<vx_int32>    { static const vx_enum value = VX_TYPE_INT32, imgType = VX_DF_IMAGE_S32; };
template<> struct TypeToEnum<vx_uint32>   { static const vx_enum value = VX_TYPE_UINT32, imgType = VX_DF_IMAGE_U32; };
template<> struct TypeToEnum<vx_int64>    { static const vx_enum value = VX_TYPE_INT64; };
template<> struct TypeToEnum<vx_uint64>   { static const vx_enum value = VX_TYPE_UINT64; };
template<> struct TypeToEnum<vx_float32>  { static const vx_enum value = VX_TYPE_FLOAT32, imgType = VX_DF_IMAGE('F', '0', '3', '2'); };
template<> struct TypeToEnum<vx_float64>  { static const vx_enum value = VX_TYPE_FLOAT64; };
//template<> struct TypeToEnum<vx_bool>     { static const vx_enum value = VX_TYPE_BOOL; };
template<> struct TypeToEnum<vx_keypoint_t> {static const vx_enum value = VX_TYPE_KEYPOINT; };
// the commented types are aliases (of integral tyes) and have conflicts with the types above
//template<> struct TypeToEnum<vx_enum>     { static const vx_enum val = VX_TYPE_ENUM; };
//template<> struct TypeToEnum<vx_size>     { static const vx_enum val = VX_TYPE_SIZE; };
//template<> struct TypeToEnum<vx_df_image> { static const vx_enum val = VX_TYPE_DF_IMAGE; };

inline bool areTypesCompatible(const vx_enum a, const vx_enum b)
{
    return enumToTypeSize(a) == enumToTypeSize(b);
}

#ifdef IVX_USE_OPENCV
inline int enumToCVType(vx_enum type)
{
    switch (type)
    {
    case VX_TYPE_CHAR: return CV_8UC1;//While OpenCV support 8S as well, 8U is supported wider
    case VX_TYPE_INT8: return CV_8SC1;
    case VX_TYPE_UINT8: return CV_8UC1;
    case VX_TYPE_INT16: return CV_16SC1;
    case VX_TYPE_UINT16: return CV_16UC1;
    case VX_TYPE_INT32: return CV_32SC1;
    case VX_TYPE_UINT32: return CV_32SC1;//That's not the best option but there is CV_32S type only
    case VX_TYPE_FLOAT32: return CV_32FC1;
    case VX_TYPE_FLOAT64: return CV_64FC1;
    case VX_TYPE_ENUM: return CV_32SC1;
    case VX_TYPE_BOOL: return CV_32SC1;
    default: throw WrapperError(std::string(__func__) + ": unsupported type enum");
    }
}
#endif

/// Helper type, provides info for OpenVX 'objects' (vx_reference extending) types
template <typename T> struct RefTypeTraits {};

class Context;
template <> struct RefTypeTraits <vx_context>
{
    typedef vx_context vxType;
    typedef Context wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_CONTEXT;
    static vx_status release(vxType& ref) { return vxReleaseContext(&ref); }
};

class Graph;
template <> struct RefTypeTraits <vx_graph>
{
    typedef vx_graph vxType;
    typedef Graph wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_GRAPH;
    static vx_status release(vxType& ref) { return vxReleaseGraph(&ref); }
};

class Node;
template <> struct RefTypeTraits <vx_node>
{
    typedef vx_node vxType;
    typedef Node wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_NODE;
    static vx_status release(vxType& ref) { return vxReleaseNode(&ref); }
};

class Kernel;
template <> struct RefTypeTraits <vx_kernel>
{
    typedef vx_kernel vxType;
    typedef Kernel wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_KERNEL;
    static vx_status release(vxType& ref) { return vxReleaseKernel(&ref); }
};

class Param;
template <> struct RefTypeTraits <vx_parameter>
{
    typedef vx_parameter vxType;
    typedef Param wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_PARAMETER;
    static vx_status release(vxType& ref) { return vxReleaseParameter(&ref); }
};

class Image;
template <> struct RefTypeTraits <vx_image>
{
    typedef vx_image vxType;
    typedef Image wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_IMAGE;
    static vx_status release(vxType& ref) { return vxReleaseImage(&ref); }
};

class Scalar;
template <> struct RefTypeTraits <vx_scalar>
{
    typedef vx_scalar vxType;
    typedef Scalar wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_SCALAR;
    static vx_status release(vxType& ref) { return vxReleaseScalar(&ref); }
};

class Array;
template <> struct RefTypeTraits <vx_array>
{
    typedef vx_array vxType;
    typedef Array wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_ARRAY;
    static vx_status release(vxType& ref) { return vxReleaseArray(&ref); }
};

class Threshold;
template <> struct RefTypeTraits <vx_threshold>
{
    typedef vx_threshold vxType;
    typedef Threshold wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_THRESHOLD;
    static vx_status release(vxType& ref) { return vxReleaseThreshold(&ref); }
};

class Convolution;
template <> struct RefTypeTraits <vx_convolution>
{
    typedef vx_convolution vxType;
    typedef Convolution wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_CONVOLUTION;
    static vx_status release(vxType& ref) { return vxReleaseConvolution(&ref); }
};

class Matrix;
template <> struct RefTypeTraits <vx_matrix>
{
    typedef vx_matrix vxType;
    typedef Matrix wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_MATRIX;
    static vx_status release(vxType& ref) { return vxReleaseMatrix(&ref); }
};

class LUT;
template <> struct RefTypeTraits <vx_lut>
{
    typedef vx_lut vxType;
    typedef LUT wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_LUT;
    static vx_status release(vxType& ref) { return vxReleaseLUT(&ref); }
};

class Pyramid;
template <> struct RefTypeTraits <vx_pyramid>
{
    typedef vx_pyramid vxType;
    typedef Pyramid wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_PYRAMID;
    static vx_status release(vxType& ref) { return vxReleasePyramid(&ref); }
};

class Distribution;
template <> struct RefTypeTraits <vx_distribution>
{
    typedef vx_distribution vxType;
    typedef Distribution wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_DISTRIBUTION;
    static vx_status release(vxType& ref) { return vxReleaseDistribution(&ref); }
};

class Remap;
template <> struct RefTypeTraits <vx_remap>
{
    typedef vx_remap vxType;
    typedef Remap wrapperType;
    static const vx_enum vxTypeEnum = VX_TYPE_REMAP;
    static vx_status release(vxType& ref) { return vxReleaseRemap(&ref); }
};

#ifdef IVX_USE_CXX98

/// Casting to vx_reference with compile-time check

// takes 'vx_reference' itself and RefWrapper<T> via 'operator vx_reference()'
inline vx_reference castToReference(vx_reference ref)
{ return ref; }

// takes vx_reference extensions that have RefTypeTraits<T> specializations
template<typename T>
inline vx_reference castToReference(const T& ref, typename RefTypeTraits<T>::vxType dummy = 0)
{ (void)dummy; return (vx_reference)ref; }

#else

template<typename T, typename = void>
struct is_ref : std::is_same<T, vx_reference>{}; // allow vx_reference

// allow RefWrapper<> types
template<typename T>
#ifndef _MSC_VER
struct is_ref<T, decltype(T().operator vx_reference(), void())> : std::true_type {};
#else
// workarounding VC14 compiler crash
struct is_ref<T, decltype(T::vxType(), void())> : std::true_type {};
#endif

// allow vx_reference extensions
template<typename T>
struct is_ref<T, decltype(RefTypeTraits<T>::vxTypeEnum, void())> : std::true_type {};

/// Casting to vx_reference with compile-time check

template<typename T>
inline vx_reference castToReference(const T& obj)
{
    static_assert(is_ref<T>::value, "unsupported conversion");
    return (vx_reference) obj;
}

#endif // IVX_USE_CXX98

inline void checkVxRef(vx_reference ref, const std::string& func, const std::string& msg)
{
    vx_status status = vxGetStatus(ref);
    if(status != VX_SUCCESS) throw RuntimeError( status, func + "() : " + msg );
}

/// Helper macro for checking the provided OpenVX 'object' and throwing a \RuntimeError in case of error
#define IVX_CHECK_REF(code) checkVxRef(castToReference(code), __func__, #code)


#ifdef IVX_USE_EXTERNAL_REFCOUNT

/// Base class for OpenVX 'objects' wrappers
template <typename T> class RefWrapper
{
public:
    typedef T vxType;
    static const vx_enum vxTypeEnum = RefTypeTraits <T>::vxTypeEnum;

    /// Default constructor
    RefWrapper() : ref(0), refcount(0)
    {}

    /// Constructor
    /// \param r OpenVX 'object' (e.g. vx_image)
    /// \param retainRef flag indicating whether to increase ref counter in constructor (false by default)
    explicit RefWrapper(T r, bool retainRef = false) : ref(0), refcount(0)
    { reset(r, retainRef); }

    /// Copy constructor
    RefWrapper(const RefWrapper& r) : ref(r.ref), refcount(r.refcount)
    { addRef(); }

#ifndef IVX_USE_CXX98
    /// Move constructor
    RefWrapper(RefWrapper&& rw) noexcept : RefWrapper()
    {
        using std::swap;
        swap(ref, rw.ref);
        swap(refcount, rw.refcount);
    }
#endif

    /// Casting to the wrapped OpenVX 'object'
    operator T() const
    { return ref; }

    /// Casting to vx_reference since every OpenVX 'object' extends it
    operator vx_reference() const
    { return castToReference(ref); }

    /// Assigning a new value (decreasing ref counter for the old one)
    /// \param r OpenVX 'object' (e.g. vx_image)
    /// \param retainRef flag indicating whether to increase ref counter in constructor (false by default)
    void reset(T r, bool retainRef = false)
    {
        release();
        ref = r;
#ifdef VX_VERSION_1_1
        if(retainRef) addRef();
#else
        // if 'retainRef' -just don't use ref-counting for v 1.0
        if(!retainRef) refcount = new int(1);
#endif
        checkRef();
    }

    /// Assigning an empty value (decreasing ref counter for the old one)
    void reset()
    { release(); }

    /// Dropping kept value without releas decreasing ref counter
    /// \return the value being dropped
    T detach()
    {
        T tmp = ref;
        ref = 0;
        release();
        return tmp;
    }

    /// Unified assignment operator (covers both copy and move cases)
    RefWrapper& operator=(RefWrapper r)
    {
        using std::swap;
        swap(ref, r.ref);
        swap(refcount, r.refcount);
        return *this;
    }

    /// Checking for non-empty
    bool operator !() const
    { return ref == 0; }

#ifndef IVX_USE_CXX98
    /// Explicit boolean evaluation (called automatically inside conditional operators only)
    explicit operator bool() const
    { return ref != 0; }
#endif

    /// Getting a context that is kept in each OpenVX 'object' (call get<Context>())
    template<typename C>
    C get() const
    {
        typedef int static_assert_context[is_same<C, Context>::value ? 1 : -1];
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return C(c, true);
    }

#ifndef IVX_USE_CXX98
    /// Getting a context that is kept in each OpenVX 'object'
    template<typename C = Context, typename = typename std::enable_if<std::is_same<C, Context>::value>::type>
    C getContext() const
    {
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return C(c, true);
    }
#endif // IVX_USE_CXX98

protected:
    T ref;
    int* refcount;

    void addRef()
    {
#ifdef VX_VERSION_1_1
        if(ref) IVX_CHECK_STATUS(vxRetainReference(castToReference(ref)));
#else //TODO: make thread-safe
        if(refcount) ++(*refcount);
#endif
    }

    void release()
    {
#ifdef VX_VERSION_1_1
        if(ref) RefTypeTraits<T>::release(ref);
#else //TODO: make thread-safe
        if(refcount && --(*refcount) == 0)
        {
            if(ref) RefTypeTraits<T>::release(ref);
            ref = 0;
            delete refcount;
            refcount = 0;
        }
#endif
    }

    void checkRef() const
    {
        IVX_CHECK_REF(ref);
        vx_enum type;
        IVX_CHECK_STATUS(vxQueryReference((vx_reference)ref, VX_REF_ATTRIBUTE_TYPE, &type, sizeof(type)));
        if (type != vxTypeEnum) throw WrapperError("incompatible reference type");
    }

    ~RefWrapper()
    { release(); }
};

#ifdef IVX_USE_CXX98

    #define IVX_REF_STD_CTORS_AND_ASSIGNMENT(Class) \
        Class() : RefWrapper() {} \
        explicit Class(Class::vxType _ref, bool retainRef = false) : RefWrapper(_ref, retainRef) {} \
        Class(const Class& _obj) : RefWrapper(_obj) {} \
        \
        Class& operator=(Class _obj) { using std::swap; swap(ref, _obj.ref); swap(refcount, _obj.refcount); return *this; }

#else

    #define IVX_REF_STD_CTORS_AND_ASSIGNMENT(Class) \
        Class() : RefWrapper() {} \
        explicit Class(Class::vxType _ref, bool retainRef = false) : RefWrapper(_ref, retainRef) {} \
        Class(const Class& _obj) : RefWrapper(_obj) {} \
        Class(Class&& _obj) : RefWrapper(std::move(_obj)) {} \
        \
        Class& operator=(Class _obj) { using std::swap; swap(ref, _obj.ref); swap(refcount, _obj.refcount); return *this; }

#endif // IVX_USE_CXX98

#else // not IVX_USE_EXTERNAL_REFCOUNT

/// Base class for OpenVX 'objects' wrappers
template <typename T> class RefWrapper
{
public:
    typedef T vxType;
    static const vx_enum vxTypeEnum = RefTypeTraits <T>::vxTypeEnum;

    /// Default constructor
    RefWrapper() : ref(0)
    {}

    /// Constructor
    /// \param r OpenVX 'object' (e.g. vx_image)
    /// \param retainRef flag indicating whether to increase ref counter in constructor (false by default)
    explicit RefWrapper(T r, bool retainRef = false) : ref(0)
    { reset(r, retainRef); }

    /// Copy constructor
    RefWrapper(const RefWrapper& r) : ref(r.ref)
    { addRef(); }

#ifndef IVX_USE_CXX98
    /// Move constructor
    RefWrapper(RefWrapper&& rw) noexcept : RefWrapper()
    {
        using std::swap;
        swap(ref, rw.ref);
    }
#endif

    /// Casting to the wrapped OpenVX 'object'
    operator T() const
    { return ref; }

    /// Casting to vx_reference since every OpenVX 'object' extends it
    operator vx_reference() const
    { return castToReference(ref); }

    /// Getting a context that is kept in each OpenVX 'object' (call get<Context>())
    template<typename C>
    C get() const
    {
        typedef int static_assert_context[is_same<C, Context>::value ? 1 : -1];
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return C(c, true);
    }

#ifndef IVX_USE_CXX98
    /// Getting a context that is kept in each OpenVX 'object'
    template<typename C = Context, typename = typename std::enable_if<std::is_same<C, Context>::value>::type>
    C getContext() const
    {
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return C(c, true);
    }
#endif // IVX_USE_CXX98

    /// Assigning a new value (decreasing ref counter for the old one)
    /// \param r OpenVX 'object' (e.g. vx_image)
    /// \param retainRef flag indicating whether to increase ref counter in constructor (false by default)
    void reset(T r, bool retainRef = false)
    {
        release();
        ref = r;
        if (retainRef) addRef();
        checkRef();
    }

    /// Assigning an empty value (decreasing ref counter for the old one)
    void reset()
    { release(); }

    /// Dropping kept value without releas decreasing ref counter
    /// \return the value being dropped
    T detach()
    {
        T tmp = ref;
        ref = 0;
        return tmp;
    }

    /// Unified assignment operator (covers both copy and move cases)
    RefWrapper& operator=(RefWrapper r)
    {
        using std::swap;
        swap(ref, r.ref);
        return *this;
    }

    /// Checking for non-empty
    bool operator !() const
    { return ref == 0; }

#ifndef IVX_USE_CXX98
    /// Explicit boolean evaluation (called automatically inside conditional operators only)
    explicit operator bool() const
    { return ref != 0; }
#endif

protected:
    T ref;

    void addRef()
    { if (ref) IVX_CHECK_STATUS(vxRetainReference((vx_reference)ref)); }

    void release()
    {
        if (ref) RefTypeTraits<T>::release(ref);
        ref = 0;
    }

    void checkRef() const
    {
        IVX_CHECK_REF(ref);
        vx_enum type;
        IVX_CHECK_STATUS(vxQueryReference((vx_reference)ref, VX_REF_ATTRIBUTE_TYPE, &type, sizeof(type)));
        if (type != vxTypeEnum) throw WrapperError("incompatible reference type");
    }

    ~RefWrapper()
    { release(); }
};

#ifdef IVX_USE_CXX98

    #define IVX_REF_STD_CTORS_AND_ASSIGNMENT(Class) \
        Class() : RefWrapper() {} \
        explicit Class(Class::vxType _ref, bool retainRef = false) : RefWrapper(_ref, retainRef) {} \
        Class(const Class& _obj) : RefWrapper(_obj) {} \
        \
        Class& operator=(Class _obj) { using std::swap; swap(ref, _obj.ref); return *this; }

#else

    #define IVX_REF_STD_CTORS_AND_ASSIGNMENT(Class) \
        Class() : RefWrapper() {} \
        explicit Class(Class::vxType _ref, bool retainRef = false) : RefWrapper(_ref, retainRef) {} \
        Class(const Class& _obj) : RefWrapper(_obj) {} \
        Class(Class&& _obj) : RefWrapper(std::move(_obj)) {} \
        \
        Class& operator=(Class _obj) { using std::swap; swap(ref, _obj.ref); return *this; }

#endif // IVX_USE_CXX98

#endif // IVX_USE_EXTERNAL_REFCOUNT

#ifndef VX_VERSION_1_1
typedef vx_border_mode_t border_t;
#else
typedef vx_border_t border_t;
#endif

/// vx_context wrapper
class Context : public RefWrapper<vx_context>
{
public:

    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Context)

    /// vxCreateContext() wrapper
    static Context create()
    { return Context(vxCreateContext()); }

    /// vxGetContext() wrapper
    template <typename T>
    static Context getFrom(const T& ref)
    {
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return Context(c, true);
    }

    /// vxLoadKernels() wrapper
    void loadKernels(const std::string& module)
    { IVX_CHECK_STATUS( vxLoadKernels(ref, module.c_str()) ); }

    /// vxQueryContext() wrapper
    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS(vxQueryContext(ref, att, &value, sizeof(value))); }

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_CONTEXT_VENDOR_ID = VX_CONTEXT_ATTRIBUTE_VENDOR_ID,
        VX_CONTEXT_VERSION = VX_CONTEXT_ATTRIBUTE_VERSION,
        VX_CONTEXT_UNIQUE_KERNELS = VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNELS,
        VX_CONTEXT_MODULES = VX_CONTEXT_ATTRIBUTE_MODULES,
        VX_CONTEXT_REFERENCES = VX_CONTEXT_ATTRIBUTE_REFERENCES,
        VX_CONTEXT_IMPLEMENTATION = VX_CONTEXT_ATTRIBUTE_IMPLEMENTATION,
        VX_CONTEXT_EXTENSIONS_SIZE = VX_CONTEXT_ATTRIBUTE_EXTENSIONS_SIZE,
        VX_CONTEXT_EXTENSIONS = VX_CONTEXT_ATTRIBUTE_EXTENSIONS,
        VX_CONTEXT_CONVOLUTION_MAX_DIMENSION = VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION,
        VX_CONTEXT_OPTICAL_FLOW_MAX_WINDOW_DIMENSION = VX_CONTEXT_ATTRIBUTE_OPTICAL_FLOW_WINDOW_MAXIMUM_DIMENSION,
        VX_CONTEXT_IMMEDIATE_BORDER = VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE,
        VX_CONTEXT_UNIQUE_KERNEL_TABLE = VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNEL_TABLE;
#endif

    /// vxQueryContext(VX_CONTEXT_VENDOR_ID) wrapper
    vx_uint16 vendorID() const
    {
        vx_uint16 v;
        query(VX_CONTEXT_VENDOR_ID, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_VERSION) wrapper
    vx_uint16 version() const
    {
        vx_uint16 v;
        query(VX_CONTEXT_VERSION, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_UNIQUE_KERNELS) wrapper
    vx_uint32 uniqueKernelsNum() const
    {
        vx_uint32 v;
        query(VX_CONTEXT_UNIQUE_KERNELS, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_MODULES) wrapper
    vx_uint32 modulesNum() const
    {
        vx_uint32 v;
        query(VX_CONTEXT_MODULES, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_REFERENCES) wrapper
    vx_uint32 refsNum() const
    {
        vx_uint32 v;
        query(VX_CONTEXT_REFERENCES, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_EXTENSIONS_SIZE) wrapper
    vx_size extensionsSize() const
    {
        vx_size v;
        query(VX_CONTEXT_EXTENSIONS_SIZE, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_CONVOLUTION_MAX_DIMENSION) wrapper
    vx_size convolutionMaxDimension() const
    {
        vx_size v;
        query(VX_CONTEXT_CONVOLUTION_MAX_DIMENSION, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_OPTICAL_FLOW_MAX_WINDOW_DIMENSION) wrapper
    vx_size opticalFlowMaxWindowSize() const
    {
        vx_size v;
        query(VX_CONTEXT_OPTICAL_FLOW_MAX_WINDOW_DIMENSION, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_IMMEDIATE_BORDER) wrapper
    border_t immediateBorder() const
    {
        border_t v;
        query(VX_CONTEXT_IMMEDIATE_BORDER, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_IMPLEMENTATION) wrapper
    std::string implName() const
    {
        std::vector<vx_char> v(VX_MAX_IMPLEMENTATION_NAME);
        IVX_CHECK_STATUS(vxQueryContext(ref, VX_CONTEXT_IMPLEMENTATION, &v[0], v.size() * sizeof(vx_char)));
        return std::string(v.data());
    }

    /// vxQueryContext(VX_CONTEXT_EXTENSIONS) wrapper
    std::string extensionsStr() const
    {
        std::vector<vx_char> v(extensionsSize());
        IVX_CHECK_STATUS(vxQueryContext(ref, VX_CONTEXT_EXTENSIONS, &v[0], v.size() * sizeof(vx_char)));
        return std::string(v.data());
    }

    /// vxQueryContext(VX_CONTEXT_UNIQUE_KERNEL_TABLE) wrapper
    std::vector<vx_kernel_info_t> kernelTable() const
    {
        std::vector<vx_kernel_info_t> v(uniqueKernelsNum());
        IVX_CHECK_STATUS(vxQueryContext(ref, VX_CONTEXT_UNIQUE_KERNEL_TABLE, &v[0], v.size() * sizeof(vx_kernel_info_t)));
        return v;
    }

#ifdef VX_VERSION_1_1
    /// vxQueryContext(VX_CONTEXT_IMMEDIATE_BORDER_POLICY) wrapper
    vx_enum immediateBorderPolicy() const
    {
        vx_enum v;
        query(VX_CONTEXT_IMMEDIATE_BORDER_POLICY, v);
        return v;
    }

    /// vxQueryContext(VX_CONTEXT_NONLINEAR_MAX_DIMENSION) wrapper
    vx_size nonlinearMaxDimension() const
    {
        vx_size v;
        query(VX_CONTEXT_NONLINEAR_MAX_DIMENSION, v);
        return v;
    }
#endif

    /// vxSetContextAttribute() wrapper
    template<typename T>
    void setAttribute(vx_enum att, const T& value)
    { IVX_CHECK_STATUS( vxSetContextAttribute(ref, att, &value, sizeof(value)) ); }

    /// vxSetContextAttribute(BORDER) wrapper
    void setImmediateBorder(const border_t& bm)
    { setAttribute(VX_CONTEXT_IMMEDIATE_BORDER, bm); }

#ifndef VX_VERSION_1_1
    /// vxSetContextAttribute(BORDER) wrapper
    void setImmediateBorder(vx_enum mode, vx_uint32 val = 0)
    { border_t bm = {mode, val}; setImmediateBorder(bm); }
#else
    /// vxSetContextAttribute(BORDER) wrapper
    void setImmediateBorder(vx_enum mode, const vx_pixel_value_t& val)
    { border_t bm = {mode, val}; setImmediateBorder(bm); }

    /// vxSetContextAttribute(BORDER) wrapper
    template <typename T>
    void setImmediateBorder(vx_enum mode, const T& _val)
    {
        vx_pixel_value_t val;
        switch (TypeToEnum<T>::value)
        {
        case VX_TYPE_UINT8:
            val.U8 = _val;
            break;
        case VX_TYPE_INT16:
            val.S16 = _val;
            break;
        case VX_TYPE_UINT16:
            val.U16 = _val;
            break;
        case VX_TYPE_INT32:
            val.S32 = _val;
            break;
        case VX_TYPE_UINT32:
            val.U32 = _val;
            break;
        default:
            throw WrapperError("Unsupported constant border value type");
        }
        setImmediateBorder(mode, val);
    }

    /// vxSetContextAttribute(BORDER) wrapper
    void setImmediateBorder(vx_enum mode)
    { vx_pixel_value_t val = {}; setImmediateBorder(mode, val); }
#endif
};

/// vx_graph wrapper
class Graph : public RefWrapper<vx_graph>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Graph);

    /// vxCreateGraph() wrapper
    static Graph create(vx_context c)
    { return Graph(vxCreateGraph(c)); }

    /// vxVerifyGraph() wrapper
    void verify()
    { IVX_CHECK_STATUS( vxVerifyGraph(ref) ); }

    /// vxProcessGraph() wrapper
    void process()
    { IVX_CHECK_STATUS( vxProcessGraph(ref) ); }

    /// vxScheduleGraph() wrapper
    void schedule()
    { IVX_CHECK_STATUS(vxScheduleGraph(ref) ); }

    /// vxWaitGraph() wrapper
    void wait()
    { IVX_CHECK_STATUS(vxWaitGraph(ref)); }
};

/// vx_kernel wrapper
class Kernel : public RefWrapper<vx_kernel>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Kernel);

    /// vxGetKernelByEnum() wrapper
    static Kernel getByEnum(vx_context c, vx_enum kernelID)
    { return Kernel(vxGetKernelByEnum(c, kernelID)); }

    /// vxGetKernelByName() wrapper
    static Kernel getByName(vx_context c, const std::string& name)
    { return Kernel(vxGetKernelByName(c, name.c_str())); }
};


/// vx_node wrapper
class Node : public RefWrapper<vx_node>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Node);

    /// vxCreateGenericNode() wrapper
    static Node create(vx_graph g, vx_kernel k)
    { return Node(vxCreateGenericNode(g, k)); }

    /// Create node for the kernel and set the parameters
    static Node create(vx_graph graph, vx_kernel kernel, const std::vector<vx_reference>& params)
    {
        Node node = Node::create(graph, kernel);
        vx_uint32 i = 0;
        for (std::vector<vx_reference>::const_iterator p = params.begin(); p != params.end(); ++p)
            node.setParameterByIndex(i++, *p);
        return node;
    }

    /// Create node for the kernel ID and set the parameters
    static Node create(vx_graph graph,  vx_enum kernelID, const std::vector<vx_reference>& params)
    { return Node::create(graph, Kernel::getByEnum(Context::getFrom(graph), kernelID), params); }

#ifdef IVX_USE_CXX98
    /// Create node for the kernel ID and set one parameter
    template<typename T0>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set two parameters
    template<typename T0, typename T1>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set three parameters
    template<typename T0, typename T1, typename T2>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set four parameters
    template<typename T0, typename T1, typename T2, typename T3>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2,
                       const T3& arg3)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        params.push_back(castToReference(arg3));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set five parameters
    template<typename T0, typename T1, typename T2, typename T3, typename T4>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2,
                       const T3& arg3, const T4& arg4)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        params.push_back(castToReference(arg3));
        params.push_back(castToReference(arg4));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set six parameters
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2,
                       const T3& arg3, const T4& arg4, const T5& arg5)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        params.push_back(castToReference(arg3));
        params.push_back(castToReference(arg4));
        params.push_back(castToReference(arg5));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set seven parameters
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2,
                       const T3& arg3, const T4& arg4, const T5& arg5,
                       const T6& arg6)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        params.push_back(castToReference(arg3));
        params.push_back(castToReference(arg4));
        params.push_back(castToReference(arg5));
        params.push_back(castToReference(arg6));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set eight parameters
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2,
                       const T3& arg3, const T4& arg4, const T5& arg5,
                       const T6& arg6, const T7& arg7)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        params.push_back(castToReference(arg3));
        params.push_back(castToReference(arg4));
        params.push_back(castToReference(arg5));
        params.push_back(castToReference(arg6));
        params.push_back(castToReference(arg7));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set nine parameters
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename T8>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2,
                       const T3& arg3, const T4& arg4, const T5& arg5,
                       const T6& arg6, const T7& arg7, const T8& arg8)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        params.push_back(castToReference(arg3));
        params.push_back(castToReference(arg4));
        params.push_back(castToReference(arg5));
        params.push_back(castToReference(arg6));
        params.push_back(castToReference(arg7));
        params.push_back(castToReference(arg8));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    /// Create node for the kernel ID and set ten parameters
    template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
             typename T6, typename T7, typename T8, typename T9>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1, const T2& arg2,
                       const T3& arg3, const T4& arg4, const T5& arg5,
                       const T6& arg6, const T7& arg7, const T8& arg8,
                       const T9& arg9)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        params.push_back(castToReference(arg2));
        params.push_back(castToReference(arg3));
        params.push_back(castToReference(arg4));
        params.push_back(castToReference(arg5));
        params.push_back(castToReference(arg6));
        params.push_back(castToReference(arg7));
        params.push_back(castToReference(arg8));
        params.push_back(castToReference(arg9));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

#else // not IVX_USE_CXX98

    /// Create node for the kernel ID and set the specified parameters
    template<typename...Ts>
    static Node create(vx_graph g, vx_enum kernelID, const Ts&...args)
    { return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), { castToReference(args)... }); }

#endif // IVX_USE_CXX98

    /// vxSetParameterByIndex() wrapper
    void setParameterByIndex(vx_uint32 index, vx_reference value)
    { IVX_CHECK_STATUS(vxSetParameterByIndex(ref, index, value)); }

    /// vxQueryNode() wrapper
    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS( vxQueryNode(ref, att, &value, sizeof(value)) ); }

#ifndef VX_VERSION_1_1
static const vx_enum
    VX_NODE_STATUS          = VX_NODE_ATTRIBUTE_STATUS,
    VX_NODE_PERFORMANCE     = VX_NODE_ATTRIBUTE_PERFORMANCE,
    VX_NODE_BORDER          = VX_NODE_ATTRIBUTE_BORDER_MODE,
    VX_NODE_LOCAL_DATA_SIZE = VX_NODE_ATTRIBUTE_LOCAL_DATA_SIZE,
    VX_NODE_LOCAL_DATA_PTR  = VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR,
    VX_BORDER_UNDEFINED     = VX_BORDER_MODE_UNDEFINED;
#endif

    /// vxQueryNode(STATUS) wrapper
    vx_status status() const
    {
        vx_status v;
        query(VX_NODE_STATUS, v);
        return v;
    }

    /// vxQueryNode(PERFORMANCE) wrapper
    vx_perf_t performance() const
    {
        vx_perf_t v;
        query(VX_NODE_PERFORMANCE, v);
        return v;
    }

    /// vxQueryNode(BORDER) wrapper
    border_t border() const
    {
        border_t v;
        v.mode = VX_BORDER_UNDEFINED;
        query(VX_NODE_BORDER, v);
        return v;
    }

    /// vxQueryNode(LOCAL_DATA_SIZE) wrapper
    vx_size dataSize() const
    {
        vx_size v;
        query(VX_NODE_LOCAL_DATA_SIZE, v);
        return v;
    }

    /// vxQueryNode(LOCAL_DATA_PTR) wrapper
    void* dataPtr() const
    {
        void* v;
        query(VX_NODE_LOCAL_DATA_PTR, v);
        return v;
    }

#ifdef VX_VERSION_1_1
    /// vxQueryNode(PARAMETERS) wrapper
    vx_uint32 paramsNum() const
    {
        vx_uint32 v;
        query(VX_NODE_PARAMETERS, v);
        return v;
    }

    /// vxQueryNode(REPLICATED) wrapper
    vx_bool isReplicated() const
    {
        vx_bool v;
        query(VX_NODE_IS_REPLICATED, v);
        return v;
    }

    /// vxQueryNode(REPLICATE_FLAGS) wrapper
    void replicateFlags(std::vector<vx_bool>& flags) const
    {
        if(flags.empty()) flags.resize(paramsNum(), vx_false_e);
        IVX_CHECK_STATUS( vxQueryNode(ref, VX_NODE_REPLICATE_FLAGS, &flags[0], flags.size()*sizeof(flags[0])) );
    }

    /// vxQueryNode(VX_NODE_VALID_RECT_RESET) wrapper
    vx_bool resetValidRect() const
    {
        vx_bool v;
        query(VX_NODE_VALID_RECT_RESET, v);
        return v;
    }
#endif // VX_VERSION_1_1

    /// vxSetNodeAttribute() wrapper
    template<typename T>
    void setAttribute(vx_enum att, const T& value)
    { IVX_CHECK_STATUS( vxSetNodeAttribute(ref, att, &value, sizeof(value)) ); }

    /// vxSetNodeAttribute(BORDER) wrapper
    void setBorder(const border_t& bm)
    { setAttribute(VX_NODE_BORDER, bm); }

#ifndef VX_VERSION_1_1
    /// vxSetNodeAttribute(BORDER) wrapper
    void setBorder(vx_enum mode, vx_uint32 val = 0)
    { vx_border_mode_t bm = {mode, val}; setBorder(bm); }
#else
    /// vxSetNodeAttribute(BORDER) wrapper
    void setBorder(vx_enum mode, const vx_pixel_value_t& val)
    { vx_border_t bm = {mode, val}; setBorder(bm); }

    /// vxSetNodeAttribute(BORDER) wrapper
    template <typename T>
    void setBorder(vx_enum mode, const T& _val)
    {
        vx_pixel_value_t val;
        switch (TypeToEnum<T>::value)
        {
        case VX_TYPE_UINT8:
            val.U8 = _val;
            break;
        case VX_TYPE_INT16:
            val.S16 = _val;
            break;
        case VX_TYPE_UINT16:
            val.U16 = _val;
            break;
        case VX_TYPE_INT32:
            val.S32 = _val;
            break;
        case VX_TYPE_UINT32:
            val.U32 = _val;
            break;
        default:
            throw WrapperError("Unsupported constant border value type");
        }
        setBorder(mode, val);
    }

    /// vxSetNodeAttribute(BORDER) wrapper
    void setBorder(vx_enum mode)
    { vx_pixel_value_t val = {}; setBorder(mode, val); }
#endif

    /// vxSetNodeAttribute(LOCAL_DATA_SIZE) wrapper
    void setDataSize(vx_size size)
    { setAttribute(VX_NODE_LOCAL_DATA_SIZE, size); }

    /// vxSetNodeAttribute(LOCAL_DATA_PTR) wrapper
    void setDataPtr(void* ptr)
    { setAttribute(VX_NODE_LOCAL_DATA_PTR, ptr); }
};


/// vx_image wrapper
class Image : public RefWrapper<vx_image>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Image);

    /// vxCreateImage() wrapper
    static Image create(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image format)
    { return Image(vxCreateImage(context, width, height, format)); }

    /// vxCreateVirtualImage() wrapper
    static Image createVirtual(vx_graph graph, vx_uint32 width = 0, vx_uint32 height = 0, vx_df_image format = VX_DF_IMAGE_VIRT)
    { return Image(vxCreateVirtualImage(graph, width, height, format)); }

#ifdef VX_VERSION_1_1
    /// vxCreateUniformImage() wrapper
    static Image createUniform(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image format, const vx_pixel_value_t& value)
    { return Image(vxCreateUniformImage(context, width, height, format, &value)); }
#else
    /// vxCreateUniformImage() wrapper
    static Image createUniform(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image format, const void* value)
    { return Image(vxCreateUniformImage(context, width, height, format, value)); }
#endif
    template <typename T>
    static Image createUniform(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image format, const T value)
    {
#if VX_VERSION > VX_VERSION_1_0
        vx_pixel_value_t pixel;
        switch (format)
        {
        case VX_DF_IMAGE_U8:pixel.U8 = (vx_uint8)value; break;
        case VX_DF_IMAGE_S16:pixel.S16 = (vx_int16)value; break;
        case VX_DF_IMAGE_U16:pixel.U16 = (vx_uint16)value; break;
        case VX_DF_IMAGE_S32:pixel.S32 = (vx_int32)value; break;
        case VX_DF_IMAGE_U32:pixel.U32 = (vx_uint32)value; break;
        default:throw ivx::WrapperError("uniform image type unsupported by this call");
        }
        return Image(vxCreateUniformImage(context, width, height, format, &pixel));
#else
        return Image(vxCreateUniformImage(context, width, height, format, &value));
#endif
    }

    /// Planes number for the specified image format (fourcc)
    /// \return 0 for unknown formats
    static vx_size planes(vx_df_image format)
    {
        switch (format)
        {
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_YUV4: return 3;
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21: return 2;
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_RGBX:
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_U8:
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
        case /*VX_DF_IMAGE_F32*/VX_DF_IMAGE('F', '0', '3', '2'):
                               return 1;
        default:               return 0;
        }
    }

    /// Create vx_imagepatch_addressing_t structure with default values
    static vx_imagepatch_addressing_t createAddressing()
    { vx_imagepatch_addressing_t ipa = VX_IMAGEPATCH_ADDR_INIT; return ipa; }

    /// Create vx_imagepatch_addressing_t structure with the provided values
    static vx_imagepatch_addressing_t createAddressing(
            vx_uint32 dimX, vx_uint32 dimY,
            vx_int32 strideX, vx_int32 strideY,
            vx_uint32 scaleX = VX_SCALE_UNITY, vx_uint32 scaleY = VX_SCALE_UNITY )
    {
        if (std::abs(strideY) < std::abs(strideX*(vx_int32)dimX))
            throw WrapperError(std::string(__func__)+"(): invalid arguments");
        vx_imagepatch_addressing_t ipa = VX_IMAGEPATCH_ADDR_INIT;
        ipa.dim_x = dimX;
        ipa.dim_y = dimY;
        ipa.stride_x = strideX;
        ipa.stride_y = strideY;
        ipa.scale_x = scaleX;
        ipa.scale_y = scaleY;
        return ipa;
    }

    /// Create vx_imagepatch_addressing_t structure for the specified image plane and its valid region
    vx_imagepatch_addressing_t createAddressing(vx_uint32 planeIdx)
    { return createAddressing(planeIdx, getValidRegion()); }

    /// Create vx_imagepatch_addressing_t structure for the specified image plane and the provided region
    vx_imagepatch_addressing_t createAddressing(vx_uint32 planeIdx, const vx_rectangle_t& rect)
    {
        vx_uint32 w = rect.end_x-rect.start_x, h = rect.end_y-rect.start_y;
        vx_size patchBytes = computePatchSize(planeIdx, rect);
        vx_imagepatch_addressing_t ipa = createAddressing(w, h, (vx_int32)(patchBytes/w/h), (vx_int32)(patchBytes/h));
        return ipa;
    }

#ifndef VX_VERSION_1_1
    static const vx_enum VX_MEMORY_TYPE_HOST = VX_IMPORT_TYPE_HOST;
#endif
    /// vxCreateImageFromHandle() wrapper
    static Image createFromHandle(
            vx_context context, vx_df_image format,
            const std::vector<vx_imagepatch_addressing_t>& addrs,
            const std::vector<void*>& ptrs, vx_enum memType = VX_MEMORY_TYPE_HOST )
    {
        vx_size num = planes(format);
        if(num == 0)
            throw WrapperError(std::string(__func__)+"(): unknown/unexpected planes number for the requested format");
        if (addrs.size() != num || ptrs.size() != num)
            throw WrapperError(std::string(__func__)+"(): incomplete input");
#ifdef VX_VERSION_1_1
        return Image(vxCreateImageFromHandle(context, format, &addrs[0], &ptrs[0], memType));
#else
        return Image( vxCreateImageFromHandle(context, format,
                     const_cast<vx_imagepatch_addressing_t*>(&addrs[0]),
                     const_cast<void**>(&ptrs[0]), memType) );
#endif
    }

    /// vxCreateImageFromHandle() wrapper for a single plane image
    static Image createFromHandle(vx_context context, vx_df_image format,const vx_imagepatch_addressing_t& addr, void* ptr)
    {
        if(planes(format) != 1) throw WrapperError(std::string(__func__)+"(): not a single plane format");
        return Image(vxCreateImageFromHandle(context, format, const_cast<vx_imagepatch_addressing_t*> (&addr), &ptr, VX_MEMORY_TYPE_HOST));
    }

#ifdef VX_VERSION_1_1
    /// vxSwapImageHandle() wrapper
    /// \param newPtrs  keeps addresses of new image planes data, can be of image planes size or empty when new pointers are not provided
    /// \param prevPtrs storage for the previous addresses of image planes data, can be of image planes size or empty when previous pointers are not needed
    void swapHandle(const std::vector<void*>& newPtrs, std::vector<void*>& prevPtrs)
    {
        vx_size num = planes();
        if(num == 0)
            throw WrapperError(std::string(__func__)+"(): unexpected planes number");
        if (!newPtrs.empty() && newPtrs.size() != num)
            throw WrapperError(std::string(__func__)+"(): unexpected number of input pointers");
        if (!prevPtrs.empty() && prevPtrs.size() != num)
            throw WrapperError(std::string(__func__)+"(): unexpected number of output pointers");
        IVX_CHECK_STATUS( vxSwapImageHandle( ref,
                                             newPtrs.empty()  ? 0 : &newPtrs[0],
                                             prevPtrs.empty() ? 0 : &prevPtrs[0],
                                             num ) );
    }

    /// vxSwapImageHandle() wrapper for a single plane image
    /// \param newPtr  an address of new image data, can be zero when new pointer is not provided
    /// \return the previuos address of image data
    void* swapHandle(void* newPtr)
    {
        if(planes() != 1) throw WrapperError(std::string(__func__)+"(): not a single plane image");
        void* prevPtr = 0;
        IVX_CHECK_STATUS( vxSwapImageHandle(ref, &newPtr, &prevPtr, 1) );
        return prevPtr;
    }

    /// vxSwapImageHandle() wrapper for the case when no new pointers provided and previous ones are not needed (retrive memory back)
    void swapHandle()
    { IVX_CHECK_STATUS( vxSwapImageHandle(ref, 0, 0, 0) ); }

    /// vxCreateImageFromChannel() wrapper
    Image createFromChannel(vx_enum channel)
    { return Image(vxCreateImageFromChannel(ref, channel)); }
#endif // VX_VERSION_1_1

    /// vxQueryImage() wrapper
    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS( vxQueryImage(ref, att, &value, sizeof(value)) ); }

#ifndef VX_VERSION_1_1
static const vx_enum
    VX_IMAGE_WIDTH  = VX_IMAGE_ATTRIBUTE_WIDTH,
    VX_IMAGE_HEIGHT = VX_IMAGE_ATTRIBUTE_HEIGHT,
    VX_IMAGE_FORMAT = VX_IMAGE_ATTRIBUTE_FORMAT,
    VX_IMAGE_PLANES = VX_IMAGE_ATTRIBUTE_PLANES,
    VX_IMAGE_SPACE  = VX_IMAGE_ATTRIBUTE_SPACE,
    VX_IMAGE_RANGE  = VX_IMAGE_ATTRIBUTE_RANGE,
    VX_IMAGE_SIZE   = VX_IMAGE_ATTRIBUTE_SIZE;
#endif

    /// vxQueryImage(VX_IMAGE_WIDTH) wrapper
    vx_uint32 width() const
    {
        vx_uint32 v;
        query(VX_IMAGE_WIDTH, v);
        return v;
    }

    /// vxQueryImage(VX_IMAGE_HEIGHT) wrapper
    vx_uint32 height() const
    {
        vx_uint32 v;
        query(VX_IMAGE_HEIGHT, v);
        return v;
    }

    /// vxQueryImage(VX_IMAGE_FORMAT) wrapper
    vx_df_image format() const
    {
        vx_df_image v;
        query(VX_IMAGE_FORMAT, v);
        return v;
    }

    /// vxQueryImage(VX_IMAGE_PLANES) wrapper
    vx_size planes() const
    {
        vx_size v;
        query(VX_IMAGE_PLANES, v);
        return v;
    }

    /// vxQueryImage(VX_IMAGE_SPACE) wrapper
    vx_enum space() const
    {
        vx_enum v;
        query(VX_IMAGE_SPACE, v);
        return v;
    }

    /// vxQueryImage(VX_IMAGE_RANGE) wrapper
    vx_enum range() const
    {
        vx_enum v;
        query(VX_IMAGE_RANGE, v);
        return v;
    }

    /// vxQueryImage(VX_IMAGE_SIZE) wrapper
    vx_size size() const
    {
        vx_size v;
        query(VX_IMAGE_SIZE, v);
        return v;
    }

#ifdef VX_VERSION_1_1
    /// vxQueryImage(VX_IMAGE_MEMORY_TYPE) wrapper
    vx_memory_type_e memType() const
    {
        vx_memory_type_e v;
        query(VX_IMAGE_MEMORY_TYPE, v);
        return v;
    }
#endif // VX_VERSION_1_1

    /// vxSetImageAttribute() wrapper
    template<typename T>
    void setAttribute(vx_enum att, T& value) const
    { IVX_CHECK_STATUS(vxSetImageAttribute(ref, att, &value, sizeof(value))); }

    /// vxSetImageAttribute(SPACE) wrapper
    void setColorSpace(const vx_enum& sp)
    { setAttribute(VX_IMAGE_SPACE, sp); }

    /// vxGetValidRegionImage() wrapper
    vx_rectangle_t getValidRegion() const
    {
        vx_rectangle_t rect;
        IVX_CHECK_STATUS( vxGetValidRegionImage(ref, &rect) );
        return rect;
    }

    /// vxComputeImagePatchSize(valid region) wrapper
    vx_size computePatchSize(vx_uint32 planeIdx)
    { return computePatchSize(planeIdx, getValidRegion()); }

    /// vxComputeImagePatchSize() wrapper
    vx_size computePatchSize(vx_uint32 planeIdx, const vx_rectangle_t& rect)
    {
        vx_size bytes = vxComputeImagePatchSize(ref, &rect, planeIdx);
        if (bytes == 0) throw WrapperError(std::string(__func__)+"(): vxComputeImagePatchSize returned 0");
        return bytes;
    }

#ifdef VX_VERSION_1_1
    /// vxSetImageValidRectangle() wrapper
    void setValidRectangle(const vx_rectangle_t& rect)
    { IVX_CHECK_STATUS( vxSetImageValidRectangle(ref, &rect) ); }
#endif // VX_VERSION_1_1

    /// Copy image plane content to the provided memory
    void copyTo(vx_uint32 planeIdx, const vx_imagepatch_addressing_t& addr, void* data)
    {
        if(!data) throw WrapperError(std::string(__func__)+"(): output pointer is 0");
        vx_rectangle_t r = getValidRegion();
        // TODO: add sizes consistency checks
        /*
        vx_uint32 w = r.end_x - r.start_x, h = r.end_y - r.start_y;
        if (w != addr.dim_x) throw WrapperError("Image::copyTo(): inconsistent dimension X");
        if (h != addr.dim_y) throw WrapperError("Image::copyTo(): inconsistent dimension Y");
        */
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyImagePatch(ref, &r, planeIdx, &addr, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#else
        vx_imagepatch_addressing_t* a = const_cast<vx_imagepatch_addressing_t*>(&addr);
        IVX_CHECK_STATUS(vxAccessImagePatch(ref, &r, planeIdx, a, &data, VX_READ_ONLY));
        IVX_CHECK_STATUS(vxCommitImagePatch(ref, 0, planeIdx, a, data));
#endif
    }

    /// Copy the provided memory data to the specified image plane
    void copyFrom(vx_uint32 planeIdx, const vx_imagepatch_addressing_t& addr, const void* data)
    {
        if (!data) throw WrapperError(std::string(__func__)+"(): input pointer is 0");
        vx_rectangle_t r = getValidRegion();
        // TODO: add sizes consistency checks
        /*
        vx_uint32 w = r.end_x - r.start_x, h = r.end_y - r.start_y;
        //vx_size patchBytes = vxComputeImagePatchSize(ref, &r, planeIdx);
        if (w != addr.dim_x) throw WrapperError("Image::copyFrom(): inconsistent dimension X");
        if (h != addr.dim_y) throw WrapperError("Image::copyFrom(): inconsistent dimension Y");
        */
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyImagePatch(ref, &r, planeIdx, &addr, (void*)data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        vx_imagepatch_addressing_t* a = const_cast<vx_imagepatch_addressing_t*>(&addr);
        IVX_CHECK_STATUS(vxAccessImagePatch(ref, &r, planeIdx, a, const_cast<void**>(&data), VX_WRITE_ONLY));
        IVX_CHECK_STATUS(vxCommitImagePatch(ref, &r, planeIdx, a, data));
#endif
    }

    /// vxCopyImagePatch() wrapper (or vxAccessImagePatch() + vxCommitImagePatch() for OpenVX 1.0)
    void copy( vx_uint32 planeIdx, vx_rectangle_t rect,
               const vx_imagepatch_addressing_t& addr, void* data,
               vx_enum usage, vx_enum memoryType = VX_MEMORY_TYPE_HOST )
    {
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyImagePatch(ref, &rect, planeIdx, &addr, (void*)data, usage, memoryType));
#else
        (void)memoryType;
        vx_imagepatch_addressing_t* a = const_cast<vx_imagepatch_addressing_t*>(&addr);
        IVX_CHECK_STATUS(vxAccessImagePatch(ref, &rect, planeIdx, a, &data, usage));
        IVX_CHECK_STATUS(vxCommitImagePatch(ref, &rect, planeIdx, a, data));
#endif
    }

    /// Convert cv::Mat type to standard image format (fourcc), throws WrapperError if not possible
    static vx_df_image matTypeToFormat(int matType)
    {
        switch (matType)
        {
            case CV_8UC4:  return VX_DF_IMAGE_RGBX;
            case CV_8UC3:  return VX_DF_IMAGE_RGB;
            case CV_8UC1:  return VX_DF_IMAGE_U8;
            case CV_16UC1: return VX_DF_IMAGE_U16;
            case CV_16SC1: return VX_DF_IMAGE_S16;
            case CV_32SC1: return VX_DF_IMAGE_S32;
            case CV_32FC1: return VX_DF_IMAGE('F', '0', '3', '2');
            default:       throw WrapperError(std::string(__func__)+"(): unsupported cv::Mat type");
        }
    }

#ifdef IVX_USE_OPENCV
    /// Convert image format (fourcc) to cv::Mat type, throws WrapperError if not possible
    static int formatToMatType(vx_df_image format, vx_uint32 planeIdx = 0)
    {
        switch (format)
        {
        case VX_DF_IMAGE_RGB:  return CV_8UC3;
        case VX_DF_IMAGE_RGBX: return CV_8UC4;
        case VX_DF_IMAGE_U8:   return CV_8UC1;
        case VX_DF_IMAGE_U16:  return CV_16UC1;
        case VX_DF_IMAGE_S16:  return CV_16SC1;
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:  return CV_32SC1;
        case VX_DF_IMAGE('F', '0', '3', '2'):
                               return CV_32FC1;
        case VX_DF_IMAGE_YUV4:
        case VX_DF_IMAGE_IYUV: return CV_8UC1;
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_YUYV: return CV_8UC2;
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21: return planeIdx == 0 ? CV_8UC1 : CV_8UC2;
        default: throw WrapperError(std::string(__func__)+"(): unsupported image format");
        }
    }

    /// Initialize cv::Mat shape to fit the specified image plane data
    void createMatForPlane(cv::Mat& m, vx_uint32 planeIdx)
    {
        vx_df_image f = format();
        //vx_uint32 w = width(), h = height();
        vx_rectangle_t r = getValidRegion();
        vx_int32 w = vx_int32(r.end_x - r.start_x), h = vx_int32(r.end_y - r.start_y);
        switch (f)
        {
        case VX_DF_IMAGE_IYUV:
            if (planeIdx == 0u) m.create(h, w, formatToMatType(f));
            else if (planeIdx == 1u || planeIdx == 2u)  m.create(h/2, w/2, formatToMatType(f));
            else throw WrapperError(std::string(__func__)+"(): wrong plane index");
            break;
        case VX_DF_IMAGE_YUV4:
            if (planeIdx == 0u || planeIdx == 1u || planeIdx == 2u) m.create(h, w, formatToMatType(f));
            else throw WrapperError(std::string(__func__)+"(): wrong plane index");
            break;
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            if (planeIdx == 0u) m.create(h, w, formatToMatType(f, 0));
            else if (planeIdx == 1u)  m.create(h/2, w/2, formatToMatType(f, 1));
            else throw WrapperError(std::string(__func__)+"(): wrong plane index");
            break;
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_RGBX:
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_U8:
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
        case /*VX_DF_IMAGE_F32*/VX_DF_IMAGE('F', '0', '3', '2'):
            if(planeIdx == 0u) m.create(h, w, formatToMatType(f));
            else throw WrapperError(std::string(__func__)+"(): wrong plane index");
            break;
        default: throw WrapperError(std::string(__func__)+"(): unsupported color format");
        }
    }

    /// Create vx_imagepatch_addressing_t corresponding to the provided cv::Mat
    static vx_imagepatch_addressing_t createAddressing(const cv::Mat& m)
    {
        if(m.empty()) throw WrapperError(std::string(__func__)+"(): empty input Mat");
        return createAddressing((vx_uint32)m.cols, (vx_uint32)m.rows, (vx_int32)m.elemSize(), (vx_int32)m.step);
    }

    /// Copy image plane content to the provided cv::Mat (reallocate if needed)
    void copyTo(vx_uint32 planeIdx, cv::Mat& m)
    {
        createMatForPlane(m, planeIdx);
        copyTo(planeIdx, createAddressing((vx_uint32)m.cols, (vx_uint32)m.rows, (vx_int32)m.elemSize(), (vx_int32)m.step), m.ptr());
    }

    /// Copy the provided cv::Mat data to the specified image plane
    void copyFrom(vx_uint32 planeIdx, const cv::Mat& m)
    {
        if(m.empty()) throw WrapperError(std::string(__func__)+"(): empty input Mat");
        // TODO: add sizes consistency checks
        //vx_rectangle_t r = getValidRegion();
        copyFrom(planeIdx, createAddressing((vx_uint32)m.cols, (vx_uint32)m.rows, (vx_int32)m.elemSize(), (vx_int32)m.step), m.ptr());
    }

/*
private:
    cv::Mat _mat; // TODO: update copy/move-c-tors, operator=() and swapHandles()
public:
    static Image createFromHandle(vx_context context, const cv::Mat& mat)
    {
        if(mat.empty()) throw WrapperError(std::string(__func__)+"(): empty cv::Mat");
        Image res = createFromHandle(context, matTypeToFormat(mat.type()), createAddressing(mat), mat.data );
        res._mat = mat;
        return res;
    }
*/
#endif //IVX_USE_OPENCV

    struct Patch;
};

/// Helper class for a mapping vx_image patch
struct Image::Patch
{
public:
    /// reference to the current vx_imagepatch_addressing_t
    const vx_imagepatch_addressing_t& addr() const
    { return _addr;}

    /// current pixels data pointer
    void* data() const
    { return _data; }

#ifdef VX_VERSION_1_1
    /// vx_memory_type_e for the current data pointer
    vx_memory_type_e memType() const
    { return _memType; }

    /// vx_map_id for the current mapping
    vx_map_id mapId() const
    { return _mapId; }
#else
    /// reference to vx_rectangle_t for the current mapping
    const vx_rectangle_t& rectangle() const
    { return _rect; }

    /// Image plane index for the current mapping
    vx_uint32 planeIndex() const
    { return _planeIdx; }
#endif // VX_VERSION_1_1

    /// vx_image for the current mapping
    vx_image image() const
    { return _img; }

    /// where this patch is  mapped
    bool isMapped() const
    { return _img != 0; }

#ifdef IVX_USE_OPENCV
    /// Reference to cv::Mat instance wrapping the mapped image data, becomes invalid after unmap()
    cv::Mat& getMat()
    { return _m; }
#endif //IVX_USE_OPENCV

protected:
    vx_imagepatch_addressing_t _addr;
    void* _data;
    vx_image _img;
#ifdef VX_VERSION_1_1
    vx_memory_type_e _memType;
    vx_map_id _mapId;
#else
    vx_rectangle_t _rect;
    vx_uint32 _planeIdx;
#endif
#ifdef IVX_USE_OPENCV
    cv::Mat _m;
#endif

public:
    /// Default constructor
    Patch() : _addr(createAddressing()), _data(0), _img(0)
#ifdef VX_VERSION_1_1
       , _memType(VX_MEMORY_TYPE_HOST), _mapId(0)
    {}
#else
       , _planeIdx(-1)
    { _rect.start_x = _rect.end_x = _rect.start_y = _rect.end_y = 0u; }
#endif

#ifndef IVX_USE_CXX98
    /// Move constructor
    Patch(Patch&& p) : Patch()
    {
        using std::swap;
        swap(_addr, p._addr);
        swap(_data, p._data);
#ifdef VX_VERSION_1_1
        swap(_memType, p._memType);
        swap(_mapId, p._mapId);
#else
        swap(_rect, p._rect);
        swap(_planeIdx, p._planeIdx);
#endif
        swap(_img, p._img);
#ifdef IVX_USE_OPENCV
        swap(_m, p._m);
#endif
    }
#endif

    /// vxMapImagePatch(VX_READ_ONLY, planeIdx valid region)
    void map(vx_image img, vx_uint32 planeIdx)
    { map(img, planeIdx, Image(img, true).getValidRegion()); }

    /// vxMapImagePatch() wrapper (or vxAccessImagePatch() for 1.0)
    void map(vx_image img, vx_uint32 planeIdx, const vx_rectangle_t& rect, vx_enum usage = VX_READ_ONLY, vx_uint32 flags = 0)
    {
        if (isMapped()) throw WrapperError(std::string(__func__)+"(): already mapped");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxMapImagePatch(img, &rect, planeIdx, &_mapId, &_addr, &_data, usage, _memType, flags) );
#else
        IVX_CHECK_STATUS(vxAccessImagePatch(img, &rect, planeIdx, &_addr, &_data, usage));
        (void)flags;
        _rect = rect;
        _planeIdx = planeIdx;
#endif
        if (_data == 0) throw WrapperError(std::string(__func__)+"(): mapped address is null");
        _img = img;
#ifdef IVX_USE_OPENCV
        vx_df_image format;
        IVX_CHECK_STATUS( vxQueryImage(_img, VX_IMAGE_FORMAT, &format, sizeof(format)) );
        int matType = formatToMatType(format);
        _m = cv::Mat( vx_int32((vx_int64)_addr.dim_y * VX_SCALE_UNITY / _addr.scale_y),
                      vx_int32((vx_int64)_addr.dim_x * VX_SCALE_UNITY / _addr.scale_x),
                      matType, _data, std::size_t(_addr.stride_y) );
#endif
    }

    /// vxUnmapImagePatch() wrapper (or vxCommitImagePatch() for 1.0)
    void unmap()
    {
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxUnmapImagePatch(_img, _mapId));
        _mapId = 0;
#else
        IVX_CHECK_STATUS(vxCommitImagePatch(_img, &_rect, _planeIdx, &_addr, _data));
        _rect.start_x = _rect.end_x = _rect.start_y = _rect.end_y = 0u;
        _planeIdx = -1;

#endif
        _img = 0;
        _data = 0;
        _addr = createAddressing();
#ifdef IVX_USE_OPENCV
        _m.release();
#endif
    }

    /// Destructor
    ~Patch()
    { try { if (_img) unmap(); } catch(...) {; /*ignore*/} }

    /// Pointer to the specified pixel data (vxFormatImagePatchAddress2d)
    void* pixelPtr(vx_uint32 x, vx_uint32 y)
    {
        if (!_data) throw WrapperError(std::string(__func__)+"(): base pointer is NULL");
        if (x >= _addr.dim_x) throw WrapperError(std::string(__func__)+"(): X out of range");
        if (y >= _addr.dim_y) throw WrapperError(std::string(__func__)+"(): Y out of range");
        return vxFormatImagePatchAddress2d(_data, x, y, &_addr);
    }

private:
    Patch(const Patch& p); // = delete
    Patch& operator=(const Patch&); // = delete
#ifndef IVX_USE_CXX98
    Patch& operator=(Patch&&); // = delete
#endif
};

/// vx_parameter wrapper
class Param : public RefWrapper<vx_parameter>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Param);
    // NYI
};

/// vx_scalar wrapper
class Scalar : public RefWrapper<vx_scalar>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Scalar);

    /// vxCreateScalar() wrapper
    static Scalar create(vx_context c, vx_enum dataType, const void *ptr)
    { return Scalar( vxCreateScalar(c, dataType, ptr) ); }

    /// vxCreateScalar() wrapper, value is passed as a value not as a pointer
    template<typename T> static Scalar create(vx_context c, vx_enum dataType, T value)
    {
        typedef int static_assert_not_pointer[is_pointer<T>::value ? -1 : 1];
        return Scalar( vxCreateScalar(c, dataType, &value) );
    }

    /// vxCreateScalar() wrapper, data type is guessed based on the passed value
    template<vx_enum E> static Scalar create(vx_context c, typename EnumToType<E>::type value)
    { return Scalar( vxCreateScalar(c, E, &value) ); }

#ifndef VX_VERSION_1_1
static const vx_enum VX_SCALAR_TYPE = VX_SCALAR_ATTRIBUTE_TYPE;
#endif
    /// Get scalar data type
    vx_enum type()
    {
        vx_enum val;
        IVX_CHECK_STATUS( vxQueryScalar(ref, VX_SCALAR_TYPE, &val, sizeof(val)) );
        return val;
    }

    /// Get scalar value
    template<typename T>
    void getValue(T& val)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, type()))
            throw WrapperError(std::string(__func__)+"(): incompatible types");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS( vxCopyScalar(ref, &val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) );
#else
        IVX_CHECK_STATUS( vxReadScalarValue(ref, &val) );
#endif
    }

    /// Get scalar value
    template<typename T>
    T getValue()
    {
        T val;
        getValue(val);
        return val;
    }


    /// Set scalar value
    template<typename T>
    void setValue(T val)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, type()))
            throw WrapperError(std::string(__func__)+"(): incompatible types");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyScalar(ref, &val, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS( vxWriteScalarValue(ref, &val) );
#endif
    }
};

/// vx_threshold wrapper
class Threshold : public RefWrapper<vx_threshold>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Threshold);

    /// vxCreateThreshold() wrapper
    static Threshold create(vx_context c, vx_enum threshType, vx_enum dataType)
    { return Threshold(vxCreateThreshold(c, threshType, dataType)); }

#ifndef VX_VERSION_1_1
static const vx_enum
    VX_THRESHOLD_TYPE            = VX_THRESHOLD_ATTRIBUTE_TYPE,
    VX_THRESHOLD_THRESHOLD_VALUE = VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE,
    VX_THRESHOLD_THRESHOLD_LOWER = VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
    VX_THRESHOLD_THRESHOLD_UPPER = VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
    VX_THRESHOLD_TRUE_VALUE      = VX_THRESHOLD_ATTRIBUTE_TRUE_VALUE,
    VX_THRESHOLD_FALSE_VALUE     = VX_THRESHOLD_ATTRIBUTE_FALSE_VALUE,
    VX_THRESHOLD_DATA_TYPE       = VX_THRESHOLD_ATTRIBUTE_DATA_TYPE;
#endif


    /// Create binary threshold with the provided value
    static Threshold createBinary(vx_context c, vx_enum dataType, vx_int32 val)
    {
        Threshold thr = create(c, VX_THRESHOLD_TYPE_BINARY, dataType);
        IVX_CHECK_STATUS( vxSetThresholdAttribute(thr.ref, VX_THRESHOLD_THRESHOLD_VALUE, &val, sizeof(val)) );
        return thr;
    }

    /// Create range threshold with the provided low and high values
    static Threshold createRange(vx_context c, vx_enum dataType, vx_int32 valLower, vx_int32 valUpper)
    {
        Threshold thr = create(c, VX_THRESHOLD_TYPE_RANGE, dataType);
        IVX_CHECK_STATUS( vxSetThresholdAttribute(thr.ref, VX_THRESHOLD_THRESHOLD_LOWER, &valLower, sizeof(valLower)) );
        IVX_CHECK_STATUS( vxSetThresholdAttribute(thr.ref, VX_THRESHOLD_THRESHOLD_UPPER, &valUpper, sizeof(valUpper)) );
        return thr;
    }

    /// vxQueryThreshold() wrapper
    template<typename T>
    void query(vx_enum att, T& val) const
    { IVX_CHECK_STATUS( vxQueryThreshold(ref, att, &val, sizeof(val)) ); }

    /// vxQueryThreshold(VX_THRESHOLD_TYPE) wrapper
    vx_enum type() const
    {
        vx_enum v;
        query(VX_THRESHOLD_TYPE, v);
        return v;
    }

    /// vxQueryThreshold(DATA_TYPE) wrapper
    vx_enum dataType() const
    {
        vx_enum v;
        query(VX_THRESHOLD_DATA_TYPE, v);
        return v;
    }

    /// vxQueryThreshold(THRESHOLD_VALUE) wrapper
    vx_int32 value() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_THRESHOLD_VALUE, v);
        return v;
    }

    /// vxQueryThreshold(THRESHOLD_LOWER) wrapper
    vx_int32 valueLower() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_THRESHOLD_LOWER, v);
        return v;
    }

    /// vxQueryThreshold(THRESHOLD_UPPER) wrapper
    vx_int32 valueUpper() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_THRESHOLD_UPPER, v);
        return v;
    }

    /// vxQueryThreshold(TRUE_VALUE) wrapper
    vx_int32 valueTrue() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_TRUE_VALUE, v);
        return v;
    }

    /// vxQueryThreshold(FALSE_VALUE) wrapper
    vx_int32 valueFalse() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_FALSE_VALUE, v);
        return v;
    }

    /// vxSetThresholdAttribute(THRESHOLD_VALUE) wrapper
    void setValue(vx_int32 &val)
    { IVX_CHECK_STATUS(vxSetThresholdAttribute(ref, VX_THRESHOLD_THRESHOLD_VALUE, &val, sizeof(val))); }

    /// vxSetThresholdAttribute(THRESHOLD_LOWER) wrapper
    void setValueLower(vx_int32 &val)
    { IVX_CHECK_STATUS(vxSetThresholdAttribute(ref, VX_THRESHOLD_THRESHOLD_LOWER, &val, sizeof(val))); }

    /// vxSetThresholdAttribute(THRESHOLD_UPPER) wrapper
    void setValueUpper(vx_int32 &val)
    { IVX_CHECK_STATUS(vxSetThresholdAttribute(ref, VX_THRESHOLD_THRESHOLD_UPPER, &val, sizeof(val))); }

    /// vxSetThresholdAttribute(TRUE_VALUE) wrapper
    void setValueTrue(vx_int32 &val)
    { IVX_CHECK_STATUS(vxSetThresholdAttribute(ref, VX_THRESHOLD_TRUE_VALUE, &val, sizeof(val))); }

    /// vxSetThresholdAttribute(FALSE_VALUE) wrapper
    void setValueFalse(vx_int32 &val)
    { IVX_CHECK_STATUS(vxSetThresholdAttribute(ref, VX_THRESHOLD_FALSE_VALUE, &val, sizeof(val))); }
};

/// vx_array wrapper
class Array : public RefWrapper<vx_array>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Array);

    /// vxCreateArray() wrapper
    static Array create(vx_context c, vx_enum type, vx_size capacity)
    { return Array(vxCreateArray(c, type, capacity)); }

    /// vxCreateVirtualArray() wrapper
    static Array createVirtual(vx_graph g, vx_enum type, vx_size capacity)
    { return Array(vxCreateVirtualArray(g, type, capacity)); }

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_MEMORY_TYPE_HOST = VX_IMPORT_TYPE_HOST,
        VX_ARRAY_ITEMTYPE   = VX_ARRAY_ATTRIBUTE_ITEMTYPE,
        VX_ARRAY_NUMITEMS   = VX_ARRAY_ATTRIBUTE_NUMITEMS,
        VX_ARRAY_CAPACITY   = VX_ARRAY_ATTRIBUTE_CAPACITY,
        VX_ARRAY_ITEMSIZE   = VX_ARRAY_ATTRIBUTE_ITEMSIZE;
#endif

    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS( vxQueryArray(ref, att, &value, sizeof(value)) ); }

    vx_enum itemType() const
    {
        vx_enum v;
        query(VX_ARRAY_ITEMTYPE, v);
        return v;
    }

    vx_size itemSize() const
    {
        vx_size v;
        query(VX_ARRAY_ITEMSIZE, v);
        return v;
    }

    vx_size capacity() const
    {
        vx_size v;
        query(VX_ARRAY_CAPACITY, v);
        return v;
    }

    vx_size itemCount() const
    {
        vx_size v;
        query(VX_ARRAY_NUMITEMS, v);
        return v;
    }

    void addItems(vx_size count, const void* ptr, vx_size stride)
    {
        IVX_CHECK_STATUS(vxAddArrayItems(ref, count, ptr, stride));
    }

    void truncateArray(vx_size new_count)
    {
        if(new_count <= itemCount())
            IVX_CHECK_STATUS(vxTruncateArray(ref, new_count));
        else
            throw WrapperError(std::string(__func__) + "(): array is too small");
    }

    void copyRangeTo(size_t start, size_t end, void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): output pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyArrayRange(ref, start, end, itemSize(), data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#else
        vx_size stride = itemSize();
        IVX_CHECK_STATUS(vxAccessArrayRange(ref, start, end, &stride, &data, VX_READ_ONLY));
        IVX_CHECK_STATUS(vxCommitArrayRange(ref, start, end, data));
#endif
    }

    void copyTo(void* data)
    { copyRangeTo(0, itemCount(), data); }

    void copyRangeFrom(size_t start, size_t end, const void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): input pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyArrayRange(ref, start, end, itemSize(), const_cast<void*>(data), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        vx_size stride = itemSize();
        IVX_CHECK_STATUS(vxAccessArrayRange(ref, start, end, &stride, const_cast<void**>(&data), VX_WRITE_ONLY));
        IVX_CHECK_STATUS(vxCommitArrayRange(ref, start, end, data));
#endif
    }

    void copyFrom(const void* data)
    { copyRangeFrom(0, itemCount(), data); }

    void copyRange(size_t start, size_t end, void* data, vx_enum usage, vx_enum memType = VX_MEMORY_TYPE_HOST)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): data pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyArrayRange(ref, start, end, itemSize(), data, usage, memType));
#else
        vx_size stride = itemSize();
        IVX_CHECK_STATUS(vxAccessArrayRange(ref, start, end, &stride, &data, usage));
        IVX_CHECK_STATUS(vxCommitArrayRange(ref, start, end, data));
        (void)memType;
#endif
    }

    void copy(void* data, vx_enum usage, vx_enum memType = VX_MEMORY_TYPE_HOST)
    { copyRange(0, itemCount(), data, usage, memType); }

    template<typename T> void addItem(const T& item)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, itemType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        addItems(1, &item, sizeof(T));
    }

    template<typename T> void addItems(const std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, itemType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        addItems(data.size(), &data[0], itemSize());
    }

    template<typename T> void copyRangeTo(size_t start, size_t end, std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, itemType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (data.empty())
            data.resize((end - start));
        else if (data.size() != (end - start))
        {
            throw WrapperError(std::string(__func__) + "(): destination size is wrong");
        }
        copyRangeTo(start, end, &data[0]);
    }

    template<typename T> void copyTo(std::vector<T>& data)
    { copyRangeTo(0, itemCount(), data); }

    template<typename T> void copyRangeFrom(size_t start, size_t end, const std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, itemType()))
            throw WrapperError(std::string(__func__) + "(): source type is wrong");
        if (data.size() != (end - start)) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        copyRangeFrom(start, end, &data[0]);
    }

    template<typename T> void copyFrom(std::vector<T>& data)
    { copyRangeFrom(0, itemCount(), data); }

#ifdef IVX_USE_OPENCV
    void addItems(cv::InputArray ia)
    {
        cv::Mat m = ia.getMat();
        if (m.type() != enumToCVType(itemType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        addItems(m.total(), m.isContinuous() ? m.ptr() : m.clone().ptr(),
                 (vx_size)(m.elemSize()));
    }

    void copyRangeTo(size_t start, size_t end, cv::Mat& m)
    {
        if (m.type() != enumToCVType(itemType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (!(
                ((vx_size)(m.rows) == (end - start) && m.cols == 1) ||
                ((vx_size)(m.cols) == (end - start) && m.rows == 1)
            ) && !m.empty())
            throw WrapperError(std::string(__func__) + "(): destination size is wrong");

        if (m.isContinuous() && (vx_size)(m.total()) == (end - start))
        {
            copyRangeTo(start, end, m.ptr());
        }
        else
        {
            cv::Mat tmp(1, (int)(end - start), enumToCVType(itemType()));
            copyRangeTo(start, end, tmp.ptr());
            if (m.empty())
                m = tmp;
            else
                tmp.copyTo(m);
        }
    }

    void copyTo(cv::Mat& m)
    { copyRangeTo(0, itemCount(), m); }

    void copyRangeFrom(size_t start, size_t end, const cv::Mat& m)
    {
        if (!(
                ((vx_size)(m.rows) == (end - start) && m.cols == 1) ||
                ((vx_size)(m.cols) == (end - start) && m.rows == 1)
             ))
            throw WrapperError(std::string(__func__) + "(): source size is wrong");
        if (m.type() != enumToCVType(itemType()))
            throw WrapperError(std::string(__func__) + "(): source type is wrong");
        copyFrom(m.isContinuous() ? m.ptr() : m.clone().ptr());
    }

    void copyFrom(const cv::Mat& m)
    { copyRangeFrom(0, itemCount(), m); }
#endif //IVX_USE_OPENCV
};

/*
* Convolution
*/
class Convolution : public RefWrapper<vx_convolution>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Convolution);

    static Convolution create(vx_context context, vx_size columns, vx_size rows)
    { return Convolution(vxCreateConvolution(context, columns, rows)); }

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_MEMORY_TYPE_HOST    = VX_IMPORT_TYPE_HOST,
        VX_CONVOLUTION_ROWS    = VX_CONVOLUTION_ATTRIBUTE_ROWS,
        VX_CONVOLUTION_COLUMNS = VX_CONVOLUTION_ATTRIBUTE_COLUMNS,
        VX_CONVOLUTION_SCALE   = VX_CONVOLUTION_ATTRIBUTE_SCALE,
        VX_CONVOLUTION_SIZE    = VX_CONVOLUTION_ATTRIBUTE_SIZE;
#endif

    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS( vxQueryConvolution(ref, att, &value, sizeof(value)) ); }

    vx_size columns() const
    {
        vx_size v;
        query(VX_CONVOLUTION_COLUMNS, v);
        return v;
    }

    vx_size rows() const
    {
        vx_size v;
        query(VX_CONVOLUTION_ROWS, v);
        return v;
    }

    vx_uint32 scale() const
    {
        vx_uint32 v;
        query(VX_CONVOLUTION_SCALE, v);
        return v;
    }

    vx_size size() const
    {
        vx_size v;
        query(VX_CONVOLUTION_SIZE, v);
        return v;
    }

    vx_enum dataType()
    {
        return VX_TYPE_INT16;
    }

    void setScale(vx_uint32 newScale)
    { IVX_CHECK_STATUS( vxSetConvolutionAttribute(ref, VX_CONVOLUTION_SCALE, &newScale, sizeof(newScale)) ); }

    void copyTo(void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): output pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyConvolutionCoefficients(ref, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxReadConvolutionCoefficients(ref, (vx_int16 *)data));
#endif
    }

    void copyFrom(const void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): input pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyConvolutionCoefficients(ref, const_cast<void*>(data), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxWriteConvolutionCoefficients(ref, (const vx_int16 *)data));
#endif
    }

    void copy(void* data, vx_enum usage, vx_enum memType = VX_MEMORY_TYPE_HOST)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): data pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyConvolutionCoefficients(ref, data, usage, memType));
#else
        if (usage == VX_READ_ONLY)
            IVX_CHECK_STATUS(vxReadConvolutionCoefficients(ref, (vx_int16 *)data));
        else if (usage == VX_WRITE_ONLY)
            IVX_CHECK_STATUS(vxWriteConvolutionCoefficients(ref, (const vx_int16 *)data));
        else
            throw WrapperError(std::string(__func__) + "(): unknown copy direction");
        (void)memType;
#endif
    }

    template<typename T> void copyTo(std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, dataType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (data.size()*sizeof(T) != size())
        {
            if (data.size() == 0)
                data.resize(size()/sizeof(T));
            else
                throw WrapperError(std::string(__func__) + "(): destination size is wrong");
        }
        copyTo(&data[0]);
    }

    template<typename T> void copyFrom(const std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, dataType()))
            throw WrapperError(std::string(__func__) + "(): source type is wrong");
        if (data.size()*sizeof(T) != size()) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        copyFrom(&data[0]);
    }

#ifdef IVX_USE_OPENCV
    void copyTo(cv::Mat& m)
    {
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (((vx_size)(m.rows) != rows() || (vx_size)(m.cols) != columns()) && !m.empty())
            throw WrapperError(std::string(__func__) + "(): destination size is wrong");

        if (m.isContinuous() && (vx_size)(m.rows) == rows() && (vx_size)(m.cols) == columns())
        {
            copyTo(m.ptr());
        }
        else
        {
            cv::Mat tmp((int)rows(), (int)columns(), enumToCVType(dataType()));
            copyTo(tmp.ptr());
            if (m.empty())
                m = tmp;
            else
                tmp.copyTo(m);
        }
    }

    void copyFrom(const cv::Mat& m)
    {
        if ((vx_size)(m.rows) != rows() || (vx_size)(m.cols) != columns()) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): source type is wrong");
        copyFrom(m.isContinuous() ? m.ptr() : m.clone().ptr());
    }
#endif //IVX_USE_OPENCV
};

/*
* Matrix
*/
class Matrix : public RefWrapper<vx_matrix>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Matrix);

    static Matrix create(vx_context context, vx_enum dataType, vx_size columns, vx_size rows)
    { return Matrix(vxCreateMatrix(context, dataType, columns, rows)); }

#ifdef VX_VERSION_1_1
    static Matrix createFromPattern(vx_context context, vx_enum pattern, vx_size columns, vx_size rows)
    { return Matrix(vxCreateMatrixFromPattern(context, pattern, columns, rows)); }
#endif

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_MEMORY_TYPE_HOST = VX_IMPORT_TYPE_HOST,
        VX_MATRIX_TYPE      = VX_MATRIX_ATTRIBUTE_TYPE,
        VX_MATRIX_ROWS      = VX_MATRIX_ATTRIBUTE_ROWS,
        VX_MATRIX_COLUMNS   = VX_MATRIX_ATTRIBUTE_COLUMNS,
        VX_MATRIX_SIZE      = VX_MATRIX_ATTRIBUTE_SIZE;
#endif

    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS( vxQueryMatrix(ref, att, &value, sizeof(value)) ); }

    vx_enum dataType() const
    {
        vx_enum v;
        query(VX_MATRIX_TYPE, v);
        return v;
    }

    vx_size columns() const
    {
        vx_size v;
        query(VX_MATRIX_COLUMNS, v);
        return v;
    }

    vx_size rows() const
    {
        vx_size v;
        query(VX_MATRIX_ROWS, v);
        return v;
    }

    vx_size size() const
    {
        vx_size v;
        query(VX_MATRIX_SIZE, v);
        return v;
    }

#ifdef VX_VERSION_1_1
    vx_coordinates2d_t origin() const
    {
        vx_coordinates2d_t v;
        query(VX_MATRIX_ORIGIN, v);
        return v;
    }

    vx_enum pattern() const
    {
        vx_enum v;
        query(VX_MATRIX_PATTERN, v);
        return v;
    }
#endif // VX_VERSION_1_1

    void copyTo(void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): output pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyMatrix(ref, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxReadMatrix(ref, data));
#endif
    }

    void copyFrom(const void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): input pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyMatrix(ref, const_cast<void*>(data), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxWriteMatrix(ref, data));
#endif
    }

    void copy(void* data, vx_enum usage, vx_enum memType = VX_MEMORY_TYPE_HOST)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): data pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyMatrix(ref, data, usage, memType));
#else
        if (usage == VX_READ_ONLY)
            IVX_CHECK_STATUS(vxReadMatrix(ref, data));
        else if (usage == VX_WRITE_ONLY)
            IVX_CHECK_STATUS(vxWriteMatrix(ref, data));
        else
            throw WrapperError(std::string(__func__) + "(): unknown copy direction");
        (void)memType;
#endif
    }

    template<typename T> void copyTo(std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, dataType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (data.size()*sizeof(T) != size())
        {
            if (data.size() == 0)
                data.resize(size()/sizeof(T));
            else
                throw WrapperError(std::string(__func__) + "(): destination size is wrong");
        }
        copyTo(&data[0]);
    }

    template<typename T> void copyFrom(const std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, dataType()))
            throw WrapperError(std::string(__func__) + "(): source type is wrong");
        if (data.size()*sizeof(T) != size()) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        copyFrom(&data[0]);
    }

#ifdef IVX_USE_OPENCV
    void copyTo(cv::Mat& m)
    {
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (((vx_size)(m.rows) != rows() || (vx_size)(m.cols) != columns()) && !m.empty())
            throw WrapperError(std::string(__func__) + "(): destination size is wrong");

        if (m.isContinuous() && (vx_size)(m.rows) == rows() && (vx_size)(m.cols) == columns())
        {
            copyTo(m.ptr());
        }
        else
        {
            cv::Mat tmp((int)rows(), (int)columns(), enumToCVType(dataType()));
            copyTo(tmp.ptr());
            if (m.empty())
                m = tmp;
            else
                tmp.copyTo(m);
        }
    }

    void copyFrom(const cv::Mat& m)
    {
        if ((vx_size)(m.rows) != rows() || (vx_size)(m.cols) != columns()) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): source type is wrong");
        copyFrom(m.isContinuous() ? m.ptr() : m.clone().ptr());
    }
#endif //IVX_USE_OPENCV
};

/*
* LUT
*/
class LUT : public RefWrapper<vx_lut>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(LUT);

#ifdef VX_VERSION_1_1
    static LUT create(vx_context context, vx_enum dataType = VX_TYPE_UINT8, vx_size count = 256)
    {
#else
    static LUT create(vx_context context)
    {
        vx_enum dataType = VX_TYPE_UINT8;
        vx_size count = 256;
#endif
        return LUT(vxCreateLUT(context, dataType, count));
    }

#ifndef VX_VERSION_1_1
    static const vx_enum VX_MEMORY_TYPE_HOST = VX_IMPORT_TYPE_HOST;
#endif

    template<typename T>
    void query(vx_enum att, T& value) const
    {
        IVX_CHECK_STATUS(vxQueryLUT(ref, att, &value, sizeof(value)));
    }

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_LUT_TYPE = VX_LUT_ATTRIBUTE_TYPE,
        VX_LUT_COUNT = VX_LUT_ATTRIBUTE_COUNT,
        VX_LUT_SIZE = VX_LUT_ATTRIBUTE_SIZE;
#endif

    vx_enum dataType() const
    {
        vx_enum v;
        query(VX_LUT_TYPE, v);
        return v;
    }

    vx_size count() const
    {
        vx_size v;
        query(VX_LUT_COUNT, v);
        return v;
    }

    vx_size size() const
    {
        vx_size v;
        query(VX_LUT_SIZE, v);
        return v;
    }

#ifdef VX_VERSION_1_1
    vx_uint32 offset() const
    {
        vx_enum v;
        query(VX_LUT_OFFSET, v);
        return v;
    }
#endif // VX_VERSION_1_1

    void copyTo(void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): output pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyLUT(ref, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxAccessLUT(ref, &data, VX_READ_ONLY));
        IVX_CHECK_STATUS(vxCommitLUT(ref, data));
#endif
    }

    void copyFrom(const void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): input pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyLUT(ref, const_cast<void*>(data), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxAccessLUT(ref, const_cast<void**>(&data), VX_WRITE_ONLY));
        IVX_CHECK_STATUS(vxCommitLUT(ref, data));
#endif
    }

    void copy(void* data, vx_enum usage, vx_enum memType = VX_MEMORY_TYPE_HOST)
    {
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyLUT(ref, data, usage, memType));
#else
        IVX_CHECK_STATUS(vxAccessLUT(ref, const_cast<void**>(&data), usage));
        IVX_CHECK_STATUS(vxCommitLUT(ref, data));
        (void)memType;
#endif
    }

    template<typename T> void copyTo(std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, dataType()))
            throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (data.size() != count())
        {
            if (data.size() == 0)
                data.resize(count());
            else
                throw WrapperError(std::string(__func__) + "(): destination size is wrong");
        }
        copyTo(&data[0]);
    }

    template<typename T> void copyFrom(const std::vector<T>& data)
    {
        if (!areTypesCompatible(TypeToEnum<T>::value, dataType()))
            throw WrapperError(std::string(__func__) + "(): source type is wrong");
        if (data.size() != count()) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        copyFrom(&data[0]);
    }

#ifdef IVX_USE_OPENCV
    void copyTo(cv::Mat& m)
    {
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (!(
            ((vx_size)(m.rows) == count() && m.cols == 1) ||
            ((vx_size)(m.cols) == count() && m.rows == 1)
            ) && !m.empty())
            throw WrapperError(std::string(__func__) + "(): destination size is wrong");

        if (m.isContinuous() && (vx_size)(m.total()) == count())
        {
            copyTo(m.ptr());
        }
        else
        {
            cv::Mat tmp(1, (int)count(), enumToCVType(dataType()));
            copyTo(tmp.ptr());
            if (m.empty())
                m = tmp;
            else
                tmp.copyTo(m);
        }
    }

    void copyFrom(const cv::Mat& m)
    {
        if (!(
                ((vx_size)(m.rows) == count() && m.cols == 1) ||
                ((vx_size)(m.cols) == count() && m.rows == 1)
           )) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): source type is wrong");
        copyFrom(m.isContinuous() ? m.ptr() : m.clone().ptr());
    }
#endif //IVX_USE_OPENCV
};

/*
 * Pyramid
 */
class Pyramid : public RefWrapper<vx_pyramid>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Pyramid)

    static Pyramid create(vx_context context, vx_size levels, vx_float32 scale,
                          vx_uint32 width, vx_uint32 height, vx_df_image format)
    {return Pyramid(vxCreatePyramid(context, levels, scale, width, height, format));}

    static Pyramid createVirtual(vx_graph graph, vx_size levels, vx_float32 scale,
                                 vx_uint32 width, vx_uint32 height, vx_df_image format)
    {return Pyramid(vxCreateVirtualPyramid(graph, levels, scale, width, height, format));}

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_PYRAMID_LEVELS = VX_PYRAMID_ATTRIBUTE_LEVELS,
        VX_PYRAMID_SCALE  = VX_PYRAMID_ATTRIBUTE_SCALE,
        VX_PYRAMID_WIDTH  = VX_PYRAMID_ATTRIBUTE_WIDTH,
        VX_PYRAMID_HEIGHT = VX_PYRAMID_ATTRIBUTE_HEIGHT,
        VX_PYRAMID_FORMAT = VX_PYRAMID_ATTRIBUTE_FORMAT;
#endif

    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS( vxQueryPyramid(ref, att, &value, sizeof(value)) ); }

    vx_size levels() const
    {
        vx_size l;
        query(VX_PYRAMID_LEVELS, l);
        return l;
    }

    vx_float32 scale() const
    {
        vx_float32 s;
        query(VX_PYRAMID_SCALE, s);
        return s;
    }

    vx_uint32 width() const
    {
        vx_uint32 v;
        query(VX_PYRAMID_WIDTH, v);
        return v;
    }

    vx_uint32 height() const
    {
        vx_uint32 v;
        query(VX_PYRAMID_HEIGHT, v);
        return v;
    }

    vx_df_image format() const
    {
        vx_df_image f;
        query(VX_PYRAMID_FORMAT, f);
        return f;
    }

    Image getLevel(vx_uint32 index)
    { return Image(vxGetPyramidLevel(ref, index)); }
};

/*
* Distribution
*/
class Distribution : public RefWrapper<vx_distribution>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Distribution);

    static Distribution create(vx_context context, vx_size numBins, vx_int32 offset, vx_uint32 range)
    {
        return Distribution(vxCreateDistribution(context, numBins, offset, range));
    }

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_MEMORY_TYPE_HOST = VX_IMPORT_TYPE_HOST,
        VX_DISTRIBUTION_DIMENSIONS = VX_DISTRIBUTION_ATTRIBUTE_DIMENSIONS,
        VX_DISTRIBUTION_OFFSET = VX_DISTRIBUTION_ATTRIBUTE_OFFSET,
        VX_DISTRIBUTION_RANGE = VX_DISTRIBUTION_ATTRIBUTE_RANGE,
        VX_DISTRIBUTION_BINS = VX_DISTRIBUTION_ATTRIBUTE_BINS,
        VX_DISTRIBUTION_WINDOW = VX_DISTRIBUTION_ATTRIBUTE_WINDOW,
        VX_DISTRIBUTION_SIZE = VX_DISTRIBUTION_ATTRIBUTE_SIZE;
#endif

    template<typename T>
    void query(vx_enum att, T& value) const
    {
        IVX_CHECK_STATUS(vxQueryDistribution(ref, att, &value, sizeof(value)));
    }

    vx_size dimensions() const
    {
        vx_size v;
        query(VX_DISTRIBUTION_DIMENSIONS, v);
        return v;
    }

    vx_int32 offset() const
    {
        vx_int32 v;
        query(VX_DISTRIBUTION_OFFSET, v);
        return v;
    }

    vx_uint32 range() const
    {
        vx_uint32 v;
        query(VX_DISTRIBUTION_RANGE, v);
        return v;
    }

    vx_size bins() const
    {
        vx_size v;
        query(VX_DISTRIBUTION_BINS, v);
        return v;
    }

    vx_uint32 window() const
    {
        vx_uint32 v;
        query(VX_DISTRIBUTION_WINDOW, v);
        return v;
    }

    vx_size size() const
    {
        vx_size v;
        query(VX_DISTRIBUTION_SIZE, v);
        return v;
    }

    vx_size dataType() const
    {
        return VX_TYPE_UINT32;
    }

    void copyTo(void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): output pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyDistribution(ref, data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxAccessDistribution(ref, &data, VX_READ_ONLY));
        IVX_CHECK_STATUS(vxCommitDistribution(ref, data));
#endif
    }

    void copyFrom(const void* data)
    {
        if (!data) throw WrapperError(std::string(__func__) + "(): input pointer is 0");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyDistribution(ref, const_cast<void*>(data), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS(vxAccessDistribution(ref, const_cast<void**>(&data), VX_WRITE_ONLY));
        IVX_CHECK_STATUS(vxCommitDistribution(ref, data));
#endif
    }

    void copy(void* data, vx_enum usage, vx_enum memType = VX_MEMORY_TYPE_HOST)
    {
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyDistribution(ref, data, usage, memType));
#else
        IVX_CHECK_STATUS(vxAccessDistribution(ref, const_cast<void**>(&data), usage));
        IVX_CHECK_STATUS(vxCommitDistribution(ref, data));
        (void)memType;
#endif
    }

    template<typename T> void copyTo(std::vector<T>& data)
    {
        if (TypeToEnum<T>::value != dataType()) throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (data.size() != bins())
        {
            if (data.size() == 0)
                data.resize(bins());
            else
                throw WrapperError(std::string(__func__) + "(): destination size is wrong");
        }
        copyTo(&data[0]);
    }

    template<typename T> void copyFrom(const std::vector<T>& data)
    {
        if (TypeToEnum<T>::value != dataType()) throw WrapperError(std::string(__func__) + "(): source type is wrong");
        if (data.size() != bins()) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        copyFrom(&data[0]);
    }

#ifdef IVX_USE_OPENCV
    void copyTo(cv::Mat& m)
    {
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): destination type is wrong");
        if (!(
            ((vx_size)(m.rows) == bins() && m.cols == 1) ||
            ((vx_size)(m.cols) == bins() && m.rows == 1)
            ) && !m.empty())
            throw WrapperError(std::string(__func__) + "(): destination size is wrong");

        if (m.isContinuous() && (vx_size)(m.total()) == bins())
        {
            copyTo(m.ptr());
        }
        else
        {
            cv::Mat tmp(1, (int)bins(), enumToCVType(dataType()));
            copyTo(tmp.ptr());
            if (m.empty())
                m = tmp;
            else
                tmp.copyTo(m);
        }
    }

    void copyFrom(const cv::Mat& m)
    {
        if (!(
            ((vx_size)(m.rows) == bins() && m.cols == 1) ||
            ((vx_size)(m.cols) == bins() && m.rows == 1)
            )) throw WrapperError(std::string(__func__) + "(): source size is wrong");
        if (m.type() != enumToCVType(dataType())) throw WrapperError(std::string(__func__) + "(): source type is wrong");
        copyFrom(m.isContinuous() ? m.ptr() : m.clone().ptr());
    }
#endif //IVX_USE_OPENCV
};

/*
* Remap
*/
class Remap : public RefWrapper<vx_remap>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Remap);

    static Remap create(vx_context context, vx_uint32 src_width, vx_uint32 src_height, vx_uint32 dst_width, vx_uint32 dst_height)
    {
        return Remap(vxCreateRemap(context, src_width, src_height, dst_width, dst_height));
    }

#ifndef VX_VERSION_1_1
    static const vx_enum
        VX_REMAP_SOURCE_WIDTH = VX_REMAP_ATTRIBUTE_SOURCE_WIDTH,
        VX_REMAP_SOURCE_HEIGHT = VX_REMAP_ATTRIBUTE_SOURCE_HEIGHT,
        VX_REMAP_DESTINATION_WIDTH = VX_REMAP_ATTRIBUTE_DESTINATION_WIDTH,
        VX_REMAP_DESTINATION_HEIGHT = VX_REMAP_ATTRIBUTE_DESTINATION_HEIGHT;
#endif

    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS(vxQueryRemap(ref, att, &value, sizeof(value))); }

    vx_uint32 srcWidth() const
    {
        vx_uint32 v;
        query(VX_REMAP_SOURCE_WIDTH, v);
        return v;
    }

    vx_uint32 srcHeight() const
    {
        vx_uint32 v;
        query(VX_REMAP_SOURCE_HEIGHT, v);
        return v;
    }

    vx_uint32 dstWidth() const
    {
        vx_uint32 v;
        query(VX_REMAP_DESTINATION_WIDTH, v);
        return v;
    }

    vx_uint32 dstHeight() const
    {
        vx_uint32 v;
        query(VX_REMAP_DESTINATION_HEIGHT, v);
        return v;
    }

    vx_uint32 srcCoordType() const
    { return VX_TYPE_FLOAT32; }

    vx_uint32 dstCoordType() const
    { return VX_TYPE_UINT32; }

    void setMapping(vx_uint32 dst_x, vx_uint32 dst_y, vx_float32 src_x, vx_float32 src_y)
    { IVX_CHECK_STATUS(vxSetRemapPoint(ref, dst_x, dst_y, src_x, src_y)); }

    void getMapping(vx_uint32 dst_x, vx_uint32 dst_y, vx_float32 &src_x, vx_float32 &src_y) const
    { IVX_CHECK_STATUS(vxGetRemapPoint(ref, dst_x, dst_y, &src_x, &src_y)); }

    void setMappings(vx_float32* map_x, size_t map_x_stride, vx_float32* map_y, size_t map_y_stride)
    {
        for (vx_uint32 y = 0; y < dstHeight(); y++)
        {
            const vx_float32* map_x_line = (vx_float32*)((char*)map_x + y*map_x_stride);
            const vx_float32* map_y_line = (vx_float32*)((char*)map_y + y*map_y_stride);
            for (vx_uint32 x = 0; x < dstWidth(); x++)
                setMapping(x, y, map_x_line[x], map_y_line[x]);
        }
    }

    void setMappings(vx_float32* map, size_t map_stride)
    {
        for (vx_uint32 y = 0; y < dstHeight(); y++)
        {
            const vx_float32* map_line = (vx_float32*)((char*)map + y*map_stride);
            for (vx_uint32 x = 0; x < 2*dstWidth(); x+=2)
                setMapping(x, y, map_line[x], map_line[x+1]);
        }
    }

#ifdef IVX_USE_OPENCV
    void setMappings(const cv::Mat& map_x, const cv::Mat& map_y)
    {
        if (map_x.type() != enumToCVType(srcCoordType()) || map_y.type() != enumToCVType(srcCoordType()))
            throw WrapperError(std::string(__func__) + "(): mapping type is wrong");
        if ((vx_uint32)(map_x.rows) != dstHeight() || (vx_uint32)(map_x.cols) != dstWidth())
            throw WrapperError(std::string(__func__) + "(): x mapping size is wrong");
        if ((vx_uint32)(map_y.rows) != dstHeight() || (vx_uint32)(map_y.cols) != dstWidth())
            throw WrapperError(std::string(__func__) + "(): y mapping size is wrong");

        for (vx_uint32 y = 0; y < dstHeight(); y++)
        {
            const vx_float32* map_x_line = map_x.ptr<vx_float32>(y);
            const vx_float32* map_y_line = map_y.ptr<vx_float32>(y);
            for (vx_uint32 x = 0; x < dstWidth(); x++)
                setMapping(x, y, map_x_line[x], map_y_line[x]);
        }
    }

    void setMappings(const cv::Mat& map)
    {
        if (map.depth() != CV_MAT_DEPTH(enumToCVType(srcCoordType())) || map.channels() != 2)
            throw WrapperError(std::string(__func__) + "(): mapping type is wrong");
        if ((vx_uint32)(map.rows) != dstHeight() || (vx_uint32)(map.cols) != dstWidth())
            throw WrapperError(std::string(__func__) + "(): x mapping size is wrong");

        for (vx_uint32 y = 0; y < dstHeight(); y++)
        {
            const vx_float32* map_line = map.ptr<vx_float32>(y);
            for (vx_uint32 x = 0; x < 2*dstWidth(); x+=2)
                setMapping(x, y, map_line[x], map_line[x+1]);
        }
    }

    void getMappings(cv::Mat& map_x, cv::Mat& map_y) const
    {
        if (map_x.type() != enumToCVType(srcCoordType()) || map_y.type() != enumToCVType(srcCoordType()))
            throw WrapperError(std::string(__func__) + "(): mapping type is wrong");
        if (((vx_uint32)(map_x.rows) != dstHeight() || (vx_uint32)(map_x.cols) != dstWidth()) && !map_x.empty())
            throw WrapperError(std::string(__func__) + "(): x mapping size is wrong");
        if (((vx_uint32)(map_y.rows) != dstHeight() || (vx_uint32)(map_y.cols) != dstWidth()) && !map_y.empty())
            throw WrapperError(std::string(__func__) + "(): y mapping size is wrong");

        if (map_x.empty())
            map_x = cv::Mat((int)dstHeight(), (int)dstWidth(), enumToCVType(srcCoordType()));
        if (map_y.empty())
            map_y = cv::Mat((int)dstHeight(), (int)dstWidth(), enumToCVType(srcCoordType()));

        for (vx_uint32 y = 0; y < dstHeight(); y++)
        {
            vx_float32* map_x_line = map_x.ptr<vx_float32>(y);
            vx_float32* map_y_line = map_y.ptr<vx_float32>(y);
            for (vx_uint32 x = 0; x < dstWidth(); x++)
                getMapping(x, y, map_x_line[x], map_y_line[x]);
        }
    }

    void getMappings(cv::Mat& map) const
    {
        if (map.depth() != CV_MAT_DEPTH(enumToCVType(srcCoordType())) || map.channels() != 2)
            throw WrapperError(std::string(__func__) + "(): mapping type is wrong");
        if (((vx_uint32)(map.rows) != dstHeight() || (vx_uint32)(map.cols) != dstWidth()) && !map.empty())
            throw WrapperError(std::string(__func__) + "(): x mapping size is wrong");

        if (map.empty())
            map = cv::Mat((int)dstHeight(), (int)dstWidth(), CV_MAKETYPE(CV_MAT_DEPTH(enumToCVType(srcCoordType())),2));

        for (vx_uint32 y = 0; y < dstHeight(); y++)
        {
            vx_float32* map_line = map.ptr<vx_float32>(y);
            for (vx_uint32 x = 0; x < 2*dstWidth(); x+=2)
                getMapping(x, y, map_line[x], map_line[x+1]);
        }
    }
#endif //IVX_USE_OPENCV
};

/// Standard nodes
namespace nodes {

/// Creates a Gaussian Filter 3x3 Node (vxGaussian3x3Node)
inline Node gaussian3x3(vx_graph graph, vx_image inImg, vx_image outImg)
{ return Node(vxGaussian3x3Node(graph, inImg, outImg)); }

} // namespace nodes

} // namespace ivx

// restore warnings
#if defined(_MSC_VER)
    #pragma warning(pop)
#elif defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif // compiler macro

#endif //IVX_HPP
