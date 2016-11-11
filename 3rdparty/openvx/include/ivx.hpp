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

#ifndef IVX_USE_CXX98
    #include <type_traits>
#else
    namespace ivx
    {
    template<typename, typename> struct is_same { static const bool value = false; };
    template<typename T> struct is_same<T, T>   { static const bool value = true; };
    }
#endif

#ifdef IVX_USE_OPENCV
    #include "opencv2/core.hpp"
#endif

namespace ivx
{

/*
* RuntimeError - OpenVX runtime errors exception class
*/
class RuntimeError : public std::runtime_error
{
public:
    explicit RuntimeError(vx_status status, const std::string& msg = "")
        : runtime_error(msg), _status(status)
    {}

    vx_status status() const
    { return _status; }

private:
    vx_status   _status;
};

/*
* WrapperError - wrappers logic errors exception class
*/
class WrapperError : public std::logic_error
{
public:
    explicit WrapperError(const std::string& msg) : logic_error(msg)
    {}
};

inline void checkVxStatus(vx_status status, const std::string& func, const std::string& msg)
{
    if(status != VX_SUCCESS) throw RuntimeError( status, func + "() : " + msg );
}


#define IVX_CHECK_STATUS(code) checkVxStatus(code, __func__, #code)


/*
* EnumToType - enum to type compile-time converter (TODO: add more types)
*/
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
#ifndef IVX_USE_CXX98
template <vx_enum E> using EnumToType_t = typename EnumToType<E>::type;
#endif

vx_size enumToTypeSize(vx_enum type)
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
    default: throw WrapperError(std::string(__func__) + ": unsupported type enum");
    }
}

/*
* TypeToEnum - type to enum compile-time converter (TODO: add more types)
*/
template<typename T> struct TypeToEnum {};
template<> struct TypeToEnum<vx_char>     { static const vx_enum value = VX_TYPE_CHAR; };
template<> struct TypeToEnum<vx_int8>     { static const vx_enum value = VX_TYPE_INT8; };
template<> struct TypeToEnum<vx_uint8>    { static const vx_enum value = VX_TYPE_UINT8; };
template<> struct TypeToEnum<vx_int16>    { static const vx_enum value = VX_TYPE_INT16; };
template<> struct TypeToEnum<vx_uint16>   { static const vx_enum value = VX_TYPE_UINT16; };
template<> struct TypeToEnum<vx_int32>    { static const vx_enum value = VX_TYPE_INT32; };
template<> struct TypeToEnum<vx_uint32>   { static const vx_enum value = VX_TYPE_UINT32; };
template<> struct TypeToEnum<vx_int64>    { static const vx_enum value = VX_TYPE_INT64; };
template<> struct TypeToEnum<vx_uint64>   { static const vx_enum value = VX_TYPE_UINT64; };
template<> struct TypeToEnum<vx_float32>  { static const vx_enum value = VX_TYPE_FLOAT32; };
template<> struct TypeToEnum<vx_float64>  { static const vx_enum value = VX_TYPE_FLOAT64; };
template<> struct TypeToEnum<vx_bool>     { static const vx_enum value = VX_TYPE_BOOL; };
// the commented types are aliases (of integral tyes) and have conflicts with the types above
//template<> struct TypeToEnum<vx_enum>     { static const vx_enum val = VX_TYPE_ENUM; };
//template<> struct TypeToEnum<vx_size>     { static const vx_enum val = VX_TYPE_SIZE; };
//template<> struct TypeToEnum<vx_df_image> { static const vx_enum val = VX_TYPE_DF_IMAGE; };

/*
* RefTypeTraits - provides info for vx_reference extending types
*/
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

/*
* Casting to vx_reference with compile-time check
*/
#ifdef IVX_USE_CXX98

// takes 'vx_reference' itself and RefWrapper<T> via 'operator vx_reference()'
vx_reference castToReference(vx_reference ref)
{ return ref; }

// takes vx_reference extensions that have RefTypeTraits<T> specializations
template<typename T>
vx_reference castToReference(const T& ref, typename RefTypeTraits<T>::vxType dummy = 0)
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

template<typename T>
vx_reference castToReference(const T& obj)
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

#define IVX_CHECK_REF(code) checkVxRef(castToReference(code), __func__, #code)

/*
* RefWrapper - base class for referenced objects wrappers
*/

#ifdef IVX_USE_EXTERNAL_REFCOUNT

template <typename T> class RefWrapper
{
public:
    typedef T vxType;
    static const vx_enum vxTypeEnum = RefTypeTraits <T>::vxTypeEnum;

    RefWrapper() : ref(0), refcount(0)
    {}

    explicit RefWrapper(T r, bool retainRef = false) : ref(0), refcount(0)
    { reset(r, retainRef); }

    RefWrapper(const RefWrapper& r) : ref(r.ref), refcount(r.refcount)
    { addRef(); }

#ifndef IVX_USE_CXX98
    RefWrapper(RefWrapper&& rw) noexcept : RefWrapper()
    {
        using std::swap;
        swap(ref, rw.ref);
        swap(refcount, rw.refcount);
    }
#endif

    operator T() const
    { return ref; }

    operator vx_reference() const
    { return castToReference(ref); }

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

    void reset()
    { release(); }

    T detach()
    {
        T tmp = ref;
        ref = 0;
        release();
        return tmp;
    }

    RefWrapper& operator=(RefWrapper r)
    {
        using std::swap;
        swap(ref, r.ref);
        swap(refcount, r.refcount);
        return *this;
    }

    bool operator !() const
    { return ref == 0; }

#ifndef IVX_USE_CXX98
    explicit operator bool() const
    { return ref != 0; }
#endif

#ifdef IVX_USE_CXX98
    template<typename C>
    C get() const
    {
        typedef int static_assert_context[is_same<C, Context>::value ? 1 : -1];
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return C(c, true);
    }
#else
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

template <typename T> class RefWrapper
{
public:
    typedef T vxType;
    static const vx_enum vxTypeEnum = RefTypeTraits <T>::vxTypeEnum;

    RefWrapper() : ref(0)
    {}

    explicit RefWrapper(T r, bool retainRef = false) : ref(0)
    { reset(r, retainRef); }

    RefWrapper(const RefWrapper& r) : ref(r.ref)
    { addRef(); }

#ifndef IVX_USE_CXX98
    RefWrapper(RefWrapper&& rw) noexcept : RefWrapper()
    {
        using std::swap;
        swap(ref, rw.ref);
    }
#endif

    operator T() const
    { return ref; }

    operator vx_reference() const
    { return castToReference(ref); }

#ifdef IVX_USE_CXX98
    template<typename C>
    C get() const
    {
        typedef int static_assert_context[is_same<C, Context>::value ? 1 : -1];
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return C(c, true);
    }
#else
    template<typename C = Context, typename = typename std::enable_if<std::is_same<C, Context>::value>::type>
    C getContext() const
    {
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return C(c, true);
    }
#endif // IVX_USE_CXX98

    void reset(T r, bool retainRef = false)
    {
        release();
        ref = r;
        if (retainRef) addRef();
        checkRef();
    }

    void reset()
    { release(); }

    T detach()
    {
        T tmp = ref;
        ref = 0;
        return tmp;
    }

    RefWrapper& operator=(RefWrapper r)
    {
        using std::swap;
        swap(ref, r.ref);
        return *this;
    }

    bool operator !() const
    { return ref == 0; }

#ifndef IVX_USE_CXX98
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

/*
* Context
*/
class Context : public RefWrapper<vx_context>
{
public:

    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Context)

    static Context create()
    { return Context(vxCreateContext()); }

    template <typename T>
    static Context getFrom(const T& ref)
    {
        vx_context c = vxGetContext(castToReference(ref));
        // vxGetContext doesn't increment ref count, let do it in wrapper c-tor
        return Context(c, true);
    }

    void loadKernels(const std::string& module)
    { IVX_CHECK_STATUS( vxLoadKernels(ref, module.c_str()) ); }
};

/*
* Graph
*/
class Graph : public RefWrapper<vx_graph>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Graph);

    static Graph create(vx_context c)
    { return Graph(vxCreateGraph(c)); }

    void verify()
    { IVX_CHECK_STATUS( vxVerifyGraph(ref) ); }

    void process()
    { IVX_CHECK_STATUS( vxProcessGraph(ref) ); }

    void schedule()
    { IVX_CHECK_STATUS(vxScheduleGraph(ref) ); }

    void wait()
    { IVX_CHECK_STATUS(vxWaitGraph(ref)); }
};

/*
* Kernel
*/
class Kernel : public RefWrapper<vx_kernel>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Kernel);

    static Kernel getByEnum(vx_context c, vx_enum kernelID)
    { return Kernel(vxGetKernelByEnum(c, kernelID)); }

    static Kernel getByName(vx_context c, const std::string& name)
    { return Kernel(vxGetKernelByName(c, name.c_str())); }
};

/*
* Node
*/
#ifdef IVX_USE_CXX98

class Node : public RefWrapper<vx_node>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Node);

    static Node create(vx_graph g, vx_kernel k)
    { return Node(vxCreateGenericNode(g, k)); }

    static Node create(vx_graph graph, vx_kernel kernel, const std::vector<vx_reference>& params)
    {
        Node node = Node::create(graph, kernel);
        vx_uint32 i = 0;
        for (std::vector<vx_reference>::const_iterator p = params.begin(); p != params.end(); ++p)
            node.setParameterByIndex(i++, *p);
        return node;
    }

    static Node create(vx_graph graph,  vx_enum kernelID, const std::vector<vx_reference>& params)
    { return Node::create(graph, Kernel::getByEnum(Context::getFrom(graph), kernelID), params); }

    template<typename T0>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

    template<typename T0, typename T1>
    static Node create(vx_graph g, vx_enum kernelID,
                       const T0& arg0, const T1& arg1)
    {
        std::vector<vx_reference> params;
        params.push_back(castToReference(arg0));
        params.push_back(castToReference(arg1));
        return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), params);
    }

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

    void setParameterByIndex(vx_uint32 index, vx_reference value)
    { IVX_CHECK_STATUS(vxSetParameterByIndex(ref, index, value)); }
};

#else // not IVX_USE_CXX98

class Node : public RefWrapper<vx_node>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Node);

    static Node create(vx_graph g, vx_kernel k)
    { return Node(vxCreateGenericNode(g, k)); }

    static Node create(vx_graph graph, vx_kernel kernel, const std::vector<vx_reference>& params)
    {
        Node node = Node::create(graph, kernel);
        vx_uint32 i = 0;
        for (const auto& p : params)
            node.setParameterByIndex(i++, p);
        return node;
    }

    static Node create(vx_graph graph,  vx_enum kernelID, const std::vector<vx_reference>& params)
    { return Node::create(graph, Kernel::getByEnum(Context::getFrom(graph), kernelID), params); }

    template<typename...Ts>
    static Node create(vx_graph g, vx_enum kernelID, const Ts&...args)
    { return create(g, Kernel::getByEnum(Context::getFrom(g), kernelID), { castToReference(args)... }); }


    void setParameterByIndex(vx_uint32 index, vx_reference value)
    { IVX_CHECK_STATUS(vxSetParameterByIndex(ref, index, value)); }
};

#endif // IVX_USE_CXX98

/*
* Image
*/
class Image : public RefWrapper<vx_image>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Image);

    static Image create(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image format)
    { return Image(vxCreateImage(context, width, height, format)); }

    static Image createVirtual(vx_graph graph, vx_uint32 width = 0, vx_uint32 height = 0, vx_df_image format = VX_DF_IMAGE_VIRT)
    { return Image(vxCreateVirtualImage(graph, width, height, format)); }

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

    static vx_imagepatch_addressing_t createAddressing()
    { vx_imagepatch_addressing_t ipa = VX_IMAGEPATCH_ADDR_INIT; return ipa; }

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

    vx_imagepatch_addressing_t createAddressing(vx_uint32 planeIdx)
    { return createAddressing(planeIdx, getValidRegion()); }

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

#ifdef VX_VERSION_1_1
    void swapHandle(const std::vector<void*>& newPtrs, std::vector<void*>& prevPtrs)
    {
        vx_size num = planes();
        if(num == 0)
            throw WrapperError(std::string(__func__)+"(): unexpected planes number");
        if (newPtrs.size() < num)
            throw WrapperError(std::string(__func__)+"(): too few input pointers");
        if (prevPtrs.empty()) prevPtrs.resize(num, 0);
        else if (prevPtrs.size() < num)
            throw WrapperError(std::string(__func__)+"(): too few output pointers");
        IVX_CHECK_STATUS( vxSwapImageHandle(ref, &newPtrs[0], &prevPtrs[0], num) );
    }

    void swapHandle(const std::vector<void*>& newPtrs)
    {
        vx_size num = planes();
        if(num == 0)
            throw WrapperError(std::string(__func__)+"(): unexpected planes number");
        if (newPtrs.size() < num)
            throw WrapperError(std::string(__func__)+"(): too few input pointers");
        IVX_CHECK_STATUS( vxSwapImageHandle(ref, &newPtrs[0], 0, num) );
    }

    void swapHandle()
    { IVX_CHECK_STATUS( vxSwapImageHandle(ref, 0, 0, 0) ); }

    Image createFromChannel(vx_enum channel)
    { return Image(vxCreateImageFromChannel(ref, channel)); }
#endif // VX_VERSION_1_1

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

    vx_uint32 width() const
    {
        vx_uint32 v;
        query(VX_IMAGE_WIDTH, v);
        return v;
    }

    vx_uint32 height() const
    {
        vx_uint32 v;
        query(VX_IMAGE_HEIGHT, v);
        return v;
    }

    vx_df_image format() const
    {
        vx_df_image v;
        query(VX_IMAGE_FORMAT, v);
        return v;
    }

    vx_size planes() const
    {
        vx_size v;
        query(VX_IMAGE_PLANES, v);
        return v;
    }

    vx_enum space() const
    {
        vx_enum v;
        query(VX_IMAGE_SPACE, v);
        return v;
    }

    vx_enum range() const
    {
        vx_enum v;
        query(VX_IMAGE_RANGE, v);
        return v;
    }

    vx_size size() const
    {
        vx_size v;
        query(VX_IMAGE_SIZE, v);
        return v;
    }

#ifdef VX_VERSION_1_1
    vx_memory_type_e memType() const
    {
        vx_memory_type_e v;
        query(VX_IMAGE_MEMORY_TYPE, v);
        return v;
    }
#endif // VX_VERSION_1_1

    vx_rectangle_t getValidRegion() const
    {
        vx_rectangle_t rect;
        IVX_CHECK_STATUS( vxGetValidRegionImage(ref, &rect) );
        return rect;
    }

    vx_size computePatchSize(vx_uint32 planeIdx)
    { return computePatchSize(planeIdx, getValidRegion()); }

    vx_size computePatchSize(vx_uint32 planeIdx, const vx_rectangle_t& rect)
    {
        vx_size bytes = vxComputeImagePatchSize(ref, &rect, planeIdx);
        if (bytes == 0) throw WrapperError(std::string(__func__)+"(): vxComputeImagePatchSize returned 0");
        return bytes;
    }

#ifdef VX_VERSION_1_1
    void setValidRectangle(const vx_rectangle_t& rect)
    { IVX_CHECK_STATUS( vxSetImageValidRectangle(ref, &rect) ); }
#endif // VX_VERSION_1_1

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

    void copy( vx_uint32 planeIdx, vx_rectangle_t rect,
               const vx_imagepatch_addressing_t& addr, void* data,
               vx_enum usage, vx_enum memType = VX_MEMORY_TYPE_HOST )
    {
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyImagePatch(ref, &rect, planeIdx, &addr, (void*)data, usage, memType));
#else
        (void)memType;
        vx_imagepatch_addressing_t* a = const_cast<vx_imagepatch_addressing_t*>(&addr);
        IVX_CHECK_STATUS(vxAccessImagePatch(ref, &rect, planeIdx, a, &data, usage));
        IVX_CHECK_STATUS(vxCommitImagePatch(ref, &rect, planeIdx, a, data));
#endif
    }

#ifdef IVX_USE_OPENCV
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
        default: return CV_USRTYPE1;
        }
    }

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

    void copyTo(vx_uint32 planeIdx, cv::Mat& m)
    {
        createMatForPlane(m, planeIdx);
        copyTo(planeIdx, createAddressing((vx_uint32)m.cols, (vx_uint32)m.rows, (vx_int32)m.elemSize(), (vx_int32)m.step), m.ptr());
    }

    void copyFrom(vx_uint32 planeIdx, const cv::Mat& m)
    {
        // TODO: add sizes consistency checks
        //vx_rectangle_t r = getValidRegion();
        copyFrom(planeIdx, createAddressing((vx_uint32)m.cols, (vx_uint32)m.rows, (vx_int32)m.elemSize(), (vx_int32)m.step), m.ptr());
    }
#endif //IVX_USE_OPENCV

    struct Patch;
};

struct Image::Patch
{
public:
    const vx_imagepatch_addressing_t& addr() const
    { return _addr;}

    void* data() const
    { return _data; }

#ifdef VX_VERSION_1_1
    vx_memory_type_e memType() const
    { return _memType; }

    vx_map_id mapId() const
    { return _mapId; }
#else
    const vx_rectangle_t& rect() const
    { return _rect; }

    vx_uint32 planeIdx() const
    { return _planeIdx; }
#endif // VX_VERSION_1_1

    vx_image img() const
    { return _img; }

    bool isMapped() const
    { return _img != 0; }

#ifdef IVX_USE_OPENCV
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
    Patch() : _addr(createAddressing()), _data(0), _img(0)
#ifdef VX_VERSION_1_1
       , _memType(VX_MEMORY_TYPE_HOST)
    {}
#else
       , _planeIdx(-1)
    { _rect.start_x = _rect.end_x = _rect.start_y = _rect.end_y = 0u; }
#endif

#ifndef IVX_USE_CXX98
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
        swap(_m, p._m);
    }
#endif

    void map(vx_image img, vx_uint32 planeIdx)
    { map(img, planeIdx, Image(img, true).getValidRegion()); }

    void map(vx_image img, vx_uint32 planeIdx, const vx_rectangle_t& rect, vx_enum usage = VX_READ_ONLY, vx_uint32 flags = 0)
    {
        if (isMapped()) throw WrapperError(std::string(__func__)+"(): already mapped");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxMapImagePatch(img, &rect, planeIdx, &_mapId, &_addr, &_data, usage, _memType, flags) );
#else
        IVX_CHECK_STATUS(vxAccessImagePatch(img, &rect, planeIdx, &_addr, &_data, usage));
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

    void unmap()
    {
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxUnmapImagePatch(_img, _mapId));
#else
        IVX_CHECK_STATUS(vxCommitImagePatch(_img, &_rect, _planeIdx, &_addr, _data));
        _rect.start_x = _rect.end_x = _rect.start_y = _rect.end_y = 0u;
        _planeIdx = -1;

#endif
        _img = 0;
        _data = 0;
#ifdef IVX_USE_OPENCV
        _m.release();
#endif
    }

    ~Patch()
    { try { if (_img) unmap(); } catch(...) {; /*ignore*/} }

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

/*
* Param
*/
class Param : public RefWrapper<vx_parameter>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Param);
    // NYI
};

/*
* Scalar
*/
class Scalar : public RefWrapper<vx_scalar>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Scalar);

    static Scalar create(vx_context c, vx_enum dataType, const void *ptr)
    { return Scalar( vxCreateScalar(c, dataType, ptr) ); }

    template<typename T> static Scalar create(vx_context c, vx_enum dataType, T value)
    { return Scalar( vxCreateScalar(c, dataType, &value) ); }

    template<vx_enum E> static Scalar create(vx_context c, typename EnumToType<E>::type value)
    { return Scalar( vxCreateScalar(c, E, &value) ); }

#ifndef VX_VERSION_1_1
static const vx_enum VX_SCALAR_TYPE = VX_SCALAR_ATTRIBUTE_TYPE;
#endif
    vx_enum type()
    {
        vx_enum val;
        IVX_CHECK_STATUS( vxQueryScalar(ref, VX_SCALAR_TYPE, &val, sizeof(val)) );
        return val;
    }

    template<typename T>
    void getValue(T& val)
    {
        if(TypeToEnum<T>::value != type()) throw WrapperError(std::string(__func__)+"(): incompatible types");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS( vxCopyScalar(ref, &val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) );
#else
        IVX_CHECK_STATUS( vxReadScalarValue(ref, &val) );
#endif
    }

    template<typename T>
    T getValue()
    {
        T val;
        getValue(val);
        return val;
    }


    template<typename T>
    void setValue(T val)
    {
        if (TypeToEnum<T>::value != type()) throw WrapperError(std::string(__func__)+"(): incompatible types");
#ifdef VX_VERSION_1_1
        IVX_CHECK_STATUS(vxCopyScalar(ref, &val, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
#else
        IVX_CHECK_STATUS( vxWriteScalarValue(ref, &val) );
#endif
    }
};

/*
* Threshold
*/
class Threshold : public RefWrapper<vx_threshold>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Threshold);

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


    static Threshold createBinary(vx_context c, vx_enum dataType, vx_int32 val)
    {
        Threshold thr = create(c, VX_THRESHOLD_TYPE_BINARY, dataType);
        IVX_CHECK_STATUS( vxSetThresholdAttribute(thr.ref, VX_THRESHOLD_THRESHOLD_VALUE, &val, sizeof(val)) );
        return thr;
    }

    static Threshold createRange(vx_context c, vx_enum dataType, vx_int32 val1, vx_int32 val2)
    {
        Threshold thr = create(c, VX_THRESHOLD_TYPE_RANGE, dataType);
        IVX_CHECK_STATUS( vxSetThresholdAttribute(thr.ref, VX_THRESHOLD_THRESHOLD_LOWER, &val1, sizeof(val1)) );
        IVX_CHECK_STATUS( vxSetThresholdAttribute(thr.ref, VX_THRESHOLD_THRESHOLD_UPPER, &val2, sizeof(val2)) );
        return thr;
    }

    template<typename T>
    void query(vx_enum att, T& value) const
    { IVX_CHECK_STATUS( vxQueryThreshold(ref, att, &value, sizeof(value)) ); }

    vx_enum type() const
    {
        vx_enum v;
        query(VX_THRESHOLD_TYPE, v);
        return v;
    }

    vx_enum dataType() const
    {
        vx_enum v;
        query(VX_THRESHOLD_DATA_TYPE, v);
        return v;
    }

    vx_int32 value() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_THRESHOLD_VALUE, v);
        return v;
    }

    vx_int32 valueLower() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_THRESHOLD_LOWER, v);
        return v;
    }

    vx_int32 valueUpper() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_THRESHOLD_UPPER, v);
        return v;
    }

    vx_int32 valueTrue() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_TRUE_VALUE, v);
        return v;
    }

    vx_int32 valueFalse() const
    {
        vx_int32 v;
        query(VX_THRESHOLD_FALSE_VALUE, v);
        return v;
    }
};

/*
* Array
*/
class Array : public RefWrapper<vx_array>
{
public:
    IVX_REF_STD_CTORS_AND_ASSIGNMENT(Array);

    static Array create(vx_context c, vx_enum type, vx_size capacity)
    { return Array(vxCreateArray(c, type, capacity)); }

    static Array createVirtual(vx_graph g, vx_enum type, vx_size capacity)
    { return Array(vxCreateVirtualArray(g, type, capacity)); }
};


/*
* standard nodes
*/
namespace nodes {

Node gaussian3x3(vx_graph graph, vx_image inImg, vx_image outImg)
{
    return Node(vxGaussian3x3Node(graph, inImg, outImg));
}

} // namespace nodes

} // namespace ivx

#endif //IVX_HPP
