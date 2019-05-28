// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_CVSTD_WRAPPER_HPP
#define OPENCV_CORE_CVSTD_WRAPPER_HPP

#include "opencv2/core/cvdef.h"

#include <string>
#include <memory>  // std::shared_ptr
#include <type_traits>  // std::enable_if

namespace cv {

using std::nullptr_t;

//! @addtogroup core_basic
//! @{

#ifdef CV_DOXYGEN

template <typename _Tp> using Ptr = std::shared_ptr<_Tp>;  // In ideal world it should look like this, but we need some compatibility workarounds below

template<typename _Tp, typename ... A1> static inline
Ptr<_Tp> makePtr(const A1&... a1) { return std::make_shared<_Tp>(a1...); }

#else  // cv::Ptr with compatibility workarounds

// It should be defined for C-API types only.
// C++ types should use regular "delete" operator.
template<typename Y> struct DefaultDeleter;
#if 0
{
    void operator()(Y* p) const;
};
#endif

namespace sfinae {
template<typename C, typename Ret, typename... Args>
struct has_parenthesis_operator
{
private:
    template<typename T>
    static CV_CONSTEXPR std::true_type check(typename std::is_same<typename std::decay<decltype(std::declval<T>().operator()(std::declval<Args>()...))>::type, Ret>::type*);

    template<typename> static CV_CONSTEXPR std::false_type check(...);

    typedef decltype(check<C>(0)) type;

public:
    static CV_CONSTEXPR bool value = type::value;
};
} // namespace sfinae

template <typename T, typename = void>
struct has_custom_delete
        : public std::false_type {};

// Force has_custom_delete to std::false_type when NVCC is compiling CUDA source files
#ifndef __CUDACC__
template <typename T>
struct has_custom_delete<T, typename std::enable_if< sfinae::has_parenthesis_operator<DefaultDeleter<T>, void, T*>::value >::type >
        : public std::true_type {};
#endif

template<typename T>
struct Ptr : public std::shared_ptr<T>
{
#if 0
    using std::shared_ptr<T>::shared_ptr;  // GCC 5.x can't handle this
#else
    inline Ptr() CV_NOEXCEPT : std::shared_ptr<T>() {}
    inline Ptr(nullptr_t) CV_NOEXCEPT : std::shared_ptr<T>(nullptr) {}
    template<typename Y, typename D> inline Ptr(Y* p, D d) : std::shared_ptr<T>(p, d) {}
    template<typename D> inline Ptr(nullptr_t, D d) : std::shared_ptr<T>(nullptr, d) {}

    template<typename Y> inline Ptr(const Ptr<Y>& r, T* ptr) CV_NOEXCEPT : std::shared_ptr<T>(r, ptr) {}

    inline Ptr(const Ptr<T>& o) CV_NOEXCEPT : std::shared_ptr<T>(o) {}
    inline Ptr(Ptr<T>&& o) CV_NOEXCEPT : std::shared_ptr<T>(std::move(o)) {}

    template<typename Y> inline Ptr(const Ptr<Y>& o) CV_NOEXCEPT : std::shared_ptr<T>(o) {}
    template<typename Y> inline Ptr(Ptr<Y>&& o) CV_NOEXCEPT : std::shared_ptr<T>(std::move(o)) {}
#endif
    inline Ptr(const std::shared_ptr<T>& o) CV_NOEXCEPT : std::shared_ptr<T>(o) {}
    inline Ptr(std::shared_ptr<T>&& o) CV_NOEXCEPT : std::shared_ptr<T>(std::move(o)) {}

    // Overload with custom DefaultDeleter: Ptr<IplImage>(...)
    template<typename Y>
    inline Ptr(const std::true_type&, Y* ptr) : std::shared_ptr<T>(ptr, DefaultDeleter<Y>()) {}

    // Overload without custom deleter: Ptr<std::string>(...);
    template<typename Y>
    inline Ptr(const std::false_type&, Y* ptr) : std::shared_ptr<T>(ptr) {}

    template<typename Y = T>
    inline Ptr(Y* ptr) : Ptr(has_custom_delete<Y>(), ptr) {}

    // Overload with custom DefaultDeleter: Ptr<IplImage>(...)
    template<typename Y>
    inline void reset(const std::true_type&, Y* ptr) { std::shared_ptr<T>::reset(ptr, DefaultDeleter<Y>()); }

    // Overload without custom deleter: Ptr<std::string>(...);
    template<typename Y>
    inline void reset(const std::false_type&, Y* ptr) { std::shared_ptr<T>::reset(ptr); }

    template<typename Y>
    inline void reset(Y* ptr) { Ptr<T>::reset(has_custom_delete<Y>(), ptr); }

    template<class Y, class Deleter>
    void reset(Y* ptr, Deleter d) { std::shared_ptr<T>::reset(ptr, d); }

    void reset() CV_NOEXCEPT { std::shared_ptr<T>::reset(); }

    Ptr& operator=(const Ptr& o) { std::shared_ptr<T>::operator =(o); return *this; }
    template<typename Y> inline Ptr& operator=(const Ptr<Y>& o) { std::shared_ptr<T>::operator =(o); return *this; }

    T* operator->() const CV_NOEXCEPT { return std::shared_ptr<T>::get();}
    typename std::add_lvalue_reference<T>::type operator*() const CV_NOEXCEPT { return *std::shared_ptr<T>::get(); }

    // OpenCV 3.x methods (not a part of standard C++ library)
    inline void release() { std::shared_ptr<T>::reset(); }
    inline operator T* () const { return std::shared_ptr<T>::get(); }
    inline bool empty() const { return std::shared_ptr<T>::get() == nullptr; }

    template<typename Y> inline
    Ptr<Y> staticCast() const CV_NOEXCEPT { return std::static_pointer_cast<Y>(*this); }

    template<typename Y> inline
    Ptr<Y> constCast() const CV_NOEXCEPT { return std::const_pointer_cast<Y>(*this); }

    template<typename Y> inline
    Ptr<Y> dynamicCast() const CV_NOEXCEPT { return std::dynamic_pointer_cast<Y>(*this); }
};

template<typename _Tp, typename ... A1> static inline
Ptr<_Tp> makePtr(const A1&... a1)
{
    static_assert( !has_custom_delete<_Tp>::value, "Can't use this makePtr with custom DefaultDeleter");
    return (Ptr<_Tp>)std::make_shared<_Tp>(a1...);
}

#endif // CV_DOXYGEN

//! @} core_basic
} // cv

#endif //OPENCV_CORE_CVSTD_WRAPPER_HPP
