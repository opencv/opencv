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

template<typename Y>
struct DefaultDeleter
{
#ifndef _MSC_VER
    void operator()(Y* p) const = delete;  // not available by default; enabled for specializations only
#else
    void operator()(Y* p) const { delete p; }
#endif
};

namespace sfinae {
template<typename C, typename Ret, typename... Args>
struct has_parenthesis_operator
{
private:
    template<typename T>
    static constexpr std::true_type check(typename std::is_same<typename std::decay<decltype(std::declval<T>().operator()(std::declval<Args>()...))>::type, Ret>::type*);

    template<typename> static constexpr std::false_type check(...);

    typedef decltype(check<C>(0)) type;

public:
    static constexpr bool value = type::value;
};
} // namespace sfinae

template <typename Y> using has_custom_delete = sfinae::has_parenthesis_operator<DefaultDeleter<Y>, void, Y*>;

template<typename T>
struct Ptr : public std::shared_ptr<T>
{
#if 0
    using std::shared_ptr<T>::shared_ptr;  // GCC 5.x can't handle this
#else
    inline Ptr() noexcept : std::shared_ptr<T>() {}
    inline Ptr(nullptr_t) noexcept : std::shared_ptr<T>(nullptr) {}
    template<typename Y, typename D> inline Ptr(Y* p, D d) : std::shared_ptr<T>(p, d) {}
    template<typename D> inline Ptr(nullptr_t, D d) : std::shared_ptr<T>(nullptr, d) {}

    template<typename Y> inline Ptr(const Ptr<Y>& r, T* ptr) noexcept : std::shared_ptr<T>(r, ptr) {}

    inline Ptr(const Ptr<T>& o) noexcept : std::shared_ptr<T>(o) {}
    inline Ptr(Ptr<T>&& o) noexcept : std::shared_ptr<T>(std::move(o)) {}

    template<typename Y> inline Ptr(const Ptr<Y>& o) noexcept : std::shared_ptr<T>(o) {}
    template<typename Y> inline Ptr(Ptr<Y>&& o) noexcept : std::shared_ptr<T>(std::move(o)) {}
#endif
    inline Ptr(const std::shared_ptr<T>& o) noexcept : std::shared_ptr<T>(o) {}
    inline Ptr(std::shared_ptr<T>&& o) noexcept : std::shared_ptr<T>(std::move(o)) {}

#ifndef _MSC_VER
    // Overload with custom DefaultDeleter: Ptr<IplImage>(...)
    template<typename Y = T, class = typename std::enable_if< has_custom_delete<Y>::value >::type>
    inline Ptr(Y* ptr) : std::shared_ptr<T>(ptr, DefaultDeleter<Y>()) {}

    // Overload without custom deleter: Ptr<std::string>(...);
    template<typename Y = T, int = sizeof(typename std::enable_if< !has_custom_delete<Y>::value, int >::type) >
    inline Ptr(Y* ptr) : std::shared_ptr<T>(ptr) {}

    // Overload with custom DefaultDeleter: Ptr<IplImage>(...)
    template<typename Y, class = typename std::enable_if< has_custom_delete<Y>::value >::type>
    inline void reset(Y* ptr) { std::shared_ptr<T>::reset(ptr, DefaultDeleter<Y>()); }

    // Overload without custom deleter: Ptr<std::string>(...);
    template<typename Y, int = sizeof(typename std::enable_if< !has_custom_delete<Y>::value, int >::type) >
    inline void reset(Y* ptr) { std::shared_ptr<T>::reset(ptr); }
#else
    template<typename Y>
    inline Ptr(Y* ptr) : std::shared_ptr<T>(ptr, DefaultDeleter<Y>()) {}

    template<typename Y>
    inline void reset(Y* ptr) { std::shared_ptr<T>::reset(ptr, DefaultDeleter<Y>()); }
#endif

    template<class Y, class Deleter>
    void reset(Y* ptr, Deleter d) { std::shared_ptr<T>::reset(ptr, d); }

    void reset() noexcept { std::shared_ptr<T>::reset(); }

    Ptr& operator=(const Ptr& o) { std::shared_ptr<T>::operator =(o); return *this; }
    template<typename Y> inline Ptr& operator=(const Ptr<Y>& o) { std::shared_ptr<T>::operator =(o); return *this; }

    T* operator->() const noexcept { return std::shared_ptr<T>::get();}
    typename std::add_lvalue_reference<T>::type operator*() const noexcept { return *std::shared_ptr<T>::get(); }

    // OpenCV 3.x methods (not a part of standart C++ library)
    inline void release() { std::shared_ptr<T>::reset(); }
    inline operator T* () const { return std::shared_ptr<T>::get(); }
    inline bool empty() const { return std::shared_ptr<T>::get() == NULL; }

    template<typename Y> inline
    Ptr<Y> staticCast() const noexcept { return std::static_pointer_cast<Y>(*this); }

    template<typename Y> inline
    Ptr<Y> constCast() const noexcept { return std::const_pointer_cast<Y>(*this); }

    template<typename Y> inline
    Ptr<Y> dynamicCast() const noexcept { return std::dynamic_pointer_cast<Y>(*this); }
};

template<typename _Tp, typename ... A1> static inline
Ptr<_Tp> makePtr(const A1&... a1)
{
#ifndef _MSC_VER
    static_assert( !has_custom_delete<_Tp>::value, "Can't use this makePtr with custom DefaultDeleter");
    return (Ptr<_Tp>)std::make_shared<_Tp>(a1...);
#else
    return Ptr<_Tp>(new _Tp(a1...), DefaultDeleter<_Tp>());
#endif
}

#endif // CV_DOXYGEN

//! @} core_basic
} // cv

#endif //OPENCV_CORE_CVSTD_WRAPPER_HPP
