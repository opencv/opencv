// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_DETAILS_EXCEPTION_PTR_H
#define OPENCV_CORE_DETAILS_EXCEPTION_PTR_H

#ifndef CV__EXCEPTION_PTR
#  if defined(__ANDROID__) && defined(ATOMIC_INT_LOCK_FREE) && ATOMIC_INT_LOCK_FREE < 2
#    define CV__EXCEPTION_PTR 0  // Not supported, details: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58938
#  elif defined(CV_CXX11)
#    define CV__EXCEPTION_PTR 1
#  elif defined(_MSC_VER)
#    define CV__EXCEPTION_PTR (_MSC_VER >= 1600)
#  elif defined(__clang__)
#    define CV__EXCEPTION_PTR 0  // C++11 only (see above)
#  elif defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define CV__EXCEPTION_PTR (__GXX_EXPERIMENTAL_CXX0X__ > 0)
#  endif
#endif
#ifndef CV__EXCEPTION_PTR
#  define CV__EXCEPTION_PTR 0
#elif CV__EXCEPTION_PTR
#  include <exception>  // std::exception_ptr
#endif

#endif // OPENCV_CORE_DETAILS_EXCEPTION_PTR_H
