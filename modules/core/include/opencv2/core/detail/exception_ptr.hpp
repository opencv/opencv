// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_DETAILS_EXCEPTION_PTR_H
#define OPENCV_CORE_DETAILS_EXCEPTION_PTR_H

#ifndef CV__EXCEPTION_PTR
#  if defined(__ANDROID__) && defined(ATOMIC_INT_LOCK_FREE) && ATOMIC_INT_LOCK_FREE < 2
#    define CV__EXCEPTION_PTR 0  // Not supported, details: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58938
#  else
#    define CV__EXCEPTION_PTR 1
#  endif
#endif
#ifndef CV__EXCEPTION_PTR
#  define CV__EXCEPTION_PTR 0
#elif CV__EXCEPTION_PTR
#  include <exception>  // std::exception_ptr
#endif

#endif // OPENCV_CORE_DETAILS_EXCEPTION_PTR_H
