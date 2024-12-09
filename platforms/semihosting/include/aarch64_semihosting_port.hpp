// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef AARCH64_BAREMETAL_PORT_HPP
#define AARCH64_BAREMETAL_PORT_HPP

#include <malloc.h> // Needed for `memalign`.
#include <sys/errno.h> // Needed for `ENOMEM`.

// -std=c++11 is missing the following definitions when targeting
// semihosting on aarch64.
#if __cplusplus == 201103L
#include <cmath>
#define M_PI 3.14159265358979323846
#define M_SQRT2 1.41421356237309504880

namespace std {
inline double cbrt(double x) {
    return ::cbrt(x);
}
inline double copysign(double mag, double sgn) {
    return ::copysign(mag, sgn);
}
} //namespace std
#endif // __cplusplus == 201103L

extern "C" {
// Redirect the implementation of `posix_memalign` to `memalign`
// as the former is
// missing at link time. https://pubs.opengroup.org/onlinepubs/9699919799/functions/posix_memalign.html
__attribute__((weak)) int posix_memalign(void **memptr, size_t alignment, size_t size) {
    void * ptr =  memalign(alignment, size);
    if (ptr != NULL) {
        *memptr = ptr;
        return 0;
    }
    return ENOMEM;
}
} // extern "C"

#endif
