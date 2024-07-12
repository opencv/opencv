#ifndef _ZBUILD_H
#define _ZBUILD_H

#define _POSIX_SOURCE 1  /* fileno */
#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L /* snprintf, posix_memalign, strdup */
#endif
#ifndef _ISOC11_SOURCE
#  define _ISOC11_SOURCE 1 /* aligned_alloc */
#endif
#ifdef __OpenBSD__
#  define _BSD_SOURCE 1
#endif

#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* Determine compiler version of C Standard */
#ifdef __STDC_VERSION__
#  if __STDC_VERSION__ >= 199901L
#    ifndef STDC99
#      define STDC99
#    endif
#  endif
#  if __STDC_VERSION__ >= 201112L
#    ifndef STDC11
#      define STDC11
#    endif
#  endif
#endif

#ifndef Z_HAS_ATTRIBUTE
#  if defined(__has_attribute)
#    define Z_HAS_ATTRIBUTE(a) __has_attribute(a)
#  else
#    define Z_HAS_ATTRIBUTE(a) 0
#  endif
#endif

#ifndef Z_FALLTHROUGH
#  if Z_HAS_ATTRIBUTE(__fallthrough__) || (defined(__GNUC__) && (__GNUC__ >= 7))
#    define Z_FALLTHROUGH __attribute__((__fallthrough__))
#  else
#    define Z_FALLTHROUGH do {} while(0) /* fallthrough */
#  endif
#endif

#ifndef Z_TARGET
#  if Z_HAS_ATTRIBUTE(__target__)
#    define Z_TARGET(x) __attribute__((__target__(x)))
#  else
#    define Z_TARGET(x)
#  endif
#endif

/* This has to be first include that defines any types */
#if defined(_MSC_VER)
#  if defined(_WIN64)
    typedef __int64 ssize_t;
#  else
    typedef long ssize_t;
#  endif

#  if defined(_WIN64)
    #define SSIZE_MAX _I64_MAX
#  else
    #define SSIZE_MAX LONG_MAX
#  endif
#endif

/* MS Visual Studio does not allow inline in C, only C++.
   But it provides __inline instead, so use that. */
#if defined(_MSC_VER) && !defined(inline) && !defined(__cplusplus)
#  define inline __inline
#endif

#if defined(ZLIB_COMPAT)
#  define PREFIX(x) x
#  define PREFIX2(x) ZLIB_ ## x
#  define PREFIX3(x) z_ ## x
#  define PREFIX4(x) x ## 64
#  define zVersion zlibVersion
#else
#  define PREFIX(x) zng_ ## x
#  define PREFIX2(x) ZLIBNG_ ## x
#  define PREFIX3(x) zng_ ## x
#  define PREFIX4(x) zng_ ## x
#  define zVersion zlibng_version
#  define z_size_t size_t
#endif

/* In zlib-compat some functions and types use unsigned long, but zlib-ng use size_t */
#if defined(ZLIB_COMPAT)
#  define z_uintmax_t unsigned long
#else
#  define z_uintmax_t size_t
#endif

/* Minimum of a and b. */
#define MIN(a, b) ((a) > (b) ? (b) : (a))
/* Maximum of a and b. */
#define MAX(a, b) ((a) < (b) ? (b) : (a))
/* Ignore unused variable warning */
#define Z_UNUSED(var) (void)(var)

#if defined(HAVE_VISIBILITY_INTERNAL)
#  define Z_INTERNAL __attribute__((visibility ("internal")))
#elif defined(HAVE_VISIBILITY_HIDDEN)
#  define Z_INTERNAL __attribute__((visibility ("hidden")))
#else
#  define Z_INTERNAL
#endif

/* Symbol versioning helpers, allowing multiple versions of a function to exist.
 * Functions using this must also be added to zlib-ng.map for each version.
 * Double @@ means this is the default for newly compiled applications to link against.
 * Single @ means this is kept for backwards compatibility.
 * This is only used for Zlib-ng native API, and only on platforms supporting this.
 */
#if defined(HAVE_SYMVER)
#  define ZSYMVER(func,alias,ver) __asm__(".symver " func ", " alias "@ZLIB_NG_" ver);
#  define ZSYMVER_DEF(func,alias,ver) __asm__(".symver " func ", " alias "@@ZLIB_NG_" ver);
#else
#  define ZSYMVER(func,alias,ver)
#  define ZSYMVER_DEF(func,alias,ver)
#endif

#ifndef __cplusplus
#  define Z_REGISTER register
#else
#  define Z_REGISTER
#endif

/* Reverse the bytes in a value. Use compiler intrinsics when
   possible to take advantage of hardware implementations. */
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
#  include <stdlib.h>
#  pragma intrinsic(_byteswap_ulong)
#  define ZSWAP16(q) _byteswap_ushort(q)
#  define ZSWAP32(q) _byteswap_ulong(q)
#  define ZSWAP64(q) _byteswap_uint64(q)

#elif defined(__clang__) || (defined(__GNUC__) && \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)))
#  define ZSWAP16(q) __builtin_bswap16(q)
#  define ZSWAP32(q) __builtin_bswap32(q)
#  define ZSWAP64(q) __builtin_bswap64(q)

#elif defined(__GNUC__) && (__GNUC__ >= 2) && defined(__linux__)
#  include <byteswap.h>
#  define ZSWAP16(q) bswap_16(q)
#  define ZSWAP32(q) bswap_32(q)
#  define ZSWAP64(q) bswap_64(q)

#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#  include <sys/endian.h>
#  define ZSWAP16(q) bswap16(q)
#  define ZSWAP32(q) bswap32(q)
#  define ZSWAP64(q) bswap64(q)
#elif defined(__OpenBSD__)
#  include <sys/endian.h>
#  define ZSWAP16(q) swap16(q)
#  define ZSWAP32(q) swap32(q)
#  define ZSWAP64(q) swap64(q)
#elif defined(__INTEL_COMPILER)
/* ICC does not provide a two byte swap. */
#  define ZSWAP16(q) ((((q) & 0xff) << 8) | (((q) & 0xff00) >> 8))
#  define ZSWAP32(q) _bswap(q)
#  define ZSWAP64(q) _bswap64(q)

#else
#  define ZSWAP16(q) ((((q) & 0xff) << 8) | (((q) & 0xff00) >> 8))
#  define ZSWAP32(q) ((((q) >> 24) & 0xff) + (((q) >> 8) & 0xff00) + \
                     (((q) & 0xff00) << 8) + (((q) & 0xff) << 24))
#  define ZSWAP64(q)                           \
         (((q & 0xFF00000000000000u) >> 56u) | \
          ((q & 0x00FF000000000000u) >> 40u) | \
          ((q & 0x0000FF0000000000u) >> 24u) | \
          ((q & 0x000000FF00000000u) >> 8u)  | \
          ((q & 0x00000000FF000000u) << 8u)  | \
          ((q & 0x0000000000FF0000u) << 24u) | \
          ((q & 0x000000000000FF00u) << 40u) | \
          ((q & 0x00000000000000FFu) << 56u))
#endif

/* Only enable likely/unlikely if the compiler is known to support it */
#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define LIKELY_NULL(x)        __builtin_expect((x) != 0, 0)
#  define LIKELY(x)             __builtin_expect(!!(x), 1)
#  define UNLIKELY(x)           __builtin_expect(!!(x), 0)
#else
#  define LIKELY_NULL(x)        x
#  define LIKELY(x)             x
#  define UNLIKELY(x)           x
#endif /* (un)likely */

#if defined(HAVE_ATTRIBUTE_ALIGNED)
#  define ALIGNED_(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#  define ALIGNED_(x) __declspec(align(x))
#endif

/* Diagnostic functions */
#ifdef ZLIB_DEBUG
#  include <stdio.h>
   extern int Z_INTERNAL z_verbose;
   extern void Z_INTERNAL z_error(const char *m);
#  define Assert(cond, msg) {if (!(cond)) z_error(msg);}
#  define Trace(x) {if (z_verbose >= 0) fprintf x;}
#  define Tracev(x) {if (z_verbose > 0) fprintf x;}
#  define Tracevv(x) {if (z_verbose > 1) fprintf x;}
#  define Tracec(c, x) {if (z_verbose > 0 && (c)) fprintf x;}
#  define Tracecv(c, x) {if (z_verbose > 1 && (c)) fprintf x;}
#else
#  define Assert(cond, msg)
#  define Trace(x)
#  define Tracev(x)
#  define Tracevv(x)
#  define Tracec(c, x)
#  define Tracecv(c, x)
#endif

#ifndef NO_UNALIGNED
#  if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__) || defined(_M_AMD64)
#    define UNALIGNED_OK
#    define UNALIGNED64_OK
#  elif defined(__i386__) || defined(__i486__) || defined(__i586__) || \
        defined(__i686__) || defined(_X86_) || defined(_M_IX86)
#    define UNALIGNED_OK
#  elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#    if (defined(__GNUC__) && defined(__ARM_FEATURE_UNALIGNED)) || !defined(__GNUC__)
#      define UNALIGNED_OK
#      define UNALIGNED64_OK
#    endif
#  elif defined(__arm__) || (_M_ARM >= 7)
#    if (defined(__GNUC__) && defined(__ARM_FEATURE_UNALIGNED)) || !defined(__GNUC__)
#      define UNALIGNED_OK
#    endif
#  elif defined(__powerpc64__) || defined(__ppc64__)
#    if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#      define UNALIGNED_OK
#      define UNALIGNED64_OK
#    endif
#  endif
#endif

#if defined(__has_feature)
#  if __has_feature(memory_sanitizer)
#    define Z_MEMORY_SANITIZER 1
#    include <sanitizer/msan_interface.h>
#  endif
#endif

#ifndef Z_MEMORY_SANITIZER
#  define __msan_unpoison(a, size) do { Z_UNUSED(a); Z_UNUSED(size); } while (0)
#endif

#endif
