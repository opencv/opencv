/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_PRIVATE_XDR_H
#define OPENEXR_PRIVATE_XDR_H

/*
 * This is only a subset of generic endian behavior. OpenEXR is
 * defined as little endian, so we only care about host <-> little
 * endian and not anything with big endian.
 */
#if defined(__linux__) || defined(__CYGWIN__)

#    include <endian.h>
#    define EXR_HOST_IS_NOT_LITTLE_ENDIAN (__BYTE_ORDER != __LITTLE_ENDIAN)

#elif defined(_WIN32) || defined(_WIN64)

#    include <stdlib.h>
#    include <windows.h>
#    define EXR_HOST_IS_NOT_LITTLE_ENDIAN (BYTE_ORDER != LITTLE_ENDIAN)
#    if EXR_HOST_IS_NOT_LITTLE_ENDIAN
#        if defined(_MSC_VER)
#            define htole16(x) _byteswap_ushort (x)
#            define le16toh(x) _byteswap_ushort (x)
#            define htole32(x) _byteswap_ulong (x)
#            define le32toh(x) _byteswap_ulong (x)
#            define htole64(x) _byteswap_uint64 (x)
#            define le64toh(x) _byteswap_uint64 (x)
#        elif defined(__GNUC__) || defined(__clang__)
#            define htole16(x) __builtin_bswap16 (x)
#            define le16toh(x) __builtin_bswap16 (x)
#            define htole32(x) __builtin_bswap32 (x)
#            define le32toh(x) __builtin_bswap32 (x)
#            define htole64(x) __builtin_bswap64 (x)
#            define le64toh(x) __builtin_bswap64 (x)
#        else
#            error Windows compiler unrecognized
#        endif
#    else
#        define htole16(x) (x)
#        define le16toh(x) (x)
#        define htole32(x) (x)
#        define le32toh(x) (x)
#        define htole64(x) (x)
#        define le64toh(x) (x)
#    endif

#elif defined(__APPLE__)

#    include <libkern/OSByteOrder.h>
#    define htole16(x) OSSwapHostToLittleInt16 (x)
#    define le16toh(x) OSSwapLittleToHostInt16 (x)
#    define htole32(x) OSSwapHostToLittleInt32 (x)
#    define le32toh(x) OSSwapLittleToHostInt32 (x)
#    define htole64(x) OSSwapHostToLittleInt64 (x)
#    define le64toh(x) OSSwapLittleToHostInt64 (x)
#    if defined(__m68k__) || defined(__POWERPC__)
#        define EXR_HOST_IS_NOT_LITTLE_ENDIAN 1
#    else
#        define EXR_HOST_IS_NOT_LITTLE_ENDIAN 0
#    endif

#elif defined(__OpenBSD__) || defined(__FreeBSD__) || defined(__NetBSD__) ||   \
    defined(__DragonFly__)

#    include <sys/endian.h>
#    define EXR_HOST_IS_NOT_LITTLE_ENDIAN (__BYTE_ORDER != __LITTLE_ENDIAN)

#else

#    include <endian.h>
#    define EXR_HOST_IS_NOT_LITTLE_ENDIAN (__BYTE_ORDER != __LITTLE_ENDIAN)

#endif

#include <string.h>

static inline uint64_t
one_to_native64 (uint64_t v)
{
    return le64toh (v);
}

static inline uint64_t
one_from_native64 (uint64_t v)
{
    return htole64 (v);
}

static inline void
priv_to_native64 (void* ptr, int n)
{
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    uint64_t* vals = (uint64_t*) ptr;
    for (int i = 0; i < n; ++i)
        vals[i] = le64toh (vals[i]);
#else
    (void) ptr;
    (void) n;
#endif
}

static inline void
priv_from_native64 (void* ptr, int n)
{
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    uint64_t* vals = (uint64_t*) ptr;
    for (int i = 0; i < n; ++i)
        vals[i] = htole64 (vals[i]);
#else
    (void) ptr;
    (void) n;
#endif
}

/**************************************/

static inline uint32_t
one_to_native32 (uint32_t v)
{
    return le32toh (v);
}

static inline uint32_t
one_from_native32 (uint32_t v)
{
    return htole32 (v);
}

static inline void
priv_to_native32 (void* ptr, int n)
{
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    uint32_t* vals = (uint32_t*) ptr;
    for (int i = 0; i < n; ++i)
        vals[i] = le32toh (vals[i]);
#else
    (void) ptr;
    (void) n;
#endif
}

static inline void
priv_from_native32 (void* ptr, int n)
{
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    uint32_t* vals = (uint32_t*) ptr;
    for (int i = 0; i < n; ++i)
        vals[i] = htole32 (vals[i]);
#else
    (void) ptr;
    (void) n;
#endif
}

static inline float
one_to_native_float (float v)
{
    union
    {
        uint32_t i;
        float    f;
    } coerce;
    coerce.f = v;
    coerce.i = one_to_native32 (coerce.i);
    return coerce.f;
}

static inline float
one_from_native_float (float v)
{
    union
    {
        uint32_t i;
        float    f;
    } coerce;
    coerce.f = v;
    coerce.i = one_from_native32 (coerce.i);
    return coerce.f;
}

/**************************************/

static inline uint16_t
one_to_native16 (uint16_t v)
{
    return le16toh (v);
}

static inline uint16_t
one_from_native16 (uint16_t v)
{
    return htole16 (v);
}

static inline void
priv_to_native16 (void* ptr, int n)
{
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    uint16_t* vals = (uint16_t*) ptr;
    for (int i = 0; i < n; ++i)
        vals[i] = le16toh (vals[i]);
#else
    (void) ptr;
    (void) n;
#endif
}

static inline void
priv_from_native16 (void* ptr, int n)
{
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    uint16_t* vals = (uint16_t*) ptr;
    for (int i = 0; i < n; ++i)
        vals[i] = htole16 (vals[i]);
#else
    (void) ptr;
    (void) n;
#endif
}

/**************************************/

static inline void
priv_to_native (void* ptr, int n, size_t eltsize)
{
    if (eltsize == 8)
        priv_to_native64 (ptr, n);
    else if (eltsize == 4)
        priv_to_native32 (ptr, n);
    else if (eltsize == 2)
        priv_to_native16 (ptr, n);
}

static inline void
priv_from_native (void* ptr, int n, size_t eltsize)
{
    if (eltsize == 8)
        priv_from_native64 (ptr, n);
    else if (eltsize == 4)
        priv_from_native32 (ptr, n);
    else if (eltsize == 2)
        priv_from_native16 (ptr, n);
}

/**************************************/

static inline void
unaligned_store16 (void* dst, uint16_t v)
{
    uint16_t xe = one_from_native16 (v);
    memcpy (dst, &xe, 2);
}

static inline void
unaligned_store32 (void* dst, uint32_t v)
{
    uint32_t xe = one_from_native32 (v);
    memcpy (dst, &xe, 4);
}

/**************************************/

static inline uint16_t
unaligned_load16 (const void* src)
{
    uint16_t tmp;
    memcpy (&tmp, src, 2);
    return one_to_native16 (tmp);
}

static inline uint32_t
unaligned_load32 (const void* src)
{
    uint32_t tmp;
    memcpy (&tmp, src, 4);
    return one_to_native32 (tmp);
}

#endif /* OPENEXR_PRIVATE_XDR_H */
