/*
 * copyright (c) 2006 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Macro definitions for various function/variable attributes
 */

#ifndef AVUTIL_ATTRIBUTES_H
#define AVUTIL_ATTRIBUTES_H

#ifdef __GNUC__
#    define AV_GCC_VERSION_AT_LEAST(x,y) (__GNUC__ > x || __GNUC__ == x && __GNUC_MINOR__ >= y)
#else
#    define AV_GCC_VERSION_AT_LEAST(x,y) 0
#endif

#ifndef av_always_inline
#if AV_GCC_VERSION_AT_LEAST(3,1)
#    define av_always_inline __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#    define av_always_inline __forceinline
#else
#    define av_always_inline inline
#endif
#endif

#ifndef av_extern_inline
#if defined(__ICL) && __ICL >= 1210 || defined(__GNUC_STDC_INLINE__)
#    define av_extern_inline extern inline
#else
#    define av_extern_inline inline
#endif
#endif

#if AV_GCC_VERSION_AT_LEAST(3,1)
#    define av_noinline __attribute__((noinline))
#else
#    define av_noinline
#endif

#if AV_GCC_VERSION_AT_LEAST(3,1)
#    define av_pure __attribute__((pure))
#else
#    define av_pure
#endif

#ifndef av_restrict
#define av_restrict restrict
#endif

#if AV_GCC_VERSION_AT_LEAST(2,6)
#    define av_const __attribute__((const))
#else
#    define av_const
#endif

#if AV_GCC_VERSION_AT_LEAST(4,3)
#    define av_cold __attribute__((cold))
#else
#    define av_cold
#endif

#if AV_GCC_VERSION_AT_LEAST(4,1)
#    define av_flatten __attribute__((flatten))
#else
#    define av_flatten
#endif

#if AV_GCC_VERSION_AT_LEAST(3,1)
#    define attribute_deprecated __attribute__((deprecated))
#else
#    define attribute_deprecated
#endif

/**
 * Disable warnings about deprecated features
 * This is useful for sections of code kept for backward compatibility and
 * scheduled for removal.
 */
#ifndef AV_NOWARN_DEPRECATED
#if AV_GCC_VERSION_AT_LEAST(4,6)
#    define AV_NOWARN_DEPRECATED(code) \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"") \
        code \
        _Pragma("GCC diagnostic pop")
#else
#    define AV_NOWARN_DEPRECATED(code) code
#endif
#endif


#if defined(__GNUC__)
#    define av_unused __attribute__((unused))
#else
#    define av_unused
#endif

/**
 * Mark a variable as used and prevent the compiler from optimizing it
 * away.  This is useful for variables accessed only from inline
 * assembler without the compiler being aware.
 */
#if AV_GCC_VERSION_AT_LEAST(3,1)
#    define av_used __attribute__((used))
#else
#    define av_used
#endif

#if AV_GCC_VERSION_AT_LEAST(3,3)
#   define av_alias __attribute__((may_alias))
#else
#   define av_alias
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
#    define av_uninit(x) x=x
#else
#    define av_uninit(x) x
#endif

#ifdef __GNUC__
#    define av_builtin_constant_p __builtin_constant_p
#    define av_printf_format(fmtpos, attrpos) __attribute__((__format__(__printf__, fmtpos, attrpos)))
#else
#    define av_builtin_constant_p(x) 0
#    define av_printf_format(fmtpos, attrpos)
#endif

#if AV_GCC_VERSION_AT_LEAST(2,5)
#    define av_noreturn __attribute__((noreturn))
#else
#    define av_noreturn
#endif

#endif /* AVUTIL_ATTRIBUTES_H */
