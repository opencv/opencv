#ifndef LIB_X86_DECOMPRESS_IMPL_H
#define LIB_X86_DECOMPRESS_IMPL_H

#include "cpu_features.h"

/*
 * BMI2 optimized version
 *
 * FIXME: with MSVC, this isn't actually compiled with BMI2 code generation
 * enabled yet.  That would require that this be moved to its own .c file.
 */
#if HAVE_BMI2_INTRIN
#  define deflate_decompress_bmi2	deflate_decompress_bmi2
#  define FUNCNAME			deflate_decompress_bmi2
#  if !HAVE_BMI2_NATIVE
#    define ATTRIBUTES			_target_attribute("bmi2")
#  endif
   /*
    * Even with __attribute__((target("bmi2"))), gcc doesn't reliably use the
    * bzhi instruction for 'word & BITMASK(count)'.  So use the bzhi intrinsic
    * explicitly.  EXTRACT_VARBITS() is equivalent to 'word & BITMASK(count)';
    * EXTRACT_VARBITS8() is equivalent to 'word & BITMASK((u8)count)'.
    * Nevertheless, their implementation using the bzhi intrinsic is identical,
    * as the bzhi instruction truncates the count to 8 bits implicitly.
    */
#  ifndef __clang__
#    include <immintrin.h>
#    ifdef ARCH_X86_64
#      define EXTRACT_VARBITS(word, count)  _bzhi_u64((word), (count))
#      define EXTRACT_VARBITS8(word, count) _bzhi_u64((word), (count))
#    else
#      define EXTRACT_VARBITS(word, count)  _bzhi_u32((word), (count))
#      define EXTRACT_VARBITS8(word, count) _bzhi_u32((word), (count))
#    endif
#  endif
#  include "../decompress_template.h"
#endif /* HAVE_BMI2_INTRIN */

#if defined(deflate_decompress_bmi2) && HAVE_BMI2_NATIVE
#define DEFAULT_IMPL	deflate_decompress_bmi2
#else
static inline decompress_func_t
arch_select_decompress_func(void)
{
#ifdef deflate_decompress_bmi2
	if (HAVE_BMI2(get_x86_cpu_features()))
		return deflate_decompress_bmi2;
#endif
	return NULL;
}
#define arch_select_decompress_func	arch_select_decompress_func
#endif

#endif /* LIB_X86_DECOMPRESS_IMPL_H */
