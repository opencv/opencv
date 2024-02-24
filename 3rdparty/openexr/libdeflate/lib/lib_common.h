/*
 * lib_common.h - internal header included by all library code
 */

#ifndef LIB_LIB_COMMON_H
#define LIB_LIB_COMMON_H

#ifdef LIBDEFLATE_H
 /*
  * When building the library, LIBDEFLATEAPI needs to be defined properly before
  * including libdeflate.h.
  */
#  error "lib_common.h must always be included before libdeflate.h"
#endif

#if defined(LIBDEFLATE_DLL) && (defined(_WIN32) || defined(__CYGWIN__))
#  define LIBDEFLATE_EXPORT_SYM  __declspec(dllexport)
#elif defined(__GNUC__)
#  define LIBDEFLATE_EXPORT_SYM  __attribute__((visibility("default")))
#else
#  define LIBDEFLATE_EXPORT_SYM
#endif

/*
 * On i386, gcc assumes that the stack is 16-byte aligned at function entry.
 * However, some compilers (e.g. MSVC) and programming languages (e.g. Delphi)
 * only guarantee 4-byte alignment when calling functions.  This is mainly an
 * issue on Windows, but it has been seen on Linux too.  Work around this ABI
 * incompatibility by realigning the stack pointer when entering libdeflate.
 * This prevents crashes in SSE/AVX code.
 */
#if defined(__GNUC__) && defined(__i386__)
#  define LIBDEFLATE_ALIGN_STACK  __attribute__((force_align_arg_pointer))
#else
#  define LIBDEFLATE_ALIGN_STACK
#endif

#define LIBDEFLATEAPI	LIBDEFLATE_EXPORT_SYM LIBDEFLATE_ALIGN_STACK

#include "../common_defs.h"

typedef void *(*malloc_func_t)(size_t);
typedef void (*free_func_t)(void *);

extern malloc_func_t libdeflate_default_malloc_func;
extern free_func_t libdeflate_default_free_func;

void *libdeflate_aligned_malloc(malloc_func_t malloc_func,
				size_t alignment, size_t size);
void libdeflate_aligned_free(free_func_t free_func, void *ptr);

#ifdef FREESTANDING
/*
 * With -ffreestanding, <string.h> may be missing, and we must provide
 * implementations of memset(), memcpy(), memmove(), and memcmp().
 * See https://gcc.gnu.org/onlinedocs/gcc/Standards.html
 *
 * Also, -ffreestanding disables interpreting calls to these functions as
 * built-ins.  E.g., calling memcpy(&v, p, WORDBYTES) will make a function call,
 * not be optimized to a single load instruction.  For performance reasons we
 * don't want that.  So, declare these functions as macros that expand to the
 * corresponding built-ins.  This approach is recommended in the gcc man page.
 * We still need the actual function definitions in case gcc calls them.
 */
void *memset(void *s, int c, size_t n);
#define memset(s, c, n)		__builtin_memset((s), (c), (n))

void *memcpy(void *dest, const void *src, size_t n);
#define memcpy(dest, src, n)	__builtin_memcpy((dest), (src), (n))

void *memmove(void *dest, const void *src, size_t n);
#define memmove(dest, src, n)	__builtin_memmove((dest), (src), (n))

int memcmp(const void *s1, const void *s2, size_t n);
#define memcmp(s1, s2, n)	__builtin_memcmp((s1), (s2), (n))

#undef LIBDEFLATE_ENABLE_ASSERTIONS
#else
#include <string.h>
#endif

/*
 * Runtime assertion support.  Don't enable this in production builds; it may
 * hurt performance significantly.
 */
#ifdef LIBDEFLATE_ENABLE_ASSERTIONS
void libdeflate_assertion_failed(const char *expr, const char *file, int line);
#define ASSERT(expr) { if (unlikely(!(expr))) \
	libdeflate_assertion_failed(#expr, __FILE__, __LINE__); }
#else
#define ASSERT(expr) (void)(expr)
#endif

#define CONCAT_IMPL(a, b)	a##b
#define CONCAT(a, b)		CONCAT_IMPL(a, b)
#define ADD_SUFFIX(name)	CONCAT(name, SUFFIX)

#endif /* LIB_LIB_COMMON_H */
