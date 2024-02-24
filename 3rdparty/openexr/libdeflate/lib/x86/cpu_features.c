/*
 * x86/cpu_features.c - feature detection for x86 CPUs
 *
 * Copyright 2016 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "../cpu_features_common.h" /* must be included first */
#include "cpu_features.h"

#if HAVE_DYNAMIC_X86_CPU_FEATURES

/*
 * With old GCC versions we have to manually save and restore the x86_32 PIC
 * register (ebx).  See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47602
 */
#if defined(ARCH_X86_32) && defined(__PIC__)
#  define EBX_CONSTRAINT "=&r"
#else
#  define EBX_CONSTRAINT "=b"
#endif

/* Execute the CPUID instruction. */
static inline void
cpuid(u32 leaf, u32 subleaf, u32 *a, u32 *b, u32 *c, u32 *d)
{
#ifdef _MSC_VER
	int result[4];

	__cpuidex(result, leaf, subleaf);
	*a = result[0];
	*b = result[1];
	*c = result[2];
	*d = result[3];
#else
	__asm__ volatile(".ifnc %%ebx, %1; mov  %%ebx, %1; .endif\n"
			 "cpuid                                  \n"
			 ".ifnc %%ebx, %1; xchg %%ebx, %1; .endif\n"
			 : "=a" (*a), EBX_CONSTRAINT (*b), "=c" (*c), "=d" (*d)
			 : "a" (leaf), "c" (subleaf));
#endif
}

/* Read an extended control register. */
static inline u64
read_xcr(u32 index)
{
#ifdef _MSC_VER
	return _xgetbv(index);
#else
	u32 d, a;

	/*
	 * Execute the "xgetbv" instruction.  Old versions of binutils do not
	 * recognize this instruction, so list the raw bytes instead.
	 *
	 * This must be 'volatile' to prevent this code from being moved out
	 * from under the check for OSXSAVE.
	 */
	__asm__ volatile(".byte 0x0f, 0x01, 0xd0" :
			 "=d" (d), "=a" (a) : "c" (index));

	return ((u64)d << 32) | a;
#endif
}

static const struct cpu_feature x86_cpu_feature_table[] = {
	{X86_CPU_FEATURE_SSE2,		"sse2"},
	{X86_CPU_FEATURE_PCLMUL,	"pclmul"},
	{X86_CPU_FEATURE_AVX,		"avx"},
	{X86_CPU_FEATURE_AVX2,		"avx2"},
	{X86_CPU_FEATURE_BMI2,		"bmi2"},
};

volatile u32 libdeflate_x86_cpu_features = 0;

/* Initialize libdeflate_x86_cpu_features. */
void libdeflate_init_x86_cpu_features(void)
{
	u32 max_leaf, a, b, c, d;
	u64 xcr0 = 0;
	u32 features = 0;

	/* EAX=0: Highest Function Parameter and Manufacturer ID */
	cpuid(0, 0, &max_leaf, &b, &c, &d);
	if (max_leaf < 1)
		goto out;

	/* EAX=1: Processor Info and Feature Bits */
	cpuid(1, 0, &a, &b, &c, &d);
	if (d & (1 << 26))
		features |= X86_CPU_FEATURE_SSE2;
	if (c & (1 << 1))
		features |= X86_CPU_FEATURE_PCLMUL;
	if (c & (1 << 27))
		xcr0 = read_xcr(0);
	if ((c & (1 << 28)) && ((xcr0 & 0x6) == 0x6))
		features |= X86_CPU_FEATURE_AVX;

	if (max_leaf < 7)
		goto out;

	/* EAX=7, ECX=0: Extended Features */
	cpuid(7, 0, &a, &b, &c, &d);
	if ((b & (1 << 5)) && ((xcr0 & 0x6) == 0x6))
		features |= X86_CPU_FEATURE_AVX2;
	if (b & (1 << 8))
		features |= X86_CPU_FEATURE_BMI2;

out:
	disable_cpu_features_for_testing(&features, x86_cpu_feature_table,
					 ARRAY_LEN(x86_cpu_feature_table));

	libdeflate_x86_cpu_features = features | X86_CPU_FEATURES_KNOWN;
}

#endif /* HAVE_DYNAMIC_X86_CPU_FEATURES */
