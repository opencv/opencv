/*
 * arm/cpu_features.c - feature detection for ARM CPUs
 *
 * Copyright 2018 Eric Biggers
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

/*
 * ARM CPUs don't have a standard way for unprivileged programs to detect CPU
 * features.  But an OS-specific way can be used when available.
 */

#ifdef __APPLE__
#  undef _ANSI_SOURCE
#  undef _DARWIN_C_SOURCE
#  define _DARWIN_C_SOURCE /* for sysctlbyname() */
#endif

#include "../cpu_features_common.h" /* must be included first */
#include "cpu_features.h"

#if HAVE_DYNAMIC_ARM_CPU_FEATURES

#ifdef __linux__
/*
 * On Linux, arm32 and arm64 CPU features can be detected by reading the
 * AT_HWCAP and AT_HWCAP2 values from /proc/self/auxv.
 *
 * Ideally we'd use the C library function getauxval(), but it's not guaranteed
 * to be available: it was only added to glibc in 2.16, and in Android it was
 * added to API level 18 for arm32 and level 21 for arm64.
 */

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#define AT_HWCAP	16
#define AT_HWCAP2	26

static void scan_auxv(unsigned long *hwcap, unsigned long *hwcap2)
{
	int fd;
	unsigned long auxbuf[32];
	int filled = 0;
	int i;

	fd = open("/proc/self/auxv", O_RDONLY);
	if (fd < 0)
		return;

	for (;;) {
		do {
			int ret = read(fd, &((char *)auxbuf)[filled],
				       sizeof(auxbuf) - filled);
			if (ret <= 0) {
				if (ret < 0 && errno == EINTR)
					continue;
				goto out;
			}
			filled += ret;
		} while (filled < 2 * sizeof(long));

		i = 0;
		do {
			unsigned long type = auxbuf[i];
			unsigned long value = auxbuf[i + 1];

			if (type == AT_HWCAP)
				*hwcap = value;
			else if (type == AT_HWCAP2)
				*hwcap2 = value;
			i += 2;
			filled -= 2 * sizeof(long);
		} while (filled >= 2 * sizeof(long));

		memmove(auxbuf, &auxbuf[i], filled);
	}
out:
	close(fd);
}

static u32 query_arm_cpu_features(void)
{
	u32 features = 0;
	unsigned long hwcap = 0;
	unsigned long hwcap2 = 0;

	scan_auxv(&hwcap, &hwcap2);

#ifdef ARCH_ARM32
	STATIC_ASSERT(sizeof(long) == 4);
	if (hwcap & (1 << 12))	/* HWCAP_NEON */
		features |= ARM_CPU_FEATURE_NEON;
	if (hwcap2 & (1 << 1))	/* HWCAP2_PMULL */
		features |= ARM_CPU_FEATURE_PMULL;
	if (hwcap2 & (1 << 4))	/* HWCAP2_CRC32 */
		features |= ARM_CPU_FEATURE_CRC32;
#else
	STATIC_ASSERT(sizeof(long) == 8);
	if (hwcap & (1 << 1))	/* HWCAP_ASIMD */
		features |= ARM_CPU_FEATURE_NEON;
	if (hwcap & (1 << 4))	/* HWCAP_PMULL */
		features |= ARM_CPU_FEATURE_PMULL;
	if (hwcap & (1 << 7))	/* HWCAP_CRC32 */
		features |= ARM_CPU_FEATURE_CRC32;
	if (hwcap & (1 << 17))	/* HWCAP_SHA3 */
		features |= ARM_CPU_FEATURE_SHA3;
	if (hwcap & (1 << 20))	/* HWCAP_ASIMDDP */
		features |= ARM_CPU_FEATURE_DOTPROD;
#endif
	return features;
}

#elif defined(__APPLE__)
/* On Apple platforms, arm64 CPU features can be detected via sysctlbyname(). */

#include <sys/types.h>
#include <sys/sysctl.h>

static const struct {
	const char *name;
	u32 feature;
} feature_sysctls[] = {
	{ "hw.optional.neon",		  ARM_CPU_FEATURE_NEON },
	{ "hw.optional.AdvSIMD",	  ARM_CPU_FEATURE_NEON },
	{ "hw.optional.arm.FEAT_PMULL",	  ARM_CPU_FEATURE_PMULL },
	{ "hw.optional.armv8_crc32",	  ARM_CPU_FEATURE_CRC32 },
	{ "hw.optional.armv8_2_sha3",	  ARM_CPU_FEATURE_SHA3 },
	{ "hw.optional.arm.FEAT_SHA3",	  ARM_CPU_FEATURE_SHA3 },
	{ "hw.optional.arm.FEAT_DotProd", ARM_CPU_FEATURE_DOTPROD },
};

static u32 query_arm_cpu_features(void)
{
	u32 features = 0;
	size_t i;

	for (i = 0; i < ARRAY_LEN(feature_sysctls); i++) {
		const char *name = feature_sysctls[i].name;
		u32 val = 0;
		size_t valsize = sizeof(val);

		if (sysctlbyname(name, &val, &valsize, NULL, 0) == 0 &&
		    valsize == sizeof(val) && val == 1)
			features |= feature_sysctls[i].feature;
	}
	return features;
}
#elif defined(_WIN32)

#include <windows.h>

static u32 query_arm_cpu_features(void)
{
	u32 features = ARM_CPU_FEATURE_NEON;

	if (IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE))
		features |= ARM_CPU_FEATURE_PMULL;
	if (IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE))
		features |= ARM_CPU_FEATURE_CRC32;

	/* FIXME: detect SHA3 and DOTPROD support too. */

	return features;
}
#else
#error "unhandled case"
#endif

static const struct cpu_feature arm_cpu_feature_table[] = {
	{ARM_CPU_FEATURE_NEON,		"neon"},
	{ARM_CPU_FEATURE_PMULL,		"pmull"},
	{ARM_CPU_FEATURE_CRC32,		"crc32"},
	{ARM_CPU_FEATURE_SHA3,		"sha3"},
	{ARM_CPU_FEATURE_DOTPROD,	"dotprod"},
};

volatile u32 libdeflate_arm_cpu_features = 0;

void libdeflate_init_arm_cpu_features(void)
{
	u32 features = query_arm_cpu_features();

	disable_cpu_features_for_testing(&features, arm_cpu_feature_table,
					 ARRAY_LEN(arm_cpu_feature_table));

	libdeflate_arm_cpu_features = features | ARM_CPU_FEATURES_KNOWN;
}

#endif /* HAVE_DYNAMIC_ARM_CPU_FEATURES */
