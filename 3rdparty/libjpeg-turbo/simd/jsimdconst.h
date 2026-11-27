/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2011, 2014, 2016, 2018, 2025-2026, D. R. Commander.
 * Copyright (C) 2016-2017, Loongson Technology Corporation Limited, BeiJing.
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* Supported CPU architectures */

#define NONE     -1
#define X86_64   0
#define I386     1
#define ARM64    2
#define ARM      3
#define POWERPC  4
#define RISCV64  5
#define MIPS64   6

/* Bitmask for supported SIMD instruction sets */

#define JSIMD_NONE       0x00
#define JSIMD_MMX        0x01
#define JSIMD_3DNOW      0x02
#define JSIMD_SSE        0x04
#define JSIMD_SSE2       0x08
#define JSIMD_NEON       0x10
#define JSIMD_RVV        0x20
#define JSIMD_ALTIVEC    0x40
#define JSIMD_AVX2       0x80
#define JSIMD_MMI        0x100
#define JSIMD_MAX        0x100
#define JSIMD_UNDEFINED  ~(JSIMD_MAX * 2U - 1U)
