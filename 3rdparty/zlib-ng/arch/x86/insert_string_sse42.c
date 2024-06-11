/* insert_string_sse42.c -- insert_string integer hash variant using SSE4.2's CRC instructions
 *
 * Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 */

#ifdef X86_SSE42
#include "../../zbuild.h"
#include <nmmintrin.h>
#include "../../deflate.h"

#define HASH_CALC(s, h, val)\
    h = _mm_crc32_u32(h, val)

#define HASH_CALC_VAR       h
#define HASH_CALC_VAR_INIT  uint32_t h = 0

#define UPDATE_HASH         update_hash_sse42
#define INSERT_STRING       insert_string_sse42
#define QUICK_INSERT_STRING quick_insert_string_sse42

#include "../../insert_string_tpl.h"
#endif
