/* insert_string_acle.c -- insert_string integer hash variant using ACLE's CRC instructions
 *
 * Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 */

#ifdef ARM_ACLE
#include "acle_intrins.h"
#include "../../zbuild.h"
#include "../../deflate.h"

#define HASH_CALC(s, h, val) \
    h = __crc32w(0, val)

#define HASH_CALC_VAR       h
#define HASH_CALC_VAR_INIT  uint32_t h = 0

#define UPDATE_HASH         Z_TARGET_CRC update_hash_acle
#define INSERT_STRING       Z_TARGET_CRC insert_string_acle
#define QUICK_INSERT_STRING Z_TARGET_CRC quick_insert_string_acle

#include "../../insert_string_tpl.h"
#endif
