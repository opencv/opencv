/* zutil.c -- target dependent utility functions for the compression library
 * Copyright (C) 1995-2017 Jean-loup Gailly
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zutil_p.h"
#include "zutil.h"

z_const char * const PREFIX(z_errmsg)[10] = {
    (z_const char *)"need dictionary",     /* Z_NEED_DICT       2  */
    (z_const char *)"stream end",          /* Z_STREAM_END      1  */
    (z_const char *)"",                    /* Z_OK              0  */
    (z_const char *)"file error",          /* Z_ERRNO         (-1) */
    (z_const char *)"stream error",        /* Z_STREAM_ERROR  (-2) */
    (z_const char *)"data error",          /* Z_DATA_ERROR    (-3) */
    (z_const char *)"insufficient memory", /* Z_MEM_ERROR     (-4) */
    (z_const char *)"buffer error",        /* Z_BUF_ERROR     (-5) */
    (z_const char *)"incompatible version",/* Z_VERSION_ERROR (-6) */
    (z_const char *)""
};

const char PREFIX3(vstring)[] =
    " zlib-ng 2.2.1";

#ifdef ZLIB_COMPAT
const char * Z_EXPORT zlibVersion(void) {
    return ZLIB_VERSION;
}
#else
const char * Z_EXPORT zlibng_version(void) {
    return ZLIBNG_VERSION;
}
#endif

unsigned long Z_EXPORT PREFIX(zlibCompileFlags)(void) {
    unsigned long flags;

    flags = 0;
    switch ((int)(sizeof(unsigned int))) {
    case 2:     break;
    case 4:     flags += 1;     break;
    case 8:     flags += 2;     break;
    default:    flags += 3;
    }
    switch ((int)(sizeof(unsigned long))) {
    case 2:     break;
    case 4:     flags += 1 << 2;        break;
    case 8:     flags += 2 << 2;        break;
    default:    flags += 3 << 2;
    }
    switch ((int)(sizeof(void *))) {
    case 2:     break;
    case 4:     flags += 1 << 4;        break;
    case 8:     flags += 2 << 4;        break;
    default:    flags += 3 << 4;
    }
    switch ((int)(sizeof(z_off_t))) {
    case 2:     break;
    case 4:     flags += 1 << 6;        break;
    case 8:     flags += 2 << 6;        break;
    default:    flags += 3 << 6;
    }
#ifdef ZLIB_DEBUG
    flags += 1 << 8;
#endif
#ifdef ZLIB_WINAPI
    flags += 1 << 10;
#endif
    /* Bit 13 reserved for DYNAMIC_CRC_TABLE */
#ifdef NO_GZCOMPRESS
    flags += 1L << 16;
#endif
#ifdef NO_GZIP
    flags += 1L << 17;
#endif
#ifdef PKZIP_BUG_WORKAROUND
    flags += 1L << 20;
#endif
    return flags;
}

#ifdef ZLIB_DEBUG
#  include <stdlib.h>
#  ifndef verbose
#    define verbose 0
#  endif
int Z_INTERNAL z_verbose = verbose;

void Z_INTERNAL z_error(const char *m) {
    fprintf(stderr, "%s\n", m);
    exit(1);
}
#endif

/* exported to allow conversion of error code to string for compress() and
 * uncompress()
 */
const char * Z_EXPORT PREFIX(zError)(int err) {
    return ERR_MSG(err);
}

void Z_INTERNAL *PREFIX(zcalloc)(void *opaque, unsigned items, unsigned size) {
    Z_UNUSED(opaque);
    return zng_alloc((size_t)items * (size_t)size);
}

void Z_INTERNAL PREFIX(zcfree)(void *opaque, void *ptr) {
    Z_UNUSED(opaque);
    zng_free(ptr);
}
