#ifndef LIB_DEFLATE_COMPRESS_H
#define LIB_DEFLATE_COMPRESS_H

#include "lib_common.h"

/*
 * DEFLATE compression is private to deflate_compress.c, but we do need to be
 * able to query the compression level for zlib and gzip header generation.
 */

struct libdeflate_compressor;

unsigned int libdeflate_get_compression_level(struct libdeflate_compressor *c);

#endif /* LIB_DEFLATE_COMPRESS_H */
