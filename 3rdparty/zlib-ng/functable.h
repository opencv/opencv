/* functable.h -- Struct containing function pointers to optimized functions
 * Copyright (C) 2017 Hans Kristian Rosbach
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef FUNCTABLE_H_
#define FUNCTABLE_H_

#include "deflate.h"
#include "crc32_fold.h"
#include "adler32_fold.h"

#ifdef ZLIB_COMPAT
typedef struct z_stream_s z_stream;
#else
typedef struct zng_stream_s zng_stream;
#endif

struct functable_s {
    void     (* force_init)         (void);
    uint32_t (* adler32)            (uint32_t adler, const uint8_t *buf, size_t len);
    uint32_t (* adler32_fold_copy)  (uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);
    uint8_t* (* chunkmemset_safe)   (uint8_t *out, unsigned dist, unsigned len, unsigned left);
    uint32_t (* chunksize)          (void);
    uint32_t (* compare256)         (const uint8_t *src0, const uint8_t *src1);
    uint32_t (* crc32)              (uint32_t crc, const uint8_t *buf, size_t len);
    void     (* crc32_fold)         (struct crc32_fold_s *crc, const uint8_t *src, size_t len, uint32_t init_crc);
    void     (* crc32_fold_copy)    (struct crc32_fold_s *crc, uint8_t *dst, const uint8_t *src, size_t len);
    uint32_t (* crc32_fold_final)   (struct crc32_fold_s *crc);
    uint32_t (* crc32_fold_reset)   (struct crc32_fold_s *crc);
    void     (* inflate_fast)       (PREFIX3(stream) *strm, uint32_t start);
    void     (* insert_string)      (deflate_state *const s, uint32_t str, uint32_t count);
    uint32_t (* longest_match)      (deflate_state *const s, Pos cur_match);
    uint32_t (* longest_match_slow) (deflate_state *const s, Pos cur_match);
    Pos      (* quick_insert_string)(deflate_state *const s, uint32_t str);
    void     (* slide_hash)         (deflate_state *s);
    uint32_t (* update_hash)        (deflate_state *const s, uint32_t h, uint32_t val);
};

Z_INTERNAL extern struct functable_s functable;

#endif
