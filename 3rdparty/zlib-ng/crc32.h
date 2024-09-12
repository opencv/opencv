/* crc32.h -- crc32 folding interface
 * Copyright (C) 2021 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#ifndef CRC32_H_
#define CRC32_H_

#define CRC32_FOLD_BUFFER_SIZE (16 * 4)
/* sizeof(__m128i) * (4 folds) */

typedef struct crc32_fold_s {
    uint8_t fold[CRC32_FOLD_BUFFER_SIZE];
    uint32_t value;
} crc32_fold;

#endif
