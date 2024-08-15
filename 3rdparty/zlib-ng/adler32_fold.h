/* adler32_fold.h -- adler32 folding interface
 * Copyright (C) 2022 Adam Stylinski
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef ADLER32_FOLD_H_
#define ADLER32_FOLD_H_

Z_INTERNAL uint32_t adler32_fold_copy_c(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len);

#endif
