/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file libavutil/fifo.h
 * a very simple circular buffer FIFO implementation
 */

#ifndef AVUTIL_FIFO_H
#define AVUTIL_FIFO_H

#include <stdint.h>
#include "avutil.h"
#include "common.h"

typedef struct AVFifoBuffer {
    uint8_t *buffer;
    uint8_t *rptr, *wptr, *end;
} AVFifoBuffer;

/**
 * Initializes an AVFifoBuffer.
 * @param *f AVFifoBuffer to initialize
 * @param size of FIFO
 * @return <0 for failure >=0 otherwise
 */
int av_fifo_init(AVFifoBuffer *f, unsigned int size);

/**
 * Frees an AVFifoBuffer.
 * @param *f AVFifoBuffer to free
 */
void av_fifo_free(AVFifoBuffer *f);

/**
 * Returns the amount of data in bytes in the AVFifoBuffer, that is the
 * amount of data you can read from it.
 * @param *f AVFifoBuffer to read from
 * @return size
 */
int av_fifo_size(AVFifoBuffer *f);

/**
 * Reads data from an AVFifoBuffer.
 * @param *f AVFifoBuffer to read from
 * @param *buf data destination
 * @param buf_size number of bytes to read
 */
int av_fifo_read(AVFifoBuffer *f, uint8_t *buf, int buf_size);

/**
 * Feeds data from an AVFifoBuffer to a user-supplied callback.
 * @param *f AVFifoBuffer to read from
 * @param buf_size number of bytes to read
 * @param *func generic read function
 * @param *dest data destination
 */
int av_fifo_generic_read(AVFifoBuffer *f, int buf_size, void (*func)(void*, void*, int), void* dest);

#if LIBAVUTIL_VERSION_MAJOR < 50
/**
 * Writes data into an AVFifoBuffer.
 * @param *f AVFifoBuffer to write to
 * @param *buf data source
 * @param size data size
 */
attribute_deprecated void av_fifo_write(AVFifoBuffer *f, const uint8_t *buf, int size);
#endif

/**
 * Feeds data from a user-supplied callback to an AVFifoBuffer.
 * @param *f AVFifoBuffer to write to
 * @param *src data source
 * @param size number of bytes to write
 * @param *func generic write function; the first parameter is src,
 * the second is dest_buf, the third is dest_buf_size.
 * func must return the number of bytes written to dest_buf, or <= 0 to
 * indicate no more data available to write.
 * If func is NULL, src is interpreted as a simple byte array for source data.
 * @return the number of bytes written to the FIFO
 */
int av_fifo_generic_write(AVFifoBuffer *f, void *src, int size, int (*func)(void*, void*, int));

#if LIBAVUTIL_VERSION_MAJOR < 50
/**
 * Resizes an AVFifoBuffer.
 * @param *f AVFifoBuffer to resize
 * @param size new AVFifoBuffer size in bytes
 * @see av_fifo_realloc2()
 */
attribute_deprecated void av_fifo_realloc(AVFifoBuffer *f, unsigned int size);
#endif

/**
 * Resizes an AVFifoBuffer.
 * @param *f AVFifoBuffer to resize
 * @param size new AVFifoBuffer size in bytes
 * @return <0 for failure, >=0 otherwise
 */
int av_fifo_realloc2(AVFifoBuffer *f, unsigned int size);

/**
 * Reads and discards the specified amount of data from an AVFifoBuffer.
 * @param *f AVFifoBuffer to read from
 * @param size amount of data to read in bytes
 */
void av_fifo_drain(AVFifoBuffer *f, int size);

static inline uint8_t av_fifo_peek(AVFifoBuffer *f, int offs)
{
    uint8_t *ptr = f->rptr + offs;
    if (ptr >= f->end)
        ptr -= f->end - f->buffer;
    return *ptr;
}
#endif /* AVUTIL_FIFO_H */
