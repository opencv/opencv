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
 * @file
 * a very simple circular buffer FIFO implementation
 */

#ifndef AVUTIL_FIFO_H
#define AVUTIL_FIFO_H

#include <stdint.h>

typedef struct AVFifoBuffer {
    uint8_t *buffer;
    uint8_t *rptr, *wptr, *end;
    uint32_t rndx, wndx;
} AVFifoBuffer;

/**
 * Initialize an AVFifoBuffer.
 * @param size of FIFO
 * @return AVFifoBuffer or NULL in case of memory allocation failure
 */
AVFifoBuffer *av_fifo_alloc(unsigned int size);

/**
 * Free an AVFifoBuffer.
 * @param *f AVFifoBuffer to free
 */
void av_fifo_free(AVFifoBuffer *f);

/**
 * Reset the AVFifoBuffer to the state right after av_fifo_alloc, in particular it is emptied.
 * @param *f AVFifoBuffer to reset
 */
void av_fifo_reset(AVFifoBuffer *f);

/**
 * Return the amount of data in bytes in the AVFifoBuffer, that is the
 * amount of data you can read from it.
 * @param *f AVFifoBuffer to read from
 * @return size
 */
int av_fifo_size(AVFifoBuffer *f);

/**
 * Return the amount of space in bytes in the AVFifoBuffer, that is the
 * amount of data you can write into it.
 * @param *f AVFifoBuffer to write into
 * @return size
 */
int av_fifo_space(AVFifoBuffer *f);

/**
 * Feed data from an AVFifoBuffer to a user-supplied callback.
 * @param *f AVFifoBuffer to read from
 * @param buf_size number of bytes to read
 * @param *func generic read function
 * @param *dest data destination
 */
int av_fifo_generic_read(AVFifoBuffer *f, void *dest, int buf_size, void (*func)(void*, void*, int));

/**
 * Feed data from a user-supplied callback to an AVFifoBuffer.
 * @param *f AVFifoBuffer to write to
 * @param *src data source; non-const since it may be used as a
 * modifiable context by the function defined in func
 * @param size number of bytes to write
 * @param *func generic write function; the first parameter is src,
 * the second is dest_buf, the third is dest_buf_size.
 * func must return the number of bytes written to dest_buf, or <= 0 to
 * indicate no more data available to write.
 * If func is NULL, src is interpreted as a simple byte array for source data.
 * @return the number of bytes written to the FIFO
 */
int av_fifo_generic_write(AVFifoBuffer *f, void *src, int size, int (*func)(void*, void*, int));

/**
 * Resize an AVFifoBuffer.
 * @param *f AVFifoBuffer to resize
 * @param size new AVFifoBuffer size in bytes
 * @return <0 for failure, >=0 otherwise
 */
int av_fifo_realloc2(AVFifoBuffer *f, unsigned int size);

/**
 * Read and discard the specified amount of data from an AVFifoBuffer.
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
