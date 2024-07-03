/* gzwrite.c -- zlib functions for writing gzip files
 * Copyright (C) 2004-2019 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zutil_p.h"
#include <stdarg.h>
#include "gzguts.h"

/* Local functions */
static int gz_init(gz_state *);
static int gz_comp(gz_state *, int);
static int gz_zero(gz_state *, z_off64_t);
static size_t gz_write(gz_state *, void const *, size_t);

/* Initialize state for writing a gzip file.  Mark initialization by setting
   state->size to non-zero.  Return -1 on a memory allocation failure, or 0 on
   success. */
static int gz_init(gz_state *state) {
    int ret;
    PREFIX3(stream) *strm = &(state->strm);

    /* allocate input buffer (double size for gzprintf) */
    state->in = (unsigned char *)zng_alloc(state->want << 1);
    if (state->in == NULL) {
        gz_error(state, Z_MEM_ERROR, "out of memory");
        return -1;
    }
    memset(state->in, 0, state->want << 1);

    /* only need output buffer and deflate state if compressing */
    if (!state->direct) {
        /* allocate output buffer */
        state->out = (unsigned char *)zng_alloc(state->want);
        if (state->out == NULL) {
            zng_free(state->in);
            gz_error(state, Z_MEM_ERROR, "out of memory");
            return -1;
        }

        /* allocate deflate memory, set up for gzip compression */
        strm->zalloc = NULL;
        strm->zfree = NULL;
        strm->opaque = NULL;
        ret = PREFIX(deflateInit2)(strm, state->level, Z_DEFLATED, MAX_WBITS + 16, DEF_MEM_LEVEL, state->strategy);
        if (ret != Z_OK) {
            zng_free(state->out);
            zng_free(state->in);
            gz_error(state, Z_MEM_ERROR, "out of memory");
            return -1;
        }
        strm->next_in = NULL;
    }

    /* mark state as initialized */
    state->size = state->want;

    /* initialize write buffer if compressing */
    if (!state->direct) {
        strm->avail_out = state->size;
        strm->next_out = state->out;
        state->x.next = strm->next_out;
    }
    return 0;
}

/* Compress whatever is at avail_in and next_in and write to the output file.
   Return -1 if there is an error writing to the output file or if gz_init()
   fails to allocate memory, otherwise 0.  flush is assumed to be a valid
   deflate() flush value.  If flush is Z_FINISH, then the deflate() state is
   reset to start a new gzip stream.  If gz->direct is true, then simply write
   to the output file without compressing, and ignore flush. */
static int gz_comp(gz_state *state, int flush) {
    int ret;
    ssize_t got;
    unsigned have;
    PREFIX3(stream) *strm = &(state->strm);

    /* allocate memory if this is the first time through */
    if (state->size == 0 && gz_init(state) == -1)
        return -1;

    /* write directly if requested */
    if (state->direct) {
        got = write(state->fd, strm->next_in, strm->avail_in);
        if (got < 0 || (unsigned)got != strm->avail_in) {
            gz_error(state, Z_ERRNO, zstrerror());
            return -1;
        }
        strm->avail_in = 0;
        return 0;
    }

    /* check for a pending reset */
    if (state->reset) {
        /* don't start a new gzip member unless there is data to write */
        if (strm->avail_in == 0)
            return 0;
        PREFIX(deflateReset)(strm);
        state->reset = 0;
    }

    /* run deflate() on provided input until it produces no more output */
    ret = Z_OK;
    do {
        /* write out current buffer contents if full, or if flushing, but if
           doing Z_FINISH then don't write until we get to Z_STREAM_END */
        if (strm->avail_out == 0 || (flush != Z_NO_FLUSH && (flush != Z_FINISH || ret == Z_STREAM_END))) {
            have = (unsigned)(strm->next_out - state->x.next);
            if (have && ((got = write(state->fd, state->x.next, (unsigned long)have)) < 0 || (unsigned)got != have)) {
                gz_error(state, Z_ERRNO, zstrerror());
                return -1;
            }
            if (strm->avail_out == 0) {
                strm->avail_out = state->size;
                strm->next_out = state->out;
                state->x.next = state->out;
            }
            state->x.next = strm->next_out;
        }

        /* compress */
        have = strm->avail_out;
        ret = PREFIX(deflate)(strm, flush);
        if (ret == Z_STREAM_ERROR) {
            gz_error(state, Z_STREAM_ERROR, "internal error: deflate stream corrupt");
            return -1;
        }
        have -= strm->avail_out;
    } while (have);

    /* if that completed a deflate stream, allow another to start */
    if (flush == Z_FINISH)
        state->reset = 1;
    /* all done, no errors */
    return 0;
}

/* Compress len zeros to output.  Return -1 on a write error or memory
   allocation failure by gz_comp(), or 0 on success. */
static int gz_zero(gz_state *state, z_off64_t len) {
    int first;
    unsigned n;
    PREFIX3(stream) *strm = &(state->strm);

    /* consume whatever's left in the input buffer */
    if (strm->avail_in && gz_comp(state, Z_NO_FLUSH) == -1)
        return -1;

    /* compress len zeros (len guaranteed > 0) */
    first = 1;
    while (len) {
        n = GT_OFF(state->size) || (z_off64_t)state->size > len ? (unsigned)len : state->size;
        if (first) {
            memset(state->in, 0, n);
            first = 0;
        }
        strm->avail_in = n;
        strm->next_in = state->in;
        state->x.pos += n;
        if (gz_comp(state, Z_NO_FLUSH) == -1)
            return -1;
        len -= n;
    }
    return 0;
}

/* Write len bytes from buf to file.  Return the number of bytes written.  If
   the returned value is less than len, then there was an error. */
static size_t gz_write(gz_state *state, void const *buf, size_t len) {
    size_t put = len;

    /* if len is zero, avoid unnecessary operations */
    if (len == 0)
        return 0;

    /* allocate memory if this is the first time through */
    if (state->size == 0 && gz_init(state) == -1)
        return 0;

    /* check for seek request */
    if (state->seek) {
        state->seek = 0;
        if (gz_zero(state, state->skip) == -1)
            return 0;
    }

    /* for small len, copy to input buffer, otherwise compress directly */
    if (len < state->size) {
        /* copy to input buffer, compress when full */
        do {
            unsigned have, copy;

            if (state->strm.avail_in == 0)
                state->strm.next_in = state->in;
            have = (unsigned)((state->strm.next_in + state->strm.avail_in) -
                              state->in);
            copy = state->size - have;
            if (copy > len)
                copy = (unsigned)len;
            memcpy(state->in + have, buf, copy);
            state->strm.avail_in += copy;
            state->x.pos += copy;
            buf = (const char *)buf + copy;
            len -= copy;
            if (len && gz_comp(state, Z_NO_FLUSH) == -1)
                return 0;
        } while (len);
    } else {
        /* consume whatever's left in the input buffer */
        if (state->strm.avail_in && gz_comp(state, Z_NO_FLUSH) == -1)
            return 0;

        /* directly compress user buffer to file */
        state->strm.next_in = (z_const unsigned char *) buf;
        do {
            unsigned n = (unsigned)-1;
            if (n > len)
                n = (unsigned)len;
            state->strm.avail_in = n;
            state->x.pos += n;
            if (gz_comp(state, Z_NO_FLUSH) == -1)
                return 0;
            len -= n;
        } while (len);
    }

    /* input was all buffered or compressed */
    return put;
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzwrite)(gzFile file, void const *buf, unsigned len) {
    gz_state *state;

    /* get internal structure */
    if (file == NULL)
        return 0;
    state = (gz_state *)file;

    /* check that we're writing and that there's no error */
    if (state->mode != GZ_WRITE || state->err != Z_OK)
        return 0;

    /* since an int is returned, make sure len fits in one, otherwise return
       with an error (this avoids a flaw in the interface) */
    if ((int)len < 0) {
        gz_error(state, Z_DATA_ERROR, "requested length does not fit in int");
        return 0;
    }

    /* write len bytes from buf (the return value will fit in an int) */
    return (int)gz_write(state, buf, len);
}

/* -- see zlib.h -- */
size_t Z_EXPORT PREFIX(gzfwrite)(void const *buf, size_t size, size_t nitems, gzFile file) {
    size_t len;
    gz_state *state;

    /* Exit early if size is zero, also prevents potential division by zero */
    if (size == 0)
        return 0;

    /* get internal structure */
    if (file == NULL)
        return 0;
    state = (gz_state *)file;

    /* check that we're writing and that there's no error */
    if (state->mode != GZ_WRITE || state->err != Z_OK)
        return 0;

    /* compute bytes to read -- error on overflow */
    len = nitems * size;
    if (len / size != nitems) {
        gz_error(state, Z_STREAM_ERROR, "request does not fit in a size_t");
        return 0;
    }

    /* write len bytes to buf, return the number of full items written */
    return len ? gz_write(state, buf, len) / size : 0;
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzputc)(gzFile file, int c) {
    unsigned have;
    unsigned char buf[1];
    gz_state *state;
    PREFIX3(stream) *strm;

    /* get internal structure */
    if (file == NULL)
        return -1;
    state = (gz_state *)file;
    strm = &(state->strm);

    /* check that we're writing and that there's no error */
    if (state->mode != GZ_WRITE || state->err != Z_OK)
        return -1;

    /* check for seek request */
    if (state->seek) {
        state->seek = 0;
        if (gz_zero(state, state->skip) == -1)
            return -1;
    }

    /* try writing to input buffer for speed (state->size == 0 if buffer not
       initialized) */
    if (state->size) {
        if (strm->avail_in == 0)
            strm->next_in = state->in;
        have = (unsigned)((strm->next_in + strm->avail_in) - state->in);
        if (have < state->size) {
            state->in[have] = (unsigned char)c;
            strm->avail_in++;
            state->x.pos++;
            return c & 0xff;
        }
    }

    /* no room in buffer or not initialized, use gz_write() */
    buf[0] = (unsigned char)c;
    if (gz_write(state, buf, 1) != 1)
        return -1;
    return c & 0xff;
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzputs)(gzFile file, const char *s) {
    size_t len, put;
    gz_state *state;

    /* get internal structure */
    if (file == NULL)
        return -1;
    state = (gz_state *)file;

    /* check that we're writing and that there's no error */
    if (state->mode != GZ_WRITE || state->err != Z_OK)
        return -1;

    /* write string */
    len = strlen(s);
    if ((int)len < 0 || (unsigned)len != len) {
        gz_error(state, Z_STREAM_ERROR, "string length does not fit in int");
        return -1;
    }
    put = gz_write(state, s, len);
    return put < len ? -1 : (int)len;
}

/* -- see zlib.h -- */
int Z_EXPORTVA PREFIX(gzvprintf)(gzFile file, const char *format, va_list va) {
    int len;
    unsigned left;
    char *next;
    gz_state *state;
    PREFIX3(stream) *strm;

    /* get internal structure */
    if (file == NULL)
        return Z_STREAM_ERROR;
    state = (gz_state *)file;
    strm = &(state->strm);

    /* check that we're writing and that there's no error */
    if (state->mode != GZ_WRITE || state->err != Z_OK)
        return Z_STREAM_ERROR;

    /* make sure we have some buffer space */
    if (state->size == 0 && gz_init(state) == -1)
        return state->err;

    /* check for seek request */
    if (state->seek) {
        state->seek = 0;
        if (gz_zero(state, state->skip) == -1)
            return state->err;
    }

    /* do the printf() into the input buffer, put length in len -- the input
       buffer is double-sized just for this function, so there is guaranteed to
       be state->size bytes available after the current contents */
    if (strm->avail_in == 0)
        strm->next_in = state->in;
    next = (char *)(state->in + (strm->next_in - state->in) + strm->avail_in);
    next[state->size - 1] = 0;
    len = vsnprintf(next, state->size, format, va);

    /* check that printf() results fit in buffer */
    if (len == 0 || (unsigned)len >= state->size || next[state->size - 1] != 0)
        return 0;

    /* update buffer and position, compress first half if past that */
    strm->avail_in += (unsigned)len;
    state->x.pos += len;
    if (strm->avail_in >= state->size) {
        left = strm->avail_in - state->size;
        strm->avail_in = state->size;
        if (gz_comp(state, Z_NO_FLUSH) == -1)
            return state->err;
        memmove(state->in, state->in + state->size, left);
        strm->next_in = state->in;
        strm->avail_in = left;
    }
    return len;
}

int Z_EXPORTVA PREFIX(gzprintf)(gzFile file, const char *format, ...) {
    va_list va;
    int ret;

    va_start(va, format);
    ret = PREFIX(gzvprintf)(file, format, va);
    va_end(va);
    return ret;
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzflush)(gzFile file, int flush) {
    gz_state *state;

    /* get internal structure */
    if (file == NULL)
        return Z_STREAM_ERROR;
    state = (gz_state *)file;

    /* check that we're writing and that there's no error */
    if (state->mode != GZ_WRITE || state->err != Z_OK)
        return Z_STREAM_ERROR;

    /* check flush parameter */
    if (flush < 0 || flush > Z_FINISH)
        return Z_STREAM_ERROR;

    /* check for seek request */
    if (state->seek) {
        state->seek = 0;
        if (gz_zero(state, state->skip) == -1)
            return state->err;
    }

    /* compress remaining data with requested flush */
    (void)gz_comp(state, flush);
    return state->err;
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzsetparams)(gzFile file, int level, int strategy) {
    gz_state *state;
    PREFIX3(stream) *strm;

    /* get internal structure */
    if (file == NULL)
        return Z_STREAM_ERROR;
    state = (gz_state *)file;
    strm = &(state->strm);

    /* check that we're writing and that there's no error */
    if (state->mode != GZ_WRITE || state->err != Z_OK || state->direct)
        return Z_STREAM_ERROR;

    /* if no change is requested, then do nothing */
    if (level == state->level && strategy == state->strategy)
        return Z_OK;

    /* check for seek request */
    if (state->seek) {
        state->seek = 0;
        if (gz_zero(state, state->skip) == -1)
            return state->err;
    }

    /* change compression parameters for subsequent input */
    if (state->size) {
        /* flush previous input with previous parameters before changing */
        if (strm->avail_in && gz_comp(state, Z_BLOCK) == -1)
            return state->err;
        PREFIX(deflateParams)(strm, level, strategy);
    }
    state->level = level;
    state->strategy = strategy;
    return Z_OK;
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzclose_w)(gzFile file) {
    int ret = Z_OK;
    gz_state *state;

    /* get internal structure */
    if (file == NULL)
        return Z_STREAM_ERROR;
    state = (gz_state *)file;

    /* check that we're writing */
    if (state->mode != GZ_WRITE)
        return Z_STREAM_ERROR;

    /* check for seek request */
    if (state->seek) {
        state->seek = 0;
        if (gz_zero(state, state->skip) == -1)
            ret = state->err;
    }

    /* flush, free memory, and close file */
    if (gz_comp(state, Z_FINISH) == -1)
        ret = state->err;
    if (state->size) {
        if (!state->direct) {
            (void)PREFIX(deflateEnd)(&(state->strm));
            zng_free(state->out);
        }
        zng_free(state->in);
    }
    gz_error(state, Z_OK, NULL);
    free(state->path);
    if (close(state->fd) == -1)
        ret = Z_ERRNO;
    zng_free(state);
    return ret;
}
