/* gzlib.c -- zlib functions common to reading and writing gzip files
 * Copyright (C) 2004-2019 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zutil_p.h"
#include "gzguts.h"

#if defined(_WIN32)
#  define LSEEK _lseeki64
#else
#if defined(_LARGEFILE64_SOURCE) && _LFS64_LARGEFILE-0
#  define LSEEK lseek64
#else
#  define LSEEK lseek
#endif
#endif

/* Local functions */
static void gz_reset(gz_state *);
static gzFile gz_open(const void *, int, const char *);

/* Reset gzip file state */
static void gz_reset(gz_state *state) {
    state->x.have = 0;              /* no output data available */
    if (state->mode == GZ_READ) {   /* for reading ... */
        state->eof = 0;             /* not at end of file */
        state->past = 0;            /* have not read past end yet */
        state->how = LOOK;          /* look for gzip header */
    }
    else                            /* for writing ... */
        state->reset = 0;           /* no deflateReset pending */
    state->seek = 0;                /* no seek request pending */
    gz_error(state, Z_OK, NULL);    /* clear error */
    state->x.pos = 0;               /* no uncompressed data yet */
    state->strm.avail_in = 0;       /* no input data yet */
}

/* Open a gzip file either by name or file descriptor. */
static gzFile gz_open(const void *path, int fd, const char *mode) {
    gz_state *state;
    size_t len;
    int oflag;
#ifdef O_CLOEXEC
    int cloexec = 0;
#endif
#ifdef O_EXCL
    int exclusive = 0;
#endif

    /* check input */
    if (path == NULL)
        return NULL;

    /* allocate gzFile structure to return */
    state = (gz_state *)zng_alloc(sizeof(gz_state));
    if (state == NULL)
        return NULL;
    state->size = 0;            /* no buffers allocated yet */
    state->want = GZBUFSIZE;    /* requested buffer size */
    state->msg = NULL;          /* no error message yet */

    /* interpret mode */
    state->mode = GZ_NONE;
    state->level = Z_DEFAULT_COMPRESSION;
    state->strategy = Z_DEFAULT_STRATEGY;
    state->direct = 0;
    while (*mode) {
        if (*mode >= '0' && *mode <= '9') {
            state->level = *mode - '0';
        } else {
            switch (*mode) {
            case 'r':
                state->mode = GZ_READ;
                break;
#ifndef NO_GZCOMPRESS
            case 'w':
                state->mode = GZ_WRITE;
                break;
            case 'a':
                state->mode = GZ_APPEND;
                break;
#endif
            case '+':       /* can't read and write at the same time */
                zng_free(state);
                return NULL;
            case 'b':       /* ignore -- will request binary anyway */
                break;
#ifdef O_CLOEXEC
            case 'e':
                cloexec = 1;
                break;
#endif
#ifdef O_EXCL
            case 'x':
                exclusive = 1;
                break;
#endif
            case 'f':
                state->strategy = Z_FILTERED;
                break;
            case 'h':
                state->strategy = Z_HUFFMAN_ONLY;
                break;
            case 'R':
                state->strategy = Z_RLE;
                break;
            case 'F':
                state->strategy = Z_FIXED;
                break;
            case 'T':
                state->direct = 1;
                break;
            default:        /* could consider as an error, but just ignore */
                {}
            }
        }
        mode++;
    }

    /* must provide an "r", "w", or "a" */
    if (state->mode == GZ_NONE) {
        zng_free(state);
        return NULL;
    }

    /* can't force transparent read */
    if (state->mode == GZ_READ) {
        if (state->direct) {
            zng_free(state);
            return NULL;
        }
        state->direct = 1;      /* for empty file */
    }

    /* save the path name for error messages */
#ifdef WIDECHAR
    if (fd == -2) {
        len = wcstombs(NULL, (const wchar_t *)path, 0);
        if (len == (size_t)-1)
            len = 0;
    } else
#endif
        len = strlen((const char *)path);
    state->path = (char *)malloc(len + 1);
    if (state->path == NULL) {
        zng_free(state);
        return NULL;
    }
#ifdef WIDECHAR
    if (fd == -2)
        if (len) {
            wcstombs(state->path, (const wchar_t *)path, len + 1);
        } else {
            *(state->path) = 0;
        }
    else
#endif
        (void)snprintf(state->path, len + 1, "%s", (const char *)path);

    /* compute the flags for open() */
    oflag =
#ifdef O_LARGEFILE
        O_LARGEFILE |
#endif
#ifdef O_BINARY
        O_BINARY |
#endif
#ifdef O_CLOEXEC
        (cloexec ? O_CLOEXEC : 0) |
#endif
        (state->mode == GZ_READ ?
         O_RDONLY :
         (O_WRONLY | O_CREAT |
#ifdef O_EXCL
          (exclusive ? O_EXCL : 0) |
#endif
          (state->mode == GZ_WRITE ?
           O_TRUNC :
           O_APPEND)));

    /* open the file with the appropriate flags (or just use fd) */
    state->fd = fd > -1 ? fd : (
#if defined(_WIN32)
        fd == -2 ? _wopen((const wchar_t *)path, oflag, 0666) :
#elif __CYGWIN__
        fd == -2 ? open(state->path, oflag, 0666) :
#endif
        open((const char *)path, oflag, 0666));
    if (state->fd == -1) {
        free(state->path);
        zng_free(state);
        return NULL;
    }
    if (state->mode == GZ_APPEND) {
        LSEEK(state->fd, 0, SEEK_END);  /* so gzoffset() is correct */
        state->mode = GZ_WRITE;         /* simplify later checks */
    }

    /* save the current position for rewinding (only if reading) */
    if (state->mode == GZ_READ) {
        state->start = LSEEK(state->fd, 0, SEEK_CUR);
        if (state->start == -1) state->start = 0;
    }

    /* initialize stream */
    gz_reset(state);

    /* return stream */
    return (gzFile)state;
}

/* -- see zlib.h -- */
gzFile Z_EXPORT PREFIX(gzopen)(const char *path, const char *mode) {
    return gz_open(path, -1, mode);
}

#ifdef ZLIB_COMPAT
gzFile Z_EXPORT PREFIX4(gzopen)(const char *path, const char *mode) {
    return gz_open(path, -1, mode);
}
#endif

/* -- see zlib.h -- */
gzFile Z_EXPORT PREFIX(gzdopen)(int fd, const char *mode) {
    char *path;         /* identifier for error messages */
    gzFile gz;

    if (fd == -1 || (path = (char *)malloc(7 + 3 * sizeof(int))) == NULL)
        return NULL;
    (void)snprintf(path, 7 + 3 * sizeof(int), "<fd:%d>", fd); /* for debugging */
    gz = gz_open(path, fd, mode);
    free(path);
    return gz;
}

/* -- see zlib.h -- */
#ifdef WIDECHAR
gzFile Z_EXPORT PREFIX(gzopen_w)(const wchar_t *path, const char *mode) {
    return gz_open(path, -2, mode);
}
#endif

int Z_EXPORT PREFIX(gzclose)(gzFile file) {
#ifndef NO_GZCOMPRESS
    gz_state *state;

    if (file == NULL)
        return Z_STREAM_ERROR;
    state = (gz_state *)file;

    return state->mode == GZ_READ ? PREFIX(gzclose_r)(file) : PREFIX(gzclose_w)(file);
#else
    return PREFIX(gzclose_r)(file);
#endif
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzbuffer)(gzFile file, unsigned size) {
    gz_state *state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return -1;
    state = (gz_state *)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return -1;

    /* make sure we haven't already allocated memory */
    if (state->size != 0)
        return -1;

    /* check and set requested size */
    if ((size << 1) < size)
        return -1;              /* need to be able to double it */
    if (size < 8)
        size = 8;               /* needed to behave well with flushing */
    state->want = size;
    return 0;
}

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzrewind)(gzFile file) {
    gz_state *state;

    /* get internal structure */
    if (file == NULL)
        return -1;
    state = (gz_state *)file;

    /* check that we're reading and that there's no error */
    if (state->mode != GZ_READ || (state->err != Z_OK && state->err != Z_BUF_ERROR))
        return -1;

    /* back up and start over */
    if (LSEEK(state->fd, state->start, SEEK_SET) == -1)
        return -1;
    gz_reset(state);
    return 0;
}

/* -- see zlib.h -- */
z_off64_t Z_EXPORT PREFIX4(gzseek)(gzFile file, z_off64_t offset, int whence) {
    unsigned n;
    z_off64_t ret;
    gz_state *state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return -1;
    state = (gz_state *)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return -1;

    /* check that there's no error */
    if (state->err != Z_OK && state->err != Z_BUF_ERROR)
        return -1;

    /* can only seek from start or relative to current position */
    if (whence != SEEK_SET && whence != SEEK_CUR)
        return -1;

    /* normalize offset to a SEEK_CUR specification */
    if (whence == SEEK_SET)
        offset -= state->x.pos;
    else if (state->seek)
        offset += state->skip;
    state->seek = 0;

    /* if within raw area while reading, just go there */
    if (state->mode == GZ_READ && state->how == COPY && state->x.pos + offset >= 0) {
        ret = LSEEK(state->fd, offset - (z_off64_t)state->x.have, SEEK_CUR);
        if (ret == -1)
            return -1;
        state->x.have = 0;
        state->eof = 0;
        state->past = 0;
        state->seek = 0;
        gz_error(state, Z_OK, NULL);
        state->strm.avail_in = 0;
        state->x.pos += offset;
        return state->x.pos;
    }

    /* calculate skip amount, rewinding if needed for back seek when reading */
    if (offset < 0) {
        if (state->mode != GZ_READ)         /* writing -- can't go backwards */
            return -1;
        offset += state->x.pos;
        if (offset < 0)                     /* before start of file! */
            return -1;
        if (PREFIX(gzrewind)(file) == -1)   /* rewind, then skip to offset */
            return -1;
    }

    /* if reading, skip what's in output buffer (one less gzgetc() check) */
    if (state->mode == GZ_READ) {
        n = GT_OFF(state->x.have) || (z_off64_t)state->x.have > offset ? (unsigned)offset : state->x.have;
        state->x.have -= n;
        state->x.next += n;
        state->x.pos += n;
        offset -= n;
    }

    /* request skip (if not zero) */
    if (offset) {
        state->seek = 1;
        state->skip = offset;
    }
    return state->x.pos + offset;
}

/* -- see zlib.h -- */
#ifdef ZLIB_COMPAT
z_off_t Z_EXPORT PREFIX(gzseek)(gzFile file, z_off_t offset, int whence) {
    z_off64_t ret;

    ret = PREFIX4(gzseek)(file, (z_off64_t)offset, whence);
    return ret == (z_off_t)ret ? (z_off_t)ret : -1;
}
#endif

/* -- see zlib.h -- */
z_off64_t Z_EXPORT PREFIX4(gztell)(gzFile file) {
    gz_state *state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return -1;
    state = (gz_state *)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return -1;

    /* return position */
    return state->x.pos + (state->seek ? state->skip : 0);
}

/* -- see zlib.h -- */
#ifdef ZLIB_COMPAT
z_off_t Z_EXPORT PREFIX(gztell)(gzFile file) {

    z_off64_t ret;

    ret = PREFIX4(gztell)(file);
    return ret == (z_off_t)ret ? (z_off_t)ret : -1;
}
#endif

/* -- see zlib.h -- */
z_off64_t Z_EXPORT PREFIX4(gzoffset)(gzFile file) {
    z_off64_t offset;
    gz_state *state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return -1;
    state = (gz_state *)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return -1;

    /* compute and return effective offset in file */
    offset = LSEEK(state->fd, 0, SEEK_CUR);
    if (offset == -1)
        return -1;
    if (state->mode == GZ_READ)             /* reading */
        offset -= state->strm.avail_in;     /* don't count buffered input */
    return offset;
}

/* -- see zlib.h -- */
#ifdef ZLIB_COMPAT
z_off_t Z_EXPORT PREFIX(gzoffset)(gzFile file) {
    z_off64_t ret;

    ret = PREFIX4(gzoffset)(file);
    return ret == (z_off_t)ret ? (z_off_t)ret : -1;
}
#endif

/* -- see zlib.h -- */
int Z_EXPORT PREFIX(gzeof)(gzFile file) {
    gz_state *state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return 0;
    state = (gz_state *)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return 0;

    /* return end-of-file state */
    return state->mode == GZ_READ ? state->past : 0;
}

/* -- see zlib.h -- */
const char * Z_EXPORT PREFIX(gzerror)(gzFile file, int *errnum) {
    gz_state *state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return NULL;
    state = (gz_state *)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return NULL;

    /* return error information */
    if (errnum != NULL)
        *errnum = state->err;
    return state->err == Z_MEM_ERROR ? "out of memory" : (state->msg == NULL ? "" : state->msg);
}

/* -- see zlib.h -- */
void Z_EXPORT PREFIX(gzclearerr)(gzFile file) {
    gz_state *state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return;
    state = (gz_state *)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return;

    /* clear error and end-of-file */
    if (state->mode == GZ_READ) {
        state->eof = 0;
        state->past = 0;
    }
    gz_error(state, Z_OK, NULL);
}

/* Create an error message in allocated memory and set state->err and
   state->msg accordingly.  Free any previous error message already there.  Do
   not try to free or allocate space if the error is Z_MEM_ERROR (out of
   memory).  Simply save the error message as a static string.  If there is an
   allocation failure constructing the error message, then convert the error to
   out of memory. */
void Z_INTERNAL gz_error(gz_state *state, int err, const char *msg) {
    /* free previously allocated message and clear */
    if (state->msg != NULL) {
        if (state->err != Z_MEM_ERROR)
            free(state->msg);
        state->msg = NULL;
    }

    /* if fatal, set state->x.have to 0 so that the gzgetc() macro fails */
    if (err != Z_OK && err != Z_BUF_ERROR)
        state->x.have = 0;

    /* set error code, and if no message, then done */
    state->err = err;
    if (msg == NULL)
        return;

    /* for an out of memory error, return literal string when requested */
    if (err == Z_MEM_ERROR)
        return;

    /* construct error message with path */
    if ((state->msg = (char *)malloc(strlen(state->path) + strlen(msg) + 3)) == NULL) {
        state->err = Z_MEM_ERROR;
        return;
    }
    (void)snprintf(state->msg, strlen(state->path) + strlen(msg) + 3, "%s%s%s", state->path, ": ", msg);
}
