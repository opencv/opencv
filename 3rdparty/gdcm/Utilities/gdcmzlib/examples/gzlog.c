/*
 * gzlog.c
 * Copyright (C) 2004 Mark Adler
 * For conditions of distribution and use, see copyright notice in gzlog.h
 * version 1.0, 26 Nov 2004
 *
 */

#include <string.h>             /* memcmp() */
#include <stdlib.h>             /* malloc(), free(), NULL */
#include <sys/types.h>          /* size_t, off_t */
#include <unistd.h>             /* read(), close(), sleep(), ftruncate(), */
                                /* lseek() */
#include <fcntl.h>              /* open() */
#include <sys/file.h>           /* flock() */
#include "zlib.h"               /* deflateInit2(), deflate(), deflateEnd() */

#include "gzlog.h"              /* interface */
#define local static

/* log object structure */
typedef struct {
    int id;                 /* object identifier */
    int fd;                 /* log file descriptor */
    off_t extra;            /* offset of extra "ap" subfield */
    off_t mark_off;         /* offset of marked data */
    off_t last_off;         /* offset of last block */
    unsigned long crc;      /* uncompressed crc */
    unsigned long len;      /* uncompressed length (modulo 2^32) */
    unsigned stored;        /* length of current stored block */
} gz_log;

#define GZLOGID 19334       /* gz_log object identifier */

#define LOCK_RETRY 1            /* retry lock once a second */
#define LOCK_PATIENCE 1200      /* try about twenty minutes before forcing */

/* acquire a lock on a file */
local int lock(int fd)
{
    int patience;

    /* try to lock every LOCK_RETRY seconds for LOCK_PATIENCE seconds */
    patience = LOCK_PATIENCE;
    do {
        if (flock(fd, LOCK_EX + LOCK_NB) == 0)
            return 0;
        (void)sleep(LOCK_RETRY);
        patience -= LOCK_RETRY;
    } while (patience > 0);

    /* we've run out of patience -- give up */
    return -1;
}

/* release lock */
local void unlock(int fd)
{
    (void)flock(fd, LOCK_UN);
}

/* release a log object */
local void log_clean(gz_log *log)
{
    unlock(log->fd);
    (void)close(log->fd);
    free(log);
}

/* read an unsigned long from a byte buffer little-endian */
local unsigned long make_ulg(unsigned char *buf)
{
    int n;
    unsigned long val;

    val = (unsigned long)(*buf++);
    for (n = 8; n < 32; n += 8)
        val += (unsigned long)(*buf++) << n;
    return val;
}

/* read an off_t from a byte buffer little-endian */
local off_t make_off(unsigned char *buf)
{
    int n;
    off_t val;

    val = (off_t)(*buf++);
    for (n = 8; n < 64; n += 8)
        val += (off_t)(*buf++) << n;
    return val;
}

/* write an unsigned long little-endian to byte buffer */
local void dice_ulg(unsigned long val, unsigned char *buf)
{
    int n;

    for (n = 0; n < 4; n++) {
        *buf++ = val & 0xff;
        val >>= 8;
    }
}

/* write an off_t little-endian to byte buffer */
local void dice_off(off_t val, unsigned char *buf)
{
    int n;

    for (n = 0; n < 8; n++) {
        *buf++ = val & 0xff;
        val >>= 8;
    }
}

/* initial, empty gzip file for appending */
local char empty_gz[] = {
    0x1f, 0x8b,                 /* magic gzip id */
    8,                          /* compression method is deflate */
    4,                          /* there is an extra field */
    0, 0, 0, 0,                 /* no modification time provided */
    0, 0xff,                    /* no extra flags, no OS */
    20, 0, 'a', 'p', 16, 0,     /* extra field with "ap" subfield */
    32, 0, 0, 0, 0, 0, 0, 0,    /* offset of uncompressed data */
    32, 0, 0, 0, 0, 0, 0, 0,    /* offset of last block */
    1, 0, 0, 0xff, 0xff,        /* empty stored block (last) */
    0, 0, 0, 0,                 /* crc */
    0, 0, 0, 0                  /* uncompressed length */
};

/* initialize a log object with locking */
void *gzlog_open(char *path)
{
    unsigned xlen;
    unsigned char temp[20];
    unsigned sub_len;
    int good;
    gz_log *log;

    /* allocate log structure */
    log = malloc(sizeof(gz_log));
    if (log == NULL)
        return NULL;
    log->id = GZLOGID;

    /* open file, creating it if necessary, and locking it */
    log->fd = open(path, O_RDWR | O_CREAT, 0600);
    if (log->fd < 0) {
        free(log);
        return NULL;
    }
    if (lock(log->fd)) {
        close(log->fd);
        free(log);
        return NULL;
    }

    /* if file is empty, write new gzip stream */
    if (lseek(log->fd, 0, SEEK_END) == 0) {
        if (write(log->fd, empty_gz, sizeof(empty_gz)) != sizeof(empty_gz)) {
            log_clean(log);
            return NULL;
        }
    }

    /* check gzip header */
    (void)lseek(log->fd, 0, SEEK_SET);
    if (read(log->fd, temp, 12) != 12 || temp[0] != 0x1f ||
        temp[1] != 0x8b || temp[2] != 8 || (temp[3] & 4) == 0) {
        log_clean(log);
        return NULL;
    }

    /* process extra field to find "ap" sub-field */
    xlen = temp[10] + (temp[11] << 8);
    good = 0;
    while (xlen) {
        if (xlen < 4 || read(log->fd, temp, 4) != 4)
            break;
        sub_len = temp[2];
        sub_len += temp[3] << 8;
        xlen -= 4;
        if (memcmp(temp, "ap", 2) == 0 && sub_len == 16) {
            good = 1;
            break;
        }
        if (xlen < sub_len)
            break;
        (void)lseek(log->fd, sub_len, SEEK_CUR);
        xlen -= sub_len;
    }
    if (!good) {
        log_clean(log);
        return NULL;
    }

    /* read in "ap" sub-field */
    log->extra = lseek(log->fd, 0, SEEK_CUR);
    if (read(log->fd, temp, 16) != 16) {
        log_clean(log);
        return NULL;
    }
    log->mark_off = make_off(temp);
    log->last_off = make_off(temp + 8);

    /* get crc, length of gzip file */
    (void)lseek(log->fd, log->last_off, SEEK_SET);
    if (read(log->fd, temp, 13) != 13 ||
        memcmp(temp, "\001\000\000\377\377", 5) != 0) {
        log_clean(log);
        return NULL;
    }
    log->crc = make_ulg(temp + 5);
    log->len = make_ulg(temp + 9);

    /* set up to write over empty last block */
    (void)lseek(log->fd, log->last_off + 5, SEEK_SET);
    log->stored = 0;
    return (void *)log;
}

/* maximum amount to put in a stored block before starting a new one */
#define MAX_BLOCK 16384

/* write a block to a log object */
int gzlog_write(void *obj, char *data, size_t len)
{
    size_t some;
    unsigned char temp[5];
    gz_log *log;

    /* check object */
    log = (gz_log *)obj;
    if (log == NULL || log->id != GZLOGID)
        return 1;

    /* write stored blocks until all of the input is written */
    do {
        some = MAX_BLOCK - log->stored;
        if (some > len)
            some = len;
        if (write(log->fd, data, some) != some)
            return 1;
        log->crc = crc32(log->crc, data, some);
        log->len += some;
        len -= some;
        data += some;
        log->stored += some;

        /* if the stored block is full, end it and start another */
        if (log->stored == MAX_BLOCK) {
            (void)lseek(log->fd, log->last_off, SEEK_SET);
            temp[0] = 0;
            dice_ulg(log->stored + ((unsigned long)(~log->stored) << 16),
                     temp + 1);
            if (write(log->fd, temp, 5) != 5)
                return 1;
            log->last_off = lseek(log->fd, log->stored, SEEK_CUR);
            (void)lseek(log->fd, 5, SEEK_CUR);
            log->stored = 0;
        }
    } while (len);
    return 0;
}

/* recompress the remaining stored deflate data in place */
local int recomp(gz_log *log)
{
    z_stream strm;
    size_t len, max;
    unsigned char *in;
    unsigned char *out;
    unsigned char temp[16];

    /* allocate space and read it all in (it's around 1 MB) */
    len = log->last_off - log->mark_off;
    max = len + (len >> 12) + (len >> 14) + 11;
    out = malloc(max);
    if (out == NULL)
        return 1;
    in = malloc(len);
    if (in == NULL) {
        free(out);
        return 1;
    }
    (void)lseek(log->fd, log->mark_off, SEEK_SET);
    if (read(log->fd, in, len) != len) {
        free(in);
        free(out);
        return 1;
    }

    /* recompress in memory, decoding stored data as we go */
    /* note: this assumes that unsigned is four bytes or more */
    /*       consider not making that assumption */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    if (deflateInit2(&strm, Z_BEST_COMPRESSION, Z_DEFLATED, -15, 8,
        Z_DEFAULT_STRATEGY) != Z_OK) {
        free(in);
        free(out);
        return 1;
    }
    strm.next_in = in;
    strm.avail_out = max;
    strm.next_out = out;
    while (len >= 5) {
        if (strm.next_in[0] != 0)
            break;
        strm.avail_in = strm.next_in[1] + (strm.next_in[2] << 8);
        strm.next_in += 5;
        len -= 5;
        if (strm.avail_in != 0) {
            if (len < strm.avail_in)
                break;
            len -= strm.avail_in;
            (void)deflate(&strm, Z_NO_FLUSH);
            if (strm.avail_in != 0 || strm.avail_out == 0)
                break;
        }
    }
    (void)deflate(&strm, Z_SYNC_FLUSH);
    (void)deflateEnd(&strm);
    free(in);
    if (len != 0 || strm.avail_out == 0) {
        free(out);
        return 1;
    }

    /* overwrite stored data with compressed data */
    (void)lseek(log->fd, log->mark_off, SEEK_SET);
    len = max - strm.avail_out;
    if (write(log->fd, out, len) != len) {
        free(out);
        return 1;
    }
    free(out);

    /* write last empty block, crc, and length */
    log->mark_off = log->last_off = lseek(log->fd, 0, SEEK_CUR);
    temp[0] = 1;
    dice_ulg(0xffffL << 16, temp + 1);
    dice_ulg(log->crc, temp + 5);
    dice_ulg(log->len, temp + 9);
    if (write(log->fd, temp, 13) != 13)
        return 1;

    /* truncate file to discard remaining stored data and old trailer */
    ftruncate(log->fd, lseek(log->fd, 0, SEEK_CUR));

    /* update extra field to point to new last empty block */
    (void)lseek(log->fd, log->extra, SEEK_SET);
    dice_off(log->mark_off, temp);
    dice_off(log->last_off, temp + 8);
    if (write(log->fd, temp, 16) != 16)
        return 1;
    return 0;
}

/* maximum accumulation of stored blocks before compressing */
#define MAX_STORED 1048576

/* close log object */
int gzlog_close(void *obj)
{
    unsigned char temp[8];
    gz_log *log;

    /* check object */
    log = (gz_log *)obj;
    if (log == NULL || log->id != GZLOGID)
        return 1;

    /* go to start of most recent block being written */
    (void)lseek(log->fd, log->last_off, SEEK_SET);

    /* if some stuff was put there, update block */
    if (log->stored) {
        temp[0] = 0;
        dice_ulg(log->stored + ((unsigned long)(~log->stored) << 16),
                 temp + 1);
        if (write(log->fd, temp, 5) != 5)
            return 1;
        log->last_off = lseek(log->fd, log->stored, SEEK_CUR);
    }

    /* write last block (empty) */
    if (write(log->fd, "\001\000\000\377\377", 5) != 5)
        return 1;

    /* write updated crc and uncompressed length */
    dice_ulg(log->crc, temp);
    dice_ulg(log->len, temp + 4);
    if (write(log->fd, temp, 8) != 8)
        return 1;

    /* put offset of that last block in gzip extra block */
    (void)lseek(log->fd, log->extra + 8, SEEK_SET);
    dice_off(log->last_off, temp);
    if (write(log->fd, temp, 8) != 8)
        return 1;

    /* if more than 1 MB stored, then time to compress it */
    if (log->last_off - log->mark_off > MAX_STORED) {
        if (recomp(log))
            return 1;
    }

    /* unlock and close file */
    log_clean(log);
    return 0;
}
