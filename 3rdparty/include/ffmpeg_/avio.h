/*
 * copyright (c) 2001 Fabrice Bellard
 *
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
#ifndef AVFORMAT_AVIO_H
#define AVFORMAT_AVIO_H

/**
 * @file libavformat/avio.h
 * unbuffered I/O operations
 *
 * @warning This file has to be considered an internal but installed
 * header, so it should not be directly included in your projects.
 */

//#include <stdint.h>

#include "common.h"

/* unbuffered I/O */

/**
 * URL Context.
 * New fields can be added to the end with minor version bumps.
 * Removal, reordering and changes to existing fields require a major
 * version bump.
 * sizeof(URLContext) must not be used outside libav*.
 */
struct URLContext {
#if LIBAVFORMAT_VERSION_MAJOR >= 53
    const AVClass *av_class; ///< information for av_log(). Set by url_open().
#endif
    struct URLProtocol *prot;
    int flags;
    int is_streamed;  /**< true if streamed (no seek possible), default = false */
    int max_packet_size;  /**< if non zero, the stream is packetized with this max packet size */
    void *priv_data;
    char *filename; /**< specified filename */
};

typedef struct URLContext URLContext;

typedef struct URLPollEntry {
    URLContext *handle;
    int events;
    int revents;
} URLPollEntry;

#define URL_RDONLY 0
#define URL_WRONLY 1
#define URL_RDWR   2

typedef int URLInterruptCB(void);

int url_open_protocol (URLContext **puc, struct URLProtocol *up,
                       const char *filename, int flags);
int url_open(URLContext **h, const char *filename, int flags);
int url_read(URLContext *h, unsigned char *buf, int size);
int url_write(URLContext *h, unsigned char *buf, int size);
int64_t url_seek(URLContext *h, int64_t pos, int whence);
int url_close(URLContext *h);
int url_exist(const char *filename);
int64_t url_filesize(URLContext *h);

/**
 * Return the maximum packet size associated to packetized file
 * handle. If the file is not packetized (stream like HTTP or file on
 * disk), then 0 is returned.
 *
 * @param h file handle
 * @return maximum packet size in bytes
 */
int url_get_max_packet_size(URLContext *h);
void url_get_filename(URLContext *h, char *buf, int buf_size);

/**
 * The callback is called in blocking functions to test regulary if
 * asynchronous interruption is needed. AVERROR(EINTR) is returned
 * in this case by the interrupted function. 'NULL' means no interrupt
 * callback is given.
 */
void url_set_interrupt_cb(URLInterruptCB *interrupt_cb);

/* not implemented */
int url_poll(URLPollEntry *poll_table, int n, int timeout);

/**
 * Pause and resume playing - only meaningful if using a network streaming
 * protocol (e.g. MMS).
 * @param pause 1 for pause, 0 for resume
 */
int av_url_read_pause(URLContext *h, int pause);

/**
 * Seek to a given timestamp relative to some component stream.
 * Only meaningful if using a network streaming protocol (e.g. MMS.).
 * @param stream_index The stream index that the timestamp is relative to.
 *        If stream_index is (-1) the timestamp should be in AV_TIME_BASE
 *        units from the beginning of the presentation.
 *        If a stream_index >= 0 is used and the protocol does not support
 *        seeking based on component streams, the call will fail with ENOTSUP.
 * @param timestamp timestamp in AVStream.time_base units
 *        or if there is no stream specified then in AV_TIME_BASE units.
 * @param flags Optional combination of AVSEEK_FLAG_BACKWARD, AVSEEK_FLAG_BYTE
 *        and AVSEEK_FLAG_ANY. The protocol may silently ignore
 *        AVSEEK_FLAG_BACKWARD and AVSEEK_FLAG_ANY, but AVSEEK_FLAG_BYTE will
 *        fail with ENOTSUP if used and not supported.
 * @return >= 0 on success
 * @see AVInputFormat::read_seek
 */
int64_t av_url_read_seek(URLContext *h, int stream_index,
                         int64_t timestamp, int flags);

/**
 * Passing this as the "whence" parameter to a seek function causes it to
 * return the filesize without seeking anywhere. Supporting this is optional.
 * If it is not supported then the seek function will return <0.
 */
#define AVSEEK_SIZE 0x10000

typedef struct URLProtocol {
    const char *name;
    int (*url_open)(URLContext *h, const char *filename, int flags);
    int (*url_read)(URLContext *h, unsigned char *buf, int size);
    int (*url_write)(URLContext *h, unsigned char *buf, int size);
    int64_t (*url_seek)(URLContext *h, int64_t pos, int whence);
    int (*url_close)(URLContext *h);
    struct URLProtocol *next;
    int (*url_read_pause)(URLContext *h, int pause);
    int64_t (*url_read_seek)(URLContext *h, int stream_index,
                             int64_t timestamp, int flags);
} URLProtocol;

#if LIBAVFORMAT_VERSION_MAJOR < 53
extern URLProtocol *first_protocol;
#endif

extern URLInterruptCB *url_interrupt_cb;

/**
 * If protocol is NULL, returns the first registered protocol,
 * if protocol is non-NULL, returns the next registered protocol after protocol,
 * or NULL if protocol is the last one.
 */
URLProtocol *av_protocol_next(URLProtocol *p);

#if LIBAVFORMAT_VERSION_MAJOR < 53
/**
 * @deprecated Use av_register_protocol() instead.
 */
attribute_deprecated int register_protocol(URLProtocol *protocol);
#endif

int av_register_protocol(URLProtocol *protocol);

/**
 * Bytestream IO Context.
 * New fields can be added to the end with minor version bumps.
 * Removal, reordering and changes to existing fields require a major
 * version bump.
 * sizeof(ByteIOContext) must not be used outside libav*.
 */
typedef struct {
    unsigned char *buffer;
    int buffer_size;
    unsigned char *buf_ptr, *buf_end;
    void *opaque;
    int (*read_packet)(void *opaque, uint8_t *buf, int buf_size);
    int (*write_packet)(void *opaque, uint8_t *buf, int buf_size);
    int64_t (*seek)(void *opaque, int64_t offset, int whence);
    int64_t pos; /**< position in the file of the current buffer */
    int must_flush; /**< true if the next seek should flush */
    int eof_reached; /**< true if eof reached */
    int write_flag;  /**< true if open for writing */
    int is_streamed;
    int max_packet_size;
    unsigned long checksum;
    unsigned char *checksum_ptr;
    unsigned long (*update_checksum)(unsigned long checksum, const uint8_t *buf, unsigned int size);
    int error;         ///< contains the error code or 0 if no error happened
    int (*read_pause)(void *opaque, int pause);
    int64_t (*read_seek)(void *opaque, int stream_index,
                         int64_t timestamp, int flags);
} ByteIOContext;

int init_put_byte(ByteIOContext *s,
                  unsigned char *buffer,
                  int buffer_size,
                  int write_flag,
                  void *opaque,
                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int64_t (*seek)(void *opaque, int64_t offset, int whence));
ByteIOContext *av_alloc_put_byte(
                  unsigned char *buffer,
                  int buffer_size,
                  int write_flag,
                  void *opaque,
                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int64_t (*seek)(void *opaque, int64_t offset, int whence));

void put_byte(ByteIOContext *s, int b);
void put_buffer(ByteIOContext *s, const unsigned char *buf, int size);
void put_le64(ByteIOContext *s, uint64_t val);
void put_be64(ByteIOContext *s, uint64_t val);
void put_le32(ByteIOContext *s, unsigned int val);
void put_be32(ByteIOContext *s, unsigned int val);
void put_le24(ByteIOContext *s, unsigned int val);
void put_be24(ByteIOContext *s, unsigned int val);
void put_le16(ByteIOContext *s, unsigned int val);
void put_be16(ByteIOContext *s, unsigned int val);
void put_tag(ByteIOContext *s, const char *tag);

void put_strz(ByteIOContext *s, const char *buf);

/**
 * fseek() equivalent for ByteIOContext.
 * @return new position or AVERROR.
 */
int64_t url_fseek(ByteIOContext *s, int64_t offset, int whence);

/**
 * Skip given number of bytes forward.
 * @param offset number of bytes
 */
void url_fskip(ByteIOContext *s, int64_t offset);

/**
 * ftell() equivalent for ByteIOContext.
 * @return position or AVERROR.
 */
int64_t url_ftell(ByteIOContext *s);

/**
 * Gets the filesize.
 * @return filesize or AVERROR
 */
int64_t url_fsize(ByteIOContext *s);

/**
 * feof() equivalent for ByteIOContext.
 * @return non zero if and only if end of file
 */
int url_feof(ByteIOContext *s);

int url_ferror(ByteIOContext *s);

int av_url_read_fpause(ByteIOContext *h, int pause);
int64_t av_url_read_fseek(ByteIOContext *h, int stream_index,
                          int64_t timestamp, int flags);

#define URL_EOF (-1)
/** @note return URL_EOF (-1) if EOF */
int url_fgetc(ByteIOContext *s);

/** @warning currently size is limited */
#ifdef __GNUC__
int url_fprintf(ByteIOContext *s, const char *fmt, ...) __attribute__ ((__format__ (__printf__, 2, 3)));
#else
int url_fprintf(ByteIOContext *s, const char *fmt, ...);
#endif

/** @note unlike fgets, the EOL character is not returned and a whole
    line is parsed. return NULL if first char read was EOF */
char *url_fgets(ByteIOContext *s, char *buf, int buf_size);

void put_flush_packet(ByteIOContext *s);


/**
 * Reads size bytes from ByteIOContext into buf.
 * @returns number of bytes read or AVERROR
 */
int get_buffer(ByteIOContext *s, unsigned char *buf, int size);

/**
 * Reads size bytes from ByteIOContext into buf.
 * This reads at most 1 packet. If that is not enough fewer bytes will be
 * returned.
 * @returns number of bytes read or AVERROR
 */
int get_partial_buffer(ByteIOContext *s, unsigned char *buf, int size);

/** @note return 0 if EOF, so you cannot use it if EOF handling is
    necessary */
int get_byte(ByteIOContext *s);
unsigned int get_le24(ByteIOContext *s);
unsigned int get_le32(ByteIOContext *s);
uint64_t get_le64(ByteIOContext *s);
unsigned int get_le16(ByteIOContext *s);

char *get_strz(ByteIOContext *s, char *buf, int maxlen);
unsigned int get_be16(ByteIOContext *s);
unsigned int get_be24(ByteIOContext *s);
unsigned int get_be32(ByteIOContext *s);
uint64_t get_be64(ByteIOContext *s);

uint64_t ff_get_v(ByteIOContext *bc);

static inline int url_is_streamed(ByteIOContext *s)
{
    return s->is_streamed;
}

/** @note when opened as read/write, the buffers are only used for
    writing */
int url_fdopen(ByteIOContext **s, URLContext *h);

/** @warning must be called before any I/O */
int url_setbufsize(ByteIOContext *s, int buf_size);
/** Reset the buffer for reading or writing.
 * @note Will drop any data currently in the buffer without transmitting it.
 * @param flags URL_RDONLY to set up the buffer for reading, or URL_WRONLY
 *        to set up the buffer for writing. */
int url_resetbuf(ByteIOContext *s, int flags);

/** @note when opened as read/write, the buffers are only used for
    writing */
int url_fopen(ByteIOContext **s, const char *filename, int flags);
int url_fclose(ByteIOContext *s);
URLContext *url_fileno(ByteIOContext *s);

/**
 * Return the maximum packet size associated to packetized buffered file
 * handle. If the file is not packetized (stream like http or file on
 * disk), then 0 is returned.
 *
 * @param s buffered file handle
 * @return maximum packet size in bytes
 */
int url_fget_max_packet_size(ByteIOContext *s);

int url_open_buf(ByteIOContext **s, uint8_t *buf, int buf_size, int flags);

/** return the written or read size */
int url_close_buf(ByteIOContext *s);

/**
 * Open a write only memory stream.
 *
 * @param s new IO context
 * @return zero if no error.
 */
int url_open_dyn_buf(ByteIOContext **s);

/**
 * Open a write only packetized memory stream with a maximum packet
 * size of 'max_packet_size'.  The stream is stored in a memory buffer
 * with a big endian 4 byte header giving the packet size in bytes.
 *
 * @param s new IO context
 * @param max_packet_size maximum packet size (must be > 0)
 * @return zero if no error.
 */
int url_open_dyn_packet_buf(ByteIOContext **s, int max_packet_size);

/**
 * Return the written size and a pointer to the buffer. The buffer
 *  must be freed with av_free().
 * @param s IO context
 * @param pbuffer pointer to a byte buffer
 * @return the length of the byte buffer
 */
int url_close_dyn_buf(ByteIOContext *s, uint8_t **pbuffer);

unsigned long ff_crc04C11DB7_update(unsigned long checksum, const uint8_t *buf,
                                    unsigned int len);
unsigned long get_checksum(ByteIOContext *s);
void init_checksum(ByteIOContext *s,
                   unsigned long (*update_checksum)(unsigned long c, const uint8_t *p, unsigned int len),
                   unsigned long checksum);

/* udp.c */
int udp_set_remote_url(URLContext *h, const char *uri);
int udp_get_local_port(URLContext *h);
int udp_get_file_handle(URLContext *h);

#endif /* AVFORMAT_AVIO_H */
