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
 * @file
 * Buffered I/O operations
 */

#include <stdint.h>

#include "libavutil/common.h"
#include "libavutil/log.h"

#include "libavformat/version.h"


#define AVIO_SEEKABLE_NORMAL 0x0001 /**< Seeking works like for a local file */

/**
 * Bytestream IO Context.
 * New fields can be added to the end with minor version bumps.
 * Removal, reordering and changes to existing fields require a major
 * version bump.
 * sizeof(AVIOContext) must not be used outside libav*.
 *
 * @note None of the function pointers in AVIOContext should be called
 *       directly, they should only be set by the client application
 *       when implementing custom I/O. Normally these are set to the
 *       function pointers specified in avio_alloc_context()
 */
typedef struct {
    unsigned char *buffer;  /**< Start of the buffer. */
    int buffer_size;        /**< Maximum buffer size */
    unsigned char *buf_ptr; /**< Current position in the buffer */
    unsigned char *buf_end; /**< End of the data, may be less than
                                 buffer+buffer_size if the read function returned
                                 less data than requested, e.g. for streams where
                                 no more data has been received yet. */
    void *opaque;           /**< A private pointer, passed to the read/write/seek/...
                                 functions. */
    int (*read_packet)(void *opaque, uint8_t *buf, int buf_size);
    int (*write_packet)(void *opaque, uint8_t *buf, int buf_size);
    int64_t (*seek)(void *opaque, int64_t offset, int whence);
    int64_t pos;            /**< position in the file of the current buffer */
    int must_flush;         /**< true if the next seek should flush */
    int eof_reached;        /**< true if eof reached */
    int write_flag;         /**< true if open for writing */
#if FF_API_OLD_AVIO
    attribute_deprecated int is_streamed;
#endif
    int max_packet_size;
    unsigned long checksum;
    unsigned char *checksum_ptr;
    unsigned long (*update_checksum)(unsigned long checksum, const uint8_t *buf, unsigned int size);
    int error;              /**< contains the error code or 0 if no error happened */
    /**
     * Pause or resume playback for network streaming protocols - e.g. MMS.
     */
    int (*read_pause)(void *opaque, int pause);
    /**
     * Seek to a given timestamp in stream with the specified stream_index.
     * Needed for some network streaming protocols which don't support seeking
     * to byte position.
     */
    int64_t (*read_seek)(void *opaque, int stream_index,
                         int64_t timestamp, int flags);
    /**
     * A combination of AVIO_SEEKABLE_ flags or 0 when the stream is not seekable.
     */
    int seekable;
} AVIOContext;

/* unbuffered I/O */

#if FF_API_OLD_AVIO
/**
 * URL Context.
 * New fields can be added to the end with minor version bumps.
 * Removal, reordering and changes to existing fields require a major
 * version bump.
 * sizeof(URLContext) must not be used outside libav*.
 * @deprecated This struct will be made private
 */
typedef struct URLContext {
    const AVClass *av_class; ///< information for av_log(). Set by url_open().
    struct URLProtocol *prot;
    int flags;
    int is_streamed;  /**< true if streamed (no seek possible), default = false */
    int max_packet_size;  /**< if non zero, the stream is packetized with this max packet size */
    void *priv_data;
    char *filename; /**< specified URL */
    int is_connected;
} URLContext;

#define URL_PROTOCOL_FLAG_NESTED_SCHEME 1 /*< The protocol name can be the first part of a nested protocol scheme */

/**
 * @deprecated This struct is to be made private. Use the higher-level
 *             AVIOContext-based API instead.
 */
typedef struct URLProtocol {
    const char *name;
    int (*url_open)(URLContext *h, const char *url, int flags);
    int (*url_read)(URLContext *h, unsigned char *buf, int size);
    int (*url_write)(URLContext *h, const unsigned char *buf, int size);
    int64_t (*url_seek)(URLContext *h, int64_t pos, int whence);
    int (*url_close)(URLContext *h);
    struct URLProtocol *next;
    int (*url_read_pause)(URLContext *h, int pause);
    int64_t (*url_read_seek)(URLContext *h, int stream_index,
                             int64_t timestamp, int flags);
    int (*url_get_file_handle)(URLContext *h);
    int priv_data_size;
    const AVClass *priv_data_class;
    int flags;
    int (*url_check)(URLContext *h, int mask);
} URLProtocol;

typedef struct URLPollEntry {
    URLContext *handle;
    int events;
    int revents;
} URLPollEntry;

/* not implemented */
attribute_deprecated int url_poll(URLPollEntry *poll_table, int n, int timeout);

/**
 * @name URL open modes
 * The flags argument to url_open and cosins must be one of the following
 * constants, optionally ORed with other flags.
 * @{
 */
#define URL_RDONLY 1  /**< read-only */
#define URL_WRONLY 2  /**< write-only */
#define URL_RDWR   (URL_RDONLY|URL_WRONLY)  /**< read-write */
/**
 * @}
 */

/**
 * Use non-blocking mode.
 * If this flag is set, operations on the context will return
 * AVERROR(EAGAIN) if they can not be performed immediately.
 * If this flag is not set, operations on the context will never return
 * AVERROR(EAGAIN).
 * Note that this flag does not affect the opening/connecting of the
 * context. Connecting a protocol will always block if necessary (e.g. on
 * network protocols) but never hang (e.g. on busy devices).
 * Warning: non-blocking protocols is work-in-progress; this flag may be
 * silently ignored.
 */
#define URL_FLAG_NONBLOCK 4

typedef int URLInterruptCB(void);
extern URLInterruptCB *url_interrupt_cb;

/**
 * @defgroup old_url_funcs Old url_* functions
 * The following functions are deprecated. Use the buffered API based on #AVIOContext instead.
 * @{
 */
attribute_deprecated int url_open_protocol (URLContext **puc, struct URLProtocol *up,
                                            const char *url, int flags);
attribute_deprecated int url_alloc(URLContext **h, const char *url, int flags);
attribute_deprecated int url_connect(URLContext *h);
attribute_deprecated int url_open(URLContext **h, const char *url, int flags);
attribute_deprecated int url_read(URLContext *h, unsigned char *buf, int size);
attribute_deprecated int url_read_complete(URLContext *h, unsigned char *buf, int size);
attribute_deprecated int url_write(URLContext *h, const unsigned char *buf, int size);
attribute_deprecated int64_t url_seek(URLContext *h, int64_t pos, int whence);
attribute_deprecated int url_close(URLContext *h);
attribute_deprecated int64_t url_filesize(URLContext *h);
attribute_deprecated int url_get_file_handle(URLContext *h);
attribute_deprecated int url_get_max_packet_size(URLContext *h);
attribute_deprecated void url_get_filename(URLContext *h, char *buf, int buf_size);
attribute_deprecated int av_url_read_pause(URLContext *h, int pause);
attribute_deprecated int64_t av_url_read_seek(URLContext *h, int stream_index,
                                              int64_t timestamp, int flags);
attribute_deprecated void url_set_interrupt_cb(int (*interrupt_cb)(void));

/**
 * returns the next registered protocol after the given protocol (the first if
 * NULL is given), or NULL if protocol is the last one.
 */
URLProtocol *av_protocol_next(URLProtocol *p);

/**
 * Register the URLProtocol protocol.
 *
 * @param size the size of the URLProtocol struct referenced
 */
attribute_deprecated int av_register_protocol2(URLProtocol *protocol, int size);
/**
 * @}
 */


typedef attribute_deprecated AVIOContext ByteIOContext;

attribute_deprecated int init_put_byte(AVIOContext *s,
                  unsigned char *buffer,
                  int buffer_size,
                  int write_flag,
                  void *opaque,
                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int64_t (*seek)(void *opaque, int64_t offset, int whence));
attribute_deprecated AVIOContext *av_alloc_put_byte(
                  unsigned char *buffer,
                  int buffer_size,
                  int write_flag,
                  void *opaque,
                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int64_t (*seek)(void *opaque, int64_t offset, int whence));

/**
 * @defgroup old_avio_funcs Old put_/get_*() functions
 * The following functions are deprecated. Use the "avio_"-prefixed functions instead.
 * @{
 */
attribute_deprecated int          get_buffer(AVIOContext *s, unsigned char *buf, int size);
attribute_deprecated int          get_partial_buffer(AVIOContext *s, unsigned char *buf, int size);
attribute_deprecated int          get_byte(AVIOContext *s);
attribute_deprecated unsigned int get_le16(AVIOContext *s);
attribute_deprecated unsigned int get_le24(AVIOContext *s);
attribute_deprecated unsigned int get_le32(AVIOContext *s);
attribute_deprecated uint64_t     get_le64(AVIOContext *s);
attribute_deprecated unsigned int get_be16(AVIOContext *s);
attribute_deprecated unsigned int get_be24(AVIOContext *s);
attribute_deprecated unsigned int get_be32(AVIOContext *s);
attribute_deprecated uint64_t     get_be64(AVIOContext *s);

attribute_deprecated void         put_byte(AVIOContext *s, int b);
attribute_deprecated void         put_nbyte(AVIOContext *s, int b, int count);
attribute_deprecated void         put_buffer(AVIOContext *s, const unsigned char *buf, int size);
attribute_deprecated void         put_le64(AVIOContext *s, uint64_t val);
attribute_deprecated void         put_be64(AVIOContext *s, uint64_t val);
attribute_deprecated void         put_le32(AVIOContext *s, unsigned int val);
attribute_deprecated void         put_be32(AVIOContext *s, unsigned int val);
attribute_deprecated void         put_le24(AVIOContext *s, unsigned int val);
attribute_deprecated void         put_be24(AVIOContext *s, unsigned int val);
attribute_deprecated void         put_le16(AVIOContext *s, unsigned int val);
attribute_deprecated void         put_be16(AVIOContext *s, unsigned int val);
attribute_deprecated void         put_tag(AVIOContext *s, const char *tag);
/**
 * @}
 */

attribute_deprecated int     av_url_read_fpause(AVIOContext *h,    int pause);
attribute_deprecated int64_t av_url_read_fseek (AVIOContext *h,    int stream_index,
                                                int64_t timestamp, int flags);

/**
 * @defgroup old_url_f_funcs Old url_f* functions
 * The following functions are deprecated, use the "avio_"-prefixed functions instead.
 * @{
 */
attribute_deprecated int url_fopen( AVIOContext **s, const char *url, int flags);
attribute_deprecated int url_fclose(AVIOContext *s);
attribute_deprecated int64_t url_fseek(AVIOContext *s, int64_t offset, int whence);
attribute_deprecated int url_fskip(AVIOContext *s, int64_t offset);
attribute_deprecated int64_t url_ftell(AVIOContext *s);
attribute_deprecated int64_t url_fsize(AVIOContext *s);
#define URL_EOF (-1)
attribute_deprecated int url_fgetc(AVIOContext *s);
attribute_deprecated int url_setbufsize(AVIOContext *s, int buf_size);
#ifdef __GNUC__
attribute_deprecated int url_fprintf(AVIOContext *s, const char *fmt, ...) __attribute__ ((__format__ (__printf__, 2, 3)));
#else
attribute_deprecated int url_fprintf(AVIOContext *s, const char *fmt, ...);
#endif
attribute_deprecated void put_flush_packet(AVIOContext *s);
attribute_deprecated int url_open_dyn_buf(AVIOContext **s);
attribute_deprecated int url_open_dyn_packet_buf(AVIOContext **s, int max_packet_size);
attribute_deprecated int url_close_dyn_buf(AVIOContext *s, uint8_t **pbuffer);
attribute_deprecated int url_fdopen(AVIOContext **s, URLContext *h);
/**
 * @}
 */

attribute_deprecated int url_ferror(AVIOContext *s);

attribute_deprecated int udp_set_remote_url(URLContext *h, const char *uri);
attribute_deprecated int udp_get_local_port(URLContext *h);

attribute_deprecated void init_checksum(AVIOContext *s,
                   unsigned long (*update_checksum)(unsigned long c, const uint8_t *p, unsigned int len),
                   unsigned long checksum);
attribute_deprecated unsigned long get_checksum(AVIOContext *s);
attribute_deprecated void put_strz(AVIOContext *s, const char *buf);
/** @note unlike fgets, the EOL character is not returned and a whole
    line is parsed. return NULL if first char read was EOF */
attribute_deprecated char *url_fgets(AVIOContext *s, char *buf, int buf_size);
/**
 * @deprecated use avio_get_str instead
 */
attribute_deprecated char *get_strz(AVIOContext *s, char *buf, int maxlen);
/**
 * @deprecated Use AVIOContext.seekable field directly.
 */
attribute_deprecated static inline int url_is_streamed(AVIOContext *s)
{
    return !s->seekable;
}
attribute_deprecated URLContext *url_fileno(AVIOContext *s);

/**
 * @deprecated use AVIOContext.max_packet_size directly.
 */
attribute_deprecated int url_fget_max_packet_size(AVIOContext *s);

attribute_deprecated int url_open_buf(AVIOContext **s, uint8_t *buf, int buf_size, int flags);

/** return the written or read size */
attribute_deprecated int url_close_buf(AVIOContext *s);

/**
 * Return a non-zero value if the resource indicated by url
 * exists, 0 otherwise.
 * @deprecated Use avio_check instead.
 */
attribute_deprecated int url_exist(const char *url);
#endif // FF_API_OLD_AVIO

/**
 * Return AVIO_FLAG_* access flags corresponding to the access permissions
 * of the resource in url, or a negative value corresponding to an
 * AVERROR code in case of failure. The returned access flags are
 * masked by the value in flags.
 *
 * @note This function is intrinsically unsafe, in the sense that the
 * checked resource may change its existence or permission status from
 * one call to another. Thus you should not trust the returned value,
 * unless you are sure that no other processes are accessing the
 * checked resource.
 */
int avio_check(const char *url, int flags);

/**
 * The callback is called in blocking functions to test regulary if
 * asynchronous interruption is needed. AVERROR_EXIT is returned
 * in this case by the interrupted function. 'NULL' means no interrupt
 * callback is given.
 */
void avio_set_interrupt_cb(int (*interrupt_cb)(void));

/**
 * Allocate and initialize an AVIOContext for buffered I/O. It must be later
 * freed with av_free().
 *
 * @param buffer Memory block for input/output operations via AVIOContext.
 *        The buffer must be allocated with av_malloc() and friends.
 * @param buffer_size The buffer size is very important for performance.
 *        For protocols with fixed blocksize it should be set to this blocksize.
 *        For others a typical size is a cache page, e.g. 4kb.
 * @param write_flag Set to 1 if the buffer should be writable, 0 otherwise.
 * @param opaque An opaque pointer to user-specific data.
 * @param read_packet  A function for refilling the buffer, may be NULL.
 * @param write_packet A function for writing the buffer contents, may be NULL.
 * @param seek A function for seeking to specified byte position, may be NULL.
 *
 * @return Allocated AVIOContext or NULL on failure.
 */
AVIOContext *avio_alloc_context(
                  unsigned char *buffer,
                  int buffer_size,
                  int write_flag,
                  void *opaque,
                  int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),
                  int64_t (*seek)(void *opaque, int64_t offset, int whence));

void avio_w8(AVIOContext *s, int b);
void avio_write(AVIOContext *s, const unsigned char *buf, int size);
void avio_wl64(AVIOContext *s, uint64_t val);
void avio_wb64(AVIOContext *s, uint64_t val);
void avio_wl32(AVIOContext *s, unsigned int val);
void avio_wb32(AVIOContext *s, unsigned int val);
void avio_wl24(AVIOContext *s, unsigned int val);
void avio_wb24(AVIOContext *s, unsigned int val);
void avio_wl16(AVIOContext *s, unsigned int val);
void avio_wb16(AVIOContext *s, unsigned int val);

/**
 * Write a NULL-terminated string.
 * @return number of bytes written.
 */
int avio_put_str(AVIOContext *s, const char *str);

/**
 * Convert an UTF-8 string to UTF-16LE and write it.
 * @return number of bytes written.
 */
int avio_put_str16le(AVIOContext *s, const char *str);

/**
 * Passing this as the "whence" parameter to a seek function causes it to
 * return the filesize without seeking anywhere. Supporting this is optional.
 * If it is not supported then the seek function will return <0.
 */
#define AVSEEK_SIZE 0x10000

/**
 * Oring this flag as into the "whence" parameter to a seek function causes it to
 * seek by any means (like reopening and linear reading) or other normally unreasonble
 * means that can be extreemly slow.
 * This may be ignored by the seek code.
 */
#define AVSEEK_FORCE 0x20000

/**
 * fseek() equivalent for AVIOContext.
 * @return new position or AVERROR.
 */
int64_t avio_seek(AVIOContext *s, int64_t offset, int whence);

/**
 * Skip given number of bytes forward
 * @return new position or AVERROR.
 */
int64_t avio_skip(AVIOContext *s, int64_t offset);

/**
 * ftell() equivalent for AVIOContext.
 * @return position or AVERROR.
 */
static av_always_inline int64_t avio_tell(AVIOContext *s)
{
    return avio_seek(s, 0, SEEK_CUR);
}

/**
 * Get the filesize.
 * @return filesize or AVERROR
 */
int64_t avio_size(AVIOContext *s);

/**
 * feof() equivalent for AVIOContext.
 * @return non zero if and only if end of file
 */
int url_feof(AVIOContext *s);

/** @warning currently size is limited */
#ifdef __GNUC__
int avio_printf(AVIOContext *s, const char *fmt, ...) __attribute__ ((__format__ (__printf__, 2, 3)));
#else
int avio_printf(AVIOContext *s, const char *fmt, ...);
#endif

void avio_flush(AVIOContext *s);


/**
 * Read size bytes from AVIOContext into buf.
 * @return number of bytes read or AVERROR
 */
int avio_read(AVIOContext *s, unsigned char *buf, int size);

/**
 * @name Functions for reading from AVIOContext
 * @{
 *
 * @note return 0 if EOF, so you cannot use it if EOF handling is
 *       necessary
 */
int          avio_r8  (AVIOContext *s);
unsigned int avio_rl16(AVIOContext *s);
unsigned int avio_rl24(AVIOContext *s);
unsigned int avio_rl32(AVIOContext *s);
uint64_t     avio_rl64(AVIOContext *s);
unsigned int avio_rb16(AVIOContext *s);
unsigned int avio_rb24(AVIOContext *s);
unsigned int avio_rb32(AVIOContext *s);
uint64_t     avio_rb64(AVIOContext *s);
/**
 * @}
 */

/**
 * Read a string from pb into buf. The reading will terminate when either
 * a NULL character was encountered, maxlen bytes have been read, or nothing
 * more can be read from pb. The result is guaranteed to be NULL-terminated, it
 * will be truncated if buf is too small.
 * Note that the string is not interpreted or validated in any way, it
 * might get truncated in the middle of a sequence for multi-byte encodings.
 *
 * @return number of bytes read (is always <= maxlen).
 * If reading ends on EOF or error, the return value will be one more than
 * bytes actually read.
 */
int avio_get_str(AVIOContext *pb, int maxlen, char *buf, int buflen);

/**
 * Read a UTF-16 string from pb and convert it to UTF-8.
 * The reading will terminate when either a null or invalid character was
 * encountered or maxlen bytes have been read.
 * @return number of bytes read (is always <= maxlen)
 */
int avio_get_str16le(AVIOContext *pb, int maxlen, char *buf, int buflen);
int avio_get_str16be(AVIOContext *pb, int maxlen, char *buf, int buflen);


/**
 * @name URL open modes
 * The flags argument to avio_open must be one of the following
 * constants, optionally ORed with other flags.
 * @{
 */
#define AVIO_FLAG_READ  1                                      /**< read-only */
#define AVIO_FLAG_WRITE 2                                      /**< write-only */
#define AVIO_FLAG_READ_WRITE (AVIO_FLAG_READ|AVIO_FLAG_WRITE)  /**< read-write pseudo flag */
/**
 * @}
 */

/**
 * Use non-blocking mode.
 * If this flag is set, operations on the context will return
 * AVERROR(EAGAIN) if they can not be performed immediately.
 * If this flag is not set, operations on the context will never return
 * AVERROR(EAGAIN).
 * Note that this flag does not affect the opening/connecting of the
 * context. Connecting a protocol will always block if necessary (e.g. on
 * network protocols) but never hang (e.g. on busy devices).
 * Warning: non-blocking protocols is work-in-progress; this flag may be
 * silently ignored.
 */
#define AVIO_FLAG_NONBLOCK 8

/**
 * Create and initialize a AVIOContext for accessing the
 * resource indicated by url.
 * @note When the resource indicated by url has been opened in
 * read+write mode, the AVIOContext can be used only for writing.
 *
 * @param s Used to return the pointer to the created AVIOContext.
 * In case of failure the pointed to value is set to NULL.
 * @param flags flags which control how the resource indicated by url
 * is to be opened
 * @return 0 in case of success, a negative value corresponding to an
 * AVERROR code in case of failure
 */
int avio_open(AVIOContext **s, const char *url, int flags);

/**
 * Close the resource accessed by the AVIOContext s and free it.
 * This function can only be used if s was opened by avio_open().
 *
 * @return 0 on success, an AVERROR < 0 on error.
 */
int avio_close(AVIOContext *s);

/**
 * Open a write only memory stream.
 *
 * @param s new IO context
 * @return zero if no error.
 */
int avio_open_dyn_buf(AVIOContext **s);

/**
 * Return the written size and a pointer to the buffer. The buffer
 * must be freed with av_free().
 * Padding of FF_INPUT_BUFFER_PADDING_SIZE is added to the buffer.
 *
 * @param s IO context
 * @param pbuffer pointer to a byte buffer
 * @return the length of the byte buffer
 */
int avio_close_dyn_buf(AVIOContext *s, uint8_t **pbuffer);

/**
 * Iterate through names of available protocols.
 * @note it is recommanded to use av_protocol_next() instead of this
 *
 * @param opaque A private pointer representing current protocol.
 *        It must be a pointer to NULL on first iteration and will
 *        be updated by successive calls to avio_enum_protocols.
 * @param output If set to 1, iterate over output protocols,
 *               otherwise over input protocols.
 *
 * @return A static string containing the name of current protocol or NULL
 */
const char *avio_enum_protocols(void **opaque, int output);

/**
 * Pause and resume playing - only meaningful if using a network streaming
 * protocol (e.g. MMS).
 * @param pause 1 for pause, 0 for resume
 */
int     avio_pause(AVIOContext *h, int pause);

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
int64_t avio_seek_time(AVIOContext *h, int stream_index,
                       int64_t timestamp, int flags);

#endif /* AVFORMAT_AVIO_H */
