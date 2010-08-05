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
 * error code definitions
 */

#ifndef AVUTIL_ERROR_H
#define AVUTIL_ERROR_H

#include <errno.h>
#include "avutil.h"

/* error handling */
#if EDOM > 0
#define AVERROR(e) (-(e))   ///< Returns a negative error code from a POSIX error code, to return from library functions.
#define AVUNERROR(e) (-(e)) ///< Returns a POSIX error code from a library function error return value.
#else
/* Some platforms have E* and errno already negated. */
#define AVERROR(e) (e)
#define AVUNERROR(e) (e)
#endif

#if LIBAVUTIL_VERSION_MAJOR < 51
#define AVERROR_INVALIDDATA AVERROR(EINVAL)  ///< Invalid data found when processing input
#define AVERROR_IO          AVERROR(EIO)     ///< I/O error
#define AVERROR_NOENT       AVERROR(ENOENT)  ///< No such file or directory
#define AVERROR_NOFMT       AVERROR(EILSEQ)  ///< Unknown format
#define AVERROR_NOMEM       AVERROR(ENOMEM)  ///< Not enough memory
#define AVERROR_NOTSUPP     AVERROR(ENOSYS)  ///< Operation not supported
#define AVERROR_NUMEXPECTED AVERROR(EDOM)    ///< Number syntax expected in filename
#define AVERROR_UNKNOWN     AVERROR(EINVAL)  ///< Unknown error
#endif

#define AVERROR_EOF         AVERROR(EPIPE)   ///< End of file

#define AVERROR_PATCHWELCOME    (-MKTAG('P','A','W','E')) ///< Not yet implemented in FFmpeg, patches welcome

#if LIBAVUTIL_VERSION_MAJOR > 50
#define AVERROR_INVALIDDATA     (-MKTAG('I','N','D','A')) ///< Invalid data found when processing input
#define AVERROR_NUMEXPECTED     (-MKTAG('N','U','E','X')) ///< Number syntax expected in filename
#endif

/**
 * Puts a description of the AVERROR code errnum in errbuf.
 * In case of failure the global variable errno is set to indicate the
 * error. Even in case of failure av_strerror() will print a generic
 * error message indicating the errnum provided to errbuf.
 *
 * @param errbuf_size the size in bytes of errbuf
 * @return 0 on success, a negative value if a description for errnum
 * cannot be found
 */
int av_strerror(int errnum, char *errbuf, size_t errbuf_size);

#endif /* AVUTIL_ERROR_H */
