/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OPJ_CIO_H
#define OPJ_CIO_H
/**
@file cio.h
@brief Implementation of a byte input-output process (CIO)

The functions in CIO.C have for goal to realize a byte input / output process.
*/

/** @defgroup CIO CIO - byte input-output stream */
/*@{*/

#include "opj_config_private.h"

/* ----------------------------------------------------------------------- */

#if defined(OPJ_BIG_ENDIAN)
#define opj_write_bytes     opj_write_bytes_BE
#define opj_read_bytes      opj_read_bytes_BE
#define opj_write_double    opj_write_double_BE
#define opj_read_double     opj_read_double_BE
#define opj_write_float     opj_write_float_BE
#define opj_read_float      opj_read_float_BE
#else
#define opj_write_bytes     opj_write_bytes_LE
#define opj_read_bytes      opj_read_bytes_LE
#define opj_write_double    opj_write_double_LE
#define opj_read_double     opj_read_double_LE
#define opj_write_float     opj_write_float_LE
#define opj_read_float      opj_read_float_LE
#endif


#define OPJ_STREAM_STATUS_OUTPUT  0x1U
#define OPJ_STREAM_STATUS_INPUT   0x2U
#define OPJ_STREAM_STATUS_END     0x4U
#define OPJ_STREAM_STATUS_ERROR   0x8U

/**
Byte input-output stream.
*/
typedef struct opj_stream_private {
    /**
     * User data, be it files, ... The actual data depends on the type of the stream.
     */
    void *                  m_user_data;

    /**
     * Pointer to function to free m_user_data (NULL at initialization)
     * when destroying the stream. If pointer is NULL the function is not
     * called and the m_user_data is not freed (even if non-NULL).
     */
    opj_stream_free_user_data_fn        m_free_user_data_fn;

    /**
     * User data length
     */
    OPJ_UINT64              m_user_data_length;

    /**
     * Pointer to actual read function (NULL at the initialization of the cio.
     */
    opj_stream_read_fn      m_read_fn;

    /**
     * Pointer to actual write function (NULL at the initialization of the cio.
     */
    opj_stream_write_fn     m_write_fn;

    /**
     * Pointer to actual skip function (NULL at the initialization of the cio.
     * There is no seek function to prevent from back and forth slow procedures.
     */
    opj_stream_skip_fn      m_skip_fn;

    /**
     * Pointer to actual seek function (if available).
     */
    opj_stream_seek_fn      m_seek_fn;

    /**
     * Actual data stored into the stream if read from. Data is read by chunk of fixed size.
     * you should never access this data directly.
     */
    OPJ_BYTE *                  m_stored_data;

    /**
     * Pointer to the current read data.
     */
    OPJ_BYTE *                  m_current_data;

    /**
    * FIXME DOC.
    */
    OPJ_OFF_T(* m_opj_skip)(struct opj_stream_private *, OPJ_OFF_T,
                            struct opj_event_mgr *);

    /**
    * FIXME DOC.
    */
    OPJ_BOOL(* m_opj_seek)(struct opj_stream_private *, OPJ_OFF_T,
                           struct opj_event_mgr *);

    /**
     * number of bytes containing in the buffer.
     */
    OPJ_SIZE_T          m_bytes_in_buffer;

    /**
     * The number of bytes read/written from the beginning of the stream
     */
    OPJ_OFF_T           m_byte_offset;

    /**
     * The size of the buffer.
     */
    OPJ_SIZE_T          m_buffer_size;

    /**
     * Flags to tell the status of the stream.
     * Used with OPJ_STREAM_STATUS_* defines.
     */
    OPJ_UINT32 m_status;

}
opj_stream_private_t;

/** @name Exported functions (see also openjpeg.h) */
/*@{*/
/* ----------------------------------------------------------------------- */
/**
 * Write some bytes to the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer      pointer the data buffer to write data to.
 * @param p_value       the value to write
 * @param p_nb_bytes    the number of bytes to write
*/
void opj_write_bytes_BE(OPJ_BYTE * p_buffer, OPJ_UINT32 p_value,
                        OPJ_UINT32 p_nb_bytes);

/**
 * Reads some bytes from the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer      pointer the data buffer to read data from.
 * @param p_value       pointer to the value that will store the data.
 * @param p_nb_bytes    the nb bytes to read.
 * @return              the number of bytes read or -1 if an error occurred.
 */
void opj_read_bytes_BE(const OPJ_BYTE * p_buffer, OPJ_UINT32 * p_value,
                       OPJ_UINT32 p_nb_bytes);

/**
 * Write some bytes to the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer      pointer the data buffer to write data to.
 * @param p_value       the value to write
 * @param p_nb_bytes    the number of bytes to write
 * @return              the number of bytes written or -1 if an error occurred
*/
void opj_write_bytes_LE(OPJ_BYTE * p_buffer, OPJ_UINT32 p_value,
                        OPJ_UINT32 p_nb_bytes);

/**
 * Reads some bytes from the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer      pointer the data buffer to read data from.
 * @param p_value       pointer to the value that will store the data.
 * @param p_nb_bytes    the nb bytes to read.
 * @return              the number of bytes read or -1 if an error occurred.
 */
void opj_read_bytes_LE(const OPJ_BYTE * p_buffer, OPJ_UINT32 * p_value,
                       OPJ_UINT32 p_nb_bytes);


/**
 * Write some bytes to the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer      pointer the data buffer to write data to.
 * @param p_value       the value to write
 */
void opj_write_double_LE(OPJ_BYTE * p_buffer, OPJ_FLOAT64 p_value);

/***
 * Write some bytes to the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer      pointer the data buffer to write data to.
 * @param p_value       the value to write
 */
void opj_write_double_BE(OPJ_BYTE * p_buffer, OPJ_FLOAT64 p_value);

/**
 * Reads some bytes from the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer      pointer the data buffer to read data from.
 * @param p_value       pointer to the value that will store the data.
 */
void opj_read_double_LE(const OPJ_BYTE * p_buffer, OPJ_FLOAT64 * p_value);

/**
 * Reads some bytes from the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer      pointer the data buffer to read data from.
 * @param p_value       pointer to the value that will store the data.
 */
void opj_read_double_BE(const OPJ_BYTE * p_buffer, OPJ_FLOAT64 * p_value);

/**
 * Reads some bytes from the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer      pointer the data buffer to read data from.
 * @param p_value       pointer to the value that will store the data.
 */
void opj_read_float_LE(const OPJ_BYTE * p_buffer, OPJ_FLOAT32 * p_value);

/**
 * Reads some bytes from the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer      pointer the data buffer to read data from.
 * @param p_value       pointer to the value that will store the data.
 */
void opj_read_float_BE(const OPJ_BYTE * p_buffer, OPJ_FLOAT32 * p_value);

/**
 * Write some bytes to the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer      pointer the data buffer to write data to.
 * @param p_value       the value to write
 */
void opj_write_float_LE(OPJ_BYTE * p_buffer, OPJ_FLOAT32 p_value);

/***
 * Write some bytes to the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer      pointer the data buffer to write data to.
 * @param p_value       the value to write
 */
void opj_write_float_BE(OPJ_BYTE * p_buffer, OPJ_FLOAT32 p_value);

/**
 * Reads some bytes from the stream.
 * @param       p_stream    the stream to read data from.
 * @param       p_buffer    pointer to the data buffer that will receive the data.
 * @param       p_size      number of bytes to read.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      the number of bytes read, or -1 if an error occurred or if the stream is at the end.
 */
OPJ_SIZE_T opj_stream_read_data(opj_stream_private_t * p_stream,
                                OPJ_BYTE * p_buffer, OPJ_SIZE_T p_size, struct opj_event_mgr * p_event_mgr);

/**
 * Writes some bytes to the stream.
 * @param       p_stream    the stream to write data to.
 * @param       p_buffer    pointer to the data buffer holds the data to be writtent.
 * @param       p_size      number of bytes to write.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      the number of bytes writtent, or -1 if an error occurred.
 */
OPJ_SIZE_T opj_stream_write_data(opj_stream_private_t * p_stream,
                                 const OPJ_BYTE * p_buffer, OPJ_SIZE_T p_size,
                                 struct opj_event_mgr * p_event_mgr);

/**
 * Writes the content of the stream buffer to the stream.
 * @param       p_stream    the stream to write data to.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      true if the data could be flushed, false else.
 */
OPJ_BOOL opj_stream_flush(opj_stream_private_t * p_stream,
                          struct opj_event_mgr * p_event_mgr);

/**
 * Skips a number of bytes from the stream.
 * @param       p_stream    the stream to skip data from.
 * @param       p_size      the number of bytes to skip.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      the number of bytes skipped, or -1 if an error occurred.
 */
OPJ_OFF_T opj_stream_skip(opj_stream_private_t * p_stream, OPJ_OFF_T p_size,
                          struct opj_event_mgr * p_event_mgr);

/**
 * Tells the byte offset on the stream (similar to ftell).
 *
 * @param       p_stream    the stream to get the information from.
 *
 * @return      the current position o fthe stream.
 */
OPJ_OFF_T opj_stream_tell(const opj_stream_private_t * p_stream);


/**
 * Get the number of bytes left before the end of the stream (similar to cio_numbytesleft).
 *
 * @param       p_stream    the stream to get the information from.
 *
 * @return      Number of bytes left before the end of the stream.
 */
OPJ_OFF_T opj_stream_get_number_byte_left(const opj_stream_private_t *
        p_stream);

/**
 * Skips a number of bytes from the stream.
 * @param       p_stream    the stream to skip data from.
 * @param       p_size      the number of bytes to skip.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      the number of bytes skipped, or -1 if an error occurred.
 */
OPJ_OFF_T opj_stream_write_skip(opj_stream_private_t * p_stream,
                                OPJ_OFF_T p_size, struct opj_event_mgr * p_event_mgr);

/**
 * Skips a number of bytes from the stream.
 * @param       p_stream    the stream to skip data from.
 * @param       p_size      the number of bytes to skip.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      the number of bytes skipped, or -1 if an error occurred.
 */
OPJ_OFF_T opj_stream_read_skip(opj_stream_private_t * p_stream,
                               OPJ_OFF_T p_size, struct opj_event_mgr * p_event_mgr);

/**
 * Skips a number of bytes from the stream.
 * @param       p_stream    the stream to skip data from.
 * @param       p_size      the number of bytes to skip.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      OPJ_TRUE if success, or OPJ_FALSE if an error occurred.
 */
OPJ_BOOL opj_stream_read_seek(opj_stream_private_t * p_stream, OPJ_OFF_T p_size,
                              struct opj_event_mgr * p_event_mgr);

/**
 * Skips a number of bytes from the stream.
 * @param       p_stream    the stream to skip data from.
 * @param       p_size      the number of bytes to skip.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      the number of bytes skipped, or -1 if an error occurred.
 */
OPJ_BOOL opj_stream_write_seek(opj_stream_private_t * p_stream,
                               OPJ_OFF_T p_size, struct opj_event_mgr * p_event_mgr);

/**
 * Seeks a number of bytes from the stream.
 * @param       p_stream    the stream to skip data from.
 * @param       p_size      the number of bytes to skip.
 * @param       p_event_mgr the user event manager to be notified of special events.
 * @return      true if the stream is seekable.
 */
OPJ_BOOL opj_stream_seek(opj_stream_private_t * p_stream, OPJ_OFF_T p_size,
                         struct opj_event_mgr * p_event_mgr);

/**
 * Tells if the given stream is seekable.
 */
OPJ_BOOL opj_stream_has_seek(const opj_stream_private_t * p_stream);

/**
 * FIXME DOC.
 */
OPJ_SIZE_T opj_stream_default_read(void * p_buffer, OPJ_SIZE_T p_nb_bytes,
                                   void * p_user_data);

/**
 * FIXME DOC.
 */
OPJ_SIZE_T opj_stream_default_write(void * p_buffer, OPJ_SIZE_T p_nb_bytes,
                                    void * p_user_data);

/**
 * FIXME DOC.
 */
OPJ_OFF_T opj_stream_default_skip(OPJ_OFF_T p_nb_bytes, void * p_user_data);

/**
 * FIXME DOC.
 */
OPJ_BOOL opj_stream_default_seek(OPJ_OFF_T p_nb_bytes, void * p_user_data);

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/


#endif /* OPJ_CIO_H */

