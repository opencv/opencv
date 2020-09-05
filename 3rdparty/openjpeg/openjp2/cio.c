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

#include "opj_includes.h"

/* ----------------------------------------------------------------------- */


/* ----------------------------------------------------------------------- */

void opj_write_bytes_BE(OPJ_BYTE * p_buffer, OPJ_UINT32 p_value,
                        OPJ_UINT32 p_nb_bytes)
{
    const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + sizeof(
                                      OPJ_UINT32) - p_nb_bytes;

    assert(p_nb_bytes > 0 && p_nb_bytes <=  sizeof(OPJ_UINT32));

    memcpy(p_buffer, l_data_ptr, p_nb_bytes);
}

void opj_write_bytes_LE(OPJ_BYTE * p_buffer, OPJ_UINT32 p_value,
                        OPJ_UINT32 p_nb_bytes)
{
    const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + p_nb_bytes - 1;
    OPJ_UINT32 i;

    assert(p_nb_bytes > 0 && p_nb_bytes <= sizeof(OPJ_UINT32));

    for (i = 0; i < p_nb_bytes; ++i) {
        *(p_buffer++) = *(l_data_ptr--);
    }
}

void opj_read_bytes_BE(const OPJ_BYTE * p_buffer, OPJ_UINT32 * p_value,
                       OPJ_UINT32 p_nb_bytes)
{
    OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value);

    assert(p_nb_bytes > 0 && p_nb_bytes <= sizeof(OPJ_UINT32));

    *p_value = 0;
    memcpy(l_data_ptr + sizeof(OPJ_UINT32) - p_nb_bytes, p_buffer, p_nb_bytes);
}

void opj_read_bytes_LE(const OPJ_BYTE * p_buffer, OPJ_UINT32 * p_value,
                       OPJ_UINT32 p_nb_bytes)
{
    OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value) + p_nb_bytes - 1;
    OPJ_UINT32 i;

    assert(p_nb_bytes > 0 && p_nb_bytes <= sizeof(OPJ_UINT32));

    *p_value = 0;
    for (i = 0; i < p_nb_bytes; ++i) {
        *(l_data_ptr--) = *(p_buffer++);
    }
}

void opj_write_double_BE(OPJ_BYTE * p_buffer, OPJ_FLOAT64 p_value)
{
    const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value);
    memcpy(p_buffer, l_data_ptr, sizeof(OPJ_FLOAT64));
}

void opj_write_double_LE(OPJ_BYTE * p_buffer, OPJ_FLOAT64 p_value)
{
    const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + sizeof(
                                      OPJ_FLOAT64) - 1;
    OPJ_UINT32 i;
    for (i = 0; i < sizeof(OPJ_FLOAT64); ++i) {
        *(p_buffer++) = *(l_data_ptr--);
    }
}

void opj_read_double_BE(const OPJ_BYTE * p_buffer, OPJ_FLOAT64 * p_value)
{
    OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value);
    memcpy(l_data_ptr, p_buffer, sizeof(OPJ_FLOAT64));
}

void opj_read_double_LE(const OPJ_BYTE * p_buffer, OPJ_FLOAT64 * p_value)
{
    OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value) + sizeof(OPJ_FLOAT64) - 1;
    OPJ_UINT32 i;
    for (i = 0; i < sizeof(OPJ_FLOAT64); ++i) {
        *(l_data_ptr--) = *(p_buffer++);
    }
}

void opj_write_float_BE(OPJ_BYTE * p_buffer, OPJ_FLOAT32 p_value)
{
    const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value);
    memcpy(p_buffer, l_data_ptr, sizeof(OPJ_FLOAT32));
}

void opj_write_float_LE(OPJ_BYTE * p_buffer, OPJ_FLOAT32 p_value)
{
    const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + sizeof(
                                      OPJ_FLOAT32) - 1;
    OPJ_UINT32 i;
    for (i = 0; i < sizeof(OPJ_FLOAT32); ++i) {
        *(p_buffer++) = *(l_data_ptr--);
    }
}

void opj_read_float_BE(const OPJ_BYTE * p_buffer, OPJ_FLOAT32 * p_value)
{
    OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value);
    memcpy(l_data_ptr, p_buffer, sizeof(OPJ_FLOAT32));
}

void opj_read_float_LE(const OPJ_BYTE * p_buffer, OPJ_FLOAT32 * p_value)
{
    OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value) + sizeof(OPJ_FLOAT32) - 1;
    OPJ_UINT32 i;
    for (i = 0; i < sizeof(OPJ_FLOAT32); ++i) {
        *(l_data_ptr--) = *(p_buffer++);
    }
}

opj_stream_t* OPJ_CALLCONV opj_stream_create(OPJ_SIZE_T p_buffer_size,
        OPJ_BOOL l_is_input)
{
    opj_stream_private_t * l_stream = 00;
    l_stream = (opj_stream_private_t*) opj_calloc(1, sizeof(opj_stream_private_t));
    if (! l_stream) {
        return 00;
    }

    l_stream->m_buffer_size = p_buffer_size;
    l_stream->m_stored_data = (OPJ_BYTE *) opj_malloc(p_buffer_size);
    if (! l_stream->m_stored_data) {
        opj_free(l_stream);
        return 00;
    }

    l_stream->m_current_data = l_stream->m_stored_data;

    if (l_is_input) {
        l_stream->m_status |= OPJ_STREAM_STATUS_INPUT;
        l_stream->m_opj_skip = opj_stream_read_skip;
        l_stream->m_opj_seek = opj_stream_read_seek;
    } else {
        l_stream->m_status |= OPJ_STREAM_STATUS_OUTPUT;
        l_stream->m_opj_skip = opj_stream_write_skip;
        l_stream->m_opj_seek = opj_stream_write_seek;
    }

    l_stream->m_read_fn = opj_stream_default_read;
    l_stream->m_write_fn = opj_stream_default_write;
    l_stream->m_skip_fn = opj_stream_default_skip;
    l_stream->m_seek_fn = opj_stream_default_seek;

    return (opj_stream_t *) l_stream;
}

opj_stream_t* OPJ_CALLCONV opj_stream_default_create(OPJ_BOOL l_is_input)
{
    return opj_stream_create(OPJ_J2K_STREAM_CHUNK_SIZE, l_is_input);
}

void OPJ_CALLCONV opj_stream_destroy(opj_stream_t* p_stream)
{
    opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;

    if (l_stream) {
        if (l_stream->m_free_user_data_fn) {
            l_stream->m_free_user_data_fn(l_stream->m_user_data);
        }
        opj_free(l_stream->m_stored_data);
        l_stream->m_stored_data = 00;
        opj_free(l_stream);
    }
}

void OPJ_CALLCONV opj_stream_set_read_function(opj_stream_t* p_stream,
        opj_stream_read_fn p_function)
{
    opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;

    if ((!l_stream) || (!(l_stream->m_status & OPJ_STREAM_STATUS_INPUT))) {
        return;
    }

    l_stream->m_read_fn = p_function;
}

void OPJ_CALLCONV opj_stream_set_seek_function(opj_stream_t* p_stream,
        opj_stream_seek_fn p_function)
{
    opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;

    if (!l_stream) {
        return;
    }
    l_stream->m_seek_fn = p_function;
}

void OPJ_CALLCONV opj_stream_set_write_function(opj_stream_t* p_stream,
        opj_stream_write_fn p_function)
{
    opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;

    if ((!l_stream) || (!(l_stream->m_status & OPJ_STREAM_STATUS_OUTPUT))) {
        return;
    }

    l_stream->m_write_fn = p_function;
}

void OPJ_CALLCONV opj_stream_set_skip_function(opj_stream_t* p_stream,
        opj_stream_skip_fn p_function)
{
    opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;

    if (! l_stream) {
        return;
    }

    l_stream->m_skip_fn = p_function;
}

void OPJ_CALLCONV opj_stream_set_user_data(opj_stream_t* p_stream,
        void * p_data, opj_stream_free_user_data_fn p_function)
{
    opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
    if (!l_stream) {
        return;
    }
    l_stream->m_user_data = p_data;
    l_stream->m_free_user_data_fn = p_function;
}

void OPJ_CALLCONV opj_stream_set_user_data_length(opj_stream_t* p_stream,
        OPJ_UINT64 data_length)
{
    opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
    if (!l_stream) {
        return;
    }
    l_stream->m_user_data_length = data_length;
}

OPJ_SIZE_T opj_stream_read_data(opj_stream_private_t * p_stream,
                                OPJ_BYTE * p_buffer, OPJ_SIZE_T p_size, opj_event_mgr_t * p_event_mgr)
{
    OPJ_SIZE_T l_read_nb_bytes = 0;
    if (p_stream->m_bytes_in_buffer >= p_size) {
        memcpy(p_buffer, p_stream->m_current_data, p_size);
        p_stream->m_current_data += p_size;
        p_stream->m_bytes_in_buffer -= p_size;
        l_read_nb_bytes += p_size;
        p_stream->m_byte_offset += (OPJ_OFF_T)p_size;
        return l_read_nb_bytes;
    }

    /* we are now in the case when the remaining data if not sufficient */
    if (p_stream->m_status & OPJ_STREAM_STATUS_END) {
        l_read_nb_bytes += p_stream->m_bytes_in_buffer;
        memcpy(p_buffer, p_stream->m_current_data, p_stream->m_bytes_in_buffer);
        p_stream->m_current_data += p_stream->m_bytes_in_buffer;
        p_stream->m_byte_offset += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
        return l_read_nb_bytes ? l_read_nb_bytes : (OPJ_SIZE_T) - 1;
    }

    /* the flag is not set, we copy data and then do an actual read on the stream */
    if (p_stream->m_bytes_in_buffer) {
        l_read_nb_bytes += p_stream->m_bytes_in_buffer;
        memcpy(p_buffer, p_stream->m_current_data, p_stream->m_bytes_in_buffer);
        p_stream->m_current_data = p_stream->m_stored_data;
        p_buffer += p_stream->m_bytes_in_buffer;
        p_size -= p_stream->m_bytes_in_buffer;
        p_stream->m_byte_offset += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
    } else {
        /* case where we are already at the end of the buffer
           so reset the m_current_data to point to the start of the
           stored buffer to get ready to read from disk*/
        p_stream->m_current_data = p_stream->m_stored_data;
    }

    for (;;) {
        /* we should read less than a chunk -> read a chunk */
        if (p_size < p_stream->m_buffer_size) {
            /* we should do an actual read on the media */
            p_stream->m_bytes_in_buffer = p_stream->m_read_fn(p_stream->m_stored_data,
                                          p_stream->m_buffer_size, p_stream->m_user_data);

            if (p_stream->m_bytes_in_buffer == (OPJ_SIZE_T) - 1) {
                /* end of stream */
                opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");

                p_stream->m_bytes_in_buffer = 0;
                p_stream->m_status |= OPJ_STREAM_STATUS_END;
                /* end of stream */
                return l_read_nb_bytes ? l_read_nb_bytes : (OPJ_SIZE_T) - 1;
            } else if (p_stream->m_bytes_in_buffer < p_size) {
                /* not enough data */
                l_read_nb_bytes += p_stream->m_bytes_in_buffer;
                memcpy(p_buffer, p_stream->m_current_data, p_stream->m_bytes_in_buffer);
                p_stream->m_current_data = p_stream->m_stored_data;
                p_buffer += p_stream->m_bytes_in_buffer;
                p_size -= p_stream->m_bytes_in_buffer;
                p_stream->m_byte_offset += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
                p_stream->m_bytes_in_buffer = 0;
            } else {
                l_read_nb_bytes += p_size;
                memcpy(p_buffer, p_stream->m_current_data, p_size);
                p_stream->m_current_data += p_size;
                p_stream->m_bytes_in_buffer -= p_size;
                p_stream->m_byte_offset += (OPJ_OFF_T)p_size;
                return l_read_nb_bytes;
            }
        } else {
            /* direct read on the dest buffer */
            p_stream->m_bytes_in_buffer = p_stream->m_read_fn(p_buffer, p_size,
                                          p_stream->m_user_data);

            if (p_stream->m_bytes_in_buffer == (OPJ_SIZE_T) - 1) {
                /*  end of stream */
                opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");

                p_stream->m_bytes_in_buffer = 0;
                p_stream->m_status |= OPJ_STREAM_STATUS_END;
                /* end of stream */
                return l_read_nb_bytes ? l_read_nb_bytes : (OPJ_SIZE_T) - 1;
            } else if (p_stream->m_bytes_in_buffer < p_size) {
                /* not enough data */
                l_read_nb_bytes += p_stream->m_bytes_in_buffer;
                p_stream->m_current_data = p_stream->m_stored_data;
                p_buffer += p_stream->m_bytes_in_buffer;
                p_size -= p_stream->m_bytes_in_buffer;
                p_stream->m_byte_offset += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
                p_stream->m_bytes_in_buffer = 0;
            } else {
                /* we have read the exact size */
                l_read_nb_bytes += p_stream->m_bytes_in_buffer;
                p_stream->m_byte_offset += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
                p_stream->m_current_data = p_stream->m_stored_data;
                p_stream->m_bytes_in_buffer = 0;
                return l_read_nb_bytes;
            }
        }
    }
}

OPJ_SIZE_T opj_stream_write_data(opj_stream_private_t * p_stream,
                                 const OPJ_BYTE * p_buffer,
                                 OPJ_SIZE_T p_size,
                                 opj_event_mgr_t * p_event_mgr)
{
    OPJ_SIZE_T l_remaining_bytes = 0;
    OPJ_SIZE_T l_write_nb_bytes = 0;

    if (p_stream->m_status & OPJ_STREAM_STATUS_ERROR) {
        return (OPJ_SIZE_T) - 1;
    }

    for (;;) {
        l_remaining_bytes = p_stream->m_buffer_size - p_stream->m_bytes_in_buffer;

        /* we have more memory than required */
        if (l_remaining_bytes >= p_size) {
            memcpy(p_stream->m_current_data, p_buffer, p_size);

            p_stream->m_current_data += p_size;
            p_stream->m_bytes_in_buffer += p_size;
            l_write_nb_bytes += p_size;
            p_stream->m_byte_offset += (OPJ_OFF_T)p_size;

            return l_write_nb_bytes;
        }

        /* we copy data and then do an actual read on the stream */
        if (l_remaining_bytes) {
            l_write_nb_bytes += l_remaining_bytes;

            memcpy(p_stream->m_current_data, p_buffer, l_remaining_bytes);

            p_stream->m_current_data = p_stream->m_stored_data;

            p_buffer += l_remaining_bytes;
            p_size -= l_remaining_bytes;
            p_stream->m_bytes_in_buffer += l_remaining_bytes;
            p_stream->m_byte_offset += (OPJ_OFF_T)l_remaining_bytes;
        }

        if (! opj_stream_flush(p_stream, p_event_mgr)) {
            return (OPJ_SIZE_T) - 1;
        }
    }

}

OPJ_BOOL opj_stream_flush(opj_stream_private_t * p_stream,
                          opj_event_mgr_t * p_event_mgr)
{
    /* the number of bytes written on the media. */
    OPJ_SIZE_T l_current_write_nb_bytes = 0;

    p_stream->m_current_data = p_stream->m_stored_data;

    while (p_stream->m_bytes_in_buffer) {
        /* we should do an actual write on the media */
        l_current_write_nb_bytes = p_stream->m_write_fn(p_stream->m_current_data,
                                   p_stream->m_bytes_in_buffer,
                                   p_stream->m_user_data);

        if (l_current_write_nb_bytes == (OPJ_SIZE_T) - 1) {
            p_stream->m_status |= OPJ_STREAM_STATUS_ERROR;
            opj_event_msg(p_event_mgr, EVT_INFO, "Error on writing stream!\n");

            return OPJ_FALSE;
        }

        p_stream->m_current_data += l_current_write_nb_bytes;
        p_stream->m_bytes_in_buffer -= l_current_write_nb_bytes;
    }

    p_stream->m_current_data = p_stream->m_stored_data;

    return OPJ_TRUE;
}

OPJ_OFF_T opj_stream_read_skip(opj_stream_private_t * p_stream,
                               OPJ_OFF_T p_size, opj_event_mgr_t * p_event_mgr)
{
    OPJ_OFF_T l_skip_nb_bytes = 0;
    OPJ_OFF_T l_current_skip_nb_bytes = 0;

    assert(p_size >= 0);

    if (p_stream->m_bytes_in_buffer >= (OPJ_SIZE_T)p_size) {
        p_stream->m_current_data += p_size;
        /* it is safe to cast p_size to OPJ_SIZE_T since it is <= m_bytes_in_buffer
        which is of type OPJ_SIZE_T */
        p_stream->m_bytes_in_buffer -= (OPJ_SIZE_T)p_size;
        l_skip_nb_bytes += p_size;
        p_stream->m_byte_offset += l_skip_nb_bytes;
        return l_skip_nb_bytes;
    }

    /* we are now in the case when the remaining data if not sufficient */
    if (p_stream->m_status & OPJ_STREAM_STATUS_END) {
        l_skip_nb_bytes += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_current_data += p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
        p_stream->m_byte_offset += l_skip_nb_bytes;
        return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_OFF_T) - 1;
    }

    /* the flag is not set, we copy data and then do an actual skip on the stream */
    if (p_stream->m_bytes_in_buffer) {
        l_skip_nb_bytes += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_current_data = p_stream->m_stored_data;
        p_size -= (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
    }

    while (p_size > 0) {
        /* Check if we are going beyond the end of file. Most skip_fn do not */
        /* check that, but we must be careful not to advance m_byte_offset */
        /* beyond m_user_data_length, otherwise */
        /* opj_stream_get_number_byte_left() will assert. */
        if ((OPJ_UINT64)(p_stream->m_byte_offset + l_skip_nb_bytes + p_size) >
                p_stream->m_user_data_length) {
            opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");

            p_stream->m_byte_offset += l_skip_nb_bytes;
            l_skip_nb_bytes = (OPJ_OFF_T)(p_stream->m_user_data_length -
                                          (OPJ_UINT64)p_stream->m_byte_offset);

            opj_stream_read_seek(p_stream, (OPJ_OFF_T)p_stream->m_user_data_length,
                                 p_event_mgr);
            p_stream->m_status |= OPJ_STREAM_STATUS_END;

            /* end if stream */
            return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_OFF_T) - 1;
        }

        /* we should do an actual skip on the media */
        l_current_skip_nb_bytes = p_stream->m_skip_fn(p_size, p_stream->m_user_data);
        if (l_current_skip_nb_bytes == (OPJ_OFF_T) - 1) {
            opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");

            p_stream->m_status |= OPJ_STREAM_STATUS_END;
            p_stream->m_byte_offset += l_skip_nb_bytes;
            /* end if stream */
            return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_OFF_T) - 1;
        }
        p_size -= l_current_skip_nb_bytes;
        l_skip_nb_bytes += l_current_skip_nb_bytes;
    }

    p_stream->m_byte_offset += l_skip_nb_bytes;

    return l_skip_nb_bytes;
}

OPJ_OFF_T opj_stream_write_skip(opj_stream_private_t * p_stream,
                                OPJ_OFF_T p_size, opj_event_mgr_t * p_event_mgr)
{
    OPJ_BOOL l_is_written = 0;
    OPJ_OFF_T l_current_skip_nb_bytes = 0;
    OPJ_OFF_T l_skip_nb_bytes = 0;

    if (p_stream->m_status & OPJ_STREAM_STATUS_ERROR) {
        return (OPJ_OFF_T) - 1;
    }

    /* we should flush data */
    l_is_written = opj_stream_flush(p_stream, p_event_mgr);
    if (! l_is_written) {
        p_stream->m_status |= OPJ_STREAM_STATUS_ERROR;
        p_stream->m_bytes_in_buffer = 0;
        return (OPJ_OFF_T) - 1;
    }
    /* then skip */

    while (p_size > 0) {
        /* we should do an actual skip on the media */
        l_current_skip_nb_bytes = p_stream->m_skip_fn(p_size, p_stream->m_user_data);

        if (l_current_skip_nb_bytes == (OPJ_OFF_T) - 1) {
            opj_event_msg(p_event_mgr, EVT_INFO, "Stream error!\n");

            p_stream->m_status |= OPJ_STREAM_STATUS_ERROR;
            p_stream->m_byte_offset += l_skip_nb_bytes;
            /* end if stream */
            return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_OFF_T) - 1;
        }
        p_size -= l_current_skip_nb_bytes;
        l_skip_nb_bytes += l_current_skip_nb_bytes;
    }

    p_stream->m_byte_offset += l_skip_nb_bytes;

    return l_skip_nb_bytes;
}

OPJ_OFF_T opj_stream_tell(const opj_stream_private_t * p_stream)
{
    return p_stream->m_byte_offset;
}

OPJ_OFF_T opj_stream_get_number_byte_left(const opj_stream_private_t * p_stream)
{
    assert(p_stream->m_byte_offset >= 0);
    assert(p_stream->m_user_data_length >= (OPJ_UINT64)p_stream->m_byte_offset);
    return p_stream->m_user_data_length ?
           (OPJ_OFF_T)(p_stream->m_user_data_length) - p_stream->m_byte_offset :
           0;
}

OPJ_OFF_T opj_stream_skip(opj_stream_private_t * p_stream, OPJ_OFF_T p_size,
                          opj_event_mgr_t * p_event_mgr)
{
    assert(p_size >= 0);
    return p_stream->m_opj_skip(p_stream, p_size, p_event_mgr);
}

OPJ_BOOL opj_stream_read_seek(opj_stream_private_t * p_stream, OPJ_OFF_T p_size,
                              opj_event_mgr_t * p_event_mgr)
{
    OPJ_ARG_NOT_USED(p_event_mgr);
    p_stream->m_current_data = p_stream->m_stored_data;
    p_stream->m_bytes_in_buffer = 0;

    if (!(p_stream->m_seek_fn(p_size, p_stream->m_user_data))) {
        p_stream->m_status |= OPJ_STREAM_STATUS_END;
        return OPJ_FALSE;
    } else {
        /* reset stream status */
        p_stream->m_status &= (~OPJ_STREAM_STATUS_END);
        p_stream->m_byte_offset = p_size;

    }

    return OPJ_TRUE;
}

OPJ_BOOL opj_stream_write_seek(opj_stream_private_t * p_stream,
                               OPJ_OFF_T p_size, opj_event_mgr_t * p_event_mgr)
{
    if (! opj_stream_flush(p_stream, p_event_mgr)) {
        p_stream->m_status |= OPJ_STREAM_STATUS_ERROR;
        return OPJ_FALSE;
    }

    p_stream->m_current_data = p_stream->m_stored_data;
    p_stream->m_bytes_in_buffer = 0;

    if (! p_stream->m_seek_fn(p_size, p_stream->m_user_data)) {
        p_stream->m_status |= OPJ_STREAM_STATUS_ERROR;
        return OPJ_FALSE;
    } else {
        p_stream->m_byte_offset = p_size;
    }

    return OPJ_TRUE;
}

OPJ_BOOL opj_stream_seek(opj_stream_private_t * p_stream, OPJ_OFF_T p_size,
                         struct opj_event_mgr * p_event_mgr)
{
    assert(p_size >= 0);
    return p_stream->m_opj_seek(p_stream, p_size, p_event_mgr);
}

OPJ_BOOL opj_stream_has_seek(const opj_stream_private_t * p_stream)
{
    return p_stream->m_seek_fn != opj_stream_default_seek;
}

OPJ_SIZE_T opj_stream_default_read(void * p_buffer, OPJ_SIZE_T p_nb_bytes,
                                   void * p_user_data)
{
    OPJ_ARG_NOT_USED(p_buffer);
    OPJ_ARG_NOT_USED(p_nb_bytes);
    OPJ_ARG_NOT_USED(p_user_data);
    return (OPJ_SIZE_T) - 1;
}

OPJ_SIZE_T opj_stream_default_write(void * p_buffer, OPJ_SIZE_T p_nb_bytes,
                                    void * p_user_data)
{
    OPJ_ARG_NOT_USED(p_buffer);
    OPJ_ARG_NOT_USED(p_nb_bytes);
    OPJ_ARG_NOT_USED(p_user_data);
    return (OPJ_SIZE_T) - 1;
}

OPJ_OFF_T opj_stream_default_skip(OPJ_OFF_T p_nb_bytes, void * p_user_data)
{
    OPJ_ARG_NOT_USED(p_nb_bytes);
    OPJ_ARG_NOT_USED(p_user_data);
    return (OPJ_OFF_T) - 1;
}

OPJ_BOOL opj_stream_default_seek(OPJ_OFF_T p_nb_bytes, void * p_user_data)
{
    OPJ_ARG_NOT_USED(p_nb_bytes);
    OPJ_ARG_NOT_USED(p_user_data);
    return OPJ_FALSE;
}
