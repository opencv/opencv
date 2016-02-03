/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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

#include "cio.h"
#include "opj_includes.h"
#include "opj_malloc.h"
#include "event.h"

/* ----------------------------------------------------------------------- */


/**
 * Write some bytes to the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer    pointer the data buffer to write data to.
 * @param p_value    the value to write
 * @param p_nb_bytes  the number of bytes to write
*/
void opj_write_bytes_BE (OPJ_BYTE * p_buffer, OPJ_UINT32 p_value, OPJ_UINT32 p_nb_bytes)
{
  const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + p_nb_bytes;
  assert(p_nb_bytes > 0 && p_nb_bytes <=  sizeof(OPJ_UINT32));
  memcpy(p_buffer,l_data_ptr,p_nb_bytes);
}

/**
 * Write some bytes to the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer    pointer the data buffer to write data to.
 * @param p_value    the value to write
 * @param p_nb_bytes  the number of bytes to write
 * @return        the number of bytes written or -1 if an error occured
*/
void opj_write_bytes_LE (OPJ_BYTE * p_buffer, OPJ_UINT32 p_value, OPJ_UINT32 p_nb_bytes)
{
  const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + p_nb_bytes - 1;
  OPJ_UINT32 i;

  assert(p_nb_bytes > 0 && p_nb_bytes <= sizeof(OPJ_UINT32));
  for
    (i=0;i<p_nb_bytes;++i)
  {
    *(p_buffer++) = *(l_data_ptr--);
  }
}

/**
 * Reads some bytes from the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer    pointer the data buffer to read data from.
 * @param p_value    pointer to the value that will store the data.
 * @param p_nb_bytes  the nb bytes to read.
 * @return        the number of bytes read or -1 if an error occured.
 */
void opj_read_bytes_BE(const OPJ_BYTE * p_buffer, OPJ_UINT32 * p_value, OPJ_UINT32 p_nb_bytes)
{
  OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value);
  assert(p_nb_bytes > 0 && p_nb_bytes <= sizeof(OPJ_UINT32));
  *p_value = 0;
  memcpy(l_data_ptr+4-p_nb_bytes,p_buffer,p_nb_bytes);
}

/**
 * Reads some bytes from the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer    pointer the data buffer to read data from.
 * @param p_value    pointer to the value that will store the data.
 * @param p_nb_bytes  the nb bytes to read.
 * @return        the number of bytes read or -1 if an error occured.
 */
void opj_read_bytes_LE(const OPJ_BYTE * p_buffer, OPJ_UINT32 * p_value, OPJ_UINT32 p_nb_bytes)
{
  OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value) + p_nb_bytes-1;
  OPJ_UINT32 i;

  assert(p_nb_bytes > 0 && p_nb_bytes <= sizeof(OPJ_UINT32));
  *p_value = 0;
  for
    (i=0;i<p_nb_bytes;++i)
  {
    *(l_data_ptr--) = *(p_buffer++);
  }
}

/**
 * Write some bytes to the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer    pointer the data buffer to write data to.
 * @param p_value    the value to write
 * @return        the number of bytes written or -1 if an error occured
 */
void opj_write_double_BE(OPJ_BYTE * p_buffer, OPJ_FLOAT64 p_value)
{
  const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value);
  memcpy(p_buffer,l_data_ptr,sizeof(OPJ_FLOAT64));
}

/**
 * Write some bytes to the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer    pointer the data buffer to write data to.
 * @param p_value    the value to write
 */
void opj_write_double_LE(OPJ_BYTE * p_buffer, OPJ_FLOAT64 p_value)
{
  const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + sizeof(OPJ_FLOAT64) - 1;
  OPJ_UINT32 i;
  for
    (i=0;i<sizeof(OPJ_FLOAT64);++i)
  {
    *(p_buffer++) = *(l_data_ptr--);
  }
}

/**
 * Reads some bytes from the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer    pointer the data buffer to read data from.
 * @param p_value    pointer to the value that will store the data.
 */
void opj_read_double_BE(const OPJ_BYTE * p_buffer, OPJ_FLOAT64 * p_value)
{
  OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value);
  memcpy(l_data_ptr,p_buffer,sizeof(OPJ_FLOAT64));
}


/**
 * Reads some bytes from the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer    pointer the data buffer to read data from.
 * @param p_value    pointer to the value that will store the data.
 */
void opj_read_double_LE(const OPJ_BYTE * p_buffer, OPJ_FLOAT64 * p_value)
{
  OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value) + sizeof(OPJ_FLOAT64)-1;
  OPJ_UINT32 i;
  for
    (i=0;i<sizeof(OPJ_FLOAT64);++i)
  {
    *(l_data_ptr--) = *(p_buffer++);
  }
}

/**
 * Write some bytes to the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer    pointer the data buffer to write data to.
 * @param p_value    the value to write
 * @return        the number of bytes written or -1 if an error occured
 */
void opj_write_float_BE(OPJ_BYTE * p_buffer, OPJ_FLOAT32 p_value)
{
  const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value);
  memcpy(p_buffer,l_data_ptr,sizeof(OPJ_FLOAT32));
}

/**
 * Write some bytes to the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer    pointer the data buffer to write data to.
 * @param p_value    the value to write
 */
void opj_write_float_LE(OPJ_BYTE * p_buffer, OPJ_FLOAT32 p_value)
{
  const OPJ_BYTE * l_data_ptr = ((const OPJ_BYTE *) &p_value) + sizeof(OPJ_FLOAT32) - 1;
  OPJ_UINT32 i;
  for
    (i=0;i<sizeof(OPJ_FLOAT32);++i)
  {
    *(p_buffer++) = *(l_data_ptr--);
  }
}

/**
 * Reads some bytes from the given data buffer, this function is used in Big Endian cpus.
 * @param p_buffer    pointer the data buffer to read data from.
 * @param p_value    pointer to the value that will store the data.
 */
void opj_read_float_BE(const OPJ_BYTE * p_buffer, OPJ_FLOAT32 * p_value)
{
  OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value);
  memcpy(l_data_ptr,p_buffer,sizeof(OPJ_FLOAT32));
}


/**
 * Reads some bytes from the given data buffer, this function is used in Little Endian cpus.
 * @param p_buffer    pointer the data buffer to read data from.
 * @param p_value    pointer to the value that will store the data.
 */
void opj_read_float_LE(const OPJ_BYTE * p_buffer, OPJ_FLOAT32 * p_value)
{
  OPJ_BYTE * l_data_ptr = ((OPJ_BYTE *) p_value) + sizeof(OPJ_FLOAT32)-1;
  OPJ_UINT32 i;
  for
    (i=0;i<sizeof(OPJ_FLOAT32);++i)
  {
    *(l_data_ptr--) = *(p_buffer++);
  }
}


/**
 * Creates an abstract stream. This function does nothing except allocating memory and initializing the abstract stream.
 * @return a stream object.
*/
opj_stream_t* OPJ_CALLCONV opj_stream_create(OPJ_UINT32 p_size,bool l_is_input)
{
  opj_stream_private_t * l_stream = 00;
  l_stream = (opj_stream_private_t*) opj_malloc(sizeof(opj_stream_private_t));
  if
    (! l_stream)
  {
    return 00;
  }
  memset(l_stream,0,sizeof(opj_stream_private_t));
  l_stream->m_buffer_size = p_size;
  l_stream->m_stored_data = (OPJ_BYTE *) opj_malloc(p_size);
  if
    (! l_stream->m_stored_data)
  {
    opj_free(l_stream);
    return 00;
  }
  l_stream->m_current_data = l_stream->m_stored_data;
  if
    (l_is_input)
  {
    l_stream->m_status |= opj_stream_e_input;
    l_stream->m_opj_skip = opj_stream_read_skip;
    l_stream->m_opj_seek = opj_stream_read_seek;
  }
  else
  {
    l_stream->m_status |= opj_stream_e_output;
    l_stream->m_opj_skip = opj_stream_write_skip;
    l_stream->m_opj_seek = opj_stream_write_seek;
  }
  l_stream->m_read_fn = opj_stream_default_read;
  l_stream->m_write_fn = opj_stream_default_write;
  l_stream->m_skip_fn = opj_stream_default_skip;
  l_stream->m_seek_fn = opj_stream_default_seek;

  return (opj_stream_t *) l_stream;
}

/**
 * Creates an abstract stream. This function does nothing except allocating memory and initializing the abstract stream.
 * @return a stream object.
*/
opj_stream_t* OPJ_CALLCONV opj_stream_default_create(bool l_is_input)
{
  return opj_stream_create(J2K_STREAM_CHUNK_SIZE,l_is_input);
}

/**
 * Destroys a stream created by opj_create_stream. This function does NOT close the abstract stream. If needed the user must
 * close its own implementation of the stream.
 */
OPJ_API void OPJ_CALLCONV opj_stream_destroy(opj_stream_t* p_stream)
{
  opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
  if
    (l_stream)
  {
    opj_free(l_stream->m_stored_data);
    l_stream->m_stored_data = 00;
    opj_free(l_stream);
  }

}

/**
 * Sets the given function to be used as a read function.
 * @param    p_stream  the stream to modify
 * @param    p_function  the function to use a read function.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_read_function(opj_stream_t* p_stream, opj_stream_read_fn p_function)
{
  opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
  if
    ((!l_stream) || (! (l_stream->m_status & opj_stream_e_input)))
  {
    return;
  }
  l_stream->m_read_fn = p_function;
}

OPJ_API void OPJ_CALLCONV opj_stream_set_seek_function(opj_stream_t* p_stream, opj_stream_seek_fn p_function)
{
  opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
  if
    (!l_stream)
  {
    return;
  }
  l_stream->m_seek_fn = p_function;
}

/**
 * Sets the given function to be used as a write function.
 * @param    p_stream  the stream to modify
 * @param    p_function  the function to use a write function.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_write_function(opj_stream_t* p_stream, opj_stream_write_fn p_function)
{
  opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
  if
    ((!l_stream )|| (! (l_stream->m_status & opj_stream_e_output)))
  {
    return;
  }
  l_stream->m_write_fn = p_function;
}

/**
 * Sets the given function to be used as a skip function.
 * @param    p_stream  the stream to modify
 * @param    p_function  the function to use a skip function.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_skip_function(opj_stream_t* p_stream, opj_stream_skip_fn p_function)
{
  opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
  if
    (! l_stream)
  {
    return;
  }
  l_stream->m_skip_fn = p_function;
}

/**
 * Sets the given data to be used as a user data for the stream.
 * @param    p_stream  the stream to modify
 * @param    p_data    the data to set.
*/
OPJ_API void OPJ_CALLCONV opj_stream_set_user_data(opj_stream_t* p_stream, void * p_data)
{
  opj_stream_private_t* l_stream = (opj_stream_private_t*) p_stream;
  l_stream->m_user_data = p_data;
}

/**
 * Reads some bytes from the stream.
 * @param    p_stream  the stream to read data from.
 * @param    p_buffer  pointer to the data buffer that will receive the data.
 * @param    p_size    number of bytes to read.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes read, or -1 if an error occured or if the stream is at the end.
 */
OPJ_UINT32 opj_stream_read_data (opj_stream_private_t * p_stream,OPJ_BYTE * p_buffer, OPJ_UINT32 p_size, opj_event_mgr_t * p_event_mgr)
{
  OPJ_UINT32 l_read_nb_bytes = 0;
  if
    (p_stream->m_bytes_in_buffer >= p_size)
  {
    memcpy(p_buffer,p_stream->m_current_data,p_size);
    p_stream->m_current_data += p_size;
    p_stream->m_bytes_in_buffer -= p_size;
    l_read_nb_bytes += p_size;
    p_stream->m_byte_offset += p_size;
    return l_read_nb_bytes;
  }

  // we are now in the case when the remaining data if not sufficient
  if
    (p_stream->m_status & opj_stream_e_end)
  {
    l_read_nb_bytes += p_stream->m_bytes_in_buffer;
    memcpy(p_buffer,p_stream->m_current_data,p_stream->m_bytes_in_buffer);
    p_stream->m_current_data += p_stream->m_bytes_in_buffer;
    p_stream->m_byte_offset += p_stream->m_bytes_in_buffer;
    p_stream->m_bytes_in_buffer = 0;
    return l_read_nb_bytes ? l_read_nb_bytes : -1;
  }

  // the flag is not set, we copy data and then do an actual read on the stream
  if
    (p_stream->m_bytes_in_buffer)
  {
    l_read_nb_bytes += p_stream->m_bytes_in_buffer;
    memcpy(p_buffer,p_stream->m_current_data,p_stream->m_bytes_in_buffer);
    p_stream->m_current_data = p_stream->m_stored_data;
    p_buffer += p_stream->m_bytes_in_buffer;
    p_size -= p_stream->m_bytes_in_buffer;
    p_stream->m_byte_offset += p_stream->m_bytes_in_buffer;
    p_stream->m_bytes_in_buffer = 0;
  }
  else
  {
    /* case where we are already at the end of the buffer
       so reset the m_current_data to point to the start of the
       stored buffer to get ready to read from disk*/
    p_stream->m_current_data = p_stream->m_stored_data;
  }


  while
    (true)
  {
    // we should read less than a chunk -> read a chunk
    if
      (p_size < p_stream->m_buffer_size)
    {
      // we should do an actual read on the media
      p_stream->m_bytes_in_buffer = p_stream->m_read_fn(p_stream->m_stored_data,p_stream->m_buffer_size,p_stream->m_user_data);
      if
        (p_stream->m_bytes_in_buffer == -1)
      {
        // end of stream
        opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");
        p_stream->m_bytes_in_buffer = 0;
        p_stream->m_status |= opj_stream_e_end;
        // end of stream
        return l_read_nb_bytes ? l_read_nb_bytes : -1;
      }
      else if
        (p_stream->m_bytes_in_buffer < p_size)
      {
        // not enough data
        l_read_nb_bytes += p_stream->m_bytes_in_buffer;
        memcpy(p_buffer,p_stream->m_current_data,p_stream->m_bytes_in_buffer);
        p_stream->m_current_data = p_stream->m_stored_data;
        p_buffer += p_stream->m_bytes_in_buffer;
        p_size -= p_stream->m_bytes_in_buffer;
        p_stream->m_byte_offset += p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
      }
      else
      {
        l_read_nb_bytes += p_size;
        memcpy(p_buffer,p_stream->m_current_data,p_size);
        p_stream->m_current_data += p_size;
        p_stream->m_bytes_in_buffer -= p_size;
        p_stream->m_byte_offset += p_size;
        return l_read_nb_bytes;
      }
    }
    else
    {
      // direct read on the dest buffer
      p_stream->m_bytes_in_buffer = p_stream->m_read_fn(p_buffer,p_size,p_stream->m_user_data);
      if
        (p_stream->m_bytes_in_buffer == -1)
      {
        // end of stream
        opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");
        p_stream->m_bytes_in_buffer = 0;
        p_stream->m_status |= opj_stream_e_end;
        // end of stream
        return l_read_nb_bytes ? l_read_nb_bytes : -1;
      }
      else if
        (p_stream->m_bytes_in_buffer < p_size)
      {
        // not enough data
        l_read_nb_bytes += p_stream->m_bytes_in_buffer;
        p_stream->m_current_data = p_stream->m_stored_data;
        p_buffer += p_stream->m_bytes_in_buffer;
        p_size -= p_stream->m_bytes_in_buffer;
        p_stream->m_byte_offset += p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
      }
      else
      {
        // we have read the exact size
        l_read_nb_bytes += p_stream->m_bytes_in_buffer;
        p_stream->m_byte_offset += p_stream->m_bytes_in_buffer;
        p_stream->m_current_data = p_stream->m_stored_data;
        p_stream->m_bytes_in_buffer = 0;
        return l_read_nb_bytes;
      }
    }
  }
}

/**
 * Writes some bytes from the stream.
 * @param    p_stream  the stream to write data to.
 * @param    p_buffer  pointer to the data buffer holds the data to be writtent.
 * @param    p_size    number of bytes to write.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes writtent, or -1 if an error occured.
 */
OPJ_UINT32 opj_stream_write_data (opj_stream_private_t * p_stream,const OPJ_BYTE * p_buffer,OPJ_UINT32 p_size, opj_event_mgr_t * p_event_mgr)
{
  OPJ_UINT32 l_remaining_bytes = 0;
  OPJ_UINT32 l_write_nb_bytes = 0;

  if
    (p_stream->m_status & opj_stream_e_error)
  {
    return -1;
  }

  while
    (true)
  {
    l_remaining_bytes = p_stream->m_buffer_size - p_stream->m_bytes_in_buffer;
    // we have more memory than required
    if
      (l_remaining_bytes >= p_size)
    {
      memcpy(p_stream->m_current_data,p_buffer,p_size);
      p_stream->m_current_data += p_size;
      p_stream->m_bytes_in_buffer += p_size;
      l_write_nb_bytes += p_size;
      p_stream->m_byte_offset += p_size;
      return l_write_nb_bytes;
    }

    // we copy data and then do an actual read on the stream
    if
      (l_remaining_bytes)
    {
      l_write_nb_bytes += l_remaining_bytes;
      memcpy(p_stream->m_current_data,p_buffer,l_remaining_bytes);
      p_stream->m_current_data = p_stream->m_stored_data;
      p_buffer += l_remaining_bytes;
      p_size -= l_remaining_bytes;
      p_stream->m_bytes_in_buffer += l_remaining_bytes;
      p_stream->m_byte_offset += l_remaining_bytes;
    }
    if
      (! opj_stream_flush(p_stream, p_event_mgr))
    {
      return -1;
    }
  }

}

/**
 * Writes the content of the stream buffer to the stream.
 * @param    p_stream  the stream to write data to.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes written, or -1 if an error occured.
 */
bool opj_stream_flush (opj_stream_private_t * p_stream, opj_event_mgr_t * p_event_mgr)
{
  // the number of bytes written on the media.
  OPJ_UINT32 l_current_write_nb_bytes = 0;
  p_stream->m_current_data = p_stream->m_stored_data;

  while
    (p_stream->m_bytes_in_buffer)
  {
    // we should do an actual write on the media
    l_current_write_nb_bytes = p_stream->m_write_fn(p_stream->m_current_data,p_stream->m_bytes_in_buffer,p_stream->m_user_data);
    if
      (l_current_write_nb_bytes == -1)
    {
      p_stream->m_status |= opj_stream_e_error;
      opj_event_msg(p_event_mgr, EVT_INFO, "Error on writting stream!\n");
      return false;
    }
    p_stream->m_current_data += l_current_write_nb_bytes;
    p_stream->m_bytes_in_buffer -= l_current_write_nb_bytes;
  }
  p_stream->m_current_data = p_stream->m_stored_data;
  return true;
}

/**
 * Skips a number of bytes from the stream.
 * @param    p_stream  the stream to skip data from.
 * @param    p_size    the number of bytes to skip.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes skipped, or -1 if an error occured.
 */
OPJ_SIZE_T opj_stream_read_skip (opj_stream_private_t * p_stream, OPJ_SIZE_T p_size, opj_event_mgr_t * p_event_mgr)
{
  OPJ_SIZE_T l_skip_nb_bytes = 0;
  OPJ_SIZE_T l_current_skip_nb_bytes = 0;

  if
    (p_stream->m_bytes_in_buffer >= p_size)
  {
    p_stream->m_current_data += p_size;
    p_stream->m_bytes_in_buffer -= p_size;
    l_skip_nb_bytes += p_size;
    p_stream->m_byte_offset += l_skip_nb_bytes;
    return l_skip_nb_bytes;
  }

  // we are now in the case when the remaining data if not sufficient
  if
    (p_stream->m_status & opj_stream_e_end)
  {
    l_skip_nb_bytes += p_stream->m_bytes_in_buffer;
    p_stream->m_current_data += p_stream->m_bytes_in_buffer;
    p_stream->m_bytes_in_buffer = 0;
    p_stream->m_byte_offset += l_skip_nb_bytes;
    return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_SIZE_T) -1;
  }

  // the flag is not set, we copy data and then do an actual skip on the stream
  if
    (p_stream->m_bytes_in_buffer)
  {
    l_skip_nb_bytes += p_stream->m_bytes_in_buffer;
    p_stream->m_current_data = p_stream->m_stored_data;
    p_size -= p_stream->m_bytes_in_buffer;
    p_stream->m_bytes_in_buffer = 0;
  }

  while
    (p_size > 0)
  {
    // we should do an actual skip on the media
    l_current_skip_nb_bytes = p_stream->m_skip_fn(p_size, p_stream->m_user_data);
    if
      (l_current_skip_nb_bytes == (OPJ_SIZE_T) -1)
    {
      opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");
      p_stream->m_status |= opj_stream_e_end;
      p_stream->m_byte_offset += l_skip_nb_bytes;
      // end if stream
      return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_SIZE_T) -1;
    }
    p_size -= l_current_skip_nb_bytes;
    l_skip_nb_bytes += l_current_skip_nb_bytes;
  }
  p_stream->m_byte_offset += l_skip_nb_bytes;
  return l_skip_nb_bytes;
}

/**
 * Skips a number of bytes from the stream.
 * @param    p_stream  the stream to skip data from.
 * @param    p_size    the number of bytes to skip.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes skipped, or -1 if an error occured.
 */
OPJ_SIZE_T opj_stream_write_skip (opj_stream_private_t * p_stream, OPJ_SIZE_T p_size, opj_event_mgr_t * p_event_mgr)
{
  bool l_is_written = 0;
  OPJ_SIZE_T l_current_skip_nb_bytes = 0;
  OPJ_SIZE_T l_skip_nb_bytes = 0;

  if
    (p_stream->m_status & opj_stream_e_error)
  {
    return (OPJ_SIZE_T) -1;
  }

  // we should flush data
  l_is_written = opj_stream_flush (p_stream, p_event_mgr);
  if
    (! l_is_written)
  {
    p_stream->m_status |= opj_stream_e_error;
    p_stream->m_bytes_in_buffer = 0;
    p_stream->m_current_data = p_stream->m_current_data;
    return (OPJ_SIZE_T) -1;
  }
  // then skip

  while
    (p_size > 0)
  {
    // we should do an actual skip on the media
    l_current_skip_nb_bytes = p_stream->m_skip_fn(p_size, p_stream->m_user_data);
    if
      (l_current_skip_nb_bytes == (OPJ_SIZE_T)-1)
    {
      opj_event_msg(p_event_mgr, EVT_INFO, "Stream error!\n");
      p_stream->m_status |= opj_stream_e_error;
      p_stream->m_byte_offset += l_skip_nb_bytes;
      // end if stream
      return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_SIZE_T)-1;
    }
    p_size -= l_current_skip_nb_bytes;
    l_skip_nb_bytes += l_current_skip_nb_bytes;
  }
  p_stream->m_byte_offset += l_skip_nb_bytes;
  return l_skip_nb_bytes;
}

/**
 * Tells the byte offset on the stream (similar to ftell).
 *
 * @param    p_stream  the stream to get the information from.
 *
 * @return    the current position o fthe stream.
 */
OPJ_SIZE_T opj_stream_tell (const opj_stream_private_t * p_stream)
{
  return p_stream->m_byte_offset;
}

/**
 * Skips a number of bytes from the stream.
 * @param    p_stream  the stream to skip data from.
 * @param    p_size    the number of bytes to skip.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes skipped, or -1 if an error occured.
 */
OPJ_SIZE_T opj_stream_skip (opj_stream_private_t * p_stream, OPJ_SIZE_T p_size, opj_event_mgr_t * p_event_mgr)
{
  return p_stream->m_opj_skip(p_stream,p_size,p_event_mgr);
}


/**
 * Skips a number of bytes from the stream.
 * @param    p_stream  the stream to skip data from.
 * @param    p_size    the number of bytes to skip.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes skipped, or -1 if an error occured.
 */
bool opj_stream_read_seek (opj_stream_private_t * p_stream, OPJ_SIZE_T p_size, opj_event_mgr_t * p_event_mgr)
{
  p_stream->m_current_data = p_stream->m_stored_data;
  p_stream->m_bytes_in_buffer = 0;
  if
    (! p_stream->m_seek_fn(p_size,p_stream->m_user_data))
  {
    p_stream->m_status |= opj_stream_e_end;
    return false;
  }
  else
  {
    // reset stream status
    p_stream->m_status &= (~opj_stream_e_end);
    p_stream->m_byte_offset = p_size;

  }
  return true;
}

/**
 * Skips a number of bytes from the stream.
 * @param    p_stream  the stream to skip data from.
 * @param    p_size    the number of bytes to skip.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    the number of bytes skipped, or -1 if an error occured.
 */
bool opj_stream_write_seek (opj_stream_private_t * p_stream, OPJ_SIZE_T p_size, opj_event_mgr_t * p_event_mgr)
{
  if
    (! opj_stream_flush(p_stream,p_event_mgr))
  {
    p_stream->m_status |= opj_stream_e_error;
    return false;
  }

  p_stream->m_current_data = p_stream->m_stored_data;
  p_stream->m_bytes_in_buffer = 0;

  if
    (! p_stream->m_seek_fn(p_size,p_stream->m_user_data))
  {
    p_stream->m_status |= opj_stream_e_error;
    return false;
  }
  else
  {
    p_stream->m_byte_offset = p_size;
  }
  return true;
}


/**
 * Seeks a number of bytes from the stream.
 * @param    p_stream  the stream to skip data from.
 * @param    p_size    the number of bytes to skip.
 * @param    p_event_mgr  the user event manager to be notified of special events.
 * @return    true if the stream is seekable.
 */
bool opj_stream_seek (opj_stream_private_t * p_stream, OPJ_SIZE_T p_size, struct opj_event_mgr * p_event_mgr)
{
  return p_stream->m_opj_seek(p_stream,p_size,p_event_mgr);
}

/**
 * Tells if the given stream is seekable.
 */
bool opj_stream_has_seek (const opj_stream_private_t * p_stream)
{
  return p_stream->m_seek_fn != opj_stream_default_seek;
}





OPJ_UINT32 opj_stream_default_read (void * p_buffer, OPJ_UINT32 p_nb_bytes, void * p_user_data)
{
  return (OPJ_UINT32) -1;
}
OPJ_UINT32 opj_stream_default_write (void * p_buffer, OPJ_UINT32 p_nb_bytes, void * p_user_data)
{
  return (OPJ_UINT32) -1;
}
OPJ_SIZE_T opj_stream_default_skip (OPJ_SIZE_T p_nb_bytes, void * p_user_data)
{
  return (OPJ_SIZE_T) -1;
}

bool opj_stream_default_seek (OPJ_SIZE_T p_nb_bytes, void * p_user_data)
{
  return false;
}
