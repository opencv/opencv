
/* pngpread.c - read a png file in push mode
 *
 * Last changed in libpng 1.2.27 [April 29, 2008]
 * For conditions of distribution and use, see copyright notice in png.h
 * Copyright (c) 1998-2008 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 */

#define PNG_INTERNAL
#include "png.h"

#ifdef PNG_PROGRESSIVE_READ_SUPPORTED

/* push model modes */
#define PNG_READ_SIG_MODE   0
#define PNG_READ_CHUNK_MODE 1
#define PNG_READ_IDAT_MODE  2
#define PNG_SKIP_MODE       3
#define PNG_READ_tEXt_MODE  4
#define PNG_READ_zTXt_MODE  5
#define PNG_READ_DONE_MODE  6
#define PNG_READ_iTXt_MODE  7
#define PNG_ERROR_MODE      8

void PNGAPI
png_process_data(png_structp png_ptr, png_infop info_ptr,
   png_bytep buffer, png_size_t buffer_size)
{
   if(png_ptr == NULL || info_ptr == NULL) return;
   png_push_restore_buffer(png_ptr, buffer, buffer_size);

   while (png_ptr->buffer_size)
   {
      png_process_some_data(png_ptr, info_ptr);
   }
}

/* What we do with the incoming data depends on what we were previously
 * doing before we ran out of data...
 */
void /* PRIVATE */
png_process_some_data(png_structp png_ptr, png_infop info_ptr)
{
   if(png_ptr == NULL) return;
   switch (png_ptr->process_mode)
   {
      case PNG_READ_SIG_MODE:
      {
         png_push_read_sig(png_ptr, info_ptr);
         break;
      }
      case PNG_READ_CHUNK_MODE:
      {
         png_push_read_chunk(png_ptr, info_ptr);
         break;
      }
      case PNG_READ_IDAT_MODE:
      {
         png_push_read_IDAT(png_ptr);
         break;
      }
#if defined(PNG_READ_tEXt_SUPPORTED)
      case PNG_READ_tEXt_MODE:
      {
         png_push_read_tEXt(png_ptr, info_ptr);
         break;
      }
#endif
#if defined(PNG_READ_zTXt_SUPPORTED)
      case PNG_READ_zTXt_MODE:
      {
         png_push_read_zTXt(png_ptr, info_ptr);
         break;
      }
#endif
#if defined(PNG_READ_iTXt_SUPPORTED)
      case PNG_READ_iTXt_MODE:
      {
         png_push_read_iTXt(png_ptr, info_ptr);
         break;
      }
#endif
      case PNG_SKIP_MODE:
      {
         png_push_crc_finish(png_ptr);
         break;
      }
      default:
      {
         png_ptr->buffer_size = 0;
         break;
      }
   }
}

/* Read any remaining signature bytes from the stream and compare them with
 * the correct PNG signature.  It is possible that this routine is called
 * with bytes already read from the signature, either because they have been
 * checked by the calling application, or because of multiple calls to this
 * routine.
 */
void /* PRIVATE */
png_push_read_sig(png_structp png_ptr, png_infop info_ptr)
{
   png_size_t num_checked = png_ptr->sig_bytes,
             num_to_check = 8 - num_checked;

   if (png_ptr->buffer_size < num_to_check)
   {
      num_to_check = png_ptr->buffer_size;
   }

   png_push_fill_buffer(png_ptr, &(info_ptr->signature[num_checked]),
      num_to_check);
   png_ptr->sig_bytes = (png_byte)(png_ptr->sig_bytes+num_to_check);

   if (png_sig_cmp(info_ptr->signature, num_checked, num_to_check))
   {
      if (num_checked < 4 &&
          png_sig_cmp(info_ptr->signature, num_checked, num_to_check - 4))
         png_error(png_ptr, "Not a PNG file");
      else
         png_error(png_ptr, "PNG file corrupted by ASCII conversion");
   }
   else
   {
      if (png_ptr->sig_bytes >= 8)
      {
         png_ptr->process_mode = PNG_READ_CHUNK_MODE;
      }
   }
}

void /* PRIVATE */
png_push_read_chunk(png_structp png_ptr, png_infop info_ptr)
{
#ifdef PNG_USE_LOCAL_ARRAYS
      PNG_CONST PNG_IHDR;
      PNG_CONST PNG_IDAT;
      PNG_CONST PNG_IEND;
      PNG_CONST PNG_PLTE;
#if defined(PNG_READ_bKGD_SUPPORTED)
      PNG_CONST PNG_bKGD;
#endif
#if defined(PNG_READ_cHRM_SUPPORTED)
      PNG_CONST PNG_cHRM;
#endif
#if defined(PNG_READ_gAMA_SUPPORTED)
      PNG_CONST PNG_gAMA;
#endif
#if defined(PNG_READ_hIST_SUPPORTED)
      PNG_CONST PNG_hIST;
#endif
#if defined(PNG_READ_iCCP_SUPPORTED)
      PNG_CONST PNG_iCCP;
#endif
#if defined(PNG_READ_iTXt_SUPPORTED)
      PNG_CONST PNG_iTXt;
#endif
#if defined(PNG_READ_oFFs_SUPPORTED)
      PNG_CONST PNG_oFFs;
#endif
#if defined(PNG_READ_pCAL_SUPPORTED)
      PNG_CONST PNG_pCAL;
#endif
#if defined(PNG_READ_pHYs_SUPPORTED)
      PNG_CONST PNG_pHYs;
#endif
#if defined(PNG_READ_sBIT_SUPPORTED)
      PNG_CONST PNG_sBIT;
#endif
#if defined(PNG_READ_sCAL_SUPPORTED)
      PNG_CONST PNG_sCAL;
#endif
#if defined(PNG_READ_sRGB_SUPPORTED)
      PNG_CONST PNG_sRGB;
#endif
#if defined(PNG_READ_sPLT_SUPPORTED)
      PNG_CONST PNG_sPLT;
#endif
#if defined(PNG_READ_tEXt_SUPPORTED)
      PNG_CONST PNG_tEXt;
#endif
#if defined(PNG_READ_tIME_SUPPORTED)
      PNG_CONST PNG_tIME;
#endif
#if defined(PNG_READ_tRNS_SUPPORTED)
      PNG_CONST PNG_tRNS;
#endif
#if defined(PNG_READ_zTXt_SUPPORTED)
      PNG_CONST PNG_zTXt;
#endif
#endif /* PNG_USE_LOCAL_ARRAYS */
   /* First we make sure we have enough data for the 4 byte chunk name
    * and the 4 byte chunk length before proceeding with decoding the
    * chunk data.  To fully decode each of these chunks, we also make
    * sure we have enough data in the buffer for the 4 byte CRC at the
    * end of every chunk (except IDAT, which is handled separately).
    */
   if (!(png_ptr->mode & PNG_HAVE_CHUNK_HEADER))
   {
      png_byte chunk_length[4];

      if (png_ptr->buffer_size < 8)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_fill_buffer(png_ptr, chunk_length, 4);
      png_ptr->push_length = png_get_uint_31(png_ptr,chunk_length);
      png_reset_crc(png_ptr);
      png_crc_read(png_ptr, png_ptr->chunk_name, 4);
      png_ptr->mode |= PNG_HAVE_CHUNK_HEADER;
   }

   if (!png_memcmp(png_ptr->chunk_name, png_IDAT, 4))
     if(png_ptr->mode & PNG_AFTER_IDAT)
        png_ptr->mode |= PNG_HAVE_CHUNK_AFTER_IDAT;

   if (!png_memcmp(png_ptr->chunk_name, png_IHDR, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         if (png_ptr->push_length != 13)
            png_error(png_ptr, "Invalid IHDR length");
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_IHDR(png_ptr, info_ptr, png_ptr->push_length);
   }
   else if (!png_memcmp(png_ptr->chunk_name, png_IEND, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_IEND(png_ptr, info_ptr, png_ptr->push_length);

      png_ptr->process_mode = PNG_READ_DONE_MODE;
      png_push_have_end(png_ptr, info_ptr);
   }
#ifdef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
   else if (png_handle_as_unknown(png_ptr, png_ptr->chunk_name))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      if (!png_memcmp(png_ptr->chunk_name, png_IDAT, 4))
         png_ptr->mode |= PNG_HAVE_IDAT;
      png_handle_unknown(png_ptr, info_ptr, png_ptr->push_length);
      if (!png_memcmp(png_ptr->chunk_name, png_PLTE, 4))
         png_ptr->mode |= PNG_HAVE_PLTE;
      else if (!png_memcmp(png_ptr->chunk_name, png_IDAT, 4))
      {
         if (!(png_ptr->mode & PNG_HAVE_IHDR))
            png_error(png_ptr, "Missing IHDR before IDAT");
         else if (png_ptr->color_type == PNG_COLOR_TYPE_PALETTE &&
                  !(png_ptr->mode & PNG_HAVE_PLTE))
            png_error(png_ptr, "Missing PLTE before IDAT");
      }
   }
#endif
   else if (!png_memcmp(png_ptr->chunk_name, png_PLTE, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_PLTE(png_ptr, info_ptr, png_ptr->push_length);
   }
   else if (!png_memcmp(png_ptr->chunk_name, png_IDAT, 4))
   {
      /* If we reach an IDAT chunk, this means we have read all of the
       * header chunks, and we can start reading the image (or if this
       * is called after the image has been read - we have an error).
       */
     if (!(png_ptr->mode & PNG_HAVE_IHDR))
       png_error(png_ptr, "Missing IHDR before IDAT");
     else if (png_ptr->color_type == PNG_COLOR_TYPE_PALETTE &&
         !(png_ptr->mode & PNG_HAVE_PLTE))
       png_error(png_ptr, "Missing PLTE before IDAT");

      if (png_ptr->mode & PNG_HAVE_IDAT)
      {
         if (!(png_ptr->mode & PNG_HAVE_CHUNK_AFTER_IDAT))
           if (png_ptr->push_length == 0)
              return;

         if (png_ptr->mode & PNG_AFTER_IDAT)
            png_error(png_ptr, "Too many IDAT's found");
      }

      png_ptr->idat_size = png_ptr->push_length;
      png_ptr->mode |= PNG_HAVE_IDAT;
      png_ptr->process_mode = PNG_READ_IDAT_MODE;
      png_push_have_info(png_ptr, info_ptr);
      png_ptr->zstream.avail_out = (uInt)png_ptr->irowbytes;
      png_ptr->zstream.next_out = png_ptr->row_buf;
      return;
   }
#if defined(PNG_READ_gAMA_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_gAMA, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_gAMA(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_sBIT_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_sBIT, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_sBIT(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_cHRM_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_cHRM, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_cHRM(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_sRGB_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_sRGB, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_sRGB(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_iCCP_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_iCCP, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_iCCP(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_sPLT_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_sPLT, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_sPLT(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_tRNS_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_tRNS, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_tRNS(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_bKGD_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_bKGD, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_bKGD(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_hIST_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_hIST, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_hIST(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_pHYs_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_pHYs, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_pHYs(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_oFFs_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_oFFs, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_oFFs(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_pCAL_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_pCAL, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_pCAL(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_sCAL_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_sCAL, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_sCAL(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_tIME_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_tIME, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_tIME(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_tEXt_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_tEXt, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_push_handle_tEXt(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_zTXt_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_zTXt, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_push_handle_zTXt(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
#if defined(PNG_READ_iTXt_SUPPORTED)
   else if (!png_memcmp(png_ptr->chunk_name, png_iTXt, 4))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_push_handle_iTXt(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif
   else
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_push_handle_unknown(png_ptr, info_ptr, png_ptr->push_length);
   }

   png_ptr->mode &= ~PNG_HAVE_CHUNK_HEADER;
}

void /* PRIVATE */
png_push_crc_skip(png_structp png_ptr, png_uint_32 skip)
{
   png_ptr->process_mode = PNG_SKIP_MODE;
   png_ptr->skip_length = skip;
}

void /* PRIVATE */
png_push_crc_finish(png_structp png_ptr)
{
   if (png_ptr->skip_length && png_ptr->save_buffer_size)
   {
      png_size_t save_size;

      if (png_ptr->skip_length < (png_uint_32)png_ptr->save_buffer_size)
         save_size = (png_size_t)png_ptr->skip_length;
      else
         save_size = png_ptr->save_buffer_size;

      png_calculate_crc(png_ptr, png_ptr->save_buffer_ptr, save_size);

      png_ptr->skip_length -= save_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->save_buffer_size -= save_size;
      png_ptr->save_buffer_ptr += save_size;
   }
   if (png_ptr->skip_length && png_ptr->current_buffer_size)
   {
      png_size_t save_size;

      if (png_ptr->skip_length < (png_uint_32)png_ptr->current_buffer_size)
         save_size = (png_size_t)png_ptr->skip_length;
      else
         save_size = png_ptr->current_buffer_size;

      png_calculate_crc(png_ptr, png_ptr->current_buffer_ptr, save_size);

      png_ptr->skip_length -= save_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->current_buffer_size -= save_size;
      png_ptr->current_buffer_ptr += save_size;
   }
   if (!png_ptr->skip_length)
   {
      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_crc_finish(png_ptr, 0);
      png_ptr->process_mode = PNG_READ_CHUNK_MODE;
   }
}

void PNGAPI
png_push_fill_buffer(png_structp png_ptr, png_bytep buffer, png_size_t length)
{
   png_bytep ptr;

   if(png_ptr == NULL) return;
   ptr = buffer;
   if (png_ptr->save_buffer_size)
   {
      png_size_t save_size;

      if (length < png_ptr->save_buffer_size)
         save_size = length;
      else
         save_size = png_ptr->save_buffer_size;

      png_memcpy(ptr, png_ptr->save_buffer_ptr, save_size);
      length -= save_size;
      ptr += save_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->save_buffer_size -= save_size;
      png_ptr->save_buffer_ptr += save_size;
   }
   if (length && png_ptr->current_buffer_size)
   {
      png_size_t save_size;

      if (length < png_ptr->current_buffer_size)
         save_size = length;
      else
         save_size = png_ptr->current_buffer_size;

      png_memcpy(ptr, png_ptr->current_buffer_ptr, save_size);
      png_ptr->buffer_size -= save_size;
      png_ptr->current_buffer_size -= save_size;
      png_ptr->current_buffer_ptr += save_size;
   }
}

void /* PRIVATE */
png_push_save_buffer(png_structp png_ptr)
{
   if (png_ptr->save_buffer_size)
   {
      if (png_ptr->save_buffer_ptr != png_ptr->save_buffer)
      {
         png_size_t i,istop;
         png_bytep sp;
         png_bytep dp;

         istop = png_ptr->save_buffer_size;
         for (i = 0, sp = png_ptr->save_buffer_ptr, dp = png_ptr->save_buffer;
            i < istop; i++, sp++, dp++)
         {
            *dp = *sp;
         }
      }
   }
   if (png_ptr->save_buffer_size + png_ptr->current_buffer_size >
      png_ptr->save_buffer_max)
   {
      png_size_t new_max;
      png_bytep old_buffer;

      if (png_ptr->save_buffer_size > PNG_SIZE_MAX -
         (png_ptr->current_buffer_size + 256))
      {
        png_error(png_ptr, "Potential overflow of save_buffer");
      }
      new_max = png_ptr->save_buffer_size + png_ptr->current_buffer_size + 256;
      old_buffer = png_ptr->save_buffer;
      png_ptr->save_buffer = (png_bytep)png_malloc(png_ptr,
         (png_uint_32)new_max);
      png_memcpy(png_ptr->save_buffer, old_buffer, png_ptr->save_buffer_size);
      png_free(png_ptr, old_buffer);
      png_ptr->save_buffer_max = new_max;
   }
   if (png_ptr->current_buffer_size)
   {
      png_memcpy(png_ptr->save_buffer + png_ptr->save_buffer_size,
         png_ptr->current_buffer_ptr, png_ptr->current_buffer_size);
      png_ptr->save_buffer_size += png_ptr->current_buffer_size;
      png_ptr->current_buffer_size = 0;
   }
   png_ptr->save_buffer_ptr = png_ptr->save_buffer;
   png_ptr->buffer_size = 0;
}

void /* PRIVATE */
png_push_restore_buffer(png_structp png_ptr, png_bytep buffer,
   png_size_t buffer_length)
{
   png_ptr->current_buffer = buffer;
   png_ptr->current_buffer_size = buffer_length;
   png_ptr->buffer_size = buffer_length + png_ptr->save_buffer_size;
   png_ptr->current_buffer_ptr = png_ptr->current_buffer;
}

void /* PRIVATE */
png_push_read_IDAT(png_structp png_ptr)
{
#ifdef PNG_USE_LOCAL_ARRAYS
   PNG_CONST PNG_IDAT;
#endif
   if (!(png_ptr->mode & PNG_HAVE_CHUNK_HEADER))
   {
      png_byte chunk_length[4];

      if (png_ptr->buffer_size < 8)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_fill_buffer(png_ptr, chunk_length, 4);
      png_ptr->push_length = png_get_uint_31(png_ptr,chunk_length);
      png_reset_crc(png_ptr);
      png_crc_read(png_ptr, png_ptr->chunk_name, 4);
      png_ptr->mode |= PNG_HAVE_CHUNK_HEADER;

      if (png_memcmp(png_ptr->chunk_name, png_IDAT, 4))
      {
         png_ptr->process_mode = PNG_READ_CHUNK_MODE;
         if (!(png_ptr->flags & PNG_FLAG_ZLIB_FINISHED))
            png_error(png_ptr, "Not enough compressed data");
         return;
      }

      png_ptr->idat_size = png_ptr->push_length;
   }
   if (png_ptr->idat_size && png_ptr->save_buffer_size)
   {
      png_size_t save_size;

      if (png_ptr->idat_size < (png_uint_32)png_ptr->save_buffer_size)
      {
         save_size = (png_size_t)png_ptr->idat_size;
         /* check for overflow */
         if((png_uint_32)save_size != png_ptr->idat_size)
            png_error(png_ptr, "save_size overflowed in pngpread");
      }
      else
         save_size = png_ptr->save_buffer_size;

      png_calculate_crc(png_ptr, png_ptr->save_buffer_ptr, save_size);
      if (!(png_ptr->flags & PNG_FLAG_ZLIB_FINISHED))
         png_process_IDAT_data(png_ptr, png_ptr->save_buffer_ptr, save_size);
      png_ptr->idat_size -= save_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->save_buffer_size -= save_size;
      png_ptr->save_buffer_ptr += save_size;
   }
   if (png_ptr->idat_size && png_ptr->current_buffer_size)
   {
      png_size_t save_size;

      if (png_ptr->idat_size < (png_uint_32)png_ptr->current_buffer_size)
      {
         save_size = (png_size_t)png_ptr->idat_size;
         /* check for overflow */
         if((png_uint_32)save_size != png_ptr->idat_size)
            png_error(png_ptr, "save_size overflowed in pngpread");
      }
      else
         save_size = png_ptr->current_buffer_size;

      png_calculate_crc(png_ptr, png_ptr->current_buffer_ptr, save_size);
      if (!(png_ptr->flags & PNG_FLAG_ZLIB_FINISHED))
        png_process_IDAT_data(png_ptr, png_ptr->current_buffer_ptr, save_size);

      png_ptr->idat_size -= save_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->current_buffer_size -= save_size;
      png_ptr->current_buffer_ptr += save_size;
   }
   if (!png_ptr->idat_size)
   {
      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_crc_finish(png_ptr, 0);
      png_ptr->mode &= ~PNG_HAVE_CHUNK_HEADER;
      png_ptr->mode |= PNG_AFTER_IDAT;
   }
}

void /* PRIVATE */
png_process_IDAT_data(png_structp png_ptr, png_bytep buffer,
   png_size_t buffer_length)
{
   int ret;

   if ((png_ptr->flags & PNG_FLAG_ZLIB_FINISHED) && buffer_length)
      png_error(png_ptr, "Extra compression data");

   png_ptr->zstream.next_in = buffer;
   png_ptr->zstream.avail_in = (uInt)buffer_length;
   for(;;)
   {
      ret = inflate(&png_ptr->zstream, Z_PARTIAL_FLUSH);
      if (ret != Z_OK)
      {
         if (ret == Z_STREAM_END)
         {
            if (png_ptr->zstream.avail_in)
               png_error(png_ptr, "Extra compressed data");
            if (!(png_ptr->zstream.avail_out))
            {
               png_push_process_row(png_ptr);
            }

            png_ptr->mode |= PNG_AFTER_IDAT;
            png_ptr->flags |= PNG_FLAG_ZLIB_FINISHED;
            break;
         }
         else if (ret == Z_BUF_ERROR)
            break;
         else
            png_error(png_ptr, "Decompression Error");
      }
      if (!(png_ptr->zstream.avail_out))
      {
         if ((
#if defined(PNG_READ_INTERLACING_SUPPORTED)
             png_ptr->interlaced && png_ptr->pass > 6) ||
             (!png_ptr->interlaced &&
#endif
             png_ptr->row_number == png_ptr->num_rows))
         {
           if (png_ptr->zstream.avail_in)
             png_warning(png_ptr, "Too much data in IDAT chunks");
           png_ptr->flags |= PNG_FLAG_ZLIB_FINISHED;
           break;
         }
         png_push_process_row(png_ptr);
         png_ptr->zstream.avail_out = (uInt)png_ptr->irowbytes;
         png_ptr->zstream.next_out = png_ptr->row_buf;
      }
      else
         break;
   }
}

void /* PRIVATE */
png_push_process_row(png_structp png_ptr)
{
   png_ptr->row_info.color_type = png_ptr->color_type;
   png_ptr->row_info.width = png_ptr->iwidth;
   png_ptr->row_info.channels = png_ptr->channels;
   png_ptr->row_info.bit_depth = png_ptr->bit_depth;
   png_ptr->row_info.pixel_depth = png_ptr->pixel_depth;

   png_ptr->row_info.rowbytes = PNG_ROWBYTES(png_ptr->row_info.pixel_depth,
       png_ptr->row_info.width);

   png_read_filter_row(png_ptr, &(png_ptr->row_info),
      png_ptr->row_buf + 1, png_ptr->prev_row + 1,
      (int)(png_ptr->row_buf[0]));

   png_memcpy_check(png_ptr, png_ptr->prev_row, png_ptr->row_buf,
      png_ptr->rowbytes + 1);

   if (png_ptr->transformations || (png_ptr->flags&PNG_FLAG_STRIP_ALPHA))
      png_do_read_transformations(png_ptr);

#if defined(PNG_READ_INTERLACING_SUPPORTED)
   /* blow up interlaced rows to full size */
   if (png_ptr->interlaced && (png_ptr->transformations & PNG_INTERLACE))
   {
      if (png_ptr->pass < 6)
/*       old interface (pre-1.0.9):
         png_do_read_interlace(&(png_ptr->row_info),
            png_ptr->row_buf + 1, png_ptr->pass, png_ptr->transformations);
 */
         png_do_read_interlace(png_ptr);

    switch (png_ptr->pass)
    {
         case 0:
         {
            int i;
            for (i = 0; i < 8 && png_ptr->pass == 0; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr); /* updates png_ptr->pass */
            }
            if (png_ptr->pass == 2) /* pass 1 might be empty */
            {
               for (i = 0; i < 4 && png_ptr->pass == 2; i++)
               {
                  png_push_have_row(png_ptr, png_bytep_NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }
            if (png_ptr->pass == 4 && png_ptr->height <= 4)
            {
               for (i = 0; i < 2 && png_ptr->pass == 4; i++)
               {
                  png_push_have_row(png_ptr, png_bytep_NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }
            if (png_ptr->pass == 6 && png_ptr->height <= 4)
            {
                png_push_have_row(png_ptr, png_bytep_NULL);
                png_read_push_finish_row(png_ptr);
            }
            break;
         }
         case 1:
         {
            int i;
            for (i = 0; i < 8 && png_ptr->pass == 1; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }
            if (png_ptr->pass == 2) /* skip top 4 generated rows */
            {
               for (i = 0; i < 4 && png_ptr->pass == 2; i++)
               {
                  png_push_have_row(png_ptr, png_bytep_NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }
            break;
         }
         case 2:
         {
            int i;
            for (i = 0; i < 4 && png_ptr->pass == 2; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }
            for (i = 0; i < 4 && png_ptr->pass == 2; i++)
            {
               png_push_have_row(png_ptr, png_bytep_NULL);
               png_read_push_finish_row(png_ptr);
            }
            if (png_ptr->pass == 4) /* pass 3 might be empty */
            {
               for (i = 0; i < 2 && png_ptr->pass == 4; i++)
               {
                  png_push_have_row(png_ptr, png_bytep_NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }
            break;
         }
         case 3:
         {
            int i;
            for (i = 0; i < 4 && png_ptr->pass == 3; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }
            if (png_ptr->pass == 4) /* skip top two generated rows */
            {
               for (i = 0; i < 2 && png_ptr->pass == 4; i++)
               {
                  png_push_have_row(png_ptr, png_bytep_NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }
            break;
         }
         case 4:
         {
            int i;
            for (i = 0; i < 2 && png_ptr->pass == 4; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }
            for (i = 0; i < 2 && png_ptr->pass == 4; i++)
            {
               png_push_have_row(png_ptr, png_bytep_NULL);
               png_read_push_finish_row(png_ptr);
            }
            if (png_ptr->pass == 6) /* pass 5 might be empty */
            {
               png_push_have_row(png_ptr, png_bytep_NULL);
               png_read_push_finish_row(png_ptr);
            }
            break;
         }
         case 5:
         {
            int i;
            for (i = 0; i < 2 && png_ptr->pass == 5; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }
            if (png_ptr->pass == 6) /* skip top generated row */
            {
               png_push_have_row(png_ptr, png_bytep_NULL);
               png_read_push_finish_row(png_ptr);
            }
            break;
         }
         case 6:
         {
            png_push_have_row(png_ptr, png_ptr->row_buf + 1);
            png_read_push_finish_row(png_ptr);
            if (png_ptr->pass != 6)
               break;
            png_push_have_row(png_ptr, png_bytep_NULL);
            png_read_push_finish_row(png_ptr);
         }
      }
   }
   else
#endif
   {
      png_push_have_row(png_ptr, png_ptr->row_buf + 1);
      png_read_push_finish_row(png_ptr);
   }
}

void /* PRIVATE */
png_read_push_finish_row(png_structp png_ptr)
{
#ifdef PNG_USE_LOCAL_ARRAYS
   /* arrays to facilitate easy interlacing - use pass (0 - 6) as index */

   /* start of interlace block */
   PNG_CONST int FARDATA png_pass_start[] = {0, 4, 0, 2, 0, 1, 0};

   /* offset to next interlace block */
   PNG_CONST int FARDATA png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1};

   /* start of interlace block in the y direction */
   PNG_CONST int FARDATA png_pass_ystart[] = {0, 0, 4, 0, 2, 0, 1};

   /* offset to next interlace block in the y direction */
   PNG_CONST int FARDATA png_pass_yinc[] = {8, 8, 8, 4, 4, 2, 2};

   /* Height of interlace block.  This is not currently used - if you need
    * it, uncomment it here and in png.h
   PNG_CONST int FARDATA png_pass_height[] = {8, 8, 4, 4, 2, 2, 1};
   */
#endif

   png_ptr->row_number++;
   if (png_ptr->row_number < png_ptr->num_rows)
      return;

   if (png_ptr->interlaced)
   {
      png_ptr->row_number = 0;
      png_memset_check(png_ptr, png_ptr->prev_row, 0,
         png_ptr->rowbytes + 1);
      do
      {
         int pass;
         pass = png_ptr->pass;
         pass++;
         if ((pass == 1 && png_ptr->width < 5) ||
             (pass == 3 && png_ptr->width < 3) ||
             (pass == 5 && png_ptr->width < 2))
           pass++;

         if (pass > 7)
            pass--;
         png_ptr->pass = (png_byte) pass;
         if (pass < 7)
           {
             png_ptr->iwidth = (png_ptr->width +
                png_pass_inc[pass] - 1 -
                png_pass_start[pass]) /
                png_pass_inc[pass];

             png_ptr->irowbytes = PNG_ROWBYTES(png_ptr->pixel_depth,
                png_ptr->iwidth) + 1;

             if (png_ptr->transformations & PNG_INTERLACE)
                break;

             png_ptr->num_rows = (png_ptr->height +
                png_pass_yinc[pass] - 1 -
                png_pass_ystart[pass]) /
                png_pass_yinc[pass];
           }
         else
           break;

      } while (png_ptr->iwidth == 0 || png_ptr->num_rows == 0);
   }
}

#if defined(PNG_READ_tEXt_SUPPORTED)
void /* PRIVATE */
png_push_handle_tEXt(png_structp png_ptr, png_infop info_ptr, png_uint_32
   length)
{
   if (!(png_ptr->mode & PNG_HAVE_IHDR) || (png_ptr->mode & PNG_HAVE_IEND))
      {
         png_error(png_ptr, "Out of place tEXt");
         info_ptr = info_ptr; /* to quiet some compiler warnings */
      }

#ifdef PNG_MAX_MALLOC_64K
   png_ptr->skip_length = 0;  /* This may not be necessary */

   if (length > (png_uint_32)65535L) /* Can't hold entire string in memory */
   {
      png_warning(png_ptr, "tEXt chunk too large to fit in memory");
      png_ptr->skip_length = length - (png_uint_32)65535L;
      length = (png_uint_32)65535L;
   }
#endif

   png_ptr->current_text = (png_charp)png_malloc(png_ptr,
         (png_uint_32)(length+1));
   png_ptr->current_text[length] = '\0';
   png_ptr->current_text_ptr = png_ptr->current_text;
   png_ptr->current_text_size = (png_size_t)length;
   png_ptr->current_text_left = (png_size_t)length;
   png_ptr->process_mode = PNG_READ_tEXt_MODE;
}

void /* PRIVATE */
png_push_read_tEXt(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->buffer_size && png_ptr->current_text_left)
   {
      png_size_t text_size;

      if (png_ptr->buffer_size < png_ptr->current_text_left)
         text_size = png_ptr->buffer_size;
      else
         text_size = png_ptr->current_text_left;
      png_crc_read(png_ptr, (png_bytep)png_ptr->current_text_ptr, text_size);
      png_ptr->current_text_left -= text_size;
      png_ptr->current_text_ptr += text_size;
   }
   if (!(png_ptr->current_text_left))
   {
      png_textp text_ptr;
      png_charp text;
      png_charp key;
      int ret;

      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_crc_finish(png_ptr);

#if defined(PNG_MAX_MALLOC_64K)
      if (png_ptr->skip_length)
         return;
#endif

      key = png_ptr->current_text;

      for (text = key; *text; text++)
         /* empty loop */ ;

      if (text < key + png_ptr->current_text_size)
         text++;

      text_ptr = (png_textp)png_malloc(png_ptr,
         (png_uint_32)png_sizeof(png_text));
      text_ptr->compression = PNG_TEXT_COMPRESSION_NONE;
      text_ptr->key = key;
#ifdef PNG_iTXt_SUPPORTED
      text_ptr->lang = NULL;
      text_ptr->lang_key = NULL;
#endif
      text_ptr->text = text;

      ret = png_set_text_2(png_ptr, info_ptr, text_ptr, 1);

      png_free(png_ptr, key);
      png_free(png_ptr, text_ptr);
      png_ptr->current_text = NULL;

      if (ret)
        png_warning(png_ptr, "Insufficient memory to store text chunk.");
   }
}
#endif

#if defined(PNG_READ_zTXt_SUPPORTED)
void /* PRIVATE */
png_push_handle_zTXt(png_structp png_ptr, png_infop info_ptr, png_uint_32
   length)
{
   if (!(png_ptr->mode & PNG_HAVE_IHDR) || (png_ptr->mode & PNG_HAVE_IEND))
      {
         png_error(png_ptr, "Out of place zTXt");
         info_ptr = info_ptr; /* to quiet some compiler warnings */
      }

#ifdef PNG_MAX_MALLOC_64K
   /* We can't handle zTXt chunks > 64K, since we don't have enough space
    * to be able to store the uncompressed data.  Actually, the threshold
    * is probably around 32K, but it isn't as definite as 64K is.
    */
   if (length > (png_uint_32)65535L)
   {
      png_warning(png_ptr, "zTXt chunk too large to fit in memory");
      png_push_crc_skip(png_ptr, length);
      return;
   }
#endif

   png_ptr->current_text = (png_charp)png_malloc(png_ptr,
       (png_uint_32)(length+1));
   png_ptr->current_text[length] = '\0';
   png_ptr->current_text_ptr = png_ptr->current_text;
   png_ptr->current_text_size = (png_size_t)length;
   png_ptr->current_text_left = (png_size_t)length;
   png_ptr->process_mode = PNG_READ_zTXt_MODE;
}

void /* PRIVATE */
png_push_read_zTXt(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->buffer_size && png_ptr->current_text_left)
   {
      png_size_t text_size;

      if (png_ptr->buffer_size < (png_uint_32)png_ptr->current_text_left)
         text_size = png_ptr->buffer_size;
      else
         text_size = png_ptr->current_text_left;
      png_crc_read(png_ptr, (png_bytep)png_ptr->current_text_ptr, text_size);
      png_ptr->current_text_left -= text_size;
      png_ptr->current_text_ptr += text_size;
   }
   if (!(png_ptr->current_text_left))
   {
      png_textp text_ptr;
      png_charp text;
      png_charp key;
      int ret;
      png_size_t text_size, key_size;

      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_crc_finish(png_ptr);

      key = png_ptr->current_text;

      for (text = key; *text; text++)
         /* empty loop */ ;

      /* zTXt can't have zero text */
      if (text >= key + png_ptr->current_text_size)
      {
         png_ptr->current_text = NULL;
         png_free(png_ptr, key);
         return;
      }

      text++;

      if (*text != PNG_TEXT_COMPRESSION_zTXt) /* check compression byte */
      {
         png_ptr->current_text = NULL;
         png_free(png_ptr, key);
         return;
      }

      text++;

      png_ptr->zstream.next_in = (png_bytep )text;
      png_ptr->zstream.avail_in = (uInt)(png_ptr->current_text_size -
         (text - key));
      png_ptr->zstream.next_out = png_ptr->zbuf;
      png_ptr->zstream.avail_out = (uInt)png_ptr->zbuf_size;

      key_size = text - key;
      text_size = 0;
      text = NULL;
      ret = Z_STREAM_END;

      while (png_ptr->zstream.avail_in)
      {
         ret = inflate(&png_ptr->zstream, Z_PARTIAL_FLUSH);
         if (ret != Z_OK && ret != Z_STREAM_END)
         {
            inflateReset(&png_ptr->zstream);
            png_ptr->zstream.avail_in = 0;
            png_ptr->current_text = NULL;
            png_free(png_ptr, key);
            png_free(png_ptr, text);
            return;
         }
         if (!(png_ptr->zstream.avail_out) || ret == Z_STREAM_END)
         {
            if (text == NULL)
            {
               text = (png_charp)png_malloc(png_ptr,
                  (png_uint_32)(png_ptr->zbuf_size - png_ptr->zstream.avail_out
                     + key_size + 1));
               png_memcpy(text + key_size, png_ptr->zbuf,
                  png_ptr->zbuf_size - png_ptr->zstream.avail_out);
               png_memcpy(text, key, key_size);
               text_size = key_size + png_ptr->zbuf_size -
                  png_ptr->zstream.avail_out;
               *(text + text_size) = '\0';
            }
            else
            {
               png_charp tmp;

               tmp = text;
               text = (png_charp)png_malloc(png_ptr, text_size +
                  (png_uint_32)(png_ptr->zbuf_size - png_ptr->zstream.avail_out
                   + 1));
               png_memcpy(text, tmp, text_size);
               png_free(png_ptr, tmp);
               png_memcpy(text + text_size, png_ptr->zbuf,
                  png_ptr->zbuf_size - png_ptr->zstream.avail_out);
               text_size += png_ptr->zbuf_size - png_ptr->zstream.avail_out;
               *(text + text_size) = '\0';
            }
            if (ret != Z_STREAM_END)
            {
               png_ptr->zstream.next_out = png_ptr->zbuf;
               png_ptr->zstream.avail_out = (uInt)png_ptr->zbuf_size;
            }
         }
         else
         {
            break;
         }

         if (ret == Z_STREAM_END)
            break;
      }

      inflateReset(&png_ptr->zstream);
      png_ptr->zstream.avail_in = 0;

      if (ret != Z_STREAM_END)
      {
         png_ptr->current_text = NULL;
         png_free(png_ptr, key);
         png_free(png_ptr, text);
         return;
      }

      png_ptr->current_text = NULL;
      png_free(png_ptr, key);
      key = text;
      text += key_size;

      text_ptr = (png_textp)png_malloc(png_ptr,
          (png_uint_32)png_sizeof(png_text));
      text_ptr->compression = PNG_TEXT_COMPRESSION_zTXt;
      text_ptr->key = key;
#ifdef PNG_iTXt_SUPPORTED
      text_ptr->lang = NULL;
      text_ptr->lang_key = NULL;
#endif
      text_ptr->text = text;

      ret = png_set_text_2(png_ptr, info_ptr, text_ptr, 1);

      png_free(png_ptr, key);
      png_free(png_ptr, text_ptr);

      if (ret)
        png_warning(png_ptr, "Insufficient memory to store text chunk.");
   }
}
#endif

#if defined(PNG_READ_iTXt_SUPPORTED)
void /* PRIVATE */
png_push_handle_iTXt(png_structp png_ptr, png_infop info_ptr, png_uint_32
   length)
{
   if (!(png_ptr->mode & PNG_HAVE_IHDR) || (png_ptr->mode & PNG_HAVE_IEND))
      {
         png_error(png_ptr, "Out of place iTXt");
         info_ptr = info_ptr; /* to quiet some compiler warnings */
      }

#ifdef PNG_MAX_MALLOC_64K
   png_ptr->skip_length = 0;  /* This may not be necessary */

   if (length > (png_uint_32)65535L) /* Can't hold entire string in memory */
   {
      png_warning(png_ptr, "iTXt chunk too large to fit in memory");
      png_ptr->skip_length = length - (png_uint_32)65535L;
      length = (png_uint_32)65535L;
   }
#endif

   png_ptr->current_text = (png_charp)png_malloc(png_ptr,
         (png_uint_32)(length+1));
   png_ptr->current_text[length] = '\0';
   png_ptr->current_text_ptr = png_ptr->current_text;
   png_ptr->current_text_size = (png_size_t)length;
   png_ptr->current_text_left = (png_size_t)length;
   png_ptr->process_mode = PNG_READ_iTXt_MODE;
}

void /* PRIVATE */
png_push_read_iTXt(png_structp png_ptr, png_infop info_ptr)
{

   if (png_ptr->buffer_size && png_ptr->current_text_left)
   {
      png_size_t text_size;

      if (png_ptr->buffer_size < png_ptr->current_text_left)
         text_size = png_ptr->buffer_size;
      else
         text_size = png_ptr->current_text_left;
      png_crc_read(png_ptr, (png_bytep)png_ptr->current_text_ptr, text_size);
      png_ptr->current_text_left -= text_size;
      png_ptr->current_text_ptr += text_size;
   }
   if (!(png_ptr->current_text_left))
   {
      png_textp text_ptr;
      png_charp key;
      int comp_flag;
      png_charp lang;
      png_charp lang_key;
      png_charp text;
      int ret;

      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_crc_finish(png_ptr);

#if defined(PNG_MAX_MALLOC_64K)
      if (png_ptr->skip_length)
         return;
#endif

      key = png_ptr->current_text;

      for (lang = key; *lang; lang++)
         /* empty loop */ ;

      if (lang < key + png_ptr->current_text_size - 3)
         lang++;

      comp_flag = *lang++;
      lang++;     /* skip comp_type, always zero */

      for (lang_key = lang; *lang_key; lang_key++)
         /* empty loop */ ;
      lang_key++;        /* skip NUL separator */

      text=lang_key;
      if (lang_key < key + png_ptr->current_text_size - 1)
      {
        for (; *text; text++)
           /* empty loop */ ;
      }

      if (text < key + png_ptr->current_text_size)
         text++;

      text_ptr = (png_textp)png_malloc(png_ptr,
         (png_uint_32)png_sizeof(png_text));
      text_ptr->compression = comp_flag + 2;
      text_ptr->key = key;
      text_ptr->lang = lang;
      text_ptr->lang_key = lang_key;
      text_ptr->text = text;
      text_ptr->text_length = 0;
      text_ptr->itxt_length = png_strlen(text);

      ret = png_set_text_2(png_ptr, info_ptr, text_ptr, 1);

      png_ptr->current_text = NULL;

      png_free(png_ptr, text_ptr);
      if (ret)
        png_warning(png_ptr, "Insufficient memory to store iTXt chunk.");
   }
}
#endif

/* This function is called when we haven't found a handler for this
 * chunk.  If there isn't a problem with the chunk itself (ie a bad chunk
 * name or a critical chunk), the chunk is (currently) silently ignored.
 */
void /* PRIVATE */
png_push_handle_unknown(png_structp png_ptr, png_infop info_ptr, png_uint_32
   length)
{
   png_uint_32 skip=0;
   png_check_chunk_name(png_ptr, png_ptr->chunk_name);

   if (!(png_ptr->chunk_name[0] & 0x20))
   {
#if defined(PNG_READ_UNKNOWN_CHUNKS_SUPPORTED)
     if(png_handle_as_unknown(png_ptr, png_ptr->chunk_name) !=
          PNG_HANDLE_CHUNK_ALWAYS
#if defined(PNG_READ_USER_CHUNKS_SUPPORTED)
          && png_ptr->read_user_chunk_fn == NULL
#endif
        )
#endif
        png_chunk_error(png_ptr, "unknown critical chunk");

     info_ptr = info_ptr; /* to quiet some compiler warnings */
   }

#if defined(PNG_READ_UNKNOWN_CHUNKS_SUPPORTED)
   if (png_ptr->flags & PNG_FLAG_KEEP_UNKNOWN_CHUNKS)
   {
#ifdef PNG_MAX_MALLOC_64K
      if (length > (png_uint_32)65535L)
      {
          png_warning(png_ptr, "unknown chunk too large to fit in memory");
          skip = length - (png_uint_32)65535L;
          length = (png_uint_32)65535L;
      }
#endif
      png_memcpy((png_charp)png_ptr->unknown_chunk.name,
                 (png_charp)png_ptr->chunk_name, 
                 png_sizeof(png_ptr->unknown_chunk.name));
      png_ptr->unknown_chunk.name[png_sizeof(png_ptr->unknown_chunk.name)-1]='\0';

      png_ptr->unknown_chunk.size = (png_size_t)length;
      if (length == 0)
         png_ptr->unknown_chunk.data = NULL;
      else
      {
         png_ptr->unknown_chunk.data = (png_bytep)png_malloc(png_ptr, length);
         png_crc_read(png_ptr, (png_bytep)png_ptr->unknown_chunk.data, length);
      }
#if defined(PNG_READ_USER_CHUNKS_SUPPORTED)
      if(png_ptr->read_user_chunk_fn != NULL)
      {
         /* callback to user unknown chunk handler */
         int ret;
         ret = (*(png_ptr->read_user_chunk_fn))
           (png_ptr, &png_ptr->unknown_chunk);
         if (ret < 0)
            png_chunk_error(png_ptr, "error in user chunk");
         if (ret == 0)
         {
            if (!(png_ptr->chunk_name[0] & 0x20))
               if(png_handle_as_unknown(png_ptr, png_ptr->chunk_name) !=
                    PNG_HANDLE_CHUNK_ALWAYS)
                  png_chunk_error(png_ptr, "unknown critical chunk");
            png_set_unknown_chunks(png_ptr, info_ptr,
               &png_ptr->unknown_chunk, 1);
         }
      }
      else
#endif
        png_set_unknown_chunks(png_ptr, info_ptr, &png_ptr->unknown_chunk, 1);
      png_free(png_ptr, png_ptr->unknown_chunk.data);
      png_ptr->unknown_chunk.data = NULL;
   }
   else
#endif
      skip=length;
   png_push_crc_skip(png_ptr, skip);
}

void /* PRIVATE */
png_push_have_info(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->info_fn != NULL)
      (*(png_ptr->info_fn))(png_ptr, info_ptr);
}

void /* PRIVATE */
png_push_have_end(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->end_fn != NULL)
      (*(png_ptr->end_fn))(png_ptr, info_ptr);
}

void /* PRIVATE */
png_push_have_row(png_structp png_ptr, png_bytep row)
{
   if (png_ptr->row_fn != NULL)
      (*(png_ptr->row_fn))(png_ptr, row, png_ptr->row_number,
         (int)png_ptr->pass);
}

void PNGAPI
png_progressive_combine_row (png_structp png_ptr,
   png_bytep old_row, png_bytep new_row)
{
#ifdef PNG_USE_LOCAL_ARRAYS
   PNG_CONST int FARDATA png_pass_dsp_mask[7] =
      {0xff, 0x0f, 0xff, 0x33, 0xff, 0x55, 0xff};
#endif
   if(png_ptr == NULL) return;
   if (new_row != NULL)    /* new_row must == png_ptr->row_buf here. */
      png_combine_row(png_ptr, old_row, png_pass_dsp_mask[png_ptr->pass]);
}

void PNGAPI
png_set_progressive_read_fn(png_structp png_ptr, png_voidp progressive_ptr,
   png_progressive_info_ptr info_fn, png_progressive_row_ptr row_fn,
   png_progressive_end_ptr end_fn)
{
   if(png_ptr == NULL) return;
   png_ptr->info_fn = info_fn;
   png_ptr->row_fn = row_fn;
   png_ptr->end_fn = end_fn;

   png_set_read_fn(png_ptr, progressive_ptr, png_push_fill_buffer);
}

png_voidp PNGAPI
png_get_progressive_ptr(png_structp png_ptr)
{
   if(png_ptr == NULL) return (NULL);
   return png_ptr->io_ptr;
}
#endif /* PNG_PROGRESSIVE_READ_SUPPORTED */
