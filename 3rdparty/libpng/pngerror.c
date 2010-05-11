
/* pngerror.c - stub functions for i/o and memory allocation
 *
 * Last changed in libpng 1.2.22 [October 13, 2007]
 * For conditions of distribution and use, see copyright notice in png.h
 * Copyright (c) 1998-2007 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This file provides a location for all error handling.  Users who
 * need special error handling are expected to write replacement functions
 * and use png_set_error_fn() to use those functions.  See the instructions
 * at each function.
 */

#define PNG_INTERNAL
#include "png.h"

#if defined(PNG_READ_SUPPORTED) || defined(PNG_WRITE_SUPPORTED)
static void /* PRIVATE */
png_default_error PNGARG((png_structp png_ptr,
  png_const_charp error_message));
#ifndef PNG_NO_WARNINGS
static void /* PRIVATE */
png_default_warning PNGARG((png_structp png_ptr,
  png_const_charp warning_message));
#endif /* PNG_NO_WARNINGS */

/* This function is called whenever there is a fatal error.  This function
 * should not be changed.  If there is a need to handle errors differently,
 * you should supply a replacement error function and use png_set_error_fn()
 * to replace the error function at run-time.
 */
#ifndef PNG_NO_ERROR_TEXT
void PNGAPI
png_error(png_structp png_ptr, png_const_charp error_message)
{
#ifdef PNG_ERROR_NUMBERS_SUPPORTED
   char msg[16];
   if (png_ptr != NULL)
   {
     if (png_ptr->flags&
       (PNG_FLAG_STRIP_ERROR_NUMBERS|PNG_FLAG_STRIP_ERROR_TEXT))
     {
       if (*error_message == '#')
       {
           int offset;
           for (offset=1; offset<15; offset++)
              if (*(error_message+offset) == ' ')
                  break;
           if (png_ptr->flags&PNG_FLAG_STRIP_ERROR_TEXT)
           {
              int i;
              for (i=0; i<offset-1; i++)
                 msg[i]=error_message[i+1];
              msg[i]='\0';
              error_message=msg;
           }
           else
              error_message+=offset;
       }
       else
       {
           if (png_ptr->flags&PNG_FLAG_STRIP_ERROR_TEXT)
           {
              msg[0]='0';
              msg[1]='\0';
              error_message=msg;
           }
       }
     }
   }
#endif
   if (png_ptr != NULL && png_ptr->error_fn != NULL)
      (*(png_ptr->error_fn))(png_ptr, error_message);

   /* If the custom handler doesn't exist, or if it returns,
      use the default handler, which will not return. */
   png_default_error(png_ptr, error_message);
}
#else
void PNGAPI
png_err(png_structp png_ptr)
{
   if (png_ptr != NULL && png_ptr->error_fn != NULL)
      (*(png_ptr->error_fn))(png_ptr, '\0');

   /* If the custom handler doesn't exist, or if it returns,
      use the default handler, which will not return. */
   png_default_error(png_ptr, '\0');
}
#endif /* PNG_NO_ERROR_TEXT */

#ifndef PNG_NO_WARNINGS
/* This function is called whenever there is a non-fatal error.  This function
 * should not be changed.  If there is a need to handle warnings differently,
 * you should supply a replacement warning function and use
 * png_set_error_fn() to replace the warning function at run-time.
 */
void PNGAPI
png_warning(png_structp png_ptr, png_const_charp warning_message)
{
   int offset = 0;
   if (png_ptr != NULL)
   {
#ifdef PNG_ERROR_NUMBERS_SUPPORTED
   if (png_ptr->flags&
     (PNG_FLAG_STRIP_ERROR_NUMBERS|PNG_FLAG_STRIP_ERROR_TEXT))
#endif
     {
       if (*warning_message == '#')
       {
           for (offset=1; offset<15; offset++)
              if (*(warning_message+offset) == ' ')
                  break;
       }
     }
     if (png_ptr != NULL && png_ptr->warning_fn != NULL)
        (*(png_ptr->warning_fn))(png_ptr, warning_message+offset);
   }
   else
      png_default_warning(png_ptr, warning_message+offset);
}
#endif /* PNG_NO_WARNINGS */


/* These utilities are used internally to build an error message that relates
 * to the current chunk.  The chunk name comes from png_ptr->chunk_name,
 * this is used to prefix the message.  The message is limited in length
 * to 63 bytes, the name characters are output as hex digits wrapped in []
 * if the character is invalid.
 */
#define isnonalpha(c) ((c) < 65 || (c) > 122 || ((c) > 90 && (c) < 97))
static PNG_CONST char png_digit[16] = {
   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
   'A', 'B', 'C', 'D', 'E', 'F'
};

#define PNG_MAX_ERROR_TEXT 64

#if !defined(PNG_NO_WARNINGS) || !defined(PNG_NO_ERROR_TEXT)
static void /* PRIVATE */
png_format_buffer(png_structp png_ptr, png_charp buffer, png_const_charp
   error_message)
{
   int iout = 0, iin = 0;

   while (iin < 4)
   {
      int c = png_ptr->chunk_name[iin++];
      if (isnonalpha(c))
      {
         buffer[iout++] = '[';
         buffer[iout++] = png_digit[(c & 0xf0) >> 4];
         buffer[iout++] = png_digit[c & 0x0f];
         buffer[iout++] = ']';
      }
      else
      {
         buffer[iout++] = (png_byte)c;
      }
   }

   if (error_message == NULL)
      buffer[iout] = '\0';
   else
   {
      buffer[iout++] = ':';
      buffer[iout++] = ' ';
      png_memcpy(buffer+iout, error_message, PNG_MAX_ERROR_TEXT);
      buffer[iout+PNG_MAX_ERROR_TEXT-1] = '\0';
   }
}

#ifdef PNG_READ_SUPPORTED
void PNGAPI
png_chunk_error(png_structp png_ptr, png_const_charp error_message)
{
   char msg[18+PNG_MAX_ERROR_TEXT];
   if (png_ptr == NULL)
     png_error(png_ptr, error_message);
   else
   {
     png_format_buffer(png_ptr, msg, error_message);
     png_error(png_ptr, msg);
   }
}
#endif /* PNG_READ_SUPPORTED */
#endif /* !defined(PNG_NO_WARNINGS) || !defined(PNG_NO_ERROR_TEXT) */

#ifndef PNG_NO_WARNINGS
void PNGAPI
png_chunk_warning(png_structp png_ptr, png_const_charp warning_message)
{
   char msg[18+PNG_MAX_ERROR_TEXT];
   if (png_ptr == NULL)
     png_warning(png_ptr, warning_message);
   else
   {
     png_format_buffer(png_ptr, msg, warning_message);
     png_warning(png_ptr, msg);
   }
}
#endif /* PNG_NO_WARNINGS */


/* This is the default error handling function.  Note that replacements for
 * this function MUST NOT RETURN, or the program will likely crash.  This
 * function is used by default, or if the program supplies NULL for the
 * error function pointer in png_set_error_fn().
 */
static void /* PRIVATE */
png_default_error(png_structp png_ptr, png_const_charp error_message)
{
#ifndef PNG_NO_CONSOLE_IO
#ifdef PNG_ERROR_NUMBERS_SUPPORTED
   if (*error_message == '#')
   {
     int offset;
     char error_number[16];
     for (offset=0; offset<15; offset++)
     {
         error_number[offset] = *(error_message+offset+1);
         if (*(error_message+offset) == ' ')
             break;
     }
     if((offset > 1) && (offset < 15))
     {
       error_number[offset-1]='\0';
       fprintf(stderr, "libpng error no. %s: %s\n", error_number,
          error_message+offset);
     }
     else
       fprintf(stderr, "libpng error: %s, offset=%d\n", error_message,offset);
   }
   else
#endif
   fprintf(stderr, "libpng error: %s\n", error_message);
#endif

#ifdef PNG_SETJMP_SUPPORTED
   if (png_ptr)
   {
#  ifdef USE_FAR_KEYWORD
   {
      jmp_buf jmpbuf;
      png_memcpy(jmpbuf, png_ptr->jmpbuf, png_sizeof(jmp_buf));
      longjmp(jmpbuf, 1);
   }
#  else
   longjmp(png_ptr->jmpbuf, 1);
#  endif
   }
#else
   PNG_ABORT();
#endif
#ifdef PNG_NO_CONSOLE_IO
   error_message = error_message; /* make compiler happy */
#endif
}

#ifndef PNG_NO_WARNINGS
/* This function is called when there is a warning, but the library thinks
 * it can continue anyway.  Replacement functions don't have to do anything
 * here if you don't want them to.  In the default configuration, png_ptr is
 * not used, but it is passed in case it may be useful.
 */
static void /* PRIVATE */
png_default_warning(png_structp png_ptr, png_const_charp warning_message)
{
#ifndef PNG_NO_CONSOLE_IO
#  ifdef PNG_ERROR_NUMBERS_SUPPORTED
   if (*warning_message == '#')
   {
     int offset;
     char warning_number[16];
     for (offset=0; offset<15; offset++)
     {
        warning_number[offset]=*(warning_message+offset+1);
        if (*(warning_message+offset) == ' ')
            break;
     }
     if((offset > 1) && (offset < 15))
     {
       warning_number[offset-1]='\0';
       fprintf(stderr, "libpng warning no. %s: %s\n", warning_number,
          warning_message+offset);
     }
     else
       fprintf(stderr, "libpng warning: %s\n", warning_message);
   }
   else
#  endif
     fprintf(stderr, "libpng warning: %s\n", warning_message);
#else
   warning_message = warning_message; /* make compiler happy */
#endif
   png_ptr = png_ptr; /* make compiler happy */
}
#endif /* PNG_NO_WARNINGS */

/* This function is called when the application wants to use another method
 * of handling errors and warnings.  Note that the error function MUST NOT
 * return to the calling routine or serious problems will occur.  The return
 * method used in the default routine calls longjmp(png_ptr->jmpbuf, 1)
 */
void PNGAPI
png_set_error_fn(png_structp png_ptr, png_voidp error_ptr,
   png_error_ptr error_fn, png_error_ptr warning_fn)
{
   if (png_ptr == NULL)
      return;
   png_ptr->error_ptr = error_ptr;
   png_ptr->error_fn = error_fn;
   png_ptr->warning_fn = warning_fn;
}


/* This function returns a pointer to the error_ptr associated with the user
 * functions.  The application should free any memory associated with this
 * pointer before png_write_destroy and png_read_destroy are called.
 */
png_voidp PNGAPI
png_get_error_ptr(png_structp png_ptr)
{
   if (png_ptr == NULL)
      return NULL;
   return ((png_voidp)png_ptr->error_ptr);
}


#ifdef PNG_ERROR_NUMBERS_SUPPORTED
void PNGAPI
png_set_strip_error_numbers(png_structp png_ptr, png_uint_32 strip_mode)
{
   if(png_ptr != NULL)
   {
     png_ptr->flags &=
       ((~(PNG_FLAG_STRIP_ERROR_NUMBERS|PNG_FLAG_STRIP_ERROR_TEXT))&strip_mode);
   }
}
#endif
#endif /* PNG_READ_SUPPORTED || PNG_WRITE_SUPPORTED */
