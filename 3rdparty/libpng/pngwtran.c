
/* pngwtran.c - transforms the data in a row for PNG writers
 *
 * Last changed in libpng 1.2.9 April 14, 2006
 * For conditions of distribution and use, see copyright notice in png.h
 * Copyright (c) 1998-2006 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 */

#define PNG_INTERNAL
#include "png.h"
#ifdef PNG_WRITE_SUPPORTED

/* Transform the data according to the user's wishes.  The order of
 * transformations is significant.
 */
void /* PRIVATE */
png_do_write_transformations(png_structp png_ptr)
{
   png_debug(1, "in png_do_write_transformations\n");

   if (png_ptr == NULL)
      return;

#if defined(PNG_WRITE_USER_TRANSFORM_SUPPORTED)
   if (png_ptr->transformations & PNG_USER_TRANSFORM)
      if(png_ptr->write_user_transform_fn != NULL)
        (*(png_ptr->write_user_transform_fn)) /* user write transform function */
          (png_ptr,                    /* png_ptr */
           &(png_ptr->row_info),       /* row_info:     */
             /*  png_uint_32 width;          width of row */
             /*  png_uint_32 rowbytes;       number of bytes in row */
             /*  png_byte color_type;        color type of pixels */
             /*  png_byte bit_depth;         bit depth of samples */
             /*  png_byte channels;          number of channels (1-4) */
             /*  png_byte pixel_depth;       bits per pixel (depth*channels) */
           png_ptr->row_buf + 1);      /* start of pixel data for row */
#endif
#if defined(PNG_WRITE_FILLER_SUPPORTED)
   if (png_ptr->transformations & PNG_FILLER)
      png_do_strip_filler(&(png_ptr->row_info), png_ptr->row_buf + 1,
         png_ptr->flags);
#endif
#if defined(PNG_WRITE_PACKSWAP_SUPPORTED)
   if (png_ptr->transformations & PNG_PACKSWAP)
      png_do_packswap(&(png_ptr->row_info), png_ptr->row_buf + 1);
#endif
#if defined(PNG_WRITE_PACK_SUPPORTED)
   if (png_ptr->transformations & PNG_PACK)
      png_do_pack(&(png_ptr->row_info), png_ptr->row_buf + 1,
         (png_uint_32)png_ptr->bit_depth);
#endif
#if defined(PNG_WRITE_SWAP_SUPPORTED)
   if (png_ptr->transformations & PNG_SWAP_BYTES)
      png_do_swap(&(png_ptr->row_info), png_ptr->row_buf + 1);
#endif
#if defined(PNG_WRITE_SHIFT_SUPPORTED)
   if (png_ptr->transformations & PNG_SHIFT)
      png_do_shift(&(png_ptr->row_info), png_ptr->row_buf + 1,
         &(png_ptr->shift));
#endif
#if defined(PNG_WRITE_SWAP_ALPHA_SUPPORTED)
   if (png_ptr->transformations & PNG_SWAP_ALPHA)
      png_do_write_swap_alpha(&(png_ptr->row_info), png_ptr->row_buf + 1);
#endif
#if defined(PNG_WRITE_INVERT_ALPHA_SUPPORTED)
   if (png_ptr->transformations & PNG_INVERT_ALPHA)
      png_do_write_invert_alpha(&(png_ptr->row_info), png_ptr->row_buf + 1);
#endif
#if defined(PNG_WRITE_BGR_SUPPORTED)
   if (png_ptr->transformations & PNG_BGR)
      png_do_bgr(&(png_ptr->row_info), png_ptr->row_buf + 1);
#endif
#if defined(PNG_WRITE_INVERT_SUPPORTED)
   if (png_ptr->transformations & PNG_INVERT_MONO)
      png_do_invert(&(png_ptr->row_info), png_ptr->row_buf + 1);
#endif
}

#if defined(PNG_WRITE_PACK_SUPPORTED)
/* Pack pixels into bytes.  Pass the true bit depth in bit_depth.  The
 * row_info bit depth should be 8 (one pixel per byte).  The channels
 * should be 1 (this only happens on grayscale and paletted images).
 */
void /* PRIVATE */
png_do_pack(png_row_infop row_info, png_bytep row, png_uint_32 bit_depth)
{
   png_debug(1, "in png_do_pack\n");
   if (row_info->bit_depth == 8 &&
#if defined(PNG_USELESS_TESTS_SUPPORTED)
       row != NULL && row_info != NULL &&
#endif
      row_info->channels == 1)
   {
      switch ((int)bit_depth)
      {
         case 1:
         {
            png_bytep sp, dp;
            int mask, v;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            sp = row;
            dp = row;
            mask = 0x80;
            v = 0;

            for (i = 0; i < row_width; i++)
            {
               if (*sp != 0)
                  v |= mask;
               sp++;
               if (mask > 1)
                  mask >>= 1;
               else
               {
                  mask = 0x80;
                  *dp = (png_byte)v;
                  dp++;
                  v = 0;
               }
            }
            if (mask != 0x80)
               *dp = (png_byte)v;
            break;
         }
         case 2:
         {
            png_bytep sp, dp;
            int shift, v;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            sp = row;
            dp = row;
            shift = 6;
            v = 0;
            for (i = 0; i < row_width; i++)
            {
               png_byte value;

               value = (png_byte)(*sp & 0x03);
               v |= (value << shift);
               if (shift == 0)
               {
                  shift = 6;
                  *dp = (png_byte)v;
                  dp++;
                  v = 0;
               }
               else
                  shift -= 2;
               sp++;
            }
            if (shift != 6)
               *dp = (png_byte)v;
            break;
         }
         case 4:
         {
            png_bytep sp, dp;
            int shift, v;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            sp = row;
            dp = row;
            shift = 4;
            v = 0;
            for (i = 0; i < row_width; i++)
            {
               png_byte value;

               value = (png_byte)(*sp & 0x0f);
               v |= (value << shift);

               if (shift == 0)
               {
                  shift = 4;
                  *dp = (png_byte)v;
                  dp++;
                  v = 0;
               }
               else
                  shift -= 4;

               sp++;
            }
            if (shift != 4)
               *dp = (png_byte)v;
            break;
         }
      }
      row_info->bit_depth = (png_byte)bit_depth;
      row_info->pixel_depth = (png_byte)(bit_depth * row_info->channels);
      row_info->rowbytes = PNG_ROWBYTES(row_info->pixel_depth,
         row_info->width);
   }
}
#endif

#if defined(PNG_WRITE_SHIFT_SUPPORTED)
/* Shift pixel values to take advantage of whole range.  Pass the
 * true number of bits in bit_depth.  The row should be packed
 * according to row_info->bit_depth.  Thus, if you had a row of
 * bit depth 4, but the pixels only had values from 0 to 7, you
 * would pass 3 as bit_depth, and this routine would translate the
 * data to 0 to 15.
 */
void /* PRIVATE */
png_do_shift(png_row_infop row_info, png_bytep row, png_color_8p bit_depth)
{
   png_debug(1, "in png_do_shift\n");
#if defined(PNG_USELESS_TESTS_SUPPORTED)
   if (row != NULL && row_info != NULL &&
#else
   if (
#endif
      row_info->color_type != PNG_COLOR_TYPE_PALETTE)
   {
      int shift_start[4], shift_dec[4];
      int channels = 0;

      if (row_info->color_type & PNG_COLOR_MASK_COLOR)
      {
         shift_start[channels] = row_info->bit_depth - bit_depth->red;
         shift_dec[channels] = bit_depth->red;
         channels++;
         shift_start[channels] = row_info->bit_depth - bit_depth->green;
         shift_dec[channels] = bit_depth->green;
         channels++;
         shift_start[channels] = row_info->bit_depth - bit_depth->blue;
         shift_dec[channels] = bit_depth->blue;
         channels++;
      }
      else
      {
         shift_start[channels] = row_info->bit_depth - bit_depth->gray;
         shift_dec[channels] = bit_depth->gray;
         channels++;
      }
      if (row_info->color_type & PNG_COLOR_MASK_ALPHA)
      {
         shift_start[channels] = row_info->bit_depth - bit_depth->alpha;
         shift_dec[channels] = bit_depth->alpha;
         channels++;
      }

      /* with low row depths, could only be grayscale, so one channel */
      if (row_info->bit_depth < 8)
      {
         png_bytep bp = row;
         png_uint_32 i;
         png_byte mask;
         png_uint_32 row_bytes = row_info->rowbytes;

         if (bit_depth->gray == 1 && row_info->bit_depth == 2)
            mask = 0x55;
         else if (row_info->bit_depth == 4 && bit_depth->gray == 3)
            mask = 0x11;
         else
            mask = 0xff;

         for (i = 0; i < row_bytes; i++, bp++)
         {
            png_uint_16 v;
            int j;

            v = *bp;
            *bp = 0;
            for (j = shift_start[0]; j > -shift_dec[0]; j -= shift_dec[0])
            {
               if (j > 0)
                  *bp |= (png_byte)((v << j) & 0xff);
               else
                  *bp |= (png_byte)((v >> (-j)) & mask);
            }
         }
      }
      else if (row_info->bit_depth == 8)
      {
         png_bytep bp = row;
         png_uint_32 i;
         png_uint_32 istop = channels * row_info->width;

         for (i = 0; i < istop; i++, bp++)
         {

            png_uint_16 v;
            int j;
            int c = (int)(i%channels);

            v = *bp;
            *bp = 0;
            for (j = shift_start[c]; j > -shift_dec[c]; j -= shift_dec[c])
            {
               if (j > 0)
                  *bp |= (png_byte)((v << j) & 0xff);
               else
                  *bp |= (png_byte)((v >> (-j)) & 0xff);
            }
         }
      }
      else
      {
         png_bytep bp;
         png_uint_32 i;
         png_uint_32 istop = channels * row_info->width;

         for (bp = row, i = 0; i < istop; i++)
         {
            int c = (int)(i%channels);
            png_uint_16 value, v;
            int j;

            v = (png_uint_16)(((png_uint_16)(*bp) << 8) + *(bp + 1));
            value = 0;
            for (j = shift_start[c]; j > -shift_dec[c]; j -= shift_dec[c])
            {
               if (j > 0)
                  value |= (png_uint_16)((v << j) & (png_uint_16)0xffff);
               else
                  value |= (png_uint_16)((v >> (-j)) & (png_uint_16)0xffff);
            }
            *bp++ = (png_byte)(value >> 8);
            *bp++ = (png_byte)(value & 0xff);
         }
      }
   }
}
#endif

#if defined(PNG_WRITE_SWAP_ALPHA_SUPPORTED)
void /* PRIVATE */
png_do_write_swap_alpha(png_row_infop row_info, png_bytep row)
{
   png_debug(1, "in png_do_write_swap_alpha\n");
#if defined(PNG_USELESS_TESTS_SUPPORTED)
   if (row != NULL && row_info != NULL)
#endif
   {
      if (row_info->color_type == PNG_COLOR_TYPE_RGB_ALPHA)
      {
         /* This converts from ARGB to RGBA */
         if (row_info->bit_depth == 8)
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;
            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               png_byte save = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = save;
            }
         }
         /* This converts from AARRGGBB to RRGGBBAA */
         else
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               png_byte save[2];
               save[0] = *(sp++);
               save[1] = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = save[0];
               *(dp++) = save[1];
            }
         }
      }
      else if (row_info->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
      {
         /* This converts from AG to GA */
         if (row_info->bit_depth == 8)
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               png_byte save = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = save;
            }
         }
         /* This converts from AAGG to GGAA */
         else
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               png_byte save[2];
               save[0] = *(sp++);
               save[1] = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = save[0];
               *(dp++) = save[1];
            }
         }
      }
   }
}
#endif

#if defined(PNG_WRITE_INVERT_ALPHA_SUPPORTED)
void /* PRIVATE */
png_do_write_invert_alpha(png_row_infop row_info, png_bytep row)
{
   png_debug(1, "in png_do_write_invert_alpha\n");
#if defined(PNG_USELESS_TESTS_SUPPORTED)
   if (row != NULL && row_info != NULL)
#endif
   {
      if (row_info->color_type == PNG_COLOR_TYPE_RGB_ALPHA)
      {
         /* This inverts the alpha channel in RGBA */
         if (row_info->bit_depth == 8)
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;
            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               /* does nothing
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               */
               sp+=3; dp = sp;
               *(dp++) = (png_byte)(255 - *(sp++));
            }
         }
         /* This inverts the alpha channel in RRGGBBAA */
         else
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               /* does nothing
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               */
               sp+=6; dp = sp;
               *(dp++) = (png_byte)(255 - *(sp++));
               *(dp++) = (png_byte)(255 - *(sp++));
            }
         }
      }
      else if (row_info->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
      {
         /* This inverts the alpha channel in GA */
         if (row_info->bit_depth == 8)
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               *(dp++) = *(sp++);
               *(dp++) = (png_byte)(255 - *(sp++));
            }
         }
         /* This inverts the alpha channel in GGAA */
         else
         {
            png_bytep sp, dp;
            png_uint_32 i;
            png_uint_32 row_width = row_info->width;

            for (i = 0, sp = dp = row; i < row_width; i++)
            {
               /* does nothing
               *(dp++) = *(sp++);
               *(dp++) = *(sp++);
               */
               sp+=2; dp = sp;
               *(dp++) = (png_byte)(255 - *(sp++));
               *(dp++) = (png_byte)(255 - *(sp++));
            }
         }
      }
   }
}
#endif

#if defined(PNG_MNG_FEATURES_SUPPORTED)
/* undoes intrapixel differencing  */
void /* PRIVATE */
png_do_write_intrapixel(png_row_infop row_info, png_bytep row)
{
   png_debug(1, "in png_do_write_intrapixel\n");
   if (
#if defined(PNG_USELESS_TESTS_SUPPORTED)
       row != NULL && row_info != NULL &&
#endif
       (row_info->color_type & PNG_COLOR_MASK_COLOR))
   {
      int bytes_per_pixel;
      png_uint_32 row_width = row_info->width;
      if (row_info->bit_depth == 8)
      {
         png_bytep rp;
         png_uint_32 i;

         if (row_info->color_type == PNG_COLOR_TYPE_RGB)
            bytes_per_pixel = 3;
         else if (row_info->color_type == PNG_COLOR_TYPE_RGB_ALPHA)
            bytes_per_pixel = 4;
         else
            return;

         for (i = 0, rp = row; i < row_width; i++, rp += bytes_per_pixel)
         {
            *(rp)   = (png_byte)((*rp     - *(rp+1))&0xff);
            *(rp+2) = (png_byte)((*(rp+2) - *(rp+1))&0xff);
         }
      }
      else if (row_info->bit_depth == 16)
      {
         png_bytep rp;
         png_uint_32 i;

         if (row_info->color_type == PNG_COLOR_TYPE_RGB)
            bytes_per_pixel = 6;
         else if (row_info->color_type == PNG_COLOR_TYPE_RGB_ALPHA)
            bytes_per_pixel = 8;
         else
            return;

         for (i = 0, rp = row; i < row_width; i++, rp += bytes_per_pixel)
         {
            png_uint_32 s0   = (*(rp  ) << 8) | *(rp+1);
            png_uint_32 s1   = (*(rp+2) << 8) | *(rp+3);
            png_uint_32 s2   = (*(rp+4) << 8) | *(rp+5);
            png_uint_32 red  = (png_uint_32)((s0-s1) & 0xffffL);
            png_uint_32 blue = (png_uint_32)((s2-s1) & 0xffffL);
            *(rp  ) = (png_byte)((red >> 8) & 0xff);
            *(rp+1) = (png_byte)(red & 0xff);
            *(rp+4) = (png_byte)((blue >> 8) & 0xff);
            *(rp+5) = (png_byte)(blue & 0xff);
         }
      }
   }
}
#endif /* PNG_MNG_FEATURES_SUPPORTED */
#endif /* PNG_WRITE_SUPPORTED */
