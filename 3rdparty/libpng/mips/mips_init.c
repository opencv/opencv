/* mips_init.c - MSA optimised filter functions
 *
 * Copyright (c) 2018-2024 Cosmin Truta
 * Copyright (c) 2016 Glenn Randers-Pehrson
 * Written by Mandar Sahastrabuddhe, 2016
 * Updated by guxiwei, 2023
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

/* Below, after checking __linux__, various non-C90 POSIX 1003.1 functions are
 * called.
 */
#define _POSIX_SOURCE 1

#include <stdio.h>
#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED

#if PNG_MIPS_MSA_IMPLEMENTATION == 1 || PNG_MIPS_MMI_IMPLEMENTATION > 0

#ifdef PNG_MIPS_MSA_CHECK_SUPPORTED /* Do MIPS MSA run-time checks */
/* WARNING: it is strongly recommended that you do not build libpng with
 * run-time checks for CPU features if at all possible.  In the case of the MIPS
 * MSA instructions there is no processor-specific way of detecting the
 * presence of the required support, therefore run-time detection is extremely
 * OS specific.
 *
 * You may set the macro PNG_MIPS_MSA_FILE to the file name of file containing
 * a fragment of C source code which defines the png_have_msa function.  There
 * are a number of implementations in contrib/mips-msa, but the only one that
 * has partial support is contrib/mips-msa/linux.c - a generic Linux
 * implementation which reads /proc/cpufino.
 */
#ifndef PNG_MIPS_MSA_FILE
#  ifdef __linux__
#     define PNG_MIPS_MSA_FILE "contrib/mips-msa/linux.c"
#  endif
#endif

#ifdef PNG_MIPS_MSA_FILE

#include <signal.h> /* for sig_atomic_t */
static int png_have_msa(png_structp png_ptr);
#include PNG_MIPS_MSA_FILE

#else  /* PNG_MIPS_MSA_FILE */
#  error PNG_MIPS_MSA_FILE undefined: no support for run-time MIPS MSA checks
#endif /* PNG_MIPS_MSA_FILE */
#endif /* PNG_MIPS_MSA_CHECK_SUPPORTED */

#ifdef PNG_MIPS_MMI_CHECK_SUPPORTED /* Do MIPS MMI run-times checks */
#ifndef PNG_MIPS_MMI_FILE
#  ifdef __linux__
#     define PNG_MIPS_MMI_FILE "contrib/mips-mmi/linux.c"
#  endif
#endif

#ifdef PNG_MIPS_MMI_FILE

#include <signal.h> /* for sig_atomic_t */
static int png_have_mmi();
#include PNG_MIPS_MMI_FILE

#else  /* PNG_MIPS_MMI_FILE */
#  error PNG_MIPS_MMI_FILE undefined: no support for run-time MIPS MMI checks
#endif /* PNG_MIPS_MMI_FILE */
#endif /* PNG_MIPS_MMI_CHECK_SUPPORTED*/

#ifndef PNG_ALIGNED_MEMORY_SUPPORTED
#  error ALIGNED_MEMORY is required; please define PNG_ALIGNED_MEMORY_SUPPORTED
#endif

/* MIPS supports two optimizations: MMI and MSA. The appropriate
 * optimization is chosen at runtime
 */
void
png_init_filter_functions_mips(png_structp pp, unsigned int bpp)
{
#if PNG_MIPS_MMI_IMPLEMENTATION  > 0
#ifdef PNG_MIPS_MMI_API_SUPPORTED
   switch ((pp->options >> PNG_MIPS_MMI) & 3)
   {
      case PNG_OPTION_UNSET:
#endif /* PNG_MIPS_MMI_API_SUPPORTED */
#ifdef PNG_MIPS_MMI_CHECK_SUPPORTED
         {
            static volatile sig_atomic_t no_mmi = -1; /* not checked */

            if (no_mmi < 0)
               no_mmi = !png_have_mmi();

            if (no_mmi)
              goto MIPS_MSA_INIT;
         }
#ifdef PNG_MIPS_MMI_API_SUPPORTED
         break;
#endif
#endif /* PNG_MIPS_MMI_CHECK_SUPPORTED */

#ifdef PNG_MIPS_MMI_API_SUPPORTED
      default: /* OFF or INVALID */
         goto MIPS_MSA_INIT;

      case PNG_OPTION_ON:
         /* Option turned on */
         break;
   }
#endif
   pp->read_filter[PNG_FILTER_VALUE_UP-1] = png_read_filter_row_up_mmi;
   if (bpp == 3)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub3_mmi;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg3_mmi;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
         png_read_filter_row_paeth3_mmi;
   }
   else if (bpp == 4)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub4_mmi;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg4_mmi;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
          png_read_filter_row_paeth4_mmi;
   }
#endif /* PNG_MIPS_MMI_IMPLEMENTATION > 0 */

MIPS_MSA_INIT:
#if PNG_MIPS_MSA_IMPLEMENTATION == 1
   /* The switch statement is compiled in for MIPS_MSA_API, the call to
    * png_have_msa is compiled in for MIPS_MSA_CHECK. If both are defined
    * the check is only performed if the API has not set the MSA option on
    * or off explicitly. In this case the check controls what happens.
    */

#ifdef PNG_MIPS_MSA_API_SUPPORTED
   switch ((pp->options >> PNG_MIPS_MSA) & 3)
   {
      case PNG_OPTION_UNSET:
         /* Allow the run-time check to execute if it has been enabled -
          * thus both API and CHECK can be turned on.  If it isn't supported
          * this case will fall through to the 'default' below, which just
          * returns.
          */
#endif /* PNG_MIPS_MSA_API_SUPPORTED */
#ifdef PNG_MIPS_MSA_CHECK_SUPPORTED
         {
            static volatile sig_atomic_t no_msa = -1; /* not checked */

            if (no_msa < 0)
               no_msa = !png_have_msa(pp);

            if (no_msa)
               return;
         }
#ifdef PNG_MIPS_MSA_API_SUPPORTED
         break;
#endif
#endif /* PNG_MIPS_MSA_CHECK_SUPPORTED */

#ifdef PNG_MIPS_MSA_API_SUPPORTED
      default: /* OFF or INVALID */
         return;

      case PNG_OPTION_ON:
         /* Option turned on */
         break;
   }
#endif

   /* IMPORTANT: any new external functions used here must be declared using
    * PNG_INTERNAL_FUNCTION in ../pngpriv.h.  This is required so that the
    * 'prefix' option to configure works:
    *
    *    ./configure --with-libpng-prefix=foobar_
    *
    * Verify you have got this right by running the above command, doing a build
    * and examining pngprefix.h; it must contain a #define for every external
    * function you add.  (Notice that this happens automatically for the
    * initialization function.)
    */
   pp->read_filter[PNG_FILTER_VALUE_UP-1] = png_read_filter_row_up_msa;

   if (bpp == 3)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub3_msa;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg3_msa;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] = png_read_filter_row_paeth3_msa;
   }

   else if (bpp == 4)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub4_msa;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg4_msa;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] = png_read_filter_row_paeth4_msa;
   }
#endif /* PNG_MIPS_MSA_IMPLEMENTATION == 1 */
   return;
}
#endif /* PNG_MIPS_MSA_IMPLEMENTATION == 1 || PNG_MIPS_MMI_IMPLEMENTATION > 0 */
#endif /* READ */
