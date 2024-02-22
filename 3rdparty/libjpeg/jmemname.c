/*
 * jmemname.c
 *
 * Copyright (C) 1992-1997, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file provides a generic implementation of the system-dependent
 * portion of the JPEG memory manager.  This implementation assumes that
 * you must explicitly construct a name for each temp file.
 * Also, the problem of determining the amount of memory available
 * is shoved onto the user.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jmemsys.h"		/* import the system-dependent declarations */

#ifndef HAVE_STDLIB_H		/* <stdlib.h> should declare malloc(),free() */
extern void * malloc JPP((size_t size));
extern void free JPP((void *ptr));
#endif

#ifndef SEEK_SET		/* pre-ANSI systems may not define this; */
#define SEEK_SET  0		/* if not, assume 0 is correct */
#endif

#ifdef DONT_USE_B_MODE		/* define mode parameters for fopen() */
#define READ_BINARY	"r"
#define RW_BINARY	"w+"
#else
#ifdef VMS			/* VMS is very nonstandard */
#define READ_BINARY	"rb", "ctx=stm"
#define RW_BINARY	"w+b", "ctx=stm"
#else				/* standard ANSI-compliant case */
#define READ_BINARY	"rb"
#define RW_BINARY	"w+b"
#endif
#endif


/*
 * Selection of a file name for a temporary file.
 * This is system-dependent!
 *
 * The code as given is suitable for most Unix systems, and it is easily
 * modified for most non-Unix systems.  Some notes:
 *  1.  The temp file is created in the directory named by TEMP_DIRECTORY.
 *      The default value is /usr/tmp, which is the conventional place for
 *      creating large temp files on Unix.  On other systems you'll probably
 *      want to change the file location.  You can do this by editing the
 *      #define, or (preferred) by defining TEMP_DIRECTORY in jconfig.h.
 *
 *  2.  If you need to change the file name as well as its location,
 *      you can override the TEMP_FILE_NAME macro.  (Note that this is
 *      actually a printf format string; it must contain %s and %d.)
 *      Few people should need to do this.
 *
 *  3.  mktemp() is used to ensure that multiple processes running
 *      simultaneously won't select the same file names.  If your system
 *      doesn't have mktemp(), define NO_MKTEMP to do it the hard way.
 *      (If you don't have <errno.h>, also define NO_ERRNO_H.)
 *
 *  4.  You probably want to define NEED_SIGNAL_CATCHER so that cjpeg.c/djpeg.c
 *      will cause the temp files to be removed if you stop the program early.
 */

#ifndef TEMP_DIRECTORY		/* can override from jconfig.h or Makefile */
#define TEMP_DIRECTORY  "/usr/tmp/" /* recommended setting for Unix */
#endif

static int next_file_num;	/* to distinguish among several temp files */

#ifdef NO_MKTEMP

#ifndef TEMP_FILE_NAME		/* can override from jconfig.h or Makefile */
#define TEMP_FILE_NAME  "%sJPG%03d.TMP"
#endif

#ifndef NO_ERRNO_H
#include <errno.h>		/* to define ENOENT */
#endif

/* ANSI C specifies that errno is a macro, but on older systems it's more
 * likely to be a plain int variable.  And not all versions of errno.h
 * bother to declare it, so we have to in order to be most portable.  Thus:
 */
#ifndef errno
extern int errno;
#endif


LOCAL(void)
select_file_name (char * fname)
{
  FILE * tfile;

  /* Keep generating file names till we find one that's not in use */
  for (;;) {
    next_file_num++;		/* advance counter */
    sprintf(fname, TEMP_FILE_NAME, TEMP_DIRECTORY, next_file_num);
    if ((tfile = fopen(fname, READ_BINARY)) == NULL) {
      /* fopen could have failed for a reason other than the file not
       * being there; for example, file there but unreadable.
       * If <errno.h> isn't available, then we cannot test the cause.
       */
#ifdef ENOENT
      if (errno != ENOENT)
	continue;
#endif
      break;
    }
    fclose(tfile);		/* oops, it's there; close tfile & try again */
  }
}

#else /* ! NO_MKTEMP */

/* Note that mktemp() requires the initial filename to end in six X's */
#ifndef TEMP_FILE_NAME		/* can override from jconfig.h or Makefile */
#define TEMP_FILE_NAME  "%sJPG%dXXXXXX"
#endif

LOCAL(void)
select_file_name (char * fname)
{
  next_file_num++;		/* advance counter */
  sprintf(fname, TEMP_FILE_NAME, TEMP_DIRECTORY, next_file_num);
  mktemp(fname);		/* make sure file name is unique */
  /* mktemp replaces the trailing XXXXXX with a unique string of characters */
}

#endif /* NO_MKTEMP */


/*
 * Memory allocation and freeing are controlled by the regular library
 * routines malloc() and free().
 */

GLOBAL(void *)
jpeg_get_small (j_common_ptr cinfo, size_t sizeofobject)
{
  return (void *) malloc(sizeofobject);
}

GLOBAL(void)
jpeg_free_small (j_common_ptr cinfo, void * object, size_t sizeofobject)
{
  free(object);
}


/*
 * "Large" objects are treated the same as "small" ones.
 * NB: although we include FAR keywords in the routine declarations,
 * this file won't actually work in 80x86 small/medium model; at least,
 * you probably won't be able to process useful-size images in only 64KB.
 */

GLOBAL(void FAR *)
jpeg_get_large (j_common_ptr cinfo, size_t sizeofobject)
{
  return (void FAR *) malloc(sizeofobject);
}

GLOBAL(void)
jpeg_free_large (j_common_ptr cinfo, void FAR * object, size_t sizeofobject)
{
  free(object);
}


/*
 * This routine computes the total memory space available for allocation.
 * It's impossible to do this in a portable way; our current solution is
 * to make the user tell us (with a default value set at compile time).
 * If you can actually get the available space, it's a good idea to subtract
 * a slop factor of 5% or so.
 */

#ifndef DEFAULT_MAX_MEM		/* so can override from makefile */
#define DEFAULT_MAX_MEM		1000000L /* default: one megabyte */
#endif

GLOBAL(long)
jpeg_mem_available (j_common_ptr cinfo, long min_bytes_needed,
		    long max_bytes_needed, long already_allocated)
{
  return cinfo->mem->max_memory_to_use - already_allocated;
}


/*
 * Backing store (temporary file) management.
 * Backing store objects are only used when the value returned by
 * jpeg_mem_available is less than the total space needed.  You can dispense
 * with these routines if you have plenty of virtual memory; see jmemnobs.c.
 */


METHODDEF(void)
read_backing_store (j_common_ptr cinfo, backing_store_ptr info,
		    void FAR * buffer_address,
		    long file_offset, long byte_count)
{
  if (fseek(info->temp_file, file_offset, SEEK_SET))
    ERREXIT(cinfo, JERR_TFILE_SEEK);
  if (JFREAD(info->temp_file, buffer_address, byte_count)
      != (size_t) byte_count)
    ERREXIT(cinfo, JERR_TFILE_READ);
}


METHODDEF(void)
write_backing_store (j_common_ptr cinfo, backing_store_ptr info,
		     void FAR * buffer_address,
		     long file_offset, long byte_count)
{
  if (fseek(info->temp_file, file_offset, SEEK_SET))
    ERREXIT(cinfo, JERR_TFILE_SEEK);
  if (JFWRITE(info->temp_file, buffer_address, byte_count)
      != (size_t) byte_count)
    ERREXIT(cinfo, JERR_TFILE_WRITE);
}


METHODDEF(void)
close_backing_store (j_common_ptr cinfo, backing_store_ptr info)
{
  fclose(info->temp_file);	/* close the file */
  unlink(info->temp_name);	/* delete the file */
/* If your system doesn't have unlink(), use remove() instead.
 * remove() is the ANSI-standard name for this function, but if
 * your system was ANSI you'd be using jmemansi.c, right?
 */
  TRACEMSS(cinfo, 1, JTRC_TFILE_CLOSE, info->temp_name);
}


/*
 * Initial opening of a backing-store object.
 */

GLOBAL(void)
jpeg_open_backing_store (j_common_ptr cinfo, backing_store_ptr info,
			 long total_bytes_needed)
{
  select_file_name(info->temp_name);
  if ((info->temp_file = fopen(info->temp_name, RW_BINARY)) == NULL)
    ERREXITS(cinfo, JERR_TFILE_CREATE, info->temp_name);
  info->read_backing_store = read_backing_store;
  info->write_backing_store = write_backing_store;
  info->close_backing_store = close_backing_store;
  TRACEMSS(cinfo, 1, JTRC_TFILE_OPEN, info->temp_name);
}


/*
 * These routines take care of any system-dependent initialization and
 * cleanup required.
 */

GLOBAL(long)
jpeg_mem_init (j_common_ptr cinfo)
{
  next_file_num = 0;		/* initialize temp file name generator */
  return DEFAULT_MAX_MEM;	/* default for max_memory_to_use */
}

GLOBAL(void)
jpeg_mem_term (j_common_ptr cinfo)
{
  /* no work */
}
