/*
 * jmemdos.c
 *
 * Copyright (C) 1992-1997, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file provides an MS-DOS-compatible implementation of the system-
 * dependent portion of the JPEG memory manager.  Temporary data can be
 * stored in extended or expanded memory as well as in regular DOS files.
 *
 * If you use this file, you must be sure that NEED_FAR_POINTERS is defined
 * if you compile in a small-data memory model; it should NOT be defined if
 * you use a large-data memory model.  This file is not recommended if you
 * are using a flat-memory-space 386 environment such as DJGCC or Watcom C.
 * Also, this code will NOT work if struct fields are aligned on greater than
 * 2-byte boundaries.
 *
 * Based on code contributed by Ge' Weijers.
 */

/*
 * If you have both extended and expanded memory, you may want to change the
 * order in which they are tried in jopen_backing_store.  On a 286 machine
 * expanded memory is usually faster, since extended memory access involves
 * an expensive protected-mode-and-back switch.  On 386 and better, extended
 * memory is usually faster.  As distributed, the code tries extended memory
 * first (what? not everyone has a 386? :-).
 *
 * You can disable use of extended/expanded memory entirely by altering these
 * definitions or overriding them from the Makefile (eg, -DEMS_SUPPORTED=0).
 */

#ifndef XMS_SUPPORTED
#define XMS_SUPPORTED  1
#endif
#ifndef EMS_SUPPORTED
#define EMS_SUPPORTED  1
#endif


#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jmemsys.h"		/* import the system-dependent declarations */

#ifndef HAVE_STDLIB_H		/* <stdlib.h> should declare these */
extern void * malloc JPP((size_t size));
extern void free JPP((void *ptr));
extern char * getenv JPP((const char * name));
#endif

#ifdef NEED_FAR_POINTERS

#ifdef __TURBOC__
/* These definitions work for Borland C (Turbo C) */
#include <alloc.h>		/* need farmalloc(), farfree() */
#define far_malloc(x)	farmalloc(x)
#define far_free(x)	farfree(x)
#else
/* These definitions work for Microsoft C and compatible compilers */
#include <malloc.h>		/* need _fmalloc(), _ffree() */
#define far_malloc(x)	_fmalloc(x)
#define far_free(x)	_ffree(x)
#endif

#else /* not NEED_FAR_POINTERS */

#define far_malloc(x)	malloc(x)
#define far_free(x)	free(x)

#endif /* NEED_FAR_POINTERS */

#ifdef DONT_USE_B_MODE		/* define mode parameters for fopen() */
#define READ_BINARY	"r"
#else
#define READ_BINARY	"rb"
#endif

#ifndef USE_MSDOS_MEMMGR	/* make sure user got configuration right */
  You forgot to define USE_MSDOS_MEMMGR in jconfig.h. /* deliberate syntax error */
#endif

#if MAX_ALLOC_CHUNK >= 65535L	/* make sure jconfig.h got this right */
  MAX_ALLOC_CHUNK should be less than 64K. /* deliberate syntax error */
#endif


/*
 * Declarations for assembly-language support routines (see jmemdosa.asm).
 *
 * The functions are declared "far" as are all their pointer arguments;
 * this ensures the assembly source code will work regardless of the
 * compiler memory model.  We assume "short" is 16 bits, "long" is 32.
 */

typedef void far * XMSDRIVER;	/* actually a pointer to code */
typedef struct {		/* registers for calling XMS driver */
	unsigned short ax, dx, bx;
	void far * ds_si;
      } XMScontext;
typedef struct {		/* registers for calling EMS driver */
	unsigned short ax, dx, bx;
	void far * ds_si;
      } EMScontext;

extern short far jdos_open JPP((short far * handle, char far * filename));
extern short far jdos_close JPP((short handle));
extern short far jdos_seek JPP((short handle, long offset));
extern short far jdos_read JPP((short handle, void far * buffer,
				unsigned short count));
extern short far jdos_write JPP((short handle, void far * buffer,
				 unsigned short count));
extern void far jxms_getdriver JPP((XMSDRIVER far *));
extern void far jxms_calldriver JPP((XMSDRIVER, XMScontext far *));
extern short far jems_available JPP((void));
extern void far jems_calldriver JPP((EMScontext far *));


/*
 * Selection of a file name for a temporary file.
 * This is highly system-dependent, and you may want to customize it.
 */

static int next_file_num;	/* to distinguish among several temp files */

LOCAL(void)
select_file_name (char * fname)
{
  const char * env;
  char * ptr;
  FILE * tfile;

  /* Keep generating file names till we find one that's not in use */
  for (;;) {
    /* Get temp directory name from environment TMP or TEMP variable;
     * if none, use "."
     */
    if ((env = (const char *) getenv("TMP")) == NULL)
      if ((env = (const char *) getenv("TEMP")) == NULL)
	env = ".";
    if (*env == '\0')		/* null string means "." */
      env = ".";
    ptr = fname;		/* copy name to fname */
    while (*env != '\0')
      *ptr++ = *env++;
    if (ptr[-1] != '\\' && ptr[-1] != '/')
      *ptr++ = '\\';		/* append backslash if not in env variable */
    /* Append a suitable file name */
    next_file_num++;		/* advance counter */
    sprintf(ptr, "JPG%03d.TMP", next_file_num);
    /* Probe to see if file name is already in use */
    if ((tfile = fopen(fname, READ_BINARY)) == NULL)
      break;
    fclose(tfile);		/* oops, it's there; close tfile & try again */
  }
}


/*
 * Near-memory allocation and freeing are controlled by the regular library
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
 * "Large" objects are allocated in far memory, if possible
 */

GLOBAL(void FAR *)
jpeg_get_large (j_common_ptr cinfo, size_t sizeofobject)
{
  return (void FAR *) far_malloc(sizeofobject);
}

GLOBAL(void)
jpeg_free_large (j_common_ptr cinfo, void FAR * object, size_t sizeofobject)
{
  far_free(object);
}


/*
 * This routine computes the total memory space available for allocation.
 * It's impossible to do this in a portable way; our current solution is
 * to make the user tell us (with a default value set at compile time).
 * If you can actually get the available space, it's a good idea to subtract
 * a slop factor of 5% or so.
 */

#ifndef DEFAULT_MAX_MEM		/* so can override from makefile */
#define DEFAULT_MAX_MEM		300000L /* for total usage about 450K */
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

/*
 * For MS-DOS we support three types of backing storage:
 *   1. Conventional DOS files.  We access these by direct DOS calls rather
 *      than via the stdio package.  This provides a bit better performance,
 *      but the real reason is that the buffers to be read or written are FAR.
 *      The stdio library for small-data memory models can't cope with that.
 *   2. Extended memory, accessed per the XMS V2.0 specification.
 *   3. Expanded memory, accessed per the LIM/EMS 4.0 specification.
 * You'll need copies of those specs to make sense of the related code.
 * The specs are available by Internet FTP from the SIMTEL archives 
 * (oak.oakland.edu and its various mirror sites).  See files
 * pub/msdos/microsoft/xms20.arc and pub/msdos/info/limems41.zip.
 */


/*
 * Access methods for a DOS file.
 */


METHODDEF(void)
read_file_store (j_common_ptr cinfo, backing_store_ptr info,
		 void FAR * buffer_address,
		 long file_offset, long byte_count)
{
  if (jdos_seek(info->handle.file_handle, file_offset))
    ERREXIT(cinfo, JERR_TFILE_SEEK);
  /* Since MAX_ALLOC_CHUNK is less than 64K, byte_count will be too. */
  if (byte_count > 65535L)	/* safety check */
    ERREXIT(cinfo, JERR_BAD_ALLOC_CHUNK);
  if (jdos_read(info->handle.file_handle, buffer_address,
		(unsigned short) byte_count))
    ERREXIT(cinfo, JERR_TFILE_READ);
}


METHODDEF(void)
write_file_store (j_common_ptr cinfo, backing_store_ptr info,
		  void FAR * buffer_address,
		  long file_offset, long byte_count)
{
  if (jdos_seek(info->handle.file_handle, file_offset))
    ERREXIT(cinfo, JERR_TFILE_SEEK);
  /* Since MAX_ALLOC_CHUNK is less than 64K, byte_count will be too. */
  if (byte_count > 65535L)	/* safety check */
    ERREXIT(cinfo, JERR_BAD_ALLOC_CHUNK);
  if (jdos_write(info->handle.file_handle, buffer_address,
		 (unsigned short) byte_count))
    ERREXIT(cinfo, JERR_TFILE_WRITE);
}


METHODDEF(void)
close_file_store (j_common_ptr cinfo, backing_store_ptr info)
{
  jdos_close(info->handle.file_handle);	/* close the file */
  remove(info->temp_name);	/* delete the file */
/* If your system doesn't have remove(), try unlink() instead.
 * remove() is the ANSI-standard name for this function, but
 * unlink() was more common in pre-ANSI systems.
 */
  TRACEMSS(cinfo, 1, JTRC_TFILE_CLOSE, info->temp_name);
}


LOCAL(boolean)
open_file_store (j_common_ptr cinfo, backing_store_ptr info,
		 long total_bytes_needed)
{
  short handle;

  select_file_name(info->temp_name);
  if (jdos_open((short far *) & handle, (char far *) info->temp_name)) {
    /* might as well exit since jpeg_open_backing_store will fail anyway */
    ERREXITS(cinfo, JERR_TFILE_CREATE, info->temp_name);
    return FALSE;
  }
  info->handle.file_handle = handle;
  info->read_backing_store = read_file_store;
  info->write_backing_store = write_file_store;
  info->close_backing_store = close_file_store;
  TRACEMSS(cinfo, 1, JTRC_TFILE_OPEN, info->temp_name);
  return TRUE;			/* succeeded */
}


/*
 * Access methods for extended memory.
 */

#if XMS_SUPPORTED

static XMSDRIVER xms_driver;	/* saved address of XMS driver */

typedef union {			/* either long offset or real-mode pointer */
	long offset;
	void far * ptr;
      } XMSPTR;

typedef struct {		/* XMS move specification structure */
	long length;
	XMSH src_handle;
	XMSPTR src;
	XMSH dst_handle;
	XMSPTR dst;
      } XMSspec;

#define ODD(X)	(((X) & 1L) != 0)


METHODDEF(void)
read_xms_store (j_common_ptr cinfo, backing_store_ptr info,
		void FAR * buffer_address,
		long file_offset, long byte_count)
{
  XMScontext ctx;
  XMSspec spec;
  char endbuffer[2];

  /* The XMS driver can't cope with an odd length, so handle the last byte
   * specially if byte_count is odd.  We don't expect this to be common.
   */

  spec.length = byte_count & (~ 1L);
  spec.src_handle = info->handle.xms_handle;
  spec.src.offset = file_offset;
  spec.dst_handle = 0;
  spec.dst.ptr = buffer_address;
  
  ctx.ds_si = (void far *) & spec;
  ctx.ax = 0x0b00;		/* EMB move */
  jxms_calldriver(xms_driver, (XMScontext far *) & ctx);
  if (ctx.ax != 1)
    ERREXIT(cinfo, JERR_XMS_READ);

  if (ODD(byte_count)) {
    read_xms_store(cinfo, info, (void FAR *) endbuffer,
		   file_offset + byte_count - 1L, 2L);
    ((char FAR *) buffer_address)[byte_count - 1L] = endbuffer[0];
  }
}


METHODDEF(void)
write_xms_store (j_common_ptr cinfo, backing_store_ptr info,
		 void FAR * buffer_address,
		 long file_offset, long byte_count)
{
  XMScontext ctx;
  XMSspec spec;
  char endbuffer[2];

  /* The XMS driver can't cope with an odd length, so handle the last byte
   * specially if byte_count is odd.  We don't expect this to be common.
   */

  spec.length = byte_count & (~ 1L);
  spec.src_handle = 0;
  spec.src.ptr = buffer_address;
  spec.dst_handle = info->handle.xms_handle;
  spec.dst.offset = file_offset;

  ctx.ds_si = (void far *) & spec;
  ctx.ax = 0x0b00;		/* EMB move */
  jxms_calldriver(xms_driver, (XMScontext far *) & ctx);
  if (ctx.ax != 1)
    ERREXIT(cinfo, JERR_XMS_WRITE);

  if (ODD(byte_count)) {
    read_xms_store(cinfo, info, (void FAR *) endbuffer,
		   file_offset + byte_count - 1L, 2L);
    endbuffer[0] = ((char FAR *) buffer_address)[byte_count - 1L];
    write_xms_store(cinfo, info, (void FAR *) endbuffer,
		    file_offset + byte_count - 1L, 2L);
  }
}


METHODDEF(void)
close_xms_store (j_common_ptr cinfo, backing_store_ptr info)
{
  XMScontext ctx;

  ctx.dx = info->handle.xms_handle;
  ctx.ax = 0x0a00;
  jxms_calldriver(xms_driver, (XMScontext far *) & ctx);
  TRACEMS1(cinfo, 1, JTRC_XMS_CLOSE, info->handle.xms_handle);
  /* we ignore any error return from the driver */
}


LOCAL(boolean)
open_xms_store (j_common_ptr cinfo, backing_store_ptr info,
		long total_bytes_needed)
{
  XMScontext ctx;

  /* Get address of XMS driver */
  jxms_getdriver((XMSDRIVER far *) & xms_driver);
  if (xms_driver == NULL)
    return FALSE;		/* no driver to be had */

  /* Get version number, must be >= 2.00 */
  ctx.ax = 0x0000;
  jxms_calldriver(xms_driver, (XMScontext far *) & ctx);
  if (ctx.ax < (unsigned short) 0x0200)
    return FALSE;

  /* Try to get space (expressed in kilobytes) */
  ctx.dx = (unsigned short) ((total_bytes_needed + 1023L) >> 10);
  ctx.ax = 0x0900;
  jxms_calldriver(xms_driver, (XMScontext far *) & ctx);
  if (ctx.ax != 1)
    return FALSE;

  /* Succeeded, save the handle and away we go */
  info->handle.xms_handle = ctx.dx;
  info->read_backing_store = read_xms_store;
  info->write_backing_store = write_xms_store;
  info->close_backing_store = close_xms_store;
  TRACEMS1(cinfo, 1, JTRC_XMS_OPEN, ctx.dx);
  return TRUE;			/* succeeded */
}

#endif /* XMS_SUPPORTED */


/*
 * Access methods for expanded memory.
 */

#if EMS_SUPPORTED

/* The EMS move specification structure requires word and long fields aligned
 * at odd byte boundaries.  Some compilers will align struct fields at even
 * byte boundaries.  While it's usually possible to force byte alignment,
 * that causes an overall performance penalty and may pose problems in merging
 * JPEG into a larger application.  Instead we accept some rather dirty code
 * here.  Note this code would fail if the hardware did not allow odd-byte
 * word & long accesses, but all 80x86 CPUs do.
 */

typedef void far * EMSPTR;

typedef union {			/* EMS move specification structure */
	long length;		/* It's easy to access first 4 bytes */
	char bytes[18];		/* Misaligned fields in here! */
      } EMSspec;

/* Macros for accessing misaligned fields */
#define FIELD_AT(spec,offset,type)  (*((type *) &(spec.bytes[offset])))
#define SRC_TYPE(spec)		FIELD_AT(spec,4,char)
#define SRC_HANDLE(spec)	FIELD_AT(spec,5,EMSH)
#define SRC_OFFSET(spec)	FIELD_AT(spec,7,unsigned short)
#define SRC_PAGE(spec)		FIELD_AT(spec,9,unsigned short)
#define SRC_PTR(spec)		FIELD_AT(spec,7,EMSPTR)
#define DST_TYPE(spec)		FIELD_AT(spec,11,char)
#define DST_HANDLE(spec)	FIELD_AT(spec,12,EMSH)
#define DST_OFFSET(spec)	FIELD_AT(spec,14,unsigned short)
#define DST_PAGE(spec)		FIELD_AT(spec,16,unsigned short)
#define DST_PTR(spec)		FIELD_AT(spec,14,EMSPTR)

#define EMSPAGESIZE	16384L	/* gospel, see the EMS specs */

#define HIBYTE(W)  (((W) >> 8) & 0xFF)
#define LOBYTE(W)  ((W) & 0xFF)


METHODDEF(void)
read_ems_store (j_common_ptr cinfo, backing_store_ptr info,
		void FAR * buffer_address,
		long file_offset, long byte_count)
{
  EMScontext ctx;
  EMSspec spec;

  spec.length = byte_count;
  SRC_TYPE(spec) = 1;
  SRC_HANDLE(spec) = info->handle.ems_handle;
  SRC_PAGE(spec)   = (unsigned short) (file_offset / EMSPAGESIZE);
  SRC_OFFSET(spec) = (unsigned short) (file_offset % EMSPAGESIZE);
  DST_TYPE(spec) = 0;
  DST_HANDLE(spec) = 0;
  DST_PTR(spec)    = buffer_address;
  
  ctx.ds_si = (void far *) & spec;
  ctx.ax = 0x5700;		/* move memory region */
  jems_calldriver((EMScontext far *) & ctx);
  if (HIBYTE(ctx.ax) != 0)
    ERREXIT(cinfo, JERR_EMS_READ);
}


METHODDEF(void)
write_ems_store (j_common_ptr cinfo, backing_store_ptr info,
		 void FAR * buffer_address,
		 long file_offset, long byte_count)
{
  EMScontext ctx;
  EMSspec spec;

  spec.length = byte_count;
  SRC_TYPE(spec) = 0;
  SRC_HANDLE(spec) = 0;
  SRC_PTR(spec)    = buffer_address;
  DST_TYPE(spec) = 1;
  DST_HANDLE(spec) = info->handle.ems_handle;
  DST_PAGE(spec)   = (unsigned short) (file_offset / EMSPAGESIZE);
  DST_OFFSET(spec) = (unsigned short) (file_offset % EMSPAGESIZE);
  
  ctx.ds_si = (void far *) & spec;
  ctx.ax = 0x5700;		/* move memory region */
  jems_calldriver((EMScontext far *) & ctx);
  if (HIBYTE(ctx.ax) != 0)
    ERREXIT(cinfo, JERR_EMS_WRITE);
}


METHODDEF(void)
close_ems_store (j_common_ptr cinfo, backing_store_ptr info)
{
  EMScontext ctx;

  ctx.ax = 0x4500;
  ctx.dx = info->handle.ems_handle;
  jems_calldriver((EMScontext far *) & ctx);
  TRACEMS1(cinfo, 1, JTRC_EMS_CLOSE, info->handle.ems_handle);
  /* we ignore any error return from the driver */
}


LOCAL(boolean)
open_ems_store (j_common_ptr cinfo, backing_store_ptr info,
		long total_bytes_needed)
{
  EMScontext ctx;

  /* Is EMS driver there? */
  if (! jems_available())
    return FALSE;

  /* Get status, make sure EMS is OK */
  ctx.ax = 0x4000;
  jems_calldriver((EMScontext far *) & ctx);
  if (HIBYTE(ctx.ax) != 0)
    return FALSE;

  /* Get version, must be >= 4.0 */
  ctx.ax = 0x4600;
  jems_calldriver((EMScontext far *) & ctx);
  if (HIBYTE(ctx.ax) != 0 || LOBYTE(ctx.ax) < 0x40)
    return FALSE;

  /* Try to allocate requested space */
  ctx.ax = 0x4300;
  ctx.bx = (unsigned short) ((total_bytes_needed + EMSPAGESIZE-1L) / EMSPAGESIZE);
  jems_calldriver((EMScontext far *) & ctx);
  if (HIBYTE(ctx.ax) != 0)
    return FALSE;

  /* Succeeded, save the handle and away we go */
  info->handle.ems_handle = ctx.dx;
  info->read_backing_store = read_ems_store;
  info->write_backing_store = write_ems_store;
  info->close_backing_store = close_ems_store;
  TRACEMS1(cinfo, 1, JTRC_EMS_OPEN, ctx.dx);
  return TRUE;			/* succeeded */
}

#endif /* EMS_SUPPORTED */


/*
 * Initial opening of a backing-store object.
 */

GLOBAL(void)
jpeg_open_backing_store (j_common_ptr cinfo, backing_store_ptr info,
			 long total_bytes_needed)
{
  /* Try extended memory, then expanded memory, then regular file. */
#if XMS_SUPPORTED
  if (open_xms_store(cinfo, info, total_bytes_needed))
    return;
#endif
#if EMS_SUPPORTED
  if (open_ems_store(cinfo, info, total_bytes_needed))
    return;
#endif
  if (open_file_store(cinfo, info, total_bytes_needed))
    return;
  ERREXITS(cinfo, JERR_TFILE_CREATE, "");
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
  /* Microsoft C, at least in v6.00A, will not successfully reclaim freed
   * blocks of size > 32Kbytes unless we give it a kick in the rear, like so:
   */
#ifdef NEED_FHEAPMIN
  _fheapmin();
#endif
}
