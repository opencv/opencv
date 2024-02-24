/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

/* implementation for unix-like file io routines (used in context.c) */
#include "openexr_config.h" 

#include <errno.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ILMTHREAD_THREADING_ENABLED
#    include <pthread.h>
#endif
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined __USE_UNIX98 || defined __USE_XOPEN2K8 ||                          \
    (defined(_XOPEN_VERSION) && _XOPEN_VERSION >= 400)
#    define CAN_USE_PREAD 1
#else
#    define CAN_USE_PREAD 0
#endif

#if CAN_USE_PREAD
struct _internal_exr_filehandle
{
    int fd;
};
#else
struct _internal_exr_filehandle
{
    int             fd;
#    ifdef ILMTHREAD_THREADING_ENABLED
    pthread_mutex_t mutex;
#    endif
};
#endif

/**************************************/

static void
default_shutdown (exr_const_context_t c, void* userdata, int failed)
{
    /* we will handle failure before here */
    struct _internal_exr_filehandle* fh = userdata;
    if (fh)
    {
        if (fh->fd >= 0) close (fh->fd);
#if !CAN_USE_PREAD
#    ifdef ILMTHREAD_THREADING_ENABLED
        pthread_mutex_destroy (&(fh->mutex));
#    endif
#endif
    }
    (void) c;
    (void) failed;
}

/**************************************/

static exr_result_t
finalize_write (struct _internal_exr_context* pf, int failed)
{
    exr_result_t rv = EXR_ERR_SUCCESS;

    /* TODO: Do we actually want to do this or leave the garbage file there */
    if (failed && pf->destroy_fn == &default_shutdown)
    {
        if (pf->tmp_filename.str)
            unlink (pf->tmp_filename.str);
        else
            unlink (pf->filename.str);
    }

    if (!failed && pf->tmp_filename.str)
    {
        int mvret = rename (pf->tmp_filename.str, pf->filename.str);
        if (mvret < 0)
            return pf->print_error (
                pf,
                EXR_ERR_FILE_ACCESS,
                "Unable to rename temporary file: %s",
                strerror (rv));
    }

    return rv;
}

/**************************************/

static int64_t
default_read_func (
    exr_const_context_t         ctxt,
    void*                       userdata,
    void*                       buffer,
    uint64_t                    sz,
    uint64_t                    offset,
    exr_stream_error_func_ptr_t error_cb)
{
    int64_t                          rv, retsz = -1;
    struct _internal_exr_filehandle* fh     = userdata;
    int                              fd     = -1;
    char*                            curbuf = (char*) buffer;
    uint64_t                         readsz = sz;

    if (sizeof (size_t) == 4)
    {
        if (sz >= (uint64_t) UINT32_MAX)
        {
            if (error_cb)
                error_cb (
                    ctxt,
                    EXR_ERR_INVALID_ARGUMENT,
                    "read request size too large for architecture");
            return retsz;
        }
    }

    if (!fh)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file handle pointer");
        return retsz;
    }

    fd = fh->fd;
    if (fd < 0)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file descriptor");
        return retsz;
    }

#if !CAN_USE_PREAD
#    ifdef ILMTHREAD_THREADING_ENABLED
    pthread_mutex_lock (&(fh->mutex));
#    endif
    {
#    if defined __USE_LARGEFILE64
        uint64_t spos = (uint64_t) lseek64 (fd, (off64_t) offset, SEEK_SET);
#    else
        uint64_t spos = (uint64_t) lseek (fd, (off_t) offset, SEEK_SET);
#    endif
        if (spos != offset)
        {
#    ifdef ILMTHREAD_THREADING_ENABLED
            pthread_mutex_unlock (&(fh->mutex));
#    endif
            if (error_cb)
            {
                if (spos == (uint64_t) -1)
                    error_cb (ctxt, EXR_ERR_READ_IO, strerror (errno));
                else
                    error_cb (
                        ctxt,
                        EXR_ERR_READ_IO,
                        "Unable to seek to requested position");
            }
            return retsz;
        }
    }
#endif

    retsz = 0;
    do
    {
#if CAN_USE_PREAD
        rv = pread (fd, curbuf, (size_t) readsz, (off_t) offset);
#else
        rv = read (fd, curbuf, (size_t) readsz);
#endif
        if (rv < 0)
        {
            if (errno == EINTR) continue;
            if (errno == EAGAIN) continue;
            retsz = -1;
            break;
        }
        if (rv == 0) break;
        retsz += rv;
        curbuf += rv;
        readsz -= (uint64_t) rv;
        offset += (uint64_t) rv;
    } while (retsz < (int64_t) sz);

#if !CAN_USE_PREAD
#    ifdef ILMTHREAD_THREADING_ENABLED
    pthread_mutex_unlock (&(fh->mutex));
#    endif
#endif
    if (retsz < 0 && error_cb)
        error_cb (
            ctxt,
            EXR_ERR_READ_IO,
            "Unable to read %" PRIu64 " bytes: %s",
            sz,
            strerror (errno));
    return retsz;
}

/**************************************/

static int64_t
default_write_func (
    exr_const_context_t         ctxt,
    void*                       userdata,
    const void*                 buffer,
    uint64_t                    sz,
    uint64_t                    offset,
    exr_stream_error_func_ptr_t error_cb)
{
    int64_t                          rv, retsz = -1;
    struct _internal_exr_filehandle* fh      = userdata;
    int                              fd      = -1;
    const uint8_t*                   curbuf  = (const uint8_t*) buffer;
    uint64_t                         writesz = sz;

    if (sizeof (size_t) < sizeof (uint64_t))
    {
        if (sz >= (uint64_t) UINT32_MAX)
        {
            if (error_cb)
                error_cb (
                    ctxt,
                    EXR_ERR_INVALID_ARGUMENT,
                    "read request size too large for architecture");
            return retsz;
        }
    }

    if (!fh)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file handle pointer");
        return retsz;
    }

    fd = fh->fd;
    if (fd < 0)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file descriptor");
        return retsz;
    }

#if !CAN_USE_PREAD
#    ifdef ILMTHREAD_THREADING_ENABLED
    pthread_mutex_lock (&(fh->mutex));
#    endif
    {
#    if defined __USE_LARGEFILE64
        uint64_t spos = (uint64_t) lseek64 (fd, (off64_t) offset, SEEK_SET);
#    else
        uint64_t spos = (uint64_t) lseek (fd, (off_t) offset, SEEK_SET);
#    endif
        if (spos != offset)
        {
#    ifdef ILMTHREAD_THREADING_ENABLED
            pthread_mutex_unlock (&(fh->mutex));
#    endif
            if (error_cb)
            {
                if (spos == (uint64_t) -1)
                    error_cb (ctxt, EXR_ERR_WRITE_IO, strerror (errno));
                else
                    error_cb (
                        ctxt,
                        EXR_ERR_WRITE_IO,
                        "Unable to seek to requested position");
            }
            return retsz;
        }
    }
#endif

    retsz = 0;
    do
    {
#if CAN_USE_PREAD
        rv = pwrite (fd, curbuf, (size_t) writesz, (off_t) offset);
#else
        rv = write (fd, curbuf, (size_t) writesz);
#endif
        if (rv < 0)
        {
            if (errno == EINTR) continue;
            if (errno == EAGAIN) continue;
            retsz = -1;
            break;
        }
        retsz += rv;
        curbuf += rv;
        writesz -= (uint64_t) rv;
        offset += (uint64_t) rv;
    } while (retsz < (int64_t) sz);

#if !CAN_USE_PREAD
#    ifdef ILMTHREAD_THREADING_ENABLED
    pthread_mutex_unlock (&(fh->mutex));
#    endif
#endif
    if (retsz != (int64_t) sz && error_cb)
        error_cb (
            ctxt,
            EXR_ERR_WRITE_IO,
            "Unable to write %" PRIu64 " bytes to stream, wrote %" PRId64
            ": %s",
            sz,
            retsz,
            strerror (errno));
    return retsz;
}

/**************************************/

static exr_result_t
default_init_read_file (struct _internal_exr_context* file)
{
    int                              fd;
    struct _internal_exr_filehandle* fh = file->user_data;

    fh->fd = -1;
#if !CAN_USE_PREAD
#    ifdef ILMTHREAD_THREADING_ENABLED
    fd = pthread_mutex_init (&(fh->mutex), NULL);
    if (fd != 0)
        return file->print_error (
            file,
            EXR_ERR_OUT_OF_MEMORY,
            "Unable to initialize file mutex: %s",
            strerror (fd));
#    endif
#endif

    file->destroy_fn = &default_shutdown;
    file->read_fn    = &default_read_func;

    fd = open (file->filename.str, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return file->print_error (
            file,
            EXR_ERR_FILE_ACCESS,
            "Unable to open file for read: %s",
            strerror (errno));

    fh->fd = fd;
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
default_init_write_file (struct _internal_exr_context* file)
{
    int                              fd;
    struct _internal_exr_filehandle* fh    = file->user_data;
    const char*                      outfn = file->tmp_filename.str;
    if (outfn == NULL) outfn = file->filename.str;

#if !CAN_USE_PREAD
#    ifdef ILMTHREAD_THREADING_ENABLED
    fd = pthread_mutex_init (&(fh->mutex), NULL);
    if (fd != 0)
        return file->print_error (
            file,
            EXR_ERR_OUT_OF_MEMORY,
            "Unable to initialize file mutex: %s",
            strerror (fd));
#    endif
#endif

    fh->fd           = -1;
    file->destroy_fn = &default_shutdown;
    file->write_fn   = &default_write_func;

    fd = open (
        outfn,
        O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    if (fd < 0)
        return file->print_error (
            file,
            EXR_ERR_FILE_ACCESS,
            "Unable to open file for write: %s",
            strerror (errno));
    fh->fd = fd;

    return EXR_ERR_SUCCESS;
}

/**************************************/

static int64_t
default_query_size_func (exr_const_context_t ctxt, void* userdata)
{
    struct stat                      sbuf;
    struct _internal_exr_filehandle* fh = userdata;
    int64_t                          sz = -1;

    if (fh->fd >= 0)
    {
        int rv = fstat (fh->fd, &sbuf);
        if (rv == 0) sz = (int64_t) sbuf.st_size;
    }

    (void) ctxt;
    return sz;
}

/**************************************/

static exr_result_t
make_temp_filename (struct _internal_exr_context* ret)
{
    /* we checked the pointers we care about before calling */
    char        tmproot[32];
    char*       tmpname;
    uint64_t    tlen, newlen;
    const char* srcfile = ret->filename.str;
    int         nwr     = snprintf (tmproot, 32, "tmp.%d", getpid ());
    if (nwr >= 32)
        return ret->report_error (
            ret,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid assumption in temporary filename");

    tlen   = strlen (tmproot);
    newlen = tlen + (uint64_t) ret->filename.length;

    if (newlen >= INT32_MAX)
        return ret->standard_error (ret, EXR_ERR_OUT_OF_MEMORY);

    tmpname = ret->alloc_fn (newlen + 1);
    if (tmpname)
    {
        const char* lastslash = strrchr (srcfile, '/');

        ret->tmp_filename.length     = (int32_t) (newlen);
        ret->tmp_filename.alloc_size = (int32_t) (newlen + 1);
        ret->tmp_filename.str        = tmpname;

        if (lastslash)
        {
            uint64_t nPrev = (uintptr_t) lastslash - (uintptr_t) srcfile + 1;
            strncpy (tmpname, srcfile, nPrev);
            strncpy (tmpname + nPrev, tmproot, tlen);
            strncpy (
                tmpname + nPrev + tlen,
                srcfile + nPrev,
                (uint64_t) (ret->filename.length) - nPrev);
            tmpname[newlen] = '\0';
        }
        else
        {
            strncpy (tmpname, tmproot, tlen);
            strncpy (tmpname + tlen, srcfile, (size_t) ret->filename.length);
            tmpname[newlen] = '\0';
        }
    }
    else
        return ret->print_error (
            ret,
            EXR_ERR_OUT_OF_MEMORY,
            "Unable to create %" PRIu64 " bytes for temporary filename",
            newlen + 1);
    return EXR_ERR_SUCCESS;
}
