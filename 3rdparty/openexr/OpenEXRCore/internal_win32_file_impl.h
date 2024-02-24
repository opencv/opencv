/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

/* implementation for windows (win32) native file io routines (used in context.c) */

#include <fileapi.h>
#include <inttypes.h>
#include <strsafe.h>
#include <windows.h>

#ifdef _MSC_VER
#    ifndef PRId64
#        define PRId64 "I64d"
#    endif
#    ifndef PRId64
#        define PRIu64 "I64u"
#    endif
#endif

static exr_result_t
print_error_helper (
    struct _internal_exr_context* pf,
    exr_result_t                  errcode,
    DWORD                         dw,
    exr_stream_error_func_ptr_t   error_cb,
    const char*                   msg)
{
    LPVOID   lpMsgBuf;
    LPVOID   lpDisplayBuf;
    uint64_t bufsz = 0;

    FormatMessage (
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID (LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0,
        NULL);

    bufsz =
        (lstrlen ((LPCTSTR) lpMsgBuf) + strlen (msg) +
         20); /* extra for format string */

    lpDisplayBuf = (LPVOID) LocalAlloc (LMEM_ZEROINIT, bufsz * sizeof (TCHAR));

    if (FAILED (StringCchPrintf (
            (LPTSTR) lpDisplayBuf,
            bufsz,
            TEXT ("%s: (%" PRId64 ") %s"),
            msg,
            (int64_t) dw,
            lpMsgBuf)))
    {
        return pf->print_error (
            pf, EXR_ERR_OUT_OF_MEMORY, "Unable to format message print");
    }

    if (error_cb)
        error_cb (
            (exr_const_context_t) pf,
            errcode,
            "%s",
            (const char*) lpDisplayBuf);
    else
        pf->print_error (pf, errcode, "%s", (const char*) lpDisplayBuf);

    LocalFree (lpMsgBuf);
    LocalFree (lpDisplayBuf);

    return errcode;
}

static exr_result_t
print_error (
    struct _internal_exr_context* pf, exr_result_t errcode, const char* msg)
{
    return print_error_helper (pf, errcode, GetLastError (), NULL, msg);
}

static exr_result_t
send_error (
    struct _internal_exr_context* pf,
    exr_result_t                  errcode,
    exr_stream_error_func_ptr_t   error_cb,
    const char*                   msg)
{
    return print_error_helper (pf, errcode, GetLastError (), error_cb, msg);
}

static wchar_t*
widen_filename (struct _internal_exr_context* file, const char* fn)
{
    int      wcSize = 0, fnlen = 0;
    wchar_t* wcFn = NULL;

    fnlen  = (int) strlen (fn);
    wcSize = MultiByteToWideChar (CP_UTF8, 0, fn, fnlen, NULL, 0);
    wcFn   = file->alloc_fn (sizeof (wchar_t) * (wcSize + 1));
    if (wcFn)
    {
        MultiByteToWideChar (CP_UTF8, 0, fn, fnlen, wcFn, wcSize);
        wcFn[wcSize] = 0;
    }
    return wcFn;
}

struct _internal_exr_filehandle
{
    HANDLE fd;
};

/**************************************/

static void
default_shutdown (exr_const_context_t c, void* userdata, int failed)
{
    /* we will handle failure before here */
    struct _internal_exr_filehandle* fh = userdata;
    if (fh)
    {
        if (fh->fd != INVALID_HANDLE_VALUE) CloseHandle (fh->fd);
        fh->fd = INVALID_HANDLE_VALUE;
    }
}

/**************************************/

static exr_result_t
finalize_write (struct _internal_exr_context* pf, int failed)
{
    /* TODO: Do we actually want to do this or leave the garbage file there */
    if (failed && pf->destroy_fn == &default_shutdown)
    {
        wchar_t* wcFn;
        if (pf->tmp_filename.str)
            wcFn = widen_filename (pf, pf->tmp_filename.str);
        else
            wcFn = widen_filename (pf, pf->filename.str);
        if (wcFn)
        {
            DeleteFileW (wcFn);
            pf->free_fn (wcFn);
        }
    }

    if (!failed && pf->tmp_filename.str)
    {
        wchar_t* wcFnTmp = widen_filename (pf, pf->tmp_filename.str);
        wchar_t* wcFn    = widen_filename (pf, pf->filename.str);
        BOOL     res     = FALSE;
        if (wcFn && wcFnTmp)
            res = ReplaceFileW (wcFn, wcFnTmp, NULL, 0, NULL, NULL);
        pf->free_fn (wcFn);
        pf->free_fn (wcFnTmp);

        if (!res)
            return print_error (
                pf,
                EXR_ERR_FILE_ACCESS,
                "Unable to rename temporary file to final destination");
    }

    return EXR_ERR_SUCCESS;
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
    int64_t                          retsz = -1;
    DWORD                            nread = 0;
    struct _internal_exr_filehandle* fh    = userdata;
    HANDLE                           fd;
    LARGE_INTEGER                    lint;
    OVERLAPPED                       overlap = {0};

    if (!fh)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file handle pointer");
        return retsz;
    }

    fd = fh->fd;
    if (fd == INVALID_HANDLE_VALUE)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file descriptor");
        return retsz;
    }

    if (sz > (uint64_t) (INT32_MAX))
    {
        if (error_cb)
            error_cb (
                ctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Read request too large for win32 api");
        return retsz;
    }

    lint.QuadPart      = offset;
    overlap.Offset     = lint.LowPart;
    overlap.OffsetHigh = lint.HighPart;
    if (ReadFile (fd, buffer, (DWORD) sz, &nread, &overlap)) { retsz = nread; }
    else
    {
        DWORD dw = GetLastError ();
        if (dw != ERROR_HANDLE_EOF)
        {
            print_error_helper (
                EXR_CTXT (ctxt),
                EXR_ERR_READ_IO,
                dw,
                error_cb,
                "Unable to read requested data");
        }
        else { retsz = nread; }
    }

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
    int64_t                          retsz = -1;
    struct _internal_exr_filehandle* fh    = userdata;
    HANDLE                           fd;
    DWORD                            nwrote = 0;
    LARGE_INTEGER                    lint;
    OVERLAPPED                       overlap = {0};

    if (!fh)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file handle pointer");
        return retsz;
    }

    fd = fh->fd;
    if (fd == INVALID_HANDLE_VALUE)
    {
        if (error_cb)
            error_cb (
                ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid file descriptor");
        return retsz;
    }

    if (sz > (uint64_t) (INT32_MAX))
    {
        if (error_cb)
            error_cb (
                ctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Read request too large for win32 api");
        return retsz;
    }

    lint.QuadPart      = offset;
    overlap.Offset     = lint.LowPart;
    overlap.OffsetHigh = lint.HighPart;
    if (WriteFile (fd, buffer, (DWORD) sz, &nwrote, &overlap))
    {
        retsz = nwrote;
    }
    else
    {
        DWORD dw = GetLastError ();
        print_error_helper (
            EXR_CTXT (ctxt),
            EXR_ERR_READ_IO,
            dw,
            error_cb,
            "Unable to write requested data");
    }

    return retsz;
}

/**************************************/

static exr_result_t
default_init_read_file (struct _internal_exr_context* file)
{
    wchar_t*                         wcFn = NULL;
    HANDLE                           fd;
    struct _internal_exr_filehandle* fh = file->user_data;

    fh->fd           = INVALID_HANDLE_VALUE;
    file->destroy_fn = &default_shutdown;
    file->read_fn    = &default_read_func;

    wcFn = widen_filename (file, file->filename.str);
    if (wcFn)
    {
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= _WIN32_WINNT_WIN8)
        fd = CreateFile2 (
            wcFn, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NULL);
#else
        fd = CreateFileW (
            wcFn,
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL, /* TBD: use overlapped? | FILE_FLAG_OVERLAPPED */
            NULL);
#endif
        file->free_fn (wcFn);

        if (fd == INVALID_HANDLE_VALUE)
            return print_error (
                file, EXR_ERR_FILE_ACCESS, "Unable to open file for read");
    }
    else
        return print_error (
            file, EXR_ERR_OUT_OF_MEMORY, "Unable to allocate unicode filename");

    fh->fd = fd;

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
default_init_write_file (struct _internal_exr_context* file)
{
    wchar_t*                         wcFn = NULL;
    struct _internal_exr_filehandle* fh   = file->user_data;
    HANDLE                           fd;
    const char*                      outfn = file->tmp_filename.str;

    if (outfn == NULL) outfn = file->filename.str;

    fh->fd           = INVALID_HANDLE_VALUE;
    file->destroy_fn = &default_shutdown;
    file->write_fn   = &default_write_func;

    wcFn = widen_filename (file, outfn);
    if (wcFn)
    {
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= _WIN32_WINNT_WIN8)
        fd = CreateFile2 (
            wcFn,
            GENERIC_WRITE | DELETE,
            0, /* no sharing */
            CREATE_ALWAYS,
            NULL);
#else
        fd = CreateFileW (
            wcFn,
            GENERIC_WRITE | DELETE,
            0, /* no sharing */
            NULL,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL, /* TBD: use overlapped? | FILE_FLAG_OVERLAPPED */
            NULL);
#endif
        file->free_fn (wcFn);

        if (fd == INVALID_HANDLE_VALUE)
            return print_error (
                file, EXR_ERR_FILE_ACCESS, "Unable to open file for write");
    }
    else
        return print_error (
            file, EXR_ERR_OUT_OF_MEMORY, "Unable to allocate unicode filename");

    fh->fd = fd;
    return EXR_ERR_SUCCESS;
}

/**************************************/

static int64_t
default_query_size_func (exr_const_context_t ctxt, void* userdata)
{
    struct _internal_exr_filehandle* fh = userdata;
    int64_t                          sz = -1;

    if (fh->fd != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER lint = {0};
        if (GetFileSizeEx (fh->fd, &lint)) { sz = lint.QuadPart; }
    }

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
    int         nwr =
        _snprintf_s (tmproot, 32, _TRUNCATE, "tmp.%d", GetCurrentProcessId ());
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
        // windows allows both
        const char* lastslash  = strrchr (srcfile, '\\');
        const char* lastslashu = strrchr (srcfile, '/');

        if (lastslash)
        {
            if (lastslashu && lastslashu > lastslash) lastslash = lastslashu;
        }
        else
            lastslash = lastslashu;

        ret->tmp_filename.length     = (int32_t) (newlen);
        ret->tmp_filename.alloc_size = (int32_t) (newlen + 1);
        ret->tmp_filename.str        = tmpname;

        if (lastslash)
        {
            uintptr_t nPrev = (uintptr_t) lastslash - (uintptr_t) srcfile + 1;
            strncpy_s (tmpname, newlen + 1, srcfile, nPrev);
            strncpy_s (tmpname + nPrev, newlen + 1 - nPrev, tmproot, tlen);
            strncpy_s (
                tmpname + nPrev + tlen,
                newlen + 1 - nPrev - tlen,
                srcfile + nPrev,
                ret->filename.length - nPrev);
            tmpname[newlen] = '\0';
        }
        else
        {
            strncpy_s (tmpname, newlen + 1, tmproot, tlen);
            strncpy_s (
                tmpname + tlen,
                newlen + 1 - tlen,
                srcfile,
                ret->filename.length);
            tmpname[newlen] = '\0';
        }
    }
    else
        return ret->print_error (
            ret,
            EXR_ERR_OUT_OF_MEMORY,
            "Unable to create %" PRIu64 " bytes for temporary filename",
            (uint64_t) newlen + 1);
    return EXR_ERR_SUCCESS;
}
