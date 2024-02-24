/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_file.h"

#include "internal_attr.h"
#include "internal_constants.h"
#include "internal_structs.h"
#include "internal_xdr.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>

/**************************************/

static exr_result_t
silent_error (
    const struct _internal_exr_context* pctxt,
    exr_result_t                        code,
    const char*                         msg)
{
    (void) pctxt;
    (void) msg;
    return code;
}

static exr_result_t
silent_standard_error (
    const struct _internal_exr_context* pctxt, exr_result_t code)
{
    (void) pctxt;
    return code;
}

static exr_result_t
silent_print_error (
    const struct _internal_exr_context* pctxt,
    exr_result_t                        code,
    const char*                         msg,
    ...)
{
    (void) pctxt;
    (void) msg;
    return code;
}

/**************************************/

struct _internal_exr_seq_scratch
{
    uint8_t* scratch;
    uint64_t curpos;
    int64_t  navail;
    uint64_t fileoff;

    exr_result_t (*sequential_read) (
        struct _internal_exr_seq_scratch*, void*, uint64_t);
    exr_result_t (*sequential_skip) (
        struct _internal_exr_seq_scratch*, int32_t);

    struct _internal_exr_context* ctxt;
};

static inline int
scratch_attr_too_big (
    struct _internal_exr_seq_scratch* scr, int32_t attrsz, int64_t fsize)
{
    int64_t acmp = (int64_t) attrsz;
    if (fsize > 0 && (acmp > scr->navail))
    {
        int64_t test = acmp - scr->navail;
        int64_t foff = (int64_t) scr->fileoff;
        if ((foff + test) > fsize) return 1;
    }
    return 0;
}

#define SCRATCH_BUFFER_SIZE 4096

static exr_result_t
scratch_seq_read (struct _internal_exr_seq_scratch* scr, void* buf, uint64_t sz)
{
    uint8_t*     outbuf  = buf;
    uint64_t     nCopied = 0;
    uint64_t     notdone = sz;
    exr_result_t rv      = -1;

    while (notdone > 0)
    {
        if (scr->navail > 0)
        {
            uint64_t nLeft = (uint64_t) scr->navail;
            uint64_t nCopy = notdone;
            if (nCopy > nLeft) nCopy = nLeft;
            memcpy (outbuf, scr->scratch + scr->curpos, nCopy);
            scr->curpos += nCopy;
            scr->navail -= (int64_t) nCopy;
            notdone -= nCopy;
            outbuf += nCopy;
            nCopied += nCopy;
        }
        else if (notdone > SCRATCH_BUFFER_SIZE)
        {
            uint64_t nPages  = notdone / SCRATCH_BUFFER_SIZE;
            int64_t  nread   = 0;
            uint64_t nToRead = nPages * SCRATCH_BUFFER_SIZE;
            rv               = scr->ctxt->do_read (
                scr->ctxt,
                outbuf,
                nToRead,
                &(scr->fileoff),
                &nread,
                EXR_MUST_READ_ALL);
            if (nread > 0)
            {
                notdone -= (uint64_t) nread;
                outbuf += nread;
                nCopied += (uint64_t) nread;
            }
            else { break; }
        }
        else
        {
            int64_t nread = 0;
            rv            = scr->ctxt->do_read (
                scr->ctxt,
                scr->scratch,
                SCRATCH_BUFFER_SIZE,
                &(scr->fileoff),
                &nread,
                EXR_ALLOW_SHORT_READ);
            if (nread > 0)
            {
                scr->navail = nread;
                scr->curpos = 0;
            }
            else
            {
                if (nread == 0)
                    rv = scr->ctxt->report_error (
                        scr->ctxt,
                        EXR_ERR_READ_IO,
                        "End of file attempting to read header");
                break;
            }
        }
    }
    if (rv == -1)
    {
        if (nCopied == sz)
            rv = EXR_ERR_SUCCESS;
        else
            rv = EXR_ERR_READ_IO;
    }
    return rv;
}

static exr_result_t
scratch_seq_skip (struct _internal_exr_seq_scratch* scr, int32_t sz)
{
    uint64_t     nCopied = 0;
    uint64_t     notdone = (uint64_t) sz;
    exr_result_t rv      = -1;

    while (notdone > 0)
    {
        if (scr->navail > 0)
        {
            uint64_t nLeft = (uint64_t) scr->navail;
            uint64_t nCopy = notdone;
            if (nCopy > nLeft) nCopy = nLeft;
            scr->curpos += nCopy;
            scr->navail -= (int64_t) nCopy;
            notdone -= nCopy;
            nCopied += nCopy;
        }
        else
        {
            int64_t nread = 0;
            rv            = scr->ctxt->do_read (
                scr->ctxt,
                scr->scratch,
                SCRATCH_BUFFER_SIZE,
                &(scr->fileoff),
                &nread,
                EXR_ALLOW_SHORT_READ);
            if (nread > 0)
            {
                scr->navail = nread;
                scr->curpos = 0;
            }
            else
            {
                if (nread == 0)
                    rv = scr->ctxt->report_error (
                        scr->ctxt,
                        EXR_ERR_READ_IO,
                        "End of file attempting to read header");
                break;
            }
        }
    }
    if (rv == -1)
    {
        if (nCopied == (uint64_t) sz)
            rv = EXR_ERR_SUCCESS;
        else
            rv = EXR_ERR_READ_IO;
    }
    return rv;
}

/**************************************/

static exr_result_t
priv_init_scratch (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scr,
    uint64_t                          offset)
{
    scr->curpos          = 0;
    scr->navail          = 0;
    scr->fileoff         = offset;
    scr->sequential_read = &scratch_seq_read;
    scr->sequential_skip = &scratch_seq_skip;
    scr->ctxt            = ctxt;
    scr->scratch         = ctxt->alloc_fn (SCRATCH_BUFFER_SIZE);
    if (scr->scratch == NULL)
        return ctxt->standard_error (ctxt, EXR_ERR_OUT_OF_MEMORY);
    return EXR_ERR_SUCCESS;
}

/**************************************/

static void
priv_destroy_scratch (struct _internal_exr_seq_scratch* scr)
{
    struct _internal_exr_context* pctxt = scr->ctxt;
    if (scr->scratch) pctxt->free_fn (scr->scratch);
}

/**************************************/

static exr_result_t
check_bad_attrsz (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    int32_t                           attrsz,
    int32_t                           eltsize,
    const char*                       aname,
    const char*                       tname,
    int32_t*                          outsz)
{
    int32_t n = attrsz;

    *outsz = n;
    if (attrsz < 0)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s', type '%s': Invalid negative size %d",
            aname,
            tname,
            attrsz);

    if (scratch_attr_too_big (scratch, attrsz, ctxt->file_size))
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s', type '%s': Invalid size %d",
            aname,
            tname,
            attrsz);

    if (eltsize > 1)
    {
        n = attrsz / eltsize;
        if (attrsz != (int32_t) (n * eltsize))
            return ctxt->print_error (
                ctxt,
                EXR_ERR_ATTR_SIZE_MISMATCH,
                "Attribute '%s': Invalid size %d (exp '%s' size 4 * n, found odd bytes %d)",
                aname,
                attrsz,
                tname,
                (attrsz % eltsize));
        *outsz = n;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
read_text (
    struct _internal_exr_context*     ctxt,
    char                              text[256],
    int32_t*                          outlen,
    int32_t                           maxlen,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       type)
{
    char         b;
    exr_result_t rv      = EXR_ERR_SUCCESS;
    int32_t      namelen = *outlen;

    while (namelen <= maxlen)
    {
        rv = scratch->sequential_read (scratch, &b, 1);
        if (rv != EXR_ERR_SUCCESS) return rv;
        text[namelen] = b;
        if (b == '\0') break;
        ++namelen;
    }
    *outlen = namelen;
    if (namelen > maxlen)
    {
        text[maxlen - 1] = '\0';
        return ctxt->print_error (
            ctxt,
            EXR_ERR_NAME_TOO_LONG,
            "Invalid %s encountered: start '%s' (max %d)",
            type,
            text,
            maxlen);
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
extract_attr_chlist (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    exr_attr_chlist_t*                attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz)
{
    char         chname[256];
    int32_t      chlen;
    int32_t      ptype, xsamp, ysamp;
    uint8_t      flags[4];
    int32_t      maxlen = ctxt->max_name_length;
    exr_result_t rv;

    rv = check_bad_attrsz (ctxt, scratch, attrsz, 1, aname, tname, &chlen);

    while (rv == EXR_ERR_SUCCESS && attrsz > 0)
    {
        chlen = 0;
        rv    = read_text (ctxt, chname, &chlen, maxlen, scratch, aname);
        if (rv != EXR_ERR_SUCCESS) break;
        attrsz -= chlen + 1;

        if (chlen == 0) break;

        if (attrsz < 16)
        {
            return ctxt->print_error (
                ctxt,
                EXR_ERR_ATTR_SIZE_MISMATCH,
                "Out of data parsing '%s', last channel '%s'",
                aname,
                chname);
        }

        rv = scratch->sequential_read (scratch, &ptype, 4);
        if (rv != EXR_ERR_SUCCESS) break;
        rv = scratch->sequential_read (scratch, &flags, 4);
        if (rv != EXR_ERR_SUCCESS) break;
        rv = scratch->sequential_read (scratch, &xsamp, 4);
        if (rv != EXR_ERR_SUCCESS) break;
        rv = scratch->sequential_read (scratch, &ysamp, 4);
        if (rv != EXR_ERR_SUCCESS) break;

        attrsz -= 16;
        ptype = (int32_t) one_to_native32 ((uint32_t) ptype);
        xsamp = (int32_t) one_to_native32 ((uint32_t) xsamp);
        ysamp = (int32_t) one_to_native32 ((uint32_t) ysamp);

        rv = exr_attr_chlist_add_with_length (
            (exr_context_t) ctxt,
            attrdata,
            chname,
            chlen,
            (exr_pixel_type_t) ptype,
            flags[0],
            xsamp,
            ysamp);
    }
    return rv;
}

/**************************************/

static exr_result_t
extract_attr_uint8 (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    uint8_t*                          attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz,
    uint8_t                           maxval)
{
    if (attrsz != 1)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s': Invalid size %d (exp '%s' size 1)",
            aname,
            attrsz,
            tname);

    if (scratch->sequential_read (scratch, attrdata, sizeof (uint8_t)))
        return ctxt->print_error (
            ctxt, EXR_ERR_READ_IO, "Unable to read '%s' %s data", aname, tname);

    if (*attrdata >= maxval)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Attribute '%s' (type '%s'): Invalid value %d (max allowed %d)",
            aname,
            tname,
            (int) *attrdata,
            (int) maxval);

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
extract_attr_64bit (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    void*                             attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz,
    int32_t                           num)
{
    exr_result_t rv;
    if (attrsz != 8 * num)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s': Invalid size %d (exp '%s' size 8 * %d (%d))",
            aname,
            attrsz,
            tname,
            num,
            8 * num);

    rv = scratch->sequential_read (scratch, attrdata, 8 * (uint64_t) num);
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt, rv, "Unable to read '%s' %s data", aname, tname);

    priv_to_native64 (attrdata, num);
    return rv;
}

/**************************************/

static exr_result_t
extract_attr_32bit (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    void*                             attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz,
    int32_t                           num)
{
    exr_result_t rv;
    if (attrsz != 4 * num)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s': Invalid size %d (exp '%s' size 4 * %d (%d))",
            aname,
            attrsz,
            tname,
            num,
            4 * num);

    rv = scratch->sequential_read (scratch, attrdata, 4 * (uint64_t) num);
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt, rv, "Unable to read '%s' %s data", aname, tname);

    priv_to_native32 (attrdata, num);
    return rv;
}

/**************************************/

static exr_result_t
extract_attr_float_vector (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    exr_attr_float_vector_t*          attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz)
{
    int32_t      n  = 0;
    exr_result_t rv = check_bad_attrsz (
        ctxt, scratch, attrsz, (int) sizeof (float), aname, tname, &n);

    /* in case of duplicate attr name in header (mostly fuzz testing) */
    exr_attr_float_vector_destroy ((exr_context_t) ctxt, attrdata);

    if (rv == EXR_ERR_SUCCESS && n > 0)
    {
        rv = exr_attr_float_vector_init ((exr_context_t) ctxt, attrdata, n);
        if (rv != EXR_ERR_SUCCESS) return rv;

        rv = scratch->sequential_read (
            scratch, EXR_CONST_CAST (void*, attrdata->arr), (uint64_t) attrsz);
        if (rv != EXR_ERR_SUCCESS)
        {
            exr_attr_float_vector_destroy ((exr_context_t) ctxt, attrdata);
            return ctxt->print_error (
                ctxt,
                EXR_ERR_READ_IO,
                "Unable to read '%s' %s data",
                aname,
                tname);
        }

        priv_to_native32 (attrdata, n);
    }

    return rv;
}

/**************************************/

static exr_result_t
extract_attr_string (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    exr_attr_string_t*                attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz,
    char*                             strptr)
{
    exr_result_t rv =
        scratch->sequential_read (scratch, (void*) strptr, (uint64_t) attrsz);

    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt, rv, "Unable to read '%s' %s data", aname, tname);

    strptr[attrsz] = '\0';

    return exr_attr_string_init_static_with_length (
        (exr_context_t) ctxt, attrdata, strptr, attrsz);
}

/**************************************/

static exr_result_t
extract_attr_string_vector (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    exr_attr_string_vector_t*         attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t       rv;
    int32_t            n, nstr, nalloced, nlen, pulled = 0;
    exr_attr_string_t *nlist, *clist, nil = {0};

    rv = check_bad_attrsz (ctxt, scratch, attrsz, 1, aname, tname, &n);
    if (rv != EXR_ERR_SUCCESS) return rv;

    nstr     = 0;
    nalloced = 0;
    clist    = NULL;
    while (pulled < attrsz)
    {
        nlen = 0;
        rv   = scratch->sequential_read (scratch, &nlen, sizeof (int32_t));
        if (rv != EXR_ERR_SUCCESS)
        {
            rv = ctxt->print_error (
                ctxt,
                rv,
                "Attribute '%s': Unable to read string length",
                aname);
            goto extract_string_vector_fail;
        }

        pulled += sizeof (int32_t);
        nlen = (int32_t) one_to_native32 ((uint32_t) nlen);
        if (nlen < 0 || (ctxt->file_size > 0 && nlen > ctxt->file_size))
        {
            rv = ctxt->print_error (
                ctxt,
                EXR_ERR_INVALID_ATTR,
                "Attribute '%s': Invalid size (%d) encountered parsing string vector",
                aname,
                nlen);
            goto extract_string_vector_fail;
        }

        if (nalloced == 0)
        {
            clist = ctxt->alloc_fn (4 * sizeof (exr_attr_string_t));
            if (clist == NULL)
            {
                rv = ctxt->standard_error (ctxt, EXR_ERR_OUT_OF_MEMORY);
                goto extract_string_vector_fail;
            }
            nalloced = 4;
        }
        if ((nstr + 1) >= nalloced)
        {
            nalloced *= 2;
            nlist = ctxt->alloc_fn (
                (size_t) (nalloced) * sizeof (exr_attr_string_t));
            if (nlist == NULL)
            {
                rv = ctxt->standard_error (ctxt, EXR_ERR_OUT_OF_MEMORY);
                goto extract_string_vector_fail;
            }
            for (int32_t i = 0; i < nstr; ++i)
                *(nlist + i) = clist[i];
            ctxt->free_fn (clist);
            clist = nlist;
        }
        nlist  = clist + nstr;
        *nlist = nil;
        nstr += 1;
        rv = exr_attr_string_init ((exr_context_t) ctxt, nlist, nlen);
        if (rv != EXR_ERR_SUCCESS) goto extract_string_vector_fail;

        rv = scratch->sequential_read (
            scratch, EXR_CONST_CAST (void*, nlist->str), (uint64_t) nlen);
        if (rv != EXR_ERR_SUCCESS)
        {
            rv = ctxt->print_error (
                ctxt,
                rv,
                "Attribute '%s': Unable to read string of length (%d)",
                aname,
                nlen);
            goto extract_string_vector_fail;
        }
        *((EXR_CONST_CAST (char*, nlist->str)) + nlen) = '\0';
        pulled += nlen;
    }

    // just in case someone injected a duplicate attribute name into the header
    exr_attr_string_vector_destroy ((exr_context_t) ctxt, attrdata);
    attrdata->n_strings  = nstr;
    attrdata->alloc_size = nalloced;
    attrdata->strings    = clist;
    return EXR_ERR_SUCCESS;
extract_string_vector_fail:
    for (int32_t i = 0; i < nstr; ++i)
        exr_attr_string_destroy ((exr_context_t) ctxt, clist + i);
    if (clist) ctxt->free_fn (clist);

    return rv;
}

/**************************************/

static exr_result_t
extract_attr_tiledesc (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    exr_attr_tiledesc_t*              attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t rv;
    if (attrsz != (int32_t) sizeof (*attrdata))
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s': Invalid size %d (exp '%s' size %d)",
            aname,
            attrsz,
            tname,
            (int32_t) sizeof (*attrdata));

    rv = scratch->sequential_read (scratch, attrdata, sizeof (*attrdata));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt, rv, "Unable to read '%s' %s data", aname, tname);

    attrdata->x_size = one_to_native32 (attrdata->x_size);
    attrdata->y_size = one_to_native32 (attrdata->y_size);

    if ((int) EXR_GET_TILE_LEVEL_MODE (*attrdata) >= (int) EXR_TILE_LAST_TYPE)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Attribute '%s': Invalid tile level specification encountered: found enum %d",
            aname,
            (int) EXR_GET_TILE_LEVEL_MODE (*attrdata));

    if ((int) EXR_GET_TILE_ROUND_MODE (*attrdata) >=
        (int) EXR_TILE_ROUND_LAST_TYPE)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Attribute '%s': Invalid tile rounding specification encountered: found enum %d",
            aname,
            (int) EXR_GET_TILE_ROUND_MODE (*attrdata));

    return rv;
}

/**************************************/

static exr_result_t
extract_attr_opaque (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    exr_attr_opaquedata_t*            attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz)
{
    int32_t      n;
    exr_result_t rv;

    rv = check_bad_attrsz (ctxt, scratch, attrsz, 1, aname, tname, &n);
    if (rv != EXR_ERR_SUCCESS) return rv;

    exr_attr_opaquedata_destroy ((exr_context_t) ctxt, attrdata);
    rv = exr_attr_opaquedata_init (
        (exr_context_t) ctxt, attrdata, (uint64_t) attrsz);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = scratch->sequential_read (
        scratch, (void*) attrdata->packed_data, (uint64_t) attrsz);
    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_opaquedata_destroy ((exr_context_t) ctxt, attrdata);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_READ_IO,
            "Attribute '%s': Unable to read opaque %s data (%d bytes)",
            aname,
            tname,
            attrsz);
    }
    return rv;
}

/**************************************/

static exr_result_t
extract_attr_preview (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch,
    exr_attr_preview_t*               attrdata,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz)
{
    uint64_t     bytes;
    uint32_t     sz[2];
    exr_result_t rv;
    int64_t      fsize = ctxt->file_size;

    /* mostly for fuzzing, but just in case there's a duplicate name */
    exr_attr_preview_destroy ((exr_context_t) ctxt, attrdata);

    if (attrsz < 8)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s': Invalid size %d (exp '%s' size >= 8)",
            aname,
            attrsz,
            tname);

    rv = scratch->sequential_read (scratch, sz, sizeof (uint32_t) * 2);
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt, rv, "Attribute '%s': Unable to read preview sizes", aname);

    sz[0] = one_to_native32 (sz[0]);
    sz[1] = one_to_native32 (sz[1]);
    bytes = 4 * sz[0] * sz[1];
    if ((uint64_t) attrsz != (8 + bytes))
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Attribute '%s': Invalid size %d (exp '%s' %u x %u * 4 + sizevals)",
            aname,
            attrsz,
            tname,
            sz[0],
            sz[1]);

    if (bytes == 0 || (fsize > 0 && bytes >= (uint64_t) fsize))
    {
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Attribute '%s', type '%s': Invalid size for preview %u x %u",
            aname,
            tname,
            sz[0],
            sz[1]);
    }

    rv = exr_attr_preview_init ((exr_context_t) ctxt, attrdata, sz[0], sz[1]);
    if (rv != EXR_ERR_SUCCESS) return rv;

    if (bytes > 0)
    {
        rv = scratch->sequential_read (
            scratch, EXR_CONST_CAST (void*, attrdata->rgba), sz[0] * sz[1] * 4);
        if (rv != EXR_ERR_SUCCESS)
        {
            exr_attr_preview_destroy ((exr_context_t) ctxt, attrdata);
            return ctxt->print_error (
                ctxt,
                rv,
                "Attribute '%s': Unable to read preview data (%d bytes)",
                aname,
                attrsz);
        }
    }

    return rv;
}

/**************************************/

static exr_result_t
check_populate_channels (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_attr_chlist_t tmpchans = {0};
    exr_result_t      rv;

    if (curpart->channels)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute 'channels' encountered");
    }

    if (0 != strcmp (tname, "chlist"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute 'channels': Invalid type '%s'",
            tname);
    }

    rv = extract_attr_chlist (
        ctxt, scratch, &(tmpchans), EXR_REQ_CHANNELS_STR, tname, attrsz);
    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_chlist_destroy ((exr_context_t) ctxt, &(tmpchans));
        return rv;
    }

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_CHANNELS_STR,
        EXR_ATTR_CHLIST,
        0,
        NULL,
        &(curpart->channels));

    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_chlist_destroy ((exr_context_t) ctxt, &tmpchans);
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'chlist'",
            EXR_REQ_CHANNELS_STR);
    }

    exr_attr_chlist_destroy ((exr_context_t) ctxt, curpart->channels->chlist);
    *(curpart->channels->chlist) = tmpchans;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_compression (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    uint8_t      data;
    exr_result_t rv;

    if (curpart->compression)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute '%s' encountered",
            EXR_REQ_COMP_STR);
    }

    if (0 != strcmp (tname, EXR_REQ_COMP_STR))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute '%s': Invalid type '%s'",
            EXR_REQ_COMP_STR,
            tname);
    }

    rv = extract_attr_uint8 (
        ctxt,
        scratch,
        &data,
        EXR_REQ_COMP_STR,
        tname,
        attrsz,
        (uint8_t) EXR_COMPRESSION_LAST_TYPE);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_COMP_STR,
        EXR_ATTR_COMPRESSION,
        0,
        NULL,
        &(curpart->compression));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'compression'",
            EXR_REQ_COMP_STR);

    curpart->compression->uc = data;
    curpart->comp_type       = (exr_compression_t) data;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_dataWindow (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_attr_box2i_t tmpdata = {0};
    exr_result_t     rv;

    if (curpart->dataWindow)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute '%s' encountered",
            EXR_REQ_DATA_STR);
    }

    if (0 != strcmp (tname, "box2i"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute '%s': Invalid type '%s'",
            EXR_REQ_DATA_STR,
            tname);
    }

    rv = extract_attr_32bit (
        ctxt, scratch, &(tmpdata), EXR_REQ_DATA_STR, tname, attrsz, 4);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_DATA_STR,
        EXR_ATTR_BOX2I,
        0,
        NULL,
        &(curpart->dataWindow));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'box2i'",
            EXR_REQ_DATA_STR);

    *(curpart->dataWindow->box2i) = tmpdata;
    curpart->data_window          = tmpdata;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_displayWindow (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_attr_box2i_t tmpdata = {0};
    exr_result_t     rv;

    if (curpart->displayWindow)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute '%s' encountered",
            EXR_REQ_DISP_STR);
    }

    if (0 != strcmp (tname, "box2i"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute '%s': Invalid type '%s'",
            EXR_REQ_DISP_STR,
            tname);
    }

    rv = extract_attr_32bit (
        ctxt, scratch, &(tmpdata), EXR_REQ_DISP_STR, tname, attrsz, 4);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_DISP_STR,
        EXR_ATTR_BOX2I,
        0,
        NULL,
        &(curpart->displayWindow));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'box2i'",
            EXR_REQ_DISP_STR);

    *(curpart->displayWindow->box2i) = tmpdata;
    curpart->display_window          = tmpdata;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_lineOrder (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    uint8_t      data;
    exr_result_t rv;

    if (curpart->lineOrder)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute '%s' encountered",
            EXR_REQ_LO_STR);
    }

    if (0 != strcmp (tname, EXR_REQ_LO_STR))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute '%s': Invalid type '%s'",
            EXR_REQ_LO_STR,
            tname);
    }

    rv = extract_attr_uint8 (
        ctxt,
        scratch,
        &data,
        EXR_REQ_LO_STR,
        tname,
        attrsz,
        (uint8_t) EXR_LINEORDER_LAST_TYPE);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_LO_STR,
        EXR_ATTR_LINEORDER,
        0,
        NULL,
        &(curpart->lineOrder));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'lineOrder'",
            EXR_REQ_LO_STR);

    curpart->lineOrder->uc = data;
    curpart->lineorder     = data;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_pixelAspectRatio (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t rv;
    union
    {
        uint32_t ival;
        float    fval;
    } tpun;

    if (curpart->pixelAspectRatio)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute '%s' encountered",
            EXR_REQ_PAR_STR);
    }

    if (0 != strcmp (tname, "float"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute '%s': Invalid type '%s'",
            EXR_REQ_PAR_STR,
            tname);
    }

    if (attrsz != sizeof (float))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Required attribute '%s': Invalid size %d (exp 4)",
            EXR_REQ_PAR_STR,
            attrsz);
    }

    rv = scratch->sequential_read (scratch, &(tpun.ival), sizeof (uint32_t));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Attribute '%s': Unable to read data (%d bytes)",
            EXR_REQ_PAR_STR,
            attrsz);

    tpun.ival = one_to_native32 (tpun.ival);

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_PAR_STR,
        EXR_ATTR_FLOAT,
        0,
        NULL,
        &(curpart->pixelAspectRatio));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'float'",
            EXR_REQ_PAR_STR);

    curpart->pixelAspectRatio->f = tpun.fval;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_screenWindowCenter (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t   rv;
    exr_attr_v2f_t tmpdata;

    if (curpart->screenWindowCenter)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute '%s' encountered",
            EXR_REQ_SCR_WC_STR);
    }

    if (0 != strcmp (tname, "v2f"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute '%s': Invalid type '%s'",
            EXR_REQ_SCR_WC_STR,
            tname);
    }

    if (attrsz != sizeof (exr_attr_v2f_t))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Required attribute '%s': Invalid size %d (exp %" PRIu64 ")",
            EXR_REQ_SCR_WC_STR,
            attrsz,
            (uint64_t) sizeof (exr_attr_v2f_t));
    }

    rv = scratch->sequential_read (scratch, &tmpdata, sizeof (exr_attr_v2f_t));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Attribute '%s': Unable to read data (%d bytes)",
            EXR_REQ_SCR_WC_STR,
            attrsz);

    priv_to_native32 (&tmpdata, 2);

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_SCR_WC_STR,
        EXR_ATTR_V2F,
        0,
        NULL,
        &(curpart->screenWindowCenter));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'v2f'",
            EXR_REQ_SCR_WC_STR);

    *(curpart->screenWindowCenter->v2f) = tmpdata;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_screenWindowWidth (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t rv;
    union
    {
        uint32_t ival;
        float    fval;
    } tpun;

    if (curpart->screenWindowWidth)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute '%s' encountered",
            EXR_REQ_SCR_WW_STR);
    }

    if (0 != strcmp (tname, "float"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute '%s': Invalid type '%s'",
            EXR_REQ_SCR_WW_STR,
            tname);
    }

    if (attrsz != sizeof (float))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_SIZE_MISMATCH,
            "Required attribute '%s': Invalid size %d (exp 4)",
            EXR_REQ_SCR_WW_STR,
            attrsz);
    }

    rv = scratch->sequential_read (scratch, &(tpun.ival), sizeof (uint32_t));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Attribute '%s': Unable to read data (%d bytes)",
            EXR_REQ_SCR_WW_STR,
            attrsz);

    tpun.ival = one_to_native32 (tpun.ival);

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_SCR_WW_STR,
        EXR_ATTR_FLOAT,
        0,
        NULL,
        &(curpart->screenWindowWidth));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'float'",
            EXR_REQ_SCR_WW_STR);

    curpart->screenWindowWidth->f = tpun.fval;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_tiles (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t        rv;
    exr_attr_tiledesc_t tmpdata = {0};

    if (curpart->tiles)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute 'tiles' encountered");
    }

    if (0 != strcmp (tname, "tiledesc"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute 'tiles': Invalid type '%s'",
            tname);
    }

    if (attrsz != sizeof (exr_attr_tiledesc_t))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute 'tiles': Invalid size %d (exp %" PRIu64 ")",
            attrsz,
            (uint64_t) sizeof (exr_attr_tiledesc_t));
    }

    rv = scratch->sequential_read (scratch, &tmpdata, sizeof (tmpdata));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->report_error (ctxt, rv, "Unable to read 'tiles' data");

    tmpdata.x_size = one_to_native32 (tmpdata.x_size);
    tmpdata.y_size = one_to_native32 (tmpdata.y_size);

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_TILES_STR,
        EXR_ATTR_TILEDESC,
        0,
        NULL,
        &(curpart->tiles));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'tiledesc'",
            EXR_REQ_TILES_STR);

    *(curpart->tiles->tiledesc) = tmpdata;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_name (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t rv;
    uint8_t*     outstr;
    int32_t      n;

    rv = check_bad_attrsz (
        ctxt, scratch, attrsz, 1, EXR_REQ_NAME_STR, tname, &n);
    if (rv != EXR_ERR_SUCCESS) return rv;

    if (curpart->name)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute 'name' encountered");
    }

    if (0 != strcmp (tname, "string"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "attribute 'name': Invalid type '%s'",
            tname);
    }

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_NAME_STR,
        EXR_ATTR_STRING,
        attrsz + 1,
        &outstr,
        &(curpart->name));
    if (rv != EXR_ERR_SUCCESS)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'string'",
            EXR_REQ_NAME_STR);
    }

    rv = scratch->sequential_read (scratch, outstr, (uint64_t) attrsz);
    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_list_remove (
            (exr_context_t) ctxt, &(curpart->attributes), curpart->name);
        curpart->name = NULL;
        return ctxt->report_error (ctxt, rv, "Unable to read 'name' data");
    }
    outstr[attrsz] = '\0';

    rv = exr_attr_string_init_static_with_length (
        (exr_context_t) ctxt,
        curpart->name->string,
        (const char*) outstr,
        attrsz);
    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_list_remove (
            (exr_context_t) ctxt, &(curpart->attributes), curpart->name);
        curpart->name = NULL;
        return ctxt->report_error (ctxt, rv, "Unable to read 'name' data");
    }

    return rv;
}

/**************************************/

static exr_result_t
check_populate_type (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t rv;
    uint8_t*     outstr;
    int32_t      n;

    rv = check_bad_attrsz (
        ctxt, scratch, attrsz, 1, EXR_REQ_TYPE_STR, tname, &n);
    if (rv != EXR_ERR_SUCCESS) return rv;

    if (curpart->type)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute 'type' encountered");
    }

    if (0 != strcmp (tname, "string"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "Required attribute 'type': Invalid type '%s'",
            tname);
    }

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_TYPE_STR,
        EXR_ATTR_STRING,
        attrsz + 1,
        &outstr,
        &(curpart->type));
    if (rv != EXR_ERR_SUCCESS)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'string'",
            EXR_REQ_TYPE_STR);
    }

    rv = scratch->sequential_read (scratch, outstr, (uint64_t) attrsz);
    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_list_remove (
            (exr_context_t) ctxt, &(curpart->attributes), curpart->type);
        curpart->type = NULL;
        return ctxt->report_error (ctxt, rv, "Unable to read 'name' data");
    }
    outstr[attrsz] = '\0';

    rv = exr_attr_string_init_static_with_length (
        (exr_context_t) ctxt,
        curpart->type->string,
        (const char*) outstr,
        attrsz);
    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_list_remove (
            (exr_context_t) ctxt, &(curpart->attributes), curpart->type);
        curpart->type = NULL;
        return ctxt->report_error (ctxt, rv, "Unable to read 'name' data");
    }

    if (strcmp ((const char*) outstr, "scanlineimage") == 0)
        curpart->storage_mode = EXR_STORAGE_SCANLINE;
    else if (strcmp ((const char*) outstr, "tiledimage") == 0)
        curpart->storage_mode = EXR_STORAGE_TILED;
    else if (strcmp ((const char*) outstr, "deepscanline") == 0)
        curpart->storage_mode = EXR_STORAGE_DEEP_SCANLINE;
    else if (strcmp ((const char*) outstr, "deeptile") == 0)
        curpart->storage_mode = EXR_STORAGE_DEEP_TILED;
    else
    {
        rv = ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "attribute 'type': Invalid type string '%s'",
            outstr);
        exr_attr_list_remove (
            (exr_context_t) ctxt, &(curpart->attributes), curpart->type);
        curpart->type = NULL;
    }

    return rv;
}

/**************************************/

static exr_result_t
check_populate_version (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t rv;

    if (curpart->version)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute 'version' encountered");
    }

    if (0 != strcmp (tname, "int"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "attribute 'version': Invalid type '%s'",
            tname);
    }

    if (attrsz != sizeof (int32_t))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "attribute 'version': Invalid size %d (exp 4)",
            attrsz);
    }

    rv = scratch->sequential_read (scratch, &attrsz, sizeof (int32_t));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->report_error (ctxt, rv, "Unable to read version data");

    attrsz = (int32_t) one_to_native32 ((uint32_t) attrsz);
    if (attrsz != 1)
        return ctxt->print_error (
            ctxt, EXR_ERR_INVALID_ATTR, "Invalid version %d: expect 1", attrsz);

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_VERSION_STR,
        EXR_ATTR_INT,
        0,
        NULL,
        &(curpart->version));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'int'",
            EXR_REQ_VERSION_STR);
    curpart->version->i = attrsz;
    return rv;
}

/**************************************/

static exr_result_t
check_populate_chunk_count (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       tname,
    int32_t                           attrsz)
{
    exr_result_t rv;

    if (curpart->chunkCount)
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Duplicate copy of required attribute 'chunkCount' encountered");
    }

    if (0 != strcmp (tname, "int"))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_ATTR_TYPE_MISMATCH,
            "attribute 'chunkCount': Invalid type '%s'",
            tname);
    }

    if (attrsz != sizeof (int32_t))
    {
        scratch->sequential_skip (scratch, attrsz);
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ATTR,
            "Required attribute 'chunkCount': Invalid size %d (exp 4)",
            attrsz);
    }

    rv = scratch->sequential_read (scratch, &attrsz, sizeof (int32_t));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->report_error (ctxt, rv, "Unable to read chunkCount data");

    rv = exr_attr_list_add_static_name (
        (exr_context_t) ctxt,
        &(curpart->attributes),
        EXR_REQ_CHUNK_COUNT_STR,
        EXR_ATTR_INT,
        0,
        NULL,
        &(curpart->chunkCount));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type 'int'",
            EXR_REQ_CHUNK_COUNT_STR);

    attrsz                 = (int32_t) one_to_native32 ((uint32_t) attrsz);
    curpart->chunkCount->i = attrsz;
    curpart->chunk_count   = attrsz;
    return rv;
}

/**************************************/

static exr_result_t
check_req_attr (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    struct _internal_exr_seq_scratch* scratch,
    const char*                       aname,
    const char*                       tname,
    int32_t                           attrsz)
{
    switch (aname[0])
    {
        case 'c':
            if (0 == strcmp (aname, EXR_REQ_CHANNELS_STR))
                return check_populate_channels (
                    ctxt, curpart, scratch, tname, attrsz);
            if (0 == strcmp (aname, EXR_REQ_COMP_STR))
                return check_populate_compression (
                    ctxt, curpart, scratch, tname, attrsz);
            if (0 == strcmp (aname, EXR_REQ_CHUNK_COUNT_STR))
                return check_populate_chunk_count (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        case 'd':
            if (0 == strcmp (aname, EXR_REQ_DATA_STR))
                return check_populate_dataWindow (
                    ctxt, curpart, scratch, tname, attrsz);
            if (0 == strcmp (aname, EXR_REQ_DISP_STR))
                return check_populate_displayWindow (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        case 'l':
            if (0 == strcmp (aname, EXR_REQ_LO_STR))
                return check_populate_lineOrder (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        case 'n':
            if (0 == strcmp (aname, EXR_REQ_NAME_STR))
                return check_populate_name (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        case 'p':
            if (0 == strcmp (aname, EXR_REQ_PAR_STR))
                return check_populate_pixelAspectRatio (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        case 's':
            if (0 == strcmp (aname, EXR_REQ_SCR_WC_STR))
                return check_populate_screenWindowCenter (
                    ctxt, curpart, scratch, tname, attrsz);
            if (0 == strcmp (aname, EXR_REQ_SCR_WW_STR))
                return check_populate_screenWindowWidth (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        case 't':
            if (0 == strcmp (aname, EXR_REQ_TILES_STR))
                return check_populate_tiles (
                    ctxt, curpart, scratch, tname, attrsz);
            if (0 == strcmp (aname, EXR_REQ_TYPE_STR))
                return check_populate_type (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        case 'v':
            if (0 == strcmp (aname, EXR_REQ_VERSION_STR))
                return check_populate_version (
                    ctxt, curpart, scratch, tname, attrsz);
            break;
        default: break;
    }

    return EXR_ERR_UNKNOWN;
}

/**************************************/

static exr_result_t
pull_attr (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_part*        curpart,
    uint8_t                           init_byte,
    struct _internal_exr_seq_scratch* scratch)
{
    char             name[256], type[256];
    exr_result_t     rv;
    int32_t          namelen = 0, typelen = 0;
    int32_t          attrsz = 0;
    exr_attribute_t* nattr  = NULL;
    uint8_t*         strptr = NULL;
    const int32_t    maxlen = ctxt->max_name_length;

    name[0] = (char) init_byte;
    namelen = 1;

    rv = read_text (ctxt, name, &namelen, maxlen, scratch, "attribute name");
    if (rv != EXR_ERR_SUCCESS) return rv;
    rv = read_text (ctxt, type, &typelen, maxlen, scratch, "attribute type");
    if (rv != EXR_ERR_SUCCESS) return rv;

    if (namelen == 0)
        return ctxt->report_error (
            ctxt,
            EXR_ERR_FILE_BAD_HEADER,
            "Invalid empty string encountered parsing attribute name");

    if (typelen == 0)
        return ctxt->print_error (
            ctxt,
            EXR_ERR_FILE_BAD_HEADER,
            "Invalid empty string encountered parsing attribute type for attribute '%s'",
            name);

    rv = scratch->sequential_read (scratch, &attrsz, sizeof (int32_t));
    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to read attribute size for attribute '%s', type '%s'",
            name,
            type);
    attrsz = (int32_t) one_to_native32 ((uint32_t) attrsz);

    rv = check_req_attr (ctxt, curpart, scratch, name, type, attrsz);
    if (rv != EXR_ERR_UNKNOWN) return rv;

    /* not a required attr, just a normal one, optimize for string type to avoid double malloc */
    if (!strcmp (type, "string"))
    {
        int32_t n;
        rv = check_bad_attrsz (ctxt, scratch, attrsz, 1, name, type, &n);
        if (rv != EXR_ERR_SUCCESS) return rv;

        rv = exr_attr_list_add (
            (exr_context_t) ctxt,
            &(curpart->attributes),
            name,
            EXR_ATTR_STRING,
            n + 1,
            &strptr,
            &nattr);
    }
    else
    {
        rv = exr_attr_list_add_by_type (
            (exr_context_t) ctxt,
            &(curpart->attributes),
            name,
            type,
            0,
            NULL,
            &nattr);
    }

    if (rv != EXR_ERR_SUCCESS)
        return ctxt->print_error (
            ctxt,
            rv,
            "Unable to initialize attribute '%s', type '%s'",
            name,
            type);

    switch (nattr->type)
    {
        case EXR_ATTR_BOX2I:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->box2i, name, type, attrsz, 4);
            break;
        case EXR_ATTR_BOX2F:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->box2f, name, type, attrsz, 4);
            break;
        case EXR_ATTR_CHLIST:
            rv = extract_attr_chlist (
                ctxt, scratch, nattr->chlist, name, type, attrsz);
            break;
        case EXR_ATTR_CHROMATICITIES:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->chromaticities, name, type, attrsz, 8);
            break;
        case EXR_ATTR_COMPRESSION:
            rv = extract_attr_uint8 (
                ctxt,
                scratch,
                &(nattr->uc),
                name,
                type,
                attrsz,
                (uint8_t) EXR_COMPRESSION_LAST_TYPE);
            break;
        case EXR_ATTR_ENVMAP:
            rv = extract_attr_uint8 (
                ctxt,
                scratch,
                &(nattr->uc),
                name,
                type,
                attrsz,
                (uint8_t) EXR_ENVMAP_LAST_TYPE);
            break;
        case EXR_ATTR_LINEORDER:
            rv = extract_attr_uint8 (
                ctxt,
                scratch,
                &(nattr->uc),
                name,
                type,
                attrsz,
                (uint8_t) EXR_LINEORDER_LAST_TYPE);
            break;
        case EXR_ATTR_DOUBLE:
            rv = extract_attr_64bit (
                ctxt, scratch, &(nattr->d), name, type, attrsz, 1);
            break;
        case EXR_ATTR_FLOAT:
            rv = extract_attr_32bit (
                ctxt, scratch, &(nattr->f), name, type, attrsz, 1);
            break;
        case EXR_ATTR_FLOAT_VECTOR:
            rv = extract_attr_float_vector (
                ctxt, scratch, nattr->floatvector, name, type, attrsz);
            break;
        case EXR_ATTR_INT:
            rv = extract_attr_32bit (
                ctxt, scratch, &(nattr->i), name, type, attrsz, 1);
            break;
        case EXR_ATTR_KEYCODE:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->keycode, name, type, attrsz, 7);
            break;
        case EXR_ATTR_M33F:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->m33f->m, name, type, attrsz, 9);
            break;
        case EXR_ATTR_M33D:
            rv = extract_attr_64bit (
                ctxt, scratch, nattr->m33d->m, name, type, attrsz, 9);
            break;
        case EXR_ATTR_M44F:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->m44f->m, name, type, attrsz, 16);
            break;
        case EXR_ATTR_M44D:
            rv = extract_attr_64bit (
                ctxt, scratch, nattr->m44d->m, name, type, attrsz, 16);
            break;
        case EXR_ATTR_PREVIEW:
            rv = extract_attr_preview (
                ctxt, scratch, nattr->preview, name, type, attrsz);
            break;
        case EXR_ATTR_RATIONAL:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->rational, name, type, attrsz, 2);
            break;
        case EXR_ATTR_STRING:
            rv = extract_attr_string (
                ctxt,
                scratch,
                nattr->string,
                name,
                type,
                attrsz,
                (char*) strptr);
            break;
        case EXR_ATTR_STRING_VECTOR:
            rv = extract_attr_string_vector (
                ctxt, scratch, nattr->stringvector, name, type, attrsz);
            break;
        case EXR_ATTR_TILEDESC:
            rv = extract_attr_tiledesc (
                ctxt, scratch, nattr->tiledesc, name, type, attrsz);
            break;
        case EXR_ATTR_TIMECODE:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->timecode, name, type, attrsz, 2);
            break;
        case EXR_ATTR_V2I:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->v2i->arr, name, type, attrsz, 2);
            break;
        case EXR_ATTR_V2F:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->v2f->arr, name, type, attrsz, 2);
            break;
        case EXR_ATTR_V2D:
            rv = extract_attr_64bit (
                ctxt, scratch, nattr->v2d->arr, name, type, attrsz, 2);
            break;
        case EXR_ATTR_V3I:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->v3i->arr, name, type, attrsz, 3);
            break;
        case EXR_ATTR_V3F:
            rv = extract_attr_32bit (
                ctxt, scratch, nattr->v3f->arr, name, type, attrsz, 3);
            break;
        case EXR_ATTR_V3D:
            rv = extract_attr_64bit (
                ctxt, scratch, nattr->v3d->arr, name, type, attrsz, 3);
            break;
        case EXR_ATTR_OPAQUE:
            rv = extract_attr_opaque (
                ctxt, scratch, nattr->opaque, name, type, attrsz);
            break;
        case EXR_ATTR_UNKNOWN:
        case EXR_ATTR_LAST_KNOWN_TYPE:
        default:
            rv = ctxt->print_error (
                ctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Invalid type '%s' for attribute '%s'",
                type,
                name);
            break;
    }
    if (rv != EXR_ERR_SUCCESS)
    {
        exr_attr_list_remove (
            (exr_context_t) ctxt, &(curpart->attributes), nattr);
    }

    return rv;
}

/**************************************/

/*  floor( log(x) / log(2) ) */
static int32_t
floor_log2 (int64_t x)
{
    int32_t y = 0;
    while (x > 1)
    {
        y += 1;
        x >>= 1;
    }
    return y;
}

/**************************************/

/*  ceil( log(x) / log(2) ) */
static int32_t
ceil_log2 (int64_t x)
{
    int32_t y = 0, r = 0;
    while (x > 1)
    {
        if (x & 1) r = 1;
        y += 1;
        x >>= 1;
    }
    return y + r;
}

/**************************************/

static int64_t
calc_level_size (
    int64_t mind, int64_t maxd, int level, exr_tile_round_mode_t rounding)
{
    int64_t dsize   = (int64_t) maxd - (int64_t) mind + 1;
    int64_t b       = ((int64_t) 1) << level;
    int64_t retsize = dsize / b;

    if (rounding == EXR_TILE_ROUND_UP && retsize * b < dsize) retsize += 1;

    if (retsize < 1) retsize = 1;
    return retsize;
}

/**************************************/

exr_result_t
internal_exr_compute_tile_information (
    struct _internal_exr_context* ctxt,
    struct _internal_exr_part*    curpart,
    int                           rebuild)
{
    exr_result_t rv = EXR_ERR_SUCCESS;
    if (curpart->storage_mode == EXR_STORAGE_SCANLINE ||
        curpart->storage_mode == EXR_STORAGE_DEEP_SCANLINE)
        return EXR_ERR_SUCCESS;

    if (rebuild && (!curpart->dataWindow || !curpart->tiles))
        return EXR_ERR_SUCCESS;

    if (!curpart->tiles)
        return ctxt->standard_error (ctxt, EXR_ERR_MISSING_REQ_ATTR);

    if (rebuild)
    {
        if (curpart->tile_level_tile_count_x)
        {
            ctxt->free_fn (curpart->tile_level_tile_count_x);
            curpart->tile_level_tile_count_x = NULL;
        }
    }

    if (curpart->tile_level_tile_count_x == NULL)
    {
        const exr_attr_box2i_t     dw       = curpart->data_window;
        const exr_attr_tiledesc_t* tiledesc = curpart->tiles->tiledesc;
        int64_t                    w, h;
        int32_t                    numX, numY;
        int32_t*                   levcntX = NULL;
        int32_t*                   levcntY = NULL;
        int32_t*                   levszX  = NULL;
        int32_t*                   levszY  = NULL;

        w = ((int64_t) dw.max.x) - ((int64_t) dw.min.x) + 1;
        h = ((int64_t) dw.max.y) - ((int64_t) dw.min.y) + 1;

        if (tiledesc->x_size == 0 || tiledesc->y_size == 0)
            return ctxt->standard_error (ctxt, EXR_ERR_INVALID_ATTR);
        switch (EXR_GET_TILE_LEVEL_MODE ((*tiledesc)))
        {
            case EXR_TILE_ONE_LEVEL: numX = numY = 1; break;
            case EXR_TILE_MIPMAP_LEVELS:
                if (EXR_GET_TILE_ROUND_MODE ((*tiledesc)) ==
                    EXR_TILE_ROUND_DOWN)
                {
                    numX = floor_log2 (w > h ? w : h) + 1;
                    numY = numX;
                }
                else
                {
                    numX = ceil_log2 (w > h ? w : h) + 1;
                    numY = numX;
                }
                break;
            case EXR_TILE_RIPMAP_LEVELS:
                if (EXR_GET_TILE_ROUND_MODE ((*tiledesc)) ==
                    EXR_TILE_ROUND_DOWN)
                {
                    numX = floor_log2 (w) + 1;
                    numY = floor_log2 (h) + 1;
                }
                else
                {
                    numX = ceil_log2 (w) + 1;
                    numY = ceil_log2 (h) + 1;
                }
                break;
            case EXR_TILE_LAST_TYPE:
            default: return ctxt->standard_error (ctxt, EXR_ERR_INVALID_ATTR);
        }

        curpart->num_tile_levels_x = numX;
        curpart->num_tile_levels_y = numY;
        levcntX                    = (int32_t*) ctxt->alloc_fn (
            2 * (size_t) (numX + numY) * sizeof (int32_t));
        if (levcntX == NULL)
            return ctxt->standard_error (ctxt, EXR_ERR_OUT_OF_MEMORY);
        levszX  = levcntX + numX;
        levcntY = levszX + numX;
        levszY  = levcntY + numY;

        for (int32_t l = 0; l < numX; ++l)
        {
            int64_t sx = calc_level_size (
                dw.min.x, dw.max.x, l, EXR_GET_TILE_ROUND_MODE ((*tiledesc)));
            if (sx < 0 || sx > (int64_t) INT32_MAX)
                return ctxt->print_error (
                    ctxt,
                    EXR_ERR_INVALID_ATTR,
                    "Invalid data window x dims (%d, %d) resulting in invalid tile level size (%" PRId64
                    ") for level %d",
                    dw.min.x,
                    dw.max.x,
                    sx,
                    l);
            levcntX[l] =
                (int32_t) (((uint64_t) sx + tiledesc->x_size - 1) / tiledesc->x_size);
            levszX[l] = (int32_t) sx;
        }

        for (int32_t l = 0; l < numY; ++l)
        {
            int64_t sy = calc_level_size (
                dw.min.y, dw.max.y, l, EXR_GET_TILE_ROUND_MODE ((*tiledesc)));
            if (sy < 0 || sy > (int64_t) INT32_MAX)
                return ctxt->print_error (
                    ctxt,
                    EXR_ERR_INVALID_ATTR,
                    "Invalid data window y dims (%d, %d) resulting in invalid tile level size (%" PRId64
                    ") for level %d",
                    dw.min.y,
                    dw.max.y,
                    sy,
                    l);
            levcntY[l] =
                (int32_t) (((uint64_t) sy + tiledesc->y_size - 1) / tiledesc->y_size);
            levszY[l] = (int32_t) sy;
        }

        curpart->tile_level_tile_count_x = levcntX;
        curpart->tile_level_tile_count_y = levcntY;
        curpart->tile_level_tile_size_x  = levszX;
        curpart->tile_level_tile_size_y  = levszY;
    }
    return rv;
}

/**************************************/

int32_t
internal_exr_compute_chunk_offset_size (struct _internal_exr_part* curpart)
{
    int32_t                  retval       = 0;
    const exr_attr_box2i_t   dw           = curpart->data_window;
    const exr_attr_chlist_t* channels     = curpart->channels->chlist;
    uint64_t                 unpackedsize = 0;
    uint64_t                 w;
    int                      hasLineSample = 0;

    w = (uint64_t) (((int64_t) dw.max.x) - ((int64_t) dw.min.x) + 1);

    if (curpart->tiles)
    {
        const exr_attr_tiledesc_t* tiledesc  = curpart->tiles->tiledesc;
        int64_t                    tilecount = 0;

        switch (EXR_GET_TILE_LEVEL_MODE ((*tiledesc)))
        {
            case EXR_TILE_ONE_LEVEL:
            case EXR_TILE_MIPMAP_LEVELS:
                for (int32_t l = 0; l < curpart->num_tile_levels_x; ++l)
                    tilecount +=
                        ((int64_t) curpart->tile_level_tile_count_x[l] *
                         (int64_t) curpart->tile_level_tile_count_y[l]);
                if (tilecount > (int64_t) INT_MAX) return -1;
                retval = (int32_t) tilecount;
                break;
            case EXR_TILE_RIPMAP_LEVELS:
                for (int32_t lx = 0; lx < curpart->num_tile_levels_x; ++lx)
                {
                    for (int32_t ly = 0; ly < curpart->num_tile_levels_y; ++ly)
                    {
                        tilecount +=
                            ((int64_t) curpart->tile_level_tile_count_x[lx] *
                             (int64_t) curpart->tile_level_tile_count_y[ly]);

                        if (tilecount > (int64_t) INT_MAX) return -1;
                    }
                }
                retval = (int32_t) tilecount;
                break;
            case EXR_TILE_LAST_TYPE:
            default: return -1;
        }

        for (int c = 0; c < channels->num_channels; ++c)
        {
            uint64_t xsamp  = (uint64_t) channels->entries[c].x_sampling;
            uint64_t ysamp  = (uint64_t) channels->entries[c].y_sampling;
            uint64_t cunpsz = 0;
            if (channels->entries[c].pixel_type == EXR_PIXEL_HALF)
                cunpsz = 2;
            else
                cunpsz = 4;
            cunpsz *= (((uint64_t) tiledesc->x_size + xsamp - 1) / xsamp);
            if (ysamp > 1)
            {
                hasLineSample = 1;
                cunpsz *= (((uint64_t) tiledesc->y_size + ysamp - 1) / ysamp);
            }
            else
                cunpsz *= (uint64_t) tiledesc->y_size;
            unpackedsize += cunpsz;
        }
        curpart->unpacked_size_per_chunk = unpackedsize;
        curpart->chan_has_line_sampling  = ((int16_t) hasLineSample);
    }
    else
    {
        uint64_t linePerChunk, h;
        switch (curpart->comp_type)
        {
            case EXR_COMPRESSION_NONE:
            case EXR_COMPRESSION_RLE:
            case EXR_COMPRESSION_ZIPS: linePerChunk = 1; break;
            case EXR_COMPRESSION_ZIP:
            case EXR_COMPRESSION_PXR24: linePerChunk = 16; break;
            case EXR_COMPRESSION_PIZ:
            case EXR_COMPRESSION_B44:
            case EXR_COMPRESSION_B44A:
            case EXR_COMPRESSION_DWAA: linePerChunk = 32; break;
            case EXR_COMPRESSION_DWAB: linePerChunk = 256; break;
            case EXR_COMPRESSION_LAST_TYPE:
            default:
                /* ERROR CONDITION */
                return -1;
        }

        for (int c = 0; c < channels->num_channels; ++c)
        {
            uint64_t xsamp  = (uint64_t) channels->entries[c].x_sampling;
            uint64_t ysamp  = (uint64_t) channels->entries[c].y_sampling;
            uint64_t cunpsz = 0;
            if (channels->entries[c].pixel_type == EXR_PIXEL_HALF)
                cunpsz = 2;
            else
                cunpsz = 4;
            cunpsz *= w / xsamp;
            cunpsz *= linePerChunk;
            if (ysamp > 1)
            {
                hasLineSample = 1;
                if (linePerChunk > 1) cunpsz *= linePerChunk / ysamp;
            }
            unpackedsize += cunpsz;
        }

        curpart->unpacked_size_per_chunk = unpackedsize;
        curpart->lines_per_chunk         = ((int16_t) linePerChunk);
        curpart->chan_has_line_sampling  = ((int16_t) hasLineSample);

        h      = (uint64_t) ((int64_t) dw.max.y - (int64_t) dw.min.y + 1);
        retval = (int32_t) ((h + linePerChunk - 1) / linePerChunk);
    }
    return retval;
}

/**************************************/

static exr_result_t
update_chunk_offsets (
    struct _internal_exr_context*     ctxt,
    struct _internal_exr_seq_scratch* scratch)
{
    struct _internal_exr_part *curpart, *prevpart;

    exr_result_t rv = EXR_ERR_SUCCESS;

    if (!ctxt->parts) return EXR_ERR_INVALID_ARGUMENT;

    ctxt->parts[0]->chunk_table_offset =
        scratch->fileoff - (uint64_t) scratch->navail;
    prevpart = ctxt->parts[0];

    for (int p = 0; p < ctxt->num_parts; ++p)
    {
        int32_t ccount;

        curpart = ctxt->parts[p];

        rv = internal_exr_compute_tile_information (ctxt, curpart, 0);
        if (rv != EXR_ERR_SUCCESS) break;

        ccount = internal_exr_compute_chunk_offset_size (curpart);
        if (ccount < 0)
        {
            rv = ctxt->print_error (
                ctxt,
                EXR_ERR_INVALID_ATTR,
                "Invalid chunk count (%d) for part '%s'",
                ccount,
                (curpart->name ? curpart->name->string->str : "<first>"));
            break;
        }

        if (curpart->chunk_count < 0)
            curpart->chunk_count = ccount;
        else if (curpart->chunk_count != ccount)
        {
            /* fatal error or just ignore it? c++ seemed to just ignore it entirely, we can at least warn */
            /* rv = */
            ctxt->print_error (
                ctxt,
                EXR_ERR_INVALID_ATTR,
                "Invalid chunk count (%d) for part '%s', expect (%d)",
                curpart->chunk_count,
                (curpart->name ? curpart->name->string->str : "<first>"),
                ccount);
            curpart->chunk_count = ccount;
        }
        if (prevpart != curpart)
            curpart->chunk_table_offset =
                prevpart->chunk_table_offset +
                sizeof (uint64_t) * (size_t) (prevpart->chunk_count);
        prevpart = curpart;
    }
    return rv;
}

/**************************************/

static exr_result_t
read_magic_and_flags (
    struct _internal_exr_context* ctxt, uint32_t* outflags, uint64_t* initpos)
{
    uint32_t     magic_and_version[2];
    uint32_t     flags;
    exr_result_t rv      = EXR_ERR_UNKNOWN;
    uint64_t     fileoff = 0;
    int64_t      nread   = 0;

    rv = ctxt->do_read (
        ctxt,
        magic_and_version,
        sizeof (uint32_t) * 2,
        &fileoff,
        &nread,
        EXR_MUST_READ_ALL);
    if (rv != EXR_ERR_SUCCESS)
    {
        ctxt->report_error (
            ctxt, EXR_ERR_READ_IO, "Unable to read magic and version flags");
        return rv;
    }

    *initpos = sizeof (uint32_t) * 2;

    priv_to_native32 (magic_and_version, 2);
    if (magic_and_version[0] != 20000630)
    {
        rv = ctxt->print_error (
            ctxt,
            EXR_ERR_FILE_BAD_HEADER,
            "File is not an OpenEXR file: magic 0x%08X (%d) flags 0x%08X",
            magic_and_version[0],
            (int) magic_and_version[0],
            magic_and_version[1]);
        return rv;
    }

    flags = magic_and_version[1];

    ctxt->version = flags & EXR_FILE_VERSION_MASK;
    if (ctxt->version != 2)
    {
        rv = ctxt->print_error (
            ctxt,
            EXR_ERR_FILE_BAD_HEADER,
            "File is of an unsupported version: %d, magic 0x%08X flags 0x%08X",
            (int) ctxt->version,
            magic_and_version[0],
            magic_and_version[1]);
        return rv;
    }

    flags = flags & ~((uint32_t) EXR_FILE_VERSION_MASK);
    if ((flags & ~((uint32_t) EXR_VALID_FLAGS)) != 0)
    {
        rv = ctxt->print_error (
            ctxt,
            EXR_ERR_FILE_BAD_HEADER,
            "File has an unsupported flags: magic 0x%08X flags 0x%08X",
            magic_and_version[0],
            magic_and_version[1]);
        return rv;
    }
    *outflags = flags;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
internal_exr_check_magic (struct _internal_exr_context* ctxt)
{
    uint32_t     flags;
    uint64_t     initpos;
    exr_result_t rv = EXR_ERR_UNKNOWN;

    rv = read_magic_and_flags (ctxt, &flags, &initpos);
    return rv;
}

/**************************************/

exr_result_t
internal_exr_parse_header (struct _internal_exr_context* ctxt)
{
    struct _internal_exr_seq_scratch scratch;
    struct _internal_exr_part*       curpart;
    uint32_t                         flags;
    uint64_t                         initpos;
    uint8_t                          next_byte;
    exr_result_t                     rv = EXR_ERR_UNKNOWN;

    if (ctxt->silent_header)
    {
        ctxt->standard_error = &silent_standard_error;
        ctxt->report_error   = &silent_error;
        ctxt->print_error    = &silent_print_error;
    }
    rv = read_magic_and_flags (ctxt, &flags, &initpos);
    if (rv != EXR_ERR_SUCCESS)
        return internal_exr_context_restore_handlers (ctxt, rv);

    rv = priv_init_scratch (ctxt, &scratch, initpos);
    if (rv != EXR_ERR_SUCCESS)
    {
        priv_destroy_scratch (&scratch);
        return internal_exr_context_restore_handlers (ctxt, rv);
    }

    curpart = ctxt->parts[0];
    if (!curpart)
    {
        rv = ctxt->report_error (
            ctxt, EXR_ERR_INVALID_ARGUMENT, "Error during file initialization");
        priv_destroy_scratch (&scratch);
        return internal_exr_context_restore_handlers (ctxt, rv);
    }

    ctxt->is_singlepart_tiled = (flags & EXR_TILED_FLAG) ? 1 : 0;
    if (ctxt->strict_header)
    {
        ctxt->max_name_length = (flags & EXR_LONG_NAMES_FLAG)
                                    ? EXR_LONGNAME_MAXLEN
                                    : EXR_SHORTNAME_MAXLEN;
    }
    else { ctxt->max_name_length = EXR_LONGNAME_MAXLEN; }
    ctxt->has_nonimage_data = (flags & EXR_NON_IMAGE_FLAG) ? 1 : 0;
    ctxt->is_multipart      = (flags & EXR_MULTI_PART_FLAG) ? 1 : 0;
    if (ctxt->is_singlepart_tiled)
    {
        if (ctxt->has_nonimage_data || ctxt->is_multipart)
        {
            if (ctxt->strict_header)
            {
                rv = ctxt->print_error (
                    ctxt,
                    EXR_ERR_FILE_BAD_HEADER,
                    "Invalid combination of version flags: single part found, but also marked as deep (%d) or multipart (%d)",
                    (int) ctxt->has_nonimage_data,
                    (int) ctxt->is_multipart);
                priv_destroy_scratch (&scratch);
                return internal_exr_context_restore_handlers (ctxt, rv);
            }
            else
            {
                // assume multipart for now
                ctxt->is_singlepart_tiled = 0;
            }
        }
        curpart->storage_mode = EXR_STORAGE_TILED;
    }
    else
        curpart->storage_mode = EXR_STORAGE_SCANLINE;

    do
    {
        rv = scratch.sequential_read (&scratch, &next_byte, 1);
        if (rv != EXR_ERR_SUCCESS)
        {
            rv = ctxt->report_error (
                ctxt, EXR_ERR_FILE_BAD_HEADER, "Unable to extract header byte");
            priv_destroy_scratch (&scratch);
            return internal_exr_context_restore_handlers (ctxt, rv);
        }

        if (next_byte == '\0')
        {
            rv = internal_exr_validate_read_part (ctxt, curpart);
            if (rv != EXR_ERR_SUCCESS)
            {
                priv_destroy_scratch (&scratch);
                return internal_exr_context_restore_handlers (ctxt, rv);
            }

            if (!ctxt->is_multipart)
            {
                /* got a terminal mark, not multipart, so finished */
                break;
            }

            rv = scratch.sequential_read (&scratch, &next_byte, 1);
            if (rv != EXR_ERR_SUCCESS)
            {
                rv = ctxt->report_error (
                    ctxt,
                    EXR_ERR_FILE_BAD_HEADER,
                    "Unable to go to next part definition");
                priv_destroy_scratch (&scratch);
                return internal_exr_context_restore_handlers (ctxt, rv);
            }

            if (next_byte == '\0')
            {
                /* got a second terminator, finished with the
                 * headers, can read chunk offsets next */
                break;
            }

            rv = internal_exr_add_part (ctxt, &curpart, NULL);
        }

        if (rv == EXR_ERR_SUCCESS)
            rv = pull_attr (ctxt, curpart, next_byte, &scratch);
        if (rv != EXR_ERR_SUCCESS)
        {
            if (ctxt->strict_header) { break; }
            rv = EXR_ERR_SUCCESS;
        }
    } while (1);

    if (rv == EXR_ERR_SUCCESS) { rv = update_chunk_offsets (ctxt, &scratch); }

    priv_destroy_scratch (&scratch);
    return internal_exr_context_restore_handlers (ctxt, rv);
}
