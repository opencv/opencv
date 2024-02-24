/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_attr.h"

#include "internal_structs.h"

#include <string.h>

/**************************************/

exr_result_t
exr_attr_string_vector_init (
    exr_context_t ctxt, exr_attr_string_vector_t* sv, int32_t nent)
{
    exr_attr_string_vector_t nil   = {0};
    exr_attr_string_t        nils  = {0};
    size_t                   bytes = (size_t) nent * sizeof (exr_attr_string_t);
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (!sv)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid reference to string vector object to assign to");

    if (nent < 0)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Received request to allocate negative sized string vector (%d entries)",
            nent);
    if (bytes > (size_t) INT32_MAX)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid too large size for string vector (%d entries)",
            nent);

    *sv = nil;
    if (bytes > 0)
    {
        sv->strings = (exr_attr_string_t*) pctxt->alloc_fn (bytes);
        if (sv->strings == NULL)
            return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);
        sv->n_strings  = nent;
        sv->alloc_size = nent;
        for (int32_t i = 0; i < nent; ++i)
            *(EXR_CONST_CAST (exr_attr_string_t*, (sv->strings + i))) = nils;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_string_vector_destroy (
    exr_context_t ctxt, exr_attr_string_vector_t* sv)
{
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (sv)
    {
        exr_attr_string_vector_t nil = {0};
        if (sv->alloc_size > 0)
        {
            exr_attr_string_t* strs =
                EXR_CONST_CAST (exr_attr_string_t*, sv->strings);
            for (int32_t i = 0; i < sv->n_strings; ++i)
                exr_attr_string_destroy (ctxt, strs + i);
            if (strs) pctxt->free_fn (strs);
        }
        *sv = nil;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_string_vector_copy (
    exr_context_t                   ctxt,
    exr_attr_string_vector_t*       sv,
    const exr_attr_string_vector_t* src)
{
    exr_result_t rv;

    if (!src) return EXR_ERR_INVALID_ARGUMENT;
    rv = exr_attr_string_vector_init (ctxt, sv, src->n_strings);
    for (int i = 0; rv == EXR_ERR_SUCCESS && i < src->n_strings; ++i)
    {
        rv = exr_attr_string_set_with_length (
            ctxt,
            EXR_CONST_CAST (exr_attr_string_t*, sv->strings + i),
            src->strings[i].str,
            src->strings[i].length);
    }
    if (rv != EXR_ERR_SUCCESS) exr_attr_string_vector_destroy (ctxt, sv);
    return rv;
}

/**************************************/

exr_result_t
exr_attr_string_vector_init_entry (
    exr_context_t ctxt, exr_attr_string_vector_t* sv, int32_t idx, int32_t len)
{
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (sv)
    {
        if (idx < 0 || idx >= sv->n_strings)
            return pctxt->print_error (
                pctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Invalid index (%d of %d) initializing string vector",
                idx,
                sv->n_strings);

        return exr_attr_string_init (
            ctxt, EXR_CONST_CAST (exr_attr_string_t*, sv->strings + idx), len);
    }

    return pctxt->print_error (
        pctxt,
        EXR_ERR_INVALID_ARGUMENT,
        "Invalid reference to string vector object to initialize index %d",
        idx);
}

/**************************************/

exr_result_t
exr_attr_string_vector_set_entry_with_length (
    exr_context_t             ctxt,
    exr_attr_string_vector_t* sv,
    int32_t                   idx,
    const char*               s,
    int32_t                   len)
{
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (!sv)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid reference to string vector object to assign to");

    if (idx < 0 || idx >= sv->n_strings)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid index (%d of %d) assigning string vector ('%s', len %d)",
            idx,
            sv->n_strings,
            s ? s : "<nil>",
            len);

    return exr_attr_string_set_with_length (
        ctxt, EXR_CONST_CAST (exr_attr_string_t*, sv->strings + idx), s, len);
}

/**************************************/

exr_result_t
exr_attr_string_vector_set_entry (
    exr_context_t             ctxt,
    exr_attr_string_vector_t* sv,
    int32_t                   idx,
    const char*               s)
{
    int32_t len = 0;
    if (s) len = (int32_t) strlen (s);
    return exr_attr_string_vector_set_entry_with_length (ctxt, sv, idx, s, len);
}

/**************************************/

exr_result_t
exr_attr_string_vector_add_entry_with_length (
    exr_context_t             ctxt,
    exr_attr_string_vector_t* sv,
    const char*               s,
    int32_t                   len)
{
    int32_t            nent;
    int                rv;
    exr_attr_string_t* nlist;
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (!sv)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid reference to string vector object to assign to");

    nent = sv->n_strings + 1;
    if (nent > sv->alloc_size)
    {
        size_t  bytes;
        int32_t allsz = sv->alloc_size * 2;

        if (sv->alloc_size >= (INT32_MAX / (int) sizeof (exr_attr_string_t)))
            return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);

        if (nent > allsz) allsz = nent + 1;
        bytes = ((size_t) allsz) * sizeof (exr_attr_string_t);
        nlist = (exr_attr_string_t*) pctxt->alloc_fn (bytes);
        if (nlist == NULL)
            return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);

        for (int32_t i = 0; i < sv->n_strings; ++i)
            *(nlist + i) = sv->strings[i];

        if (sv->alloc_size > 0)
            pctxt->free_fn (EXR_CONST_CAST (void*, sv->strings));
        sv->strings    = nlist;
        sv->alloc_size = allsz;
    }
    else
    {
        /* that means we own this and can write into, cast away const */
        nlist = EXR_CONST_CAST (exr_attr_string_t*, sv->strings);
    }

    rv = exr_attr_string_create_with_length (
        ctxt, nlist + sv->n_strings, s, len);
    if (rv == EXR_ERR_SUCCESS) sv->n_strings = nent;
    return rv;
}

/**************************************/

exr_result_t
exr_attr_string_vector_add_entry (
    exr_context_t ctxt, exr_attr_string_vector_t* sv, const char* s)
{
    int32_t len = 0;
    if (s) len = (int32_t) strlen (s);
    return exr_attr_string_vector_add_entry_with_length (ctxt, sv, s, len);
}
