/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_attr.h"

#include "internal_structs.h"

#include <string.h>

/**************************************/

exr_result_t
exr_attr_preview_init (
    exr_context_t ctxt, exr_attr_preview_t* p, uint32_t w, uint32_t h)
{
    exr_attr_preview_t nil   = {0};
    uint64_t           bytes = (uint64_t) w * (uint64_t) h * (uint64_t) 4;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (bytes > (size_t) INT32_MAX)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid very large size for preview image (%u x %u - %" PRIu64
            " bytes)",
            w,
            h,
            (uint64_t) bytes);

    if (!p)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid reference to preview object to initialize");

    *p = nil;
    if (bytes > 0)
    {
        p->rgba = (uint8_t*) pctxt->alloc_fn (bytes);
        if (p->rgba == NULL)
            return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);
        p->alloc_size = bytes;
        p->width      = w;
        p->height     = h;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_preview_create (
    exr_context_t       ctxt,
    exr_attr_preview_t* p,
    uint32_t            w,
    uint32_t            h,
    const uint8_t*      d)
{
    exr_result_t rv = exr_attr_preview_init (ctxt, p, w, h);
    if (rv == EXR_ERR_SUCCESS)
    {
        size_t copybytes = w * h * 4;
        if (copybytes > 0)
            memcpy (EXR_CONST_CAST (uint8_t*, p->rgba), d, copybytes);
    }
    return rv;
}

/**************************************/

exr_result_t
exr_attr_preview_destroy (exr_context_t ctxt, exr_attr_preview_t* p)
{
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (p)
    {
        exr_attr_preview_t nil = {0};
        if (p->rgba && p->alloc_size > 0)
            pctxt->free_fn (EXR_CONST_CAST (uint8_t*, p->rgba));
        *p = nil;
    }
    return EXR_ERR_SUCCESS;
}
