/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_attr.h"

#include "internal_structs.h"

#include <string.h>

/**************************************/

/* allocates ram, but does not fill any data */
exr_result_t
exr_attr_float_vector_init (
    exr_context_t ctxt, exr_attr_float_vector_t* fv, int32_t nent)
{
    exr_attr_float_vector_t nil   = {0};
    size_t                  bytes = (size_t) (nent) * sizeof (float);

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (nent < 0)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Received request to allocate negative sized float vector (%d entries)",
            nent);
    if (bytes > (size_t) INT32_MAX)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid too large size for float vector (%d entries)",
            nent);
    if (!fv)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid reference to float vector object to initialize");

    *fv = nil;
    if (bytes > 0)
    {
        fv->arr = (float*) pctxt->alloc_fn (bytes);
        if (fv->arr == NULL)
            return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);
        fv->length     = nent;
        fv->alloc_size = nent;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_float_vector_init_static (
    exr_context_t            ctxt,
    exr_attr_float_vector_t* fv,
    const float*             arr,
    int32_t                  nent)
{
    exr_attr_float_vector_t nil = {0};

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (nent < 0)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Received request to allocate negative sized float vector (%d entries)",
            nent);
    if (!fv)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid reference to float vector object to initialize");
    if (!arr)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid reference to float array object to initialize");

    *fv            = nil;
    fv->arr        = arr;
    fv->length     = nent;
    fv->alloc_size = 0;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_float_vector_create (
    exr_context_t            ctxt,
    exr_attr_float_vector_t* fv,
    const float*             arr,
    int32_t                  nent)
{
    exr_result_t rv = EXR_ERR_UNKNOWN;
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (!fv || !arr)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid (NULL) arguments to float vector create");

    rv = exr_attr_float_vector_init (ctxt, fv, nent);
    if (rv == EXR_ERR_SUCCESS && nent > 0)
        memcpy (
            EXR_CONST_CAST (float*, fv->arr),
            arr,
            (size_t) (nent) * sizeof (float));
    return rv;
}

/**************************************/

exr_result_t
exr_attr_float_vector_destroy (exr_context_t ctxt, exr_attr_float_vector_t* fv)
{
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);
    if (fv)
    {
        exr_attr_float_vector_t nil = {0};
        if (fv->arr && fv->alloc_size > 0)
            pctxt->free_fn (EXR_CONST_CAST (void*, fv->arr));
        *fv = nil;
    }
    return EXR_ERR_SUCCESS;
}
