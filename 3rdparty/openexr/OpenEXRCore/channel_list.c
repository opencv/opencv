/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_attr.h"

#include "internal_constants.h"
#include "internal_structs.h"

#include <string.h>

/**************************************/

exr_result_t
exr_attr_chlist_init (exr_context_t ctxt, exr_attr_chlist_t* clist, int nchans)
{
    exr_attr_chlist_t        nil = {0};
    exr_attr_chlist_entry_t* nlist;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (!clist)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid channel list pointer to chlist_add_with_length");

    if (nchans < 0)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Negative number of channels requested (%d)",
            nchans);

    *clist = nil;

    if (nchans > 0)
    {
        nlist = (exr_attr_chlist_entry_t*) pctxt->alloc_fn (
            sizeof (*nlist) * (size_t) nchans);
        if (nlist == NULL)
            return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);
    }
    else
        nlist = NULL;
    clist->entries     = nlist;
    clist->num_alloced = nchans;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_chlist_add (
    exr_context_t              ctxt,
    exr_attr_chlist_t*         clist,
    const char*                name,
    exr_pixel_type_t           ptype,
    exr_perceptual_treatment_t islinear,
    int32_t                    xsamp,
    int32_t                    ysamp)
{
    int32_t len = 0;
    if (name) len = (int32_t) strlen (name);
    return exr_attr_chlist_add_with_length (
        ctxt, clist, name, len, ptype, islinear, xsamp, ysamp);
}

/**************************************/

exr_result_t
exr_attr_chlist_add_with_length (
    exr_context_t              ctxt,
    exr_attr_chlist_t*         clist,
    const char*                name,
    int32_t                    namelen,
    exr_pixel_type_t           ptype,
    exr_perceptual_treatment_t islinear,
    int32_t                    xsamp,
    int32_t                    ysamp)
{
    exr_attr_chlist_entry_t  nent = {0};
    exr_attr_chlist_entry_t *nlist, *olist;
    int                      newcount, insertpos;
    int32_t                  maxlen;
    exr_result_t             rv;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    maxlen = pctxt->max_name_length;

    if (!clist)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid channel list pointer to chlist_add_with_length");

    if (!name || name[0] == '\0' || namelen == 0)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Channel name must not be empty, received '%s'",
            (name ? name : "<NULL>"));

    if (namelen > maxlen)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_NAME_TOO_LONG,
            "Channel name must shorter than length allowed by file (%d), received '%s' (%d)",
            maxlen,
            name,
            namelen);

    if (ptype != EXR_PIXEL_UINT && ptype != EXR_PIXEL_HALF &&
        ptype != EXR_PIXEL_FLOAT)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid pixel type specified (%d) adding channel '%s' to list",
            (int) ptype,
            name);

    if (islinear != EXR_PERCEPTUALLY_LOGARITHMIC &&
        islinear != EXR_PERCEPTUALLY_LINEAR)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid perceptual linear flag value (%d) adding channel '%s' to list",
            (int) islinear,
            name);

    if (xsamp <= 0 || ysamp <= 0)
        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid pixel sampling (x %d y %d) adding channel '%s' to list",
            xsamp,
            ysamp,
            name);

    insertpos = 0;
    olist     = EXR_CONST_CAST (exr_attr_chlist_entry_t*, clist->entries);
    for (int32_t c = 0; c < clist->num_channels; ++c)
    {
        int ord = strcmp (name, olist[c].name.str);
        if (ord < 0)
        {
            insertpos = c;
            break;
        }
        else if (ord == 0)
        {
            return pctxt->print_error (
                pctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Attempt to add duplicate channel '%s' to channel list",
                name);
        }
        else
            insertpos = c + 1;
    }

    /* temporarily use newcount as a return value check */
    rv = exr_attr_string_create_with_length (ctxt, &(nent.name), name, namelen);
    if (rv != EXR_ERR_SUCCESS) return rv;

    newcount        = clist->num_channels + 1;
    nent.pixel_type = ptype;
    nent.p_linear   = (uint8_t) islinear;
    nent.x_sampling = xsamp;
    nent.y_sampling = ysamp;

    if (newcount > clist->num_alloced)
    {
        int nsz = clist->num_alloced * 2;
        if (newcount > nsz) nsz = newcount + 1;
        nlist = (exr_attr_chlist_entry_t*) pctxt->alloc_fn (
            sizeof (*nlist) * (size_t) nsz);
        if (nlist == NULL)
        {
            exr_attr_string_destroy (ctxt, &(nent.name));
            return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);
        }
        clist->num_alloced = nsz;
    }
    else
        nlist = EXR_CONST_CAST (exr_attr_chlist_entry_t*, clist->entries);

    /* since we can re-use same memory, have to have slightly more
     * complex logic to avoid overwrites, find where we will insert
     * and copy entries after that first */

    /* shift old entries further first */
    for (int i = newcount - 1; i > insertpos; --i)
        nlist[i] = olist[i - 1];
    nlist[insertpos] = nent;
    if (nlist != olist)
    {
        for (int i = 0; i < insertpos; ++i)
            nlist[i] = olist[i];
    }

    clist->num_channels = newcount;
    clist->entries      = nlist;
    if (nlist != olist) pctxt->free_fn (olist);
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_chlist_duplicate (
    exr_context_t ctxt, exr_attr_chlist_t* chl, const exr_attr_chlist_t* srcchl)
{
    exr_result_t rv;
    int          numchans;

    if (!chl || !srcchl) return EXR_ERR_INVALID_ARGUMENT;

    numchans = srcchl->num_channels;
    rv       = exr_attr_chlist_init (ctxt, chl, numchans);
    if (rv != EXR_ERR_SUCCESS) return rv;

    for (int c = 0; c < numchans; ++c)
    {
        const exr_attr_chlist_entry_t* cur = srcchl->entries + c;

        rv = exr_attr_chlist_add_with_length (
            ctxt,
            chl,
            cur->name.str,
            cur->name.length,
            cur->pixel_type,
            (exr_perceptual_treatment_t) cur->p_linear,
            cur->x_sampling,
            cur->y_sampling);
        if (rv != EXR_ERR_SUCCESS)
        {
            exr_attr_chlist_destroy (ctxt, chl);
            return rv;
        }
    }
    return rv;
}

/**************************************/

exr_result_t
exr_attr_chlist_destroy (exr_context_t ctxt, exr_attr_chlist_t* clist)
{
    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (clist)
    {
        exr_attr_chlist_t        nil = {0};
        int                      nc  = clist->num_channels;
        exr_attr_chlist_entry_t* entries =
            EXR_CONST_CAST (exr_attr_chlist_entry_t*, clist->entries);

        for (int i = 0; i < nc; ++i)
            exr_attr_string_destroy (ctxt, &(entries[i].name));
        if (entries) pctxt->free_fn (entries);
        *clist = nil;
    }
    return EXR_ERR_SUCCESS;
}
