/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef _LARGEFILE64_SOURCE
// define this if it hasn't been defined elsewhere
#    define _LARGEFILE64_SOURCE
#endif

#include "openexr_config.h"
#include "openexr_context.h"

#include "openexr_part.h"

#include "internal_constants.h"
#include "internal_file.h"
#include "backward_compatibility.h"

#if defined(_WIN32) || defined(_WIN64)
#    include "internal_win32_file_impl.h"
#else
#    include "internal_posix_file_impl.h"
#endif

/**************************************/

static exr_result_t
dispatch_read (
    const struct _internal_exr_context* ctxt,
    void*                               buf,
    uint64_t                            sz,
    uint64_t*                           offsetp,
    int64_t*                            nread,
    enum _INTERNAL_EXR_READ_MODE        rmode)
{
    int64_t      rval = -1;
    exr_result_t rv   = EXR_ERR_UNKNOWN;

    if (nread) *nread = rval;

    if (!ctxt) return EXR_ERR_MISSING_CONTEXT_ARG;

    if (!offsetp)
        return ctxt->report_error (
            ctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "read requested with no output offset pointer");

    if (ctxt->read_fn)
        rval = ctxt->read_fn (
            (exr_const_context_t) ctxt,
            ctxt->user_data,
            buf,
            sz,
            *offsetp,
            (exr_stream_error_func_ptr_t) ctxt->print_error);
    else
        return ctxt->standard_error (ctxt, EXR_ERR_NOT_OPEN_READ);

    if (nread) *nread = rval;
    if (rval > 0) *offsetp += (uint64_t) rval;

    if (rval == (int64_t) sz || (rmode == EXR_ALLOW_SHORT_READ && rval >= 0))
        rv = EXR_ERR_SUCCESS;
    else
        rv = EXR_ERR_READ_IO;

    return rv;
}

/**************************************/

static exr_result_t
dispatch_write (
    struct _internal_exr_context* ctxt,
    const void*                   buf,
    uint64_t                      sz,
    uint64_t*                     offsetp)
{
    int64_t rval = -1;

    if (!ctxt) return EXR_ERR_MISSING_CONTEXT_ARG;

    if (!offsetp)
        return ctxt->report_error (
            ctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "write requested with no output offset pointer");

    if (ctxt->write_fn)
        rval = ctxt->write_fn (
            (exr_const_context_t) ctxt,
            ctxt->user_data,
            buf,
            sz,
            *offsetp,
            (exr_stream_error_func_ptr_t) ctxt->print_error);
    else
        return ctxt->standard_error (ctxt, EXR_ERR_NOT_OPEN_WRITE);

    if (rval > 0) *offsetp += (uint64_t) rval;

    return (rval == (int64_t) sz) ? EXR_ERR_SUCCESS : EXR_ERR_WRITE_IO;
}

/**************************************/

static exr_result_t
process_query_size (
    struct _internal_exr_context* ctxt, exr_context_initializer_t* inits)
{
    if (inits->size_fn)
    {
        ctxt->file_size =
            (inits->size_fn) ((exr_const_context_t) ctxt, ctxt->user_data);
    }
    else { ctxt->file_size = -1; }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static inline exr_context_initializer_t
fill_context_data (const exr_context_initializer_t* ctxtdata)
{
    exr_context_initializer_t inits = EXR_DEFAULT_CONTEXT_INITIALIZER;
    if (ctxtdata)
    {
        inits.error_handler_fn = ctxtdata->error_handler_fn;
        inits.alloc_fn         = ctxtdata->alloc_fn;
        inits.free_fn          = ctxtdata->free_fn;
        inits.user_data        = ctxtdata->user_data;
        inits.read_fn          = ctxtdata->read_fn;
        inits.size_fn          = ctxtdata->size_fn;
        inits.write_fn         = ctxtdata->write_fn;
        inits.destroy_fn       = ctxtdata->destroy_fn;
        inits.max_image_width  = ctxtdata->max_image_width;
        inits.max_image_height = ctxtdata->max_image_height;
        inits.max_tile_width   = ctxtdata->max_tile_width;
        inits.max_tile_height  = ctxtdata->max_tile_height;
        if (ctxtdata->size >= sizeof (struct _exr_context_initializer_v2))
        {
            inits.zip_level   = ctxtdata->zip_level;
            inits.dwa_quality = ctxtdata->dwa_quality;
        }
        if (ctxtdata->size >= sizeof (struct _exr_context_initializer_v3))
        {
            inits.flags = ctxtdata->flags;
        }
    }

    internal_exr_update_default_handlers (&inits);
    return inits;
}

/**************************************/

exr_result_t
exr_test_file_header (
    const char* filename, const exr_context_initializer_t* ctxtdata)
{
    exr_result_t                  rv    = EXR_ERR_SUCCESS;
    struct _internal_exr_context* ret   = NULL;
    exr_context_initializer_t     inits = fill_context_data (ctxtdata);

    if (filename && filename[0] != '\0')
    {
        rv = internal_exr_alloc_context (
            &ret,
            &inits,
            EXR_CONTEXT_READ,
            sizeof (struct _internal_exr_filehandle));
        if (rv == EXR_ERR_SUCCESS)
        {
            ret->do_read = &dispatch_read;

            rv = exr_attr_string_create (
                (exr_context_t) ret, &(ret->filename), filename);
            if (rv == EXR_ERR_SUCCESS)
            {
                if (!inits.read_fn)
                {
                    inits.size_fn = &default_query_size_func;
                    rv            = default_init_read_file (ret);
                }

                if (rv == EXR_ERR_SUCCESS)
                    rv = process_query_size (ret, &inits);
                if (rv == EXR_ERR_SUCCESS) rv = internal_exr_check_magic (ret);
            }

            exr_finish ((exr_context_t*) &ret);
        }
        else
            rv = EXR_ERR_OUT_OF_MEMORY;
    }
    else
    {
        inits.error_handler_fn (
            NULL,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid filename passed to test file header function");
        rv = EXR_ERR_INVALID_ARGUMENT;
    }
    return rv;
}

/**************************************/

exr_result_t
exr_finish (exr_context_t* pctxt)
{
    struct _internal_exr_context* ctxt;
    exr_result_t                  rv = EXR_ERR_SUCCESS;

    if (!pctxt) return EXR_ERR_MISSING_CONTEXT_ARG;

    ctxt = EXR_CTXT (*pctxt);
    if (ctxt)
    {
        int failed = 0;
        if (ctxt->mode == EXR_CONTEXT_WRITE ||
            ctxt->mode == EXR_CONTEXT_WRITING_DATA)
            failed = 1;

        if (ctxt->mode != EXR_CONTEXT_READ) rv = finalize_write (ctxt, failed);

        if (ctxt->destroy_fn)
            ctxt->destroy_fn (*pctxt, ctxt->user_data, failed);

        internal_exr_destroy_context (ctxt);
    }
    *pctxt = NULL;

    return rv;
}

/**************************************/

exr_result_t
exr_start_read (
    exr_context_t*                   ctxt,
    const char*                      filename,
    const exr_context_initializer_t* ctxtdata)
{
    exr_result_t                  rv    = EXR_ERR_UNKNOWN;
    struct _internal_exr_context* ret   = NULL;
    exr_context_initializer_t     inits = fill_context_data (ctxtdata);

    if (!ctxt)
    {
        if (!(inits.flags & EXR_CONTEXT_FLAG_SILENT_HEADER_PARSE))
            inits.error_handler_fn (
                NULL,
                EXR_ERR_INVALID_ARGUMENT,
                "Invalid context handle passed to start_read function");
        return EXR_ERR_INVALID_ARGUMENT;
    }

    if (filename && filename[0] != '\0')
    {
        rv = internal_exr_alloc_context (
            &ret,
            &inits,
            EXR_CONTEXT_READ,
            sizeof (struct _internal_exr_filehandle));
        if (rv == EXR_ERR_SUCCESS)
        {
            ret->do_read = &dispatch_read;

            rv = exr_attr_string_create (
                (exr_context_t) ret, &(ret->filename), filename);
            if (rv == EXR_ERR_SUCCESS)
            {
                if (!inits.read_fn)
                {
                    inits.size_fn = &default_query_size_func;
                    rv            = default_init_read_file (ret);
                }

                if (rv == EXR_ERR_SUCCESS)
                    rv = process_query_size (ret, &inits);
                if (rv == EXR_ERR_SUCCESS) rv = internal_exr_parse_header (ret);
            }

            if (rv != EXR_ERR_SUCCESS) exr_finish ((exr_context_t*) &ret);
        }
        else
            rv = EXR_ERR_OUT_OF_MEMORY;
    }
    else
    {
        if (!(inits.flags & EXR_CONTEXT_FLAG_SILENT_HEADER_PARSE))
            inits.error_handler_fn (
                NULL,
                EXR_ERR_INVALID_ARGUMENT,
                "Invalid filename passed to start_read function");
        rv = EXR_ERR_INVALID_ARGUMENT;
    }

    *ctxt = (exr_context_t) ret;
    return rv;
}

/**************************************/

exr_result_t
exr_start_write (
    exr_context_t*                   ctxt,
    const char*                      filename,
    exr_default_write_mode_t         default_mode,
    const exr_context_initializer_t* ctxtdata)
{
    int                           rv    = EXR_ERR_UNKNOWN;
    struct _internal_exr_context* ret   = NULL;
    exr_context_initializer_t     inits = fill_context_data (ctxtdata);

    if (!ctxt)
    {
        inits.error_handler_fn (
            NULL,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid context handle passed to start_read function");
        return EXR_ERR_INVALID_ARGUMENT;
    }

    if (filename && filename[0] != '\0')
    {
        rv = internal_exr_alloc_context (
            &ret,
            &inits,
            EXR_CONTEXT_WRITE,
            sizeof (struct _internal_exr_filehandle));
        if (rv == EXR_ERR_SUCCESS)
        {
            ret->do_write = &dispatch_write;

            rv = exr_attr_string_create (
                (exr_context_t) ret, &(ret->filename), filename);

            if (rv == EXR_ERR_SUCCESS)
            {
                if (!inits.write_fn)
                {
                    if (default_mode == EXR_INTERMEDIATE_TEMP_FILE)
                        rv = make_temp_filename (ret);
                    if (rv == EXR_ERR_SUCCESS)
                        rv = default_init_write_file (ret);
                }
            }

            if (rv != EXR_ERR_SUCCESS) exr_finish ((exr_context_t*) &ret);
        }
        else
            rv = EXR_ERR_OUT_OF_MEMORY;
    }
    else
    {
        inits.error_handler_fn (
            NULL,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid filename passed to start_write function");
        rv = EXR_ERR_INVALID_ARGUMENT;
    }

    *ctxt = (exr_context_t) ret;
    return rv;
}

/**************************************/

exr_result_t
exr_start_inplace_header_update (
    exr_context_t*                   ctxt,
    const char*                      filename,
    const exr_context_initializer_t* ctxtdata)
{
    /* TODO: not yet implemented */
    (void) ctxt;
    (void) filename;
    (void) ctxtdata;
    return EXR_ERR_INVALID_ARGUMENT;
}

/**************************************/

exr_result_t
exr_get_file_name (exr_const_context_t ctxt, const char** name)
{
    EXR_PROMOTE_CONST_CONTEXT_OR_ERROR (ctxt);

    /* not changeable after construction, no locking needed */
    if (name)
    {
        *name = pctxt->filename.str;
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }

    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
        pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));
}

/**************************************/

exr_result_t
exr_get_user_data (exr_const_context_t ctxt, void** userdata)
{
    EXR_PROMOTE_CONST_CONTEXT_OR_ERROR (ctxt);

    /* not changeable after construction, no locking needed */
    if (userdata)
    {
        *userdata = pctxt->real_user_data;
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }

    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
        pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));
}

/**************************************/

exr_result_t
exr_register_attr_type_handler (
    exr_context_t ctxt,
    const char*   type,
    exr_result_t (*unpack_func_ptr) (
        exr_context_t ctxt,
        const void*   data,
        int32_t       attrsize,
        int32_t*      outsize,
        void**        outbuffer),
    exr_result_t (*pack_func_ptr) (
        exr_context_t ctxt,
        const void*   data,
        int32_t       datasize,
        int32_t*      outsize,
        void*         outbuffer),
    void (*destroy_unpacked_func_ptr) (
        exr_context_t ctxt, void* data, int32_t datasize))
{
    exr_attribute_t*      ent;
    exr_result_t          rv;
    int32_t               tlen, mlen = EXR_SHORTNAME_MAXLEN;
    size_t                slen;
    exr_attribute_list_t* curattrs;

    EXR_PROMOTE_LOCKED_CONTEXT_OR_ERROR (ctxt);

    mlen = (int32_t) pctxt->max_name_length;

    if (!type || type[0] == '\0')
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid type to register_attr_handler"));

    slen = strlen (type);
    if (slen > (size_t) mlen)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_NAME_TOO_LONG,
            "Provided type name '%s' too long for file (len %d, max %d)",
            type,
            (int) slen,
            mlen));
    tlen = (int32_t) slen;

    if (internal_exr_is_standard_type (type))
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Provided type name '%s' is a reserved / internal type name",
            type));

    rv = exr_attr_list_find_by_name (
        ctxt, &(pctxt->custom_handlers), type, &ent);
    if (rv == EXR_ERR_SUCCESS)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Attribute handler for '%s' previously registered",
            type));

    ent = NULL;
    rv  = exr_attr_list_add_by_type (
        ctxt, &(pctxt->custom_handlers), type, type, 0, NULL, &ent);
    if (rv != EXR_ERR_SUCCESS)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            rv,
            "Unable to register custom handler for type '%s'",
            type));

    ent->opaque->unpack_func_ptr           = unpack_func_ptr;
    ent->opaque->pack_func_ptr             = pack_func_ptr;
    ent->opaque->destroy_unpacked_func_ptr = destroy_unpacked_func_ptr;

    for (int p = 0; p < pctxt->num_parts; ++p)
    {
        curattrs = &(pctxt->parts[p]->attributes);
        if (curattrs)
        {
            int nattr = curattrs->num_attributes;
            for (int a = 0; a < nattr; ++a)
            {
                ent = curattrs->entries[a];
                if (ent->type_name_length == tlen &&
                    0 == strcmp (ent->type_name, type))
                {
                    ent->opaque->unpack_func_ptr = unpack_func_ptr;
                    ent->opaque->pack_func_ptr   = pack_func_ptr;
                    ent->opaque->destroy_unpacked_func_ptr =
                        destroy_unpacked_func_ptr;
                }
            }
        }
    }

    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_set_longname_support (exr_context_t ctxt, int onoff)
{
    uint8_t oldval, newval;

    EXR_PROMOTE_LOCKED_CONTEXT_OR_ERROR (ctxt);

    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    oldval = pctxt->max_name_length;
    newval = EXR_SHORTNAME_MAXLEN;
    if (onoff) newval = EXR_LONGNAME_MAXLEN;

    if (oldval > newval)
    {
        for (int p = 0; p < pctxt->num_parts; ++p)
        {
            struct _internal_exr_part* curp = pctxt->parts[p];
            for (int a = 0; a < curp->attributes.num_attributes; ++a)
            {
                exr_attribute_t* curattr = curp->attributes.entries[a];
                if (curattr->name_length > newval ||
                    curattr->type_name_length > newval)
                {
                    return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                        pctxt,
                        EXR_ERR_NAME_TOO_LONG,
                        "Part %d, attribute '%s' (type '%s') has a name too long for new longname setting (%d)",
                        curp->part_index,
                        curattr->name,
                        curattr->type_name,
                        (int) newval));
                }
                if (curattr->type == EXR_ATTR_CHLIST)
                {
                    exr_attr_chlist_t* chs = curattr->chlist;
                    for (int c = 0; c < chs->num_channels; ++c)
                    {
                        if (chs->entries[c].name.length > newval)
                        {
                            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                                pctxt,
                                EXR_ERR_NAME_TOO_LONG,
                                "Part %d, channel '%s' has a name too long for new longname setting (%d)",
                                curp->part_index,
                                chs->entries[c].name.str,
                                (int) newval));
                        }
                    }
                }
            }
        }
    }
    pctxt->max_name_length = newval;
    return EXR_UNLOCK_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
}

/**************************************/

exr_result_t
exr_write_header (exr_context_t ctxt)
{
    exr_result_t rv = EXR_ERR_SUCCESS;
    EXR_PROMOTE_LOCKED_CONTEXT_OR_ERROR (ctxt);

    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    if (pctxt->num_parts == 0)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt,
            EXR_ERR_FILE_BAD_HEADER,
            "No parts defined in file prior to writing data"));

    for (int p = 0; rv == EXR_ERR_SUCCESS && p < pctxt->num_parts; ++p)
    {
        struct _internal_exr_part* curp = pctxt->parts[p];

        int32_t ccount = 0;

        if (!curp->channels)
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MISSING_REQ_ATTR,
                "Part %d is missing channel list",
                p));

        rv = internal_exr_compute_tile_information (pctxt, curp, 0);
        if (rv != EXR_ERR_SUCCESS) break;

        ccount = internal_exr_compute_chunk_offset_size (curp);

        curp->chunk_count = ccount;

        if (pctxt->has_nonimage_data || pctxt->is_multipart)
        {
            EXR_UNLOCK (pctxt);
            rv = exr_attr_set_int (ctxt, p, EXR_REQ_CHUNK_COUNT_STR, ccount);
            EXR_LOCK (pctxt);
            if (rv != EXR_ERR_SUCCESS) break;
        }

        rv = internal_exr_validate_write_part (pctxt, curp);
    }

    pctxt->output_file_offset = 0;

    if (rv == EXR_ERR_SUCCESS) rv = internal_exr_write_header (pctxt);

    if (rv == EXR_ERR_SUCCESS)
    {
        pctxt->mode               = EXR_CONTEXT_WRITING_DATA;
        pctxt->cur_output_part    = 0;
        pctxt->last_output_chunk  = -1;
        pctxt->output_chunk_count = 0;
        for (int p = 0; rv == EXR_ERR_SUCCESS && p < pctxt->num_parts; ++p)
        {
            struct _internal_exr_part* curp = pctxt->parts[p];
            curp->chunk_table_offset        = pctxt->output_file_offset;
            pctxt->output_file_offset +=
                (uint64_t) (curp->chunk_count) * sizeof (uint64_t);
        }
    }

    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}
