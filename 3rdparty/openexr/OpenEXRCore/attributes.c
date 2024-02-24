/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_attr.h"

#include "internal_constants.h"
#include "internal_structs.h"

#include <string.h>

struct _internal_exr_attr_map
{
    const char*          name;
    uint32_t             name_len;
    exr_attribute_type_t type;
    size_t               exp_size;
};

static struct _internal_exr_attr_map the_predefined_attr_typenames[] = {
    {"box2i", 5, EXR_ATTR_BOX2I, sizeof (exr_attr_box2i_t)},
    {"box2f", 5, EXR_ATTR_BOX2F, sizeof (exr_attr_box2f_t)},
    {"chlist", 6, EXR_ATTR_CHLIST, sizeof (exr_attr_chlist_t)},
    {"chromaticities",
     14,
     EXR_ATTR_CHROMATICITIES,
     sizeof (exr_attr_chromaticities_t)},
    {"compression", 11, EXR_ATTR_COMPRESSION, 0},
    {"double", 6, EXR_ATTR_DOUBLE, 0},
    {"envmap", 6, EXR_ATTR_ENVMAP, 0},
    {"float", 5, EXR_ATTR_FLOAT, 0},
    {"floatvector",
     11,
     EXR_ATTR_FLOAT_VECTOR,
     sizeof (exr_attr_float_vector_t)},
    {"int", 3, EXR_ATTR_INT, 0},
    {"keycode", 7, EXR_ATTR_KEYCODE, sizeof (exr_attr_keycode_t)},
    {"lineOrder", 9, EXR_ATTR_LINEORDER, 0},
    {"m33f", 4, EXR_ATTR_M33F, sizeof (exr_attr_m33f_t)},
    {"m33d", 4, EXR_ATTR_M33D, sizeof (exr_attr_m33d_t)},
    {"m44f", 4, EXR_ATTR_M44F, sizeof (exr_attr_m44f_t)},
    {"m44d", 4, EXR_ATTR_M44D, sizeof (exr_attr_m44d_t)},
    {"preview", 7, EXR_ATTR_PREVIEW, sizeof (exr_attr_preview_t)},
    {"rational", 8, EXR_ATTR_RATIONAL, sizeof (exr_attr_rational_t)},
    {"string", 6, EXR_ATTR_STRING, sizeof (exr_attr_string_t)},
    {"stringvector",
     12,
     EXR_ATTR_STRING_VECTOR,
     sizeof (exr_attr_string_vector_t)},
    {"tiledesc", 8, EXR_ATTR_TILEDESC, sizeof (exr_attr_tiledesc_t)},
    {"timecode", 8, EXR_ATTR_TIMECODE, sizeof (exr_attr_timecode_t)},
    {"v2i", 3, EXR_ATTR_V2I, sizeof (exr_attr_v2i_t)},
    {"v2f", 3, EXR_ATTR_V2F, sizeof (exr_attr_v2f_t)},
    {"v2d", 3, EXR_ATTR_V2D, sizeof (exr_attr_v2d_t)},
    {"v3i", 3, EXR_ATTR_V3I, sizeof (exr_attr_v3i_t)},
    {"v3f", 3, EXR_ATTR_V3F, sizeof (exr_attr_v3f_t)},
    {"v3d", 3, EXR_ATTR_V3D, sizeof (exr_attr_v3d_t)}};
static int the_predefined_attr_count = sizeof (the_predefined_attr_typenames) /
                                       sizeof (struct _internal_exr_attr_map);

/**************************************/

static exr_result_t
attr_init (struct _internal_exr_context* ctxt, exr_attribute_t* nattr)
{
    switch (nattr->type)
    {
        case EXR_ATTR_BOX2I: {
            exr_attr_box2i_t nil = {0};
            *(nattr->box2i)      = nil;
            break;
        }
        case EXR_ATTR_BOX2F: {
            exr_attr_box2f_t nil = {0};
            *(nattr->box2f)      = nil;
            break;
        }
        case EXR_ATTR_CHLIST: {
            exr_attr_chlist_t nil = {0};
            *(nattr->chlist)      = nil;
            break;
        }
        case EXR_ATTR_CHROMATICITIES: {
            exr_attr_chromaticities_t nil = {0};
            *(nattr->chromaticities)      = nil;
            break;
        }
        case EXR_ATTR_COMPRESSION:
        case EXR_ATTR_ENVMAP:
        case EXR_ATTR_LINEORDER: nattr->uc = 0; break;
        case EXR_ATTR_DOUBLE: nattr->d = 0.0; break;
        case EXR_ATTR_FLOAT: nattr->f = 0.0f; break;
        case EXR_ATTR_FLOAT_VECTOR: {
            exr_attr_float_vector_t nil = {0};
            *(nattr->floatvector)       = nil;
            break;
        }
        case EXR_ATTR_INT: nattr->i = 0; break;
        case EXR_ATTR_KEYCODE: {
            exr_attr_keycode_t nil = {0};
            *(nattr->keycode)      = nil;
            break;
        }
        case EXR_ATTR_M33F: {
            exr_attr_m33f_t nil = {0};
            *(nattr->m33f)      = nil;
            break;
        }
        case EXR_ATTR_M33D: {
            exr_attr_m33d_t nil = {0};
            *(nattr->m33d)      = nil;
            break;
        }
        case EXR_ATTR_M44F: {
            exr_attr_m44f_t nil = {0};
            *(nattr->m44f)      = nil;
            break;
        }
        case EXR_ATTR_M44D: {
            exr_attr_m44f_t nil = {0};
            *(nattr->m44f)      = nil;
            break;
        }
        case EXR_ATTR_PREVIEW: {
            exr_attr_preview_t nil = {0};
            *(nattr->preview)      = nil;
            break;
        }
        case EXR_ATTR_RATIONAL: {
            exr_attr_rational_t nil = {0};
            *(nattr->rational)      = nil;
            break;
        }
        case EXR_ATTR_STRING: {
            exr_attr_string_t nil = {0};
            *(nattr->string)      = nil;
            break;
        }
        case EXR_ATTR_STRING_VECTOR: {
            exr_attr_string_vector_t nil = {0};
            *(nattr->stringvector)       = nil;
            break;
        }
        case EXR_ATTR_TILEDESC: {
            exr_attr_tiledesc_t nil = {0};
            *(nattr->tiledesc)      = nil;
            break;
        }
        case EXR_ATTR_TIMECODE: {
            exr_attr_timecode_t nil = {0};
            *(nattr->timecode)      = nil;
            break;
        }
        case EXR_ATTR_V2I: {
            exr_attr_v2i_t nil = {0};
            *(nattr->v2i)      = nil;
            break;
        }
        case EXR_ATTR_V2F: {
            exr_attr_v2f_t nil = {0};
            *(nattr->v2f)      = nil;
            break;
        }
        case EXR_ATTR_V2D: {
            exr_attr_v2d_t nil = {0};
            *(nattr->v2d)      = nil;
            break;
        }
        case EXR_ATTR_V3I: {
            exr_attr_v3i_t nil = {0};
            *(nattr->v3i)      = nil;
            break;
        }
        case EXR_ATTR_V3F: {
            exr_attr_v3f_t nil = {0};
            *(nattr->v3f)      = nil;
            break;
        }
        case EXR_ATTR_V3D: {
            exr_attr_v3d_t nil = {0};
            *(nattr->v3d)      = nil;
            break;
        }
        case EXR_ATTR_OPAQUE: {
            exr_attr_opaquedata_t nil = {0};
            *(nattr->opaque)          = nil;
            break;
        }
        case EXR_ATTR_UNKNOWN:
        case EXR_ATTR_LAST_KNOWN_TYPE:
        default:
            if (ctxt)
                ctxt->print_error (
                    ctxt,
                    EXR_ERR_INVALID_ARGUMENT,
                    "Invalid / unimplemented type (%s) in attr_init",
                    nattr->type_name);
            return EXR_ERR_INVALID_ARGUMENT;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
attr_destroy (struct _internal_exr_context* ctxt, exr_attribute_t* attr)
{
    exr_result_t rv = EXR_ERR_SUCCESS;
    switch (attr->type)
    {
        case EXR_ATTR_CHLIST:
            rv = exr_attr_chlist_destroy ((exr_context_t) ctxt, attr->chlist);
            break;
        case EXR_ATTR_FLOAT_VECTOR:
            rv = exr_attr_float_vector_destroy (
                (exr_context_t) ctxt, attr->floatvector);
            break;
        case EXR_ATTR_PREVIEW:
            rv = exr_attr_preview_destroy ((exr_context_t) ctxt, attr->preview);
            break;
        case EXR_ATTR_STRING:
            rv = exr_attr_string_destroy ((exr_context_t) ctxt, attr->string);
            break;
        case EXR_ATTR_STRING_VECTOR:
            rv = exr_attr_string_vector_destroy (
                (exr_context_t) ctxt, attr->stringvector);
            break;
        case EXR_ATTR_OPAQUE:
            rv = exr_attr_opaquedata_destroy (
                (exr_context_t) ctxt, attr->opaque);
            break;
        case EXR_ATTR_BOX2I:
        case EXR_ATTR_BOX2F:
        case EXR_ATTR_CHROMATICITIES:
        case EXR_ATTR_COMPRESSION:
        case EXR_ATTR_ENVMAP:
        case EXR_ATTR_LINEORDER:
        case EXR_ATTR_DOUBLE:
        case EXR_ATTR_FLOAT:
        case EXR_ATTR_INT:
        case EXR_ATTR_KEYCODE:
        case EXR_ATTR_M33F:
        case EXR_ATTR_M33D:
        case EXR_ATTR_M44F:
        case EXR_ATTR_M44D:
        case EXR_ATTR_RATIONAL:
        case EXR_ATTR_TILEDESC:
        case EXR_ATTR_TIMECODE:
        case EXR_ATTR_V2I:
        case EXR_ATTR_V2F:
        case EXR_ATTR_V2D:
        case EXR_ATTR_V3I:
        case EXR_ATTR_V3F:
        case EXR_ATTR_V3D:
        case EXR_ATTR_UNKNOWN:
        case EXR_ATTR_LAST_KNOWN_TYPE:
        default: break;
    }
    /* we don't care about the string because they were built into the
     * allocation block of the attribute as necessary */
    ctxt->free_fn (attr);
    return rv;
}

/**************************************/

int
internal_exr_is_standard_type (const char* typen)
{
    for (int i = 0; i < the_predefined_attr_count; ++i)
    {
        if (0 == strcmp (typen, the_predefined_attr_typenames[i].name))
            return 1;
    }
    return 0;
}

/**************************************/

exr_result_t
exr_attr_list_destroy (exr_context_t ctxt, exr_attribute_list_t* list)
{
    exr_attribute_list_t nil = {0};
    exr_result_t         arv;
    exr_result_t         rv = EXR_ERR_SUCCESS;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (list)
    {
        if (list->entries)
        {
            for (int i = 0; i < list->num_attributes; ++i)
            {
                arv = attr_destroy (pctxt, list->entries[i]);
                if (arv != EXR_ERR_SUCCESS) rv = arv;
            }
            pctxt->free_fn (list->entries);
        }
        *list = nil;
    }
    return rv;
}

/**************************************/

exr_result_t
exr_attr_list_compute_size (
    exr_context_t ctxt, exr_attribute_list_t* list, uint64_t* out)
{
    uint64_t     retval = 0;
    exr_result_t rv     = EXR_ERR_SUCCESS;

    INTERN_EXR_PROMOTE_CONST_CONTEXT_OR_ERROR (ctxt);

    if (!list)
        return pctxt->report_error (
            pctxt, EXR_ERR_INVALID_ARGUMENT, "Missing list to compute size");

    if (!out)
        return pctxt->report_error (
            pctxt, EXR_ERR_INVALID_ARGUMENT, "Expected output pointer");

    *out = 0;
    for (int i = 0; i < list->num_attributes; ++i)
    {
        const exr_attribute_t* cur = list->entries[i];
        retval += (size_t) cur->name_length + 1;
        retval += (size_t) cur->type_name_length + 1;
        retval += sizeof (int32_t);
        switch (cur->type)
        {
            case EXR_ATTR_BOX2I: retval += sizeof (*(cur->box2i)); break;
            case EXR_ATTR_BOX2F: retval += sizeof (*(cur->box2f)); break;
            case EXR_ATTR_CHLIST:
                for (int c = 0; c < cur->chlist->num_channels; ++c)
                {
                    retval += (size_t) cur->chlist->entries[c].name.length + 1;
                    retval += sizeof (int32_t) * 4;
                }
                break;
            case EXR_ATTR_CHROMATICITIES:
                retval += sizeof (*(cur->chromaticities));
                break;
            case EXR_ATTR_COMPRESSION:
            case EXR_ATTR_ENVMAP:
            case EXR_ATTR_LINEORDER: retval += sizeof (uint8_t); break;
            case EXR_ATTR_DOUBLE: retval += sizeof (double); break;
            case EXR_ATTR_FLOAT: retval += sizeof (float); break;
            case EXR_ATTR_FLOAT_VECTOR:
                retval += sizeof (float) * (size_t) (cur->floatvector->length);
                break;
            case EXR_ATTR_INT: retval += sizeof (int32_t); break;
            case EXR_ATTR_KEYCODE: retval += sizeof (*(cur->keycode)); break;
            case EXR_ATTR_M33F: retval += sizeof (*(cur->m33f)); break;
            case EXR_ATTR_M33D: retval += sizeof (*(cur->m33d)); break;
            case EXR_ATTR_M44F: retval += sizeof (*(cur->m44f)); break;
            case EXR_ATTR_M44D: retval += sizeof (*(cur->m44d)); break;
            case EXR_ATTR_PREVIEW:
                retval += (size_t) cur->preview->width *
                          (size_t) cur->preview->height * (size_t) 4;
                break;
            case EXR_ATTR_RATIONAL: retval += sizeof (*(cur->rational)); break;
            case EXR_ATTR_STRING: retval += (size_t) cur->string->length; break;
            case EXR_ATTR_STRING_VECTOR:
                for (int s = 0; s < cur->stringvector->n_strings; ++s)
                {
                    retval += (size_t) cur->stringvector->strings[s].length;
                    retval += sizeof (int32_t);
                }
                break;
            case EXR_ATTR_TILEDESC: retval += sizeof (*(cur->tiledesc)); break;
            case EXR_ATTR_TIMECODE: retval += sizeof (*(cur->timecode)); break;
            case EXR_ATTR_V2I: retval += sizeof (*(cur->v2i)); break;
            case EXR_ATTR_V2F: retval += sizeof (*(cur->v2f)); break;
            case EXR_ATTR_V2D: retval += sizeof (*(cur->v2d)); break;
            case EXR_ATTR_V3I: retval += sizeof (*(cur->v3i)); break;
            case EXR_ATTR_V3F: retval += sizeof (*(cur->v3f)); break;
            case EXR_ATTR_V3D: retval += sizeof (*(cur->v3d)); break;
            case EXR_ATTR_OPAQUE:
                if (cur->opaque->packed_data)
                    retval += (size_t) cur->opaque->size;
                else if (cur->opaque->unpacked_data)
                {
                    int32_t sz = 0;
                    rv =
                        exr_attr_opaquedata_pack (ctxt, cur->opaque, &sz, NULL);
                    if (rv != EXR_ERR_SUCCESS) return rv;

                    retval += (size_t) sz;
                }
                break;
            case EXR_ATTR_UNKNOWN:
            case EXR_ATTR_LAST_KNOWN_TYPE:
            default:
                return pctxt->print_error (
                    pctxt,
                    EXR_ERR_INVALID_ARGUMENT,
                    "Invalid / unhandled type '%s' for attribute '%s', unable to compute size",
                    cur->type_name,
                    cur->name);
        }
    }

    *out = retval;
    return rv;
}

/**************************************/

exr_result_t
exr_attr_list_find_by_name (
    exr_const_context_t   ctxt,
    exr_attribute_list_t* list,
    const char*           name,
    exr_attribute_t**     out)
{
    exr_attribute_t** it    = NULL;
    exr_attribute_t** first = NULL;
    exr_attribute_t** end   = NULL;
    int               step, count, cmp;
    INTERN_EXR_PROMOTE_CONST_CONTEXT_OR_ERROR (ctxt);

    if (!out)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid output pointer passed to find_by_name");

    if (!name || name[0] == '\0')
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid name passed to find_by_name");

    if (!list)
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid list pointer passed to find_by_name");

    if (list->sorted_entries)
    {
        first = list->sorted_entries;
        count = list->num_attributes;
        end   = first + count;
        /* lower bound search w/ equality check */
        while (count > 0)
        {
            it   = first;
            step = count / 2;
            it += step;
            cmp = strcmp ((*it)->name, name);
            if (cmp == 0)
            {
                // early exit
                *out = (*it);
                return EXR_ERR_SUCCESS;
            }

            if (cmp < 0)
            {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }

        if (first && first < end && 0 == strcmp ((*first)->name, name))
        {
            *out = (*first);
            return EXR_ERR_SUCCESS;
        }
    }

    return EXR_ERR_NO_ATTR_BY_NAME;
}

/**************************************/

static exr_result_t
add_to_list (
    struct _internal_exr_context* ctxt,
    exr_attribute_list_t*         list,
    exr_attribute_t*              nattr,
    const char*                   name)
{
    int               cattrsz = list->num_attributes;
    int               nattrsz = cattrsz + 1;
    int               insertpos;
    exr_attribute_t** attrs        = list->entries;
    exr_attribute_t** sorted_attrs = list->sorted_entries;
    exr_result_t      rv           = EXR_ERR_SUCCESS;

    (void) name;
    if (nattrsz > list->num_alloced)
    {
        size_t nsize = (size_t) (list->num_alloced) * 2;
        if ((size_t) nattrsz > nsize) nsize = (size_t) (nattrsz) + 1;
        attrs = (exr_attribute_t**) ctxt->alloc_fn (
            sizeof (exr_attribute_t*) * nsize * 2);
        if (!attrs)
        {
            ctxt->free_fn (nattr);
            return ctxt->standard_error (ctxt, EXR_ERR_OUT_OF_MEMORY);
        }

        list->num_alloced = (int32_t) nsize;
        sorted_attrs      = attrs + nsize;

        for (int i = 0; i < cattrsz; ++i)
        {
            attrs[i]        = list->entries[i];
            sorted_attrs[i] = list->sorted_entries[i];
        }

        if (list->entries) ctxt->free_fn (list->entries);
        list->entries        = attrs;
        list->sorted_entries = sorted_attrs;
    }
    attrs[cattrsz]        = nattr;
    sorted_attrs[cattrsz] = nattr;
    insertpos             = cattrsz - 1;

    // FYI: qsort is shockingly slow, just do a quick search and
    // bubble it up until it's in the correct location
    while (insertpos >= 0)
    {
        exr_attribute_t* prev = sorted_attrs[insertpos];

        if (strcmp (nattr->name, prev->name) >= 0) break;

        sorted_attrs[insertpos + 1] = prev;
        sorted_attrs[insertpos]     = nattr;
        --insertpos;
    }

    list->num_attributes = nattrsz;
    rv                   = attr_init (ctxt, nattr);
    if (rv != EXR_ERR_SUCCESS)
        exr_attr_list_remove ((exr_context_t) ctxt, list, nattr);
    return rv;
}

/**************************************/

static exr_result_t
validate_attr_arguments (
    struct _internal_exr_context* ctxt,
    exr_attribute_list_t*         list,
    const char*                   name,
    int32_t                       data_len,
    uint8_t**                     data_ptr,
    exr_attribute_t**             attr)
{
    exr_attribute_t* nattr = NULL;
    exr_result_t     rv;
    if (!list)
    {
        return ctxt->report_error (
            ctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid list pointer to attr_list_add");
    }

    if (!attr)
    {
        return ctxt->report_error (
            ctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid output attribute pointer location to attr_list_add");
    }

    *attr = NULL;

    if (data_len < 0)
    {
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Extra data storage requested negative length (%d)",
            data_len);
    }
    else if (data_len > 0 && !data_ptr)
    {
        return ctxt->print_error (
            ctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Extra data storage output pointer must be provided when requesting extra data (%d)",
            data_len);
    }
    else if (data_ptr)
        *data_ptr = NULL;

    if (!name || name[0] == '\0')
    {
        return ctxt->report_error (
            ctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid name to add_by_type");
    }

    /* is it already in the list? */
    rv = exr_attr_list_find_by_name (
        (exr_const_context_t) ctxt, list, name, &nattr);

    if (rv == EXR_ERR_SUCCESS)
    {
        if (data_ptr && data_len > 0)
        {
            return ctxt->print_error (
                ctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Attribute '%s' (type %s) already in list but requesting additional data",
                name,
                nattr->type_name);
        }

        *attr = nattr;
        return -1;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static void
check_attr_handler (struct _internal_exr_context* pctxt, exr_attribute_t* attr)
{
    if (attr->type == EXR_ATTR_OPAQUE)
    {
        exr_attribute_t* handler = NULL;
        exr_result_t     rv      = exr_attr_list_find_by_name (
            (exr_const_context_t) pctxt,
            &(pctxt->custom_handlers),
            attr->type_name,
            &handler);
        if (rv == EXR_ERR_SUCCESS && handler)
        {
            attr->opaque->unpack_func_ptr = handler->opaque->unpack_func_ptr;
            attr->opaque->pack_func_ptr   = handler->opaque->pack_func_ptr;
            attr->opaque->destroy_unpacked_func_ptr =
                handler->opaque->destroy_unpacked_func_ptr;
        }
    }
}

/**************************************/

static exr_result_t
create_attr_block (
    struct _internal_exr_context* pctxt,
    exr_attribute_t**             attr,
    size_t                        dblocksize,
    int32_t                       data_len,
    uint8_t**                     data_ptr,
    const char*                   name,
    int32_t                       nlen,
    const char*                   type,
    int32_t                       tlen)
{
    size_t           alignpad1, alignpad2;
    size_t           attrblocksz = sizeof (exr_attribute_t);
    uint8_t*         ptr;
    exr_attribute_t* nattr;
    exr_attribute_t  nil = {0};
    // not all compilers have this :(
    //const size_t ptralign = _Alignof(void*);
    const size_t ptralign = 8;

    if (nlen > 0) attrblocksz += (size_t) (nlen + 1);
    if (tlen > 0) attrblocksz += (size_t) (tlen + 1);

    if (dblocksize > 0)
    {
        alignpad1 = ptralign - (attrblocksz % ptralign);
        if (alignpad1 == ptralign) alignpad1 = 0;
        attrblocksz += alignpad1;
        attrblocksz += dblocksize;
    }
    else
        alignpad1 = 0;

    if (data_len > 0)
    {
        /* align the extra data to a pointer */
        alignpad2 = ptralign - (attrblocksz % ptralign);
        if (alignpad2 == ptralign) alignpad2 = 0;
        attrblocksz += alignpad2;
        attrblocksz += (size_t) data_len;
    }
    else
        alignpad2 = 0;

    ptr = (uint8_t*) pctxt->alloc_fn (attrblocksz);
    if (!ptr) return pctxt->standard_error (pctxt, EXR_ERR_OUT_OF_MEMORY);

    nattr  = (exr_attribute_t*) ptr;
    *nattr = nil;
    *attr  = nattr;
    ptr += sizeof (exr_attribute_t);
    if (nlen > 0)
    {
        memcpy (ptr, name, (size_t) (nlen + 1));
        nattr->name        = (char*) ptr;
        nattr->name_length = (uint8_t) nlen;

        ptr += nlen + 1;
    }
    if (tlen > 0)
    {
        memcpy (ptr, type, (size_t) (tlen + 1));
        nattr->type_name        = (char*) ptr;
        nattr->type_name_length = (uint8_t) tlen;

        ptr += tlen + 1;
    }
    ptr += alignpad1;
    if (dblocksize > 0)
    {
        nattr->rawptr = ptr;
        ptr += dblocksize;
    }
    if (data_ptr)
    {
        if (data_len > 0)
        {
            ptr += alignpad2;
            *data_ptr = ptr;
        }
        else
            *data_ptr = NULL;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_attr_list_add_by_type (
    exr_context_t         ctxt,
    exr_attribute_list_t* list,
    const char*           name,
    const char*           type,
    int32_t               data_len,
    uint8_t**             data_ptr,
    exr_attribute_t**     attr)
{
    const struct _internal_exr_attr_map* known = NULL;

    exr_result_t     rval = EXR_ERR_INVALID_ARGUMENT;
    int32_t          nlen, tlen, mlen;
    size_t           slen;
    exr_attribute_t* nattr = NULL;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (!type || type[0] == '\0')
    {
        return pctxt->report_error (
            pctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid type to add_by_type");
    }

    rval =
        validate_attr_arguments (pctxt, list, name, data_len, data_ptr, attr);
    if (rval != EXR_ERR_SUCCESS)
    {
        if (rval < 0)
        {
            if (0 != strcmp (type, (*attr)->type_name))
            {
                nattr = *attr;
                *attr = NULL;
                return pctxt->print_error (
                    pctxt,
                    EXR_ERR_INVALID_ARGUMENT,
                    "Entry '%s' already in list but with different type ('%s' vs requested '%s')",
                    name,
                    nattr->type_name,
                    type);
            }
            return EXR_ERR_SUCCESS;
        }
        return rval;
    }

    slen = strlen (name);
    mlen = (int32_t) pctxt->max_name_length;

    if (slen > (size_t) mlen)
    {
        return pctxt->print_error (
            pctxt,
            EXR_ERR_NAME_TOO_LONG,
            "Provided name '%s' too long for file (len %d, max %d)",
            name,
            (int) slen,
            mlen);
    }
    nlen = (int32_t) slen;

    slen = strlen (type);
    if (slen > (size_t) mlen)
    {
        return pctxt->print_error (
            pctxt,
            EXR_ERR_NAME_TOO_LONG,
            "Provided type name '%s' too long for file (len %d, max %d)",
            type,
            (int) slen,
            mlen);
    }
    tlen = (int32_t) slen;

    for (int i = 0; i < the_predefined_attr_count; ++i)
    {
        if (0 == strcmp (type, the_predefined_attr_typenames[i].name))
        {
            known = &(the_predefined_attr_typenames[i]);
            break;
        }
    }

    if (known)
    {
        rval = create_attr_block (
            pctxt,
            &nattr,
            known->exp_size,
            data_len,
            data_ptr,
            name,
            nlen,
            NULL,
            0);

        if (rval == EXR_ERR_SUCCESS)
        {
            nattr->type_name        = known->name;
            nattr->type_name_length = (uint8_t) known->name_len;
            nattr->type             = known->type;
        }
    }
    else
    {
        rval = create_attr_block (
            pctxt,
            &nattr,
            sizeof (exr_attr_opaquedata_t),
            data_len,
            data_ptr,
            name,
            nlen,
            type,
            tlen);

        if (rval == EXR_ERR_SUCCESS) nattr->type = EXR_ATTR_OPAQUE;
    }
    if (rval == EXR_ERR_SUCCESS) rval = add_to_list (pctxt, list, nattr, name);
    if (rval == EXR_ERR_SUCCESS)
    {
        *attr = nattr;
        check_attr_handler (pctxt, nattr);
    }
    else if (data_ptr)
        *data_ptr = NULL;

    return rval;
}

/**************************************/

exr_result_t
exr_attr_list_add (
    exr_context_t         ctxt,
    exr_attribute_list_t* list,
    const char*           name,
    exr_attribute_type_t  type,
    int32_t               data_len,
    uint8_t**             data_ptr,
    exr_attribute_t**     attr)
{
    const struct _internal_exr_attr_map* known = NULL;

    exr_result_t     rval = EXR_ERR_INVALID_ARGUMENT;
    int32_t          nlen, tidx, mlen;
    size_t           slen;
    exr_attribute_t* nattr = NULL;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    rval =
        validate_attr_arguments (pctxt, list, name, data_len, data_ptr, attr);
    if (rval != EXR_ERR_SUCCESS)
    {
        if (rval < 0)
        {
            if ((*attr)->type != type)
            {
                nattr = *attr;
                *attr = NULL;
                return pctxt->print_error (
                    pctxt,
                    EXR_ERR_INVALID_ARGUMENT,
                    "Entry '%s' already in list but with different type ('%s')",
                    name,
                    nattr->type_name);
            }
            return EXR_ERR_SUCCESS;
        }
        return rval;
    }

    slen = strlen (name);
    mlen = (int32_t) pctxt->max_name_length;
    if (slen > (size_t) mlen)
    {
        return pctxt->print_error (
            pctxt,
            EXR_ERR_NAME_TOO_LONG,
            "Provided name '%s' too long for file (len %d, max %d)",
            name,
            (int) slen,
            mlen);
    }
    nlen = (int32_t) slen;

    tidx = ((int) type) - 1;
    if (tidx < 0 || tidx >= the_predefined_attr_count)
    {
        if (type == EXR_ATTR_OPAQUE)
            return pctxt->print_error (
                pctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Invalid type enum for '%s': the opaque type is not actually a built-in type",
                name);

        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid type enum for '%s' in create by builtin type (type %d)",
            name,
            (int) type);
    }

    known = &(the_predefined_attr_typenames[tidx]);

    rval = create_attr_block (
        pctxt,
        &nattr,
        known->exp_size,
        data_len,
        data_ptr,
        name,
        nlen,
        NULL,
        0);

    if (rval == EXR_ERR_SUCCESS)
    {
        nattr->type_name        = known->name;
        nattr->type_name_length = (uint8_t) known->name_len;
        nattr->type             = known->type;
        rval                    = add_to_list (pctxt, list, nattr, name);
    }

    if (rval == EXR_ERR_SUCCESS)
    {
        *attr = nattr;
        check_attr_handler (pctxt, nattr);
    }
    else if (data_ptr)
        *data_ptr = NULL;
    return rval;
}

/**************************************/

exr_result_t
exr_attr_list_add_static_name (
    exr_context_t         ctxt,
    exr_attribute_list_t* list,
    const char*           name,
    exr_attribute_type_t  type,
    int32_t               data_len,
    uint8_t**             data_ptr,
    exr_attribute_t**     attr)
{
    const struct _internal_exr_attr_map* known = NULL;

    int              rval = EXR_ERR_INVALID_ARGUMENT;
    int32_t          nlen, tidx, mlen;
    size_t           slen;
    exr_attribute_t* nattr = NULL;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    rval =
        validate_attr_arguments (pctxt, list, name, data_len, data_ptr, attr);
    if (rval != EXR_ERR_SUCCESS)
    {
        if (rval < 0)
        {
            if ((*attr)->type != type)
            {
                nattr = *attr;
                *attr = NULL;
                return pctxt->print_error (
                    pctxt,
                    EXR_ERR_INVALID_ARGUMENT,
                    "Entry '%s' already in list but with different type ('%s')",
                    name,
                    nattr->type_name);
            }
            return EXR_ERR_SUCCESS;
        }
        return rval;
    }

    mlen = (int32_t) pctxt->max_name_length;
    slen = strlen (name);
    if (slen > (size_t) mlen)
    {
        return pctxt->print_error (
            pctxt,
            EXR_ERR_NAME_TOO_LONG,
            "Provided name '%s' too long for file (len %d, max %d)",
            name,
            (int) slen,
            mlen);
    }
    nlen = (int32_t) slen;

    tidx = ((int) type) - 1;
    if (tidx < 0 || tidx >= the_predefined_attr_count)
    {
        if (type == EXR_ATTR_OPAQUE)
            return pctxt->print_error (
                pctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Invalid type enum for '%s': the opaque type is not actually a built-in type",
                name);

        return pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid type enum for '%s' in create by builtin type (type %d)",
            name,
            (int) type);
    }
    known = &(the_predefined_attr_typenames[tidx]);

    rval = create_attr_block (
        pctxt, &nattr, known->exp_size, data_len, data_ptr, NULL, 0, NULL, 0);

    if (rval == EXR_ERR_SUCCESS)
    {
        nattr->name             = name;
        nattr->type_name        = known->name;
        nattr->name_length      = (uint8_t) nlen;
        nattr->type_name_length = (uint8_t) known->name_len;
        nattr->type             = known->type;
        rval                    = add_to_list (pctxt, list, nattr, name);
    }

    if (rval == EXR_ERR_SUCCESS)
    {
        *attr = nattr;
        check_attr_handler (pctxt, nattr);
    }
    else if (data_ptr)
        *data_ptr = NULL;
    return rval;
}

/**************************************/

exr_result_t
exr_attr_list_remove (
    exr_context_t ctxt, exr_attribute_list_t* list, exr_attribute_t* attr)
{
    int               cattrsz, attridx = -1;
    exr_attribute_t** attrs;

    INTERN_EXR_PROMOTE_CONTEXT_OR_ERROR (ctxt);

    if (!attr)
    {
        return pctxt->report_error (
            pctxt, EXR_ERR_INVALID_ARGUMENT, "NULL attribute passed to remove");
    }

    if (!list)
    {
        return pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid list pointer to remove attribute");
    }

    cattrsz = list->num_attributes;
    attrs   = list->entries;
    for (int i = 0; i < cattrsz; ++i)
    {
        if (attrs[i] == attr)
        {
            attridx = i;
            break;
        }
    }

    if (attridx == -1)
    {
        return pctxt->report_error (
            pctxt, EXR_ERR_INVALID_ARGUMENT, "Attribute not in list");
    }

    list->entries[attridx] = NULL;
    for (int i = attridx; i < (cattrsz - 1); ++i)
        attrs[i] = attrs[i + 1];
    list->num_attributes = cattrsz - 1;

    attrs   = list->sorted_entries;
    attridx = 0;
    for (int i = 0; i < cattrsz; ++i)
    {
        if (attrs[i] == attr) continue;
        attrs[attridx++] = attrs[i];
    }

    return attr_destroy (pctxt, attr);
}
