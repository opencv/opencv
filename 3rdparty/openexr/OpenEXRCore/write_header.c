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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**************************************/

static exr_result_t
save_attr_sz (struct _internal_exr_context* ctxt, size_t sz)
{
    int32_t isz;

    if (sz > (size_t) INT32_MAX)
        return ctxt->standard_error (ctxt, EXR_ERR_INVALID_ARGUMENT);

    isz = (int32_t) sz;
    priv_from_native32 (&isz, 1);

    return ctxt->do_write (
        ctxt, &isz, sizeof (int32_t), &(ctxt->output_file_offset));
}

/**************************************/

static exr_result_t
save_attr_32 (struct _internal_exr_context* ctxt, void* ptr, int n)
{
    priv_from_native32 (ptr, n);

    return ctxt->do_write (
        ctxt,
        ptr,
        sizeof (int32_t) * (uint64_t) (n),
        &(ctxt->output_file_offset));
}

/**************************************/

static exr_result_t
save_attr_64 (struct _internal_exr_context* ctxt, void* ptr, int n)
{
    priv_from_native64 (ptr, n);

    return ctxt->do_write (
        ctxt,
        ptr,
        sizeof (int64_t) * (uint64_t) (n),
        &(ctxt->output_file_offset));
}

/**************************************/

static exr_result_t
save_attr_uint8 (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;

    rv = save_attr_sz (ctxt, sizeof (uint8_t));
    if (rv == EXR_ERR_SUCCESS)
        rv = ctxt->do_write (
            ctxt, &(a->uc), sizeof (uint8_t), &(ctxt->output_file_offset));
    return rv;
}

/**************************************/

static exr_result_t
save_attr_float (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    float        tmp = a->f;

    rv = save_attr_sz (ctxt, sizeof (float));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 1);
    return rv;
}

/**************************************/

static exr_result_t
save_attr_int (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    int32_t      tmp = a->i;

    rv = save_attr_sz (ctxt, sizeof (int32_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 1);
    return rv;
}

/**************************************/

static exr_result_t
save_attr_double (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    double       tmp = a->d;

    rv = save_attr_sz (ctxt, sizeof (double));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_64 (ctxt, &tmp, 1);
    return rv;
}

/**************************************/

static exr_result_t
save_box2i (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t     rv;
    exr_attr_box2i_t tmp = *(a->box2i);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_box2i_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 4);
    return rv;
}

/**************************************/

static exr_result_t
save_box2f (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t     rv;
    exr_attr_box2f_t tmp = *(a->box2f);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_box2f_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 4);
    return rv;
}

/**************************************/

static exr_result_t
save_chlist (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    size_t       attrsz = 0;
    int32_t      ptype;
    uint8_t      eol;
    uint8_t      flags[4];
    int32_t      samps[2];

    for (int c = 0; c < a->chlist->num_channels; ++c)
    {
        const exr_attr_chlist_entry_t* centry = a->chlist->entries + c;
        attrsz += (size_t) (centry->name.length + 1);
        attrsz += 16;
    }
    // for end of list marker
    attrsz += 1;

    rv = save_attr_sz (ctxt, attrsz);

    for (int c = 0; rv == EXR_ERR_SUCCESS && c < a->chlist->num_channels; ++c)
    {
        const exr_attr_chlist_entry_t* centry = a->chlist->entries + c;

        ptype    = (int32_t) (centry->pixel_type);
        samps[0] = centry->x_sampling;
        samps[1] = centry->y_sampling;

        flags[0] = centry->p_linear;
        flags[1] = flags[2] = flags[3] = 0;

        priv_from_native32 (&ptype, 1);
        priv_from_native32 (samps, 2);

        rv = ctxt->do_write (
            ctxt,
            centry->name.str,
            (uint64_t) (centry->name.length + 1),
            &(ctxt->output_file_offset));
        if (rv != EXR_ERR_SUCCESS) break;
        rv = ctxt->do_write (
            ctxt, &ptype, sizeof (int32_t), &(ctxt->output_file_offset));
        if (rv != EXR_ERR_SUCCESS) break;
        rv = ctxt->do_write (
            ctxt, flags, sizeof (uint8_t) * 4, &(ctxt->output_file_offset));
        if (rv != EXR_ERR_SUCCESS) break;
        rv = ctxt->do_write (
            ctxt, samps, sizeof (int32_t) * 2, &(ctxt->output_file_offset));
    }
    if (rv == EXR_ERR_SUCCESS)
    {
        eol = 0;
        rv  = ctxt->do_write (
            ctxt, &eol, sizeof (uint8_t), &(ctxt->output_file_offset));
    }
    return rv;
}

/**************************************/

static exr_result_t
save_chromaticities (
    struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t              rv;
    exr_attr_chromaticities_t tmp = *(a->chromaticities);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_chromaticities_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 8);
    return rv;
}

/**************************************/

static exr_result_t
save_float_vector (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;

    rv =
        save_attr_sz (ctxt, sizeof (float) * (size_t) (a->floatvector->length));
    if (rv == EXR_ERR_SUCCESS && a->floatvector->length > 0)
    {
        if (a->floatvector->alloc_size > 0)
        {
            /* we own the data, so we can swap it, then swap it back */
            rv = save_attr_32 (
                ctxt,
                EXR_CONST_CAST (void*, a->floatvector->arr),
                a->floatvector->length);
            priv_to_native32 (
                EXR_CONST_CAST (void*, a->floatvector->arr),
                a->floatvector->length);
        }
        else
        {
            /* might be static data, take a copy first */
            float* tmp = ctxt->alloc_fn (
                (size_t) (a->floatvector->length) * sizeof (float));
            if (tmp == NULL)
                return ctxt->standard_error (ctxt, EXR_ERR_OUT_OF_MEMORY);
            memcpy (
                tmp,
                a->floatvector->arr,
                (size_t) (a->floatvector->length) * sizeof (float));
            rv = save_attr_32 (ctxt, tmp, a->floatvector->length);
            ctxt->free_fn (tmp);
        }
    }

    return rv;
}

/**************************************/

static exr_result_t
save_keycode (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t       rv;
    exr_attr_keycode_t tmp = *(a->keycode);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_keycode_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 7);
    return rv;
}

/**************************************/

static exr_result_t
save_m33f (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t    rv;
    exr_attr_m33f_t tmp = *(a->m33f);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_m33f_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 9);
    return rv;
}

/**************************************/

static exr_result_t
save_m33d (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t    rv;
    exr_attr_m33d_t tmp = *(a->m33d);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_m33d_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_64 (ctxt, &tmp, 9);
    return rv;
}

/**************************************/

static exr_result_t
save_m44f (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t    rv;
    exr_attr_m44f_t tmp = *(a->m44f);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_m44f_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 16);
    return rv;
}

/**************************************/

static exr_result_t
save_m44d (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t    rv;
    exr_attr_m44d_t tmp = *(a->m44d);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_m44d_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_64 (ctxt, &tmp, 16);
    return rv;
}

/**************************************/

static exr_result_t
save_preview (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    uint32_t     sizes[2];
    size_t       prevsize = 0;

    sizes[0] = a->preview->width;
    sizes[1] = a->preview->height;
    prevsize = 4 * sizes[0] * sizes[1];

    rv = save_attr_sz (ctxt, sizeof (uint32_t) * 2 + prevsize);

    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, sizes, 2);
    if (rv == EXR_ERR_SUCCESS)
        rv = ctxt->do_write (
            ctxt, a->preview->rgba, prevsize, &(ctxt->output_file_offset));
    return rv;
}

/**************************************/

static exr_result_t
save_rational (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t        rv;
    exr_attr_rational_t tmp = *(a->rational);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_rational_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 2);
    return rv;
}

/**************************************/

static exr_result_t
save_string (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t       rv;
    exr_attr_string_t* tmp = a->string;

    rv = save_attr_sz (ctxt, (size_t) tmp->length);
    if (rv == EXR_ERR_SUCCESS)
        rv = ctxt->do_write (
            ctxt,
            tmp->str,
            (uint64_t) (tmp->length),
            &(ctxt->output_file_offset));
    return rv;
}

/**************************************/

static exr_result_t
save_string_vector (
    struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    size_t       attrsz = 0;

    for (int i = 0; i < a->stringvector->n_strings; ++i)
    {
        attrsz += sizeof (int32_t);
        attrsz += (size_t) a->stringvector->strings[i].length;
    }

    rv = save_attr_sz (ctxt, attrsz);

    for (int i = 0; rv == EXR_ERR_SUCCESS && i < a->stringvector->n_strings;
         ++i)
    {
        const exr_attr_string_t* s = a->stringvector->strings + i;

        rv = save_attr_sz (ctxt, (size_t) s->length);
        if (rv == EXR_ERR_SUCCESS)
            rv = ctxt->do_write (
                ctxt,
                s->str,
                (uint64_t) s->length,
                &(ctxt->output_file_offset));
    }

    return rv;
}

/**************************************/

static exr_result_t
save_tiledesc (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    uint32_t     sizes[2];

    sizes[0] = a->tiledesc->x_size;
    sizes[1] = a->tiledesc->y_size;

    rv = save_attr_sz (ctxt, sizeof (uint32_t) * 2 + 1);

    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, sizes, 2);
    if (rv == EXR_ERR_SUCCESS)
        rv = ctxt->do_write (
            ctxt,
            &(a->tiledesc->level_and_round),
            sizeof (uint8_t),
            &(ctxt->output_file_offset));
    return rv;
}

/**************************************/

static exr_result_t
save_timecode (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t        rv;
    exr_attr_timecode_t tmp = *(a->timecode);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_timecode_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 2);
    return rv;
}

/**************************************/

static exr_result_t
save_v2i (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t   rv;
    exr_attr_v2i_t tmp = *(a->v2i);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_v2i_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 2);
    return rv;
}

/**************************************/

static exr_result_t
save_v2f (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t   rv;
    exr_attr_v2f_t tmp = *(a->v2f);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_v2f_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 2);
    return rv;
}

/**************************************/

static exr_result_t
save_v2d (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t   rv;
    exr_attr_v2d_t tmp = *(a->v2d);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_v2d_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_64 (ctxt, &tmp, 2);
    return rv;
}

/**************************************/

static exr_result_t
save_v3i (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t   rv;
    exr_attr_v3i_t tmp = *(a->v3i);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_v3i_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 3);
    return rv;
}

/**************************************/

static exr_result_t
save_v3f (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t   rv;
    exr_attr_v3f_t tmp = *(a->v3f);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_v3f_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_32 (ctxt, &tmp, 3);
    return rv;
}

/**************************************/

static exr_result_t
save_v3d (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t   rv;
    exr_attr_v3d_t tmp = *(a->v3d);

    rv = save_attr_sz (ctxt, sizeof (exr_attr_v3d_t));
    if (rv == EXR_ERR_SUCCESS) rv = save_attr_64 (ctxt, &tmp, 3);
    return rv;
}

/**************************************/

static exr_result_t
save_opaque (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;
    int32_t      sz    = 0;
    void*        pdata = NULL;

    rv =
        exr_attr_opaquedata_pack ((exr_context_t) ctxt, a->opaque, &sz, &pdata);
    if (rv != EXR_ERR_SUCCESS) return rv;

    rv = save_attr_sz (ctxt, (uint64_t) sz);
    if (rv == EXR_ERR_SUCCESS && sz > 0)
        rv = ctxt->do_write (
            ctxt, pdata, (uint64_t) sz, &(ctxt->output_file_offset));
    return rv;
}

/**************************************/

static exr_result_t
save_attr (struct _internal_exr_context* ctxt, const exr_attribute_t* a)
{
    exr_result_t rv;

    rv = ctxt->do_write (
        ctxt, a->name, a->name_length + 1, &(ctxt->output_file_offset));
    if (rv != EXR_ERR_SUCCESS) return rv;
    rv = ctxt->do_write (
        ctxt,
        a->type_name,
        a->type_name_length + 1,
        &(ctxt->output_file_offset));
    if (rv != EXR_ERR_SUCCESS) return rv;

    switch (a->type)
    {
        case EXR_ATTR_BOX2I: rv = save_box2i (ctxt, a); break;
        case EXR_ATTR_BOX2F: rv = save_box2f (ctxt, a); break;
        case EXR_ATTR_CHLIST: rv = save_chlist (ctxt, a); break;
        case EXR_ATTR_CHROMATICITIES: rv = save_chromaticities (ctxt, a); break;
        case EXR_ATTR_COMPRESSION: rv = save_attr_uint8 (ctxt, a); break;
        case EXR_ATTR_DOUBLE: rv = save_attr_double (ctxt, a); break;
        case EXR_ATTR_ENVMAP: rv = save_attr_uint8 (ctxt, a); break;
        case EXR_ATTR_FLOAT: rv = save_attr_float (ctxt, a); break;
        case EXR_ATTR_FLOAT_VECTOR: rv = save_float_vector (ctxt, a); break;
        case EXR_ATTR_INT: rv = save_attr_int (ctxt, a); break;
        case EXR_ATTR_KEYCODE: rv = save_keycode (ctxt, a); break;
        case EXR_ATTR_LINEORDER: rv = save_attr_uint8 (ctxt, a); break;
        case EXR_ATTR_M33F: rv = save_m33f (ctxt, a); break;
        case EXR_ATTR_M33D: rv = save_m33d (ctxt, a); break;
        case EXR_ATTR_M44F: rv = save_m44f (ctxt, a); break;
        case EXR_ATTR_M44D: rv = save_m44d (ctxt, a); break;
        case EXR_ATTR_PREVIEW: rv = save_preview (ctxt, a); break;
        case EXR_ATTR_RATIONAL: rv = save_rational (ctxt, a); break;
        case EXR_ATTR_STRING: rv = save_string (ctxt, a); break;
        case EXR_ATTR_STRING_VECTOR: rv = save_string_vector (ctxt, a); break;
        case EXR_ATTR_TILEDESC: rv = save_tiledesc (ctxt, a); break;
        case EXR_ATTR_TIMECODE: rv = save_timecode (ctxt, a); break;
        case EXR_ATTR_V2I: rv = save_v2i (ctxt, a); break;
        case EXR_ATTR_V2F: rv = save_v2f (ctxt, a); break;
        case EXR_ATTR_V2D: rv = save_v2d (ctxt, a); break;
        case EXR_ATTR_V3I: rv = save_v3i (ctxt, a); break;
        case EXR_ATTR_V3F: rv = save_v3f (ctxt, a); break;
        case EXR_ATTR_V3D: rv = save_v3d (ctxt, a); break;
        case EXR_ATTR_OPAQUE: rv = save_opaque (ctxt, a); break;

        case EXR_ATTR_UNKNOWN:
        case EXR_ATTR_LAST_KNOWN_TYPE:
        default: rv = ctxt->standard_error (ctxt, EXR_ERR_INVALID_ATTR); break;
    }
    return rv;
}

/**************************************/

exr_result_t
internal_exr_write_header (struct _internal_exr_context* ctxt)
{
    exr_result_t rv;
    uint32_t     magic_and_version[2];
    uint32_t     flags;
    uint8_t      next_byte;

    flags = 2; // EXR_VERSION
    if (ctxt->is_multipart) flags |= EXR_MULTI_PART_FLAG;
    if (ctxt->max_name_length > EXR_SHORTNAME_MAXLEN)
        flags |= EXR_LONG_NAMES_FLAG;
    if (ctxt->has_nonimage_data) flags |= EXR_NON_IMAGE_FLAG;
    if (ctxt->is_singlepart_tiled) flags |= EXR_TILED_FLAG;

    magic_and_version[0] = 20000630;
    magic_and_version[1] = flags;

    priv_from_native32 (magic_and_version, 2);

    rv = ctxt->do_write (
        ctxt,
        magic_and_version,
        sizeof (uint32_t) * 2,
        &(ctxt->output_file_offset));
    if (rv != EXR_ERR_SUCCESS) return rv;

    for (int p = 0; rv == EXR_ERR_SUCCESS && p < ctxt->num_parts; ++p)
    {
        struct _internal_exr_part* curp = ctxt->parts[p];
        if (ctxt->legacy_header)
        {
            for (int a = 0; a < curp->attributes.num_attributes; ++a)
            {
                exr_attribute_t* curattr = curp->attributes.sorted_entries[a];
                if (0 == (flags & (EXR_MULTI_PART_FLAG | EXR_NON_IMAGE_FLAG)) &&
                    1 == ctxt->num_parts)
                {
                    if (0 == strcmp (curattr->name, "type") ||
                        0 == strcmp (curattr->name, "name"))
                    {
                        /* old file wouldn't have had this */
                        continue;
                    }
                }
                rv = save_attr (ctxt, curattr);
                if (rv != EXR_ERR_SUCCESS) break;
            }
        }
        else
        {
            for (int a = 0; a < curp->attributes.num_attributes; ++a)
            {
                rv = save_attr (ctxt, curp->attributes.entries[a]);
                if (rv != EXR_ERR_SUCCESS) break;
            }
        }

        /* indicate this part is finished */
        if (rv == EXR_ERR_SUCCESS)
        {
            next_byte = 0;
            rv        = ctxt->do_write (
                ctxt,
                &next_byte,
                sizeof (uint8_t),
                &(ctxt->output_file_offset));
        }
    }

    /* for multipart write a double terminator at the end */
    if (rv == EXR_ERR_SUCCESS && ctxt->is_multipart)
    {
        next_byte = 0;
        rv        = ctxt->do_write (
            ctxt, &next_byte, sizeof (uint8_t), &(ctxt->output_file_offset));
    }

    return rv;
}
