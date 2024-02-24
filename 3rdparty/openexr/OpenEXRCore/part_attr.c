/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "openexr_part.h"

#include "internal_attr.h"
#include "internal_constants.h"
#include "internal_file.h"
#include "internal_structs.h"

#include <stdio.h>
#include <string.h>

/**************************************/

exr_result_t
exr_get_attribute_count (
    exr_const_context_t ctxt, int part_index, int32_t* count)
{
    int32_t cnt;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    cnt = part->attributes.num_attributes;
    EXR_UNLOCK_WRITE (pctxt);

    if (!count) return pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT);
    *count = cnt;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_get_attribute_by_index (
    exr_const_context_t         ctxt,
    int                         part_index,
    exr_attr_list_access_mode_t mode,
    int32_t                     idx,
    const exr_attribute_t**     outattr)
{
    exr_attribute_t** srclist;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (!outattr)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));

    if (idx < 0 || idx >= part->attributes.num_attributes)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_ARGUMENT_OUT_OF_RANGE));

    if (mode == EXR_ATTR_LIST_SORTED_ORDER)
        srclist = part->attributes.sorted_entries;
    else if (mode == EXR_ATTR_LIST_FILE_ORDER)
        srclist = part->attributes.entries;
    else
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));

    *outattr = srclist[idx];
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
}

/**************************************/

exr_result_t
exr_get_attribute_by_name (
    exr_const_context_t     ctxt,
    int                     part_index,
    const char*             name,
    const exr_attribute_t** outattr)
{
    exr_attribute_t* tmpptr;
    exr_result_t     rv;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (!outattr)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));

    rv = exr_attr_list_find_by_name (
        EXR_CONST_CAST (exr_context_t, ctxt),
        EXR_CONST_CAST (exr_attribute_list_t*, &(part->attributes)),
        name,
        &tmpptr);
    if (rv == EXR_ERR_SUCCESS) *outattr = tmpptr;
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_attribute_list (
    exr_const_context_t         ctxt,
    int                         part_index,
    exr_attr_list_access_mode_t mode,
    int32_t*                    count,
    const exr_attribute_t**     outlist)
{
    exr_attribute_t** srclist;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (!count)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));

    if (mode == EXR_ATTR_LIST_SORTED_ORDER)
        srclist = part->attributes.sorted_entries;
    else if (mode == EXR_ATTR_LIST_FILE_ORDER)
        srclist = part->attributes.entries;
    else
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));

    if (outlist && *count >= part->attributes.num_attributes)
        memcpy (
            EXR_CONST_CAST (exr_attribute_t**, outlist),
            srclist,
            sizeof (exr_attribute_t*) *
                (size_t) part->attributes.num_attributes);
    *count = part->attributes.num_attributes;
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
}

/**************************************/

exr_result_t
exr_attr_declare_by_type (
    exr_context_t     ctxt,
    int               part_index,
    const char*       name,
    const char*       type,
    exr_attribute_t** outattr)
{
    exr_result_t rv;
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    rv = exr_attr_list_add_by_type (
        ctxt, &(part->attributes), name, type, 0, NULL, outattr);
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_attr_declare (
    exr_context_t        ctxt,
    int                  part_index,
    const char*          name,
    exr_attribute_type_t type,
    exr_attribute_t**    outattr)
{
    exr_result_t rv;
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    rv = exr_attr_list_add (
        ctxt, &(part->attributes), name, type, 0, NULL, outattr);
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_initialize_required_attr (
    exr_context_t           ctxt,
    int                     part_index,
    const exr_attr_box2i_t* displayWindow,
    const exr_attr_box2i_t* dataWindow,
    float                   pixelaspectratio,
    const exr_attr_v2f_t*   screenWindowCenter,
    float                   screenWindowWidth,
    exr_lineorder_t         lineorder,
    exr_compression_t       ctype)
{
    exr_result_t rv;

    rv = exr_set_compression (ctxt, part_index, ctype);
    if (rv != EXR_ERR_SUCCESS) return rv;
    rv = exr_set_data_window (ctxt, part_index, dataWindow);
    if (rv != EXR_ERR_SUCCESS) return rv;
    rv = exr_set_display_window (ctxt, part_index, displayWindow);
    if (rv != EXR_ERR_SUCCESS) return rv;
    rv = exr_set_lineorder (ctxt, part_index, lineorder);
    if (rv != EXR_ERR_SUCCESS) return rv;
    rv = exr_set_pixel_aspect_ratio (ctxt, part_index, pixelaspectratio);
    if (rv != EXR_ERR_SUCCESS) return rv;
    rv = exr_set_screen_window_center (ctxt, part_index, screenWindowCenter);
    if (rv != EXR_ERR_SUCCESS) return rv;

    return exr_set_screen_window_width (ctxt, part_index, screenWindowWidth);
}

/**************************************/

exr_result_t
exr_initialize_required_attr_simple (
    exr_context_t     ctxt,
    int               part_index,
    int32_t           width,
    int32_t           height,
    exr_compression_t ctype)
{
    exr_attr_box2i_t dispWindow = {
        .min = {.x = 0, .y = 0}, .max = {.x = (width - 1), .y = (height - 1)}};
    exr_attr_v2f_t swc = {.x = 0.f, .y = 0.f};
    return exr_initialize_required_attr (
        ctxt,
        part_index,
        &dispWindow,
        &dispWindow,
        1.f,
        &swc,
        1.f,
        EXR_LINEORDER_INCREASING_Y,
        ctype);
}

/**************************************/

static exr_result_t
copy_attr (
    exr_context_t                 ctxt,
    struct _internal_exr_context* pctxt,
    struct _internal_exr_part*    part,
    const exr_attribute_t*        srca,
    int*                          update_tiles)
{
    exr_result_t         rv    = EXR_ERR_UNKNOWN;
    const char*          aname = srca->name;
    exr_attribute_t*     attr  = NULL;
    exr_attribute_type_t type  = srca->type;
    switch (aname[0])
    {
        case 'c':
            if (0 == strcmp (aname, EXR_REQ_CHANNELS_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_CHANNELS_STR,
                    type,
                    0,
                    NULL,
                    &(part->channels));
                attr = part->channels;
            }
            else if (0 == strcmp (aname, EXR_REQ_COMP_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_COMP_STR,
                    type,
                    0,
                    NULL,
                    &(part->compression));
                attr = part->compression;
                if (rv == EXR_ERR_SUCCESS)
                    part->comp_type = (exr_compression_t) srca->uc;
            }
            else if (0 == strcmp (aname, EXR_REQ_CHUNK_COUNT_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_CHUNK_COUNT_STR,
                    type,
                    0,
                    NULL,
                    &(part->chunkCount));
                attr = part->chunkCount;
            }
            break;
        case 'd':
            if (0 == strcmp (aname, EXR_REQ_DATA_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_DATA_STR,
                    type,
                    0,
                    NULL,
                    &(part->dataWindow));
                attr = part->dataWindow;
                if (rv == EXR_ERR_SUCCESS) part->data_window = *(srca->box2i);
                *update_tiles = 1;
            }
            else if (0 == strcmp (aname, EXR_REQ_DISP_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_DISP_STR,
                    type,
                    0,
                    NULL,
                    &(part->displayWindow));
                attr = part->displayWindow;
                if (rv == EXR_ERR_SUCCESS)
                    part->display_window = *(srca->box2i);
            }
            break;
        case 'l':
            if (0 == strcmp (aname, EXR_REQ_LO_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_LO_STR,
                    type,
                    0,
                    NULL,
                    &(part->lineOrder));
                attr = part->lineOrder;
                if (rv == EXR_ERR_SUCCESS)
                    part->lineorder = (exr_lineorder_t) srca->uc;
            }
            break;
        case 'n':
            if (0 == strcmp (aname, EXR_REQ_NAME_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_NAME_STR,
                    type,
                    0,
                    NULL,
                    &(part->name));
                attr = part->name;
            }
            break;
        case 'p':
            if (0 == strcmp (aname, EXR_REQ_PAR_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_PAR_STR,
                    type,
                    0,
                    NULL,
                    &(part->pixelAspectRatio));
                attr = part->pixelAspectRatio;
            }
            break;
        case 's':
            if (0 == strcmp (aname, EXR_REQ_SCR_WC_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_SCR_WC_STR,
                    type,
                    0,
                    NULL,
                    &(part->screenWindowCenter));
                attr = part->screenWindowCenter;
            }
            else if (0 == strcmp (aname, EXR_REQ_SCR_WW_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_SCR_WW_STR,
                    type,
                    0,
                    NULL,
                    &(part->screenWindowWidth));
                attr = part->screenWindowWidth;
            }
            break;
        case 't':
            if (0 == strcmp (aname, EXR_REQ_TILES_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_TILES_STR,
                    type,
                    0,
                    NULL,
                    &(part->tiles));
                attr          = part->tiles;
                *update_tiles = 1;
            }
            else if (0 == strcmp (aname, EXR_REQ_TYPE_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_TYPE_STR,
                    type,
                    0,
                    NULL,
                    &(part->type));
                attr = part->type;
            }
            break;
        case 'v':
            if (0 == strcmp (aname, EXR_REQ_VERSION_STR))
            {
                rv = exr_attr_list_add_static_name (
                    ctxt,
                    &(part->attributes),
                    EXR_REQ_VERSION_STR,
                    type,
                    0,
                    NULL,
                    &(part->version));
                attr = part->version;
            }
            break;
        default: break;
    }

    if (rv == EXR_ERR_UNKNOWN && !attr)
    {
        rv = exr_attr_list_add (
            ctxt, &(part->attributes), aname, type, 0, NULL, &(attr));
    }

    if (rv != EXR_ERR_SUCCESS) return rv;

    switch (type)
    {
        case EXR_ATTR_BOX2I: *(attr->box2i) = *(srca->box2i); break;
        case EXR_ATTR_BOX2F: *(attr->box2f) = *(srca->box2f); break;
        case EXR_ATTR_CHLIST:
            rv = exr_attr_chlist_duplicate (ctxt, attr->chlist, srca->chlist);
            break;
        case EXR_ATTR_CHROMATICITIES:
            *(attr->chromaticities) = *(srca->chromaticities);
            break;
        case EXR_ATTR_COMPRESSION: attr->uc = srca->uc; break;
        case EXR_ATTR_DOUBLE: attr->d = srca->d; break;
        case EXR_ATTR_ENVMAP: attr->uc = srca->uc; break;
        case EXR_ATTR_FLOAT: attr->f = srca->f; break;
        case EXR_ATTR_FLOAT_VECTOR:
            rv = exr_attr_float_vector_create (
                ctxt,
                attr->floatvector,
                srca->floatvector->arr,
                srca->floatvector->length);
            break;
        case EXR_ATTR_INT: attr->i = srca->i; break;
        case EXR_ATTR_KEYCODE: *(attr->keycode) = *(srca->keycode); break;
        case EXR_ATTR_LINEORDER: attr->uc = srca->uc; break;
        case EXR_ATTR_M33F: *(attr->m33f) = *(srca->m33f); break;
        case EXR_ATTR_M33D: *(attr->m33d) = *(srca->m33d); break;
        case EXR_ATTR_M44F: *(attr->m44f) = *(srca->m44f); break;
        case EXR_ATTR_M44D: *(attr->m44d) = *(srca->m44d); break;
        case EXR_ATTR_PREVIEW:
            rv = exr_attr_preview_create (
                ctxt,
                attr->preview,
                srca->preview->width,
                srca->preview->height,
                srca->preview->rgba);
            break;
        case EXR_ATTR_RATIONAL: *(attr->rational) = *(srca->rational); break;
        case EXR_ATTR_STRING:
            rv = exr_attr_string_create_with_length (
                ctxt, attr->string, srca->string->str, srca->string->length);
            break;
        case EXR_ATTR_STRING_VECTOR:
            rv = exr_attr_string_vector_copy (
                ctxt, attr->stringvector, srca->stringvector);
            break;
        case EXR_ATTR_TILEDESC: *(attr->tiledesc) = *(srca->tiledesc); break;
        case EXR_ATTR_TIMECODE: *(attr->timecode) = *(srca->timecode); break;
        case EXR_ATTR_V2I: *(attr->v2i) = *(srca->v2i); break;
        case EXR_ATTR_V2F: *(attr->v2f) = *(srca->v2f); break;
        case EXR_ATTR_V2D: *(attr->v2d) = *(srca->v2d); break;
        case EXR_ATTR_V3I: *(attr->v3i) = *(srca->v3i); break;
        case EXR_ATTR_V3F: *(attr->v3f) = *(srca->v3f); break;
        case EXR_ATTR_V3D: *(attr->v3d) = *(srca->v3d); break;
        case EXR_ATTR_OPAQUE:
            rv = exr_attr_opaquedata_copy (ctxt, attr->opaque, srca->opaque);
            break;
        case EXR_ATTR_UNKNOWN:
        case EXR_ATTR_LAST_KNOWN_TYPE:
        default:
            rv = pctxt->standard_error (pctxt, EXR_ERR_INVALID_ATTR);
            break;
    }

    if (rv != EXR_ERR_SUCCESS)
        exr_attr_list_remove (ctxt, &(part->attributes), attr);

    return rv;
}

/**************************************/

exr_result_t
exr_copy_unset_attributes (
    exr_context_t       ctxt,
    int                 part_index,
    exr_const_context_t source,
    int                 src_part_index)
{
    exr_result_t                        rv;
    const struct _internal_exr_context* srcctxt = EXR_CCTXT (source);
    struct _internal_exr_part*          srcpart;
    int                                 update_tiles = 0;
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (!srcctxt)
        return EXR_UNLOCK_AND_RETURN_PCTXT (EXR_ERR_MISSING_CONTEXT_ARG);
    if (srcctxt != pctxt) EXR_LOCK (srcctxt);

    if (src_part_index < 0 || src_part_index >= srcctxt->num_parts)
    {
        if (srcctxt != pctxt) EXR_UNLOCK (srcctxt);
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_ARGUMENT_OUT_OF_RANGE,
            "Source part index (%d) out of range",
            src_part_index));
    }

    srcpart = srcctxt->parts[src_part_index];

    rv = EXR_ERR_SUCCESS;
    for (int a = 0;
         rv == EXR_ERR_SUCCESS && a < srcpart->attributes.num_attributes;
         ++a)
    {
        const exr_attribute_t* srca = srcpart->attributes.entries[a];
        exr_attribute_t*       attr = NULL;

        rv = exr_attr_list_find_by_name (
            ctxt,
            (exr_attribute_list_t*) &(part->attributes),
            srca->name,
            &attr);
        if (rv == EXR_ERR_NO_ATTR_BY_NAME)
        {
            rv = copy_attr (ctxt, pctxt, part, srca, &update_tiles);
        }
        else { rv = EXR_ERR_SUCCESS; }
    }

    if (update_tiles)
        rv = internal_exr_compute_tile_information (pctxt, part, 1);

    if (srcctxt != pctxt) EXR_UNLOCK (srcctxt);
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

#define REQ_ATTR_GET_IMPL(name, entry, t)                                      \
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);            \
    if (!out)                                                                  \
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (         \
            pctxt, EXR_ERR_INVALID_ARGUMENT, "NULL output for '%s'", #name));  \
    if (part->name)                                                            \
    {                                                                          \
        if (part->name->type != t)                                             \
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (     \
                pctxt,                                                         \
                EXR_ERR_FILE_BAD_HEADER,                                       \
                "Invalid required attribute type '%s' for '%s'",               \
                part->name->type_name,                                         \
                #name));                                                       \
        *out = part->name->entry;                                              \
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);            \
    }                                                                          \
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_NO_ATTR_BY_NAME)

#define REQ_ATTR_GET_IMPL_DEREF(name, entry, t)                                \
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);            \
    if (!out)                                                                  \
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (         \
            pctxt, EXR_ERR_INVALID_ARGUMENT, "NULL output for '%s'", #name));  \
    if (part->name)                                                            \
    {                                                                          \
        if (part->name->type != t)                                             \
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (     \
                pctxt,                                                         \
                EXR_ERR_FILE_BAD_HEADER,                                       \
                "Invalid required attribute type '%s' for '%s'",               \
                part->name->type_name,                                         \
                #name));                                                       \
        *out = *(part->name->entry);                                           \
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);            \
    }                                                                          \
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_NO_ATTR_BY_NAME)

#define REQ_ATTR_FIND_CREATE(name, t)                                          \
    exr_attribute_t* attr = NULL;                                              \
    exr_result_t     rv   = EXR_ERR_SUCCESS;                                   \
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);           \
    if (pctxt->mode == EXR_CONTEXT_READ)                                       \
        return EXR_UNLOCK_AND_RETURN_PCTXT (                                   \
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));            \
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)                               \
        return EXR_UNLOCK_AND_RETURN_PCTXT (                                   \
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));       \
    if (!part->name)                                                           \
    {                                                                          \
        rv = exr_attr_list_add (                                               \
            ctxt, &(part->attributes), #name, t, 0, NULL, &(part->name));      \
    }                                                                          \
    else if (part->name->type != t)                                            \
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (               \
            pctxt,                                                             \
            EXR_ERR_FILE_BAD_HEADER,                                           \
            "Invalid required attribute type '%s' for '%s'",                   \
            part->name->type_name,                                             \
            #name));                                                           \
    attr = part->name

/**************************************/

exr_result_t
exr_get_channels (
    exr_const_context_t ctxt, int part_index, const exr_attr_chlist_t** out)
{
    REQ_ATTR_GET_IMPL (channels, chlist, EXR_ATTR_CHLIST);
}

/**************************************/

exr_result_t
exr_add_channel (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    exr_pixel_type_t           ptype,
    exr_perceptual_treatment_t islinear,
    int32_t                    xsamp,
    int32_t                    ysamp)
{
    REQ_ATTR_FIND_CREATE (channels, EXR_ATTR_CHLIST);
    if (rv == EXR_ERR_SUCCESS)
    {
        rv = exr_attr_chlist_add (
            ctxt, attr->chlist, name, ptype, islinear, xsamp, ysamp);
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_set_channels (
    exr_context_t ctxt, int part_index, const exr_attr_chlist_t* channels)
{
    if (!channels)
        return EXR_CTXT (ctxt)->report_error (
            EXR_CTXT (ctxt),
            EXR_ERR_INVALID_ARGUMENT,
            "No channels provided for channel list");

    {
        REQ_ATTR_FIND_CREATE (channels, EXR_ATTR_CHLIST);
        if (rv == EXR_ERR_SUCCESS)
        {
            exr_attr_chlist_t clist;

            rv = exr_attr_chlist_duplicate (ctxt, &clist, channels);
            if (rv != EXR_ERR_SUCCESS) return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

            exr_attr_chlist_destroy (ctxt, attr->chlist);
            *(attr->chlist) = clist;
        }
        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    }
}

/**************************************/

exr_result_t
exr_get_compression (
    exr_const_context_t ctxt, int part_index, exr_compression_t* out)
{
    REQ_ATTR_GET_IMPL (compression, uc, EXR_ATTR_COMPRESSION);
}

/**************************************/

exr_result_t
exr_set_compression (
    exr_context_t ctxt, int part_index, exr_compression_t ctype)
{
    REQ_ATTR_FIND_CREATE (compression, EXR_ATTR_COMPRESSION);
    if (rv == EXR_ERR_SUCCESS)
    {
        attr->uc        = (uint8_t) ctype;
        part->comp_type = ctype;
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_data_window (
    exr_const_context_t ctxt, int part_index, exr_attr_box2i_t* out)
{
    REQ_ATTR_GET_IMPL_DEREF (dataWindow, box2i, EXR_ATTR_BOX2I);
}

/**************************************/

exr_result_t
exr_set_data_window (
    exr_context_t ctxt, int part_index, const exr_attr_box2i_t* dw)
{
    if (!dw)
        return EXR_CTXT (ctxt)->report_error (
            EXR_CTXT (ctxt),
            EXR_ERR_INVALID_ARGUMENT,
            "Missing value for data window assignment");

    {
        REQ_ATTR_FIND_CREATE (dataWindow, EXR_ATTR_BOX2I);

        if (rv == EXR_ERR_SUCCESS)
        {
            *(attr->box2i)    = *dw;
            part->data_window = *dw;

            rv = internal_exr_compute_tile_information (pctxt, part, 1);
        }

        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    }
}

/**************************************/

exr_result_t
exr_get_display_window (
    exr_const_context_t ctxt, int part_index, exr_attr_box2i_t* out)
{
    REQ_ATTR_GET_IMPL_DEREF (displayWindow, box2i, EXR_ATTR_BOX2I);
}

/**************************************/

exr_result_t
exr_set_display_window (
    exr_context_t ctxt, int part_index, const exr_attr_box2i_t* dw)
{
    if (!dw)
        return EXR_CTXT (ctxt)->report_error (
            EXR_CTXT (ctxt),
            EXR_ERR_INVALID_ARGUMENT,
            "Missing value for data window assignment");

    {
        REQ_ATTR_FIND_CREATE (displayWindow, EXR_ATTR_BOX2I);
        if (rv == EXR_ERR_SUCCESS)
        {
            *(attr->box2i)       = *dw;
            part->display_window = *dw;
        }

        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    }
}

/**************************************/

exr_result_t
exr_get_lineorder (
    exr_const_context_t ctxt, int part_index, exr_lineorder_t* out)
{
    REQ_ATTR_GET_IMPL (lineOrder, uc, EXR_ATTR_LINEORDER);
}

/**************************************/

exr_result_t
exr_set_lineorder (exr_context_t ctxt, int part_index, exr_lineorder_t lo)
{
    if (lo >= EXR_LINEORDER_LAST_TYPE)
        return EXR_CTXT (ctxt)->print_error (
            EXR_CTXT (ctxt),
            EXR_ERR_ARGUMENT_OUT_OF_RANGE,
            "'lineOrder' value for line order (%d) out of range (%d - %d)",
            (int) lo,
            0,
            (int) EXR_LINEORDER_LAST_TYPE);

    {
        REQ_ATTR_FIND_CREATE (lineOrder, EXR_ATTR_LINEORDER);
        if (rv == EXR_ERR_SUCCESS)
        {
            attr->uc        = (uint8_t) lo;
            part->lineorder = lo;
        }

        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    }
}

/**************************************/

exr_result_t
exr_get_pixel_aspect_ratio (
    exr_const_context_t ctxt, int part_index, float* out)
{
    REQ_ATTR_GET_IMPL (pixelAspectRatio, f, EXR_ATTR_FLOAT);
}

/**************************************/

exr_result_t
exr_set_pixel_aspect_ratio (exr_context_t ctxt, int part_index, float par)
{
    REQ_ATTR_FIND_CREATE (pixelAspectRatio, EXR_ATTR_FLOAT);
    if (rv == EXR_ERR_SUCCESS) attr->f = par;
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_screen_window_center (
    exr_const_context_t ctxt, int part_index, exr_attr_v2f_t* out)
{
    REQ_ATTR_GET_IMPL_DEREF (screenWindowCenter, v2f, EXR_ATTR_V2F);
}

/**************************************/

exr_result_t
exr_set_screen_window_center (
    exr_context_t ctxt, int part_index, const exr_attr_v2f_t* swc)
{
    REQ_ATTR_FIND_CREATE (screenWindowCenter, EXR_ATTR_V2F);
    if (rv != EXR_ERR_SUCCESS) return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    if (!swc)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Missing value for data window assignment"));

    attr->v2f->x = swc->x;
    attr->v2f->y = swc->y;
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_screen_window_width (
    exr_const_context_t ctxt, int part_index, float* out)
{
    REQ_ATTR_GET_IMPL (screenWindowWidth, f, EXR_ATTR_FLOAT);
}

/**************************************/

exr_result_t
exr_set_screen_window_width (exr_context_t ctxt, int part_index, float ssw)
{
    REQ_ATTR_FIND_CREATE (screenWindowWidth, EXR_ATTR_FLOAT);
    if (rv == EXR_ERR_SUCCESS) attr->f = ssw;
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_tile_descriptor (
    exr_const_context_t    ctxt,
    int                    part_index,
    uint32_t*              xsize,
    uint32_t*              ysize,
    exr_tile_level_mode_t* level,
    exr_tile_round_mode_t* round)
{
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (part->tiles)
    {
        const exr_attr_tiledesc_t* out = part->tiles->tiledesc;

        if (part->tiles->type != EXR_ATTR_TILEDESC)
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_FILE_BAD_HEADER,
                "Invalid required attribute type '%s' for 'tiles'",
                part->tiles->type_name));

        if (xsize) *xsize = out->x_size;
        if (ysize) *ysize = out->y_size;
        if (level) *level = EXR_GET_TILE_LEVEL_MODE (*out);
        if (round) *round = EXR_GET_TILE_ROUND_MODE (*out);
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_NO_ATTR_BY_NAME);
}

/**************************************/

exr_result_t
exr_set_tile_descriptor (
    exr_context_t         ctxt,
    int                   part_index,
    uint32_t              x_size,
    uint32_t              y_size,
    exr_tile_level_mode_t level_mode,
    exr_tile_round_mode_t round_mode)
{
    exr_result_t     rv   = EXR_ERR_SUCCESS;
    exr_attribute_t* attr = NULL;
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    if (pctxt->mode == EXR_CONTEXT_READ)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));
    if (part->storage_mode == EXR_STORAGE_SCANLINE ||
        part->storage_mode == EXR_STORAGE_DEEP_SCANLINE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt,
            EXR_ERR_TILE_SCAN_MIXEDAPI,
            "Attempt to set tile descriptor on scanline part"));

    if (!part->tiles)
    {
        rv = exr_attr_list_add (
            ctxt,
            &(part->attributes),
            "tiles",
            EXR_ATTR_TILEDESC,
            0,
            NULL,
            &(part->tiles));
    }
    else if (part->tiles->type != EXR_ATTR_TILEDESC)
    {
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_FILE_BAD_HEADER,
            "Invalid required attribute type '%s' for '%s'",
            part->tiles->type_name,
            "tiles"));
    }

    attr = part->tiles;

    if (rv == EXR_ERR_SUCCESS)
    {
        attr->tiledesc->x_size = x_size;
        attr->tiledesc->y_size = y_size;
        attr->tiledesc->level_and_round =
            EXR_PACK_TILE_LEVEL_ROUND (level_mode, round_mode);

        rv = internal_exr_compute_tile_information (pctxt, part, 1);
    }

    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_name (exr_const_context_t ctxt, int part_index, const char** out)
{
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    if (!out)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt, EXR_ERR_INVALID_ARGUMENT, "NULL output for 'name'"));

    if (part->name)
    {
        if (part->name->type != EXR_ATTR_STRING)
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_FILE_BAD_HEADER,
                "Invalid required attribute type '%s' for 'name'",
                part->name->type_name));
        *out = part->name->string->str;
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_NO_ATTR_BY_NAME);
}

exr_result_t
exr_set_name (exr_context_t ctxt, int part_index, const char* val)
{
    size_t bytes;
    REQ_ATTR_FIND_CREATE (name, EXR_ATTR_STRING);

    if (!val || val[0] == '\0')
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid string passed trying to set 'name'"));

    bytes = strlen (val);

    if (bytes >= (size_t) INT32_MAX)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "String too large to store (%" PRIu64 " bytes) into 'name'",
            (uint64_t) bytes));

    if (rv == EXR_ERR_SUCCESS)
    {
        if (attr->string->length == (int32_t) bytes &&
            attr->string->alloc_size > 0)
        {
            /* we own the string... */
            memcpy (EXR_CONST_CAST (void*, attr->string->str), val, bytes);
        }
        else if (pctxt->mode != EXR_CONTEXT_WRITE)
        {
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MODIFY_SIZE_CHANGE,
                "Existing string 'name' has length %d, requested %d, unable to change",
                attr->string->length,
                (int32_t) bytes));
        }
        else
        {
            rv = exr_attr_string_set_with_length (
                ctxt, attr->string, val, (int32_t) bytes);
        }
    }

    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_version (exr_const_context_t ctxt, int part_index, int32_t* out)
{
    REQ_ATTR_GET_IMPL (version, i, EXR_ATTR_INT);
}

/**************************************/

exr_result_t
exr_set_version (exr_context_t ctxt, int part_index, int32_t val)
{
    /* version number for deep data, expect 1 */
    if (val <= 0 || val > 1) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;

    {
        REQ_ATTR_FIND_CREATE (version, EXR_ATTR_INT);
        if (rv == EXR_ERR_SUCCESS) { attr->i = val; }
        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    }
}

/**************************************/

exr_result_t
exr_set_chunk_count (exr_context_t ctxt, int part_index, int32_t val)
{
    REQ_ATTR_FIND_CREATE (chunkCount, EXR_ATTR_INT);
    if (rv == EXR_ERR_SUCCESS)
    {
        attr->i           = val;
        part->chunk_count = val;
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

#define ATTR_FIND_ATTR(t, entry)                                               \
    exr_attribute_t* attr;                                                     \
    exr_result_t     rv;                                                       \
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);            \
    if (!name || name[0] == '\0')                                              \
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->report_error (        \
            pctxt,                                                             \
            EXR_ERR_INVALID_ARGUMENT,                                          \
            "Invalid name for " #entry " attribute query"));                   \
    rv = exr_attr_list_find_by_name (                                          \
        ctxt,                                                                  \
        EXR_CONST_CAST (exr_attribute_list_t*, &(part->attributes)),           \
        name,                                                                  \
        &attr);                                                                \
    if (rv != EXR_ERR_SUCCESS) return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv);  \
    if (attr->type != t)                                                       \
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (             \
        pctxt,                                                                 \
        EXR_ERR_ATTR_TYPE_MISMATCH,                                            \
        "'%s' requested type '" #entry                                         \
        "', but stored attributes is type '%s'",                               \
        name,                                                                  \
        attr->type_name))

#define ATTR_GET_IMPL(t, entry)                                                \
    ATTR_FIND_ATTR (t, entry);                                                 \
    if (!out)                                                                  \
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (         \
            pctxt, EXR_ERR_INVALID_ARGUMENT, "NULL output for '%s'", name));   \
    *out = attr->entry;                                                        \
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv)

#define ATTR_GET_IMPL_DEREF(t, entry)                                          \
    ATTR_FIND_ATTR (t, entry);                                                 \
    if (!out)                                                                  \
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (         \
            pctxt, EXR_ERR_INVALID_ARGUMENT, "NULL output for '%s'", name));   \
    *out = *(attr->entry);                                                     \
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv)

#define ATTR_FIND_CREATE(t, entry)                                             \
    exr_attribute_t* attr = NULL;                                              \
    exr_result_t     rv   = EXR_ERR_SUCCESS;                                   \
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);           \
    if (pctxt->mode == EXR_CONTEXT_READ)                                       \
        return EXR_UNLOCK_AND_RETURN_PCTXT (                                   \
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));            \
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)                               \
        return EXR_UNLOCK_AND_RETURN_PCTXT (                                   \
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));       \
    rv = exr_attr_list_find_by_name (                                          \
        ctxt, (exr_attribute_list_t*) &(part->attributes), name, &attr);       \
    if (rv == EXR_ERR_NO_ATTR_BY_NAME)                                         \
    {                                                                          \
        if (pctxt->mode != EXR_CONTEXT_WRITE)                                  \
            return EXR_UNLOCK_AND_RETURN_PCTXT (rv);                           \
                                                                               \
        rv = exr_attr_list_add (                                               \
            ctxt, &(part->attributes), name, t, 0, NULL, &(attr));             \
    }                                                                          \
    else if (rv == EXR_ERR_SUCCESS)                                            \
    {                                                                          \
        if (attr->type != t)                                                   \
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (           \
                pctxt,                                                         \
                EXR_ERR_ATTR_TYPE_MISMATCH,                                    \
                "'%s' requested type '" #entry                                 \
                "', but stored attributes is type '%s'",                       \
                name,                                                          \
                attr->type_name));                                             \
    }                                                                          \
    else                                                                       \
        return EXR_UNLOCK_AND_RETURN_PCTXT (rv)

#define ATTR_SET_IMPL(t, entry)                                                \
    ATTR_FIND_CREATE (t, entry);                                               \
    if (rv == EXR_ERR_SUCCESS) attr->entry = val;                              \
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv)

#define ATTR_SET_IMPL_DEREF(t, entry)                                          \
    ATTR_FIND_CREATE (t, entry);                                               \
    if (!val)                                                                  \
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (               \
            pctxt,                                                             \
            EXR_ERR_INVALID_ARGUMENT,                                          \
            "No input value for setting '%s', type '%s'",                      \
            name,                                                              \
            #entry));                                                          \
    if (rv == EXR_ERR_SUCCESS) *(attr->entry) = *val;                          \
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv)

/**************************************/

exr_result_t
exr_attr_get_box2i (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_box2i_t*   out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_BOX2I, box2i);
}

exr_result_t
exr_attr_set_box2i (
    exr_context_t           ctxt,
    int                     part_index,
    const char*             name,
    const exr_attr_box2i_t* val)
{
    if (name && 0 == strcmp (name, EXR_REQ_DATA_STR))
        return exr_set_data_window (ctxt, part_index, val);
    if (name && 0 == strcmp (name, EXR_REQ_DISP_STR))
        return exr_set_display_window (ctxt, part_index, val);

    {
        ATTR_SET_IMPL_DEREF (EXR_ATTR_BOX2I, box2i);
    }
}

/**************************************/

exr_result_t
exr_attr_get_box2f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_box2f_t*   out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_BOX2F, box2f);
}

exr_result_t
exr_attr_set_box2f (
    exr_context_t           ctxt,
    int                     part_index,
    const char*             name,
    const exr_attr_box2f_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_BOX2F, box2f);
}

/**************************************/

exr_result_t
exr_attr_get_channels (
    exr_const_context_t       ctxt,
    int                       part_index,
    const char*               name,
    const exr_attr_chlist_t** out)
{
    ATTR_GET_IMPL (EXR_ATTR_CHLIST, chlist);
}

exr_result_t
exr_attr_set_channels (
    exr_context_t            ctxt,
    int                      part_index,
    const char*              name,
    const exr_attr_chlist_t* channels)
{
    exr_attribute_t* attr = NULL;
    exr_result_t     rv   = EXR_ERR_SUCCESS;

    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (name && 0 == strcmp (name, EXR_REQ_CHANNELS_STR))
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            exr_set_channels (ctxt, part_index, channels));

    /* do not support updating channels during update operation... */
    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    if (!channels)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "No input values for setting '%s', type 'chlist'",
            name));

    rv = exr_attr_list_find_by_name (
        ctxt, (exr_attribute_list_t*) &(part->attributes), name, &attr);

    if (rv == EXR_ERR_NO_ATTR_BY_NAME)
    {
        rv = exr_attr_list_add (
            ctxt, &(part->attributes), name, EXR_ATTR_CHLIST, 0, NULL, &(attr));
    }

    if (rv == EXR_ERR_SUCCESS)
    {
        exr_attr_chlist_t clist;
        int               numchans;

        if (!channels)
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
                pctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "No channels provided for channel list"));

        numchans = channels->num_channels;
        rv       = exr_attr_chlist_init (ctxt, &clist, numchans);
        if (rv != EXR_ERR_SUCCESS) return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

        for (int c = 0; c < numchans; ++c)
        {
            const exr_attr_chlist_entry_t* cur = channels->entries + c;

            rv = exr_attr_chlist_add_with_length (
                ctxt,
                &clist,
                cur->name.str,
                cur->name.length,
                cur->pixel_type,
                cur->p_linear,
                cur->x_sampling,
                cur->y_sampling);
            if (rv != EXR_ERR_SUCCESS)
            {
                exr_attr_chlist_destroy (ctxt, &clist);
                return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
            }
        }

        exr_attr_chlist_destroy (ctxt, attr->chlist);
        *(attr->chlist) = clist;
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_attr_get_chromaticities (
    exr_const_context_t        ctxt,
    int                        part_index,
    const char*                name,
    exr_attr_chromaticities_t* out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_CHROMATICITIES, chromaticities);
}

exr_result_t
exr_attr_set_chromaticities (
    exr_context_t                    ctxt,
    int                              part_index,
    const char*                      name,
    const exr_attr_chromaticities_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_CHROMATICITIES, chromaticities);
}

/**************************************/

exr_result_t
exr_attr_get_compression (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_compression_t*  out)
{
    ATTR_GET_IMPL (EXR_ATTR_COMPRESSION, uc);
}

exr_result_t
exr_attr_set_compression (
    exr_context_t     ctxt,
    int               part_index,
    const char*       name,
    exr_compression_t cval)
{
    uint8_t val = (uint8_t) cval;
    if (cval >= EXR_COMPRESSION_LAST_TYPE)
        return EXR_CTXT (ctxt)->print_error (
            EXR_CTXT (ctxt),
            EXR_ERR_ARGUMENT_OUT_OF_RANGE,
            "'%s' value for compression type (%d) out of range (%d - %d)",
            name,
            (int) cval,
            0,
            (int) EXR_COMPRESSION_LAST_TYPE);

    if (name && 0 == strcmp (name, EXR_REQ_COMP_STR))
        return exr_set_compression (ctxt, part_index, cval);

    {
        ATTR_SET_IMPL (EXR_ATTR_COMPRESSION, uc);
    }
}

/**************************************/

exr_result_t
exr_attr_get_double (
    exr_const_context_t ctxt, int part_index, const char* name, double* out)
{
    ATTR_GET_IMPL (EXR_ATTR_DOUBLE, d);
}

exr_result_t
exr_attr_set_double (
    exr_context_t ctxt, int part_index, const char* name, double val)
{
    ATTR_SET_IMPL (EXR_ATTR_DOUBLE, d);
}

/**************************************/

exr_result_t
exr_attr_get_envmap (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_envmap_t*       out)
{
    ATTR_GET_IMPL (EXR_ATTR_ENVMAP, uc);
}

exr_result_t
exr_attr_set_envmap (
    exr_context_t ctxt, int part_index, const char* name, exr_envmap_t eval)
{
    uint8_t val = (uint8_t) eval;
    if (eval >= EXR_ENVMAP_LAST_TYPE)
        return EXR_CTXT (ctxt)->print_error (
            EXR_CTXT (ctxt),
            EXR_ERR_ARGUMENT_OUT_OF_RANGE,
            "'%s' value for envmap (%d) out of range (%d - %d)",
            name,
            (int) eval,
            0,
            (int) EXR_ENVMAP_LAST_TYPE);

    {
        ATTR_SET_IMPL (EXR_ATTR_ENVMAP, uc);
    }
}

/**************************************/

exr_result_t
exr_attr_get_float (
    exr_const_context_t ctxt, int part_index, const char* name, float* out)
{
    ATTR_GET_IMPL (EXR_ATTR_FLOAT, f);
}

exr_result_t
exr_attr_set_float (
    exr_context_t ctxt, int part_index, const char* name, float val)
{
    if (name && 0 == strcmp (name, EXR_REQ_PAR_STR))
        return exr_set_pixel_aspect_ratio (ctxt, part_index, val);
    if (name && 0 == strcmp (name, EXR_REQ_SCR_WW_STR))
        return exr_set_screen_window_width (ctxt, part_index, val);

    {
        ATTR_SET_IMPL (EXR_ATTR_FLOAT, f);
    }
}

exr_result_t
exr_attr_get_float_vector (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    int32_t*            sz,
    const float**       out)
{
    ATTR_FIND_ATTR (EXR_ATTR_FLOAT_VECTOR, floatvector);
    if (sz) *sz = attr->floatvector->length;
    if (out) *out = attr->floatvector->arr;
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv);
}

exr_result_t
exr_attr_set_float_vector (
    exr_context_t ctxt,
    int           part_index,
    const char*   name,
    int32_t       sz,
    const float*  val)
{
    exr_attribute_t* attr  = NULL;
    exr_result_t     rv    = EXR_ERR_SUCCESS;
    size_t           bytes = (size_t) sz * sizeof (float);

    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (pctxt->mode == EXR_CONTEXT_READ)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));

    if (sz < 0 || bytes > (size_t) INT32_MAX)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid size (%d) for float vector '%s'",
            sz,
            name));

    if (!val)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "No input values for setting '%s', type 'floatvector'",
            name));

    rv = exr_attr_list_find_by_name (
        ctxt, (exr_attribute_list_t*) &(part->attributes), name, &attr);

    if (rv == EXR_ERR_NO_ATTR_BY_NAME)
    {
        if (pctxt->mode != EXR_CONTEXT_WRITE)
            return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

        rv = exr_attr_list_add (
            ctxt,
            &(part->attributes),
            name,
            EXR_ATTR_FLOAT_VECTOR,
            0,
            NULL,
            &(attr));
        if (rv == EXR_ERR_SUCCESS)
            rv =
                exr_attr_float_vector_create (ctxt, attr->floatvector, val, sz);
    }
    else if (rv == EXR_ERR_SUCCESS)
    {
        if (attr->type != EXR_ATTR_FLOAT_VECTOR)
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_ATTR_TYPE_MISMATCH,
                "'%s' requested type 'floatvector', but attribute is type '%s'",
                name,
                attr->type_name));
        if (attr->floatvector->length == sz &&
            attr->floatvector->alloc_size > 0)
        {
            memcpy (EXR_CONST_CAST (void*, attr->floatvector->arr), val, bytes);
        }
        else if (pctxt->mode != EXR_CONTEXT_WRITE)
        {
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MODIFY_SIZE_CHANGE,
                "Existing float vector '%s' has %d, requested %d, unable to change",
                name,
                attr->floatvector->length,
                sz));
        }
        else
        {
            exr_attr_float_vector_destroy (ctxt, attr->floatvector);
            rv =
                exr_attr_float_vector_create (ctxt, attr->floatvector, val, sz);
        }
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_attr_get_int (
    exr_const_context_t ctxt, int part_index, const char* name, int32_t* out)
{
    ATTR_GET_IMPL (EXR_ATTR_INT, i);
}

exr_result_t
exr_attr_set_int (
    exr_context_t ctxt, int part_index, const char* name, int32_t val)
{
    if (name && !strcmp (name, EXR_REQ_VERSION_STR))
        return exr_set_version (ctxt, part_index, val);
    if (name && !strcmp (name, EXR_REQ_CHUNK_COUNT_STR))
        return exr_set_chunk_count (ctxt, part_index, val);

    {
        ATTR_SET_IMPL (EXR_ATTR_INT, i);
    }
}

/**************************************/

exr_result_t
exr_attr_get_keycode (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_keycode_t* out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_KEYCODE, keycode);
}

exr_result_t
exr_attr_set_keycode (
    exr_context_t             ctxt,
    int                       part_index,
    const char*               name,
    const exr_attr_keycode_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_KEYCODE, keycode);
}

/**************************************/

exr_result_t
exr_attr_get_lineorder (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_lineorder_t*    out)
{
    ATTR_GET_IMPL (EXR_ATTR_LINEORDER, uc);
}

exr_result_t
exr_attr_set_lineorder (
    exr_context_t ctxt, int part_index, const char* name, exr_lineorder_t lval)
{
    uint8_t val = (uint8_t) lval;
    if (lval >= EXR_LINEORDER_LAST_TYPE)
        return EXR_CTXT (ctxt)->print_error (
            EXR_CTXT (ctxt),
            EXR_ERR_ARGUMENT_OUT_OF_RANGE,
            "'%s' value for line order enum (%d) out of range (%d - %d)",
            name,
            (int) lval,
            0,
            (int) EXR_LINEORDER_LAST_TYPE);

    if (name && 0 == strcmp (name, EXR_REQ_LO_STR))
        return exr_set_lineorder (ctxt, part_index, val);

    {
        ATTR_SET_IMPL (EXR_ATTR_LINEORDER, uc);
    }
}

/**************************************/

exr_result_t
exr_attr_get_m33f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m33f_t*    out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_M33F, m33f);
}

exr_result_t
exr_attr_set_m33f (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m33f_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_M33F, m33f);
}

/**************************************/

exr_result_t
exr_attr_get_m33d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m33d_t*    out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_M33D, m33d);
}

exr_result_t
exr_attr_set_m33d (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m33d_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_M33D, m33d);
}

/**************************************/

exr_result_t
exr_attr_get_m44f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m44f_t*    out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_M44F, m44f);
}

exr_result_t
exr_attr_set_m44f (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m44f_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_M44F, m44f);
}

/**************************************/

exr_result_t
exr_attr_get_m44d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m44d_t*    out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_M44D, m44d);
}

exr_result_t
exr_attr_set_m44d (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m44d_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_M44D, m44d);
}

/**************************************/

exr_result_t
exr_attr_get_preview (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_preview_t* out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_PREVIEW, preview);
}

exr_result_t
exr_attr_set_preview (
    exr_context_t             ctxt,
    int                       part_index,
    const char*               name,
    const exr_attr_preview_t* val)
{
    exr_attribute_t* attr = NULL;
    exr_result_t     rv   = EXR_ERR_SUCCESS;

    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (pctxt->mode == EXR_CONTEXT_READ)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));

    rv = exr_attr_list_find_by_name (
        ctxt, (exr_attribute_list_t*) &(part->attributes), name, &attr);

    if (!val)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "No input value for setting '%s', type 'preview'",
            name));

    if (rv == EXR_ERR_NO_ATTR_BY_NAME)
    {
        if (pctxt->mode != EXR_CONTEXT_WRITE)
            return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

        rv = exr_attr_list_add (
            ctxt,
            &(part->attributes),
            name,
            EXR_ATTR_PREVIEW,
            0,
            NULL,
            &(attr));
        if (rv == EXR_ERR_SUCCESS)
            rv = exr_attr_preview_create (
                ctxt, attr->preview, val->width, val->height, val->rgba);
    }
    else if (rv == EXR_ERR_SUCCESS)
    {
        if (attr->type != EXR_ATTR_PREVIEW)
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_ATTR_TYPE_MISMATCH,
                "'%s' requested type 'preview', but attribute is type '%s'",
                name,
                attr->type_name));

        if (attr->preview->width == val->width &&
            attr->preview->height == val->height &&
            attr->preview->alloc_size > 0)
        {
            size_t copybytes = val->width * val->height * 4;
            memcpy (
                EXR_CONST_CAST (void*, attr->preview->rgba),
                val->rgba,
                copybytes);
        }
        else if (pctxt->mode != EXR_CONTEXT_WRITE)
        {
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MODIFY_SIZE_CHANGE,
                "Existing preview '%s' is %u x %u, requested is %u x %u, unable to change",
                name,
                attr->preview->width,
                attr->preview->height,
                val->width,
                val->height));
        }
        else
        {
            exr_attr_preview_destroy (ctxt, attr->preview);
            rv = exr_attr_preview_create (
                ctxt, attr->preview, val->width, val->height, val->rgba);
        }
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_attr_get_rational (
    exr_const_context_t  ctxt,
    int                  part_index,
    const char*          name,
    exr_attr_rational_t* out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_RATIONAL, rational);
}

exr_result_t
exr_attr_set_rational (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    const exr_attr_rational_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_RATIONAL, rational);
}

/**************************************/

exr_result_t
exr_attr_get_string (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    int32_t*            length,
    const char**        out)
{
    ATTR_FIND_ATTR (EXR_ATTR_STRING, string);
    if (length) *length = attr->string->length;
    if (out) *out = attr->string->str;
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv);
}

exr_result_t
exr_attr_set_string (
    exr_context_t ctxt, int part_index, const char* name, const char* val)
{
    size_t           bytes;
    exr_attribute_t* attr = NULL;
    exr_result_t     rv   = EXR_ERR_SUCCESS;

    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (name && !strcmp (name, EXR_REQ_NAME_STR))
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            exr_set_name (ctxt, part_index, name));

    if (name && !strcmp (name, EXR_REQ_TYPE_STR))
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Part type attribute must be implicitly only when adding a part"));

    if (pctxt->mode == EXR_CONTEXT_READ)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));

    rv = exr_attr_list_find_by_name (
        ctxt, (exr_attribute_list_t*) &(part->attributes), name, &attr);

    bytes = val ? strlen (val) : 0;

    if (bytes > (size_t) INT32_MAX)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "String too large to store (%" PRIu64 " bytes) into '%s'",
            (uint64_t) bytes,
            name));

    if (rv == EXR_ERR_NO_ATTR_BY_NAME)
    {
        if (pctxt->mode != EXR_CONTEXT_WRITE)
            return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

        rv = exr_attr_list_add (
            ctxt, &(part->attributes), name, EXR_ATTR_STRING, 0, NULL, &(attr));
        if (rv == EXR_ERR_SUCCESS)
            rv = exr_attr_string_create_with_length (
                ctxt, attr->string, val, (int32_t) bytes);
    }
    else if (rv == EXR_ERR_SUCCESS)
    {
        if (attr->type != EXR_ATTR_STRING)
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_ATTR_TYPE_MISMATCH,
                "'%s' requested type 'string', but attribute is type '%s'",
                name,
                attr->type_name));
        if (attr->string->length == (int32_t) bytes &&
            attr->string->alloc_size > 0)
        {
            if (val)
                memcpy (EXR_CONST_CAST (void*, attr->string->str), val, bytes);
        }
        else if (pctxt->mode != EXR_CONTEXT_WRITE)
        {
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MODIFY_SIZE_CHANGE,
                "Existing string '%s' has length %d, requested %d, unable to change",
                name,
                attr->string->length,
                (int32_t) bytes));
        }
        else
        {
            rv = exr_attr_string_set_with_length (
                ctxt, attr->string, val, (int32_t) bytes);
        }
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

exr_result_t
exr_attr_get_string_vector (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    int32_t*            size,
    const char**        out)
{
    ATTR_FIND_ATTR (EXR_ATTR_STRING_VECTOR, stringvector);
    if (!size)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "size parameter required to query stringvector"));
    if (out)
    {
        if (*size < attr->stringvector->n_strings)
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "'%s' array buffer too small (%d) to hold string values (%d)",
                name,
                *size,
                attr->stringvector->n_strings));
        for (int32_t i = 0; i < attr->stringvector->n_strings; ++i)
            out[i] = attr->stringvector->strings[i].str;
    }
    *size = attr->stringvector->n_strings;
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv);
}

exr_result_t
exr_attr_set_string_vector (
    exr_context_t ctxt,
    int           part_index,
    const char*   name,
    int32_t       size,
    const char**  val)
{
    exr_attribute_t* attr = NULL;
    exr_result_t     rv   = EXR_ERR_SUCCESS;

    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (pctxt->mode == EXR_CONTEXT_READ)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));

    if (size < 0)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid size (%d) for string vector '%s'",
            size,
            name));

    if (!val)
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "No input string values for setting '%s', type 'stringvector'",
            name));

    rv = exr_attr_list_find_by_name (
        ctxt, (exr_attribute_list_t*) &(part->attributes), name, &attr);

    if (rv == EXR_ERR_NO_ATTR_BY_NAME)
    {
        if (pctxt->mode != EXR_CONTEXT_WRITE)
            return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

        rv = exr_attr_list_add (
            ctxt,
            &(part->attributes),
            name,
            EXR_ATTR_STRING_VECTOR,
            0,
            NULL,
            &(attr));
        if (rv == EXR_ERR_SUCCESS)
            rv = exr_attr_string_vector_init (ctxt, attr->stringvector, size);
        for (int32_t i = 0; rv == EXR_ERR_SUCCESS && i < size; ++i)
            rv = exr_attr_string_vector_set_entry (
                ctxt, attr->stringvector, i, val[i]);
    }
    else if (rv == EXR_ERR_SUCCESS)
    {
        if (attr->type != EXR_ATTR_STRING_VECTOR)
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_ATTR_TYPE_MISMATCH,
                "'%s' requested type 'stringvector', but attribute is type '%s'",
                name,
                attr->type_name));
        if (attr->stringvector->n_strings == size &&
            attr->stringvector->alloc_size > 0)
        {
            if (pctxt->mode != EXR_CONTEXT_WRITE)
            {
                for (int32_t i = 0; rv == EXR_ERR_SUCCESS && i < size; ++i)
                {
                    size_t curlen;
                    if (!val[i])
                        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                            pctxt,
                            EXR_ERR_INVALID_ARGUMENT,
                            "'%s' received NULL string in string vector",
                            name));

                    curlen = strlen (val[i]);
                    if (curlen !=
                        (size_t) attr->stringvector->strings[i].length)
                        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                            pctxt,
                            EXR_ERR_INVALID_ARGUMENT,
                            "'%s' string %d in string vector is different size (old %d new %d), unable to update",
                            name,
                            i,
                            attr->stringvector->strings[i].length,
                            (int32_t) curlen));

                    rv = exr_attr_string_vector_set_entry_with_length (
                        ctxt, attr->stringvector, i, val[i], (int32_t) curlen);
                }
            }
            else
            {
                for (int32_t i = 0; rv == EXR_ERR_SUCCESS && i < size; ++i)
                    rv = exr_attr_string_vector_set_entry (
                        ctxt, attr->stringvector, i, val[i]);
            }
        }
        else if (pctxt->mode != EXR_CONTEXT_WRITE)
        {
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MODIFY_SIZE_CHANGE,
                "Existing string vector '%s' has %d strings, but given %d, unable to change",
                name,
                attr->stringvector->n_strings,
                size));
        }
        else
        {
            for (int32_t i = 0; rv == EXR_ERR_SUCCESS && i < size; ++i)
                rv = exr_attr_string_vector_set_entry (
                    ctxt, attr->stringvector, i, val[i]);
        }
    }
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_attr_get_tiledesc (
    exr_const_context_t  ctxt,
    int                  part_index,
    const char*          name,
    exr_attr_tiledesc_t* out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_TILEDESC, tiledesc);
}

exr_result_t
exr_attr_set_tiledesc (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    const exr_attr_tiledesc_t* val)
{
    if (name && 0 == strcmp (name, EXR_REQ_TILES_STR))
    {
        if (!val) return EXR_ERR_INVALID_ARGUMENT;
        return exr_set_tile_descriptor (
            ctxt,
            part_index,
            val->x_size,
            val->y_size,
            EXR_GET_TILE_LEVEL_MODE (*val),
            EXR_GET_TILE_ROUND_MODE (*val));
    }

    {
        ATTR_SET_IMPL_DEREF (EXR_ATTR_TILEDESC, tiledesc);
    }
}

/**************************************/

exr_result_t
exr_attr_get_timecode (
    exr_const_context_t  ctxt,
    int                  part_index,
    const char*          name,
    exr_attr_timecode_t* out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_TIMECODE, timecode);
}

exr_result_t
exr_attr_set_timecode (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    const exr_attr_timecode_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_TIMECODE, timecode);
}

/**************************************/

exr_result_t
exr_attr_get_v2i (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v2i_t*     out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_V2I, v2i);
}

exr_result_t
exr_attr_set_v2i (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v2i_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_V2I, v2i);
}

/**************************************/

exr_result_t
exr_attr_get_v2f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v2f_t*     out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_V2F, v2f);
}

exr_result_t
exr_attr_set_v2f (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v2f_t* val)
{
    if (name && 0 == strcmp (name, EXR_REQ_SCR_WC_STR))
        return exr_set_screen_window_center (ctxt, part_index, val);

    {
        ATTR_SET_IMPL_DEREF (EXR_ATTR_V2F, v2f);
    }
}

/**************************************/

exr_result_t
exr_attr_get_v2d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v2d_t*     out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_V2D, v2d);
}

exr_result_t
exr_attr_set_v2d (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v2d_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_V2D, v2d);
}

/**************************************/

exr_result_t
exr_attr_get_v3i (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v3i_t*     out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_V3I, v3i);
}

exr_result_t
exr_attr_set_v3i (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v3i_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_V3I, v3i);
}

/**************************************/

exr_result_t
exr_attr_get_v3f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v3f_t*     out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_V3F, v3f);
}

exr_result_t
exr_attr_set_v3f (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v3f_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_V3F, v3f);
}

/**************************************/

exr_result_t
exr_attr_get_v3d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v3d_t*     out)
{
    ATTR_GET_IMPL_DEREF (EXR_ATTR_V3D, v3d);
}

exr_result_t
exr_attr_set_v3d (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v3d_t* val)
{
    ATTR_SET_IMPL_DEREF (EXR_ATTR_V3D, v3d);
}

/**************************************/

exr_result_t
exr_attr_get_user (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    const char**        type,
    int32_t*            size,
    const void**        out)
{
    ATTR_FIND_ATTR (EXR_ATTR_OPAQUE, opaque);

    if (rv == EXR_ERR_SUCCESS)
    {
        if (type) *type = attr->type_name;

        if (attr->opaque->pack_func_ptr)
        {
            if (size) *size = attr->opaque->unpacked_size;
            if (out) *out = attr->opaque->unpacked_data;
        }
        else
        {
            if (size) *size = attr->opaque->packed_alloc_size;
            if (out) *out = attr->opaque->packed_data;
        }
    }

    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (rv);
}

exr_result_t
exr_attr_set_user (
    exr_context_t ctxt,
    int           part_index,
    const char*   name,
    const char*   type,
    int32_t       size,
    const void*   out)
{
    exr_attr_opaquedata_t* opq;
    exr_attribute_t*       attr = NULL;
    exr_result_t           rv   = EXR_ERR_SUCCESS;
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    if (pctxt->mode == EXR_CONTEXT_READ)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));
    if (pctxt->mode == EXR_CONTEXT_WRITING_DATA)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_ALREADY_WROTE_ATTRS));
    rv = exr_attr_list_find_by_name (
        ctxt, (exr_attribute_list_t*) &(part->attributes), name, &attr);
    if (rv == EXR_ERR_NO_ATTR_BY_NAME)
    {
        if (pctxt->mode != EXR_CONTEXT_WRITE)
            return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
        rv = exr_attr_list_add_by_type (
            ctxt, &(part->attributes), name, type, 0, NULL, &(attr));
    }
    else if (rv == EXR_ERR_SUCCESS)
    {
        if (attr->type != EXR_ATTR_OPAQUE)
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_ATTR_TYPE_MISMATCH,
                "'%s' requested type '%s', but stored attributes is type '%s'",
                name,
                type,
                attr->type_name));
    }
    else
        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

    opq = attr->opaque;
    if (opq->pack_func_ptr)
    {
        rv = exr_attr_opaquedata_set_unpacked (
            ctxt, attr->opaque, EXR_CONST_CAST (void*, out), size);
        if (rv == EXR_ERR_SUCCESS)
            rv = exr_attr_opaquedata_pack (ctxt, attr->opaque, NULL, NULL);
    }
    else
        rv = exr_attr_opaquedata_set_packed (ctxt, attr->opaque, out, size);
    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}
