/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "openexr_part.h"

#include "internal_attr.h"
#include "internal_constants.h"
#include "internal_structs.h"

#include <string.h>

/**************************************/

exr_result_t
exr_get_count (exr_const_context_t ctxt, int* count)
{
    int cnt;
    EXR_PROMOTE_CONST_CONTEXT_OR_ERROR (ctxt);
    cnt = pctxt->num_parts;
    EXR_UNLOCK_WRITE (pctxt);

    if (!count) return pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT);

    *count = cnt;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_get_storage (exr_const_context_t ctxt, int part_index, exr_storage_t* out)
{
    exr_storage_t smode;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    smode = part->storage_mode;
    EXR_UNLOCK_WRITE (pctxt);

    if (!out) return pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT);

    *out = smode;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_add_part (
    exr_context_t ctxt,
    const char*   partname,
    exr_storage_t type,
    int*          new_index)
{
    exr_result_t rv;
    int32_t      attrsz  = -1;
    const char*  typestr = NULL;

    struct _internal_exr_part* part = NULL;

    EXR_PROMOTE_LOCKED_CONTEXT_OR_ERROR (ctxt);

    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    rv = internal_exr_add_part (pctxt, &part, new_index);
    if (rv != EXR_ERR_SUCCESS) return EXR_UNLOCK_AND_RETURN_PCTXT (rv);

    part->storage_mode = type;
    switch (type)
    {
        case EXR_STORAGE_SCANLINE:
            typestr = "scanlineimage";
            attrsz  = 13;
            break;
        case EXR_STORAGE_TILED:
            typestr = "tiledimage";
            attrsz  = 10;
            break;
        case EXR_STORAGE_DEEP_SCANLINE:
            typestr = "deepscanline";
            attrsz  = 12;
            break;
        case EXR_STORAGE_DEEP_TILED:
            typestr = "deeptile";
            attrsz  = 8;
            break;
        case EXR_STORAGE_LAST_TYPE:
        default:
            internal_exr_revert_add_part (pctxt, &part, new_index);
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_INVALID_ARGUMENT,
                "Invalid storage type %d for new part",
                (int) type));
    }

    rv = exr_attr_list_add_static_name (
        ctxt,
        &(part->attributes),
        EXR_REQ_TYPE_STR,
        EXR_ATTR_STRING,
        0,
        NULL,
        &(part->type));

    if (rv != EXR_ERR_SUCCESS)
    {
        internal_exr_revert_add_part (pctxt, &part, new_index);
        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    }

    rv = exr_attr_string_init_static_with_length (
        ctxt, part->type->string, typestr, attrsz);

    if (rv != EXR_ERR_SUCCESS)
    {
        internal_exr_revert_add_part (pctxt, &part, new_index);
        return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
    }

    /* make sure we put in SOME sort of partname */
    if (!partname) partname = "";
    if (partname && partname[0] != '\0')
    {
        size_t pnamelen = strlen (partname);
        if (pnamelen >= INT32_MAX)
        {
            internal_exr_revert_add_part (pctxt, &part, new_index);
            return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_INVALID_ATTR,
                "Part name '%s': Invalid name length %" PRIu64,
                partname,
                (uint64_t) pnamelen));
        }

        rv = exr_attr_list_add_static_name (
            ctxt,
            &(part->attributes),
            EXR_REQ_NAME_STR,
            EXR_ATTR_STRING,
            0,
            NULL,
            &(part->name));

        if (rv == EXR_ERR_SUCCESS)
            rv = exr_attr_string_create_with_length (
                ctxt, part->name->string, partname, (int32_t) pnamelen);
    }

    if (rv == EXR_ERR_SUCCESS &&
        (type == EXR_STORAGE_DEEP_TILED || type == EXR_STORAGE_DEEP_SCANLINE))
    {
        rv = exr_attr_list_add_static_name (
            ctxt,
            &(part->attributes),
            EXR_REQ_VERSION_STR,
            EXR_ATTR_INT,
            0,
            NULL,
            &(part->version));
        if (rv == EXR_ERR_SUCCESS) part->version->i = 1;
        pctxt->has_nonimage_data = 1;
    }

    if (rv == EXR_ERR_SUCCESS)
    {
        if (pctxt->num_parts > 1) pctxt->is_multipart = 1;

        if (!pctxt->has_nonimage_data && pctxt->num_parts == 1 &&
            type == EXR_STORAGE_TILED)
            pctxt->is_singlepart_tiled = 1;
        else
            pctxt->is_singlepart_tiled = 0;
    }
    else
        internal_exr_revert_add_part (pctxt, &part, new_index);

    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_tile_levels (
    exr_const_context_t ctxt, int part_index, int* levelsx, int* levelsy)
{
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (part->storage_mode == EXR_STORAGE_TILED ||
        part->storage_mode == EXR_STORAGE_DEEP_TILED)
    {
        if (!part->tiles || part->num_tile_levels_x <= 0 ||
            part->num_tile_levels_y <= 0 || !part->tile_level_tile_count_x ||
            !part->tile_level_tile_count_y)
        {
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MISSING_REQ_ATTR,
                "Tile data missing or corrupt"));
        }

        if (levelsx) *levelsx = part->num_tile_levels_x;
        if (levelsy) *levelsy = part->num_tile_levels_y;
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }

    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
        pctxt->standard_error (pctxt, EXR_ERR_TILE_SCAN_MIXEDAPI));
}

/**************************************/

exr_result_t
exr_get_tile_sizes (
    exr_const_context_t ctxt,
    int                 part_index,
    int                 levelx,
    int                 levely,
    int32_t*            tilew,
    int32_t*            tileh)
{
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (part->storage_mode == EXR_STORAGE_TILED ||
        part->storage_mode == EXR_STORAGE_DEEP_TILED)
    {
        const exr_attr_tiledesc_t* tiledesc;

        if (!part->tiles || part->num_tile_levels_x <= 0 ||
            part->num_tile_levels_y <= 0 || !part->tile_level_tile_count_x ||
            !part->tile_level_tile_count_y)
        {
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MISSING_REQ_ATTR,
                "Tile data missing or corrupt"));
        }

        if (levelx < 0 || levely < 0 || levelx >= part->num_tile_levels_x ||
            levely >= part->num_tile_levels_y)
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
                pctxt->standard_error (pctxt, EXR_ERR_ARGUMENT_OUT_OF_RANGE));

        tiledesc = part->tiles->tiledesc;
        if (tilew)
        {
            int32_t levw = part->tile_level_tile_size_x[levelx];
            if (tiledesc->x_size < (uint32_t) levw)
                *tilew = (int32_t) tiledesc->x_size;
            else
                *tilew = levw;
        }
        if (tileh)
        {
            int32_t levh = part->tile_level_tile_size_y[levely];
            if (tiledesc->y_size < (uint32_t) levh)
                *tileh = (int32_t) tiledesc->y_size;
            else
                *tileh = levh;
        }
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }

    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
        pctxt->standard_error (pctxt, EXR_ERR_TILE_SCAN_MIXEDAPI));
}

/**************************************/

exr_result_t
exr_get_level_sizes (
    exr_const_context_t ctxt,
    int                 part_index,
    int                 levelx,
    int                 levely,
    int32_t*            levw,
    int32_t*            levh)
{
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (part->storage_mode == EXR_STORAGE_TILED ||
        part->storage_mode == EXR_STORAGE_DEEP_TILED)
    {
        if (!part->tiles || part->num_tile_levels_x <= 0 ||
            part->num_tile_levels_y <= 0 || !part->tile_level_tile_count_x ||
            !part->tile_level_tile_count_y)
        {
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->print_error (
                pctxt,
                EXR_ERR_MISSING_REQ_ATTR,
                "Tile data missing or corrupt"));
        }

        if (levelx < 0 || levely < 0 || levelx >= part->num_tile_levels_x ||
            levely >= part->num_tile_levels_y)
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
                pctxt->standard_error (pctxt, EXR_ERR_ARGUMENT_OUT_OF_RANGE));

        if (levw) *levw = part->tile_level_tile_size_x[levelx];
        if (levh) *levh = part->tile_level_tile_size_y[levely];
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }

    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
        pctxt->standard_error (pctxt, EXR_ERR_TILE_SCAN_MIXEDAPI));
}

/**************************************/

exr_result_t
exr_get_chunk_count (exr_const_context_t ctxt, int part_index, int32_t* out)
{
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (!out)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT));

    if (part->dataWindow)
    {
        if (part->storage_mode == EXR_STORAGE_TILED ||
            part->storage_mode == EXR_STORAGE_DEEP_TILED)
        {
            if (part->tiles)
            {
                *out = part->chunk_count;
                return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
            }
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->report_error (
                pctxt,
                EXR_ERR_MISSING_REQ_ATTR,
                "Tile data missing or corrupt"));
        }
        else if (
            part->storage_mode == EXR_STORAGE_SCANLINE ||
            part->storage_mode == EXR_STORAGE_DEEP_SCANLINE)
        {
            if (part->compression)
            {
                *out = part->chunk_count;
                return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
            }
            return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->report_error (
                pctxt,
                EXR_ERR_MISSING_REQ_ATTR,
                "Missing scanline chunk compression information"));
        }
    }

    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (pctxt->report_error (
        pctxt,
        EXR_ERR_MISSING_REQ_ATTR,
        "Missing data window for chunk information"));
}

/**************************************/

exr_result_t
exr_get_scanlines_per_chunk (
    exr_const_context_t ctxt, int part_index, int32_t* out)
{
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (!out)
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_INVALID_ARGUMENT);

    if (part->storage_mode == EXR_STORAGE_SCANLINE ||
        part->storage_mode == EXR_STORAGE_DEEP_SCANLINE)
    {
        *out = part->lines_per_chunk;
        return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
    }
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (
        pctxt->standard_error (pctxt, EXR_ERR_SCAN_TILE_MIXEDAPI));
}

/**************************************/

exr_result_t
exr_get_chunk_unpacked_size (
    exr_const_context_t ctxt, int part_index, uint64_t* out)
{
    uint64_t sz;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    sz = part->unpacked_size_per_chunk;
    EXR_UNLOCK_WRITE (pctxt);

    if (!out) return pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT);

    *out = sz;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_get_zip_compression_level (
    exr_const_context_t ctxt, int part_index, int* level)
{
    int l;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    l = part->zip_compression_level;
    EXR_UNLOCK_WRITE (pctxt);

    if (!level) return pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT);
    *level = l;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_set_zip_compression_level (exr_context_t ctxt, int part_index, int level)
{
    exr_result_t rv;
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    if (level >= -1 && level < 10)
    {
        part->zip_compression_level = level;
        rv                          = EXR_ERR_SUCCESS;
    }
    else
    {
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt, EXR_ERR_INVALID_ARGUMENT, "Invalid zip level specified"));
    }

    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}

/**************************************/

exr_result_t
exr_get_dwa_compression_level (
    exr_const_context_t ctxt, int part_index, float* level)
{
    float l;
    EXR_PROMOTE_CONST_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);
    l = part->dwa_compression_level;
    EXR_UNLOCK_WRITE (pctxt);

    if (!level) return pctxt->standard_error (pctxt, EXR_ERR_INVALID_ARGUMENT);
    *level = l;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
exr_set_dwa_compression_level (exr_context_t ctxt, int part_index, float level)
{
    exr_result_t rv;
    EXR_PROMOTE_LOCKED_CONTEXT_AND_PART_OR_ERROR (ctxt, part_index);

    if (pctxt->mode != EXR_CONTEXT_WRITE)
        return EXR_UNLOCK_AND_RETURN_PCTXT (
            pctxt->standard_error (pctxt, EXR_ERR_NOT_OPEN_WRITE));

    if (level > 0.f && level <= 100.f)
    {
        part->dwa_compression_level = level;
        rv                          = EXR_ERR_SUCCESS;
    }
    else
    {
        return EXR_UNLOCK_AND_RETURN_PCTXT (pctxt->report_error (
            pctxt,
            EXR_ERR_INVALID_ARGUMENT,
            "Invalid dwa quality level specified"));
    }

    return EXR_UNLOCK_AND_RETURN_PCTXT (rv);
}
