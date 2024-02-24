/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "openexr_debug.h"

#include "internal_constants.h"
#include "internal_structs.h"
#include "openexr_attr.h"

#include <stdio.h>
#include <string.h>

/**************************************/

static void
print_attr (const exr_attribute_t* a, int verbose)
{
    if (!a) return;

    printf ("%s: ", a->name);
    if (verbose) printf ("%s ", a->type_name);
    switch (a->type)
    {
        case EXR_ATTR_BOX2I:
            printf (
                "[ %d, %d - %d %d ] %d x %d",
                a->box2i->min.x,
                a->box2i->min.y,
                a->box2i->max.x,
                a->box2i->max.y,
                a->box2i->max.x - a->box2i->min.x + 1,
                a->box2i->max.y - a->box2i->min.y + 1);
            break;
        case EXR_ATTR_BOX2F:
            printf (
                "[ %g, %g - %g %g ]",
                (double) a->box2f->min.x,
                (double) a->box2f->min.y,
                (double) a->box2f->max.x,
                (double) a->box2f->max.y);
            break;
        case EXR_ATTR_CHLIST:
            printf ("%d channels\n", a->chlist->num_channels);
            for (int c = 0; c < a->chlist->num_channels; ++c)
            {
                if (c > 0) printf ("\n");
                printf (
                    "   '%s': %s samp %d %d",
                    a->chlist->entries[c].name.str,
                    (a->chlist->entries[c].pixel_type == EXR_PIXEL_UINT
                         ? "uint"
                         : (a->chlist->entries[c].pixel_type == EXR_PIXEL_HALF
                                ? "half"
                            : a->chlist->entries[c].pixel_type ==
                                    EXR_PIXEL_FLOAT
                                ? "float"
                                : "<UNKNOWN>")),
                    a->chlist->entries[c].x_sampling,
                    a->chlist->entries[c].y_sampling);
            }
            break;
        case EXR_ATTR_CHROMATICITIES:
            printf (
                "r[%g, %g] g[%g, %g] b[%g, %g] w[%g, %g]",
                (double) a->chromaticities->red_x,
                (double) a->chromaticities->red_y,
                (double) a->chromaticities->green_x,
                (double) a->chromaticities->green_y,
                (double) a->chromaticities->blue_x,
                (double) a->chromaticities->blue_y,
                (double) a->chromaticities->white_x,
                (double) a->chromaticities->white_y);
            break;
        case EXR_ATTR_COMPRESSION: {
            static char* compressionnames[] = {
                "none",
                "rle",
                "zips",
                "zip",
                "piz",
                "pxr24",
                "b44",
                "b44a",
                "dwaa",
                "dwab"};
            printf (
                "'%s'", (a->uc < 10 ? compressionnames[a->uc] : "<UNKNOWN>"));
            if (verbose) printf (" (0x%02X)", a->uc);
            break;
        }
        case EXR_ATTR_DOUBLE: printf ("%g", a->d); break;
        case EXR_ATTR_ENVMAP:
            printf ("%s", a->uc == 0 ? "latlong" : "cube");
            break;
        case EXR_ATTR_FLOAT: printf ("%g", (double) a->f); break;
        case EXR_ATTR_FLOAT_VECTOR:
            printf ("[%d entries]:\n   ", a->floatvector->length);
            for (int i = 0; i < a->floatvector->length; ++i)
                printf (" %g", (double) a->floatvector->arr[i]);
            break;
        case EXR_ATTR_INT: printf ("%d", a->i); break;
        case EXR_ATTR_KEYCODE:
            printf (
                "mfgc %d film %d prefix %d count %d perf_off %d ppf %d ppc %d",
                a->keycode->film_mfc_code,
                a->keycode->film_type,
                a->keycode->prefix,
                a->keycode->count,
                a->keycode->perf_offset,
                a->keycode->perfs_per_frame,
                a->keycode->perfs_per_count);
            break;
        case EXR_ATTR_LINEORDER:
            printf (
                "%d (%s)",
                (int) a->uc,
                a->uc == EXR_LINEORDER_INCREASING_Y
                    ? "increasing"
                    : (a->uc == EXR_LINEORDER_DECREASING_Y
                           ? "decreasing"
                           : (a->uc == EXR_LINEORDER_RANDOM_Y ? "random"
                                                              : "<UNKNOWN>")));
            break;
        case EXR_ATTR_M33F:
            printf (
                "[ [%g %g %g] [%g %g %g] [%g %g %g] ]",
                (double) a->m33f->m[0],
                (double) a->m33f->m[1],
                (double) a->m33f->m[2],
                (double) a->m33f->m[3],
                (double) a->m33f->m[4],
                (double) a->m33f->m[5],
                (double) a->m33f->m[6],
                (double) a->m33f->m[7],
                (double) a->m33f->m[8]);
            break;
        case EXR_ATTR_M33D:
            printf (
                "[ [%g %g %g] [%g %g %g] [%g %g %g] ]",
                a->m33d->m[0],
                a->m33d->m[1],
                a->m33d->m[2],
                a->m33d->m[3],
                a->m33d->m[4],
                a->m33d->m[5],
                a->m33d->m[6],
                a->m33d->m[7],
                a->m33d->m[8]);
            break;
        case EXR_ATTR_M44F:
            printf (
                "[ [%g %g %g %g] [%g %g %g %g] [%g %g %g %g] [%g %g %g %g] ]",
                (double) a->m44f->m[0],
                (double) a->m44f->m[1],
                (double) a->m44f->m[2],
                (double) a->m44f->m[3],
                (double) a->m44f->m[4],
                (double) a->m44f->m[5],
                (double) a->m44f->m[6],
                (double) a->m44f->m[7],
                (double) a->m44f->m[8],
                (double) a->m44f->m[9],
                (double) a->m44f->m[10],
                (double) a->m44f->m[11],
                (double) a->m44f->m[12],
                (double) a->m44f->m[13],
                (double) a->m44f->m[14],
                (double) a->m44f->m[15]);
            break;
        case EXR_ATTR_M44D:
            printf (
                "[ [%g %g %g %g] [%g %g %g %g] [%g %g %g %g] [%g %g %g %g] ]",
                a->m44d->m[0],
                a->m44d->m[1],
                a->m44d->m[2],
                a->m44d->m[3],
                a->m44d->m[4],
                a->m44d->m[5],
                a->m44d->m[6],
                a->m44d->m[7],
                a->m44d->m[8],
                a->m44d->m[9],
                a->m44d->m[10],
                a->m44d->m[11],
                a->m44d->m[12],
                a->m44d->m[13],
                a->m44d->m[14],
                a->m44d->m[15]);
            break;
        case EXR_ATTR_PREVIEW:
            printf ("%u x %u", a->preview->width, a->preview->height);
            break;
        case EXR_ATTR_RATIONAL:
            printf ("%d / %u", a->rational->num, a->rational->denom);
            if (a->rational->denom != 0)
                printf (
                    " (%g)",
                    (double) (a->rational->num) /
                        (double) (a->rational->denom));
            break;
        case EXR_ATTR_STRING:
            printf ("'%s'", a->string->str ? a->string->str : "<NULL>");
            break;
        case EXR_ATTR_STRING_VECTOR:
            printf ("[%d entries]:\n", a->stringvector->n_strings);
            for (int i = 0; i < a->stringvector->n_strings; ++i)
            {
                if (i > 0) printf ("\n");
                printf ("    '%s'", a->stringvector->strings[i].str);
            }
            break;
        case EXR_ATTR_TILEDESC: {
            static const char* lvlModes[] = {
                "single image", "mipmap", "ripmap"};
            uint8_t lvlMode =
                (uint8_t) EXR_GET_TILE_LEVEL_MODE (*(a->tiledesc));
            uint8_t rndMode =
                (uint8_t) EXR_GET_TILE_ROUND_MODE (*(a->tiledesc));
            printf (
                "size %u x %u level %u (%s) round %u (%s)",
                a->tiledesc->x_size,
                a->tiledesc->y_size,
                lvlMode,
                lvlMode < 3 ? lvlModes[lvlMode] : "<UNKNOWN>",
                rndMode,
                rndMode == 0 ? "down" : "up");
            break;
        }
        case EXR_ATTR_TIMECODE:
            printf (
                "time %u user %u",
                a->timecode->time_and_flags,
                a->timecode->user_data);
            break;
        case EXR_ATTR_V2I: printf ("[ %d, %d ]", a->v2i->x, a->v2i->y); break;
        case EXR_ATTR_V2F:
            printf ("[ %g, %g ]", (double) a->v2f->x, (double) a->v2f->y);
            break;
        case EXR_ATTR_V2D: printf ("[ %g, %g ]", a->v2d->x, a->v2d->y); break;
        case EXR_ATTR_V3I:
            printf ("[ %d, %d, %d ]", a->v3i->x, a->v3i->y, a->v3i->z);
            break;
        case EXR_ATTR_V3F:
            printf (
                "[ %g, %g, %g ]",
                (double) a->v3f->x,
                (double) a->v3f->y,
                (double) a->v3f->z);
            break;
        case EXR_ATTR_V3D:
            printf ("[ %g, %g, %g ]", a->v3d->x, a->v3d->y, a->v3d->z);
            break;
        case EXR_ATTR_OPAQUE: {
            uintptr_t faddr_unpack = (uintptr_t) a->opaque->unpack_func_ptr;
            uintptr_t faddr_pack   = (uintptr_t) a->opaque->pack_func_ptr;
            uintptr_t faddr_destroy =
                (uintptr_t) a->opaque->destroy_unpacked_func_ptr;
            printf (
                "(size %d unp size %d hdlrs %p %p %p)",
                a->opaque->size,
                a->opaque->unpacked_size,
                (void*) faddr_unpack,
                (void*) faddr_pack,
                (void*) faddr_destroy);
            break;
        }
        case EXR_ATTR_UNKNOWN:
        case EXR_ATTR_LAST_KNOWN_TYPE:
        default: printf ("<ERROR Unknown type '%s'>", a->type_name); break;
    }
}

/**************************************/

exr_result_t
exr_print_context_info (exr_const_context_t ctxt, int verbose)
{
    EXR_PROMOTE_CONST_CONTEXT_OR_ERROR (ctxt);
    if (verbose)
    {
        printf (
            "File '%s': ver %d flags%s%s%s%s\n",
            pctxt->filename.str,
            (int) pctxt->version,
            pctxt->is_singlepart_tiled ? " singletile" : "",
            pctxt->max_name_length == EXR_LONGNAME_MAXLEN ? " longnames"
                                                          : " shortnames",
            pctxt->has_nonimage_data ? " deep" : "",
            pctxt->is_multipart ? " multipart" : "");
        printf (" parts: %d\n", pctxt->num_parts);
    }
    else { printf ("File '%s':\n", pctxt->filename.str); }

    for (int partidx = 0; partidx < pctxt->num_parts; ++partidx)
    {
        const struct _internal_exr_part* curpart = pctxt->parts[partidx];
        if (verbose || pctxt->is_multipart || curpart->name)
            printf (
                " part %d: %s\n",
                partidx + 1,
                curpart->name ? curpart->name->string->str : "<single>");
        if (verbose)
        {
            for (int a = 0; a < curpart->attributes.num_attributes; ++a)
            {
                if (a > 0) printf ("\n");
                printf ("  ");
                print_attr (curpart->attributes.entries[a], verbose);
            }
            printf ("\n");
        }
        else
        {
            if (curpart->type)
            {
                printf ("  ");
                print_attr (curpart->type, verbose);
            }
            printf ("  ");
            print_attr (curpart->compression, verbose);
            if (curpart->tiles)
            {
                printf ("\n  ");
                print_attr (curpart->tiles, verbose);
            }
            printf ("\n  ");
            print_attr (curpart->displayWindow, verbose);
            printf ("\n  ");
            print_attr (curpart->dataWindow, verbose);
            printf ("\n  ");
            print_attr (curpart->channels, verbose);
            printf ("\n");
        }
        if (curpart->tiles)
        {
            printf (
                "  tiled image has levels: x %d y %d\n",
                curpart->num_tile_levels_x,
                curpart->num_tile_levels_y);
            printf ("    x tile count:");
            for (int l = 0; l < curpart->num_tile_levels_x; ++l)
                printf (
                    " %d (sz %d)",
                    curpart->tile_level_tile_count_x[l],
                    curpart->tile_level_tile_size_x[l]);
            printf ("\n    y tile count:");
            for (int l = 0; l < curpart->num_tile_levels_y; ++l)
                printf (
                    " %d (sz %d)",
                    curpart->tile_level_tile_count_y[l],
                    curpart->tile_level_tile_size_y[l]);
            printf ("\n");
        }
    }
    return EXR_UNLOCK_WRITE_AND_RETURN_PCTXT (EXR_ERR_SUCCESS);
}
