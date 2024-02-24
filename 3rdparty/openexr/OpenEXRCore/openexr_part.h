/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_PART_H
#define OPENEXR_PART_H

#include "openexr_context.h"

#include "openexr_attr.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** 
 * @defgroup PartInfo Part related definitions.
 *
 * A part is a separate entity in the OpenEXR file. This was
 * formalized in the OpenEXR 2.0 timeframe to allow there to be a
 * clear set of eyes for stereo, or just a simple list of AOVs within
 * a single OpenEXR file. Prior, it was managed by name convention,
 * but with a multi-part file, they are clearly separate types, and
 * can have separate behavior.
 *
 * This is a set of functions to query, or set up when writing, that
 * set of parts within a file. This remains backward compatible to
 * OpenEXR files from before this change, in that a file with a single
 * part is a subset of a multi-part file. As a special case, creating
 * a file with a single part will write out as if it is a file which
 * is not multi-part aware, so as to be compatible with those old
 * libraries.
 *
 * @{
 */

/** @brief Query how many parts are in the file. */
EXR_EXPORT exr_result_t exr_get_count (exr_const_context_t ctxt, int* count);

/** @brief Query the part name for the specified part.
 *
 * NB: If this file is a single part file and name has not been set, this
 * will return `NULL`.
 */
EXR_EXPORT exr_result_t
exr_get_name (exr_const_context_t ctxt, int part_index, const char** out);

/** @brief Query the storage type for the specified part. */
EXR_EXPORT exr_result_t
exr_get_storage (exr_const_context_t ctxt, int part_index, exr_storage_t* out);

/** @brief Define a new part in the file. */
EXR_EXPORT exr_result_t exr_add_part (
    exr_context_t ctxt,
    const char*   partname,
    exr_storage_t type,
    int*          new_index);

/** @brief Query how many levels are in the specified part.
 *
 * If the part is a tiled part, fill in how many tile levels are present.
 *
 * Return `ERR_SUCCESS` on success, an error otherwise (i.e. if the part
 * is not tiled).
 *
 * It is valid to pass `NULL` to either of the @p levelsx or @p levelsy
 * arguments, which enables testing if this part is a tiled part, or
 * if you don't need both (i.e. in the case of a mip-level tiled
 * image)
 */
EXR_EXPORT exr_result_t exr_get_tile_levels (
    exr_const_context_t ctxt,
    int                 part_index,
    int32_t*            levelsx,
    int32_t*            levelsy);

/** @brief Query the tile size for a particular level in the specified part.
 *
 * If the part is a tiled part, fill in the tile size for the
 * specified part/level.
 *
 * Return `ERR_SUCCESS` on success, an error otherwise (i.e. if the
 * part is not tiled).
 *
 * It is valid to pass `NULL` to either of the @p tilew or @p tileh
 * arguments, which enables testing if this part is a tiled part, or
 * if you don't need both (i.e. in the case of a mip-level tiled
 * image)
 */
EXR_EXPORT exr_result_t exr_get_tile_sizes (
    exr_const_context_t ctxt,
    int                 part_index,
    int                 levelx,
    int                 levely,
    int32_t*            tilew,
    int32_t*            tileh);

/** @brief Query the data sizes for a particular level in the specified part.
 *
 * If the part is a tiled part, fill in the width/height for the
 * specified levels.
 *
 * Return `ERR_SUCCESS` on success, an error otherwise (i.e. if the part
 * is not tiled).
 *
 * It is valid to pass `NULL` to either of the @p levw or @p levh
 * arguments, which enables testing if this part is a tiled part, or
 * if you don't need both for some reason.
 */
EXR_EXPORT exr_result_t exr_get_level_sizes (
    exr_const_context_t ctxt,
    int                 part_index,
    int                 levelx,
    int                 levely,
    int32_t*            levw,
    int32_t*            levh);

/** Return the number of chunks contained in this part of the file.
 *
 * As in the technical documentation for OpenEXR, the chunk is the
 * generic term for a pixel data block. This is the atomic unit that
 * this library uses to negotiate data to and from a context.
 * 
 * This should be used as a basis for splitting up how a file is
 * processed. Depending on the compression, a different number of
 * scanlines are encoded in each chunk, and since those need to be
 * encoded/decoded as a block, the chunk should be the basis for I/O
 * as well.
 */
EXR_EXPORT exr_result_t
exr_get_chunk_count (exr_const_context_t ctxt, int part_index, int32_t* out);

/** Return the number of scanlines chunks for this file part.
 *
 * When iterating over a scanline file, this may be an easier metric
 * for multi-threading or other access than only negotiating chunk
 * counts, and so is provided as a utility.
 */
EXR_EXPORT exr_result_t exr_get_scanlines_per_chunk (
    exr_const_context_t ctxt, int part_index, int32_t* out);

/** Return the maximum unpacked size of a chunk for the file part.
 *
 * This may be used ahead of any actual reading of data, so can be
 * used to pre-allocate buffers for multiple threads in one block or
 * whatever your application may require.
 */
EXR_EXPORT exr_result_t exr_get_chunk_unpacked_size (
    exr_const_context_t ctxt, int part_index, uint64_t* out);

/** @brief Retrieve the zip compression level used for the specified part.
 *
 * This only applies when the compression method involves using zip
 * compression (zip, zips, some modes of DWAA/DWAB).
 *
 * This value is NOT persisted in the file, and only exists for the
 * lifetime of the context, so will be at the default value when just
 * reading a file.
 */
EXR_EXPORT exr_result_t exr_get_zip_compression_level (
    exr_const_context_t ctxt, int part_index, int* level);

/** @brief Set the zip compression method used for the specified part.
 *
 * This only applies when the compression method involves using zip
 * compression (zip, zips, some modes of DWAA/DWAB).
 *
 * This value is NOT persisted in the file, and only exists for the
 * lifetime of the context, so this value will be ignored when
 * reading a file.
 */
EXR_EXPORT exr_result_t
exr_set_zip_compression_level (exr_context_t ctxt, int part_index, int level);

/** @brief Retrieve the dwa compression level used for the specified part.
 *
 * This only applies when the compression method is DWAA/DWAB.
 *
 * This value is NOT persisted in the file, and only exists for the
 * lifetime of the context, so will be at the default value when just
 * reading a file.
 */
EXR_EXPORT exr_result_t exr_get_dwa_compression_level (
    exr_const_context_t ctxt, int part_index, float* level);

/** @brief Set the dwa compression method used for the specified part.
 *
 * This only applies when the compression method is DWAA/DWAB.
 *
 * This value is NOT persisted in the file, and only exists for the
 * lifetime of the context, so this value will be ignored when
 * reading a file.
 */
EXR_EXPORT exr_result_t
exr_set_dwa_compression_level (exr_context_t ctxt, int part_index, float level);

/**************************************/

/** @defgroup PartMetadata Functions to get and set metadata for a particular part.
 * @{
 *
 */

/** @brief Query the count of attributes in a part. */
EXR_EXPORT exr_result_t exr_get_attribute_count (
    exr_const_context_t ctxt, int part_index, int32_t* count);

typedef enum exr_attr_list_access_mode
{
    EXR_ATTR_LIST_FILE_ORDER,  /**< Order they appear in the file */
    EXR_ATTR_LIST_SORTED_ORDER /**< Alphabetically sorted */
} exr_attr_list_access_mode_t;

/** @brief Query a particular attribute by index. */
EXR_EXPORT exr_result_t exr_get_attribute_by_index (
    exr_const_context_t         ctxt,
    int                         part_index,
    exr_attr_list_access_mode_t mode,
    int32_t                     idx,
    const exr_attribute_t**     outattr);

/** @brief Query a particular attribute by name. */
EXR_EXPORT exr_result_t exr_get_attribute_by_name (
    exr_const_context_t     ctxt,
    int                     part_index,
    const char*             name,
    const exr_attribute_t** outattr);

/** @brief Query the list of attributes in a part.
 *
 * This retrieves a list of attributes currently defined in a part.
 *
 * If outlist is `NULL`, this function still succeeds, filling only the
 * count. In this manner, the user can allocate memory for the list of
 * attributes, then re-call this function to get the full list.
 */
EXR_EXPORT exr_result_t exr_get_attribute_list (
    exr_const_context_t         ctxt,
    int                         part_index,
    exr_attr_list_access_mode_t mode,
    int32_t*                    count,
    const exr_attribute_t**     outlist);

/** Declare an attribute within the specified part.
 *
 * Only valid when a file is opened for write.
 */
EXR_EXPORT exr_result_t exr_attr_declare_by_type (
    exr_context_t     ctxt,
    int               part_index,
    const char*       name,
    const char*       type,
    exr_attribute_t** newattr);

/** @brief Declare an attribute within the specified part.
 *
 * Only valid when a file is opened for write.
 */
EXR_EXPORT exr_result_t exr_attr_declare (
    exr_context_t        ctxt,
    int                  part_index,
    const char*          name,
    exr_attribute_type_t type,
    exr_attribute_t**    newattr);

/** 
 * @defgroup RequiredAttributeHelpers Required Attribute Utililities
 *
 * @brief These are a group of functions for attributes that are
 * required to be in every part of every file.
 *
 * @{
 */

/** @brief Initialize all required attributes for all files.
 *
 * NB: other file types do require other attributes, such as the tile
 * description for a tiled file.
 */
EXR_EXPORT exr_result_t exr_initialize_required_attr (
    exr_context_t           ctxt,
    int                     part_index,
    const exr_attr_box2i_t* displayWindow,
    const exr_attr_box2i_t* dataWindow,
    float                   pixelaspectratio,
    const exr_attr_v2f_t*   screenWindowCenter,
    float                   screenWindowWidth,
    exr_lineorder_t         lineorder,
    exr_compression_t       ctype);

/** @brief Initialize all required attributes to default values:
 *
 * - `displayWindow` is set to (0, 0 -> @p width - 1, @p height - 1)
 * - `dataWindow` is set to (0, 0 -> @p width - 1, @p height - 1)
 * - `pixelAspectRatio` is set to 1.0
 * - `screenWindowCenter` is set to 0.f, 0.f
 * - `screenWindowWidth` is set to 1.f
 * - `lineorder` is set to `INCREASING_Y`
 * - `compression` is set to @p ctype
 */
EXR_EXPORT exr_result_t exr_initialize_required_attr_simple (
    exr_context_t     ctxt,
    int               part_index,
    int32_t           width,
    int32_t           height,
    exr_compression_t ctype);

/** @brief Copy the attributes from one part to another.
 *
 * This allows one to quickly unassigned attributes from one source to another.
 *
 * If an attribute in the source part has not been yet set in the
 * destination part, the item will be copied over.
 *
 * For example, when you add a part, the storage type and name
 * attributes are required arguments to the definition of a new part,
 * but channels has not yet been assigned. So by calling this with an
 * input file as the source, you can copy the channel definitions (and
 * any other unassigned attributes from the source).
 */
EXR_EXPORT exr_result_t exr_copy_unset_attributes (
    exr_context_t       ctxt,
    int                 part_index,
    exr_const_context_t source,
    int                 src_part_index);

/** @brief Retrieve the list of channels. */
EXR_EXPORT exr_result_t exr_get_channels (
    exr_const_context_t ctxt, int part_index, const exr_attr_chlist_t** chlist);

/** @brief Define a new channel to the output file part.
 *
 * The @p percept parameter is used for lossy compression techniques
 * to indicate that the value represented is closer to linear (1) or
 * closer to logarithmic (0). For r, g, b, luminance, this is normally
 * 0.
 */
EXR_EXPORT int exr_add_channel (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    exr_pixel_type_t           ptype,
    exr_perceptual_treatment_t percept,
    int32_t                    xsamp,
    int32_t                    ysamp);

/** @brief Copy the channels from another source.
 *
 * Useful if you are manually constructing the list or simply copying
 * from an input file.
 */
EXR_EXPORT exr_result_t exr_set_channels (
    exr_context_t ctxt, int part_index, const exr_attr_chlist_t* channels);

/** @brief Retrieve the compression method used for the specified part. */
EXR_EXPORT exr_result_t exr_get_compression (
    exr_const_context_t ctxt, int part_index, exr_compression_t* compression);
/** @brief Set the compression method used for the specified part. */
EXR_EXPORT exr_result_t exr_set_compression (
    exr_context_t ctxt, int part_index, exr_compression_t ctype);

/** @brief Retrieve the data window for the specified part. */
EXR_EXPORT exr_result_t exr_get_data_window (
    exr_const_context_t ctxt, int part_index, exr_attr_box2i_t* out);
/** @brief Set the data window for the specified part. */
EXR_EXPORT int exr_set_data_window (
    exr_context_t ctxt, int part_index, const exr_attr_box2i_t* dw);

/** @brief Retrieve the display window for the specified part. */
EXR_EXPORT exr_result_t exr_get_display_window (
    exr_const_context_t ctxt, int part_index, exr_attr_box2i_t* out);
/** @brief Set the display window for the specified part. */
EXR_EXPORT int exr_set_display_window (
    exr_context_t ctxt, int part_index, const exr_attr_box2i_t* dw);

/** @brief Retrieve the line order for storing data in the specified part (use 0 for single part images). */
EXR_EXPORT exr_result_t exr_get_lineorder (
    exr_const_context_t ctxt, int part_index, exr_lineorder_t* out);
/** @brief Set the line order for storing data in the specified part (use 0 for single part images). */
EXR_EXPORT exr_result_t
exr_set_lineorder (exr_context_t ctxt, int part_index, exr_lineorder_t lo);

/** @brief Retrieve the pixel aspect ratio for the specified part (use 0 for single part images). */
EXR_EXPORT exr_result_t exr_get_pixel_aspect_ratio (
    exr_const_context_t ctxt, int part_index, float* par);
/** @brief Set the pixel aspect ratio for the specified part (use 0 for single part images). */
EXR_EXPORT exr_result_t
exr_set_pixel_aspect_ratio (exr_context_t ctxt, int part_index, float par);

/** @brief Retrieve the screen oriented window center for the specified part (use 0 for single part images). */
EXR_EXPORT exr_result_t exr_get_screen_window_center (
    exr_const_context_t ctxt, int part_index, exr_attr_v2f_t* wc);
/** @brief Set the screen oriented window center for the specified part (use 0 for single part images). */
EXR_EXPORT int exr_set_screen_window_center (
    exr_context_t ctxt, int part_index, const exr_attr_v2f_t* wc);

/** @brief Retrieve the screen oriented window width for the specified part (use 0 for single part images). */
EXR_EXPORT exr_result_t exr_get_screen_window_width (
    exr_const_context_t ctxt, int part_index, float* out);
/** @brief Set the screen oriented window width for the specified part (use 0 for single part images). */
EXR_EXPORT exr_result_t
exr_set_screen_window_width (exr_context_t ctxt, int part_index, float ssw);

/** @brief Retrieve the tiling info for a tiled part (use 0 for single part images). */
EXR_EXPORT exr_result_t exr_get_tile_descriptor (
    exr_const_context_t    ctxt,
    int                    part_index,
    uint32_t*              xsize,
    uint32_t*              ysize,
    exr_tile_level_mode_t* level,
    exr_tile_round_mode_t* round);

/** @brief Set the tiling info for a tiled part (use 0 for single part images). */
EXR_EXPORT exr_result_t exr_set_tile_descriptor (
    exr_context_t         ctxt,
    int                   part_index,
    uint32_t              x_size,
    uint32_t              y_size,
    exr_tile_level_mode_t level_mode,
    exr_tile_round_mode_t round_mode);

EXR_EXPORT exr_result_t
exr_set_name (exr_context_t ctxt, int part_index, const char* val);

EXR_EXPORT exr_result_t
exr_get_version (exr_const_context_t ctxt, int part_index, int32_t* out);

EXR_EXPORT exr_result_t
exr_set_version (exr_context_t ctxt, int part_index, int32_t val);

EXR_EXPORT exr_result_t
exr_set_chunk_count (exr_context_t ctxt, int part_index, int32_t val);

/** @} */ /* required attr group. */

/** 
 * @defgroup BuiltinAttributeHelpers Attribute utilities for builtin types
 *
 * @brief These are a group of functions for attributes that use the builtin types.
 *
 * @{
 */

EXR_EXPORT exr_result_t exr_attr_get_box2i (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_box2i_t*   outval);

EXR_EXPORT exr_result_t exr_attr_set_box2i (
    exr_context_t           ctxt,
    int                     part_index,
    const char*             name,
    const exr_attr_box2i_t* val);

EXR_EXPORT exr_result_t exr_attr_get_box2f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_box2f_t*   outval);

EXR_EXPORT exr_result_t exr_attr_set_box2f (
    exr_context_t           ctxt,
    int                     part_index,
    const char*             name,
    const exr_attr_box2f_t* val);

/** @brief Zero-copy query of channel data.
 *
 * Do not free or manipulate the @p chlist data, or use
 * after the lifetime of the context.
 */
EXR_EXPORT exr_result_t exr_attr_get_channels (
    exr_const_context_t       ctxt,
    int                       part_index,
    const char*               name,
    const exr_attr_chlist_t** chlist);

/** @brief This allows one to quickly copy the channels from one file
 * to another.
 */
EXR_EXPORT exr_result_t exr_attr_set_channels (
    exr_context_t            ctxt,
    int                      part_index,
    const char*              name,
    const exr_attr_chlist_t* channels);

EXR_EXPORT exr_result_t exr_attr_get_chromaticities (
    exr_const_context_t        ctxt,
    int                        part_index,
    const char*                name,
    exr_attr_chromaticities_t* chroma);

EXR_EXPORT exr_result_t exr_attr_set_chromaticities (
    exr_context_t                    ctxt,
    int                              part_index,
    const char*                      name,
    const exr_attr_chromaticities_t* chroma);

EXR_EXPORT exr_result_t exr_attr_get_compression (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_compression_t*  out);

EXR_EXPORT exr_result_t exr_attr_set_compression (
    exr_context_t     ctxt,
    int               part_index,
    const char*       name,
    exr_compression_t comp);

EXR_EXPORT exr_result_t exr_attr_get_double (
    exr_const_context_t ctxt, int part_index, const char* name, double* out);

EXR_EXPORT exr_result_t exr_attr_set_double (
    exr_context_t ctxt, int part_index, const char* name, double val);

EXR_EXPORT exr_result_t exr_attr_get_envmap (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_envmap_t*       out);

EXR_EXPORT exr_result_t exr_attr_set_envmap (
    exr_context_t ctxt, int part_index, const char* name, exr_envmap_t emap);

EXR_EXPORT exr_result_t exr_attr_get_float (
    exr_const_context_t ctxt, int part_index, const char* name, float* out);

EXR_EXPORT exr_result_t exr_attr_set_float (
    exr_context_t ctxt, int part_index, const char* name, float val);

/** @brief Zero-copy query of float data.
 *
 * Do not free or manipulate the @p out data, or use after the
 * lifetime of the context.
 */
EXR_EXPORT exr_result_t exr_attr_get_float_vector (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    int32_t*            sz,
    const float**       out);

EXR_EXPORT exr_result_t exr_attr_set_float_vector (
    exr_context_t ctxt,
    int           part_index,
    const char*   name,
    int32_t       sz,
    const float*  vals);

EXR_EXPORT exr_result_t exr_attr_get_int (
    exr_const_context_t ctxt, int part_index, const char* name, int32_t* out);

EXR_EXPORT exr_result_t exr_attr_set_int (
    exr_context_t ctxt, int part_index, const char* name, int32_t val);

EXR_EXPORT exr_result_t exr_attr_get_keycode (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_keycode_t* out);

EXR_EXPORT exr_result_t exr_attr_set_keycode (
    exr_context_t             ctxt,
    int                       part_index,
    const char*               name,
    const exr_attr_keycode_t* kc);

EXR_EXPORT exr_result_t exr_attr_get_lineorder (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_lineorder_t*    out);

EXR_EXPORT exr_result_t exr_attr_set_lineorder (
    exr_context_t ctxt, int part_index, const char* name, exr_lineorder_t lo);

EXR_EXPORT exr_result_t exr_attr_get_m33f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m33f_t*    out);

EXR_EXPORT exr_result_t exr_attr_set_m33f (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m33f_t* m);

EXR_EXPORT exr_result_t exr_attr_get_m33d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m33d_t*    out);

EXR_EXPORT exr_result_t exr_attr_set_m33d (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m33d_t* m);

EXR_EXPORT exr_result_t exr_attr_get_m44f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m44f_t*    out);

EXR_EXPORT exr_result_t exr_attr_set_m44f (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m44f_t* m);

EXR_EXPORT exr_result_t exr_attr_get_m44d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_m44d_t*    out);

EXR_EXPORT exr_result_t exr_attr_set_m44d (
    exr_context_t          ctxt,
    int                    part_index,
    const char*            name,
    const exr_attr_m44d_t* m);

EXR_EXPORT exr_result_t exr_attr_get_preview (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_preview_t* out);

EXR_EXPORT exr_result_t exr_attr_set_preview (
    exr_context_t             ctxt,
    int                       part_index,
    const char*               name,
    const exr_attr_preview_t* p);

EXR_EXPORT exr_result_t exr_attr_get_rational (
    exr_const_context_t  ctxt,
    int                  part_index,
    const char*          name,
    exr_attr_rational_t* out);

EXR_EXPORT exr_result_t exr_attr_set_rational (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    const exr_attr_rational_t* r);

/** @brief Zero-copy query of string value.
 *
 * Do not modify the string pointed to by @p out, and do not use
 * after the lifetime of the context.
 */
EXR_EXPORT exr_result_t exr_attr_get_string (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    int32_t*            length,
    const char**        out);

EXR_EXPORT exr_result_t exr_attr_set_string (
    exr_context_t ctxt, int part_index, const char* name, const char* s);

/** @brief Zero-copy query of string data.
 *
 * Do not free the strings pointed to by the array.
 *
 * Must provide @p size.
 *
 * \p out must be a ``const char**`` array large enough to hold
 * the string pointers for the string vector when provided.
 */
EXR_EXPORT exr_result_t exr_attr_get_string_vector (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    int32_t*            size,
    const char**        out);

EXR_EXPORT exr_result_t exr_attr_set_string_vector (
    exr_context_t ctxt,
    int           part_index,
    const char*   name,
    int32_t       size,
    const char**  sv);

EXR_EXPORT exr_result_t exr_attr_get_tiledesc (
    exr_const_context_t  ctxt,
    int                  part_index,
    const char*          name,
    exr_attr_tiledesc_t* out);

EXR_EXPORT exr_result_t exr_attr_set_tiledesc (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    const exr_attr_tiledesc_t* td);

EXR_EXPORT exr_result_t exr_attr_get_timecode (
    exr_const_context_t  ctxt,
    int                  part_index,
    const char*          name,
    exr_attr_timecode_t* out);

EXR_EXPORT exr_result_t exr_attr_set_timecode (
    exr_context_t              ctxt,
    int                        part_index,
    const char*                name,
    const exr_attr_timecode_t* tc);

EXR_EXPORT exr_result_t exr_attr_get_v2i (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v2i_t*     out);

EXR_EXPORT exr_result_t exr_attr_set_v2i (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v2i_t* v);

EXR_EXPORT exr_result_t exr_attr_get_v2f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v2f_t*     out);

EXR_EXPORT exr_result_t exr_attr_set_v2f (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v2f_t* v);

EXR_EXPORT exr_result_t exr_attr_get_v2d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v2d_t*     out);

EXR_EXPORT exr_result_t exr_attr_set_v2d (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v2d_t* v);

EXR_EXPORT exr_result_t exr_attr_get_v3i (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v3i_t*     out);

EXR_EXPORT exr_result_t exr_attr_set_v3i (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v3i_t* v);

EXR_EXPORT exr_result_t exr_attr_get_v3f (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v3f_t*     out);

EXR_EXPORT exr_result_t exr_attr_set_v3f (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v3f_t* v);

EXR_EXPORT exr_result_t exr_attr_get_v3d (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    exr_attr_v3d_t*     out);

EXR_EXPORT exr_result_t exr_attr_set_v3d (
    exr_context_t         ctxt,
    int                   part_index,
    const char*           name,
    const exr_attr_v3d_t* v);

EXR_EXPORT exr_result_t exr_attr_get_user (
    exr_const_context_t ctxt,
    int                 part_index,
    const char*         name,
    const char**        type,
    int32_t*            size,
    const void**        out);

EXR_EXPORT exr_result_t exr_attr_set_user (
    exr_context_t ctxt,
    int           part_index,
    const char*   name,
    const char*   type,
    int32_t       size,
    const void*   out);

/** @} */ /* built-in attr group */

/** @} */ /* metadata group */

/** @} */ /* part group */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_PART_H */
