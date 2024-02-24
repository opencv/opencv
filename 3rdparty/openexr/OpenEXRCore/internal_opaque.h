/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_ATTR_OPAQUE_H
#define OPENEXR_ATTR_OPAQUE_H

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @addtogroup InternalAttributeFunctions
 * @{
 */

exr_result_t exr_attr_opaquedata_init (
    exr_context_t ctxt, exr_attr_opaquedata_t* odata, size_t sz);
exr_result_t exr_attr_opaquedata_create (
    exr_context_t          ctxt,
    exr_attr_opaquedata_t* odata,
    size_t                 sz,
    const void*            values);
exr_result_t
exr_attr_opaquedata_destroy (exr_context_t ctxt, exr_attr_opaquedata_t* ud);

exr_result_t exr_attr_opaquedata_copy (
    exr_context_t                ctxt,
    exr_attr_opaquedata_t*       ud,
    const exr_attr_opaquedata_t* srcud);

/** If an unpack routine was registered, this unpacks the opaque data, returning the pointer and size.
 *
 * The unpacked pointer is stored internally and will be freed during destroy */
exr_result_t exr_attr_opaquedata_unpack (
    exr_context_t ctxt, exr_attr_opaquedata_t*, int32_t* sz, void** unpacked);
/** If a pack routine was registered, this packs the opaque data, returning the pointer and size.
 *
 * The packed pointer is stored internally and will be freed during destroy */
exr_result_t exr_attr_opaquedata_pack (
    exr_context_t ctxt, exr_attr_opaquedata_t*, int32_t* sz, void** packed);

/** Assigns unpacked data
 *
 * Assuming the appropriate handlers have been registered, assigns the
 * unpacked data to the provided value. This memory will be freed at
 * destruction time using the destroy pointer
 */
exr_result_t exr_attr_opaquedata_set_unpacked (
    exr_context_t ctxt, exr_attr_opaquedata_t*, void* unpacked, int32_t sz);

exr_result_t exr_attr_opaquedata_set_packed (
    exr_context_t ctxt, exr_attr_opaquedata_t*, const void* packed, int32_t sz);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_ATTR_OPAQUE_H */
