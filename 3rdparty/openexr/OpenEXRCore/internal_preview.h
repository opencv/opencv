/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_ATTR_PREVIEW_H
#define OPENEXR_ATTR_PREVIEW_H

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @addtogroup InternalAttributeFunctions
 * @{
 */

/** @brief Allocates memory for a w * h * 4 entry in the preview
 *
 * This presumes the attr_preview passed in is uninitialized prior to this call
 *
 * @param ctxt context for associated preview attribute (used for error reporting)
 * @param p pointer to attribute to fill. Assumed uninitialized
 * @param w width of preview image
 * @param h height of preview image
 *
 * @return 0 on success, error code otherwise
 */
exr_result_t exr_attr_preview_init (
    exr_context_t ctxt, exr_attr_preview_t* p, uint32_t w, uint32_t h);

/** @brief Allocates memory for a w * h * 4 entry in the preview and fills with provided data
 *
 * This presumes the attr_preview passed in is uninitialized prior to this call.
 *
 * @param ctxt context for associated preview attribute (used for error reporting)
 * @param p pointer to attribute to fill. Assumed uninitialized
 * @param w width of preview image
 * @param h height of preview image
 * @param d input w * h * 4 bytes of data to copy
 *
 * @return 0 on success, error code otherwise
 */
exr_result_t exr_attr_preview_create (
    exr_context_t       ctxt,
    exr_attr_preview_t* p,
    uint32_t            w,
    uint32_t            h,
    const uint8_t*      d);

/** @brief Frees memory for the preview attribute if memory is owned by the preview attr */
exr_result_t
exr_attr_preview_destroy (exr_context_t ctxt, exr_attr_preview_t* p);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_ATTR_PREVIEW_H */
