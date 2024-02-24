/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_ATTR_CHLIST_H
#define OPENEXR_ATTR_CHLIST_H

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @addtogroup InternalAttributeFunctions
 * @{
 */

/** @brief initialize a channel list with a number of channels to be added later */
exr_result_t
exr_attr_chlist_init (exr_context_t ctxt, exr_attr_chlist_t* chl, int nchans);

/** @brief Add a channel to the channel list */
exr_result_t exr_attr_chlist_add (
    exr_context_t              ctxt,
    exr_attr_chlist_t*         chl,
    const char*                name,
    exr_pixel_type_t           ptype,
    exr_perceptual_treatment_t percept,
    int32_t                    xsamp,
    int32_t                    ysamp);
/** @brief Add a channel to the channel list */
exr_result_t exr_attr_chlist_add_with_length (
    exr_context_t              ctxt,
    exr_attr_chlist_t*         chl,
    const char*                name,
    int32_t                    namelen,
    exr_pixel_type_t           ptype,
    exr_perceptual_treatment_t percept,
    int32_t                    xsamp,
    int32_t                    ysamp);

/** @brief initializes a channel list and duplicates from the source */
exr_result_t exr_attr_chlist_duplicate (
    exr_context_t            ctxt,
    exr_attr_chlist_t*       chl,
    const exr_attr_chlist_t* srcchl);

/** @brief Frees memory for the channel list and all channels inside */
exr_result_t exr_attr_chlist_destroy (exr_context_t ctxt, exr_attr_chlist_t*);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_ATTR_CHLIST_H */
