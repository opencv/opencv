/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_ATTR_STRING_VECTOR_H
#define OPENEXR_ATTR_STRING_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @addtogroup InternalAttributeFunctions
 * @{
 */

/** @brief Allocates memory for a list of strings of length nelt
 *
 * This presumes the attr_string_vector passed in is uninitialized prior to this call
 *
 * @param ctxt context for associated string vector (used for error reporting)
 * @param sv pointer to attribute to initialize. Assumed uninitialized
 * @param nelt desired size of string vector
 *
 * @return 0 on success, error code otherwise
 */
exr_result_t exr_attr_string_vector_init (
    exr_context_t ctxt, exr_attr_string_vector_t* sv, int32_t nelt);

/** @brief Frees memory for the channel list and all channels inside */
exr_result_t exr_attr_string_vector_destroy (
    exr_context_t ctxt, exr_attr_string_vector_t* sv);

exr_result_t exr_attr_string_vector_copy (
    exr_context_t                   ctxt,
    exr_attr_string_vector_t*       sv,
    const exr_attr_string_vector_t* src);

/** @brief Allocates memory for a particular string within the list
 *
 * This enables one to pre-allocate, then read directly into the string
 *
 * @param ctxt context for associated string vector (used for error reporting)
 * @param sv pointer to string vector. It should have been resized ahead of calling
 * @param idx index of the string to initialize
 * @param length desired size of string 
 *
 * @return 0 on success, error code otherwise
 */
exr_result_t exr_attr_string_vector_init_entry (
    exr_context_t             ctxt,
    exr_attr_string_vector_t* sv,
    int32_t                   idx,
    int32_t                   length);

/** @brief Set a string within the string vector */
exr_result_t exr_attr_string_vector_set_entry_with_length (
    exr_context_t             ctxt,
    exr_attr_string_vector_t* sv,
    int32_t                   idx,
    const char*               s,
    int32_t                   length);
/** @brief Set a string within the string vector */
exr_result_t exr_attr_string_vector_set_entry (
    exr_context_t             ctxt,
    exr_attr_string_vector_t* sv,
    int32_t                   idx,
    const char*               s);

/** @brief Append a string to the string vector */
exr_result_t exr_attr_string_vector_add_entry_with_length (
    exr_context_t             ctxt,
    exr_attr_string_vector_t* sv,
    const char*               s,
    int32_t                   length);
/** @brief Append a string to the string vector */
exr_result_t exr_attr_string_vector_add_entry (
    exr_context_t ctxt, exr_attr_string_vector_t* sv, const char* s);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_ATTR_STRING_VECTOR_H */
