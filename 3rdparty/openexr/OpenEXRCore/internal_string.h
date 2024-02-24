/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_ATTR_STRING_H
#define OPENEXR_ATTR_STRING_H

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @addtogroup InternalAttributeFunctions
 * @{
 */

/** Initializes storage for a string of the provided length
 *
 * This function assumes the string is uninitialized, so make sure use
 * attr_string_destroy to free any string prior to calling init
 */
exr_result_t
exr_attr_string_init (exr_context_t ctxt, exr_attr_string_t* s, int32_t length);

/** Initializes a string with a static string (will not be freed)
 *
 * NB: As a performance optimization, no extra validation of length is
 * performed other than ensuring it is >= 0
 *
 * This function assumes the string is uninitialized, so make sure use
 * attr_string_destroy to free any string prior to calling init
 */
exr_result_t exr_attr_string_init_static_with_length (
    exr_context_t ctxt, exr_attr_string_t* s, const char* v, int32_t length);

/** Initializes a string with a static string (will not be freed).
 *
 * passes through to attr_string_init_static_with_length
 */
exr_result_t exr_attr_string_init_static (
    exr_context_t ctxt, exr_attr_string_t* s, const char* v);

/** Initializes and assigns a string value to the string with a precomputed length
 *
 * This function assumes the string is uninitialized, so make sure use
 * attr_string_destroy to free any string prior to calling init
 */
exr_result_t exr_attr_string_create_with_length (
    exr_context_t ctxt, exr_attr_string_t* s, const char* v, int32_t length);
/** Initializes and assigns a string value to the string
 *
 * This function assumes the string is uninitialized, so make sure use
 * attr_string_destroy to free any string prior to calling init
 */
exr_result_t exr_attr_string_create (
    exr_context_t ctxt, exr_attr_string_t* s, const char* v);

/** Assigns a string value to the string given a precomputed length, potentially resizing it */
exr_result_t exr_attr_string_set_with_length (
    exr_context_t ctxt, exr_attr_string_t* s, const char* v, int32_t length);

/** Assigns a string value to the string, potentially resizing it */
exr_result_t
exr_attr_string_set (exr_context_t ctxt, exr_attr_string_t* s, const char* v);

/** Frees any owned memory associated with the string */
exr_result_t exr_attr_string_destroy (exr_context_t ctxt, exr_attr_string_t* s);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_ATTR_STRING_H */
