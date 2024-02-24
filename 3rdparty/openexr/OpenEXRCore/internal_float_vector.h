/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_ATTR_FLOAT_VECTOR_H
#define OPENEXR_ATTR_FLOAT_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @addtogroup InternalAttributeFunctions
 * @{
 */

/** Allocates storage for a float vector with the provided number of entries.
 *
 * Leaves the float data uninitialized
 */
exr_result_t exr_attr_float_vector_init (
    exr_context_t ctxt, exr_attr_float_vector_t* fv, int32_t nent);
/** Initializes a float vector with the provided number of entries and sets the pointer to the provided pointer
 *
 * This will result in a float vector pointing at a float array that
 * is owned by the calling application and will not be freed, and is
 * expected to outlive the lifetime of the attribute.
 */
exr_result_t exr_attr_float_vector_init_static (
    exr_context_t            ctxt,
    exr_attr_float_vector_t* fv,
    const float*             arr,
    int32_t                  nent);
/** Allocates storage for a float vector with the provided number of entries and initializes */
exr_result_t exr_attr_float_vector_create (
    exr_context_t            ctxt,
    exr_attr_float_vector_t* fv,
    const float*             arr,
    int32_t                  nent);

/** Frees any owned storage for a float vector */
exr_result_t
exr_attr_float_vector_destroy (exr_context_t ctxt, exr_attr_float_vector_t* fv);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_ATTR_FLOAT_VECTOR_H */
