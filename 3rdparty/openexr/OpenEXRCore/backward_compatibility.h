/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_BACKWARD_COMPATIBILITY_H
#define OPENEXR_BACKWARD_COMPATIBILITY_H

struct _exr_context_initializer_v1
{
    size_t                        size;
    exr_error_handler_cb_t        error_handler_fn;
    exr_memory_allocation_func_t  alloc_fn;
    exr_memory_free_func_t        free_fn;
    void*                         user_data;
    exr_read_func_ptr_t           read_fn;
    exr_query_size_func_ptr_t     size_fn;
    exr_write_func_ptr_t          write_fn;
    exr_destroy_stream_func_ptr_t destroy_fn;
    int                           max_image_width;
    int                           max_image_height;
    int                           max_tile_width;
    int                           max_tile_height;
};

struct _exr_context_initializer_v2
{
    size_t                        size;
    exr_error_handler_cb_t        error_handler_fn;
    exr_memory_allocation_func_t  alloc_fn;
    exr_memory_free_func_t        free_fn;
    void*                         user_data;
    exr_read_func_ptr_t           read_fn;
    exr_query_size_func_ptr_t     size_fn;
    exr_write_func_ptr_t          write_fn;
    exr_destroy_stream_func_ptr_t destroy_fn;
    int                           max_image_width;
    int                           max_image_height;
    int                           max_tile_width;
    int                           max_tile_height;
    int                           zip_level;
    float                         dwa_quality;
};

#endif /* OPENEXR_BACKWARD_COMPATIBILITY_H */
