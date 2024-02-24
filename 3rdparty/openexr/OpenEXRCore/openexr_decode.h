/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_DECODE_H
#define OPENEXR_CORE_DECODE_H

#include "openexr_chunkio.h"
#include "openexr_coding.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** Can be bit-wise or'ed into the decode_flags in the decode pipeline.
 *
 * Indicates that the sample count table should be decoded to a an
 * individual sample count list (n, m, o, ...), with an extra int at
 * the end containing the total samples.
 *
 * Without this (i.e. a value of 0 in that bit), indicates the sample
 * count table should be decoded to a cumulative list (n, n+m, n+m+o,
 * ...), which is the on-disk representation.
 */
#define EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL ((uint16_t) (1 << 0))

/** Can be bit-wise or'ed into the decode_flags in the decode pipeline.
 *
 * Indicates that the data in the channel pointers to decode to is not
 * a direct pointer, but instead is a pointer-to-pointers. In this
 * mode, the user_pixel_stride and user_line_stride are used to
 * advance the pointer offsets for each pixel in the output, but the
 * user_bytes_per_element and user_data_type are used to put
 * (successive) entries into each destination pointer (if not `NULL`).
 *
 * So each channel pointer must then point to an array of
 * chunk.width * chunk.height pointers.
 *
 * With this, you can only extract desired pixels (although all the
 * pixels must be initially decompressed) to handle such operations
 * like proxying where you might want to read every other pixel.
 *
 * If this is NOT set (0), the default unpacking routine assumes the
 * data will be planar and contiguous (each channel is a separate
 * memory block), ignoring user_line_stride and user_pixel_stride.
 */
#define EXR_DECODE_NON_IMAGE_DATA_AS_POINTERS ((uint16_t) (1 << 1))

/**
 * When reading non-image data (i.e. deep), only read the sample table.
 */
#define EXR_DECODE_SAMPLE_DATA_ONLY ((uint16_t) (1 << 2))

/**
 * Struct meant to be used on a per-thread basis for reading exr data
 *
 * As should be obvious, this structure is NOT thread safe, but rather
 * meant to be used by separate threads, which can all be accessing
 * the same context concurrently.
 */
typedef struct _exr_decode_pipeline
{
    /** The output channel information for this chunk.
     *
     * User is expected to fill the channel pointers for the desired
     * output channels (any that are `NULL` will be skipped) if you are
     * going to use exr_decoding_choose_default_routines(). If all that is
     * desired is to read and decompress the data, this can be left
     * uninitialized.
     *
     * Describes the channel information. This information is
     * allocated dynamically during exr_decoding_initialize().
     */
    exr_coding_channel_info_t* channels;
    int16_t                    channel_count;

    /** Decode flags to control the behavior. */
    uint16_t decode_flags;

    /** Copy of the parameters given to the initialize/update for
     * convenience.
     */
    int                 part_index;
    exr_const_context_t context;
    exr_chunk_info_t    chunk;

    /** Can be used by the user to pass custom context data through
     * the decode pipeline.
     */
    void* decoding_user_data;

    /** The (compressed) buffer.
     *
     * If `NULL`, will be allocated during the run of the pipeline.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* packed_buffer;

    /** Used when re-using the same decode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough.
     */
    size_t packed_alloc_size;

    /** The decompressed buffer (unpacked_size from the chunk block
     * info), but still packed into storage order, only needed for
     * compressed files.
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* unpacked_buffer;

    /** Used when re-using the same decode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough.
     */
    size_t unpacked_alloc_size;

    /** For deep or other non-image data: packed sample table
     * (compressed, raw on disk representation).
     */
    void*  packed_sample_count_table;
    size_t packed_sample_count_alloc_size;

    /** Usable, native sample count table. Depending on the flag set
     * above, will be decoded to either a cumulative list (n, n+m,
     * n+m+o, ...), or an individual table (n, m, o, ...). As an
     * optimization, if the latter individual count table is chosen,
     * an extra int32_t will be allocated at the end of the table to
     * contain the total count of samples, so the table will be n+1
     * samples in size.
     */
    int32_t* sample_count_table;
    size_t   sample_count_alloc_size;

    /** A scratch buffer of unpacked_size for intermediate results.
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* scratch_buffer_1;

    /** Used when re-using the same decode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough.
     */
    size_t scratch_alloc_size_1;

    /** Some decompression routines may need a second scratch buffer (zlib).
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* scratch_buffer_2;

    /** Used when re-using the same decode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough.
     */
    size_t scratch_alloc_size_2;

    /** Enable a custom allocator for the different buffers (if
     * decoding on a GPU). If `NULL`, will use the allocator from the
     * context.
     */
    void* (*alloc_fn) (exr_transcoding_pipeline_buffer_id_t, size_t);

    /** Enable a custom allocator for the different buffers (if
     * decoding on a GPU). If `NULL`, will use the allocator from the
     * context.
     */
    void (*free_fn) (exr_transcoding_pipeline_buffer_id_t, void*);

    /** Function chosen to read chunk data from the context.
     *
     * Initialized to a default generic read routine, may be updated
     * based on channel information when 
     * exr_decoding_choose_default_routines() is called. This is done such that
     * if the file is uncompressed and the output channel data is
     * planar and the same type, the read function can read straight
     * into the output channels, getting closer to a zero-copy
     * operation. Otherwise a more traditional read, decompress, then
     * unpack pipeline will be used with a default reader.
     *
     * This is allowed to be overridden, but probably is not necessary
     * in most scenarios.
     */
    exr_result_t (*read_fn) (struct _exr_decode_pipeline* pipeline);

    /** Function chosen based on the compression type of the part to
     * decompress data.
     *
     * If the user has a custom decompression method for the
     * compression on this part, this can be changed after
     * initialization.
     *
     * If only compressed data is desired, then assign this to `NULL`
     * after initialization.
     */
    exr_result_t (*decompress_fn) (struct _exr_decode_pipeline* pipeline);

    /** Function which can be provided if you have bespoke handling for
     * non-image data and need to re-allocate the data to handle the
     * about-to-be unpacked data.
     *
     * If left `NULL`, will assume the memory pointed to by the channel
     * pointers is sufficient.
     */
    exr_result_t (*realloc_nonimage_data_fn) (
        struct _exr_decode_pipeline* pipeline);

    /** Function chosen based on the output layout of the channels of the part to
     * decompress data.
     *
     * This will be `NULL` after initialization, until the user
     * specifies a custom routine, or initializes the channel data and
     * calls exr_decoding_choose_default_routines().
     *
     * If only compressed data is desired, then leave or assign this
     * to `NULL` after initialization.
     */
    exr_result_t (*unpack_and_convert_fn) (
        struct _exr_decode_pipeline* pipeline);

    /** Small stash of channel info values. This is faster than calling
     * malloc when the channel count in the part is small (RGBAZ),
     * which is super common, however if there are a large number of
     * channels, it will allocate space for that, so do not rely on
     * this being used.
     */
    exr_coding_channel_info_t _quick_chan_store[5];
} exr_decode_pipeline_t;

/** @brief Simple macro to initialize an empty decode pipeline. */
#define EXR_DECODE_PIPELINE_INITIALIZER                                        \
    {                                                                          \
        0                                                                      \
    }

/** Initialize the decoding pipeline structure with the channel info
 * for the specified part, and the first block to be read.
 *
 * NB: The decode->unpack_and_convert_fn field will be `NULL` after this. If that
 * stage is desired, initialize the channel output information and
 * call exr_decoding_choose_default_routines().
 */
EXR_EXPORT
exr_result_t exr_decoding_initialize (
    exr_const_context_t     ctxt,
    int                     part_index,
    const exr_chunk_info_t* cinfo,
    exr_decode_pipeline_t*  decode);

/** Given an initialized decode pipeline, find appropriate functions
 * to read and shuffle/convert data into the defined channel outputs.
 *
 * Calling this is not required if custom routines will be used, or if
 * just the raw compressed data is desired. Although in that scenario,
 * it is probably easier to just read the chunk directly using 
 * exr_read_chunk().
 */
EXR_EXPORT
exr_result_t exr_decoding_choose_default_routines (
    exr_const_context_t ctxt, int part_index, exr_decode_pipeline_t* decode);

/** Given a decode pipeline previously initialized, update it for the
 * new chunk to be read.
 *
 * In this manner, memory buffers can be re-used to avoid continual
 * malloc/free calls. Further, it allows the previous choices for
 * the various functions to be quickly re-used.
 */
EXR_EXPORT
exr_result_t exr_decoding_update (
    exr_const_context_t     ctxt,
    int                     part_index,
    const exr_chunk_info_t* cinfo,
    exr_decode_pipeline_t*  decode);

/** Execute the decoding pipeline. */
EXR_EXPORT
exr_result_t exr_decoding_run (
    exr_const_context_t ctxt, int part_index, exr_decode_pipeline_t* decode);

/** Free any intermediate memory in the decoding pipeline.
 *
 * This does *not* free any pointers referred to in the channel info
 * areas, but rather only the intermediate buffers and memory needed
 * for the structure itself.
 */
EXR_EXPORT
exr_result_t
exr_decoding_destroy (exr_const_context_t ctxt, exr_decode_pipeline_t* decode);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_CORE_DECODE_H */
