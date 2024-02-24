/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_ENCODE_H
#define OPENEXR_CORE_ENCODE_H

#include "openexr_chunkio.h"
#include "openexr_coding.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** Can be bit-wise or'ed into the decode_flags in the decode pipeline.
 *
 * Indicates that the sample count table should be encoded from an
 * individual sample count list (n, m, o, ...), meaning it will have
 * to compute the cumulative counts on the fly.
 *
 * Without this (i.e. a value of 0 in that bit), indicates the sample
 * count table is already a cumulative list (n, n+m, n+m+o, ...),
 * which is the on-disk representation.
 */
#define EXR_ENCODE_DATA_SAMPLE_COUNTS_ARE_INDIVIDUAL ((uint16_t) (1 << 0))

/** Can be bit-wise or'ed into the decode_flags in the decode pipeline.
 *
 * Indicates that the data in the channel pointers to encode from is not
 * a direct pointer, but instead is a pointer-to-pointers. In this
 * mode, the user_pixel_stride and user_line_stride are used to
 * advance the pointer offsets for each pixel in the output, but the
 * user_bytes_per_element and user_data_type are used to put
 * (successive) entries into each destination.
 *
 * So each channel pointer must then point to an array of
 * chunk.width * chunk.height pointers. If an entry is
 * `NULL`, 0 samples will be placed in the output.
 *
 * If this is NOT set (0), the default packing routine assumes the
 * data will be planar and contiguous (each channel is a separate
 * memory block), ignoring user_line_stride and user_pixel_stride and
 * advancing only by the sample counts and bytes per element.
 */
#define EXR_ENCODE_NON_IMAGE_DATA_AS_POINTERS ((uint16_t) (1 << 1))

/** Struct meant to be used on a per-thread basis for writing exr data.
 *
 * As should be obvious, this structure is NOT thread safe, but rather
 * meant to be used by separate threads, which can all be accessing
 * the same context concurrently.
 */
typedef struct _exr_encode_pipeline
{
    /** The output channel information for this chunk.
     *
     * User is expected to fill the channel pointers for the input
     * channels. For writing, all channels must be initialized prior
     * to using exr_encoding_choose_default_routines(). If a custom pack routine
     * is written, that is up to the implementor.
     *
     * Describes the channel information. This information is
     * allocated dynamically during exr_encoding_initialize().
     */
    exr_coding_channel_info_t* channels;
    int16_t                    channel_count;

    /** Encode flags to control the behavior. */
    uint16_t encode_flags;

    /** Copy of the parameters given to the initialize/update for convenience. */
    int                 part_index;
    exr_const_context_t context;
    exr_chunk_info_t    chunk;

    /** Can be used by the user to pass custom context data through
     * the encode pipeline.
     */
    void* encoding_user_data;

    /** The packed buffer where individual channels have been put into here.
     *
     * If `NULL`, will be allocated during the run of the pipeline.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* packed_buffer;

    /** Differing from the allocation size, the number of actual bytes */
    uint64_t packed_bytes;

    /** Used when re-using the same encode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough
     *
     * If `NULL`, will be allocated during the run of the pipeline.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    size_t packed_alloc_size;

    /** For deep data. NB: the members NOT const because we need to
     * temporarily swap it to xdr order and restore it (to avoid a
     * duplicate buffer allocation).
     *
     * Depending on the flag set above, will be treated either as a
     * cumulative list (n, n+m, n+m+o, ...), or an individual table
     * (n, m, o, ...). */
    int32_t* sample_count_table;

    /** Allocated table size (to avoid re-allocations). Number of
     * samples must always be width * height for the chunk.
     */
    size_t sample_count_alloc_size;

    /** Packed sample table (compressed, raw on disk representation)
     * for deep or other non-image data.
     */
    void* packed_sample_count_table;

    /** Number of bytes to write (actual size) for the
     * packed_sample_count_table.
     */
    size_t packed_sample_count_bytes;

    /** Allocated size (to avoid re-allocations) for the
     * packed_sample_count_table.
     */
    size_t packed_sample_count_alloc_size;

    /** The compressed buffer, only needed for compressed files.
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* compressed_buffer;

    /** Must be filled in as the pipeline runs to inform the writing
     * software about the compressed size of the chunk (if it is an
     * uncompressed file or the compression would make the file
     * larger, it is expected to be the packed_buffer)
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to zero here. Be cognizant of any
     * custom allocators.
     */
    size_t compressed_bytes;

    /** Used when re-using the same encode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough.
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to zero here. Be cognizant of any
     * custom allocators.
     */
    size_t compressed_alloc_size;

    /** A scratch buffer for intermediate results.
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* scratch_buffer_1;

    /** Used when re-using the same encode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough.
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    size_t scratch_alloc_size_1;

    /** Some compression routines may need a second scratch buffer.
     *
     * If `NULL`, will be allocated during the run of the pipeline when
     * needed.
     *
     * If the caller wishes to take control of the buffer, simple
     * adopt the pointer and set it to `NULL` here. Be cognizant of any
     * custom allocators.
     */
    void* scratch_buffer_2;

    /** Used when re-using the same encode pipeline struct to know if
     * chunk is changed size whether current buffer is large enough.
     */
    size_t scratch_alloc_size_2;

    /** Enable a custom allocator for the different buffers (if
     * encoding on a GPU). If `NULL`, will use the allocator from the
     * context.
     */
    void* (*alloc_fn) (exr_transcoding_pipeline_buffer_id_t, size_t);

    /** Enable a custom allocator for the different buffers (if
     * encoding on a GPU). If `NULL`, will use the allocator from the
     * context.
     */
    void (*free_fn) (exr_transcoding_pipeline_buffer_id_t, void*);

    /** Function chosen based on the output layout of the channels of the part to
     * decompress data.
     *
     * If the user has a custom method for the
     * compression on this part, this can be changed after
     * initialization.
     */
    exr_result_t (*convert_and_pack_fn) (struct _exr_encode_pipeline* pipeline);

    /** Function chosen based on the compression type of the part to
     * compress data.
     *
     * If the user has a custom compression method for the compression
     * type on this part, this can be changed after initialization.
     */
    exr_result_t (*compress_fn) (struct _exr_encode_pipeline* pipeline);

    /** This routine is used when waiting for other threads to finish
     * writing previous chunks such that this thread can write this
     * chunk. This is used for parts which have a specified chunk
     * ordering (increasing/decreasing y) and the chunks can not be
     * written randomly (as could be true for uncompressed).
     *
     * This enables the calling application to contribute thread time
     * to other computation as needed, or just use something like
     * pthread_yield().
     *
     * By default, this routine will be assigned to a function which
     * returns an error, failing the encode immediately. In this way,
     * it assumes that there is only one thread being used for
     * writing.
     *
     * It is up to the user to provide an appropriate routine if
     * performing multi-threaded writing.
     */
    exr_result_t (*yield_until_ready_fn) (
        struct _exr_encode_pipeline* pipeline);

    /** Function chosen to write chunk data to the context.
     *
     * This is allowed to be overridden, but probably is not necessary
     * in most scenarios.
     */
    exr_result_t (*write_fn) (struct _exr_encode_pipeline* pipeline);

    /** Small stash of channel info values. This is faster than calling
     * malloc when the channel count in the part is small (RGBAZ),
     * which is super common, however if there are a large number of
     * channels, it will allocate space for that, so do not rely on
     * this being used.
     */
    exr_coding_channel_info_t _quick_chan_store[5];
} exr_encode_pipeline_t;

/** @brief Simple macro to initialize an empty decode pipeline. */
#define EXR_ENCODE_PIPELINE_INITIALIZER                                        \
    {                                                                          \
        0                                                                      \
    }

/** Initialize the encoding pipeline structure with the channel info
 * for the specified part based on the chunk to be written.
 *
 * NB: The encode_pipe->pack_and_convert_fn field will be `NULL` after this. If that
 * stage is desired, initialize the channel output information and
 * call exr_encoding_choose_default_routines().
 */
EXR_EXPORT
exr_result_t exr_encoding_initialize (
    exr_const_context_t     ctxt,
    int                     part_index,
    const exr_chunk_info_t* cinfo,
    exr_encode_pipeline_t*  encode_pipe);

/** Given an initialized encode pipeline, find an appropriate
 * function to shuffle and convert data into the defined channel
 * outputs.
 *
 * Calling this is not required if a custom routine will be used, or
 * if just the raw decompressed data is desired.
 */
EXR_EXPORT
exr_result_t exr_encoding_choose_default_routines (
    exr_const_context_t    ctxt,
    int                    part_index,
    exr_encode_pipeline_t* encode_pipe);

/** Given a encode pipeline previously initialized, update it for the
 * new chunk to be written.
 *
 * In this manner, memory buffers can be re-used to avoid continual
 * malloc/free calls. Further, it allows the previous choices for
 * the various functions to be quickly re-used.
 */
EXR_EXPORT
exr_result_t exr_encoding_update (
    exr_const_context_t     ctxt,
    int                     part_index,
    const exr_chunk_info_t* cinfo,
    exr_encode_pipeline_t*  encode_pipe);

/** Execute the encoding pipeline. */
EXR_EXPORT
exr_result_t exr_encoding_run (
    exr_const_context_t    ctxt,
    int                    part_index,
    exr_encode_pipeline_t* encode_pipe);

/** Free any intermediate memory in the encoding pipeline.
 *
 * This does NOT free any pointers referred to in the channel info
 * areas, but rather only the intermediate buffers and memory needed
 * for the structure itself.
 */
EXR_EXPORT
exr_result_t exr_encoding_destroy (
    exr_const_context_t ctxt, exr_encode_pipeline_t* encode_pipe);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_CORE_ENCODE_H */
