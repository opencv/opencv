/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CONTEXT_H
#define OPENEXR_CONTEXT_H

#include "openexr_errors.h"

#include "openexr_base.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** 
 * @defgroup Context Context related definitions
 *
 * A context is a single instance of an OpenEXR file or stream. Beyond
 * a particular file or stream handle, it also has separate controls
 * for error handling and memory allocation. This is done to enable
 * encoding or decoding on mixed hardware.
 *
 * @{
 */

/** Opaque context handle
 *
 * The implementation of this is partly opaque to provide better
 * version portability, and all accesses to relevant data should
 * happen using provided functions. This handle serves as a container
 * and identifier for all the metadata and parts associated with a
 * file and/or stream.
 */

typedef struct _priv_exr_context_t*       exr_context_t;
typedef const struct _priv_exr_context_t* exr_const_context_t;

/** 
 * @defgroup ContextFunctions OpenEXR Context Stream/File Functions
 *
 * @brief These are a group of function interfaces used to customize
 * the error handling, memory allocations, or I/O behavior of an
 * OpenEXR context.
 *
 * @{
 */

/** @brief Stream error notifier
 *
 *  This function pointer is provided to the stream functions by the
 *  library such that they can provide a nice error message to the
 *  user during stream operations.
 */
typedef exr_result_t (*exr_stream_error_func_ptr_t) (
    exr_const_context_t ctxt, exr_result_t code, const char* fmt, ...)
    EXR_PRINTF_FUNC_ATTRIBUTE;

/** @brief Error callback function
 *
 *  Because a file can be read from using many threads at once, it is
 *  difficult to store an error message for later retrieval. As such,
 *  when a file is constructed, a callback function can be provided
 *  which delivers an error message for the calling application to
 *  handle. This will then be delivered on the same thread causing the
 *  error.
 */
typedef void (*exr_error_handler_cb_t) (
    exr_const_context_t ctxt, exr_result_t code, const char* msg);

/** Destroy custom stream function pointer
 *
 *  Generic callback to clean up user data for custom streams.
 *  This is called when the file is closed and expected not to
 *  error.
 *
 *  @param failed Indicates the write operation failed, the
 *                implementor may wish to cleanup temporary files
 *  @param ctxt The context
 *  @param userdata The userdata
 */
typedef void (*exr_destroy_stream_func_ptr_t) (
    exr_const_context_t ctxt, void* userdata, int failed);

/** Query stream size function pointer
 *
 * Used to query the size of the file, or amount of data representing
 * the openexr file in the data stream.
 *
 * This is used to validate requests against the file. If the size is
 * unavailable, return -1, which will disable these validation steps
 * for this file, although appropriate memory safeguards must be in
 * place in the calling application.
 */
typedef int64_t (*exr_query_size_func_ptr_t) (
    exr_const_context_t ctxt, void* userdata);

/** @brief Read custom function pointer
 *
 * Used to read data from a custom output. Expects similar semantics to
 * pread or ReadFile with overlapped data under win32.
 *
 * It is required that this provides thread-safe concurrent access to
 * the same file. If the stream/input layer you are providing does
 * not have this guarantee, your are responsible for providing
 * appropriate serialization of requests.
 *
 * A file should be expected to be accessed in the following pattern:
 *  - upon open, the header and part information attributes will be read
 *  - upon the first image read request, the offset tables will be read
 *    multiple threads accessing this concurrently may actually read
 *    these values at the same time
 *  - chunks can then be read in any order as preferred by the
 *    application
 *
 * While this should mean that the header will be read in 'stream'
 * order (no seeks required), no guarantee is made beyond that to
 * retrieve image/deep data in order. So if the backing file is
 * truly a stream, it is up to the provider to implement appropriate
 * caching of data to give the appearance of being able to seek/read
 * atomically.
 */
typedef int64_t (*exr_read_func_ptr_t) (
    exr_const_context_t         ctxt,
    void*                       userdata,
    void*                       buffer,
    uint64_t                    sz,
    uint64_t                    offset,
    exr_stream_error_func_ptr_t error_cb);

/** Write custom function pointer
 *
 *  Used to write data to a custom output. Expects similar semantics to
 *  pwrite or WriteFile with overlapped data under win32.
 *
 *  It is required that this provides thread-safe concurrent access to
 *  the same file. While it is unlikely that multiple threads will
 *  be used to write data for compressed forms, it is possible.
 *
 *  A file should be expected to be accessed in the following pattern:
 *  - upon open, the header and part information attributes is constructed.
 *
 *  - when the write_header routine is called, the header becomes immutable
 *    and is written to the file. This computes the space to store the chunk
 *    offsets, but does not yet write the values.
 *
 *  - Image chunks are written to the file, and appear in the order
 *    they are written, not in the ordering that is required by the
 *    chunk offset table (unless written in that order). This may vary
 *    slightly if the size of the chunks is not directly known and
 *    tight packing of data is necessary.
 *
 *  - at file close, the chunk offset tables are written to the file.
 */
typedef int64_t (*exr_write_func_ptr_t) (
    exr_const_context_t         ctxt,
    void*                       userdata,
    const void*                 buffer,
    uint64_t                    sz,
    uint64_t                    offset,
    exr_stream_error_func_ptr_t error_cb);

/** @brief Struct used to pass function pointers into the context
 * initialization routines.
 *
 * This partly exists to avoid the chicken and egg issue around
 * creating the storage needed for the context on systems which want
 * to override the malloc/free routines.
 *
 * However, it also serves to make a tidier/simpler set of functions
 * to create and start processing exr files.
 *
 * The size member is required for version portability.
 *
 * It can be initialized using \c EXR_DEFAULT_CONTEXT_INITIALIZER.
 *
 * \code{.c}
 * exr_context_initializer_t myctxtinit = DEFAULT_CONTEXT_INITIALIZER;
 * myctxtinit.error_cb = &my_super_cool_error_callback_function;
 * ...
 * \endcode
 *
 */
typedef struct _exr_context_initializer_v3
{
    /** @brief Size member to tag initializer for version stability.
     *
     * This should be initialized to the size of the current
     * structure. This allows EXR to add functions or other
     * initializers in the future, and retain version compatibility
     */
    size_t size;

    /** @brief Error callback function pointer
     *
     * The error callback is allowed to be `NULL`, and will use a
     * default print which outputs to \c stderr.
     *
     * @sa exr_error_handler_cb_t
     */
    exr_error_handler_cb_t error_handler_fn;

    /** Custom allocator, if `NULL`, will use malloc. @sa exr_memory_allocation_func_t */
    exr_memory_allocation_func_t alloc_fn;

    /** Custom deallocator, if `NULL`, will use free. @sa exr_memory_free_func_t */
    exr_memory_free_func_t free_fn;

    /** Blind data passed to custom read, size, write, destroy
     * functions below. Up to user to manage this pointer.
     */
    void* user_data;

    /** @brief Custom read routine.
     *
     * This is only used during read or update contexts. If this is
     * provided, it is expected that the caller has previously made
     * the stream available, and placed whatever stream/file data
     * into \c user_data above.
     *
     * If this is `NULL`, and the context requested is for reading an
     * exr file, an internal implementation is provided for reading
     * from normal filesystem files, and the filename provided is
     * attempted to be opened as such.
     *
     * Expected to be `NULL` for a write-only operation, but is ignored
     * if it is provided.
     *
     * For update contexts, both read and write functions must be
     * provided if either is.
     *
     * @sa exr_read_func_ptr_t
     */
    exr_read_func_ptr_t read_fn;

    /** @brief Custom size query routine.
     *
     * Used to provide validation when reading header values. If this
     * is not provided, but a custom read routine is provided, this
     * will disable some of the validation checks when parsing the
     * image header.
     *
     * Expected to be `NULL` for a write-only operation, but is ignored
     * if it is provided.
     *
     * @sa exr_query_size_func_ptr_t
     */
    exr_query_size_func_ptr_t size_fn;

    /** @brief Custom write routine.
     *
     * This is only used during write or update contexts. If this is
     * provided, it is expected that the caller has previously made
     * the stream available, and placed whatever stream/file data
     * into \c user_data above.
     *
     * If this is `NULL`, and the context requested is for writing an
     * exr file, an internal implementation is provided for reading
     * from normal filesystem files, and the filename provided is
     * attempted to be opened as such.
     *
     * For update contexts, both read and write functions must be
     * provided if either is.
     *
     * @sa exr_write_func_ptr_t
     */
    exr_write_func_ptr_t write_fn;

    /** @brief Optional function to destroy the user data block of a custom stream.
     *
     * Allows one to free any user allocated data, and close any handles.
     *
     * @sa exr_destroy_stream_func_ptr_t
     * */
    exr_destroy_stream_func_ptr_t destroy_fn;

    /** Initialize a field specifying what the maximum image width
     * allowed by the context is. See exr_set_default_maximum_image_size() to
     * understand how this interacts with global defaults.
     */
    int max_image_width;

    /** Initialize a field specifying what the maximum image height
     * allowed by the context is. See exr_set_default_maximum_image_size() to
     * understand how this interacts with global defaults.
     */
    int max_image_height;

    /** Initialize a field specifying what the maximum tile width
     * allowed by the context is. See exr_set_default_maximum_tile_size() to
     * understand how this interacts with global defaults.
     */
    int max_tile_width;

    /** Initialize a field specifying what the maximum tile height
     * allowed by the context is. See exr_set_default_maximum_tile_size() to
     * understand how this interacts with global defaults.
     */
    int max_tile_height;

    /** Initialize a field specifying what the default zip compression level should be
     * for this context. See exr_set_default_zip_compresion_level() to
     * set it for all contexts.
     */
    int zip_level;

    /** Initialize the default dwa compression quality. See
     * exr_set_default_dwa_compression_quality() to set the default
     * for all contexts.
     */
    float dwa_quality;

    /** Initialize with a bitwise or of the various context flags
     */
    int flags;

    uint8_t pad[4];
} exr_context_initializer_t;

/** @brief context flag which will enforce strict header validation
 * checks and may prevent reading of files which could otherwise be
 * processed.
 */
#define EXR_CONTEXT_FLAG_STRICT_HEADER (1 << 0)

/** @brief Disables error messages while parsing headers
 *
 * The return values will remain the same, but error reporting will be
 * skipped. This is only valid for reading contexts
 */
#define EXR_CONTEXT_FLAG_SILENT_HEADER_PARSE (1 << 1)

/** @brief Disables reconstruction logic upon corrupt / missing data chunks
 *
 * This will disable the reconstruction logic that searches through an
 * incomplete file, and will instead just return errors at read
 * time. This is only valid for reading contexts
 */
#define EXR_CONTEXT_FLAG_DISABLE_CHUNK_RECONSTRUCTION (1 << 2)

/** @brief Writes an old-style, sorted header with minimal information */
#define EXR_CONTEXT_FLAG_WRITE_LEGACY_HEADER (1 << 3)

/* clang-format off */
/** @brief Simple macro to initialize the context initializer with default values. */
#define EXR_DEFAULT_CONTEXT_INITIALIZER                                        \
    { sizeof (exr_context_initializer_t), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1.f, 0, { 0, 0, 0, 0 } }
/* clang-format on */

/** @} */ /* context function pointer declarations */

/** @brief Check the magic number of the file and report
 * `EXR_ERR_SUCCESS` if the file appears to be a valid file (or at least
 * has the correct magic number and can be read).
 */
EXR_EXPORT exr_result_t exr_test_file_header (
    const char* filename, const exr_context_initializer_t* ctxtdata);

/** @brief Close and free any internally allocated memory,
 * calling any provided destroy function for custom streams.
 *
 * If the file was opened for write, first save the chunk offsets
 * or any other unwritten data.
 */
EXR_EXPORT exr_result_t exr_finish (exr_context_t* ctxt);

/** @brief Create and initialize a read-only exr read context.
 *
 * If a custom read function is provided, the filename is for
 * informational purposes only, the system assumes the user has
 * previously opened a stream, file, or whatever and placed relevant
 * data in userdata to access that.
 *
 * One notable attribute of the context is that once it has been
 * created and returned a successful code, it has parsed all the
 * header data. This is done as one step such that it is easier to
 * provide a safe context for multiple threads to request data from
 * the same context concurrently.
 *
 * Once finished reading data, use exr_finish() to clean up
 * the context.
 *
 * If you have custom I/O requirements, see the initializer context
 * documentation \ref exr_context_initializer_t. The @p ctxtdata parameter
 * is optional, if `NULL`, default values will be used.
 */
EXR_EXPORT exr_result_t exr_start_read (
    exr_context_t*                   ctxt,
    const char*                      filename,
    const exr_context_initializer_t* ctxtdata);

/** @brief Enum describing how default files are handled during write. */
typedef enum exr_default_write_mode
{
    EXR_WRITE_FILE_DIRECTLY =
        0, /**< Overwrite filename provided directly, deleted upon error. */
    EXR_INTERMEDIATE_TEMP_FILE =
        1 /**< Create a temporary file, renaming it upon successful write, leaving original upon error */
} exr_default_write_mode_t;

/** @brief Create and initialize a write-only context. 
 *
 * If a custom write function is provided, the filename is for
 * informational purposes only, and the @p default_mode parameter will be
 * ignored. As such, the system assumes the user has previously opened
 * a stream, file, or whatever and placed relevant data in userdata to
 * access that.
 *
 * Multi-Threading: To avoid issues with creating multi-part EXR
 * files, the library approaches writing as a multi-step process, so
 * the same concurrent guarantees can not be made for writing a
 * file. The steps are:
 *
 * 1. Context creation (this function)
 *
 * 2. Part definition (required attributes and additional metadata)
 *
 * 3. Transition to writing data (this "commits" the part definitions,
 * any changes requested after will result in an error)
 *
 * 4. Write part data in sequential order of parts (part<sub>0</sub>
 * -> part<sub>N-1</sub>).
 *
 * 5. Within each part, multiple threads can be encoding and writing
 * data concurrently. For some EXR part definitions, this may be able
 * to write data concurrently when it can predict the chunk sizes, or
 * data is allowed to be padded. For others, it may need to
 * temporarily cache chunks until the data is received to flush in
 * order. The concurrency around this is handled by the library
 *
 * 6. Once finished writing data, use exr_finish() to clean
 * up the context, which will flush any unwritten data such as the
 * final chunk offset tables, and handle the temporary file flags.
 *
 * If you have custom I/O requirements, see the initializer context
 * documentation \ref exr_context_initializer_t. The @p ctxtdata
 * parameter is optional, if `NULL`, default values will be used.
 */
EXR_EXPORT exr_result_t exr_start_write (
    exr_context_t*                   ctxt,
    const char*                      filename,
    exr_default_write_mode_t         default_mode,
    const exr_context_initializer_t* ctxtdata);

/** @brief Create a new context for updating an exr file in place.
 *
 * This is a custom mode that allows one to modify the value of a
 * metadata entry, although not to change the size of the header, or
 * any of the image data.
 *
 * If you have custom I/O requirements, see the initializer context
 * documentation \ref exr_context_initializer_t. The @p ctxtdata parameter
 * is optional, if `NULL`, default values will be used.
 */
EXR_EXPORT exr_result_t exr_start_inplace_header_update (
    exr_context_t*                   ctxt,
    const char*                      filename,
    const exr_context_initializer_t* ctxtdata);

/** @brief Retrieve the file name the context is for as provided
 * during the start routine.
 *
 * Do not free the resulting string.
 */
EXR_EXPORT exr_result_t
exr_get_file_name (exr_const_context_t ctxt, const char** name);

/** @brief Query the user data the context was constructed with. This
 * is perhaps useful in the error handler callback to jump back into
 * an object the user controls.
 */
EXR_EXPORT exr_result_t
exr_get_user_data (exr_const_context_t ctxt, void** userdata);

/** Any opaque attribute data entry of the specified type is tagged
 * with these functions enabling downstream users to unpack (or pack)
 * the data.
 *
 * The library handles the memory packed data internally, but the
 * handler is expected to allocate and manage memory for the
 * *unpacked* buffer (the library will call the destroy function).
 *
 * NB: the pack function will be called twice (unless there is a
 * memory failure), the first with a `NULL` buffer, requesting the
 * maximum size (or exact size if known) for the packed buffer, then
 * the second to fill the output packed buffer, at which point the
 * size can be re-updated to have the final, precise size to put into
 * the file.
 */
EXR_EXPORT exr_result_t exr_register_attr_type_handler (
    exr_context_t ctxt,
    const char*   type,
    exr_result_t (*unpack_func_ptr) (
        exr_context_t ctxt,
        const void*   data,
        int32_t       attrsize,
        int32_t*      outsize,
        void**        outbuffer),
    exr_result_t (*pack_func_ptr) (
        exr_context_t ctxt,
        const void*   data,
        int32_t       datasize,
        int32_t*      outsize,
        void*         outbuffer),
    void (*destroy_unpacked_func_ptr) (
        exr_context_t ctxt, void* data, int32_t datasize));

/** @brief Enable long name support in the output context */
EXR_EXPORT exr_result_t
exr_set_longname_support (exr_context_t ctxt, int onoff);

/** @brief Write the header data.
 *
 * Opening a new output file has a small initialization state problem
 * compared to opening for read/update: we need to enable the user
 * to specify an arbitrary set of metadata across an arbitrary number
 * of parts. To avoid having to create the list of parts and entire
 * metadata up front, prior to calling the above exr_start_write(),
 * allow the data to be set, then once this is called, it switches
 * into a mode where the library assumes the data is now valid.
 * 
 * It will recompute the number of chunks that will be written, and
 * reset the chunk offsets. If you modify file attributes or part
 * information after a call to this, it will error.
 */
EXR_EXPORT exr_result_t exr_write_header (exr_context_t ctxt);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_CONTEXT_H */
