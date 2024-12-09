// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef PLUGIN_WRITER_API_HPP
#define PLUGIN_WRITER_API_HPP

#include <opencv2/core/cvdef.h>
#include <opencv2/core/llapi/llapi.h>

#if !defined(BUILD_PLUGIN)

/// increased for backward-compatible changes, e.g. add new function
/// Caller API <= Plugin API -> plugin is fully compatible
/// Caller API > Plugin API -> plugin is not fully compatible, caller should use extra checks to use plugins with older API
#define WRITER_API_VERSION 1

/// increased for incompatible changes, e.g. remove function argument
/// Caller ABI == Plugin ABI -> plugin is compatible
/// Caller ABI > Plugin ABI -> plugin is not compatible, caller should use shim code to use old ABI plugins (caller may know how lower ABI works, so it is possible)
/// Caller ABI < Plugin ABI -> plugin can't be used (plugin should provide interface with lower ABI to handle that)
#define WRITER_ABI_VERSION 1

#else // !defined(BUILD_PLUGIN)

#if !defined(WRITER_ABI_VERSION) || !defined(WRITER_API_VERSION)
#error "Plugin must define WRITER_ABI_VERSION and WRITER_API_VERSION before including plugin_writer_api.hpp"
#endif

#endif // !defined(BUILD_PLUGIN)


#ifdef __cplusplus
extern "C" {
#endif

typedef struct CvPluginWriter_t* CvPluginWriter;

struct OpenCV_VideoIO_Writer_Plugin_API_v1_0_api_entries
{
    /** OpenCV capture ID (VideoCaptureAPIs)
    @note API-ENTRY 1, API-Version == 0
     */
    int id;

    /** @brief Try to open video writer

    @param filename Destination location
    @param fourcc FOURCC code
    @param fps FPS
    @param width frame width
    @param height frame height
    @param isColor true if video stream should save color frames
    @param[out] handle pointer on Writer handle

    @note API-CALL 2, API-Version == 0
     */
    CvResult (CV_API_CALL *Writer_open)(const char* filename, int fourcc, double fps, int width, int height, int isColor,
                                         CV_OUT CvPluginWriter* handle);

    /** @brief Release Writer handle

    @param handle Writer handle

    @note API-CALL 3, API-Version == 0
     */
    CvResult (CV_API_CALL *Writer_release)(CvPluginWriter handle);

    /** @brief Get property value

    @param handle Writer handle
    @param prop Property index
    @param[out] val property value

    @note API-CALL 4, API-Version == 0
     */
    CvResult (CV_API_CALL *Writer_getProperty)(CvPluginWriter handle, int prop, CV_OUT double* val);

    /** @brief Set property value

    @param handle Writer handle
    @param prop Property index
    @param val property value

    @note API-CALL 5, API-Version == 0
     */
    CvResult (CV_API_CALL *Writer_setProperty)(CvPluginWriter handle, int prop, double val);

    /** @brief Write frame

    @param handle Writer handle
    @param data frame data
    @param step step in bytes
    @param width frame width in pixels
    @param height frame height
    @param cn number of channels per pixel

    @note API-CALL 6, API-Version == 0
     */
    CvResult (CV_API_CALL *Writer_write)(CvPluginWriter handle, const unsigned char *data, int step, int width, int height, int cn);
}; // OpenCV_VideoIO_Writer_Plugin_API_v1_0_api_entries

struct OpenCV_VideoIO_Writer_Plugin_API_v1_1_api_entries
{
    /** @brief Try to open video writer

    @param filename Destination location
    @param fourcc FOURCC code
    @param fps FPS
    @param width frame width
    @param height frame height
    @param params pointer on 2*n_params array of 'key,value' pairs
    @param n_params number of passed parameters
    @param[out] handle pointer on Writer handle

    @note API-CALL 7, API-Version == 1
     */
    CvResult (CV_API_CALL* Writer_open_with_params)(
        const char* filename, int fourcc, double fps, int width, int height,
        int* params, unsigned n_params,
        CV_OUT CvPluginWriter* handle
    );
}; // OpenCV_VideoIO_Writer_Plugin_API_v1_1_api_entries

typedef struct OpenCV_VideoIO_Writer_Plugin_API_v1_0
{
    OpenCV_API_Header api_header;
    struct OpenCV_VideoIO_Writer_Plugin_API_v1_0_api_entries v0;
} OpenCV_VideoIO_Writer_Plugin_API_v1_0;

typedef struct OpenCV_VideoIO_Writer_Plugin_API_v1_1
{
    OpenCV_API_Header api_header;
    struct OpenCV_VideoIO_Writer_Plugin_API_v1_0_api_entries v0;
    struct OpenCV_VideoIO_Writer_Plugin_API_v1_1_api_entries v1;
} OpenCV_VideoIO_Writer_Plugin_API_v1_1;


#if WRITER_ABI_VERSION == 1 && WRITER_API_VERSION == 1
typedef struct OpenCV_VideoIO_Writer_Plugin_API_v1_1 OpenCV_VideoIO_Writer_Plugin_API;
#elif WRITER_ABI_VERSION == 1 && WRITER_API_VERSION == 0
typedef struct OpenCV_VideoIO_Writer_Plugin_API_v1_0 OpenCV_VideoIO_Writer_Plugin_API;
#else
#error "Not supported configuration: check WRITER_ABI_VERSION/WRITER_API_VERSION"
#endif

#ifdef BUILD_PLUGIN

CV_PLUGIN_EXPORTS
const OpenCV_VideoIO_Writer_Plugin_API* CV_API_CALL opencv_videoio_writer_plugin_init_v1
        (int requested_abi_version, int requested_api_version, void* reserved /*NULL*/) CV_NOEXCEPT;

#else  // BUILD_PLUGIN
typedef const OpenCV_VideoIO_Writer_Plugin_API* (CV_API_CALL *FN_opencv_videoio_writer_plugin_init_t)
        (int requested_abi_version, int requested_api_version, void* reserved /*NULL*/);
#endif  // BUILD_PLUGIN


#ifdef __cplusplus
}
#endif

#endif // PLUGIN_WRITER_API_HPP
