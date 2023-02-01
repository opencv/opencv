// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_CORE_LLAPI_LLAPI_H
#define OPENCV_CORE_LLAPI_LLAPI_H
/**
@addtogroup core_lowlevel_api

API for OpenCV external plugins:
- HAL accelerators
- VideoIO camera backends / decoders / encoders
- Imgcodecs encoders / decoders

Plugins are usually built separately or before OpenCV (OpenCV can depend on them - like HAL libraries).

Using this approach OpenCV provides some basic low level functionality for external plugins.

@note Preview only (no backward compatibility)

@{
*/

#ifndef CV_API_CALL
//! calling convention (including callbacks)
#define CV_API_CALL
#endif

#ifndef CV_PLUGIN_EXPORTS
#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define CV_PLUGIN_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define CV_PLUGIN_EXPORTS __attribute__ ((visibility ("default")))
#endif
#endif

typedef enum cvResult
{
    CV_ERROR_FAIL = -1,                          //!< Some error occurred (TODO Require to fill exception information)
    CV_ERROR_OK = 0                              //!< No error
} CvResult;

typedef struct OpenCV_API_Header_t
{
    /** @brief valid size of this structure
     @details `assert(api.header.valid_size >= sizeof(OpenCV_<Name>_API_v<N>));`
     */
    size_t valid_size;
    unsigned min_api_version;                    //!< backward compatible API version
    unsigned api_version;                        //!< provided API version (features)
    unsigned opencv_version_major;               //!< compiled OpenCV version
    unsigned opencv_version_minor;               //!< compiled OpenCV version
    unsigned opencv_version_patch;               //!< compiled OpenCV version
    const char* opencv_version_status;           //!< compiled OpenCV version
    const char* api_description;                 //!< API description (debug purposes only)
} OpenCV_API_Header;



#if 0

typedef int (CV_API_CALL *cv_example_callback1_cb_t)(unsigned const char* cb_result, void* cb_context);

struct OpenCV_Example_API_v1
{
    OpenCV_API_Header header;

    /** @brief Some API call

    @param param1 description1
    @param param2 description2

    @note API-CALL 1, API-Version >=1
     */
    CvResult (CV_API_CALL *Request1)(int param1, const char* param2);

    /** @brief Register callback

    @param callback function to handle callback
    @param cb_context context data passed to callback function
    @param[out] cb_handle callback id (used to unregister callback)

    @note API-CALL 2, API-Version >=1
     */
    CvResult (CV_API_CALL *RegisterCallback)(cv_example_callback1_cb_t callback, void* cb_context, CV_OUT unsigned* cb_handle);

    /** @brief Unregister callback

    @param cb_handle callback handle

    @note API-CALL 3, API-Version >=1
     */
    CvResult (CV_API_CALL *UnegisterCallback)(unsigned cb_handle);

    ...
};
#endif // 0

//! @}

#endif // OPENCV_CORE_LLAPI_LLAPI_H
