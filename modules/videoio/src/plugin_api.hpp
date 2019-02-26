// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef PLUGIN_API_HPP
#define PLUGIN_API_HPP

// increase for backward-compatible changes, e.g. add new function
// Main API <= Plugin API -> plugin is compatible
#define API_VERSION 1
// increase for incompatible changes, e.g. remove function argument
// Main ABI == Plugin ABI -> plugin is compatible
#define ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

// common
typedef void cv_get_version_t(int & major, int & minor, int & patch, int & api, int & abi);
typedef int cv_domain_t();

// capture
typedef bool cv_open_capture_t(const char * filename, int camera_index, void * &handle);
typedef bool cv_get_cap_prop_t(void * handle, int prop, double & val);
typedef bool cv_set_cap_prop_t(void * handle, int prop, double val);
typedef bool cv_grab_t(void * handle);
// callback function type
typedef bool cv_retrieve_cb_t(unsigned char * data, int step, int width, int height, int cn, void * userdata);
typedef bool cv_retrieve_t(void * handle, int idx, cv_retrieve_cb_t * cb, void * userdata);
typedef bool cv_release_capture_t(void * handle);

// writer
typedef bool cv_open_writer_t(const char * filename, int fourcc, double fps, int width, int height, int isColor, void * &handle);
typedef bool cv_get_wri_prop_t(void * handle, int prop, double & val);
typedef bool cv_set_wri_prop_t(void * handle, int prop, double val);
typedef bool cv_write_t(void * handle, const unsigned char * data, int step, int width, int height, int cn);
typedef bool cv_release_writer_t(void * handle);

#ifdef BUILD_PLUGIN

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define CV_PLUGIN_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define CV_PLUGIN_EXPORTS __attribute__ ((visibility ("default")))
#endif

CV_PLUGIN_EXPORTS cv_get_version_t cv_get_version;
CV_PLUGIN_EXPORTS cv_domain_t cv_domain;

CV_PLUGIN_EXPORTS cv_open_capture_t cv_open_capture;
CV_PLUGIN_EXPORTS cv_get_cap_prop_t cv_get_cap_prop;
CV_PLUGIN_EXPORTS cv_set_cap_prop_t cv_set_cap_prop;
CV_PLUGIN_EXPORTS cv_grab_t cv_grab;
CV_PLUGIN_EXPORTS cv_retrieve_t cv_retrieve;
CV_PLUGIN_EXPORTS cv_release_capture_t cv_release_capture;

CV_PLUGIN_EXPORTS cv_open_writer_t cv_open_writer;
CV_PLUGIN_EXPORTS cv_get_wri_prop_t cv_get_wri_prop;
CV_PLUGIN_EXPORTS cv_set_wri_prop_t cv_set_wri_prop;
CV_PLUGIN_EXPORTS cv_write_t cv_write;
CV_PLUGIN_EXPORTS cv_release_writer_t cv_release_writer;

#endif


#ifdef __cplusplus
}
#endif

#endif // PLUGIN_API_HPP
