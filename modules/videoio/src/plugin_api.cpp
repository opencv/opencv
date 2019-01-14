// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef BUILD_PLUGIN

#include "plugin_api.hpp"
#include "opencv2/core/version.hpp"

void cv_get_version(int & major, int & minor, int & patch, int & api, int & abi)
{
    major = CV_VERSION_MAJOR;
    minor = CV_VERSION_MINOR;
    patch = CV_VERSION_REVISION;
    api = API_VERSION;
    abi = ABI_VERSION;
}

#endif // BUILD_PLUGIN
