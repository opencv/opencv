// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "backend.hpp"

#include <opencv2/core/private.hpp>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#ifdef NDEBUG
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#else
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#endif
#include <opencv2/core/utils/logger.hpp>

#include "factory.hpp"

#include "plugin_api.hpp"
#include "plugin_wrapper.impl.hpp"


namespace cv { namespace dnn_backend {

NetworkBackend::~NetworkBackend()
{
    // nothing
}

}}  // namespace cv::dnn_backend
