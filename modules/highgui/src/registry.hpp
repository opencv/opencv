// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HIGHGUI_REGISTRY_HPP
#define OPENCV_HIGHGUI_REGISTRY_HPP

#include "factory.hpp"

namespace cv { namespace highgui_backend {

struct BackendInfo
{
    int priority;     // 1000-<index*10> - default builtin priority
                      // 0 - disabled (OPENCV_UI_PRIORITY_<name> = 0)
                      // >10000 - prioritized list (OPENCV_UI_PRIORITY_LIST)
    std::string name;
    std::shared_ptr<IUIBackendFactory> backendFactory;
};

const std::vector<BackendInfo>& getBackendsInfo();

}} // namespace

#endif // OPENCV_HIGHGUI_REGISTRY_HPP
