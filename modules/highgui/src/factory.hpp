// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UI_FACTORY_HPP
#define OPENCV_UI_FACTORY_HPP

#include "backend.hpp"

namespace cv { namespace highgui_backend {

class IUIBackendFactory
{
public:
    virtual ~IUIBackendFactory() {}
    virtual std::shared_ptr<cv::highgui_backend::UIBackend> create() const = 0;
};


class StaticBackendFactory CV_FINAL: public IUIBackendFactory
{
protected:
    std::function<std::shared_ptr<cv::highgui_backend::UIBackend>(void)> create_fn_;

public:
    StaticBackendFactory(std::function<std::shared_ptr<cv::highgui_backend::UIBackend>(void)>&& create_fn)
        : create_fn_(create_fn)
    {
        // nothing
    }

    ~StaticBackendFactory() CV_OVERRIDE {}

    std::shared_ptr<cv::highgui_backend::UIBackend> create() const CV_OVERRIDE
    {
        return create_fn_();
    }
};

//
// PluginUIBackendFactory is implemented in plugin_wrapper
//

std::shared_ptr<IUIBackendFactory> createPluginUIBackendFactory(const std::string& baseName);

}}  // namespace

#endif  // OPENCV_UI_FACTORY_HPP
