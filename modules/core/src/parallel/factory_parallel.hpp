// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_PARALLEL_FACTORY_HPP
#define OPENCV_CORE_PARALLEL_FACTORY_HPP

#include "opencv2/core/parallel/parallel_backend.hpp"

namespace cv { namespace parallel {

class IParallelBackendFactory
{
public:
    virtual ~IParallelBackendFactory() {}
    virtual std::shared_ptr<cv::parallel::ParallelForAPI> create() const = 0;
};


class StaticBackendFactory CV_FINAL: public IParallelBackendFactory
{
protected:
    std::function<std::shared_ptr<cv::parallel::ParallelForAPI>(void)> create_fn_;

public:
    StaticBackendFactory(std::function<std::shared_ptr<cv::parallel::ParallelForAPI>(void)>&& create_fn)
        : create_fn_(create_fn)
    {
        // nothing
    }

    ~StaticBackendFactory() CV_OVERRIDE {}

    std::shared_ptr<cv::parallel::ParallelForAPI> create() const CV_OVERRIDE
    {
        return create_fn_();
    }
};

//
// PluginBackendFactory is implemented in plugin_wrapper.cpp
//

std::shared_ptr<IParallelBackendFactory> createPluginParallelBackendFactory(const std::string& baseName);

}}  // namespace

#endif  // OPENCV_CORE_PARALLEL_FACTORY_HPP
