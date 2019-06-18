// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "workspace.hpp"
#include "memory.hpp"

#include <cstddef>
#include <memory>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    class Workspace::Impl {
    public:
        ManagedPtr<unsigned char> ptr;
    };

    Workspace::Workspace() : impl(std::make_shared<Impl>()) { }

    void Workspace::require(std::size_t bytes) {
        if (bytes > impl->ptr.size())
            impl->ptr.reset(bytes);
    }

    std::size_t Workspace::size() const noexcept { return impl->ptr.size(); }

    DevicePtr<unsigned char> WorkspaceAccessor::get(const Workspace& workspace) {
        return DevicePtr<unsigned char>(workspace.impl->ptr.get());
    }

}}}} /* cv::dnn::cuda4dnn::csl */
