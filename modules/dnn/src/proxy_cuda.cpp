// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "op_cuda.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /* when the dnn module is compiled without CUDA support, the CSL components are not compiled
    ** however, the headers which are exposed require proxy implementations to compile
    */

#ifndef HAVE_CUDA
    class StreamAccessor { };
    class Stream::UniqueStream { };

    Stream::Stream() { }
    Stream::Stream(const Stream&) noexcept = default;
    Stream::Stream(Stream&&) noexcept = default;
    Stream::Stream(bool create) { }
    Stream& Stream::operator=(const Stream&) noexcept = default;
    Stream& Stream::operator=(Stream&&) noexcept = default;
    void Stream::synchronize() const { }
    bool Stream::busy() const { return false; }
    Stream::operator bool() const noexcept { return false; }

    namespace cublas {
        class HandleAccessor { };
        class Handle::UniqueHandle { };

        Handle::Handle() { }
        Handle::Handle(const Handle&) noexcept = default;
        Handle::Handle(Handle&&) noexcept = default;
        Handle::Handle(Stream strm) { }
        Handle& Handle::operator=(const Handle&) noexcept = default;
        Handle& Handle::operator=(Handle&&) noexcept = default;
        Handle::operator bool() const noexcept { return false; }
    }

    namespace cudnn {
        class HandleAccessor { };
        class Handle::UniqueHandle { };

        Handle::Handle() { }
        Handle::Handle(const Handle&) noexcept = default;
        Handle::Handle(Handle&&) noexcept = default;
        Handle::Handle(Stream strm) { }
        Handle& Handle::operator=(const Handle&) noexcept = default;
        Handle& Handle::operator=(Handle&&) noexcept = default;
        Handle::operator bool() const noexcept { return false; }
    }

    class WorkspaceAccessor { };
    class Workspace::Impl { };

    Workspace::Workspace() { }
    void Workspace::require(std::size_t bytes) { }
    std::size_t Workspace::size() const noexcept { return 0; }

#endif

}}}} /* cv::dnn::cuda4dnn::csl */
