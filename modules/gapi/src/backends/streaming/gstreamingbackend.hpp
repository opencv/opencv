//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you ("License"). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit
// this software or the related documents without Intel's prior written
// permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef OPENCV_GAPI_COPY_KERNEL_HPP
#define OPENCV_GAPI_COPY_KERNEL_HPP

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/streaming.hpp>
#include "gstreamingkernel.hpp"

namespace cv {
namespace gimpl {

class RMatMediaBGRAdapter : public cv::RMat::Adapter
{
    cv::MediaFrame m_frame;
public:
    RMatMediaBGRAdapter(cv::MediaFrame frame) : m_frame(frame) { };

    virtual cv::RMat::View access(cv::RMat::Access a) override
    {
        auto view = m_frame.access(a == cv::RMat::Access::W ? cv::MediaFrame::Access::W
                                                            : cv::MediaFrame::Access::R);

        return cv::RMat::View(desc(),
                              reinterpret_cast<uchar*>(view.ptr[0]),
                              view.stride[0],
                              [=](){});
    }

    virtual cv::GMatDesc desc() const override
    {
        const auto& desc = m_frame.desc();
        GAPI_Assert(desc.fmt == cv::MediaFormat::BGR);
        return cv::GMatDesc{CV_8U, 3, desc.size};
    }
};

struct Copy: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GCopy;

    static gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public cv::gapi::streaming::IActor
    {
        public:
            explicit Actor() {}
            virtual void run(cv::gimpl::GIslandExecutable::IInput  &in,
                             cv::gimpl::GIslandExecutable::IOutput &out) override;
    };

    static cv::gapi::streaming::IActor::Ptr create()
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor());
    }

    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; };
};

struct BGR: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GBGR;
    static gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public cv::gapi::streaming::IActor {
        public:
            explicit Actor() {}
            virtual void run(cv::gimpl::GIslandExecutable::IInput &in,
                             cv::gimpl::GIslandExecutable::IOutput&out) override;
    };

    static cv::gapi::streaming::IActor::Ptr create()
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor());
    }
    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; };
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_COPY_KERNEL_HPtruct Copy: public cv::detail::KernelTag {
