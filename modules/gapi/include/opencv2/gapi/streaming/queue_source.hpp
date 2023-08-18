// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_QUEUE_SOURCE_HPP
#define OPENCV_GAPI_STREAMING_QUEUE_SOURCE_HPP

#include <memory>                      // shared_ptr
#include <type_traits>                 // is_base_of

#include <opencv2/gapi/gmetaarg.hpp>   // GMetaArg + all descr_of

namespace cv {
namespace gapi {
namespace wip {
struct Data; // "forward-declaration" of GRunArg

class GAPI_EXPORTS QueueSourceBase: public cv::gapi::wip::IStreamSource {
    class Priv;
    std::shared_ptr<Priv> m_priv;

public:
    explicit QueueSourceBase(const cv::GMetaArg &m);
    void push(const Data &data);
    virtual bool pull(Data &data) override;
    virtual void halt() override;
    virtual GMetaArg descr_of() const override;
    virtual ~QueueSourceBase() = default;
};

/**
 * @brief Queued streaming pipeline source.
 *
 */
template<class T>
class QueueSource final: public QueueSourceBase
{
public:
    using Meta = decltype(cv::descr_of(T{}));
    explicit QueueSource(Meta m) : QueueSourceBase(GMetaArg{m}) {
    }
    void push(T t) {
        QueueSourceBase::push(Data{t});
    }
};

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_SOURCE_HPP
