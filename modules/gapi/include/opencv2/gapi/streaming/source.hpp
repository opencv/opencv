// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_SOURCE_HPP
#define OPENCV_GAPI_STREAMING_SOURCE_HPP

#include <memory>                      // shared_ptr
#include <type_traits>                 // is_base_of

#include <opencv2/gapi/gmetaarg.hpp>   // GMetaArg


namespace cv {
namespace gapi {
namespace wip {
struct Data; // forward-declaration of Data to avoid circular dependencies

/**
 * @brief Abstract streaming pipeline source.
 *
 * Implement this interface if you want customize the way how data is
 * streaming into GStreamingCompiled.
 *
 * Objects implementing this interface can be passed to
 * GStreamingCompiled using setSource() with cv::gin(). Regular
 * compiled graphs (GCompiled) don't support input objects of this
 * type.
 *
 * Default cv::VideoCapture-based implementation is available, see
 * cv::gapi::wip::GCaptureSource.
 *
 * @note stream sources are passed to G-API via shared pointers, so
 *  please use ptr() when passing a IStreamSource implementation to
 *  cv::gin().
 */
class IStreamSource: public std::enable_shared_from_this<IStreamSource>
{
public:
    using Ptr = std::shared_ptr<IStreamSource>;
    Ptr ptr() { return shared_from_this(); }
    virtual bool pull(Data &data) = 0;
    virtual GMetaArg descr_of() const = 0;
    virtual void halt() {
        // Do nothing by default to maintain compatibility with the existing sources...
        // In fact needs to be decorated atop of the child classes to maintain the behavior
        // FIXME: Make it mandatory in OpenCV 5.0
    };
    virtual ~IStreamSource() = default;
};

template<class T, class... Args>
IStreamSource::Ptr inline make_src(Args&&... args)
{
    static_assert(std::is_base_of<IStreamSource, T>::value,
                  "T must implement the cv::gapi::IStreamSource interface!");
    auto src_ptr = std::make_shared<T>(std::forward<Args>(args)...);
    return src_ptr->ptr();
}

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_SOURCE_HPP
