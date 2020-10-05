// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_MEDIA_HPP
#define OPENCV_GAPI_MEDIA_HPP

#include <memory>     // unique_ptr<>, shared_ptr<>
#include <array>      // array<>
#include <functional> // function<>
#include <utility>    // forward<>()

#include <opencv2/gapi/gframe.hpp>

namespace cv {

class GAPI_EXPORTS MediaFrame {
public:
    enum class Access { R, W };
    class IAdapter;
    class View;
    using AdapterPtr = std::unique_ptr<IAdapter>;

    MediaFrame();
    explicit MediaFrame(AdapterPtr &&);
    template<class T, class... Args> static cv::MediaFrame Create(Args&&...);

    View access(Access) const;
    cv::GFrameDesc desc() const;

private:
    struct Priv;
    std::shared_ptr<Priv> m;
};

template<class T, class... Args>
inline cv::MediaFrame cv::MediaFrame::Create(Args&&... args) {
    std::unique_ptr<T> ptr(new T(std::forward<Args>(args)...));
    return cv::MediaFrame(std::move(ptr));
}

class GAPI_EXPORTS MediaFrame::View final {
public:
    static constexpr const size_t MAX_PLANES = 4;
    using Ptrs     = std::array<void*, MAX_PLANES>;
    using Strides  = std::array<std::size_t, MAX_PLANES>; // in bytes
    using Callback = std::function<void()>;

    View(Ptrs&& ptrs, Strides&& strs, Callback &&cb = [](){});
    View(const View&) = delete;
    View(View&&) = default;
    ~View();

    Ptrs    ptr;
    Strides stride;

private:
    Callback m_cb;
};

class GAPI_EXPORTS MediaFrame::IAdapter {
public:
    virtual ~IAdapter() = 0;
    virtual cv::GFrameDesc meta() const = 0;
    virtual MediaFrame::View access(MediaFrame::Access) = 0;
};

} //namespace cv

#endif // OPENCV_GAPI_MEDIA_HPP
