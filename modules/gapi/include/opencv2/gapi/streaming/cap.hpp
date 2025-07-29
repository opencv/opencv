// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_CAP_HPP
#define OPENCV_GAPI_STREAMING_CAP_HPP

/**
 * YOUR ATTENTION PLEASE!
 *
 * This is a header-only implementation of cv::VideoCapture-based
 * Stream source.  It is not built by default with G-API as G-API
 * doesn't depend on videoio module.
 *
 * If you want to use it in your application, please make sure
 * videioio is available in your OpenCV package and is linked to your
 * application.
 *
 * Note for developers: please don't put videoio dependency in G-API
 * because of this file.
 */
#include <chrono>
#include <map>

#include <opencv2/videoio.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>

namespace cv {
namespace gapi {
namespace wip {

/**
 * @brief OpenCV's VideoCapture-based streaming source.
 *
 * This class implements IStreamSource interface.
 * Its constructor takes the same parameters as cv::VideoCapture does.
 *
 * Please make sure that videoio OpenCV module is available before using
 * this in your application (G-API doesn't depend on it directly).
 *
 * @note stream sources are passed to G-API via shared pointers, so
 *  please gapi::make_src<> to create objects and ptr() to pass a
 *  GCaptureSource to cv::gin().
 */
class GCaptureSource: public IStreamSource
{
public:
    explicit GCaptureSource(int id, const std::map<int, double> &properties = {})
        : cap(id) { prep(properties); }

    explicit GCaptureSource(const std::string &path,
                            const std::map<int, double> &properties = {})
        : cap(path) { prep(properties); }

    void set(int propid, double value) {
        cap.set(propid, value);
    }

    // TODO: Add more constructor overloads to make it
    // fully compatible with VideoCapture's interface.

protected:
    cv::VideoCapture cap;
    cv::Mat first;
    bool first_pulled = false;
    int64_t counter = 0;

    void prep(const std::map<int, double> &properties)
    {
        for (const auto &it : properties) {
            cap.set(it.first, it.second);
        }

        // Prepare first frame to report its meta to engine
        // when needed
        GAPI_Assert(first.empty());
        cv::Mat tmp;
        if (!cap.read(tmp))
        {
            GAPI_Error("Couldn't grab the very first frame");
        }
        // NOTE: Some decode/media VideoCapture backends continue
        // owning the video buffer under cv::Mat so in order to
        // process it safely in a highly concurrent pipeline, clone()
        // is the only right way.
        first = tmp.clone();
    }

    virtual bool pull(cv::gapi::wip::Data &data) override
    {
        if (!first_pulled)
        {
            GAPI_Assert(!first.empty());
            first_pulled = true;
            data = first; // no need to clone here since it was cloned already
        }
        else
        {
            if (!cap.isOpened()) return false;

            cv::Mat frame;
            if (!cap.read(frame))
            {
                // end-of-stream happened
                return false;
            }
            // Same reason to clone as in prep()
            data = frame.clone();
        }
        // Tag data with seq_id/ts
        const auto now = std::chrono::system_clock::now();
        const auto dur = std::chrono::duration_cast<std::chrono::microseconds>
            (now.time_since_epoch());
        data.meta[cv::gapi::streaming::meta_tag::timestamp] = int64_t{dur.count()};
        data.meta[cv::gapi::streaming::meta_tag::seq_id]    = int64_t{counter++};
        return true;
    }

    virtual GMetaArg descr_of() const override
    {
        GAPI_Assert(!first.empty());
        return cv::GMetaArg{cv::descr_of(first)};
    }
};

// NB: Overload for using from python
GAPI_EXPORTS_W cv::Ptr<IStreamSource>
inline make_capture_src(const std::string& path,
                        const std::map<int, double>& properties = {})
{
    return make_src<GCaptureSource>(path, properties);
}

// NB: Overload for using from python
GAPI_EXPORTS_W cv::Ptr<IStreamSource>
inline make_capture_src(const int id,
                        const std::map<int, double>& properties = {})
{
    return make_src<GCaptureSource>(id, properties);
}

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_CAP_HPP
