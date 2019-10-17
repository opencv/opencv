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

#include <opencv2/videoio.hpp>
#include <opencv2/gapi/garg.hpp>

namespace cv {
namespace gapi {
namespace wip {

/**
 * @brief OpenCV's VideoCapture-based streaming source.
 *
 * This class implements IStreamSource interface.
 * Its constructor takes the same parameters as cv::VideoCapture does.
 *
 * Please make sure that videoio OpenCV module is avaiable before using
 * this in your application (G-API doesn't depend on it directly).
 *
 * @note stream sources are passed to G-API via shared pointers, so
 *  please gapi::make_src<> to create objects and ptr() to pass a
 *  GCaptureSource to cv::gin().
 */
class GCaptureSource: public IStreamSource
{
public:
    explicit GCaptureSource(int id) : cap(id) {}
    explicit GCaptureSource(const std::string &path) : cap(path) {}

    // TODO: Add more constructor overloads to make it
    // fully compatible with VideoCapture's interface.

protected:
    cv::VideoCapture cap;
    virtual bool pull(cv::gapi::wip::Data &data) override
    {
        if (!cap.isOpened()) return false;
        cv::Mat frame;
        if (!cap.read(frame))
        {
            // end-of-stream happened
            return false;
        }

        // NOTE: Some decode/media VideoCapture backends continue
        // owning the video buffer under cv::Mat so in order to
        // process it safely in a highly concurrent pipeline, clone()
        // is the only right way.
        data = frame.clone();
        return true;
    }
};

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_CAP_HPP
