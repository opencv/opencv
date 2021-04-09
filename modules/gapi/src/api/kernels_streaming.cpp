// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"

#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#include <opencv2/gapi/core.hpp>

cv::GMat cv::gapi::streaming::desync(const cv::GMat &g) {
    // FIXME: this is a limited implementation of desync
    // The real implementation must be generic (template) and
    // reside in desync.hpp (and it is detail::desync<>())

    // FIXME: Put a copy here to solve the below problem
    // FIXME: Because of the copy, the desync functionality is limited
    // to GMat only (we don't have generic copy kernel for other
    // object types)
    return cv::gapi::copy(detail::desync(g));

    // FIXME
    //
    // If consumed by multiple different islands (OCV and Fluid by
    // example, an object needs to be desynchronized individually
    // for every path.
    //
    // This is a limitation of the current implementation. It works
    // this way: every "desync" link from the main path to a new
    // desync path gets its "DesyncQueue" object which stores only the
    // last value written before of the desync object (DO) it consumes
    // (the container of type "last written value" or LWV.
    //
    //                         LWV
    // [Sync path] -> desync() - - > DO -> [ISL0 @ Desync path #1]
    //
    // At the same time, generally, every island in the streaming
    // graph gets its individual input as a queue (so normally, a
    // writer pushes the same output MULTIPLE TIMES if it has mutliple
    // readers):
    //
    //                         LWV
    // [Sync path] -> desync() - - > DO1 -> [ISL0 @ Desync path #1]
    //                       : LWV
    //                       ' - - > DO2 -> [ISL1 @ Desync path #1]
    //
    // For users, it may seem legit to use desync here only once, and
    // it MUST BE legit once the problem is fixed.
    // But the problem with the current implementation is that islands
    // on the same desync path get different desync queues and in fact
    // stay desynchronized between each other. One shouldn't consider
    // this as a single desync path anymore.
    // If these two ISLs are then merged e.g. with add(a,b), the
    // results will be inconsistent, given that the latency of ISL0
    // and ISL1 may be different. This is not the same frame anymore
    // coming as `a` and `b` to add(a,b) because of it.
    //
    // To make things clear, we forbid this now and ask to call
    // desync one more time to allow that. It is bad since the graph
    // structure and island layout depends on kernel packages used,
    // not on the sole GComputation structure. This needs to be fixed!
    // Here's the working configuration:
    //
    //                         LWV
    // [Sync path] -> desync() - - > DO1 -> [ISL0 @ Desync path #1]
    //            :            LWV
    //            '-> desync() - - > DO2 -> [ISL1 @ Desync path #2] <-(!)
    //
    // Put an operation right after desync() is a quick workaround to
    // this synchronization problem. There will be one "last_written_value"
    // connected to a desynchronized data object, and this sole last_written_value
    // object will feed both branches of the streaming executable.
}

cv::GMat cv::gapi::streaming::BGR(const cv::GFrame& in) {
    return cv::gapi::streaming::GBGR::on(in);
}

cv::GMat cv::gapi::streaming::Y(const cv::GFrame& in){
    return cv::gapi::streaming::GY::on(in);
}

cv::GMat cv::gapi::streaming::UV(const cv::GFrame& in){
    return cv::gapi::streaming::GUV::on(in);
}
