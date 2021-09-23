// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

Device::Device(Ptr device_ptr, AccelType device_type) :
    ptr(device_ptr),
    type(device_type) {
}

Device::~Device() {
}

Device::Ptr Device::get_ptr() const {
    return ptr;
}

AccelType Device::get_type() const {
    return type;
}


Context::Context(Ptr ctx_ptr, AccelType ctx_type) :
    ptr(ctx_ptr),
    type(ctx_type) {
}

Context::~Context() {
}

Context::Ptr Context::get_ptr() const {
    return ptr;
}

AccelType Context::get_type() const {
    return type;
}



IDeviceSelector::Score::Score(Type val) :
    value(val) {
}

IDeviceSelector::Score::~Score() {
}

IDeviceSelector::Score::operator Type () const {
    return value;
}
IDeviceSelector::Score::Type IDeviceSelector::Score::get() const {
    return value;
}


IDeviceSelector::~IDeviceSelector() {
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
