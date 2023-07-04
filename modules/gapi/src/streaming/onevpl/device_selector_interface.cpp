// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <stdexcept>
#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>
#include <opencv2/gapi/own/assert.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

const char* to_cstring(AccelType type) {

    switch(type) {
        case AccelType::HOST:
            return "HOST";
        case AccelType::DX11:
            return "DX11";
        case AccelType::VAAPI:
            return "VAAPI";
        default:
            GAPI_DbgAssert(false && "Unexpected AccelType");
            break;
   }
   return "UNKNOWN";
}

Device::Device(Ptr device_ptr, const std::string& device_name, AccelType device_type) :
    name(device_name),
    ptr(device_ptr),
    type(device_type) {
}

Device::~Device() {
}

const std::string& Device::get_name() const {
    return name;
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

namespace detail
{
struct DeviceContextCreator : public IDeviceSelector {
    DeviceScoreTable select_devices() const override { return {};}
    DeviceContexts select_context() override { return {};}

    template<typename Entity, typename ...Args>
    static Entity create_entity(Args &&...args) {
        return IDeviceSelector::create<Entity>(std::forward<Args>(args)...);
    }
};
}

Device create_host_device() {
    return detail::DeviceContextCreator::create_entity<Device>(nullptr,
                                                               "CPU",
                                                               AccelType::HOST);
}

Context create_host_context() {
    return detail::DeviceContextCreator::create_entity<Context>(nullptr,
                                                                AccelType::HOST);
}

Device create_dx11_device(Device::Ptr device_ptr,
                          const std::string& device_name) {
    return detail::DeviceContextCreator::create_entity<Device>(device_ptr,
                                                               device_name,
                                                               AccelType::DX11);
}

Context create_dx11_context(Context::Ptr ctx_ptr) {
    return detail::DeviceContextCreator::create_entity<Context>(ctx_ptr,
                                                                AccelType::DX11);
}

Device create_vaapi_device(Device::Ptr device_ptr,
                           const std::string& device_name) {
    return detail::DeviceContextCreator::create_entity<Device>(device_ptr,
                                                               device_name,
                                                               AccelType::VAAPI);
}

Context create_vaapi_context(Context::Ptr ctx_ptr) {
    return detail::DeviceContextCreator::create_entity<Context>(ctx_ptr,
                                                                AccelType::VAAPI);
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
