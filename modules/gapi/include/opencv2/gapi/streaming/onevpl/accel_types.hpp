// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCEL_TYPES_HPP
#define GAPI_STREAMING_ONEVPL_ACCEL_TYPES_HPP

#include <limits>
#include <string>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

enum class AccelType: uint8_t {
    HOST,
    DX11,
    VAAPI,

    LAST_VALUE = std::numeric_limits<uint8_t>::max()
};

GAPI_EXPORTS const char* to_cstring(AccelType type);

struct IDeviceSelector;
struct GAPI_EXPORTS Device {
    friend struct IDeviceSelector;
    using Ptr = void*;

    ~Device();
    const std::string& get_name() const;
    Ptr get_ptr() const;
    AccelType get_type() const;
private:
    Device(Ptr device_ptr, const std::string& device_name,
           AccelType device_type);

    std::string name;
    Ptr ptr;
    AccelType type;
};

struct GAPI_EXPORTS Context {
    friend struct IDeviceSelector;
    using Ptr = void*;

    ~Context();
    Ptr get_ptr() const;
    AccelType get_type() const;
private:
    Context(Ptr ctx_ptr, AccelType ctx_type);
    Ptr ptr;
    AccelType type;
};

GAPI_EXPORTS Device create_host_device();
GAPI_EXPORTS Context create_host_context();

GAPI_EXPORTS Device create_dx11_device(Device::Ptr device_ptr,
                                       const std::string& device_name);
GAPI_EXPORTS Context create_dx11_context(Context::Ptr ctx_ptr);

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ACCEL_TYPES_HPP
