// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_INTERFACE_HPP
#define GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_INTERFACE_HPP

#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

enum class AccelType : uint8_t {
    HOST,
    DX11
};

struct IDeviceSelector;
struct GAPI_EXPORTS Device {
    friend struct IDeviceSelector;
    using Ptr = void*;

    ~Device();
    Ptr get_ptr() const;
    AccelType get_type() const;
private:
    Device(Ptr device_ptr, AccelType device_type);
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

struct GAPI_EXPORTS IDeviceSelector {
    using Ptr = std::shared_ptr<IDeviceSelector>;

    struct GAPI_EXPORTS Score {
        friend struct IDeviceSelector;
        using Type = uint16_t;
        static constexpr Type Max = std::numeric_limits<Type>::max();
        static constexpr Type Min = std::numeric_limits<Type>::min();

        Score(Type val);
        ~Score();

        operator Type () const;
        Type get() const;
        friend bool operator< (Score lhs, Score rhs) {
            return lhs.get() < rhs.get();
        }
    private:
        Type value;
    };

    using DeviceScoreTable = std::map<Score, Device>;

    virtual ~IDeviceSelector();
    virtual DeviceScoreTable select_devices() const = 0;
    virtual DeviceScoreTable select_spare_devices() const = 0;
    virtual Context select_context(const DeviceScoreTable& selected_devices) = 0;
    virtual Context get_last_context() const = 0;
protected:
    template<typename Entity, typename ...Args>
    static Entity create(Args &&...args) {
        return Entity(std::forward<Args>(args)...);
    }
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
