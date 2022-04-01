// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_INTERFACE_HPP
#define GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_INTERFACE_HPP

#include <limits>
#include <map>
#include <string>
#include <vector>

#include <opencv2/gapi/streaming/onevpl/accel_types.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct GAPI_EXPORTS IDeviceSelector {
    using Ptr = std::shared_ptr<IDeviceSelector>;

    struct GAPI_EXPORTS Score {
        friend struct IDeviceSelector;
        using Type = int16_t;
        static constexpr Type MaxActivePriority = std::numeric_limits<Type>::max();
        static constexpr Type MinActivePriority = 0;
        static constexpr Type MaxPassivePriority = MinActivePriority - 1;
        static constexpr Type MinPassivePriority = std::numeric_limits<Type>::min();

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
    using DeviceContexts = std::vector<Context>;

    virtual ~IDeviceSelector();
    virtual DeviceScoreTable select_devices() const = 0;
    virtual DeviceContexts select_context() = 0;
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

#endif // GAPI_STREAMING_ONEVPL_DEVICE_SELECTOR_INTERFACE_HPP
