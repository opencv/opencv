// LibCamera compatibility fixes for different versions
#pragma once

#include <libcamera/version.h>

// Helper macros for libcamera API compatibility
#if LIBCAMERA_VERSION_MAJOR == 0 && LIBCAMERA_VERSION_MINOR >= 2
    // For newer libcamera versions, some APIs have changed
    #define CONTROL_ID(ctrl) ctrl.id()
    #define PROPERTY_ID(prop) prop.id()
#else
    // For older libcamera versions
    #define CONTROL_ID(ctrl) ctrl
    #define PROPERTY_ID(prop) prop
#endif

// Fix for Control set API
namespace libcamera_compat {
    template<typename T>
    void setControl(libcamera::ControlList& controls, const libcamera::Control<T>& control, const T& value) {
#if LIBCAMERA_VERSION_MAJOR == 0 && LIBCAMERA_VERSION_MINOR >= 2
        controls.set(control.id(), libcamera::ControlValue(value));
#else
        controls.set(control, value);
#endif
    }
}
