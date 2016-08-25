#include "openvx_hal.hpp"

vxContext * vxContext::getContext()
{
    // not thread safe
    static vxContext instance;
    return &instance;
}
