#include "precomp.hpp"
#include "opencv2/core/hal/backend_registry.hpp"
#include <vector>

namespace cv { namespace hal {

namespace {

//
// getRegistry()
// Returns a reference to the global backend list.
// Static local — guaranteed initialized before first use.
// Avoids static initialization order problems.
//
std::vector<Backend*>& getRegistry() {
    static std::vector<Backend*> registry;
    return registry;
}

} // anonymous namespace

//
// registerBackend()
// Adds a backend to the end of the registry.
// Backends are checked in registration order.
// Does not take ownership of the pointer.
//
void registerBackend(Backend* backend) {
    CV_Assert(backend != nullptr);
    getRegistry().push_back(backend);
}

//
// findBackend()
// Walks the registry in order.
// Returns first backend where support(op_id) is true.
// Returns nullptr if no backend supports this operation.
//
Backend* findBackend(int op_id) {
    for (Backend* b : getRegistry()) {
        if (b->support(op_id))
            return b;
    }
    return nullptr;
}

//
// clearBackends()
// Empties the registry.
// Used in unit tests only.
//
void clearBackends() {
    getRegistry().clear();
}

}} // cv::hal
