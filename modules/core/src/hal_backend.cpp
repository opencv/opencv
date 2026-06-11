#include "precomp.hpp"
#include "opencv2/core/hal/backend_registry.hpp"
#include <vector>
#include <dlfcn.h>      // dlopen, dlsym, dlclose
#include <mutex>        // std::once_flag, std::call_once
#include <string>
#include <cstdlib>      // getenv

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

// =============================================================
// Plugin loading
// Loads GPU backend plugins (.so) at runtime via dlopen.
// Runs once at startup before any cv:: function is called.
// =============================================================

namespace {

// factory function signature — what plugins export
typedef Backend* (*BackendFactory)();

// Try to load one plugin by filename.
// Returns true if loaded and registered.
bool tryLoadPlugin(const std::string& path) {

    // RTLD_LAZY  — resolve symbols when first used
    // RTLD_GLOBAL— plugin can see opencv_core symbols
    void* handle = dlopen(path.c_str(),
                          RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        // plugin not found at this path — not an error,
        // just means it is not installed here
        return false;
    }

    // clear any stale dlerror
    dlerror();

    // look for the factory function by exact name
    BackendFactory factory = reinterpret_cast<BackendFactory>(
        dlsym(handle, "cv_hal_createCudaBackend"));

    const char* err = dlerror();
    if (err != nullptr || factory == nullptr) {
        // wrong plugin — symbol not present
        dlclose(handle);
        return false;
    }

    // call factory — get a Backend*
    Backend* backend = factory();
    if (backend == nullptr) {
        dlclose(handle);
        return false;
    }

    // register the backend
    // handle stays open — we need the .so loaded for life
    registerBackend(backend);
    return true;
}

// Load all available GPU backend plugins.
void doLoadBackendPlugins() {

    // candidate plugin filenames
    std::vector<std::string> candidates;

    // 1. explicit path from environment variable
    const char* envPath = std::getenv("OPENCV_GPU_BACKEND_PATH");
    if (envPath != nullptr) {
        candidates.push_back(std::string(envPath));
    }

    // 2. bare filename — relies on rpath / LD_LIBRARY_PATH
    candidates.push_back("libopencv_cuda_backend.so");

    // try each candidate — stop after first success
    for (const std::string& path : candidates) {
        if (tryLoadPlugin(path)) {
            break;
        }
    }
    // if none loaded — registry stays empty — CPU fallback
}

} // anonymous namespace

// =============================================================
// loadBackendPlugins()
// Thread-safe, runs the loader exactly once.
// =============================================================
void loadBackendPlugins() {
    static std::once_flag flag;
    std::call_once(flag, doLoadBackendPlugins);
}

// =============================================================
// Static initializer — runs loadBackendPlugins() at startup
// when libopencv_core.so is loaded, before main().
// =============================================================
namespace {
struct BackendPluginInitializer {
    BackendPluginInitializer() {
        loadBackendPlugins();
    }
};
static BackendPluginInitializer g_backendPluginInit;
} // anonymous namespace

}} // cv::hal
