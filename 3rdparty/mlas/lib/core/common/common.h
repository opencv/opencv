// Shim for ORT's core/common/common.h. Provides the small subset of
// macros that MLAS's q4_dq.cpp / q4common.h use: ORT_ENFORCE and ORT_THROW.
// Upstream's common.h pulls in logging, status, exceptions, and lots more
// — none of which MLAS itself needs. We map both macros to throwing
// std::runtime_error since MLAS is built without exception-disable in
// our CMake (see mlasi.h's MLAS_NO_EXCEPTION guard).

#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace onnxruntime {

// Concatenate stream-like arguments into a single string. Supports the
// same `operator<<` chain that ORT_ENFORCE uses for its diagnostic.
template <typename... Args>
inline std::string MlasShimMakeMessage(const Args&... args) {
    std::ostringstream oss;
    using expand = int[];
    (void)expand{0, ((void)(oss << args), 0)...};
    return oss.str();
}

}  // namespace onnxruntime

#define ORT_THROW(...)                                                   \
    do {                                                                 \
        throw std::runtime_error(                                        \
            ::onnxruntime::MlasShimMakeMessage(__VA_ARGS__));            \
    } while (0)

#define ORT_ENFORCE(cond, ...)                                           \
    do {                                                                 \
        if (!(cond)) {                                                   \
            throw std::runtime_error(                                    \
                ::onnxruntime::MlasShimMakeMessage(                      \
                    "ORT_ENFORCE(" #cond ") failed: ", ##__VA_ARGS__));  \
        }                                                                \
    } while (0)

#define ORT_NOT_IMPLEMENTED(...) ORT_THROW("not implemented: ", ##__VA_ARGS__)
