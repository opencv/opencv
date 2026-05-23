// Shim for ORT's core/common/narrow.h — used by the vendored MLAS (cast.cpp).
// Upstream provides a checked narrowing cast a la gsl::narrow. The MLAS
// translation units here only #include the header; they do not actually
// invoke narrow<T>(...). We provide a minimal definition anyway so the file
// compiles cleanly and any future MLAS update that does call narrow keeps
// working.
//
// This file is intentionally tiny so OpenCV can keep a stable shim while
// upstream MLAS evolves.

#pragma once

#include <stdexcept>
#include <type_traits>

namespace onnxruntime {

template <typename T, typename U>
constexpr T narrow(U u) {
    static_assert(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
                  "narrow<T>(U): T and U must be arithmetic types");
    const T t = static_cast<T>(u);
    if (static_cast<U>(t) != u ||
        ((t < T{}) != (u < U{}))) {
        throw std::runtime_error("onnxruntime::narrow: narrowing failed");
    }
    return t;
}

}  // namespace onnxruntime
