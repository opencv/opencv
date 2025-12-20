export module opencv.core;

// This minimal module interface re-exports the existing public C API for
// the core module. It is only built when the toolchain supports C++20
// modules and OPENCV_ENABLE_CXX_MODULES is enabled.

export import <opencv2/core.hpp>;
