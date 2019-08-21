// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_NVTX_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_NVTX_HPP

#include <nvtx3/nvToolsExt.h>

//#define CUDA4DNN_ENABLE_NVTX

#if defined CUDA4DNN_ENABLE_NVTX && defined NDEBUG
#ifdef _MSC_VER
#pragma message("WARNING: NVTX enabled in release build")
#else
#warning "NVTX enabled in release build"
#endif
#endif

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace nvtx {

    void mark(const char* str) {
#ifdef CUDA4DNN_ENABLE_NVTX
        nvtxMarkA(str);
#endif
    }

    class Range {
    public:
        Range() noexcept : active { false } { }
        Range(const char* str) : active { false } {
            start(str);
        }

        ~Range() {
            end();
        }

        void start(const char* str) {
            end();

#ifdef CUDA4DNN_ENABLE_NVTX
            id = nvtxRangeStartA(str);
            active = true;
#endif
        }

        void end() {
#ifdef CUDA4DNN_ENABLE_NVTX
            if (active) {
                nvtxRangeEnd(id);
                active = false;
            }
#endif
        }

    private:
#ifdef CUDA4DNN_ENABLE_NVTX
        nvtxRangeId_t id;
#endif

        bool active;
    };

}}}}} /* cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_NVTX_HPP */
