/**
 * simoncatbot-opencv RPP HAL - Unified Dispatch Layer
 *
 * Architecture:
 *   1. GPU path: Uses RPP HIP backend (amdhip64)
 *   2. CPU path: Uses RPP HOST backend (multithreaded CPU kernels)
 *   3. Fallback: Returns CV_HAL_ERROR_NOT_IMPLEMENTED → OpenCV native
 */

#ifndef __RPP_HAL_UTILS_HPP__
#define __RPP_HAL_UTILS_HPP__

#include <opencv2/core.hpp>
#include <rpp/rppt.h>
#include <rpp/rppdefs.h>

namespace cv { namespace hal { namespace rpp {

// ---------------------------------------------------------------------------
// Availability checks
// ---------------------------------------------------------------------------

/** Returns true if any RPP backend is usable (GPU or CPU) */
bool isRppAvailable();

/** Returns true if RPP HIP GPU backend is usable */
bool isRppGpuAvailable();

/** Returns true if RPP HOST CPU backend is usable */
bool isRppCpuAvailable();

// ---------------------------------------------------------------------------
// Descriptor builders (shared between GPU and CPU paths)
// ---------------------------------------------------------------------------

/** Build RpptDesc for NHWC layout (matches OpenCV interleaved) */
void buildRppDescNHWC(RpptDesc& desc, int width, int height, int channels, int depth);

/** Build RpptROI for full image */
void buildFullRoi(RpptROI& roi, int width, int height);

// ---------------------------------------------------------------------------
// Data type conversion
// ---------------------------------------------------------------------------

RpptDataType cvDepthToRppDataType(int cvDepth);

// ---------------------------------------------------------------------------
// Memory helpers for GPU path
// ---------------------------------------------------------------------------

/** Allocate contiguous device buffer and copy from OpenCV mat (row-by-row) */
bool uploadRawToHip(const void* host_ptr, size_t step, int w, int h, int depth, int cn, void** out_dev_ptr);

/** Copy from contiguous device buffer back to OpenCV mat (row-by-row) */
bool downloadRawFromHip(void* dev_ptr, void* host_ptr, size_t step, int w, int h, int depth, int cn);

/** Free HIP device pointer */
void freeHipPtr(void* devPtr);

// ---------------------------------------------------------------------------
// RPP Handle helpers
// ---------------------------------------------------------------------------

/** Create RPP handle for GPU backend. Returns nullptr on failure. */
rppHandle_t createRppGpuHandle(size_t batchSize = 1);

/** Create RPP handle for CPU backend. Returns nullptr on failure. */
rppHandle_t createRppCpuHandle(size_t batchSize = 1, Rpp32u numThreads = 0);

/** Destroy RPP handle (HIP backend) — works around internal hipFree bug */
void destroyRppGpuHandle(rppHandle_t handle);

/** Destroy RPP handle (HOST backend) */
void destroyRppCpuHandle(rppHandle_t handle);

}}} // namespace cv::hal::rpp

#endif // __RPP_HAL_UTILS_HPP__
