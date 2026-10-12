// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_UTILS_HOST_DEVICE_BUFFER_HPP
#define OPENCV_UTILS_HOST_DEVICE_BUFFER_HPP

#include <opencv2/core.hpp>

namespace cv { namespace utils {

//! @addtogroup core_utils
//! @{

/** @brief Host, CUDA and OpenCL storage for one value, moved between them only by explicit copies.
@note This class is considered private and should be used only in OpenCV itself. API can be changed. */
class CV_EXPORTS HostDeviceBuffer
{
public:
    enum Location { NONE = 0, HOST = 1, CUDA = 2, OPENCL = 4 };

    HostDeviceBuffer();
    //! Move-only: a copy would share the storage but track validity separately.
    HostDeviceBuffer(const HostDeviceBuffer&) = delete;
    HostDeviceBuffer& operator=(const HostDeviceBuffer&) = delete;
    //! The moved-from buffer is left with no current value.
    HostDeviceBuffer(HostDeviceBuffer&& other) CV_NOEXCEPT;
    HostDeviceBuffer& operator=(HostDeviceBuffer&& other) CV_NOEXCEPT;

    //! Host storage for a value that will be fully overwritten; nothing is copied.
    Mat& hostForWrite(const MatShape& shape, int type);
    //! The current value on the host, downloaded first when the host copy is stale.
    const Mat& hostForRead();

    //! CUDA storage for a value that will be fully overwritten, or NULL when CUDA can't be used.
    UMat* cudaForWrite(const MatShape& shape, int type);
    //! The current value on CUDA, uploaded through the host when stale, or NULL when CUDA can't be used.
    UMat* cudaForRead();

    //! OpenCL storage for a value that will be fully overwritten, or NULL when OpenCL can't be used.
    UMat* oclForWrite(const MatShape& shape, int type);
    //! The current value on OpenCL, uploaded through the host when stale, or NULL when OpenCL can't be used.
    UMat* oclForRead();

    //! Makes the value current at the first of CUDA -> OpenCL -> HOST that works, starting at @p first.
    Location fallbackForRead(Location first = CUDA);
    //! Fits storage at the first of CUDA -> OpenCL -> HOST that works, starting at @p first.
    Location fallbackForWrite(const MatShape& shape, int type, Location first = CUDA);

    //! Marks @p where as the only current copy, once its producer has finished writing.
    void commit(Location where);
    //! Forgets the current value but keeps the allocations for the next one.
    void reset();
    //! Frees all storage and forgets any backend that failed.
    void release();

    //! Bitmask of the locations that hold the current value.
    int valid() const { return valid_; }

private:
    UMat* deviceForWrite(Location where, const MatShape& shape, int type);
    UMat* deviceForRead(Location where);

    Mat host_;
    UMat cuda_, ocl_;
    int valid_;
    int failed_;  // backends that failed once and are skipped until release()
};

//! @}

}} // cv::utils::

#endif
