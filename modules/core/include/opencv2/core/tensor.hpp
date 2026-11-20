// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_TENSOR_HPP
#define OPENCV_CORE_TENSOR_HPP

#include "opencv2/core/mat.hpp"
#include "opencv2/core/compute_device.hpp"

namespace cv {

//! @addtogroup core_basic
//! @{

/** @brief An n-dimensional array whose data is on the CPU or on a device, with one interface for both.

A Tensor holds a Mat when its data is on the CPU and a UMat when it is on a device (see
ComputeDevice). Wrapping a Mat or a UMat shares its buffer without copying, and Mat and UMat stay
fully usable alongside it.

A Tensor can be passed to any cv:: function that takes InputArray or OutputArray. A CPU tensor
behaves exactly like a Mat there, and a device tensor like a UMat on that device, so an OpenCL
tensor takes the OpenCL path of the T-API.

Where outputs go:
- a non-empty tensor keeps its device;
- an empty tensor passed to a cv:: function gets the device of this thread's DeviceScope, or the
  CPU outside a scope;
- clone(), copyTo(Tensor&) and convertTo(Tensor&, ...) give an empty destination the device of
  the source.

Data moves between devices only in to(), in copyTo() or convertTo() across devices, in getMat()
of a device tensor, and where one cv:: call mixes arrays on different devices.

@code
cv::Tensor img = cv::Tensor(cv::imread("in.png")).to("cuda:0");
cv::DeviceScope scope(img.device());
cv::Tensor gray;
cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);       // gray is created on cuda:0
cv::Mat result = gray.getMat(cv::ACCESS_READ);     // downloads; the device copy stays valid
@endcode

The element type means what it means in Mat: a depth and a number of channels.

@sa ComputeDevice, DeviceScope, Mat, UMat
*/
class CV_EXPORTS Tensor
{
public:
    //! an empty tensor
    Tensor();

    /** @brief Allocates a tensor on the current device (ComputeDevice::current()). */
    Tensor(const MatShape& shape, int type);

    /** @brief Allocates a tensor on @p device. */
    Tensor(const MatShape& shape, int type, const ComputeDevice& device);

    /** @brief Allocates a tensor on the current device and sets every element to @p value. */
    Tensor(const MatShape& shape, int type, const Scalar& value);

    /** @brief Allocates a tensor on @p device and sets every element to @p value. */
    Tensor(const MatShape& shape, int type, const Scalar& value, const ComputeDevice& device);

    /** @brief A CPU tensor sharing the buffer of @p m. */
    explicit Tensor(const Mat& m);

    /** @brief A tensor sharing the buffer of @p u, on the device that holds it. */
    explicit Tensor(const UMat& u);

    /** @brief The device that holds the data; the CPU for an empty tensor. */
    ComputeDevice device() const;

    /** @brief This tensor on @p device: the same data when it is there already, otherwise a copy. */
    Tensor to(const ComputeDevice& device) const;

    /** @overload
    The device is ComputeDevice::find(spec). */
    Tensor to(const String& spec) const;

    /** @brief A Mat header giving host access to the data.

    For a CPU tensor this is the tensor's own Mat. For a device tensor the data is mapped to host
    memory while the returned Mat lives (see UMat::getMat); on CUDA, ACCESS_READ leaves the device
    copy valid. Release the Mat before the tensor is used on its device again. */
    Mat getMat(AccessFlag access) const;

    /** @brief A UMat header over the data: the tensor's own UMat on a device, Mat::getUMat() on the CPU. */
    UMat getUMat(AccessFlag access) const;

    MatShape shape() const;     //!< the shape, with its layout; channels are in type()
    int type() const;
    int depth() const;
    int channels() const;
    int dims() const;
    size_t total() const;
    size_t elemSize() const;
    bool empty() const;
    bool isContinuous() const;

    /** @brief A tensor of @p newShape over the same data and channels, see Mat::reshape. */
    Tensor reshape(const MatShape& newShape) const;

    /** @brief A view of a sub-array, sharing the data. */
    Tensor operator()(const std::vector<Range>& ranges) const;

    /** @overload
    For 2-D tensors. */
    Tensor operator()(const Rect& roi) const;

    /** @brief Allocates new data unless the tensor has this shape and type already.
    The data stays on the tensor's device; an empty tensor gets the current device. */
    void create(const MatShape& shape, int type);

    /** @brief Drops this header's reference to the data, leaving an empty tensor. */
    void release();

    /** @brief A deep copy on the same device. */
    Tensor clone() const;

    /** @brief Copies the data to @p dst, which keeps its device or, when empty, gets this one's. */
    void copyTo(Tensor& dst) const;

    /** @overload */
    void copyTo(OutputArray dst) const;

    /** @brief Converts the elements to depth @p rtype (see Mat::convertTo).
    @p dst keeps its device or, when empty, gets this one's. */
    void convertTo(Tensor& dst, int rtype, double alpha = 1, double beta = 0) const;

    /** @overload */
    void convertTo(OutputArray dst, int rtype, double alpha = 1, double beta = 0) const;

    /** @brief Sets the elements to @p value, only where @p mask is nonzero when a mask is given. */
    Tensor& setTo(InputArray value, InputArray mask = noArray());

    /** @brief A tensor of zeros on @p device. */
    static Tensor zeros(const MatShape& shape, int type, const ComputeDevice& device = ComputeDevice::current());

private:
    friend _InputArray;
    friend _OutputArray;
    friend _InputOutputArray;

    bool usesMat() const;
    bool usesUMat() const;
    bool prepareOutput();
    void allocate(const MatShape& shape, int type, const ComputeDevice& device);

    Mat m_;     // the data of a CPU tensor
    UMat u_;    // the data of a device tensor
};

//! @} core_basic

} // namespace cv

#endif // OPENCV_CORE_TENSOR_HPP
