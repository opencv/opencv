/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_ARRAY_HELPERS_HPP
#define OPENCV_CORE_ARRAY_HELPERS_HPP

#include "opencv2/core/base.hpp"
#include "opencv2/core/check.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/mat_def.hpp"
#include "opencv2/core/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

namespace cv {
template<class> struct _InputArrayOps;

struct _InputArrayOpsBase {
    virtual Mat getMat_(const _InputArray& self, int idx) const = 0;
    virtual UMat getUMat(const _InputArray& self, int i) const = 0;
    virtual std::vector<Mat> getMatVector(const _InputArray& self) const = 0;
    virtual std::vector<UMat> getUMatVector(const _InputArray& self) const = 0;
    virtual std::vector<cuda::GpuMat> getGpuMatVector(const _InputArray& self) const = 0;
    virtual cuda::GpuMat getGpuMat(const _InputArray& self) const = 0;
    virtual ogl::Buffer getOGlBuffer(const _InputArray& self) const = 0;
    virtual int dims(const _InputArray& self, int i) const = 0;
    virtual Size size(const _InputArray& self, int idx) const = 0;
    virtual int sizend(const _InputArray& self, int* arraySize, int i) const = 0;
    virtual bool sameSize(const _InputArray& self, const _InputArray& other) const = 0;

    virtual int type(const _InputArray& self, int idx) const = 0;
    virtual int isContinuous(const _InputArray& self, int idx) const = 0;
    virtual int isSubmatrix(const _InputArray& self, int idx) const = 0;
    virtual int empty(const _InputArray& self) const = 0;
    virtual void copyTo(const _InputArray& self, const _OutputArray& arr) const = 0;
    virtual void copyTo(const _InputArray& self, const _OutputArray& arr, const _InputArray& mask) const = 0;
    virtual std::size_t offset(const _InputArray& self, std::size_t idx) const = 0;
    virtual std::size_t step(const _InputArray& self, std::size_t idx) const = 0;

    virtual bool isMat() const = 0;
    virtual bool isUMat() const = 0;
    virtual bool isMatVector() const = 0;
    virtual bool isUMatVector() const = 0;
    virtual bool isMatx() const = 0;
    virtual bool isVector() const = 0;
    virtual bool isGpuMat() const = 0;
    virtual bool isGpuMatVector() const = 0;

    virtual void create(const _OutputArray& arr,
                        Size sz,
                        int mtype,
                        int i,
                        bool allowTransposed,
                        _OutputArray::DepthMask fixedDepthMask,
                        bool fixedSize,
                        bool fixedType) const = 0;
    virtual void create(const _OutputArray& arr,
                        int rows,
                        int cols,
                        int mtype,
                        int i,
                        bool allowTransposed,
                        _OutputArray::DepthMask fixedDepthMask,
                        bool fixedSize,
                        bool fixedType) const = 0;
    virtual void create(const _OutputArray& arr,
                        int d,
                        const int* sizes,
                        int mtype,
                        int i,
                        bool allowTransposed,
                        _OutputArray::DepthMask fixedDepthMask,
                        bool fixedSize,
                        bool fixedType) const = 0;
    virtual void release(const _OutputArray& self, bool fixedSize) const = 0;
    virtual void clear(const _OutputArray& self, bool fixedSize) const = 0;
    virtual Mat& getMatRef(const _OutputArray& self, int i) const = 0;
    virtual UMat& getUMatRef(const _OutputArray& self, int i) const = 0;
    virtual ogl::Buffer& getOGlBufferRef(const _OutputArray& self) const = 0;
    virtual cuda::HostMem& getHostMemRef(const _OutputArray& self) const = 0;
    virtual cuda::GpuMat& getGpuMatRef(const _OutputArray& self) const = 0;
    virtual std::vector<cuda::GpuMat>& getGpuMatVecRef(const _OutputArray& self) const = 0;
    virtual void setTo(const _OutputArray& self, const _InputArray& arr, const _InputArray& mask) const = 0;
    virtual void assign(const _OutputArray& self, const UMat& u) const = 0;
    virtual void assign(const _OutputArray& self, const Mat& m) const = 0;
    virtual void assign(const _OutputArray& self, const std::vector<Mat>& other) const = 0;
    virtual void assign(const _OutputArray& self, const std::vector<UMat>& other) const = 0;
    virtual void move(const _OutputArray& self, UMat& u, bool fixed_size) const = 0;
    virtual void move(const _OutputArray& self, Mat& m, bool fixed_size) const = 0;

    template <class T>
    static const T& as(const void* const data) {
        return *static_cast<const T*>(data);
    }

    template <class T>
    static T& as(void* const data) {
        return *static_cast<T*>(data);
    }
protected:
    ~_InputArrayOpsBase() = default;

    [[noreturn]] static void noCudaError();
    [[noreturn]] static void noOpenGLError();
    [[noreturn]] static void unsupportedTypeError();
    [[noreturn]] static void customError(Error::Code error, std::string_view message);
    [[noreturn]] static void createError(std::size_t size);
};

template<class T>
inline constexpr bool is_mat_ = false;

template<class T>
inline constexpr bool is_mat_<Mat_<T>> = true;

template<class T>
inline constexpr bool is_mat = std::is_same_v<T, Mat> || is_mat_<T>;

template<class T>
inline constexpr bool is_umat = std::is_same_v<T, UMat>;

template<class T>
inline constexpr bool is_matx = false;

template<class T>
inline constexpr bool is_cpu_matrix = is_mat<T> || is_umat<T>;

template<class T, int M, int N>
inline constexpr bool is_matx<Matx<T, M, N>> = true;

template<class T>
inline constexpr bool is_none = std::is_null_pointer_v<T>;

template<class T>
inline constexpr bool is_vector = false;

template<class T, class Alloc>
inline constexpr bool is_vector<std::vector<T, Alloc>> = true;

template<class T>
inline constexpr bool is_array_of_mat = false;

template<class T, std::size_t N>
inline constexpr bool is_array_of_mat<std::array<T, N>> = is_mat<T>;

template<class T>
inline constexpr bool is_opengl_buffer = std::is_same_v<T, ogl::Buffer>;

template<class T>
inline constexpr bool is_cuda_host_mem = std::is_same_v<T, cuda::HostMem>;

template<class T>
inline constexpr bool is_cuda_gpu_mat = std::is_same_v<T, cuda::GpuMat>;

template<class Expected, class T, bool = is_vector<T>>
inline constexpr bool is_vector_of = false;

template<class Expected, class T, class Alloc>
inline constexpr bool is_vector_of<Expected, std::vector<T, Alloc>, true> = std::is_same_v<T, Expected>;

template<class T>
inline constexpr bool is_vector_of_mat = is_vector_of<Mat, T>;

template<class T, class Alloc>
inline constexpr bool is_vector_of_mat<std::vector<Mat_<T>, Alloc>> = true;

template<class T>
inline constexpr bool is_cuda_type = is_cuda_host_mem<T> || is_cuda_gpu_mat<T> || is_vector_of<cuda::GpuMat,T>;

#ifdef HAVE_CUDA
inline constexpr bool have_cuda = true;
#else
inline constexpr bool have_cuda = false;
#endif // HAVE_CUDA

#ifdef HAVE_OPENGL
inline constexpr bool have_opengl = true;
#else
inline constexpr bool have_opengl = false;
#endif // HAVE_OPENGL

template<class T>
struct _InputArrayOps : _InputArrayOpsBase {
    Mat getMat_(const _InputArray& self, int i) const override;

    UMat getUMat(const _InputArray& self, const int i) const override
    {
        void* const p = self.getObj();
        const int flags = self.getFlags();

        if constexpr (is_umat<T>) {
            const auto& mat = as<T>(p);
            return i < 0 ? mat : mat.row(i);
        }
        else if constexpr (is_vector_of<UMat, T>) {
            const auto& v = as<T>(p);
            CV_Assert(0 <= i);

            const auto index = static_cast<std::size_t>(i);
            CV_Assert(index < v.size());
            return v[index];
        }
        else if constexpr (is_mat<T>) {
            const auto& mat = as<T>(p);
            const AccessFlag access = flags & ACCESS_MASK;
            return i < 0 ? mat.getUMat(access) : mat.row(i).getUMat(access);
        }
        else if constexpr (is_none<T>) {
            return UMat();
        }
        else {
            return getMat_(self, i).getUMat(flags & ACCESS_MASK);
        }
    }

    std::vector<Mat> getMatVector(const _InputArray& self) const override
    {
        void* const p = self.getObj();
        const int flags = self.getFlags();
        const Size sz = self.getSz();

        if constexpr (is_mat<T>) {
            const auto& mat = as<Mat>(p);
            std::vector<Mat> result;
            const auto size = static_cast<std::size_t>(mat.size[0]);
            result.reserve(size);
            for (std::size_t i = 0; i < size; ++i) {
                if (mat.dims == 2) {
                    result.emplace_back(1, mat.cols, mat.type(), (void*)mat.ptr(i));
                }
                else {
                    result.emplace_back(mat.dims - 1, &mat.size[1], mat.type(), (void*)mat.ptr(i), &mat.step[1]);
                }
            }

            return result;
        }
        else if constexpr (is_matx<T>) {
            const auto& mat = as<T>(p);
            std::vector<Mat> result;
            const auto size = static_cast<std::size_t>(sz.height);
            result.reserve(size);
            for (std::size_t i = 0; i < size; ++i) {
                result.emplace_back(1, sz.width, CV_MAT_TYPE(flags), mat.val[sz.width * i]);
            }
            return result;
        }
        else if constexpr (std::is_scalar_v<T>) {
            return std::vector<Mat>(1, Mat(1, 1, CV_MAT_TYPE(flags), as<T>(p)));
        }
        else if constexpr (is_vector<T>) {
            using value_type = typename T::value_type;
            auto& v = as<T>(p);

            if constexpr (is_vector<value_type>) {
                const int type = CV_MAT_TYPE(flags);
                std::vector<Mat> result;
                result.reserve(v.size());
                for (std::size_t i = 0; i < v.size(); ++i) {
                    const Size s = size(self, i);
                    if constexpr (is_vector_of<bool, value_type>) {
                        std::vector<char> as_bytes(v[i].begin(), v[i].end());
                        result.emplace_back(s, type, as_bytes.data());
                    }
                    else {
                        result.emplace_back(s, type, v[i].data());
                    }
                }
                return result;
            }
            else if constexpr (is_mat<value_type>) {
                return {v.begin(), v.end()};
            }
            else if constexpr (is_umat<value_type>) {
                std::vector<Mat> result;
                result.reserve(v.size());
                for (const UMat& mat : v) {
                    result.push_back(mat.getMat(flags & ACCESS_MASK));
                }

                return result;
            }
            else if constexpr (!is_vector_of<bool, T>) {
                const auto width = static_cast<std::size_t>(size(self, -1).width);
                const int depth = CV_MAT_DEPTH(flags);
                const int cn = CV_MAT_CN(flags);

                std::vector<Mat> result;
                result.reserve(width);
                for (std::size_t i = 0; i < width; ++i) {
                    result.emplace_back(1, cn, depth, (void*)&v[i]);
                }

                return result;
            }
            else {
                unsupportedTypeError();
            }
        }
        else if constexpr (is_none<T>) {
            return {};
        }
        else if constexpr (is_array_of_mat<T>) {
            const auto& a = as<T>(p);
            return std::vector<Mat>(a.begin(), a.end());
        }
        else {
            unsupportedTypeError();
        }
    }

    std::vector<UMat> getUMatVector(const _InputArray& self) const override
    {
        void* const p = self.getObj();
        const int flags = self.getFlags();

        if constexpr (is_none<T>) {
            return {};
        }
        else if constexpr (is_vector_of_mat<T> || is_array_of_mat<T>) {
            const auto& v = as<T>(p);
            std::vector<UMat> result;
            result.reserve(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) {
                result.emplace_back(v[i].getUMat(flags & ACCESS_MASK));
            }
            return result;
        }
        else if constexpr (is_vector_of<UMat, T>) {
            return as<T>(p);
        }
        else if constexpr (is_mat<T>) {
            return {as<Mat>(p).getUMat(flags & ACCESS_MASK)};
        }
        else if constexpr (is_umat<T>) {
            return {as<UMat>(p)};
        }
        else {
            unsupportedTypeError();
        }
    }

    std::vector<cuda::GpuMat> getGpuMatVector(const _InputArray& self) const override
    {
        if constexpr (have_cuda) {
            if constexpr (is_vector_of<cuda::GpuMat, T>) {
                return as<std::vector<cuda::GpuMat>>(self.getObj());
            }
            else {
                unsupportedTypeError();
            }
        }
        else {
            noCudaError();
        }
    }

    cuda::GpuMat getGpuMat(const _InputArray& self) const override
    {
        if constexpr (have_cuda) {
            if constexpr (is_cuda_gpu_mat<T>) {
                return as<T>(self.getObj());
            }
            else if constexpr (is_cuda_host_mem<T>) {
                return as<T>(self.getObj())->createGpuMatHeader();
            }
            else if constexpr (is_opengl_buffer<T>) {
                customError(Error::StsNotImplemented, "You should explicitly call mapDevice/unmapDevice methods for ogl::Buffer object");
            }
            else if constexpr (is_none<T>) {
                return cuda::GpuMat();
            }
            else {
                customError(Error::StsNotImplemented, "getGpuMat is available only for cuda::GpuMat and cuda::HostMem");
            }
        }
        else {
            noCudaError();
        }
    }

    ogl::Buffer getOGlBuffer(const _InputArray& self) const override
    {
        if constexpr (is_opengl_buffer<T>) {
            return as<ogl::Buffer>(self.getObj());
        }
        else {
            customError(Error::StsNotImplemented, "getOGlBuffer is only available for ogl::Buffer");
        }
    }

    int dims(const _InputArray& self, int const i) const override
    {
        if constexpr (is_mat<T> || is_umat<T>) {
            CV_Assert(i < 0);
            return as<T>(self.getObj()).dims;
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            CV_Assert(i < 0);
            return 2;
        }
        else if constexpr (is_vector<T>) {
            const auto& v = as<T>(self.getObj());
            const auto size = static_cast<int>(v.size());

            if constexpr (is_vector<typename T::value_type>) {
                if (i < 0) {
                    return 1;
                }

                CV_Assert(i < size);
                return 2;
            }
            else if constexpr (is_mat<T> || is_umat<T>) {
                if (i < 0) {
                    return 1;
                }

                CV_Assert(i < size);
                return v[static_cast<std::size_t>(i)].dims;
            }
            else {
                CV_Assert(i < 0);
                return 2;
            }
        }
        else if constexpr (is_none<T>) {
            return 0;
        }
        else if constexpr (is_array_of_mat<T>) {
            if (i < 0) {
                return 1;
            }

            const auto& a = as<T>(self.getObj());
            const auto index = static_cast<std::size_t>(i);
            CV_Assert(index < a.size());
            return a[index].dims;
        }
        else if constexpr (is_opengl_buffer<T> || is_cuda_gpu_mat<T> || is_cuda_host_mem<T>) {
            CV_Assert(i < 0);
            return 2;
        }
        else {
            customError(Error::StsNotImplemented, "Unkown/unsupported type array");
        }
    }

    Size size(const _InputArray& self, const int i) const override
    {
        if constexpr (is_mat<T> || is_umat<T>) {
            CV_Assert(i < 0);
            return as<T>(self.getObj()).size();
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            CV_Assert(i < 0);
            return self.getSz();
        }
        else if constexpr (is_vector<T>) {
            const auto& v = as<T>(self.getObj());
            const auto size = static_cast<int>(v.size());

            if constexpr (is_vector<typename T::value_type>) {
                if (i < 0) {
                    return v.empty() ? Size() : Size(size, 1);
                }

                CV_Assert(i < size);
                return Size(static_cast<int>(v[static_cast<std::size_t>(i)].size()), 1);
            }
            else if constexpr (is_vector_of<bool, T>) {
                CV_Assert(i < 0);
                return Size(size, 1);
            }
            else if constexpr (is_vector_of<cuda::GpuMat, T>) {
                if constexpr (have_cuda) {
                    if (i < 0) {
                        return v.empty() ? Size() : Size(size, 1);
                    }

                    CV_Assert(i < size);
                    return v[static_cast<std::size_t>(i)].size();
                }
                else {
                    noCudaError();
                }
            }
            else {
                CV_Assert(i < size);
                if constexpr (is_vector_of_mat<T> || is_vector_of<UMat, T>) {
                    if (i < 0) {
                        return v.empty() ? Size() : Size(size, 1);
                    }

                    CV_Assert(i < size);
                    return v[i].size();
                }
                else {
                    return Size(size, 1);
                }
            }
        }
        else if constexpr (is_none<T>) {
            return Size();
        }
        else if constexpr (is_array_of_mat<T>) {
            const auto& a = as<T>(self.getObj());
            if (i < 0) {
                return a.size() == 0 ? Size() : Size(a.size(), 1);
            }

            const auto index = static_cast<std::size_t>(i);
            CV_Assert(index < a.size());
            return a[index].size();
        }
        else if constexpr (is_opengl_buffer<T> || is_cuda_host_mem<T>) {
            CV_Assert(i < 0);
            return as<T>(self.getObj()).size();
        }
        else {
            unsupportedTypeError();
        }
    }

    int sizend(const _InputArray& self, int* const arraySize, const int i) const override
    {
        if constexpr (is_none<T>) {
            return 0;
        }
        else if constexpr (is_mat<T> || is_umat<T>) {
            CV_Assert(i < 0);
            const auto& mat = as<T>(self.getObj());
            if (arraySize != nullptr) {
                std::copy_n(mat.size.p, mat.dims, arraySize);
            }

            return mat.dims;
        }
        else if constexpr (is_vector_of_mat<T> || is_vector_of<UMat, T> || is_array_of_mat<T>) {
            if (i >= 0) {
                CV_Assert(i < self.getSz().height);
                const auto& mat = as<T>(self.getObj())[i];

                if (arraySize) {
                    std::copy_n(mat.size.p, mat.dims, arraySize);
                }

                return mat.dims;
            }
        }
        // else
        constexpr int dimSize = 2;
        CV_CheckLE(dims(self, i), dimSize, "Not supported");
        Size sz2d = size(self, i);
        if (arraySize != nullptr) {
            arraySize[0] = sz2d.height;
            arraySize[1] = sz2d.width;
        }

        return dimSize;
    }

    bool sameSize(const _InputArray& self, const _InputArray& other) const override
    {
        if constexpr (is_mat<T> || is_umat<T>) {
            auto& x = as<T>(self.getObj());

            if (other.isMat()) {
                auto& y = as<Mat>(other.getObj());
                return x.size == y.size;
            }

            if (other.isUMat()) {
                auto& y = as<UMat>(other.getObj());
                return x.size == y.size;
            }

            return (x.dims <= 2 || other.dims() <= 2) && x.size() == other.size();
        }
        else {
            return other.dims() <= 2 && self.size() == other.size();
        }
    }

    int type(const _InputArray& self, const int i) const override
    {
        const int flags = self.getFlags();
        if constexpr (is_mat<T> || is_umat<T>) {
            return as<T>(self.getObj()).type();
        }
        else if constexpr (is_vector_of_mat<T> || is_vector_of<UMat, T> || is_vector_of<cuda::GpuMat, T> || is_array_of_mat<T>) {
            if constexpr (is_vector_of<cuda::GpuMat, T> && !have_cuda) {
                noCudaError();
            }
            else {
                const auto& mats = as<T>(self.getObj());
                if (mats.empty()) {
                    CV_Assert((flags & _InputArray::FIXED_TYPE) != 0);
                    return CV_MAT_TYPE(flags);
                }

                const auto isize = static_cast<std::ptrdiff_t>(mats.size());
                CV_Assert(i < isize);
                return mats[std::clamp(0, i, i)].type();
            }
        }
        else if constexpr (is_matx<T> || is_vector<T> || std::is_scalar_v<T>) {
            return CV_MAT_TYPE(flags);
        }
        else if constexpr (is_opengl_buffer<T> || is_cuda_gpu_mat<T> || is_cuda_host_mem<T>) {
            return as<T>(self.getObj()).type();
        }
        else {
            unsupportedTypeError();
        }
    }

    int isContinuous(const _InputArray& self, const int i) const override
    {
        if constexpr (is_mat<T> || is_umat<T> || is_cuda_gpu_mat<T>) {
            return i < 0 ? as<T>(self.getObj()).isContinuous() : true;
        }
        else if constexpr (is_vector_of_mat<T> || is_vector_of<UMat, T> || is_array_of_mat<T>) {
            CV_Assert(i >= 0);
            const auto& v = as<T>(self.getObj());
            const auto index = static_cast<std::size_t>(i);
            CV_Assert(index < v.size());
            return v[index].isContinuous();
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T> || (is_vector<T> && !is_vector_of<cuda::GpuMat, T>) || is_none<T>) {
            return true;
        }
        else {
            unsupportedTypeError();
        }
    }

    int isSubmatrix(const _InputArray& self, const int i) const override
    {
        if constexpr (is_mat<T> || is_umat<T>) {
            return i < 0 ? as<T>(self.getObj()).isSubmatrix() : false;
        }
        else if constexpr (is_vector_of_mat<T> || is_vector_of<UMat, T> || is_array_of_mat<T>) {
            CV_Assert(i >= 0);
            const auto& mats = as<T>(self.getObj());

            const auto index = static_cast<std::size_t>(i);
            CV_Assert(index < mats.size());
            return mats[index].isSubmatrix();
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T> || (is_vector<T> && !is_vector_of<cuda::GpuMat, T>) || is_none<T>) {
            return false;
        }
        else {
            unsupportedTypeError();
        }
    }

    int empty(const _InputArray& self) const override
    {
        constexpr bool has_empty = is_mat<T> || is_umat<T> || is_vector<T> || is_array_of_mat<T> || is_opengl_buffer<T> || is_cuda_gpu_mat<T> || is_cuda_host_mem<T>;
        if constexpr (has_empty) {
            return as<T>(self.getObj()).empty();
        }
        else if constexpr (is_none<T>) {
            return true;
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            return false;
        }
        else {
            unsupportedTypeError();
        }
    }

    void copyTo(const _InputArray& self, const _OutputArray& arr) const override
    {
        if constexpr (is_none<T>) {
            arr.release();
        }
        else if constexpr (is_mat<T> || is_matx<T> || std::is_scalar_v<T> || is_vector<T>) {
            Mat m = getMat_(self, -1);
            m.copyTo(arr);
        }
        else if constexpr (is_umat<T> || (is_cuda_gpu_mat<T> && have_cuda)) {
            as<T>(self.getObj()).copyTo(arr);
        }
        else {
            unsupportedTypeError();
        }
    }

    void copyTo(const _InputArray& self, const _OutputArray& arr, const _InputArray& mask) const override
    {
        if constexpr (is_none<T>) {
            arr.release();
        }
        else if constexpr (is_mat<T> || is_matx<T> || std::is_scalar_v<T> || is_vector<T>) {
            getMat_(self, -1).copyTo(arr, mask);
        }
        else if constexpr (is_umat<T> || is_cuda_gpu_mat<T>) {
            if constexpr (is_cuda_gpu_mat<T> && !have_cuda) {
                noCudaError();
            }
            else {
                as<T>(self.getObj()).copyTo(arr, mask);
            }
        }
        else {
            unsupportedTypeError();
        }
    }

    std::size_t offset(const _InputArray& self, const std::size_t i) const override
    {
        if constexpr (is_mat<T>) {
            CV_Assert(i < 0);
            const auto& mat = as<T>(self.getObj());
            return static_cast<std::size_t>(mat.ptr() - mat.datastart);
        }
        else if constexpr (is_umat<T>) {
            CV_Assert(i < 0);
            return as<T>(self.getObj()).offset;
        }
        else if constexpr (is_cuda_gpu_mat<T>) {
            CV_Assert(i < 0);
            const auto& mat = as<T>(self.getObj());
            return static_cast<std::size_t>(mat.data - mat.datastart);
        }
        else if constexpr (is_vector_of_mat<T> || is_array_of_mat<T>) {
            const auto& mats = as<T>(self.getObj());
            CV_Assert(i >= 0);
            CV_Assert(i < mats.size());
            return static_cast<std::size_t>(mats[i].ptr() - mats[i].datastart);
        }
        else if constexpr (is_vector_of<UMat, T>) {
            const auto& mats = as<T>(self.getObj());
            CV_Assert(i >= 0);
            CV_Assert(i < mats.size());
            return mats[i].offset;
        }
        else if constexpr (is_vector_of<cuda::GpuMat, T>) {
            const auto& mats = as<T>(self.getObj());
            CV_Assert(i >= 0);
            CV_Assert(i < mats.size());
            return static_cast<std::size_t>(mats[i].data - mats[i].datastart);
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T> || is_vector<T> || is_none<T>) {
            return 0;
        }
        else {
            unsupportedTypeError();
        }
    }

    std::size_t step(const _InputArray& self, const std::size_t i) const override
    {
        if constexpr (is_mat<T> || is_umat<T> || is_cuda_gpu_mat<T>) {
            CV_Assert(i < 0);
            return as<T>(self.getObj()).step;
        }
        else if constexpr (is_vector_of_mat<T> || is_vector_of<UMat, T> || is_vector_of<cuda::GpuMat, T> || is_array_of_mat<T>) {
            const auto& mats = as<T>(self.getObj());
            CV_Assert(i >= 0);
            CV_Assert(i < mats.size());
            return mats[i].step;
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T> || is_vector<T> || is_none<T>) {
            return 0;
        }
        else {
            unsupportedTypeError();
        }
    }

    bool isMat() const override
    {
        return is_mat<T>;
    }

    bool isUMat() const override
    {
        return is_umat<T>;
    }

    bool isMatVector() const override
    {
        return is_vector_of_mat<T>;
    }

    bool isUMatVector() const override
    {
        return is_vector_of<UMat, T>;
    }

    bool isMatx() const override
    {
        // Scalar values have historically been treated as a Matx with Size(1, 1)
        return is_matx<T> || std::is_scalar_v<T>;
    }

    bool isVector() const override
    {
        return is_vector<T>;
    }

    bool isGpuMat() const override
    {
        return is_cuda_gpu_mat<T>;
    }

    bool isGpuMatVector() const override
    {
        return is_vector_of<cuda::GpuMat, T>;
    }

    void create(const _OutputArray& arr,
                const Size sz,
                const int mtype,
                const int i,
                const bool allowTransposed,
                const _OutputArray::DepthMask fixedDepthMask,
                const bool fixedSize,
                const bool fixedType) const override
    {
        if (i < 0 && !allowTransposed && fixedDepthMask == 0) {
            if constexpr (is_mat<T> || is_umat<T> || is_cuda_gpu_mat<T> || is_cuda_host_mem<T> || is_opengl_buffer<T>) {
                auto& mat = as<T>(arr.getObj());
                CV_Assert(!fixedSize || mat.size() == sz);
                CV_Assert(!fixedType || mat.type() == mtype);

                if constexpr ((is_cuda_gpu_mat<T> || is_cuda_host_mem<T>) && !have_cuda) {
                    noCudaError();
                }
                else if constexpr (is_opengl_buffer<T> && !have_opengl) {
                    noOpenGLError();
                }
                else {
                    mat.create(sz, mtype);
                }
            }
            else {
                goto fallback;
            }
        }
        else {
        fallback:
            const std::array<int, 2> sizes{sz.height, sz.width};
            create(arr, 2, sizes.data(), mtype, i, allowTransposed, fixedDepthMask, fixedSize, fixedType);
        }
    }

    void create(const _OutputArray& arr,
                const int rows,
                const int cols,
                const int mtype,
                const int i,
                const bool allowTransposed,
                const _OutputArray::DepthMask fixedDepthMask,
                const bool fixedSize,
                const bool fixedType) const override
    {
        if (i < 0 && !allowTransposed && fixedDepthMask == 0) {
            if constexpr (is_mat<T> || is_umat<T> || is_cuda_gpu_mat<T> || is_cuda_host_mem<T> || is_opengl_buffer<T>) {
                if constexpr ((is_cuda_gpu_mat<T> || is_cuda_host_mem<T>) && !have_cuda) {
                    noCudaError();
                }
                else if constexpr (is_opengl_buffer<T> && !have_opengl) {
                    noOpenGLError();
                }
                else {
                    auto& mat = as<T>(arr.getObj());
                    CV_Assert(!fixedSize || (mat.size)() == Size(cols, rows));
                    CV_Assert(!fixedType || mat.type() == mtype);
                    mat.create(rows, cols, mtype);
                }
            }
            else {
                goto fallback;
            }
        }
        else {
        fallback:
            const std::array<int, 2> sizes{rows, cols};
            create(arr, 2, sizes.data(), mtype, i, allowTransposed, fixedDepthMask, fixedSize, fixedType);
        }
    }

    void create(const _OutputArray& arr,
                int d,
                const int* sizes,
                int mtype,
                const int i,
                const bool allowTransposed,
                const _OutputArray::DepthMask fixedDepthMask,
                const bool fixedSize,
                const bool fixedType) const override
    {
        const std::array<int, 2> sizebuf{sizes[0], 1};
        if (d == 1) {
            d = 2;
            sizes = sizebuf.data();
        }

        const int flags = arr.getFlags();
        const Size sz = arr.getSz();
        const auto index = static_cast<std::size_t>(i);
        if constexpr (is_mat<T> || is_umat<T>) {
            CV_Assert(i < 0);
            auto& mat = as<T>(arr.getObj());
            CV_Assert(!(mat.empty() && fixedType && fixedSize) && "Can't reallocate empty matrix with locked layout (probably due to misused 'const' modifier)");

            const bool both2d = d == 2 && mat.dims == 2;
            const bool same_type = mat.type() == mtype;
            const bool same_cols = mat.cols == sizes[0];
            const bool same_rows = mat.rows == sizes[1];
            if (allowTransposed && !mat.empty() && both2d && same_type && same_cols && same_rows && mat.isContinuous()) {
                return;
            }

            if (fixedType) {
                if (CV_MAT_CN(mtype) == mat.channels() && (1 << CV_MAT_TYPE(flags) & fixedDepthMask) != 0) {
                    mtype = mat.type();
                }
                else {
                    CV_CheckTypeEQ(mat.type(), CV_MAT_TYPE(mtype), "Can't reallocate matrix with locked size (probably due to misused 'const' modifier)");
                }
            }

            if (fixedSize) {
                CV_CheckTypeEQ(mat.dims, d, "Can't reallocate matrix with locked size (probably due to misused 'const' modifier)");
                for (int j = 0; j < d; ++j) {
                    CV_CheckEQ(mat.size[j], sizes[j], "Can't reallocate matrix with locked size (probably due to misused 'const' modifier)");
                }
            }

            mat.create(d, sizes, mtype);
            return;
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            CV_Assert(i < 0);
            const int type0 = CV_MAT_TYPE(flags);

            // TODO: rename to a meaningful name (what does this check involve?)
            const bool something = CV_MAT_CN(mtype) == 1 && ((1 << type0) & fixedDepthMask) != 0;
            CV_Assert(mtype == type0 || something);
            CV_CheckLE(d, 2, ""); // TODO: a good diagnostic should go here

            const Size requested_size{d == 2 ? sizes[1] : 1, d >= 1 ? sizes[0] : 1};
            if (sz.width == 1 || sz.height == 1) {
                // CV_Assert(allowTransposed && "1D arrays assume allowTransposed=true (see #4159)");
                const int actual_total = std::max(sz.width, sz.height);
                const int expected_total = std::max(requested_size.width, requested_size.height);

                // TODO: a good diagnostic should go here
                CV_Check(requested_size, expected_total == actual_total, "");
            }
            else if (allowTransposed) {
                const int is_transposed = requested_size.height == sz.width && requested_size.width == sz.height;
                // TODO: a good diagnostic should go here
                CV_Check(requested_size, requested_size == sz || is_transposed, "");
            }
            else {
                // TODO: a good diagnostic should go here
                CV_CheckEQ(requested_size, sz, "");
            }
        }
        else if constexpr (is_vector<T> || is_array_of_mat<T>) {
            using value_type = typename T::value_type;
            if constexpr (is_mat<value_type> || is_umat<value_type>) {
                auto& mats = as<T>(arr.getObj());
                if (i < 0) {
                    CV_Assert(d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0] * sizes[1] == 0));
                    const std::size_t len = sizes[0] * sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0;
                    const std::size_t len0 = mats.size();

                    if constexpr (is_vector<T>) {
                        CV_Assert(!fixedSize || len == len0);
                        mats.resize(len);
                    }
                    else {
                        CV_Assert(len == len0);
                    }

                    if (fixedType) {
                        int _type = CV_MAT_TYPE(flags);
                        for (std::size_t j = len0; j < len; ++j) {
                            if (mats[j].type() == _type) {
                                continue;
                            }

                            CV_Assert(mats[j].empty());
                            mats[j].flags = (mats[j].flags & ~CV_MAT_TYPE_MASK) | _type;
                        }
                    }

                    return;
                }

                CV_Assert(index < mats.size());
                auto& mat = mats[index];

                if (allowTransposed) {
                    if (!mat.isContinuous()) {
                        CV_Assert(!fixedType && !fixedSize);
                        mat.release();
                    }

                    const bool both_2d = d == 2 && mat.dims == 2;
                    const bool same_type = mat.type() == mtype;
                    const bool same_rows = mat.rows == sizes[1];
                    const bool same_cols = mat.cols == sizes[0];
                    if (both_2d && matrix_data(mat) != nullptr && same_type && same_rows && same_cols) {
                        return;
                    }
                }

                if (fixedType) {
                    const bool same_channels = CV_MAT_CN(mtype) == mat.channels();
                    // TODO: come up with a better name for this
                    const bool something = ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0;
                    if (same_channels && something) {
                        mtype = mat.type();
                    }
                    else {
                        CV_Assert(CV_MAT_TYPE(mtype) == mat.type());
                    }
                }

                if (fixedSize) {
                    CV_Assert(mat.dims == d);

                    // TODO: check this isn't going to result in a buffer overrun
                    for (int j = 0; j < d; ++j) {
                        CV_Assert(mat.size[j] == sizes[j]);
                    }
                }

                if constexpr (is_mat_<value_type>) {
                    mat.Mat::create(d, sizes, mtype);
                }
                else {
                    mat.create(d, sizes, mtype);
                }
            }
            else {
                CV_Assert(d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0] * sizes[1] == 0));
                const std::size_t len = sizes[0] * sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0;
                const int type0 = CV_MAT_TYPE(flags);
                const bool same_type = mtype == type0;
                const bool same_column = CV_MAT_CN(mtype) == CV_MAT_CN(type0);
                const bool something = ((1 << type0) & fixedDepthMask) != 0;
                CV_Assert(same_type || (same_column && something));

                // constexpr auto valid_sizes = std::array{1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 128, 256, 512};
                auto& v = as<T>(arr.getObj());

                if constexpr (!is_vector<value_type>) {
                    CV_Assert(i < 0);
                    CV_Assert(!fixedSize || len == sizeof(value_type));
                    // if (!std::binary_search(valid_sizes.begin(), valid_sizes.end(), sizeof(value_type))) {
                    //     createError(sizeof(value_type));
                    // }
                    v.resize(len);
                }
                else {
                    if (i < 0) {
                        CV_Assert(!fixedSize || len == v.size());
                        v.resize(len);
                        return;
                    }

                    CV_Assert(index < v.size());
                    CV_Assert(!fixedSize || len == sizeof(typename value_type::value_type));
                    // if (std::binary_search(valid_sizes.begin(), valid_sizes.end(), sizeof(typename value_type::value_type))) {
                    //     createError(sizeof(typename value_type::value_type));
                    // }
                    v[index].resize(len);
                }
            }
        }
        else if constexpr (is_none<T>) {
            customError(Error::StsNullPtr, "create() called for the missing output array");
        }
        else {
            unsupportedTypeError();
        }
    }

    void release(const _OutputArray& self, const bool fixedSize) const override
    {
        CV_Assert(!fixedSize);
        if constexpr (is_mat<T> || is_umat<T> || is_cuda_type<T> || is_opengl_buffer<T>) {
            if constexpr (is_cuda_type<T> && !have_cuda) {
                noCudaError();
            }
            else if constexpr (is_opengl_buffer<T> && !have_opengl) {
                noOpenGLError();
            }
            else {
                as<T>(self.getObj()).release();
            }
        }
        else if constexpr (is_vector<T>) {
            if constexpr (is_cuda_type<T> && !have_cuda) {
                noCudaError();
            }
            else {
                as<T>(self.getObj()).clear();
            }
        }
        else {
            unsupportedTypeError();
        }
    }

    void clear(const _OutputArray& self, const bool fixedSize) const override
    {
        if constexpr (is_mat<T>) {
            CV_Assert(!fixedSize);
            as<T>(self.getObj()).resize(0);
        }
        else {
            release(self, fixedSize);
        }
    }

    Mat& getMatRef(const _OutputArray& self, const int i) const override
    {
        if constexpr (is_mat<T>) {
            CV_Assert(i < 0);
            return as<T>(self.getObj());
        }
        else if constexpr (is_vector_of_mat<T> || is_array_of_mat<T>) {
            auto& mats = as<T>(self.getObj());
            CV_Assert(i >= 0);

            const auto index = static_cast<std::size_t>(i);
            CV_Assert(index < mats.size());
            return mats[index];
        }
        else {
            unsupportedTypeError();
        }
    }

    UMat& getUMatRef(const _OutputArray& self, const int i) const override
    {
        if constexpr (is_umat<T>) {
            CV_Assert(i < 0);
            return as<T>(self.getObj());
        }
        else if constexpr (is_vector_of<UMat, T>) {
            auto& mats = as<T>(self.getObj());
            CV_Assert(i >= 0);
            const auto index = static_cast<std::size_t>(i);
            CV_Assert(index < mats.size());
            return mats[index];
        }
        else {
            unsupportedTypeError();
        }
    }

    ogl::Buffer& getOGlBufferRef(const _OutputArray& self) const override
    {
        if constexpr (is_opengl_buffer<T>) {
            return as<T>(self.getObj());
        }
        else {
            unsupportedTypeError();
        }
    }

    cuda::HostMem& getHostMemRef(const _OutputArray& self) const override
    {
        if constexpr (is_cuda_host_mem<T>) {
            return as<T>(self.getObj());
        }
        else {
            unsupportedTypeError();
        }
    }

    cuda::GpuMat& getGpuMatRef(const _OutputArray& self) const override
    {
        if constexpr (is_cuda_gpu_mat<T>) {
            return as<T>(self.getObj());
        }
        else {
            unsupportedTypeError();
        }
    }

    std::vector<cuda::GpuMat>& getGpuMatVecRef(const _OutputArray& self) const override
    {
        if constexpr (is_vector_of<cuda::GpuMat, T>) {
            return as<T>(self.getObj());
        }
        else {
            unsupportedTypeError();
        }
    }

    void setTo(const _OutputArray& self, const _InputArray& arr, const _InputArray& mask) const override
    {
        if constexpr (is_none<T>) {
            return;
        }
        else if constexpr (is_mat<T> || is_matx<T> || std::is_scalar_v<T> || is_vector<T>) {
            getMat_(self, -1).setTo(arr, mask);
        }
        else if constexpr (is_umat<T>) {
            as<T>(self.getObj()).setTo(arr, mask);
        }
        else if (is_cuda_gpu_mat<T>) {
            if constexpr (!have_cuda) {
                noCudaError();
            }
            else {
                const Mat value = arr.getMat();
                // CV_Assert(checkScalar(value, type(), arr.kind(+, _InputArray::CUDA_GPU_MAT)));
                as<T>(self.getObj()).setTo(Scalar(Vec<double, 4>(value.ptr<double>())), mask);
            }
        }
        else {
            unsupportedTypeError();
        }
    }

    void assign(const _OutputArray& self, const UMat& u) const override
    {
        if constexpr (is_umat<T>) {
            as<T>(self.getObj()) = u;
        }
        else if constexpr (is_mat<T>) {
            u.copyTo(as<T>(self.getObj())); // TODO: check u.getMat()
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            u.copyTo(getMat_(self, -1)); // TODO: check u.getMat()
        }
        else {
            unsupportedTypeError();
        }
    }

    void assign(const _OutputArray& self, const Mat& m) const override
    {
        if constexpr (is_mat<T>) {
            as<T>(self.getObj()) = m;
        }
        else if constexpr (is_umat<T>) {
            m.copyTo(as<T>(self.getObj())); // TODO: check m.getMat()
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            m.copyTo(getMat_(self, -1)); // TODO: check m.getMat()
        }
        else {
            unsupportedTypeError();
        }
    }

    template<class U>
    static void assign(const _OutputArray& self, const std::vector<U>& other)
    {
        if constexpr (is_vector_of_mat<T> || is_vector_of<UMat, T>) {
            auto& mats = as<T>(self.getObj());
            CV_Assert(mats.size() == other.size());

            for (std::size_t i = 0; i < mats.size(); ++i) {
                auto& mat = mats[i];
                const auto& other_mat = other[i];
                if (mat.u != nullptr && mat.u == other_mat.u) {
                    continue;
                }

                other_mat.copyTo(mat);
            }
        }
        else {
            unsupportedTypeError();
        }
    }

    void assign(const _OutputArray& self, const std::vector<Mat>& other) const override
    {
        _InputArrayOps::assign<>(self, other);
    }

    void assign(const _OutputArray& self, const std::vector<UMat>& other) const override
    {
        _InputArrayOps::assign<>(self, other);
    }

    void move(const _OutputArray& self, UMat& u, const bool fixed_size) const override
    {
        if (fixed_size) {
            // TODO: performance warning
            assign(self, u);
            return;
        }

        if constexpr (is_umat<T>) {
            as<T>(self.getObj()) = std::move(u);
        }
        else if constexpr (is_mat<T>) {
            u.copyTo(as<T>(self.getObj())); // TODO: check u.getMat()
            u.release();
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            u.copyTo(getMat_(self, -1)); // TODO: check u.getMat()
            u.release();
        }
        else {
            unsupportedTypeError();
        }
    }

    void move(const _OutputArray& self, Mat& m, const bool fixed_size) const override
    {
        if (fixed_size) {
            // TODO: performance warning
            assign(self, m);
            return;
        }

        if constexpr (is_mat<T>) {
            as<T>(self.getObj()) = std::move(m);
        }
        else if constexpr (is_umat<T>) {
            m.copyTo(as<T>(self.getObj())); // TODO: check u.getMat()
            m.release();
        }
        else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
            m.copyTo(getMat_(self, -1)); // TODO: check u.getMat()
            m.release();
        }
        else {
            unsupportedTypeError();
        }
    }
};

inline constexpr _InputArrayOps<std::nullptr_t> _InputArrayOps_none;
inline constexpr _InputArrayOps<UMat> _InputArrayOps_umat;
inline constexpr _InputArrayOps<ogl::Buffer> _InputArrayOps_opengl_buffer;
inline constexpr _InputArrayOps<cuda::HostMem> _InputArrayOps_cuda_host_mem;
inline constexpr _InputArrayOps<cuda::GpuMat> _InputArrayOps_cuda_gpu_mat;

inline constexpr _InputArrayOps<Mat> _InputArrayOps_mat;

template<class T>
inline constexpr _InputArrayOps<T> _InputArrayOps_matx;

template <class T>
inline constexpr _InputArrayOps<T> _InputArrayOps_vector;

template <class T>
inline constexpr _InputArrayOps<T> _InputArrayOps_std_array_mat;

template <class T>
inline constexpr _InputArrayOps<T> _InputArrayOps_scalar;

template<class T>
Mat _InputArrayOps<T>::getMat_(const _InputArray& self, const int i) const
{
    void* const p = self.getObj();
    const int flags = self.getFlags();
    const Size sz = self.getSz();
    const AccessFlag access = flags & ACCESS_MASK;

    if constexpr (is_mat<T>) {
        return i < 0 ? as<T>(p) : as<T>(p).row(i);
    }
    else if constexpr (is_umat<T>) {
        return i < 0 ? as<T>(p).getMat(access) : as<T>(p).getMat(access).row(i);
    }
    else if constexpr (is_none<T>) {
        return Mat();
    }
    else if constexpr (is_matx<T> || std::is_scalar_v<T>) {
        CV_Assert(i < 0);
        return Mat(sz, flags, p);
    }
    else if constexpr (is_vector<T> || is_array_of_mat<T>) {
        auto& v = as<T>(p);
        const auto index = static_cast<std::size_t>(i);

        if constexpr (is_vector<typename T::value_type>) {
            int t = type(self, i);
            CV_Assert(0 <= i);
            CV_Assert(static_cast<std::size_t>(i) < v.size());
            return !v.empty() ? Mat(size(self, i), t, v.data()) : Mat();
        }
        else if constexpr (is_vector_of<bool, T>) {
            CV_Assert(i < 0);
            constexpr int type = CV_8U;
            if (v.empty()) {
                return {};
            }

            Mat m(1, v.size(), type);
            for (std::size_t j = 0; j < v.size(); ++j) {
                m.data[j] = v[j];
            }

            return m;
        }
        else if constexpr (is_vector_of_mat<T> || is_array_of_mat<T>) {
            CV_Assert(0 <= index);
            CV_Assert(index < v.size());
            return v[index];
        }
        else if constexpr (is_vector_of<UMat, T>) {
            CV_Assert(0 <= index);
            CV_Assert(index < v.size());
            return v[index].getMat(access);
        }
        else {
            CV_Assert(i < 0);
            const int type = CV_MAT_TYPE(flags);
            return v.empty() ? Mat() : Mat(size(self, i), type, v.data());
        }
    }
    else if constexpr (is_opengl_buffer<T>) {
        CV_Assert(i < 0);
        customError(cv::Error::StsNotImplemented,
                "You should call 'ogl::Buffer::mapHost' and 'ogl::Buffer::unmapHost' for 'ogl::Buffer' objects");
    }
    else if constexpr (is_cuda_gpu_mat<T>) {
        CV_Assert(i < 0);
        customError(
            cv::Error::StsNotImplemented,
            "You should explicitly call 'cuda::GpuMat::download' for 'cuda::GpuMat' objects");
    }
    else if constexpr (is_cuda_host_mem<T>) {
        CV_Assert(i < 0);
        const auto& cuda_mem = as<T>(p);
        return cuda_mem.createMatHeader();
    }
    else {
        unsupportedTypeError();
    }
}

inline _InputArray::_InputArray()
: flags(NONE)
, obj(nullptr)
, ops(&_InputArrayOps_none)
{}

template<typename T>
inline _InputArray::_InputArray(int _flags, T* _obj, Size sz)
: flags(_flags)
, obj(_obj)
, sz(sz)
{
    if constexpr (is_mat<T>) {
        ops = &_InputArrayOps_mat;
    }
    else if constexpr (is_umat<T>) {
        ops = &_InputArrayOps_umat;
    }
    else if constexpr (is_opengl_buffer<T>) {
        ops = &_InputArrayOps_opengl_buffer;
    }
    else if constexpr (is_cuda_host_mem<T>) {
        ops = &_InputArrayOps_cuda_host_mem;
    }
    else if constexpr (is_cuda_gpu_mat<T>) {
        ops = &_InputArrayOps_cuda_gpu_mat;
    }
    else if constexpr (is_matx<T>) {
        ops = &_InputArrayOps_matx<T>;
    }
    else if constexpr (is_vector<T>) {
        ops = &_InputArrayOps_vector<T>;
    }
    else if constexpr (is_array_of_mat<T>) {
        ops = &_InputArrayOps_std_array_mat<T>;
    }
    else if constexpr (std::is_scalar_v<T>) {
        ops = &_InputArrayOps_scalar<T>;
    }
}

}

#pragma GCC diagnostic pop

#endif // OPENCV_CORE_ARRAY_HELPERS_HPP
