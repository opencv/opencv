// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudev.hpp"

using namespace cv;
using namespace cv::cuda;

GpuData::GpuData(const size_t _size)
    : data(nullptr), size(_size)
{
    CV_CUDEV_SAFE_CALL(cudaMalloc(&data, _size));
}

GpuData::~GpuData()
{
    CV_CUDEV_SAFE_CALL(cudaFree(data));
}

/////////////////////////////////////////////////////
/// create

void GpuMatND::create(SizeArray _size, int _type)
{
    {
        auto elements_nonzero = [](SizeArray& v)
        {
            return std::all_of(v.begin(), v.end(),
                [](unsigned u){ return u > 0; });
        };
        CV_Assert(!_size.empty());
        CV_Assert(elements_nonzero(_size));
    }

    _type &= Mat::TYPE_MASK;

    if (size == _size && type() == _type && !empty() && !external() && isContinuous() && !isSubmatrix())
        return;

    release();

    setFields(std::move(_size), _type);

    data_ = std::make_shared<GpuData>(totalMemSize());
    data = data_->data;
    offset = 0;
}

/////////////////////////////////////////////////////
/// release

void GpuMatND::release()
{
    data = nullptr;
    data_.reset();

    flags = dims = offset = 0;
    size.clear();
    step.clear();
}

/////////////////////////////////////////////////////
/// clone

static bool next(uchar*& d, const uchar*& s, std::vector<int>& idx, const int dims, const GpuMatND& dst, const GpuMatND& src)
{
    int inc = dims-3;

    while (true)
    {
        if (idx[inc] == src.size[inc] - 1)
        {
            if (inc == 0)
            {
                return false;
            }

            idx[inc] = 0;
            d -= (dst.size[inc] - 1) * dst.step[inc];
            s -= (src.size[inc] - 1) * src.step[inc];
            inc--;
        }
        else
        {
            idx[inc]++;
            d += dst.step[inc];
            s += src.step[inc];
            break;
        }
    }

    return true;
}

GpuMatND GpuMatND::clone() const
{
    CV_DbgAssert(!empty());

    GpuMatND ret(size, type());

    if (isContinuous())
    {
        CV_CUDEV_SAFE_CALL(cudaMemcpy(ret.getDevicePtr(), getDevicePtr(), ret.totalMemSize(), cudaMemcpyDeviceToDevice));
    }
    else
    {
        // 1D arrays are always continuous

        if (dims == 2)
        {
            CV_CUDEV_SAFE_CALL(
                cudaMemcpy2D(ret.getDevicePtr(), ret.step[0], getDevicePtr(), step[0],
                    size[1]*step[1], size[0], cudaMemcpyDeviceToDevice)
            );
        }
        else
        {
            std::vector<int> idx(dims-2, 0);

            uchar* d = ret.getDevicePtr();
            const uchar* s = getDevicePtr();

            // iterate each 2D plane
            do
            {
                CV_CUDEV_SAFE_CALL(
                    cudaMemcpy2DAsync(
                        d, ret.step[dims-2], s, step[dims-2],
                        size[dims-1]*step[dims-1], size[dims-2], cudaMemcpyDeviceToDevice)
                );
            }
            while (next(d, s, idx, dims, ret, *this));

            CV_CUDEV_SAFE_CALL(cudaStreamSynchronize(0));
        }
    }

    return ret;
}

GpuMatND GpuMatND::clone(Stream& stream) const
{
    CV_DbgAssert(!empty());

    GpuMatND ret(size, type());

    cudaStream_t _stream = StreamAccessor::getStream(stream);

    if (isContinuous())
    {
        CV_CUDEV_SAFE_CALL(cudaMemcpyAsync(ret.getDevicePtr(), getDevicePtr(), ret.totalMemSize(), cudaMemcpyDeviceToDevice, _stream));
    }
    else
    {
        // 1D arrays are always continuous

        if (dims == 2)
        {
            CV_CUDEV_SAFE_CALL(
                cudaMemcpy2DAsync(ret.getDevicePtr(), ret.step[0], getDevicePtr(), step[0],
                    size[1]*step[1], size[0], cudaMemcpyDeviceToDevice, _stream)
            );
        }
        else
        {
            std::vector<int> idx(dims-2, 0);

            uchar* d = ret.getDevicePtr();
            const uchar* s = getDevicePtr();

            // iterate each 2D plane
            do
            {
                CV_CUDEV_SAFE_CALL(
                    cudaMemcpy2DAsync(
                        d, ret.step[dims-2], s, step[dims-2],
                        size[dims-1]*step[dims-1], size[dims-2], cudaMemcpyDeviceToDevice, _stream)
                );
            }
            while (next(d, s, idx, dims, ret, *this));
        }
    }

    return ret;
}

/////////////////////////////////////////////////////
/// upload

void GpuMatND::upload(InputArray src)
{
    Mat mat = src.getMat();

    CV_DbgAssert(!mat.empty());

    if (!mat.isContinuous())
        mat = mat.clone();

    SizeArray _size(mat.dims);
    std::copy_n(mat.size.p, mat.dims, _size.data());

    create(std::move(_size), mat.type());

    CV_CUDEV_SAFE_CALL(cudaMemcpy(getDevicePtr(), mat.data, totalMemSize(), cudaMemcpyHostToDevice));
}

void GpuMatND::upload(InputArray src, Stream& stream)
{
    Mat mat = src.getMat();

    CV_DbgAssert(!mat.empty());

    if (!mat.isContinuous())
        mat = mat.clone();

    SizeArray _size(mat.dims);
    std::copy_n(mat.size.p, mat.dims, _size.data());

    create(std::move(_size), mat.type());

    cudaStream_t _stream = StreamAccessor::getStream(stream);
    CV_CUDEV_SAFE_CALL(cudaMemcpyAsync(getDevicePtr(), mat.data, totalMemSize(), cudaMemcpyHostToDevice, _stream));
}

/////////////////////////////////////////////////////
/// download

void GpuMatND::download(OutputArray dst) const
{
    CV_DbgAssert(!empty());

    dst.create(dims, size.data(), type());
    Mat mat = dst.getMat();

    GpuMatND gmat = *this;

    if (!gmat.isContinuous())
        gmat = gmat.clone();

    CV_CUDEV_SAFE_CALL(cudaMemcpy(mat.data, gmat.getDevicePtr(), mat.total() * mat.elemSize(), cudaMemcpyDeviceToHost));
}

void GpuMatND::download(OutputArray dst, Stream& stream) const
{
    CV_DbgAssert(!empty());

    dst.create(dims, size.data(), type());
    Mat mat = dst.getMat();

    GpuMatND gmat = *this;

    if (!gmat.isContinuous())
        gmat = gmat.clone(stream);

    cudaStream_t _stream = StreamAccessor::getStream(stream);
    CV_CUDEV_SAFE_CALL(cudaMemcpyAsync(mat.data, gmat.getDevicePtr(), mat.total() * mat.elemSize(), cudaMemcpyDeviceToHost, _stream));
}

#endif
