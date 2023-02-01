// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_COMMON_HPP__
#define __OPENCV_DNN_COMMON_HPP__

#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN
#define IS_DNN_OPENCL_TARGET(id) (id == DNN_TARGET_OPENCL || id == DNN_TARGET_OPENCL_FP16)
Mutex& getInitializationMutex();
void initializeLayerFactory();

namespace detail {
#define CALL_MEMBER_FN(object, ptrToMemFn)  ((object).*(ptrToMemFn))

struct NetImplBase
{
    const int networkId;  // network global identifier
    mutable int networkDumpCounter;  // dump counter
    int dumpLevel;  // level of information dumps (initialized through OPENCV_DNN_NETWORK_DUMP parameter)

    NetImplBase();

    std::string getDumpFileNameBase() const;
};

}  // namespace detail


typedef std::vector<MatShape> ShapesVec;

static inline std::string toString(const ShapesVec& shapes, const std::string& name = std::string())
{
    std::ostringstream ss;
    if (!name.empty())
        ss << name << ' ';
    ss << '[';
    for(size_t i = 0, n = shapes.size(); i < n; ++i)
        ss << ' ' << toString(shapes[i]);
    ss << " ]";
    return ss.str();
}

static inline std::string toString(const Mat& blob, const std::string& name = std::string())
{
    std::ostringstream ss;
    if (!name.empty())
        ss << name << ' ';
    if (blob.empty())
    {
        ss << "<empty>";
    }
    else if (blob.dims == 1)
    {
        Mat blob_ = blob;
        blob_.dims = 2;  // hack
        ss << blob_.t();
    }
    else
    {
        ss << blob.reshape(1, 1);
    }
    return ss.str();
}

CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace

#endif  // __OPENCV_DNN_COMMON_HPP__
