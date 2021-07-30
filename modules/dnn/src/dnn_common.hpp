// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_COMMON_HPP__
#define __OPENCV_DNN_COMMON_HPP__

#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
#define IS_DNN_OPENCL_TARGET(id) (id == DNN_TARGET_OPENCL || id == DNN_TARGET_OPENCL_FP16)
Mutex& getInitializationMutex();
void initializeLayerFactory();

namespace detail {
#define CALL_MEMBER_FN(object, ptrToMemFn)  ((object).*(ptrToMemFn))

class NotImplemented : public Layer
{
public:
    static Ptr<Layer> create(const LayerParams &params);

    static void Register();
    static void unRegister();
};

struct NetImplBase
{
    const int networkId;  // network global identifier
    int networkDumpCounter;  // dump counter
    int dumpLevel;  // level of information dumps (initialized through OPENCV_DNN_NETWORK_DUMP parameter)

    NetImplBase();

    std::string getDumpFileNameBase();
};

}  // namespace detail

CV__DNN_INLINE_NS_END
}}  // namespace

#endif  // __OPENCV_DNN_COMMON_HPP__
