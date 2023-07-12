// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_COMMON_HPP__
#define __OPENCV_DNN_COMMON_HPP__

#include <unordered_map>
#include <unordered_set>

#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
#define IS_DNN_OPENCL_TARGET(id) (id == DNN_TARGET_OPENCL || id == DNN_TARGET_OPENCL_FP16)
#define IS_DNN_CPU_TARGET(id) (id == DNN_TARGET_CPU || id == DNN_TARGET_CPU_FP16)
#define IS_DNN_VULKAN_TARGET(id) (id == DNN_TARGET_VULKAN)
Mutex& getInitializationMutex();
void initializeLayerFactory();

extern bool DNN_DIAGNOSTICS_RUN;
extern bool DNN_SKIP_REAL_IMPORT;

//
// dnn_params.cpp
//

/// Network dump level
size_t getParam_DNN_NETWORK_DUMP();

/// This parameter is useful to run with valgrind memory errors detection
bool getParam_DNN_DISABLE_MEMORY_OPTIMIZATIONS();

#ifdef HAVE_OPENCL
bool getParam_DNN_OPENCL_ALLOW_ALL_DEVICES();
#endif

int getParam_DNN_BACKEND_DEFAULT();

// Additional checks (slowdowns execution!)
bool getParam_DNN_CHECK_NAN_INF();
bool getParam_DNN_CHECK_NAN_INF_DUMP();
bool getParam_DNN_CHECK_NAN_INF_RAISE_ERROR();


inline namespace detail {

typedef std::vector<MatShape> ShapesVec;

struct LayerShapes
{
    ShapesVec in, out, internal;
    // No guarantees that layer which support in-place computations
    // will be computed in-place (input.data_ptr == output.data_ptr).
    // If layer said that it could work in-place and layers after it
    // no longer use input blob, we'll set output = input.
    bool supportInPlace;
    LayerShapes() {supportInPlace = false;}
};


#define CALL_MEMBER_FN(object, ptrToMemFn)  ((object).*(ptrToMemFn))

class NotImplemented : public Layer
{
public:
    static Ptr<Layer> create(const LayerParams &params);

    static void Register();
    static void unRegister();
};

template <typename Importer, typename ... Args>
Net readNet(Args&& ... args)
{
    Net net;
    Importer importer(net, std::forward<Args>(args)...);
    return net;
}

template <typename Importer, typename ... Args>
Net readNetDiagnostic(Args&& ... args)
{
    Net maybeDebugNet = readNet<Importer>(std::forward<Args>(args)...);
    if (DNN_DIAGNOSTICS_RUN && !DNN_SKIP_REAL_IMPORT)
    {
        // if we just imported the net in diagnostic mode, disable it and import again
        enableModelDiagnostics(false);
        Net releaseNet = readNet<Importer>(std::forward<Args>(args)...);
        enableModelDiagnostics(true);
        return releaseNet;
    }
    return maybeDebugNet;
}

class LayerHandler
{
public:
    void addMissing(const std::string& name, const std::string& type);
    bool contains(const std::string& type) const;
    void printMissing() const;

protected:
    LayerParams getNotImplementedParams(const std::string& name, const std::string& op);

private:
    std::unordered_map<std::string, std::unordered_set<std::string>> layers;
};

struct NetImplBase
{
    const int networkId;  // network global identifier
    mutable int networkDumpCounter;  // dump counter
    int dumpLevel;  // level of information dumps (initialized through OPENCV_DNN_NETWORK_DUMP parameter)

    NetImplBase();

    std::string getDumpFileNameBase() const;
};

}  // namespace detail


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

// Scalefactor is a common parameter used for data scaling. In OpenCV, we often use Scalar to represent it.
// Because 0 is meaningless in scalefactor.
// If the scalefactor is (x, 0, 0, 0), we convert it to (x, x, x, x). The following func will do this hack.
static inline Scalar_<double> broadcastRealScalar(const Scalar_<double>& _scale)
{
    Scalar_<double> scale = _scale;
    if (scale[1] == 0 && scale[2] == 0 && scale[3] == 0)
    {
        CV_Assert(scale[0] != 0 && "Scalefactor of 0 is meaningless.");
        scale = Scalar_<double>::all(scale[0]);
    }

    return scale;
}


CV__DNN_INLINE_NS_END

namespace accessor {
class DnnNetAccessor
{
public:
    static inline Ptr<Net::Impl>& getImplPtrRef(Net& net)
    {
        return net.impl;
    }
};
}

}}  // namespace

#endif  // __OPENCV_DNN_COMMON_HPP__
