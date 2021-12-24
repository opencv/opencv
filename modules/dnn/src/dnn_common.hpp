// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_COMMON_HPP__
#define __OPENCV_DNN_COMMON_HPP__

#include <unordered_set>
#include <unordered_map>

#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
#define IS_DNN_OPENCL_TARGET(id) (id == DNN_TARGET_OPENCL || id == DNN_TARGET_OPENCL_FP16)
Mutex& getInitializationMutex();
void initializeLayerFactory();

extern bool DNN_DIAGNOSTICS_RUN;
extern bool DNN_SKIP_REAL_IMPORT;

namespace detail {
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
    void printMissing();

protected:
    LayerParams getNotImplementedParams(const std::string& name, const std::string& op);

private:
    std::unordered_map<std::string, std::unordered_set<std::string>> layers;
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


CV__DNN_INLINE_NS_END
}}  // namespace

#endif  // __OPENCV_DNN_COMMON_HPP__
