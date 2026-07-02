// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/dnn/layer_reg.private.hpp>  // getLayerFactoryImpl


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Mutex& getLayerFactoryMutex()
{
    static Mutex* volatile instance = NULL;
    if (instance == NULL)
    {
        cv::AutoLock lock(getInitializationMutex());
        if (instance == NULL)
            instance = new Mutex();
    }
    return *instance;
}

static LayerFactory_Impl& getLayerFactoryImpl_()
{
    static LayerFactory_Impl impl;
    return impl;
}

LayerFactory_Impl& getLayerFactoryImpl()
{
    static LayerFactory_Impl* volatile instance = NULL;
    if (instance == NULL)
    {
        cv::AutoLock lock(getLayerFactoryMutex());
        if (instance == NULL)
        {
            instance = &getLayerFactoryImpl_();
            initializeLayerFactory();
        }
    }
    return *instance;
}

void LayerFactory::registerLayer(const String& type, Constructor constructor)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    LayerFactory_Impl::iterator it = getLayerFactoryImpl().find(type);

    if (it != getLayerFactoryImpl().end())
    {
        if (it->second.back() == constructor)
            CV_Error(cv::Error::StsBadArg, "Layer \"" + type + "\" already was registered");
        it->second.push_back(constructor);
    }
    getLayerFactoryImpl().insert(std::make_pair(type, std::vector<Constructor>(1, constructor)));
}

void LayerFactory::unregisterLayer(const String& type)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());

    LayerFactory_Impl::iterator it = getLayerFactoryImpl().find(type);
    if (it != getLayerFactoryImpl().end())
    {
        if (it->second.size() > 1)
            it->second.pop_back();
        else
            getLayerFactoryImpl().erase(it);
    }
}

bool LayerFactory::isLayerRegistered(const std::string& type)
{
    cv::AutoLock lock(getLayerFactoryMutex());
    auto& registeredLayers = getLayerFactoryImpl();
    return registeredLayers.find(type) != registeredLayers.end();
}

Ptr<Layer> LayerFactory::createLayerInstance(const String& type, LayerParams& params)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());

    cv::AutoLock lock(getLayerFactoryMutex());
    LayerFactory_Impl::const_iterator it = getLayerFactoryImpl().find(type);

    if (it != getLayerFactoryImpl().end())
    {
        CV_Assert(!it->second.empty());
        return it->second.back()(params);
    }
    else
    {
        return Ptr<Layer>();  // NULL
    }
}

typedef std::map<std::string, LayerFactory::OpConstructor> OpFactory_Impl;
typedef std::map<std::string, std::map<int, LayerFactory::ExecConstructor> > ExecFactory_Impl;

static OpFactory_Impl& getOpFactoryImpl()
{
    static OpFactory_Impl impl;
    return impl;
}

static ExecFactory_Impl& getExecFactoryImpl()
{
    static ExecFactory_Impl impl;
    return impl;
}

void LayerFactory::registerOp(const String& type, OpConstructor constructor)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());
    CV_Assert(constructor);
    cv::AutoLock lock(getLayerFactoryMutex());
    getOpFactoryImpl()[type] = constructor;  // last registration wins
}

Ptr<LayerInfo> LayerFactory::createOp(const String& type, const LayerParams& params)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());
    cv::AutoLock lock(getLayerFactoryMutex());
    OpFactory_Impl& impl = getOpFactoryImpl();
    OpFactory_Impl::const_iterator it = impl.find(type);
    if (it != impl.end())
        return it->second(params);
    return Ptr<LayerInfo>();  // NULL: no LayerInfo constructor for this type yet
}

void LayerFactory::registerExec(const String& type, int backendId, ExecConstructor constructor)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());
    CV_Assert(constructor);
    cv::AutoLock lock(getLayerFactoryMutex());
    getExecFactoryImpl()[type][backendId] = constructor;
}

Ptr<Layer> LayerFactory::createExec(const String& type, int backendId,
                                    const Ptr<LayerInfo>& data, void* backendCtx)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(type, "type", type.c_str());
    ExecConstructor ctor = nullptr;
    {
        cv::AutoLock lock(getLayerFactoryMutex());
        ExecFactory_Impl& impl = getExecFactoryImpl();
        ExecFactory_Impl::const_iterator it = impl.find(type);
        if (it != impl.end()) {
            auto bit = it->second.find(backendId);
            if (bit != it->second.end())
                ctor = bit->second;
        }
    }
    if (ctor)
        return ctor(data, backendCtx);
    return Ptr<Layer>();
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
