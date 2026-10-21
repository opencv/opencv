// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_CUDA
#include "op_cuda.hpp"
#include "cuda4dnn/init.hpp"
#include "net_impl.hpp"
#include <opencv2/dnn/layer.details.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class CUDALegacyExec : public Layer
{
public:
    CUDALegacyExec(const Ptr<Layer>& impl_, void* ctx_) : impl(impl_), ctx(ctx_) {}

    static Ptr<Layer> create(const Ptr<LayerInfo>& data, void* backendCtx)
    {
        Ptr<Layer> impl = data.dynamicCast<Layer>();
        if (!impl || !backendCtx || !impl->supportBackend(DNN_BACKEND_CUDA))
            return Ptr<Layer>();  // unsupported -> CPU fallback
        Ptr<CUDALegacyExec> e(new CUDALegacyExec(impl, backendCtx));
        e->name = impl->name;
        e->type = impl->type;
        e->inputs = impl->inputs;
        e->outputs = impl->outputs;
        return e;
    }

    void finalize(InputArrayOfArrays inputs, OutputArrayOfArrays outputs) CV_OVERRIDE
    {
        impl->finalize(inputs, outputs);
        // the engine re-finalizes on a new input signature; initCUDA() must see the new shapes
        node.release();
        cudaNode.release();
    }

    void forwardCUDA(InputArrayOfArrays inputs_,
                     OutputArrayOfArrays outputs_,
                     void* workspace) CV_OVERRIDE
    {
        std::vector<UMat> inputs, outputs;
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        cuda4dnn::csl::Workspace& ws = *reinterpret_cast<cuda4dnn::csl::Workspace*>(workspace);
        // a dynamic op's node bakes in input values (e.g. Slice offsets) that change without a new shape
        if (!node || impl->dynamicOutputShapes()) {
            impl->preferableTarget = preferableTarget;  // initCUDA may pick FP16/FP32 by target
            cuda4dnn::csl::CSLContext context = *reinterpret_cast<cuda4dnn::csl::CSLContext*>(ctx);
            node = impl->initCUDA(&context, inputs_, outputs_);
            CV_Assert(node);
            cudaNode = node.dynamicCast<CUDABackendNode>();
            CV_Assert(cudaNode);
            ws.require(cudaNode->get_workspace_memory_in_bytes());
        }
        cudaNode->forward(inputs, outputs, ws);
    }

    Ptr<Layer> impl;
    void* ctx;
    Ptr<BackendNode> node;
    Ptr<CUDABackendNode> cudaNode;
};

void registerCudaCommonExecs()
{
    CV_DNN_REGISTER_EXEC_CLASS(ReLU,        DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(ReLU6,       DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(NaryEltwise, DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Flatten,     DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Reshape2,    DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(BatchNorm2,  DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(MaxPool,     DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Gemm,        DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Pooling,     DNN_BACKEND_CUDA, CUDALegacyExec);  // GlobalAveragePool/GlobalMaxPool
    CV_DNN_REGISTER_EXEC_CLASS(AveragePool, DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Attention,       DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Cast2,           DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(ConstantOfShape, DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Expand2,         DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Gather2,         DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Not,             DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Reduce2,         DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Shape,           DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Slice2,          DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Softmax,         DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Tile2,           DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Transpose,       DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Concat2,             DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Gelu,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(LayerNormalization2, DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(MatMul,              DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Split2,              DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Sqrt,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Squeeze,             DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Unsqueeze,           DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Sigmoid,             DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Swish,               DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Mish,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(TanH,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(ELU,                 DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Celu,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Selu,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Erf,                 DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(GeluApproximation,   DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(HardSwish,           DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(HardSigmoid,         DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(ThresholdedRelu,     DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Exp,                 DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Log,                 DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Power,               DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(AbsVal,              DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Sign,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(BNLL,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Softplus,            DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Softsign,            DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Shrink,              DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Ceil,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Floor,               DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Round,               DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Sin,                 DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Sinh,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Cos,                 DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Cosh,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Tan,                 DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Acos,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Acosh,               DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Asin,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Asinh,               DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Atan,                DNN_BACKEND_CUDA, CUDALegacyExec);
    CV_DNN_REGISTER_EXEC_CLASS(Atanh,               DNN_BACKEND_CUDA, CUDALegacyExec);
}


void Net::Impl::initCUDABackend()
{
    CV_Assert(preferableBackend == DNN_BACKEND_CUDA);

    if (!cudaInfo) /* we need to check only once */
        cuda4dnn::checkVersions();

    if (cuda4dnn::getDeviceCount() <= 0)
        CV_Error(Error::StsError, "No CUDA capable device found.");

    if (cuda4dnn::getDevice() < 0)
        CV_Error(Error::StsError, "No CUDA capable device selected.");

    if (!cuda4dnn::isDeviceCompatible())
        CV_Error(Error::GpuNotSupported, "OpenCV was not built to work with the selected device. Please check CUDA_ARCH_PTX or CUDA_ARCH_BIN in your build configuration.");

    if (preferableTarget == DNN_TARGET_CUDA_FP16 && !cuda4dnn::doesDeviceSupportFP16())
    {
        CV_LOG_WARNING(NULL, "The selected CUDA device does not support FP16 target; switching to FP32 target.");
        preferableTarget = DNN_TARGET_CUDA;
    }

    if (!cudaInfo)
    {
        cuda4dnn::csl::CSLContext context;
        context.stream = cuda4dnn::csl::Stream(true);
        context.cublas_handle = cuda4dnn::csl::cublas::Handle(context.stream);
        context.cudnn_handle = cuda4dnn::csl::cudnn::Handle(context.stream);

        cudaInfo = std::make_unique<CudaInfo_t>(std::move(context));
    }

    cudaInfo->workspace = cuda4dnn::csl::Workspace();  // release workspace memory if any

    for (auto& layer : layers)
    {
        auto& ld = layer.second;

        if (ld.id == 0 && netInputLayer->supportBackend(preferableBackend))
        {
            for (auto& wrapper : ld.inputBlobsWrappers)
            {
                auto cudaWrapper = wrapper.dynamicCast<CUDABackendWrapper>();
                cudaWrapper->setStream(cudaInfo->context.stream);
            }
        }

        for (auto& wrapper : ld.outputBlobsWrappers)
        {
            auto cudaWrapper = wrapper.dynamicCast<CUDABackendWrapper>();
            cudaWrapper->setStream(cudaInfo->context.stream);
        }
    }

    for (auto& layer : layers)
    {
        auto& ld = layer.second;
        auto& layerInstance = ld.layerInstance;

        if (!layerInstance->supportBackend(DNN_BACKEND_CUDA))
        {
            std::ostringstream os;
            os << "CUDA backend will fallback to the CPU implementation for the layer \"" << ld.name
               << "\" of type " << ld.type << '\n';
            CV_LOG_INFO(NULL, os.str().c_str());
            continue;
        }

        /* we make a copy so that `initCUDA` doesn't modify `cudaInfo->context` */
        auto context = cudaInfo->context;
        auto node = layerInstance->initCUDA(&context, ld.inputBlobsWrappers, ld.outputBlobsWrappers);
        ld.backendNodes[DNN_BACKEND_CUDA] = node;

        if(!node.empty())
        {
            auto cudaNode = node.dynamicCast<CUDABackendNode>();
            cudaInfo->workspace.require(cudaNode->get_workspace_memory_in_bytes());
        }
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // HAVE_CUDA

namespace cv { namespace dnn {

bool haveCUDA()
{
#ifdef HAVE_CUDA
    int dev = 0;
    static bool ret = (cudaGetDevice(&dev) == cudaSuccess);
    return ret;
#else
    return false;
#endif
}

}}  // namespace cv::dnn
