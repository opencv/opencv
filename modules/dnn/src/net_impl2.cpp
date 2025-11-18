// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"
#ifdef HAVE_CUDA
#include "cuda/layer_cudnn.hpp"
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

std::string modelFormatToString(ModelFormat modelFormat)
{
    return
        modelFormat == DNN_MODEL_ONNX ? "ONNX" :
        modelFormat == DNN_MODEL_TF ? "TF" :
        modelFormat == DNN_MODEL_TFLITE ? "TFLite" :
        modelFormat == DNN_MODEL_CAFFE ? "Caffe" : "Unknown/Generic";
}

std::string argKindToString(ArgKind kind)
{
    return
        kind == DNN_ARG_CONST ? "Const" :
        kind == DNN_ARG_INPUT ? "Input" :
        kind == DNN_ARG_OUTPUT ? "Output" :
        kind == DNN_ARG_TEMP ? "Temp" :
        kind == DNN_ARG_PATTERN ? "Pattern" : "???";
}

ArgData::ArgData()
{
    kind = DNN_ARG_EMPTY;
    type = -1;
}

class GraphImpl : public Graph
{
public:
    GraphImpl(Net::Impl* netimpl, const std::string& name,
              const std::vector<Arg>& inputs)
    {
        netimpl_ = netimpl;
        name_ = name;
        inputs_ = inputs;
    }

    virtual ~GraphImpl()
    {
    }

    virtual std::string name() const override { return name_; }
    virtual bool empty() const override { return prog_.empty(); }
    virtual void clear() override
    {
        prog_.clear();
    }

    /*Ptr<Graph> clone(Net* newnet) const
    {
        Graph g = std::make_shared<GraphData>((newnet ? *newnet : *net_), name_, inputs_, ispattern_);
        g->outputs_ = outputs_;
        g->backend_ = backend_;
        // don't copy optigraph_. It has to be re-created
        for (auto n : prog_) {
            g->prog_.push_back(n->clone(g->net_));
        }
        return g;
    }*/

    virtual const std::vector<Arg>& append(Ptr<Layer>& layer,
                const std::vector<std::string>& outnames) override
    {
        CV_Assert(layer);
        int i, noutputs = (int)outnames.size();
        //CV_Assert(layer->minNumOutputs() <= noutputs && noutputs <= layer->maxNumOutputs());

        layer->outputs.resize(noutputs);
        for (i = 0; i < noutputs; i++) {
            Arg outarg = netimpl_->getArg(outnames[i]);
            ArgKind kind = netimpl_->argKind(outarg);
            CV_Assert(kind == DNN_ARG_TEMP || kind == DNN_ARG_OUTPUT);
            layer->outputs[i] = outarg;
        }

        prog_.push_back(layer);
        return layer->outputs;
    }

    virtual Arg append(Ptr<Layer>& layer,
               const std::string& outname) override
    {
        std::vector<std::string> outnames = {outname};
        const std::vector<Arg>& outputs = append(layer, outnames);
        CV_Assert(outputs.size() == 1);
        return outputs[0];
    }

    virtual std::ostream& dump(std::ostream& strm, int indent, bool comma) override
    {
        CV_Assert(netimpl_);
        size_t ninputs = inputs_.size(), noutputs = outputs_.size();
        int delta_indent = netimpl_->dump_indent;
        int subindent = indent + delta_indent;
        int argindent = subindent + delta_indent;
        strm << "{\n";
        prindent(strm, subindent);
        strm << "name: ";
        if (name_.empty())
            strm << "<noname>\n";
        else
            strm << '\"' << name_ << "\"\n";
        prindent(strm, subindent);
        strm << "inputs: [\n";
        for (size_t i = 0; i < ninputs; i++) {
            netimpl_->dumpArg(strm, inputs_[i], argindent, i+1 < ninputs, true);
        }
        prindent(strm, subindent);
        strm << "],\n";
        prindent(strm, subindent);
        strm << "outputs: [\n";
        for (size_t i = 0; i < noutputs; i++) {
            netimpl_->dumpArg(strm, outputs_[i], argindent, i+1 < noutputs, true);
        }
        prindent(strm, subindent);
        strm << "],\n";
        prindent(strm, subindent);
        strm << "nodes: [\n";
        size_t nlayers = prog_.size();
        for (size_t i = 0; i < nlayers; i++) {
            prindent(strm, argindent);
            strm << "// op #" << i << "\n";
            const Ptr<Layer>& layer = prog_[i];
            layer->dump(strm, argindent, i+1 < nlayers);
        }
        prindent(strm, subindent);
        strm << "]\n";
        prindent(strm, indent);
        strm << '}';
        if (comma)
            strm << ',';
        strm << '\n';
        return strm;
    }

    //virtual Net* net() const override { return net_; }
    virtual const std::vector<Arg>& inputs() const override { return inputs_; }
    virtual const std::vector<Arg>& outputs() const override { return outputs_; }

    virtual void setOutputs(const std::vector<Arg>& outputs) override {
        CV_Assert(netimpl_);
        netimpl_->checkArgs(outputs);
        outputs_ = outputs;
    }
    virtual const std::vector<Ptr<Layer> >& prog() const override { return prog_; }
    virtual void setProg(const std::vector<Ptr<Layer> >& newprog) override { prog_ = newprog; }

protected:
    Net::Impl* netimpl_;
    std::string name_;
    std::vector<Arg> inputs_;
    std::vector<Arg> outputs_;
    std::vector<Ptr<Layer> > prog_;
};

Ptr<Graph> Graph::create(void* netimpl, const std::string& name,
                         const std::vector<Arg>& inputs)
{
    return Ptr<Graph>(new GraphImpl(reinterpret_cast<Net::Impl*>(netimpl), name, inputs));
}

Graph::~Graph() {}

bool Net::Impl::isConstArg(Arg arg) const
{
    return argKind(arg) == DNN_ARG_CONST;
}

#ifdef HAVE_CUDA
cudnnTensorDescriptor_t Net::Impl::argTensorCuDNN(Arg arg,
                                                  int N, int C, int H, int W,
                                                  int n_stride, int c_stride, int h_stride, int w_stride,
                                                  cudnnDataType_t dtype)
{
    CV_Assert((size_t)arg.idx < args.size());
    if (cudnnTensors.size() < args.size())
        cudnnTensors.resize(args.size(), nullptr);
    if (!cudnnTensors[arg.idx])
    {
        cudnnTensorDescriptor_t desc = nullptr;
        cudnnStatus_t st = cudnnCreateTensorDescriptor(&desc);
        CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnCreateTensorDescriptor failed");
        cudnnTensors[arg.idx] = desc;
    }
    // Configure descriptor with explicit strides
    cudnnStatus_t st = cudnnSetTensor4dDescriptorEx(cudnnTensors[arg.idx],
                                                    dtype, N, C, H, W,
                                                    n_stride, c_stride, h_stride, w_stride);
    CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnSetTensor4dDescriptorEx failed");
    return cudnnTensors[arg.idx];
}

cudnnFilterDescriptor_t Net::Impl::filterDescCuDNN(const std::string& key,
                                                   cudnnDataType_t dtype, cudnnTensorFormat_t layout,
                                                   int outC, int inC, int kH, int kW)
{
    cudnnFilterDescriptor_t& desc = cudnnFilters[key];
    if (!desc) {
        cudnnStatus_t st = cudnnCreateFilterDescriptor(&desc);
        CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnCreateFilterDescriptor failed");
    }
    cudnnStatus_t st = cudnnSetFilter4dDescriptor(desc, dtype, layout, outC, inC, kH, kW);
    CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnSetFilter4dDescriptor failed");
    return desc;
}

cudnnConvolutionDescriptor_t Net::Impl::convDescCuDNN(const std::string& key,
                                                      int padH, int padW, int strideH, int strideW,
                                                      int dilH, int dilW, int groups, cudnnDataType_t computeType)
{
    cudnnConvolutionDescriptor_t& desc = cudnnConvs[key];
    if (!desc) {
        cudnnStatus_t st = cudnnCreateConvolutionDescriptor(&desc);
        CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnCreateConvolutionDescriptor failed");
    }
    cudnnStatus_t st = cudnnSetConvolution2dDescriptor(desc, padH, padW, strideH, strideW, dilH, dilW,
                                                       CUDNN_CROSS_CORRELATION, computeType);
    CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnSetConvolution2dDescriptor failed");
#if CUDNN_MAJOR >= 7
    cudnnSetConvolutionMathType(desc, CUDNN_DEFAULT_MATH);
#endif
    if (groups > 1) {
        cudnnSetConvolutionGroupCount(desc, groups);
    }
    return desc;
}

cudnnActivationDescriptor_t Net::Impl::activationDescCuDNN(const std::string& key,
                                                           cudnnActivationMode_t mode,
                                                           cudnnNanPropagation_t nanOpt,
                                                           double coef)
{
    cudnnActivationDescriptor_t& desc = cudnnActs[key];
    if (!desc) {
        cudnnStatus_t st = cudnnCreateActivationDescriptor(&desc);
        CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnCreateActivationDescriptor failed");
    }
    cudnnStatus_t st = cudnnSetActivationDescriptor(desc, mode, nanOpt, coef);
    CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnSetActivationDescriptor failed");
    return desc;
}

cudnnOpTensorDescriptor_t Net::Impl::opTensorDescCuDNN(const std::string& key,
                                                       cudnnOpTensorOp_t op,
                                                       cudnnDataType_t dtype,
                                                       cudnnNanPropagation_t nanOpt)
{
    cudnnOpTensorDescriptor_t& desc = cudnnOpTensors[key];
    if (!desc) {
        cudnnStatus_t st = cudnnCreateOpTensorDescriptor(&desc);
        CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnCreateOpTensorDescriptor failed");
    }
    cudnnStatus_t st = cudnnSetOpTensorDescriptor(desc, op, dtype, nanOpt);
    CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnSetOpTensorDescriptor failed");
    return desc;
}

cudnnPoolingDescriptor_t Net::Impl::poolingDescCuDNN(const std::string& key,
                                                     cudnnPoolingMode_t mode,
                                                     cudnnNanPropagation_t nanOpt,
                                                     int kH, int kW,
                                                     int padH, int padW,
                                                     int strideH, int strideW)
{
    cudnnPoolingDescriptor_t& desc = cudnnPools[key];
    if (!desc) {
        cudnnStatus_t st = cudnnCreatePoolingDescriptor(&desc);
        CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnCreatePoolingDescriptor failed");
    }
    cudnnStatus_t st = cudnnSetPooling2dDescriptor(desc, mode, nanOpt, kH, kW, padH, padW, strideH, strideW);
    CV_Assert(st == CUDNN_STATUS_SUCCESS && "cudnnSetPooling2dDescriptor failed");
    return desc;
}
#endif

const ArgData& Net::Impl::argData(Arg arg) const
{
    CV_Assert((size_t)arg.idx < args.size());
    return args[arg.idx];
}

const std::string& Net::Impl::argName(Arg arg) const
{
    return argData(arg).name;
}

ArgKind Net::Impl::argKind(Arg arg) const
{
    return argData(arg).kind;
}

Mat& Net::Impl::argTensor(Arg arg) const
{
    const ArgData& adata = argData(arg);
    if (adata.kind == DNN_ARG_TEMP) {
        CV_Assert(__tensors__.at(arg.idx).empty());
        int bufidx = bufidxs.at(arg.idx);
        CV_Assert(bufidx >= 0);
        return const_cast<Mat&>(buffers.at(bufidx));
    }
    return const_cast<Mat&>(__tensors__.at(arg.idx));
}

Arg Net::Impl::getArg(const std::string& name)
{
    auto it = argnames.find(name);
    if (it != argnames.end()) {
        return Arg((int)it->second);
    }
    return newArg(name, DNN_ARG_TEMP);
}

bool Net::Impl::haveArg(const std::string& name) const
{
    return argnames.find(name) != argnames.end();
}

Arg Net::Impl::newConstArg(const std::string& name, const Mat& m)
{
    if (name.empty()) {
        CV_Assert(m.empty());
        return Arg();
    }
    Arg arg = newArg(name, DNN_ARG_CONST, true);
    __tensors__[arg.idx] = m;
    ArgData& adata = args[arg.idx];
    adata.type = m.type();
    adata.shape = m.shape();
    return arg;
}

Arg Net::Impl::newArg(const std::string& name, ArgKind kind, bool allowEmptyName)
{
    CV_Assert(allowEmptyName || !name.empty());
    int idx = (int)args.size();

    if (!name.empty()) {
        CV_Assert(argnames.find(name) == argnames.end());
        argnames.insert(std::make_pair(name, (int64_t)idx));
    }

    ArgData adata;
    adata.name = name;
    adata.kind = kind;
    args.push_back(adata);
    __tensors__.push_back(Mat());
    bufidxs.push_back(-1);

    return Arg(idx);
}

int Net::Impl::findDim(const std::string& dimname, bool insert)
{
    if (!dimname.empty()) {
        auto it = dimnames.find(dimname);
        if (it != dimnames.end()) {
            return (int)it->second;
        }
    }
    if (!insert) {
        CV_Error_(Error::StsObjectNotFound, ("symbolic dimension '%s' is not found",
                                             dimname.empty() ? "<some unique name>" : dimname.c_str()));
    }
    int value = -(int)dimnames_vec.size() - 1;
    std::string inserted_dimname = dimname.empty() ? format("N!%d", -value) : dimname;
    dimnames.insert(std::make_pair(inserted_dimname, (int64_t)value));
    dimnames_vec.push_back(inserted_dimname);
    return value;
}

Ptr<Graph> Net::Impl::newGraph(const std::string& name_, const std::vector<Arg>& inpargs, bool ismain)
{
    if (ismain)
        globGraphIdx = 0;
    std::string name = name_;
    if (name_.empty())
        name = ismain ? std::string("main") : format("subgraph_%d", globGraphIdx);
    globGraphIdx++;
    Ptr<Graph> graph = Graph::create(this, name, inpargs);
    if (ismain)
        mainGraph = graph;
    return graph;
}

void Net::Impl::prepareForInference()
{
    if (!prepared) {
        constFold();
        //inferTypes();
        //constArgs();
        //inferShapes(true);
        //fuse();
        //useBlockLayout();
        //inferShapes(true);
        assignBuffers();
        totalLayers = updateGraphOfs(mainGraph, 0, true);
        prepared = true;
        finalizeLayers = true;
    }
}

void Net::Impl::allocateLayerOutputs(
                          const Ptr<Layer>& layer,
                          const std::vector<int>& inpTypes,
                          const std::vector<MatShape>& inpShapes,
                          std::vector<int>& outTypes,
                          std::vector<MatShape>& outShapes,
                          std::vector<std::pair<uchar*, size_t> >& outOrigData,
                          std::vector<Mat>& outputs,
                          std::vector<int>& tempTypes,
                          std::vector<MatShape>& tempShapes,
                          std::vector<Mat>& temps,
                          std::vector<Mat>& globalTemps,
                          bool useBufferPool)
{
    // In theory, when
    // 1) useBufferPool==true,
    // 2) the buffers in the pool are already big enough (e.g. when we already run inference a few times to let them grow)
    // 3) getMemoryShapes() and getTypes() are implemented efficiently without any memory allocations
    // the method allocateLayerOutputs() should not allocate any memory either.
    //
    // Well, currently it still may do so, because Mat::fit() may create e.g. 4D tensor on top of 1D buffer and then
    // MatSize and MatStep will require dynamic memory allocation (those are very small buffers though).
    // But we plan to make MatSize and MatStep lighter so that they don't use dynamic memory.
    size_t noutputs = layer->outputs.size();
    outShapes.clear();
    outTypes.clear();
    tempShapes.clear();
    tempTypes.clear();
    layer->getMemoryShapes(inpShapes, (int)noutputs, outShapes, tempShapes);
    layer->getTypes(inpTypes, (int)noutputs, (int)tempShapes.size(), outTypes, tempTypes);
    CV_Assert(tempShapes.size() == tempTypes.size());
    CV_Assert(outShapes.size() == outTypes.size());
    CV_Assert(outShapes.size() == noutputs);
    outputs.assign(noutputs, Mat());
    outOrigData.resize(noutputs);
    for (size_t i = 0; i < noutputs; i++) {
        Arg out = layer->outputs[i];
        if (useBufferPool) {
            Mat& out_t = argTensor(out);
            out_t.fit(outShapes[i], outTypes[i]);
            outputs[i] = out_t;
        } else {
            outputs[i].fit(outShapes[i], outTypes[i]);
        }
        outOrigData[i].first = outputs[i].u ? outputs[i].u->data : nullptr;
        outOrigData[i].second = outputs[i].u ? outputs[i].u->size : 0;
    }
    // [TODO] probably there should be a smarter algorithm that e.g. sorts
    // temp buffers by size in decreasing order and assigns global temps accordingly
    // in order to minimize the total size of temp buffers
    size_t ntemps = tempShapes.size();
    temps.resize(ntemps);
    globalTemps.resize(std::max(ntemps, globalTemps.size()));
    for (size_t i = 0; i < ntemps; i++) {
        globalTemps[i].fit(tempShapes[i], tempTypes[i]);
        temps[i] = globalTemps[i];
    }
}

#ifdef HAVE_CUDA
namespace {
static inline cv::cuda::GpuMat dnn_createGpuMatFromShape(const MatShape& shp, int type, cv::cuda::GpuMatND& storageND)
{
    MatShape ndsize;
    ndsize.reserve(shp.dims);
    size_t total = 1;
    for (int d = 0; d < shp.dims; ++d) {
        int dim = shp[d] > 0 ? shp[d] : 1;
        ndsize.push_back(dim);
        total *= (size_t)dim;
    }
    storageND.fit(ndsize, type);

    int rows = ndsize.empty() ? 1 : ndsize[0];
    if (rows <= 0) rows = 1;
    size_t cols_sz = rows > 0 ? (total / (size_t)rows) : total;
    int cols = (int)cols_sz; if (cols <= 0) cols = 1;
    size_t esz = (size_t)CV_ELEM_SIZE(type);
    size_t step_bytes = (size_t)cols * esz;
    return cv::cuda::GpuMat(rows, cols, type, storageND.getDevicePtr(), step_bytes);
}
} // anonymous namespace

static inline void dnn_downloadIntoND(const cv::cuda::GpuMat& gm,
                                      cv::Mat& nd,
                                      const MatShape& ndShape,
                                      int dstType)
{
    if (nd.shape() != ndShape || nd.type() != dstType)
        nd.fit(ndShape, dstType);
    CV_Assert(nd.isContinuous());

    const int rows = gm.rows > 0 ? gm.rows : 1;
    const int cols = gm.cols > 0 ? gm.cols : 1;

    cv::Mat nd2d = (nd.dims <= 2) ? nd : nd.reshape(CV_MAT_CN(dstType), rows); // alias, no alloc
    CV_Assert(nd2d.rows == rows && nd2d.cols == cols && nd2d.type() == gm.type());
    gm.download(nd2d);
}
void Net::Impl::allocateLayerGpuOutputs(
                          const Ptr<Layer>& layer,
                          const std::vector<int>& inpTypes,
                          const std::vector<MatShape>& inpShapes,
                          std::vector<int>& outTypes,
                          std::vector<MatShape>& outShapes,
                          std::vector<int>& tempTypes,
                          std::vector<MatShape>& tempShapes,
                          std::vector<cv::cuda::GpuMat>& outGpuMats,
                          std::vector<cv::cuda::GpuMat>& tempGpuMats)
{
    size_t noutputs = layer->outputs.size();
    outShapes.clear();
    outTypes.clear();
    tempShapes.clear();
    tempTypes.clear();
    layer->getMemoryShapes(inpShapes, (int)noutputs, outShapes, tempShapes);
    layer->getTypes(inpTypes, (int)noutputs, (int)tempShapes.size(), outTypes, tempTypes);
    CV_Assert(tempShapes.size() == tempTypes.size());
    CV_Assert(outShapes.size() == outTypes.size());
    CV_Assert(outShapes.size() == noutputs);

    outGpuMats.resize(noutputs);

    if (gpuTensorsND.size() < args.size()) gpuTensorsND.resize(args.size());
    size_t maxBufIdx = 0;
    for (size_t i = 0; i < noutputs; ++i) {
        Arg out = layer->outputs[i];
        if (args[out.idx].kind == DNN_ARG_TEMP) {
            int b = bufidxs.at(out.idx);
            if (b >= 0) maxBufIdx = std::max(maxBufIdx, (size_t)b);
        }
    }
    if (gpuBuffersND.size() <= maxBufIdx) gpuBuffersND.resize(maxBufIdx + 1);

    for (size_t i = 0; i < noutputs; i++) {
        try {
            const MatShape& shp = outShapes[i];
            int type = outTypes[i];

            Arg out = layer->outputs[i];
            const ArgData& ad = args.at(out.idx);
            if (ad.kind == DNN_ARG_TEMP) {
                int b = bufidxs.at(out.idx);
                CV_Assert(b >= 0);
                outGpuMats[i] = dnn_createGpuMatFromShape(shp, type, gpuBuffersND[(size_t)b]);
            } else {
                outGpuMats[i] = dnn_createGpuMatFromShape(shp, type, gpuTensorsND[out.idx]);
            }
        } catch (const cv::Exception& e) {
            CV_LOG_WARNING(NULL, "DNN/CUDA: failed to prepare output GpuMat #" << i << ": " << e.what());
            outGpuMats[i].release();
        }
    }

    size_t ntemps = tempShapes.size();
    tempGpuMats.resize(ntemps);
    layerTempGpuND.resize(ntemps);
    for (size_t i = 0; i < ntemps; i++) {
        try {
            const MatShape& shp = tempShapes[i];
            int type = tempTypes[i];
            tempGpuMats[i] = dnn_createGpuMatFromShape(shp, type, layerTempGpuND[i]);
        } catch (const cv::Exception& e) {
            CV_LOG_WARNING(NULL, "DNN/CUDA: failed to prepare temp GpuMat #" << i << ": " << e.what());
            tempGpuMats[i].release();
        }
    }
}
#endif

void Net::Impl::forwardMainGraph(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
{
    if (!mainGraph) {
        CV_Error(Error::StsNullPtr, "the model was not loaded");
    }
    // ************ uncomment one of the lines below for debugging **********
    //tracingMode = DNN_TRACE_OP;
    //tracingMode = DNN_TRACE_ALL;
    // [TODO] initialize profile, tracer, symbolic shapes etc.
    size_t nsymdims = dimnames_vec.size();
    dimvalues.assign(nsymdims, -1);
    layersTimings.assign(totalLayers + 1, 0.);

    forwardGraph(mainGraph, inputs, outputs, true);

    // reset finalizeLayer so that layers are only initialized once.
    // [TODO] if a target or backend change or there are some other important
    // global changes in configuration, finalizeLayers should be set to 'true' again
    finalizeLayers = false;
}

Mat Net::Impl::forwardWithSingleOutput(const std::string& outname)
{
    if (!mainGraph) {
        CV_Error(Error::StsNullPtr, "the model was not loaded");
    }
    const std::vector<Arg>& outargs = mainGraph->outputs();
    CV_Assert(outargs.size() > 0);
    if (!outname.empty()) {
        const ArgData& outdata = args.at(outargs[0].idx);
        CV_Assert(outdata.name == outname);
    }
    std::vector<Mat> inps={}, outs;
    forwardMainGraph(inps, outs);
    return outs[0];
}

void Net::Impl::forwardWithMultipleOutputs(OutputArrayOfArrays outblobs, const std::vector<std::string>& outnames)
{
    if (!mainGraph) {
        CV_Error(Error::StsNullPtr, "the model was not loaded");
    }
    const std::vector<Arg>& outargs = mainGraph->outputs();
    std::vector<int> outidxs;
    int i, j, noutputs = (int)outargs.size();
    if (!outnames.empty()) {
        CV_CheckEQ((int)outnames.size(), noutputs, "the number of requested and actual outputs must be the same");
        if (noutputs == 1 && outnames[0].empty())
            ;
        else {
            for (i = 0; i < noutputs; i++) {
                const std::string& outname = outnames[i];
                for (j = 0; j < noutputs; j++) {
                    const ArgData& adata = args.at(outargs[j].idx);
                    if (adata.name == outname) {
                        outidxs.push_back((int)j);
                        break;
                    }
                }
                if (j == noutputs) {
                    CV_Error_(Error::StsObjectNotFound, ("the required output '%s' is not found", outname.c_str()));
                }
            }
        }
    }
    std::vector<Mat> inps={}, outs;
    forwardMainGraph(inps, outs);
    CV_Assert(outs.size() == noutputs);
    std::vector<Mat>* outMats = nullptr;
    std::vector<UMat>* outUMats = nullptr;
    _InputArray::KindFlag outKind = outblobs.kind();
    if (outKind == _InputArray::STD_VECTOR_MAT) {
        outMats = &outblobs.getMatVecRef();
        outMats->resize(noutputs);
    } else if (outKind == _InputArray::STD_VECTOR_UMAT) {
        outUMats = &outblobs.getUMatVecRef();
        outUMats->resize(noutputs);
    } else if (outKind == _InputArray::MAT || outKind == _InputArray::UMAT) {
        CV_Assert(noutputs == 1);
    } else {
        CV_Error(Error::StsBadArg, "outputs must be Mat, UMat, a vector of Mat's or a vector of UMat's");
    }
    for (i = 0; i < noutputs; i++) {
        int j = outidxs.empty() ? i : outidxs[i];
        Mat src = outs[j];
        if (outMats) {
            src.copyTo(outMats->at(i));
        } else if (outUMats) {
            src.copyTo(outUMats->at(i));
        } else {
            src.copyTo(outblobs);
        }
    }
}

/*void Net::Impl::checkAndUpdateDim(const Ptr<Graph>& g, const Ptr<Layer>& layer, Arg inp, int j, int value)
{
    const ArgData& adata = args[inp.idx];
    int64_t value0 = adata.size.size[j];
    if (value0 >= 0) {
        if (value0 != value) {
            CV_Error_(Error::StsBadArg, ("graph '%s': node '%s': %d-th dimension of argument '%s' is wrong: %lld given, %lld expected",
                                        g->name().data(), node ? node->name().data() : "none (graph input)", j, adata.name.c_str(), value, value0));
        }
    } else {
        int64_t idx = -value0-1;
        CV_Assert(0 <= idx && idx < (int64_t)dimvalues.size());
        value0 = dimvalues[idx];
        if (value0 < 0) {
            dimvalues[idx] = value;
        } else if (value0 != value) {
            CV_Error_(Error::StsBadArg,
            ("graph '%s': node '%s': %d-th dimension '%s' of argument '%s' is wrong: %lld given, but '%s' is already set to %lld",
                    g->name().data(), node ? node->name().data() : "none (graph input)",
                    j, dimnames_[idx].c_str(), adata.name.c_str(),
                    value, dimnames_[idx].c_str(), value0));
        }
    }
}*/

void Net::Impl::traceArg(std::ostream& strm_, const char* prefix, size_t i, Arg arg, bool dumpdata)
{
    const int PPRINT_CONTEXT = 3;
    const int PPRINT_CONST_THRESHOLD = 16;
    const int PPRINT_ALL_THRESHOLD = 100;
    const Mat& m = argTensor(arg);
    const ArgData& adata = args.at(arg.idx);
    bool constArg = adata.kind == DNN_ARG_CONST;
    // [TODO] replace with type compatibility check
    // CV_Assert(m.type() == adata.type);
    strm_ << prefix << " " << i << ". Name: " << (arg.idx > 0 ? adata.name.c_str() : "<empty>") << "\n";
    if (arg.idx == 0)
        return;
    strm_ << "  Buf: " << bufidxs.at(arg.idx) << "\n";
    strm_ << "  Type: " << typeToString(adata.type) << " \n";
    MatShape shape = m.shape();
    strm_ << "  Shape: " << shape;
    if (constArg && m.total() <= PPRINT_CONST_THRESHOLD) {
        strm_ << " /* ";
        pprint(strm_, m, 0, PPRINT_CONTEXT, PPRINT_CONST_THRESHOLD, '{');
        strm_ << " */";
    }
    strm_ << "\n  Layout: " << layoutToString(shape.layout) << "\n";
    if (dumpdata && !constArg) {
        // [TODO] when we support block layout, block-layout tensor
        // should be converted to the original layout before printing it
        pprint(strm_, m, 0, PPRINT_CONTEXT, PPRINT_ALL_THRESHOLD, '[');
        strm_ << "\n";
    }
}

void Net::Impl::setMainGraphInput(InputArray m, const std::string& inpname)
{
    CV_Assert(mainGraph);
    const std::vector<Arg>& gr_inputs = mainGraph->inputs();
    size_t i, ninputs = gr_inputs.size();
    if (inpname.empty()) {
        CV_Assert(ninputs == 1 && "empty name can only be used to set input if there is just one input");
        i = 0;
    } else {
        for (i = 0; i < ninputs; i++) {
            const ArgData& adata = args.at(gr_inputs[i].idx);
            CV_Assert(adata.kind == DNN_ARG_INPUT);
            if (adata.name == inpname)
                break;
        }
        if ((i == ninputs) && (!isdigit(inpname[0]) || !sscanf(inpname.c_str(), "%zu", &i))) {
            CV_Error_(Error::StsObjectNotFound, ("input '%s' is not found", inpname.c_str()));
        }
    }
    setGraphInput(mainGraph, i, m.getMat());
}

void Net::Impl::setGraphInput(Ptr<Graph>& graph, size_t idx, const Mat& m)
{
    int mtype = m.type();
    MatShape mshape = m.shape();
    const std::vector<Arg>& gr_inputs = graph->inputs();
    if (idx >= gr_inputs.size())
    {
        return;
    }
    Arg inp = gr_inputs[idx];
    const ArgData& adata = args.at(inp.idx);
    /*
     [TODO] add more detailed shape check
     if (adata.shape.dims != mshape.dims) {
     CV_Error_(Error::StsBadArg, ("wrong dimensionality of argument '%s': %d given, %d expected",
     adata.name.c_str(), tsize.ndims, adata.size.ndims));
     }

     for (int k = 0; k < mshape.dims; k++) {
        checkAndUpdateDim(graph, Node(), inp, k, tsize.size[k]);
     }
    */

    if (adata.kind == DNN_ARG_INPUT) {
        int adata_type = adata.type;
        if ((adata_type == CV_16F || adata_type == CV_16BF) && !enableFP16)
            adata_type = CV_32F;

        if (adata_type != mtype &&
            !((adata_type == CV_64F || adata_type == CV_32F || adata_type == CV_16F || adata_type == CV_16BF) &&
              (mtype == CV_64F || mtype == CV_32F || mtype == CV_16F || mtype == CV_16BF)) &&
            !((adata_type == CV_8U || adata_type == CV_8S || adata_type == CV_16U || adata_type == CV_16S || adata_type == CV_32S || adata_type == CV_32U || adata_type == CV_64S || adata_type == CV_64U) &&
              (mtype == CV_8U || mtype == CV_8S || mtype == CV_16U || mtype == CV_16S || mtype == CV_32S || mtype == CV_32U || mtype == CV_64S || mtype == CV_64U)) &&
            !(adata.type == CV_16BF && mtype == CV_16U) && !(adata.type == CV_16F && mtype == CV_16U) &&
            !m.empty())
        {
            CV_Error_(Error::StsBadArg, ("incompatible type of input tensor #%zu '%s': %s given, %s expected",
                                         idx, adata.name.c_str(), typeToString(mtype).c_str(),
                                         typeToString(adata.type).c_str()));
        }
        Mat& inp_t = argTensor(inp);
        if (inp_t.shape() != mshape || inp_t.type() != adata_type)
            finalizeLayers = true;
        inp_t.fit(mshape, adata_type);

        if (adata.type == CV_16BF && mtype == CV_16U)
        {
            Mat tmp(mshape, CV_16BF, (void*)m.data);
            tmp.convertTo(inp_t, adata_type);
        }
        else if (adata.type == CV_16F && mtype == CV_16U)
        {
            Mat tmp(mshape, CV_16F, (void*)m.data);
            tmp.convertTo(inp_t, adata_type);
        }
        else
        {
            m.convertTo(inp_t, adata_type);
        }
    } else if (adata.kind == DNN_ARG_TEMP) {
        int bufidx = bufidxs.at(inp.idx);
        Mat& buf = buffers.at(bufidx);
        buf.fit(mshape, mtype); // minimize reallocations
        m.copyTo(buf);
        ensureBufferWrapper(bufidx);
    } else {
        CV_Error_(Error::StsBadArg, ("graph %s: argument '%s' must be 'INPUT' or 'TEMP', not '%s'",
                                     graph->name().data(), adata.name.c_str(),
                                     argKindToString(adata.kind).c_str()));
    }
}

void Net::Impl::forwardGraph(Ptr<Graph>& graph, InputArrayOfArrays inputs_,
                             OutputArrayOfArrays outputs_, bool isMainGraph)
{
    auto graphofs_it = graphofs.find(graph->name());
    if (graphofs_it == graphofs.end()) {
        CV_Error_(Error::StsObjectNotFound, ("graph '%s' does not belong to the model", graph->name().c_str()));
    }
    std::ostream& strm_ = dump_strm ? *dump_strm : std::cout;
    const std::vector<Ptr<Layer> >& prog = graph->prog();
    size_t i, nops = prog.size();
    const std::vector<Arg>& gr_inputs = graph->inputs();
    const std::vector<Arg>& gr_outputs = graph->outputs();

    size_t n_gr_inputs = gr_inputs.size(), n_gr_outputs = gr_outputs.size();
    std::vector<Mat> inpMats, outMats, tempMats;
#ifdef HAVE_CUDA
    static bool cudaInitFailed = false;
    bool supportGPU = (preferableBackend == DNN_BACKEND_CUDA) && haveCUDA() && !cudaInitFailed; // ignore target in new engine
    std::vector<cv::cuda::GpuMat> outGpuMats, tempGpuMats;
    if (supportGPU) {
        for (size_t opidx_check = 0; opidx_check < nops; ++opidx_check) {
            const Ptr<Layer>& layer_check = prog.at(opidx_check);
            if (!layer_check)
                continue;
            if (!layer_check->supportBackend(DNN_BACKEND_CUDA)) {
                supportGPU = false;
                break;
            }
        }
        if (!supportGPU) {
            CV_LOG_WARNING(NULL, "DNN/CUDA: full-graph CPU fallback enabled for graph '" << graph->name() << "'");
        }
    }
#endif

    std::vector<int> inpTypes, outTypes, tempTypes;
    std::vector<std::pair<uchar*, size_t> > outOrigData;
    std::vector<MatShape> inpShapes, outShapes, tempShapes;
    double tickfreq = getTickFrequency();
    int64_t timestamp = 0;

    size_t graph_ofs = (size_t)graphofs_it->second;
    CV_Assert(graph_ofs + nops <= totalLayers);
    if (!inputs_.empty()) {
        if (inputs_.total() != n_gr_inputs) {
            CV_Error_(Error::StsBadArg, ("wrong number of inputs in graph '%s': %zu given, %zu expected",
                                         graph->name().data(), inputs_.total(), n_gr_inputs));
        }
#ifdef HAVE_CUDA
        _InputArray::KindFlag inKind = inputs_.kind();
        bool usedGpuInputs = false;
        if (supportGPU && inKind == _InputArray::STD_VECTOR_CUDA_GPU_MAT) {
            std::vector<cv::cuda::GpuMat> inGpu; inputs_.getGpuMatVector(inGpu);
            CV_CheckEQ(inGpu.size(), n_gr_inputs, "Unexpected number of GPU inputs");
            if (gpuTensors.size() < args.size()) gpuTensors.resize(args.size());
            for (i = 0; i < n_gr_inputs; i++) {
                Arg inp = gr_inputs[i];
                CV_Assert(args.at(inp.idx).kind == DNN_ARG_INPUT);
                gpuTensors[inp.idx] = inGpu[i];
            }
            usedGpuInputs = true;
        }
        if (!usedGpuInputs)
#endif
        {
            for (i = 0; i < n_gr_inputs; i++) {
                Mat m = inputs_.getMat((int)i);
                setGraphInput(graph, i, m);
            }
        }
    }

    for (size_t opidx = 0; opidx < nops; opidx++) {
        const Ptr<Layer>& layer = prog.at(opidx);
        if (!layer)
            continue;
        const std::vector<Arg>& inputs = layer->inputs;
        const std::vector<Arg>& outputs = layer->outputs;
        size_t ninputs = inputs.size(), noutputs = outputs.size();

        inpMats.resize(ninputs);
        inpTypes.resize(ninputs);
        inpShapes.resize(ninputs);
        outMats.clear();
        outOrigData.clear();
#ifdef HAVE_CUDA
        outGpuMats.clear();
#endif

        for (i = 0; i < ninputs; i++) {
            Arg inp = inputs[i];
            //const ArgData& adata = args[inp.idx];
            const Mat& m = argTensor(inp);
            inpMats[i] = m;
            inpTypes[i] = m.type();
            inpShapes[i] = m.shape();
        }
#ifdef HAVE_CUDA
        if (supportGPU) {
            for (i = 0; i < ninputs; i++) {
                if (inpMats[i].empty()) {
                    Arg ainp = inputs[i];
                    const ArgData& ad = args.at(ainp.idx);
                    if (ad.type >= 0) inpTypes[i] = ad.type;
                    if (!ad.shape.empty()) inpShapes[i] = ad.shape;
                }
            }
        }
#endif

        if (tracingMode != DNN_TRACE_NONE) {
            strm_ << "-----------\n";
            strm_ << "'" << graph->name() << "' [" << opidx << "/" << nops << "]. " << layer->type << " node: " << layer->name << "\n";
            for (i = 0; i < ninputs; i++) {
                Arg inp = inputs[i];
                traceArg(strm_, "Input", i, inp, false);
            }
        }
        bool dynamicOutShapes = layer->dynamicOutputShapes();
        if (!dynamicOutShapes) {
#ifdef HAVE_CUDA
            if (supportGPU) {
                allocateLayerGpuOutputs(layer, inpTypes, inpShapes, outTypes, outShapes,
                                        tempTypes, tempShapes, outGpuMats, tempGpuMats); // only GPU mats
            } else
#endif
            {
                allocateLayerOutputs(layer, inpTypes, inpShapes, outTypes, outShapes, outOrigData, outMats,
                                 tempTypes, tempShapes, tempMats, scratchBufs, true);
            }
        } else {
            outMats.resize(noutputs);
            for (i = 0; i < noutputs; i++) {
                Arg out = outputs[i];
                outMats[i] = argTensor(out);
            }
            tempMats = scratchBufs;
        }

        timestamp = getTickCount();

#ifdef HAVE_CUDA
        bool ranOnGPU = false;
#endif

        std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
        if (!subgraphs) {
#ifdef HAVE_CUDA
            if (supportGPU && !dynamicOutShapes) {
                try {
                    std::vector<cv::cuda::GpuMat> inGpu; inGpu.reserve(ninputs);
                    for (size_t ii = 0; ii < ninputs; ++ii) {
                        Arg ainp = inputs[ii];
                        const ArgData& ainpData = args.at(ainp.idx);
                        if (ainpData.kind == DNN_ARG_TEMP) {
                            int ibuf = bufidxs.at(ainp.idx);
                            CV_Assert(ibuf >= 0);
                            ensureBufferWrapper(ibuf);
                            inGpu.push_back(gpuBuffers[(size_t)ibuf]);
                        } else {
                            if (gpuTensors.size() <= (size_t)ainp.idx) gpuTensors.resize(args.size());
                            if (gpuTensors[ainp.idx].empty()) {
                                (void)argGpuMat(ainp);
                            }
                            inGpu.push_back(gpuTensors[ainp.idx]);
                        }
                    }
                    std::vector<cv::cuda::GpuMat> tmpGpu = tempGpuMats;
                    if (!cudaInfo) {
                        if (!ensureCudaReady()) {
                            cudaInitFailed = true;
                            CV_Error(Error::StsError, "DNN/CUDA: initCUDABackend failed");
                        }
                    }
                    if (finalizeLayers) {
                        std::vector<Mat> finalizeOuts(outShapes.size());
                        for (size_t oi = 0; oi < outShapes.size(); ++oi) finalizeOuts[oi].fit(outShapes[oi], outTypes[oi]);

                        std::vector<Mat> cpuFinalizeInps; cpuFinalizeInps.resize(ninputs);
                        for (size_t ii = 0; ii < ninputs; ++ii) {
                            MatShape ish = ii < inpShapes.size() ? inpShapes[ii] : MatShape();
                            int ty = (ii < inpTypes.size() && inpTypes[ii] >= 0) ? inpTypes[ii] : CV_32F;
                            if (CV_MAT_DEPTH(ty) != CV_32F) ty = CV_MAKETYPE(CV_32F, CV_MAT_CN(ty));
                            Mat& dst = cpuFinalizeInps[ii];
                            if (!ish.empty()) dst.fit(ish, ty);
                            else dst.create(1, 1, ty);
                        }
                        layer->finalize(cpuFinalizeInps, finalizeOuts);
                    }
                    layer->forward((InputArrayOfArrays)inGpu, (OutputArrayOfArrays)outGpuMats, (OutputArrayOfArrays)tmpGpu);
                    ranOnGPU = true;
                } catch (const cv::Exception& e) {
                    CV_LOG_WARNING(NULL, "DNN/CUDA: pure GpuMat forward failed for layer '" << layer->name << "' (" << layer->type << "): " << e.what() << ". Falling back to CPU.");
                    outGpuMats.clear();
                    tempGpuMats.clear();
                    // Ensure CPU inputs are materialized from GPU if needed (download straight into N-D storage)
                    for (size_t ii = 0; ii < ninputs; ++ii) {
                        Arg ainp = inputs[ii];
                        const ArgData& ad = args.at(ainp.idx);
                        MatShape ish = ii < inpShapes.size() ? inpShapes[ii] : ad.shape;
                        Mat& host = argTensor(ainp);
                        // Choose N-D shape and destination type
                        MatShape ndShape = !ish.empty() ? ish : host.shape();
                        const cv::cuda::GpuMat* gmPtr = nullptr;
                        if (ad.kind == DNN_ARG_TEMP) {
                            int ibuf = bufidxs.at(ainp.idx);
                            if (ibuf >= 0 && (size_t)ibuf < gpuBuffers.size() && !gpuBuffers[(size_t)ibuf].empty())
                                gmPtr = &gpuBuffers[(size_t)ibuf];
                        } else {
                            if (gpuTensors.size() > (size_t)ainp.idx && !gpuTensors[ainp.idx].empty())
                                gmPtr = &gpuTensors[ainp.idx];
                        }

                        if (gmPtr) {
                            // Fallback if shape inference was missing: synthesize a 2-D shape
                            if (ndShape.empty()) {
                                int rows = gmPtr->rows > 0 ? gmPtr->rows : 1;
                                int cols = gmPtr->cols > 0 ? gmPtr->cols : 1;
                                ndShape = MatShape({ rows, cols });
                            }
                            const int dstType = gmPtr->type();
                            dnn_downloadIntoND(*gmPtr, host, ndShape, dstType);
                        }
                        inpMats[ii] = host; // host now owns the downloaded storage with correct N-D shape
                    }
                    // Prepare CPU outputs depending on shape mode
                    if (!dynamicOutShapes) {
                        allocateLayerOutputs(layer, inpTypes, inpShapes, outTypes, outShapes, outOrigData, outMats,
                                             tempTypes, tempShapes, tempMats, scratchBufs, true);
                    } else {
                        outMats.resize(noutputs);
                        for (size_t i2 = 0; i2 < noutputs; i2++) {
                            Arg out2 = outputs[i2];
                            outMats[i2] = argTensor(out2);
                        }
                        tempMats = scratchBufs;
                    }
                }
            }
#endif
        }
        else {
            Ptr<IfLayer> iflayer = layer.dynamicCast<IfLayer>();
            if (iflayer) {
                int branch = iflayer->branch(inpMats[0]);
                Ptr<Graph> subgraph = subgraphs->at(branch);
                std::vector<Mat> branchInputs;
                if (inpMats.size() > 1)
                    branchInputs.assign(inpMats.begin() + 1, inpMats.end());
                forwardGraph(subgraph, branchInputs, outMats, false);
            }
            else {
                CV_Error_(Error::StsNotImplemented,
                          ("unknown layer type '%s' with subgraphs", layer->type.c_str()));
            }
        }
#ifdef HAVE_CUDA
        if (!(supportGPU && ranOnGPU))
#endif
        {
            try {
                // If GPU attempt allocated only GPU outputs, allocate CPU outputs before CPU forward
#ifdef HAVE_CUDA
                if (!dynamicOutShapes && (supportGPU && !ranOnGPU)) {
                    if (outMats.empty()) {
                        allocateLayerOutputs(layer, inpTypes, inpShapes, outTypes, outShapes, outOrigData, outMats,
                                             tempTypes, tempShapes, tempMats, scratchBufs, true);
                    }
                }
#endif
                if (finalizeLayers) {
                    layer->finalize((InputArrayOfArrays)inpMats, (OutputArrayOfArrays)outMats);
                }
                layer->forward((InputArrayOfArrays)inpMats, (OutputArrayOfArrays)outMats, (OutputArrayOfArrays)tempMats);
                // Safety: ensure Reshape outputs are materialized even if the layer didn't copy
                if (layer->type == "Reshape") {
                    size_t copyCount = std::min(noutputs, ninputs);
                    for (size_t ci = 0; ci < copyCount; ++ci) {
                        const Mat& src = inpMats[ci];
                        Mat& dst = outMats[ci];
                        if (!src.empty() && !dst.empty() && dst.data != src.data) {
                            Mat reshaped = src.reshape(1, dst.dims, dst.size.p);
                            reshaped.copyTo(dst);
                        }
                    }
                }
            } catch (const cv::Exception& e) {
                CV_Error_(Error::StsError, ("DNN: CPU forward failed for layer '%s' (%s): %s", layer->name.c_str(), layer->type.c_str(), e.what()));
            }
            CV_Assert(outMats.size() == noutputs);
        }

        for (i = 0; i < noutputs; i++) {
            Arg out = outputs[i];
            ArgData& adata = args[out.idx];
#ifdef HAVE_CUDA
            if (supportGPU && ranOnGPU) {
                CV_Assert(i < outTypes.size());
                CV_Assert(i < outShapes.size());
                adata.type = outTypes[i];
                adata.shape = outShapes[i];
                if (adata.kind == DNN_ARG_TEMP) {
                    int bufidx = bufidxs.at(out.idx);
                    if ((size_t)bufidx >= gpuBuffers.size())
                        gpuBuffers.resize((size_t)bufidx + 1);
                    if ((size_t)i < outGpuMats.size())
                        gpuBuffers[(size_t)bufidx] = outGpuMats[i];
                    if ((size_t)bufidx < buffers.size())
                        buffers[(size_t)bufidx].release();
                } else {
                    if (gpuTensors.size() <= (size_t)out.idx)
                        gpuTensors.resize(args.size());
                    if ((size_t)i < outGpuMats.size())
                        gpuTensors[out.idx] = outGpuMats[i];
                    __tensors__[out.idx].release();
                }
                continue;
            }
#endif

            const Mat& m = outMats[i];
            adata.type = m.type();
            adata.shape = m.shape();
#ifdef HAVE_CUDA
            // If this op ran on CPU (i.e., !(supportGPU && ranOnGPU)), clear any stale GPU storage
            // so later code doesn't mistakenly prefer old gpuTensors/gpuBuffers.
            if (supportGPU) {
                if (adata.kind == DNN_ARG_TEMP) {
                    int bufidx_clear = bufidxs.at(out.idx);
                    if ((size_t)bufidx_clear < gpuBuffers.size()) {
                        gpuBuffers[(size_t)bufidx_clear].release();
                    }
                } else {
                    if (gpuTensors.size() > (size_t)out.idx) {
                        gpuTensors[out.idx].release();
                    }
                }
            }
#endif
            if (adata.kind == DNN_ARG_TEMP) {
                int bufidx = bufidxs.at(out.idx);
                Mat& buf = buffers.at(bufidx);
                if (!dynamicOutShapes) {
                    CV_Assert_N(buf.u == m.u,
                                buf.shape() == m.shape(),
                                buf.type() == m.type());
                } else if (!buf.u || m.u->size > buf.u->size) {
                    buf = m;
                } else {
                    buf.fit(m.shape(), m.type());
                    m.copyTo(buf);
                }
            } else {
                __tensors__.at(out.idx) = m;
            }
        }

        timestamp = getTickCount() - timestamp;
        layersTimings[opidx + graph_ofs + 1] += timestamp;

        if (tracingMode != DNN_TRACE_NONE) {
#ifdef HAVE_CUDA
            const char* device_str = (supportGPU && ranOnGPU) ? "CUDA" : "CPU";
#else
            const char* device_str = "CPU";
#endif
            strm_ << "TIME (\"" << layer->name << "\", \"" << layer->type << "\", " << device_str << "): "
                  << format("%.2fms", (double)timestamp*1000./tickfreq) << "\n";
            for (i = 0; i < noutputs; i++) {
                Arg out = outputs[i];
                traceArg(strm_, "Output", i, out, tracingMode == DNN_TRACE_ALL);
            }
        }
    }
    std::vector<Mat>& outputsVec = outputs_.getMatVecRef();
    outputsVec.resize(n_gr_outputs);
    for (i = 0; i < n_gr_outputs; i++) {
        Arg out = gr_outputs[i];
        const Mat& outm = argTensor(out);
#ifdef HAVE_CUDA
        if (supportGPU) {
            try {
                // Only download from GPU if the host tensor is empty.
                // If CPU fallback produced this tensor, outm is non-empty and should be preferred.
                if (outm.empty() &&
                    gpuTensors.size() > (size_t)out.idx &&
                    !gpuTensors[out.idx].empty()) {
                    const cv::cuda::GpuMat& gm = gpuTensors[out.idx];
                    Mat& dst = outputsVec[i];
                    const ArgData& ad = args.at(out.idx);
                    MatShape ndShape = ad.shape;
                    if (ndShape.empty()) {
                        int rows = gm.rows > 0 ? gm.rows : 1, cols = gm.cols > 0 ? gm.cols : 1;
                        ndShape = MatShape({ rows, cols });
                    }
                    dnn_downloadIntoND(gm, dst, ndShape, gm.type());
                    continue;
                }
            } catch (const cv::Exception& e) {
                CV_LOG_WARNING(NULL, "DNN/CUDA: failed to download graph output to host: " << e.what());
            }
        }
#endif
        if (isMainGraph) {
            outputsVec[i].fit(outm.shape(), outm.type());
            outm.copyTo(outputsVec[i]);
        } else {
            outputsVec[i] = outm;
        }
    }
}


void Net::Impl::updateUseCounts(const Ptr<Graph>& graph, std::vector<int>& usecounts) const
{
    if (!graph)
        return;
    const std::vector<Arg>& gr_outputs = graph->outputs();
    for (const Arg& output: gr_outputs) {
        CV_Assert(output.idx < (int)usecounts.size());
        usecounts[output.idx]++;
    }
    const std::vector<Ptr<Layer> >& prog = graph->prog();
    for (const Ptr<Layer>& layer: prog) {
        const std::vector<Arg>& inputs = layer->inputs;
        for (const Arg& input: inputs) {
            CV_Assert(input.idx < (int)usecounts.size());
            usecounts[input.idx]++;
        }
        const std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
        if (subgraphs) {
            for (const Ptr<Graph>& subgraph: *subgraphs) {
                updateUseCounts(subgraph, usecounts);
            }
        }
    }
}

void Net::Impl::useCounts(std::vector<int>& usecounts) const
{
    size_t nargs = args.size();
    usecounts.assign(nargs, 0);
    usecounts[0] = 1; // empty Arg() is always useful
    updateUseCounts(mainGraph, usecounts);
}

int Net::Impl::updateGraphOfs(const Ptr<Graph>& graph, int currofs, bool ismain)
{
    CV_Assert(currofs >= 0);
    if (ismain) {
        graphofs.clear();
        allgraphs.clear();
        layerNameToId.clear();
    }
    const std::vector<Ptr<Layer> >& prog = graph->prog();
    size_t i, nops = prog.size();
    int subgraph_ofs = currofs + (int)nops;
    std::string name = graph->name();
    graphofs.insert(std::make_pair(name, currofs));
    allgraphs.push_back(graph);
    for (i = 0; i < nops; i++) {
        const Ptr<Layer>& layer = prog[i];
        layerNameToId.insert(std::make_pair(layer->name, currofs + (int)i));
        const std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
        if (subgraphs) {
            for (const Ptr<Graph>& subgraph : *subgraphs) {
                subgraph_ofs = updateGraphOfs(subgraph, subgraph_ofs, false);
            }
        }
    }
    return subgraph_ofs;
}

bool Net::Impl::tryInferShapes(const std::vector<MatShape>& suggestedInpShapes,
                               const std::vector<MatType>& suggestedInpTypes,
                               LayerShapes& result,
                               std::vector<MatShape>& shapeCache,
                               std::vector<MatType>& typeCache) const
{
    result.in.clear();
    result.out.clear();
    result.inTypes.clear();
    result.outTypes.clear();

    CV_Assert(mainGraph);
    size_t nargs = args.size();
    shapeCache.assign(nargs, MatShape());
    typeCache.assign(nargs, -1);

    const std::vector<Arg>& inputs = mainGraph->inputs();
    const std::vector<Arg>& outputs = mainGraph->outputs();

    size_t ninputs = inputs.size();
    size_t noutputs = outputs.size();

    size_t nsuggestedShapes = suggestedInpShapes.size();
    size_t nsuggestedTypes = suggestedInpTypes.size();
    CV_Assert(nsuggestedShapes == 0 || nsuggestedShapes == ninputs ||

              // workaround, but this is not quite correct usage of the function
              (nsuggestedShapes == 1 && suggestedInpShapes[0].empty())
              );
    CV_Assert(nsuggestedTypes <= 1 || nsuggestedTypes == ninputs);
    bool dynamicInputShapes = false;

    result.in.resize(ninputs);
    result.inTypes.resize(ninputs);

    for (size_t i = 0; i < ninputs; i++) {
        Arg inp = inputs[i];
        const ArgData& adata = args.at(inp.idx);
        CV_Assert(adata.kind == DNN_ARG_INPUT);

        int type;
        MatShape shape;
        const Mat& tensor = argTensor(inp);
        if (!tensor.empty()) {
            type = tensor.type();
            shape = tensor.shape();
        } else {
            type = adata.type;
            shape = adata.shape;
        }

        if (nsuggestedTypes) {
            int suggestedType = suggestedInpTypes[i < nsuggestedTypes ? i : 0];
            if (suggestedType == -1)
                suggestedType = type;
            if (adata.type == type ||
                ((adata.type == CV_32F || adata.type == CV_16F || adata.type == CV_16BF) &&
                 (suggestedType == CV_32F || suggestedType == CV_16F || suggestedType == CV_16BF)))
                ;
            else {
                CV_Error_(Error::StsBadArg, ("mismatched type for model input '%s': %s provided, %s expected",
                                             adata.name.c_str(), typeToString(suggestedType).c_str(),
                                             typeToString(adata.type).c_str()));
            }
            type = suggestedType;
        }

        if (nsuggestedShapes) {
            MatShape suggestedShape = suggestedInpShapes[i < nsuggestedShapes ? i : 0];
            if (suggestedShape.empty()) {
                suggestedShape = shape;
            }
            // [TODO] shut up it for now;
            // too many ONNX conformance tests
            // depend on this "liberal" behaviour
            //
            // CV_Assert(suggestedShape.dims == adata.shape.dims);
            shape = suggestedShape;
        }

        typeCache[inp.idx] = type;
        shapeCache[inp.idx] = shape;

        if (shape.hasSymbols()) {
            CV_LOG_WARNING(NULL, format("the shape of model input '%s' includes symbols. Shape inference is impossible without prior calls to setInput()",
                adata.name.c_str()));
            dynamicInputShapes = true;
            shape = MatShape();
        }

        result.inTypes[i] = type;
        result.in[i] = shape;
    }

    bool inferenced = false;
    if (!dynamicInputShapes)
        inferenced = tryInferGraphShapes(mainGraph, shapeCache, typeCache);
    bool missingOutputs = false;

    result.outTypes.resize(noutputs, -1);
    result.out.resize(noutputs);

    for (size_t i = 0; i < noutputs; i++) {
        Arg out = outputs[i];
        const ArgData adata = args.at(out.idx);
        int type = typeCache.at(out.idx);
        MatShape shape = shapeCache.at(out.idx);
        if (type < 0) {
            if (!inferenced)
                type = adata.type;
            if (type < 0) {
                CV_LOG_WARNING(NULL, format("type for output '%s' was not inferred",                        adata.name.c_str()));
                missingOutputs = true;
            }
        }

        result.outTypes[i] = type;
        result.out[i] = shape;
    }

    return inferenced && !missingOutputs;
}

// [TODO]
// The current 'pure' shape inference is quite fragile, it does not handle any dynamic cases
// or even some seemingly dynamic cases.
// It would be nice maybe to some optional speculative forward() with some dummy inputs when
// straight-forward shape inference mechanism failed.
bool Net::Impl::tryInferGraphShapes(const Ptr<Graph>& graph,
                                    std::vector<MatShape>& shapeCache,
                                    std::vector<MatType>& typeCache) const
{
    if (!graph)
        return true;

    const std::vector<Ptr<Layer> >& prog = graph->prog();

    std::vector<MatShape> inpShapes, outShapes, tempShapes;
    std::vector<int> inpTypes, outTypes, tempTypes;

    for (const Ptr<Layer>& layer: prog) {
        if (!layer)
            continue;

        const std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
        if (subgraphs) {
            CV_LOG_WARNING(NULL, format("shape inference for the model with subgraphs (node %s (%s)) is not supported yet", layer->name.c_str(), layer->type.c_str()));
        }

        if (layer->dynamicOutputShapes()) {
            CV_LOG_WARNING(NULL, format("DNN/InferShape: Layer '%s' (%s) output shapes cannot be inferenced without running forward()", layer->name.c_str(), layer->type.c_str()));
            return false;
        }

        const std::vector<Arg>& inputs = layer->inputs;
        const std::vector<Arg>& outputs = layer->outputs;

        int ninputs = (int)inputs.size();
        int noutputs = (int)outputs.size();

        inpShapes.resize(ninputs);
        inpTypes.resize(ninputs);
        outShapes.clear();
        outTypes.clear();
        tempShapes.clear();
        tempTypes.clear();

        for (int i = 0; i < ninputs; i++) {
            Arg inp = inputs[i];
            const ArgData& adata = args.at(inp.idx);
            MatShape shape;
            int type;

            if (adata.kind == DNN_ARG_CONST || adata.kind == DNN_ARG_EMPTY) {
                shape = adata.shape;
                type = adata.type;

                // unnecessary, but nice to have for consistency
                shapeCache[inp.idx] = shape;
                typeCache[inp.idx] = type;
            } else {
                shape = shapeCache[inp.idx];
                type = typeCache[inp.idx];
                if (type < 0) {
                    CV_Error_(Error::StsInternal, ("input '%s' of operation '%s' (%s) does not have a proper type",
                                                   adata.name.c_str(), layer->name.c_str(), layer->type.c_str()));
                }
            }
            inpShapes[i] = shape;
            inpTypes[i] = type;
        }

        layer->getMemoryShapes(inpShapes, noutputs, outShapes, tempShapes);
        CV_Assert((int)outShapes.size() == noutputs);
        layer->getTypes(inpTypes, noutputs, (int)tempShapes.size(), outTypes, tempTypes);
        CV_Assert((int)outTypes.size() == noutputs);

        for (int i = 0; i < noutputs; i++) {
            Arg out = outputs[i];
            if (out.idx == 0)
                continue;
            shapeCache[out.idx] = outShapes[i];
            typeCache[out.idx] = outTypes[i];
        }
    }

    return true;
}

void Net::Impl::checkArgs(const std::vector<Arg>& args_) const
{
    for (const Arg& a: args_) {
        checkArg(a);
    }
}

void Net::Impl::checkArg(Arg a) const
{
    CV_Assert(a.idx >= 0);
    CV_Assert(a.idx < (int)args.size());
}

std::ostream& Net::Impl::dumpDim(std::ostream& strm, int value) const
{
    if (value >= 0) {
        strm << value;
    } else {
        size_t idx = -value;
        if (idx < dimnames_vec.size())
            strm << dimnames_vec[idx];
        else
            strm << "sym(" << idx << ")";
    }
    return strm;
}

std::ostream& Net::Impl::dumpTypeShape(std::ostream& strm, int type, const MatShape& shape) const
{
    if (shape.empty()) {
        strm << "<empty>";
    } else {
        strm << typeToString(type);
        if (shape.dims > 0 && shape.layout != DATA_LAYOUT_UNKNOWN) {
            strm << " " << layoutToString(shape.layout);
        }
        strm << " [";
        for (int i = 0; i < shape.dims; i++) {
            strm << (i > 0 ? " x " : "");
            dumpDim(strm, shape[i]);
        }
        strm << "]";
    }
    return strm;
}

std::ostream& Net::Impl::dumpArg(std::ostream& strm, Arg arg, int indent,
                                 bool comma, bool dump_details) const
{
    checkArg(arg);
    const ArgData& adata = args.at(arg.idx);
    prindent(strm, indent);
    if (arg.empty()) {
        strm << "<empty>" << (comma ? "," : "");
    } else {
        strm << '\"' << adata.name << (comma ? "\"," : "\"");
        if (dump_details && arg.idx > 0) {
            strm << " // ";
            strm << (adata.kind == DNN_ARG_INPUT ? "<Input>" :
                     adata.kind == DNN_ARG_OUTPUT ? "<Output>" :
                     adata.kind == DNN_ARG_CONST ? "<Const>" :
                     adata.kind == DNN_ARG_TEMP ? "<Temp>" :
                     "<Uknown kind>");
            if (adata.type >= 0) {
                strm << " ";
                dumpTypeShape(strm, adata.type, adata.shape);
            }
            if (adata.kind == DNN_ARG_TEMP && ((size_t)arg.idx < bufidxs.size()))
                strm << " (buf #" << bufidxs[arg.idx] << ")";
        }
    }
    strm << "\n";
    return strm;
}

std::ostream& Net::Impl::dump(std::ostream& strm)
{
    int indent = dump_indent;
    strm << "{\n";
    prindent(strm, indent);
    strm << "model_format: \"" << modelFormatToString(modelFormat) << "\",\n";
    prindent(strm, indent);
    strm << "layout: \"" << layoutToString(originalLayout) << "\",\n";
    if (mainGraph) {
        prindent(strm, indent);
        strm << "main_graph: ";
        mainGraph->dump(strm, indent, false);
    }
    strm << "}\n";
    return strm;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
