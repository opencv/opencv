// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_ENGINE_HPP__
#define __OPENCV_DNN_ENGINE_HPP__

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

enum { DNN_ARG_CONST=0, DNN_ARG_INPUT=1, DNN_ARG_OUTPUT=2, DNN_ARG_TEMP=3 };
enum { DNN_BUF_READONLY=1, DNN_BUF_WRITEONLY=2, DNN_BUF_RW=3 };

size_t totalBytes(const TensorShape& shape, int typ);

struct LayerArg
{
    LayerArg();
    std::string name;
    int kind;
    TensorShape shape;
    int typ;
};

struct MemoryManager;

struct Device
{
    enum { DEV_CPU=0, DEV_IGPU=1, DEV_DGPU=2, DEV_NPU=3 };
    virtual ~Device();
    virtual int kind() const = 0;
    virtual std::string name() const = 0;
    virtual bool supportType(int typ) const = 0;
    virtual bool zeroCopy() const = 0;
    virtual MemoryManager* defaultMemoryManager() = 0;
};

Device* getCPUDevice();

struct MemoryManager
{
    virtual ~MemoryManager();
    virtual void* allocate(Device* device, size_t bufsize) = 0;
    virtual void release(Device* device, void* handle) = 0;
    virtual void* map(Device* device, void* handle, size_t size, int access=DNN_BUF_RW) = 0;
    virtual void unmap(Device* device, void* handle, void* ptr, size_t size, int access=DNN_BUF_RW) = 0;
    virtual void copyFromDevice(Device* device, void* handle, size_t offset, size_t size, void* dst) = 0;
    virtual void copyToDevice(Device* device, const void* src, void* handle, size_t offset, size_t size) = 0;
};

struct Buffer
{
    struct Shared
    {
        Shared();
        void* ptr;
        int refcount;
        int mapcount;
    };
    Buffer();
    Buffer(const Buffer& buf);
    Buffer(const void* data, size_t size, bool copy);
    Buffer& operator = (const Buffer& buf);
    ~Buffer();

    static Buffer allocate(size_t size, MemoryManager* mm=0, Device* device=0);
    void fit(size_t size);
    void set(const void* data, size_t size, bool copy);
    void release();
    void* map(int access=DNN_BUF_RW);
    void unmap(int access=DNN_BUF_RW);

    Device* device;
    MemoryManager* mm;
    Shared* shared;
    void* handle;
    size_t size;
};

// temporary solution while Mat cannot be 0-D or 1-D array.
struct Tensor
{
    Tensor();
    Tensor(const Tensor& t);
    Tensor& operator = (const Tensor& t);
    ~Tensor();

    explicit Tensor(const TensorShape& shape, int typ);
    explicit Tensor(const TensorShape& shape, int typ, void* data, bool copy);
    explicit Tensor(InputArray arr, int ndims, bool copy);

    void release();
    void fit(const TensorShape& shape, int typ);
    void set(const TensorShape& shape, int typ, void* data, bool copy);
    void set(InputArray arr, int ndims, bool copy);
    size_t total() const;
    bool empty() const;
    void* data() const;
    Mat getMat();
    void* map(int access=DNN_BUF_RW);
    void unmap(int access=DNN_BUF_RW);

    TensorShape shape;
    int typ;
    Buffer buf;
};

typedef Ptr<Layer> PLayer;
struct Graph;
typedef Ptr<Graph> PGraph;

void fitMat(Mat& m, size_t size);
void dump(const Tensor& t, int border=3, int maxsz_all=100, bool braces=true);

struct Node
{
    Node() {}
    Node(const PLayer& op_, const std::vector<int>& inputs_,
         const std::vector<int>& outputs_,
         const std::vector<PGraph>& subgraphs_=std::vector<PGraph>())
        : op(op_), inputs(inputs_),
        outputs(outputs_), subgraphs(subgraphs_) {}
    PLayer op;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<PGraph> subgraphs;
};

struct Graph
{
    bool empty() const;
    void clear();
    std::string name;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<Node> prog;
};

typedef std::pair<int64_t, std::string> OnnxOpSet;

struct TensorDim
{
    TensorDim() : value(-1) {}
    explicit TensorDim(const std::string& p) : param(p), value(-1) {}
    explicit TensorDim(const char* p) : param(p), value(-1) {}
    explicit TensorDim(int64_t v) : param(), value(v) {}
    bool empty() { return param.empty() && value <= 0; }
    std::string param;
    int64_t value;
};

struct ArgInfo
{
    ArgInfo() : typ(-1) {}
    std::string name;
    int typ;
    std::vector<TensorDim> shape;
};

struct OnnxInfo
{
    int64_t IRVersion;
    std::string producer;
    std::string domain;
    std::string docString;
    std::vector<OnnxOpSet> opsets;
};

enum {
    DNN_MODEL_GENERIC = 0,
    DNN_MODEL_ONNX = 1,
    DNN_MODEL_TF = 2
};

typedef std::unordered_map<std::string, int> NamesHash;

struct Net2::Impl
{
    Impl();
    ~Impl();

    void clear();
    void forward(InputArrayOfArrays inputBlobs, OutputArrayOfArrays outputBlobs);
    void forwardGraph(const Graph& graph);
    void useCounts(std::vector<int>& usecounts) const;
    void updateUseCounts(std::vector<int>& usecounts, const Graph& graph) const;
    void assignBuffers();
    int addConstTensor(const std::string& name, const Tensor& t, int idx=-1);
    int addArg(int argkind, const ArgInfo& arginfo);
    int64_t findDim(const std::string& dimname);
    int findArg(const std::string& argname);
    int findOutputArg(const std::string& argname);
    bool isConst(int argidx) const;
    int kind(int argidx) const;
    bool empty() const;
    void dump() const;
    void dumpGraph(const Graph& graph,
                   const std::string& indent,
                   bool comma) const;
    void dumpAttrValue(const DictValue& value) const;
    void dumpNode(const Layer& layer,
                  const std::vector<int>& inpargs,
                  const std::vector<int>& outargs,
                  const std::vector<PGraph>& subgraphs,
                  const std::string& indent,
                  bool comma) const;
    void dumpArg(const String& prefix, int i,
                 int argidx, bool dumpdata) const;
    void dumpArgInfo(int argidx, const std::string& indent, bool comma) const;
    bool useFP16() const;
    void set(int propId, double value);
    double get(int propId) const;

    Net2* net;
    int modelFormat;
    OnnxInfo onnxInfo;

    NamesHash argnames;
    NamesHash dimnames;
    std::vector<std::string> dimnames_;
    std::vector<LayerArg> args;
    std::vector<Tensor> tensors;
    std::vector<int> bufidxs;
    std::vector<Buffer> buffers;
    Graph graph;
    DataLayout defaultLayout;
    bool enableFP16;
    bool haveFP16;
    bool trace;
    bool profile;
    bool traceProfile;

    Buffer scratchBuf;
    std::vector<int64_t> perfProfileTime;
    std::vector<int> perfProfileCount;
    std::string delta_indent = "   ";

    Device* defaultDevice;
    MemoryManager* defaultMemoryManager;
};

CV__DNN_INLINE_NS_END
}}

#endif
