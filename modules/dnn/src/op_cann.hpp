// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_OP_CANN_HPP
#define OPENCV_DNN_OP_CANN_HPP

#ifdef HAVE_CANN
#include "acl/acl.h" // acl* functions
#include "graph/graph.h" // ge::Graph; ge::Operator from operator.h
#include "graph/ge_error_codes.h" // GRAPH_SUCCESS, ...

/* We dont need these if we do not build graph with initCann
#include "all_ops.h" // ge::Conv2D, ...
#include "graph/tensor.h" // ge::Shape, ge::Tensor, ge::TensorDesc
#include "graph/types.h" // DT_FLOAT, ... ; FORMAT_NCHW, ...
*/

#include "ge/ge_api_types.h" // ge::ir_option::SOC_VERSION
#include "ge/ge_ir_build.h" // build graph

// parser
#include "parser/onnx_parser.h" // aclgrphParseONNX
#include "parser/tensorflow_parser.h" // aclgrphParseTensorflow
#include "parser/caffe_parser.h" // aclgrphParseCaffe

#endif // HAVE_CANN

#include <vector>

#ifdef HAVE_CANN
#define ACL_CHECK_RET(f) \
{ \
    if (f != ACL_SUCCESS) \
    { \
        CV_LOG_ERROR(NULL, "CANN check failed, ret = " << f); \
        CV_Error(Error::StsError, "CANN check failed"); \
    } \
}
#define ACL_CHECK_GRAPH_RET(f) \
{ \
    if (f != ge::GRAPH_SUCCESS) \
    { \
        CV_LOG_ERROR(NULL, "CANN graph check failed, ret = " << f); \
        CV_Error(Error::StsError, "CANN graph check failed"); \
    } \
}

#endif

namespace cv { namespace dnn {

#ifdef HAVE_CANN

CV__DNN_INLINE_NS_BEGIN

void switchToCannBackend(Net& net);

CV__DNN_INLINE_NS_END

    // void initAcl();
    void initAclGraphBuilder();

    class AclEnvGuard {
    public:
        explicit AclEnvGuard();
        ~AclEnvGuard();
        static std::shared_ptr<AclEnvGuard> GetAclEnv();

    private:
        static std::shared_ptr<AclEnvGuard> global_acl_env_;
        static std::mutex global_acl_env_mutex_;
    };

    struct CannGraph
    {
        explicit CannGraph() { graph = ge::Graph("graph"); }

        ge::Graph graph;

        bool loadFromONNX(const String& modelFile);
        bool loadFromONNXFromMem(const char* buffer, size_t length);
        bool loadFromTensorFlow(const String& modelFile);
        bool loadFromCaffe(const String& modelFile, const String& weightFile);
    };

    class CannClient
    {
    public:
        explicit CannClient(int deviceId = 0) { init(deviceId); acl_env = AclEnvGuard::GetAclEnv(); }
        ~CannClient(); // release private members
        void init(int deviceId);
        static void finalize();

        bool empty() const;

        void buildModelFromGraph(ge::Graph& graph);
        void loadModel(); // call aclInit before this API is called

        void setInput(const Mat& input, const String& name = String());
        void forward();
        void fetchOutput(Mat& output, const String& name);
        void fetchOutput(Mat& output, const size_t idx);

        size_t getOutputNum() const;

    private:
        void createInputDataset();
        void createOutputDataset();

        void destroyDataset(aclmdlDataset** dataset);

        std::shared_ptr<AclEnvGuard> acl_env;

        uint32_t model_id{0};
        aclmdlDesc* model_desc{nullptr};
        std::vector<uint8_t> model;
        aclmdlDataset* inputs{nullptr};
        aclmdlDataset* outputs{nullptr};

        int device_id{0};
        aclrtContext context{nullptr};
        // aclrtStream stream{nullptr};
    };

#endif // HAVE_CANN

}} // namespace cv::dnn

#endif // OPENCV_DNN_OP_CANN_HPP
