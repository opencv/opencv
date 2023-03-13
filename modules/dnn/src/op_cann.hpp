// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_OP_CANN_HPP
#define OPENCV_DNN_OP_CANN_HPP

#ifdef HAVE_CANN
#include "acl/acl.h" // acl* functions
#include "graph/graph.h" // ge::Graph; ge::Operator from operator.h
#include "graph/ge_error_codes.h" // GRAPH_SUCCESS, ...

#include "op_proto/built-in/inc/all_ops.h" // ge::Conv2D, ...
#include "graph/tensor.h" // ge::Shape, ge::Tensor, ge::TensorDesc
#include "graph/types.h" // DT_FLOAT, ... ; FORMAT_NCHW, ...

#include "ge/ge_api_types.h" // ge::ir_option::SOC_VERSION
#include "ge/ge_ir_build.h" // build graph

// for fork()
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

    class CannNet;

    class AclEnvGuard {
    public:
        explicit AclEnvGuard();
        ~AclEnvGuard();
        static std::shared_ptr<AclEnvGuard> GetAclEnv();

    private:
        static std::shared_ptr<AclEnvGuard> global_acl_env_;
        static std::mutex global_acl_env_mutex_;
    };

    class CannConstOp
    {
    public:
        CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);
        std::shared_ptr<ge::op::Const> getOp() { return op_; }
        std::shared_ptr<ge::TensorDesc> getTensorDesc() { return desc_; }
    private:
        std::shared_ptr<ge::op::Const> op_;
        std::shared_ptr<ge::TensorDesc> desc_;
    };

    class CannBackendNode : public BackendNode
    {
    public:
        CannBackendNode(const std::shared_ptr<ge::Operator>& op);
        std::shared_ptr<ge::Operator> getOp();
        std::shared_ptr<CannNet> net;
    private:
        std::shared_ptr<ge::Operator> op_;
    };

    class CannBackendWrapper : public BackendWrapper
    {
    public:
        CannBackendWrapper(const Mat& m);
        ~CannBackendWrapper() { }

        std::shared_ptr<ge::TensorDesc> getTensorDesc() { return desc_; }

        virtual void copyToHost() CV_OVERRIDE;

        virtual void setHostDirty() CV_OVERRIDE;

        Mat* host;
        std::shared_ptr<ge::TensorDesc> desc_;
        std::string name;
    };

    class CannNet
    {
    public:
        explicit CannNet(int deviceId = 0)
            : device_id(deviceId)
        {
            init();
            acl_env = AclEnvGuard::GetAclEnv();
        }
        ~CannNet(); // release private members

        bool empty() const;

        void loadModelBuffer(std::shared_ptr<ge::ModelBufferData> modelBuffer);

        void bindInputWrappers(const std::vector<Ptr<BackendWrapper>>& inputWrappers);
        void bindOutputWrappers(const std::vector<Ptr<BackendWrapper>>& outputWrappers);

        void forward();

        size_t getInputNum() const;
        size_t getOutputNum() const;

    private:
        void init();

        void loadToDevice(); // call aclInit before this API is called
        void createInputDataset();
        void createOutputDataset();

        int getOutputIndexByName(const std::string& name);

        void destroyDataset(aclmdlDataset** dataset);

        std::shared_ptr<AclEnvGuard> acl_env;

        std::vector<Ptr<CannBackendWrapper>> input_wrappers;
        std::vector<Ptr<CannBackendWrapper>> output_wrappers;

        uint32_t model_id{0};
        aclmdlDesc* model_desc{nullptr};
        std::vector<uint8_t> model;
        aclmdlDataset* inputs{nullptr};
        aclmdlDataset* outputs{nullptr};

        int device_id{0};
        aclrtContext context{nullptr};
    };

#endif // HAVE_CANN

}} // namespace cv::dnn

#endif // OPENCV_DNN_OP_CANN_HPP
