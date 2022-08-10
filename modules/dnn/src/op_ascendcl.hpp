// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_DNN_OP_ASCENDCL_HPP
#define OPENCV_DNN_OP_ASCENDCL_HPP

#include <opencv2/dnn/shape_utils.hpp>
#ifdef HAVE_ASCENDCL
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "ascendcl/include/ascendcl.hpp"
#endif // HAVE_ASCENDCL
#include <vector>

namespace cv
{
namespace dnn
{
#ifdef HAVE_ASCENDCL

    void copyToTensor(ascendcl::Tensor& dst, const Mat& src);
    void copyToMat(Mat& dst, ascendcl::Tensor& src);

    class CannInfo
    {
    public:
        CannInfo() { }
        void init(int deviceId = 0);
        ~CannInfo();
        aclrtStream getStream() const;
        void syncStream();
    private:
        int device_id{0};
        aclrtContext context{nullptr};
        aclrtStream stream{nullptr};
    };

    class AscendCLBackendNode : public BackendNode
    {
    public:
        AscendCLBackendNode(const aclrtStream stream,
                            const std::vector<Ptr<BackendWrapper> >& inputsWrapper,
                            const std::shared_ptr<ascendcl::Operator>& op,
                            const std::vector<Ptr<BackendWrapper> >& blobsWrapper = std::vector<Ptr<BackendWrapper> >());

        bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> outs);

    private:
        aclrtStream stream_;
        std::vector<Ptr<BackendWrapper> > inputs_wrapper;
        std::vector<Ptr<BackendWrapper> > blobs_wrapper;
        std::shared_ptr<ascendcl::Operator> operator_;
    };

    class AscendCLBackendWrapper : public BackendWrapper
    {
    public:
        AscendCLBackendWrapper(Mat m);
        AscendCLBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat m);

        void createTensor();

        virtual void copyToHost() CV_OVERRIDE;
        virtual void setHostDirty() CV_OVERRIDE;
        void setDeviceDirty();
        void copyToDevice();

        std::shared_ptr<ascendcl::Tensor> getTensor();
        int getShapeAt(int axis) const;

        Mat getMat() const;

    private:
        bool is_tensor;
        std::shared_ptr<ascendcl::Tensor> tensor;
        Mat host;
        bool hostDirty;
        bool deviceDirty;
    };
#endif // HAVE_ASCENDCL

    void forwardAscendCL(std::vector<Ptr<BackendWrapper> >& outputs, const Ptr<BackendNode>& node);

    bool haveAscendCL();

} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_OP_ASCENDCL_HPP
