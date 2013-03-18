#include "perf_precomp.hpp"

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::superres;
using namespace cv::gpu;

namespace
{
    class OneFrameSource_CPU : public FrameSource
    {
    public:
        explicit OneFrameSource_CPU(const Mat& frame) : frame_(frame) {}

        void nextFrame(OutputArray frame)
        {
            frame.getMatRef() = frame_;
        }

        void reset()
        {
        }

    private:
        Mat frame_;
    };

    class OneFrameSource_GPU : public FrameSource
    {
    public:
        explicit OneFrameSource_GPU(const GpuMat& frame) : frame_(frame) {}

        void nextFrame(OutputArray frame)
        {
            frame.getGpuMatRef() = frame_;
        }

        void reset()
        {
        }

    private:
        GpuMat frame_;
    };

    class ZeroOpticalFlow : public DenseOpticalFlowExt
    {
    public:
        void calc(InputArray frame0, InputArray, OutputArray flow1, OutputArray flow2)
        {
            cv::Size size = frame0.size();

            if (!flow2.needed())
            {
                flow1.create(size, CV_32FC2);

                if (flow1.kind() == cv::_InputArray::GPU_MAT)
                    flow1.getGpuMatRef().setTo(cv::Scalar::all(0));
                else
                    flow1.getMatRef().setTo(cv::Scalar::all(0));
            }
            else
            {
                flow1.create(size, CV_32FC1);
                flow2.create(size, CV_32FC1);

                if (flow1.kind() == cv::_InputArray::GPU_MAT)
                    flow1.getGpuMatRef().setTo(cv::Scalar::all(0));
                else
                    flow1.getMatRef().setTo(cv::Scalar::all(0));

                if (flow2.kind() == cv::_InputArray::GPU_MAT)
                    flow2.getGpuMatRef().setTo(cv::Scalar::all(0));
                else
                    flow2.getMatRef().setTo(cv::Scalar::all(0));
            }
        }

        void collectGarbage()
        {
        }
    };
}

PERF_TEST_P(Size_MatType, SuperResolution_BTVL1,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_8UC1), MatType(CV_8UC3))))
{
    declare.time(5 * 60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    declare.in(frame, WARMUP_RNG);

    const int scale = 2;
    const int iterations = 50;
    const int temporalAreaRadius = 1;
    Ptr<DenseOpticalFlowExt> opticalFlow(new ZeroOpticalFlow);

    if (PERF_RUN_GPU())
    {
        Ptr<SuperResolution> superRes = createSuperResolution_BTVL1_GPU();

        superRes->set("scale", scale);
        superRes->set("iterations", iterations);
        superRes->set("temporalAreaRadius", temporalAreaRadius);
        superRes->set("opticalFlow", opticalFlow);

        superRes->setInput(new OneFrameSource_GPU(GpuMat(frame)));

        GpuMat dst;
        superRes->nextFrame(dst);

        TEST_CYCLE_N(10) superRes->nextFrame(dst);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        Ptr<SuperResolution> superRes = createSuperResolution_BTVL1();

        superRes->set("scale", scale);
        superRes->set("iterations", iterations);
        superRes->set("temporalAreaRadius", temporalAreaRadius);
        superRes->set("opticalFlow", opticalFlow);

        superRes->setInput(new OneFrameSource_CPU(frame));

        Mat dst;
        superRes->nextFrame(dst);

        TEST_CYCLE_N(10) superRes->nextFrame(dst);

        CPU_SANITY_CHECK(dst);
    }
}
