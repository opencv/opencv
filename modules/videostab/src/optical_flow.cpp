#include "precomp.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/videostab/optical_flow.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

void SparsePyrLkOptFlowEstimator::run(
        InputArray frame0, InputArray frame1, InputArray points0, InputOutputArray points1,
        OutputArray status, OutputArray errors)
{
    calcOpticalFlowPyrLK(frame0, frame1, points0, points1, status, errors, winSize_, maxLevel_);
}


DensePyrLkOptFlowEstimatorGpu::DensePyrLkOptFlowEstimatorGpu()
{
    CV_Assert(gpu::getCudaEnabledDeviceCount() > 0);
}


void DensePyrLkOptFlowEstimatorGpu::run(
        InputArray frame0, InputArray frame1, InputOutputArray flowX, InputOutputArray flowY,
        OutputArray errors)
{
    frame0_.upload(frame0.getMat());
    frame1_.upload(frame1.getMat());

    optFlowEstimator_.winSize = winSize_;
    optFlowEstimator_.maxLevel = maxLevel_;
    if (errors.needed())
    {
        optFlowEstimator_.dense(frame0_, frame1_, flowX_, flowY_, &errors_);
        errors_.download(errors.getMatRef());
    }
    else
        optFlowEstimator_.dense(frame0_, frame1_, flowX_, flowY_);

    flowX_.download(flowX.getMatRef());
    flowY_.download(flowY.getMatRef());
}


} // namespace videostab
} // namespace cv
