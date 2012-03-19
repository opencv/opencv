#ifndef __OPENCV_VIDEOSTAB_OPTICAL_FLOW_HPP__
#define __OPENCV_VIDEOSTAB_OPTICAL_FLOW_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace cv
{
namespace videostab
{

class ISparseOptFlowEstimator
{
public:
    virtual ~ISparseOptFlowEstimator() {}
    virtual void run(
            InputArray frame0, InputArray frame1, InputArray points0, InputOutputArray points1,
            OutputArray status, OutputArray errors) = 0;
};

class IDenseOptFlowEstimator
{
public:
    virtual ~IDenseOptFlowEstimator() {}
    virtual void run(
            InputArray frame0, InputArray frame1, InputOutputArray flowX, InputOutputArray flowY,
            OutputArray errors) = 0;
};

class PyrLkOptFlowEstimatorBase
{
public:
    PyrLkOptFlowEstimatorBase() { setWinSize(Size(21, 21)); setMaxLevel(3); }

    void setWinSize(Size val) { winSize_ = val; }
    Size winSize() const { return winSize_; }

    void setMaxLevel(int val) { maxLevel_ = val; }
    int maxLevel() const { return maxLevel_; }

protected:
    Size winSize_;
    int maxLevel_;
};

class SparsePyrLkOptFlowEstimator
        : public PyrLkOptFlowEstimatorBase, public ISparseOptFlowEstimator
{
public:
    virtual void run(
            InputArray frame0, InputArray frame1, InputArray points0, InputOutputArray points1,
            OutputArray status, OutputArray errors);
};

class DensePyrLkOptFlowEstimatorGpu
        : public PyrLkOptFlowEstimatorBase, public IDenseOptFlowEstimator
{
public:
    DensePyrLkOptFlowEstimatorGpu();

    virtual void run(
            InputArray frame0, InputArray frame1, InputOutputArray flowX, InputOutputArray flowY,
            OutputArray errors);
private:
    gpu::PyrLKOpticalFlow optFlowEstimator_;
    gpu::GpuMat frame0_, frame1_, flowX_, flowY_, errors_;
};

} // namespace videostab
} // namespace cv

#endif
