#ifndef OPENCV_FEATURES2D_DISK_HPP
#define OPENCV_FEATURES2D_DISK_HPP

#include "opencv2/features2d.hpp"

namespace cv {

/** @brief Class implementing the DISK (Deep Image Structure and Keypoints) feature detector

Wrapping the inference of the ONNX model provided by LightGlue-ONNX.
*/
class CV_EXPORTS_W DISK : public Feature2D
{
public:
    /**
      @param modelPath Path to the ONNX model file (e.g. disk_standalone.onnx).
      @param useGPU If true, attempts to use CUDA backend.
     */
    CV_WRAP static Ptr<DISK> create(const String& modelPath, bool useGPU = false);

    CV_WRAP virtual String getDefaultName() const CV_OVERRIDE;
};

} // namespace cv

#endif // OPENCV_FEATURES2D_DISK_HPP