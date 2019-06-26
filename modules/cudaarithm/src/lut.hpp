// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __CUDAARITHM_LUT_HPP__
#define __CUDAARITHM_LUT_HPP__

#include "opencv2/cudaarithm.hpp"

#include <cuda_runtime.h>

namespace cv { namespace cuda {

class LookUpTableImpl : public LookUpTable
{
public:
    LookUpTableImpl(InputArray lut);
    ~LookUpTableImpl();

    void transform(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) CV_OVERRIDE;

private:
    GpuMat d_lut;
    cudaTextureObject_t texLutTableObj;
    bool cc30;
};

} }

#endif // __CUDAARITHM_LUT_HPP__
