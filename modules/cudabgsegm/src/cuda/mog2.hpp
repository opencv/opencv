// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CUDA_MOG2_H
#define OPENCV_CUDA_MOG2_H

#include "opencv2/core/cuda.hpp"

struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

namespace cv { namespace cuda {

class Stream;

namespace device { namespace mog2 {

typedef struct
{
    float Tb_;
    float TB_;
    float Tg_;
    float varInit_;
    float varMin_;
    float varMax_;
    float tau_;
    int nmixtures_;
    unsigned char shadowVal_;
} Constants;

void mog2_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzf variance, PtrStepSzb mean, float alphaT, float prune, bool detectShadows, const Constants *const constants, cudaStream_t stream);
void getBackgroundImage2_gpu(int cn, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, const Constants *const constants, cudaStream_t stream);

} } } }

#endif /* OPENCV_CUDA_MOG2_H */
