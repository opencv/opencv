#ifndef OPENCV_CUDA_MOG2_H
#define OPENCV_CUDA_MOG2_H

#include "opencv2/core/cuda.hpp"

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

} } } }

#endif /* OPENCV_CUDA_MOG2_H */
