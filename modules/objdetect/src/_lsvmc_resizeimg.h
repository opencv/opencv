#ifndef _LSVM_RESIZEIMG_H_
#define _LSVM_RESIZEIMG_H_

#include "_lsvmc_types.h"

namespace cv
{
namespace lsvmcascade
{

IplImage * resize_opencv (IplImage * img, float scale);
}
}

#endif
