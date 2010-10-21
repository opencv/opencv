#ifndef _LSVM_RESIZEIMG_H_
#define _LSVM_RESIZEIMG_H_

#include "_lsvm_types.h"

IplImage * resize_opencv (IplImage * img, float scale);
IplImage * resize_article_dp1(IplImage * img, float scale, const int k);
IplImage * resize_article_dp(IplImage * img, float scale, const int k);

#endif
