#pragma once

#include "opencv2/core.hpp"

namespace cv
{

CV_EXPORTS_W void setErrorVerbosity(bool verbose);

}

#if 0

namespace cv
{
CV_EXPORTS_W void add(InputArray src1, Scalar src2, OutputArray dst, InputArray mask=noArray(), ElemDepth ddepth = CV_DEPTH_AUTO);

CV_EXPORTS_W void subtract(InputArray src1, Scalar src2, OutputArray dst, InputArray mask=noArray(), ElemDepth ddepth = CV_DEPTH_AUTO);

CV_EXPORTS_W void multiply(InputArray src1, Scalar src2, OutputArray dst, double scale=1, ElemDepth ddepth = CV_DEPTH_AUTO);

CV_EXPORTS_W void divide(InputArray src1, Scalar src2, OutputArray dst, double scale=1, ElemDepth ddepth = CV_DEPTH_AUTO);

CV_EXPORTS_W void absdiff(InputArray src1, Scalar src2, OutputArray dst);

CV_EXPORTS_W void compare(InputArray src1, Scalar src2, OutputArray dst, int cmpop);

CV_EXPORTS_W void min(InputArray src1, Scalar src2, OutputArray dst);

CV_EXPORTS_W void max(InputArray src1, Scalar src2, OutputArray dst);

}
#endif //0
