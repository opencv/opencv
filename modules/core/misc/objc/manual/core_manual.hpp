#pragma once

#include "opencv2/core.hpp"

#ifdef OPENCV_BINDINGS_PARSER

namespace cv
{
CV_EXPORTS_W void add(InputArray src1, Scalar srcScalar, OutputArray dst, InputArray mask=noArray(), int dtype=-1);

CV_EXPORTS_W void subtract(InputArray src1, Scalar srcScalar, OutputArray dst, InputArray mask=noArray(), int dtype=-1);

CV_EXPORTS_W void multiply(InputArray src1, Scalar srcScalar, OutputArray dst, double scale=1, int dtype=-1);

CV_EXPORTS_W void divide(InputArray src1, Scalar srcScalar, OutputArray dst, double scale=1, int dtype=-1);

CV_EXPORTS_W void absdiff(InputArray src1, Scalar srcScalar, OutputArray dst);

CV_EXPORTS_W void compare(InputArray src1, Scalar srcScalar, OutputArray dst, int cmpop);

CV_EXPORTS_W void min(InputArray src1, Scalar srcScalar, OutputArray dst);

CV_EXPORTS_W void max(InputArray src1, Scalar srcScalar, OutputArray dst);

}
#endif
