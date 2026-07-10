// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include "reduce.simd.hpp"
#include "reduce.simd_declarations.hpp"

namespace cv {

typedef void (*ReduceSumFunc)(const Mat& src, Mat& dst);
ReduceSumFunc getReduceCSumFunc(int sdepth, int ddepth);
ReduceSumFunc getReduceCAvgFunc(int sdepth, int ddepth);
ReduceSumFunc getReduceCMaxFunc(int sdepth, int ddepth);
ReduceSumFunc getReduceCMinFunc(int sdepth, int ddepth);
ReduceSumFunc getReduceCSum2Func(int sdepth, int ddepth);
ReduceSumFunc getReduceRSumFunc(int sdepth, int ddepth);

ReduceSumFunc getReduceCSumFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCSumFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceSumFunc getReduceCAvgFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCAvgFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceSumFunc getReduceCMaxFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCMaxFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceSumFunc getReduceCMinFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCMinFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceSumFunc getReduceCSum2Func(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCSum2Func, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceSumFunc getReduceRSumFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceRSumFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
