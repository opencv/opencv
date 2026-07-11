// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include "reduce.simd.hpp"
#include "reduce.simd_declarations.hpp"

namespace cv {

typedef void (*ReduceFunc)( const Mat& src, Mat& dst );
ReduceFunc getReduceCSumFunc(int sdepth, int ddepth);
ReduceFunc getReduceCAvgFunc(int sdepth, int ddepth);
ReduceFunc getReduceCMaxFunc(int sdepth, int ddepth);
ReduceFunc getReduceCMinFunc(int sdepth, int ddepth);
ReduceFunc getReduceCSum2Func(int sdepth, int ddepth);
ReduceFunc getReduceRSumFunc(int sdepth, int ddepth);

ReduceFunc getReduceCSumFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCSumFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceFunc getReduceCAvgFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCAvgFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceFunc getReduceCMaxFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCMaxFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceFunc getReduceCMinFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCMinFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceFunc getReduceCSum2Func(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceCSum2Func, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

ReduceFunc getReduceRSumFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getReduceRSumFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
