// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<LookUpTable> cv::cuda::createLookUpTable(InputArray) { throw_no_cuda(); return Ptr<LookUpTable>(); }

#else /* !defined (HAVE_CUDA) || defined (CUDA_DISABLER) */

// lut.hpp includes cuda_runtime.h and can only be included when we have CUDA
#include "lut.hpp"

Ptr<LookUpTable> cv::cuda::createLookUpTable(InputArray lut)
{
    return makePtr<LookUpTableImpl>(lut);
}

#endif
