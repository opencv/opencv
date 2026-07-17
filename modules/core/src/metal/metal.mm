// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_private.hpp"

#ifdef HAVE_METAL

namespace cv {
namespace metal {

bool haveMetal()
{
    return getMetalContext()->valid();
}

MatAllocator* getMetalAllocator()
{
    CV_SINGLETON_LAZY_INIT(MatAllocator, getMetalAllocator_())
}

} // namespace metal
} // namespace cv

#endif // HAVE_METAL
