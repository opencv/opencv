// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_CORE_SRC_UMATRIX_HPP
#define OPENCV_CORE_SRC_UMATRIX_HPP

namespace cv {

class MatAllocator;

struct CV_EXPORTS UMatDataAutoLock
{
    explicit UMatDataAutoLock(UMatData* u);
    UMatDataAutoLock(UMatData* u1, UMatData* u2);
    ~UMatDataAutoLock();
    UMatData* u1;
    UMatData* u2;
};

namespace umat {

bool useMetal();
bool useOpenCL();
MatAllocator* getAcceleratedAllocator();
bool haveAcceleratedAllocator();
bool isAcceleratedAllocator(const MatAllocator* allocator);

} // namespace umat

}

#endif // OPENCV_CORE_SRC_UMATRIX_HPP
