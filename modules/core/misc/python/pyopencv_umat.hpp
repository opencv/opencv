#ifdef HAVE_OPENCV_CORE

#include "opencv2/core/mat.hpp"

typedef std::vector<Range> vector_Range;

CV_PY_TO_CLASS(UMat);
CV_PY_FROM_CLASS(UMat);

static bool cv_mappable_to(const Ptr<Mat>& src, Ptr<UMat>& dst)
{
    //dst.reset(new UMat(src->getUMat(ACCESS_RW)));
    dst.reset(new UMat());
    src->copyTo(*dst);
    return true;
}

static void* cv_UMat_queue()
{
    return cv::ocl::Queue::getDefault().ptr();
}

static void* cv_UMat_context()
{
    return cv::ocl::Context::getDefault().ptr();
}

static Mat cv_UMat_get(const UMat* _self)
{
    Mat m;
    m.allocator = &g_numpyAllocator;
    _self->copyTo(m);
    return m;
}

#endif
