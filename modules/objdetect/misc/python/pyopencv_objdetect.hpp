#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

CV_PY_FROM_ENUM(HOGDescriptor::HistogramNormType);
CV_PY_TO_ENUM(HOGDescriptor::HistogramNormType);
CV_PY_FROM_ENUM(HOGDescriptor::DescriptorStorageFormat);
CV_PY_TO_ENUM(HOGDescriptor::DescriptorStorageFormat);

#endif
