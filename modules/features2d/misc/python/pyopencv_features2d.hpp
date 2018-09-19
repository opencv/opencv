#ifdef HAVE_OPENCV_FEATURES2D
typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;
typedef AKAZE::DescriptorType AKAZE_DescriptorType;

CV_PY_FROM_ENUM(AKAZE::DescriptorType);
CV_PY_TO_ENUM(AKAZE::DescriptorType);
#endif