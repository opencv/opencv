#ifdef HAVE_OPENCV_FEATURES2D
typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;
typedef AKAZE::DescriptorType AKAZE_DescriptorType;
typedef AgastFeatureDetector::DetectorType AgastFeatureDetector_DetectorType;
typedef FastFeatureDetector::DetectorType FastFeatureDetector_DetectorType;
typedef DescriptorMatcher::MatcherType DescriptorMatcher_MatcherType;
typedef KAZE::DiffusivityType KAZE_DiffusivityType;
typedef ORB::ScoreType ORB_ScoreType;

CV_PY_FROM_ENUM(AKAZE::DescriptorType);
CV_PY_TO_ENUM(AKAZE::DescriptorType);
CV_PY_FROM_ENUM(AgastFeatureDetector::DetectorType);
CV_PY_TO_ENUM(AgastFeatureDetector::DetectorType);
CV_PY_FROM_ENUM(DrawMatchesFlags);
CV_PY_TO_ENUM(DrawMatchesFlags);
CV_PY_FROM_ENUM(FastFeatureDetector::DetectorType);
CV_PY_TO_ENUM(FastFeatureDetector::DetectorType);
CV_PY_FROM_ENUM(DescriptorMatcher::MatcherType);
CV_PY_TO_ENUM(DescriptorMatcher::MatcherType);
CV_PY_FROM_ENUM(KAZE::DiffusivityType);
CV_PY_TO_ENUM(KAZE::DiffusivityType);
CV_PY_FROM_ENUM(ORB::ScoreType);
CV_PY_TO_ENUM(ORB::ScoreType);
#endif