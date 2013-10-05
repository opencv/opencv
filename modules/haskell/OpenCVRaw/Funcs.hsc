{-# LANGUAGE ForeignFunctionInterface #-}
#include <bindings.dsl.h>
#include <opencv_generated.hpp>
module OpenCVRaw.Funcs where
#strict_import
import Foreign.C
import Foreign.C.Types
import OpenCVRaw.Types
#ccall cv_create_BFMatcher , CInt -> CInt -> IO (Ptr <BFMatcher>)
#ccall cv_create_BRISK , CInt -> CInt -> CFloat -> IO (Ptr <BRISK>)
#ccall cv_create_BRISK5 , Ptr <vector_float> -> Ptr <vector_int> -> CFloat -> CFloat -> Ptr <vector_int> -> IO (Ptr <BRISK>)
#ccall cv_CamShift , Ptr <Mat> -> Ptr <Rect> -> Ptr <TermCriteria> -> IO (Ptr <RotatedRect>)
#ccall cv_Canny , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> CInt -> CInt -> IO ()
#ccall cv_create_CascadeClassifier , IO (Ptr <CascadeClassifier>)
#ccall cv_create_CascadeClassifier1 , Ptr <String> -> IO (Ptr <CascadeClassifier>)
#ccall cv_create_CvANN_MLP , IO (Ptr <CvANN_MLP>)
#ccall cv_create_CvANN_MLP4 , Ptr <Mat> -> CInt -> CDouble -> CDouble -> IO (Ptr <CvANN_MLP>)
#ccall cv_create_CvBoost , IO (Ptr <CvBoost>)
#ccall cv_create_CvBoost8 , Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvBoostParams> -> IO (Ptr <CvBoost>)
#ccall cv_create_CvDTree , IO (Ptr <CvDTree>)
#ccall cv_create_CvERTrees , IO (Ptr <CvERTrees>)
#ccall cv_create_CvGBTrees , IO (Ptr <CvGBTrees>)
#ccall cv_create_CvGBTrees8 , Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvGBTreesParams> -> IO (Ptr <CvGBTrees>)
#ccall cv_create_CvKNearest , IO (Ptr <CvKNearest>)
#ccall cv_create_CvKNearest5 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO (Ptr <CvKNearest>)
#ccall cv_create_CvNormalBayesClassifier , IO (Ptr <CvNormalBayesClassifier>)
#ccall cv_create_CvNormalBayesClassifier4 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO (Ptr <CvNormalBayesClassifier>)
#ccall cv_create_CvRTrees , IO (Ptr <CvRTrees>)
#ccall cv_create_CvSVM , IO (Ptr <CvSVM>)
#ccall cv_create_CvSVM5 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvSVMParams> -> IO (Ptr <CvSVM>)
#ccall cv_create_EM , CInt -> CInt -> Ptr <TermCriteria> -> IO (Ptr <EM>)
#ccall cv_create_FastFeatureDetector , CInt -> CInt -> IO (Ptr <FastFeatureDetector>)
#ccall cv_create_FastFeatureDetector3 , CInt -> CInt -> CInt -> IO (Ptr <FastFeatureDetector>)
#ccall cv_create_FlannBasedMatcher , Ptr <IndexParams> -> Ptr <SearchParams> -> IO (Ptr <FlannBasedMatcher>)
#ccall cv_create_GFTTDetector , CInt -> CDouble -> CDouble -> CInt -> CInt -> CDouble -> IO (Ptr <GFTTDetector>)
#ccall cv_GaussianBlur , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_create_GridAdaptedFeatureDetector , Ptr <FeatureDetector> -> CInt -> CInt -> CInt -> IO (Ptr <GridAdaptedFeatureDetector>)
#ccall cv_create_HOGDescriptor , IO (Ptr <HOGDescriptor>)
#ccall cv_create_HOGDescriptor1 , Ptr <String> -> IO (Ptr <HOGDescriptor>)
#ccall cv_create_HOGDescriptor11 , Ptr <Size> -> Ptr <Size> -> Ptr <Size> -> Ptr <Size> -> CInt -> CInt -> CDouble -> CInt -> CDouble -> CInt -> CInt -> IO (Ptr <HOGDescriptor>)
#ccall cv_HoughCircles , Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> CDouble -> CDouble -> CDouble -> CInt -> CInt -> IO ()
#ccall cv_HoughLines , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> CInt -> CDouble -> CDouble -> IO ()
#ccall cv_HoughLinesP , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> CInt -> CDouble -> CDouble -> IO ()
#ccall cv_HuMoments , Ptr <Moments> -> Ptr <Mat> -> IO ()
#ccall cv_create_Index , IO (Ptr <Index>)
#ccall cv_create_Index3 , Ptr <Mat> -> Ptr <IndexParams> -> Ptr <flann_distance_t> -> IO (Ptr <Index>)
#ccall cv_create_KDTree , IO (Ptr <KDTree>)
#ccall cv_create_KDTree2 , Ptr <Mat> -> CInt -> IO (Ptr <KDTree>)
#ccall cv_create_KDTree3 , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO (Ptr <KDTree>)
#ccall cv_create_KalmanFilter , IO (Ptr <KalmanFilter>)
#ccall cv_create_KalmanFilter4 , CInt -> CInt -> CInt -> CInt -> IO (Ptr <KalmanFilter>)
#ccall cv_LUT , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_Laplacian , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_create_MSER , CInt -> CInt -> CInt -> CDouble -> CDouble -> CInt -> CDouble -> CDouble -> CInt -> IO (Ptr <MSER>)
#ccall cv_Mahalanobis , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CDouble
#ccall cv_create_ORB , CInt -> CFloat -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO (Ptr <ORB>)
#ccall cv_PCABackProject , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_PCACompute , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_PCACompute4 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> IO ()
#ccall cv_PCAProject , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_PSNR , Ptr <Mat> -> Ptr <Mat> -> IO CDouble
#ccall cv_create_Params , IO (Ptr <Params>)
#ccall cv_create_PyramidAdaptedFeatureDetector , Ptr <FeatureDetector> -> CInt -> IO (Ptr <PyramidAdaptedFeatureDetector>)
#ccall cv_RQDecomp3x3 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO (Ptr <Vec3d>)
#ccall cv_Rodrigues , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_SVBackSubst , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_SVDecomp , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_Scharr , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_create_SimpleBlobDetector , Ptr <Params> -> IO (Ptr <SimpleBlobDetector>)
#ccall cv_Sobel , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_create_StarDetector , CInt -> CInt -> CInt -> CInt -> CInt -> IO (Ptr <StarDetector>)
#ccall cv_create_StereoVar , IO (Ptr <StereoVar>)
#ccall cv_create_StereoVar12 , CInt -> CDouble -> CInt -> CInt -> CInt -> CInt -> CDouble -> CFloat -> CFloat -> CInt -> CInt -> CInt -> IO (Ptr <StereoVar>)
#ccall cv_create_Subdiv2D , IO (Ptr <Subdiv2D>)
#ccall cv_create_Subdiv2D1 , Ptr <Rect> -> IO (Ptr <Subdiv2D>)
#ccall cv_create_VideoCapture , IO (Ptr <VideoCapture>)
#ccall cv_create_VideoCapture1 , CInt -> IO (Ptr <VideoCapture>)
#ccall cv_create_VideoWriter , IO (Ptr <VideoWriter>)
#ccall cv_create_VideoWriter5 , Ptr <String> -> CInt -> CDouble -> Ptr <Size> -> CInt -> IO (Ptr <VideoWriter>)
#ccall cv_Algorithm__create , Ptr <Algorithm> -> Ptr <String> -> IO (Ptr <Algorithm>)
#ccall cv_absdiff , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_accumulate , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_accumulateProduct , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_accumulateSquare , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_accumulateWeighted , Ptr <Mat> -> Ptr <Mat> -> CDouble -> Ptr <Mat> -> IO ()
#ccall cv_adaptiveBilateralFilter , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CDouble -> Ptr <Point> -> CInt -> IO ()
#ccall cv_adaptiveThreshold , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CInt -> CInt -> CInt -> CDouble -> IO ()
#ccall cv_add , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_DescriptorMatcher_add1 , Ptr <DescriptorMatcher> -> Ptr <vector_Mat> -> IO ()
#ccall cv_addWeighted , Ptr <Mat> -> CDouble -> Ptr <Mat> -> CDouble -> CDouble -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_CLAHE_apply , Ptr <CLAHE> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_BackgroundSubtractor_apply3 , Ptr <BackgroundSubtractor> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> IO ()
#ccall cv_applyColorMap , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_approxPolyDP , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CInt -> IO ()
#ccall cv_arcLength , Ptr <Mat> -> CInt -> IO CDouble
#ccall cv_batchDistance , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> CInt -> CInt -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_bilateralFilter , Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_bitwise_and , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_bitwise_not , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_bitwise_or , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_bitwise_xor , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_blur , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> Ptr <Point> -> CInt -> IO ()
#ccall cv_borderInterpolate , CInt -> CInt -> CInt -> IO CInt
#ccall cv_boundingRect , Ptr <Mat> -> IO (Ptr <Rect>)
#ccall cv_boxFilter , Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <Size> -> Ptr <Point> -> CInt -> CInt -> IO ()
#ccall cv_boxPoints , Ptr <RotatedRect> -> Ptr <Mat> -> IO ()
#ccall cv_KDTree_build , Ptr <KDTree> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_Index_build3 , Ptr <Index> -> Ptr <Mat> -> Ptr <IndexParams> -> Ptr <flann_distance_t> -> IO ()
#ccall cv_buildOpticalFlowPyramid , Ptr <Mat> -> Ptr <vector_Mat> -> Ptr <Size> -> CInt -> CInt -> CInt -> CInt -> CInt -> IO CInt
#ccall cv_calcBackProject , Ptr <vector_Mat> -> Ptr <vector_int> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_float> -> CDouble -> IO ()
#ccall cv_calcCovarMatrix , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_calcGlobalOrientation , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> IO CDouble
#ccall cv_calcHist , Ptr <vector_Mat> -> Ptr <vector_int> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_int> -> Ptr <vector_float> -> CInt -> IO ()
#ccall cv_calcMotionGradient , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_calcOpticalFlowFarneback , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> CInt -> CInt -> CInt -> CInt -> CDouble -> CInt -> IO ()
#ccall cv_calcOpticalFlowPyrLK , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CInt -> Ptr <TermCriteria> -> CInt -> CDouble -> IO ()
#ccall cv_calcOpticalFlowSF , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_calcOpticalFlowSF16 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CInt -> CDouble -> CDouble -> CDouble -> CInt -> CDouble -> CDouble -> CDouble -> IO ()
#ccall cv_calibrateCamera , Ptr <vector_Mat> -> Ptr <vector_Mat> -> Ptr <Size> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_Mat> -> Ptr <vector_Mat> -> CInt -> Ptr <TermCriteria> -> IO CDouble
#ccall cv_calibrationMatrixValues , Ptr <Mat> -> Ptr <Size> -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> Ptr <Point2d> -> CDouble -> IO ()
#ccall cv_cartToPolar , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_chamerMatching , Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_vector_Point> -> Ptr <vector_float> -> CDouble -> CInt -> CDouble -> CInt -> CInt -> CInt -> CDouble -> CDouble -> CDouble -> CDouble -> IO CInt
#ccall cv_HOGDescriptor_checkDetectorSize , Ptr <HOGDescriptor> -> IO CInt
#ccall cv_checkRange , Ptr <Mat> -> CInt -> Ptr <Point> -> CDouble -> CDouble -> IO CInt
#ccall cv_circle , Ptr <Mat> -> Ptr <Point> -> CInt -> Ptr <Scalar> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_CvNormalBayesClassifier_clear , Ptr <CvNormalBayesClassifier> -> IO ()
#ccall cv_DescriptorMatcher_clear0 , Ptr <DescriptorMatcher> -> IO ()
#ccall cv_clipLine , Ptr <Rect> -> Ptr <Point> -> Ptr <Point> -> IO CInt
#ccall cv_compare , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_compareHist , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO CDouble
#ccall cv_LineSegmentDetector_compareSegments , Ptr <LineSegmentDetector> -> Ptr <Size> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CInt
#ccall cv_completeSymm , Ptr <Mat> -> CInt -> IO ()
#ccall cv_composeRT , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_StereoMatcher_compute , Ptr <StereoMatcher> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_Feature2D_compute3 , Ptr <Feature2D> -> Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <Mat> -> IO ()
#ccall cv_HOGDescriptor_compute5 , Ptr <HOGDescriptor> -> Ptr <Mat> -> Ptr <vector_float> -> Ptr <Size> -> Ptr <Size> -> Ptr <vector_Point> -> IO ()
#ccall cv_computeCorrespondEpilines , Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_HOGDescriptor_computeGradient , Ptr <HOGDescriptor> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> Ptr <Size> -> IO ()
#ccall cv_connectedComponents , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO CInt
#ccall cv_connectedComponentsWithStats , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO CInt
#ccall cv_contourArea , Ptr <Mat> -> CInt -> IO CDouble
#ccall cv_convertMaps , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_convertPointsFromHomogeneous , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_convertPointsToHomogeneous , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_convertScaleAbs , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> IO ()
#ccall cv_convexHull , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_convexityDefects , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_copyMakeBorder , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> CInt -> CInt -> Ptr <Scalar> -> IO ()
#ccall cv_cornerEigenValsAndVecs , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_cornerHarris , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CDouble -> CInt -> IO ()
#ccall cv_cornerMinEigenVal , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_cornerSubPix , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> Ptr <Size> -> Ptr <TermCriteria> -> IO ()
#ccall cv_KalmanFilter_correct , Ptr <KalmanFilter> -> Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_correctMatches , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_countNonZero , Ptr <Mat> -> IO CInt
#ccall cv_CvANN_MLP_create , Ptr <CvANN_MLP> -> Ptr <Mat> -> CInt -> CDouble -> CDouble -> IO ()
#ccall cv_DescriptorMatcher_create1 , Ptr <DescriptorMatcher> -> Ptr <String> -> IO (Ptr <DescriptorMatcher>)
#ccall cv_createBackgroundSubtractorGMG , CInt -> CDouble -> IO (Ptr <BackgroundSubtractorGMG>)
#ccall cv_createBackgroundSubtractorMOG , CInt -> CInt -> CDouble -> CDouble -> IO (Ptr <BackgroundSubtractorMOG>)
#ccall cv_createBackgroundSubtractorMOG2 , CInt -> CDouble -> CInt -> IO (Ptr <BackgroundSubtractorMOG2>)
#ccall cv_createCLAHE , CDouble -> Ptr <Size> -> IO (Ptr <CLAHE>)
#ccall cv_createEigenFaceRecognizer , CInt -> CDouble -> IO (Ptr <FaceRecognizer>)
#ccall cv_createFisherFaceRecognizer , CInt -> CDouble -> IO (Ptr <FaceRecognizer>)
#ccall cv_createHanningWindow , Ptr <Mat> -> Ptr <Size> -> CInt -> IO ()
#ccall cv_createLBPHFaceRecognizer , CInt -> CInt -> CInt -> CInt -> CDouble -> IO (Ptr <FaceRecognizer>)
#ccall cv_createLineSegmentDetector , CInt -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CInt -> IO (Ptr <LineSegmentDetector>)
#ccall cv_createStereoBM , CInt -> CInt -> IO (Ptr <StereoBM>)
#ccall cv_createStereoSGBM , CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> IO (Ptr <StereoSGBM>)
#ccall cv_cvtColor , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_dct , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_decomposeEssentialMat , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_decomposeProjectionMatrix , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_demosaicing , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_destroyAllWindows , IO ()
#ccall cv_destroyWindow , Ptr <String> -> IO ()
#ccall cv_LineSegmentDetector_detect , Ptr <LineSegmentDetector> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_FeatureDetector_detect3 , Ptr <FeatureDetector> -> Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <Mat> -> IO ()
#ccall cv_HOGDescriptor_detect7 , Ptr <HOGDescriptor> -> Ptr <Mat> -> Ptr <vector_Point> -> Ptr <vector_double> -> CDouble -> Ptr <Size> -> Ptr <Size> -> Ptr <vector_Point> -> IO ()
#ccall cv_CascadeClassifier_detectMultiScale , Ptr <CascadeClassifier> -> Ptr <Mat> -> Ptr <vector_Rect> -> CDouble -> CInt -> CInt -> Ptr <Size> -> Ptr <Size> -> IO ()
#ccall cv_CascadeClassifier_detectMultiScale10 , Ptr <CascadeClassifier> -> Ptr <Mat> -> Ptr <vector_Rect> -> Ptr <vector_int> -> Ptr <vector_double> -> CDouble -> CInt -> CInt -> Ptr <Size> -> Ptr <Size> -> CInt -> IO ()
#ccall cv_CascadeClassifier_detectMultiScale8 , Ptr <CascadeClassifier> -> Ptr <Mat> -> Ptr <vector_Rect> -> Ptr <vector_int> -> CDouble -> CInt -> CInt -> Ptr <Size> -> Ptr <Size> -> IO ()
#ccall cv_HOGDescriptor_detectMultiScale9 , Ptr <HOGDescriptor> -> Ptr <Mat> -> Ptr <vector_Rect> -> Ptr <vector_double> -> CDouble -> Ptr <Size> -> Ptr <Size> -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_determinant , Ptr <Mat> -> IO CDouble
#ccall cv_dft , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_dilate , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Point> -> CInt -> CInt -> Ptr <Scalar> -> IO ()
#ccall cv_KDTree_dims , Ptr <KDTree> -> IO CInt
#ccall cv_distanceTransform , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_distanceTransform4 , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_divide , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> CInt -> IO ()
#ccall cv_divide4 , CDouble -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_drawChessboardCorners , Ptr <Mat> -> Ptr <Size> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_drawContours , Ptr <Mat> -> Ptr <vector_Mat> -> CInt -> Ptr <Scalar> -> CInt -> CInt -> Ptr <Mat> -> CInt -> Ptr <Point> -> IO ()
#ccall cv_drawDataMatrixCodes , Ptr <Mat> -> Ptr <vector_String> -> Ptr <Mat> -> IO ()
#ccall cv_drawKeypoints , Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <Mat> -> Ptr <Scalar> -> CInt -> IO ()
#ccall cv_drawMatches , Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <vector_DMatch> -> Ptr <Mat> -> Ptr <Scalar> -> Ptr <Scalar> -> Ptr <vector_char> -> CInt -> IO ()
#ccall cv_drawMatches10 , Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <vector_vector_DMatch> -> Ptr <Mat> -> Ptr <Scalar> -> Ptr <Scalar> -> Ptr <vector_vector_char> -> CInt -> IO ()
#ccall cv_LineSegmentDetector_drawSegments , Ptr <LineSegmentDetector> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_Subdiv2D_edgeDst , Ptr <Subdiv2D> -> CInt -> Ptr <Point2f> -> IO CInt
#ccall cv_Subdiv2D_edgeOrg , Ptr <Subdiv2D> -> CInt -> Ptr <Point2f> -> IO CInt
#ccall cv_eigen , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CInt
#ccall cv_ellipse , Ptr <Mat> -> Ptr <Point> -> Ptr <Size> -> CDouble -> CDouble -> CDouble -> Ptr <Scalar> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_ellipse2Poly , Ptr <Point> -> Ptr <Size> -> CInt -> CInt -> CInt -> CInt -> Ptr <vector_Point> -> IO ()
#ccall cv_ellipse5 , Ptr <Mat> -> Ptr <RotatedRect> -> Ptr <Scalar> -> CInt -> CInt -> IO ()
#ccall cv_FeatureDetector_empty , Ptr <FeatureDetector> -> IO CInt
#ccall cv_CascadeClassifier_empty0 , Ptr <CascadeClassifier> -> IO CInt
#ccall cv_equalizeHist , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_erode , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Point> -> CInt -> CInt -> Ptr <Scalar> -> IO ()
#ccall cv_estimateAffine3D , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> IO CInt
#ccall cv_estimateRigidTransform , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO (Ptr <Mat>)
#ccall cv_exp , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_extractChannel , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_fillConvexPoly , Ptr <Mat> -> Ptr <Mat> -> Ptr <Scalar> -> CInt -> CInt -> IO ()
#ccall cv_fillPoly , Ptr <Mat> -> Ptr <vector_Mat> -> Ptr <Scalar> -> CInt -> CInt -> Ptr <Point> -> IO ()
#ccall cv_filter2D , Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Point> -> CDouble -> CInt -> IO ()
#ccall cv_filterSpeckles , Ptr <Mat> -> CDouble -> CInt -> CDouble -> Ptr <Mat> -> IO ()
#ccall cv_findChessboardCorners , Ptr <Mat> -> Ptr <Size> -> Ptr <Mat> -> CInt -> IO CInt
#ccall cv_findCirclesGrid , Ptr <Mat> -> Ptr <Size> -> Ptr <Mat> -> CInt -> Ptr <FeatureDetector> -> IO CInt
#ccall cv_findContours , Ptr <Mat> -> Ptr <vector_Mat> -> Ptr <Mat> -> CInt -> CInt -> Ptr <Point> -> IO ()
#ccall cv_findDataMatrix , Ptr <Mat> -> Ptr <vector_String> -> Ptr <Mat> -> Ptr <vector_Mat> -> IO ()
#ccall cv_findEssentialMat , Ptr <Mat> -> Ptr <Mat> -> CDouble -> Ptr <Point2d> -> CInt -> CDouble -> CDouble -> Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_findFundamentalMat , Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> CDouble -> Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_findHomography , Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_KDTree_findNearest , Ptr <KDTree> -> Ptr <Mat> -> CInt -> CInt -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CInt
#ccall cv_Subdiv2D_findNearest2 , Ptr <Subdiv2D> -> Ptr <Point2f> -> Ptr <Point2f> -> IO CInt
#ccall cv_findNonZero , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_KDTree_findOrthoRange , Ptr <KDTree> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_findTransformECC , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <TermCriteria> -> IO CDouble
#ccall cv_CvKNearest_find_nearest , Ptr <CvKNearest> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CFloat
#ccall cv_fitEllipse , Ptr <Mat> -> IO (Ptr <RotatedRect>)
#ccall cv_fitLine , Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> CDouble -> CDouble -> IO ()
#ccall cv_flip , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_floodFill , Ptr <Mat> -> Ptr <Mat> -> Ptr <Point> -> Ptr <Scalar> -> Ptr <Rect> -> Ptr <Scalar> -> Ptr <Scalar> -> CInt -> IO CInt
#ccall cv_VideoWriter_fourcc , Ptr <VideoWriter> -> CChar -> CChar -> CChar -> CChar -> IO CInt
#ccall cv_gemm , Ptr <Mat> -> Ptr <Mat> -> CDouble -> Ptr <Mat> -> CDouble -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_BRISK_generateKernel , Ptr <BRISK> -> Ptr <vector_float> -> Ptr <vector_int> -> CFloat -> CFloat -> Ptr <vector_int> -> IO ()
#ccall cv_VideoCapture_get , Ptr <VideoCapture> -> CInt -> IO CDouble
#ccall cv_getAffineTransform , Ptr <Mat> -> Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_Algorithm_getAlgorithm , Ptr <Algorithm> -> Ptr <String> -> IO (Ptr <Algorithm>)
#ccall cv_Index_getAlgorithm0 , Ptr <Index> -> IO (Ptr <flann_algorithm_t>)
#ccall cv_Algorithm_getBool , Ptr <Algorithm> -> Ptr <String> -> IO CInt
#ccall cv_HOGDescriptor_getDaimlerPeopleDetector , Ptr <HOGDescriptor> -> IO (Ptr <vector_float>)
#ccall cv_getDefaultNewCameraMatrix , Ptr <Mat> -> Ptr <Size> -> CInt -> IO (Ptr <Mat>)
#ccall cv_HOGDescriptor_getDefaultPeopleDetector , Ptr <HOGDescriptor> -> IO (Ptr <vector_float>)
#ccall cv_getDerivKernels , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> CInt -> CInt -> IO ()
#ccall cv_HOGDescriptor_getDescriptorSize , Ptr <HOGDescriptor> -> IO CSize
#ccall cv_Index_getDistance , Ptr <Index> -> IO (Ptr <flann_distance_t>)
#ccall cv_Algorithm_getDouble , Ptr <Algorithm> -> Ptr <String> -> IO CDouble
#ccall cv_Subdiv2D_getEdge , Ptr <Subdiv2D> -> CInt -> CInt -> IO CInt
#ccall cv_Subdiv2D_getEdgeList , Ptr <Subdiv2D> -> Ptr <vector_Vec4f> -> IO ()
#ccall cv_getGaborKernel , Ptr <Size> -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CInt -> IO (Ptr <Mat>)
#ccall cv_getGaussianKernel , CInt -> CDouble -> CInt -> IO (Ptr <Mat>)
#ccall cv_Algorithm_getInt , Ptr <Algorithm> -> Ptr <String> -> IO CInt
#ccall cv_Algorithm_getList , Ptr <Algorithm> -> Ptr <vector_String> -> IO ()
#ccall cv_Algorithm_getMat , Ptr <Algorithm> -> Ptr <String> -> IO (Ptr <Mat>)
#ccall cv_Algorithm_getMatVector , Ptr <Algorithm> -> Ptr <String> -> IO (Ptr <vector_Mat>)
#ccall cv_getOptimalDFTSize , CInt -> IO CInt
#ccall cv_getOptimalNewCameraMatrix , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CDouble -> Ptr <Size> -> Ptr <Rect> -> CInt -> IO (Ptr <Mat>)
#ccall cv_Algorithm_getParams , Ptr <Algorithm> -> Ptr <vector_String> -> IO ()
#ccall cv_getPerspectiveTransform , Ptr <Mat> -> Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_KDTree_getPoints , Ptr <KDTree> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_getRectSubPix , Ptr <Mat> -> Ptr <Size> -> Ptr <Point2f> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_getRotationMatrix2D , Ptr <Point2f> -> CDouble -> CDouble -> IO (Ptr <Mat>)
#ccall cv_Algorithm_getString , Ptr <Algorithm> -> Ptr <String> -> IO (Ptr <String>)
#ccall cv_getStructuringElement , CInt -> Ptr <Size> -> Ptr <Point> -> IO (Ptr <Mat>)
#ccall cv_getTextSize , Ptr <String> -> CInt -> CDouble -> CInt -> Ptr CInt -> IO (Ptr <Size>)
#ccall cv_getTrackbarPos , Ptr <String> -> Ptr <String> -> IO CInt
#ccall cv_DescriptorMatcher_getTrainDescriptors , Ptr <DescriptorMatcher> -> IO (Ptr <vector_Mat>)
#ccall cv_Subdiv2D_getTriangleList , Ptr <Subdiv2D> -> Ptr <vector_Vec6f> -> IO ()
#ccall cv_getValidDisparityROI , Ptr <Rect> -> Ptr <Rect> -> CInt -> CInt -> CInt -> IO (Ptr <Rect>)
#ccall cv_CvDTree_getVarImportance , Ptr <CvDTree> -> IO (Ptr <Mat>)
#ccall cv_CvRTrees_getVarImportance0 , Ptr <CvRTrees> -> IO (Ptr <Mat>)
#ccall cv_Subdiv2D_getVertex , Ptr <Subdiv2D> -> CInt -> Ptr CInt -> IO (Ptr <Point2f>)
#ccall cv_Subdiv2D_getVoronoiFacetList , Ptr <Subdiv2D> -> Ptr <vector_int> -> Ptr <vector_vector_Point2f> -> Ptr <vector_Point2f> -> IO ()
#ccall cv_HOGDescriptor_getWinSigma , Ptr <HOGDescriptor> -> IO CDouble
#ccall cv_getWindowProperty , Ptr <String> -> CInt -> IO CDouble
#ccall cv_CvSVM_get_support_vector_count , Ptr <CvSVM> -> IO CInt
#ccall cv_CvSVM_get_var_count , Ptr <CvSVM> -> IO CInt
#ccall cv_goodFeaturesToTrack , Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> CDouble -> Ptr <Mat> -> CInt -> CInt -> CDouble -> IO ()
#ccall cv_VideoCapture_grab , Ptr <VideoCapture> -> IO CInt
#ccall cv_grabCut , Ptr <Mat> -> Ptr <Mat> -> Ptr <Rect> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_groupRectangles , Ptr <vector_Rect> -> Ptr <vector_int> -> CInt -> CDouble -> IO ()
#ccall cv_hconcat , Ptr <vector_Mat> -> Ptr <Mat> -> IO ()
#ccall cv_idct , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_idft , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_imdecode , Ptr <Mat> -> CInt -> IO (Ptr <Mat>)
#ccall cv_imencode , Ptr <String> -> Ptr <Mat> -> Ptr <vector_uchar> -> Ptr <vector_int> -> IO CInt
#ccall cv_imread , Ptr <String> -> CInt -> IO (Ptr <Mat>)
#ccall cv_imshow , Ptr <String> -> Ptr <Mat> -> IO ()
#ccall cv_imwrite , Ptr <String> -> Ptr <Mat> -> Ptr <vector_int> -> IO CInt
#ccall cv_inRange , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_initCameraMatrix2D , Ptr <vector_Mat> -> Ptr <vector_Mat> -> Ptr <Size> -> CDouble -> IO (Ptr <Mat>)
#ccall cv_Subdiv2D_initDelaunay , Ptr <Subdiv2D> -> Ptr <Rect> -> IO ()
#ccall cv_initUndistortRectifyMap , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_initWideAngleProjMap , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CInt -> CInt -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> IO CFloat
#ccall cv_Subdiv2D_insert , Ptr <Subdiv2D> -> Ptr <Point2f> -> IO CInt
#ccall cv_Subdiv2D_insert1 , Ptr <Subdiv2D> -> Ptr <vector_Point2f> -> IO ()
#ccall cv_insertChannel , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_integral , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_integral4 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_integral5 , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_intersectConvexConvex , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO CFloat
#ccall cv_invert , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO CDouble
#ccall cv_invertAffineTransform , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_isContourConvex , Ptr <Mat> -> IO CInt
#ccall cv_VideoCapture_isOpened , Ptr <VideoCapture> -> IO CInt
#ccall cv_VideoWriter_isOpened0 , Ptr <VideoWriter> -> IO CInt
#ccall cv_EM_isTrained , Ptr <EM> -> IO CInt
#ccall cv_kmeans , Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <TermCriteria> -> CInt -> CInt -> Ptr <Mat> -> IO CDouble
#ccall cv_DescriptorMatcher_knnMatch , Ptr <DescriptorMatcher> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_vector_DMatch> -> CInt -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_DescriptorMatcher_knnMatch5 , Ptr <DescriptorMatcher> -> Ptr <Mat> -> Ptr <vector_vector_DMatch> -> CInt -> Ptr <vector_Mat> -> CInt -> IO ()
#ccall cv_Index_knnSearch , Ptr <Index> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <SearchParams> -> IO ()
#ccall cv_line , Ptr <Mat> -> Ptr <Point> -> Ptr <Point> -> Ptr <Scalar> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_Index_load , Ptr <Index> -> Ptr <Mat> -> Ptr <String> -> IO CInt
#ccall cv_FaceRecognizer_load1 , Ptr <FaceRecognizer> -> Ptr <String> -> IO ()
#ccall cv_HOGDescriptor_load2 , Ptr <HOGDescriptor> -> Ptr <String> -> Ptr <String> -> IO CInt
#ccall cv_Subdiv2D_locate , Ptr <Subdiv2D> -> Ptr <Point2f> -> CInt -> CInt -> IO CInt
#ccall cv_log , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_magnitude , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_matMulDeriv , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_DescriptorMatcher_match , Ptr <DescriptorMatcher> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_DMatch> -> Ptr <Mat> -> IO ()
#ccall cv_DescriptorMatcher_match3 , Ptr <DescriptorMatcher> -> Ptr <Mat> -> Ptr <vector_DMatch> -> Ptr <vector_Mat> -> IO ()
#ccall cv_matchShapes , Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> IO CDouble
#ccall cv_matchTemplate , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_max , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_mean , Ptr <Mat> -> Ptr <Mat> -> IO (Ptr <Scalar>)
#ccall cv_meanShift , Ptr <Mat> -> Ptr <Rect> -> Ptr <TermCriteria> -> IO CInt
#ccall cv_meanStdDev , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_medianBlur , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_merge , Ptr <vector_Mat> -> Ptr <Mat> -> IO ()
#ccall cv_min , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_minAreaRect , Ptr <Mat> -> IO (Ptr <RotatedRect>)
#ccall cv_minEnclosingCircle , Ptr <Mat> -> Ptr <Point2f> -> CFloat -> IO ()
#ccall cv_minEnclosingTriangle , Ptr <Mat> -> Ptr <Mat> -> IO CDouble
#ccall cv_minMaxLoc , Ptr <Mat> -> Ptr CDouble -> Ptr CDouble -> Ptr <Point> -> Ptr <Point> -> Ptr <Mat> -> IO ()
#ccall cv_mixChannels , Ptr <vector_Mat> -> Ptr <vector_Mat> -> Ptr <vector_int> -> IO ()
#ccall cv_moments , Ptr <Mat> -> CInt -> IO (Ptr <Moments>)
#ccall cv_morphologyEx , Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Point> -> CInt -> CInt -> Ptr <Scalar> -> IO ()
#ccall cv_moveWindow , Ptr <String> -> CInt -> CInt -> IO ()
#ccall cv_mulSpectrums , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_mulTransposed , Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> CDouble -> CInt -> IO ()
#ccall cv_multiply , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> CInt -> IO ()
#ccall cv_namedWindow , Ptr <String> -> CInt -> IO ()
#ccall cv_Subdiv2D_nextEdge , Ptr <Subdiv2D> -> CInt -> IO CInt
#ccall cv_norm , Ptr <Mat> -> CInt -> Ptr <Mat> -> IO CDouble
#ccall cv_norm4 , Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> IO CDouble
#ccall cv_normalize , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> CInt -> CInt -> Ptr <Mat> -> IO ()
#ccall cv_VideoCapture_open , Ptr <VideoCapture> -> Ptr <String> -> IO CInt
#ccall cv_VideoCapture_open1 , Ptr <VideoCapture> -> CInt -> IO CInt
#ccall cv_VideoWriter_open5 , Ptr <VideoWriter> -> Ptr <String> -> CInt -> CDouble -> Ptr <Size> -> CInt -> IO CInt
#ccall cv_Feature2D_call , Ptr <Feature2D> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_KeyPoint> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_StarDetector_call , Ptr <StarDetector> -> Ptr <Mat> -> Ptr <vector_KeyPoint> -> IO ()
#ccall cv_StereoVar_call , Ptr <StereoVar> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_Algorithm_paramHelp , Ptr <Algorithm> -> Ptr <String> -> IO (Ptr <String>)
#ccall cv_Algorithm_paramType , Ptr <Algorithm> -> Ptr <String> -> IO CInt
#ccall cv_patchNaNs , Ptr <Mat> -> CDouble -> IO ()
#ccall cv_perspectiveTransform , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_phase , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_phaseCorrelate , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr CDouble -> IO (Ptr <Point2d>)
#ccall cv_pointPolygonTest , Ptr <Mat> -> Ptr <Point2f> -> CInt -> IO CDouble
#ccall cv_polarToCart , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_polylines , Ptr <Mat> -> Ptr <vector_Mat> -> CInt -> Ptr <Scalar> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_pow , Ptr <Mat> -> CDouble -> Ptr <Mat> -> IO ()
#ccall cv_preCornerDetect , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_CvNormalBayesClassifier_predict , Ptr <CvNormalBayesClassifier> -> Ptr <Mat> -> Ptr <Mat> -> IO CFloat
#ccall cv_KalmanFilter_predict1 , Ptr <KalmanFilter> -> Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_CvANN_MLP_predict2 , Ptr <CvANN_MLP> -> Ptr <Mat> -> Ptr <Mat> -> IO CFloat
#ccall cv_FaceRecognizer_predict3 , Ptr <FaceRecognizer> -> Ptr <Mat> -> CInt -> CDouble -> IO ()
#ccall cv_CvGBTrees_predict4 , Ptr <CvGBTrees> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Range> -> CInt -> IO CFloat
#ccall cv_CvBoost_predict5 , Ptr <CvBoost> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Range> -> CInt -> CInt -> IO CFloat
#ccall cv_CvRTrees_predict_prob , Ptr <CvRTrees> -> Ptr <Mat> -> Ptr <Mat> -> IO CFloat
#ccall cv_projectPoints , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> IO ()
#ccall cv_CvBoost_prune , Ptr <CvBoost> -> Ptr <CvSlice> -> IO ()
#ccall cv_putText , Ptr <Mat> -> Ptr <String> -> Ptr <Point> -> CInt -> CDouble -> Ptr <Scalar> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_pyrDown , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CInt -> IO ()
#ccall cv_pyrMeanShiftFiltering , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> CInt -> Ptr <TermCriteria> -> IO ()
#ccall cv_pyrUp , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CInt -> IO ()
#ccall cv_Index_radiusSearch , Ptr <Index> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> CInt -> Ptr <SearchParams> -> IO CInt
#ccall cv_randShuffle , Ptr <Mat> -> CDouble -> Ptr <RNG> -> IO ()
#ccall cv_randn , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_randu , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_VideoCapture_read , Ptr <VideoCapture> -> Ptr <Mat> -> IO CInt
#ccall cv_recoverPose , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> Ptr <Point2d> -> Ptr <Mat> -> IO CInt
#ccall cv_rectangle , Ptr <Mat> -> Ptr <Point> -> Ptr <Point> -> Ptr <Scalar> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_rectify3Collinear , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_Mat> -> Ptr <vector_Mat> -> Ptr <Size> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> Ptr <Size> -> Ptr <Rect> -> Ptr <Rect> -> CInt -> IO CFloat
#ccall cv_reduce , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_Index_release , Ptr <Index> -> IO ()
#ccall cv_VideoWriter_release0 , Ptr <VideoWriter> -> IO ()
#ccall cv_remap , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> Ptr <Scalar> -> IO ()
#ccall cv_repeat , Ptr <Mat> -> CInt -> CInt -> Ptr <Mat> -> IO ()
#ccall cv_reprojectImageTo3D , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO ()
#ccall cv_resize , Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CDouble -> CDouble -> CInt -> IO ()
#ccall cv_resizeWindow , Ptr <String> -> CInt -> CInt -> IO ()
#ccall cv_VideoCapture_retrieve , Ptr <VideoCapture> -> Ptr <Mat> -> CInt -> IO CInt
#ccall cv_Subdiv2D_rotateEdge , Ptr <Subdiv2D> -> CInt -> CInt -> IO CInt
#ccall cv_rotatedRectangleIntersection , Ptr <RotatedRect> -> Ptr <RotatedRect> -> Ptr <Mat> -> IO CInt
#ccall cv_Index_save , Ptr <Index> -> Ptr <String> -> IO ()
#ccall cv_FaceRecognizer_save1 , Ptr <FaceRecognizer> -> Ptr <String> -> IO ()
#ccall cv_HOGDescriptor_save2 , Ptr <HOGDescriptor> -> Ptr <String> -> Ptr <String> -> IO ()
#ccall cv_scaleAdd , Ptr <Mat> -> CDouble -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_segmentMotion , Ptr <Mat> -> Ptr <Mat> -> Ptr <vector_Rect> -> CDouble -> CDouble -> IO ()
#ccall cv_sepFilter2D , Ptr <Mat> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Point> -> CDouble -> CInt -> IO ()
#ccall cv_VideoCapture_set , Ptr <VideoCapture> -> CInt -> CDouble -> IO CInt
#ccall cv_Algorithm_setAlgorithm , Ptr <Algorithm> -> Ptr <String> -> Ptr <Algorithm> -> IO ()
#ccall cv_BackgroundSubtractorGMG_setBackgroundPrior , Ptr <BackgroundSubtractorGMG> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG_setBackgroundRatio , Ptr <BackgroundSubtractorMOG> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setBackgroundRatio1 , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_StereoMatcher_setBlockSize , Ptr <StereoMatcher> -> CInt -> IO ()
#ccall cv_Algorithm_setBool , Ptr <Algorithm> -> Ptr <String> -> CInt -> IO ()
#ccall cv_CLAHE_setClipLimit , Ptr <CLAHE> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setComplexityReductionThreshold , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorGMG_setDecisionThreshold , Ptr <BackgroundSubtractorGMG> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorGMG_setDefaultLearningRate , Ptr <BackgroundSubtractorGMG> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setDetectShadows , Ptr <BackgroundSubtractorMOG2> -> CInt -> IO ()
#ccall cv_StereoMatcher_setDisp12MaxDiff , Ptr <StereoMatcher> -> CInt -> IO ()
#ccall cv_Algorithm_setDouble , Ptr <Algorithm> -> Ptr <String> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG_setHistory , Ptr <BackgroundSubtractorMOG> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setHistory1 , Ptr <BackgroundSubtractorMOG2> -> CInt -> IO ()
#ccall cv_setIdentity , Ptr <Mat> -> Ptr <Scalar> -> IO ()
#ccall cv_Algorithm_setInt , Ptr <Algorithm> -> Ptr <String> -> CInt -> IO ()
#ccall cv_Algorithm_setMat , Ptr <Algorithm> -> Ptr <String> -> Ptr <Mat> -> IO ()
#ccall cv_Algorithm_setMatVector , Ptr <Algorithm> -> Ptr <String> -> Ptr <vector_Mat> -> IO ()
#ccall cv_BackgroundSubtractorGMG_setMaxFeatures , Ptr <BackgroundSubtractorGMG> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorGMG_setMaxVal , Ptr <BackgroundSubtractorGMG> -> CDouble -> IO ()
#ccall cv_StereoMatcher_setMinDisparity , Ptr <StereoMatcher> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorGMG_setMinVal , Ptr <BackgroundSubtractorGMG> -> CDouble -> IO ()
#ccall cv_StereoSGBM_setMode , Ptr <StereoSGBM> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorMOG_setNMixtures , Ptr <BackgroundSubtractorMOG> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setNMixtures1 , Ptr <BackgroundSubtractorMOG2> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorMOG_setNoiseSigma , Ptr <BackgroundSubtractorMOG> -> CDouble -> IO ()
#ccall cv_StereoMatcher_setNumDisparities , Ptr <StereoMatcher> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorGMG_setNumFrames , Ptr <BackgroundSubtractorGMG> -> CInt -> IO ()
#ccall cv_StereoSGBM_setP1 , Ptr <StereoSGBM> -> CInt -> IO ()
#ccall cv_StereoSGBM_setP2 , Ptr <StereoSGBM> -> CInt -> IO ()
#ccall cv_StereoBM_setPreFilterCap , Ptr <StereoBM> -> CInt -> IO ()
#ccall cv_StereoSGBM_setPreFilterCap1 , Ptr <StereoSGBM> -> CInt -> IO ()
#ccall cv_StereoBM_setPreFilterSize , Ptr <StereoBM> -> CInt -> IO ()
#ccall cv_StereoBM_setPreFilterType , Ptr <StereoBM> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorGMG_setQuantizationLevels , Ptr <BackgroundSubtractorGMG> -> CInt -> IO ()
#ccall cv_StereoBM_setROI1 , Ptr <StereoBM> -> Ptr <Rect> -> IO ()
#ccall cv_StereoBM_setROI2 , Ptr <StereoBM> -> Ptr <Rect> -> IO ()
#ccall cv_HOGDescriptor_setSVMDetector , Ptr <HOGDescriptor> -> Ptr <Mat> -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setShadowThreshold , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setShadowValue , Ptr <BackgroundSubtractorMOG2> -> CInt -> IO ()
#ccall cv_StereoBM_setSmallerBlockSize , Ptr <StereoBM> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorGMG_setSmoothingRadius , Ptr <BackgroundSubtractorGMG> -> CInt -> IO ()
#ccall cv_StereoMatcher_setSpeckleRange , Ptr <StereoMatcher> -> CInt -> IO ()
#ccall cv_StereoMatcher_setSpeckleWindowSize , Ptr <StereoMatcher> -> CInt -> IO ()
#ccall cv_Algorithm_setString , Ptr <Algorithm> -> Ptr <String> -> Ptr <String> -> IO ()
#ccall cv_StereoBM_setTextureThreshold , Ptr <StereoBM> -> CInt -> IO ()
#ccall cv_CLAHE_setTilesGridSize , Ptr <CLAHE> -> Ptr <Size> -> IO ()
#ccall cv_setTrackbarPos , Ptr <String> -> Ptr <String> -> CInt -> IO ()
#ccall cv_StereoBM_setUniquenessRatio , Ptr <StereoBM> -> CInt -> IO ()
#ccall cv_StereoSGBM_setUniquenessRatio1 , Ptr <StereoSGBM> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorGMG_setUpdateBackgroundModel , Ptr <BackgroundSubtractorGMG> -> CInt -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setVarInit , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setVarMax , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setVarMin , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setVarThreshold , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_BackgroundSubtractorMOG2_setVarThresholdGen , Ptr <BackgroundSubtractorMOG2> -> CDouble -> IO ()
#ccall cv_setWindowProperty , Ptr <String> -> CInt -> CDouble -> IO ()
#ccall cv_solve , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO CInt
#ccall cv_solveCubic , Ptr <Mat> -> Ptr <Mat> -> IO CInt
#ccall cv_solvePnP , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> IO CInt
#ccall cv_solvePnPRansac , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CFloat -> CInt -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_solvePoly , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO CDouble
#ccall cv_sort , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_sortIdx , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_split , Ptr <Mat> -> Ptr <vector_Mat> -> IO ()
#ccall cv_sqrt , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_startWindowThread , IO CInt
#ccall cv_stereoCalibrate , Ptr <vector_Mat> -> Ptr <vector_Mat> -> Ptr <vector_Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <TermCriteria> -> CInt -> IO CDouble
#ccall cv_stereoRectify , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> CDouble -> Ptr <Size> -> Ptr <Rect> -> Ptr <Rect> -> IO ()
#ccall cv_stereoRectifyUncalibrated , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> Ptr <Mat> -> Ptr <Mat> -> CDouble -> IO CInt
#ccall cv_subtract , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_sum , Ptr <Mat> -> IO (Ptr <Scalar>)
#ccall cv_Subdiv2D_symEdge , Ptr <Subdiv2D> -> CInt -> IO CInt
#ccall cv_threshold , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> CInt -> IO CDouble
#ccall cv_trace , Ptr <Mat> -> IO (Ptr <Scalar>)
#ccall cv_CvNormalBayesClassifier_train , Ptr <CvNormalBayesClassifier> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> CInt -> IO CInt
#ccall cv_DescriptorMatcher_train0 , Ptr <DescriptorMatcher> -> IO ()
#ccall cv_FaceRecognizer_train2 , Ptr <FaceRecognizer> -> Ptr <vector_Mat> -> Ptr <Mat> -> IO ()
#ccall cv_EM_train4 , Ptr <EM> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CInt
#ccall cv_CvSVM_train5 , Ptr <CvSVM> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvSVMParams> -> IO CInt
#ccall cv_CvANN_MLP_train6 , Ptr <CvANN_MLP> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvANN_MLP_TrainParams> -> CInt -> IO CInt
#ccall cv_CvERTrees_train8 , Ptr <CvERTrees> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvRTParams> -> IO CInt
#ccall cv_CvGBTrees_train9 , Ptr <CvGBTrees> -> Ptr <Mat> -> CInt -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvGBTreesParams> -> CInt -> IO CInt
#ccall cv_EM_trainE , Ptr <EM> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CInt
#ccall cv_EM_trainM , Ptr <EM> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO CInt
#ccall cv_CvSVM_train_auto , Ptr <CvSVM> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <CvSVMParams> -> CInt -> Ptr <CvParamGrid> -> Ptr <CvParamGrid> -> Ptr <CvParamGrid> -> Ptr <CvParamGrid> -> Ptr <CvParamGrid> -> Ptr <CvParamGrid> -> CInt -> IO CInt
#ccall cv_transform , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_transpose , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_triangulatePoints , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_undistort , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_undistortPoints , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_FaceRecognizer_update , Ptr <FaceRecognizer> -> Ptr <vector_Mat> -> Ptr <Mat> -> IO ()
#ccall cv_updateMotionHistory , Ptr <Mat> -> Ptr <Mat> -> CDouble -> CDouble -> IO ()
#ccall cv_validateDisparity , Ptr <Mat> -> Ptr <Mat> -> CInt -> CInt -> CInt -> IO ()
#ccall cv_vconcat , Ptr <vector_Mat> -> Ptr <Mat> -> IO ()
#ccall cv_waitKey , CInt -> IO CInt
#ccall cv_warpAffine , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CInt -> CInt -> Ptr <Scalar> -> IO ()
#ccall cv_warpPerspective , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> Ptr <Size> -> CInt -> CInt -> Ptr <Scalar> -> IO ()
#ccall cv_watershed , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_VideoWriter_write , Ptr <VideoWriter> -> Ptr <Mat> -> IO ()
