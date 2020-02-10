//
// This file is auto-generated. Please don't modify it!
//

#import "Imgproc.h"
#import "CVObjcUtil.h"

#import "CLAHE.h"
#import "DoubleOut.h"
#import "FloatOut.h"
#import "FloatVector.h"
#import "GeneralizedHoughBallard.h"
#import "GeneralizedHoughGuil.h"
#import "Int4.h"
#import "IntOut.h"
#import "IntVector.h"
#import "LineSegmentDetector.h"
#import "Mat.h"
#import "Moments.h"
#import "Point2d.h"
#import "Point2f.h"
#import "Point2i.h"
#import "Rect2i.h"
#import "RotatedRect.h"
#import "Scalar.h"
#import "Size2i.h"
#import "TermCriteria.h"

@implementation Imgproc

+ (NSString*)getVersion {
    return @"4.2.0-pre";
}

+ (NSString*)getNativeLibraryName {
    return @"opencv_objc420";    
}

+ (int)getVersionMajor {
    return 4;
}

+ (int)getVersionMinor {
    return 2;
}

+ (int)getVersionRevision {
    return 0;
}

+ (NSString*)getVersionStatus {
    return @"-pre";
}



+ (int)CV_GAUSSIAN_5x5 {
    return 7;
}

+ (int)CV_SCHARR {
    return -1;
}

+ (int)CV_MAX_SOBEL_KSIZE {
    return 7;
}

+ (int)CV_RGBA2mRGBA {
    return 125;
}

+ (int)CV_mRGBA2RGBA {
    return 126;
}

+ (int)CV_WARP_FILL_OUTLIERS {
    return 8;
}

+ (int)CV_WARP_INVERSE_MAP {
    return 16;
}

+ (int)CV_CHAIN_CODE {
    return 0;
}

+ (int)CV_LINK_RUNS {
    return 5;
}

+ (int)CV_POLY_APPROX_DP {
    return 0;
}

+ (int)CV_CONTOURS_MATCH_I1 {
    return 1;
}

+ (int)CV_CONTOURS_MATCH_I2 {
    return 2;
}

+ (int)CV_CONTOURS_MATCH_I3 {
    return 3;
}

+ (int)CV_CLOCKWISE {
    return 1;
}

+ (int)CV_COUNTER_CLOCKWISE {
    return 2;
}

+ (int)CV_COMP_CORREL {
    return 0;
}

+ (int)CV_COMP_CHISQR {
    return 1;
}

+ (int)CV_COMP_INTERSECT {
    return 2;
}

+ (int)CV_COMP_BHATTACHARYYA {
    return 3;
}

+ (int)CV_COMP_HELLINGER {
    return Imgproc.CV_COMP_BHATTACHARYYA;
}

+ (int)CV_COMP_CHISQR_ALT {
    return 4;
}

+ (int)CV_COMP_KL_DIV {
    return 5;
}

+ (int)CV_DIST_MASK_3 {
    return 3;
}

+ (int)CV_DIST_MASK_5 {
    return 5;
}

+ (int)CV_DIST_MASK_PRECISE {
    return 0;
}

+ (int)CV_DIST_LABEL_CCOMP {
    return 0;
}

+ (int)CV_DIST_LABEL_PIXEL {
    return 1;
}

+ (int)CV_DIST_USER {
    return -1;
}

+ (int)CV_DIST_L1 {
    return 1;
}

+ (int)CV_DIST_L2 {
    return 2;
}

+ (int)CV_DIST_C {
    return 3;
}

+ (int)CV_DIST_L12 {
    return 4;
}

+ (int)CV_DIST_FAIR {
    return 5;
}

+ (int)CV_DIST_WELSCH {
    return 6;
}

+ (int)CV_DIST_HUBER {
    return 7;
}

+ (int)CV_CANNY_L2_GRADIENT {
    return (1 << 31);
}

+ (int)CV_HOUGH_STANDARD {
    return 0;
}

+ (int)CV_HOUGH_PROBABILISTIC {
    return 1;
}

+ (int)CV_HOUGH_MULTI_SCALE {
    return 2;
}

+ (int)CV_HOUGH_GRADIENT {
    return 3;
}



//
//  Mat cv::getAffineTransform(vector_Point2f src, vector_Point2f dst)
//
+ (Mat*)getAffineTransform:(NSArray<Point2f*>*)src dst:(NSArray<Point2f*>*)dst {
    OBJC2CV(cv::Point2f, Point2f, srcVector, src);
    OBJC2CV(cv::Point2f, Point2f, dstVector, dst);
    cv::Mat* retVal = new cv::Mat(cv::getAffineTransform(srcVector, dstVector));
    return [Mat fromNativePtr:retVal];
}


//
//  Mat cv::getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi = CV_PI*0.5, int ktype = CV_64F)
//
+ (Mat*)getGaborKernel:(Size2i*)ksize sigma:(double)sigma theta:(double)theta lambd:(double)lambd gamma:(double)gamma psi:(double)psi ktype:(int)ktype {
    cv::Mat* retVal = new cv::Mat(cv::getGaborKernel(ksize.nativeRef, sigma, theta, lambd, gamma, psi, ktype));
    return [Mat fromNativePtr:retVal];
}

+ (Mat*)getGaborKernel:(Size2i*)ksize sigma:(double)sigma theta:(double)theta lambd:(double)lambd gamma:(double)gamma psi:(double)psi {
    cv::Mat* retVal = new cv::Mat(cv::getGaborKernel(ksize.nativeRef, sigma, theta, lambd, gamma, psi));
    return [Mat fromNativePtr:retVal];
}

+ (Mat*)getGaborKernel:(Size2i*)ksize sigma:(double)sigma theta:(double)theta lambd:(double)lambd gamma:(double)gamma {
    cv::Mat* retVal = new cv::Mat(cv::getGaborKernel(ksize.nativeRef, sigma, theta, lambd, gamma));
    return [Mat fromNativePtr:retVal];
}


//
//  Mat cv::getGaussianKernel(int ksize, double sigma, int ktype = CV_64F)
//
+ (Mat*)getGaussianKernel:(int)ksize sigma:(double)sigma ktype:(int)ktype {
    cv::Mat* retVal = new cv::Mat(cv::getGaussianKernel(ksize, sigma, ktype));
    return [Mat fromNativePtr:retVal];
}

+ (Mat*)getGaussianKernel:(int)ksize sigma:(double)sigma {
    cv::Mat* retVal = new cv::Mat(cv::getGaussianKernel(ksize, sigma));
    return [Mat fromNativePtr:retVal];
}


//
//  Mat cv::getPerspectiveTransform(Mat src, Mat dst, int solveMethod = DECOMP_LU)
//
+ (Mat*)getPerspectiveTransform:(Mat*)src dst:(Mat*)dst solveMethod:(int)solveMethod {
    cv::Mat* retVal = new cv::Mat(cv::getPerspectiveTransform(src.nativeRef, dst.nativeRef, solveMethod));
    return [Mat fromNativePtr:retVal];
}

+ (Mat*)getPerspectiveTransform:(Mat*)src dst:(Mat*)dst {
    cv::Mat* retVal = new cv::Mat(cv::getPerspectiveTransform(src.nativeRef, dst.nativeRef));
    return [Mat fromNativePtr:retVal];
}


//
//  Mat cv::getRotationMatrix2D(Point2f center, double angle, double scale)
//
+ (Mat*)getRotationMatrix2D:(Point2f*)center angle:(double)angle scale:(double)scale {
    cv::Mat* retVal = new cv::Mat(cv::getRotationMatrix2D(center.nativeRef, angle, scale));
    return [Mat fromNativePtr:retVal];
}


//
//  Mat cv::getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1))
//
+ (Mat*)getStructuringElement:(int)shape ksize:(Size2i*)ksize anchor:(Point2i*)anchor {
    cv::Mat* retVal = new cv::Mat(cv::getStructuringElement(shape, ksize.nativeRef, anchor.nativeRef));
    return [Mat fromNativePtr:retVal];
}

+ (Mat*)getStructuringElement:(int)shape ksize:(Size2i*)ksize {
    cv::Mat* retVal = new cv::Mat(cv::getStructuringElement(shape, ksize.nativeRef));
    return [Mat fromNativePtr:retVal];
}


//
//  Moments cv::moments(Mat array, bool binaryImage = false)
//
+ (Moments*)moments:(Mat*)array binaryImage:(BOOL)binaryImage {
    cv::Moments retVal = cv::moments(array.nativeRef, binaryImage);
    return [Moments fromNative:retVal];
}

+ (Moments*)moments:(Mat*)array {
    cv::Moments retVal = cv::moments(array.nativeRef);
    return [Moments fromNative:retVal];
}


//
//  Point2d cv::phaseCorrelate(Mat src1, Mat src2, Mat window = Mat(), double* response = 0)
//
+ (Point2d*)phaseCorrelate:(Mat*)src1 src2:(Mat*)src2 window:(Mat*)window response:(DoubleOut*)response {
    cv::Point2d retVal = cv::phaseCorrelate(src1.nativeRef, src2.nativeRef, window.nativeRef, response.ptr);
    return [Point2d fromNative:retVal];
}

+ (Point2d*)phaseCorrelate:(Mat*)src1 src2:(Mat*)src2 window:(Mat*)window {
    cv::Point2d retVal = cv::phaseCorrelate(src1.nativeRef, src2.nativeRef, window.nativeRef);
    return [Point2d fromNative:retVal];
}

+ (Point2d*)phaseCorrelate:(Mat*)src1 src2:(Mat*)src2 {
    cv::Point2d retVal = cv::phaseCorrelate(src1.nativeRef, src2.nativeRef);
    return [Point2d fromNative:retVal];
}


//
//  Ptr_CLAHE cv::createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8))
//
+ (CLAHE*)createCLAHE:(double)clipLimit tileGridSize:(Size2i*)tileGridSize {
    cv::CLAHE* retVal = cv::createCLAHE(clipLimit, tileGridSize.nativeRef);
    return [CLAHE fromNative:retVal];
}

+ (CLAHE*)createCLAHE:(double)clipLimit {
    cv::CLAHE* retVal = cv::createCLAHE(clipLimit);
    return [CLAHE fromNative:retVal];
}

+ (CLAHE*)createCLAHE {
    cv::CLAHE* retVal = cv::createCLAHE();
    return [CLAHE fromNative:retVal];
}


//
//  Ptr_GeneralizedHoughBallard cv::createGeneralizedHoughBallard()
//
+ (GeneralizedHoughBallard*)createGeneralizedHoughBallard {
    cv::GeneralizedHoughBallard* retVal = cv::createGeneralizedHoughBallard();
    return [GeneralizedHoughBallard fromNative:retVal];
}


//
//  Ptr_GeneralizedHoughGuil cv::createGeneralizedHoughGuil()
//
+ (GeneralizedHoughGuil*)createGeneralizedHoughGuil {
    cv::GeneralizedHoughGuil* retVal = cv::createGeneralizedHoughGuil();
    return [GeneralizedHoughGuil fromNative:retVal];
}


//
//  Ptr_LineSegmentDetector cv::createLineSegmentDetector(int _refine = LSD_REFINE_STD, double _scale = 0.8, double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5, double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024)
//
+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine _scale:(double)_scale _sigma_scale:(double)_sigma_scale _quant:(double)_quant _ang_th:(double)_ang_th _log_eps:(double)_log_eps _density_th:(double)_density_th _n_bins:(int)_n_bins {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th, _n_bins);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine _scale:(double)_scale _sigma_scale:(double)_sigma_scale _quant:(double)_quant _ang_th:(double)_ang_th _log_eps:(double)_log_eps _density_th:(double)_density_th {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine _scale:(double)_scale _sigma_scale:(double)_sigma_scale _quant:(double)_quant _ang_th:(double)_ang_th _log_eps:(double)_log_eps {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine _scale:(double)_scale _sigma_scale:(double)_sigma_scale _quant:(double)_quant _ang_th:(double)_ang_th {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine _scale:(double)_scale _sigma_scale:(double)_sigma_scale _quant:(double)_quant {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine _scale:(double)_scale _sigma_scale:(double)_sigma_scale {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine, _scale, _sigma_scale);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine _scale:(double)_scale {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine, _scale);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector:(int)_refine {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector(_refine);
    return [LineSegmentDetector fromNative:retVal];
}

+ (LineSegmentDetector*)createLineSegmentDetector {
    cv::LineSegmentDetector* retVal = cv::createLineSegmentDetector();
    return [LineSegmentDetector fromNative:retVal];
}


//
//  Rect cv::boundingRect(Mat array)
//
+ (Rect2i*)boundingRect:(Mat*)array {
    cv::Rect retVal = cv::boundingRect(array.nativeRef);
    return [Rect2i fromNative:retVal];
}


//
//  RotatedRect cv::fitEllipse(vector_Point2f points)
//
+ (RotatedRect*)fitEllipse:(NSArray<Point2f*>*)points {
    OBJC2CV(cv::Point2f, Point2f, pointsVector, points);
    cv::RotatedRect retVal = cv::fitEllipse(pointsVector);
    return [RotatedRect fromNative:retVal];
}


//
//  RotatedRect cv::fitEllipseAMS(Mat points)
//
+ (RotatedRect*)fitEllipseAMS:(Mat*)points {
    cv::RotatedRect retVal = cv::fitEllipseAMS(points.nativeRef);
    return [RotatedRect fromNative:retVal];
}


//
//  RotatedRect cv::fitEllipseDirect(Mat points)
//
+ (RotatedRect*)fitEllipseDirect:(Mat*)points {
    cv::RotatedRect retVal = cv::fitEllipseDirect(points.nativeRef);
    return [RotatedRect fromNative:retVal];
}


//
//  RotatedRect cv::minAreaRect(vector_Point2f points)
//
+ (RotatedRect*)minAreaRect:(NSArray<Point2f*>*)points {
    OBJC2CV(cv::Point2f, Point2f, pointsVector, points);
    cv::RotatedRect retVal = cv::minAreaRect(pointsVector);
    return [RotatedRect fromNative:retVal];
}


//
//  bool cv::clipLine(Rect imgRect, Point& pt1, Point& pt2)
//
+ (BOOL)clipLine:(Rect2i*)imgRect pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 {
    bool retVal = cv::clipLine(imgRect.nativeRef, pt1.nativeRef, pt2.nativeRef);
    return retVal;
}


//
//  bool cv::isContourConvex(vector_Point contour)
//
+ (BOOL)isContourConvex:(NSArray<Point2i*>*)contour {
    OBJC2CV(cv::Point, Point2i, contourVector, contour);
    bool retVal = cv::isContourConvex(contourVector);
    return retVal;
}


//
//  double cv::arcLength(vector_Point2f curve, bool closed)
//
+ (double)arcLength:(NSArray<Point2f*>*)curve closed:(BOOL)closed {
    OBJC2CV(cv::Point2f, Point2f, curveVector, curve);
    double retVal = cv::arcLength(curveVector, closed);
    return retVal;
}


//
//  double cv::compareHist(Mat H1, Mat H2, int method)
//
+ (double)compareHist:(Mat*)H1 H2:(Mat*)H2 method:(int)method {
    double retVal = cv::compareHist(H1.nativeRef, H2.nativeRef, method);
    return retVal;
}


//
//  double cv::contourArea(Mat contour, bool oriented = false)
//
+ (double)contourArea:(Mat*)contour oriented:(BOOL)oriented {
    double retVal = cv::contourArea(contour.nativeRef, oriented);
    return retVal;
}

+ (double)contourArea:(Mat*)contour {
    double retVal = cv::contourArea(contour.nativeRef);
    return retVal;
}


//
//  double cv::getFontScaleFromHeight(int fontFace, int pixelHeight, int thickness = 1)
//
+ (double)getFontScaleFromHeight:(int)fontFace pixelHeight:(int)pixelHeight thickness:(int)thickness {
    double retVal = cv::getFontScaleFromHeight(fontFace, pixelHeight, thickness);
    return retVal;
}

+ (double)getFontScaleFromHeight:(int)fontFace pixelHeight:(int)pixelHeight {
    double retVal = cv::getFontScaleFromHeight(fontFace, pixelHeight);
    return retVal;
}


//
//  double cv::matchShapes(Mat contour1, Mat contour2, int method, double parameter)
//
+ (double)matchShapes:(Mat*)contour1 contour2:(Mat*)contour2 method:(int)method parameter:(double)parameter {
    double retVal = cv::matchShapes(contour1.nativeRef, contour2.nativeRef, method, parameter);
    return retVal;
}


//
//  double cv::minEnclosingTriangle(Mat points, Mat& triangle)
//
+ (double)minEnclosingTriangle:(Mat*)points triangle:(Mat*)triangle {
    double retVal = cv::minEnclosingTriangle(points.nativeRef, triangle.nativeRef);
    return retVal;
}


//
//  double cv::pointPolygonTest(vector_Point2f contour, Point2f pt, bool measureDist)
//
+ (double)pointPolygonTest:(NSArray<Point2f*>*)contour pt:(Point2f*)pt measureDist:(BOOL)measureDist {
    OBJC2CV(cv::Point2f, Point2f, contourVector, contour);
    double retVal = cv::pointPolygonTest(contourVector, pt.nativeRef, measureDist);
    return retVal;
}


//
//  double cv::threshold(Mat src, Mat& dst, double thresh, double maxval, int type)
//
+ (double)threshold:(Mat*)src dst:(Mat*)dst thresh:(double)thresh maxval:(double)maxval type:(int)type {
    double retVal = cv::threshold(src.nativeRef, dst.nativeRef, thresh, maxval, type);
    return retVal;
}


//
//  float cv::intersectConvexConvex(Mat _p1, Mat _p2, Mat& _p12, bool handleNested = true)
//
+ (float)intersectConvexConvex:(Mat*)_p1 _p2:(Mat*)_p2 _p12:(Mat*)_p12 handleNested:(BOOL)handleNested {
    float retVal = cv::intersectConvexConvex(_p1.nativeRef, _p2.nativeRef, _p12.nativeRef, handleNested);
    return retVal;
}

+ (float)intersectConvexConvex:(Mat*)_p1 _p2:(Mat*)_p2 _p12:(Mat*)_p12 {
    float retVal = cv::intersectConvexConvex(_p1.nativeRef, _p2.nativeRef, _p12.nativeRef);
    return retVal;
}


//
//  float cv::wrapperEMD(Mat signature1, Mat signature2, int distType, Mat cost = Mat(),  _hidden_ & lowerBound = cv::Ptr<float>(), Mat& flow = Mat())
//
+ (float)EMD:(Mat*)signature1 signature2:(Mat*)signature2 distType:(int)distType cost:(Mat*)cost flow:(Mat*)flow {
    float retVal = cv::wrapperEMD(signature1.nativeRef, signature2.nativeRef, distType, cost.nativeRef, cv::Ptr<float>(), flow.nativeRef);
    return retVal;
}

+ (float)EMD:(Mat*)signature1 signature2:(Mat*)signature2 distType:(int)distType cost:(Mat*)cost {
    float retVal = cv::wrapperEMD(signature1.nativeRef, signature2.nativeRef, distType, cost.nativeRef, cv::Ptr<float>());
    return retVal;
}

+ (float)EMD:(Mat*)signature1 signature2:(Mat*)signature2 distType:(int)distType {
    float retVal = cv::wrapperEMD(signature1.nativeRef, signature2.nativeRef, distType);
    return retVal;
}


//
//  int cv::connectedComponents(Mat image, Mat& labels, int connectivity, int ltype, int ccltype)
//
+ (int)connectedComponentsWithAlgorithm:(Mat*)image labels:(Mat*)labels connectivity:(int)connectivity ltype:(int)ltype ccltype:(int)ccltype {
    int retVal = cv::connectedComponents(image.nativeRef, labels.nativeRef, connectivity, ltype, ccltype);
    return retVal;
}


//
//  int cv::connectedComponents(Mat image, Mat& labels, int connectivity = 8, int ltype = CV_32S)
//
+ (int)connectedComponents:(Mat*)image labels:(Mat*)labels connectivity:(int)connectivity ltype:(int)ltype {
    int retVal = cv::connectedComponents(image.nativeRef, labels.nativeRef, connectivity, ltype);
    return retVal;
}

+ (int)connectedComponents:(Mat*)image labels:(Mat*)labels connectivity:(int)connectivity {
    int retVal = cv::connectedComponents(image.nativeRef, labels.nativeRef, connectivity);
    return retVal;
}

+ (int)connectedComponents:(Mat*)image labels:(Mat*)labels {
    int retVal = cv::connectedComponents(image.nativeRef, labels.nativeRef);
    return retVal;
}


//
//  int cv::connectedComponentsWithStats(Mat image, Mat& labels, Mat& stats, Mat& centroids, int connectivity, int ltype, int ccltype)
//
+ (int)connectedComponentsWithStatsWithAlgorithm:(Mat*)image labels:(Mat*)labels stats:(Mat*)stats centroids:(Mat*)centroids connectivity:(int)connectivity ltype:(int)ltype ccltype:(int)ccltype {
    int retVal = cv::connectedComponentsWithStats(image.nativeRef, labels.nativeRef, stats.nativeRef, centroids.nativeRef, connectivity, ltype, ccltype);
    return retVal;
}


//
//  int cv::connectedComponentsWithStats(Mat image, Mat& labels, Mat& stats, Mat& centroids, int connectivity = 8, int ltype = CV_32S)
//
+ (int)connectedComponentsWithStats:(Mat*)image labels:(Mat*)labels stats:(Mat*)stats centroids:(Mat*)centroids connectivity:(int)connectivity ltype:(int)ltype {
    int retVal = cv::connectedComponentsWithStats(image.nativeRef, labels.nativeRef, stats.nativeRef, centroids.nativeRef, connectivity, ltype);
    return retVal;
}

+ (int)connectedComponentsWithStats:(Mat*)image labels:(Mat*)labels stats:(Mat*)stats centroids:(Mat*)centroids connectivity:(int)connectivity {
    int retVal = cv::connectedComponentsWithStats(image.nativeRef, labels.nativeRef, stats.nativeRef, centroids.nativeRef, connectivity);
    return retVal;
}

+ (int)connectedComponentsWithStats:(Mat*)image labels:(Mat*)labels stats:(Mat*)stats centroids:(Mat*)centroids {
    int retVal = cv::connectedComponentsWithStats(image.nativeRef, labels.nativeRef, stats.nativeRef, centroids.nativeRef);
    return retVal;
}


//
//  int cv::floodFill(Mat& image, Mat& mask, Point seedPoint, Scalar newVal, Rect* rect = 0, Scalar loDiff = Scalar(), Scalar upDiff = Scalar(), int flags = 4)
//
+ (int)floodFill:(Mat*)image mask:(Mat*)mask seedPoint:(Point2i*)seedPoint newVal:(Scalar*)newVal rect:(Rect2i*)rect loDiff:(Scalar*)loDiff upDiff:(Scalar*)upDiff flags:(int)flags {
    int retVal = cv::floodFill(image.nativeRef, mask.nativeRef, seedPoint.nativeRef, newVal.nativeRef, &(rect.nativeRef), loDiff.nativeRef, upDiff.nativeRef, flags);
    return retVal;
}

+ (int)floodFill:(Mat*)image mask:(Mat*)mask seedPoint:(Point2i*)seedPoint newVal:(Scalar*)newVal rect:(Rect2i*)rect loDiff:(Scalar*)loDiff upDiff:(Scalar*)upDiff {
    int retVal = cv::floodFill(image.nativeRef, mask.nativeRef, seedPoint.nativeRef, newVal.nativeRef, &(rect.nativeRef), loDiff.nativeRef, upDiff.nativeRef);
    return retVal;
}

+ (int)floodFill:(Mat*)image mask:(Mat*)mask seedPoint:(Point2i*)seedPoint newVal:(Scalar*)newVal rect:(Rect2i*)rect loDiff:(Scalar*)loDiff {
    int retVal = cv::floodFill(image.nativeRef, mask.nativeRef, seedPoint.nativeRef, newVal.nativeRef, &(rect.nativeRef), loDiff.nativeRef);
    return retVal;
}

+ (int)floodFill:(Mat*)image mask:(Mat*)mask seedPoint:(Point2i*)seedPoint newVal:(Scalar*)newVal rect:(Rect2i*)rect {
    int retVal = cv::floodFill(image.nativeRef, mask.nativeRef, seedPoint.nativeRef, newVal.nativeRef, &(rect.nativeRef));
    return retVal;
}

+ (int)floodFill:(Mat*)image mask:(Mat*)mask seedPoint:(Point2i*)seedPoint newVal:(Scalar*)newVal {
    int retVal = cv::floodFill(image.nativeRef, mask.nativeRef, seedPoint.nativeRef, newVal.nativeRef);
    return retVal;
}


//
//  int cv::rotatedRectangleIntersection(RotatedRect rect1, RotatedRect rect2, Mat& intersectingRegion)
//
+ (int)rotatedRectangleIntersection:(RotatedRect*)rect1 rect2:(RotatedRect*)rect2 intersectingRegion:(Mat*)intersectingRegion {
    int retVal = cv::rotatedRectangleIntersection(rect1.nativeRef, rect2.nativeRef, intersectingRegion.nativeRef);
    return retVal;
}


//
//  void cv::Canny(Mat dx, Mat dy, Mat& edges, double threshold1, double threshold2, bool L2gradient = false)
//
+ (void)Canny:(Mat*)dx dy:(Mat*)dy edges:(Mat*)edges threshold1:(double)threshold1 threshold2:(double)threshold2 L2gradient:(BOOL)L2gradient {
    cv::Canny(dx.nativeRef, dy.nativeRef, edges.nativeRef, threshold1, threshold2, L2gradient);
}

+ (void)Canny:(Mat*)dx dy:(Mat*)dy edges:(Mat*)edges threshold1:(double)threshold1 threshold2:(double)threshold2 {
    cv::Canny(dx.nativeRef, dy.nativeRef, edges.nativeRef, threshold1, threshold2);
}


//
//  void cv::Canny(Mat image, Mat& edges, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false)
//
+ (void)Canny:(Mat*)image edges:(Mat*)edges threshold1:(double)threshold1 threshold2:(double)threshold2 apertureSize:(int)apertureSize L2gradient:(BOOL)L2gradient {
    cv::Canny(image.nativeRef, edges.nativeRef, threshold1, threshold2, apertureSize, L2gradient);
}

+ (void)Canny:(Mat*)image edges:(Mat*)edges threshold1:(double)threshold1 threshold2:(double)threshold2 apertureSize:(int)apertureSize {
    cv::Canny(image.nativeRef, edges.nativeRef, threshold1, threshold2, apertureSize);
}

+ (void)Canny:(Mat*)image edges:(Mat*)edges threshold1:(double)threshold1 threshold2:(double)threshold2 {
    cv::Canny(image.nativeRef, edges.nativeRef, threshold1, threshold2);
}


//
//  void cv::GaussianBlur(Mat src, Mat& dst, Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT)
//
+ (void)GaussianBlur:(Mat*)src dst:(Mat*)dst ksize:(Size2i*)ksize sigmaX:(double)sigmaX sigmaY:(double)sigmaY borderType:(int)borderType {
    cv::GaussianBlur(src.nativeRef, dst.nativeRef, ksize.nativeRef, sigmaX, sigmaY, borderType);
}

+ (void)GaussianBlur:(Mat*)src dst:(Mat*)dst ksize:(Size2i*)ksize sigmaX:(double)sigmaX sigmaY:(double)sigmaY {
    cv::GaussianBlur(src.nativeRef, dst.nativeRef, ksize.nativeRef, sigmaX, sigmaY);
}

+ (void)GaussianBlur:(Mat*)src dst:(Mat*)dst ksize:(Size2i*)ksize sigmaX:(double)sigmaX {
    cv::GaussianBlur(src.nativeRef, dst.nativeRef, ksize.nativeRef, sigmaX);
}


//
//  void cv::HoughCircles(Mat image, Mat& circles, int method, double dp, double minDist, double param1 = 100, double param2 = 100, int minRadius = 0, int maxRadius = 0)
//
+ (void)HoughCircles:(Mat*)image circles:(Mat*)circles method:(int)method dp:(double)dp minDist:(double)minDist param1:(double)param1 param2:(double)param2 minRadius:(int)minRadius maxRadius:(int)maxRadius {
    cv::HoughCircles(image.nativeRef, circles.nativeRef, method, dp, minDist, param1, param2, minRadius, maxRadius);
}

+ (void)HoughCircles:(Mat*)image circles:(Mat*)circles method:(int)method dp:(double)dp minDist:(double)minDist param1:(double)param1 param2:(double)param2 minRadius:(int)minRadius {
    cv::HoughCircles(image.nativeRef, circles.nativeRef, method, dp, minDist, param1, param2, minRadius);
}

+ (void)HoughCircles:(Mat*)image circles:(Mat*)circles method:(int)method dp:(double)dp minDist:(double)minDist param1:(double)param1 param2:(double)param2 {
    cv::HoughCircles(image.nativeRef, circles.nativeRef, method, dp, minDist, param1, param2);
}

+ (void)HoughCircles:(Mat*)image circles:(Mat*)circles method:(int)method dp:(double)dp minDist:(double)minDist param1:(double)param1 {
    cv::HoughCircles(image.nativeRef, circles.nativeRef, method, dp, minDist, param1);
}

+ (void)HoughCircles:(Mat*)image circles:(Mat*)circles method:(int)method dp:(double)dp minDist:(double)minDist {
    cv::HoughCircles(image.nativeRef, circles.nativeRef, method, dp, minDist);
}


//
//  void cv::HoughLines(Mat image, Mat& lines, double rho, double theta, int threshold, double srn = 0, double stn = 0, double min_theta = 0, double max_theta = CV_PI)
//
+ (void)HoughLines:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold srn:(double)srn stn:(double)stn min_theta:(double)min_theta max_theta:(double)max_theta {
    cv::HoughLines(image.nativeRef, lines.nativeRef, rho, theta, threshold, srn, stn, min_theta, max_theta);
}

+ (void)HoughLines:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold srn:(double)srn stn:(double)stn min_theta:(double)min_theta {
    cv::HoughLines(image.nativeRef, lines.nativeRef, rho, theta, threshold, srn, stn, min_theta);
}

+ (void)HoughLines:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold srn:(double)srn stn:(double)stn {
    cv::HoughLines(image.nativeRef, lines.nativeRef, rho, theta, threshold, srn, stn);
}

+ (void)HoughLines:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold srn:(double)srn {
    cv::HoughLines(image.nativeRef, lines.nativeRef, rho, theta, threshold, srn);
}

+ (void)HoughLines:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold {
    cv::HoughLines(image.nativeRef, lines.nativeRef, rho, theta, threshold);
}


//
//  void cv::HoughLinesP(Mat image, Mat& lines, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0)
//
+ (void)HoughLinesP:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold minLineLength:(double)minLineLength maxLineGap:(double)maxLineGap {
    cv::HoughLinesP(image.nativeRef, lines.nativeRef, rho, theta, threshold, minLineLength, maxLineGap);
}

+ (void)HoughLinesP:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold minLineLength:(double)minLineLength {
    cv::HoughLinesP(image.nativeRef, lines.nativeRef, rho, theta, threshold, minLineLength);
}

+ (void)HoughLinesP:(Mat*)image lines:(Mat*)lines rho:(double)rho theta:(double)theta threshold:(int)threshold {
    cv::HoughLinesP(image.nativeRef, lines.nativeRef, rho, theta, threshold);
}


//
//  void cv::HoughLinesPointSet(Mat _point, Mat& _lines, int lines_max, int threshold, double min_rho, double max_rho, double rho_step, double min_theta, double max_theta, double theta_step)
//
+ (void)HoughLinesPointSet:(Mat*)_point _lines:(Mat*)_lines lines_max:(int)lines_max threshold:(int)threshold min_rho:(double)min_rho max_rho:(double)max_rho rho_step:(double)rho_step min_theta:(double)min_theta max_theta:(double)max_theta theta_step:(double)theta_step {
    cv::HoughLinesPointSet(_point.nativeRef, _lines.nativeRef, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step);
}


//
//  void cv::HuMoments(Moments m, Mat& hu)
//
+ (void)HuMoments:(Moments*)m hu:(Mat*)hu {
    cv::HuMoments(m.nativeRef, hu.nativeRef);
}


//
//  void cv::Laplacian(Mat src, Mat& dst, int ddepth, int ksize = 1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
//
+ (void)Laplacian:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(int)ksize scale:(double)scale delta:(double)delta borderType:(int)borderType {
    cv::Laplacian(src.nativeRef, dst.nativeRef, ddepth, ksize, scale, delta, borderType);
}

+ (void)Laplacian:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(int)ksize scale:(double)scale delta:(double)delta {
    cv::Laplacian(src.nativeRef, dst.nativeRef, ddepth, ksize, scale, delta);
}

+ (void)Laplacian:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(int)ksize scale:(double)scale {
    cv::Laplacian(src.nativeRef, dst.nativeRef, ddepth, ksize, scale);
}

+ (void)Laplacian:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(int)ksize {
    cv::Laplacian(src.nativeRef, dst.nativeRef, ddepth, ksize);
}

+ (void)Laplacian:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth {
    cv::Laplacian(src.nativeRef, dst.nativeRef, ddepth);
}


//
//  void cv::Scharr(Mat src, Mat& dst, int ddepth, int dx, int dy, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
//
+ (void)Scharr:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy scale:(double)scale delta:(double)delta borderType:(int)borderType {
    cv::Scharr(src.nativeRef, dst.nativeRef, ddepth, dx, dy, scale, delta, borderType);
}

+ (void)Scharr:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy scale:(double)scale delta:(double)delta {
    cv::Scharr(src.nativeRef, dst.nativeRef, ddepth, dx, dy, scale, delta);
}

+ (void)Scharr:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy scale:(double)scale {
    cv::Scharr(src.nativeRef, dst.nativeRef, ddepth, dx, dy, scale);
}

+ (void)Scharr:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy {
    cv::Scharr(src.nativeRef, dst.nativeRef, ddepth, dx, dy);
}


//
//  void cv::Sobel(Mat src, Mat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
//
+ (void)Sobel:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy ksize:(int)ksize scale:(double)scale delta:(double)delta borderType:(int)borderType {
    cv::Sobel(src.nativeRef, dst.nativeRef, ddepth, dx, dy, ksize, scale, delta, borderType);
}

+ (void)Sobel:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy ksize:(int)ksize scale:(double)scale delta:(double)delta {
    cv::Sobel(src.nativeRef, dst.nativeRef, ddepth, dx, dy, ksize, scale, delta);
}

+ (void)Sobel:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy ksize:(int)ksize scale:(double)scale {
    cv::Sobel(src.nativeRef, dst.nativeRef, ddepth, dx, dy, ksize, scale);
}

+ (void)Sobel:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy ksize:(int)ksize {
    cv::Sobel(src.nativeRef, dst.nativeRef, ddepth, dx, dy, ksize);
}

+ (void)Sobel:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth dx:(int)dx dy:(int)dy {
    cv::Sobel(src.nativeRef, dst.nativeRef, ddepth, dx, dy);
}


//
//  void cv::accumulate(Mat src, Mat& dst, Mat mask = Mat())
//
+ (void)accumulate:(Mat*)src dst:(Mat*)dst mask:(Mat*)mask {
    cv::accumulate(src.nativeRef, dst.nativeRef, mask.nativeRef);
}

+ (void)accumulate:(Mat*)src dst:(Mat*)dst {
    cv::accumulate(src.nativeRef, dst.nativeRef);
}


//
//  void cv::accumulateProduct(Mat src1, Mat src2, Mat& dst, Mat mask = Mat())
//
+ (void)accumulateProduct:(Mat*)src1 src2:(Mat*)src2 dst:(Mat*)dst mask:(Mat*)mask {
    cv::accumulateProduct(src1.nativeRef, src2.nativeRef, dst.nativeRef, mask.nativeRef);
}

+ (void)accumulateProduct:(Mat*)src1 src2:(Mat*)src2 dst:(Mat*)dst {
    cv::accumulateProduct(src1.nativeRef, src2.nativeRef, dst.nativeRef);
}


//
//  void cv::accumulateSquare(Mat src, Mat& dst, Mat mask = Mat())
//
+ (void)accumulateSquare:(Mat*)src dst:(Mat*)dst mask:(Mat*)mask {
    cv::accumulateSquare(src.nativeRef, dst.nativeRef, mask.nativeRef);
}

+ (void)accumulateSquare:(Mat*)src dst:(Mat*)dst {
    cv::accumulateSquare(src.nativeRef, dst.nativeRef);
}


//
//  void cv::accumulateWeighted(Mat src, Mat& dst, double alpha, Mat mask = Mat())
//
+ (void)accumulateWeighted:(Mat*)src dst:(Mat*)dst alpha:(double)alpha mask:(Mat*)mask {
    cv::accumulateWeighted(src.nativeRef, dst.nativeRef, alpha, mask.nativeRef);
}

+ (void)accumulateWeighted:(Mat*)src dst:(Mat*)dst alpha:(double)alpha {
    cv::accumulateWeighted(src.nativeRef, dst.nativeRef, alpha);
}


//
//  void cv::adaptiveThreshold(Mat src, Mat& dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)
//
+ (void)adaptiveThreshold:(Mat*)src dst:(Mat*)dst maxValue:(double)maxValue adaptiveMethod:(int)adaptiveMethod thresholdType:(int)thresholdType blockSize:(int)blockSize C:(double)C {
    cv::adaptiveThreshold(src.nativeRef, dst.nativeRef, maxValue, adaptiveMethod, thresholdType, blockSize, C);
}


//
//  void cv::applyColorMap(Mat src, Mat& dst, Mat userColor)
//
+ (void)applyColorMap:(Mat*)src dst:(Mat*)dst userColor:(Mat*)userColor {
    cv::applyColorMap(src.nativeRef, dst.nativeRef, userColor.nativeRef);
}


//
//  void cv::applyColorMap(Mat src, Mat& dst, int colormap)
//
+ (void)applyColorMap:(Mat*)src dst:(Mat*)dst colormap:(int)colormap {
    cv::applyColorMap(src.nativeRef, dst.nativeRef, colormap);
}


//
//  void cv::approxPolyDP(vector_Point2f curve, vector_Point2f& approxCurve, double epsilon, bool closed)
//
+ (void)approxPolyDP:(NSArray<Point2f*>*)curve approxCurve:(NSMutableArray<Point2f*>*)approxCurve epsilon:(double)epsilon closed:(BOOL)closed {
    OBJC2CV(cv::Point2f, Point2f, curveVector, curve);
    OBJC2CV(cv::Point2f, Point2f, approxCurveVector, approxCurve);
    cv::approxPolyDP(curveVector, approxCurveVector, epsilon, closed);
    CV2OBJC(cv::Point2f, Point2f, approxCurveVector, approxCurve);
}


//
//  void cv::arrowedLine(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1)
//
+ (void)arrowedLine:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness line_type:(int)line_type shift:(int)shift tipLength:(double)tipLength {
    cv::arrowedLine(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness, line_type, shift, tipLength);
}

+ (void)arrowedLine:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness line_type:(int)line_type shift:(int)shift {
    cv::arrowedLine(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness, line_type, shift);
}

+ (void)arrowedLine:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness line_type:(int)line_type {
    cv::arrowedLine(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness, line_type);
}

+ (void)arrowedLine:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness {
    cv::arrowedLine(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness);
}

+ (void)arrowedLine:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color {
    cv::arrowedLine(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef);
}


//
//  void cv::bilateralFilter(Mat src, Mat& dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT)
//
+ (void)bilateralFilter:(Mat*)src dst:(Mat*)dst d:(int)d sigmaColor:(double)sigmaColor sigmaSpace:(double)sigmaSpace borderType:(int)borderType {
    cv::bilateralFilter(src.nativeRef, dst.nativeRef, d, sigmaColor, sigmaSpace, borderType);
}

+ (void)bilateralFilter:(Mat*)src dst:(Mat*)dst d:(int)d sigmaColor:(double)sigmaColor sigmaSpace:(double)sigmaSpace {
    cv::bilateralFilter(src.nativeRef, dst.nativeRef, d, sigmaColor, sigmaSpace);
}


//
//  void cv::blur(Mat src, Mat& dst, Size ksize, Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT)
//
+ (void)blur:(Mat*)src dst:(Mat*)dst ksize:(Size2i*)ksize anchor:(Point2i*)anchor borderType:(int)borderType {
    cv::blur(src.nativeRef, dst.nativeRef, ksize.nativeRef, anchor.nativeRef, borderType);
}

+ (void)blur:(Mat*)src dst:(Mat*)dst ksize:(Size2i*)ksize anchor:(Point2i*)anchor {
    cv::blur(src.nativeRef, dst.nativeRef, ksize.nativeRef, anchor.nativeRef);
}

+ (void)blur:(Mat*)src dst:(Mat*)dst ksize:(Size2i*)ksize {
    cv::blur(src.nativeRef, dst.nativeRef, ksize.nativeRef);
}


//
//  void cv::boxFilter(Mat src, Mat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), bool normalize = true, int borderType = BORDER_DEFAULT)
//
+ (void)boxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize anchor:(Point2i*)anchor normalize:(BOOL)normalize borderType:(int)borderType {
    cv::boxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef, anchor.nativeRef, normalize, borderType);
}

+ (void)boxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize anchor:(Point2i*)anchor normalize:(BOOL)normalize {
    cv::boxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef, anchor.nativeRef, normalize);
}

+ (void)boxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize anchor:(Point2i*)anchor {
    cv::boxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef, anchor.nativeRef);
}

+ (void)boxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize {
    cv::boxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef);
}


//
//  void cv::boxPoints(RotatedRect box, Mat& points)
//
+ (void)boxPoints:(RotatedRect*)box points:(Mat*)points {
    cv::boxPoints(box.nativeRef, points.nativeRef);
}


//
//  void cv::calcBackProject(vector_Mat images, vector_int channels, Mat hist, Mat& dst, vector_float ranges, double scale)
//
+ (void)calcBackProject:(NSArray<Mat*>*)images channels:(IntVector*)channels hist:(Mat*)hist dst:(Mat*)dst ranges:(FloatVector*)ranges scale:(double)scale {
    OBJC2CV(cv::Mat, Mat, imagesVector, images);
    cv::calcBackProject(imagesVector, channels.vector, hist.nativeRef, dst.nativeRef, ranges.vector, scale);
}


//
//  void cv::calcHist(vector_Mat images, vector_int channels, Mat mask, Mat& hist, vector_int histSize, vector_float ranges, bool accumulate = false)
//
+ (void)calcHist:(NSArray<Mat*>*)images channels:(IntVector*)channels mask:(Mat*)mask hist:(Mat*)hist histSize:(IntVector*)histSize ranges:(FloatVector*)ranges accumulate:(BOOL)accumulate {
    OBJC2CV(cv::Mat, Mat, imagesVector, images);
    cv::calcHist(imagesVector, channels.vector, mask.nativeRef, hist.nativeRef, histSize.vector, ranges.vector, accumulate);
}

+ (void)calcHist:(NSArray<Mat*>*)images channels:(IntVector*)channels mask:(Mat*)mask hist:(Mat*)hist histSize:(IntVector*)histSize ranges:(FloatVector*)ranges {
    OBJC2CV(cv::Mat, Mat, imagesVector, images);
    cv::calcHist(imagesVector, channels.vector, mask.nativeRef, hist.nativeRef, histSize.vector, ranges.vector);
}


//
//  void cv::circle(Mat& img, Point center, int radius, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
//
+ (void)circle:(Mat*)img center:(Point2i*)center radius:(int)radius color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    cv::circle(img.nativeRef, center.nativeRef, radius, color.nativeRef, thickness, lineType, shift);
}

+ (void)circle:(Mat*)img center:(Point2i*)center radius:(int)radius color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::circle(img.nativeRef, center.nativeRef, radius, color.nativeRef, thickness, lineType);
}

+ (void)circle:(Mat*)img center:(Point2i*)center radius:(int)radius color:(Scalar*)color thickness:(int)thickness {
    cv::circle(img.nativeRef, center.nativeRef, radius, color.nativeRef, thickness);
}

+ (void)circle:(Mat*)img center:(Point2i*)center radius:(int)radius color:(Scalar*)color {
    cv::circle(img.nativeRef, center.nativeRef, radius, color.nativeRef);
}


//
//  void cv::convertMaps(Mat map1, Mat map2, Mat& dstmap1, Mat& dstmap2, int dstmap1type, bool nninterpolation = false)
//
+ (void)convertMaps:(Mat*)map1 map2:(Mat*)map2 dstmap1:(Mat*)dstmap1 dstmap2:(Mat*)dstmap2 dstmap1type:(int)dstmap1type nninterpolation:(BOOL)nninterpolation {
    cv::convertMaps(map1.nativeRef, map2.nativeRef, dstmap1.nativeRef, dstmap2.nativeRef, dstmap1type, nninterpolation);
}

+ (void)convertMaps:(Mat*)map1 map2:(Mat*)map2 dstmap1:(Mat*)dstmap1 dstmap2:(Mat*)dstmap2 dstmap1type:(int)dstmap1type {
    cv::convertMaps(map1.nativeRef, map2.nativeRef, dstmap1.nativeRef, dstmap2.nativeRef, dstmap1type);
}


//
//  void cv::convexHull(vector_Point points, vector_int& hull, bool clockwise = false,  _hidden_  returnPoints = true)
//
+ (void)convexHull:(NSArray<Point2i*>*)points hull:(IntVector*)hull clockwise:(BOOL)clockwise {
    OBJC2CV(cv::Point, Point2i, pointsVector, points);
    cv::convexHull(pointsVector, hull.vector, clockwise, true);
}

+ (void)convexHull:(NSArray<Point2i*>*)points hull:(IntVector*)hull {
    OBJC2CV(cv::Point, Point2i, pointsVector, points);
    cv::convexHull(pointsVector, hull.vector);
}


//
//  void cv::convexityDefects(vector_Point contour, vector_int convexhull, vector_Vec4i& convexityDefects)
//
+ (void)convexityDefects:(NSArray<Point2i*>*)contour convexhull:(IntVector*)convexhull convexityDefects:(NSMutableArray<Int4*>*)convexityDefects {
    OBJC2CV(cv::Point, Point2i, contourVector, contour);
    OBJC2CV(cv::Vec4i, Int4, convexityDefectsVector, convexityDefects);
    cv::convexityDefects(contourVector, convexhull.vector, convexityDefectsVector);
    CV2OBJC(cv::Vec4i, Int4, convexityDefectsVector, convexityDefects);
}


//
//  void cv::cornerEigenValsAndVecs(Mat src, Mat& dst, int blockSize, int ksize, int borderType = BORDER_DEFAULT)
//
+ (void)cornerEigenValsAndVecs:(Mat*)src dst:(Mat*)dst blockSize:(int)blockSize ksize:(int)ksize borderType:(int)borderType {
    cv::cornerEigenValsAndVecs(src.nativeRef, dst.nativeRef, blockSize, ksize, borderType);
}

+ (void)cornerEigenValsAndVecs:(Mat*)src dst:(Mat*)dst blockSize:(int)blockSize ksize:(int)ksize {
    cv::cornerEigenValsAndVecs(src.nativeRef, dst.nativeRef, blockSize, ksize);
}


//
//  void cv::cornerHarris(Mat src, Mat& dst, int blockSize, int ksize, double k, int borderType = BORDER_DEFAULT)
//
+ (void)cornerHarris:(Mat*)src dst:(Mat*)dst blockSize:(int)blockSize ksize:(int)ksize k:(double)k borderType:(int)borderType {
    cv::cornerHarris(src.nativeRef, dst.nativeRef, blockSize, ksize, k, borderType);
}

+ (void)cornerHarris:(Mat*)src dst:(Mat*)dst blockSize:(int)blockSize ksize:(int)ksize k:(double)k {
    cv::cornerHarris(src.nativeRef, dst.nativeRef, blockSize, ksize, k);
}


//
//  void cv::cornerMinEigenVal(Mat src, Mat& dst, int blockSize, int ksize = 3, int borderType = BORDER_DEFAULT)
//
+ (void)cornerMinEigenVal:(Mat*)src dst:(Mat*)dst blockSize:(int)blockSize ksize:(int)ksize borderType:(int)borderType {
    cv::cornerMinEigenVal(src.nativeRef, dst.nativeRef, blockSize, ksize, borderType);
}

+ (void)cornerMinEigenVal:(Mat*)src dst:(Mat*)dst blockSize:(int)blockSize ksize:(int)ksize {
    cv::cornerMinEigenVal(src.nativeRef, dst.nativeRef, blockSize, ksize);
}

+ (void)cornerMinEigenVal:(Mat*)src dst:(Mat*)dst blockSize:(int)blockSize {
    cv::cornerMinEigenVal(src.nativeRef, dst.nativeRef, blockSize);
}


//
//  void cv::cornerSubPix(Mat image, Mat& corners, Size winSize, Size zeroZone, TermCriteria criteria)
//
+ (void)cornerSubPix:(Mat*)image corners:(Mat*)corners winSize:(Size2i*)winSize zeroZone:(Size2i*)zeroZone criteria:(TermCriteria*)criteria {
    cv::cornerSubPix(image.nativeRef, corners.nativeRef, winSize.nativeRef, zeroZone.nativeRef, criteria.nativeRef);
}


//
//  void cv::createHanningWindow(Mat& dst, Size winSize, int type)
//
+ (void)createHanningWindow:(Mat*)dst winSize:(Size2i*)winSize type:(int)type {
    cv::createHanningWindow(dst.nativeRef, winSize.nativeRef, type);
}


//
//  void cv::cvtColor(Mat src, Mat& dst, int code, int dstCn = 0)
//
+ (void)cvtColor:(Mat*)src dst:(Mat*)dst code:(int)code dstCn:(int)dstCn {
    cv::cvtColor(src.nativeRef, dst.nativeRef, code, dstCn);
}

+ (void)cvtColor:(Mat*)src dst:(Mat*)dst code:(int)code {
    cv::cvtColor(src.nativeRef, dst.nativeRef, code);
}


//
//  void cv::cvtColorTwoPlane(Mat src1, Mat src2, Mat& dst, int code)
//
+ (void)cvtColorTwoPlane:(Mat*)src1 src2:(Mat*)src2 dst:(Mat*)dst code:(int)code {
    cv::cvtColorTwoPlane(src1.nativeRef, src2.nativeRef, dst.nativeRef, code);
}


//
//  void cv::demosaicing(Mat src, Mat& dst, int code, int dstCn = 0)
//
+ (void)demosaicing:(Mat*)src dst:(Mat*)dst code:(int)code dstCn:(int)dstCn {
    cv::demosaicing(src.nativeRef, dst.nativeRef, code, dstCn);
}

+ (void)demosaicing:(Mat*)src dst:(Mat*)dst code:(int)code {
    cv::demosaicing(src.nativeRef, dst.nativeRef, code);
}


//
//  void cv::dilate(Mat src, Mat& dst, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
//
+ (void)dilate:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations borderType:(int)borderType borderValue:(Scalar*)borderValue {
    cv::dilate(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef, iterations, borderType, borderValue.nativeRef);
}

+ (void)dilate:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations borderType:(int)borderType {
    cv::dilate(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef, iterations, borderType);
}

+ (void)dilate:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations {
    cv::dilate(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef, iterations);
}

+ (void)dilate:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor {
    cv::dilate(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef);
}

+ (void)dilate:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel {
    cv::dilate(src.nativeRef, dst.nativeRef, kernel.nativeRef);
}


//
//  void cv::distanceTransform(Mat src, Mat& dst, Mat& labels, int distanceType, int maskSize, int labelType = DIST_LABEL_CCOMP)
//
+ (void)distanceTransformWithLabels:(Mat*)src dst:(Mat*)dst labels:(Mat*)labels distanceType:(int)distanceType maskSize:(int)maskSize labelType:(int)labelType {
    cv::distanceTransform(src.nativeRef, dst.nativeRef, labels.nativeRef, distanceType, maskSize, labelType);
}

+ (void)distanceTransformWithLabels:(Mat*)src dst:(Mat*)dst labels:(Mat*)labels distanceType:(int)distanceType maskSize:(int)maskSize {
    cv::distanceTransform(src.nativeRef, dst.nativeRef, labels.nativeRef, distanceType, maskSize);
}


//
//  void cv::distanceTransform(Mat src, Mat& dst, int distanceType, int maskSize, int dstType = CV_32F)
//
+ (void)distanceTransform:(Mat*)src dst:(Mat*)dst distanceType:(int)distanceType maskSize:(int)maskSize dstType:(int)dstType {
    cv::distanceTransform(src.nativeRef, dst.nativeRef, distanceType, maskSize, dstType);
}

+ (void)distanceTransform:(Mat*)src dst:(Mat*)dst distanceType:(int)distanceType maskSize:(int)maskSize {
    cv::distanceTransform(src.nativeRef, dst.nativeRef, distanceType, maskSize);
}


//
//  void cv::drawContours(Mat& image, vector_vector_Point contours, int contourIdx, Scalar color, int thickness = 1, int lineType = LINE_8, Mat hierarchy = Mat(), int maxLevel = INT_MAX, Point offset = Point())
//
+ (void)drawContours:(Mat*)image contours:(NSArray<NSArray<Point2i*>*>*)contours contourIdx:(int)contourIdx color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType hierarchy:(Mat*)hierarchy maxLevel:(int)maxLevel offset:(Point2i*)offset {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::drawContours(image.nativeRef, contoursVector2, contourIdx, color.nativeRef, thickness, lineType, hierarchy.nativeRef, maxLevel, offset.nativeRef);
}

+ (void)drawContours:(Mat*)image contours:(NSArray<NSArray<Point2i*>*>*)contours contourIdx:(int)contourIdx color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType hierarchy:(Mat*)hierarchy maxLevel:(int)maxLevel {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::drawContours(image.nativeRef, contoursVector2, contourIdx, color.nativeRef, thickness, lineType, hierarchy.nativeRef, maxLevel);
}

+ (void)drawContours:(Mat*)image contours:(NSArray<NSArray<Point2i*>*>*)contours contourIdx:(int)contourIdx color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType hierarchy:(Mat*)hierarchy {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::drawContours(image.nativeRef, contoursVector2, contourIdx, color.nativeRef, thickness, lineType, hierarchy.nativeRef);
}

+ (void)drawContours:(Mat*)image contours:(NSArray<NSArray<Point2i*>*>*)contours contourIdx:(int)contourIdx color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::drawContours(image.nativeRef, contoursVector2, contourIdx, color.nativeRef, thickness, lineType);
}

+ (void)drawContours:(Mat*)image contours:(NSArray<NSArray<Point2i*>*>*)contours contourIdx:(int)contourIdx color:(Scalar*)color thickness:(int)thickness {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::drawContours(image.nativeRef, contoursVector2, contourIdx, color.nativeRef, thickness);
}

+ (void)drawContours:(Mat*)image contours:(NSArray<NSArray<Point2i*>*>*)contours contourIdx:(int)contourIdx color:(Scalar*)color {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::drawContours(image.nativeRef, contoursVector2, contourIdx, color.nativeRef);
}


//
//  void cv::drawMarker(Mat& img, Point position, Scalar color, int markerType = MARKER_CROSS, int markerSize = 20, int thickness = 1, int line_type = 8)
//
+ (void)drawMarker:(Mat*)img position:(Point2i*)position color:(Scalar*)color markerType:(int)markerType markerSize:(int)markerSize thickness:(int)thickness line_type:(int)line_type {
    cv::drawMarker(img.nativeRef, position.nativeRef, color.nativeRef, markerType, markerSize, thickness, line_type);
}

+ (void)drawMarker:(Mat*)img position:(Point2i*)position color:(Scalar*)color markerType:(int)markerType markerSize:(int)markerSize thickness:(int)thickness {
    cv::drawMarker(img.nativeRef, position.nativeRef, color.nativeRef, markerType, markerSize, thickness);
}

+ (void)drawMarker:(Mat*)img position:(Point2i*)position color:(Scalar*)color markerType:(int)markerType markerSize:(int)markerSize {
    cv::drawMarker(img.nativeRef, position.nativeRef, color.nativeRef, markerType, markerSize);
}

+ (void)drawMarker:(Mat*)img position:(Point2i*)position color:(Scalar*)color markerType:(int)markerType {
    cv::drawMarker(img.nativeRef, position.nativeRef, color.nativeRef, markerType);
}

+ (void)drawMarker:(Mat*)img position:(Point2i*)position color:(Scalar*)color {
    cv::drawMarker(img.nativeRef, position.nativeRef, color.nativeRef);
}


//
//  void cv::ellipse(Mat& img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
//
+ (void)ellipse:(Mat*)img center:(Point2i*)center axes:(Size2i*)axes angle:(double)angle startAngle:(double)startAngle endAngle:(double)endAngle color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    cv::ellipse(img.nativeRef, center.nativeRef, axes.nativeRef, angle, startAngle, endAngle, color.nativeRef, thickness, lineType, shift);
}

+ (void)ellipse:(Mat*)img center:(Point2i*)center axes:(Size2i*)axes angle:(double)angle startAngle:(double)startAngle endAngle:(double)endAngle color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::ellipse(img.nativeRef, center.nativeRef, axes.nativeRef, angle, startAngle, endAngle, color.nativeRef, thickness, lineType);
}

+ (void)ellipse:(Mat*)img center:(Point2i*)center axes:(Size2i*)axes angle:(double)angle startAngle:(double)startAngle endAngle:(double)endAngle color:(Scalar*)color thickness:(int)thickness {
    cv::ellipse(img.nativeRef, center.nativeRef, axes.nativeRef, angle, startAngle, endAngle, color.nativeRef, thickness);
}

+ (void)ellipse:(Mat*)img center:(Point2i*)center axes:(Size2i*)axes angle:(double)angle startAngle:(double)startAngle endAngle:(double)endAngle color:(Scalar*)color {
    cv::ellipse(img.nativeRef, center.nativeRef, axes.nativeRef, angle, startAngle, endAngle, color.nativeRef);
}


//
//  void cv::ellipse(Mat& img, RotatedRect box, Scalar color, int thickness = 1, int lineType = LINE_8)
//
+ (void)ellipse:(Mat*)img box:(RotatedRect*)box color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::ellipse(img.nativeRef, box.nativeRef, color.nativeRef, thickness, lineType);
}

+ (void)ellipse:(Mat*)img box:(RotatedRect*)box color:(Scalar*)color thickness:(int)thickness {
    cv::ellipse(img.nativeRef, box.nativeRef, color.nativeRef, thickness);
}

+ (void)ellipse:(Mat*)img box:(RotatedRect*)box color:(Scalar*)color {
    cv::ellipse(img.nativeRef, box.nativeRef, color.nativeRef);
}


//
//  void cv::ellipse2Poly(Point center, Size axes, int angle, int arcStart, int arcEnd, int delta, vector_Point& pts)
//
+ (void)ellipse2Poly:(Point2i*)center axes:(Size2i*)axes angle:(int)angle arcStart:(int)arcStart arcEnd:(int)arcEnd delta:(int)delta pts:(NSMutableArray<Point2i*>*)pts {
    OBJC2CV(cv::Point, Point2i, ptsVector, pts);
    cv::ellipse2Poly(center.nativeRef, axes.nativeRef, angle, arcStart, arcEnd, delta, ptsVector);
    CV2OBJC(cv::Point, Point2i, ptsVector, pts);
}


//
//  void cv::equalizeHist(Mat src, Mat& dst)
//
+ (void)equalizeHist:(Mat*)src dst:(Mat*)dst {
    cv::equalizeHist(src.nativeRef, dst.nativeRef);
}


//
//  void cv::erode(Mat src, Mat& dst, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
//
+ (void)erode:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations borderType:(int)borderType borderValue:(Scalar*)borderValue {
    cv::erode(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef, iterations, borderType, borderValue.nativeRef);
}

+ (void)erode:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations borderType:(int)borderType {
    cv::erode(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef, iterations, borderType);
}

+ (void)erode:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations {
    cv::erode(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef, iterations);
}

+ (void)erode:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel anchor:(Point2i*)anchor {
    cv::erode(src.nativeRef, dst.nativeRef, kernel.nativeRef, anchor.nativeRef);
}

+ (void)erode:(Mat*)src dst:(Mat*)dst kernel:(Mat*)kernel {
    cv::erode(src.nativeRef, dst.nativeRef, kernel.nativeRef);
}


//
//  void cv::fillConvexPoly(Mat& img, vector_Point points, Scalar color, int lineType = LINE_8, int shift = 0)
//
+ (void)fillConvexPoly:(Mat*)img points:(NSArray<Point2i*>*)points color:(Scalar*)color lineType:(int)lineType shift:(int)shift {
    OBJC2CV(cv::Point, Point2i, pointsVector, points);
    cv::fillConvexPoly(img.nativeRef, pointsVector, color.nativeRef, lineType, shift);
}

+ (void)fillConvexPoly:(Mat*)img points:(NSArray<Point2i*>*)points color:(Scalar*)color lineType:(int)lineType {
    OBJC2CV(cv::Point, Point2i, pointsVector, points);
    cv::fillConvexPoly(img.nativeRef, pointsVector, color.nativeRef, lineType);
}

+ (void)fillConvexPoly:(Mat*)img points:(NSArray<Point2i*>*)points color:(Scalar*)color {
    OBJC2CV(cv::Point, Point2i, pointsVector, points);
    cv::fillConvexPoly(img.nativeRef, pointsVector, color.nativeRef);
}


//
//  void cv::fillPoly(Mat& img, vector_vector_Point pts, Scalar color, int lineType = LINE_8, int shift = 0, Point offset = Point())
//
+ (void)fillPoly:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts color:(Scalar*)color lineType:(int)lineType shift:(int)shift offset:(Point2i*)offset {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::fillPoly(img.nativeRef, ptsVector2, color.nativeRef, lineType, shift, offset.nativeRef);
}

+ (void)fillPoly:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts color:(Scalar*)color lineType:(int)lineType shift:(int)shift {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::fillPoly(img.nativeRef, ptsVector2, color.nativeRef, lineType, shift);
}

+ (void)fillPoly:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts color:(Scalar*)color lineType:(int)lineType {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::fillPoly(img.nativeRef, ptsVector2, color.nativeRef, lineType);
}

+ (void)fillPoly:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts color:(Scalar*)color {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::fillPoly(img.nativeRef, ptsVector2, color.nativeRef);
}


//
//  void cv::filter2D(Mat src, Mat& dst, int ddepth, Mat kernel, Point anchor = Point(-1,-1), double delta = 0, int borderType = BORDER_DEFAULT)
//
+ (void)filter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernel:(Mat*)kernel anchor:(Point2i*)anchor delta:(double)delta borderType:(int)borderType {
    cv::filter2D(src.nativeRef, dst.nativeRef, ddepth, kernel.nativeRef, anchor.nativeRef, delta, borderType);
}

+ (void)filter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernel:(Mat*)kernel anchor:(Point2i*)anchor delta:(double)delta {
    cv::filter2D(src.nativeRef, dst.nativeRef, ddepth, kernel.nativeRef, anchor.nativeRef, delta);
}

+ (void)filter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernel:(Mat*)kernel anchor:(Point2i*)anchor {
    cv::filter2D(src.nativeRef, dst.nativeRef, ddepth, kernel.nativeRef, anchor.nativeRef);
}

+ (void)filter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernel:(Mat*)kernel {
    cv::filter2D(src.nativeRef, dst.nativeRef, ddepth, kernel.nativeRef);
}


//
//  void cv::findContours(Mat image, vector_vector_Point& contours, Mat& hierarchy, int mode, int method, Point offset = Point())
//
+ (void)findContours:(Mat*)image contours:(NSMutableArray<NSMutableArray<Point2i*>*>*)contours hierarchy:(Mat*)hierarchy mode:(int)mode method:(int)method offset:(Point2i*)offset {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::findContours(image.nativeRef, contoursVector2, hierarchy.nativeRef, mode, method, offset.nativeRef);
    CV2OBJC2(cv::Point, Point2i, contoursVector2, contours);
}

+ (void)findContours:(Mat*)image contours:(NSMutableArray<NSMutableArray<Point2i*>*>*)contours hierarchy:(Mat*)hierarchy mode:(int)mode method:(int)method {
    OBJC2CV2(cv::Point, Point2i, contoursVector2, contours);
    cv::findContours(image.nativeRef, contoursVector2, hierarchy.nativeRef, mode, method);
    CV2OBJC2(cv::Point, Point2i, contoursVector2, contours);
}


//
//  void cv::fitLine(Mat points, Mat& line, int distType, double param, double reps, double aeps)
//
+ (void)fitLine:(Mat*)points line:(Mat*)line distType:(int)distType param:(double)param reps:(double)reps aeps:(double)aeps {
    cv::fitLine(points.nativeRef, line.nativeRef, distType, param, reps, aeps);
}


//
//  void cv::getDerivKernels(Mat& kx, Mat& ky, int dx, int dy, int ksize, bool normalize = false, int ktype = CV_32F)
//
+ (void)getDerivKernels:(Mat*)kx ky:(Mat*)ky dx:(int)dx dy:(int)dy ksize:(int)ksize normalize:(BOOL)normalize ktype:(int)ktype {
    cv::getDerivKernels(kx.nativeRef, ky.nativeRef, dx, dy, ksize, normalize, ktype);
}

+ (void)getDerivKernels:(Mat*)kx ky:(Mat*)ky dx:(int)dx dy:(int)dy ksize:(int)ksize normalize:(BOOL)normalize {
    cv::getDerivKernels(kx.nativeRef, ky.nativeRef, dx, dy, ksize, normalize);
}

+ (void)getDerivKernels:(Mat*)kx ky:(Mat*)ky dx:(int)dx dy:(int)dy ksize:(int)ksize {
    cv::getDerivKernels(kx.nativeRef, ky.nativeRef, dx, dy, ksize);
}


//
//  void cv::getRectSubPix(Mat image, Size patchSize, Point2f center, Mat& patch, int patchType = -1)
//
+ (void)getRectSubPix:(Mat*)image patchSize:(Size2i*)patchSize center:(Point2f*)center patch:(Mat*)patch patchType:(int)patchType {
    cv::getRectSubPix(image.nativeRef, patchSize.nativeRef, center.nativeRef, patch.nativeRef, patchType);
}

+ (void)getRectSubPix:(Mat*)image patchSize:(Size2i*)patchSize center:(Point2f*)center patch:(Mat*)patch {
    cv::getRectSubPix(image.nativeRef, patchSize.nativeRef, center.nativeRef, patch.nativeRef);
}


//
//  void cv::goodFeaturesToTrack(Mat image, vector_Point& corners, int maxCorners, double qualityLevel, double minDistance, Mat mask, int blockSize, int gradientSize, bool useHarrisDetector = false, double k = 0.04)
//
+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance mask:(Mat*)mask blockSize:(int)blockSize gradientSize:(int)gradientSize useHarrisDetector:(BOOL)useHarrisDetector k:(double)k {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance, mask.nativeRef, blockSize, gradientSize, useHarrisDetector, k);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}

+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance mask:(Mat*)mask blockSize:(int)blockSize gradientSize:(int)gradientSize useHarrisDetector:(BOOL)useHarrisDetector {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance, mask.nativeRef, blockSize, gradientSize, useHarrisDetector);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}

+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance mask:(Mat*)mask blockSize:(int)blockSize gradientSize:(int)gradientSize {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance, mask.nativeRef, blockSize, gradientSize);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}


//
//  void cv::goodFeaturesToTrack(Mat image, vector_Point& corners, int maxCorners, double qualityLevel, double minDistance, Mat mask = Mat(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04)
//
+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance mask:(Mat*)mask blockSize:(int)blockSize useHarrisDetector:(BOOL)useHarrisDetector k:(double)k {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance, mask.nativeRef, blockSize, useHarrisDetector, k);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}

+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance mask:(Mat*)mask blockSize:(int)blockSize useHarrisDetector:(BOOL)useHarrisDetector {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance, mask.nativeRef, blockSize, useHarrisDetector);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}

+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance mask:(Mat*)mask blockSize:(int)blockSize {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance, mask.nativeRef, blockSize);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}

+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance mask:(Mat*)mask {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance, mask.nativeRef);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}

+ (void)goodFeaturesToTrack:(Mat*)image corners:(NSMutableArray<Point2i*>*)corners maxCorners:(int)maxCorners qualityLevel:(double)qualityLevel minDistance:(double)minDistance {
    OBJC2CV(cv::Point, Point2i, cornersVector, corners);
    cv::goodFeaturesToTrack(image.nativeRef, cornersVector, maxCorners, qualityLevel, minDistance);
    CV2OBJC(cv::Point, Point2i, cornersVector, corners);
}


//
//  void cv::grabCut(Mat img, Mat& mask, Rect rect, Mat& bgdModel, Mat& fgdModel, int iterCount, int mode = GC_EVAL)
//
+ (void)grabCut:(Mat*)img mask:(Mat*)mask rect:(Rect2i*)rect bgdModel:(Mat*)bgdModel fgdModel:(Mat*)fgdModel iterCount:(int)iterCount mode:(int)mode {
    cv::grabCut(img.nativeRef, mask.nativeRef, rect.nativeRef, bgdModel.nativeRef, fgdModel.nativeRef, iterCount, mode);
}

+ (void)grabCut:(Mat*)img mask:(Mat*)mask rect:(Rect2i*)rect bgdModel:(Mat*)bgdModel fgdModel:(Mat*)fgdModel iterCount:(int)iterCount {
    cv::grabCut(img.nativeRef, mask.nativeRef, rect.nativeRef, bgdModel.nativeRef, fgdModel.nativeRef, iterCount);
}


//
//  void cv::integral(Mat src, Mat& sum, Mat& sqsum, Mat& tilted, int sdepth = -1, int sqdepth = -1)
//
+ (void)integral3:(Mat*)src sum:(Mat*)sum sqsum:(Mat*)sqsum tilted:(Mat*)tilted sdepth:(int)sdepth sqdepth:(int)sqdepth {
    cv::integral(src.nativeRef, sum.nativeRef, sqsum.nativeRef, tilted.nativeRef, sdepth, sqdepth);
}

+ (void)integral3:(Mat*)src sum:(Mat*)sum sqsum:(Mat*)sqsum tilted:(Mat*)tilted sdepth:(int)sdepth {
    cv::integral(src.nativeRef, sum.nativeRef, sqsum.nativeRef, tilted.nativeRef, sdepth);
}

+ (void)integral3:(Mat*)src sum:(Mat*)sum sqsum:(Mat*)sqsum tilted:(Mat*)tilted {
    cv::integral(src.nativeRef, sum.nativeRef, sqsum.nativeRef, tilted.nativeRef);
}


//
//  void cv::integral(Mat src, Mat& sum, Mat& sqsum, int sdepth = -1, int sqdepth = -1)
//
+ (void)integral2:(Mat*)src sum:(Mat*)sum sqsum:(Mat*)sqsum sdepth:(int)sdepth sqdepth:(int)sqdepth {
    cv::integral(src.nativeRef, sum.nativeRef, sqsum.nativeRef, sdepth, sqdepth);
}

+ (void)integral2:(Mat*)src sum:(Mat*)sum sqsum:(Mat*)sqsum sdepth:(int)sdepth {
    cv::integral(src.nativeRef, sum.nativeRef, sqsum.nativeRef, sdepth);
}

+ (void)integral2:(Mat*)src sum:(Mat*)sum sqsum:(Mat*)sqsum {
    cv::integral(src.nativeRef, sum.nativeRef, sqsum.nativeRef);
}


//
//  void cv::integral(Mat src, Mat& sum, int sdepth = -1)
//
+ (void)integral:(Mat*)src sum:(Mat*)sum sdepth:(int)sdepth {
    cv::integral(src.nativeRef, sum.nativeRef, sdepth);
}

+ (void)integral:(Mat*)src sum:(Mat*)sum {
    cv::integral(src.nativeRef, sum.nativeRef);
}


//
//  void cv::invertAffineTransform(Mat M, Mat& iM)
//
+ (void)invertAffineTransform:(Mat*)M iM:(Mat*)iM {
    cv::invertAffineTransform(M.nativeRef, iM.nativeRef);
}


//
//  void cv::line(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
//
+ (void)line:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    cv::line(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness, lineType, shift);
}

+ (void)line:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::line(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness, lineType);
}

+ (void)line:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness {
    cv::line(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness);
}

+ (void)line:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color {
    cv::line(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef);
}


//
//  void cv::linearPolar(Mat src, Mat& dst, Point2f center, double maxRadius, int flags)
//
+ (void)linearPolar:(Mat*)src dst:(Mat*)dst center:(Point2f*)center maxRadius:(double)maxRadius flags:(int)flags {
    cv::linearPolar(src.nativeRef, dst.nativeRef, center.nativeRef, maxRadius, flags);
}


//
//  void cv::logPolar(Mat src, Mat& dst, Point2f center, double M, int flags)
//
+ (void)logPolar:(Mat*)src dst:(Mat*)dst center:(Point2f*)center M:(double)M flags:(int)flags {
    cv::logPolar(src.nativeRef, dst.nativeRef, center.nativeRef, M, flags);
}


//
//  void cv::matchTemplate(Mat image, Mat templ, Mat& result, int method, Mat mask = Mat())
//
+ (void)matchTemplate:(Mat*)image templ:(Mat*)templ result:(Mat*)result method:(int)method mask:(Mat*)mask {
    cv::matchTemplate(image.nativeRef, templ.nativeRef, result.nativeRef, method, mask.nativeRef);
}

+ (void)matchTemplate:(Mat*)image templ:(Mat*)templ result:(Mat*)result method:(int)method {
    cv::matchTemplate(image.nativeRef, templ.nativeRef, result.nativeRef, method);
}


//
//  void cv::medianBlur(Mat src, Mat& dst, int ksize)
//
+ (void)medianBlur:(Mat*)src dst:(Mat*)dst ksize:(int)ksize {
    cv::medianBlur(src.nativeRef, dst.nativeRef, ksize);
}


//
//  void cv::minEnclosingCircle(vector_Point2f points, Point2f& center, float& radius)
//
+ (void)minEnclosingCircle:(NSArray<Point2f*>*)points center:(Point2f*)center radius:(FloatOut*)radius {
    OBJC2CV(cv::Point2f, Point2f, pointsVector, points);
    cv::minEnclosingCircle(pointsVector, center.nativeRef, *(float*)(radius.ptr));
}


//
//  void cv::morphologyEx(Mat src, Mat& dst, int op, Mat kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, Scalar borderValue = morphologyDefaultBorderValue())
//
+ (void)morphologyEx:(Mat*)src dst:(Mat*)dst op:(int)op kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations borderType:(int)borderType borderValue:(Scalar*)borderValue {
    cv::morphologyEx(src.nativeRef, dst.nativeRef, op, kernel.nativeRef, anchor.nativeRef, iterations, borderType, borderValue.nativeRef);
}

+ (void)morphologyEx:(Mat*)src dst:(Mat*)dst op:(int)op kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations borderType:(int)borderType {
    cv::morphologyEx(src.nativeRef, dst.nativeRef, op, kernel.nativeRef, anchor.nativeRef, iterations, borderType);
}

+ (void)morphologyEx:(Mat*)src dst:(Mat*)dst op:(int)op kernel:(Mat*)kernel anchor:(Point2i*)anchor iterations:(int)iterations {
    cv::morphologyEx(src.nativeRef, dst.nativeRef, op, kernel.nativeRef, anchor.nativeRef, iterations);
}

+ (void)morphologyEx:(Mat*)src dst:(Mat*)dst op:(int)op kernel:(Mat*)kernel anchor:(Point2i*)anchor {
    cv::morphologyEx(src.nativeRef, dst.nativeRef, op, kernel.nativeRef, anchor.nativeRef);
}

+ (void)morphologyEx:(Mat*)src dst:(Mat*)dst op:(int)op kernel:(Mat*)kernel {
    cv::morphologyEx(src.nativeRef, dst.nativeRef, op, kernel.nativeRef);
}


//
//  void cv::polylines(Mat& img, vector_vector_Point pts, bool isClosed, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
//
+ (void)polylines:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts isClosed:(BOOL)isClosed color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::polylines(img.nativeRef, ptsVector2, isClosed, color.nativeRef, thickness, lineType, shift);
}

+ (void)polylines:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts isClosed:(BOOL)isClosed color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::polylines(img.nativeRef, ptsVector2, isClosed, color.nativeRef, thickness, lineType);
}

+ (void)polylines:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts isClosed:(BOOL)isClosed color:(Scalar*)color thickness:(int)thickness {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::polylines(img.nativeRef, ptsVector2, isClosed, color.nativeRef, thickness);
}

+ (void)polylines:(Mat*)img pts:(NSArray<NSArray<Point2i*>*>*)pts isClosed:(BOOL)isClosed color:(Scalar*)color {
    OBJC2CV2(cv::Point, Point2i, ptsVector2, pts);
    cv::polylines(img.nativeRef, ptsVector2, isClosed, color.nativeRef);
}


//
//  void cv::preCornerDetect(Mat src, Mat& dst, int ksize, int borderType = BORDER_DEFAULT)
//
+ (void)preCornerDetect:(Mat*)src dst:(Mat*)dst ksize:(int)ksize borderType:(int)borderType {
    cv::preCornerDetect(src.nativeRef, dst.nativeRef, ksize, borderType);
}

+ (void)preCornerDetect:(Mat*)src dst:(Mat*)dst ksize:(int)ksize {
    cv::preCornerDetect(src.nativeRef, dst.nativeRef, ksize);
}


//
//  void cv::putText(Mat& img, String text, Point org, int fontFace, double fontScale, Scalar color, int thickness = 1, int lineType = LINE_8, bool bottomLeftOrigin = false)
//
+ (void)putText:(Mat*)img text:(NSString*)text org:(Point2i*)org fontFace:(int)fontFace fontScale:(double)fontScale color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType bottomLeftOrigin:(BOOL)bottomLeftOrigin {
    cv::putText(img.nativeRef, cv::String(text.UTF8String), org.nativeRef, fontFace, fontScale, color.nativeRef, thickness, lineType, bottomLeftOrigin);
}

+ (void)putText:(Mat*)img text:(NSString*)text org:(Point2i*)org fontFace:(int)fontFace fontScale:(double)fontScale color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::putText(img.nativeRef, cv::String(text.UTF8String), org.nativeRef, fontFace, fontScale, color.nativeRef, thickness, lineType);
}

+ (void)putText:(Mat*)img text:(NSString*)text org:(Point2i*)org fontFace:(int)fontFace fontScale:(double)fontScale color:(Scalar*)color thickness:(int)thickness {
    cv::putText(img.nativeRef, cv::String(text.UTF8String), org.nativeRef, fontFace, fontScale, color.nativeRef, thickness);
}

+ (void)putText:(Mat*)img text:(NSString*)text org:(Point2i*)org fontFace:(int)fontFace fontScale:(double)fontScale color:(Scalar*)color {
    cv::putText(img.nativeRef, cv::String(text.UTF8String), org.nativeRef, fontFace, fontScale, color.nativeRef);
}


//
//  void cv::pyrDown(Mat src, Mat& dst, Size dstsize = Size(), int borderType = BORDER_DEFAULT)
//
+ (void)pyrDown:(Mat*)src dst:(Mat*)dst dstsize:(Size2i*)dstsize borderType:(int)borderType {
    cv::pyrDown(src.nativeRef, dst.nativeRef, dstsize.nativeRef, borderType);
}

+ (void)pyrDown:(Mat*)src dst:(Mat*)dst dstsize:(Size2i*)dstsize {
    cv::pyrDown(src.nativeRef, dst.nativeRef, dstsize.nativeRef);
}

+ (void)pyrDown:(Mat*)src dst:(Mat*)dst {
    cv::pyrDown(src.nativeRef, dst.nativeRef);
}


//
//  void cv::pyrMeanShiftFiltering(Mat src, Mat& dst, double sp, double sr, int maxLevel = 1, TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1))
//
+ (void)pyrMeanShiftFiltering:(Mat*)src dst:(Mat*)dst sp:(double)sp sr:(double)sr maxLevel:(int)maxLevel termcrit:(TermCriteria*)termcrit {
    cv::pyrMeanShiftFiltering(src.nativeRef, dst.nativeRef, sp, sr, maxLevel, termcrit.nativeRef);
}

+ (void)pyrMeanShiftFiltering:(Mat*)src dst:(Mat*)dst sp:(double)sp sr:(double)sr maxLevel:(int)maxLevel {
    cv::pyrMeanShiftFiltering(src.nativeRef, dst.nativeRef, sp, sr, maxLevel);
}

+ (void)pyrMeanShiftFiltering:(Mat*)src dst:(Mat*)dst sp:(double)sp sr:(double)sr {
    cv::pyrMeanShiftFiltering(src.nativeRef, dst.nativeRef, sp, sr);
}


//
//  void cv::pyrUp(Mat src, Mat& dst, Size dstsize = Size(), int borderType = BORDER_DEFAULT)
//
+ (void)pyrUp:(Mat*)src dst:(Mat*)dst dstsize:(Size2i*)dstsize borderType:(int)borderType {
    cv::pyrUp(src.nativeRef, dst.nativeRef, dstsize.nativeRef, borderType);
}

+ (void)pyrUp:(Mat*)src dst:(Mat*)dst dstsize:(Size2i*)dstsize {
    cv::pyrUp(src.nativeRef, dst.nativeRef, dstsize.nativeRef);
}

+ (void)pyrUp:(Mat*)src dst:(Mat*)dst {
    cv::pyrUp(src.nativeRef, dst.nativeRef);
}


//
//  void cv::rectangle(Mat& img, Point pt1, Point pt2, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
//
+ (void)rectangle:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    cv::rectangle(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness, lineType, shift);
}

+ (void)rectangle:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::rectangle(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness, lineType);
}

+ (void)rectangle:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color thickness:(int)thickness {
    cv::rectangle(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef, thickness);
}

+ (void)rectangle:(Mat*)img pt1:(Point2i*)pt1 pt2:(Point2i*)pt2 color:(Scalar*)color {
    cv::rectangle(img.nativeRef, pt1.nativeRef, pt2.nativeRef, color.nativeRef);
}


//
//  void cv::rectangle(Mat& img, Rect rec, Scalar color, int thickness = 1, int lineType = LINE_8, int shift = 0)
//
+ (void)rectangle:(Mat*)img rec:(Rect2i*)rec color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    cv::rectangle(img.nativeRef, rec.nativeRef, color.nativeRef, thickness, lineType, shift);
}

+ (void)rectangle:(Mat*)img rec:(Rect2i*)rec color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::rectangle(img.nativeRef, rec.nativeRef, color.nativeRef, thickness, lineType);
}

+ (void)rectangle:(Mat*)img rec:(Rect2i*)rec color:(Scalar*)color thickness:(int)thickness {
    cv::rectangle(img.nativeRef, rec.nativeRef, color.nativeRef, thickness);
}

+ (void)rectangle:(Mat*)img rec:(Rect2i*)rec color:(Scalar*)color {
    cv::rectangle(img.nativeRef, rec.nativeRef, color.nativeRef);
}


//
//  void cv::remap(Mat src, Mat& dst, Mat map1, Mat map2, int interpolation, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
//
+ (void)remap:(Mat*)src dst:(Mat*)dst map1:(Mat*)map1 map2:(Mat*)map2 interpolation:(int)interpolation borderMode:(int)borderMode borderValue:(Scalar*)borderValue {
    cv::remap(src.nativeRef, dst.nativeRef, map1.nativeRef, map2.nativeRef, interpolation, borderMode, borderValue.nativeRef);
}

+ (void)remap:(Mat*)src dst:(Mat*)dst map1:(Mat*)map1 map2:(Mat*)map2 interpolation:(int)interpolation borderMode:(int)borderMode {
    cv::remap(src.nativeRef, dst.nativeRef, map1.nativeRef, map2.nativeRef, interpolation, borderMode);
}

+ (void)remap:(Mat*)src dst:(Mat*)dst map1:(Mat*)map1 map2:(Mat*)map2 interpolation:(int)interpolation {
    cv::remap(src.nativeRef, dst.nativeRef, map1.nativeRef, map2.nativeRef, interpolation);
}


//
//  void cv::resize(Mat src, Mat& dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR)
//
+ (void)resize:(Mat*)src dst:(Mat*)dst dsize:(Size2i*)dsize fx:(double)fx fy:(double)fy interpolation:(int)interpolation {
    cv::resize(src.nativeRef, dst.nativeRef, dsize.nativeRef, fx, fy, interpolation);
}

+ (void)resize:(Mat*)src dst:(Mat*)dst dsize:(Size2i*)dsize fx:(double)fx fy:(double)fy {
    cv::resize(src.nativeRef, dst.nativeRef, dsize.nativeRef, fx, fy);
}

+ (void)resize:(Mat*)src dst:(Mat*)dst dsize:(Size2i*)dsize fx:(double)fx {
    cv::resize(src.nativeRef, dst.nativeRef, dsize.nativeRef, fx);
}

+ (void)resize:(Mat*)src dst:(Mat*)dst dsize:(Size2i*)dsize {
    cv::resize(src.nativeRef, dst.nativeRef, dsize.nativeRef);
}


//
//  void cv::sepFilter2D(Mat src, Mat& dst, int ddepth, Mat kernelX, Mat kernelY, Point anchor = Point(-1,-1), double delta = 0, int borderType = BORDER_DEFAULT)
//
+ (void)sepFilter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernelX:(Mat*)kernelX kernelY:(Mat*)kernelY anchor:(Point2i*)anchor delta:(double)delta borderType:(int)borderType {
    cv::sepFilter2D(src.nativeRef, dst.nativeRef, ddepth, kernelX.nativeRef, kernelY.nativeRef, anchor.nativeRef, delta, borderType);
}

+ (void)sepFilter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernelX:(Mat*)kernelX kernelY:(Mat*)kernelY anchor:(Point2i*)anchor delta:(double)delta {
    cv::sepFilter2D(src.nativeRef, dst.nativeRef, ddepth, kernelX.nativeRef, kernelY.nativeRef, anchor.nativeRef, delta);
}

+ (void)sepFilter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernelX:(Mat*)kernelX kernelY:(Mat*)kernelY anchor:(Point2i*)anchor {
    cv::sepFilter2D(src.nativeRef, dst.nativeRef, ddepth, kernelX.nativeRef, kernelY.nativeRef, anchor.nativeRef);
}

+ (void)sepFilter2D:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth kernelX:(Mat*)kernelX kernelY:(Mat*)kernelY {
    cv::sepFilter2D(src.nativeRef, dst.nativeRef, ddepth, kernelX.nativeRef, kernelY.nativeRef);
}


//
//  void cv::spatialGradient(Mat src, Mat& dx, Mat& dy, int ksize = 3, int borderType = BORDER_DEFAULT)
//
+ (void)spatialGradient:(Mat*)src dx:(Mat*)dx dy:(Mat*)dy ksize:(int)ksize borderType:(int)borderType {
    cv::spatialGradient(src.nativeRef, dx.nativeRef, dy.nativeRef, ksize, borderType);
}

+ (void)spatialGradient:(Mat*)src dx:(Mat*)dx dy:(Mat*)dy ksize:(int)ksize {
    cv::spatialGradient(src.nativeRef, dx.nativeRef, dy.nativeRef, ksize);
}

+ (void)spatialGradient:(Mat*)src dx:(Mat*)dx dy:(Mat*)dy {
    cv::spatialGradient(src.nativeRef, dx.nativeRef, dy.nativeRef);
}


//
//  void cv::sqrBoxFilter(Mat src, Mat& dst, int ddepth, Size ksize, Point anchor = Point(-1, -1), bool normalize = true, int borderType = BORDER_DEFAULT)
//
+ (void)sqrBoxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize anchor:(Point2i*)anchor normalize:(BOOL)normalize borderType:(int)borderType {
    cv::sqrBoxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef, anchor.nativeRef, normalize, borderType);
}

+ (void)sqrBoxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize anchor:(Point2i*)anchor normalize:(BOOL)normalize {
    cv::sqrBoxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef, anchor.nativeRef, normalize);
}

+ (void)sqrBoxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize anchor:(Point2i*)anchor {
    cv::sqrBoxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef, anchor.nativeRef);
}

+ (void)sqrBoxFilter:(Mat*)src dst:(Mat*)dst ddepth:(int)ddepth ksize:(Size2i*)ksize {
    cv::sqrBoxFilter(src.nativeRef, dst.nativeRef, ddepth, ksize.nativeRef);
}


//
//  void cv::warpAffine(Mat src, Mat& dst, Mat M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
//
+ (void)warpAffine:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize flags:(int)flags borderMode:(int)borderMode borderValue:(Scalar*)borderValue {
    cv::warpAffine(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef, flags, borderMode, borderValue.nativeRef);
}

+ (void)warpAffine:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize flags:(int)flags borderMode:(int)borderMode {
    cv::warpAffine(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef, flags, borderMode);
}

+ (void)warpAffine:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize flags:(int)flags {
    cv::warpAffine(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef, flags);
}

+ (void)warpAffine:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize {
    cv::warpAffine(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef);
}


//
//  void cv::warpPerspective(Mat src, Mat& dst, Mat M, Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT, Scalar borderValue = Scalar())
//
+ (void)warpPerspective:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize flags:(int)flags borderMode:(int)borderMode borderValue:(Scalar*)borderValue {
    cv::warpPerspective(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef, flags, borderMode, borderValue.nativeRef);
}

+ (void)warpPerspective:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize flags:(int)flags borderMode:(int)borderMode {
    cv::warpPerspective(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef, flags, borderMode);
}

+ (void)warpPerspective:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize flags:(int)flags {
    cv::warpPerspective(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef, flags);
}

+ (void)warpPerspective:(Mat*)src dst:(Mat*)dst M:(Mat*)M dsize:(Size2i*)dsize {
    cv::warpPerspective(src.nativeRef, dst.nativeRef, M.nativeRef, dsize.nativeRef);
}


//
//  void cv::warpPolar(Mat src, Mat& dst, Size dsize, Point2f center, double maxRadius, int flags)
//
+ (void)warpPolar:(Mat*)src dst:(Mat*)dst dsize:(Size2i*)dsize center:(Point2f*)center maxRadius:(double)maxRadius flags:(int)flags {
    cv::warpPolar(src.nativeRef, dst.nativeRef, dsize.nativeRef, center.nativeRef, maxRadius, flags);
}


//
//  void cv::watershed(Mat image, Mat& markers)
//
+ (void)watershed:(Mat*)image markers:(Mat*)markers {
    cv::watershed(image.nativeRef, markers.nativeRef);
}


// C++: Size getTextSize(const String& text, int fontFace, double fontScale, int thickness, int* baseLine);
+ (Size2i*)getTextSize:(NSString*)text fontFace:(int)fontFace fontScale:(double)fontScale thickness:(int)thickness baseLine:(IntOut*)baseLine {

    cv::Size rsize = cv::getTextSize(cv::String(text.UTF8String), fontFace, fontScale, thickness, baseLine.ptr);

    return [Size2i fromNative: rsize];

}

@end


