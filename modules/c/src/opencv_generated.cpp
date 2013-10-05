#include "opencv_generated.hpp"
using namespace cv;
using namespace std;
using namespace flann;
using namespace cvflann;
extern "C" {
BFMatcher* cv_create_BFMatcher(int normType, bool crossCheck) {
	return new BFMatcher(normType, crossCheck);
}
BRISK* cv_create_BRISK(int thresh, int octaves, float patternScale) {
	return new BRISK(thresh, octaves, patternScale);
}
BRISK* cv_create_BRISK5(vector_float* radiusList, vector_int* numberList, float dMax, float dMin, vector_int* indexChange) {
	return new BRISK(*radiusList, *numberList, dMax, dMin, *indexChange);
}
RotatedRect* cv_CamShift(Mat* probImage, Rect* window, TermCriteria* criteria) {
	return new RotatedRect(cv::CamShift(*probImage, *window, *criteria));
}
void cv_Canny(Mat* image, Mat* edges, double threshold1, double threshold2, int apertureSize, bool L2gradient) {
	cv::Canny(*image, *edges, threshold1, threshold2, apertureSize, L2gradient);
}
CascadeClassifier* cv_create_CascadeClassifier() {
	return new CascadeClassifier();
}
CascadeClassifier* cv_create_CascadeClassifier1(String* filename) {
	return new CascadeClassifier(*filename);
}
CvANN_MLP* cv_create_CvANN_MLP() {
	return new CvANN_MLP();
}
CvANN_MLP* cv_create_CvANN_MLP4(Mat* layerSizes, int activateFunc, double fparam1, double fparam2) {
	return new CvANN_MLP(*layerSizes, activateFunc, fparam1, fparam2);
}
CvBoost* cv_create_CvBoost() {
	return new CvBoost();
}
CvBoost* cv_create_CvBoost8(Mat* trainData, int tflag, Mat* responses, Mat* varIdx, Mat* sampleIdx, Mat* varType, Mat* missingDataMask, CvBoostParams* params) {
	return new CvBoost(*trainData, tflag, *responses, *varIdx, *sampleIdx, *varType, *missingDataMask, *params);
}
CvDTree* cv_create_CvDTree() {
	return new CvDTree();
}
CvERTrees* cv_create_CvERTrees() {
	return new CvERTrees();
}
CvGBTrees* cv_create_CvGBTrees() {
	return new CvGBTrees();
}
CvGBTrees* cv_create_CvGBTrees8(Mat* trainData, int tflag, Mat* responses, Mat* varIdx, Mat* sampleIdx, Mat* varType, Mat* missingDataMask, CvGBTreesParams* params) {
	return new CvGBTrees(*trainData, tflag, *responses, *varIdx, *sampleIdx, *varType, *missingDataMask, *params);
}
CvKNearest* cv_create_CvKNearest() {
	return new CvKNearest();
}
CvKNearest* cv_create_CvKNearest5(Mat* trainData, Mat* responses, Mat* sampleIdx, bool isRegression, int max_k) {
	return new CvKNearest(*trainData, *responses, *sampleIdx, isRegression, max_k);
}
CvNormalBayesClassifier* cv_create_CvNormalBayesClassifier() {
	return new CvNormalBayesClassifier();
}
CvNormalBayesClassifier* cv_create_CvNormalBayesClassifier4(Mat* trainData, Mat* responses, Mat* varIdx, Mat* sampleIdx) {
	return new CvNormalBayesClassifier(*trainData, *responses, *varIdx, *sampleIdx);
}
CvRTrees* cv_create_CvRTrees() {
	return new CvRTrees();
}
CvSVM* cv_create_CvSVM() {
	return new CvSVM();
}
CvSVM* cv_create_CvSVM5(Mat* trainData, Mat* responses, Mat* varIdx, Mat* sampleIdx, CvSVMParams* params) {
	return new CvSVM(*trainData, *responses, *varIdx, *sampleIdx, *params);
}
EM* cv_create_EM(int nclusters, int covMatType, TermCriteria* termCrit) {
	return new EM(nclusters, covMatType, *termCrit);
}
FastFeatureDetector* cv_create_FastFeatureDetector(int threshold, bool nonmaxSuppression) {
	return new FastFeatureDetector(threshold, nonmaxSuppression);
}
FastFeatureDetector* cv_create_FastFeatureDetector3(int threshold, bool nonmaxSuppression, int type) {
	return new FastFeatureDetector(threshold, nonmaxSuppression, type);
}
FlannBasedMatcher* cv_create_FlannBasedMatcher(flann_IndexParams* indexParams, flann_SearchParams* searchParams) {
	return new FlannBasedMatcher(Ptr<flann_IndexParams>(indexParams), Ptr<flann_SearchParams>(searchParams));
}
GFTTDetector* cv_create_GFTTDetector(int maxCorners, double qualityLevel, double minDistance, int blockSize, bool useHarrisDetector, double k) {
	return new GFTTDetector(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k);
}
void cv_GaussianBlur(Mat* src, Mat* dst, Size* ksize, double sigmaX, double sigmaY, int borderType) {
	cv::GaussianBlur(*src, *dst, *ksize, sigmaX, sigmaY, borderType);
}
GridAdaptedFeatureDetector* cv_create_GridAdaptedFeatureDetector(FeatureDetector* detector, int maxTotalKeypoints, int gridRows, int gridCols) {
	return new GridAdaptedFeatureDetector(Ptr<FeatureDetector>(detector), maxTotalKeypoints, gridRows, gridCols);
}
HOGDescriptor* cv_create_HOGDescriptor() {
	return new HOGDescriptor();
}
HOGDescriptor* cv_create_HOGDescriptor1(String* filename) {
	return new HOGDescriptor(*filename);
}
HOGDescriptor* cv_create_HOGDescriptor11(Size* _winSize, Size* _blockSize, Size* _blockStride, Size* _cellSize, int _nbins, int _derivAperture, double _winSigma, int _histogramNormType, double _L2HysThreshold, bool _gammaCorrection, int _nlevels) {
	return new HOGDescriptor(*_winSize, *_blockSize, *_blockStride, *_cellSize, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold, _gammaCorrection, _nlevels);
}
void cv_HoughCircles(Mat* image, Mat* circles, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius) {
	cv::HoughCircles(*image, *circles, method, dp, minDist, param1, param2, minRadius, maxRadius);
}
void cv_HoughLines(Mat* image, Mat* lines, double rho, double theta, int threshold, double srn, double stn) {
	cv::HoughLines(*image, *lines, rho, theta, threshold, srn, stn);
}
void cv_HoughLinesP(Mat* image, Mat* lines, double rho, double theta, int threshold, double minLineLength, double maxLineGap) {
	cv::HoughLinesP(*image, *lines, rho, theta, threshold, minLineLength, maxLineGap);
}
void cv_HuMoments(Moments* m, Mat* hu) {
	cv::HuMoments(*m, *hu);
}
Index* cv_create_Index() {
	return new Index();
}
Index* cv_create_Index3(Mat* features, IndexParams* params, cvflann_flann_distance_t* distType) {
	return new Index(*features, *params, *distType);
}
KDTree* cv_create_KDTree() {
	return new KDTree();
}
KDTree* cv_create_KDTree2(Mat* points, bool copyAndReorderPoints) {
	return new KDTree(*points, copyAndReorderPoints);
}
KDTree* cv_create_KDTree3(Mat* points, Mat* _labels, bool copyAndReorderPoints) {
	return new KDTree(*points, *_labels, copyAndReorderPoints);
}
KalmanFilter* cv_create_KalmanFilter() {
	return new KalmanFilter();
}
KalmanFilter* cv_create_KalmanFilter4(int dynamParams, int measureParams, int controlParams, int type) {
	return new KalmanFilter(dynamParams, measureParams, controlParams, type);
}
void cv_LUT(Mat* src, Mat* lut, Mat* dst) {
	cv::LUT(*src, *lut, *dst);
}
void cv_Laplacian(Mat* src, Mat* dst, int ddepth, int ksize, double scale, double delta, int borderType) {
	cv::Laplacian(*src, *dst, ddepth, ksize, scale, delta, borderType);
}
MSER* cv_create_MSER(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size) {
	return new MSER(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin, _edge_blur_size);
}
double cv_Mahalanobis(Mat* v1, Mat* v2, Mat* icovar) {
	return cv::Mahalanobis(*v1, *v2, *icovar);
}
ORB* cv_create_ORB(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize) {
	return new ORB(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);
}
void cv_PCABackProject(Mat* data, Mat* mean, Mat* eigenvectors, Mat* result) {
	cv::PCABackProject(*data, *mean, *eigenvectors, *result);
}
void cv_PCACompute(Mat* data, Mat* mean, Mat* eigenvectors, int maxComponents) {
	cv::PCACompute(*data, *mean, *eigenvectors, maxComponents);
}
void cv_PCACompute4(Mat* data, Mat* mean, Mat* eigenvectors, double retainedVariance) {
	cv::PCACompute(*data, *mean, *eigenvectors, retainedVariance);
}
void cv_PCAProject(Mat* data, Mat* mean, Mat* eigenvectors, Mat* result) {
	cv::PCAProject(*data, *mean, *eigenvectors, *result);
}
double cv_PSNR(Mat* src1, Mat* src2) {
	return cv::PSNR(*src1, *src2);
}
Params* cv_create_Params() {
	return new Params();
}
PyramidAdaptedFeatureDetector* cv_create_PyramidAdaptedFeatureDetector(FeatureDetector* detector, int maxLevel) {
	return new PyramidAdaptedFeatureDetector(Ptr<FeatureDetector>(detector), maxLevel);
}
Vec3d* cv_RQDecomp3x3(Mat* src, Mat* mtxR, Mat* mtxQ, Mat* Qx, Mat* Qy, Mat* Qz) {
	return new Vec3d(cv::RQDecomp3x3(*src, *mtxR, *mtxQ, *Qx, *Qy, *Qz));
}
void cv_Rodrigues(Mat* src, Mat* dst, Mat* jacobian) {
	cv::Rodrigues(*src, *dst, *jacobian);
}
void cv_SVBackSubst(Mat* w, Mat* u, Mat* vt, Mat* rhs, Mat* dst) {
	cv::SVBackSubst(*w, *u, *vt, *rhs, *dst);
}
void cv_SVDecomp(Mat* src, Mat* w, Mat* u, Mat* vt, int flags) {
	cv::SVDecomp(*src, *w, *u, *vt, flags);
}
void cv_Scharr(Mat* src, Mat* dst, int ddepth, int dx, int dy, double scale, double delta, int borderType) {
	cv::Scharr(*src, *dst, ddepth, dx, dy, scale, delta, borderType);
}
SimpleBlobDetector* cv_create_SimpleBlobDetector(SimpleBlobDetector_Params* parameters) {
	return new SimpleBlobDetector(*parameters);
}
void cv_Sobel(Mat* src, Mat* dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType) {
	cv::Sobel(*src, *dst, ddepth, dx, dy, ksize, scale, delta, borderType);
}
StarDetector* cv_create_StarDetector(int _maxSize, int _responseThreshold, int _lineThresholdProjected, int _lineThresholdBinarized, int _suppressNonmaxSize) {
	return new StarDetector(_maxSize, _responseThreshold, _lineThresholdProjected, _lineThresholdBinarized, _suppressNonmaxSize);
}
StereoVar* cv_create_StereoVar() {
	return new StereoVar();
}
StereoVar* cv_create_StereoVar12(int levels, double pyrScale, int nIt, int minDisp, int maxDisp, int poly_n, double poly_sigma, float fi, float lambda, int penalization, int cycle, int flags) {
	return new StereoVar(levels, pyrScale, nIt, minDisp, maxDisp, poly_n, poly_sigma, fi, lambda, penalization, cycle, flags);
}
Subdiv2D* cv_create_Subdiv2D() {
	return new Subdiv2D();
}
Subdiv2D* cv_create_Subdiv2D1(Rect* rect) {
	return new Subdiv2D(*rect);
}
VideoCapture* cv_create_VideoCapture() {
	return new VideoCapture();
}
VideoCapture* cv_create_VideoCapture1(int device) {
	return new VideoCapture(device);
}
VideoWriter* cv_create_VideoWriter() {
	return new VideoWriter();
}
VideoWriter* cv_create_VideoWriter5(String* filename, int fourcc, double fps, Size* frameSize, bool isColor) {
	return new VideoWriter(*filename, fourcc, fps, *frameSize, isColor);
}
Algorithm* cv_Algorithm__create(Algorithm* self, String* name) {
	return &*self->_create(*name);
}
void cv_absdiff(Mat* src1, Mat* src2, Mat* dst) {
	cv::absdiff(*src1, *src2, *dst);
}
void cv_accumulate(Mat* src, Mat* dst, Mat* mask) {
	cv::accumulate(*src, *dst, *mask);
}
void cv_accumulateProduct(Mat* src1, Mat* src2, Mat* dst, Mat* mask) {
	cv::accumulateProduct(*src1, *src2, *dst, *mask);
}
void cv_accumulateSquare(Mat* src, Mat* dst, Mat* mask) {
	cv::accumulateSquare(*src, *dst, *mask);
}
void cv_accumulateWeighted(Mat* src, Mat* dst, double alpha, Mat* mask) {
	cv::accumulateWeighted(*src, *dst, alpha, *mask);
}
void cv_adaptiveBilateralFilter(Mat* src, Mat* dst, Size* ksize, double sigmaSpace, Point* anchor, int borderType) {
	cv::adaptiveBilateralFilter(*src, *dst, *ksize, sigmaSpace, *anchor, borderType);
}
void cv_adaptiveThreshold(Mat* src, Mat* dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C) {
	cv::adaptiveThreshold(*src, *dst, maxValue, adaptiveMethod, thresholdType, blockSize, C);
}
void cv_add(Mat* src1, Mat* src2, Mat* dst, Mat* mask, int dtype) {
	cv::add(*src1, *src2, *dst, *mask, dtype);
}
void cv_DescriptorMatcher_add1(DescriptorMatcher* self, vector_Mat* descriptors) {
	self->add(*descriptors);
}
void cv_addWeighted(Mat* src1, double alpha, Mat* src2, double beta, double gamma, Mat* dst, int dtype) {
	cv::addWeighted(*src1, alpha, *src2, beta, gamma, *dst, dtype);
}
void cv_CLAHE_apply(CLAHE* self, Mat* src, Mat* dst) {
	self->apply(*src, *dst);
}
void cv_BackgroundSubtractor_apply3(BackgroundSubtractor* self, Mat* image, Mat* fgmask, double learningRate) {
	self->apply(*image, *fgmask, learningRate);
}
void cv_applyColorMap(Mat* src, Mat* dst, int colormap) {
	cv::applyColorMap(*src, *dst, colormap);
}
void cv_approxPolyDP(Mat* curve, Mat* approxCurve, double epsilon, bool closed) {
	cv::approxPolyDP(*curve, *approxCurve, epsilon, closed);
}
double cv_arcLength(Mat* curve, bool closed) {
	return cv::arcLength(*curve, closed);
}
void cv_batchDistance(Mat* src1, Mat* src2, Mat* dist, int dtype, Mat* nidx, int normType, int K, Mat* mask, int update, bool crosscheck) {
	cv::batchDistance(*src1, *src2, *dist, dtype, *nidx, normType, K, *mask, update, crosscheck);
}
void cv_bilateralFilter(Mat* src, Mat* dst, int d, double sigmaColor, double sigmaSpace, int borderType) {
	cv::bilateralFilter(*src, *dst, d, sigmaColor, sigmaSpace, borderType);
}
void cv_bitwise_and(Mat* src1, Mat* src2, Mat* dst, Mat* mask) {
	cv::bitwise_and(*src1, *src2, *dst, *mask);
}
void cv_bitwise_not(Mat* src, Mat* dst, Mat* mask) {
	cv::bitwise_not(*src, *dst, *mask);
}
void cv_bitwise_or(Mat* src1, Mat* src2, Mat* dst, Mat* mask) {
	cv::bitwise_or(*src1, *src2, *dst, *mask);
}
void cv_bitwise_xor(Mat* src1, Mat* src2, Mat* dst, Mat* mask) {
	cv::bitwise_xor(*src1, *src2, *dst, *mask);
}
void cv_blur(Mat* src, Mat* dst, Size* ksize, Point* anchor, int borderType) {
	cv::blur(*src, *dst, *ksize, *anchor, borderType);
}
int cv_borderInterpolate(int p, int len, int borderType) {
	return cv::borderInterpolate(p, len, borderType);
}
Rect* cv_boundingRect(Mat* points) {
	return new Rect(cv::boundingRect(*points));
}
void cv_boxFilter(Mat* src, Mat* dst, int ddepth, Size* ksize, Point* anchor, bool normalize, int borderType) {
	cv::boxFilter(*src, *dst, ddepth, *ksize, *anchor, normalize, borderType);
}
void cv_boxPoints(RotatedRect* box, Mat* points) {
	cv::boxPoints(*box, *points);
}
void cv_KDTree_build(KDTree* self, Mat* points, bool copyAndReorderPoints) {
	self->build(*points, copyAndReorderPoints);
}
void cv_Index_build3(Index* self, Mat* features, IndexParams* params, cvflann_flann_distance_t* distType) {
	self->build(*features, *params, *distType);
}
int cv_buildOpticalFlowPyramid(Mat* img, vector_Mat* pyramid, Size* winSize, int maxLevel, bool withDerivatives, int pyrBorder, int derivBorder, bool tryReuseInputImage) {
	return cv::buildOpticalFlowPyramid(*img, *pyramid, *winSize, maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage);
}
void cv_calcBackProject(vector_Mat* images, vector_int* channels, Mat* hist, Mat* dst, vector_float* ranges, double scale) {
	cv::calcBackProject(*images, *channels, *hist, *dst, *ranges, scale);
}
void cv_calcCovarMatrix(Mat* samples, Mat* covar, Mat* mean, int flags, int ctype) {
	cv::calcCovarMatrix(*samples, *covar, *mean, flags, ctype);
}
double cv_calcGlobalOrientation(Mat* orientation, Mat* mask, Mat* mhi, double timestamp, double duration) {
	return cv::calcGlobalOrientation(*orientation, *mask, *mhi, timestamp, duration);
}
void cv_calcHist(vector_Mat* images, vector_int* channels, Mat* mask, Mat* hist, vector_int* histSize, vector_float* ranges, bool accumulate) {
	cv::calcHist(*images, *channels, *mask, *hist, *histSize, *ranges, accumulate);
}
void cv_calcMotionGradient(Mat* mhi, Mat* mask, Mat* orientation, double delta1, double delta2, int apertureSize) {
	cv::calcMotionGradient(*mhi, *mask, *orientation, delta1, delta2, apertureSize);
}
void cv_calcOpticalFlowFarneback(Mat* prev, Mat* next, Mat* flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags) {
	cv::calcOpticalFlowFarneback(*prev, *next, *flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
}
void cv_calcOpticalFlowPyrLK(Mat* prevImg, Mat* nextImg, Mat* prevPts, Mat* nextPts, Mat* status, Mat* err, Size* winSize, int maxLevel, TermCriteria* criteria, int flags, double minEigThreshold) {
	cv::calcOpticalFlowPyrLK(*prevImg, *nextImg, *prevPts, *nextPts, *status, *err, *winSize, maxLevel, *criteria, flags, minEigThreshold);
}
void cv_calcOpticalFlowSF(Mat* from, Mat* to, Mat* flow, int layers, int averaging_block_size, int max_flow) {
	cv::calcOpticalFlowSF(*from, *to, *flow, layers, averaging_block_size, max_flow);
}
void cv_calcOpticalFlowSF16(Mat* from, Mat* to, Mat* flow, int layers, int averaging_block_size, int max_flow, double sigma_dist, double sigma_color, int postprocess_window, double sigma_dist_fix, double sigma_color_fix, double occ_thr, int upscale_averaging_radius, double upscale_sigma_dist, double upscale_sigma_color, double speed_up_thr) {
	cv::calcOpticalFlowSF(*from, *to, *flow, layers, averaging_block_size, max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, upscale_averaging_radius, upscale_sigma_dist, upscale_sigma_color, speed_up_thr);
}
double cv_calibrateCamera(vector_Mat* objectPoints, vector_Mat* imagePoints, Size* imageSize, Mat* cameraMatrix, Mat* distCoeffs, vector_Mat* rvecs, vector_Mat* tvecs, int flags, TermCriteria* criteria) {
	return cv::calibrateCamera(*objectPoints, *imagePoints, *imageSize, *cameraMatrix, *distCoeffs, *rvecs, *tvecs, flags, *criteria);
}
void cv_calibrationMatrixValues(Mat* cameraMatrix, Size* imageSize, double apertureWidth, double apertureHeight, double fovx, double fovy, double focalLength, Point2d* principalPoint, double aspectRatio) {
	cv::calibrationMatrixValues(*cameraMatrix, *imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, *principalPoint, aspectRatio);
}
void cv_cartToPolar(Mat* x, Mat* y, Mat* magnitude, Mat* angle, bool angleInDegrees) {
	cv::cartToPolar(*x, *y, *magnitude, *angle, angleInDegrees);
}
int cv_chamerMatching(Mat* img, Mat* templ, vector_vector_Point* results, vector_float* cost, double templScale, int maxMatches, double minMatchDistance, int padX, int padY, int scales, double minScale, double maxScale, double orientationWeight, double truncate) {
	return cv::chamerMatching(*img, *templ, *results, *cost, templScale, maxMatches, minMatchDistance, padX, padY, scales, minScale, maxScale, orientationWeight, truncate);
}
bool cv_HOGDescriptor_checkDetectorSize(HOGDescriptor* self) {
	return self->checkDetectorSize();
}
bool cv_checkRange(Mat* a, bool quiet, Point* pos, double minVal, double maxVal) {
	return cv::checkRange(*a, quiet, pos, minVal, maxVal);
}
void cv_circle(Mat* img, Point* center, int radius, Scalar* color, int thickness, int lineType, int shift) {
	cv::circle(*img, *center, radius, *color, thickness, lineType, shift);
}
void cv_CvNormalBayesClassifier_clear(CvNormalBayesClassifier* self) {
	self->clear();
}
void cv_DescriptorMatcher_clear0(DescriptorMatcher* self) {
	self->clear();
}
bool cv_clipLine(Rect* imgRect, Point* pt1, Point* pt2) {
	return cv::clipLine(*imgRect, *pt1, *pt2);
}
void cv_compare(Mat* src1, Mat* src2, Mat* dst, int cmpop) {
	cv::compare(*src1, *src2, *dst, cmpop);
}
double cv_compareHist(Mat* H1, Mat* H2, int method) {
	return cv::compareHist(*H1, *H2, method);
}
int cv_LineSegmentDetector_compareSegments(LineSegmentDetector* self, Size* size, Mat* lines1, Mat* lines2, Mat* _image) {
	return self->compareSegments(*size, *lines1, *lines2, *_image);
}
void cv_completeSymm(Mat* mtx, bool lowerToUpper) {
	cv::completeSymm(*mtx, lowerToUpper);
}
void cv_composeRT(Mat* rvec1, Mat* tvec1, Mat* rvec2, Mat* tvec2, Mat* rvec3, Mat* tvec3, Mat* dr3dr1, Mat* dr3dt1, Mat* dr3dr2, Mat* dr3dt2, Mat* dt3dr1, Mat* dt3dt1, Mat* dt3dr2, Mat* dt3dt2) {
	cv::composeRT(*rvec1, *tvec1, *rvec2, *tvec2, *rvec3, *tvec3, *dr3dr1, *dr3dt1, *dr3dr2, *dr3dt2, *dt3dr1, *dt3dt1, *dt3dr2, *dt3dt2);
}
void cv_StereoMatcher_compute(StereoMatcher* self, Mat* left, Mat* right, Mat* disparity) {
	self->compute(*left, *right, *disparity);
}
void cv_Feature2D_compute3(Feature2D* self, Mat* image, vector_KeyPoint* keypoints, Mat* descriptors) {
	self->compute(*image, *keypoints, *descriptors);
}
void cv_HOGDescriptor_compute5(HOGDescriptor* self, Mat* img, vector_float* descriptors, Size* winStride, Size* padding, vector_Point* locations) {
	self->compute(*img, *descriptors, *winStride, *padding, *locations);
}
void cv_computeCorrespondEpilines(Mat* points, int whichImage, Mat* F, Mat* lines) {
	cv::computeCorrespondEpilines(*points, whichImage, *F, *lines);
}
void cv_HOGDescriptor_computeGradient(HOGDescriptor* self, Mat* img, Mat* grad, Mat* angleOfs, Size* paddingTL, Size* paddingBR) {
	self->computeGradient(*img, *grad, *angleOfs, *paddingTL, *paddingBR);
}
int cv_connectedComponents(Mat* image, Mat* labels, int connectivity, int ltype) {
	return cv::connectedComponents(*image, *labels, connectivity, ltype);
}
int cv_connectedComponentsWithStats(Mat* image, Mat* labels, Mat* stats, Mat* centroids, int connectivity, int ltype) {
	return cv::connectedComponentsWithStats(*image, *labels, *stats, *centroids, connectivity, ltype);
}
double cv_contourArea(Mat* contour, bool oriented) {
	return cv::contourArea(*contour, oriented);
}
void cv_convertMaps(Mat* map1, Mat* map2, Mat* dstmap1, Mat* dstmap2, int dstmap1type, bool nninterpolation) {
	cv::convertMaps(*map1, *map2, *dstmap1, *dstmap2, dstmap1type, nninterpolation);
}
void cv_convertPointsFromHomogeneous(Mat* src, Mat* dst) {
	cv::convertPointsFromHomogeneous(*src, *dst);
}
void cv_convertPointsToHomogeneous(Mat* src, Mat* dst) {
	cv::convertPointsToHomogeneous(*src, *dst);
}
void cv_convertScaleAbs(Mat* src, Mat* dst, double alpha, double beta) {
	cv::convertScaleAbs(*src, *dst, alpha, beta);
}
void cv_convexHull(Mat* points, Mat* hull, bool clockwise, bool returnPoints) {
	cv::convexHull(*points, *hull, clockwise, returnPoints);
}
void cv_convexityDefects(Mat* contour, Mat* convexhull, Mat* convexityDefects) {
	cv::convexityDefects(*contour, *convexhull, *convexityDefects);
}
void cv_copyMakeBorder(Mat* src, Mat* dst, int top, int bottom, int left, int right, int borderType, Scalar* value) {
	cv::copyMakeBorder(*src, *dst, top, bottom, left, right, borderType, *value);
}
void cv_cornerEigenValsAndVecs(Mat* src, Mat* dst, int blockSize, int ksize, int borderType) {
	cv::cornerEigenValsAndVecs(*src, *dst, blockSize, ksize, borderType);
}
void cv_cornerHarris(Mat* src, Mat* dst, int blockSize, int ksize, double k, int borderType) {
	cv::cornerHarris(*src, *dst, blockSize, ksize, k, borderType);
}
void cv_cornerMinEigenVal(Mat* src, Mat* dst, int blockSize, int ksize, int borderType) {
	cv::cornerMinEigenVal(*src, *dst, blockSize, ksize, borderType);
}
void cv_cornerSubPix(Mat* image, Mat* corners, Size* winSize, Size* zeroZone, TermCriteria* criteria) {
	cv::cornerSubPix(*image, *corners, *winSize, *zeroZone, *criteria);
}
Mat* cv_KalmanFilter_correct(KalmanFilter* self, Mat* measurement) {
	return new Mat(self->correct(*measurement));
}
void cv_correctMatches(Mat* F, Mat* points1, Mat* points2, Mat* newPoints1, Mat* newPoints2) {
	cv::correctMatches(*F, *points1, *points2, *newPoints1, *newPoints2);
}
int cv_countNonZero(Mat* src) {
	return cv::countNonZero(*src);
}
void cv_CvANN_MLP_create(CvANN_MLP* self, Mat* layerSizes, int activateFunc, double fparam1, double fparam2) {
	self->create(*layerSizes, activateFunc, fparam1, fparam2);
}
DescriptorMatcher* cv_DescriptorMatcher_create1(DescriptorMatcher* self, String* descriptorMatcherType) {
	return &*self->create(*descriptorMatcherType);
}
BackgroundSubtractorGMG* cv_createBackgroundSubtractorGMG(int initializationFrames, double decisionThreshold) {
	return &*cv::createBackgroundSubtractorGMG(initializationFrames, decisionThreshold);
}
BackgroundSubtractorMOG* cv_createBackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma) {
	return &*cv::createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma);
}
BackgroundSubtractorMOG2* cv_createBackgroundSubtractorMOG2(int history, double varThreshold, bool detectShadows) {
	return &*cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
}
CLAHE* cv_createCLAHE(double clipLimit, Size* tileGridSize) {
	return &*cv::createCLAHE(clipLimit, *tileGridSize);
}
FaceRecognizer* cv_createEigenFaceRecognizer(int num_components, double threshold) {
	return &*cv::createEigenFaceRecognizer(num_components, threshold);
}
FaceRecognizer* cv_createFisherFaceRecognizer(int num_components, double threshold) {
	return &*cv::createFisherFaceRecognizer(num_components, threshold);
}
void cv_createHanningWindow(Mat* dst, Size* winSize, int type) {
	cv::createHanningWindow(*dst, *winSize, type);
}
FaceRecognizer* cv_createLBPHFaceRecognizer(int radius, int neighbors, int grid_x, int grid_y, double threshold) {
	return &*cv::createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold);
}
LineSegmentDetector* cv_createLineSegmentDetector(int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th, double _log_eps, double _density_th, int _n_bins) {
	return &*cv::createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th, _n_bins);
}
StereoBM* cv_createStereoBM(int numDisparities, int blockSize) {
	return &*cv::createStereoBM(numDisparities, blockSize);
}
StereoSGBM* cv_createStereoSGBM(int minDisparity, int numDisparities, int blockSize, int P1, int P2, int disp12MaxDiff, int preFilterCap, int uniquenessRatio, int speckleWindowSize, int speckleRange, int mode) {
	return &*cv::createStereoSGBM(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
}
void cv_cvtColor(Mat* src, Mat* dst, int code, int dstCn) {
	cv::cvtColor(*src, *dst, code, dstCn);
}
void cv_dct(Mat* src, Mat* dst, int flags) {
	cv::dct(*src, *dst, flags);
}
void cv_decomposeEssentialMat(Mat* E, Mat* R1, Mat* R2, Mat* t) {
	cv::decomposeEssentialMat(*E, *R1, *R2, *t);
}
void cv_decomposeProjectionMatrix(Mat* projMatrix, Mat* cameraMatrix, Mat* rotMatrix, Mat* transVect, Mat* rotMatrixX, Mat* rotMatrixY, Mat* rotMatrixZ, Mat* eulerAngles) {
	cv::decomposeProjectionMatrix(*projMatrix, *cameraMatrix, *rotMatrix, *transVect, *rotMatrixX, *rotMatrixY, *rotMatrixZ, *eulerAngles);
}
void cv_demosaicing(Mat* _src, Mat* _dst, int code, int dcn) {
	cv::demosaicing(*_src, *_dst, code, dcn);
}
void cv_destroyAllWindows() {
	cv::destroyAllWindows();
}
void cv_destroyWindow(String* winname) {
	cv::destroyWindow(*winname);
}
void cv_LineSegmentDetector_detect(LineSegmentDetector* self, Mat* _image, Mat* _lines, Mat* width, Mat* prec, Mat* nfa) {
	self->detect(*_image, *_lines, *width, *prec, *nfa);
}
void cv_FeatureDetector_detect3(FeatureDetector* self, Mat* image, vector_KeyPoint* keypoints, Mat* mask) {
	self->detect(*image, *keypoints, *mask);
}
void cv_HOGDescriptor_detect7(HOGDescriptor* self, Mat* img, vector_Point* foundLocations, vector_double* weights, double hitThreshold, Size* winStride, Size* padding, vector_Point* searchLocations) {
	self->detect(*img, *foundLocations, *weights, hitThreshold, *winStride, *padding, *searchLocations);
}
void cv_CascadeClassifier_detectMultiScale(CascadeClassifier* self, Mat* image, vector_Rect* objects, double scaleFactor, int minNeighbors, int flags, Size* minSize, Size* maxSize) {
	self->detectMultiScale(*image, *objects, scaleFactor, minNeighbors, flags, *minSize, *maxSize);
}
void cv_CascadeClassifier_detectMultiScale10(CascadeClassifier* self, Mat* image, vector_Rect* objects, vector_int* rejectLevels, vector_double* levelWeights, double scaleFactor, int minNeighbors, int flags, Size* minSize, Size* maxSize, bool outputRejectLevels) {
	self->detectMultiScale(*image, *objects, *rejectLevels, *levelWeights, scaleFactor, minNeighbors, flags, *minSize, *maxSize, outputRejectLevels);
}
void cv_CascadeClassifier_detectMultiScale8(CascadeClassifier* self, Mat* image, vector_Rect* objects, vector_int* numDetections, double scaleFactor, int minNeighbors, int flags, Size* minSize, Size* maxSize) {
	self->detectMultiScale(*image, *objects, *numDetections, scaleFactor, minNeighbors, flags, *minSize, *maxSize);
}
void cv_HOGDescriptor_detectMultiScale9(HOGDescriptor* self, Mat* img, vector_Rect* foundLocations, vector_double* foundWeights, double hitThreshold, Size* winStride, Size* padding, double scale, double finalThreshold, bool useMeanshiftGrouping) {
	self->detectMultiScale(*img, *foundLocations, *foundWeights, hitThreshold, *winStride, *padding, scale, finalThreshold, useMeanshiftGrouping);
}
double cv_determinant(Mat* mtx) {
	return cv::determinant(*mtx);
}
void cv_dft(Mat* src, Mat* dst, int flags, int nonzeroRows) {
	cv::dft(*src, *dst, flags, nonzeroRows);
}
void cv_dilate(Mat* src, Mat* dst, Mat* kernel, Point* anchor, int iterations, int borderType, Scalar* borderValue) {
	cv::dilate(*src, *dst, *kernel, *anchor, iterations, borderType, *borderValue);
}
int cv_KDTree_dims(KDTree* self) {
	return self->dims();
}
void cv_distanceTransform(Mat* src, Mat* dst, Mat* labels, int distanceType, int maskSize, int labelType) {
	cv::distanceTransform(*src, *dst, *labels, distanceType, maskSize, labelType);
}
void cv_distanceTransform4(Mat* src, Mat* dst, int distanceType, int maskSize) {
	cv::distanceTransform(*src, *dst, distanceType, maskSize);
}
void cv_divide(Mat* src1, Mat* src2, Mat* dst, double scale, int dtype) {
	cv::divide(*src1, *src2, *dst, scale, dtype);
}
void cv_divide4(double scale, Mat* src2, Mat* dst, int dtype) {
	cv::divide(scale, *src2, *dst, dtype);
}
void cv_drawChessboardCorners(Mat* image, Size* patternSize, Mat* corners, bool patternWasFound) {
	cv::drawChessboardCorners(*image, *patternSize, *corners, patternWasFound);
}
void cv_drawContours(Mat* image, vector_Mat* contours, int contourIdx, Scalar* color, int thickness, int lineType, Mat* hierarchy, int maxLevel, Point* offset) {
	cv::drawContours(*image, *contours, contourIdx, *color, thickness, lineType, *hierarchy, maxLevel, *offset);
}
void cv_drawDataMatrixCodes(Mat* image, vector_String* codes, Mat* corners) {
	cv::drawDataMatrixCodes(*image, *codes, *corners);
}
void cv_drawKeypoints(Mat* image, vector_KeyPoint* keypoints, Mat* outImage, Scalar* color, int flags) {
	cv::drawKeypoints(*image, *keypoints, *outImage, *color, flags);
}
void cv_drawMatches(Mat* img1, vector_KeyPoint* keypoints1, Mat* img2, vector_KeyPoint* keypoints2, vector_DMatch* matches1to2, Mat* outImg, Scalar* matchColor, Scalar* singlePointColor, vector_char* matchesMask, int flags) {
	cv::drawMatches(*img1, *keypoints1, *img2, *keypoints2, *matches1to2, *outImg, *matchColor, *singlePointColor, *matchesMask, flags);
}
void cv_drawMatches10(Mat* img1, vector_KeyPoint* keypoints1, Mat* img2, vector_KeyPoint* keypoints2, vector_vector_DMatch* matches1to2, Mat* outImg, Scalar* matchColor, Scalar* singlePointColor, vector_vector_char* matchesMask, int flags) {
	cv::drawMatches(*img1, *keypoints1, *img2, *keypoints2, *matches1to2, *outImg, *matchColor, *singlePointColor, *matchesMask, flags);
}
void cv_LineSegmentDetector_drawSegments(LineSegmentDetector* self, Mat* _image, Mat* lines) {
	self->drawSegments(*_image, *lines);
}
int cv_Subdiv2D_edgeDst(Subdiv2D* self, int edge, Point2f* dstpt) {
	return self->edgeDst(edge, dstpt);
}
int cv_Subdiv2D_edgeOrg(Subdiv2D* self, int edge, Point2f* orgpt) {
	return self->edgeOrg(edge, orgpt);
}
bool cv_eigen(Mat* src, Mat* eigenvalues, Mat* eigenvectors) {
	return cv::eigen(*src, *eigenvalues, *eigenvectors);
}
void cv_ellipse(Mat* img, Point* center, Size* axes, double angle, double startAngle, double endAngle, Scalar* color, int thickness, int lineType, int shift) {
	cv::ellipse(*img, *center, *axes, angle, startAngle, endAngle, *color, thickness, lineType, shift);
}
void cv_ellipse2Poly(Point* center, Size* axes, int angle, int arcStart, int arcEnd, int delta, vector_Point* pts) {
	cv::ellipse2Poly(*center, *axes, angle, arcStart, arcEnd, delta, *pts);
}
void cv_ellipse5(Mat* img, RotatedRect* box, Scalar* color, int thickness, int lineType) {
	cv::ellipse(*img, *box, *color, thickness, lineType);
}
bool cv_FeatureDetector_empty(FeatureDetector* self) {
	return self->empty();
}
bool cv_CascadeClassifier_empty0(CascadeClassifier* self) {
	return self->empty();
}
void cv_equalizeHist(Mat* src, Mat* dst) {
	cv::equalizeHist(*src, *dst);
}
void cv_erode(Mat* src, Mat* dst, Mat* kernel, Point* anchor, int iterations, int borderType, Scalar* borderValue) {
	cv::erode(*src, *dst, *kernel, *anchor, iterations, borderType, *borderValue);
}
int cv_estimateAffine3D(Mat* src, Mat* dst, Mat* out, Mat* inliers, double ransacThreshold, double confidence) {
	return cv::estimateAffine3D(*src, *dst, *out, *inliers, ransacThreshold, confidence);
}
Mat* cv_estimateRigidTransform(Mat* src, Mat* dst, bool fullAffine) {
	return new Mat(cv::estimateRigidTransform(*src, *dst, fullAffine));
}
void cv_exp(Mat* src, Mat* dst) {
	cv::exp(*src, *dst);
}
void cv_extractChannel(Mat* src, Mat* dst, int coi) {
	cv::extractChannel(*src, *dst, coi);
}
void cv_fillConvexPoly(Mat* img, Mat* points, Scalar* color, int lineType, int shift) {
	cv::fillConvexPoly(*img, *points, *color, lineType, shift);
}
void cv_fillPoly(Mat* img, vector_Mat* pts, Scalar* color, int lineType, int shift, Point* offset) {
	cv::fillPoly(*img, *pts, *color, lineType, shift, *offset);
}
void cv_filter2D(Mat* src, Mat* dst, int ddepth, Mat* kernel, Point* anchor, double delta, int borderType) {
	cv::filter2D(*src, *dst, ddepth, *kernel, *anchor, delta, borderType);
}
void cv_filterSpeckles(Mat* img, double newVal, int maxSpeckleSize, double maxDiff, Mat* buf) {
	cv::filterSpeckles(*img, newVal, maxSpeckleSize, maxDiff, *buf);
}
bool cv_findChessboardCorners(Mat* image, Size* patternSize, Mat* corners, int flags) {
	return cv::findChessboardCorners(*image, *patternSize, *corners, flags);
}
bool cv_findCirclesGrid(Mat* image, Size* patternSize, Mat* centers, int flags, FeatureDetector* blobDetector) {
	return cv::findCirclesGrid(*image, *patternSize, *centers, flags, Ptr<FeatureDetector>(blobDetector));
}
void cv_findContours(Mat* image, vector_Mat* contours, Mat* hierarchy, int mode, int method, Point* offset) {
	cv::findContours(*image, *contours, *hierarchy, mode, method, *offset);
}
void cv_findDataMatrix(Mat* image, vector_String* codes, Mat* corners, vector_Mat* dmtx) {
	cv::findDataMatrix(*image, *codes, *corners, *dmtx);
}
Mat* cv_findEssentialMat(Mat* points1, Mat* points2, double focal, Point2d* pp, int method, double prob, double threshold, Mat* mask) {
	return new Mat(cv::findEssentialMat(*points1, *points2, focal, *pp, method, prob, threshold, *mask));
}
Mat* cv_findFundamentalMat(Mat* points1, Mat* points2, int method, double param1, double param2, Mat* mask) {
	return new Mat(cv::findFundamentalMat(*points1, *points2, method, param1, param2, *mask));
}
Mat* cv_findHomography(Mat* srcPoints, Mat* dstPoints, int method, double ransacReprojThreshold, Mat* mask) {
	return new Mat(cv::findHomography(*srcPoints, *dstPoints, method, ransacReprojThreshold, *mask));
}
int cv_KDTree_findNearest(KDTree* self, Mat* vec, int K, int Emax, Mat* neighborsIdx, Mat* neighbors, Mat* dist, Mat* labels) {
	return self->findNearest(*vec, K, Emax, *neighborsIdx, *neighbors, *dist, *labels);
}
int cv_Subdiv2D_findNearest2(Subdiv2D* self, Point2f* pt, Point2f* nearestPt) {
	return self->findNearest(*pt, nearestPt);
}
void cv_findNonZero(Mat* src, Mat* idx) {
	cv::findNonZero(*src, *idx);
}
void cv_KDTree_findOrthoRange(KDTree* self, Mat* minBounds, Mat* maxBounds, Mat* neighborsIdx, Mat* neighbors, Mat* labels) {
	self->findOrthoRange(*minBounds, *maxBounds, *neighborsIdx, *neighbors, *labels);
}
double cv_findTransformECC(Mat* templateImage, Mat* inputImage, Mat* warpMatrix, int motionType, TermCriteria* criteria) {
	return cv::findTransformECC(*templateImage, *inputImage, *warpMatrix, motionType, *criteria);
}
float cv_CvKNearest_find_nearest(CvKNearest* self, Mat* samples, int k, Mat* results, Mat* neighborResponses, Mat* dists) {
	return self->find_nearest(*samples, k, *results, *neighborResponses, *dists);
}
RotatedRect* cv_fitEllipse(Mat* points) {
	return new RotatedRect(cv::fitEllipse(*points));
}
void cv_fitLine(Mat* points, Mat* line, int distType, double param, double reps, double aeps) {
	cv::fitLine(*points, *line, distType, param, reps, aeps);
}
void cv_flip(Mat* src, Mat* dst, int flipCode) {
	cv::flip(*src, *dst, flipCode);
}
int cv_floodFill(Mat* image, Mat* mask, Point* seedPoint, Scalar* newVal, Rect* rect, Scalar* loDiff, Scalar* upDiff, int flags) {
	return cv::floodFill(*image, *mask, *seedPoint, *newVal, rect, *loDiff, *upDiff, flags);
}
int cv_VideoWriter_fourcc(VideoWriter* self, char c1, char c2, char c3, char c4) {
	return self->fourcc(c1, c2, c3, c4);
}
void cv_gemm(Mat* src1, Mat* src2, double alpha, Mat* src3, double gamma, Mat* dst, int flags) {
	cv::gemm(*src1, *src2, alpha, *src3, gamma, *dst, flags);
}
void cv_BRISK_generateKernel(BRISK* self, vector_float* radiusList, vector_int* numberList, float dMax, float dMin, vector_int* indexChange) {
	self->generateKernel(*radiusList, *numberList, dMax, dMin, *indexChange);
}
double cv_VideoCapture_get(VideoCapture* self, int propId) {
	return self->get(propId);
}
Mat* cv_getAffineTransform(Mat* src, Mat* dst) {
	return new Mat(cv::getAffineTransform(*src, *dst));
}
Algorithm* cv_Algorithm_getAlgorithm(Algorithm* self, String* name) {
	return &*self->getAlgorithm(*name);
}
cvflann_flann_algorithm_t* cv_Index_getAlgorithm0(Index* self) {
	return new cvflann_flann_algorithm_t(self->getAlgorithm());
}
bool cv_Algorithm_getBool(Algorithm* self, String* name) {
	return self->getBool(*name);
}
vector_float* cv_HOGDescriptor_getDaimlerPeopleDetector(HOGDescriptor* self) {
	return new vector_float(self->getDaimlerPeopleDetector());
}
Mat* cv_getDefaultNewCameraMatrix(Mat* cameraMatrix, Size* imgsize, bool centerPrincipalPoint) {
	return new Mat(cv::getDefaultNewCameraMatrix(*cameraMatrix, *imgsize, centerPrincipalPoint));
}
vector_float* cv_HOGDescriptor_getDefaultPeopleDetector(HOGDescriptor* self) {
	return new vector_float(self->getDefaultPeopleDetector());
}
void cv_getDerivKernels(Mat* kx, Mat* ky, int dx, int dy, int ksize, bool normalize, int ktype) {
	cv::getDerivKernels(*kx, *ky, dx, dy, ksize, normalize, ktype);
}
size_t cv_HOGDescriptor_getDescriptorSize(HOGDescriptor* self) {
	return self->getDescriptorSize();
}
cvflann_flann_distance_t* cv_Index_getDistance(Index* self) {
	return new cvflann_flann_distance_t(self->getDistance());
}
double cv_Algorithm_getDouble(Algorithm* self, String* name) {
	return self->getDouble(*name);
}
int cv_Subdiv2D_getEdge(Subdiv2D* self, int edge, int nextEdgeType) {
	return self->getEdge(edge, nextEdgeType);
}
void cv_Subdiv2D_getEdgeList(Subdiv2D* self, vector_Vec4f* edgeList) {
	self->getEdgeList(*edgeList);
}
Mat* cv_getGaborKernel(Size* ksize, double sigma, double theta, double lambd, double gamma, double psi, int ktype) {
	return new Mat(cv::getGaborKernel(*ksize, sigma, theta, lambd, gamma, psi, ktype));
}
Mat* cv_getGaussianKernel(int ksize, double sigma, int ktype) {
	return new Mat(cv::getGaussianKernel(ksize, sigma, ktype));
}
int cv_Algorithm_getInt(Algorithm* self, String* name) {
	return self->getInt(*name);
}
void cv_Algorithm_getList(Algorithm* self, vector_String* algorithms) {
	self->getList(*algorithms);
}
Mat* cv_Algorithm_getMat(Algorithm* self, String* name) {
	return new Mat(self->getMat(*name));
}
vector_Mat* cv_Algorithm_getMatVector(Algorithm* self, String* name) {
	return new vector_Mat(self->getMatVector(*name));
}
int cv_getOptimalDFTSize(int vecsize) {
	return cv::getOptimalDFTSize(vecsize);
}
Mat* cv_getOptimalNewCameraMatrix(Mat* cameraMatrix, Mat* distCoeffs, Size* imageSize, double alpha, Size* newImgSize, Rect* validPixROI, bool centerPrincipalPoint) {
	return new Mat(cv::getOptimalNewCameraMatrix(*cameraMatrix, *distCoeffs, *imageSize, alpha, *newImgSize, validPixROI, centerPrincipalPoint));
}
void cv_Algorithm_getParams(Algorithm* self, vector_String* names) {
	self->getParams(*names);
}
Mat* cv_getPerspectiveTransform(Mat* src, Mat* dst) {
	return new Mat(cv::getPerspectiveTransform(*src, *dst));
}
void cv_KDTree_getPoints(KDTree* self, Mat* idx, Mat* pts, Mat* labels) {
	self->getPoints(*idx, *pts, *labels);
}
void cv_getRectSubPix(Mat* image, Size* patchSize, Point2f* center, Mat* patch, int patchType) {
	cv::getRectSubPix(*image, *patchSize, *center, *patch, patchType);
}
Mat* cv_getRotationMatrix2D(Point2f* center, double angle, double scale) {
	return new Mat(cv::getRotationMatrix2D(*center, angle, scale));
}
String* cv_Algorithm_getString(Algorithm* self, String* name) {
	return new String(self->getString(*name));
}
Mat* cv_getStructuringElement(int shape, Size* ksize, Point* anchor) {
	return new Mat(cv::getStructuringElement(shape, *ksize, *anchor));
}
Size* cv_getTextSize(String* text, int fontFace, double fontScale, int thickness, int* baseLine) {
	return new Size(cv::getTextSize(*text, fontFace, fontScale, thickness, baseLine));
}
int cv_getTrackbarPos(String* trackbarname, String* winname) {
	return cv::getTrackbarPos(*trackbarname, *winname);
}
vector_Mat* cv_DescriptorMatcher_getTrainDescriptors(DescriptorMatcher* self) {
	return new vector_Mat(self->getTrainDescriptors());
}
void cv_Subdiv2D_getTriangleList(Subdiv2D* self, vector_Vec6f* triangleList) {
	self->getTriangleList(*triangleList);
}
Rect* cv_getValidDisparityROI(Rect* roi1, Rect* roi2, int minDisparity, int numberOfDisparities, int SADWindowSize) {
	return new Rect(cv::getValidDisparityROI(*roi1, *roi2, minDisparity, numberOfDisparities, SADWindowSize));
}
Mat* cv_CvDTree_getVarImportance(CvDTree* self) {
	return new Mat(self->getVarImportance());
}
Mat* cv_CvRTrees_getVarImportance0(CvRTrees* self) {
	return new Mat(self->getVarImportance());
}
Point2f* cv_Subdiv2D_getVertex(Subdiv2D* self, int vertex, int* firstEdge) {
	return new Point2f(self->getVertex(vertex, firstEdge));
}
void cv_Subdiv2D_getVoronoiFacetList(Subdiv2D* self, vector_int* idx, vector_vector_Point2f* facetList, vector_Point2f* facetCenters) {
	self->getVoronoiFacetList(*idx, *facetList, *facetCenters);
}
double cv_HOGDescriptor_getWinSigma(HOGDescriptor* self) {
	return self->getWinSigma();
}
double cv_getWindowProperty(String* winname, int prop_id) {
	return cv::getWindowProperty(*winname, prop_id);
}
int cv_CvSVM_get_support_vector_count(CvSVM* self) {
	return self->get_support_vector_count();
}
int cv_CvSVM_get_var_count(CvSVM* self) {
	return self->get_var_count();
}
void cv_goodFeaturesToTrack(Mat* image, Mat* corners, int maxCorners, double qualityLevel, double minDistance, Mat* mask, int blockSize, bool useHarrisDetector, double k) {
	cv::goodFeaturesToTrack(*image, *corners, maxCorners, qualityLevel, minDistance, *mask, blockSize, useHarrisDetector, k);
}
bool cv_VideoCapture_grab(VideoCapture* self) {
	return self->grab();
}
void cv_grabCut(Mat* img, Mat* mask, Rect* rect, Mat* bgdModel, Mat* fgdModel, int iterCount, int mode) {
	cv::grabCut(*img, *mask, *rect, *bgdModel, *fgdModel, iterCount, mode);
}
void cv_groupRectangles(vector_Rect* rectList, vector_int* weights, int groupThreshold, double eps) {
	cv::groupRectangles(*rectList, *weights, groupThreshold, eps);
}
void cv_hconcat(vector_Mat* src, Mat* dst) {
	cv::hconcat(*src, *dst);
}
void cv_idct(Mat* src, Mat* dst, int flags) {
	cv::idct(*src, *dst, flags);
}
void cv_idft(Mat* src, Mat* dst, int flags, int nonzeroRows) {
	cv::idft(*src, *dst, flags, nonzeroRows);
}
Mat* cv_imdecode(Mat* buf, int flags) {
	return new Mat(cv::imdecode(*buf, flags));
}
bool cv_imencode(String* ext, Mat* img, vector_uchar* buf, vector_int* params) {
	return cv::imencode(*ext, *img, *buf, *params);
}
Mat* cv_imread(String* filename, int flags) {
	return new Mat(cv::imread(*filename, flags));
}
void cv_imshow(String* winname, Mat* mat) {
	cv::imshow(*winname, *mat);
}
bool cv_imwrite(String* filename, Mat* img, vector_int* params) {
	return cv::imwrite(*filename, *img, *params);
}
void cv_inRange(Mat* src, Mat* lowerb, Mat* upperb, Mat* dst) {
	cv::inRange(*src, *lowerb, *upperb, *dst);
}
Mat* cv_initCameraMatrix2D(vector_Mat* objectPoints, vector_Mat* imagePoints, Size* imageSize, double aspectRatio) {
	return new Mat(cv::initCameraMatrix2D(*objectPoints, *imagePoints, *imageSize, aspectRatio));
}
void cv_Subdiv2D_initDelaunay(Subdiv2D* self, Rect* rect) {
	self->initDelaunay(*rect);
}
void cv_initUndistortRectifyMap(Mat* cameraMatrix, Mat* distCoeffs, Mat* R, Mat* newCameraMatrix, Size* size, int m1type, Mat* map1, Mat* map2) {
	cv::initUndistortRectifyMap(*cameraMatrix, *distCoeffs, *R, *newCameraMatrix, *size, m1type, *map1, *map2);
}
float cv_initWideAngleProjMap(Mat* cameraMatrix, Mat* distCoeffs, Size* imageSize, int destImageWidth, int m1type, Mat* map1, Mat* map2, int projType, double alpha) {
	return cv::initWideAngleProjMap(*cameraMatrix, *distCoeffs, *imageSize, destImageWidth, m1type, *map1, *map2, projType, alpha);
}
int cv_Subdiv2D_insert(Subdiv2D* self, Point2f* pt) {
	return self->insert(*pt);
}
void cv_Subdiv2D_insert1(Subdiv2D* self, vector_Point2f* ptvec) {
	self->insert(*ptvec);
}
void cv_insertChannel(Mat* src, Mat* dst, int coi) {
	cv::insertChannel(*src, *dst, coi);
}
void cv_integral(Mat* src, Mat* sum, int sdepth) {
	cv::integral(*src, *sum, sdepth);
}
void cv_integral4(Mat* src, Mat* sum, Mat* sqsum, int sdepth) {
	cv::integral(*src, *sum, *sqsum, sdepth);
}
void cv_integral5(Mat* src, Mat* sum, Mat* sqsum, Mat* tilted, int sdepth) {
	cv::integral(*src, *sum, *sqsum, *tilted, sdepth);
}
float cv_intersectConvexConvex(Mat* _p1, Mat* _p2, Mat* _p12, bool handleNested) {
	return cv::intersectConvexConvex(*_p1, *_p2, *_p12, handleNested);
}
double cv_invert(Mat* src, Mat* dst, int flags) {
	return cv::invert(*src, *dst, flags);
}
void cv_invertAffineTransform(Mat* M, Mat* iM) {
	cv::invertAffineTransform(*M, *iM);
}
bool cv_isContourConvex(Mat* contour) {
	return cv::isContourConvex(*contour);
}
bool cv_VideoCapture_isOpened(VideoCapture* self) {
	return self->isOpened();
}
bool cv_VideoWriter_isOpened0(VideoWriter* self) {
	return self->isOpened();
}
bool cv_EM_isTrained(EM* self) {
	return self->isTrained();
}
double cv_kmeans(Mat* data, int K, Mat* bestLabels, TermCriteria* criteria, int attempts, int flags, Mat* centers) {
	return cv::kmeans(*data, K, *bestLabels, *criteria, attempts, flags, *centers);
}
void cv_DescriptorMatcher_knnMatch(DescriptorMatcher* self, Mat* queryDescriptors, Mat* trainDescriptors, vector_vector_DMatch* matches, int k, Mat* mask, bool compactResult) {
	self->knnMatch(*queryDescriptors, *trainDescriptors, *matches, k, *mask, compactResult);
}
void cv_DescriptorMatcher_knnMatch5(DescriptorMatcher* self, Mat* queryDescriptors, vector_vector_DMatch* matches, int k, vector_Mat* masks, bool compactResult) {
	self->knnMatch(*queryDescriptors, *matches, k, *masks, compactResult);
}
void cv_Index_knnSearch(Index* self, Mat* query, Mat* indices, Mat* dists, int knn, SearchParams* params) {
	self->knnSearch(*query, *indices, *dists, knn, *params);
}
void cv_line(Mat* img, Point* pt1, Point* pt2, Scalar* color, int thickness, int lineType, int shift) {
	cv::line(*img, *pt1, *pt2, *color, thickness, lineType, shift);
}
bool cv_Index_load(Index* self, Mat* features, String* filename) {
	return self->load(*features, *filename);
}
void cv_FaceRecognizer_load1(FaceRecognizer* self, String* filename) {
	self->load(*filename);
}
bool cv_HOGDescriptor_load2(HOGDescriptor* self, String* filename, String* objname) {
	return self->load(*filename, *objname);
}
int cv_Subdiv2D_locate(Subdiv2D* self, Point2f* pt, int edge, int vertex) {
	return self->locate(*pt, edge, vertex);
}
void cv_log(Mat* src, Mat* dst) {
	cv::log(*src, *dst);
}
void cv_magnitude(Mat* x, Mat* y, Mat* magnitude) {
	cv::magnitude(*x, *y, *magnitude);
}
void cv_matMulDeriv(Mat* A, Mat* B, Mat* dABdA, Mat* dABdB) {
	cv::matMulDeriv(*A, *B, *dABdA, *dABdB);
}
void cv_DescriptorMatcher_match(DescriptorMatcher* self, Mat* queryDescriptors, Mat* trainDescriptors, vector_DMatch* matches, Mat* mask) {
	self->match(*queryDescriptors, *trainDescriptors, *matches, *mask);
}
void cv_DescriptorMatcher_match3(DescriptorMatcher* self, Mat* queryDescriptors, vector_DMatch* matches, vector_Mat* masks) {
	self->match(*queryDescriptors, *matches, *masks);
}
double cv_matchShapes(Mat* contour1, Mat* contour2, int method, double parameter) {
	return cv::matchShapes(*contour1, *contour2, method, parameter);
}
void cv_matchTemplate(Mat* image, Mat* templ, Mat* result, int method) {
	cv::matchTemplate(*image, *templ, *result, method);
}
void cv_max(Mat* src1, Mat* src2, Mat* dst) {
	cv::max(*src1, *src2, *dst);
}
Scalar* cv_mean(Mat* src, Mat* mask) {
	return new Scalar(cv::mean(*src, *mask));
}
int cv_meanShift(Mat* probImage, Rect* window, TermCriteria* criteria) {
	return cv::meanShift(*probImage, *window, *criteria);
}
void cv_meanStdDev(Mat* src, Mat* mean, Mat* stddev, Mat* mask) {
	cv::meanStdDev(*src, *mean, *stddev, *mask);
}
void cv_medianBlur(Mat* src, Mat* dst, int ksize) {
	cv::medianBlur(*src, *dst, ksize);
}
void cv_merge(vector_Mat* mv, Mat* dst) {
	cv::merge(*mv, *dst);
}
void cv_min(Mat* src1, Mat* src2, Mat* dst) {
	cv::min(*src1, *src2, *dst);
}
RotatedRect* cv_minAreaRect(Mat* points) {
	return new RotatedRect(cv::minAreaRect(*points));
}
void cv_minEnclosingCircle(Mat* points, Point2f* center, float radius) {
	cv::minEnclosingCircle(*points, *center, radius);
}
double cv_minEnclosingTriangle(Mat* points, Mat* triangle) {
	return cv::minEnclosingTriangle(*points, *triangle);
}
void cv_minMaxLoc(Mat* src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, Mat* mask) {
	cv::minMaxLoc(*src, minVal, maxVal, minLoc, maxLoc, *mask);
}
void cv_mixChannels(vector_Mat* src, vector_Mat* dst, vector_int* fromTo) {
	cv::mixChannels(*src, *dst, *fromTo);
}
Moments* cv_moments(Mat* array, bool binaryImage) {
	return new Moments(cv::moments(*array, binaryImage));
}
void cv_morphologyEx(Mat* src, Mat* dst, int op, Mat* kernel, Point* anchor, int iterations, int borderType, Scalar* borderValue) {
	cv::morphologyEx(*src, *dst, op, *kernel, *anchor, iterations, borderType, *borderValue);
}
void cv_moveWindow(String* winname, int x, int y) {
	cv::moveWindow(*winname, x, y);
}
void cv_mulSpectrums(Mat* a, Mat* b, Mat* c, int flags, bool conjB) {
	cv::mulSpectrums(*a, *b, *c, flags, conjB);
}
void cv_mulTransposed(Mat* src, Mat* dst, bool aTa, Mat* delta, double scale, int dtype) {
	cv::mulTransposed(*src, *dst, aTa, *delta, scale, dtype);
}
void cv_multiply(Mat* src1, Mat* src2, Mat* dst, double scale, int dtype) {
	cv::multiply(*src1, *src2, *dst, scale, dtype);
}
void cv_namedWindow(String* winname, int flags) {
	cv::namedWindow(*winname, flags);
}
int cv_Subdiv2D_nextEdge(Subdiv2D* self, int edge) {
	return self->nextEdge(edge);
}
double cv_norm(Mat* src1, int normType, Mat* mask) {
	return cv::norm(*src1, normType, *mask);
}
double cv_norm4(Mat* src1, Mat* src2, int normType, Mat* mask) {
	return cv::norm(*src1, *src2, normType, *mask);
}
void cv_normalize(Mat* src, Mat* dst, double alpha, double beta, int norm_type, int dtype, Mat* mask) {
	cv::normalize(*src, *dst, alpha, beta, norm_type, dtype, *mask);
}
bool cv_VideoCapture_open(VideoCapture* self, String* filename) {
	return self->open(*filename);
}
bool cv_VideoCapture_open1(VideoCapture* self, int device) {
	return self->open(device);
}
bool cv_VideoWriter_open5(VideoWriter* self, String* filename, int fourcc, double fps, Size* frameSize, bool isColor) {
	return self->open(*filename, fourcc, fps, *frameSize, isColor);
}
void cv_Feature2D_call(Feature2D* self, Mat* image, Mat* mask, vector_KeyPoint* keypoints, Mat* descriptors, bool useProvidedKeypoints) {
	self->operator ()(*image, *mask, *keypoints, *descriptors, useProvidedKeypoints);
}
void cv_StarDetector_call(StarDetector* self, Mat* image, vector_KeyPoint* keypoints) {
	self->operator ()(*image, *keypoints);
}
void cv_StereoVar_call(StereoVar* self, Mat* left, Mat* right, Mat* disp) {
	self->operator ()(*left, *right, *disp);
}
String* cv_Algorithm_paramHelp(Algorithm* self, String* name) {
	return new String(self->paramHelp(*name));
}
int cv_Algorithm_paramType(Algorithm* self, String* name) {
	return self->paramType(*name);
}
void cv_patchNaNs(Mat* a, double val) {
	cv::patchNaNs(*a, val);
}
void cv_perspectiveTransform(Mat* src, Mat* dst, Mat* m) {
	cv::perspectiveTransform(*src, *dst, *m);
}
void cv_phase(Mat* x, Mat* y, Mat* angle, bool angleInDegrees) {
	cv::phase(*x, *y, *angle, angleInDegrees);
}
Point2d* cv_phaseCorrelate(Mat* src1, Mat* src2, Mat* window, double* response) {
	return new Point2d(cv::phaseCorrelate(*src1, *src2, *window, response));
}
double cv_pointPolygonTest(Mat* contour, Point2f* pt, bool measureDist) {
	return cv::pointPolygonTest(*contour, *pt, measureDist);
}
void cv_polarToCart(Mat* magnitude, Mat* angle, Mat* x, Mat* y, bool angleInDegrees) {
	cv::polarToCart(*magnitude, *angle, *x, *y, angleInDegrees);
}
void cv_polylines(Mat* img, vector_Mat* pts, bool isClosed, Scalar* color, int thickness, int lineType, int shift) {
	cv::polylines(*img, *pts, isClosed, *color, thickness, lineType, shift);
}
void cv_pow(Mat* src, double power, Mat* dst) {
	cv::pow(*src, power, *dst);
}
void cv_preCornerDetect(Mat* src, Mat* dst, int ksize, int borderType) {
	cv::preCornerDetect(*src, *dst, ksize, borderType);
}
float cv_CvNormalBayesClassifier_predict(CvNormalBayesClassifier* self, Mat* samples, Mat* results) {
	return self->predict(*samples, results);
}
Mat* cv_KalmanFilter_predict1(KalmanFilter* self, Mat* control) {
	return new Mat(self->predict(*control));
}
float cv_CvANN_MLP_predict2(CvANN_MLP* self, Mat* inputs, Mat* outputs) {
	return self->predict(*inputs, *outputs);
}
void cv_FaceRecognizer_predict3(FaceRecognizer* self, Mat* src, int label, double confidence) {
	self->predict(*src, label, confidence);
}
float cv_CvGBTrees_predict4(CvGBTrees* self, Mat* sample, Mat* missing, Range* slice, int k) {
	return self->predict(*sample, *missing, *slice, k);
}
float cv_CvBoost_predict5(CvBoost* self, Mat* sample, Mat* missing, Range* slice, bool rawMode, bool returnSum) {
	return self->predict(*sample, *missing, *slice, rawMode, returnSum);
}
float cv_CvRTrees_predict_prob(CvRTrees* self, Mat* sample, Mat* missing) {
	return self->predict_prob(*sample, *missing);
}
void cv_projectPoints(Mat* objectPoints, Mat* rvec, Mat* tvec, Mat* cameraMatrix, Mat* distCoeffs, Mat* imagePoints, Mat* jacobian, double aspectRatio) {
	cv::projectPoints(*objectPoints, *rvec, *tvec, *cameraMatrix, *distCoeffs, *imagePoints, *jacobian, aspectRatio);
}
void cv_CvBoost_prune(CvBoost* self, CvSlice* slice) {
	self->prune(*slice);
}
void cv_putText(Mat* img, String* text, Point* org, int fontFace, double fontScale, Scalar* color, int thickness, int lineType, bool bottomLeftOrigin) {
	cv::putText(*img, *text, *org, fontFace, fontScale, *color, thickness, lineType, bottomLeftOrigin);
}
void cv_pyrDown(Mat* src, Mat* dst, Size* dstsize, int borderType) {
	cv::pyrDown(*src, *dst, *dstsize, borderType);
}
void cv_pyrMeanShiftFiltering(Mat* src, Mat* dst, double sp, double sr, int maxLevel, TermCriteria* termcrit) {
	cv::pyrMeanShiftFiltering(*src, *dst, sp, sr, maxLevel, *termcrit);
}
void cv_pyrUp(Mat* src, Mat* dst, Size* dstsize, int borderType) {
	cv::pyrUp(*src, *dst, *dstsize, borderType);
}
int cv_Index_radiusSearch(Index* self, Mat* query, Mat* indices, Mat* dists, double radius, int maxResults, SearchParams* params) {
	return self->radiusSearch(*query, *indices, *dists, radius, maxResults, *params);
}
void cv_randShuffle(Mat* dst, double iterFactor, RNG* rng) {
	cv::randShuffle(*dst, iterFactor, rng);
}
void cv_randn(Mat* dst, Mat* mean, Mat* stddev) {
	cv::randn(*dst, *mean, *stddev);
}
void cv_randu(Mat* dst, Mat* low, Mat* high) {
	cv::randu(*dst, *low, *high);
}
bool cv_VideoCapture_read(VideoCapture* self, Mat* image) {
	return self->read(*image);
}
int cv_recoverPose(Mat* E, Mat* points1, Mat* points2, Mat* R, Mat* t, double focal, Point2d* pp, Mat* mask) {
	return cv::recoverPose(*E, *points1, *points2, *R, *t, focal, *pp, *mask);
}
void cv_rectangle(Mat* img, Point* pt1, Point* pt2, Scalar* color, int thickness, int lineType, int shift) {
	cv::rectangle(*img, *pt1, *pt2, *color, thickness, lineType, shift);
}
float cv_rectify3Collinear(Mat* cameraMatrix1, Mat* distCoeffs1, Mat* cameraMatrix2, Mat* distCoeffs2, Mat* cameraMatrix3, Mat* distCoeffs3, vector_Mat* imgpt1, vector_Mat* imgpt3, Size* imageSize, Mat* R12, Mat* T12, Mat* R13, Mat* T13, Mat* R1, Mat* R2, Mat* R3, Mat* P1, Mat* P2, Mat* P3, Mat* Q, double alpha, Size* newImgSize, Rect* roi1, Rect* roi2, int flags) {
	return cv::rectify3Collinear(*cameraMatrix1, *distCoeffs1, *cameraMatrix2, *distCoeffs2, *cameraMatrix3, *distCoeffs3, *imgpt1, *imgpt3, *imageSize, *R12, *T12, *R13, *T13, *R1, *R2, *R3, *P1, *P2, *P3, *Q, alpha, *newImgSize, roi1, roi2, flags);
}
void cv_reduce(Mat* src, Mat* dst, int dim, int rtype, int dtype) {
	cv::reduce(*src, *dst, dim, rtype, dtype);
}
void cv_Index_release(Index* self) {
	self->release();
}
void cv_VideoWriter_release0(VideoWriter* self) {
	self->release();
}
void cv_remap(Mat* src, Mat* dst, Mat* map1, Mat* map2, int interpolation, int borderMode, Scalar* borderValue) {
	cv::remap(*src, *dst, *map1, *map2, interpolation, borderMode, *borderValue);
}
void cv_repeat(Mat* src, int ny, int nx, Mat* dst) {
	cv::repeat(*src, ny, nx, *dst);
}
void cv_reprojectImageTo3D(Mat* disparity, Mat* _3dImage, Mat* Q, bool handleMissingValues, int ddepth) {
	cv::reprojectImageTo3D(*disparity, *_3dImage, *Q, handleMissingValues, ddepth);
}
void cv_resize(Mat* src, Mat* dst, Size* dsize, double fx, double fy, int interpolation) {
	cv::resize(*src, *dst, *dsize, fx, fy, interpolation);
}
void cv_resizeWindow(String* winname, int width, int height) {
	cv::resizeWindow(*winname, width, height);
}
bool cv_VideoCapture_retrieve(VideoCapture* self, Mat* image, int flag) {
	return self->retrieve(*image, flag);
}
int cv_Subdiv2D_rotateEdge(Subdiv2D* self, int edge, int rotate) {
	return self->rotateEdge(edge, rotate);
}
int cv_rotatedRectangleIntersection(RotatedRect* rect1, RotatedRect* rect2, Mat* intersectingRegion) {
	return cv::rotatedRectangleIntersection(*rect1, *rect2, *intersectingRegion);
}
void cv_Index_save(Index* self, String* filename) {
	self->save(*filename);
}
void cv_FaceRecognizer_save1(FaceRecognizer* self, String* filename) {
	self->save(*filename);
}
void cv_HOGDescriptor_save2(HOGDescriptor* self, String* filename, String* objname) {
	self->save(*filename, *objname);
}
void cv_scaleAdd(Mat* src1, double alpha, Mat* src2, Mat* dst) {
	cv::scaleAdd(*src1, alpha, *src2, *dst);
}
void cv_segmentMotion(Mat* mhi, Mat* segmask, vector_Rect* boundingRects, double timestamp, double segThresh) {
	cv::segmentMotion(*mhi, *segmask, *boundingRects, timestamp, segThresh);
}
void cv_sepFilter2D(Mat* src, Mat* dst, int ddepth, Mat* kernelX, Mat* kernelY, Point* anchor, double delta, int borderType) {
	cv::sepFilter2D(*src, *dst, ddepth, *kernelX, *kernelY, *anchor, delta, borderType);
}
bool cv_VideoCapture_set(VideoCapture* self, int propId, double value) {
	return self->set(propId, value);
}
void cv_Algorithm_setAlgorithm(Algorithm* self, String* name, Algorithm* value) {
	self->setAlgorithm(*name, Ptr<Algorithm>(value));
}
void cv_BackgroundSubtractorGMG_setBackgroundPrior(BackgroundSubtractorGMG* self, double bgprior) {
	self->setBackgroundPrior(bgprior);
}
void cv_BackgroundSubtractorMOG_setBackgroundRatio(BackgroundSubtractorMOG* self, double backgroundRatio) {
	self->setBackgroundRatio(backgroundRatio);
}
void cv_BackgroundSubtractorMOG2_setBackgroundRatio1(BackgroundSubtractorMOG2* self, double ratio) {
	self->setBackgroundRatio(ratio);
}
void cv_StereoMatcher_setBlockSize(StereoMatcher* self, int blockSize) {
	self->setBlockSize(blockSize);
}
void cv_Algorithm_setBool(Algorithm* self, String* name, bool value) {
	self->setBool(*name, value);
}
void cv_CLAHE_setClipLimit(CLAHE* self, double clipLimit) {
	self->setClipLimit(clipLimit);
}
void cv_BackgroundSubtractorMOG2_setComplexityReductionThreshold(BackgroundSubtractorMOG2* self, double ct) {
	self->setComplexityReductionThreshold(ct);
}
void cv_BackgroundSubtractorGMG_setDecisionThreshold(BackgroundSubtractorGMG* self, double thresh) {
	self->setDecisionThreshold(thresh);
}
void cv_BackgroundSubtractorGMG_setDefaultLearningRate(BackgroundSubtractorGMG* self, double lr) {
	self->setDefaultLearningRate(lr);
}
void cv_BackgroundSubtractorMOG2_setDetectShadows(BackgroundSubtractorMOG2* self, bool detectShadows) {
	self->setDetectShadows(detectShadows);
}
void cv_StereoMatcher_setDisp12MaxDiff(StereoMatcher* self, int disp12MaxDiff) {
	self->setDisp12MaxDiff(disp12MaxDiff);
}
void cv_Algorithm_setDouble(Algorithm* self, String* name, double value) {
	self->setDouble(*name, value);
}
void cv_BackgroundSubtractorMOG_setHistory(BackgroundSubtractorMOG* self, int nframes) {
	self->setHistory(nframes);
}
void cv_BackgroundSubtractorMOG2_setHistory1(BackgroundSubtractorMOG2* self, int history) {
	self->setHistory(history);
}
void cv_setIdentity(Mat* mtx, Scalar* s) {
	cv::setIdentity(*mtx, *s);
}
void cv_Algorithm_setInt(Algorithm* self, String* name, int value) {
	self->setInt(*name, value);
}
void cv_Algorithm_setMat(Algorithm* self, String* name, Mat* value) {
	self->setMat(*name, *value);
}
void cv_Algorithm_setMatVector(Algorithm* self, String* name, vector_Mat* value) {
	self->setMatVector(*name, *value);
}
void cv_BackgroundSubtractorGMG_setMaxFeatures(BackgroundSubtractorGMG* self, int maxFeatures) {
	self->setMaxFeatures(maxFeatures);
}
void cv_BackgroundSubtractorGMG_setMaxVal(BackgroundSubtractorGMG* self, double val) {
	self->setMaxVal(val);
}
void cv_StereoMatcher_setMinDisparity(StereoMatcher* self, int minDisparity) {
	self->setMinDisparity(minDisparity);
}
void cv_BackgroundSubtractorGMG_setMinVal(BackgroundSubtractorGMG* self, double val) {
	self->setMinVal(val);
}
void cv_StereoSGBM_setMode(StereoSGBM* self, int mode) {
	self->setMode(mode);
}
void cv_BackgroundSubtractorMOG_setNMixtures(BackgroundSubtractorMOG* self, int nmix) {
	self->setNMixtures(nmix);
}
void cv_BackgroundSubtractorMOG2_setNMixtures1(BackgroundSubtractorMOG2* self, int nmixtures) {
	self->setNMixtures(nmixtures);
}
void cv_BackgroundSubtractorMOG_setNoiseSigma(BackgroundSubtractorMOG* self, double noiseSigma) {
	self->setNoiseSigma(noiseSigma);
}
void cv_StereoMatcher_setNumDisparities(StereoMatcher* self, int numDisparities) {
	self->setNumDisparities(numDisparities);
}
void cv_BackgroundSubtractorGMG_setNumFrames(BackgroundSubtractorGMG* self, int nframes) {
	self->setNumFrames(nframes);
}
void cv_StereoSGBM_setP1(StereoSGBM* self, int P1) {
	self->setP1(P1);
}
void cv_StereoSGBM_setP2(StereoSGBM* self, int P2) {
	self->setP2(P2);
}
void cv_StereoBM_setPreFilterCap(StereoBM* self, int preFilterCap) {
	self->setPreFilterCap(preFilterCap);
}
void cv_StereoSGBM_setPreFilterCap1(StereoSGBM* self, int preFilterCap) {
	self->setPreFilterCap(preFilterCap);
}
void cv_StereoBM_setPreFilterSize(StereoBM* self, int preFilterSize) {
	self->setPreFilterSize(preFilterSize);
}
void cv_StereoBM_setPreFilterType(StereoBM* self, int preFilterType) {
	self->setPreFilterType(preFilterType);
}
void cv_BackgroundSubtractorGMG_setQuantizationLevels(BackgroundSubtractorGMG* self, int nlevels) {
	self->setQuantizationLevels(nlevels);
}
void cv_StereoBM_setROI1(StereoBM* self, Rect* roi1) {
	self->setROI1(*roi1);
}
void cv_StereoBM_setROI2(StereoBM* self, Rect* roi2) {
	self->setROI2(*roi2);
}
void cv_HOGDescriptor_setSVMDetector(HOGDescriptor* self, Mat* _svmdetector) {
	self->setSVMDetector(*_svmdetector);
}
void cv_BackgroundSubtractorMOG2_setShadowThreshold(BackgroundSubtractorMOG2* self, double threshold) {
	self->setShadowThreshold(threshold);
}
void cv_BackgroundSubtractorMOG2_setShadowValue(BackgroundSubtractorMOG2* self, int value) {
	self->setShadowValue(value);
}
void cv_StereoBM_setSmallerBlockSize(StereoBM* self, int blockSize) {
	self->setSmallerBlockSize(blockSize);
}
void cv_BackgroundSubtractorGMG_setSmoothingRadius(BackgroundSubtractorGMG* self, int radius) {
	self->setSmoothingRadius(radius);
}
void cv_StereoMatcher_setSpeckleRange(StereoMatcher* self, int speckleRange) {
	self->setSpeckleRange(speckleRange);
}
void cv_StereoMatcher_setSpeckleWindowSize(StereoMatcher* self, int speckleWindowSize) {
	self->setSpeckleWindowSize(speckleWindowSize);
}
void cv_Algorithm_setString(Algorithm* self, String* name, String* value) {
	self->setString(*name, *value);
}
void cv_StereoBM_setTextureThreshold(StereoBM* self, int textureThreshold) {
	self->setTextureThreshold(textureThreshold);
}
void cv_CLAHE_setTilesGridSize(CLAHE* self, Size* tileGridSize) {
	self->setTilesGridSize(*tileGridSize);
}
void cv_setTrackbarPos(String* trackbarname, String* winname, int pos) {
	cv::setTrackbarPos(*trackbarname, *winname, pos);
}
void cv_StereoBM_setUniquenessRatio(StereoBM* self, int uniquenessRatio) {
	self->setUniquenessRatio(uniquenessRatio);
}
void cv_StereoSGBM_setUniquenessRatio1(StereoSGBM* self, int uniquenessRatio) {
	self->setUniquenessRatio(uniquenessRatio);
}
void cv_BackgroundSubtractorGMG_setUpdateBackgroundModel(BackgroundSubtractorGMG* self, bool update) {
	self->setUpdateBackgroundModel(update);
}
void cv_BackgroundSubtractorMOG2_setVarInit(BackgroundSubtractorMOG2* self, double varInit) {
	self->setVarInit(varInit);
}
void cv_BackgroundSubtractorMOG2_setVarMax(BackgroundSubtractorMOG2* self, double varMax) {
	self->setVarMax(varMax);
}
void cv_BackgroundSubtractorMOG2_setVarMin(BackgroundSubtractorMOG2* self, double varMin) {
	self->setVarMin(varMin);
}
void cv_BackgroundSubtractorMOG2_setVarThreshold(BackgroundSubtractorMOG2* self, double varThreshold) {
	self->setVarThreshold(varThreshold);
}
void cv_BackgroundSubtractorMOG2_setVarThresholdGen(BackgroundSubtractorMOG2* self, double varThresholdGen) {
	self->setVarThresholdGen(varThresholdGen);
}
void cv_setWindowProperty(String* winname, int prop_id, double prop_value) {
	cv::setWindowProperty(*winname, prop_id, prop_value);
}
bool cv_solve(Mat* src1, Mat* src2, Mat* dst, int flags) {
	return cv::solve(*src1, *src2, *dst, flags);
}
int cv_solveCubic(Mat* coeffs, Mat* roots) {
	return cv::solveCubic(*coeffs, *roots);
}
bool cv_solvePnP(Mat* objectPoints, Mat* imagePoints, Mat* cameraMatrix, Mat* distCoeffs, Mat* rvec, Mat* tvec, bool useExtrinsicGuess, int flags) {
	return cv::solvePnP(*objectPoints, *imagePoints, *cameraMatrix, *distCoeffs, *rvec, *tvec, useExtrinsicGuess, flags);
}
void cv_solvePnPRansac(Mat* objectPoints, Mat* imagePoints, Mat* cameraMatrix, Mat* distCoeffs, Mat* rvec, Mat* tvec, bool useExtrinsicGuess, int iterationsCount, float reprojectionError, int minInliersCount, Mat* inliers, int flags) {
	cv::solvePnPRansac(*objectPoints, *imagePoints, *cameraMatrix, *distCoeffs, *rvec, *tvec, useExtrinsicGuess, iterationsCount, reprojectionError, minInliersCount, *inliers, flags);
}
double cv_solvePoly(Mat* coeffs, Mat* roots, int maxIters) {
	return cv::solvePoly(*coeffs, *roots, maxIters);
}
void cv_sort(Mat* src, Mat* dst, int flags) {
	cv::sort(*src, *dst, flags);
}
void cv_sortIdx(Mat* src, Mat* dst, int flags) {
	cv::sortIdx(*src, *dst, flags);
}
void cv_split(Mat* m, vector_Mat* mv) {
	cv::split(*m, *mv);
}
void cv_sqrt(Mat* src, Mat* dst) {
	cv::sqrt(*src, *dst);
}
int cv_startWindowThread() {
	return cv::startWindowThread();
}
double cv_stereoCalibrate(vector_Mat* objectPoints, vector_Mat* imagePoints1, vector_Mat* imagePoints2, Mat* cameraMatrix1, Mat* distCoeffs1, Mat* cameraMatrix2, Mat* distCoeffs2, Size* imageSize, Mat* R, Mat* T, Mat* E, Mat* F, TermCriteria* criteria, int flags) {
	return cv::stereoCalibrate(*objectPoints, *imagePoints1, *imagePoints2, *cameraMatrix1, *distCoeffs1, *cameraMatrix2, *distCoeffs2, *imageSize, *R, *T, *E, *F, *criteria, flags);
}
void cv_stereoRectify(Mat* cameraMatrix1, Mat* distCoeffs1, Mat* cameraMatrix2, Mat* distCoeffs2, Size* imageSize, Mat* R, Mat* T, Mat* R1, Mat* R2, Mat* P1, Mat* P2, Mat* Q, int flags, double alpha, Size* newImageSize, Rect* validPixROI1, Rect* validPixROI2) {
	cv::stereoRectify(*cameraMatrix1, *distCoeffs1, *cameraMatrix2, *distCoeffs2, *imageSize, *R, *T, *R1, *R2, *P1, *P2, *Q, flags, alpha, *newImageSize, validPixROI1, validPixROI2);
}
bool cv_stereoRectifyUncalibrated(Mat* points1, Mat* points2, Mat* F, Size* imgSize, Mat* H1, Mat* H2, double threshold) {
	return cv::stereoRectifyUncalibrated(*points1, *points2, *F, *imgSize, *H1, *H2, threshold);
}
void cv_subtract(Mat* src1, Mat* src2, Mat* dst, Mat* mask, int dtype) {
	cv::subtract(*src1, *src2, *dst, *mask, dtype);
}
Scalar* cv_sum(Mat* src) {
	return new Scalar(cv::sum(*src));
}
int cv_Subdiv2D_symEdge(Subdiv2D* self, int edge) {
	return self->symEdge(edge);
}
double cv_threshold(Mat* src, Mat* dst, double thresh, double maxval, int type) {
	return cv::threshold(*src, *dst, thresh, maxval, type);
}
Scalar* cv_trace(Mat* mtx) {
	return new Scalar(cv::trace(*mtx));
}
bool cv_CvNormalBayesClassifier_train(CvNormalBayesClassifier* self, Mat* trainData, Mat* responses, Mat* varIdx, Mat* sampleIdx, bool update) {
	return self->train(*trainData, *responses, *varIdx, *sampleIdx, update);
}
void cv_DescriptorMatcher_train0(DescriptorMatcher* self) {
	self->train();
}
void cv_FaceRecognizer_train2(FaceRecognizer* self, vector_Mat* src, Mat* labels) {
	self->train(*src, *labels);
}
bool cv_EM_train4(EM* self, Mat* samples, Mat* logLikelihoods, Mat* labels, Mat* probs) {
	return self->train(*samples, *logLikelihoods, *labels, *probs);
}
bool cv_CvSVM_train5(CvSVM* self, Mat* trainData, Mat* responses, Mat* varIdx, Mat* sampleIdx, CvSVMParams* params) {
	return self->train(*trainData, *responses, *varIdx, *sampleIdx, *params);
}
int cv_CvANN_MLP_train6(CvANN_MLP* self, Mat* inputs, Mat* outputs, Mat* sampleWeights, Mat* sampleIdx, CvANN_MLP_TrainParams* params, int flags) {
	return self->train(*inputs, *outputs, *sampleWeights, *sampleIdx, *params, flags);
}
bool cv_CvERTrees_train8(CvERTrees* self, Mat* trainData, int tflag, Mat* responses, Mat* varIdx, Mat* sampleIdx, Mat* varType, Mat* missingDataMask, CvRTParams* params) {
	return self->train(*trainData, tflag, *responses, *varIdx, *sampleIdx, *varType, *missingDataMask, *params);
}
bool cv_CvGBTrees_train9(CvGBTrees* self, Mat* trainData, int tflag, Mat* responses, Mat* varIdx, Mat* sampleIdx, Mat* varType, Mat* missingDataMask, CvGBTreesParams* params, bool update) {
	return self->train(*trainData, tflag, *responses, *varIdx, *sampleIdx, *varType, *missingDataMask, *params, update);
}
bool cv_EM_trainE(EM* self, Mat* samples, Mat* means0, Mat* covs0, Mat* weights0, Mat* logLikelihoods, Mat* labels, Mat* probs) {
	return self->trainE(*samples, *means0, *covs0, *weights0, *logLikelihoods, *labels, *probs);
}
bool cv_EM_trainM(EM* self, Mat* samples, Mat* probs0, Mat* logLikelihoods, Mat* labels, Mat* probs) {
	return self->trainM(*samples, *probs0, *logLikelihoods, *labels, *probs);
}
bool cv_CvSVM_train_auto(CvSVM* self, Mat* trainData, Mat* responses, Mat* varIdx, Mat* sampleIdx, CvSVMParams* params, int k_fold, CvParamGrid* Cgrid, CvParamGrid* gammaGrid, CvParamGrid* pGrid, CvParamGrid* nuGrid, CvParamGrid* coeffGrid, CvParamGrid* degreeGrid, bool balanced) {
	return self->train_auto(*trainData, *responses, *varIdx, *sampleIdx, *params, k_fold, *Cgrid, *gammaGrid, *pGrid, *nuGrid, *coeffGrid, *degreeGrid, balanced);
}
void cv_transform(Mat* src, Mat* dst, Mat* m) {
	cv::transform(*src, *dst, *m);
}
void cv_transpose(Mat* src, Mat* dst) {
	cv::transpose(*src, *dst);
}
void cv_triangulatePoints(Mat* projMatr1, Mat* projMatr2, Mat* projPoints1, Mat* projPoints2, Mat* points4D) {
	cv::triangulatePoints(*projMatr1, *projMatr2, *projPoints1, *projPoints2, *points4D);
}
void cv_undistort(Mat* src, Mat* dst, Mat* cameraMatrix, Mat* distCoeffs, Mat* newCameraMatrix) {
	cv::undistort(*src, *dst, *cameraMatrix, *distCoeffs, *newCameraMatrix);
}
void cv_undistortPoints(Mat* src, Mat* dst, Mat* cameraMatrix, Mat* distCoeffs, Mat* R, Mat* P) {
	cv::undistortPoints(*src, *dst, *cameraMatrix, *distCoeffs, *R, *P);
}
void cv_FaceRecognizer_update(FaceRecognizer* self, vector_Mat* src, Mat* labels) {
	self->update(*src, *labels);
}
void cv_updateMotionHistory(Mat* silhouette, Mat* mhi, double timestamp, double duration) {
	cv::updateMotionHistory(*silhouette, *mhi, timestamp, duration);
}
void cv_validateDisparity(Mat* disparity, Mat* cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp) {
	cv::validateDisparity(*disparity, *cost, minDisparity, numberOfDisparities, disp12MaxDisp);
}
void cv_vconcat(vector_Mat* src, Mat* dst) {
	cv::vconcat(*src, *dst);
}
int cv_waitKey(int delay) {
	return cv::waitKey(delay);
}
void cv_warpAffine(Mat* src, Mat* dst, Mat* M, Size* dsize, int flags, int borderMode, Scalar* borderValue) {
	cv::warpAffine(*src, *dst, *M, *dsize, flags, borderMode, *borderValue);
}
void cv_warpPerspective(Mat* src, Mat* dst, Mat* M, Size* dsize, int flags, int borderMode, Scalar* borderValue) {
	cv::warpPerspective(*src, *dst, *M, *dsize, flags, borderMode, *borderValue);
}
void cv_watershed(Mat* image, Mat* markers) {
	cv::watershed(*image, *markers);
}
void cv_VideoWriter_write(VideoWriter* self, Mat* image) {
	self->write(*image);
}
}