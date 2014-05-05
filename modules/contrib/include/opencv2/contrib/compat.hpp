/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CONTRIB_COMPAT_HPP__
#define __OPENCV_CONTRIB_COMPAT_HPP__

#include "opencv2/core/core_c.h"

#ifdef __cplusplus

/****************************************************************************************\
*                                   Adaptive Skin Detector                               *
\****************************************************************************************/

class CV_EXPORTS CvAdaptiveSkinDetector
{
private:
    enum {
        GSD_HUE_LT = 3,
        GSD_HUE_UT = 33,
        GSD_INTENSITY_LT = 15,
        GSD_INTENSITY_UT = 250
    };

    class CV_EXPORTS Histogram
    {
    private:
        enum {
            HistogramSize = (GSD_HUE_UT - GSD_HUE_LT + 1)
        };

    protected:
        int findCoverageIndex(double surfaceToCover, int defaultValue = 0);

    public:
        CvHistogram *fHistogram;
        Histogram();
        virtual ~Histogram();

        void findCurveThresholds(int &x1, int &x2, double percent = 0.05);
        void mergeWith(Histogram *source, double weight);
    };

    int nStartCounter, nFrameCount, nSkinHueLowerBound, nSkinHueUpperBound, nMorphingMethod, nSamplingDivider;
    double fHistogramMergeFactor, fHuePercentCovered;
    Histogram histogramHueMotion, skinHueHistogram;
    IplImage *imgHueFrame, *imgSaturationFrame, *imgLastGrayFrame, *imgMotionFrame, *imgFilteredFrame;
    IplImage *imgShrinked, *imgTemp, *imgGrayFrame, *imgHSVFrame;

protected:
    void initData(IplImage *src, int widthDivider, int heightDivider);
    void adaptiveFilter();

public:

    enum {
        MORPHING_METHOD_NONE = 0,
        MORPHING_METHOD_ERODE = 1,
        MORPHING_METHOD_ERODE_ERODE = 2,
        MORPHING_METHOD_ERODE_DILATE = 3
    };

    CvAdaptiveSkinDetector(int samplingDivider = 1, int morphingMethod = MORPHING_METHOD_NONE);
    virtual ~CvAdaptiveSkinDetector();

    virtual void process(IplImage *inputBGRImage, IplImage *outputHueMask);
};


/****************************************************************************************\
 *                                  Fuzzy MeanShift Tracker                               *
 \****************************************************************************************/

class CV_EXPORTS CvFuzzyPoint {
public:
    double x, y, value;

    CvFuzzyPoint(double _x, double _y);
};

class CV_EXPORTS CvFuzzyCurve {
private:
    std::vector<CvFuzzyPoint> points;
    double value, centre;

    bool between(double x, double x1, double x2);

public:
    CvFuzzyCurve();
    ~CvFuzzyCurve();

    void setCentre(double _centre);
    double getCentre();
    void clear();
    void addPoint(double x, double y);
    double calcValue(double param);
    double getValue();
    void setValue(double _value);
};

class CV_EXPORTS CvFuzzyFunction {
public:
    std::vector<CvFuzzyCurve> curves;

    CvFuzzyFunction();
    ~CvFuzzyFunction();
    void addCurve(CvFuzzyCurve *curve, double value = 0);
    void resetValues();
    double calcValue();
    CvFuzzyCurve *newCurve();
};

class CV_EXPORTS CvFuzzyRule {
private:
    CvFuzzyCurve *fuzzyInput1, *fuzzyInput2;
    CvFuzzyCurve *fuzzyOutput;
public:
    CvFuzzyRule();
    ~CvFuzzyRule();
    void setRule(CvFuzzyCurve *c1, CvFuzzyCurve *c2, CvFuzzyCurve *o1);
    double calcValue(double param1, double param2);
    CvFuzzyCurve *getOutputCurve();
};

class CV_EXPORTS CvFuzzyController {
private:
    std::vector<CvFuzzyRule*> rules;
public:
    CvFuzzyController();
    ~CvFuzzyController();
    void addRule(CvFuzzyCurve *c1, CvFuzzyCurve *c2, CvFuzzyCurve *o1);
    double calcOutput(double param1, double param2);
};

class CV_EXPORTS CvFuzzyMeanShiftTracker
{
private:
    class FuzzyResizer
    {
    private:
        CvFuzzyFunction iInput, iOutput;
        CvFuzzyController fuzzyController;
    public:
        FuzzyResizer();
        int calcOutput(double edgeDensity, double density);
    };

    class SearchWindow
    {
    public:
        FuzzyResizer *fuzzyResizer;
        int x, y;
        int width, height, maxWidth, maxHeight, ellipseHeight, ellipseWidth;
        int ldx, ldy, ldw, ldh, numShifts, numIters;
        int xGc, yGc;
        long m00, m01, m10, m11, m02, m20;
        double ellipseAngle;
        double density;
        unsigned int depthLow, depthHigh;
        int verticalEdgeLeft, verticalEdgeRight, horizontalEdgeTop, horizontalEdgeBottom;

        SearchWindow();
        ~SearchWindow();
        void setSize(int _x, int _y, int _width, int _height);
        void initDepthValues(IplImage *maskImage, IplImage *depthMap);
        bool shift();
        void extractInfo(IplImage *maskImage, IplImage *depthMap, bool initDepth);
        void getResizeAttribsEdgeDensityLinear(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh);
        void getResizeAttribsInnerDensity(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh);
        void getResizeAttribsEdgeDensityFuzzy(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh);
        bool meanShift(IplImage *maskImage, IplImage *depthMap, int maxIteration, bool initDepth);
    };

public:
    enum TrackingState
    {
        tsNone          = 0,
        tsSearching     = 1,
        tsTracking      = 2,
        tsSetWindow     = 3,
        tsDisabled      = 10
    };

    enum ResizeMethod {
        rmEdgeDensityLinear     = 0,
        rmEdgeDensityFuzzy      = 1,
        rmInnerDensity          = 2
    };

    enum {
        MinKernelMass           = 1000
    };

    SearchWindow kernel;
    int searchMode;

private:
    enum
    {
        MaxMeanShiftIteration   = 5,
        MaxSetSizeIteration     = 5
    };

    void findOptimumSearchWindow(SearchWindow &searchWindow, IplImage *maskImage, IplImage *depthMap, int maxIteration, int resizeMethod, bool initDepth);

public:
    CvFuzzyMeanShiftTracker();
    ~CvFuzzyMeanShiftTracker();

    void track(IplImage *maskImage, IplImage *depthMap, int resizeMethod, bool resetSearch, int minKernelMass = MinKernelMass);
};


namespace cv
{

typedef bool (*BundleAdjustCallback)(int iteration, double norm_error, void* user_data);

class CV_EXPORTS LevMarqSparse {
public:
    LevMarqSparse();
    LevMarqSparse(int npoints, // number of points
                  int ncameras, // number of cameras
                  int nPointParams, // number of params per one point  (3 in case of 3D points)
                  int nCameraParams, // number of parameters per one camera
                  int nErrParams, // number of parameters in measurement vector
                  // for 1 point at one camera (2 in case of 2D projections)
                  Mat& visibility, // visibility matrix. rows correspond to points, columns correspond to cameras
                  // 1 - point is visible for the camera, 0 - invisible
                  Mat& P0, // starting vector of parameters, first cameras then points
                  Mat& X, // measurements, in order of visibility. non visible cases are skipped
                  TermCriteria criteria, // termination criteria

                  // callback for estimation of Jacobian matrices
                  void (*fjac)(int i, int j, Mat& point_params,
                                         Mat& cam_params, Mat& A, Mat& B, void* data),
                  // callback for estimation of backprojection errors
                  void (*func)(int i, int j, Mat& point_params,
                                         Mat& cam_params, Mat& estim, void* data),
                  void* data, // user-specific data passed to the callbacks
                  BundleAdjustCallback cb, void* user_data
                  );

    virtual ~LevMarqSparse();

    virtual void run( int npoints, // number of points
                     int ncameras, // number of cameras
                     int nPointParams, // number of params per one point  (3 in case of 3D points)
                     int nCameraParams, // number of parameters per one camera
                     int nErrParams, // number of parameters in measurement vector
                     // for 1 point at one camera (2 in case of 2D projections)
                     Mat& visibility, // visibility matrix. rows correspond to points, columns correspond to cameras
                     // 1 - point is visible for the camera, 0 - invisible
                     Mat& P0, // starting vector of parameters, first cameras then points
                     Mat& X, // measurements, in order of visibility. non visible cases are skipped
                     TermCriteria criteria, // termination criteria

                     // callback for estimation of Jacobian matrices
                     void (CV_CDECL * fjac)(int i, int j, Mat& point_params,
                                            Mat& cam_params, Mat& A, Mat& B, void* data),
                     // callback for estimation of backprojection errors
                     void (CV_CDECL * func)(int i, int j, Mat& point_params,
                                            Mat& cam_params, Mat& estim, void* data),
                     void* data // user-specific data passed to the callbacks
                     );

    virtual void clear();

    // useful function to do simple bundle adjustment tasks
    static void bundleAdjust(std::vector<Point3d>& points, // positions of points in global coordinate system (input and output)
                             const std::vector<std::vector<Point2d> >& imagePoints, // projections of 3d points for every camera
                             const std::vector<std::vector<int> >& visibility, // visibility of 3d points for every camera
                             std::vector<Mat>& cameraMatrix, // intrinsic matrices of all cameras (input and output)
                             std::vector<Mat>& R, // rotation matrices of all cameras (input and output)
                             std::vector<Mat>& T, // translation vector of all cameras (input and output)
                             std::vector<Mat>& distCoeffs, // distortion coefficients of all cameras (input and output)
                             const TermCriteria& criteria=
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON),
                             BundleAdjustCallback cb = 0, void* user_data = 0);

public:
    virtual void optimize(CvMat &_vis); //main function that runs minimization

    //iteratively asks for measurement for visible camera-point pairs
    void ask_for_proj(CvMat &_vis,bool once=false);
    //iteratively asks for Jacobians for every camera_point pair
    void ask_for_projac(CvMat &_vis);

    CvMat* err; //error X-hX
    double prevErrNorm, errNorm;
    double lambda;
    CvTermCriteria criteria;
    int iters;

    CvMat** U; //size of array is equal to number of cameras
    CvMat** V; //size of array is equal to number of points
    CvMat** inv_V_star; //inverse of V*

    CvMat** A;
    CvMat** B;
    CvMat** W;

    CvMat* X; //measurement
    CvMat* hX; //current measurement extimation given new parameter vector

    CvMat* prevP; //current already accepted parameter.
    CvMat* P; // parameters used to evaluate function with new params
    // this parameters may be rejected

    CvMat* deltaP; //computed increase of parameters (result of normal system solution )

    CvMat** ea; // sum_i  AijT * e_ij , used as right part of normal equation
    // length of array is j = number of cameras
    CvMat** eb; // sum_j  BijT * e_ij , used as right part of normal equation
    // length of array is i = number of points

    CvMat** Yj; //length of array is i = num_points

    CvMat* S; //big matrix of block Sjk  , each block has size num_cam_params x num_cam_params

    CvMat* JtJ_diag; //diagonal of JtJ,  used to backup diagonal elements before augmentation

    CvMat* Vis_index; // matrix which element is index of measurement for point i and camera j

    int num_cams;
    int num_points;
    int num_err_param;
    int num_cam_param;
    int num_point_param;

    //target function and jacobian pointers, which needs to be initialized
    void (*fjac)(int i, int j, Mat& point_params, Mat& cam_params, Mat& A, Mat& B, void* data);
    void (*func)(int i, int j, Mat& point_params, Mat& cam_params, Mat& estim, void* data);

    void* data;

    BundleAdjustCallback cb;
    void* user_data;
};

} // cv

#endif /* __cplusplus */

#endif /* __OPENCV_CONTRIB_COMPAT_HPP__ */
